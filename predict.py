"""
Run the exported TFLite models on recorded RESpeck CSVs exactly as the app will:
samples arrive one by one at 12.5 Hz, the newest window is classified every `step_samples`,
predictions are smoothed, and the social model only runs while the posture is static.

  # one-shot: classify every window of a recording and print an overall answer
  python predict.py datasetJan2026/daily_activity/running/N_12.csv

  # LIVE replay in the terminal at the real sensor speed (12.5 samples/s)
  python predict.py --live datasetJan2026/social_signal/coughing/N_20.csv

  # several recordings back to back = activity changes, replayed 10x faster
  python predict.py --live --speed 10 \
      datasetJan2026/daily_activity/sittingStanding/N_3.csv \
      datasetJan2026/daily_activity/normalWalking/N_7.csv \
      datasetJan2026/daily_activity/lyingBack/N_11.csv

  # a random demo session of N recordings
  python predict.py --live --speed 10 --random 4

Options: --models DIR (default cw3_out), --smooth N (majority vote over the last N
predictions, default 3; 1 = off), --speed X (1 = real time, 0 = as fast as possible).
The true class is taken from the folder name, so accuracy and reaction time are reported.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # hide TensorFlow start-up noise
import argparse
import contextlib
import io
import json
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np

import cw3lib as L
import cw3models as M

STATIC = set(L.STATIC_CLASSES)


# ----------------------------------------------------------------------------- models
class Classifier:
    """One TFLite model + its metadata, fed one sample at a time (like the app's HarClassifier)."""

    def __init__(self, models_dir: Path, name: str):
        self.ok = (Path(models_dir) / f"{name}_meta.json").exists()
        if not self.ok:
            return
        self.model = M.load_predictor(models_dir, name)      # TFLite or LightGBM, from the meta file
        self.meta = self.model.meta
        self.W, self.S, self.classes = self.model.W, self.model.S, self.model.classes
        self.buf = deque(maxlen=self.W)
        self.n = 0

    def reset(self):
        self.buf.clear(); self.n = 0

    def add(self, xyz):
        """Add one sample; returns (class, confidence, ms) when a prediction is due, else None.
        Like the app: first prediction once the window is full, then every step_samples."""
        self.buf.append(xyz); self.n += 1
        if self.n < self.W or (self.n - self.W) % self.S:
            return None
        t0 = time.perf_counter()
        p = self.model.predict_one(np.asarray(self.buf, np.float32))
        ms = (time.perf_counter() - t0) * 1000
        return self.classes[int(p.argmax())], float(p.max()), ms


def smoothed(history: deque) -> str:
    return Counter(history).most_common(1)[0][0]


# ----------------------------------------------------------------------------- helpers
def true_labels(path: Path):
    """(activity_truth, social_truth) from the folder name; None where unknown."""
    cls, task = path.parent.name, path.parent.parent.name
    if task == "daily_activity" and cls in L.ACTIVITY_CLASSES:
        return cls, None
    if task == "social_signal" and cls in L.SOCIAL_CLASSES:
        return None, cls          # social recordings are stationary, posture unknown
    return None, None


def held_out(data_dir: Path, folds_csv: str, fold_ids):
    """Set of recording paths in the given folds (both tasks share the folds file format)."""
    keep = set()
    for task in ("activity", "social"):
        cfg = L.TASKS[task]
        recs = L.list_recordings(data_dir, cfg)
        with contextlib.redirect_stdout(io.StringIO()):          # the other task's file matches nothing
            f = L.load_folds(recs, folds_csv)
        keep |= {str(p) for p in recs.path[f.isin(fold_ids)]}
    return keep


def pick_random(data_dir: Path, n: int, seed: int, only=None):
    rng = random.Random(seed)
    pools = [[p for p in sorted((data_dir / t / c).glob("*.csv")) if only is None or str(p) in only]
             for t, cs in [("daily_activity", L.ACTIVITY_CLASSES), ("social_signal", L.SOCIAL_CLASSES)]
             for c in cs]
    return [rng.choice(rng.choice([p for p in pools if p])) for _ in range(n)]


# ----------------------------------------------------------------------------- live replay
def live(files, har: Classifier, soc: Classifier, speed: float, smooth: int):
    dt = 0 if speed <= 0 else 1 / (L.SAMPLING_FREQ * speed)
    act_hist, soc_hist = deque(maxlen=smooth), deque(maxlen=smooth)
    scores = []          # (file, kind, truth, n_correct, n_total, first_correct_s)
    t_global, last_act, last_soc, infer_ms = 0.0, "…", "…", []

    print(f"replaying {len(files)} recording(s) at {'max' if dt == 0 else f'{speed:g}x'} speed, "
          f"smoothing over {smooth} prediction(s)\n")
    print(f"{'time':>7}  {'true':22s} {'activity (smoothed)':26s} {'social signal (smoothed)':28s}")
    for f in files:
        f = Path(f)
        sig = L.load_recording(f)
        a_true, s_true = true_labels(f)
        truth = a_true or s_true or "?"
        print(f"{'─' * 92}\n▶ {f.parent.name}/{f.name}  ({len(sig) / L.SAMPLING_FREQ:.0f} s)")
        t_start, a_ok, a_n, s_ok, s_n, first_ok = t_global, 0, 0, 0, 0, None
        for xyz in sig:
            t_global += 1 / L.SAMPLING_FREQ
            r_act = har.add(xyz)
            r_soc = soc.add(xyz) if soc.ok else None
            if r_soc:
                infer_ms.append(r_soc[2])
                if last_act in STATIC:
                    soc_hist.append(r_soc[0])
                    last_soc = f"{smoothed(soc_hist)} ({r_soc[1]:.0%})"
            if r_act:
                infer_ms.append(r_act[2])
                act_hist.append(r_act[0])
                last_act = smoothed(act_hist)
                if last_act not in STATIC:
                    soc_hist.clear(); last_soc = "– (moving)"
                elif not soc_hist:
                    last_soc = "– (filling 10 s window)"
                # score this update against the truth of the recording being played
                ok_a = a_true is not None and last_act == a_true
                ok_s = s_true is not None and soc_hist and smoothed(soc_hist) == s_true
                if a_true is not None:
                    a_n += 1; a_ok += ok_a
                if s_true is not None and soc_hist:
                    s_n += 1; s_ok += bool(ok_s)
                if first_ok is None and (ok_a or ok_s):
                    first_ok = t_global - t_start
                mark = "✓" if (ok_a or ok_s) else ("✗" if (a_true or (s_true and soc_hist)) else " ")
                act_txt = f"{last_act} ({r_act[1]:.0%})"
                print(f"{t_global:6.1f}s  {truth:22s} {act_txt:26s} {last_soc:28s} {mark}")
            if dt:
                time.sleep(dt)
        if a_true:
            scores.append((f.name, "activity", a_true, a_ok, a_n, first_ok))
        if s_true:
            scores.append((f.name, "social", s_true, s_ok, s_n, first_ok))

    print(f"{'─' * 92}\nSUMMARY (smoothed predictions shown on screen)")
    for name, kind, truth, ok, n, first in scores:
        acc = f"{ok}/{n} = {ok / n:.0%}" if n else "no predictions"
        react = f"{first:.1f} s" if first is not None else "never"
        print(f"  {name:28s} {kind:8s} true={truth:18s} correct {acc:14s} first correct after {react}")
    if infer_ms:
        print(f"  model inference: {np.mean(infer_ms):.2f} ms per window on this computer")
    print("Note: the first prediction needs one full window (4 s activity, 10 s social), and after a\n"
          "change the old activity stays in the window for up to one window length.")


# ----------------------------------------------------------------------------- one-shot
def oneshot(files, har: Classifier, soc: Classifier):
    for f in files:
        f = Path(f); sig = L.load_recording(f)
        har.reset(); soc.ok and soc.reset()
        acts, socs = [], []
        for xyz in sig:
            r = har.add(xyz)
            s = soc.add(xyz) if soc.ok else None
            if r: acts.append(r[0])
            if s and acts and acts[-1] in STATIC: socs.append(s[0])
        a = Counter(acts).most_common(1)[0][0] if acts else "too short"
        s = Counter(socs).most_common(1)[0][0] if socs else "n/a"
        print(f"{f.parent.name}/{f.name}: activity = {a} ({len(acts)} windows), social = {s} ({len(socs)} windows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--models", default="cw3_out")
    ap.add_argument("--live", action="store_true", help="replay sample by sample")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 10 = 10x faster, 0 = no waiting")
    ap.add_argument("--smooth", type=int, default=3, help="majority vote over the last N predictions")
    ap.add_argument("--random", type=int, default=0, help="play N random recordings from --data")
    ap.add_argument("--data", default="datasetJan2026")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--har", default="har_hybrid", help="activity model name (e.g. har_gbm, har_minirocket)")
    ap.add_argument("--social", default="social_hybrid", help="social model name (e.g. social_gbm)")
    ap.add_argument("--folds", nargs="+", default=None,
                    help="with --random: fold CSV file(s) to pick from (e.g. both Learn files)")
    ap.add_argument("--fold", default=None, help="with --random: only recordings in these folds, e.g. 10 or 9,10")
    args = ap.parse_args()

    d = Path(args.models)
    har, soc = Classifier(d, args.har), Classifier(d, args.social)
    if not har.ok:
        sys.exit(f"no {args.har}_meta.json in {d} – run: python run_cw3.py --task activity --mode final")
    if not soc.ok:
        print(f"(no {args.social} model in {d} – social signals will not be shown)")
    for c in (har, soc):
        if c.ok and c.meta.get("trained_on_folds") is not None:
            print(f"{c.model.name}: {c.model.backend}, trained on folds {c.meta['trained_on_folds']}")
    files = list(args.files)
    if args.random:
        only = None
        if args.fold:
            from run_cw3 import parse_folds
            only = set()
            for fcsv in (args.folds or [None]):
                only |= held_out(Path(args.data), fcsv, parse_folds(args.fold))
            print(f"picking from {len(only)} recordings in fold(s) {args.fold}")
        files += pick_random(Path(args.data), args.random,
                             args.seed if args.seed is not None else int(time.time()), only)
    if not files:
        sys.exit("give one or more CSV files, or --random N")
    if args.live or args.random:
        live(files, har, soc, args.speed, max(1, args.smooth))
    else:
        oneshot(files, har, soc)


if __name__ == "__main__":
    main()
