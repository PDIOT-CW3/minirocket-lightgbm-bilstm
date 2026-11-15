"""
PDIoT CW3 – cross-validate, train, export and test the models.

Models (--model):  hybrid (CNN-GRU, default) | bilstm (multibranch CNN-BiLSTM) | gbm (LightGBM)
                   | minirocket | rf (CV baseline only)
Modes  (--mode):
  cv     10-fold subject-wise cross-validation (1 test fold, 1 validation fold, 8 training)
  final  train on ALL folds and export for the app / predict.py   (= train with every fold)
  train  train on chosen folds only (--train-folds, optional --val-fold) and export
  test   load an exported model (--name) and score it on chosen folds (--test-folds)
  both   cv, then final

Examples
  python run_cw3.py --task social --mode cv --model gbm --folds social_folds.csv
  python run_cw3.py --task social --mode final --model minirocket --folds social_folds.csv

  # separate training and testing: train on folds 1-9, test on fold 10
  python run_cw3.py --task social --mode train --model gbm --folds social_folds.csv --train-folds 1-8 --val-fold 9
  python run_cw3.py --task social --mode test  --name social_gbm --folds social_folds.csv --test-folds 10

  # test someone else's TFLite model (Lab 3 format) on your data
  python run_cw3.py --task activity --mode test --models their_models --name har_model
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import cw3lib as L
import cw3models as M

PREFIX = {"activity": "har", "social": "social"}


def parse_folds(spec):
    """'1-8,10' -> [1, 2, ..., 8, 10]"""
    if spec is None:
        return None
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def report(name, cfg, accs, cms, extra, out_dir, raw=None):
    summary = L.summarise(accs, cfg)
    summary.update(extra)
    summary["confusion_matrix_summed"] = np.sum(cms, 0).tolist()
    if raw:
        summary["uncalibrated"] = L.summarise(raw, cfg)["groups"]
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    n = len(accs)
    print(f"\n=== {name}: " + (f"mean over {n} folds" if n > 1 else "result") + " ===")
    for c, v in summary["per_class"].items():
        print(f"  {c:18s} {v['mean']:.3f}" + (f" ± {v['std']:.3f}" if n > 1 else ""))
    for g, v in summary["groups"].items():
        r = summary.get("uncalibrated", {}).get(g)
        print(f"  GROUP {g:12s} {v['mean']:.3f}" + (f" ± {v['std']:.3f}" if n > 1 else "")
              + (f"   (uncalibrated {r['mean']:.3f})" if r else ""))
    print("  summed confusion matrix (rows = true, cols = predicted):")
    for c, row in zip(cfg.classes, summary["confusion_matrix_summed"]):
        print(f"  {c[:12]:12s} {row}")
    print(f"saved {out_dir / (name + '.json')}")
    return summary


def calibrate(eng, cfg, X_va, y_va):
    if not cfg.calibrate or X_va is None or not len(X_va):
        return None
    return L.fit_logit_offsets(eng.predict_proba(X_va), y_va, len(cfg.classes), min_weight=cfg.calib_min_weight)


# ----------------------------------------------------------------------------- cv
def run_cv(cfg, recs, folds, args, out_dir):
    fold_ids = sorted(int(f) for f in folds.unique() if f >= 0)
    if args.quick:
        fold_ids = fold_ids[:2]
    import time
    use_val = args.model != "rf"
    accs, raw, cms, best_epochs, offsets = [], [], [], [], []
    print(f"CV model={args.model}, {len(fold_ids)} folds" + (" (val = next fold)" if use_val else "")
          + f", test step {cfg.step}")

    def group_means(a):
        return "  ".join(f"{g} {np.nanmean([a[cfg.classes.index(c)] for c in m]):.3f}" for g, m in cfg.groups.items())

    for i, k in enumerate(fold_ids):
        t0 = time.time()
        test = recs[folds == k]
        rest = recs[(folds != k) & (folds >= 0)]
        val_k = fold_ids[(fold_ids.index(k) + 1) % len(fold_ids)] if use_val else None
        val = rest[folds.loc[rest.index] == val_k] if use_val else rest.iloc[:0]
        train = rest.drop(val.index)
        X_tr, y_tr, _ = L.windows_from_recordings(train, cfg)
        X_te, y_te, _ = L.windows_from_recordings(test, cfg, step=cfg.step)
        X_va, y_va = (L.windows_from_recordings(val, cfg, step=cfg.step)[:2] if use_val else (None, None))
        eng = M.ENGINES[args.model](cfg, args)
        best_epochs.append(eng.fit(X_tr, y_tr, X_va, y_va))
        p_te = eng.predict_proba(X_te)
        raw.append(L.per_class_accuracy(y_te, p_te.argmax(1), len(cfg.classes)))
        b = calibrate(eng, cfg, X_va, y_va)
        if b is not None:
            offsets.append(b.tolist())
            p_te = L.apply_offsets(p_te, b)
        y_hat = p_te.argmax(1)
        acc = L.per_class_accuracy(y_te, y_hat, len(cfg.classes))
        accs.append(acc); cms.append(L.confusion(y_te, y_hat, len(cfg.classes)))
        print(f"fold {k} ({i + 1:2d}/{len(fold_ids)})  "
              + "  ".join(f"{c[:10]} {a:.2f}" for c, a in zip(cfg.classes, acc))
              + f" | {group_means(acc)}" + (f" (uncalibrated: {group_means(raw[-1])})" if b is not None else "")
              + (f" | best {best_epochs[-1]}" if use_val else "") + f" | {time.time() - t0:.0f} s")
    extra = {"model": args.model, "folds": fold_ids, "best_epochs": best_epochs, "logit_offsets": offsets,
             "per_fold_class_accuracy": np.round(np.array(accs), 4).tolist(),
             "per_fold_class_accuracy_uncalibrated": np.round(np.array(raw), 4).tolist(),
             "window": cfg.window, "step": cfg.step}
    return report(f"{cfg.name}_{args.model}_cv", cfg, accs, cms, extra, out_dir,
                  raw if offsets else None)


# ----------------------------------------------------------------------------- train / final
def run_train(cfg, recs, folds, args, out_dir, cv_summary=None, all_folds=False):
    if args.model == "rf":
        print("rf is a CV baseline only (not exported)"); return
    valid = sorted(int(f) for f in folds.unique() if f >= 0)
    val_fold = None if all_folds else args.val_fold
    train_ids = valid if all_folds or args.train_folds is None else parse_folds(args.train_folds)
    train_ids = [f for f in train_ids if f != val_fold]
    if not (folds >= 0).any():
        train_ids = valid
    train = recs[folds.isin(train_ids)]
    val = recs[folds == val_fold] if val_fold is not None else recs.iloc[:0]
    name = args.name or f"{PREFIX[cfg.name]}_{args.model}" + ("_int8" if args.quantize else "")

    cv_file = out_dir / f"{cfg.name}_{args.model}_cv.json"
    if cv_summary is None and cv_file.exists():
        cv_summary = json.loads(cv_file.read_text())
    X, y, _ = L.windows_from_recordings(train, cfg)
    eng = M.ENGINES[args.model](cfg, args)
    if len(val):                                        # early stopping + calibration on --val-fold
        X_va, y_va, _ = L.windows_from_recordings(val, cfg, step=cfg.step)
        print(f"{name}: train folds {train_ids} ({len(train)} recs, {len(X)} windows), "
              f"validation fold {val_fold} ({len(val)} recs)")
        best = eng.fit(X, y, X_va, y_va)
        offsets = calibrate(eng, cfg, X_va, y_va)
    else:                                               # fixed length from CV (median best epoch)
        epochs = None
        if cv_summary and cv_summary.get("best_epochs"):
            epochs = max(1, int(np.median(cv_summary["best_epochs"])))
            src = f"median of CV in {cv_file.name}"
        else:
            src = "default – run --mode cv first or give --val-fold"
        if args.quick:
            epochs = 2
        print(f"{name}: train folds {train_ids} ({len(train)} recs, {len(X)} windows), "
              f"{'rounds' if args.model == 'gbm' else 'epochs'}: {epochs or 'default'} ({src})")
        best = eng.fit(X, y, epochs=epochs)
        offsets = None
        if cfg.calibrate and cv_summary and cv_summary.get("logit_offsets"):
            offsets = np.mean(cv_summary["logit_offsets"], axis=0)
            print(f"   class offsets (mean over CV folds): {np.round(offsets, 2)}")
    extra = {"trained_on_folds": train_ids if (folds >= 0).any() else "all",
             "validation_fold": val_fold, "folds_file": args.folds, "model_family": args.model,
             "best_epochs_or_rounds": best}
    X_chk, y_chk, _ = L.windows_from_recordings(train, cfg, step=cfg.step)
    eng.export(out_dir, name, offsets, extra, X_chk, y_chk)
    print(f"done – test it with:  python run_cw3.py --task {cfg.name} --mode test --name {name} "
          f"--models {out_dir}" + (f" --folds {args.folds}" if args.folds else "") + " --test-folds <fold>")


# ----------------------------------------------------------------------------- test
def run_test(cfg, recs, folds, args):
    models_dir = Path(args.models or args.out)
    name = args.name or f"{PREFIX[cfg.name]}_{args.model}"
    pred = M.load_predictor(models_dir, name)
    if pred.classes != list(cfg.classes):
        missing = set(cfg.classes) ^ set(pred.classes)
        if missing:
            raise SystemExit(f"model classes {pred.classes} do not match task classes ({missing})")
    # use the model's own window/step, map its class order to ours
    cfg.window, cfg.step = pred.W, pred.S
    order = [pred.classes.index(c) for c in cfg.classes]
    test_ids = parse_folds(args.test_folds)
    sel = recs[folds.isin(test_ids)] if test_ids else recs[folds >= 0] if (folds >= 0).any() else recs
    trained = pred.meta.get("trained_on_folds")
    if trained is not None:
        overlap = sorted(set(test_ids or []) & set(trained if isinstance(trained, list) else []))
        if trained == "all" or overlap or (not test_ids and isinstance(trained, list)):
            print(f"!! WARNING: this model was trained on folds {trained} – testing on training data "
                  f"gives optimistic accuracy. Use --mode train with a held-out fold for a fair test.")
    accs, cms = [], []
    groups = test_ids if (test_ids and args.per_fold) else [None]
    for k in groups:
        part = sel if k is None else sel[folds.loc[sel.index] == k]
        X, y, _ = L.windows_from_recordings(part, cfg, step=cfg.step)
        print(f"testing {name} ({pred.backend}) on {len(part)} recordings / {len(X)} windows"
              + (f" (fold {k})" if k is not None else f" (folds {test_ids or 'all'})"))
        p = pred.predict(X)[:, order]
        y_hat = p.argmax(1)
        accs.append(L.per_class_accuracy(y, y_hat, len(cfg.classes)))
        cms.append(L.confusion(y, y_hat, len(cfg.classes)))
    out_dir = Path(args.out)
    tag = "_folds" + "-".join(map(str, test_ids)) if test_ids else "_all"
    return report(f"{name}_test{tag}", cfg, accs, cms,
                  {"model_name": name, "test_folds": test_ids or "all", "trained_on_folds": trained},
                  out_dir)


def main():
    try:                                     # show every line immediately, even when piped / logged
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="datasetJan2026", help="folder with daily_activity/ and social_signal/")
    p.add_argument("--task", choices=["activity", "social"], required=True)
    p.add_argument("--mode", choices=["cv", "final", "train", "test", "both"], default="cv")
    p.add_argument("--model", choices=list(M.ENGINES), default="hybrid")
    p.add_argument("--folds", default=None, help="Learn fold CSV (activity, file, fold)")
    p.add_argument("--out", default="cw3_out")
    p.add_argument("--name", default=None, help="model name for train/final/test (default har_<model> / social_<model>)")
    p.add_argument("--models", default=None, help="test: folder with the model (default --out)")
    p.add_argument("--train-folds", default=None, help="train: e.g. 1-8 (default all except --val-fold)")
    p.add_argument("--val-fold", type=int, default=None, help="train: fold for early stopping + calibration")
    p.add_argument("--test-folds", default=None, help="test: e.g. 10 or 9,10 (default all)")
    p.add_argument("--per-fold", action="store_true", help="test: score each test fold separately")
    p.add_argument("--epochs", type=int, default=40, help="max epochs (hybrid, minirocket head)")
    p.add_argument("--gbm-rounds", type=int, default=2000, help="max boosting rounds (gbm)")
    p.add_argument("--mr-features", type=int, default=4032, help="MiniRocket features (multiple of 84)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--window", type=int, default=None, help="override window (samples)")
    p.add_argument("--step", type=int, default=None, help="override app step (samples)")
    p.add_argument("--quantize", action="store_true", help="int8 weights, float32 input/output (TFLite)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0,
                   help="Keras training output: 2 one line per epoch (default), 1 progress bar, 0 silent")
    p.add_argument("--quick", action="store_true", help="2 folds, 2 epochs – smoke test only")
    p.add_argument("--calibrate", choices=["on", "off"], default=None,
                   help="per-class logit offsets (default: on for social, off for activity)")
    p.add_argument("--calib-min-weight", type=float, default=None,
                   help="calibration: weight of the worst class (default 0.5; try 2 to balance classes)")
    p.add_argument("--weight-power", type=float, default=None,
                   help="class-weight strength: 1 balanced, 0.5 sqrt, 0 none")
    args = p.parse_args()
    if args.quick:
        args.epochs, args.gbm_rounds = 2, 20

    cfg = L.TASKS[args.task]
    if args.window:
        cfg.window = args.window
    if args.step:
        cfg.step = args.step
    if args.weight_power is not None:
        cfg.weight_power = args.weight_power
    if args.calibrate:
        cfg.calibrate = args.calibrate == "on"
    if args.calib_min_weight is not None:
        cfg.calib_min_weight = args.calib_min_weight
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    L.set_seed(args.seed)

    recs = L.list_recordings(Path(args.data), cfg)
    folds = L.load_folds(recs, args.folds, seed=args.seed)
    print(f"{cfg.name}: {len(recs)} recordings, folds {sorted(int(f) for f in folds.unique() if f >= 0)}, "
          f"window {cfg.window} ({cfg.window / L.SAMPLING_FREQ:.1f} s), app step {cfg.step}")

    if args.mode == "test":
        run_test(cfg, recs, folds, args); return
    summary = None
    if args.mode in ("cv", "both"):
        summary = run_cv(cfg, recs, folds, args, out_dir)
    if args.mode in ("final", "both"):
        run_train(cfg, recs, folds, args, out_dir, summary, all_folds=True)
    if args.mode == "train":
        run_train(cfg, recs, folds, args, out_dir)


if __name__ == "__main__":
    main()
