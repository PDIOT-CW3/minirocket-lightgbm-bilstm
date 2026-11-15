# PDIoT Coursework 3 – model training pipeline

Two hybrid models for the RESpeck (12.5 Hz, 3-axis accelerometer), exported in the exact
format the Lab 4 app template loads: **raw g in, softmax out, normalisation inside the model**.

|                   | Task 1 · physical activity                                                | Task 2 · social signals                                                                   |
| ----------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Classes           | 11, in Lab 3 order                                                        | breathingNormally, coughing, hyperventilation, other                                      |
| Window / app step | 50 samples (4 s) / 25 (2 s)                                               | 125 samples (10 s) / 25 (2 s)                                                             |
| Input to network  | x, y, z, \|a\| (gravity **kept**: posture lives in it)                    | x, y, z, \|a\| with window mean removed (gravity **removed**: posture-invariant)          |
| Deep branch       | 3 × Conv1D + BatchNorm → GRU(64, unrolled)                                | 3 × Conv1D + BatchNorm (2 pools) → GRU(64, unrolled)                                      |
| Feature branch    | mean/std/min/max/mean\|diff\| per channel (orientation → static postures) | log power spectrum 0.05–2.5 Hz (fixed DFT matrix) + jerk statistics + window statistics   |
| Class imbalance   | balanced class weights                                                    | no weights + per-class logit offsets fitted on validation subjects (built into the model) |
| Augmentation      | rotation ±10°, scaling 0.9–1.1, jitter                                    | rotation ±20°, scaling, jitter                                                            |
| TFLite size       | ≈ 310 kB float                                                            | ≈ 400 kB float                                                                            |

Both convert with **standard TFLite builtins only** (the converter is set to fail otherwise), so
no Select-TF-ops/Flex library is needed in the app.

## Files

- `cw3lib.py` – data loading/cleaning, folds, custom layers, models, training, CW3 metric, TFLite export and verification.
- `cw3models.py` – the extra models (LightGBM on rich features, MiniRocket) and `load_predictor()`, which
  loads any exported model (TFLite or LightGBM, ours or another group's Lab-3-format TFLite).
- `run_cw3.py` – command-line runner: cross-validation, training on chosen folds, final export, testing.

## Setup

```
pip install tensorflow scikit-learn pandas lightgbm   # TF 2.16+; tested with TF 2.21 / Keras 3
```

Put the unzipped `datasetJan2026/` folder next to the scripts (or pass `--data PATH`).
On Colab: upload the two `.py` files and the zip, `!unzip -q datasetJan2026.zip`, then run the
commands below with `!python ...`.

## Usage

```
# 1. Cross-validation with the OFFICIAL Learn folds (required for the report)
python run_cw3.py --task activity --mode cv --folds <learn_folds.csv>
python run_cw3.py --task social   --mode cv --folds <learn_folds.csv>

# 2. Random-forest baseline on the same folds (comparison for the report)
python run_cw3.py --task activity --mode cv --model rf --folds <learn_folds.csv>
python run_cw3.py --task social   --mode cv --model rf --folds <learn_folds.csv>

# 3. Final models for the app (uses the median best epoch and mean offsets from step 1)
python run_cw3.py --task activity --mode final
python run_cw3.py --task social   --mode final

# smoke test (2 folds, 2 epochs, ~3 min)
python run_cw3.py --task activity --mode both --quick
```

### Other models: LightGBM and MiniRocket (`--model gbm | minirocket`)

|                                        | `gbm`                                                                                                                                        | `minirocket`                                                                                   |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| What                                   | LightGBM (gradient-boosted trees) on ~230 hand-crafted features per window                                                                   | MiniRocket (Dempster et al. 2021): 4032 fixed random-convolution features + softmax regression |
| Features                               | stats, skew/kurtosis, jerk, spectrum + dominant frequency, spectral entropy, band powers, autocorrelation period, peaks/s, axis correlations | learned automatically ("proportion of positive values" of 84 kernels × dilations)              |
| Output                                 | `<name>.txt` + meta + `<name>_importance.json` (top features, useful for the report)                                                         | `<name>.tflite` + meta + example – **works in the app** like the hybrid                        |
| Runs in                                | `predict.py`, `respeck_live.py`, `--mode test` (not the Flutter app)                                                                         | everywhere, incl. the app                                                                      |
| Speed                                  | ~30 s training per fold, ~1.5 ms per window                                                                                                  | ~3 min per fold, ~3 ms per window                                                              |
| Social, fold 0 (dev folds), calibrated | **80.6 / 84.9 / 81.8 / 78.4 %**                                                                                                              | 78.9 / 81.6 / 78.6 / 75.0 %                                                                    |

Same commands as the hybrid, just add `--model`:

```
python run_cw3.py --task social --mode cv    --model gbm        --folds <social_folds.csv>
python run_cw3.py --task social --mode final --model minirocket --folds <social_folds.csv>
```

Final names: `har_gbm`, `social_gbm`, `har_minirocket`, `social_minirocket` (or `--name`).

### Multibranch CNN-BiLSTM (`--model bilstm`)

After Challa et al., _"A multibranch CNN-BiLSTM model for human activity recognition using wearable
sensor data"_: three parallel conv branches (kernel sizes 3 / 7 / 11, two Conv1D+BN+ReLU each, max-pool),
concatenated → BiLSTM(64) → BiLSTM(32) → Dense(64) → softmax. Adapted to this project: gravity removed
(social) or |a| added (activity) and normalisation inside the model (raw g in), pooled to ≤ 31 time
steps and unrolled LSTMs so it converts to builtin TFLite ops, trained with the hybrid's recipe
(augmentation, early stopping, logit offsets). Exports `<name>.tflite` like the hybrid (works in the app).

```
python run_cw3.py --task social --mode cv --model bilstm --folds folds_social_signal.csv
```

Training prints one line per epoch (loss, accuracy, val_loss, val_accuracy) by default; `--verbose 1`
shows a progress bar, `--verbose 0` is silent. Output is line-buffered, so it also appears live when
piped, e.g. `... | Tee-Object cw3_out\log_social_bilstm_cv.txt`.

### CNN-BiLSTM + LightGBM features (`--model bilstm_feat`)

LightGBM scores best in CV, probably because of its hand-crafted features, but it can't run in the
app. `bilstm_feat` keeps the CNN-BiLSTM trunk unchanged and adds a feature branch. The `RichFeatures` layer
computes the same ~230 features as `rich_features()` (skew/kurtosis, percentiles, spectral
entropy/centroid, band powers, autocorrelation period, peaks/s, cross-axis correlation) **inside the
model** with builtin TFLite ops, then Normalization → Dense(64). The two branches are fused → Dense(64) → softmax.
It exports to `.tflite` like the other networks and works in the app. `--model bilstm` is unchanged and
writes its own files; this model writes `<task>_bilstm_feat_*`.

```
python run_cw3.py --task social --mode cv --model bilstm_feat --folds folds_social_signal.csv
python compare_cv.py --task social --models minirocket bilstm bilstm_feat gbm
python run_cw3.py --task social --mode final --model bilstm_feat --folds folds_social_signal.csv
```

### Comparing models (`compare_cv.py`)

```
python compare_cv.py --task social --models minirocket hybrid bilstm
```

Reads `cw3_out/<task>_<model>_cv.json`, checks the folds match, and writes per-class accuracy, the
CW3 group mean (calibrated and uncalibrated), per-fold numbers, paired t-test / Wilcoxon over folds
and confusion matrices to `cw3_out/<task>_cv_comparison.{md,json}`.

### Separate training and testing (`--mode train` / `--mode test`)

Fold numbers are exactly as in the Learn file (e.g. 1–10).

```
# train on folds 1-8, early stopping + calibration on fold 9, export
python run_cw3.py --task social --mode train --model gbm --folds <social_folds.csv> --train-folds 1-8 --val-fold 9

# score that saved model on the untouched fold 10 (per-class accuracy, group mean, confusion matrix)
python run_cw3.py --task social --mode test --name social_gbm --folds <social_folds.csv> --test-folds 10

# several test folds, reported separately
python run_cw3.py --task social --mode test --name social_gbm --folds <social_folds.csv> --test-folds 9,10 --per-fold

# test ANY Lab-3-format model, e.g. another group's TFLite in their_models/
python run_cw3.py --task activity --mode test --models their_models --name har_model
```

Without `--val-fold`, training uses the median epochs/rounds and mean offsets from that model's CV json.
The meta file records `trained_on_folds`; `--mode test` warns if you test on folds the model was
trained on. Results are saved as `<name>_test_folds<k>.json`.

To _watch_ a held-out test in the terminal, replay only recordings from the test fold:

```
python predict.py --har har_gbm --social social_gbm --random 5 --speed 10 \
    --folds <activity_folds.csv> <social_folds.csv> --fold 10
```

### Testing the final models on recordings (`predict.py`)

Needs the `--mode final` models in `cw3_out/`. It runs them exactly as the app will: one sample at a
time at 12.5 Hz, a prediction every 2 s, majority-vote smoothing over the last 3 predictions, and the
social model only while the posture is static. True labels come from the folder names.

```
# one answer per recording
python predict.py datasetJan2026/daily_activity/running/N_12.csv

# live replay at real sensor speed (one line every 2 s)
python predict.py --live datasetJan2026/social_signal/coughing/N_20.csv

# several recordings back to back = activity changes, 10x faster
python predict.py --live --speed 10 datasetJan2026/daily_activity/sittingStanding/N_3.csv datasetJan2026/daily_activity/normalWalking/N_7.csv

# random demo session of 5 recordings, as fast as possible
python predict.py --random 5 --speed 0
```

The summary shows, per recording, how many on-screen predictions were correct and how long it took
after the recording started until the first correct one (reaction time). Options: `--smooth N`
(1 = off), `--speed X` (1 = real time, 0 = no waiting), `--models DIR`, `--seed N` with `--random`,
`--har NAME` / `--social NAME` to pick models (e.g. `har_gbm`, `social_minirocket`).

### Live from the RESpeck in the terminal (`respeck_live.py`, no app)

Connects to the sensor over Bluetooth from the laptop (`pip install bleak`), decodes its packets the
same way as the app's `lib/ble.dart`, and runs both final models live with the same windows,
smoothing and static-posture gate. Disconnect the RESpeck from any phone first: it accepts one
connection at a time.

```
python respeck_live.py --scan                         # list nearby RESpecks
python respeck_live.py --id F0:5B:12:34:56:78         # ID from the QR code on the sensor
python respeck_live.py                                # or just the nearest RESpeck
python respeck_live.py --id ... --record s1.csv       # also save the raw data (replay later with predict.py)
python respeck_live.py --id ... --duration 120        # stop after 2 minutes
python respeck_live.py --id ... --social social_gbm   # use another model (also --har)
```

Every 2 s it prints the activity, the social signal, battery, the actual sample rate (should be
about 12.5 Hz), lost packets and processing time. Ctrl+C prints the time spent per activity.

Useful options: `--window/--step` (samples), `--weight-power 1|0.5|0`, `--calibrate on|off`,
`--quantize` (int8 weights, float32 I/O – still works with the template), `--out DIR`.

**Fold numbers** are kept as in the Learn file (1–10); generated folds are 0–9.

**Folds file.** `load_folds()` expects one row per recording with columns `activity, file, fold`
(as described in Lab 3 B9). If the Learn file uses other column names, edit the two `next(...)`
lines in `load_folds()`. Without `--folds`, folds are generated per activity from the `N_<id>.csv`
subject IDs – fine for development, **not** for reported results.

**What CV reports** (saved to `cw3_out/<task>_<model>_cv.json`): accuracy per class averaged over
folds, the group means exactly as CW3 defines them (static = 5 classes, dynamic = 6 classes,
each social class), per-fold numbers, uncalibrated scores, and the summed confusion matrix. Test
windows use the app's step (2 s), i.e. the windows the phone would actually see. Inside each
fold, one _training_ fold is held out for early stopping and calibration – the test fold is never
used for any decision.

Runtime: about 4–6 min per fold per task on a 2-core CPU, so roughly an hour per 10-fold run.

## Data cleaning done automatically

- `miscMovement/0.csv` has an extra `timestamp` column – only the three axes are read.
- `running (26).csv` has 3 missing values – interpolated.
- Files longer than 30 s (up to 1312 samples) – windows are capped per recording
  (40 for activity, 20 for social) so they do not dominate training.
- Values saturate at ±2 g in dynamic activities; augmentation clips to ±2 g to match.

## Output for the app

`cw3_out/` gets, per model, `<name>.tflite`, `<name>_meta.json` and `<name>_example.json`
(same fields as Lab 3), plus `<name>.keras` for re-conversion. `verify_tflite()` checks that
TFLite and Keras agree and times inference. Copy the three files into `assets/models/`.

The social model's meta file has an extra field
`"run_when": {"activity_model_predicts_one_of": [static classes]}` – all social training data is
stationary, so the app should run it (a second `HarClassifier`, Lab 4 Part 3.1) only while the
activity model reports a static posture, and buffer 125 samples for it.

## Results from development runs (generated folds, not the Learn folds)

| Run                                                      | Result                                                       |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| Activity, 2 folds × 2 epochs (smoke test)                | static 96.4 %, dynamic 84.2 %                                |
| Social, fold 0, 25 epochs, balanced weights              | 98.6 / 83.7 / 70.2 / 30.6 % (other collapses into breathing) |
| Social, fold 0, no weights                               | 94.0 / 72.2 / 62.3 / 62.8 %                                  |
| Social, fold 0, no weights **+ logit offsets** (default) | **79.1 / 81.5 / 74.1 / 70.5 %**                              |
| Social, fold 0, random forest on the same features       | 82.8 / 82.1 / 61.9 / 68.9 %                                  |

Order: breathingNormally / coughing / hyperventilation / other. The biggest remaining confusion
is hyperventilation and "other" (talking, eating, singing, laughing) being read as normal
breathing. Tested and not helpful: adding raw orientation to the social features, spectra up to
6.25 Hz, windows of 6 s or 15 s, and logit offsets for the activity model (overfits with 11 classes).

4% idle - 1 MIN
7-8% max spikes - 5 min
