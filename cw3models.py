"""
Extra models for CW3, behind one common interface so run_cw3.py can cross-validate, train,
export and test any of them the same way:

  hybrid      CNN-GRU + feature branch (cw3lib)                               -> .tflite
  gbm         LightGBM on a rich hand-crafted feature set                    -> .txt (LightGBM)
  minirocket  MiniRocket random convolution features + linear softmax head   -> .tflite
  rf          random forest on the hybrid's feature branch (CV baseline only)
  bilstm      multibranch CNN-BiLSTM                                         -> .tflite
  bilstm_feat CNN-BiLSTM + LightGBM's features computed in-model             -> .tflite

Every exported model gets a <name>_meta.json in the Lab 3 format (+ "backend") so predict.py,
respeck_live.py and `run_cw3.py --mode test` load them all with load_predictor().
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import keras
from keras import layers, ops

import cw3lib as L

reg = L.reg


# ======================================================================================
# 1. Rich hand-crafted features (for LightGBM)
# ======================================================================================
CH = ["x", "y", "z", "mag"]
BANDS = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 6.25)]


def _rich_chunk(X: np.ndarray, task: str):
    """X (n, W, 3) raw g -> (features (n, F), names). All vectorised numpy."""
    n, W, _ = X.shape
    fs = L.SAMPLING_FREQ
    X = X.astype(np.float64)
    mag = np.sqrt((X ** 2).sum(-1, keepdims=True) + 1e-12)
    base = np.concatenate([X, mag], -1)                      # (n, W, 4) with gravity
    dyn = base - base.mean(1, keepdims=True)                 # gravity removed
    src = base if task == "activity" else dyn                # activity keeps posture info
    F, names = [], []

    def add(arr, label):                                     # arr (n, 4) -> one name per channel
        F.append(arr); names.extend(f"{label}_{c}" for c in CH)

    # --- distribution of the signal
    sd = dyn.std(1)
    if task == "activity":
        add(base.mean(1), "mean")                            # orientation = posture
    add(sd, "std")
    add(src.min(1), "min"); add(src.max(1), "max"); add(src.max(1) - src.min(1), "range")
    q25, q50, q75 = np.percentile(src, [25, 50, 75], axis=1)
    add(q50, "median"); add(q75 - q25, "iqr")
    add((dyn ** 3).mean(1) / (sd ** 3 + 1e-9), "skew")     # coughs: asymmetric spikes
    add((dyn ** 4).mean(1) / (sd ** 4 + 1e-9), "kurtosis")
    add(np.sqrt((dyn ** 2).mean(1)), "rms")
    add((np.diff(np.sign(dyn), axis=1) != 0).mean(1) * fs, "zero_cross_per_s")

    # --- jerk (sample-to-sample change)
    d = np.diff(base, axis=1); ad = np.abs(d)
    add(ad.mean(1), "jerk_mean"); add(d.std(1), "jerk_std"); add(ad.max(1), "jerk_max")
    add((ad > 3 * ad.std(1, keepdims=True) + 1e-4).mean(1), "jerk_spike_frac")

    # --- spectrum (Hann window, same scaling as cw3lib.SpectralPower)
    freqs = np.fft.rfftfreq(W, 1 / fs)
    P = np.abs(np.fft.rfft(dyn * np.hanning(W)[None, :, None], axis=1)) ** 2   # (n, K, 4)
    use = freqs >= 0.1
    Pu, fu = P[:, use], freqs[use]
    tot = Pu.sum(1) + 1e-12
    pn = Pu / tot[:, None]
    add(np.log(tot), "log_power")
    add(fu[Pu.argmax(1)], "dominant_freq")                   # breathing rate / step rate
    add(Pu.max(1) / tot, "dominant_frac")                    # how rhythmic
    add(-(pn * np.log(pn + 1e-12)).sum(1) / np.log(len(fu)), "spectral_entropy")
    add((fu[None, :, None] * pn).sum(1), "spectral_centroid")
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            bp = P[:, m].sum(1)
            add(np.log(bp + 1e-9), f"band_{lo:g}_{hi:g}Hz_log")
            add(bp / tot, f"band_{lo:g}_{hi:g}Hz_frac")
    fmax = 2.5 if task == "social" else 6.25                 # fine spectrum (as the hybrid)
    for k in np.where((freqs >= 0.05) & (freqs <= fmax))[0]:
        add(np.log(P[:, k] + 1e-4), f"spec_{freqs[k]:.2f}Hz")

    # --- periodicity (autocorrelation): breathing is regular, talking/eating is not
    R = np.fft.irfft(np.abs(np.fft.rfft(dyn, n=2 * W, axis=1)) ** 2, axis=1)[:, :W]
    R = R / (R[:, :1] + 1e-12)
    lo, hi = int(0.8 * fs), min(int(6 * fs), W // 2)
    if hi > lo + 1:
        seg = R[:, lo:hi]
        add(seg.max(1), "autocorr_peak"); add((seg.argmax(1) + lo) / fs, "autocorr_period_s")

    # --- peaks (breaths / steps / coughs)
    mid = dyn[:, 1:-1]
    pk = (mid > dyn[:, :-2]) & (mid >= dyn[:, 2:]) & (mid > 0.5 * sd[:, None, :])
    add(pk.sum(1) / (W / fs), "peaks_per_s")

    # --- cross-axis correlation (how the chest moves as a whole)
    z = dyn / (sd[:, None, :] + 1e-9)
    for i, j in itertools.combinations(range(3), 2):
        F.append((z[:, :, i] * z[:, :, j]).mean(1)[:, None]); names.append(f"corr_{CH[i]}{CH[j]}")

    return np.concatenate(F, 1).astype(np.float32), names


def rich_features(X: np.ndarray, task: str, chunk: int = 8000):
    out, names = [], None
    for i in range(0, len(X), chunk):
        f, names = _rich_chunk(X[i:i + chunk], task)
        out.append(f)
    return np.concatenate(out), names


@reg
class RichFeatures(layers.Layer):
    """Keras port of rich_features() (same features, same order) for the `bilstm_feat` model:
    raw g (B, W, 3) -> (B, F). Only builtin TFLite ops: FFTs and the autocorrelation are fixed
    DFT matrices (MatMul, as cw3lib.SpectralPower), percentiles use sort (TopK), dominant
    frequency / period use ArgMax. It does not have to match numpy bit-for-bit (the network
    learns from its own outputs), but it does to float32 precision."""

    def __init__(self, window: int, task: str = "social", fs: float = L.SAMPLING_FREQ, **kw):
        super().__init__(**kw)
        self.window, self.task, self.fs = window, task, fs
        W = window
        freqs = np.fft.rfftfreq(W, 1 / fs)
        t = np.arange(W)
        hann = np.hanning(W)[:, None]
        ang = 2 * np.pi * np.outer(t, np.arange(len(freqs))) / W
        self._c = {"cos": hann * np.cos(ang), "sin": hann * np.sin(ang)}
        # spectral features use bins with f >= 0.1 Hz (contiguous suffix of the rfft bins)
        self.u0 = int(np.argmax(freqs >= 0.1))
        fu = freqs[self.u0:]
        self.fu0, self.df, self.n_fu = float(fu[0]), float(freqs[1]), len(fu)
        self._c["fu"] = fu[None, None, :]
        self.bands = [(lo, hi) for lo, hi in BANDS if ((freqs >= lo) & (freqs < hi)).any()]
        self._c["band_mask"] = np.stack([((freqs >= lo) & (freqs < hi)).astype(float)
                                         for lo, hi in self.bands], 1)              # (K, n_bands)
        fmax = 2.5 if task == "social" else 6.25
        fine = np.where((freqs >= 0.05) & (freqs <= fmax))[0]
        self.fine0, self.fine1 = int(fine[0]), int(fine[-1]) + 1                   # contiguous
        # autocorrelation = irfft(|rfft(x, n=2W)|^2): forward DFT (W, W+1), inverse cos (W+1, lags)
        N = 2 * W
        j = np.arange(N // 2 + 1)
        ang2 = 2 * np.pi * np.outer(t, j) / N
        self._c["ac_cos"], self._c["ac_sin"] = np.cos(ang2), np.sin(ang2)
        self.lo, self.hi = int(0.8 * fs), min(int(6 * fs), W // 2)
        self.has_ac = self.hi > self.lo + 1
        lags = np.r_[0, np.arange(self.lo, self.hi)] if self.has_ac else np.array([0])
        wj = np.where((j == 0) | (j == N // 2), 1.0, 2.0)[:, None]
        self._c["ac_inv"] = wj * np.cos(2 * np.pi * np.outer(j, lags) / N) / N

    def build(self, input_shape):
        self.c = {k: self.add_weight(name=k, shape=v.shape, trainable=False,
                                     initializer=keras.initializers.Constant(v.astype(np.float32)))
                  for k, v in self._c.items()}

    def call(self, x):
        W, fs = self.window, self.fs
        f32 = lambda v: ops.cast(v, "float32")
        mag = ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + 1e-12)
        base = ops.concatenate([x, mag], axis=-1)                       # (B, W, 4)
        dyn = base - ops.mean(base, axis=1, keepdims=True)
        src = base if self.task == "activity" else dyn
        F = []

        # --- distribution of the signal
        sd = ops.std(dyn, axis=1)
        if self.task == "activity":
            F.append(ops.mean(base, axis=1))
        mn, mx = ops.min(src, axis=1), ops.max(src, axis=1)
        F += [sd, mn, mx, mx - mn]
        s = ops.sort(ops.transpose(src, (0, 2, 1)), axis=-1)            # (B, 4, W) ascending

        def pct(q):                                                     # numpy 'linear' percentile
            pos = q * (W - 1)
            i0 = int(np.floor(pos)); fr = pos - i0; i1 = min(i0 + 1, W - 1)
            return s[:, :, i0] * (1 - fr) + s[:, :, i1] * fr
        q25, q50, q75 = pct(0.25), pct(0.5), pct(0.75)
        d2 = dyn * dyn
        F += [q50, q75 - q25,
              ops.mean(d2 * dyn, axis=1) / (sd * sd * sd + 1e-9),
              ops.mean(d2 * d2, axis=1) / (sd * sd * sd * sd + 1e-9),
              ops.sqrt(ops.mean(d2, axis=1))]
        sgn = f32(ops.greater(dyn, 0.0)) - f32(ops.less(dyn, 0.0))
        F.append(ops.mean(f32(ops.not_equal(sgn[:, 1:], sgn[:, :-1])), axis=1) * fs)

        # --- jerk
        d = base[:, 1:] - base[:, :-1]
        ad = ops.abs(d)
        F += [ops.mean(ad, axis=1), ops.std(d, axis=1), ops.max(ad, axis=1),
              ops.mean(f32(ops.greater(ad, 3.0 * ops.std(ad, axis=1, keepdims=True) + 1e-4)), axis=1)]

        # --- spectrum (Hann window)
        xt = ops.transpose(dyn, (0, 2, 1))                              # (B, 4, W)
        P = ops.square(ops.matmul(xt, self.c["cos"])) + ops.square(ops.matmul(xt, self.c["sin"]))  # (B, 4, K)
        Pu = P[:, :, self.u0:]
        tot = ops.sum(Pu, axis=-1) + 1e-12
        pn = Pu / tot[:, :, None]
        F += [ops.log(tot),
              f32(ops.argmax(Pu, axis=-1)) * self.df + self.fu0,
              ops.max(Pu, axis=-1) / tot,
              -ops.sum(pn * ops.log(pn + 1e-12), axis=-1) / float(np.log(self.n_fu)),
              ops.sum(self.c["fu"] * pn, axis=-1)]
        bp = ops.matmul(P, self.c["band_mask"])                         # (B, 4, n_bands)
        for b in range(len(self.bands)):
            F += [ops.log(bp[:, :, b] + 1e-9), bp[:, :, b] / tot]
        fine = ops.log(P[:, :, self.fine0:self.fine1] + 1e-4)           # (B, 4, nk) -> bin-major
        F.append(ops.reshape(ops.transpose(fine, (0, 2, 1)), (-1, 4 * (self.fine1 - self.fine0))))

        # --- periodicity (autocorrelation over lags 0.8 - 6 s)
        if self.has_ac:
            S = ops.square(ops.matmul(xt, self.c["ac_cos"])) + ops.square(ops.matmul(xt, self.c["ac_sin"]))
            R = ops.matmul(S, self.c["ac_inv"])                         # (B, 4, 1 + lags)
            seg = R[:, :, 1:] / (R[:, :, :1] + 1e-12)
            F += [ops.max(seg, axis=-1), (f32(ops.argmax(seg, axis=-1)) + self.lo) / fs]

        # --- peaks
        mid = dyn[:, 1:-1]
        pk = (f32(ops.greater(mid, dyn[:, :-2])) * f32(ops.greater_equal(mid, dyn[:, 2:]))
              * f32(ops.greater(mid, 0.5 * sd[:, None, :])))
        F.append(ops.sum(pk, axis=1) / (W / fs))

        # --- cross-axis correlation
        z = dyn / (sd[:, None, :] + 1e-9)
        for i, k in itertools.combinations(range(3), 2):
            F.append(ops.mean(z[:, :, i] * z[:, :, k], axis=1, keepdims=True))
        return ops.concatenate(F, axis=-1)

    def get_config(self):
        return {**super().get_config(), "window": self.window, "task": self.task, "fs": self.fs}


# ======================================================================================
# 2. MiniRocket (Dempster, Schmidt & Webb 2021) as a Keras layer -> builtin TFLite ops
# ======================================================================================
_MR_KERNELS = np.array([[2.0 if i in idx else -1.0 for i in range(9)]
                        for idx in itertools.combinations(range(9), 3)], np.float32)   # (84, 9)


def _mr_dilations(window: int, n_features: int, max_dil_per_kernel: int = 32):
    per_kernel = max(1, n_features // 84)
    true_max = min(per_kernel, max_dil_per_kernel)
    mult = per_kernel / true_max
    max_exp = np.log2((window - 1) / 8)
    dil, counts = np.unique(np.floor(np.logspace(0, max_exp, true_max, base=2)).astype(int),
                            return_counts=True)
    nq = (counts * mult).astype(int)
    rem = per_kernel - nq.sum()
    i = 0
    while rem > 0:
        nq[i] += 1; rem -= 1; i = (i + 1) % len(nq)
    return dil.tolist(), nq.tolist()


@reg
class MiniRocketFeatures(layers.Layer):
    """(B, W, C) -> (B, 84 * sum(n_quantiles)) "proportion of positive values" features.
    84 fixed kernels (weights -1/2), several dilations, a random channel subset per
    kernel/dilation, and biases taken from quantiles of training convolution outputs.
    Only Conv / Greater / Cast / Mean ops, so it converts to builtin TFLite."""

    def __init__(self, n_channels, dilations, n_quantiles, channel_masks, biases, **kw):
        super().__init__(**kw)
        self.n_channels, self.dilations, self.n_quantiles = n_channels, list(dilations), list(n_quantiles)
        self.channel_masks = [np.asarray(m, np.float32).tolist() for m in channel_masks]
        self.bias_values = [np.asarray(b, np.float32).tolist() for b in biases]

    def build(self, input_shape):
        self.kern, self.bias = [], []
        for i, (d, nq) in enumerate(zip(self.dilations, self.n_quantiles)):
            mask = np.asarray(self.channel_masks[i], np.float32)                     # (84, C)
            w = _MR_KERNELS.T[:, None, :] * mask.T[None, :, :]                       # (9, C, 84)
            self.kern.append(self.add_weight(name=f"k{d}", shape=w.shape, trainable=False,
                                             initializer=keras.initializers.Constant(w)))
            b = np.asarray(self.bias_values[i], np.float32).reshape(1, 1, 84, nq)
            self.bias.append(self.add_weight(name=f"b{d}", shape=b.shape, trainable=False,
                                             initializer=keras.initializers.Constant(b)))

    def conv(self, x, i):
        return ops.conv(x, self.kern[i], strides=1, padding="same", dilation_rate=self.dilations[i])

    def call(self, x):
        out = []
        for i, nq in enumerate(self.n_quantiles):
            c = ops.expand_dims(self.conv(x, i), -1)                                 # (B, W, 84, 1)
            ppv = ops.mean(ops.cast(ops.greater(c, self.bias[i]), "float32"), axis=1)
            out.append(ops.reshape(ppv, (-1, 84 * nq)))
        return ops.concatenate(out, axis=-1)

    def get_config(self):
        return {**super().get_config(), "n_channels": self.n_channels, "dilations": self.dilations,
                "n_quantiles": self.n_quantiles, "channel_masks": self.channel_masks,
                "biases": self.bias_values}


def fit_minirocket(sig: np.ndarray, n_features: int, seed: int = 0) -> MiniRocketFeatures:
    """Choose dilations, channel subsets and biases from training signals sig (n, W, C)."""
    rng = np.random.default_rng(seed)
    n, W, C = sig.shape
    dil, nq = _mr_dilations(W, n_features)
    phi = (np.sqrt(5) + 1) / 2
    q_all = (np.arange(1, 84 * sum(nq) + 1) * phi) % 1                # low-discrepancy quantiles
    masks, biases, qi = [], [], 0
    for d, k in zip(dil, nq):
        m = np.zeros((84, C), np.float32)
        for j in range(84):
            nc = int(2 ** rng.uniform(0, np.log2(min(C, 9) + 1)))
            m[j, rng.choice(C, size=max(1, min(nc, C)), replace=False)] = 1
        masks.append(m)
        biases.append(np.zeros((84, k), np.float32))
    layer = MiniRocketFeatures(C, dil, nq, masks, biases)
    layer.build((None, W, C))
    for i, (d, k) in enumerate(zip(dil, nq)):
        ex = sig[rng.integers(0, n, size=84)]                             # one random example per kernel
        c = keras.ops.convert_to_numpy(layer.conv(ex, i))                 # (84, W, 84)
        own = c[np.arange(84), :, np.arange(84)]                          # example j through kernel j
        b = np.stack([np.quantile(own[j], q_all[qi + j * k: qi + (j + 1) * k]) for j in range(84)])
        qi += 84 * k
        biases[i] = b.astype(np.float32)
    return MiniRocketFeatures(C, dil, nq, masks, biases, name="minirocket")


# ======================================================================================
# 2b. Multibranch CNN-BiLSTM (after Challa et al., "A multibranch CNN-BiLSTM model for human
#     activity recognition using wearable sensor data"), adapted to the RESpeck / app contract
# ======================================================================================
BILSTM_KERNELS = (3, 7, 11)


def build_cnn_bilstm(X_train: np.ndarray, cfg: L.TaskConfig, filters: int = 32,
                     lstm_units=(64, 32)) -> keras.Model:
    """Parallel conv branches with small / medium / large kernels (multi-scale local patterns),
    concatenated and fed to two stacked BiLSTMs (long-range temporal context).
    Adaptations: gravity removed for social (posture-invariant) / magnitude added for activity,
    normalisation inside the model (raw g in), pooling to <= 31 steps and unrolled LSTMs so the
    model converts to builtin TFLite ops."""
    inp = keras.Input(shape=(cfg.window, 3), name="window")
    ch = _channels_layer(cfg.name)
    sig = ch(inp)
    x = L._adapted_norm("normalize", L._eager(ch, X_train))(sig)
    branches = []
    for k in BILSTM_KERNELS:
        b = x
        for j in (1, 2):
            b = layers.Conv1D(filters, k, padding="same", use_bias=False, name=f"k{k}_conv{j}")(b)
            b = layers.BatchNormalization(name=f"k{k}_bn{j}")(b)
            b = layers.ReLU(name=f"k{k}_relu{j}")(b)
        branches.append(layers.MaxPooling1D(2, name=f"k{k}_pool")(b))
    x = layers.Concatenate(name="branches")(branches)
    if cfg.name == "social":                              # 125 -> 62 -> 31 time steps
        x = layers.MaxPooling1D(2, name="pool2")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units[0], return_sequences=True, unroll=True), name="bilstm1")(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units[1], unroll=True), name="bilstm2")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(cfg.classes), activation="softmax", name="probabilities")(x)
    prefix = "har" if cfg.name == "activity" else "social"
    return keras.Model(inp, out, name=f"{prefix}_cnn_bilstm")


# 2c. CNN-BiLSTM + LightGBM's hand-crafted features (`bilstm_feat`). build_cnn_bilstm above is
#     left untouched; this model repeats its trunk and adds a feature branch.
def _bilstm_trunk(x, cfg: L.TaskConfig, filters: int, lstm_units):
    """The multibranch conv -> 2x BiLSTM trunk of build_cnn_bilstm (up to the post-LSTM dropout)."""
    branches = []
    for k in BILSTM_KERNELS:
        b = x
        for j in (1, 2):
            b = layers.Conv1D(filters, k, padding="same", use_bias=False, name=f"k{k}_conv{j}")(b)
            b = layers.BatchNormalization(name=f"k{k}_bn{j}")(b)
            b = layers.ReLU(name=f"k{k}_relu{j}")(b)
        branches.append(layers.MaxPooling1D(2, name=f"k{k}_pool")(b))
    x = layers.Concatenate(name="branches")(branches)
    if cfg.name == "social":                              # 125 -> 62 -> 31 time steps
        x = layers.MaxPooling1D(2, name="pool2")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units[0], return_sequences=True, unroll=True), name="bilstm1")(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units[1], unroll=True), name="bilstm2")(x)
    return layers.Dropout(0.3)(x)


def build_cnn_bilstm_feat(X_train: np.ndarray, cfg: L.TaskConfig, filters: int = 32,
                          lstm_units=(64, 32)) -> keras.Model:
    """deep branch    : the CNN-BiLSTM trunk (as build_cnn_bilstm)
    feature branch : RichFeatures (the ~230 LightGBM features, computed in-model) -> Dense(64)
    fused -> Dense(64) -> softmax. Raw g in, softmax out; builtin TFLite ops only."""
    inp = keras.Input(shape=(cfg.window, 3), name="window")
    ch = _channels_layer(cfg.name)
    x = L._adapted_norm("normalize", L._eager(ch, X_train))(ch(inp))
    deep = _bilstm_trunk(x, cfg, filters, lstm_units)

    rf = RichFeatures(cfg.window, cfg.name, name="rich_features")
    feats = L._adapted_norm("normalize_features", L._eager(rf, X_train))(rf(inp))
    feats = layers.Dense(64, activation="relu", name="feat_dense")(feats)
    feats = layers.Dropout(0.3)(feats)

    h = layers.Concatenate(name="fuse")([deep, feats])
    h = layers.Dense(64, activation="relu", name="head_dense")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(len(cfg.classes), activation="softmax", name="probabilities")(h)
    prefix = "har" if cfg.name == "activity" else "social"
    return keras.Model(inp, out, name=f"{prefix}_cnn_bilstm_feat")


# ======================================================================================
# 3. Engines: fit / predict_proba / export with one interface
# ======================================================================================
def _channels_layer(task: str):
    return L.AddMagnitude(name="add_magnitude") if task == "activity" else L.DynamicChannels(name="remove_gravity")


class Engine:
    kind = "?"
    exportable = True

    def __init__(self, cfg: L.TaskConfig, args):
        self.cfg, self.args = cfg, args

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, epochs=None) -> int: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def export(self, out_dir: Path, name: str, offsets, extra_meta: dict, X_chk, y_chk) -> Path: ...


class HybridEngine(Engine):
    kind = "hybrid"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, epochs=None):
        self.model, best = L.train_model(self.cfg, X_tr, y_tr, X_va, y_va, epochs=epochs or self.args.epochs,
                                         batch_size=self.args.batch, seed=self.args.seed,
                                         verbose=getattr(self.args, "verbose", 0))
        return best

    def predict_proba(self, X):
        return self.model.predict(X, batch_size=1024, verbose=0)

    def export(self, out_dir, name, offsets, extra_meta, X_chk, y_chk):
        model = L.with_offsets(self.model, offsets) if offsets is not None else self.model
        model.save(Path(out_dir) / f"{name}.keras")
        path = L.convert_to_tflite(model, name, self.cfg, out_dir, quantize=self.args.quantize,
                                   extra_meta=extra_meta)
        L.verify_tflite(path, model, X_chk, y_chk, self.cfg, name)
        return path


class CNNBiLSTMEngine(HybridEngine):
    """Multibranch CNN-BiLSTM, trained with the hybrid's recipe (augmentation, early stopping,
    logit offsets); exported to TFLite the same way."""
    kind = "bilstm"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, epochs=None):
        self.model, best = L.train_model(self.cfg, X_tr, y_tr, X_va, y_va, epochs=epochs or self.args.epochs,
                                         batch_size=self.args.batch, seed=self.args.seed,
                                         verbose=getattr(self.args, "verbose", 0), builder=build_cnn_bilstm)
        return best


class CNNBiLSTMFeatEngine(HybridEngine):
    """CNN-BiLSTM + in-model LightGBM features; same training recipe and TFLite export."""
    kind = "bilstm_feat"

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, epochs=None):
        self.model, best = L.train_model(self.cfg, X_tr, y_tr, X_va, y_va, epochs=epochs or self.args.epochs,
                                         batch_size=self.args.batch, seed=self.args.seed,
                                         verbose=getattr(self.args, "verbose", 0), builder=build_cnn_bilstm_feat)
        return best

    def export(self, out_dir, name, offsets, extra_meta, X_chk, y_chk):
        model = L.with_offsets(self.model, offsets) if offsets is not None else self.model
        model.save(Path(out_dir) / f"{name}.keras")
        # The TFLite file gets a fixed batch of 1 (as the app calls it). With a dynamic batch, XNNPACK –
        # TFLite's default CPU delegate, also on Android – computes this graph wrongly (the plain
        # kernels are fine); with batch 1 it matches Keras.
        inp1 = keras.Input(batch_shape=(1, self.cfg.window, 3), name="window")
        lite = keras.Model(inp1, model(inp1), name=model.name)
        path = L.convert_to_tflite(lite, name, self.cfg, out_dir, quantize=self.args.quantize,
                                   extra_meta=extra_meta)
        L.verify_tflite(path, model, X_chk, y_chk, self.cfg, name)
        return path


class MiniRocketEngine(Engine):
    """MiniRocket features + softmax regression (the paper's logistic-regression variant for
    large training sets), trained with Adam and early stopping on the validation fold."""
    kind = "minirocket"

    def _features(self, X):
        out = []
        for i in range(0, len(X), 256):
            out.append(keras.ops.convert_to_numpy(self.mr(self.ch(X[i:i + 256]))))
        return np.concatenate(out)

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, epochs=None):
        L.set_seed(self.args.seed)
        self.ch = _channels_layer(self.cfg.name)
        sig = keras.ops.convert_to_numpy(self.ch(X_tr[:4096]))
        self.mr = fit_minirocket(sig, self.args.mr_features, self.args.seed)
        F_tr = self._features(X_tr)
        n_cls = len(self.cfg.classes)
        self.norm = layers.Normalization(name="normalize_features"); self.norm.adapt(F_tr)
        self.drop = layers.Dropout(0.2)
        self.dense = layers.Dense(n_cls, activation="softmax", name="probabilities",
                                  kernel_regularizer=keras.regularizers.l2(1e-4))
        fin = keras.Input(shape=(F_tr.shape[1],))
        head = keras.Model(fin, self.dense(self.drop(self.norm(fin))))
        head.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
        cw = L.balanced_weights(y_tr, n_cls, self.cfg.weight_power)
        val, cbs = None, []
        if X_va is not None and len(X_va):
            val = (self._features(X_va), y_va, cw[y_va])
            cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)]
        ep = epochs or self.args.epochs
        h = head.fit(F_tr, y_tr, sample_weight=cw[y_tr], batch_size=256, epochs=ep,
                     validation_data=val, callbacks=cbs, verbose=getattr(self.args, "verbose", 0), shuffle=True)
        best = int(np.argmin(h.history["val_loss"])) + 1 if val else ep
        self.head = head
        return best

    def full_model(self):
        inp = keras.Input(shape=(self.cfg.window, 3), name="window")
        out = self.dense(self.norm(self.mr(self.ch(inp))))           # dropout is a no-op at inference
        return keras.Model(inp, out, name="minirocket")

    def predict_proba(self, X):
        return self.head.predict(self._features(X), batch_size=2048, verbose=0)

    def export(self, out_dir, name, offsets, extra_meta, X_chk, y_chk):
        model = self.full_model()
        if offsets is not None:
            model = L.with_offsets(model, offsets)
        model.save(Path(out_dir) / f"{name}.keras")
        path = L.convert_to_tflite(model, name, self.cfg, out_dir, quantize=self.args.quantize,
                                   extra_meta=extra_meta)
        L.verify_tflite(path, model, X_chk, y_chk, self.cfg, name)
        return path


class GBMEngine(Engine):
    """LightGBM on rich_features(); early stopping on the validation fold (rounds = 'epochs')."""
    kind = "gbm"

    def params(self):
        return dict(objective="multiclass", learning_rate=0.05, num_leaves=31, min_child_samples=40,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                    n_jobs=-1, random_state=self.args.seed, verbose=-1)

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, epochs=None):
        import lightgbm as lgb
        F_tr, self.names = rich_features(X_tr, self.cfg.name)
        cw = L.balanced_weights(y_tr, len(self.cfg.classes), self.cfg.weight_power)
        kw, cbs = {}, []
        has_val = X_va is not None and len(X_va)
        n_rounds = epochs or (self.args.gbm_rounds if has_val else 400)
        if has_val:
            F_va, _ = rich_features(X_va, self.cfg.name)
            kw = dict(eval_set=[(F_va, y_va)], eval_sample_weight=[cw[y_va]])
            cbs = [lgb.early_stopping(50, verbose=False)]
        self.model = lgb.LGBMClassifier(n_estimators=n_rounds, **self.params())
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(F_tr, y_tr, sample_weight=cw[y_tr], callbacks=cbs, **kw)
        best = int(self.model.best_iteration_ or n_rounds) if cbs else n_rounds
        return best

    def predict_proba(self, X):
        F, _ = rich_features(X, self.cfg.name)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict_proba(F)

    def export(self, out_dir, name, offsets, extra_meta, X_chk, y_chk):
        out_dir = Path(out_dir)
        path = out_dir / f"{name}.txt"
        self.model.booster_.save_model(str(path))
        imp = self.model.booster_.feature_importance("gain")
        top = sorted(zip(self.names, imp.tolist()), key=lambda t: -t[1])[:30]
        (out_dir / f"{name}_importance.json").write_text(json.dumps(
            [{"feature": f, "gain": round(g, 1)} for f, g in top], indent=2))
        meta = L.base_meta(self.cfg, path.name, "lightgbm_" + self.cfg.name)
        meta.update({"backend": "lightgbm", "task_name": self.cfg.name, "features": "cw3models.rich_features",
                     "logit_offsets": None if offsets is None else [float(b) for b in offsets],
                     "normalization": "features computed in Python (cw3models.rich_features) from raw g"})
        meta.update(extra_meta)
        (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
        p = load_predictor(out_dir, name)
        agree = (p.predict(X_chk[:500]).argmax(1) ==
                 _offset_probs(self.predict_proba(X_chk[:500]), offsets).argmax(1)).mean()
        t0 = time.time(); [p.predict_one(w) for w in X_chk[:100]]
        print(f"   wrote {path.name} ({path.stat().st_size / 1024:.0f} kB), {name}_meta.json, {name}_importance.json; "
              f"reload check {agree:.1%}, {(time.time() - t0) * 10:.2f} ms/window incl. features")
        print("   note: LightGBM runs in predict.py / respeck_live.py; the Flutter app needs a TFLite model")
        return path


class RFEngine(Engine):
    kind = "rf"
    exportable = False

    def fit(self, X_tr, y_tr, X_va=None, y_va=None, epochs=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                            random_state=self.args.seed)
        self.model.fit(L.handcrafted_features(X_tr, self.cfg), y_tr)
        return 0

    def predict_proba(self, X):
        return self.model.predict_proba(L.handcrafted_features(X, self.cfg))


ENGINES = {"hybrid": HybridEngine, "bilstm": CNNBiLSTMEngine, "gbm": GBMEngine,
           "minirocket": MiniRocketEngine, "rf": RFEngine, "bilstm_feat": CNNBiLSTMFeatEngine}


# ======================================================================================
# 4. Loading any exported model (ours or someone else's Lab-3-format TFLite)
# ======================================================================================
def _offset_probs(p, offsets):
    if offsets is None:
        return p
    z = np.log(p + 1e-7) + np.asarray(offsets)
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


class Predictor:
    """meta-driven: window_samples, step_samples, output.classes, backend (tflite default)."""

    def __init__(self, models_dir: Path, name: str):
        models_dir = Path(models_dir)
        self.meta = json.loads((models_dir / f"{name}_meta.json").read_text())
        self.name = name
        self.W, self.S = int(self.meta["window_samples"]), int(self.meta["step_samples"])
        self.classes = list(self.meta["output"]["classes"])
        self.backend = self.meta.get("backend", "tflite")
        if self.backend == "tflite":
            self._m = L.TFLiteModel(models_dir / self.meta["model_file"])
        elif self.backend == "lightgbm":
            import lightgbm as lgb
            self._m = lgb.Booster(model_file=str(models_dir / self.meta["model_file"]))
            self.task = self.meta["task_name"]
            self.offsets = self.meta.get("logit_offsets")
        else:
            raise ValueError(f"unknown backend {self.backend}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        if self.backend == "tflite":
            return self._m.predict(X)
        p = self._m.predict(rich_features(X, self.task)[0])
        return _offset_probs(p, self.offsets)

    def predict_one(self, w):
        return self.predict(np.asarray(w, np.float32)[None])[0]


def load_predictor(models_dir, name) -> Predictor:
    return Predictor(models_dir, name)
