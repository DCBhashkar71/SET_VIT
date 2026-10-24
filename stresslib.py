"""
stresslib.py - single source of truth for the stress-classification study.

Design rules:
  * The independent unit is the SUBJECT (15 WESAD participants). Primary protocol = leave-one-subject-out
    (LOSO); every metric is computed per held-out subject and inference is done on the 15 per-subject values.
  * Every fitted quantity (scaler, class weights, Youden threshold, rule thresholds, stacking meta-features,
    imputation means) is fitted on the training subjects of the current fold only.
  * The cascade is defined with an unambiguous margin (units of training-fold SD of HR around the
    training-fold Youden threshold) and the full margin frontier is reported - nothing is "selected" on test data.
  * Primary feature set = 4 chest biometrics (hr, hrv, eda, temp_chest). Environmental features are only used as
    a NEGATIVE CONTROL (see env_negative_control in run_experiments.py).
"""
from __future__ import annotations

import copy
import json
import pickle
import platform
import sys
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

try:  # optional
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False

BIO = ["hr", "hrv", "eda", "temp_chest"]
ENV = ["co2_noisy", "lux_noisy", "posture_cm"]
WIN_S, STEP_S, FS = 30, 15, 700
# WESAD protocol labels: 0 undefined, 1 baseline, 2 stress (TSST), 3 amusement, 4 meditation, 5-7 transient.
# NORMAL = baseline + amusement; STRESS = TSST; meditation and undefined/transient segments are excluded.
LABEL_TO_Y = {1: 0, 3: 0, 2: 1}


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)


# ----------------------------------------------------------------------------------------------------------
# ECG -> HR / HRV   (one filter pass over the full recording, then per-window peak detection + RR cleaning)
# ----------------------------------------------------------------------------------------------------------
def bandpass(x, fs=FS, lo=5.0, hi=15.0, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def detect_r_peaks(filtered, fs=FS):
    """Peak detection on a band-passed ECG segment: squared signal, 80 ms moving average, adaptive height
    (25% of the 99th percentile), 300 ms refractory period, then refinement to the local |max| (+-60 ms)."""
    e = filtered ** 2
    w = max(1, int(0.08 * fs))
    s = np.convolve(e, np.ones(w) / w, mode="same")
    thr = 0.25 * np.percentile(s, 99)
    cand, _ = find_peaks(s, height=thr, distance=int(0.3 * fs))
    r = int(0.06 * fs)
    out = []
    for c in cand:
        lo, hi = max(0, c - r), min(len(filtered), c + r + 1)
        out.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
    return np.unique(np.asarray(out, dtype=int))


def hrv_features(peaks, fs=FS, min_valid_frac=0.8):
    """HR (from mean valid RR), RMSSD, SDNN, pNN50 with inter-beat correction.
    An RR interval is invalid if outside 300-2000 ms or >25% away from the local (5-beat) median.
    RMSSD/pNN50 use only successive differences whose two RR intervals are both valid, so one missed or spurious
    beat cannot create a huge difference. Returns dict with ok flag and a rejection reason."""
    res = {"ok": False, "reason": ""}
    if len(peaks) < 10:
        res["reason"] = "few_peaks"
        return res
    rr = np.diff(peaks) / fs * 1000.0
    valid = (rr >= 300) & (rr <= 2000)
    med = pd.Series(rr).rolling(5, center=True, min_periods=1).median().to_numpy()
    valid &= np.abs(rr - med) <= 0.25 * med
    frac = float(valid.mean())
    if frac < min_valid_frac:
        res["reason"] = "rr_artifact_fraction"
        return res
    pair = valid[:-1] & valid[1:]
    if pair.sum() < 10:
        res["reason"] = "few_valid_pairs"
        return res
    d = np.diff(rr)[pair]
    rrv = rr[valid]
    hr = 60000.0 / rrv.mean()
    if not (30 <= hr <= 220):
        res["reason"] = "hr_out_of_range"
        return res
    res.update(ok=True, hr=hr, hrv=float(np.sqrt(np.mean(d ** 2))), sdnn=float(rrv.std(ddof=1)),
               pnn50=float(100 * np.mean(np.abs(d) > 50)), rr_valid_frac=frac, n_beats=int(len(peaks)))
    return res


def validate_detector(det_peaks, ref_peaks, fs=FS, tol_ms=50):
    """Sensitivity / PPV / median timing error of detected vs reference R-peaks (use MIT-BIH annotations,
    manually annotated WESAD excerpts, or NeuroKit2 as an independent detector)."""
    det, ref = np.asarray(det_peaks), np.asarray(ref_peaks)
    tol = tol_ms / 1000 * fs
    used, tp, errs = set(), 0, []
    for r in ref:
        j = np.searchsorted(det, r)
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(det) and k not in used and abs(det[k] - r) <= tol:
                if best is None or abs(det[k] - r) < abs(det[best] - r):
                    best = k
        if best is not None:
            used.add(best); tp += 1; errs.append((det[best] - r) / fs * 1000)
    return {"sensitivity": tp / max(1, len(ref)), "ppv": tp / max(1, len(det)),
            "median_abs_err_ms": float(np.median(np.abs(errs))) if errs else np.nan, "n_ref": int(len(ref)),
            "n_det": int(len(det))}


def extract_subject(path, sid, win_s=WIN_S, step_s=STEP_S, fs=FS, purity=0.9):
    """Returns the COMPLETE window log of one subject (status in kept / qc_fail / mixed_label / unmapped_label)."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ecg = d["signal"]["chest"]["ECG"].flatten().astype(float)
    eda = d["signal"]["chest"]["EDA"].flatten()
    temp = d["signal"]["chest"]["Temp"].flatten()
    lab = np.asarray(d["label"]).flatten()
    n = min(len(ecg), len(lab))
    filt = bandpass(ecg[:n], fs)
    wn, st = int(win_s * fs), int(step_s * fs)
    rows = []
    for a in range(0, n - wn + 1, st):
        wl = lab[a:a + wn]
        vals, cnt = np.unique(wl, return_counts=True)
        dom, pur = int(vals[np.argmax(cnt)]), cnt.max() / len(wl)
        rec = {"subject": sid, "t_start": a / fs, "label_raw": dom, "purity": pur}
        if pur < purity:
            rec["status"] = "mixed_label"
        elif dom not in LABEL_TO_Y:
            rec["status"] = "unmapped_label"
        else:
            h = hrv_features(detect_r_peaks(filt[a:a + wn], fs), fs)
            if h["ok"]:
                rec.update(status="kept", y=LABEL_TO_Y[dom], hr=h["hr"], hrv=h["hrv"], sdnn=h["sdnn"],
                           pnn50=h["pnn50"], rr_valid_frac=h["rr_valid_frac"], eda=float(np.mean(eda[a:a + wn])),
                           temp_chest=float(np.mean(temp[a:a + wn])))
            else:
                rec.update(status="qc_fail", reason=h["reason"], y=LABEL_TO_Y[dom])
        rows.append(rec)
    return pd.DataFrame(rows)


def time_ecg_pipeline(ecg_windows, fs=FS, reps=1):
    """Measured cost (ms per 30 s window) of band-pass + peak detection + HRV features on THIS machine."""
    ts = []
    for w in ecg_windows:
        t0 = time.perf_counter()
        for _ in range(reps):
            hrv_features(detect_r_peaks(bandpass(w, fs), fs), fs)
        ts.append((time.perf_counter() - t0) / reps * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "n": len(ts)}


# ----------------------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------------------
def load_features(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"wrist_temp": "temp_chest"})  # the channel is the CHEST thermistor
    if "y" not in df:
        df["y"] = (df["ground_truth"] == "STRESSED").astype(int)
    if "t_start" not in df:
        warnings.warn("legacy CSV without t_start: reconstructing time from row order (15 s step). "
                      "Re-run extract_wesad.py for exact timing and the QC log.")
        df["t_start"] = df.groupby("subject").cumcount() * STEP_S
    return df.sort_values(["subject", "t_start"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------
# Models (all expose fit(X_df, y, groups) / score(X_df) / predict(X_df); all fit everything on train only)
# ----------------------------------------------------------------------------------------------------------
class YoudenRule:
    """Single-feature threshold chosen by Youden's J on the training subjects."""
    def __init__(self, col="hr"):
        self.col = col

    def fit(self, X, y, g=None):
        fpr, tpr, thr = roc_curve(y, X[self.col].to_numpy())
        self.t = float(thr[np.argmax(tpr - fpr)])
        self.sd = float(X[self.col].std())
        return self

    def score(self, X):
        return X[self.col].to_numpy()

    def predict(self, X):
        return (X[self.col].to_numpy() > self.t).astype(int)


class TunedRule:
    """HR>a AND HRV<b with a, b chosen on the training subjects (b=inf allowed => HR-only)."""
    def fit(self, X, y, g=None):
        hr, hrv = X["hr"].to_numpy(), X["hrv"].to_numpy()
        a_grid = np.quantile(hr, np.linspace(.05, .95, 19))
        b_grid = np.r_[np.quantile(hrv, np.linspace(.05, .95, 19)), np.inf]
        best = (-1, None, None)
        for a in a_grid:
            for b in b_grid:
                f = macro_f1(y, ((hr > a) & (hrv < b)).astype(int))
                if f > best[0]:
                    best = (f, a, b)
        _, self.a, self.b = best
        return self

    def score(self, X):
        return X["hr"].to_numpy()

    def predict(self, X):
        return ((X["hr"].to_numpy() > self.a) & (X["hrv"].to_numpy() < self.b)).astype(int)


class Skl:
    """sklearn estimator + scaler fitted on training data; balanced class handling for EVERY model
    (class_weight where available, balanced sample_weight for GB/XGB/HGB, random oversampling for MLP,
    which has neither class_weight nor - in older versions - sample_weight)."""
    def __init__(self, est, cols=BIO, sw=False, oversample=False, seed=0):
        self.est, self.cols, self.sw, self.oversample, self.seed = est, list(cols), sw, oversample, seed

    def _prep(self, X):
        return self.scaler.transform(X[self.cols].to_numpy())

    def fit(self, X, y, g=None):
        self.scaler = StandardScaler().fit(X[self.cols].to_numpy())
        Xs, y = self._prep(X), np.asarray(y)
        self.model = clone(self.est)
        if self.oversample:
            rng = np.random.default_rng(self.seed)
            idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
            small, big = (idx1, idx0) if len(idx1) < len(idx0) else (idx0, idx1)
            extra = rng.choice(small, len(big) - len(small), replace=True)
            idx = np.r_[np.arange(len(y)), extra]
            self.model.fit(Xs[idx], y[idx])
        elif self.sw:
            self.model.fit(Xs, y, sample_weight=compute_sample_weight("balanced", y))
        else:
            self.model.fit(Xs, y)
        return self

    def score(self, X):
        Xs = self._prep(X)
        return self.model.predict_proba(Xs)[:, 1] if hasattr(self.model, "predict_proba") else self.model.decision_function(Xs)

    def predict(self, X):
        return self.model.predict(self._prep(X)).astype(int)


class Stacking:
    """Leakage-safe stacking: out-of-fold base predictions come from an INNER SUBJECT-GROUPED CV on the training
    subjects; meta-learner = balanced logistic regression on [OOF probabilities (+ optionally scaled features)]."""
    def __init__(self, bases, inner_splits=3, passthrough=True, seed=0, cols=BIO):
        self.bases, self.inner_splits, self.passthrough, self.seed, self.cols = bases, inner_splits, passthrough, seed, cols

    def _meta_X(self, base_scores, X):
        return np.hstack([base_scores, self.scaler.transform(X[self.cols].to_numpy())]) if self.passthrough else base_scores

    def fit(self, X, y, g):
        y, g = np.asarray(y), np.asarray(g)
        self.scaler = StandardScaler().fit(X[self.cols].to_numpy())
        oof = np.zeros((len(y), len(self.bases)))
        cv = StratifiedGroupKFold(self.inner_splits, shuffle=True, random_state=self.seed)
        for tr, va in cv.split(X, y, g):
            for i, b in enumerate(self.bases.values()):
                m = copy.deepcopy(b).fit(X.iloc[tr], y[tr])
                oof[va, i] = m.score(X.iloc[va])
        self.fitted = [copy.deepcopy(b).fit(X, y) for b in self.bases.values()]
        self.meta = LogisticRegression(max_iter=3000, class_weight="balanced").fit(self._meta_X(oof, X), y)
        return self

    def _base_scores(self, X):
        return np.column_stack([m.score(X) for m in self.fitted])

    def score(self, X):
        return self.meta.predict_proba(self._meta_X(self._base_scores(X), X))[:, 1]

    def predict(self, X):
        return self.meta.predict(self._meta_X(self._base_scores(X), X)).astype(int)


def make_models(seed=0, quick=False, n_jobs=-1):
    n = 100 if quick else 200
    lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
    rf = lambda: RandomForestClassifier(n, class_weight="balanced", random_state=seed, n_jobs=n_jobs)
    gb = lambda: GradientBoostingClassifier(random_state=seed)
    xg = (lambda: XGBClassifier(n_estimators=n, max_depth=4, eval_metric="logloss", random_state=seed)) if HAVE_XGB \
        else (lambda: HistGradientBoostingClassifier(max_depth=4, random_state=seed))
    bases = lambda: {"rf": Skl(rf()), "xgb": Skl(xg(), sw=True), "gb": Skl(gb(), sw=True), "lr": Skl(lr())}
    return {
        "Youden_HR": YoudenRule("hr"),
        "TunedRule_HR_HRV": TunedRule(),
        "LR_HR": Skl(lr(), cols=["hr"]),
        "LR_HR_EDA": Skl(lr(), cols=["hr", "eda"]),
        "LR_bio4": Skl(lr()),
        "SVM": Skl(SVC(class_weight="balanced")),
        "RF": Skl(rf()),
        "GB": Skl(gb(), sw=True),
        "XGB": Skl(xg(), sw=True),
        "MLP": Skl(MLPClassifier((64, 32), max_iter=500, random_state=seed), oversample=True, seed=seed),
        "Stacking": Stacking(bases(), passthrough=True, seed=seed),
        "Stacking_nopass": Stacking(bases(), passthrough=False, seed=seed),
    }


# ----------------------------------------------------------------------------------------------------------
# Metrics and subject-level inference
# ----------------------------------------------------------------------------------------------------------
def subj_metrics(y, score, pred):
    y = np.asarray(y)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    d = dict(macro_f1=macro_f1(y, pred), bal_acc=balanced_accuracy_score(y, pred), acc=accuracy_score(y, pred),
             f1_stress=f[1], prec_stress=p[1], rec_stress=r[1], f1_normal=f[0])
    if len(np.unique(y)) == 2:
        d["auroc"] = roc_auc_score(y, score)
        d["auprc"] = average_precision_score(y, score)
    return d


def boot_ci(x, B=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def signflip_p(d, n_mc=200000, seed=0):
    """Two-sided paired sign-flip permutation test on the mean of per-subject differences.
    Exact enumeration for n <= 20 subjects; Monte Carlo (n_mc random sign patterns, +1 correction) for larger n."""
    d = np.asarray(d, float)
    n = len(d)
    if np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    if n <= 20:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_mc // 20000):
        s = rng.choice([-1.0, 1.0], size=(20000, n))
        hits += int((np.abs((s * d).mean(1)) >= obs - 1e-12).sum())
    m = (n_mc // 20000) * 20000
    return float((hits + 1) / (m + 1))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired_table(wide, metric_name, delta=0.05, pairs=None, B=10000):
    """wide: subjects x models DataFrame of one metric. Returns mean diff, bootstrap CI over subjects, dz,
    exact sign-flip p, Holm-adjusted p (family = all rows) and whether the CI lies inside +-delta."""
    cols = list(wide.columns)
    pairs = pairs or [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    rows = []
    for a, b in pairs:
        d = (wide[a] - wide[b]).dropna().to_numpy()
        lo, hi = boot_ci(d, B)
        rows.append(dict(metric=metric_name, A=a, B=b, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                         dz=d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan, p_signflip=signflip_p(d),
                         within_pm_delta=bool(lo > -delta and hi < delta), n_subjects=len(d)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p_signflip"].to_numpy())
    return t


def alarm_burden(g_sorted, pred, step_s=STEP_S):
    """Consecutive positive windows (no gap in time) merge into ONE alarm event. A false alarm event contains no
    true-stress window. Rate = false-alarm events per hour of NORMAL-labelled recording time."""
    t, y, p = g_sorted["t_start"].to_numpy(), g_sorted["y"].to_numpy(), np.asarray(pred)
    events, cur_hit, prev_t = [], None, None
    for ti, yi, pi in zip(t, y, p):
        contiguous = prev_t is not None and abs(ti - prev_t - step_s) < 1e-6
        if pi:
            if cur_hit is None or not contiguous:
                if cur_hit is not None:
                    events.append(cur_hit)
                cur_hit = False
            cur_hit = cur_hit or bool(yi)
        else:
            if cur_hit is not None:
                events.append(cur_hit); cur_hit = None
        prev_t = ti
    if cur_hit is not None:
        events.append(cur_hit)
    hours = (y == 0).sum() * step_s / 3600
    return dict(events=len(events), false_events=int(sum(1 for e in events if not e)),
                fa_per_hour=sum(1 for e in events if not e) / hours if hours else np.nan)


# ----------------------------------------------------------------------------------------------------------
# Cost profiling (desktop profiling only - NOT a microcontroller measurement)
# ----------------------------------------------------------------------------------------------------------
def profile_model(model, Xone, reps=300, warm=30):
    for _ in range(warm):
        model.predict(Xone)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        model.predict(Xone)
        ts.append((time.perf_counter_ns() - t0) / 1000)
    tracemalloc.start()
    model.predict(Xone)
    peak = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    return dict(latency_us_median=float(np.median(ts)), latency_us_q25=float(np.quantile(ts, .25)),
                latency_us_q75=float(np.quantile(ts, .75)), serialized_kb=len(pickle.dumps(model)) / 1024,
                py_peak_alloc_kb=peak, reps=reps, batch_size=1)


def environment_manifest():
    import sklearn, scipy
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                xgboost_available=HAVE_XGB)
