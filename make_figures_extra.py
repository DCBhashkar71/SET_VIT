"""
make_figures_extra.py - figures for the additional experiments (reads outputs of extra_experiments.py).
    python make_figures_extra.py --extra outputs_extra --features outputs_revised\\wesad_features.csv
FigA decomposition (resting-HR spread vs stress shift; pooled vs within-subject AUROC) | FigB calibration curve |
FigC learning curve | FigD smoothing trade-off | FigE HR-HRV scatter by class | FigF per-subject AUROC vs macro-F1
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--extra", default="outputs_extra")
ap.add_argument("--features", default="outputs_revised/wesad_features.csv"); ap.add_argument("--main", default="outputs_revised")
a = ap.parse_args(); o = a.extra
def ci(v):
    v = np.asarray(v, float); b = v[np.random.default_rng(0).integers(0, len(v), (5000, len(v)))].mean(1)
    return v.mean(), np.quantile(b, .025), np.quantile(b, .975)

# A
sub = pd.read_csv(f"{o}/e1_per_subject.csv").sort_values("hr_base_mean"); summ = json.load(open(f"{o}/e1_summary.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
y = np.arange(len(sub))
ax[0].hlines(y, sub.hr_base_mean, sub.hr_stress_mean, color="#999")
ax[0].scatter(sub.hr_base_mean, y, label="rest (normal windows)", color="#4C72B0"); ax[0].scatter(sub.hr_stress_mean, y, label="stress", color="#C44E52")
ax[0].set_yticks(y); ax[0].set_yticklabels(sub.subject.astype(int)); ax[0].set_xlabel("Mean heart rate (bpm)"); ax[0].set_ylabel("Subject"); ax[0].legend(fontsize=8)
feats = [c for c in ("hr", "eda") if c in summ]; w = .27
for j, (k, lab) in enumerate((("auc_pooled", "pooled"), ("auc_within_subject_mean", "within-subject (mean)"), ("auc_pooled_after_oracle_centering", "pooled after per-subject centring"))):
    ax[1].bar(np.arange(len(feats)) + j * w, [summ[f][k] for f in feats], w, label=lab)
ax[1].set_xticks(np.arange(len(feats)) + w); ax[1].set_xticklabels([f.upper() for f in feats]); ax[1].set_ylim(.4, 1); ax[1].set_ylabel("AUROC (oriented)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{o}/figA_baseline_shift.png", dpi=300); plt.close()

# B
r = pd.read_csv(f"{o}/e2_calibration_per_subject.csv")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for m, col in (("LR", "#4C72B0"), ("GB", "#55A868"), ("Youden_HR", "#C44E52")):
    for var, ls in (("raw", "--"), ("centred", "-")):
        xs, ys, lo, hi = [], [], [], []
        for n, g in r[(r.model == m) & (r.variant == var) & (r.n_cal > 0)].groupby("n_cal"):
            mu, l, h = ci(g.macro_f1); xs.append(n); ys.append(mu); lo.append(l); hi.append(h)
        ax.plot(xs, ys, ls, marker="o", color=col, label=f"{m} {var}"); 
        if var == "centred": ax.fill_between(xs, lo, hi, color=col, alpha=.12)
ax.set_xlabel("Calibration windows (resting, 15 s step)"); ax.set_ylabel("Macro-F1 on non-overlapping later windows"); ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(f"{o}/figB_calibration.png", dpi=300); plt.close()

# C
lc = pd.read_csv(f"{o}/e6_learning_curve.csv")
fig, ax = plt.subplots(figsize=(5.5, 4))
for m, g in lc.groupby("model"):
    ax.plot(g.n_train, g.macro_f1, marker="o", label=m); ax.fill_between(g.n_train, g.ci_lo, g.ci_hi, alpha=.12)
ax.set_xlabel("Number of training subjects"); ax.set_ylabel("Macro-F1 on held-out subject"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{o}/figC_learning_curve.png", dpi=300); plt.close()

# D
s = pd.read_csv(f"{o}/e5_smoothing_summary.csv")
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
for m, g in s.groupby("model"):
    ax[0].plot(g.fa_per_hour, g.macro_f1, marker="o", label=m)
    for _, rr in g.iterrows(): ax[0].annotate(f"k={int(rr.k)}", (rr.fa_per_hour, rr.macro_f1), fontsize=6)
    ax[1].plot(g.k, g.median_delay_s, marker="o", label=m)
ax[0].set_xlabel("False-alarm events per hour of normal time"); ax[0].set_ylabel("Macro-F1"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("Causal smoothing length k (windows)"); ax[1].set_ylabel("Median detection delay (s)")
plt.tight_layout(); plt.savefig(f"{o}/figD_smoothing.png", dpi=300); plt.close()

# E
if os.path.exists(a.features) and "hrv" in pd.read_csv(a.features, nrows=1):
    f = pd.read_csv(a.features); fig, ax = plt.subplots(figsize=(5, 4.2))
    for yv, c, l in ((0, "#4C72B0", "normal"), (1, "#C44E52", "stress")):
        g = f[f.y == yv]; ax.scatter(g.hr, g.hrv, s=5, alpha=.3, color=c, label=l)
    ax.set_xlabel("Heart rate (bpm)"); ax.set_ylabel("RMSSD (ms, 30-s window)"); ax.legend(markerscale=3)
    plt.tight_layout(); plt.savefig(f"{o}/figE_hr_hrv_scatter.png", dpi=300); plt.close()

# F
pm = f"{a.main}/loso_per_subject.csv"
if os.path.exists(pm):
    p = pd.read_csv(pm); p = p[p.model == "Stacking"]
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(p.auroc, p.macro_f1)
    for _, rr in p.iterrows(): ax.annotate(int(rr.subject), (rr.auroc, rr.macro_f1), fontsize=7)
    ax.set_xlabel("Within-subject AUROC (stacking, LOSO)"); ax.set_ylabel("Macro-F1 at the population threshold")
    plt.tight_layout(); plt.savefig(f"{o}/figF_auroc_vs_f1.png", dpi=300); plt.close()
print("extra figures written to", o)
