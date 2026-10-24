"""
make_figures.py - main figures, drawn from the CSVs written by run_experiments.py.
    python make_figures.py --out outputs_revised
Fig2 per-subject dot plot (no fold-SD error bars) | Fig3 pooled confusion matrix of a chosen model, with the
counts-are-pooled caveat | Fig4 cascade frontier with CI band | Fig5 test-time dropout (positive = degradation) |
Fig6 pooled ROC/PR | Fig7 environment negative-control null distribution.
Axis labels are human-readable; all CIs are bootstrap over SUBJECTS.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

NICE = {"Youden_HR": "Youden HR rule", "TunedRule_HR_HRV": "Tuned HR/HRV rule", "LR_HR": "LR (HR)", "LR_HR_EDA": "LR (HR+EDA)",
        "LR_bio4": "LR (4 biometrics)", "SVM": "SVM", "RF": "Random forest", "GB": "Gradient boosting", "XGB": "XGBoost",
        "MLP": "MLP", "Stacking": "Stacking ensemble", "Stacking_nopass": "Stacking (no passthrough)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs_revised"); ap.add_argument("--cm-model", default="Stacking")
    a = ap.parse_args(); o = a.out
    ps = pd.read_csv(f"{o}/loso_per_subject.csv"); summ = pd.read_csv(f"{o}/loso_summary.csv")
    order = summ.sort_values("macro_f1_mean")["model"].tolist()
    # Fig 2
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, m in enumerate(order):
        v = ps[ps.model == m].macro_f1.to_numpy()
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(0).uniform(-.18, .18, len(v)), s=14, alpha=.6, color="#4C72B0")
        r = summ[summ.model == m].iloc[0]
        ax.errorbar(r.macro_f1_mean, i, xerr=[[r.macro_f1_mean - r.macro_f1_ci_lo], [r.macro_f1_ci_hi - r.macro_f1_mean]], fmt="D", color="k", capsize=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE.get(m, m) for m in order])
    ax.set_xlabel("Macro-F1 per held-out subject (dots)\nmean with 95% subject-bootstrap CI (diamond)")
    plt.tight_layout(); plt.savefig(f"{o}/fig2_per_subject.png", dpi=300); plt.close()
    # Fig 3, 6
    oof = pd.read_csv(f"{o}/oof_predictions.csv"); m = a.cm_model
    cm = confusion_matrix(oof.y, oof[f"pred_{m}"].astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Stress"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Stress"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{NICE.get(m, m)}\nLOSO out-of-sample windows (n={len(oof)}, each predicted once)", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{o}/fig3_confusion.png", dpi=300); plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for mm in ["Youden_HR", "LR_HR_EDA", "RF", "Stacking"]:
        f, t, _ = roc_curve(oof.y, oof[f"score_{mm}"]); axs[0].plot(f, t, label=NICE[mm])
        p, r, _ = precision_recall_curve(oof.y, oof[f"score_{mm}"]); axs[1].plot(r, p, label=NICE[mm])
    axs[0].plot([0, 1], [0, 1], "k:"); axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{o}/fig6_roc_pr.png", dpi=300); plt.close()
    # Fig 4
    c = pd.read_csv(f"{o}/cascade_frontier.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for s2, g in c.groupby("stage2"):
        axs[0].plot(g.invoked_mean * 100, g.macro_f1, marker="o", label=f"Cascade -> {NICE.get(s2, s2)}")
        axs[0].fill_between(g.invoked_mean * 100, g.ci_lo, g.ci_hi, alpha=.12)
        axs[0].plot(g.invoked_mean * 100, g.f1_random_routing, ls="--", alpha=.6)
    g0 = c.iloc[0]; axs[0].axhline(g0.f1_stage1_only, color="gray", ls=":", label="Stage 1 only (Youden HR)")
    axs[0].set_xlabel("Windows sent to stage 2 (%)"); axs[0].set_ylabel("Macro-F1 (mean over subjects, 95% CI)"); axs[0].legend(fontsize=7)
    axs[0].set_title("dashed = random routing at the same rate", fontsize=8)
    cp = f"{o}/cost_profile_single_thread.csv"
    if os.path.exists(cp):
        lat = pd.read_csv(cp).set_index("model").latency_us_median
        for s2, g in c.groupby("stage2"):
            ms = (lat["Youden_HR"] + g.invoked_mean * lat[s2]) / 1000
            axs[1].plot(ms, g.macro_f1, marker="o", label=NICE.get(s2, s2))
        axs[1].set_xscale("log"); axs[1].legend(fontsize=7)
    axs[1].set_xlabel("Expected model latency per window\n(ms, single-thread desktop CPU, batch=1)")
    axs[1].set_ylabel("Macro-F1")
    plt.tight_layout(); plt.savefig(f"{o}/fig4_cascade_frontier.png", dpi=300); plt.close()
    # Fig 5
    d = pd.read_csv(f"{o}/dropout.csv"); d["delta"] = d.f1_full - d.f1_dropped
    fig, ax = plt.subplots(figsize=(7, 4)); models = d.model.unique(); w = .8 / len(models)
    drops = d.dropped.unique()
    for k, mm in enumerate(models):
        means, los, his = [], [], []
        for dr in drops:
            v = d[(d.model == mm) & (d.dropped == dr)].delta.to_numpy(); rng = np.random.default_rng(0)
            b = v[rng.integers(0, len(v), (5000, len(v)))].mean(1); means.append(v.mean()); los.append(np.quantile(b, .025)); his.append(np.quantile(b, .975))
        x = np.arange(len(drops)) + k * w
        ax.bar(x, means, w, yerr=[np.array(means) - los, np.array(his) - means], label=NICE.get(mm, mm), capsize=2)
    ax.set_xticks(np.arange(len(drops)) + .4 - w / 2); ax.set_xticklabels(drops, rotation=20); ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("Macro-F1 loss when sensor fails\n(positive = degradation)"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{o}/fig5_dropout.png", dpi=300); plt.close()
    # Fig 7
    ep = f"{o}/env_negative_control.csv"
    if os.path.exists(ep):
        e = pd.read_csv(ep); nul = e[e["mode"] == "permuted"].delta_fused_minus_bio; seq = e[e["mode"] == "sequential"].delta_fused_minus_bio.iloc[0]
        fig, ax = plt.subplots(figsize=(5.5, 3.5)); ax.hist(nul, bins=12, color="#999"); ax.axvline(seq, color="r", label="sequential alignment")
        ax.set_xlabel("Macro-F1(bio+env) - macro-F1(bio), random alignments"); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{o}/fig7_env_negative_control.png", dpi=300); plt.close()
    print("figures written to", o)


if __name__ == "__main__":
    main()
