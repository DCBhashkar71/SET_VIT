"""
make_manuscript_figures.py - regenerates five figures of the manuscript from the result files you already have.

    python make_manuscript_figures.py

Default folders (change with the options if yours differ):
    --revised  outputs_revised     needs qc_summary.csv, pairwise_tests.csv, loso_summary.csv, cross_dataset_revised.csv
    --sp       outputs_sp          needs stress_predict_qc.csv
    --sp-extra outputs_sp_extra    needs e1_per_subject.csv, e2_calibration_per_subject.csv   (Stress-Predict run of extra_experiments.py)
    --pooled   outputs_pooled      needs pooled_calibration.csv                               (output of combine_datasets.py)
    --out      figs_manuscript     the five PNG files are written here

Figures written:
    fig_qc_windows.png            windows kept per participant (WESAD, Stress-Predict)
    fig_pairwise_heatmap.png      paired macro-F1 differences between models
    figSP_baseline_calibration.png  Stress-Predict baseline shift and calibration curves
    fig_pooled_calibration.png    calibration gain per dataset and pooled
    fig_cross_dataset.png         single-feature and transfer AUROC (WESAD <-> SWELL-KW)
Use --out outputs_revised if you want the images saved in that folder.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NICE = {"LR": "LR", "Youden_HR": "Youden HR", "TunedRule_HR_HRV": "Tuned HR/HRV", "LR_HR": "LR (HR)", "LR_HR_EDA": "LR (HR+EDA)",
        "LR_bio4": "LR (4 biometrics)", "SVM": "SVM", "RF": "RF", "GB": "GB", "XGB": "XGBoost", "MLP": "MLP",
        "Stacking": "Stacking", "Stacking_nopass": "Stacking (no pass.)"}


def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing input file: {path}\nCheck the folder option or run the step that creates it.")
    return pd.read_csv(path)


def boot_ci(v, B=5000, seed=0):
    v = np.asarray(v, float)
    b = v[np.random.default_rng(seed).integers(0, len(v), (B, len(v)))].mean(1)
    return v.mean(), np.quantile(b, .025), np.quantile(b, .975)


def fig_qc(a, out):
    q = need(os.path.join(a.revised, "qc_summary.csv"))
    sp = need(os.path.join(a.sp, "stress_predict_qc.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    p = q.pivot(index="subject", columns="class", values="kept")
    x = p.index.astype(int).astype(str)
    ax[0].bar(x, p["NORMAL"], color="#4C72B0", label="normal (kept)")
    ax[0].bar(x, p["STRESS"], bottom=p["NORMAL"], color="#C44E52", label="stress (kept)")
    rej = q.groupby("subject").rejected.sum()
    for i, s in enumerate(p.index):
        if rej[s] > 0:
            ax[0].text(i, p.loc[s].sum() + 2, f"-{int(rej[s])}", ha="center", fontsize=8)
    ax[0].set_title("WESAD (15 participants)", fontsize=10); ax[0].set_xlabel("Participant"); ax[0].set_ylabel("Windows (30 s, 15 s step)"); ax[0].legend(fontsize=7)
    rest = sp.windows_kept - sp.stress_windows
    ax[1].bar(sp.subject.astype(str), rest, color="#4C72B0", label="rest (kept)")
    ax[1].bar(sp.subject.astype(str), sp.stress_windows, bottom=rest, color="#C44E52", label="stress (kept)")
    ax[1].set_title("Stress-Predict (34 participants)", fontsize=10); ax[1].set_xlabel("Participant"); ax[1].set_ylabel("Windows (30 s, 15 s step)")
    ax[1].tick_params(axis="x", labelsize=6, rotation=90); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_qc_windows.png"), dpi=300); plt.close()


def fig_heatmap(a, out):
    p = need(os.path.join(a.revised, "pairwise_tests.csv")); p = p[p.metric == "macro_f1"]
    s = need(os.path.join(a.revised, "loso_summary.csv")).sort_values("macro_f1_mean", ascending=False)
    order = s.model.tolist(); n = len(order)
    M = np.full((n, n), np.nan); H = np.full((n, n), np.nan)
    for _, r in p.iterrows():
        i, j = order.index(r.A), order.index(r.B)
        M[i, j] = r.mean_diff; M[j, i] = -r.mean_diff; H[i, j] = H[j, i] = r.p_holm
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5, fontweight="bold" if H[i, j] < 0.05 else None)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([NICE.get(m, m) for m in order], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([NICE.get(m, m) for m in order], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046).set_label("Macro-F1 difference (row minus column)")
    n_sig = int((p.p_holm < 0.05).sum())
    ax.set_title(f"Paired per-participant differences ({n_sig} significant after Holm correction; bold = significant)", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_pairwise_heatmap.png"), dpi=300); plt.close()


def fig_sp(a, out):
    e1 = need(os.path.join(a.sp_extra, "e1_per_subject.csv")).sort_values("hr_base_mean")
    r = need(os.path.join(a.sp_extra, "e2_calibration_per_subject.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6)); y = np.arange(len(e1))
    ax[0].hlines(y, e1.hr_base_mean, e1.hr_stress_mean, color="#999")
    ax[0].scatter(e1.hr_base_mean, y, s=14, color="#4C72B0", label="rest")
    ax[0].scatter(e1.hr_stress_mean, y, s=14, color="#C44E52", label="stress tasks")
    ax[0].set_yticks(y[::3]); ax[0].set_yticklabels(e1.subject.astype(int).values[::3], fontsize=7)
    ax[0].set_xlabel("Mean wrist heart rate (bpm)"); ax[0].set_ylabel("Participant"); ax[0].legend(fontsize=8)
    for m, c in (("LR", "#4C72B0"), ("Youden_HR", "#C44E52")):
        for var, ls in (("raw", "--"), ("centred", "-")):
            xs, ys, lo, hi = [], [], [], []
            for k, g in r[(r.model == m) & (r.variant == var) & (r.n_cal > 0)].groupby("n_cal"):
                mu, l, h = boot_ci(g.macro_f1); xs.append(k); ys.append(mu); lo.append(l); hi.append(h)
            ax[1].plot(xs, ys, ls, marker="o", color=c, label=f"{NICE[m]} {var}")
            if var == "centred":
                ax[1].fill_between(xs, lo, hi, color=c, alpha=.12)
    ax[1].set_xlabel("Rest windows used for calibration (15 s step)"); ax[1].set_ylabel("Macro-F1 on later non-overlapping windows"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out, "figSP_baseline_calibration.png"), dpi=300); plt.close()


def fig_pooled(a, out):
    pc = need(os.path.join(a.pooled, "pooled_calibration.csv"))
    pc["lab"] = pc.model.map(NICE) + " / " + pc.n_cal.astype(int).astype(str) + " windows"
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); yy = np.arange(len(pc))[::-1]
    for col, c, dy, lab in (("WESAD", "#4C72B0", 0.22, "WESAD (n=15)"), ("StressPredict", "#55A868", 0.0, "Stress-Predict (n=34)"), ("pooled", "#C44E52", -0.22, "pooled (per participant)")):
        if col == "pooled":
            m, lo, hi = pc.pooled_subject_weighted, pc.pooled_sw_lo, pc.pooled_sw_hi
        else:
            m, lo, hi = pc[f"{col}_mean"], pc[f"{col}_lo"], pc[f"{col}_hi"]
        ax.errorbar(m, yy + dy, xerr=[m - lo, hi - m], fmt="o", ms=4, color=c, capsize=2, label=lab)
    ax.axvline(0, color="k", lw=.8); ax.set_yticks(yy); ax.set_yticklabels(pc.lab, fontsize=8)
    ax.set_xlabel("Macro-F1 gain from baseline centring (95% participant-bootstrap CI)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_pooled_calibration.png"), dpi=300, bbox_inches="tight"); plt.close()


def fig_cross(a, out):
    x = need(os.path.join(a.revised, "cross_dataset_revised.csv"))
    sf = x[x.kind == "single_feature_AUC"]
    tr = x[(x.kind == "transfer") & (x.scaling == "source_only") & (x.features == "HR+RMSSD") & (x.scheme.str.startswith("TP_vs_NoStress"))]
    rows = [("WESAD 30 s", "WESAD_30s"), ("WESAD 60 s", "WESAD_60s"), ("WESAD 300 s", "WESAD_300s"), ("SWELL-KW (TP vs no stress)", "SWELL")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9)); w = 0.35
    for k, (feat, c) in enumerate((("HR", "#C44E52"), ("RMSSD", "#4C72B0"))):
        vals = []
        for lab, ds in rows:
            g = sf[(sf.dataset == ds) & (sf.feature == feat)]
            if ds == "SWELL":
                g = g[g.scheme.str.startswith("TP_vs_NoStress")]
            vals.append(g.auc.iloc[0])
        ax[0].bar(np.arange(len(rows)) + k * w, vals, w, color=c, label=feat)
    ax[0].axhline(0.5, color="k", lw=.8, ls="--"); ax[0].set_xticks(np.arange(len(rows)) + w / 2)
    ax[0].set_xticklabels([r[0] for r in rows], rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("AUROC of the raw feature\n(stress = positive)"); ax[0].legend(fontsize=8)
    for d, c in (("WESAD->SWELL", "#C44E52"), ("SWELL->WESAD", "#4C72B0")):
        g = tr[tr.direction == d].sort_values("wesad_window_s"); ax[1].plot(g.wesad_window_s, g.auc, "o-", color=c, label=d)
        if g.ci_lo.notna().any():
            ax[1].fill_between(g.wesad_window_s, g.ci_lo, g.ci_hi, color=c, alpha=.15)
    ax[1].axhline(0.5, color="k", lw=.8, ls="--"); ax[1].set_xscale("log"); ax[1].set_xticks([30, 60, 300]); ax[1].set_xticklabels(["30", "60", "300"])
    ax[1].set_xlabel("WESAD window length (s)"); ax[1].set_ylabel("Transfer AUROC\n(source-only scaling)"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_cross_dataset.png"), dpi=300); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revised", default="outputs_revised"); ap.add_argument("--sp", default="outputs_sp")
    ap.add_argument("--sp-extra", dest="sp_extra", default="outputs_sp_extra"); ap.add_argument("--pooled", default="outputs_pooled")
    ap.add_argument("--out", default="figs_manuscript")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for f in (fig_qc, fig_heatmap, fig_sp, fig_pooled, fig_cross):
        f(a, a.out); print("written:", f.__name__)
    print("done ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
