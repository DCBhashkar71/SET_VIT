"""
make_figures_final.py - schematic and scatter figures.

    python make_figures_final.py --features outputs_revised\\wesad_features.csv ^
        --oof outputs_revised\\oof_predictions.csv --sp outputs_sp\\stress_predict_features.csv --out figs_final

fig1_pipeline.png          study flowchart (real data streams, LOSO nesting, inner grouped CV for stacking, negative control kept apart)
fig_hr_hrv_margin.png      WESAD HR vs RMSSD by class with the cascade stage-1 threshold and margin bands (+-0.5, +-0.75 SD of HR)
fig_sp_hr_eda.png          Stress-Predict HR vs EDA by class (no HRV in that dataset)
fig_env_alignment.png      schematic of the environmental alignments: sequential vs random alignment
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, text, fc="#E8EEF7", ec="#33507A", ls="-", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, wrap=True)


def arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333", ls=ls))


def fig1(out):
    fig, ax = plt.subplots(figsize=(10, 6.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .02, .78, .22, .16, "WESAD (lab TSST)\nchest ECG 700 Hz, EDA, temp\n15 participants", bold=True)
    box(ax, .02, .52, .22, .16, "Stress-Predict (replication)\nwrist HR 1 Hz, EDA, temp\n34 participants", bold=True)
    box(ax, .30, .78, .20, .16, "30 s windows, 15 s step\nlabel purity >= 0.9\nNORMAL = baseline+amusement\nSTRESS = TSST")
    box(ax, .30, .52, .20, .16, "30 s windows, 15 s step\nrest/relax = 0, Stroop,\ninterview, hyperventilation = 1")
    box(ax, .56, .78, .20, .16, "ECG band-pass, R-peaks,\nRR cleaning -> HR, RMSSD\n+ EDA mean, chest temp\n(QC log per window)")
    box(ax, .56, .52, .20, .16, "HR (Empatica), EDA mean,\nwrist temp mean")
    arrow(ax, .24, .86, .30, .86); arrow(ax, .50, .86, .56, .86); arrow(ax, .24, .60, .30, .60); arrow(ax, .50, .60, .56, .60)
    box(ax, .02, .20, .74, .24, "", fc="#FFF8E5", ec="#B08900")
    ax.text(.03, .415, "Leave-one-subject-out (outer loop): every fitted quantity uses the training subjects only", fontsize=9, fontweight="bold")
    box(ax, .04, .24, .22, .13, "Training subjects:\nscaler, class weights,\nYouden / rule thresholds", fc="#FFFFFF")
    box(ax, .29, .24, .22, .13, "Stacking: inner subject-grouped\nCV -> out-of-fold base\npredictions -> meta-learner", fc="#FFFFFF")
    box(ax, .54, .24, .20, .13, "Held-out subject:\npredict -> per-subject\nmetrics (macro-F1, AUROC)", fc="#FFFFFF")
    arrow(ax, .26, .305, .29, .305); arrow(ax, .51, .305, .54, .305)
    for yy in (.86, .60):                       # both feature tables feed the LOSO block (elbow connector)
        ax.plot([.76, .785], [yy, yy], color="#333", lw=1.4)
    ax.plot([.785, .785], [.60, .405], color="#333", lw=1.4); ax.plot([.785, .785], [.86, .60], color="#333", lw=1.4)
    arrow(ax, .785, .405, .745, .405)
    box(ax, .80, .22, .18, .20, "Subject-level inference:\nbootstrap CI over subjects,\nexact sign-flip test, Holm;\noracle / calibration /\nsmoothing analyses", fc="#E6F4EA", ec="#2E7D32")
    arrow(ax, .74, .32, .80, .32)
    box(ax, .02, .02, .74, .12, "NEGATIVE CONTROL ONLY (no inference on real sensors): room-occupancy CO2/light + simulated posture,\nattached to WESAD windows by random alignment (20 seeds) vs a sequential alignment",
        fc="#F5F5F5", ec="#777", ls="--")
    box(ax, .80, .02, .18, .12, "Exploratory:\nWESAD <-> SWELL-KW\ntransfer (derived features)", fc="#F5F5F5", ec="#777", ls="--")
    plt.tight_layout(); plt.savefig(f"{out}/fig1_pipeline.png", dpi=300); plt.close()


def scatter_margin(feat, oof, out):
    f = pd.read_csv(feat); o = pd.read_csv(oof)
    t = float(o["youden_t"].mean()); sd = float(o["hr_sd_train"].mean())
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for yv, c, l in ((0, "#4C72B0", "normal"), (1, "#C44E52", "stress")):
        g = f[f.y == yv]; ax.scatter(g.hr, g.hrv, s=5, alpha=.3, color=c, label=l)
    top = ax.get_ylim()[1]
    for m, a, yy in ((0.75, .10, .90), (0.5, .18, .97)):
        ax.axvspan(t - m * sd, t + m * sd, color="#999", alpha=a, lw=0)
        ax.text(t - m * sd - 0.5, top * yy, f"-{m} SD", fontsize=7, ha="right", va="top")
        ax.text(t + m * sd + 0.5, top * yy, f"+{m} SD", fontsize=7, ha="left", va="top")
    ax.axvline(t, color="k", lw=1.2, ls="--", label=f"stage-1 HR threshold (mean over folds: {t:.0f} bpm)")
    ax.set_xlabel("Heart rate (bpm)"); ax.set_ylabel("RMSSD (ms, 30-s window)"); ax.legend(fontsize=7, markerscale=2)
    ax.set_title("Windows inside the shaded bands are routed to stage 2", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out}/fig_hr_hrv_margin.png", dpi=300); plt.close()


def scatter_sp(sp, out):
    f = pd.read_csv(sp); fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for yv, c, l in ((0, "#4C72B0", "rest"), (1, "#C44E52", "stress task")):
        g = f[f.y == yv]; ax.scatter(g.hr, g.eda, s=3, alpha=.15, color=c, label=l)
    ax.set_xlabel("Heart rate (bpm, wrist)"); ax.set_ylabel("EDA (uS, 30-s mean)"); ax.set_yscale("log"); ax.legend(markerscale=4, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out}/fig_sp_hr_eda.png", dpi=300); plt.close()


def env_schematic(out):
    fig, ax = plt.subplots(figsize=(9, 3.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, .02, .62, .24, .26, "Room-occupancy series\n(ordered rows 1..N)", bold=True)
    box(ax, .36, .66, .28, .22, "SEQUENTIAL: first n rows in\norder -> features are a function\nof row position (and subject)", fc="#FDECEA", ec="#B71C1C")
    box(ax, .36, .30, .28, .22, "NEGATIVE CONTROL: random rows\nin random order (20 seeds)\n-> independent of labels", fc="#E6F4EA", ec="#2E7D32")
    box(ax, .74, .46, .24, .26, "WESAD windows\n(ordered by subject,\nthen time)", bold=True)
    arrow(ax, .26, .75, .36, .77); arrow(ax, .26, .68, .36, .42); arrow(ax, .64, .77, .74, .62); arrow(ax, .64, .41, .74, .52)
    ax.text(.5, .08, "Posture = simulated AR process with a time-dependent fatigue term; noise SD 8 ppm (CO2), 6 lux (light)", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out}/fig_env_alignment.png", dpi=300); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs_revised/wesad_features.csv"); ap.add_argument("--oof", default="outputs_revised/oof_predictions.csv")
    ap.add_argument("--sp", default="outputs_sp/stress_predict_features.csv"); ap.add_argument("--out", default="figs_final")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fig1(a.out); env_schematic(a.out)
    if os.path.exists(a.features) and os.path.exists(a.oof): scatter_margin(a.features, a.oof, a.out)
    if os.path.exists(a.sp): scatter_sp(a.sp, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
