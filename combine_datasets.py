"""
combine_datasets.py - pooled (exploratory) estimate of the baseline-calibration effect across two datasets.

    python combine_datasets.py --a outputs_extra\\e2_calibration_per_subject.csv ^
                               --b outputs_sp_extra\\e2_calibration_per_subject.csv ^
                               --names WESAD StressPredict --out outputs_pooled

For each (model, n_cal): per-subject paired difference (centred - raw macro-F1) in each dataset, then
  * per-dataset mean with subject-bootstrap CI
  * pooled mean over all subjects (subject-weighted) and mean of the two dataset means (dataset-weighted),
    both with STRATIFIED bootstrap CIs (subjects resampled within their own dataset)
  * difference between datasets (heterogeneity) with bootstrap CI
  * pooled sign-flip p-value (Monte Carlo, all subjects)
This is an exploratory summary: the two datasets differ in sensor, stressor and label definition, and the
pre-declared PRIMARY test is the per-dataset one (LR, n_cal = 25). Report both.
"""
import argparse, os
import numpy as np
import pandas as pd
import stresslib as sl


def diffs(path, model, n_cal, metric="macro_f1"):
    r = pd.read_csv(path)
    r = r[(r.model == model) & (r.n_cal == n_cal)]
    w = r.pivot(index="subject", columns="variant", values=metric).dropna()
    return (w["centred"] - w["raw"]).to_numpy()


def strat_boot(dA, dB, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = dA[rng.integers(0, len(dA), (B, len(dA)))]; b = dB[rng.integers(0, len(dB), (B, len(dB)))]
    subj_w = (a.sum(1) + b.sum(1)) / (len(dA) + len(dB))
    data_w = (a.mean(1) + b.mean(1)) / 2
    het = a.mean(1) - b.mean(1)
    q = lambda x: (float(np.quantile(x, .025)), float(np.quantile(x, .975)))
    return q(subj_w), q(data_w), q(het)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--names", nargs=2, default=["A", "B"]); ap.add_argument("--out", default="outputs_pooled")
    ap.add_argument("--metric", default="macro_f1")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = []
    for model, n_cal in [("LR", 5), ("LR", 10), ("LR", 25), ("GB", 25), ("Youden_HR", 5), ("Youden_HR", 10), ("Youden_HR", 25)]:
        try:
            dA, dB = diffs(a.a, model, n_cal, a.metric), diffs(a.b, model, n_cal, a.metric)
        except Exception as e:  # model or n_cal missing in one file
            print("skip", model, n_cal, e); continue
        (sl_lo, sl_hi), (dw_lo, dw_hi), (h_lo, h_hi) = strat_boot(dA, dB)
        lA, hA = sl.boot_ci(dA); lB, hB = sl.boot_ci(dB)
        allv = np.r_[dA, dB]
        rows.append(dict(model=model, n_cal=n_cal, metric=a.metric,
                         **{f"{a.names[0]}_n": len(dA), f"{a.names[0]}_mean": dA.mean(), f"{a.names[0]}_lo": lA, f"{a.names[0]}_hi": hA,
                            f"{a.names[1]}_n": len(dB), f"{a.names[1]}_mean": dB.mean(), f"{a.names[1]}_lo": lB, f"{a.names[1]}_hi": hB},
                         pooled_subject_weighted=allv.mean(), pooled_sw_lo=sl_lo, pooled_sw_hi=sl_hi,
                         pooled_dataset_weighted=(dA.mean() + dB.mean()) / 2, pooled_dw_lo=dw_lo, pooled_dw_hi=dw_hi,
                         difference_between_datasets=dA.mean() - dB.mean(), diff_lo=h_lo, diff_hi=h_hi,
                         p_signflip_pooled=sl.signflip_p(allv)))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(a.out, "pooled_calibration.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
