"""
cross_dataset.py - exploratory WESAD <-> SWELL-KW transfer with the confounds made explicit.

    python cross_dataset.py --wesad-raw data/wesad_raw --swell data/swell --out outputs_revised

Design choices that limit confounding:
  1. WESAD HR/RMSSD is recomputed at several window lengths (30/60/300 s) so the window-length confound can be
     examined directly instead of asserted.
  2. Scaling is fitted on the SOURCE data only (primary). Target-fitted scaling is reported separately and labelled
     TRANSDUCTIVE.
  3. AUCs are reported as obtained. No polarity flipping. The direction of each single feature in each dataset is
     reported on its own (AUC of raw HR, raw RMSSD), which shows whether the sign problem is a property of the data.
  4. SWELL labels: (a) time pressure vs no stress (interruption rows DROPPED), (b) time pressure + interruption vs no
     stress, (c) interruption counted as 'no stress', for reference only.
  5. If SWELL has a subject column it is auto-detected and used for cluster bootstrap CIs; otherwise SWELL rows are
     flagged as non-independent and no CI is claimed.
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import stresslib as sl

SUBJECT_CANDIDATES = ["subject", "subject_id", "Subject", "participant", "pp", "PP", "id", "ID"]


def cluster_boot_auc(y, s, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(B):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[pick])) == 2:
            out.append(roc_auc_score(y[pick], s[pick]))
    return float(np.quantile(out, .025)), float(np.quantile(out, .975))


def wesad_windows(raw, win_s, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    parts = []
    for f in sorted(os.listdir(raw), key=lambda f: int(f[1:-4])):
        if f.endswith(".pkl"):
            d = sl.extract_subject(os.path.join(raw, f), int(f[1:-4]), win_s, win_s / 2)
            parts.append(d[d.status == "kept"])
    out = pd.concat(parts)
    out.to_csv(cache, index=False)
    return out


def load_swell(folder):
    parts = [pd.read_csv(os.path.join(folder, f)) for f in ("train.csv", "test.csv") if os.path.exists(os.path.join(folder, f))]
    sw = pd.concat(parts, ignore_index=True)
    subj = next((c for c in SUBJECT_CANDIDATES if c in sw.columns), None)
    print("SWELL columns:", list(sw.columns)[:40], "| subject column:", subj)
    return sw, subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad-raw", default="data/wesad_raw")
    ap.add_argument("--swell", default="data/swell")
    ap.add_argument("--out", default="outputs_revised")
    ap.add_argument("--windows", type=int, nargs="+", default=[30, 60, 300])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sw, subj = load_swell(a.swell)
    schemes = {
        "TP_vs_NoStress (interruption dropped)": (sw[sw.condition != "interruption"], lambda d: (d.condition == "time pressure").astype(int)),
        "TP+INT_vs_NoStress": (sw, lambda d: (d.condition != "no stress").astype(int)),
        "TP_vs_rest (interruption counted as no stress)": (sw, lambda d: (d.condition == "time pressure").astype(int)),
    }
    rows = []
    # (3) single-feature polarity in each dataset
    for sname, (d, fy) in schemes.items():
        y = fy(d).to_numpy()
        for feat in ("HR", "RMSSD"):
            rows.append(dict(kind="single_feature_AUC", dataset="SWELL", scheme=sname, feature=feat, auc=roc_auc_score(y, d[feat])))
    for W in a.windows:
        we = wesad_windows(a.wesad_raw, W, os.path.join(a.out, f"wesad_windows_{W}s.csv"))
        for feat, col in (("HR", "hr"), ("RMSSD", "hrv")):
            y = we.y.astype(int).to_numpy()
            lo, hi = cluster_boot_auc(y, we[col].to_numpy(), we.subject.to_numpy())
            rows.append(dict(kind="single_feature_AUC", dataset=f"WESAD_{W}s", scheme="stress_vs_normal", feature=feat,
                             auc=roc_auc_score(y, we[col]), ci_lo=lo, ci_hi=hi, n_windows=len(we)))
        # transfer
        for sname, (d, fy) in schemes.items():
            ysw = fy(d).to_numpy()
            for feats_w, feats_s, tag in ((["hr", "hrv"], ["HR", "RMSSD"], "HR+RMSSD"), (["hr"], ["HR"], "HR_only"), (["hrv"], ["RMSSD"], "RMSSD_only")):
                for scaling in ("source_only", "target_fitted_TRANSDUCTIVE"):
                    # WESAD -> SWELL
                    sc_s = StandardScaler().fit(we[feats_w]); clf = LogisticRegression(class_weight="balanced", max_iter=1000)
                    clf.fit(sc_s.transform(we[feats_w]), we.y.astype(int))
                    sc_t = sc_s if scaling == "source_only" else StandardScaler().fit(d[feats_s])
                    auc = roc_auc_score(ysw, clf.decision_function(sc_t.transform(d[feats_s].to_numpy())))
                    r = dict(kind="transfer", direction="WESAD->SWELL", wesad_window_s=W, scheme=sname, features=tag, scaling=scaling, auc=auc)
                    if subj:
                        r["ci_lo"], r["ci_hi"] = cluster_boot_auc(ysw, clf.decision_function(sc_t.transform(d[feats_s].to_numpy())), d[subj].to_numpy())
                    rows.append(r)
                    # SWELL -> WESAD
                    sc_s2 = StandardScaler().fit(d[feats_s]); clf2 = LogisticRegression(class_weight="balanced", max_iter=1000)
                    clf2.fit(sc_s2.transform(d[feats_s].to_numpy()), ysw)
                    sc_t2 = sc_s2 if scaling == "source_only" else StandardScaler().fit(we[feats_w])
                    s2 = clf2.decision_function(sc_t2.transform(we[feats_w].to_numpy()))
                    lo, hi = cluster_boot_auc(we.y.astype(int).to_numpy(), s2, we.subject.to_numpy())
                    rows.append(dict(kind="transfer", direction="SWELL->WESAD", wesad_window_s=W, scheme=sname, features=tag,
                                     scaling=scaling, auc=roc_auc_score(we.y.astype(int), s2), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "cross_dataset_revised.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    if not subj:
        print("\nNOTE: no subject column in the SWELL file -> SWELL rows are not independent; no CI reported for SWELL-side "
              "estimates and any within-SWELL score is optimistic. Obtain participant IDs from the original 4TU release.")


if __name__ == "__main__":
    main()
