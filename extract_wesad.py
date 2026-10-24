"""
extract_wesad.py - re-extract HR/HRV/EDA/temperature windows from raw WESAD pickles with a COMPLETE QC log.

    python extract_wesad.py --raw data/wesad_raw --out outputs_revised [--win 30 --step 15]

Outputs
  wesad_features.csv     kept windows (subject, t_start, label_raw, y, hr, hrv, sdnn, pnn50, eda, temp_chest, ...)
  window_log.csv         EVERY candidate window with status: kept / qc_fail(+reason) / mixed_label / unmapped_label
  qc_summary.csv         per subject x class: candidate windows, kept, rejected, rejection %
  extraction_timing.json measured ms per 30 s ECG window (filter + peaks + HRV) on this machine
Optional: if neurokit2 is installed, HR/RMSSD agreement with an independent detector is written to
detector_agreement.csv.
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
import stresslib as sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/wesad_raw")
    ap.add_argument("--out", default="outputs_revised")
    ap.add_argument("--win", type=float, default=sl.WIN_S)
    ap.add_argument("--step", type=float, default=sl.STEP_S)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted([f for f in os.listdir(a.raw) if f.endswith(".pkl")], key=lambda f: int(f[1:-4]))
    logs = []
    for f in files:
        sid = int(f[1:-4])
        d = sl.extract_subject(os.path.join(a.raw, f), sid, a.win, a.step)
        logs.append(d)
        print(f"S{sid}: {int((d.status == 'kept').sum())} kept / {len(d)} windows "
              f"(qc_fail {int((d.status == 'qc_fail').sum())})")
    log = pd.concat(logs, ignore_index=True)
    if "reason" not in log:
        log["reason"] = ""
    log.to_csv(os.path.join(a.out, "window_log.csv"), index=False)

    kept = log[log.status == "kept"].drop(columns=["status", "reason"], errors="ignore").copy()
    kept["y"] = kept["y"].astype(int)
    kept.to_csv(os.path.join(a.out, "wesad_features.csv"), index=False)

    cand = log[log.status.isin(["kept", "qc_fail"])].copy()
    cand["class"] = cand["y"].map({0: "NORMAL", 1: "STRESS"})
    qc = (cand.groupby(["subject", "class"])
          .agg(candidate=("status", "size"), kept=("status", lambda s: int((s == "kept").sum()))).reset_index())
    qc["rejected"] = qc["candidate"] - qc["kept"]
    qc["rejected_pct"] = 100 * qc["rejected"] / qc["candidate"]
    qc.to_csv(os.path.join(a.out, "qc_summary.csv"), index=False)
    print("\nRejection by reason:\n", log[log.status == "qc_fail"]["reason"].value_counts())
    print("\nOverall candidate->kept per class:\n", qc.groupby("class")[["candidate", "kept"]].sum())

    # timing on a sample of kept windows
    rng = np.random.default_rng(0)
    sample = kept.sample(min(40, len(kept)), random_state=0)
    wins = []
    for _, r in sample.iterrows():
        with open(os.path.join(a.raw, f"S{int(r.subject)}.pkl"), "rb") as fh:
            ecg = pickle.load(fh, encoding="latin1")["signal"]["chest"]["ECG"].flatten().astype(float)
        s0 = int(r.t_start * sl.FS)
        wins.append(ecg[s0:s0 + int(a.win * sl.FS)])
    json.dump(sl.time_ecg_pipeline(wins), open(os.path.join(a.out, "extraction_timing.json"), "w"), indent=1)

    try:
        import neurokit2 as nk
        rows = []
        for sid, g in kept.groupby("subject"):
            with open(os.path.join(a.raw, f"S{sid}.pkl"), "rb") as fh:
                ecg = pickle.load(fh, encoding="latin1")["signal"]["chest"]["ECG"].flatten().astype(float)
            for _, r in g.sample(min(15, len(g)), random_state=0).iterrows():
                s0 = int(r.t_start * sl.FS)
                w = ecg[s0:s0 + int(a.win * sl.FS)]
                mine = sl.detect_r_peaks(sl.bandpass(w))
                ref = nk.ecg_peaks(nk.ecg_clean(w, sampling_rate=sl.FS), sampling_rate=sl.FS)[1]["ECG_R_Peaks"]
                v = sl.validate_detector(mine, ref)
                v.update(subject=sid, t_start=r.t_start)
                rows.append(v)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "detector_agreement.csv"), index=False)
        print("detector_agreement.csv written (reference = NeuroKit2; NOT ground truth - add manual/MIT-BIH checks)")
    except ImportError:
        print("neurokit2 not installed: skipped detector agreement (pip install neurokit2)")


if __name__ == "__main__":
    main()
