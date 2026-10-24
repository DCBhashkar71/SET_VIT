"""
build_stress_predict.py - harmonise the Stress-Predict dataset (Iqbal et al., Sensors 2022) for the replication.

    python build_stress_predict.py --root data\\Stress-Predict-Dataset-main --out outputs_sp

Inputs : Processed_data/Improved_All_Combined_hr_rsp_binary.csv  (per-second Empatica HR + binary label, 34 participants)
         Raw_data/S??/EDA.csv, TEMP.csv                          (4 Hz, first row = unix start, second row = rate)
Output : wesad-style window table  subject, t_start, y, hr, eda, temp_wrist   (+ qc summary)
Windows: 30 s, 15 s step, label purity >= 0.9, >= 80 % valid HR seconds.
Labels : 0 = baseline / rest / relax / questionnaire, 1 = stress task (Stroop, interview, hyperventilation), as
         provided by the dataset authors. Stress tasks differ from WESAD's TSST; heart rate is the Empatica-computed 1 Hz HR
         (wrist PPG), NOT ECG, and there is no HRV feature here.
"""
import argparse, os
import numpy as np
import pandas as pd

WIN, STEP, PURITY, MIN_HR_FRAC = 30, 15, 0.9, 0.8


def load_low_rate(path):
    with open(path) as f:
        t0 = float(f.readline().split(",")[0]); fs = float(f.readline().split(",")[0])
    x = pd.read_csv(path, skiprows=2, header=None).iloc[:, 0].to_numpy(float)
    return t0, fs, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Stress-Predict-Dataset-main"); ap.add_argument("--out", default="outputs_sp")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    proc = pd.read_csv(os.path.join(a.root, "Processed_data", "Improved_All_Combined_hr_rsp_binary.csv"))
    rows, qc = [], []
    for pid, g in proc.groupby("Participant"):
        g = g.drop_duplicates("Time(sec)").set_index("Time(sec)").sort_index()
        full = g.reindex(np.arange(g.index.min(), g.index.max() + 1))
        hr, lab = full["HR"].to_numpy(float), full["Label"].to_numpy(float)
        t0abs = int(full.index[0])
        folder = os.path.join(a.root, "Raw_data", f"S{int(pid):02d}")
        eda = load_low_rate(os.path.join(folder, "EDA.csv")); tmp = load_low_rate(os.path.join(folder, "TEMP.csv"))
        n_all = n_kept = 0
        for s in range(0, len(full) - WIN + 1, STEP):
            n_all += 1
            l = lab[s:s + WIN]; h = hr[s:s + WIN]
            if np.isnan(l).mean() > 0.1:
                continue
            lv = l[~np.isnan(l)]; y = int(round(lv.mean())); pur = (lv == y).mean()
            if pur < PURITY or np.isnan(h).mean() > 1 - MIN_HR_FRAC:
                continue
            ta = t0abs + s
            feats = {}
            for name, (t0, fs, x) in (("eda", eda), ("temp_wrist", tmp)):
                i0, i1 = int(round((ta - t0) * fs)), int(round((ta + WIN - t0) * fs))
                if i0 < 0 or i1 > len(x):
                    feats = None; break
                feats[name] = float(np.mean(x[i0:i1]))
            if feats is None:
                continue
            n_kept += 1
            rows.append(dict(subject=int(pid), t_start=float(s), y=y, hr=float(np.nanmean(h)), **feats))
        qc.append(dict(subject=int(pid), seconds=len(full), windows_candidate=n_all, windows_kept=n_kept,
                       stress_windows=int(sum(r["y"] for r in rows if r["subject"] == pid)), n_stress_seconds=int(np.nansum(lab == 1))))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "stress_predict_features.csv"), index=False)
    q = pd.DataFrame(qc); q.to_csv(os.path.join(a.out, "stress_predict_qc.csv"), index=False)
    print(q.to_string(index=False)); print(df.groupby("y").size()); print(df[["hr", "eda", "temp_wrist"]].describe().round(2))


if __name__ == "__main__":
    main()
