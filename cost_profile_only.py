"""
cost_profile_only.py - re-measure batch-size-1 latency/size with SINGLE-THREADED models (n_jobs=1).
The first cost_profile.csv used RandomForest(n_jobs=-1): thread start-up (~50 ms) dominated and does not
represent a single-core embedded CPU. Run:
    python cost_profile_only.py --features outputs_revised\\wesad_features.csv --out outputs_revised
Writes cost_profile_single_thread.csv (overwrite-safe) and prints the table.
"""
import argparse, copy
import pandas as pd
import stresslib as sl

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="outputs_revised/wesad_features.csv")
ap.add_argument("--out", default="outputs_revised")
a = ap.parse_args()
df = sl.load_features(a.features).dropna(subset=sl.BIO).reset_index(drop=True)
models = sl.make_models(0, False, n_jobs=1)
rows = {}
for name, proto in models.items():
    m = copy.deepcopy(proto).fit(df, df.y.to_numpy(), df.subject.to_numpy())
    rows[name] = sl.profile_model(m, df.iloc[[0]], reps=300)
out = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "model"})
out.to_csv(f"{a.out}/cost_profile_single_thread.csv", index=False)
print(out[["model", "latency_us_median", "latency_us_q25", "latency_us_q75", "serialized_kb"]].round(1).to_string(index=False))
