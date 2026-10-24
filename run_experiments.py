"""
run_experiments.py - ONE scripted pipeline that regenerates every table/figure input of the paper.

    python run_experiments.py --features outputs_revised/wesad_features.csv --room data/room_occupancy.csv \
                              --out outputs_revised [--quick]

Sections (all subject-level, LOSO, everything fitted on training subjects only):
  A  main model comparison        -> loso_per_subject.csv, loso_summary.csv, pairwise_tests.csv, oof_predictions.csv
  B  cascade frontier             -> cascade_frontier.csv, cascade_tests.csv
  C  test-time sensor dropout     -> dropout.csv
  D  personalisation              -> personalisation.csv, personalisation_tests.csv
  E  false-alarm burden           -> alarm_burden.csv
  F  environmental negative ctrl  -> env_negative_control.csv
  G  cost / size profile          -> cost_profile.csv
  run_manifest.json (versions, subjects, windows per subject/class, seeds)
"""
import argparse, copy, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import stresslib as sl

MARGINS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]  # pre-specified grid, units = training-fold SD of HR
STAGE2 = ["Stacking", "RF", "LR_bio4"]
DROPOUT_MODELS = ["Stacking", "RF", "LR_bio4", "Youden_HR"]
DELTA = 0.05  # equivalence margin on macro-F1: a PRE-SPECIFIED, to-be-justified choice, not a data-driven one


def run_loso(df, models, seed=0, dropout_models=()):
    subjects = sorted(df.subject.unique())
    oof = pd.DataFrame(index=df.index)
    meta_cols = {"youden_t": np.nan, "hr_sd_train": np.nan}
    for k, v in meta_cols.items():
        oof[k] = v
    drop_rows = []
    drop_sets = {f"-{c}": [c] for c in sl.BIO}
    drop_sets["-hr&hrv(ECG lead)"] = ["hr", "hrv"]
    for s in subjects:
        tr, te = df.subject != s, df.subject == s
        Xtr, ytr, gtr = df[tr], df.loc[tr, "y"].to_numpy(), df.loc[tr, "subject"].to_numpy()
        Xte, yte = df[te], df.loc[te, "y"].to_numpy()
        for name, proto in models.items():
            m = copy.deepcopy(proto).fit(Xtr, ytr, gtr)
            oof.loc[te, f"score_{name}"] = m.score(Xte)
            pred = m.predict(Xte)
            oof.loc[te, f"pred_{name}"] = pred
            if name == "Youden_HR":
                oof.loc[te, "youden_t"] = m.t
                oof.loc[te, "hr_sd_train"] = m.sd
            if name in dropout_models:
                f_full = sl.macro_f1(yte, pred)
                for lab, cols in drop_sets.items():
                    Xd = Xte.copy()
                    Xd[cols] = Xtr[cols].mean().to_numpy()  # TRAIN-fold means; model NOT retrained (= sensor failure)
                    drop_rows.append(dict(subject=s, model=name, dropped=lab, f1_full=f_full,
                                          f1_dropped=sl.macro_f1(yte, m.predict(Xd))))
        print(f"LOSO subject {s} done", flush=True)
    return oof, pd.DataFrame(drop_rows)


def per_subject(df, oof, models):
    rows = []
    for s, g in df.groupby("subject"):
        for name in models:
            d = sl.subj_metrics(g.y, oof.loc[g.index, f"score_{name}"], oof.loc[g.index, f"pred_{name}"].astype(int))
            d.update(subject=s, model=name)
            rows.append(d)
    return pd.DataFrame(rows)


def summarise(ps, metrics=("macro_f1", "bal_acc", "acc", "auroc", "auprc", "f1_stress", "prec_stress", "rec_stress")):
    rows = []
    for name, g in ps.groupby("model"):
        r = {"model": name, "n_subjects": len(g)}
        for m in metrics:
            if m in g:
                v = g[m].dropna().to_numpy()
                lo, hi = sl.boot_ci(v)
                r.update({f"{m}_mean": v.mean(), f"{m}_sd_across_subjects": v.std(ddof=1), f"{m}_ci_lo": lo, f"{m}_ci_hi": hi})
        rows.append(r)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def cascade(df, oof, prof, ecg_ms, rng_seed=0, n_rand=20):
    """Stage 1 = training-fold Youden HR rule. Ambiguous <=> |HR - t| <= m * SD_HR(train). Ambiguous windows go to
    stage 2 (full model); the rest keep the stage-1 decision. Also reports a RANDOM-routing control at the same
    invocation rate, so any gain over 'mixture of two models' can be separated from the value of the routing."""
    rng = np.random.default_rng(rng_seed)
    rows, per_subj = [], []
    for s2 in STAGE2:
        for m in MARGINS:
            for s, g in df.groupby("subject"):
                o = oof.loc[g.index]
                amb = (np.abs(g.hr - o.youden_t) <= m * o.hr_sd_train).to_numpy()
                p1, p2 = o["pred_Youden_HR"].to_numpy().astype(int), o[f"pred_{s2}"].to_numpy().astype(int)
                casc = np.where(amb, p2, p1)
                fr = np.mean([sl.macro_f1(g.y, np.where(rng.random(len(g)) < amb.mean(), p2, p1)) for _ in range(n_rand)])
                per_subj.append(dict(stage2=s2, margin=m, subject=s, macro_f1=sl.macro_f1(g.y, casc),
                                     f1_random_routing=fr, f1_stage1=sl.macro_f1(g.y, p1), f1_stage2=sl.macro_f1(g.y, p2),
                                     invoked=amb.mean()))
    ps = pd.DataFrame(per_subj)
    tests = []
    for (s2, m), g in ps.groupby(["stage2", "margin"]):
        lat1 = prof["Youden_HR"]["latency_us_median"]; lat2 = prof[s2]["latency_us_median"]
        inv = g.invoked.mean()
        lo, hi = sl.boot_ci(g.macro_f1)
        r = dict(stage2=s2, margin_sd=m, invoked_mean=inv, macro_f1=g.macro_f1.mean(), ci_lo=lo, ci_hi=hi,
                 f1_random_routing=g.f1_random_routing.mean(), f1_stage1_only=g.f1_stage1.mean(),
                 f1_stage2_only=g.f1_stage2.mean(), exp_model_latency_us=lat1 + inv * lat2,
                 exp_total_latency_us_incl_ecg=lat1 + inv * lat2 + (ecg_ms * 1000 if ecg_ms else np.nan))
        for ref, col in (("stage1", "f1_stage1"), ("stage2", "f1_stage2"), ("random_routing", "f1_random_routing")):
            d = (g.macro_f1 - g[col]).to_numpy()
            l, h = sl.boot_ci(d)
            r.update({f"diff_vs_{ref}": d.mean(), f"diff_vs_{ref}_lo": l, f"diff_vs_{ref}_hi": h, f"p_vs_{ref}": sl.signflip_p(d)})
        tests.append(r)
    t = pd.DataFrame(tests)
    for ref in ("stage1", "stage2", "random_routing"):
        t[f"p_holm_vs_{ref}"] = sl.holm(t[f"p_vs_{ref}"].to_numpy())
    return t, ps


def personalisation(df, oof, n_cals=(10, 25)):
    """Calibration = first n NORMAL windows (chronological). Evaluation = windows starting after the calibration
    windows have ended (no overlap with calibration). Same evaluation windows for every rule."""
    rows = []
    subs = sorted(df.subject.unique())
    for n_cal in n_cals:
        info = {}
        for s in subs:
            g = df[df.subject == s].sort_values("t_start")
            cal = g[g.y == 0].head(n_cal)
            if len(cal) < n_cal:
                continue
            info[s] = dict(mu=cal.hr.mean(), sd=cal.hr.std(ddof=1), ev=g[g.t_start >= cal.t_start.max() + sl.WIN_S])
        for s in info:
            xs = np.concatenate([info[o]["ev"].hr.to_numpy() - info[o]["mu"] for o in info if o != s])
            ys = np.concatenate([info[o]["ev"].y.to_numpy() for o in info if o != s])
            fpr, tpr, thr = roc_curve(ys, xs)
            t = thr[np.argmax(tpr - fpr)]
            ev, mu, sd = info[s]["ev"], info[s]["mu"], info[s]["sd"]
            rules = {"Population_Youden_HR": oof.loc[ev.index, "pred_Youden_HR"].to_numpy().astype(int),
                     "Centered_HR_Youden": (ev.hr.to_numpy() - mu > t).astype(int),
                     "Mean+1SD": (ev.hr.to_numpy() > mu + sd).astype(int),
                     "Mean+2SD": (ev.hr.to_numpy() > mu + 2 * sd).astype(int),
                     "Population_Stacking": oof.loc[ev.index, "pred_Stacking"].to_numpy().astype(int)}
            for r, p in rules.items():
                rows.append(dict(n_cal=n_cal, subject=s, rule=r, macro_f1=sl.macro_f1(ev.y, p), n_eval=len(ev)))
    ps = pd.DataFrame(rows)
    tests = []
    for n_cal, g in ps.groupby("n_cal"):
        w = g.pivot(index="subject", columns="rule", values="macro_f1")
        pairs = [(c, "Population_Youden_HR") for c in w.columns if c != "Population_Youden_HR"]
        t = sl.paired_table(w, "macro_f1", DELTA, pairs)
        t.insert(0, "n_cal", n_cal)
        tests.append(t)
    return ps, pd.concat(tests)


def simulate_posture(n, rng, d0=56, alpha=20, beta=5, lam=0.03):
    """Posture simulation (equation given in the manuscript) with an explicit RNG."""
    p = np.zeros(n); p[0] = d0; sp = 0.0
    for t in range(1, n):
        fatigue = min(1, t / 3600)
        sp = np.clip(sp + 0.05 * (0 - sp) + rng.normal(0, 0.1), 0, 1)
        p[t] = (1 - lam) * p[t - 1] + lam * (d0 - alpha * fatigue - beta * sp) + rng.normal(0, 1.5)
    return p


def env_negative_control(df, room_csv, n_perm, quick):
    """Fusion test as a NEGATIVE CONTROL. 'sequential' = the first n room rows in
    order => features are functions of row position); 'permuted' draws random room rows in random order, the true
    null. If sequential falls inside the permuted null distribution, the sequential-alignment result carried no signal."""
    room = pd.read_csv(room_csv)
    room.columns = [c.strip() for c in room.columns]
    n = len(df)
    ntree = 100 if quick else 200
    rows = []

    def loso_f1(d, cols):
        f = []
        for s in sorted(d.subject.unique()):
            tr, te = d.subject != s, d.subject == s
            sc = sl.StandardScaler().fit(d.loc[tr, cols])
            m = RandomForestClassifier(ntree, class_weight="balanced", random_state=0, n_jobs=-1)
            m.fit(sc.transform(d.loc[tr, cols]), d.loc[tr, "y"])
            f.append(sl.macro_f1(d.loc[te, "y"], m.predict(sc.transform(d.loc[te, cols]))))
        return float(np.mean(f))

    f_bio = loso_f1(df, sl.BIO)
    for mode, seed in [("sequential", 42)] + [("permuted", 1000 + i) for i in range(n_perm)]:
        rng = np.random.default_rng(seed)
        co2 = room["CO2"].to_numpy() + rng.normal(0, 8, len(room))
        lux = room["Light"].to_numpy() + rng.normal(0, 6, len(room))
        post = simulate_posture(len(room), rng)
        idx = np.arange(n) % len(room) if mode == "sequential" else rng.permutation(len(room))[:n]
        d = df.copy()
        d["co2_noisy"], d["lux_noisy"], d["posture_cm"] = co2[idx], lux[idx], post[idx]
        r = dict(mode=mode, seed=seed, f1_bio=f_bio, f1_fused=loso_f1(d, sl.BIO + sl.ENV), f1_env_only=loso_f1(d, sl.ENV))
        r["delta_fused_minus_bio"] = r["f1_fused"] - f_bio
        rows.append(r)
        print(f"env control {mode} {seed}: delta={r['delta_fused_minus_bio']:+.4f}", flush=True)
    out = pd.DataFrame(rows)
    nul = out[out["mode"] == "permuted"]["delta_fused_minus_bio"]
    seq = out[out["mode"] == "sequential"]["delta_fused_minus_bio"].iloc[0]
    out.attrs["summary"] = dict(sequential_delta=float(seq), null_mean=float(nul.mean()), null_sd=float(nul.std()),
                                frac_null_below_sequential=float((nul <= seq).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs_revised/wesad_features.csv")
    ap.add_argument("--room", default="data/room_occupancy.csv")
    ap.add_argument("--out", default="outputs_revised")
    ap.add_argument("--quick", action="store_true", help="fewer trees / permutations (smoke test only)")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warnings.filterwarnings("ignore")
    df = sl.load_features(a.features)
    df = df.dropna(subset=sl.BIO).reset_index(drop=True)
    models = sl.make_models(a.seed, a.quick)

    # ---- A: main comparison
    oof, drop = run_loso(df, models, a.seed, DROPOUT_MODELS)
    oof_out = pd.concat([df[["subject", "t_start", "y", "hr"]], oof], axis=1)
    oof_out.to_csv(f"{a.out}/oof_predictions.csv", index=False)
    ps = per_subject(df, oof, models)
    ps.to_csv(f"{a.out}/loso_per_subject.csv", index=False)
    summ = summarise(ps)
    summ.to_csv(f"{a.out}/loso_summary.csv", index=False)
    pt = pd.concat([sl.paired_table(ps.pivot(index="subject", columns="model", values=m), m, DELTA)
                    for m in ("macro_f1", "bal_acc")])
    pt.to_csv(f"{a.out}/pairwise_tests.csv", index=False)
    print(summ[["model", "macro_f1_mean", "macro_f1_ci_lo", "macro_f1_ci_hi", "bal_acc_mean", "auroc_mean"]].to_string(index=False))

    # ---- G: cost profile (fit on all data once; batch size 1)
    prof = {}
    Xone = df.iloc[[0]]
    for name, proto in models.items():
        mfit = copy.deepcopy(proto).fit(df, df.y.to_numpy(), df.subject.to_numpy())
        prof[name] = sl.profile_model(mfit, Xone, reps=100 if a.quick else 300)
    cost = pd.DataFrame(prof).T.reset_index().rename(columns={"index": "model"})
    cost.to_csv(f"{a.out}/cost_profile.csv", index=False)
    ecg_ms = None
    tj = f"{a.out}/extraction_timing.json"
    if os.path.exists(tj):
        ecg_ms = json.load(open(tj))["median_ms"]

    # ---- B: cascade
    ct, cps = cascade(df, oof, prof, ecg_ms)
    ct.to_csv(f"{a.out}/cascade_frontier.csv", index=False)
    cps.to_csv(f"{a.out}/cascade_per_subject.csv", index=False)

    # ---- C: dropout (positive = degradation)
    drop["delta_f1_degradation"] = drop.f1_full - drop.f1_dropped
    drop.to_csv(f"{a.out}/dropout.csv", index=False)

    # ---- D: personalisation
    pp, ptests = personalisation(df, oof)
    pp.to_csv(f"{a.out}/personalisation.csv", index=False)
    ptests.to_csv(f"{a.out}/personalisation_tests.csv", index=False)

    # ---- E: alarm burden
    rows = []
    for name in ["Youden_HR", "LR_HR_EDA", "LR_bio4", "RF", "XGB", "Stacking"]:
        for s, g in df.groupby("subject"):
            r = sl.alarm_burden(g, oof.loc[g.index, f"pred_{name}"].astype(int).to_numpy()); r.update(model=name, subject=s)
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{a.out}/alarm_burden.csv", index=False)

    # ---- F: environment negative control
    if os.path.exists(a.room):
        env = env_negative_control(df, a.room, 3 if a.quick else a.n_perm, a.quick)
        env.to_csv(f"{a.out}/env_negative_control.csv", index=False)
        json.dump(env.attrs["summary"], open(f"{a.out}/env_negative_control_summary.json", "w"), indent=1)
    else:
        print("room csv not found - skipped negative control")

    manifest = dict(env=sl.environment_manifest(), seed=a.seed, quick=a.quick, margins=MARGINS, delta=DELTA,
                    subjects=[int(s) for s in sorted(df.subject.unique())],
                    windows_per_subject_class=df.groupby(["subject", "y"]).size().unstack(fill_value=0).to_dict())
    json.dump(manifest, open(f"{a.out}/run_manifest.json", "w"), indent=1, default=str)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
