"""
extra_experiments.py - the additional experiments (baseline shift, calibration, smoothing, learning curve).

    python extra_experiments.py --features outputs_revised\\wesad_features.csv --out outputs_extra
    (optional) --stacking            add the stacking ensemble to E2/E5 (slower)
    (optional) --only e1 e2 e5       run a subset
    (replication on another dataset) --features other_harmonised.csv --cols hr eda --tag replication --out outputs_repl

Central question: is the bottleneck model complexity, or between-subject baseline shift?

E1  variance decomposition: between-subject spread of resting HR/EDA vs the stress effect; pooled vs within-subject AUROC
E2  baseline calibration: subject-centred features (supervised: first n rest windows; label-free: subject z-score) for
    LR / GB (/ Stacking) / Youden, n_cal in {5,10,25,50}; calibration and evaluation windows never overlap
E3  responders / non-responders: per-subject effect size and AUROC vs population-model performance
E4  oracle ceiling: macro-F1 with the best SUBJECT-SPECIFIC HR threshold vs the population Youden threshold
E5  causal temporal smoothing: macro-F1, false-alarm events/hour, detection delay, missed episodes
E6  learning curve over the number of TRAINING subjects (3..14)
E7  protocol variants for literature comparison: normal-vs-stress, baseline-vs-stress, 3-class
E8  minimum detectable difference (power) from loso_per_subject.csv (optional input)

All inference is per-subject (bootstrap CI over subjects, exact sign-flip test, Holm).
PRIMARY (declare in analysis_plan.txt BEFORE running): E2, LR, n_cal = 25, centred vs raw macro-F1.
NOTE: a centred-HR gain was already seen in run_experiments.py, so E2 is exploratory unless replicated on an
independent dataset (--features other.csv --tag replication).
"""
import argparse, copy, json, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import stresslib as sl

DELTA = 0.05
N_CALS = [5, 10, 25, 50]
SMOOTH_K = [1, 3, 5, 9]          # 1 = no smoothing; windows are 30 s with 15 s step
LC_SIZES = [3, 5, 8, 11, 14]


def auc_safe(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def mk_models(cols, seed=0, stacking=False):
    lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
    gb = lambda: GradientBoostingClassifier(random_state=seed)
    m = {"LR": sl.Skl(lr(), cols=cols), "GB": sl.Skl(gb(), cols=cols, sw=True)}
    if stacking:   # SAME four base learners as the main table (RF-200, XGBoost/HGB, GB, LR) so "Stacking" means one thing
        xg = (lambda: sl.XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss", random_state=seed)) if sl.HAVE_XGB \
            else (lambda: sl.HistGradientBoostingClassifier(max_depth=4, random_state=seed))
        bases = {"rf": sl.Skl(sl.RandomForestClassifier(200, class_weight="balanced", random_state=seed, n_jobs=1), cols=cols),
                 "xgb": sl.Skl(xg(), cols=cols, sw=True), "gb": sl.Skl(gb(), cols=cols, sw=True), "lr": sl.Skl(lr(), cols=cols)}
        m["Stacking"] = sl.Stacking(bases, passthrough=True, seed=seed, cols=cols)
    return m


def fit_score(proto, Xtr, ytr, gtr, Xte):
    m = copy.deepcopy(proto).fit(Xtr, ytr, gtr)
    return m.score(Xte), m.predict(Xte)


def paired(rows, keys, a="centred", b="raw", metric="macro_f1"):
    """rows: long DataFrame; paired difference a-b across subjects within each key combination."""
    out = []
    for k, g in rows.groupby(keys):
        w = g.pivot(index="subject", columns="variant", values=metric)
        d = (w[a] - w[b]).dropna().to_numpy()
        if len(d) < 3:
            continue
        lo, hi = sl.boot_ci(d)
        out.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "metric": metric, "mean_diff": d.mean(),
                    "ci_lo": lo, "ci_hi": hi, "dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
                    "p_signflip": sl.signflip_p(d), "n_subjects": len(d)})
    t = pd.DataFrame(out)
    if len(t):
        t["p_holm"] = sl.holm(t["p_signflip"].to_numpy())
    return t


# ---------------------------------------------------------------------------------------------- LOSO base scores
def loso_scores(df, cols, models):
    oof = pd.DataFrame(index=df.index)
    oof["youden_t"] = np.nan
    for s in sorted(df.subject.unique()):
        tr, te = df.subject != s, df.subject == s
        yt = df.loc[tr, "y"].to_numpy(); gt = df.loc[tr, "subject"].to_numpy()
        yo = sl.YoudenRule("hr").fit(df[tr], yt)
        oof.loc[te, "score_Youden_HR"] = yo.score(df[te]); oof.loc[te, "pred_Youden_HR"] = yo.predict(df[te])
        oof.loc[te, "youden_t"] = yo.t
        for name, proto in models.items():
            sc, pr = fit_score(proto, df[tr], yt, gt, df[te])
            oof.loc[te, f"score_{name}"] = sc; oof.loc[te, f"pred_{name}"] = pr
    return oof


# ---------------------------------------------------------------------------------------------- E1
def e1(df, cols, out):
    base = {c: df[df.y == 0].groupby("subject")[c].mean() for c in cols}
    rows = []
    for s, g in df.groupby("subject"):
        n, st = g[g.y == 0], g[g.y == 1]
        r = dict(subject=s)
        for c in cols:
            sdp = np.sqrt((n[c].var(ddof=1) + st[c].var(ddof=1)) / 2)
            r[f"{c}_base_mean"] = n[c].mean(); r[f"{c}_stress_mean"] = st[c].mean()
            r[f"{c}_delta"] = st[c].mean() - n[c].mean(); r[f"{c}_d"] = r[f"{c}_delta"] / sdp if sdp > 0 else np.nan
            r[f"{c}_auc_within_raw"] = auc_safe(g.y, g[c])
        rows.append(r)
    sub = pd.DataFrame(rows)
    sub.to_csv(f"{out}/e1_per_subject.csv", index=False)
    summ = {}
    for c in cols:
        sign = 1 if auc_safe(df.y, df[c]) >= .5 else -1          # orient by the pooled direction
        cen = df[c] - df.subject.map(base[c])
        aw = np.array([auc_safe(g.y, sign * g[c]) for _, g in df.groupby("subject")])
        summ[c] = dict(between_subject_sd_of_resting_mean=float(sub[f"{c}_base_mean"].std(ddof=1)),
                       mean_stress_delta=float(sub[f"{c}_delta"].mean()), sd_of_stress_delta=float(sub[f"{c}_delta"].std(ddof=1)),
                       ratio_between_sd_to_mean_delta=float(sub[f"{c}_base_mean"].std(ddof=1) / abs(sub[f"{c}_delta"].mean())),
                       auc_pooled=float(auc_safe(df.y, sign * df[c])), auc_within_subject_mean=float(np.nanmean(aw)),
                       auc_within_subject_ci=list(sl.boot_ci(aw[~np.isnan(aw)])),
                       auc_pooled_after_oracle_centering=float(auc_safe(df.y, sign * cen)))
    json.dump(summ, open(f"{out}/e1_summary.json", "w"), indent=1)
    print("E1", json.dumps(summ, indent=1)[:1500])
    return sub


# ---------------------------------------------------------------------------------------------- E2
def e2(df, cols, out, models, n_cals=None):
    rows = []
    for n_cal in (n_cals or N_CALS):
        info = {}
        for s, g in df.groupby("subject"):
            g = g.sort_values("t_start"); first_st = g.t_start[g.y == 1].min()
            cal = g[(g.y == 0) & (g.t_start + sl.WIN_S <= first_st)].head(n_cal)   # rest BEFORE the first stress window
            if len(cal) < n_cal:
                continue
            ev = g[g.t_start >= cal.t_start.max() + sl.WIN_S]
            if ev.y.nunique() < 2:
                continue
            info[s] = dict(mu=cal[cols].mean(), ev=ev)
        for s in info:
            tr_ids = [o for o in info if o != s]
            Tr = pd.concat([info[o]["ev"] for o in tr_ids]); Te = info[s]["ev"]
            Trc, Tec = Tr.copy(), Te.copy()
            for o in tr_ids:
                idx = info[o]["ev"].index; Trc.loc[idx, cols] = info[o]["ev"][cols].to_numpy() - info[o]["mu"].to_numpy()
            Tec[cols] = Te[cols].to_numpy() - info[s]["mu"].to_numpy()
            ytr, gtr = Tr.y.to_numpy(), Tr.subject.to_numpy()
            variants = {"raw": (Tr, Te), "centred": (Trc, Tec)}
            for var, (A, B) in variants.items():
                yo = sl.YoudenRule("hr").fit(A, ytr)
                p = yo.predict(B)
                rows.append(dict(n_cal=n_cal, subject=s, model="Youden_HR", variant=var, **sl.subj_metrics(Te.y, yo.score(B), p)))
                for name, proto in models.items():
                    sc, pr = fit_score(proto, A, ytr, gtr, B)
                    rows.append(dict(n_cal=n_cal, subject=s, model=name, variant=var, **sl.subj_metrics(Te.y, sc, pr)))
        print(f"E2 n_cal={n_cal} done ({len(info)} subjects)", flush=True)
    # label-free retrospective subject z-score (uses the subject's whole session; no labels)
    z = df.copy()
    for s, g in df.groupby("subject"):
        z.loc[g.index, cols] = (g[cols] - g[cols].mean()) / g[cols].std(ddof=1).replace(0, 1)
    for s in sorted(df.subject.unique()):
        tr, te = df.subject != s, df.subject == s
        for var, D in (("raw", df), ("centred", z)):
            ytr, gtr = D.loc[tr, "y"].to_numpy(), D.loc[tr, "subject"].to_numpy()
            yo = sl.YoudenRule("hr").fit(D[tr], ytr)
            rows.append(dict(n_cal=0, subject=s, model="Youden_HR", variant=var, **sl.subj_metrics(D.loc[te, "y"], yo.score(D[te]), yo.predict(D[te]))))
            for name, proto in models.items():
                sc, pr = fit_score(proto, D[tr], ytr, gtr, D[te])
                rows.append(dict(n_cal=0, subject=s, model=name, variant=var, **sl.subj_metrics(D.loc[te, "y"], sc, pr)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/e2_calibration_per_subject.csv", index=False)
    tests = pd.concat([paired(res, ["n_cal", "model"], metric=m) for m in ("macro_f1", "auroc")])
    tests["primary"] = (tests.model == "LR") & (tests.n_cal == 25) & (tests.metric == "macro_f1")
    tests.to_csv(f"{out}/e2_calibration_tests.csv", index=False)
    print(tests[tests.metric == "macro_f1"][["n_cal", "model", "mean_diff", "ci_lo", "ci_hi", "p_signflip", "p_holm", "primary"]].round(3).to_string(index=False))
    print("(n_cal = 0 is the LABEL-FREE retrospective subject z-score)")


# ---------------------------------------------------------------------------------------------- E3 / E4
def e3_e4(df, cols, out, oof, sub):
    rows = []
    for s, g in df.groupby("subject"):
        yo = g.y.to_numpy()
        fpr_f1 = []
        ths = np.unique(g.hr.to_numpy())
        best = max(sl.macro_f1(yo, (g.hr.to_numpy() > t).astype(int)) for t in ths)
        r = dict(subject=s, f1_population_youden=sl.macro_f1(yo, oof.loc[g.index, "pred_Youden_HR"].astype(int)),
                 f1_oracle_subject_threshold=best, f1_LR=sl.macro_f1(yo, oof.loc[g.index, "pred_LR"].astype(int)),
                 auroc_LR=auc_safe(yo, oof.loc[g.index, "score_LR"]))
        rows.append(r)
    t = pd.DataFrame(rows).merge(sub[["subject", "hr_d", "hr_auc_within_raw"] + ([f"{c}_d" for c in cols if c == "eda"])], on="subject")
    t["responder_auc_ge_0.8"] = t.hr_auc_within_raw >= 0.8
    t.to_csv(f"{out}/e3_e4_responders_oracle.csv", index=False)
    d = (t.f1_oracle_subject_threshold - t.f1_population_youden).to_numpy()
    lo, hi = sl.boot_ci(d)
    rho = spearmanr(t.hr_d, t.f1_LR)
    summ = dict(oracle_minus_population_mean=float(d.mean()), ci=[lo, hi], p_signflip=sl.signflip_p(d),
                oracle_f1_mean=float(t.f1_oracle_subject_threshold.mean()),
                population_f1_mean=float(t.f1_population_youden.mean()),
                n_nonresponders_auc_lt_0_8=int((~t["responder_auc_ge_0.8"]).sum()),
                spearman_hr_effect_vs_LR_f1=[float(rho.statistic), float(rho.pvalue)])
    json.dump(summ, open(f"{out}/e3_e4_summary.json", "w"), indent=1)
    print("E3/E4", summ)


# ---------------------------------------------------------------------------------------------- E5
def causal_smooth(g, score, k):
    t = g.t_start.to_numpy()
    seg = np.r_[0, np.cumsum(np.abs(np.diff(t) - sl.STEP_S) > 1e-6)]
    return pd.Series(score).groupby(seg).transform(lambda x: x.rolling(k, min_periods=1).mean()).to_numpy()


def e5(df, oof, out, names):
    rows = []
    for name in names:
        for k in SMOOTH_K:
            for s, g in df.groupby("subject"):
                g = g.sort_values("t_start"); o = oof.loc[g.index]
                sm = causal_smooth(g, o[f"score_{name}"].to_numpy(), k)
                pred = (sm > o["youden_t"].to_numpy()).astype(int) if name == "Youden_HR" else (sm > 0.5).astype(int)
                ab = sl.alarm_burden(g, pred)
                tt, yy = g.t_start.to_numpy(), g.y.to_numpy()
                ep = np.r_[0, np.cumsum((np.abs(np.diff(tt) - sl.STEP_S) > 1e-6) | (np.diff(yy) != 0))]   # contiguous runs
                delays, det = [], 0
                episodes = [e for e in np.unique(ep) if yy[ep == e][0] == 1]
                for e in episodes:
                    idx = np.where(ep == e)[0]; hit = idx[pred[idx] == 1]
                    if len(hit):
                        det += 1; delays.append(tt[hit[0]] + sl.WIN_S - tt[idx[0]])
                rows.append(dict(model=name, k=k, subject=s, macro_f1=sl.macro_f1(g.y, pred), fa_per_hour=ab["fa_per_hour"],
                                 episodes=len(episodes), detected_frac=det / max(1, len(episodes)),
                                 delay_s=float(np.median(delays)) if delays else np.nan))
    r = pd.DataFrame(rows)
    r.to_csv(f"{out}/e5_smoothing_per_subject.csv", index=False)
    s = r.groupby(["model", "k"]).agg(macro_f1=("macro_f1", "mean"), fa_per_hour=("fa_per_hour", "mean"),
                                      detected=("detected_frac", "mean"), median_delay_s=("delay_s", "median")).reset_index()
    s.to_csv(f"{out}/e5_smoothing_summary.csv", index=False)
    print(s.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- E6
def e6(df, cols, out, reps, seed=0):
    rng = np.random.default_rng(seed)
    models = mk_models(cols, seed)
    subs = sorted(df.subject.unique()); rows = []
    for s in subs:
        te = df[df.subject == s]
        for n in LC_SIZES:
            for rep in range(reps):
                pick = rng.choice([o for o in subs if o != s], n, replace=False)
                tr = df[df.subject.isin(pick)]
                if tr.y.nunique() < 2:
                    continue
                for name, proto in models.items():
                    sc, pr = fit_score(proto, tr, tr.y.to_numpy(), tr.subject.to_numpy(), te)
                    rows.append(dict(subject=s, n_train=n, rep=rep, model=name, **sl.subj_metrics(te.y, sc, pr)))
        print(f"E6 subject {s} done", flush=True)
    r = pd.DataFrame(rows)
    ps = r.groupby(["model", "n_train", "subject"])[["macro_f1", "auroc"]].mean().reset_index()   # average reps per subject first
    ps.to_csv(f"{out}/e6_learning_curve_per_subject.csv", index=False)
    rows2 = []
    for (m, n), g in ps.groupby(["model", "n_train"]):
        lo, hi = sl.boot_ci(g.macro_f1)
        rows2.append(dict(model=m, n_train=n, macro_f1=g.macro_f1.mean(), ci_lo=lo, ci_hi=hi, auroc=g.auroc.mean()))
    pd.DataFrame(rows2).to_csv(f"{out}/e6_learning_curve.csv", index=False)
    print(pd.DataFrame(rows2).round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- E7
def e7(df, cols, out):
    if "label_raw" not in df:
        print("E7 skipped: no label_raw column"); return
    rows = []
    def run(sub_df, ycol, labels, tag):
        for s in sorted(sub_df.subject.unique()):
            tr, te = sub_df[sub_df.subject != s], sub_df[sub_df.subject == s]
            sc = StandardScaler().fit(tr[cols])
            for name, est in (("LR", LogisticRegression(max_iter=3000, class_weight="balanced")),
                              ("GB", GradientBoostingClassifier(random_state=0))):
                from sklearn.utils.class_weight import compute_sample_weight
                if name == "GB":
                    est.fit(sc.transform(tr[cols]), tr[ycol], sample_weight=compute_sample_weight("balanced", tr[ycol]))
                else:
                    est.fit(sc.transform(tr[cols]), tr[ycol])
                p = est.predict(sc.transform(te[cols]))
                rows.append(dict(protocol=tag, model=name, subject=s,
                                 macro_f1=f1_score(te[ycol], p, average="macro", labels=labels, zero_division=0),
                                 acc=accuracy_score(te[ycol], p), bal_acc=balanced_accuracy_score(te[ycol], p)))
    d = df.copy()
    run(d, "y", [0, 1], "binary: baseline+amusement vs stress")
    b = d[d.label_raw.isin([1, 2])].copy(); b["yb"] = (b.label_raw == 2).astype(int)
    run(b, "yb", [0, 1], "binary: baseline vs stress")
    t = d[d.label_raw.isin([1, 2, 3])].copy()
    run(t, "label_raw", [1, 2, 3], "3-class: baseline/stress/amusement")
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e7_protocols_per_subject.csv", index=False)
    s = r.groupby(["protocol", "model"]).agg(**{m: (m, "mean") for m in ("macro_f1", "acc", "bal_acc")}).reset_index()
    s.to_csv(f"{out}/e7_protocols.csv", index=False); print(s.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- E8
def e8(path, out, ref="Stacking"):
    p = pd.read_csv(path); w = p.pivot(index="subject", columns="model", values="macro_f1"); n = len(w)
    k = tdist.ppf(.975, n - 1) + tdist.ppf(.80, n - 1)
    rows = [dict(vs=ref, other=c, sd_paired_diff=(w[ref] - w[c]).std(ddof=1), mde_80pct_power=k * (w[ref] - w[c]).std(ddof=1) / np.sqrt(n))
            for c in w.columns if c != ref]
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e8_power.csv", index=False); print(r.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs_revised/wesad_features.csv")
    ap.add_argument("--cols", nargs="+", default=sl.BIO)
    ap.add_argument("--out", default="outputs_extra")
    ap.add_argument("--only", nargs="+", default=["e1", "e2", "e3", "e5", "e6", "e7", "e8"])
    ap.add_argument("--tag", default="", help="free-text label stored in run_info.json (e.g. replication)")
    ap.add_argument("--n-cals", type=int, nargs="+", default=N_CALS)
    ap.add_argument("--models", nargs="+", default=["LR", "GB"], help="subset of LR GB (Stacking via --stacking)")
    ap.add_argument("--stacking", action="store_true")
    ap.add_argument("--lc-reps", type=int, default=10)
    ap.add_argument("--loso-per-subject", default="outputs_revised/loso_per_subject.csv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore"); os.makedirs(a.out, exist_ok=True)
    df = sl.load_features(a.features).dropna(subset=a.cols).reset_index(drop=True)
    cols = a.cols
    json.dump(dict(tag=a.tag, features=a.features, cols=cols, n_subjects=int(df.subject.nunique()), n_windows=int(len(df))), open(f"{a.out}/run_info.json", "w"))
    print(f"{df.subject.nunique()} subjects, {len(df)} windows, features {cols}")
    models = {k: v for k, v in mk_models(cols, 0, a.stacking).items() if k in a.models or k == 'Stacking'}
    sub = e1(df, cols, a.out) if "e1" in a.only or "e3" in a.only else None
    if "e2" in a.only: e2(df, cols, a.out, models, a.n_cals)
    oof = None
    if any(x in a.only for x in ("e3", "e5")):
        oof = loso_scores(df, cols, models); oof.to_csv(f"{a.out}/base_oof.csv", index=False)
    if "e3" in a.only: e3_e4(df, cols, a.out, oof, sub)
    if "e5" in a.only: e5(df, oof, a.out, ["Youden_HR"] + list(models))
    if "e6" in a.only: e6(df, cols, a.out, a.lc_reps)
    if "e7" in a.only: e7(df, cols, a.out)
    if "e8" in a.only and os.path.exists(a.loso_per_subject): e8(a.loso_per_subject, a.out)


if __name__ == "__main__":
    main()
