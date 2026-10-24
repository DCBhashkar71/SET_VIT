"""
sensitivity_fairness.py - sensitivity to class weighting and to hyper-parameter tuning.

    python sensitivity_fairness.py --features outputs_revised\\wesad_features.csv --out outputs_sens

A  Gradient boosting WITHOUT class weighting  vs WITH balanced sample weights
   (as now): per-subject macro-F1, stress recall, AUROC, paired difference with CI, so the direction and size of the
   old asymmetry is quantified.
B  Hyper-parameter tuning: default settings (what the main tables use) vs settings chosen by an INNER subject-grouped
   CV on the training subjects only (nested within LOSO). Chosen settings per fold are saved.
All comparisons: LOSO, per-subject, bootstrap CI over subjects, exact sign-flip p (Holm within each table).
"""
import argparse, copy, itertools, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import stresslib as sl

LR_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
GB_GRID = [{"max_depth": d, "n_estimators": n} for d in (2, 3, 4) for n in (50, 100, 200)]


def mk(kind, params, seed=0):
    if kind == "LR":
        return sl.Skl(LogisticRegression(max_iter=2000, class_weight="balanced", **params))
    return sl.Skl(GradientBoostingClassifier(random_state=seed, **params), sw=True)


class Tuned:
    """Inner subject-grouped CV (4 folds) over a small grid, maximising macro-F1 on the training subjects only."""
    def __init__(self, kind, grid):
        self.kind, self.grid = kind, grid

    def fit(self, X, y, g):
        y, g = np.asarray(y), np.asarray(g)
        best, self.best = -1, None
        cv = GroupKFold(n_splits=min(4, len(np.unique(g))))
        for p in self.grid:
            f = []
            for tr, va in cv.split(X, y, g):
                m = mk(self.kind, p).fit(X.iloc[tr], y[tr])
                f.append(sl.macro_f1(y[va], m.predict(X.iloc[va])))
            if np.mean(f) > best:
                best, self.best = np.mean(f), p
        self.m = mk(self.kind, self.best).fit(X, y)
        return self

    def score(self, X): return self.m.score(X)
    def predict(self, X): return self.m.predict(X)


def run(df, models, chosen=None):
    rows = []
    for s in sorted(df.subject.unique()):
        tr, te = df.subject != s, df.subject == s
        for name, proto in models.items():
            m = copy.deepcopy(proto).fit(df[tr], df.loc[tr, "y"].to_numpy(), df.loc[tr, "subject"].to_numpy())
            d = sl.subj_metrics(df.loc[te, "y"], m.score(df[te]), m.predict(df[te]))
            d.update(subject=s, model=name)
            rows.append(d)
            if chosen is not None and hasattr(m, "best"):
                chosen.append(dict(subject=s, model=name, **m.best))
        print("subject", s, "done", flush=True)
    return pd.DataFrame(rows)


def paired(ps, pairs, out, tag):
    tabs = []
    for metric in ("macro_f1", "rec_stress", "auroc"):
        w = ps.pivot(index="subject", columns="model", values=metric)
        tabs.append(sl.paired_table(w, metric, 0.05, pairs))
    t = pd.concat(tabs); t.to_csv(f"{out}/{tag}_paired.csv", index=False)
    print(t[["metric", "A", "B", "mean_diff", "ci_lo", "ci_hi", "p_signflip", "p_holm"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs_revised/wesad_features.csv"); ap.add_argument("--out", default="outputs_sens")
    ap.add_argument("--cols", nargs="+", default=sl.BIO)
    a = ap.parse_args(); warnings.filterwarnings("ignore"); os.makedirs(a.out, exist_ok=True)
    df = sl.load_features(a.features).dropna(subset=a.cols).reset_index(drop=True)
    # A: class-weighting asymmetry
    A = {"GB_unweighted": sl.Skl(GradientBoostingClassifier(random_state=0), cols=a.cols, sw=False),
         "GB_weighted": sl.Skl(GradientBoostingClassifier(random_state=0), cols=a.cols, sw=True)}
    psA = run(df, A); psA.to_csv(f"{a.out}/A_gb_weighting_per_subject.csv", index=False)
    print("\n== A: GB weighted - unweighted"); paired(psA, [("GB_weighted", "GB_unweighted")], a.out, "A_gb_weighting")
    # B: tuning
    B = {"LR_default": mk("LR", {}), "LR_tuned": Tuned("LR", LR_GRID), "GB_default": mk("GB", {}), "GB_tuned": Tuned("GB", GB_GRID)}
    chosen = []
    psB = run(df, B, chosen); psB.to_csv(f"{a.out}/B_tuning_per_subject.csv", index=False)
    pd.DataFrame(chosen).to_csv(f"{a.out}/B_chosen_hyperparameters.csv", index=False)
    print("\n== B: tuned - default"); paired(psB, [("LR_tuned", "LR_default"), ("GB_tuned", "GB_default")], a.out, "B_tuning")
    print(pd.DataFrame(chosen).groupby("model").agg(lambda s: s.value_counts().to_dict()).T)


if __name__ == "__main__":
    main()
