"""Walk-forward predictor: forward-return *edge* + winner *confidence*.

Two heads, both run through the SAME expanding-window walk-forward loop:

  * edge head      -> a per-name score used for SELECTION (ranking)
  * confidence head -> P(this stock beats the cross-sectional median next month)
                       used for position SIZING

THE NO-LOOKAHEAD CONTRACT (read carefully — this is the auditable core):

A training "sample" is one (month s, ticker) row. Its features come from data
available at month-end s. Its target is the return s -> s+1, realised only at
month-end s+1. To predict month t we may train on a sample at month s ONLY IF
s + 1 <= t  (i.e. s < t). So: train on every row with date < t, predict the
rows at date == t. The model never touches a return at or after t.

The classifier label is also lookahead-free: a sample's "winner" label compares
its own realised target to the median of targets *in that same past month s*,
which is fully known by t.

THREE INTERCHANGEABLE METHODS (``method=`` argument), all honouring the contract:

  * "gbm"        : GradientBoosting regressor (edge) + classifier (confidence).
                   The original; predicts a pointwise forward return.
  * "lambdarank" : LightGBM LGBMRanker (objective=lambdarank) optimises the
                   per-month RANKING directly + an LGBMClassifier for confidence.
                   The edge "score" is ordinal within a month (use the order, not
                   the magnitude). Usually the better fit for cross-sectional
                   selection.
  * "mlp"        : sklearn MLPRegressor (edge) + MLPClassifier (confidence), each
                   behind a StandardScaler. Experimental — small/regularised and
                   seed-unstable; expect it to overfit at current breadth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

import config as C

METHODS = ("gbm", "lambdarank", "mlp")


def _winner_labels(train: pd.DataFrame) -> np.ndarray:
    """1 if a sample's target beats the cross-sectional median of its own month.

    Computed per training month, so it uses only that month's realised returns
    — no information from the prediction month leaks in.
    """
    med = train.groupby(level=0)["target"].transform("median")
    return (train["target"] > med).astype(int).to_numpy()


def _grades(train: pd.DataFrame, k: int) -> np.ndarray:
    """Per-month quantile buckets of the target -> integer relevance grades 0..k-1.

    Higher grade = higher realised forward return that month. Used as the
    lambdarank relevance label. Degenerate months (too few names / ties) -> 0.
    """
    def _q(s: pd.Series) -> pd.Series:
        try:
            g = pd.qcut(s, k, labels=False, duplicates="drop")
        except ValueError:
            g = pd.Series(0, index=s.index)
        return g.fillna(0).astype(int)
    return train.groupby(level=0)["target"].transform(_q).to_numpy()


def _params(base: dict, seed: int | None) -> dict:
    p = dict(base)
    if seed is not None:
        p["random_state"] = seed
    return p


def _confidence_proba(clf, X_cur, y_win) -> np.ndarray:
    """Run a fitted classifier's P(class==1); neutral 0.5 for degenerate months."""
    if len(np.unique(y_win)) < 2:
        return np.full(X_cur.shape[0], 0.5)
    return clf.predict_proba(X_cur)[:, list(clf.classes_).index(1)]


# --- per-month fit functions (pure; run in worker processes) ----------------

def _fit_gbm(Xtr, y_ret, y_win, Xcur, seed):
    p = _params(C.GBM_PARAMS, seed)
    reg = GradientBoostingRegressor(**p)
    reg.fit(Xtr, y_ret)
    pred = reg.predict(Xcur)
    if len(np.unique(y_win)) < 2:
        conf = np.full(Xcur.shape[0], 0.5)
    else:
        clf = GradientBoostingClassifier(**p)
        clf.fit(Xtr, y_win)
        conf = _confidence_proba(clf, Xcur, y_win)
    return pred, conf


def _fit_lambdarank(Xtr, grades, y_win, group, Xcur, seed):
    p = _params(C.LGBM_PARAMS, seed)
    k = int(C.RANK_GRADES)
    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        label_gain=[2 ** i - 1 for i in range(k)],
        n_jobs=1, verbose=-1, **p)
    ranker.fit(Xtr, grades, group=group)
    pred = ranker.predict(Xcur)  # ordinal score within the month; higher=better
    if len(np.unique(y_win)) < 2:
        conf = np.full(Xcur.shape[0], 0.5)
    else:
        clf = lgb.LGBMClassifier(n_jobs=1, verbose=-1, **p)
        clf.fit(Xtr, y_win)
        conf = _confidence_proba(clf, Xcur, y_win)
    return pred, conf


def _fit_mlp(Xtr, y_ret, y_win, Xcur, seed):
    """Seed-ENSEMBLED MLP: average MLP_SEEDS independently-seeded nets to kill the
    single-seed variance that otherwise makes the neural net unreliable.
    StandardScaler is refit inside each pipeline on train rows only (no leakage).
    """
    base = seed if seed is not None else C.MLP_PARAMS.get("random_state", 42)
    n = max(1, int(getattr(C, "MLP_SEEDS", 1)))

    preds = []
    for s in range(n):
        p = _params(C.MLP_PARAMS, base + 1000 * s)
        reg = make_pipeline(StandardScaler(), MLPRegressor(**p))
        reg.fit(Xtr, y_ret)
        preds.append(reg.predict(Xcur))
    pred = np.mean(preds, axis=0)

    if len(np.unique(y_win)) < 2:
        conf = np.full(Xcur.shape[0], 0.5)
    else:
        probs = []
        for s in range(n):
            p = _params(C.MLP_PARAMS, base + 1000 * s)
            clf = make_pipeline(StandardScaler(), MLPClassifier(**p))
            clf.fit(Xtr, y_win)
            m = clf.named_steps["mlpclassifier"]
            probs.append(clf.predict_proba(Xcur)[:, list(m.classes_).index(1)])
        conf = np.mean(probs, axis=0)
    return pred, conf


def _run_task(task: dict):
    """Dispatch one month's fit to the chosen method. Returns [(idx, pred, conf)]."""
    method, idx, seed = task["method"], task["idx"], task["seed"]
    if method == "lambdarank":
        pred, conf = _fit_lambdarank(task["Xtr"], task["grades"], task["y_win"],
                                     task["group"], task["Xcur"], seed)
    elif method == "mlp":
        pred, conf = _fit_mlp(task["Xtr"], task["y_ret"], task["y_win"],
                              task["Xcur"], seed)
    else:
        pred, conf = _fit_gbm(task["Xtr"], task["y_ret"], task["y_win"],
                              task["Xcur"], seed)
    return [(i, float(p), float(c)) for i, p, c in zip(idx, pred, conf)]


def walk_forward_predict(samples: pd.DataFrame, feature_cols: list[str],
                         method: str = "gbm",
                         seed: int | None = None) -> pd.DataFrame:
    """Expanding-window walk-forward; returns predicted edge + confidence.

    Parameters
    ----------
    samples : DataFrame indexed by (date, ticker) with the feature columns plus
        a ``target`` column (forward 1-month return).
    feature_cols : columns fed to the model(s).
    method : "gbm" | "lambdarank" | "mlp" (see module docstring).
    seed : overrides the method's ``random_state`` (used by retrain.py to re-fit
        the same data into a different model). None => the config default.

    Returns
    -------
    DataFrame indexed by (date, ticker) with columns ``pred`` (edge / ranking
    score) and ``confidence`` (probability in [0,1]), one row per scored
    (decision month, ticker) from MIN_TRAIN_MONTHS onward.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    df = samples.dropna(subset=feature_cols + ["target"]).copy().sort_index(level=0)
    feat_only = samples.dropna(subset=feature_cols).copy()  # predict rows may lack target

    # Decision months come from the FEATURE frame, not the target-filtered one:
    # the most recent month-end has a NaN forward-return target (it hasn't
    # happened yet), so it's absent from df — but it's exactly the month the live
    # trader must act on. Training still uses only df rows dated < t (below), so
    # predicting this targetless month introduces no lookahead.
    months = sorted(feat_only.index.get_level_values(0).unique())
    df_dates = df.index.get_level_values(0)
    fo_dates = feat_only.index.get_level_values(0)

    def _tasks():
        """Yield one pre-sliced task per decision month (lazy => bounded memory).

        Each task's training slice uses ONLY rows with date < t, so parallel
        execution can't break the no-lookahead contract.
        """
        for t in months:
            train = df[df_dates < t]
            if train.index.get_level_values(0).nunique() < C.MIN_TRAIN_MONTHS:
                continue  # not enough history yet
            cur = feat_only[fo_dates == t]
            if cur.empty:
                continue
            task = {
                "method": method, "seed": seed,
                "Xtr": train[feature_cols].to_numpy(),
                "Xcur": cur[feature_cols].to_numpy(),
                "idx": list(cur.index),
                "y_ret": train["target"].to_numpy(),
                "y_win": _winner_labels(train),
            }
            if method == "lambdarank":
                # rows are sorted by date (level 0) => groups are contiguous.
                task["group"] = train.groupby(level=0).size().tolist()
                task["grades"] = _grades(train, int(C.RANK_GRADES))
            yield task

    # Fan the independent per-month fits across all cores. Generator + lazy
    # dispatch keeps only a few training slices materialised at a time.
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_run_task)(task) for task in _tasks()
    )

    rows: dict[tuple, dict] = {}
    for res in results:
        for idx, p, c in res:
            rows[idx] = {"pred": p, "confidence": c}

    out = pd.DataFrame.from_dict(rows, orient="index")
    if not out.empty:
        out.index = pd.MultiIndex.from_tuples(out.index, names=["date", "ticker"])
    return out
