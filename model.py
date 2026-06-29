"""Walk-forward predictor: forward-return *edge* + winner *confidence*.

Two heads, both run through the SAME expanding-window walk-forward loop:

  * regressor  -> predicted forward 1-month return ("edge", used for selection)
  * classifier -> P(this stock beats the cross-sectional median next month)
                  ("confidence", used for position sizing)

THE NO-LOOKAHEAD CONTRACT (read carefully — this is the auditable core):

A training "sample" is one (month s, ticker) row. Its features come from data
available at month-end s. Its target is the return s -> s+1, realised only at
month-end s+1. To predict month t we may train on a sample at month s ONLY IF
s + 1 <= t  (i.e. s < t). So: train on every row with date < t, predict the
rows at date == t. The model never touches a return at or after t.

The classifier label is also lookahead-free: a sample's "winner" label compares
its own realised target to the median of targets *in that same past month s*,
which is fully known by t.
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import config as C


def _winner_labels(train: pd.DataFrame) -> pd.Series:
    """1 if a sample's target beats the cross-sectional median of its own month.

    Computed per training month, so it uses only that month's realised returns
    — no information from the prediction month leaks in.
    """
    med = train.groupby(level=0)["target"].transform("median")
    return (train["target"] > med).astype(int)


def _fit_predict_month(X_train, y_ret, y_win, idx, X_cur, params):
    """Train both heads on one month's expanding window and score the candidates.

    Pure function of pre-sliced arrays so it runs in a separate process. The
    slices are built in the caller using only data with date < t, so the
    no-lookahead contract holds regardless of execution order.
    """
    reg = GradientBoostingRegressor(**params)
    reg.fit(X_train, y_ret)
    pred = reg.predict(X_cur)

    if len(set(y_win)) < 2:  # degenerate month -> neutral confidence
        conf = [0.5] * len(idx)
    else:
        clf = GradientBoostingClassifier(**params)
        clf.fit(X_train, y_win)
        conf = clf.predict_proba(X_cur)[:, list(clf.classes_).index(1)]
    return [(i, float(p), float(c)) for i, p, c in zip(idx, pred, conf)]


def walk_forward_predict(samples: pd.DataFrame, feature_cols: list[str],
                         params: dict | None = None) -> pd.DataFrame:
    """Expanding-window walk-forward; returns predicted return + confidence.

    Parameters
    ----------
    samples : DataFrame indexed by (date, ticker) with the feature columns plus
        a ``target`` column (forward 1-month return).
    feature_cols : columns fed to both models. Backtest A passes price-only
        features; Backtest B passes all seven.
    params : GBM hyper-parameters (defaults to ``config.GBM_PARAMS``). Override
        the ``random_state`` to re-train the same data into a different model —
        the basis of the re-train robustness test (see ``retrain.py``).

    Returns
    -------
    DataFrame indexed by (date, ticker) with columns ``pred`` (forward return)
    and ``confidence`` (probability in [0,1]), one row per scored
    (decision month, ticker) from MIN_TRAIN_MONTHS onward.
    """
    gbm_params = params if params is not None else C.GBM_PARAMS
    df = samples.dropna(subset=feature_cols + ["target"]).copy()
    feat_only = samples.dropna(subset=feature_cols).copy()  # predict rows may lack target

    months = sorted(df.index.get_level_values(0).unique())
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
            yield (train[feature_cols].values, train["target"].values,
                   _winner_labels(train).values, list(cur.index),
                   cur[feature_cols].values, gbm_params)

    # Fan the independent per-month fits across all cores. Generator + lazy
    # dispatch keeps only a few training slices materialised at a time.
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_fit_predict_month)(*task) for task in _tasks()
    )

    rows: dict[tuple, dict] = {}
    for res in results:
        for idx, p, c in res:
            rows[idx] = {"pred": p, "confidence": c}

    out = pd.DataFrame.from_dict(rows, orient="index")
    if not out.empty:
        out.index = pd.MultiIndex.from_tuples(out.index, names=["date", "ticker"])
    return out
