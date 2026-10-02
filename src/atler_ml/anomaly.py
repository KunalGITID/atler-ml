"""Task 3: which expenses are unusual?

Planted anomalies are ordinary merchants at 4-10x their usual amount (1% of
everyday spends). Every method only looks at the past: an expense is judged
against earlier ones, as it would be on the day it happened.

- atler: unusualness (core/insights.ts): Tukey's fence on earlier spends at the
  same merchant (by category before ATLER#28), at least 2x the median,
  Rs 200 or more, 5+ earlier spends.
- robust-z: log amount against the median/MAD of earlier amounts in the same
  category AND at the same merchant; flagged when either z > 3.5.
- iforest: Isolation Forest over the same "how far from usual" features,
  per user, flagging its top 1%. Unsupervised: it never sees the labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from . import bridge

MIN_HISTORY = 5
Z = 3.5


def _past_z(log_amounts: np.ndarray) -> np.ndarray:
    """For each position, robust z of the value against the ones before it (nan with <5)."""
    z = np.full(len(log_amounts), np.nan)
    for i in range(MIN_HISTORY, len(log_amounts)):
        past = log_amounts[:i]
        med = np.median(past)
        mad = 1.4826 * np.median(np.abs(past - med))
        z[i] = (log_amounts[i] - med) / max(mad, 0.15)  # floor: a merchant you pay the same every time
    return z


def run(df: pd.DataFrame, merchants: pd.Series) -> pd.DataFrame:
    df = df.assign(merchant_name=merchants.fillna(df["description"]), log=np.log(df["amount"] / 100))
    users = sorted(df["user"].unique())
    assert df["user"].is_monotonic_increasing  # so the batched answers line up with the rows
    jobs = [{"task": "unusual", "payments": [
        {"id": i, "name": n, "amount": int(a), "on": d.strftime("%Y-%m-%d"), "categoryId": c}
        for i, n, a, d, c in zip(r["id"], r["merchant_name"], r["amount"], r["on"], r["category"])]}
        for u in users for r in [df[df["user"] == u]]]
    df["atler_score"] = pd.Series([t for out in bridge.run(jobs) for t in out], index=df.index, dtype=float).fillna(0)

    df["z_cat"] = np.nan
    df["z_merchant"] = np.nan
    for _, g in df.groupby(["user", "category"]):
        df.loc[g.index, "z_cat"] = _past_z(g["log"].to_numpy())
    for _, g in df.groupby(["user", "merchant_name"]):
        df.loc[g.index, "z_merchant"] = _past_z(g["log"].to_numpy())
    df["z_score"] = df[["z_cat", "z_merchant"]].max(axis=1).fillna(0)

    df["iforest_score"] = 0.0
    for _, g in df.groupby("user"):
        x = g[["log", "z_cat", "z_merchant"]].fillna(0).to_numpy()
        model = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(x)
        # Only "too big" is unusual here, not "too small".
        df.loc[g.index, "iforest_score"] = np.where(g["z_merchant"].fillna(g["z_cat"]).fillna(0) > 0,
                                                    -model.score_samples(x), 0)
        df.loc[g.index, "iforest_flag"] = (model.predict(x) == -1) & (df.loc[g.index, "iforest_score"] > 0)
    return df


THRESHOLDS = [2, 2.5, 3, 3.5, 4]


def threshold_curve(df: pd.DataFrame) -> pd.DataFrame:
    """ATLER's alerts if the "times the usual" bar were higher than 2x."""
    flagged, n, users = df[df["atler_score"] > 0], df["anomaly"].sum(), df["user"].nunique()
    return pd.DataFrame([{"at least": f"{t}x", "precision": flagged.loc[flagged["atler_score"] >= t, "anomaly"].mean(),
                          "recall": flagged.loc[flagged["atler_score"] >= t, "anomaly"].sum() / n,
                          "alerts per user per year": round((flagged["atler_score"] >= t).sum() / users, 1)}
                         for t in THRESHOLDS])


def score(df: pd.DataFrame) -> pd.DataFrame:
    flags = {"atler": df["atler_score"] > 0,
             "robust-z": df["z_score"] > Z, "iforest": df["iforest_flag"].astype(bool)}
    scores = {"atler": df["atler_score"],
              "robust-z": df["z_score"], "iforest": df["iforest_score"]}
    y = df["anomaly"]
    return pd.DataFrame([{
        "method": m,
        "precision": precision_score(y, flags[m], zero_division=0),
        "recall": recall_score(y, flags[m]),
        "f1": f1_score(y, flags[m]),
        "pr_auc": average_precision_score(y, scores[m]),
        "flagged": int(flags[m].sum()),
    } for m in flags])
