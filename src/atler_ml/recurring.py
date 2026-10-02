"""Task 2: which charges in a statement are subscriptions?

- atler: findRecurring (core/import/statement.ts) as it is in ATLER_DIR.
- rules: ATLER's original rhythm rules (before the model; numbers in results/).
- gbm: gradient boosting over ATLER's own candidates and features
  (recurringCandidates / seriesFeatures, called through the bridge, so there's
  one implementation), trained on other users (GroupKFold by user).

A detection counts when most of its charges belong to one real subscription
not already found. Recall is over subscriptions with 3+ charges (2+ yearly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from . import bridge

TRUE_CYCLES = {"month": {("month", 1)}, "year": {("year", 1)}, "28d": {("day", 28)},
               "91d": {("month", 3), ("day", 91), ("day", 84)}}
RHYTHMS = {7: ("day", 7), 14: ("day", 14), 28: ("day", 28), 30.4: ("month", 1), 56: ("day", 56),
           84: ("day", 84), 91.3: ("month", 3), 182.6: ("month", 6), 365: ("year", 1)}
MATCH = {7: 1, 14: 1, 28: 1, 30.4: 2.5, 56: 2, 84: 3, 91.3: 4, 182.6: 7, 365: 12}


def model() -> GradientBoostingClassifier:
    # Small on purpose: it ships to phones as JSON.
    return GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=0)


def _debits(rows: pd.DataFrame) -> list[dict]:
    return [{"on": d.strftime("%Y-%m-%d"), "description": s, "amount": int(a)}
            for d, s, a in zip(rows["on"], rows["description"], rows["amount"])]


def atler(df: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """ATLER's merchant name for every row, and what findRecurring found per user."""
    users = sorted(df["user"].unique())
    today = df["on"].max().strftime("%Y-%m-%d")
    out = bridge.run([{"task": "recurring", "today": today, "debits": _debits(df[df["user"] == u])} for u in users])
    names = pd.Series(index=df.index, dtype=object)
    found = []
    for u, res in zip(users, out):
        names.loc[df.index[df["user"] == u]] = res["merchants"]
        found += [{**f, "user": u} for f in res["found"]]
    return names, found


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """ATLER's candidate series with its features, labelled from the ground truth."""
    users = sorted(df["user"].unique())
    today = df["on"].max().strftime("%Y-%m-%d")
    frames = [df[df["user"] == u] for u in users]
    out = bridge.run([{"task": "candidates", "today": today, "debits": _debits(f)} for f in frames])
    rows = []
    for u, f, cands in zip(users, frames, out):
        for c in cands:
            idx = f.index[c["rows"]]
            subs = df.loc[idx, "sub_id"].dropna()
            label = bool(len(subs)) and (subs == subs.mode().iloc[0]).sum() / len(idx) >= 0.6
            rows.append({"user": u, "merchant": c["name"], "whole": c["whole"], "rows": idx, "label": label,
                         **dict(enumerate(c["features"]))})
    return pd.DataFrame(rows)


def feature_cols(cands: pd.DataFrame) -> list[int]:
    return [c for c in cands.columns if isinstance(c, int)]


def _cycle(gap: float) -> tuple[str, int]:
    exact = [r for r in RHYTHMS if abs(gap - r) <= MATCH[r]]
    return RHYTHMS[exact[0] if exact else min(RHYTHMS, key=lambda r: abs(gap - r))]


def pick(cands: pd.DataFrame, sure: float = 0.5) -> pd.DataFrame:
    """One answer per merchant, as findRecurring does: the whole series if it scores, else its amounts that do."""
    picked = []
    for _, g in cands.groupby(["user", "merchant"]):
        whole = g[g["whole"] & (g["p"] >= sure)]
        picked += [whole.iloc[0]] if len(whole) else [r for _, r in g[~g["whole"] & (g["p"] >= sure)].iterrows()]
    return pd.DataFrame([{"user": r["user"], "rows": r["rows"], "cycle": _cycle(r[1]), "score": r["p"]} for r in picked])


def run(df: pd.DataFrame, folds: int = 5) -> dict:
    names, found = atler(df)
    subs = df.dropna(subset=["sub_id"]).groupby("sub_id")
    eligible = {s for s, g in subs if len(g) >= 3 or (len(g) == 2 and g["cycle"].iloc[0] == "year")}
    cycle_of = df.dropna(subset=["sub_id"]).groupby("sub_id")["cycle"].first().to_dict()

    atler_res = _summarise(atler_detections(df, names, found), df, eligible, cycle_of)

    cands = candidates(df)
    cols = feature_cols(cands)
    cands["p"] = 0.0
    for tr, te in GroupKFold(n_splits=folds).split(cands, groups=cands["user"]):
        m = model().fit(cands.iloc[tr][cols], cands.iloc[tr]["label"])
        cands.loc[cands.index[te], "p"] = m.predict_proba(cands.iloc[te][cols])[:, 1]
    gbm_res = _summarise(pick(cands), df, eligible, cycle_of)
    return {"atler": atler_res, "gbm": gbm_res, "eligible": len(eligible), "candidates": cands, "names": names}


def atler_detections(df: pd.DataFrame, names: pd.Series, found: list[dict]) -> pd.DataFrame:
    rows = []
    for f in found:
        u = f["user"]
        g = df[(df["user"] == u) & (names == f["name"])]
        # ATLER may have split this merchant by amount: keep rows near the
        # price unless the series rose in steps (then take all of them).
        if f["priceChange"] is None:
            g = g[(g["amount"] - f["price"]).abs() <= np.maximum(200, 0.1 * g["amount"])]
        rows.append({"user": u, "rows": g.index, "cycle": (f["cycle"]["unit"], f["cycle"]["every"]),
                     "score": f["confidence"] / 100})
    return pd.DataFrame(rows)


def _summarise(det: pd.DataFrame, df: pd.DataFrame, eligible: set[str], cycle_of: dict) -> dict:
    if det.empty:
        return {"precision": 0, "recall": 0, "f1": 0, "cycle_accuracy": 0, "detections": 0, "false_positives": 0}
    matched: set[str] = set()
    hits = []
    for rows in det["rows"]:
        subs = df.loc[rows, "sub_id"].dropna()
        top = subs.mode().iloc[0] if len(subs) else None
        ok = top is not None and (subs == top).sum() / len(rows) >= 0.6 and top not in matched
        if ok:
            matched.add(top)
        hits.append(top if ok else None)
    det = det.assign(sub_id=hits)
    tp = det.dropna(subset=["sub_id"])
    p = len(tp) / len(det)
    r = len(matched & eligible) / len(eligible)
    cyc = np.mean([c in TRUE_CYCLES[cycle_of[s]] for s, c in zip(tp["sub_id"], tp["cycle"])]) if len(tp) else 0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0,
            "cycle_accuracy": float(cyc), "detections": len(det), "false_positives": len(det) - len(tp)}
