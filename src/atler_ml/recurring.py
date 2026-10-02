"""Task 2: which charges in a statement are subscriptions?

- atler: findRecurring (core/import/statement.ts): group by merchant name,
  match the median gap to a billing rhythm, check gaps and amounts are steady.
- gbm: the same candidates (merchant groups, split by amount like ATLER does),
  described by a dozen features and scored by gradient boosting trained on
  other users (GroupKFold by user, so a user is never in their own training set).

A detection counts when most of its charges belong to one real subscription
not already found. Recall is over subscriptions with 3+ charges (2+ yearly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from . import bridge

RHYTHMS = {7: ("day", 7), 14: ("day", 14), 28: ("day", 28), 30.4: ("month", 1), 56: ("day", 56),
           84: ("day", 84), 91.3: ("month", 3), 182.6: ("month", 6), 365: ("year", 1)}
TRUE_CYCLES = {"month": {("month", 1)}, "year": {("year", 1)}, "28d": {("day", 28)},
               "91d": {("month", 3), ("day", 91), ("day", 84)}}
FEATURES = ["n", "median_gap", "gap_iqr_ratio", "gap_regular", "rhythm_dist", "amount_cv", "amount_steady",
            "amount_steps", "log_amount", "dom_spread", "span_ratio", "days_since_last_ratio"]


def atler(df: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """ATLER's merchant name for every row, and what findRecurring found per user."""
    users = sorted(df["user"].unique())
    jobs = [{"task": "recurring", "today": df["on"].max().strftime("%Y-%m-%d"),
             "debits": [{"on": d.strftime("%Y-%m-%d"), "description": s, "amount": int(a)}
                        for d, s, a in zip(r["on"], r["description"], r["amount"])]}
            for u in users for r in [df[df["user"] == u]]]
    out = bridge.run(jobs)
    names = pd.Series(index=df.index, dtype=object)
    found = []
    for u, res in zip(users, out):
        idx = df.index[df["user"] == u]
        names.loc[idx] = res["merchants"]
        found += [{**f, "user": u} for f in res["found"]]
    return names, found


def _match(cands: list[tuple[int, pd.Index]], df: pd.DataFrame, eligible: set[str]) -> list[dict]:
    """For each detected candidate (user, rows): TP if 60%+ of its rows are one unmatched subscription."""
    matched: set[str] = set()
    out = []
    for _, rows in cands:
        subs = df.loc[rows, "sub_id"].dropna()
        top = subs.mode().iloc[0] if len(subs) else None
        share = (subs == top).sum() / len(rows) if top else 0
        ok = top is not None and share >= 0.6 and top not in matched
        if ok:
            matched.add(top)
        out.append({"sub_id": top if ok else None, "tp": ok})
    return out


def atler_detections(df: pd.DataFrame, names: pd.Series, found: list[dict]) -> pd.DataFrame:
    rows = []
    for f in found:
        u = f["user"]
        mask = (df["user"] == u) & (names == f["name"])
        # ATLER may have split this merchant by amount: keep rows near the
        # price unless the series rose in steps (then take all of them).
        g = df[mask]
        if f["priceChange"] is None:
            g = g[(g["amount"] - f["price"]).abs() <= np.maximum(200, 0.1 * g["amount"])]
        rows.append({"user": u, "rows": g.index, "cycle": (f["cycle"]["unit"], f["cycle"]["every"]),
                     "score": f["confidence"] / 100})
    return pd.DataFrame(rows)


def _features(g: pd.DataFrame, today: pd.Timestamp) -> dict:
    on = g["on"].sort_values()
    gaps = on.diff().dt.days.dropna().to_numpy()
    amounts = g["amount"].to_numpy(dtype=float)
    med = float(np.median(gaps)) if len(gaps) else 0
    near = min(RHYTHMS, key=lambda r: abs(med - r))
    steps = int(np.sum(np.abs(np.diff(amounts)) > np.maximum(200, 0.1 * amounts[1:]))) if len(amounts) > 1 else 0
    dom = on.dt.day.to_numpy()
    span = (on.iloc[-1] - on.iloc[0]).days or 1
    return {
        "n": len(g), "median_gap": med,
        "gap_iqr_ratio": float(np.subtract(*np.percentile(gaps, [75, 25])) / med) if med else 9,
        "gap_regular": float(np.mean(np.abs(gaps - med) <= max(2, 0.1 * med))) if len(gaps) else 0,
        "rhythm_dist": abs(med - near) / near,
        "amount_cv": float(amounts.std() / amounts.mean()),
        "amount_steady": float(np.mean(np.abs(amounts - np.median(amounts)) <= np.maximum(200, 0.1 * amounts))),
        "amount_steps": steps,
        "log_amount": float(np.log(np.median(amounts))),
        "dom_spread": float(np.std(dom)),
        "span_ratio": span / max(med, 1) / max(len(g) - 1, 1),
        "days_since_last_ratio": (today - on.iloc[-1]).days / max(med, 1),
    }


def _nearest_cycle(gap: float) -> tuple[str, int]:
    return RHYTHMS[min(RHYTHMS, key=lambda r: abs(gap - r))]


def candidates(df: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    """Merchant groups, plus amount buckets inside each (two plans at one merchant)."""
    today = df["on"].max()
    out = []
    for (u, name), g in df.assign(merchant=names).dropna(subset=["merchant"]).groupby(["user", "merchant"]):
        groups = [g]
        buckets: list[list[int]] = []
        for i, a in zip(g.index, g["amount"]):
            b = next((b for b in buckets if abs(a - np.median(df.loc[b, "amount"])) <= max(200, 0.1 * a)), None)
            (b.append(i) if b is not None else buckets.append([i]))
        if len(buckets) > 1:
            groups += [g.loc[b] for b in buckets]
        for gg in groups:
            if len(gg) < 2:
                continue
            f = _features(gg, today)
            if len(gg) == 2 and abs(f["median_gap"] - 365) > 15:
                continue
            subs = gg["sub_id"].dropna()
            label = bool(len(subs)) and (subs == subs.mode().iloc[0]).sum() / len(gg) >= 0.6
            out.append({"user": u, "merchant": name, "rows": gg.index, "whole": gg is g, "label": label, **f})
    return pd.DataFrame(out)


def run(df: pd.DataFrame, folds: int = 5) -> dict:
    names, found = atler(df)
    subs = df.dropna(subset=["sub_id"]).groupby("sub_id")
    eligible = {s for s, g in subs if len(g) >= 3 or (len(g) == 2 and g["cycle"].iloc[0] == "year")}
    cycle_of = df.dropna(subset=["sub_id"]).groupby("sub_id")["cycle"].first().to_dict()

    # ATLER
    atler_res = _summarise(atler_detections(df, names, found), df, eligible, cycle_of)

    # GBM, out-of-fold by user
    cands = candidates(df, names)
    cands["p"] = 0.0
    for tr, te in GroupKFold(n_splits=folds).split(cands, groups=cands["user"]):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        model.fit(cands.iloc[tr][FEATURES], cands.iloc[tr]["label"])
        cands.loc[cands.index[te], "p"] = model.predict_proba(cands.iloc[te][FEATURES])[:, 1]
    # One answer per merchant: the whole group if it scores, else its buckets that do.
    picked = []
    for _, g in cands.groupby(["user", "merchant"]):
        whole = g[g["whole"]]
        if len(whole) and whole["p"].iloc[0] >= 0.5:
            picked.append(whole.iloc[0])
        else:
            picked += [r for _, r in g[~g["whole"] & (g["p"] >= 0.5)].iterrows()]
    gbm = pd.DataFrame([{"user": r["user"], "rows": r["rows"], "cycle": _nearest_cycle(r["median_gap"]), "score": r["p"]}
                        for r in picked])
    gbm_res = _summarise(gbm, df, eligible, cycle_of)
    return {"atler": atler_res, "gbm": gbm_res, "eligible": len(eligible), "candidates": cands, "names": names}


def _summarise(det: pd.DataFrame, df: pd.DataFrame, eligible: set[str], cycle_of: dict) -> dict:
    if det.empty:
        return {"precision": 0, "recall": 0, "f1": 0, "cycle_accuracy": 0, "detections": 0}
    m = pd.DataFrame(_match(list(zip(det["user"], det["rows"])), df, eligible))
    det = pd.concat([det.reset_index(drop=True), m], axis=1)
    tp = det[det["tp"]]
    found = set(tp["sub_id"]) & eligible
    p = len(tp) / len(det)
    r = len(found) / len(eligible)
    cyc = np.mean([c in TRUE_CYCLES[cycle_of[s]] for s, c in zip(tp["sub_id"], tp["cycle"])]) if len(tp) else 0
    fps = det[~det["tp"]]
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0,
            "cycle_accuracy": float(cyc), "detections": len(det), "false_positives": len(fps)}
