"""Does answering unusual-spend alerts make them better? (ATLER's alertFeedback.ts)

Simulated users answer a share of the alerts they're shown, truthfully, and
ATLER raises its bars from those answers. Precision is reported per quarter,
since learning needs answers to accumulate. The answers are the only thing
the learning sees: it never sees how the anomalies were planted.

    uv run python -m atler_ml.feedback
"""

from __future__ import annotations

import os

import pandas as pd

from . import bridge, recurring, synth


def simulate(df: pd.DataFrame, names: pd.Series, answer_rate: float, learn: bool) -> pd.Series:
    users = sorted(df["user"].unique())
    jobs = [{"task": "feedback", "answerRate": answer_rate, "learn": learn, "payments": [
        {"id": i, "name": n, "amount": int(a), "on": d.strftime("%Y-%m-%d"), "categoryId": c, "anomaly": bool(x)}
        for i, n, a, d, c, x in zip(r["id"], names[r.index].fillna(r["description"]), r["amount"], r["on"], r["category"], r["anomaly"])]}
        for u in users for r in [df[df["user"] == u]]]
    return pd.Series([t for out in bridge.run(jobs) for t in out], index=df.index, dtype=float)


def main() -> None:
    rows = []
    for seed in [int(s) for s in os.environ.get("SEEDS", "7,11").split(",")]:
        df = synth.generate(users=40, seed=seed)
        names, _ = recurring.atler(df)
        for label, rate, learn in (("no answers (2x)", 0.0, False), ("answers 30% of alerts", 0.3, True),
                                   ("answers 70% of alerts", 0.7, True)):
            shown = simulate(df, names, rate, learn).notna()
            q = df["on"].dt.quarter.map({4: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"})  # the year starts in October
            for quarter in ("Q1", "Q2", "Q3", "Q4"):
                m = q == quarter
                rows.append({"seed": seed, "user": label, "quarter": quarter,
                             "precision": df.loc[m & shown, "anomaly"].mean(),
                             "recall": (m & shown & df["anomaly"]).sum() / (m & df["anomaly"]).sum(),
                             "alerts": (m & shown).sum() / df["user"].nunique()})
    out = pd.DataFrame(rows).groupby(["user", "quarter"], sort=False)[["precision", "recall", "alerts"]].mean()
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
