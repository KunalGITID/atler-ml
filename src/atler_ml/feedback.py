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


def simulate(df: pd.DataFrame, names: pd.Series, answer_rate: float, learn: bool, review_from: str | None = None) -> pd.Series:
    users = sorted(df["user"].unique())
    def payments(r: pd.DataFrame) -> list[dict]:
        shown_as = names[r.index].fillna(r["description"])
        return [{"id": i, "name": n, "amount": int(a), "on": d.strftime("%Y-%m-%d"), "categoryId": c, "anomaly": bool(x)}
                for i, n, a, d, c, x in zip(r["id"], shown_as, r["amount"], r["on"], r["category"], r["anomaly"])]

    jobs = [{"task": "feedback", "answerRate": answer_rate, "learn": learn, "payments": payments(df[df["user"] == u]),
             **({"reviewFrom": review_from} if review_from else {})} for u in users]
    return pd.Series([t for out in bridge.run(jobs) for t in out], index=df.index, dtype=float)


def main() -> None:
    rows = []
    for seed in [int(s) for s in os.environ.get("SEEDS", "7,11").split(",")]:
        df = synth.generate(users=40, seed=seed)
        names, _ = recurring.atler(df)
        # The first 3 months arrive as an imported statement; alerts are live after that.
        start = df["on"].min() + pd.Timedelta(days=92)
        live = df["on"] >= start
        quarter = ((df["on"] - df["on"].min()).dt.days // 92 + 1).clip(upper=4).map(lambda q: f"Q{q}")
        for label, rate, learn, review in (("no answers (2x)", 0.0, False, False),
                                           ("answers 50%", 0.5, True, False), ("answers 50% + review", 0.5, True, True),
                                           ("answers 70%", 0.7, True, False), ("answers 70% + review", 0.7, True, True),
                                           ("answers 90%", 0.9, True, False), ("answers 90% + review", 0.9, True, True)):
            shown = simulate(df, names, rate, learn, start.strftime("%Y-%m-%d") if review else None).notna() & live
            for q in ("Q2", "Q3", "Q4"):
                m = quarter == q
                rows.append({"seed": seed, "user": label, "quarter": q, "precision": df.loc[m & shown, "anomaly"].mean(),
                             "recall": (m & shown & df["anomaly"]).sum() / (m & df["anomaly"]).sum(),
                             "alerts": (m & shown).sum() / df["user"].nunique()})
    out = pd.DataFrame(rows).groupby(["user", "quarter"], sort=False)[["precision", "recall", "alerts"]].mean()
    print(out.round(3).unstack("quarter").to_string())


if __name__ == "__main__":
    main()
