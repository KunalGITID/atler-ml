"""Train the models ATLER ships and write them into an ATLER checkout.

    uv run python -m atler_ml.export subscriptions    # -> src/core/import/subscriptionModel.json
    uv run python -m atler_ml.export category-prior   # -> src/core/categoryPrior.json

Training uses seeds the benchmark never evaluates on (TRAIN_SEEDS), so the
numbers in results/ are for statements the shipped model hasn't seen. After
writing, the model is run through ATLER's TypeScript and checked against
scikit-learn's own probabilities.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import bridge, recurring, synth

TRAIN_SEEDS = [101, 102, 103]


def _round(x: float) -> float:
    return float(f"{x:.6g}")


def export_trees(m) -> dict:
    """sklearn GradientBoostingClassifier (binary) -> the TreeModel JSON statement.ts reads."""
    p = m.init_.class_prior_[1]
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        leaf = t.children_left == -1
        trees.append({
            "feature": np.where(leaf, -1, t.feature).tolist(),
            "threshold": [_round(v) if not lf else 0 for v, lf in zip(t.threshold, leaf)],
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "value": [_round(v) for v in t.value[:, 0, 0]],
        })
    return {"init": _round(float(np.log(p / (1 - p)))), "rate": m.learning_rate, "trees": trees}


def subscriptions() -> None:
    import pandas as pd
    cands = pd.concat([recurring.candidates(synth.generate(users=40, seed=s)).assign(user=lambda d, s=s: d["user"] + 1000 * s)
                       for s in TRAIN_SEEDS], ignore_index=True)
    cols = recurring.feature_cols(cands)
    m = recurring.model().fit(cands[cols], cands["label"])
    names = bridge.run([{"task": "features"}])[0]
    out = {"features": names, **export_trees(m)}
    path = bridge.atler_dir() / "src/core/import/subscriptionModel.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"{len(cands)} candidates ({cands['label'].mean():.0%} subscriptions) -> {path} ({path.stat().st_size / 1024:.0f} KB)")

    # Same answers from TypeScript as from scikit-learn?
    sample = cands.sample(300, random_state=0)
    ts = bridge.run([{"task": "treeProbability", "x": sample[cols].to_numpy().tolist()}])[0]
    gap = np.abs(np.array(ts) - m.predict_proba(sample[cols])[:, 1]).max()
    assert gap < 1e-4, f"TypeScript and sklearn disagree by {gap}"
    print(f"TypeScript matches sklearn (max difference {gap:.1e})")


def category_prior() -> None:
    import pandas as pd

    from . import prior
    df = pd.concat([synth.generate(users=40, seed=s).assign(user=lambda d, s=s: d["user"] + 1000 * s) for s in TRAIN_SEEDS],
                   ignore_index=True)
    f, lr = prior.train(df, keep=1000)
    path = bridge.atler_dir() / "src/core/categoryPrior.json"
    path.write_text(json.dumps(prior.export(f, lr), separators=(",", ":")) + "\n")
    print(f"{len(df)} lines, {len(f.index)} features -> {path} ({path.stat().st_size / 1024:.0f} KB)")

    # Same answers from TypeScript? (Weights are rounded to 3 decimals in the JSON, so allow a little.)
    texts = df["description"].sample(300, random_state=0).tolist()
    ts = bridge.run([{"task": "priorProbabilities", "texts": texts}])[0]
    py = lr.predict_proba(f.transform(texts))
    known = [i for i, t in enumerate(ts) if t is not None]
    gap = np.abs(np.array([ts[i] for i in known]) - py[known]).max()
    agree = (np.array([ts[i] for i in known]).argmax(1) == py[known].argmax(1)).mean()
    assert gap < 0.02 and agree == 1, f"TypeScript and sklearn disagree (max {gap}, top-1 agreement {agree})"
    print(f"TypeScript matches sklearn on {len(known)} lines (max difference {gap:.1e}, same top class on all)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["subscriptions", "category-prior"])
    args = ap.parse_args()
    {"subscriptions": subscriptions, "category-prior": category_prior}[args.model]()


if __name__ == "__main__":
    main()
