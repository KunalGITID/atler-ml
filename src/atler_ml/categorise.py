"""Task 1: which category does a statement line belong to?

The setting is ATLER's: you've filed your first k expenses yourself, and the
app guesses the rest. Everything is per user and chronological (train on your
first k, test on everything after), because that's what the phone sees.

Methods
- keywords: ATLER's regex hints (import/sms.ts suggestCategory). No learning.
- atler-nb: ATLER's on-phone Naive Bayes (core/classify.ts) on your k.
- atler-chain: NB when it's >= 60% sure, else keywords (core/suggest.ts before the prior).
- atler-suggest: ATLER's real suggestCategoryId as it is in ATLER_DIR, with the
  shipped category prior when that checkout has one.
- tfidf-lr: char n-gram TF-IDF + logistic regression on your k.
- prior+you: a model trained on *other* users (shipped with the app, so it
  works on day one) blended with tfidf-lr on yours, weight k / (k + 20).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline, make_union

from . import bridge

KS = [10, 25, 50, 100, 200]
SURE = 0.6
SHRINK = 20


def _pipeline() -> object:
    features = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z]{2,}", lowercase=True),
    )
    return make_pipeline(features, LogisticRegression(C=10, max_iter=2000))


def _proba(model, names: list[str], classes: list[str]) -> np.ndarray:
    """Probabilities over `classes`, in that order (0 for classes the model never saw)."""
    out = np.zeros((len(names), len(classes)))
    p = model.predict_proba(names)
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = p[:, j]
    return out


def run(df: pd.DataFrame, ks: list[int] = KS, folds: int = 5) -> pd.DataFrame:
    classes = sorted(df["category"].unique())
    users = sorted(df["user"].unique())

    # The shipped prior: for each user, a model that never saw them.
    prior: dict[int, object] = {}
    for train_idx, test_idx in GroupKFold(n_splits=min(folds, len(users))).split(df, groups=df["user"]):
        model = _pipeline().fit(df["description"].iloc[train_idx].tolist(), df["category"].iloc[train_idx])
        for u in df["user"].iloc[test_idx].unique():
            prior[u] = model

    jobs, meta = [], []
    for u in users:
        rows = df[df["user"] == u]
        for k in ks:
            if len(rows) <= k + 20:
                continue
            train, test = rows.iloc[:k], rows.iloc[k:]
            jobs.append({"task": "classify",
                         "train": [{"name": n, "categoryId": c} for n, c in zip(train["description"], train["category"])],
                         "test": test["description"].tolist()})
            meta.append((u, k, train, test))
    nb = bridge.run(jobs)
    has_prior = (bridge.atler_dir() / "src/core/categoryPrior.json").exists()
    classes_ = sorted(df["category"].unique())
    suggested = bridge.run([{"task": "suggest", "categories": classes_, "prior": has_prior, "test": test["description"].tolist(),
                             "filed": [{"name": n, "category": c} for n, c in zip(train["description"], train["category"])]}
                            for (_, _, train, test) in meta])
    kw_names = df["description"].tolist()
    keywords = dict(zip(df["id"], bridge.run([{"task": "keywords", "names": kw_names}])[0]))

    preds: list[pd.DataFrame] = []
    for (u, k, train, test), nb_out, sug in zip(meta, nb, suggested):
        names = test["description"].tolist()
        kw = [keywords[i] for i in test["id"]]
        atler_nb = [p["categoryId"] if p else None for p in nb_out]
        chain = [p["categoryId"] if p and p["confidence"] >= SURE else h for p, h in zip(nb_out, kw)]

        p_prior = _proba(prior[u], names, classes)
        if train["category"].nunique() >= 2:
            mine = _pipeline().fit(train["description"].tolist(), train["category"])
            p_mine = _proba(mine, names, classes)
        else:  # one category so far: that's all it can say
            p_mine = np.zeros_like(p_prior)
            p_mine[:, classes.index(train["category"].iloc[0])] = 1
        w = k / (k + SHRINK)
        blended = w * p_mine + (1 - w) * p_prior

        preds.append(pd.DataFrame({
            "user": u, "k": k, "truth": test["category"].to_numpy(),
            "keywords": kw, "atler-nb": atler_nb, "atler-chain": chain, "atler-suggest": sug,
            "tfidf-lr": np.array(classes)[p_mine.argmax(1)],
            "prior+you": np.array(classes)[blended.argmax(1)],
            "prior-only": np.array(classes)[p_prior.argmax(1)],
        }))
    return pd.concat(preds, ignore_index=True)


METHODS = ["keywords", "atler-nb", "atler-chain", "atler-suggest", "tfidf-lr", "prior-only", "prior+you"]


def score(preds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and macro-F1 per method and k (no guess counts as wrong)."""
    rows = []
    for k, g in preds.groupby("k"):
        for m in METHODS:
            guess = g[m].fillna("(none)")
            rows.append({"k": k, "method": m,
                         "accuracy": accuracy_score(g["truth"], guess),
                         "macro_f1": f1_score(g["truth"], guess, average="macro",
                                              labels=sorted(g["truth"].unique()), zero_division=0),
                         "answered": g[m].notna().mean()})
    return pd.DataFrame(rows)
