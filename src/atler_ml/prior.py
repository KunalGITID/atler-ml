"""The category prior ATLER ships: logistic regression over character n-grams.

Trained on other people's statements, so suggestions work from your first
expense. The featurizer is tiny on purpose so it can be written the same way in
TypeScript (src/core/categoryPrior.ts): lowercase, letters only, then for each
word "w:word" plus the 3-5 character n-grams of " word ".

Only features seen in 3+ users' statements are kept: the prior should know
"swiggy" and "medicals", not one person's corner shop (that is what your own
filing is for).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

MIN_USERS = 3


def tokens(text: str) -> list[str]:
    out = []
    for w in re.sub(r"[^a-z]+", " ", text.lower()).split():
        if len(w) < 2:
            continue
        out.append("w:" + w)
        padded = f" {w} "
        for n in (3, 4, 5):
            out += [padded[i:i + n] for i in range(len(padded) - n + 1)]
    return out


class Featurizer:
    """TF-IDF (sublinear tf, smooth idf, L2) over `tokens`, written out so TypeScript can match it exactly."""

    def fit(self, texts: list[str], users: list[int]) -> Featurizer:
        seen: dict[str, set[int]] = defaultdict(set)
        df: Counter[str] = Counter()
        for t, u in zip(texts, users):
            toks = set(tokens(t))
            df.update(toks)
            for tok in toks:
                seen[tok].add(u)
        vocab = sorted(t for t, us in seen.items() if len(us) >= MIN_USERS)
        self.index = {t: i for i, t in enumerate(vocab)}
        n = len(texts)
        self.idf = np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in vocab])
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            counts = Counter(i for tok in tokens(t) if (i := self.index.get(tok)) is not None)
            if not counts:
                continue
            idx = np.fromiter(counts, dtype=int)
            v = (1 + np.log(np.fromiter(counts.values(), dtype=float))) * self.idf[idx]
            v /= np.linalg.norm(v)
            rows += [r] * len(idx)
            cols += idx.tolist()
            vals += v.tolist()
        return csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.index)))


def train(df: pd.DataFrame, keep: int = 4000) -> tuple[Featurizer, LogisticRegression]:
    """Fit, then keep only the `keep` features that matter most (it ships to phones) and refit."""
    texts, users, y = df["description"].tolist(), df["user"].tolist(), df["category"]
    f = Featurizer().fit(texts, users)
    lr = LogisticRegression(C=10, max_iter=3000).fit(f.transform(texts), y)
    strongest = np.argsort(-np.abs(lr.coef_).max(axis=0))[:keep]
    chosen = set(strongest.tolist())
    vocab = sorted(t for t, i in f.index.items() if i in chosen)
    f.index = {t: i for i, t in enumerate(vocab)}
    f.idf = _idf_for(vocab, texts)
    lr = LogisticRegression(C=10, max_iter=3000).fit(f.transform(texts), y)
    return f, lr


def _idf_for(vocab: list[str], texts: list[str]) -> np.ndarray:
    df: Counter[str] = Counter()
    keep = set(vocab)
    for t in texts:
        df.update(set(tokens(t)) & keep)
    n = len(texts)
    return np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in vocab])


def export(f: Featurizer, lr: LogisticRegression) -> dict:
    """{classes, bias[c], features: {token: [idf, w_c0, w_c1, ...]}} with weights rounded to 3 decimals."""
    vocab = sorted(f.index, key=f.index.get)
    return {
        "classes": lr.classes_.tolist(),
        "bias": [round(float(b), 3) for b in lr.intercept_],
        "features": {t: [round(float(f.idf[i]), 3), *[round(float(w), 3) for w in lr.coef_[:, i]]]
                     for t, i in ((t, f.index[t]) for t in vocab)},
    }
