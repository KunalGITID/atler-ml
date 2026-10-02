import pandas as pd

from atler_ml import prior, synth


def test_tokens_match_the_typescript_spec():
    # the same example as ATLER's src/core/categoryPrior.test.ts
    assert prior.tokens("UPI/DR/123/Go") == ["w:upi", " up", "upi", "pi ", " upi", "upi ", " upi ",
                                             "w:dr", " dr", "dr ", " dr ", "w:go", " go", "go ", " go "]
    assert prior.tokens("a 1 *") == []


def test_a_word_only_one_user_has_never_becomes_a_feature():
    df = synth.generate(users=6, seed=4)
    mine = pd.DataFrame({"description": ["UPI/DR/1/ZQXWV STALL/HDFC"] * 30, "category": "Food", "user": 0})
    f, _ = prior.train(pd.concat([df, mine], ignore_index=True), keep=10_000)
    assert "w:zqxwv" not in f.index
    assert "w:swiggy" in f.index


def test_export_shape_and_l2_norm():
    df = synth.generate(users=6, seed=5)
    f, lr = prior.train(df, keep=300)
    out = prior.export(f, lr)
    assert len(f.index) <= 300
    assert len(out["classes"]) == len(out["bias"]) == df["category"].nunique()
    assert all(len(v) == 1 + len(out["classes"]) for v in out["features"].values())
    x = f.transform(df["description"].head(50).tolist())
    norms = x.multiply(x).sum(axis=1).A1
    assert all(abs(n - 1) < 1e-9 or n == 0 for n in norms)
