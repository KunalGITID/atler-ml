from atler_ml import synth


def test_same_seed_same_statements():
    a, b = synth.generate(users=2, seed=3), synth.generate(users=2, seed=3)
    assert a.equals(b)


def test_shape_of_the_data():
    df = synth.generate(users=3, seed=1)
    assert df["user"].is_monotonic_increasing
    assert (df["amount"] > 0).all()
    assert set(df["category"]) <= set(synth.CATEGORIES)
    assert df["anomaly"].sum() > 0
    # anomalies are planted on everyday spends only
    assert df.loc[df["anomaly"], "sub_id"].isna().all()


def test_subscriptions_keep_their_rhythm():
    df = synth.generate(users=5, seed=2)
    monthly = df[df["cycle"] == "month"].groupby("sub_id")["on"].apply(lambda s: s.diff().dt.days.dropna())
    assert monthly.between(26, 34).mean() > 0.95


def test_narrations_look_like_bank_lines():
    import random
    rng = random.Random(0)
    assert synth.narration(rng, "NETFLIX", "mandate").split()[0].split("/")[0] in {"NACH", "SI", "ACH", "NETFLIX"}
    assert "UPI" in synth.narration(rng, "ZEPTO", "upi")
