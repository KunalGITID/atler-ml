"""ATLER's TypeScript core, called from Python. Needs Node 23+ and an ATLER checkout."""
import shutil

import pytest

from atler_ml import bridge

pytestmark = pytest.mark.skipif(not shutil.which("node") or not bridge.atler_dir().exists(), reason="needs node + ATLER")


def test_classify_and_keywords():
    nb, kw = bridge.run([
        {"task": "classify", "train": [{"name": "SWIGGY order", "categoryId": "Food"},
                                        {"name": "UBER trip", "categoryId": "Transport"}], "test": ["SWIGGY*BLR 8823"]},
        {"task": "keywords", "names": ["UPI/DR/1/ZOMATO/HDFC", "something else"]},
    ])
    assert nb[0]["categoryId"] == "Food"
    assert kw == ["Food", None]


def test_recurring_finds_netflix():
    debits = [{"on": f"2026-0{m}-05", "description": f"NACH/NETFLIX/{m}", "amount": 19900} for m in range(1, 7)]
    out = bridge.run([{"task": "recurring", "debits": debits, "today": "2026-06-20"}])[0]
    assert out["found"][0]["name"] == "Netflix"
    assert out["found"][0]["cycle"] == {"unit": "month", "every": 1}
