# atler-ml

[![CI](https://github.com/KunalGITID/atler-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/KunalGITID/atler-ml/actions/workflows/ci.yml)

**Does machine learning beat the hand-written rules in [ATLER](https://github.com/KunalGITID/ATLER)?**

ATLER is my expense and subscription tracker. It runs entirely on the phone and has three "smart" features,
each built from heuristics:

| Feature | What ATLER does today | File in ATLER |
|---|---|---|
| Category suggestions | regex keyword hints, then a per-user Naive Bayes trained on what you've filed | `src/core/suggest.ts`, `classify.ts` |
| Subscription finder | groups a bank statement by merchant, matches the median gap to a billing rhythm | `src/core/import/statement.ts` |
| Unusual-spend alerts | Tukey's fence (Q3 + 1.5 IQR) on earlier spends in the same category | `src/core/insights.ts` |

This repo benchmarks each one against scikit-learn models on the same data, with the same rules about what the
model may see. The ATLER side is not a Python re-implementation: `bridge/atler.ts` imports ATLER's real
TypeScript core and runs it under Node, so the numbers describe the shipped code.

## Results

40 synthetic users, one year each, 35,371 transactions. Full tables: [`results/RESULTS.md`](results/RESULTS.md).

![accuracy vs number of filed expenses](results/categorise.png)

**1. Category suggestions.** Every user files their first *k* expenses; the model guesses the rest.

| accuracy (%) | k = 10 | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|
| ATLER keywords | 34.6 | 34.7 | 34.7 | 34.8 | 35.1 |
| ATLER Naive Bayes | 41.4 | 59.3 | 72.0 | 82.9 | 91.1 |
| ATLER chain (NB, else keywords) | 49.6 | 58.8 | 69.5 | 81.3 | 88.8 |
| TF-IDF + logistic regression | 43.9 | 63.6 | 77.5 | 88.4 | 95.6 |
| prior trained on *other* users | 87.8 | 87.8 | 87.9 | 87.9 | 88.0 |
| **prior + your filings** | **88.1** | **92.1** | **93.3** | **92.3** | **96.5** |

- On-phone learning needs about 100 labels to become useful, and most people give up before filing 100 expenses.
- A model trained on other people's statements works from the first expense, because Swiggy is Swiggy for
  everyone. Blending it with your own filings (weight *k* / (*k* + 20)) keeps the day-one accuracy and still
  learns *your* corner shop, which no one else's data contains.
- Character n-grams beat ATLER's word tokens because they cope with `SWIGGY*BLR`, `BUNDLTECHNOLOGIES` and
  similar mangled names.

**2. Subscription finder** (154 real recurring series)

| | precision | recall | F1 | cycle correct |
|---|---:|---:|---:|---:|
| ATLER rules | 76.6 | 85.1 | 80.6 | **96.9** |
| gradient boosting on 12 series features | **85.1** | **96.1** | **90.2** | 90.5 |

The model finds more of the subscriptions and makes fewer false alarms. ATLER's rhythm table is still better at
naming the billing cycle, so the two combine well: the model decides *whether* a series is a subscription, and
ATLER's rules decide its cycle.

**3. Unusual spending** (277 planted anomalies, 4 to 10 times a merchant's usual amount)

| | precision | recall | F1 | PR-AUC | flagged |
|---|---:|---:|---:|---:|---:|
| ATLER (by category) | 10.4 | **84.5** | 18.5 | 14.4 | 2,254 |
| ATLER's rule, by merchant | 16.1 | 59.6 | 25.4 | **45.2** | 1,022 |
| robust z (category + merchant) | 19.1 | 61.7 | 29.1 | 15.4 | 897 |
| Isolation Forest | **37.8** | 36.8 | **37.3** | 35.1 | 270 |

This is the clearest problem found. Grouped by category, ATLER flags about 56 expenses per user per year and 9
in 10 of those alerts are false. A big DMart shop looks "unusual" next to small corner-shop runs, although it
is ordinary for DMart. Comparing against the same merchant triples the ranking quality (PR-AUC 14 → 45).

## What goes back into ATLER

- [ ] Unusual spend: compare against the same merchant once it has 5+ past spends, and fall back to the
      category before that. This is a small change in `insights.ts`.
- [ ] Category suggestions: switch the tokens to character n-grams and ship a small prior (a few hundred KB of
      logistic-regression weights as JSON) so suggestions work from the first expense. Nothing leaves the phone.
- [ ] Subscription finder: add a scored "is this a subscription" step in front of the rhythm table.

## Caveats

- **The data is synthetic.** ATLER keeps money data on the phone and real statements are private, so
  `src/atler_ml/synth.py` generates statements in the style ATLER imports: UPI, POS and NACH narrations, legal
  names that differ from brand names (Bundl = Swiggy, Eternal = Zomato), posting delays, price rises,
  cancelled plans, and local shops unique to each user. Each result depends on these assumptions. In particular:
  - The prior's lead depends on how much users' merchants overlap.
  - The anomaly numbers depend on how much everyday amounts vary.
- The anomalies are *planted*, so "recall" means recall of this one kind of anomaly.
- One seed, one run. Variance across seeds isn't reported yet.

## Run it

Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 23 or newer, and an ATLER checkout (by default
next to this repo, or wherever `ATLER_DIR` points).

```sh
git clone https://github.com/KunalGITID/ATLER ../ATLER
uv sync
uv run pytest
uv run python -m atler_ml.bench          # full run, about 2 min; writes results/
uv run python -m atler_ml.bench --quick  # 8 users, as in CI
```

## Layout

```
bridge/atler.ts          runs ATLER's TypeScript core on JSON jobs
src/atler_ml/synth.py    synthetic statements + ground truth
src/atler_ml/categorise.py, recurring.py, anomaly.py   one file per task
src/atler_ml/bench.py    runs everything, writes results/RESULTS.md
```

## Next

- Forecasting: compare ATLER's 6-month mean against seasonal-naive and ETS, with backtested interval coverage.
- Results across 5 seeds, with confidence intervals.
- Model card for the shipped category prior.
