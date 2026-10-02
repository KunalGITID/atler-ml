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

The tables show 40 synthetic users, one year each (seed 7). Full tables are in [`results/RESULTS.md`](results/RESULTS.md).
"ATLER" here means current `main`, including the changes that came out of this repo
([#28](https://github.com/KunalGITID/ATLER/pull/28)–[#31](https://github.com/KunalGITID/ATLER/pull/31) and
[#32](https://github.com/KunalGITID/ATLER/pull/32)). The shipped models were trained on seeds 101–103 and never saw
these statements.

**Generator v2 (October 2026).** Each merchant now has its own spread: metro fares and fees barely move, while Amazon
and IRCTC swing a lot. Before v2, every merchant varied by the same amount, which made unusual spending look easier
to catch than it is. The before/after numbers in ATLER #28–#31 were measured on v1.

![accuracy vs number of filed expenses](results/categorise.png)

**1. Category suggestions** ([#30](https://github.com/KunalGITID/ATLER/pull/30)). Every user files their first *k*
expenses, and the model guesses the rest.

| accuracy (%) | k = 10 | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|
| before #30 (Naive Bayes, else keywords) | 53.7 | 62.6 | 71.5 | 81.6 | 89.7 |
| **ATLER (shipped prior + your Naive Bayes)** | **87.9** | **88.4** | **88.6** | **91.6** | **95.4** |
| prior + your own logistic regression (best here) | 89.7 | 94.7 | 91.1 | 91.6 | 95.7 |

The prior is trained on other people's statements, so it works from the first expense; Swiggy is Swiggy for
everyone. It only keeps character n-gram features seen in 3+ users' statements. Your filing gets weight
*k* / (*k* + 150). It ships as 1,000 features (81 KB of JSON, 15.7 KB gzipped).

**2. Subscription finder** ([#29](https://github.com/KunalGITID/ATLER/pull/29)), over 154 real recurring series

| | precision | recall | F1 | cycle correct |
|---|---:|---:|---:|---:|
| **ATLER (name fix + shipped tree model)** | **96.7** | **96.1** | **96.4** | **100** |
| the same model, trained here out-of-fold | 94.3 | 96.8 | 95.5 | 100 |

On v1, the rules ATLER had before #29 scored F1 81–85. `merchantName` kept bank words (`DR`, `Paid`, `PUR`), which
split one payee into several. The model is 80 depth-3 trees over 11 regularity features, with no price and no
merchant, shipped as 30 KB of JSON.

**3. Unusual spending** (planted anomalies, 4 to 10 times a merchant's typical amount)

With no answers, ATLER flags spends at least 2× your usual at the same place
([#28](https://github.com/KunalGITID/ATLER/pull/28)). That is 34 alerts per user per year, and only 16% are real.
A higher fixed bar doesn't rescue it on v2: 4× gives 48% precision, and fitting it to the planted range would be
circular anyway. So ATLER learns the bar from **your answers**
([#31](https://github.com/KunalGITID/ATLER/pull/31)). It asks "Expected?" on the alert card and right after you add
an unusual expense, and offers a one-off review of your 5 biggest past jumps
([#32](https://github.com/KunalGITID/ATLER/pull/32)).

`feedback.py` runs ATLER's own `unusualness`, `learnBars` and `pastJumps` with simulated users who answer a share of
alerts truthfully. The first 3 months arrive as an imported statement; alerts are live after that. Seeds 7 and 11:

| precision | Q2 | Q3 | Q4 | recall Q4 | alerts per user, Q4 |
|---|---:|---:|---:|---:|---:|
| no answers (2×) | 15% | 14% | 19% | 83% | 9.4 |
| answers 50% of alerts | 20% | 33% | 53% | 54% | 2.2 |
| answers 50% + review of past jumps | 38% | 40% | 60% | 50% | 1.8 |
| answers 70% of alerts | 38% | 60% | 66% | 41% | 1.3 |
| **answers 70% + review of past jumps** | **45%** | **65%** | **71%** | 44% | 1.3 |
| answers 90% + review of past jumps | 45% | 65% | 72% | 43% | 1.3 |

The biggest lever is how many alerts get answered, which is why ATLER asks right after you add a spend. The review
lifts the first months. The cost is recall: fewer alerts, and some real ones are no longer shown.

**What didn't help (on this data), so it isn't in ATLER:**
- **A per-user model with more signals** (z-score against the place's own spread, history length, share of the
  month's spending). Real anomalies are rare, so after a year most users still don't have the dozen-plus answers
  with 3+ of each kind needed to fit even a 4-feature logistic regression. It never switched on.
- **Variance-aware gates** (Iglewicz–Hoaglin modified z ≥ 3.5, or z ≥ 2 on top of the learned bar). No better
  frontier than plain "times your usual" here, because the anomalies are *planted* as multiples of a place's typical
  amount. Real anomalies may behave differently; only real answers can show that (see **Next**).
- Learning rules for the bar: per-threshold precision counting and a 1-D logistic fit both lost to the midpoint rule.

## Caveats

- **The data is synthetic.** ATLER keeps money data on the phone and real statements are private, so
  `src/atler_ml/synth.py` generates statements in the style ATLER imports: UPI, POS and NACH narrations, legal
  names that differ from brand names (Bundl = Swiggy, Eternal = Zomato), posting delays, price rises,
  cancelled plans, and local shops unique to each user. The shipped models are trained on other seeds of the
  same generator, so every "after" number is in-distribution, and real-world numbers will be lower. In particular:
  - The category prior only knows the brands in the generator, plus generic words like *medicals* and *stores*.
  - The anomaly numbers depend on how much everyday amounts vary.
- The anomalies are *planted*, so "recall" means recall of this one kind of anomaly.
- The tables use seed 7. The subscription and suggestion results were re-checked on seeds 11 and 23, and the
  feedback simulation uses seeds 7 and 11.
- The simulated users answer perfectly truthfully. Real ones won't.

## Run it

Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 23 or newer, and an ATLER checkout (by default
next to this repo, or wherever `ATLER_DIR` points).

```sh
git clone https://github.com/KunalGITID/ATLER ../ATLER
uv sync
uv run pytest
uv run python -m atler_ml.bench          # full run, about 2 min; writes results/
uv run python -m atler_ml.bench --quick  # 8 users, as in CI
uv run python -m atler_ml.export subscriptions   # retrain + write ATLER's subscriptionModel.json
uv run python -m atler_ml.export category-prior  # retrain + write ATLER's categoryPrior.json
uv run python -m atler_ml.feedback               # answering alerts, by quarter
```

## Layout

```
bridge/atler.ts          runs ATLER's TypeScript core on JSON jobs
src/atler_ml/synth.py    synthetic statements + ground truth
src/atler_ml/categorise.py, recurring.py, anomaly.py   one file per task
src/atler_ml/prior.py    the category prior's featurizer (mirrored in ATLER's categoryPrior.ts)
src/atler_ml/feedback.py simulates people answering unusual-spend alerts
src/atler_ml/export.py   trains the models ATLER ships and writes them into an ATLER checkout
src/atler_ml/bench.py    runs everything, writes results/RESULTS.md
```

## Next

- **Real answers, opt-in.** ATLER's You screen shows "N of the M you answered were really unusual". An opt-in,
  anonymous share of (times your usual, history length, the place's spread, answer) would check this simulation
  against reality, re-test the variance-aware ideas that didn't help on synthetic data, and give new users a
  starting bar learned from real people.
- Forecasting: compare ATLER's 6-month mean against seasonal-naive and ETS, with backtested interval coverage.
- Results across 5 seeds, with confidence intervals.
- Model card for the shipped category prior.
