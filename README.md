# SENTINEL — Intelligent Merchant & Product Risk Engine

**TVS Credit E.P.I.C 8.0 · Problem Statement (g)**

Forecasts merchant deterioration *before* it hits the book, and converts every score into
a concrete financing action: exposure cap, EMI tenure cap, holdback percentage.

Built on the **real Olist Brazilian E-Commerce dataset** — 98,666 orders, 3,095 merchants,
32,951 products, 99,224 reviews, Sep 2016 – Oct 2018. No synthetic data anywhere.

---

## Headline results

**Purged walk-forward across 24 expanding-window folds**, against two baselines.

| Scorer | Mean AUC | Verdict |
|---|---|---|
| **SENTINEL** (0.30 model + 0.70 shrunk rate) | **0.6567** | wins 18/24 folds, **p = 0.0024** |
| Empirical-Bayes shrunk rate alone | 0.6451 | strong, simple |
| Raw bad rate (naive persistence) | 0.6363 | the baseline to beat |
| Gradient-boosted model alone | 0.6181 | **worst of the four** |

| Business metric (out-of-sample) | Value |
|---|---|
| **Early-warning lead time (median)** | **49 days** |
| Deteriorating merchants caught early | 38% (80 of 211) |
| **Policy capture at 20% of book declined** | **27.8%** vs 27.1% naive |
| Policy backtest | wins 36/55 snapshots, p = 0.0021 |
| Products scored / flagged for delisting | 3,998 / 79 |
| Deseasonalised demand shocks detected | 118 |
| **Multi-horizon separation (30d / 60d / 90d)** | **AUC 0.655 / 0.671 / 0.674** |

### Multi-horizon validation — early-warning vs nowcast

The score separates **better** at longer horizons. Merchant deterioration builds slowly enough to be seen months out; the 30-day window is the noisiest place to measure it.

| Horizon | Base rate | AUC | Lift @20% | Verdict |
|---|---|---|---|---|
| **30 days** | 0.195 | 0.6547 | 1.71x | Short-term nowcast |
| **60 days** | 0.164 | 0.6714 | 1.86x | Strong forward separation |
| **90 days** | 0.145 | **0.6735** | **1.90x** | **Peak separation — true early warning** |

### Abstention & score uncertainty — tested and failed

Every score carries a 95% interval derived from the binomial standard error of the merchant's underlying bad rate, damped by empirical-Bayes shrinkage. We tested refusing to score merchants whose interval spans two bands to improve precision:

| Population | Share | AUC | Lift @20% | Mean width | Verdict |
|---|---|---|---|---|---|
| **All** | 100% | **0.6547** | **1.71x** | 0.395 | Full cohort benchmark |
| Confident | 46.7% | 0.6386 | 1.61x | 0.310 | **Slightly worse than full book — rejected as gate** |
| Abstained | 53.3% | 0.6025 | 1.47x | 0.470 | Thin evidence (spans multiple bands) |

*Decision*: We do **not** gate on confidence. We surface the interval on the merchant file as context and report the failed gate openly.

### Segment stability — beats naive across all sizes

| Segment | Median orders | SENTINEL AUC | Naive AUC | Margin |
|---|---|---|---|---|
| **Small** | 15 | **0.6358** | 0.6154 | **+0.0204 (widest margin)** |
| **Medium** | 33 | **0.6495** | 0.6494 | +0.0001 |
| **Large** | 85 | **0.6903** | 0.6836 | +0.0067 |

### Read this before you read the numbers

Three findings shaped this project more than any modelling choice.

**1. The naive baseline is strong.** Ranking merchants by their current bad rate scores
0.636. Any early-warning system that cannot beat one column of arithmetic
does not deserve to exist. We tested it first, not last.

**2. Our first validation was wrong.** An earlier version reported AUC 0.700, winning 11 of
12 folds. That was measured without purging: training snapshots whose 30-day outcome window
overlapped the test window leaked future information. After adding the purge, the same model
scored 0.640 and won 2 of 12 — it did not beat the baseline at all. The number in this README
is the purged one.

**The in-sample lead time was also inflated.** v3 measured lead time by scoring the whole
panel, including rows the model trained on. Rebuilt on strictly out-of-sample scores
(`oos.py`), the honest figures are **49-day median lead** but only
**38% caught early** — the lead is longer than we claimed, the hit rate lower.

**The first policy backtest was measured wrong.** Comparing scorers at equal *merchant
count* flatters any scorer that prefers large merchants, because it silently declines far
more money for the same headcount. Rebuilt at equal *financed value declined*: SENTINEL
captures 27.8% of bad financed value at a 20% decline budget vs
27.1% for naive, winning 36/55 snapshots (p = 0.0021).

**3. What finally beat it was shrinkage, not the model.** A merchant with six orders at a 33%
bad rate is mostly noise. Empirical-Bayes shrinkage pulls each rate toward what that merchant's
own product-and-region mix predicts, in proportion to how little volume backs it. That alone
beats naive persistence (0.6451, p < 0.05). The gradient-boosted model earns its
place as a 30% correction on top — not as a replacement. Every blend weight from 0.2 to 0.5
beats the baseline significantly, so the choice is not knife-edge.

We also built and **rejected** an exposure-weighted target scoring AUC 0.855, because the single
column `financed_value` scored 0.858 on it. It was predicting merchant size, not risk.

**4. Multi-horizon validation proves this is an early warning system, not a nowcast.**
AUC rises monotonically from 0.6547 at 30 days to 0.6714 at 60 days and 0.6735 at 90 days.
Lift at 20% budget rises from 1.71x to 1.90x. Deterioration is visible months before it peaks.

**5. We tested gating on confidence intervals, and it failed.**
Refusing to score merchants with wide intervals drops confident AUC to 0.6386 vs 0.6547 for the full
population. We display intervals for transparency but score 100% of eligible merchants.

---

## What it does

1. **Weak-label construction.** Olist has no ground-truth fraud flag, so a bad order is
   defined as: `order_status` in (canceled, unavailable) **OR** delivered >5 days past the
   promised date **OR** review score ≤ 2. Portfolio bad-order rate = 15.2%.

2. **Rolling seller panel.** Every 14 days, aggregate each merchant's trailing 90 days into
   45 features; the target is whether their *next 30 days* exceed a 25% bad-order rate.
   5,778 merchant-snapshots across 613 merchants with enough volume to score.

3. **Feature families** — fulfilment (approval lag, carrier lag, delay p90), returns
   (cancel rate), feedback (review score, review coverage), **complaint NLP** (Portuguese
   keyword taxonomy: not-received / late / damaged / wrong-item / counterfeit / refund),
   catalogue quality (photos, description length), **EMI behaviour** (mean installments,
   % ≥6 installments, financed value), concentration (top-customer share, repeat rate),
   and **category-peer z-scores** that catch a merchant drifting from its own category norm.

4. **Blended percentile scorer.** Final score = 0.30 x percentile(model probability)
   + 0.70 x percentile(shrunk empirical-Bayes rate), with the weight chosen on a held-out
   validation slice that never touched the test set. Percentiles are taken against
   frozen reference distributions, so a single merchant can be scored in isolation
   in real time, not only in a batch. Risk bands are book positions: CRITICAL is the
   top 5% of the merchant book, HIGH the next 10%, WATCH the next 15%.

5. **Automated risk memo.** Every merchant gets an analyst-style written assessment
   naming its drivers against peer norms, complaint mix, trajectory and binding action.
   Generated deterministically from model outputs and benchmarks, so it is fully
   auditable and cannot hallucinate a number that is not in the data.

6. **Collusion graph.** Bipartite merchant ↔ `customer_unique_id` network; 44 merchant pairs
   share an abnormal number of unique customers — the fake-transaction ring signal.

7. **Action layer.** The differentiator. Score → risk band → *exposure multiplier, max EMI
   tenure, holdback %*. CRITICAL suspends new financing; HIGH caps at 25% of current exposure,
   3-month tenure, 15% holdback; WATCH caps at 60%, 6-month tenure, 7% holdback.

8. **Product risk engine** (`analytics.py`). 3,998 products scored, each shrunk toward its
   own category, ranked by exposure at risk, with a delist / restrict / review action.

9. **Seasonality and demand-shift detection.** Category share of platform volume is
   deseasonalised *and* detrended against platform growth. 118 genuine consumer-behaviour
   shocks detected, plus CUSUM-style volume breaks per merchant.

10. **Drift monitoring.** Population Stability Index per feature between the training and live
    eras, with the conventional 0.25 retrain line.

---

## Architecture

```
7 Olist tables → order master (weak labels) → rolling seller panel (45 feats)
                        │                              │
                        ├── category / regional        ├── LightGBM forecaster
                        │   anomaly z-scores           ├── SHAP explainability
                        ├── complaint NLP taxonomy     └── lead-time backtest
                        └── collusion graph
                                       ↓
                    ACTION LAYER: exposure cap · tenure cap · holdback
                                       ↓
                     Streamlit command centre + exportable action file
```

---

## Mapping to a TVS Credit LMS

| Olist field | TVS equivalent |
|---|---|
| `seller_id` | Dealer / merchant ID |
| `payment_installments` | EMI tenure |
| `payment_value` where installments ≥ 2 | Financed value / exposure |
| `order_status` (canceled, unavailable) | Disbursal cancellation / reversal |
| delivered vs `order_estimated_delivery_date` | Fulfilment SLA breach |
| `review_score`, `review_comment_message` | Post-sale service feedback / complaint ticket |
| `customer_zip_code_prefix`, `seller_state` | Branch / region / pincode |
| `product_category_name` | Asset category (mobile, consumer durable) |

Nothing in the pipeline is Brazil-specific except the Portuguese complaint lexicon, which is
one dictionary swap.

---

## Run it

```bash
pip install -r requirements.txt
bash get_data.sh              # only if ./data is missing
python pipeline.py            # ~3 min: builds panel, trains, scores, writes artifacts/
python oos.py                 # honest out-of-sample scoring + policy backtest
python analytics.py           # product risk, seasonality, demand shifts, drift
python depth.py               # uncertainty intervals, multi-horizon (30/60/90d), segment stability
python finalize_metrics.py    # merges OOS & analytics into metrics.json (MUST be last)
python export_web.py          # builds web/data.json
uvicorn serve:app --port 8000 # risk register UI + scoring API -> open http://localhost:8000
streamlit run app.py          # optional analyst view
```

The UI is served at `/`, the API under `/score`, `/watchlist` and `/health`. Fonts are vendored
locally and all charts are hand-rolled SVG, so the demo needs **no internet connection**.

**API** — proves LMS integration. `POST /score {"seller_id": "..."}` returns score, band,
approved limit, max tenure, holdback, an APPROVE/REFER/DECLINE decision and the written
memo. `GET /watchlist?band=HIGH` returns the review queue. `GET /health` for liveness.

Dashboard views: Command Centre · Merchant Deep-Dive (SHAP + trajectory) ·
Early-Warning Alerts (exportable CSV) · Collusion Graph · Product & Category Risk ·
Model Validation.

---

## Limitations — state these to the jury before they ask

- **Weak supervision.** The bad-order label is constructed, not ground truth. We do not
  claim it equals return fraud. It is a defensible proxy for merchant-driven portfolio loss.
- **AUC ~0.66 is the honest ceiling on this data.** We tested order-level models,
  autoregressive lag features, five alternative label definitions and a hyperparameter
  sweep. A large share of merchant badness is logistics randomness that is not predictable
  from past behaviour. We report ranking quality, lead time and a significance test against
  two baselines rather than inflating a headline.
- **Brazilian proxy data**, chosen because it is the only public source carrying merchant,
  fulfilment, review and EMI-installment fields in one relational schema.
- **No leakage.** Every feature is computed from a strictly past window; the train/test
  split is temporal, never random.
