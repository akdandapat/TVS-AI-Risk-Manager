# SENTINEL — Intelligent Merchant & Product Risk Engine

**TVS Credit E.P.I.C 8.0 · Problem Statement (g)**

Forecasts merchant deterioration *before* it hits the book, and converts every score into
a concrete financing action: exposure cap, EMI tenure cap, holdback percentage.

Built on the **real Olist Brazilian E-Commerce dataset** — 98,666 orders, 3,095 merchants,
32,951 products, 99,224 reviews, Sep 2016 – Oct 2018. No synthetic data anywhere.

---

## Headline results (out-of-sample, time-based split)

| Metric | Value |
|---|---|
| **Early-warning lead time (median)** | **28 days** |
| **Deteriorating merchants caught early** | **50%** (87 of 174 eligible) |
| Top-decile lift | **2.19x** |
| Precision at 0.60 threshold | 31.5% vs 12.1% base — **2.60x lift** |
| Financed value in bad orders captured | **23%** (R$ 106,590 of R$ 464,642) |
| ROC-AUC | 0.658 |
| Model | LightGBM, 45 features, trained ≤ 2018-04-07, tested strictly after |

**Read the lead time, not the AUC.** A risk committee does not act on AUC. It acts on
"which 10% of my merchant book do I review this week, and how many days of warning do I get."
We give 28 days and 2.19x concentration.

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

4. **Collusion graph.** Bipartite merchant ↔ `customer_unique_id` network; 44 merchant pairs
   share an abnormal number of unique customers — the fake-transaction ring signal.

5. **Action layer.** The differentiator. Score → risk band → *exposure multiplier, max EMI
   tenure, holdback %*. CRITICAL suspends new financing; HIGH caps at 25% of current exposure,
   3-month tenure, 15% holdback; WATCH caps at 60%, 6-month tenure, 7% holdback.

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
bash get_data.sh          # pulls the 7 Olist tables into ./data
python pipeline.py        # ~90s: builds panel, trains, scores, writes artifacts/
streamlit run app.py      # dashboard
```

Dashboard views: Command Centre · Merchant Deep-Dive (SHAP + trajectory) ·
Early-Warning Alerts (exportable CSV) · Collusion Graph · Product & Category Risk ·
Model Validation.

---

## Limitations — state these to the jury before they ask

- **Weak supervision.** The bad-order label is constructed, not ground truth. We do not
  claim it equals return fraud. It is a defensible proxy for merchant-driven portfolio loss.
- **AUC 0.658 is the honest ceiling on this data.** We tested order-level models,
  autoregressive lag features, and three alternative label definitions; a large share of
  merchant badness is logistics randomness and is not predictable from purchase-time
  information. We report ranking quality and lead time instead of inflating a headline.
- **Brazilian proxy data**, chosen because it is the only public source carrying merchant,
  fulfilment, review and EMI-installment fields in one relational schema.
- **No leakage.** Every feature is computed from a strictly past window; the train/test
  split is temporal, never random.
