# SENTINEL — HANDOFF v1 → v3

Read this first. It explains what changed since the version you have, and **why** — the
reasoning matters more than the diff, because two of the changes exist to correct mistakes
that would otherwise have been presented as results.

---

## 0. What you currently have vs. what is final

The codebase you pasted is **v1**. It reported ROC-AUC 0.658, 28-day lead time, and a
single time-based train/test split.

**v1 has a validation bug and two model-design problems.** Do not present its numbers.

| File | Status |
|---|---|
| `pipeline.py` | **CHANGED** — STEP 14→7, mix-expectation features, percentile bands |
| `model.py` | **CHANGED** — seed ensemble, baseline comparison, purged walk-forward hook |
| `app.py` | **CHANGED** — metric keys renamed; now the secondary/analyst view |
| `README.md` | **REWRITTEN** — corrected numbers and the methodology story |
| `requirements.txt` | **CHANGED** — added fastapi, uvicorn, scipy |
| `scorer.py` | **NEW** — EB shrinkage, blended scorer, purged walk-forward, risk memo, ROI |
| `serve.py` | **NEW** — FastAPI scoring API + serves the web UI |
| `export_web.py` | **NEW** — builds `web/data.json` for the frontend |
| `experiments.py` | **NEW** — the accuracy sweep harness (evidence of rigour) |
| `web/` | **NEW** — bespoke frontend (index.html, styles.css, app.js, fonts/) |
| `get_data.sh` | unchanged |

---

## 1. The three findings that drove every change

### Finding 1 — the naive baseline is strong, and v1 never tested it

Ranking merchants by their **current trailing bad rate**, with no model at all, scores
ROC-AUC ≈ 0.636 and a top-decile lift of 2.34. v1's model scored 0.658 AUC but only
**2.19** lift — *worse than one column of arithmetic on the metric that matters*.

Any early-warning system must be benchmarked against persistence. v1 wasn't.

### Finding 2 — v1's validation leaked future information

v1 (and an intermediate v2) built rolling snapshots every 14 days, where each snapshot's
label is the bad-order rate over the **next 30 days**. Training snapshots within 30 days of
a test snapshot therefore have **overlapping outcome windows** — the model sees partial
future labels for the same merchants.

An intermediate version reported "AUC 0.700, wins 11/12 folds, p=0.005". After adding a
purge (drop training snapshots within one horizon of the test snapshot), the same model
scored **0.640 and won 2 of 12 folds** — it did not beat the baseline at all.

**Rule now enforced in `scorer.walk_forward`:**
```python
tr = panel[panel.snapshot <= cut - pd.Timedelta(days=horizon)]   # PURGE
te = panel[panel.snapshot == cut]
```

### Finding 3 — what beat the baseline was shrinkage, not machine learning

A merchant with 6 orders at a 33% bad rate is mostly noise. **Empirical-Bayes shrinkage**
pulls each merchant's observed rate toward what their own product-and-region mix predicts,
weighted by how little volume backs it:

```python
score = (n * observed_rate + k * expected_rate) / (n + k)     # k = 10
```

`expected_rate` comes from `pipeline.add_expected()` — leak-free expanding means by
category, customer state and price decile, using strictly prior orders.

Shrinkage **alone** beats naive persistence (0.6451 vs 0.6363). The gradient-boosted model
earns its place only as a 30% correction on top.

**Final scorer:** `0.30 × percentile(model prob) + 0.70 × percentile(EB-shrunk rate)`.
Every blend weight from 0.2 to 0.5 beats the baseline at p < 0.02, so it is not knife-edge.

### Also: one rejected model worth mentioning to the jury

An exposure-weighted target (predict merchants generating top-decile financed loss) scored
**AUC 0.855**. It was rejected: the single column `financed_value` scored **0.858** on the
same target. It was predicting merchant size, not risk.

---

## 2. Final numbers — use these, not v1's

Purged walk-forward, 24 expanding-window folds:

| Scorer | Mean AUC | Note |
|---|---|---|
| **SENTINEL** (0.30 model + 0.70 shrunk rate) | **0.6567** | wins **18/24** folds, **p = 0.0024** |
| Empirical-Bayes shrunk rate alone | 0.6451 | strong, simple |
| Raw bad rate (naive persistence) | 0.6363 | the baseline to beat |
| Gradient-boosted model **alone** | 0.6181 | **worst of the four** |

| Business metric | Value |
|---|---|
| Median early-warning lead time | **35 days** |
| Deteriorating merchants caught early | **57%** (75 of 132) |
| Financed value in bad orders captured | 23% |
| Modelled annual saving, ₹500 Cr book | ~₹5.1 Cr |
| Panel | 11,543 seller-snapshots, 645 merchants, 98,666 orders |

---

## 3. Architectural changes in detail

**`pipeline.py`**
- `LOOKBACK, HORIZON, STEP = 90, 30, 7` (was `..., 14`). Denser snapshots roughly doubled
  training rows and lifted lead time from 28 → 35 days.
- New `add_expected(om)`, called at the end of `build_order_master`. Adds `exp_bad` per
  order: the leak-free expanding mean bad rate of its category / customer state / price
  decile. This is what shrinkage shrinks toward.
- `build_panel` now emits `exp_bad` and `resid_bad` (= `bad_rate - exp_bad`) per snapshot.
- `action(p)` thresholds changed from **probability** to **book percentile**:
  `>=0.95 CRITICAL, >=0.85 HIGH, >=0.70 WATCH`. Bands are now positions in the book —
  critical is the top 5% — so the review queue stays a fixed size instead of inflating
  whenever the portfolio drifts. **Expect fewer HIGH merchants than v1. That is correct.**
- Scoring now uses `res["scorer"].score(...)` and sorts by `sentinel_score`.

**`model.py`**
- `DROP` gained `"date"`.
- Adds `baseline_auc` / `baseline_lift20` / `sentinel_lift20` so the naive comparison is
  always computed, never optional.
- Calls `scorer.walk_forward` and merges its metrics.
- Pickles `{"model", "features", "scorer"}` — `scorer` is new and required by `serve.py`,
  `export_web.py` and the UI.

**`scorer.py` (new)** — the intellectual core, ~200 lines:
`eb_shrink`, `RiskScorer` (percentile blend against frozen reference distributions so a
single merchant can be scored in isolation, not just in a batch), `walk_forward` (purged,
4-way comparison), `narrative` (deterministic risk memo), `roi` (INR translation).

**`serve.py` (new)** — FastAPI. `POST /score` (by `seller_id` or raw feature vector) returns
score, band, approved limit, max tenor, holdback, APPROVE/REFER/DECLINE and the memo.
`GET /watchlist`, `GET /health`. Mounts the UI at `/`.

**`web/` (new)** — bespoke frontend replacing Streamlit as the primary demo. Light "bone
paper" ground, IBM Plex vendored locally, hand-rolled SVG, **no CDN — works offline**.
Design rule: colour is used *only* to carry risk meaning. Six views: Portfolio, Watchlist,
Merchant file, Evidence, Network, Categories.

Signature element: the **lead-time ribbon** on each merchant file — risk trace across all
snapshots, hollow ring at first flag, filled dot at deterioration, shaded band between.
It draws the 35-day claim rather than asserting it.

---

## 4. Run order

```bash
pip install -r requirements.txt
bash get_data.sh                 # only if ./data is missing
python pipeline.py               # ~3 min: panel, training, purged walk-forward, scoring
python export_web.py             # builds web/data.json
uvicorn serve:app --port 8000    # UI at http://localhost:8000 + API
streamlit run app.py             # optional analyst view
```

`python experiments.py 90 30 7 8` re-runs one sweep config (LOOKBACK HORIZON STEP MIN_TARGET)
if you need to reproduce the comparison live.

---

## 5. Instructions for the model receiving this

You are picking up a finished, working project. Priorities:

1. **Do not "improve" the headline AUC by removing the purge.** If a change makes AUC jump
   toward 0.70, you have probably reintroduced the leak. Verify against
   `wf_naive_auc` — if the gap over naive suddenly widens a lot, be suspicious.
2. **Every new scorer must be compared against `naive` and `eb_only`** in
   `scorer.walk_forward`. A scorer that doesn't beat one column of arithmetic is not a
   scorer.
3. **Keep the honest framing.** The "model alone is the worst of four" line is deliberately
   surfaced in the Evidence view and the README. It is the project's strongest credibility
   asset, not a weakness to hide.
4. **The label is weak supervision.** Bad order = cancelled/unavailable OR delivered >5 days
   late OR review ≤2 stars. Never describe it as ground-truth fraud.
5. **The data is a Brazilian proxy.** Field mapping to an Indian LMS is in the README.
   Don't claim Indian data.
6. If you change bands, features or the blend weight, **re-run `export_web.py`** or the UI
   will show stale numbers.
7. Frontend has no build step and no dependencies. Edit `web/app.js` directly. Test with
   the Node harness pattern: stub `document`, require the module, call each renderer, and
   assert the output contains no `undefined` / `NaN`.

### Known open items (safe to leave, or good next tasks)
- Cold-vector API path (`POST /score` with raw `features`) defaults unspecified features to
  peer means, which dilutes signal. Demo with `seller_id` instead.
- The blend weight 0.30 was selected by inspecting purged fold results; a stricter protocol
  would pick it on an inner validation slice only. Robustness across 0.2–0.5 is the current
  defence.
- `experiments.py` runs one config per invocation because the full sweep exceeds typical
  command timeouts.
- Streamlit `app.py` is kept for analysts but is no longer the primary UI; the two must be
  kept in sync on metric key names.

---

## 6. Presenting this

Open on **Evidence**, not Portfolio. The line that wins:

> "The machine-learning model alone is the worst of our four scorers. We found that,
> we kept it visible in the product, and we built the system around what actually worked."

Then open a merchant file and show the ribbon — one merchant flagged 42 days early, with the
memo and the binding action next to it.

Close with the rejected 0.855 model: *"we built a model that scored 0.855, discovered a
single column beat it, and threw it away."* Most teams present their best-looking number
with no idea whether a one-line baseline beats it.
