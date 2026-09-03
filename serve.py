"""SENTINEL real-time scoring API.  Run: uvicorn serve:app --port 8000

Demonstrates LMS integration: an underwriting system posts a merchant's trailing
feature vector and receives a score, band, and binding financing limits in <50ms.
"""
import os, json, pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

A = os.environ.get("SENTINEL_OUT", "artifacts")
B = pickle.load(open(f"{A}/model.pkl", "rb"))
SCORER, FEATS = B["scorer"], B["features"]
SCORES = pd.read_parquet(f"{A}/seller_scores.parquet").set_index("seller_id")

import sys; sys.path.insert(0, os.path.dirname(__file__))
from pipeline import action
from scorer import narrative

PEER = SCORES[FEATS].mean()

# --- Part 4: reference data for the extended API
def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default

PRODUCTS = _try(lambda: pd.read_parquet(f"{A}/product_risk.parquet"))
FRONTIER = _try(lambda: pd.read_csv(f"{A}/policy_frontier.csv"))
METRICS = _try(lambda: json.load(open(f"{A}/metrics.json")), {})
BREAKS = _try(lambda: pd.read_csv(f"{A}/volume_breaks.csv"))
SHIFTS = _try(lambda: pd.read_csv(f"{A}/demand_shifts.csv"))
DEPTH = _try(lambda: json.load(open(f"{A}/depth_metrics.json")), {})
app = FastAPI(title="SENTINEL Merchant Risk API", version="2.0")

# serve the risk register UI at /
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
if os.path.isdir(WEB):
    from fastapi.staticfiles import StaticFiles
    app.mount("/ui", StaticFiles(directory=WEB, html=True), name="ui")

    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse("/ui/index.html")


class ScoreRequest(BaseModel):
    seller_id: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    exposure: Optional[float] = None


def _decide(row: pd.Series, exposure: float):
    X = pd.DataFrame([row[FEATS]])
    score = float(SCORER.score(X)[0])
    prob = float(SCORER.prob(X)[0])
    band, mult, tenure, hold, rec = action(score)
    return {"sentinel_score": round(score, 4), "model_probability": round(prob, 4),
            "risk_band": band, "current_exposure": round(exposure, 2),
            "approved_limit": round(exposure * mult, 2),
            "max_tenure_months": tenure, "holdback_pct": hold,
            "recommendation": rec,
            "decision": "DECLINE" if band == "CRITICAL" else
                        "REFER" if band == "HIGH" else "APPROVE"}


@app.get("/health")
def health():
    return {"status": "ok", "merchants_cached": len(SCORES), "features": len(FEATS)}


@app.post("/score")
def score(req: ScoreRequest):
    if req.seller_id:
        if req.seller_id not in SCORES.index:
            raise HTTPException(404, "merchant not found")
        row = SCORES.loc[req.seller_id]
        exp = req.exposure if req.exposure is not None else float(row["current_exposure"])
    elif req.features:
        row = pd.Series({f: req.features.get(f, PEER[f]) for f in FEATS})
        exp = req.exposure or 0.0
    else:
        raise HTTPException(400, "provide seller_id or features")
    out = _decide(row, exp)
    if req.seller_id:
        out["memo"] = narrative({**row.to_dict(), **out,
                                 "seller_id": req.seller_id,
                                 "revised_limit": out["approved_limit"]}, PEER)
    return out


class BatchRequest(BaseModel):
    seller_ids: list[str]
    exposure: Optional[Dict[str, float]] = None


class SimulateRequest(BaseModel):
    decline_pct: float = 0.20          # share of financed value declined
    portfolio_inr_cr: float = 500.0
    margin: float = 0.06
    recovery: float = 0.35
    scorer: str = "s_sentinel"


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    """Score up to 500 merchants in one call — the nightly LMS refresh path."""
    if len(req.seller_ids) > 500:
        raise HTTPException(400, "max 500 merchants per batch")
    out, missing = [], []
    for sid in req.seller_ids:
        if sid not in SCORES.index:
            missing.append(sid); continue
        row = SCORES.loc[sid]
        exp = (req.exposure or {}).get(sid, float(row["current_exposure"]))
        out.append({"seller_id": sid, **_decide(row, exp)})
    return {"scored": len(out), "missing": missing, "results": out}


@app.get("/product/{product_id}")
def product(product_id: str):
    if PRODUCTS is None:
        raise HTTPException(503, "product risk not built — run analytics.py")
    r = PRODUCTS[PRODUCTS.product_id == product_id]
    if r.empty:
        raise HTTPException(404, "product not found")
    r = r.iloc[0]
    return {"product_id": product_id, "category": str(r["category"]),
            "orders": int(r["n_orders"]), "raw_bad_rate": round(float(r["bad_rate"]), 4),
            "shrunk_risk": round(float(r["risk"]), 4),
            "category_norm": round(float(r["cat_bad"]), 4),
            "excess_over_category": round(float(r["excess"]), 4),
            "financed": round(float(r["financed"]), 2),
            "exposure_at_risk": round(float(r["exposure_at_risk"]), 2),
            "action": r["action"],
            "decision": "DECLINE" if str(r["action"]).startswith("DELIST")
            else "REFER" if str(r["action"]).startswith(("RESTRICT", "REVIEW"))
            else "APPROVE"}


@app.get("/products/watchlist")
def products_watchlist(action: str = "DELIST", limit: int = 25):
    if PRODUCTS is None:
        raise HTTPException(503, "product risk not built — run analytics.py")
    v = PRODUCTS[PRODUCTS.action.str.startswith(action.upper())] \
        .nlargest(limit, "exposure_at_risk")
    return v[["product_id", "category", "n_orders", "risk", "financed",
              "exposure_at_risk", "action"]].to_dict("records")


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """What-if for a credit committee: pick a decline budget, see the trade."""
    if FRONTIER is None:
        raise HTTPException(503, "policy backtest not built — run oos.py")
    g = FRONTIER[FRONTIER.scorer == req.scorer].sort_values("declined")
    if g.empty:
        raise HTTPException(400, f"unknown scorer {req.scorer}")
    import numpy as np
    cap = float(np.interp(req.decline_pct, g.declined, g.captured))
    bad_rate = METRICS.get("bad_order_rate", 0.152)
    gross = req.portfolio_inr_cr * bad_rate
    avoided = gross * cap * req.recovery
    forgone = req.portfolio_inr_cr * (1 - bad_rate) * req.decline_pct * req.margin
    return {"scorer": req.scorer, "decline_pct": req.decline_pct,
            "bad_value_captured_pct": round(cap, 4),
            "portfolio_inr_cr": req.portfolio_inr_cr,
            "gross_bad_loss_inr_cr": round(gross, 3),
            "loss_avoided_inr_cr": round(avoided, 3),
            "margin_forgone_inr_cr": round(forgone, 3),
            "net_benefit_inr_cr": round(avoided - forgone, 3),
            "assumptions": {"bad_order_rate": bad_rate, "margin": req.margin,
                            "recovery": req.recovery}}


@app.get("/alerts")
def alerts(limit: int = 25):
    """Everything that changed recently and needs a human look."""
    vb = [] if BREAKS is None else BREAKS.head(limit)[
        ["seller_id", "snapshot", "n_orders", "v_mean", "v_z", "kind"]].to_dict("records")
    ds = [] if SHIFTS is None else SHIFTS.reindex(
        SHIFTS.z.abs().sort_values(ascending=False).index).head(limit)[
        ["category", "ym", "n", "z", "shift", "bad_rate"]].to_dict("records")
    crit = SCORES[SCORES.risk_band.isin(["CRITICAL", "HIGH"])]
    return {"merchant_alerts": len(crit), "volume_breaks": vb, "demand_shifts": ds}


@app.get("/metrics")
def metrics():
    """Model card endpoint — what this model is and how well it works."""
    keep = ["wf_sentinel_auc", "wf_naive_auc", "wf_eb_auc", "wf_model_only_auc",
            "wf_folds", "wf_auc_wins", "wf_pvalue", "median_lead_days",
            "pct_caught_early", "policy_capture_sentinel", "policy_capture_naive",
            "policy_p", "n_orders", "bad_order_rate", "n_products_scored"]
    return {"model": METRICS.get("engine"), "features": len(FEATS),
            "merchants_scored": len(SCORES),
            "performance": {k: METRICS.get(k) for k in keep if k in METRICS},
            "horizons": DEPTH.get("horizons", {}),
            "segments": DEPTH.get("segments", {}).get("size_seg", {}),
            "known_limitations": [
                "Bad-order label is weak supervision, not ground-truth fraud.",
                "Brazilian proxy data; field mapping to an Indian LMS is in the README.",
                "Confidence-based abstention was tested and did not improve ranking.",
                "exp_bad and approval_lag show PSI > 1.7; schedule retraining."]}


@app.get("/watchlist")
def watchlist(band: str = "HIGH", limit: int = 25):
    v = SCORES[SCORES.risk_band == band].head(limit)
    return v[["sentinel_score", "risk_band", "bad_rate", "n_orders",
              "current_exposure", "revised_limit", "recommendation"]] \
        .reset_index().to_dict("records")
