"""SENTINEL real-time scoring API.  Run: uvicorn serve:app --port 8000

Demonstrates LMS integration: an underwriting system posts a merchant's trailing
feature vector and receives a score, band, and binding financing limits in <50ms.
"""
import os, pickle
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


@app.get("/watchlist")
def watchlist(band: str = "HIGH", limit: int = 25):
    v = SCORES[SCORES.risk_band == band].head(limit)
    return v[["sentinel_score", "risk_band", "bad_rate", "n_orders",
              "current_exposure", "revised_limit", "recommendation"]] \
        .reset_index().to_dict("records")
