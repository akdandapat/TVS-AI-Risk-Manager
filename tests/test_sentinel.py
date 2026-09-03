"""Regression tests. Run: pytest -q

These exist to stop the three mistakes this project actually made:
purging removed, policy compared at merchant count, in-sample metrics promoted.
"""
import json, os, subprocess, sys
import numpy as np
import pandas as pd
import pytest

A = os.environ.get("SENTINEL_OUT", "artifacts")
pytestmark = pytest.mark.skipif(not os.path.exists(f"{A}/metrics.json"),
                                reason="run the pipeline first")


@pytest.fixture(scope="module")
def M():
    return json.load(open(f"{A}/metrics.json"))


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(f"{A}/panel.parquet")


@pytest.fixture(scope="module")
def scored():
    return pd.read_parquet(f"{A}/panel_scored.parquet")


# ---------------------------------------------------------------- leakage
def test_no_future_columns_in_features():
    import pickle
    feats = pickle.load(open(f"{A}/model.pkl", "rb"))["features"]
    leaky = [f for f in feats if f.startswith("fut_") or f == "target"]
    assert not leaky, f"future-derived columns leaked into features: {leaky}"


def test_walkforward_is_purged(M):
    assert M.get("wf_purged") is True, "walk-forward purge flag missing"


def test_sentinel_beats_naive_but_not_absurdly(M):
    s, n = M["wf_sentinel_auc"], M["wf_naive_auc"]
    assert s > n, "SENTINEL must beat naive persistence"
    assert s - n < 0.06, (
        f"gap over naive is {s-n:.4f}. A gap this large usually means the purge was "
        "removed and outcome windows are overlapping. Check scorer.walk_forward.")


def test_model_alone_is_reported(M):
    assert "wf_model_only_auc" in M, "model-only baseline must stay visible"


# ----------------------------------------------------------------- policy
def test_policy_frontier_shape():
    f = pd.read_csv(f"{A}/policy_frontier.csv")
    assert set(f.scorer.unique()) == {"s_sentinel", "s_eb", "s_naive", "s_model"}
    assert f.groupby("scorer").size().nunique() == 1, "uneven depth grid"


def test_policy_capture_monotonic():
    f = pd.read_csv(f"{A}/policy_frontier.csv")
    for s, g in f.groupby("scorer"):
        g = g.sort_values("declined")
        assert g.captured.is_monotonic_increasing, f"{s} capture curve is not monotonic"


def test_policy_beats_no_skill():
    """Capture must exceed the decline budget, or the scorer has no skill at all."""
    f = pd.read_csv(f"{A}/policy_frontier.csv")
    g = f[f.scorer == "s_sentinel"]
    assert (g.captured > g.declined).all(), "SENTINEL is below the no-skill diagonal"


# ---------------------------------------------------------------- metrics
def test_oos_metrics_promoted(M):
    assert "insample_median_lead_days" in M, "finalize_metrics.py did not run"
    assert M["median_lead_days"] == M["oos_median_lead_days"], (
        "headline lead time must be the out-of-sample figure, not the in-sample one")


def test_finalize_is_idempotent(M):
    before = M["insample_median_lead_days"]
    subprocess.run([sys.executable, "finalize_metrics.py"], check=True,
                   capture_output=True)
    after = json.load(open(f"{A}/metrics.json"))["insample_median_lead_days"]
    assert before == after, "re-running finalize_metrics overwrote in-sample values"


# ------------------------------------------------------------------ depth
def test_longer_horizons_hold():
    d = json.load(open(f"{A}/depth_metrics.json"))
    h = d["horizons"]
    assert h["h90"]["auc"] > h["h30"]["auc"] - 0.02, (
        "signal collapses at 90 days — this would be a nowcast, not early warning")


def test_beats_naive_in_every_size_segment():
    d = json.load(open(f"{A}/depth_metrics.json"))["segments"]["size_seg"]
    for seg, v in d.items():
        if "auc" in v:
            assert v["auc"] >= v["naive_auc"] - 0.005, f"loses to naive on {seg}"


def test_abstention_result_still_reported():
    d = json.load(open(f"{A}/depth_metrics.json"))
    assert "abstention" in d and "confident" in d["abstention"], (
        "the failed abstention test must remain visible, not be deleted")


# --------------------------------------------------------------- sanity
def test_scores_in_range(scored):
    assert scored.oos_score.between(0, 1).all()


def test_no_duplicate_seller_snapshots(panel):
    assert not panel.duplicated(["seller_id", "snapshot"]).any()


def test_web_payload_complete():
    p = "web/data.json"
    if not os.path.exists(p):
        pytest.skip("run export_web.py")
    d = json.load(open(p))
    for k in ["merchants", "products", "policy_frontier", "depth", "drift",
              "demand_shifts", "seasonal", "walkforward"]:
        assert k in d and len(d[k]), f"web payload missing {k}"
