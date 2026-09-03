"""Builds web/data.json — everything the frontend needs, precomputed."""
import json, os, pickle
import numpy as np
import pandas as pd

A = os.environ.get("SENTINEL_OUT", "artifacts")
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB, exist_ok=True)
FLAG = 0.70   # WATCH band and above = "flagged"


def jnum(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


def main():
    M = json.load(open(f"{A}/metrics.json"))
    S = pd.read_parquet(f"{A}/seller_scores.parquet")
    P = pd.read_parquet(f"{A}/panel.parquet")
    OM = pd.read_parquet(f"{A}/order_master.parquet")
    E = pd.read_csv(f"{A}/collusion_edges.csv")
    WF = pd.read_csv(f"{A}/walkforward.csv")
    IMP = pd.read_csv(f"{A}/feature_importance.csv", index_col=0)
    PR = pd.read_parquet(f"{A}/product_risk.parquet")
    PF = pd.read_csv(f"{A}/policy_frontier.csv")
    PS = pd.read_csv(f"{A}/policy_summary.csv").rename(columns={"Unnamed: 0": "scorer"})
    DS = pd.read_csv(f"{A}/demand_shifts.csv")
    DSER = pd.read_csv(f"{A}/demand_series.csv")
    SI = pd.read_csv(f"{A}/seasonal_index.csv")
    VB = pd.read_csv(f"{A}/volume_breaks.csv")
    DR = pd.read_csv(f"{A}/drift.csv")
    B = pickle.load(open(f"{A}/model.pkl", "rb"))
    SC, FEATS = B["scorer"], B["features"]

    from scorer import narrative
    PEER = S[FEATS].mean()

    # ---- score every historical snapshot so we can draw risk trajectories
    P = P.sort_values(["seller_id", "snapshot"]).copy()
    P["score"] = SC.score(P)
    snaps = sorted(P["snapshot"].unique())
    sidx = {s: i for i, s in enumerate(snaps)}

    # ---- SHAP drivers
    try:
        import shap
        ex = shap.TreeExplainer(SC.models[0])
        sv = ex.shap_values(S[FEATS].astype(float).values)
        sv = sv[1] if isinstance(sv, list) else sv
    except Exception:
        sv = np.zeros((len(S), len(FEATS)))

    merchants = []
    for i, (_, r) in enumerate(S.iterrows()):
        h = P[P.seller_id == r.seller_id]
        traj = [{"t": sidx[t], "d": str(t)[:10], "s": jnum(sc, 3),
                 "b": jnum(br, 3), "y": int(y)}
                for t, sc, br, y in zip(h.snapshot, h.score, h.bad_rate, h.target)]
        first_flag = next((p["t"] for p in traj if p["s"] and p["s"] >= FLAG), None)
        first_bad = next((p["t"] for p in traj if p["y"] == 1), None)
        lead = None
        if first_flag is not None and first_bad is not None and first_flag < first_bad:
            lead = int((pd.Timestamp(snaps[first_bad])
                        - pd.Timestamp(snaps[first_flag])).days)
        d = pd.DataFrame({"f": FEATS, "v": sv[i]})
        d["a"] = d.v.abs()
        drivers = [{"f": x.f, "v": jnum(x.v, 4)}
                   for x in d.nlargest(8, "a").itertuples()]
        merchants.append({
            "id": r.seller_id, "short": r.seller_id[:10],
            "score": jnum(r.sentinel_score, 3), "band": r.risk_band,
            "prob": jnum(r.risk_prob, 3), "eb": jnum(r.get("eb_rate"), 3),
            "bad_rate": jnum(r.bad_rate, 4), "exp_bad": jnum(r.get("exp_bad"), 4),
            "late_rate": jnum(r.late_rate, 4), "cancel_rate": jnum(r.cancel_rate, 4),
            "badreview_rate": jnum(r.badreview_rate, 4),
            "mean_review": jnum(r.mean_review, 2), "mean_delay": jnum(r.mean_delay, 2),
            "n_orders": int(r.n_orders), "n_products": int(r.n_products),
            "tenure": int(r.tenure_days) if pd.notna(r.tenure_days) else None,
            "exposure": jnum(r.current_exposure, 2),
            "limit": jnum(r.revised_limit, 2),
            "tenure_cap": int(r.max_tenure_months), "holdback": jnum(r.holdback_pct, 3),
            "rec": r.recommendation, "category": str(r.category),
            "cx": {k[3:]: jnum(r[k], 3) for k in S.columns if k.startswith("cx_")},
            "traj": traj, "first_flag": first_flag, "first_bad": first_bad,
            "lead": lead,
            "drivers": drivers,
            "memo": narrative(r.to_dict(), PEER),
        })

    cat = (OM.groupby("category").agg(
        orders=("order_id", "count"), bad=("is_bad", "mean"),
        cancel=("is_cancelled", "mean"), late=("is_late", "mean"),
        rev=("is_badreview", "mean"), financed=("financed_value", "sum")).reset_index())
    cat = cat[cat.orders >= 200].sort_values("bad", ascending=False)

    st = (OM.groupby("customer_state").agg(
        orders=("order_id", "count"), bad=("is_bad", "mean")).reset_index())
    st = st[st.orders >= 200].sort_values("bad", ascending=False)

    trend = (OM.set_index("date").resample("ME")
               .agg(bad=("is_bad", "mean"), orders=("order_id", "count")).reset_index())
    trend = trend[trend.orders > 100]

    risk_of = S.set_index("seller_id")["sentinel_score"].to_dict()
    edges = E.head(80)
    nodes = sorted(pd.unique(edges[["seller_id_x", "seller_id_y"]].values.ravel()))

    out = {
        "metrics": M,
        "generated": pd.Timestamp.now().strftime("%d %b %Y"),
        "snapshots": [str(s)[:10] for s in snaps],
        "merchants": merchants,
        "bands": S["risk_band"].value_counts().to_dict(),
        "band_exposure": S.groupby("risk_band")["current_exposure"].sum().to_dict(),
        "walkforward": [{"fold": r.fold, "sentinel": jnum(r.sentinel_auc, 4),
                         "naive": jnum(r.naive_auc, 4), "eb": jnum(r.eb_only_auc, 4),
                         "model": jnum(r.model_only_auc, 4)}
                        for r in WF.itertuples()],
        "categories": [{"name": r.category, "orders": int(r.orders),
                        "bad": jnum(r.bad, 4), "cancel": jnum(r.cancel, 4),
                        "late": jnum(r.late, 4), "rev": jnum(r.rev, 4),
                        "financed": jnum(r.financed, 0)} for r in cat.itertuples()],
        "states": [{"s": r.customer_state, "orders": int(r.orders),
                    "bad": jnum(r.bad, 4)} for r in st.itertuples()],
        "trend": [{"d": str(r.date)[:7], "bad": jnum(r.bad, 4),
                   "orders": int(r.orders)} for r in trend.itertuples()],
        "graph": {"nodes": [{"id": n, "short": n[:8], "risk": jnum(risk_of.get(n, 0), 3)}
                            for n in nodes],
                  "edges": [{"a": r.seller_id_x, "b": r.seller_id_y,
                             "w": int(r.shared_customers)} for r in edges.itertuples()]},
        "importance": [{"f": i, "v": jnum(v, 1)}
                       for i, v in IMP["importance"].head(16).items()],
        # ---- Part 2 additions ----
        "products": [{
            "id": r.product_id, "short": r.product_id[:10], "cat": str(r.category),
            "n": int(r.n_orders), "bad": jnum(r.bad_rate, 4), "risk": jnum(r.risk, 4),
            "cat_bad": jnum(r.cat_bad, 4), "excess": jnum(r.excess, 4),
            "financed": jnum(r.financed, 0), "ear": jnum(r.exposure_at_risk, 0),
            "cancel": jnum(r.cancel_rate, 4), "late": jnum(r.late_rate, 4),
            "review": jnum(r.mean_review, 2), "sellers": int(r.n_sellers),
            "price": jnum(r.mean_price, 0), "pctile": jnum(r.pctile, 3),
            "action": r.action, "act": r.action.split(" ")[0]}
            for r in PR.nlargest(140, "exposure_at_risk").itertuples()],
        "product_summary": {
            "scored": int(len(PR)),
            "delist": int(PR.action.str.startswith("DELIST").sum()),
            "restrict": int(PR.action.str.startswith("RESTRICT").sum()),
            "review": int(PR.action.str.startswith("REVIEW").sum()),
            "ear_total": jnum(PR.exposure_at_risk.sum(), 0),
            "ear_flagged": jnum(
                PR.loc[~PR.action.str.startswith("OK"), "exposure_at_risk"].sum(), 0)},

        "policy_frontier": [{"scorer": r.scorer, "d": jnum(r.declined, 3),
                             "c": jnum(r.captured, 4)} for r in PF.itertuples()],
        "policy_summary": [{"scorer": r.scorer, "mean_capture": jnum(r.mean_capture, 4),
                            "cap20": jnum(r.capture_at_20pct, 4),
                            "wins": int(r.wins_vs_naive), "n": int(r.n_snapshots),
                            "p": jnum(r.p_vs_naive, 4)} for r in PS.itertuples()],

        "seasonal": [{"cat": r.category, "m": int(r.month),
                      "v": jnum(r.seasonal_index, 3)} for r in SI.itertuples()],
        "demand_series": [{"cat": r.category, "ym": str(r.ym),
                           "share": jnum(r.share, 5), "z": jnum(r.z, 2),
                           "bad": jnum(r.bad_rate, 4), "n": int(r.n)}
                          for r in DSER.itertuples()],
        "demand_shifts": [{"cat": r.category, "ym": str(r.ym), "n": int(r.n),
                           "z": jnum(r.z, 2), "kind": r.shift,
                           "bad": jnum(r.bad_rate, 4)}
                          for r in DS.reindex(DS.z.abs().sort_values(
                              ascending=False).index).head(60).itertuples()],
        "volume_breaks": [{"id": r.seller_id, "short": r.seller_id[:10],
                           "snap": str(r.snapshot)[:10], "n": int(r.n_orders),
                           "mean": jnum(r.v_mean, 1), "z": jnum(r.v_z, 2),
                           "bad": jnum(r.bad_rate, 4),
                           "expo": jnum(r.financed_value, 0), "kind": r.kind}
                          for r in VB.head(50).itertuples()],
        "drift": [{"f": r.feature, "psi": jnum(r.psi, 4), "status": r.status}
                  for r in DR.itertuples()],
    }
    path = os.path.join(WEB, "data.json")
    json.dump(out, open(path, "w"), separators=(",", ":"), default=str)
    print(f"wrote {path}  ({os.path.getsize(path)/1e6:.2f} MB)")
    print(f"  merchants={len(merchants)}  with lead time={sum(1 for m in merchants if m['lead'])}"
          f"  folds={len(out['walkforward'])}")


if __name__ == "__main__":
    main()
