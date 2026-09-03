"""
SENTINEL - Intelligent Merchant & Product Risk Engine
TVS Credit E.P.I.C 8.0 | Problem Statement (g)

Forecasts merchant deterioration BEFORE it hits the book.
Target: given seller behaviour in [t-90, t), predict bad-order rate in [t, t+30).
"""
import os, json, warnings
import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore")
DATA = os.environ.get("SENTINEL_DATA", "data")
OUT = os.environ.get("SENTINEL_OUT", "artifacts")
os.makedirs(OUT, exist_ok=True)

LATE_TOLERANCE_DAYS = 5
BAD_RATE_THRESHOLD = 0.25      # next-30d bad rate that defines "deteriorated"
MIN_ORDERS_FEATURE = 5         # seller must have >=5 orders in lookback
MIN_ORDERS_TARGET = 8          # and >=8 forward (label-noise control)
LOOKBACK, HORIZON, STEP = 90, 30, 7


# ---------------------------------------------------------------- 1. LOAD
def load():
    p = lambda f: os.path.join(DATA, f)
    orders = pd.read_csv(p("olist_orders_dataset.csv"))
    items = pd.read_csv(p("olist_order_items_dataset.csv"))
    prods = pd.read_csv(p("olist_products_dataset.csv"))
    custs = pd.read_csv(p("olist_customers_dataset.csv"))
    pays = pd.read_csv(p("olist_order_payments_dataset.csv"))
    revs = pd.read_csv(p("olist_order_reviews_dataset.csv"))
    sell = pd.read_csv(p("olist_sellers_dataset.csv"))
    for c in ["order_purchase_timestamp", "order_approved_at",
              "order_delivered_carrier_date", "order_delivered_customer_date",
              "order_estimated_delivery_date"]:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
    return orders, items, prods, custs, pays, revs, sell


# ------------------------------------------------- 2. LABEL + ORDER MASTER
COMPLAINT_LEX = {
    "not_received": ["nao recebi", "não recebi", "nao chegou", "não chegou",
                     "nunca chegou", "nao entregue", "não entregue"],
    "late":         ["atraso", "atrasado", "demorou", "demora", "prazo"],
    "damaged":      ["quebrado", "danificado", "avariado", "estragado", "defeito"],
    "wrong_item":   ["errado", "diferente", "nao era", "não era", "trocado"],
    "counterfeit":  ["falso", "falsificado", "nao original", "não original", "pirata"],
    "refund":       ["estorno", "reembolso", "devolver", "devolucao", "devolução",
                     "cancelar", "dinheiro de volta"],
}


def strip_accents(s):
    return (s.str.normalize("NFKD")
             .str.encode("ascii", errors="ignore").str.decode("utf-8").str.lower())


def build_order_master(orders, items, prods, custs, pays, revs, sell):
    # --- review aggregation (one order can have >1 review row)
    revs["review_comment_message"] = revs["review_comment_message"].fillna("")
    txt = strip_accents(revs["review_comment_message"].astype(str))
    for tag, kws in COMPLAINT_LEX.items():
        revs["cx_" + tag] = txt.apply(lambda s: int(any(k in s for k in kws)))
    cx_cols = ["cx_" + k for k in COMPLAINT_LEX]
    ragg = revs.groupby("order_id").agg(
        review_score=("review_score", "mean"),
        has_review=("review_score", "size"),
        **{c: (c, "max") for c in cx_cols}).reset_index()

    # --- payments
    pagg = pays.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        installments=("payment_installments", "max"),
        n_pay_methods=("payment_type", "nunique")).reset_index()

    # --- order level
    o = orders.merge(custs[["customer_id", "customer_unique_id",
                            "customer_state", "customer_zip_code_prefix"]],
                     on="customer_id", how="left")
    o = o.merge(ragg, on="order_id", how="left").merge(pagg, on="order_id", how="left")

    o["delay_days"] = (o["order_delivered_customer_date"]
                       - o["order_estimated_delivery_date"]).dt.total_seconds() / 86400
    o["approval_lag_h"] = (o["order_approved_at"]
                           - o["order_purchase_timestamp"]).dt.total_seconds() / 3600
    o["carrier_lag_d"] = (o["order_delivered_carrier_date"]
                          - o["order_approved_at"]).dt.total_seconds() / 86400

    o["is_cancelled"] = o["order_status"].isin(["canceled", "unavailable"]).astype(int)
    o["is_late"] = (o["delay_days"] > LATE_TOLERANCE_DAYS).fillna(False).astype(int)
    o["is_badreview"] = (o["review_score"] <= 2).fillna(False).astype(int)
    o["is_bad"] = ((o["is_cancelled"] + o["is_late"] + o["is_badreview"]) > 0).astype(int)

    # --- attach seller (primary seller = first line item) + product attrs
    it = items.merge(prods, on="product_id", how="left")
    iagg = it.groupby("order_id").agg(
        seller_id=("seller_id", "first"),
        n_sellers=("seller_id", "nunique"),
        n_items=("order_item_id", "count"),
        price=("price", "sum"),
        freight=("freight_value", "sum"),
        product_id=("product_id", "first"),
        category=("product_category_name", "first"),
        photos_qty=("product_photos_qty", "mean"),
        desc_len=("product_description_lenght", "mean"),
        weight_g=("product_weight_g", "mean")).reset_index()

    o = o.merge(iagg, on="order_id", how="left")
    o = o.merge(sell, on="seller_id", how="left")
    o = o[o["seller_id"].notna()].copy()
    o["freight_ratio"] = o["freight"] / o["price"].replace(0, np.nan)
    o["cross_state"] = (o["customer_state"] != o["seller_state"]).astype(int)
    o["financed_value"] = np.where(o["installments"].fillna(1) >= 2,
                                   o["payment_value"].fillna(0), 0.0)
    o["date"] = o["order_purchase_timestamp"]
    o = add_expected(o)
    return o, cx_cols


def add_expected(om):
    """Leak-free expected badness from the ORDER MIX (category / region / price band).

    Separates "this merchant is bad" from "this merchant sells hard categories into
    hard regions". Every rate uses strictly prior orders only.
    """
    om = om.sort_values("date").reset_index(drop=True)
    om["price_bkt"] = pd.qcut(om["price"].rank(method="first"), 10,
                              labels=False, duplicates="drop")
    parts = []
    for key in ["category", "customer_state", "price_bkt"]:
        r = om.groupby(key)["is_bad"].transform(lambda s: s.shift(1).expanding().mean())
        parts.append(r)
    om["exp_bad"] = pd.concat(parts, axis=1).mean(axis=1)
    om["exp_bad"] = om["exp_bad"].fillna(om["is_bad"].mean())
    return om


# ------------------------------------------------------- 3. PANEL BUILDER
def _agg_window(df, cx_cols):
    g = df.groupby("seller_id")
    out = g.agg(
        n_orders=("order_id", "count"),
        gmv=("price", "sum"),
        financed_value=("financed_value", "sum"),
        bad_rate=("is_bad", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        late_rate=("is_late", "mean"),
        badreview_rate=("is_badreview", "mean"),
        mean_delay=("delay_days", "mean"),
        p90_delay=("delay_days", lambda s: s.quantile(0.9)),
        mean_review=("review_score", "mean"),
        review_cov=("has_review", lambda s: s.notna().mean()),
        approval_lag=("approval_lag_h", "median"),
        carrier_lag=("carrier_lag_d", "median"),
        mean_price=("price", "mean"),
        freight_ratio=("freight_ratio", "median"),
        mean_inst=("installments", "mean"),
        n_products=("product_id", "nunique"),
        n_categories=("category", "nunique"),
        photos_qty=("photos_qty", "mean"),
        desc_len=("desc_len", "mean"),
        n_cust=("customer_unique_id", "nunique"),
        n_cust_states=("customer_state", "nunique"),
        cross_state_rate=("cross_state", "mean"),
        multi_seller_rate=("n_sellers", lambda s: (s > 1).mean()),
        **{c: (c, "mean") for c in cx_cols})
    # concentration: share of orders from the single largest customer
    top = (df.groupby(["seller_id", "customer_unique_id"]).size()
             .groupby("seller_id").max())
    out["top_cust_share"] = top / out["n_orders"]
    out["pct_inst_ge6"] = (df.assign(f=(df["installments"].fillna(1) >= 6).astype(int))
                             .groupby("seller_id")["f"].mean())
    out["repeat_cust_rate"] = 1 - out["n_cust"] / out["n_orders"]
    return out


def build_panel(om, cx_cols):
    tmin, tmax = om["date"].min(), om["date"].max()
    start = tmin + timedelta(days=LOOKBACK)
    snaps, rows = [], []
    t = start
    while t + timedelta(days=HORIZON) <= tmax:
        snaps.append(t); t += timedelta(days=STEP)

    first_seen = om.groupby("seller_id")["date"].min()
    cat_of = om.groupby("seller_id")["category"].agg(
        lambda s: s.mode().iat[0] if len(s.mode()) else "unknown")

    for t in snaps:
        win = om[(om["date"] >= t - timedelta(days=LOOKBACK)) & (om["date"] < t)]
        rec = om[(om["date"] >= t - timedelta(days=30)) & (om["date"] < t)]
        fut = om[(om["date"] >= t) & (om["date"] < t + timedelta(days=HORIZON))]
        if len(win) == 0 or len(fut) == 0:
            continue
        A = _agg_window(win, cx_cols)
        A = A[A["n_orders"] >= MIN_ORDERS_FEATURE]
        if A.empty:
            continue
        R = rec.groupby("seller_id").agg(bad_rate_30=("is_bad", "mean"),
                                         n_orders_30=("order_id", "count"),
                                         mean_delay_30=("delay_days", "mean"))
        F = fut.groupby("seller_id").agg(fut_bad=("is_bad", "mean"),
                                         fut_n=("order_id", "count"),
                                         fut_financed=("financed_value", "sum"),
                                         fut_bad_financed=(
                                             "financed_value",
                                             lambda s: s[fut.loc[s.index, "is_bad"] == 1].sum()))
        df = A.join(R, how="left").join(F, how="inner")
        df = df[df["fut_n"] >= MIN_ORDERS_TARGET]
        if df.empty:
            continue
        df["snapshot"] = t
        df["exp_bad"] = win.groupby("seller_id")["exp_bad"].mean().reindex(df.index)
        df["resid_bad"] = df["bad_rate"] - df["exp_bad"]
        df["category"] = cat_of.reindex(df.index).values
        df["tenure_days"] = (t - first_seen.reindex(df.index)).dt.days
        # velocity + momentum
        df["velocity_ratio"] = (df["n_orders_30"].fillna(0) * 3) / df["n_orders"]
        df["bad_momentum"] = df["bad_rate_30"].fillna(df["bad_rate"]) - df["bad_rate"]
        df["delay_momentum"] = df["mean_delay_30"].fillna(df["mean_delay"]) - df["mean_delay"]
        # category-peer z-scores (regional/category anomaly detection)
        for col in ["bad_rate", "late_rate", "mean_delay", "mean_price", "cancel_rate"]:
            gm = df.groupby("category")[col].transform("mean")
            gs = df.groupby("category")[col].transform("std").replace(0, np.nan)
            df[f"z_{col}"] = ((df[col] - gm) / gs).fillna(0)
        rows.append(df.reset_index())

    panel = pd.concat(rows, ignore_index=True)
    panel["target"] = (panel["fut_bad"] >= BAD_RATE_THRESHOLD).astype(int)
    return panel


# -------------------------------------------------------- 4. COLLUSION GRAPH
def collusion_graph(om, min_shared=3):
    sc = (om.groupby(["seller_id", "customer_unique_id"]).size()
            .reset_index(name="n"))
    m = sc.merge(sc, on="customer_unique_id")
    m = m[m["seller_id_x"] < m["seller_id_y"]]
    edges = (m.groupby(["seller_id_x", "seller_id_y"])
               .agg(shared_customers=("customer_unique_id", "nunique"))
               .reset_index())
    edges = edges[edges["shared_customers"] >= min_shared] \
        .sort_values("shared_customers", ascending=False)
    return edges


# ------------------------------------------------------------ 5. ACTION LAYER
def action(p):
    """p is the SENTINEL percentile score (0-1): band = position in the book."""
    if p >= 0.95:
        return ("CRITICAL", 0.00, 0, 1.00,
                "Suspend new financing. Recover outstanding. Field audit.")
    if p >= 0.85:
        return ("HIGH", 0.25, 3, 0.15,
                "Cap exposure to 25% of current. Max 3-mo tenure. 15% holdback.")
    if p >= 0.70:
        return ("WATCH", 0.60, 6, 0.07,
                "Cap exposure to 60%. Max 6-mo tenure. 7% holdback. Weekly review.")
    return ("HEALTHY", 1.00, 12, 0.00, "Full limits. Standard monitoring.")


# ------------------------------------------------------------------- 6. MAIN
def main():
    print("[1/6] loading Olist tables ...")
    raw = load()
    print("[2/6] building order master + weak labels ...")
    om, cx_cols = build_order_master(*raw)
    om.to_parquet(f"{OUT}/order_master.parquet", index=False)
    print(f"      {len(om):,} orders | bad-order rate = {om['is_bad'].mean():.3f}")

    print("[3/6] building seller panel (rolling snapshots) ...")
    panel = build_panel(om, cx_cols)
    panel.to_parquet(f"{OUT}/panel.parquet", index=False)
    print(f"      {len(panel):,} seller-snapshots | "
          f"{panel['seller_id'].nunique():,} sellers | "
          f"deterioration rate = {panel['target'].mean():.3f}")

    print("[4/6] training forecaster (time-based split) ...")
    from model import train
    res = train(panel)

    print("[5/6] building collusion graph ...")
    edges = collusion_graph(om)
    edges.to_csv(f"{OUT}/collusion_edges.csv", index=False)
    print(f"      {len(edges):,} suspicious seller pairs")

    print("[6/6] scoring latest snapshot + action layer ...")
    latest = panel[panel["snapshot"] == panel["snapshot"].max()].copy()
    latest["eb_rate"] = __import__("scorer").eb_shrink(latest)
    latest["risk_prob"] = res["scorer"].prob(latest)
    latest["sentinel_score"] = res["scorer"].score(latest)
    acts = latest["sentinel_score"].apply(action)
    latest["risk_band"] = [a[0] for a in acts]
    latest["exposure_multiplier"] = [a[1] for a in acts]
    latest["max_tenure_months"] = [a[2] for a in acts]
    latest["holdback_pct"] = [a[3] for a in acts]
    latest["recommendation"] = [a[4] for a in acts]
    latest["current_exposure"] = latest["financed_value"]
    latest["revised_limit"] = latest["current_exposure"] * latest["exposure_multiplier"]
    latest = latest.sort_values("sentinel_score", ascending=False)
    latest.to_parquet(f"{OUT}/seller_scores.parquet", index=False)

    summary = {**res["metrics"],
               "n_orders": int(len(om)),
               "bad_order_rate": float(om["is_bad"].mean()),
               "n_seller_snapshots": int(len(panel)),
               "n_sellers": int(panel["seller_id"].nunique()),
               "deterioration_rate": float(panel["target"].mean()),
               "n_collusion_edges": int(len(edges)),
               "band_counts": latest["risk_band"].value_counts().to_dict(),
               "exposure_at_risk": float(
                   latest.loc[latest.risk_band.isin(["CRITICAL", "HIGH"]),
                              "current_exposure"].sum())}
    from scorer import roi
    summary["roi_inr"] = roi(summary)
    json.dump(summary, open(f"{OUT}/metrics.json", "w"), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
