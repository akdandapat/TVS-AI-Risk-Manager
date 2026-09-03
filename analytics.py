"""Product risk, seasonal/demand-shift detection, and drift monitoring.

Closes the parts of PS (g) the merchant model does not cover: "product performance",
"seasonal trends", "sudden shifts in consumer buying behaviour".
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

OUT = os.environ.get("SENTINEL_OUT", "/home/claude/sentinel/artifacts")
EB_K_PRODUCT = 8


# ---------------------------------------------------------- 1. PRODUCT RISK
def product_risk(om, min_orders=5):
    """Per-product risk, shrunk toward its category. A product with 5 orders and two
    complaints is not twice as risky as one with 200 orders and 40 - shrinkage says so."""
    p = om[om["product_id"].notna()].copy()
    cat = p.groupby("category").agg(cat_bad=("is_bad", "mean"),
                                    cat_n=("order_id", "count")).reset_index()
    g = p.groupby(["product_id", "category"]).agg(
        n_orders=("order_id", "count"),
        bad_rate=("is_bad", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        late_rate=("is_late", "mean"),
        badreview_rate=("is_badreview", "mean"),
        mean_review=("review_score", "mean"),
        mean_price=("price", "mean"),
        financed=("financed_value", "sum"),
        mean_delay=("delay_days", "mean"),
        photos=("photos_qty", "mean"),
        desc_len=("desc_len", "mean"),
        n_sellers=("seller_id", "nunique"),
        first_seen=("date", "min"), last_seen=("date", "max"),
        **{c: (c, "mean") for c in om.columns if c.startswith("cx_")}).reset_index()
    g = g[g["n_orders"] >= min_orders].merge(cat, on="category", how="left")
    g["cat_bad"] = g["cat_bad"].fillna(om["is_bad"].mean())
    g["risk"] = ((g["n_orders"] * g["bad_rate"] + EB_K_PRODUCT * g["cat_bad"])
                 / (g["n_orders"] + EB_K_PRODUCT))
    g["excess"] = g["risk"] - g["cat_bad"]          # worse than its own category
    g["exposure_at_risk"] = g["risk"] * g["financed"]
    g["pctile"] = g["risk"].rank(pct=True)
    g["action"] = np.select(
        [g.pctile >= 0.98, g.pctile >= 0.92, g.pctile >= 0.80],
        ["DELIST — stop financing this SKU",
         "RESTRICT — cash only, no EMI",
         "REVIEW — cap tenure at 3 months"],
        default="OK — standard terms")
    g = g.sort_values("exposure_at_risk", ascending=False)
    g.to_parquet(f"{OUT}/product_risk.parquet", index=False)
    return g


# ------------------------------------------- 2. SEASONALITY / DEMAND SHIFT
def seasonal_profile(om, min_orders=300):
    """Monthly seasonal index per category, computed on the category's SHARE of platform
    volume so that platform growth is not mistaken for seasonality."""
    m = om.copy()
    m["ym"] = m["date"].dt.to_period("M")
    m["month"] = m["date"].dt.month
    v = m.groupby(["category", "ym"]).size().reset_index(name="n")
    keep = m.groupby("category").size()
    keep = keep[keep >= min_orders].index
    v = v[v["category"].isin(keep)]
    v["month"] = v["ym"].dt.month
    tot = m.groupby("ym").size().rename("platform_n")
    v = v.merge(tot, on="ym", how="left")
    v["share"] = v["n"] / v["platform_n"].replace(0, np.nan)
    v["cat_mean"] = v.groupby("category")["share"].transform("mean")
    v["ratio"] = v["share"] / v["cat_mean"].replace(0, np.nan)
    idx = v.groupby(["category", "month"])["ratio"].mean().reset_index(name="seasonal_index")
    idx.to_csv(f"{OUT}/seasonal_index.csv", index=False)
    return idx, v


def demand_shifts(om, idx, min_hist=4, z_flag=2.0, min_volume=25, min_cv=0.03):
    """Deseasonalised volume shocks per category: what a genuine consumer-behaviour
    shift looks like once you strip out the fact that December is always busy."""
    m = om.copy()
    m["ym"] = m["date"].dt.to_period("M")
    v = m.groupby(["category", "ym"]).size().reset_index(name="n")
    # Detrend against platform growth: Olist roughly 10x'd over the window, so raw
    # volume surges just rediscover that. A genuine consumer-behaviour shift is a change
    # in a category's SHARE of the platform, after removing its seasonal pattern.
    tot = m.groupby("ym").size().rename("platform_n")
    v = v.merge(tot, on="ym", how="left")
    v["share"] = v["n"] / v["platform_n"].replace(0, np.nan)
    v["month"] = v["ym"].dt.month
    v = v.merge(idx, on=["category", "month"], how="left")
    v["seasonal_index"] = v["seasonal_index"].fillna(1.0)
    v["adj"] = v["share"] / v["seasonal_index"].replace(0, np.nan)
    v = v.sort_values(["category", "ym"])
    g = v.groupby("category")["adj"]
    v["roll_mean"] = g.transform(lambda s: s.shift(1).rolling(min_hist, min_periods=2).mean())
    v["roll_std"] = g.transform(lambda s: s.shift(1).rolling(min_hist, min_periods=2).std())
    # Guard against a near-zero rolling std producing meaningless z-scores: floor the
    # denominator at a minimum coefficient of variation, and require real volume.
    floor = (v["roll_mean"] * min_cv).clip(lower=1e-6)
    v["roll_std"] = v["roll_std"].fillna(0).clip(lower=0)
    v["denom"] = np.maximum(v["roll_std"], floor)
    v["z"] = ((v["adj"] - v["roll_mean"]) / v["denom"]).replace([np.inf, -np.inf], np.nan)
    ok = (v["n"] >= min_volume) & (v["platform_n"] >= 300) & v["z"].notna()
    v["shift"] = np.where(ok & (v["z"] >= z_flag), "SURGE",
                          np.where(ok & (v["z"] <= -z_flag), "COLLAPSE", ""))
    # pair the shift with what happened to quality in the same month
    q = m.groupby(["category", "ym"])["is_bad"].mean().reset_index(name="bad_rate")
    v = v.merge(q, on=["category", "ym"], how="left")
    v["ym"] = v["ym"].astype(str)
    flags = v[v["shift"] != ""].copy()
    v.to_csv(f"{OUT}/demand_series.csv", index=False)
    flags.to_csv(f"{OUT}/demand_shifts.csv", index=False)
    return v, flags


def merchant_volume_breaks(panel, z=2.5, min_volume=10, min_cv=0.05):
    """CUSUM-style break detection on a merchant's own order velocity. A merchant whose
    volume triples in a fortnight is either a success story or a fraud ring."""
    p = panel.sort_values(["seller_id", "snapshot"]).copy()
    g = p.groupby("seller_id")["n_orders"]
    p["v_mean"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    p["v_std"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    floor = (p["v_mean"] * min_cv).clip(lower=1.0)
    p["v_std"] = p["v_std"].fillna(0).clip(lower=0)
    p["v_z"] = ((p["n_orders"] - p["v_mean"]) / np.maximum(p["v_std"], floor)) \
        .replace([np.inf, -np.inf], np.nan)
    p = p[(p["v_mean"] >= min_volume) & p["v_z"].notna()]
    br = p[p["v_z"].abs() >= z][
        ["seller_id", "snapshot", "n_orders", "v_mean", "v_z", "bad_rate", "financed_value"]]
    br = br.assign(kind=np.where(br["v_z"] > 0, "VOLUME SURGE", "VOLUME COLLAPSE")) \
           .sort_values("snapshot", ascending=False)
    br.to_csv(f"{OUT}/volume_breaks.csv", index=False)
    return br


# ------------------------------------------------------------- 3. DRIFT (PSI)
def psi(expected, actual, bins=10):
    qs = np.nanquantile(expected, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return 0.0
    e = np.histogram(expected, bins=qs)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=qs)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(panel, feats, split_frac=0.70, top=25):
    """Population Stability Index between the training era and the live era.
    >0.25 is the conventional 'retrain now' line."""
    cut = panel["snapshot"].quantile(split_frac)
    tr, te = panel[panel.snapshot <= cut], panel[panel.snapshot > cut]
    rows = [{"feature": f, "psi": psi(tr[f].values, te[f].values)}
            for f in feats if pd.api.types.is_numeric_dtype(panel[f])]
    d = pd.DataFrame(rows).sort_values("psi", ascending=False)
    d["status"] = np.select([d.psi >= 0.25, d.psi >= 0.10],
                            ["UNSTABLE", "WATCH"], default="STABLE")
    d.to_csv(f"{OUT}/drift.csv", index=False)
    return d.head(top), d


if __name__ == "__main__":
    om = pd.read_parquet(f"{OUT}/order_master.parquet")
    panel = pd.read_parquet(f"{OUT}/panel.parquet")
    import pickle
    feats = pickle.load(open(f"{OUT}/model.pkl", "rb"))["features"]

    print("[1/4] product risk ...")
    pr = product_risk(om)
    print(f"      {len(pr):,} products scored | "
          f"{(pr.action.str.startswith('DELIST')).sum()} delist, "
          f"{(pr.action.str.startswith('RESTRICT')).sum()} restrict")
    print(pr.head(5)[["product_id", "category", "n_orders", "bad_rate",
                      "risk", "financed", "action"]].to_string(index=False))

    print("\n[2/4] seasonality ...")
    idx, _ = seasonal_profile(om)
    print(f"      seasonal index for {idx.category.nunique()} categories")

    print("[3/4] demand shifts ...")
    series, flags = demand_shifts(om, idx)
    print(f"      {len(flags)} deseasonalised shocks flagged")
    if len(flags):
        print(flags.nlargest(5, "z")[["category", "ym", "n", "z", "shift", "bad_rate"]]
              .to_string(index=False))

    print("\n[4/4] volume breaks + drift ...")
    br = merchant_volume_breaks(panel)
    topd, alld = drift_report(panel, feats)
    print(f"      {len(br)} merchant volume breaks | "
          f"{(alld.status=='UNSTABLE').sum()} unstable features, "
          f"{(alld.status=='WATCH').sum()} on watch")
    print(topd.head(6).to_string(index=False))

    json.dump({"n_products_scored": int(len(pr)),
               "n_products_delist": int((pr.action.str.startswith("DELIST")).sum()),
               "n_products_restrict": int((pr.action.str.startswith("RESTRICT")).sum()),
               "product_exposure_at_risk": float(pr.exposure_at_risk.sum()),
               "n_demand_shifts": int(len(flags)),
               "n_volume_breaks": int(len(br)),
               "n_features_unstable": int((alld.status == "UNSTABLE").sum()),
               "max_psi": float(alld.psi.max())},
              open(f"{OUT}/analytics_metrics.json", "w"), indent=2)
