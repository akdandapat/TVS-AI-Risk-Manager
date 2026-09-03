"""Model depth: uncertainty + abstention, multi-horizon validation, segment stability.

Three questions a credit-risk committee asks that v4 cannot answer:
  1. How confident are you in THIS merchant's score?
  2. Does the warning hold at 60 and 90 days, or only at 30?
  3. Does it work for small merchants, or only where you have volume?
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import roc_auc_score

OUT = os.environ.get("SENTINEL_OUT", "artifacts" if os.path.exists("artifacts") else "/home/claude/sentinel/artifacts")
if not os.path.exists(OUT) and os.path.exists("artifacts"):
    OUT = "artifacts"
BANDS = [(0.95, "CRITICAL"), (0.85, "HIGH"), (0.70, "WATCH"), (0.0, "HEALTHY")]
Z = 1.96


def band_of(s):
    for t, n in BANDS:
        if s >= t:
            return n
    return "HEALTHY"


# ------------------------------------------------- 1. UNCERTAINTY + ABSTENTION
def score_interval(sc, k=10, z=Z):
    """A confidence interval on the SENTINEL score, driven by how much evidence
    backs the merchant's observed rate.

    The dominant term (70% of the score) is the empirical-Bayes shrunk rate. Its
    uncertainty is the binomial standard error of the underlying bad rate, damped
    by the same shrinkage that produced it. A merchant with 6 orders gets a wide
    interval; one with 300 gets a narrow one. We then map the rate bounds back
    through the same empirical percentile transform used to build the score.
    """
    d = sc.copy()
    n = d["n_orders"].clip(lower=1).values
    p = d["bad_rate"].fillna(0).clip(0, 1).values
    prior = d["exp_bad"].fillna(p.mean()).values
    se = np.sqrt(np.clip(p * (1 - p), 1e-9, None) / n)
    lo_r = np.clip(p - z * se, 0, 1)
    hi_r = np.clip(p + z * se, 0, 1)
    shrink = lambda r: (n * r + k * prior) / (n + k)
    eb_lo, eb_hi = shrink(lo_r), shrink(hi_r)

    ref = np.sort(d["oos_eb"].values)
    pctl = lambda v: np.searchsorted(ref, v, side="right") / max(len(ref), 1)
    model_term = 0.30 * (np.searchsorted(np.sort(d["oos_prob"].values),
                                         d["oos_prob"].values, side="right")
                         / max(len(d), 1))
    d["score_lo"] = np.clip(model_term + 0.70 * pctl(eb_lo), 0, 1)
    d["score_hi"] = np.clip(model_term + 0.70 * pctl(eb_hi), 0, 1)
    d["score_width"] = d["score_hi"] - d["score_lo"]
    d["band_lo"] = [band_of(s) for s in d["score_lo"]]
    d["band_hi"] = [band_of(s) for s in d["score_hi"]]
    d["confident"] = (d["band_lo"] == d["band_hi"])
    return d


def abstention_report(d, score_col="oos_score"):
    """Does refusing to score the uncertain merchants actually help?

    If it does, precision on the confident subset beats precision overall, and we
    can honestly tell a credit committee which files to trust.
    """
    out = {}
    for label, sub in [("all", d), ("confident", d[d.confident]),
                       ("abstained", d[~d.confident])]:
        if len(sub) < 50 or sub["target"].nunique() < 2:
            out[label] = {"n": int(len(sub))}
            continue
        k = max(int(0.20 * len(sub)), 1)
        top = np.argsort(-sub[score_col].values)[:k]
        base = sub["target"].mean()
        out[label] = {
            "n": int(len(sub)),
            "share": float(len(sub) / len(d)),
            "base_rate": float(base),
            "auc": float(roc_auc_score(sub["target"], sub[score_col])),
            "lift20": float(sub["target"].iloc[top].mean() / base),
            "median_orders": float(sub["n_orders"].median()),
            "mean_width": float(sub["score_width"].mean())}
    return out


# ------------------------------------------------------- 2. MULTI-HORIZON
def add_horizons(panel, om, horizons=(30, 60, 90), thresh=0.25, min_n=8):
    """Recompute forward outcomes at 60 and 90 days for the SAME snapshots.

    If the merchant score only predicts the next 30 days it is a nowcast. If it
    still separates at 90 days it is genuinely an early-warning system.
    """
    p = panel.copy()
    for h in horizons:
        if h == 30:
            p["target_30"] = p["target"]
            p["fut_n_30"] = p["fut_n"]
            continue
        recs = []
        for t, g in p.groupby("snapshot"):
            fut = om[(om["date"] >= t) & (om["date"] < t + timedelta(days=h))]
            a = fut.groupby("seller_id").agg(fb=("is_bad", "mean"),
                                             fn=("order_id", "count"))
            a = a.reindex(g["seller_id"].values)
            recs.append(pd.DataFrame({"seller_id": g["seller_id"].values,
                                      "snapshot": t,
                                      f"fut_bad_{h}": a["fb"].values,
                                      f"fut_n_{h}": a["fn"].values}))
        r = pd.concat(recs, ignore_index=True)
        p = p.merge(r, on=["seller_id", "snapshot"], how="left")
        p[f"target_{h}"] = (p[f"fut_bad_{h}"] >= thresh).astype("float")
        p.loc[p[f"fut_n_{h}"].fillna(0) < min_n, f"target_{h}"] = np.nan
    return p


def horizon_report(p, score_col="oos_score", horizons=(30, 60, 90)):
    out = {}
    for h in horizons:
        col = f"target_{h}"
        s = p[p[col].notna() & p[score_col].notna()]
        if len(s) < 200 or s[col].nunique() < 2:
            out[f"h{h}"] = {"n": int(len(s))}
            continue
        y = s[col].astype(int)
        k = max(int(0.20 * len(s)), 1)
        top = np.argsort(-s[score_col].values)[:k]
        base = y.mean()
        # per-snapshot AUC so we can report dispersion honestly
        aucs = [roc_auc_score(g[col].astype(int), g[score_col])
                for _, g in s.groupby("snapshot")
                if g[col].nunique() > 1 and len(g) >= 40]
        out[f"h{h}"] = {"n": int(len(s)), "base_rate": float(base),
                        "auc": float(roc_auc_score(y, s[score_col])),
                        "auc_mean_per_snapshot": float(np.mean(aucs)) if aucs else None,
                        "auc_std_per_snapshot": float(np.std(aucs)) if aucs else None,
                        "lift20": float(y.iloc[top].mean() / base)}
    return out


# ---------------------------------------------------- 3. SEGMENT STABILITY
def segment_report(d, score_col="oos_score"):
    """A model that only works on large merchants is not deployable: the small ones
    are exactly where a lender has least other information."""
    d = d.copy()
    d["size_seg"] = pd.qcut(d["n_orders"].rank(method="first"), 3,
                            labels=["small", "medium", "large"])
    d["tenure_seg"] = pd.cut(d["tenure_days"].fillna(0),
                             [-1, 180, 365, 1e9], labels=["<6mo", "6-12mo", "12mo+"])
    d["expo_seg"] = pd.qcut(d["financed_value"].fillna(0).rank(method="first"), 3,
                            labels=["low", "mid", "high"])
    out = {}
    for dim in ["size_seg", "tenure_seg", "expo_seg"]:
        rows = {}
        for seg, g in d.groupby(dim, observed=True):
            if len(g) < 100 or g["target"].nunique() < 2:
                rows[str(seg)] = {"n": int(len(g))}
                continue
            k = max(int(0.20 * len(g)), 1)
            top = np.argsort(-g[score_col].values)[:k]
            base = g["target"].mean()
            rows[str(seg)] = {
                "n": int(len(g)), "base_rate": float(base),
                "auc": float(roc_auc_score(g["target"], g[score_col])),
                "naive_auc": float(roc_auc_score(g["target"], g["oos_naive"])),
                "lift20": float(g["target"].iloc[top].mean() / base),
                "median_orders": float(g["n_orders"].median())}
        out[dim] = rows
    return out


# ------------------------------------------------------------------- MAIN
def main():
    sc = pd.read_parquet(f"{OUT}/panel_scored.parquet")
    om = pd.read_parquet(f"{OUT}/order_master.parquet")

    print("[1/3] score intervals + abstention ...")
    d = score_interval(sc)
    d.to_parquet(f"{OUT}/panel_confidence.parquet", index=False)
    ab = abstention_report(d)
    for kk, v in ab.items():
        if "auc" in v:
            print(f"      {kk:10s} n={v['n']:5d} ({v['share']:5.1%})  AUC={v['auc']:.4f}  "
                  f"lift20={v['lift20']:.2f}  med orders={v['median_orders']:.0f}  "
                  f"width={v['mean_width']:.3f}")

    print("[2/3] multi-horizon (30 / 60 / 90 days) ...")
    p = add_horizons(d, om)
    p.to_parquet(f"{OUT}/panel_horizons.parquet", index=False)
    hz = horizon_report(p)
    for kk, v in hz.items():
        if "auc" in v:
            sd = f" +-{v['auc_std_per_snapshot']:.3f}" if v.get("auc_std_per_snapshot") else ""
            print(f"      {kk:5s} n={v['n']:5d}  base={v['base_rate']:.3f}  "
                  f"AUC={v['auc']:.4f}{sd}  lift20={v['lift20']:.2f}")

    print("[3/3] segment stability ...")
    seg = segment_report(d)
    for dim, rows in seg.items():
        print(f"      {dim}")
        for s, v in rows.items():
            if "auc" in v:
                print(f"        {s:8s} n={v['n']:5d} AUC={v['auc']:.4f} "
                      f"(naive {v['naive_auc']:.4f})  lift20={v['lift20']:.2f} "
                      f"med orders={v['median_orders']:.0f}")

    json.dump({"abstention": ab, "horizons": hz, "segments": seg},
              open(f"{OUT}/depth_metrics.json", "w"), indent=2, default=str)
    print("\nwrote depth_metrics.json, panel_confidence.parquet, panel_horizons.parquet")


if __name__ == "__main__":
    main()
