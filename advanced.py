"""Advanced analyses that turn a score into an argument.

1. SURVIVAL   — how long until a merchant in each band actually deteriorates.
2. GAMING     — could a merchant manipulate its way to a good score?
3. RULES      — the model distilled to something a credit officer can apply by hand.
4. LEVERS     — for a given merchant, which fixable behaviour moves the score most.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, _tree

OUT = os.environ.get("SENTINEL_OUT", "artifacts")
BANDS = [("CRITICAL", 0.95, 1.01), ("HIGH", 0.85, 0.95),
         ("WATCH", 0.70, 0.85), ("HEALTHY", 0.0, 0.70)]

# Features a merchant could plausibly manipulate: reviews can be solicited, bought
# or suppressed; catalogue copy and photos are edited freely. Delivery timestamps,
# cancellations and payment records come from systems the merchant does not own.
GAMEABLE = {"mean_review", "badreview_rate", "review_cov", "photos_qty", "desc_len",
            "cx_not_received", "cx_late", "cx_damaged", "cx_wrong_item",
            "cx_counterfeit", "cx_refund", "z_bad_rate", "bad_rate", "bad_rate_30",
            "bad_momentum", "exp_bad", "resid_bad"}

# Features a merchant can genuinely improve through operations - these are the
# ones worth telling a dealer about.
LEVERS = ["mean_delay", "p90_delay", "carrier_lag", "approval_lag", "late_rate",
          "cancel_rate", "freight_ratio", "photos_qty", "desc_len"]


# ------------------------------------------------------------- 1. SURVIVAL
def survival(sc, horizon_days=180):
    """Kaplan-Meier: time from being scored into a band to first deterioration.

    A risk band should not just rank merchants - it should tell you how much time
    you have. Merchants entering CRITICAL should fail sooner than those in WATCH.
    """
    sc = sc.sort_values(["seller_id", "snapshot"])
    rows = []
    for sid, g in sc.groupby("seller_id"):
        g = g.reset_index(drop=True)
        seen = set()
        for i, r in g.iterrows():
            b = next(n for n, lo, hi in BANDS if lo <= r["oos_score"] < hi)
            if b in seen:
                continue
            seen.add(b)
            fut = g.iloc[i + 1:]
            bad = fut[fut["target"] == 1]
            if len(bad):
                t = (bad["snapshot"].iloc[0] - r["snapshot"]).days
                rows.append({"band": b, "t": t, "event": 1})
            elif len(fut):
                t = (fut["snapshot"].iloc[-1] - r["snapshot"]).days
                rows.append({"band": b, "t": t, "event": 0})
    d = pd.DataFrame(rows)
    curves, summary = [], {}
    for b, g in d.groupby("band"):
        g = g.sort_values("t")
        n = len(g)
        surv, at_risk = 1.0, n
        pts = [{"t": 0, "s": 1.0}]
        for t, grp in g.groupby("t"):
            ev = int(grp["event"].sum())
            if at_risk > 0 and ev:
                surv *= (1 - ev / at_risk)
            pts.append({"t": int(t), "s": round(surv, 4), "at_risk": int(at_risk)})
            at_risk -= len(grp)
        curves += [{"band": b, **p} for p in pts if p["t"] <= horizon_days]
        # median survival, and share failing within 90 days
        med = next((p["t"] for p in pts if p["s"] <= 0.5), None)
        s90 = next((p["s"] for p in reversed(pts) if p["t"] <= 90), 1.0)
        summary[b] = {"n": int(n), "events": int(g["event"].sum()),
                      "median_days_to_deterioration": med,
                      "pct_deteriorated_by_90d": round(1 - s90, 4)}
    pd.DataFrame(curves).to_csv(f"{OUT}/survival_curves.csv", index=False)
    return summary, curves


# --------------------------------------------------------------- 2. GAMING
def gaming(panel, feats, horizon=30, start_frac=0.45, step=3, seeds=3):
    """Retrain using only features the merchant cannot manipulate.

    If performance collapses, the score is a review-sentiment detector wearing a
    risk-model costume, and any dealer who buys reviews defeats it.
    """
    from scorer import CFG
    import lightgbm as lgb
    hard = [f for f in feats if f not in GAMEABLE]
    soft = [f for f in feats if f in GAMEABLE]
    snaps = np.sort(panel["snapshot"].unique())
    rows = []
    for cut in snaps[int(len(snaps) * start_frac)::step]:
        tr = panel[panel.snapshot <= cut - pd.Timedelta(days=horizon)]
        te = panel[panel.snapshot == cut]
        if len(te) < 100 or te["target"].nunique() < 2 or tr["target"].sum() < 60:
            continue
        r = {"fold": str(cut)[:10], "n": len(te)}
        for name, fs in [("all", feats), ("hard_only", hard), ("gameable_only", soft)]:
            p = np.mean([lgb.LGBMClassifier(**CFG, random_state=s, verbose=-1)
                         .fit(tr[fs].astype(float), tr["target"])
                         .predict_proba(te[fs].astype(float))[:, 1]
                         for s in range(seeds)], axis=0)
            r[name] = roc_auc_score(te["target"], p)
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/gaming.csv", index=False)
    from scipy import stats
    return {"n_folds": int(len(d)), "n_hard": len(hard), "n_gameable": len(soft),
            "auc_all": float(d["all"].mean()),
            "auc_hard_only": float(d["hard_only"].mean()),
            "auc_gameable_only": float(d["gameable_only"].mean()),
            "retention": float(d["hard_only"].mean() / d["all"].mean()),
            "p_hard_vs_all": float(stats.ttest_rel(d["hard_only"], d["all"]).pvalue),
            "hard_features": hard[:40]}


# ---------------------------------------------------------------- 3. RULES
def extract_rules(panel, feats, max_depth=4, min_lift=1.25, min_leaf=100):
    """Distil the model into a handful of if/then rules.

    Two uses: a credit officer can apply the policy the day the model goes down,
    and a committee can sanity-check that the logic is not absurd.
    """
    X = panel[feats].astype(float).fillna(0)
    y = panel["target"]
    t = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                               random_state=0).fit(X, y)
    tree, base = t.tree_, y.mean()
    rules = []

    def walk(node, conds):
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            # sklearn >=1.4 stores tree_.value as class PROPORTIONS, not counts.
            # Sample counts must come from n_node_samples or every rule is dropped.
            n = int(tree.n_node_samples[node])
            v = tree.value[node][0]
            rate = float(v[1] / v.sum()) if v.sum() else 0.0
            if n >= min_leaf and rate / base >= min_lift:
                rules.append({"conditions": conds[:], "n": n,
                              "bad_rate": round(rate, 4),
                              "lift": round(rate / base, 2)})
            return
        f = feats[tree.feature[node]]
        thr = round(float(tree.threshold[node]), 4)
        walk(tree.children_left[node], conds + [f"{f} <= {thr}"])
        walk(tree.children_right[node], conds + [f"{f} > {thr}"])

    walk(0, [])
    rules.sort(key=lambda r: -r["lift"])
    covered = sum(r["n"] for r in rules)
    json.dump({"base_rate": round(float(base), 4), "rules": rules,
               "coverage": round(covered / len(panel), 4)},
              open(f"{OUT}/rules.json", "w"), indent=2)
    return {"base_rate": float(base), "n_rules": len(rules),
            "coverage": float(covered / len(panel)), "rules": rules}


# --------------------------------------------------------------- 4. LEVERS
def levers(scores, scorer, feats, peer, top_n=60):
    """For each merchant: which fixable behaviour would move the score most?

    Turns a decline into a conversation. TVS would rather rehabilitate a dealer
    than lose the volume, and this is the list of what to ask them to fix.
    """
    out = {}
    avail = [l for l in LEVERS if l in feats]
    for _, r in scores.head(top_n).iterrows():
        base = float(scorer.prob(pd.DataFrame([r[feats]]))[0])
        deltas = []
        for f in avail:
            if pd.isna(r[f]):
                continue
            row = r.copy()
            row[f] = peer[f]                       # move to portfolio norm
            new = float(scorer.prob(pd.DataFrame([row[feats]]))[0])
            if base - new > 0.002:
                deltas.append({"feature": f, "current": round(float(r[f]), 3),
                               "target": round(float(peer[f]), 3),
                               "score_drop": round(base - new, 4)})
        deltas.sort(key=lambda x: -x["score_drop"])
        out[r["seller_id"]] = {"base_prob": round(base, 4), "levers": deltas[:4]}
    json.dump(out, open(f"{OUT}/levers.json", "w"), indent=2)
    return out


# ------------------------------------------------------------------- MAIN
def main():
    import pickle
    sc = pd.read_parquet(f"{OUT}/panel_scored.parquet")
    panel = pd.read_parquet(f"{OUT}/panel.parquet")
    S = pd.read_parquet(f"{OUT}/seller_scores.parquet")
    B = pickle.load(open(f"{OUT}/model.pkl", "rb"))
    feats, scorer = B["features"], B["scorer"]

    print("[1/4] survival by risk band ...")
    surv, _ = survival(sc)
    for b in ["CRITICAL", "HIGH", "WATCH", "HEALTHY"]:
        if b in surv:
            v = surv[b]
            md = v["median_days_to_deterioration"]
            print(f"      {b:9s} n={v['n']:5d}  median days to deterioration="
                  f"{md if md else '>window':>8}  by 90d={v['pct_deteriorated_by_90d']:.1%}")

    print("\n[2/4] gaming resistance ...")
    g = gaming(panel, feats)
    print(f"      all {len(feats)} features      AUC {g['auc_all']:.4f}")
    print(f"      {g['n_hard']} hard features only  AUC {g['auc_hard_only']:.4f}  "
          f"({g['retention']:.1%} retained)")
    print(f"      {g['n_gameable']} gameable only      AUC {g['auc_gameable_only']:.4f}")

    print("\n[3/4] rule extraction ...")
    r = extract_rules(panel, feats)
    print(f"      base rate {r['base_rate']:.3f} | {r['n_rules']} rules | "
          f"coverage {r['coverage']:.1%}")
    for x in r["rules"][:3]:
        print(f"        IF {' AND '.join(x['conditions'])}")
        print(f"           -> {x['bad_rate']:.1%} bad ({x['lift']}x), n={x['n']}")

    print("\n[4/4] improvement levers ...")
    peer = S[feats].median()
    lv = levers(S, scorer, feats, peer)
    ex = [(k, v) for k, v in lv.items() if v["levers"]][:3]
    for k, v in ex:
        top = v["levers"][0]
        print(f"      {k[:12]}: fix {top['feature']} "
              f"({top['current']} -> {top['target']}) drops risk {top['score_drop']:.3f}")

    json.dump({"survival": surv, "gaming": g,
               "rules": {k: v for k, v in r.items() if k != "rules"},
               "n_levers_scored": len(lv)},
              open(f"{OUT}/advanced_metrics.json", "w"), indent=2, default=str)
    print("\nwrote advanced_metrics.json, survival_curves.csv, gaming.csv, "
          "rules.json, levers.json")


if __name__ == "__main__":
    main()
