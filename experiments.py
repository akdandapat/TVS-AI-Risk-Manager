"""Accuracy experiments. Purged walk-forward so overlapping windows can't leak."""
import warnings, itertools, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pipeline as P

CFG = dict(n_estimators=400, learning_rate=0.04, num_leaves=15, min_child_samples=40,
           subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0)
DROP = {"seller_id", "snapshot", "category", "target", "fut_bad", "fut_n",
        "fut_financed", "fut_bad_financed", "exp_bad", "date"}


# ---------------------------------------------- mix-adjusted expectation
def add_expected(om):
    """Leak-free expected badness from the seller's ORDER MIX (category / region / price).

    Separates 'this merchant is bad' from 'this merchant sells hard categories to
    hard regions'. Every rate uses strictly prior orders only.
    """
    om = om.sort_values("date").reset_index(drop=True)
    om["price_bkt"] = pd.qcut(om["price"].rank(method="first"), 10,
                              labels=False, duplicates="drop")
    parts = []
    for key in ["category", "customer_state", "price_bkt"]:
        r = om.groupby(key)["is_bad"].transform(lambda s: s.shift(1).expanding().mean())
        parts.append(r.fillna(om["is_bad"].expanding().mean().shift(1)))
    om["exp_bad"] = pd.concat(parts, axis=1).mean(axis=1)
    om["exp_bad"] = om["exp_bad"].fillna(om["is_bad"].mean())
    return om


def add_residual_feats(panel, om):
    """Per seller-snapshot: expected badness of their mix, and the residual."""
    out = []
    for t, g in panel.groupby("snapshot"):
        win = om[(om["date"] >= t - pd.Timedelta(days=LB)) & (om["date"] < t)]
        e = win.groupby("seller_id")["exp_bad"].mean()
        g = g.copy()
        g["exp_bad"] = g["seller_id"].map(e)
        g["resid_bad"] = g["bad_rate"] - g["exp_bad"]
        g["resid_ratio"] = g["bad_rate"] / g["exp_bad"].replace(0, np.nan)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------ purged walk-forward
def purged_wf(panel, horizon, depth=0.20, start=0.45, step=2, seeds=3):
    feats = [c for c in panel.columns
             if c not in DROP and pd.api.types.is_numeric_dtype(panel[c])]
    snaps = np.sort(panel["snapshot"].unique())
    rows = []
    for cut in snaps[int(len(snaps) * start)::step]:
        # PURGE: drop training snapshots whose target window overlaps the test window
        tr = panel[panel.snapshot <= cut - pd.Timedelta(days=horizon)]
        te = panel[panel.snapshot == cut]
        if len(te) < 100 or te.target.nunique() < 2 or tr.target.sum() < 60:
            continue
        pr = np.mean([lgb.LGBMClassifier(**CFG, random_state=s, verbose=-1)
                      .fit(tr[feats].astype(float), tr.target)
                      .predict_proba(te[feats].astype(float))[:, 1]
                      for s in range(seeds)], axis=0)
        bl = (0.75 * pd.Series(pr).rank(pct=True).values
              + 0.25 * te.bad_rate.fillna(0).rank(pct=True).values)
        nb = te.bad_rate.fillna(0).values
        b, k = te.target.mean(), max(int(depth * len(te)), 1)
        f = lambda s: (roc_auc_score(te.target, s),
                       te.target.iloc[np.argsort(-np.asarray(s))[:k]].mean() / b)
        ma, ml = f(bl); ba, bla = f(nb)
        rows.append(dict(fold=str(cut)[:10], n=len(te), m_auc=ma, m_lift=ml,
                         b_auc=ba, b_lift=bla))
    d = pd.DataFrame(rows)
    if len(d) < 3:
        return d, None
    from scipy import stats
    return d, dict(folds=len(d), m_auc=d.m_auc.mean(), m_sd=d.m_auc.std(),
                   b_auc=d.b_auc.mean(), m_lift=d.m_lift.mean(),
                   b_lift=d.b_lift.mean(), wins=int((d.m_auc > d.b_auc).sum()),
                   p=float(stats.ttest_rel(d.m_auc, d.b_auc).pvalue))


def run_one(LB, HZ, ST, MT, om, cx):
    P.LOOKBACK, P.HORIZON, P.STEP, P.MIN_ORDERS_TARGET = LB, HZ, ST, MT
    globals()["LB"] = LB
    pan = P.build_panel(om, cx)
    out = []
    if True:
        for use_res in (False, True):
            p2 = add_residual_feats(pan, om) if use_res else pan.copy()
            if not use_res:
                p2 = p2.drop(columns=[c for c in ("exp_bad", "resid_bad", "resid_ratio")
                                      if c in p2], errors="ignore")
            d, s = purged_wf(p2, HZ)
            if s is None:
                continue
            tag = f"LB{LB} HZ{HZ} ST{ST} MT{MT} {'+resid' if use_res else 'base'}"
            print(f"{tag:34s} n={len(p2):6d} folds={s['folds']:2d} "
                  f"AUC={s['m_auc']:.4f}+-{s['m_sd']:.3f} (naive {s['b_auc']:.4f}) "
                  f"lift={s['m_lift']:.2f}v{s['b_lift']:.2f} "
                  f"wins={s['wins']}/{s['folds']} p={s['p']:.4f}", flush=True)
            out.append(dict(tag=tag, n=len(p2), **s))
    return out, pan


if __name__ == "__main__":
    import sys, os
    LB, HZ, ST, MT = map(int, sys.argv[1:5])
    om = pd.read_parquet("artifacts/om_exp.parquet")
    cx = json.load(open("artifacts/cx.json"))
    out, _ = run_one(LB, HZ, ST, MT, om, cx)
    f = "artifacts/experiments.csv"
    df = pd.DataFrame(out)
    if os.path.exists(f):
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.to_csv(f, index=False)
