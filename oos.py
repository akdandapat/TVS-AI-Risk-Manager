"""Out-of-sample scoring + policy backtest.

Everything downstream (lead time, alerts, policy P&L) must use scores produced by a model
that never saw the snapshot it is scoring. The v3 lead-time metric scored the whole panel
including training rows, which inflated it. This module fixes that.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

OUT = os.environ.get("SENTINEL_OUT", "/home/claude/sentinel/artifacts")
from scorer import CFG, W_MODEL, eb_shrink, _fit_ensemble

DROP = {"seller_id", "snapshot", "category", "target", "date",
        "fut_bad", "fut_n", "fut_financed", "fut_bad_financed"}


def feature_cols(panel):
    return [c for c in panel.columns
            if c not in DROP and pd.api.types.is_numeric_dtype(panel[c])]


# ------------------------------------------------------ 1. OOS SCORING
def score_out_of_sample(panel, horizon=30, start_frac=0.35, seeds=3, refit_every=2):
    """Expanding-window, purged. Returns panel rows with an honest `oos_score`.

    The model is refit every `refit_every` snapshots (a real risk team retrains on a
    cadence, not continuously) and always excludes snapshots whose outcome window
    overlaps the one being scored.
    """
    feats = feature_cols(panel)
    snaps = np.sort(panel["snapshot"].unique())
    eval_snaps = snaps[int(len(snaps) * start_frac):]
    out, models, k = [], None, 0
    for i, cut in enumerate(eval_snaps):
        tr = panel[panel.snapshot <= cut - pd.Timedelta(days=horizon)]
        te = panel[panel.snapshot == cut]
        if len(te) == 0 or tr["target"].sum() < 40:
            continue
        if models is None or i % refit_every == 0:
            models = _fit_ensemble(tr, feats, seeds=seeds)
            ref_prob = np.mean([m.predict_proba(tr[feats].astype(float))[:, 1]
                                for m in models], axis=0)
            ref_prob = np.sort(ref_prob)
            ref_eb = np.sort(eb_shrink(tr))
            k += 1
        prob = np.mean([m.predict_proba(te[feats].astype(float))[:, 1]
                        for m in models], axis=0)
        ebv = eb_shrink(te)
        pctl = lambda ref, v: np.searchsorted(ref, v, side="right") / max(len(ref), 1)
        g = te.copy()
        g["oos_prob"] = prob
        g["oos_eb"] = ebv
        g["oos_score"] = (W_MODEL * pctl(ref_prob, prob)
                          + (1 - W_MODEL) * pctl(ref_eb, ebv))
        g["oos_naive"] = g["bad_rate"].fillna(0).values
        out.append(g)
    sc = pd.concat(out, ignore_index=True)
    sc.to_parquet(f"{OUT}/panel_scored.parquet", index=False)
    print(f"      scored {len(sc):,} snapshot-rows out-of-sample "
          f"({sc.snapshot.nunique()} snapshots, {k} model refits)")
    return sc


# --------------------------------------------------- 2. HONEST LEAD TIME
def lead_time_oos(sc, thresh=0.70, min_prior=2):
    """Same definition as before, but on out-of-sample scores only."""
    leads, eligible, missed = [], 0, 0
    for _, g in sc.sort_values("snapshot").groupby("seller_id"):
        bad = g[g["target"] == 1]
        if bad.empty:
            continue
        first_bad = bad["snapshot"].iloc[0]
        prior = g[g["snapshot"] < first_bad]
        if len(prior) < min_prior:
            continue
        eligible += 1
        flagged = prior[prior["oos_score"] >= thresh]
        if len(flagged):
            leads.append((first_bad - flagged["snapshot"].iloc[0]).days)
        else:
            missed += 1
    if not leads:
        return {"oos_ew_eligible": eligible, "oos_pct_caught_early": 0.0,
                "oos_median_lead_days": 0.0}
    L = np.array(leads)
    return {"oos_ew_threshold": thresh, "oos_ew_eligible": int(eligible),
            "oos_ew_caught": int(len(L)), "oos_ew_missed": int(missed),
            "oos_pct_caught_early": float(len(L) / eligible),
            "oos_median_lead_days": float(np.median(L)),
            "oos_mean_lead_days": float(L.mean()),
            "oos_max_lead_days": int(L.max())}


# ----------------------------------------------------- 3. POLICY BACKTEST
# The decision a risk team makes is: which merchants do we stop funding?
# Every scorer must be judged at EQUAL FINANCED VALUE DECLINED - not equal merchant
# count. Comparing at equal merchant count flatters any scorer that simply prefers
# large merchants, because it silently declines far more money for the same headcount.
GRID = np.arange(0.05, 0.61, 0.05)


def capture_curve(sc, col, grid=GRID):
    """Per snapshot: decline highest-scoring merchants until x% of financed value is
    cut; record the share of BAD financed value that fell inside the declined set."""
    rows = []
    for _, g in sc.groupby("snapshot"):
        g = g.sort_values(col, ascending=False)
        e = g["fut_financed"].fillna(0)
        b = g["fut_bad_financed"].fillna(0)
        if e.sum() <= 0 or b.sum() <= 0:
            continue
        ce, cb = e.cumsum() / e.sum(), b.cumsum() / b.sum()
        rows.append([float(np.interp(x, ce, cb)) for x in grid])
    return np.array(rows)


def run_policy(sc, margin=0.06, recovery=0.35, depth=0.20):
    """Capture curves + net benefit at a fixed decline budget."""
    from scipy import stats
    sc = sc.copy()
    R = lambda g, c: g[c].rank(pct=True)
    sc["s_sentinel"] = sc.groupby("snapshot", group_keys=False).apply(
        lambda g: W_MODEL * R(g, "oos_prob") + (1 - W_MODEL) * R(g, "oos_eb"))
    sc["s_eb"] = sc.groupby("snapshot", group_keys=False).apply(lambda g: R(g, "oos_eb"))
    sc["s_naive"] = sc.groupby("snapshot", group_keys=False).apply(lambda g: R(g, "oos_naive"))
    sc["s_model"] = sc.groupby("snapshot", group_keys=False).apply(lambda g: R(g, "oos_prob"))
    cols = ["s_sentinel", "s_eb", "s_naive", "s_model"]
    C = {c: capture_curve(sc, c) for c in cols}

    rows = []
    for c in cols:
        for i, x in enumerate(GRID):
            rows.append({"scorer": c, "declined": float(x),
                         "captured": float(C[c][:, i].mean())})
    pd.DataFrame(rows).to_csv(f"{OUT}/policy_frontier.csv", index=False)

    di = int(np.argmin(np.abs(GRID - depth)))
    tot_bad = float(sc["fut_bad_financed"].sum())
    tot_good = float((sc["fut_financed"] - sc["fut_bad_financed"]).clip(lower=0).sum())
    base = C["s_naive"].mean(axis=1)
    summary = {}
    for c in cols:
        cap = float(C[c][:, di].mean())
        avoided = cap * tot_bad
        forgone = depth * tot_good
        mine = C[c].mean(axis=1)
        summary[c] = {
            "mean_capture": float(C[c].mean()),
            f"capture_at_{int(depth*100)}pct": cap,
            "bad_avoided": avoided, "good_forgone": forgone,
            "net_benefit": avoided * recovery - forgone * margin,
            "wins_vs_naive": int((mine > base).sum()), "n_snapshots": int(len(mine)),
            "p_vs_naive": float(stats.ttest_rel(mine, base).pvalue)
            if c != "s_naive" else None}
    pd.DataFrame(summary).T.to_csv(f"{OUT}/policy_summary.csv")
    return {"policy": summary, "policy_depth": depth,
            "policy_margin": margin, "policy_recovery": recovery,
            "total_bad_financed": tot_bad, "total_good_financed": tot_good}


if __name__ == "__main__":
    panel = pd.read_parquet(f"{OUT}/panel.parquet")
    print("[1/3] out-of-sample scoring ...")
    sc = score_out_of_sample(panel)
    print("[2/3] honest lead time ...")
    lt = lead_time_oos(sc)
    print(json.dumps(lt, indent=2))
    print("[3/3] policy backtest ...")
    pol = run_policy(sc)
    print(f"  {'scorer':12s}{'capture@20%':>13s}{'mean cap':>10s}{'wins':>9s}{'p':>9s}")
    for k, v in pol["policy"].items():
        p = f"{v['p_vs_naive']:.4f}" if v["p_vs_naive"] is not None else "  --"
        print(f"  {k:12s}{v['capture_at_20pct']:12.1%}{v['mean_capture']:10.3f}"
              f"{v['wins_vs_naive']:6d}/{v['n_snapshots']}{p:>9s}")
    json.dump({**lt, **{k: v for k, v in pol.items()}},
              open(f"{OUT}/oos_metrics.json", "w"), indent=2, default=str)
