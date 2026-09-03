"""Merge out-of-sample, policy and analytics metrics into artifacts/metrics.json.

Run AFTER pipeline.py, oos.py and analytics.py. Idempotent: safe to run repeatedly.

The important bit is the promotion at the bottom. pipeline.py writes an IN-SAMPLE lead
time (it scores the whole panel, including rows the model trained on). That number is
inflated. This script preserves it under `insample_*` and promotes the out-of-sample
figures to the headline keys the UI and README read.
"""
import json, os

OUT = os.environ.get("SENTINEL_OUT", "artifacts")


def load(name, required=True):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        if required:
            raise SystemExit(f"missing {p} — run the pipeline steps in order "
                             "(pipeline.py -> oos.py -> analytics.py -> finalize_metrics.py)")
        return {}
    return json.load(open(p))


def main():
    m = load("metrics.json")
    o = load("oos_metrics.json")
    a = load("analytics_metrics.json", required=False)

    # preserve the in-sample figures ONCE, so re-running never overwrites them
    m.setdefault("insample_median_lead_days", m.get("median_lead_days"))
    m.setdefault("insample_pct_caught_early", m.get("pct_caught_early"))

    # promote out-of-sample as the headline
    m["median_lead_days"] = o["oos_median_lead_days"]
    m["pct_caught_early"] = o["oos_pct_caught_early"]
    m["ew_caught_early"] = o["oos_ew_caught"]
    m["ew_eligible"] = o["oos_ew_eligible"]

    for k, v in o.items():
        if k.startswith(("oos_", "policy", "total_")):
            m[k] = v
    m.update(a)

    p = o.get("policy", {})
    if p:
        m["policy_capture_sentinel"] = p["s_sentinel"]["capture_at_20pct"]
        m["policy_capture_eb"] = p["s_eb"]["capture_at_20pct"]
        m["policy_capture_naive"] = p["s_naive"]["capture_at_20pct"]
        m["policy_capture_model"] = p["s_model"]["capture_at_20pct"]
        m["policy_wins"] = p["s_sentinel"]["wins_vs_naive"]
        m["policy_n"] = p["s_sentinel"]["n_snapshots"]
        m["policy_p"] = p["s_sentinel"]["p_vs_naive"]

    json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=str)

    print("metrics.json finalised")
    print(f"  lead time      {m['median_lead_days']:.0f} days out-of-sample "
          f"(in-sample was {m['insample_median_lead_days']:.0f})")
    print(f"  caught early   {m['pct_caught_early']:.1%} "
          f"(in-sample was {m['insample_pct_caught_early']:.1%})")
    if p:
        print(f"  policy@20%     {m['policy_capture_sentinel']:.1%} sentinel vs "
              f"{m['policy_capture_naive']:.1%} naive "
              f"| wins {m['policy_wins']}/{m['policy_n']} p={m['policy_p']:.4f}")
    if a:
        print(f"  products       {m['n_products_scored']:,} scored, "
              f"{m['n_products_delist']} delist, {m['n_products_restrict']} restrict")
        print(f"  demand shocks  {m['n_demand_shifts']} | "
              f"volume breaks {m['n_volume_breaks']} | "
              f"unstable features {m['n_features_unstable']}")


if __name__ == "__main__":
    main()
