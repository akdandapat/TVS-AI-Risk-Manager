"""SENTINEL — Merchant & Product Risk Command Centre.  Run: streamlit run app.py"""
import json, os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

A = os.environ.get("SENTINEL_OUT", "artifacts")
st.set_page_config(page_title="SENTINEL | Merchant Risk", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
.stApp{background:#0b0e14}
h1,h2,h3,h4,label,p,span,div{color:#e6e9ef!important;font-family:'Inter',system-ui,sans-serif}
[data-testid="stMetricValue"]{font-size:1.9rem;font-weight:700}
[data-testid="stMetric"]{background:#141924;border:1px solid #222938;
 border-radius:10px;padding:14px 16px}
.band-CRITICAL{background:#3b0d12;border-left:4px solid #f2385a;padding:10px 14px;
 border-radius:6px;margin:6px 0}
.band-HIGH{background:#3a2410;border-left:4px solid #ff8c42;padding:10px 14px;
 border-radius:6px;margin:6px 0}
.band-WATCH{background:#3a350f;border-left:4px solid #ffd166;padding:10px 14px;
 border-radius:6px;margin:6px 0}
.band-HEALTHY{background:#0f2a1c;border-left:4px solid #2ecc71;padding:10px 14px;
 border-radius:6px;margin:6px 0}
</style>""", unsafe_allow_html=True)

PALETTE = {"CRITICAL": "#f2385a", "HIGH": "#ff8c42",
           "WATCH": "#ffd166", "HEALTHY": "#2ecc71"}


@st.cache_data
def load():
    m = json.load(open(f"{A}/metrics.json"))
    s = pd.read_parquet(f"{A}/seller_scores.parquet")
    p = pd.read_parquet(f"{A}/panel.parquet")
    o = pd.read_parquet(f"{A}/order_master.parquet")
    e = pd.read_csv(f"{A}/collusion_edges.csv")
    i = pd.read_csv(f"{A}/feature_importance.csv", index_col=0)
    b = pickle.load(open(f"{A}/model.pkl", "rb"))
    return m, s, p, o, e, i, b


M, S, P, OM, E, IMP, BUNDLE = load()
MODEL, FEATS = BUNDLE["model"], BUNDLE["features"]

st.title("SENTINEL")
st.caption("Intelligent Merchant & Product Risk Engine  ·  TVS Credit E.P.I.C 8.0 — PS (g)  "
           "·  98,666 real orders · 3,095 merchants")

page = st.sidebar.radio("View", ["Command Centre", "Merchant Deep-Dive",
                                 "Early-Warning Alerts", "Collusion Graph",
                                 "Product & Category Risk", "Model Validation"])
st.sidebar.markdown("---")
st.sidebar.metric("Portfolio exposure at risk", f"R$ {M['exposure_at_risk']:,.0f}")
st.sidebar.metric("Early-warning lead time", f"{M['median_lead_days']:.0f} days")
st.sidebar.caption(f"Model: {M['engine']} · AUC {M['roc_auc']:.3f} · "
                   f"decile lift {M['top_decile_lift']:.2f}x")


# ------------------------------------------------------------- COMMAND CENTRE
if page == "Command Centre":
    c = st.columns(5)
    bc = M["band_counts"]
    c[0].metric("Merchants scored", f"{len(S):,}")
    c[1].metric("Critical", bc.get("CRITICAL", 0))
    c[2].metric("High risk", bc.get("HIGH", 0))
    c[3].metric("Watchlist", bc.get("WATCH", 0))
    c[4].metric("Exposure at risk", f"R$ {M['exposure_at_risk']/1000:,.0f}k")

    l, r = st.columns([3, 2])
    with l:
        st.subheader("Risk distribution")
        d = S["risk_band"].value_counts().reindex(
            ["CRITICAL", "HIGH", "WATCH", "HEALTHY"]).fillna(0).reset_index()
        d.columns = ["band", "n"]
        fig = px.bar(d, x="n", y="band", orientation="h", color="band",
                     color_discrete_map=PALETTE, text="n")
        fig.update_layout(showlegend=False, height=280, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e9ef",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Exposure concentration by risk band")
        ex = S.groupby("risk_band")["current_exposure"].sum().reindex(
            ["CRITICAL", "HIGH", "WATCH", "HEALTHY"]).fillna(0).reset_index()
        fig2 = px.pie(ex, values="current_exposure", names="risk_band", hole=.55,
                      color="risk_band", color_discrete_map=PALETTE)
        fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#e6e9ef", margin=dict(t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with r:
        st.subheader("Action queue — top 10")
        for _, x in S.head(10).iterrows():
            st.markdown(
                f"<div class='band-{x.risk_band}'><b>{x.seller_id[:12]}…</b> "
                f"&nbsp;<code>{x.risk_prob:.0%}</code> &nbsp;<b>{x.risk_band}</b><br>"
                f"<small>{x.recommendation}<br>Exposure R$ {x.current_exposure:,.0f} "
                f"→ revised limit R$ {x.revised_limit:,.0f}</small></div>",
                unsafe_allow_html=True)

    st.subheader("Portfolio trend — bad-order rate over time")
    t = (OM.set_index("date").resample("ME")
           .agg(bad_rate=("is_bad", "mean"), orders=("order_id", "count")).reset_index())
    t = t[t.orders > 100]
    fig3 = go.Figure()
    fig3.add_bar(x=t.date, y=t.orders, name="Orders", marker_color="#243049", yaxis="y2")
    fig3.add_scatter(x=t.date, y=t.bad_rate, name="Bad-order rate",
                     line=dict(color="#f2385a", width=3))
    fig3.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e9ef",
                       yaxis=dict(title="Bad rate", tickformat=".0%"),
                       yaxis2=dict(overlaying="y", side="right", title="Orders",
                                   showgrid=False),
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------------------- DEEP DIVE
elif page == "Merchant Deep-Dive":
    sid = st.selectbox("Merchant", S["seller_id"].tolist(),
                       format_func=lambda x: f"{x[:14]}…  "
                       f"({S.set_index('seller_id').loc[x,'risk_band']})")
    row = S.set_index("seller_id").loc[sid]
    c = st.columns(5)
    c[0].metric("Risk probability", f"{row.risk_prob:.1%}")
    c[1].metric("Band", row.risk_band)
    c[2].metric("Bad rate (90d)", f"{row.bad_rate:.1%}")
    c[3].metric("Orders (90d)", int(row.n_orders))
    c[4].metric("Exposure", f"R$ {row.current_exposure:,.0f}")

    st.markdown(f"<div class='band-{row.risk_band}'><b>Recommended action:</b> "
                f"{row.recommendation}<br>Revised limit "
                f"<b>R$ {row.revised_limit:,.0f}</b> · max tenure "
                f"<b>{int(row.max_tenure_months)} months</b> · holdback "
                f"<b>{row.holdback_pct:.0%}</b></div>", unsafe_allow_html=True)

    l, r = st.columns(2)
    with l:
        st.subheader("Why this score (SHAP)")
        try:
            import shap
            ex = shap.TreeExplainer(MODEL)
            sv = ex.shap_values(row[FEATS].astype(float).values.reshape(1, -1))
            sv = sv[1][0] if isinstance(sv, list) else sv[0]
            d = pd.DataFrame({"f": FEATS, "v": sv}).assign(a=lambda x: x.v.abs()) \
                  .nlargest(12, "a").sort_values("v")
            fig = px.bar(d, x="v", y="f", orientation="h",
                         color=d.v > 0, color_discrete_map={True: "#f2385a", False: "#2ecc71"})
            fig.update_layout(showlegend=False, height=420,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e6e9ef", xaxis_title="→ pushes risk up",
                              yaxis_title="", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            d = IMP.head(12).reset_index()
            d.columns = ["f", "v"]
            st.bar_chart(d.set_index("f"))

    with r:
        st.subheader("Behaviour trajectory")
        h = P[P.seller_id == sid].sort_values("snapshot")
        if len(h) > 1:
            fig = go.Figure()
            fig.add_scatter(x=h.snapshot, y=h.bad_rate, name="Bad rate",
                            line=dict(color="#f2385a", width=3))
            fig.add_scatter(x=h.snapshot, y=h.late_rate, name="Late rate",
                            line=dict(color="#ff8c42", dash="dot"))
            fig.add_scatter(x=h.snapshot, y=h.cancel_rate, name="Cancel rate",
                            line=dict(color="#ffd166", dash="dash"))
            fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e9ef",
                              yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Complaint mix")
        cx = [c for c in S.columns if c.startswith("cx_")]
        cd = pd.DataFrame({"type": [c[3:] for c in cx],
                           "rate": [row[c] for c in cx]})
        st.dataframe(cd.sort_values("rate", ascending=False),
                     hide_index=True, use_container_width=True)


# ------------------------------------------------------------------ ALERTS
elif page == "Early-Warning Alerts":
    st.subheader("Live alert feed")
    st.caption(f"Median lead time **{M['median_lead_days']:.0f} days** · "
               f"{M['pct_caught_early']:.0%} of deteriorating merchants caught before the fact")
    band = st.multiselect("Bands", ["CRITICAL", "HIGH", "WATCH"],
                          default=["CRITICAL", "HIGH"])
    v = S[S.risk_band.isin(band)][
        ["seller_id", "risk_prob", "risk_band", "bad_rate", "late_rate",
         "cancel_rate", "n_orders", "current_exposure", "revised_limit",
         "max_tenure_months", "holdback_pct", "recommendation"]]
    st.dataframe(v.style.format({
        "risk_prob": "{:.1%}", "bad_rate": "{:.1%}", "late_rate": "{:.1%}",
        "cancel_rate": "{:.1%}", "current_exposure": "R$ {:,.0f}",
        "revised_limit": "R$ {:,.0f}", "holdback_pct": "{:.0%}"}),
        use_container_width=True, height=520, hide_index=True)
    st.download_button("Export action file (CSV)", v.to_csv(index=False),
                       "sentinel_actions.csv")


# ------------------------------------------------------------------- GRAPH
elif page == "Collusion Graph":
    st.subheader("Shared-customer network — fake-transaction ring detection")
    st.caption("Edges connect merchants sharing an unusual number of unique customers.")
    top = E.head(60)
    nodes = pd.unique(top[["seller_id_x", "seller_id_y"]].values.ravel())
    ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}
    risk = S.set_index("seller_id")["risk_prob"].to_dict()
    ex, ey = [], []
    for _, r in top.iterrows():
        x0, y0 = pos[r.seller_id_x]; x1, y1 = pos[r.seller_id_y]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_scatter(x=ex, y=ey, mode="lines",
                    line=dict(color="#31405e", width=1), hoverinfo="none")
    fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                    mode="markers", marker=dict(
                        size=14, color=[risk.get(n, 0) for n in nodes],
                        colorscale="Inferno", showscale=True,
                        colorbar=dict(title="Risk")),
                    text=[f"{n[:10]}… risk {risk.get(n,0):.0%}" for n in nodes],
                    hoverinfo="text")
    fig.update_layout(height=560, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(E.head(25), hide_index=True, use_container_width=True)


# --------------------------------------------------------- PRODUCT / CATEGORY
elif page == "Product & Category Risk":
    st.subheader("Category risk league table")
    c = (OM.groupby("category").agg(
        orders=("order_id", "count"), bad_rate=("is_bad", "mean"),
        cancel=("is_cancelled", "mean"), late=("is_late", "mean"),
        badrev=("is_badreview", "mean"), gmv=("price", "sum"),
        financed=("financed_value", "sum")).reset_index())
    c = c[c.orders >= 200].sort_values("bad_rate", ascending=False)
    fig = px.scatter(c, x="orders", y="bad_rate", size="financed", color="bad_rate",
                     hover_name="category", color_continuous_scale="Inferno",
                     log_x=True, size_max=45)
    fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e9ef",
                      yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(c.style.format({
        "bad_rate": "{:.1%}", "cancel": "{:.2%}", "late": "{:.1%}",
        "badrev": "{:.1%}", "gmv": "R$ {:,.0f}", "financed": "R$ {:,.0f}"}),
        use_container_width=True, height=380, hide_index=True)

    st.subheader("Regional anomaly — bad rate by customer state")
    g = OM.groupby("customer_state").agg(
        orders=("order_id", "count"), bad=("is_bad", "mean")).reset_index()
    g = g[g.orders >= 200].sort_values("bad", ascending=False)
    fig2 = px.bar(g, x="customer_state", y="bad", color="bad",
                  color_continuous_scale="Inferno")
    fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e9ef",
                       yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------------------- VALIDATION
else:
    st.subheader("Model validation — time-based out-of-sample")
    st.caption(f"Trained on data up to {M['split_date'][:10]}, tested strictly after. "
               "No future information used in any feature.")
    c = st.columns(4)
    c[0].metric("ROC-AUC", f"{M['roc_auc']:.3f}")
    c[1].metric("Top-decile lift", f"{M['top_decile_lift']:.2f}x")
    c[2].metric("Median lead time", f"{M['median_lead_days']:.0f} d")
    c[3].metric("Caught early", f"{M['pct_caught_early']:.0%}")

    st.markdown("#### Operating points")
    ops = pd.DataFrame(M["operating_points"]).T.reset_index()
    ops.columns = ["threshold", "flagged", "precision", "recall", "lift"]
    st.dataframe(ops.style.format({"precision": "{:.1%}", "recall": "{:.1%}",
                                   "lift": "{:.2f}x"}),
                 hide_index=True, use_container_width=True)

    st.markdown("#### Loss-avoided simulation")
    c = st.columns(3)
    c[0].metric("Financed value in bad orders (test)",
                f"R$ {M['total_bad_financed_brl']:,.0f}")
    c[1].metric("Captured by flagged merchants", f"R$ {M['loss_avoided_brl']:,.0f}")
    c[2].metric("Share of loss captured", f"{M['pct_loss_captured']:.1%}")

    st.markdown("#### Top drivers")
    d = IMP.head(15).reset_index()
    d.columns = ["feature", "importance"]
    fig = px.bar(d.sort_values("importance"), x="importance", y="feature",
                 orientation="h", color="importance", color_continuous_scale="Inferno")
    fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e9ef",
                      showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Honest limitations (say this to the jury)"):
        st.markdown("""
- **Weak supervision.** Olist has no ground-truth "return fraud" flag. We construct a
  composite bad-order label (cancelled/unavailable, delivered >5 days past promise,
  or review ≤2 stars). We do not claim it equals fraud.
- **AUC ≈ 0.66 is the honest ceiling here.** Much of merchant badness is logistics
  randomness. We therefore optimise and report *ranking* quality (2.2x top-decile lift)
  and *lead time*, which is what a risk committee actually acts on.
- **Brazilian proxy data.** Chosen because it is the only public source carrying
  merchant, fulfilment, review and EMI-installment fields in one relational schema.
  Field mapping to a TVS LMS is in the README.
- **No leakage.** All features come from strictly past windows; the split is temporal.
""")
