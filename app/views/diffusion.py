import json

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from data import COMMUNITIES, out, topic_labels
from config import site_dirs


def render(site: str):
    ui.page_header("Knowledge diffusion", f"How knowledge spreads through the {COMMUNITIES[site]} community")
    summ, exp, im, curves, valid = (out(site, "diffusion_summary.json"), out(site, "diffusion_exposure_test.csv"),
                                    out(site, "influence_maximization.csv"), out(site, "topic_adoption_curves.csv"),
                                    out(site, "diffusion_validation.csv"))
    l2h = summ["learner_to_helper"]
    z, p = summ.get("exposure_stouffer_z"), summ.get("exposure_stouffer_p")
    ui.kpis([
        ("Exposure effect", f"z = {z:.2f}" if z is not None else "–", f"p = {p:.3f} (all topics combined)" if p is not None else ""),
        ("Topics with effect", f"{summ['exposure_topics_significant']} / {summ['exposure_topics_tested']}", "p < 0.05"),
        ("Learners → helpers", ui.fmt_pct(l2h["pct_became_helpers"], 1),
         f"χ² p = {l2h['p_value']:.3f}" if l2h["p_value"] >= 0.001 else "χ² p < 0.001"),
        ("Simulation vs reality", f"ρ = {summ['validation']['simulated_ic_spread']:.2f}", "Spearman correlation"),
    ])

    st.markdown("## Does contact spread topics?")
    st.caption("For each topic: the share of people who, before first joining it, had talked with someone who had "
               "already joined, compared with 200 random re-orderings of the joining times.")
    e = exp.sort_values("observed_exposed_share")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=100 * e["null_mean"], y=e["label"], mode="markers", name="Expected by chance",
                             marker=dict(color=ui.OTHER, size=11)))
    fig.add_trace(go.Scatter(x=100 * e["observed_exposed_share"], y=e["label"], mode="markers", name="Observed",
                             marker=dict(color=ui.COMMUNITY_COLOR[site], size=11), customdata=e[["p_value"]],
                             hovertemplate="%{y}<br>observed %{x:.1f}%<br>p = %{customdata[0]:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="% of adopters exposed to an earlier adopter", legend=dict(orientation="h", y=1.08))
    ui.chart(fig, max(360, 30 * len(e)))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("## Topic adoption")
        names = topic_labels(site)
        options = sorted(curves["topic"].unique())
        pick = st.multiselect("Topics", options, default=options[:3], format_func=lambda t: names.get(t, t), max_selections=6)
        fig = go.Figure()
        for i, t in enumerate(pick):
            d = curves[curves.topic == t]
            fig.add_trace(go.Scatter(x=d["period"], y=d["cumulative_adopters"], name=names.get(t, t), mode="lines",
                                     line=dict(color=ui.SERIES[i], width=2)))
        fig.update_layout(yaxis_title="People who joined the topic", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(nticks=8, tickangle=0)
        ui.chart(fig, 400)
    with right:
        st.markdown("## Who should seed knowledge?")
        model = st.segmented_control("Diffusion model", ["IC", "LT"], default="IC",
                                     format_func=lambda s: {"IC": "Independent Cascade", "LT": "Linear Threshold"}[s]) or "IC"
        d = im[im.model == model]
        order = ["CELF greedy", "Top ExpertiseRank", "Top out-degree", "Top betweenness", "Top PageRank", "Random"]
        fig = go.Figure()
        for i, s in enumerate(order):
            x = d[d.strategy == s].sort_values("k")
            fig.add_trace(go.Scatter(x=x["k"], y=x["spread_pct_of_users"], name=s, mode="lines+markers",
                                     line=dict(color=ui.OTHER if s == "Random" else ui.SERIES[i], width=2,
                                               dash="dash" if s == "Random" else "solid")))
        fig.update_layout(xaxis_title="Seed users", yaxis_title="% of people reached", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.25))
        ui.chart(fig, 400)
        seeds = json.loads((site_dirs(site)["outputs"] / "seed_sets.json").read_text())[model]["CELF greedy"]
        m = out(site, "user_metrics.csv").set_index("user_id")
        st.markdown("**Best seed users (CELF):** " + ", ".join(str(m.at[s, "user_name"]) for s in seeds if s in m.index))

    st.markdown("## Does the simulation match reality?")
    fig = px.scatter(valid, x="simulated_ic_spread", y="observed_reach", hover_name="user_name", log_x=True, log_y=True,
                     color_discrete_sequence=[ui.COMMUNITY_COLOR[site]])
    fig.update_layout(xaxis_title="Simulated spread from the user", yaxis_title="Observed reach within one year")
    fig.update_xaxes(dtick=1)  # log axes: label only 1, 10, 100, 1000
    fig.update_yaxes(dtick=1)
    ui.chart(fig, 420)
    ui.callout(f"Of {l2h['n_learners']:,} learners, {l2h['n_became_helpers']:,} later answered someone else's question "
               f"(median {l2h['median_days_to_become_helper'] or 0:.0f} days). Whether their first question was answered made "
               f"{'a significant' if l2h['p_value'] < 0.05 else 'no significant'} difference "
               f"({ui.fmt_pct(l2h['pct_helper_if_first_question_answered'], 1)} if answered vs "
               f"{ui.fmt_pct(l2h['pct_helper_if_first_question_unanswered'], 1)} if not, p = {l2h['p_value']:.3f}).")
