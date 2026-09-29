import plotly.graph_objects as go
import streamlit as st

import ui
from data import COMMUNITIES, out, processed, raw


def render(site: str):
    name = COMMUNITIES[site]
    r = raw(site)
    q = processed(site, "questions_clean.parquet")
    comp, stats, dif = out(site, "topic_model_comparison.csv"), out(site, "network_stats.json"), out(site, "diffusion_summary.json")
    topics, pred, rec = out(site, "topics.csv"), out(site, "prediction_metrics.csv"), out(site, "recommender_eval.csv")

    ui.hero(
        "Learning community analytics",
        f"How the {name} community learns, helps and shares knowledge",
        "NLP topic modeling, knowledge networks and diffusion analysis on the public Stack Exchange "
        "Q&A data, with tools that predict whether a question will be answered and who can answer it.",
        chips=[f"{name} Stack Exchange", f"{q.created_at.min():%b %Y} – {q.created_at.max():%b %Y}",
               "BERTopic · NetworkX · IC/LT diffusion · machine learning"],
    )
    ui.kpis([
        ("Questions", f"{len(r['questions']):,}", "asked by learners"),
        ("Answers", f"{len(r['answers']):,}", f"{len(r['comments']):,} comments"),
        ("People", f"{len(r['users']):,}", "askers and helpers"),
        ("Answered", ui.fmt_pct(q.is_answered.mean()), f"{ui.fmt_pct(q.has_accepted.mean())} with an accepted answer"),
        ("Median wait", f"{q.hours_to_first_answer.median():.0f} h", "until the first answer"),
    ])

    per = q.groupby(q["created_at"].dt.to_period("M")).size()
    fig = go.Figure(go.Scatter(x=per.index.to_timestamp(), y=per.values, mode="lines", fill="tozeroy",
                               line=dict(color=ui.COMMUNITY_COLOR[site], width=2),
                               hovertemplate="%{x|%b %Y}: %{y} questions<extra></extra>"))
    ui.chart(fig, 300, "Questions per month")

    st.markdown("## Key findings")
    items = []
    if comp is not None:
        b, l = comp.iloc[1], comp.iloc[0]
        items.append(("🧠", "Discussion topics", f"{int(b.n_topics)} topics",
                      f"BERTopic agrees with the community's own tag categories better than LDA "
                      f"(NMI {b.nmi_vs_tags:.2f} vs {l.nmi_vs_tags:.2f})."))
    if topics is not None:
        hard = topics.sort_values("difficulty_index", ascending=False).iloc[0]
        items.append(("🔥", "Hardest topic", hard.label,
                      f"{hard.pct_unanswered:.0f}% of its questions stay unanswered; median wait "
                      f"{hard.median_hours_to_answer:.0f} hours."))
    if stats is not None:
        items.append(("🕸️", "Help is concentrated", ui.fmt_pct(stats["help_share_top1pct"]),
                      f"of all help comes from the top 1% of users; {stats['n_communities_10plus']} sub-communities "
                      f"(modularity {stats['modularity']:.2f})."))
    if dif is not None:
        z = dif.get("exposure_stouffer_z")
        items.append(("🌊", "Social exposure", f"z = {z:.2f}" if z is not None else "–",
                      f"{dif['exposure_topics_significant']} of {dif['exposure_topics_tested']} topics show more "
                      "contact with earlier adopters than chance (time-shuffle test)."))
        rho = dif["validation"]["simulated_ic_spread"]
        items.append(("🎯", "Simulation check", f"ρ = {rho:.2f}",
                      "Simulated influence (Independent Cascade) matches how far each helper's knowledge travelled within a year."
                      if rho >= 0.3 else
                      "Simulated influence does not predict real reach here: within a year most active helpers reach a similar "
                      "share of the community."))
    if pred is not None and rec is not None:
        best = pred.iloc[1:].sort_values("roc_auc").iloc[-1]
        hyb = rec.set_index("method").loc["Hybrid (embedding)"]
        items.append(("🤖", "Smart tools", f"Hit@10 {ui.fmt_pct(hyb['hit@10'])}",
                      f"The expert finder finds a real answerer in its top 10; answer prediction reaches "
                      f"ROC-AUC {best.roc_auc:.2f} ({best.model})."))
    ui.cards(items)

    st.markdown("## How it works")
    ui.steps([
        ("Collect", "Stack Exchange API → PostgreSQL"),
        ("Clean", "HTML, code, lemmas, confusion score"),
        ("Topics", "LDA vs BERTopic"),
        ("Network", "Who helps whom, experts, communities"),
        ("Diffusion", "Shuffle test, IC/LT, CELF"),
        ("Predict", "Will it be answered? Who can answer?"),
        ("Compare", "Data Science vs AI"),
    ])
