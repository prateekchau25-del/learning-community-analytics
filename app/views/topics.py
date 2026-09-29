import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from data import COMMUNITIES, out, processed, topic_labels


def render(site: str):
    ui.page_header("Discussion topics", f"What the {COMMUNITIES[site]} community talks about, and where learners struggle")
    topics, comp, tmap, trends = (out(site, "topics.csv"), out(site, "topic_model_comparison.csv"),
                                  out(site, "topic_map.csv"), out(site, "topic_trends.csv"))
    names = topic_labels(site)
    lda, bt = comp.iloc[0], comp.iloc[1]
    ui.kpis([
        ("BERTopic topics", int(bt.n_topics), f"LDA: {int(lda.n_topics)}"),
        ("NPMI coherence", f"{bt.npmi:.2f}", f"LDA: {lda.npmi:.2f}"),
        ("Topic diversity", f"{bt.diversity:.2f}", f"LDA: {lda.diversity:.2f}"),
        ("Match with tags (NMI)", f"{bt.nmi_vs_tags:.2f}", f"LDA: {lda.nmi_vs_tags:.2f}"),
    ])

    st.markdown("## Topic explorer")
    left, right = st.columns([3, 2], gap="large")
    order = topics.sort_values("n_questions", ascending=False)
    with right:
        choice = st.selectbox("Choose a topic", order["topic"], format_func=lambda t: f"{names[t]}")
        row = topics.set_index("topic").loc[choice]
        cat = row.category if isinstance(row.category, str) and row.category_purity >= 0.5 else "Mixed"
        ui.kpis([("Questions", int(row.n_questions), cat),
                 ("Unanswered", f"{row.pct_unanswered:.0f}%", f"median wait {row.median_hours_to_answer:.0f} h")])
        st.markdown("**Top words**")
        ui.words(str(row.top_words).split(", ")[:10])
        q = processed(site, "questions_clean.parquet")
        qt = processed(site, "question_topics.parquet")
        sample = q.merge(qt[qt.topic == choice][["post_id"]], on="post_id").nlargest(6, "view_count")
        st.markdown("**Most viewed questions**")
        for r in sample.itertuples():
            st.markdown(f"- [{r.title}](https://{site}.stackexchange.com/q/{r.post_id}) · {r.view_count:,} views")
    with left:
        tm = tmap.assign(group=np.where(tmap["topic"] == choice, names[choice], "Other topics"))
        fig = px.scatter(tm, x="x", y="y", color="group", hover_name="title",
                         color_discrete_map={names[choice]: ui.COMMUNITY_COLOR[site], "Other topics": "#3A3450"},
                         category_orders={"group": ["Other topics", names[choice]]})
        fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(legend=dict(orientation="h", y=-0.05))
        ui.chart(fig, 520, "Every question placed by meaning (UMAP of sentence embeddings)")

    st.markdown("## Where learners struggle")
    d = topics.sort_values("difficulty_index")
    fig = go.Figure(go.Bar(
        x=d["difficulty_index"], y=d["label"], orientation="h",
        marker_color=[ui.SERIES[7] if v > 0 else ui.SERIES[0] for v in d["difficulty_index"]],
        customdata=d[["pct_unanswered", "median_hours_to_answer", "n_questions"]],
        hovertemplate="%{y}<br>difficulty %{x:.2f}<br>unanswered %{customdata[0]:.0f}%<br>"
                      "median wait %{customdata[1]:.0f} h<br>%{customdata[2]} questions<extra></extra>"))
    ui.chart(fig, max(380, 26 * len(d)), "Difficulty index: unanswered rate + waiting time + learner confusion (higher = harder)")

    st.markdown("## How topics changed")
    trends["year"] = trends["period"].str[:4]
    y = trends.groupby(["topic", "year"])["n_questions"].sum().reset_index()
    y = y[y.groupby("year")["n_questions"].transform("sum") >= 100]
    y["share"] = 100 * y["n_questions"] / y.groupby("year")["n_questions"].transform("sum")
    y["label"] = y["topic"].map(names)
    heat = y.pivot(index="label", columns="year", values="share").fillna(0)
    if heat.shape[1] >= 2:
        fig = px.imshow(heat, color_continuous_scale=ui.BLUES, aspect="auto",
                        labels=dict(x="", y="", color="% of year"))
        fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1f}% of questions<extra></extra>")
        ui.chart(fig, max(380, 26 * len(heat)), "Share of each year's questions by topic (years with 100+ questions)")
    else:
        st.info("Not enough years with 100+ questions to show a trend.")
