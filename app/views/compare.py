import plotly.graph_objects as go
import streamlit as st

import ui
from data import COMMUNITIES, comparison, processed

PCT_ROWS = ["Answered (%)", "Accepted answer (%)", "Questions with code (%)", "Questions with maths (%)",
            "Users who help (%)", "Help from top 1% (%)", "Help from top 10% (%)", "Learners who became helpers (%)",
            "IC reach, 10 seeds (%)", "LT reach, 10 seeds (%)", "Expert finder Hit@10 (%)"]


def fmt(measure, v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    if "(%)" in measure:
        return f"{x:.1f}%"
    if measure in ("Questions", "Answers", "Comments", "Users", "Network users", "Help links",
                   "BERTopic topics", "Communities (10+ users)"):
        return f"{x:,.0f}"
    if "hours" in measure:
        return f"{x:.1f} h"
    return f"{x:.3f}"


def render(sites):
    if len(sites) < 2:
        st.info("The comparison needs results for both communities. Run: python src/run_pipeline.py")
        return
    a, b = sites
    ui.page_header("Data Science vs Artificial Intelligence",
                   "Two learning communities side by side: what they discuss, how they help, how knowledge spreads")
    ov, tests, mix = comparison("overview.csv"), comparison("tests.csv"), comparison("category_mix.csv")
    shared, shared_sum = comparison("shared_users.csv"), comparison("shared_users_summary.json")
    ov = ov.set_index("measure")

    cols = st.columns(2, gap="large")
    for col, s in zip(cols, sites):
        with col:
            st.markdown(f"<h3 style='color:{ui.COMMUNITY_COLOR[s]};margin-bottom:0'>● {COMMUNITIES[s]}</h3>",
                        unsafe_allow_html=True)
            ui.kpis([("Questions", fmt("Questions", ov.loc["Questions", s]), f"{fmt('Users', ov.loc['Users', s])} people"),
                     ("Answered", fmt("Answered (%)", ov.loc["Answered (%)", s]),
                      f"median wait {fmt('Median hours to first answer', ov.loc['Median hours to first answer', s])}"),
                     ("Top 1% give", fmt("Help from top 1% (%)", ov.loc["Help from top 1% (%)", s]), "of all help")])

    st.markdown("## Key measures")
    rows = [r for r in PCT_ROWS if r in ov.index]
    fig = go.Figure()
    for s in sites:
        fig.add_trace(go.Bar(name=COMMUNITIES[s], y=[r.replace(" (%)", "") for r in rows],
                             x=[float(ov.loc[r, s]) for r in rows], orientation="h",
                             marker_color=ui.COMMUNITY_COLOR[s], hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(barmode="group", xaxis_title="%", legend=dict(orientation="h", y=1.06))
    fig.update_yaxes(autorange="reversed")
    ui.chart(fig, 520)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("## What they talk about")
        m = mix.set_index("category")
        m = m.loc[m.sum(axis=1).sort_values(ascending=True).index]
        fig = go.Figure()
        for s in sites:
            fig.add_trace(go.Bar(name=COMMUNITIES[s], y=m.index, x=m[s], orientation="h",
                                 marker_color=ui.COMMUNITY_COLOR[s], hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_layout(barmode="group", xaxis_title="% of questions", legend=dict(orientation="h", y=1.08))
        ui.chart(fig, 480)
    with right:
        st.markdown("## Are the differences real?")
        t = tests.drop(columns=["statistic", "test"]).copy()
        t["significant"] = t["p_value"].map(lambda p: "✅ yes" if p < 0.05 else "no")
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "measure": "Measure", "p_value": st.column_config.NumberColumn("p", format="%.4f"),
            a: st.column_config.NumberColumn(COMMUNITIES[a], format="%.3f"),
            b: st.column_config.NumberColumn(COMMUNITIES[b], format="%.3f")})
        ui.callout("Answered rate: chi-square test. Waiting time, confusion and length: Mann-Whitney U test on medians.")

    st.markdown("## Knowledge bridges: people active in both communities")
    ui.kpis([("Shared people", f"{shared_sum['shared_users']:,}", "same Stack Exchange account"),
             ("Answer in both", f"{shared_sum['helpers_in_both']:,}", "carry knowledge across"),
             (f"Help in {COMMUNITIES[a]}", ui.fmt_pct(shared_sum[f"share_of_help_{a}"]), "given by shared people"),
             (f"Help in {COMMUNITIES[b]}", ui.fmt_pct(shared_sum[f"share_of_help_{b}"]), "given by shared people")])
    show = shared.head(15).rename(columns={
        f"role_{a}": f"Role ({COMMUNITIES[a]})", f"out_strength_{a}": f"Help ({COMMUNITIES[a]})",
        f"role_{b}": f"Role ({COMMUNITIES[b]})", f"out_strength_{b}": f"Help ({COMMUNITIES[b]})", "user_name": "Person"})
    st.dataframe(show[["Person", f"Role ({COMMUNITIES[a]})", f"Help ({COMMUNITIES[a]})",
                       f"Role ({COMMUNITIES[b]})", f"Help ({COMMUNITIES[b]})"]], hide_index=True, width="stretch")

    with st.expander("All measures"):
        table = ov.copy()
        for s in sites:
            table[s] = [fmt(m, v) for m, v in zip(table.index, table[s])]
        st.dataframe(table.rename(columns=COMMUNITIES), width="stretch")
