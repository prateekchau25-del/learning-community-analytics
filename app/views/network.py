import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from data import COMMUNITIES, out, topic_labels

ROLE_COLOR = {"Core expert": ui.SERIES[0], "Bridge": ui.SERIES[1], "Helper": ui.SERIES[2]}


@st.cache_data(show_spinner="Drawing the network ...")
def network_html(site: str, n_top: int) -> str:
    from pyvis.network import Network
    m, edges = out(site, "user_metrics.csv"), out(site, "help_edges.csv")
    top = m.nlargest(n_top, "out_strength")
    ids = set(top["user_id"])
    sub = edges[edges.source_user.isin(ids) & edges.target_user.isin(ids)]
    # Drop people with no link to the others: they float far away and shrink the fitted view.
    linked = set(sub.source_user) | set(sub.target_user)
    top = top[top["user_id"].isin(linked)]
    net = Network(height="600px", width="100%", directed=True, bgcolor=ui.SURFACE, font_color=ui.INK,
                  cdn_resources="in_line")
    # Straight edges and a bounded stabilisation keep large networks fast; the view is then fitted.
    net.set_options("""{
      "physics": {"barnesHut": {"gravitationalConstant": -14000, "springLength": 260, "avoidOverlap": 0.4},
                  "stabilization": {"enabled": true, "iterations": 400, "fit": true}},
      "edges": {"smooth": false, "arrows": {"to": {"enabled": true, "scaleFactor": 0.35}}, "width": 0.6},
      "interaction": {"hover": true}
    }""")
    named = set(top.head(15)["user_id"])  # label only the top helpers so the picture stays readable
    for r in top.itertuples():
        net.add_node(int(r.user_id), label=str(r.user_name) if r.user_id in named else " ",
                     font={"color": ui.INK, "size": 40, "strokeWidth": 6, "strokeColor": ui.SURFACE},  # large: readable when zoomed out
                     size=6 + 1.3 * np.sqrt(r.out_strength),
                     color=ROLE_COLOR.get(r.role, ui.OTHER),
                     title=f"{r.user_name}\nrole: {r.role}\nhelped: {r.out_strength}\nanswers: {r.n_answers}")
    for r in sub.itertuples():
        net.add_edge(int(r.source_user), int(r.target_user), value=int(r.weight), color="#3B3552")
    # generate_html avoids writing a file (pyvis would use the Windows code page and fail on names).
    html = net.generate_html(notebook=False)
    # Dark page behind the canvas (pyvis leaves the page white).
    html = html.replace("border: 1px solid lightgray", "border: 0")  # pyvis draws a light frame by default
    # After the layout settles, freeze it and fit it to the frame.
    html = html.replace("</body>", "<script>network.once('stabilizationIterationsDone', function () {"
                                   "network.setOptions({physics: false}); network.fit(); });</script></body>", 1)
    return html.replace("<body>", f"<body style='margin:0;background:{ui.SURFACE}'>", 1)


def render(site: str):
    ui.page_header("Knowledge network", f"Who helps whom in the {COMMUNITIES[site]} community")
    stats, m, over, val = (out(site, "network_stats.json"), out(site, "user_metrics.csv"),
                           out(site, "network_over_time.csv"), out(site, "network_validation.csv"))
    ui.kpis([
        ("People in network", f"{stats['n_users']:,}", f"{stats['n_edges']:,} helper → learner links"),
        ("Help from top 1%", ui.fmt_pct(stats["help_share_top1pct"]), f"top 10%: {ui.fmt_pct(stats['help_share_top10pct'])}"),
        ("People who help", ui.fmt_pct(stats["pct_users_who_help"]), f"Gini {stats['gini_help_given']:.2f}"),
        ("Sub-communities", stats["n_communities_10plus"], f"modularity {stats['modularity']:.2f}"),
        ("Reciprocity", f"{stats['reciprocity']:.3f}", "help flows one way"),
    ])

    st.markdown("## Interactive network")
    n_top = st.slider("People shown (most active helpers)", 30, 200, 90, step=10)
    legend = " · ".join(f"<span style='color:{c}'>●</span> {r}" for r, c in ROLE_COLOR.items())
    st.markdown(f"<span style='color:{ui.MUTED};font-size:.9rem'>{legend} · <span style='color:{ui.OTHER}'>●</span> "
                "learner / other · size = help given · drag, zoom and hover</span>", unsafe_allow_html=True)
    st.components.v1.html(network_html(site, n_top), height=620)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("## Expert leaderboard")
        names = topic_labels(site)
        board = m.head(15)[["user_name", "role", "out_strength", "n_answers", "n_accepted", "expertise_rank", "main_help_topic"]].copy()
        board["main_help_topic"] = board["main_help_topic"].map(names)
        st.dataframe(board, hide_index=True, width="stretch", column_config={
            "user_name": "Expert", "role": "Role", "out_strength": st.column_config.NumberColumn("Help given"),
            "n_answers": "Answers", "n_accepted": "Accepted",
            "expertise_rank": st.column_config.ProgressColumn("ExpertiseRank", format="%.4f",
                                                              min_value=0, max_value=float(board.expertise_rank.max())),
            "main_help_topic": "Main topic"})
    with right:
        st.markdown("## Roles")
        roles = m["role"].value_counts()
        fig = go.Figure(go.Bar(x=roles.values, y=roles.index, orientation="h", cliponaxis=False,
                               marker_color=[ROLE_COLOR.get(r, ui.OTHER) for r in roles.index],
                               text=roles.values, textposition="outside",
                               hovertemplate="%{y}: %{x} people<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, roles.max() * 1.18])
        ui.chart(fig, 330)

    st.markdown("## Over time")
    metric = st.segmented_control("Measure", ["active_users", "n_helpers", "help_share_top10pct", "modularity"],
                                  default="active_users", format_func=lambda s: s.replace("_", " ").replace("pct", "%"))
    metric = metric or "active_users"
    fig = go.Figure(go.Scatter(x=over["year"], y=over[metric], mode="lines+markers",
                               line=dict(color=ui.COMMUNITY_COLOR[site], width=2), marker=dict(size=8)))
    fig.update_xaxes(dtick=1, tickformat="d")
    ui.chart(fig, 320)

    with st.expander("Validation: do network scores agree with Stack Exchange reputation?"):
        st.dataframe(val, hide_index=True, width="stretch")
        ui.callout("Among people who answered at least once. top20_overlap = share of the metric's top 20 who are "
                   "also in the top 20 by accepted answers or reputation.")
