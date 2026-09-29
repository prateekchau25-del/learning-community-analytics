"""
Step 3: build the knowledge network (who helps whom) and measure its structure.

Graph: directed, edge helper -> learner, weight = number of interactions. Only
knowledge-giving interactions are used (answers, and comments by someone other than the
thread's asker), see data_loader.help_interactions.

Outputs:
    outputs/user_metrics.csv        centrality, community and role of every user
    outputs/help_edges.csv          aggregated weighted edge list
    outputs/network_stats.json      global statistics of the network
    outputs/network_over_time.csv   yearly snapshots
    outputs/community_topics.csv    what each community talks about
    outputs/topic_network_edges.csv topics linked by shared helpers
    outputs/network_validation.csv  do centrality scores agree with reputation?
"""

import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED
from data_loader import help_interactions, load_raw


def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def aggregate_edges(interactions: pd.DataFrame) -> pd.DataFrame:
    return (interactions.groupby(["source_user", "target_user"])
            .agg(weight=("id", "size"),
                 n_answers=("type", lambda s: int((s == "answer").sum())),
                 first_at=("created_at", "min"), last_at=("created_at", "max"))
            .reset_index())


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for e in edges.itertuples():
        g.add_edge(int(e.source_user), int(e.target_user), weight=int(e.weight))
    return g


def user_activity(raw) -> pd.DataFrame:
    q, a, c = raw["questions"], raw["answers"], raw["comments"]
    act = pd.DataFrame({
        "n_questions": q.groupby("user_id").size(),
        "n_answers": a.groupby("user_id").size(),
        "n_accepted": a[a["is_accepted"].astype(bool)].groupby("user_id").size(),
        "n_comments": c.groupby("user_id").size(),
        "answer_score": a.groupby("user_id")["score"].sum(),
    }).fillna(0).astype(int)
    first_seen = pd.concat([q[["user_id", "created_at"]], a[["user_id", "created_at"]],
                            c[["user_id", "created_at"]]]).groupby("user_id")["created_at"]
    act["first_seen"] = first_seen.min()
    act["last_seen"] = first_seen.max()
    act.index = act.index.astype("int64")
    return act


def participation_coefficient(ug: nx.Graph, community: dict) -> dict:
    """Share of a node's links that go to other communities (0 = inward, ~1 = bridge)."""
    out = {}
    for node in ug:
        nbrs = list(ug[node])
        if not nbrs:
            out[node] = 0.0
            continue
        counts = pd.Series([community[n] for n in nbrs]).value_counts(normalize=True)
        out[node] = float(1 - (counts ** 2).sum())
    return out


def assign_roles(m: pd.DataFrame) -> pd.Series:
    """Rule-based roles, from most to least specific."""
    top_help = m["out_strength"].quantile(0.98)
    top_betw = m["betweenness"].quantile(0.95)
    role = pd.Series("Peripheral learner", index=m.index)
    role[(m["out_strength"] >= 1) & (m["in_strength"] >= 1)] = "Active participant"
    role[(m["out_strength"] >= 3) & (m["out_strength"] > m["in_strength"])] = "Helper"
    role[(m["betweenness"] >= top_betw) & (m["participation_coef"] >= 0.3) & (m["out_strength"] >= 3)] = "Bridge"
    role[(m["out_strength"] >= max(top_help, 10)) & (m["n_answers"] >= 10)] = "Core expert"
    role[(m["out_strength"] == 0) & (m["in_strength"] == 0)] = "Isolated"
    return role


def global_stats(g: nx.DiGraph, ug: nx.Graph, communities, m: pd.DataFrame) -> dict:
    giant = max(nx.weakly_connected_components(g), key=len)
    help_given = m["out_strength"].sort_values(ascending=False)
    total = help_given.sum()
    n = len(help_given)
    out_deg = np.array([d for _, d in g.out_degree() if d > 0])
    # Slope of the log-log complementary CDF: a straight line means a heavy-tailed (scale-free-like) network.
    vals = np.sort(np.unique(out_deg))
    ccdf = np.array([(out_deg >= v).mean() for v in vals])
    slope = float(np.polyfit(np.log(vals), np.log(ccdf), 1)[0]) if len(vals) > 2 else None
    helpers = (m["out_strength"] > 0).sum()
    return {
        "n_users": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_interactions": int(sum(d["weight"] for _, _, d in g.edges(data=True))),
        "density": nx.density(g),
        "reciprocity": nx.reciprocity(g),
        "avg_clustering": nx.average_clustering(ug),
        "transitivity": nx.transitivity(ug),
        "degree_assortativity": nx.degree_assortativity_coefficient(ug),
        "giant_component_share": len(giant) / g.number_of_nodes(),
        "n_communities": len(communities),
        "n_communities_10plus": sum(len(c) >= 10 for c in communities),
        "modularity": nx.community.modularity(ug, communities, weight="weight"),
        "n_helpers": int(helpers),
        "pct_users_who_help": helpers / n,
        "help_share_top1pct": float(help_given.iloc[: max(1, n // 100)].sum() / total),
        "help_share_top10pct": float(help_given.iloc[: max(1, n // 10)].sum() / total),
        "gini_help_given": gini(help_given),
        "out_degree_ccdf_slope": slope,
    }


def snapshots(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, part in interactions.groupby(interactions["created_at"].dt.year):
        e = aggregate_edges(part)
        g = build_graph(e)
        if g.number_of_nodes() < 5:
            continue
        ug = g.to_undirected()
        comms = nx.community.louvain_communities(ug, weight="weight", seed=RANDOM_SEED)
        strength = pd.Series(dict(g.out_degree(weight="weight"))).sort_values(ascending=False)
        rows.append({
            "year": int(year), "active_users": g.number_of_nodes(), "edges": g.number_of_edges(),
            "interactions": int(e["weight"].sum()), "density": nx.density(g),
            "reciprocity": nx.reciprocity(g), "avg_clustering": nx.average_clustering(ug),
            "giant_component_share": len(max(nx.weakly_connected_components(g), key=len)) / g.number_of_nodes(),
            "modularity": nx.community.modularity(ug, comms, weight="weight"),
            "n_helpers": int((strength > 0).sum()),
            "help_share_top10pct": float(strength.iloc[: max(1, len(strength) // 10)].sum() / strength.sum()),
        })
    return pd.DataFrame(rows)


def topic_participation(raw, qt: pd.DataFrame) -> pd.DataFrame:
    """One row per (user, topic, role): who asks about and who helps with each topic."""
    q = raw["questions"][["post_id", "user_id"]].merge(qt[["post_id", "topic"]], on="post_id")
    a = raw["answers"][["question_id", "user_id"]].merge(
        qt[["post_id", "topic"]], left_on="question_id", right_on="post_id")
    asks = q.assign(role="ask")[["user_id", "topic", "role"]]
    helps = a.assign(role="help")[["user_id", "topic", "role"]]
    return pd.concat([asks, helps]).dropna(subset=["user_id"])


def run():
    raw = load_raw()
    inter = help_interactions(raw)

    edges = aggregate_edges(inter)
    edges.to_csv(OUTPUT_DIR / "help_edges.csv", index=False)
    g = build_graph(edges)
    ug = nx.Graph()
    for u, v, d in g.edges(data=True):
        w = ug[u][v]["weight"] + d["weight"] if ug.has_edge(u, v) else d["weight"]
        ug.add_edge(u, v, weight=w)
    print(f"[network] {g.number_of_nodes()} users, {g.number_of_edges()} helper->learner links")

    # ---- centrality
    m = pd.DataFrame(index=pd.Index(list(g.nodes), name="user_id"))
    m["out_degree"] = pd.Series(dict(g.out_degree()))
    m["in_degree"] = pd.Series(dict(g.in_degree()))
    m["out_strength"] = pd.Series(dict(g.out_degree(weight="weight")))
    m["in_strength"] = pd.Series(dict(g.in_degree(weight="weight")))
    # ExpertiseRank: PageRank on learner -> helper links, so helping people who themselves
    # help others counts more (Zhang, Ackerman & Adamic, 2007).
    m["expertise_rank"] = pd.Series(nx.pagerank(g.reverse(copy=False), weight="weight"))
    m["pagerank"] = pd.Series(nx.pagerank(g, weight="weight"))
    m["betweenness"] = pd.Series(nx.betweenness_centrality(ug, k=min(1000, len(ug)), seed=RANDOM_SEED))
    m["clustering"] = pd.Series(nx.clustering(ug))
    simple = nx.Graph(ug)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    m["core_number"] = pd.Series(nx.core_number(simple))

    # ---- communities
    communities = nx.community.louvain_communities(ug, weight="weight", seed=RANDOM_SEED)
    communities = sorted(communities, key=len, reverse=True)
    community_of = {node: i for i, c in enumerate(communities) for node in c}
    m["community"] = pd.Series(community_of)
    m["participation_coef"] = pd.Series(participation_coefficient(ug, community_of))

    # ---- activity, reputation, roles
    users = raw["users"].dropna(subset=["user_id"]).assign(user_id=lambda d: d.user_id.astype("int64"))
    m = m.join(users.set_index("user_id")[["user_name", "user_reputation"]]).join(user_activity(raw))
    m[["n_questions", "n_answers", "n_accepted", "n_comments", "answer_score"]] = (
        m[["n_questions", "n_answers", "n_accepted", "n_comments", "answer_score"]].fillna(0).astype(int))
    m["role"] = assign_roles(m)

    # Main topic each user helps with (from the topic model, when available).
    qt_path = PROCESSED_DIR / "question_topics.parquet"
    if qt_path.exists():
        qt = pd.read_parquet(qt_path)
        part = topic_participation(raw, qt)
        part["user_id"] = part["user_id"].astype("int64")
        main_help = part[part.role == "help"].groupby("user_id")["topic"].agg(lambda s: s.value_counts().index[0])
        m["main_help_topic"] = main_help.astype("Int64")
        m["n_topics_helped"] = part[part.role == "help"].groupby("user_id")["topic"].nunique()
        m["n_topics_helped"] = m["n_topics_helped"].fillna(0).astype(int)

        # Community profiles: which topics each community's members ask about or help with.
        cp = part.merge(m[["community"]], left_on="user_id", right_index=True)
        prof = cp.groupby(["community", "topic"]).size().rename("n").reset_index()
        prof["share"] = prof["n"] / prof.groupby("community")["n"].transform("sum")
        prof = prof[prof["community"].isin(range(15))].sort_values(["community", "share"], ascending=[True, False])
        prof.groupby("community").head(5).to_csv(OUTPUT_DIR / "community_topics.csv", index=False)

        # Topic network: two topics are linked when the same people help with both.
        helpers = part[part.role == "help"].drop_duplicates(["user_id", "topic"])
        pairs = helpers.merge(helpers, on="user_id")
        pairs = pairs[pairs.topic_x < pairs.topic_y]
        tn = pairs.groupby(["topic_x", "topic_y"]).size().rename("shared_helpers").reset_index()
        tn.rename(columns={"topic_x": "topic_a", "topic_y": "topic_b"}).to_csv(
            OUTPUT_DIR / "topic_network_edges.csv", index=False)

    m = m.sort_values("expertise_rank", ascending=False)
    m.reset_index().to_csv(OUTPUT_DIR / "user_metrics.csv", index=False)

    # ---- communities summary
    comm = (m.groupby("community")
            .agg(size=("out_strength", "size"), help_given=("out_strength", "sum"),
                 top_member=("user_name", "first"))
            .sort_values("size", ascending=False))
    comm.reset_index().to_csv(OUTPUT_DIR / "communities.csv", index=False)

    # ---- global stats, validation, time
    stats = global_stats(g, ug, communities, m)
    stats["role_counts"] = m["role"].value_counts().to_dict()
    with open(OUTPUT_DIR / "network_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=float)

    # Validate on helpers only: most users never answer, and thousands of tied zeros would
    # swamp the rank correlation.
    val = []
    helpers = m[(m["n_answers"] > 0) & m["user_reputation"].notna()]
    for metric in ["expertise_rank", "pagerank", "out_strength", "betweenness", "core_number"]:
        for target in ["user_reputation", "n_accepted"]:
            rho, p = spearmanr(helpers[metric], helpers[target])
            top_metric = set(helpers.nlargest(20, metric).index)
            top_target = set(helpers.nlargest(20, target).index)
            val.append({"metric": metric, "compared_with": target, "n_helpers": len(helpers),
                        "spearman_rho": rho, "p_value": p,
                        "top20_overlap": len(top_metric & top_target) / 20})
    pd.DataFrame(val).round(4).to_csv(OUTPUT_DIR / "network_validation.csv", index=False)

    snapshots(inter).round(4).to_csv(OUTPUT_DIR / "network_over_time.csv", index=False)

    print(f"[network] {stats['n_communities']} communities (modularity {stats['modularity']:.2f}); "
          f"top 10% of users give {stats['help_share_top10pct']:.0%} of all help")
    return m


if __name__ == "__main__":
    run()
