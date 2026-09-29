"""
Step 7: static figures for the report (PNG, 200 dpi) in outputs/figures/.

Colour roles follow one validated categorical order (never cycled); magnitude uses one
blue ramp, and signed values use a blue/red diverging pair around grey.
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import COMMUNITY_NAME, FIGURE_DIR, OUTPUT_DIR, PROCESSED_DIR  # noqa: E402

warnings.filterwarnings("ignore")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b4b2aa"
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEG, NEUTRAL, POS = "#2a78d6", "#f0efec", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2,
})


# Short names for the tag categories, used where labels must stay compact.
CATEGORY_SHORT = {
    "Generative AI & LLMs": "GenAI", "Natural Language Processing": "NLP", "Computer Vision": "Vision",
    "Reinforcement Learning": "RL", "Time Series & Sequences": "Time series", "Deep Learning": "DL",
    "Classical ML & Modelling": "ML", "Data Wrangling & Tools": "Data/tools", "Statistics & Theory": "Stats",
    "AI Philosophy & Ethics": "Ethics/AGI",
}


def style_axes(ax, grid="y"):
    """Keep only the value-axis grid."""
    ax.grid(axis="x" if grid == "y" else "y", visible=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def topic_names():
    t = pd.read_csv(OUTPUT_DIR / "topics.csv")
    return dict(zip(t["topic"], t["label"]))


# ------------------------------------------------------------------ community & topics

def fig_activity():
    q = pd.read_parquet(PROCESSED_DIR / "questions_clean.parquet")
    per = q.groupby(q["created_at"].dt.to_period("Q")).size()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(per.index.astype(str), per.values, color=SERIES[0], width=0.8)
    peak = per.idxmax()
    ax.annotate(f"peak: {per.max()} questions in {peak}", xy=(list(per.index).index(peak), per.max()),
                xytext=(10, -5), textcoords="offset points", color=TEXT_2, fontsize=9, va="top")
    ax.set_title(f"Questions asked per quarter, {COMMUNITY_NAME} Stack Exchange")
    ax.set_ylabel("Questions")
    ax.set_xticks(range(0, len(per), 4))
    ax.set_xticklabels([str(p) for p in per.index[::4]])
    ax.grid(axis="x", visible=False)
    save(fig, "01_activity_over_time")


def fig_lda_selection():
    s = pd.read_csv(OUTPUT_DIR / "lda_model_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, title in [(axes[0], "npmi", "NPMI coherence (higher = better)"),
                           (axes[1], "diversity", "Topic diversity (higher = better)")]:
        ax.plot(s["k"], s[col], color=SERIES[0], marker="o", markersize=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Number of LDA topics (k)")
        ax.set_xticks(s["k"])
    save(fig, "02_lda_model_selection")


def fig_model_comparison():
    c = pd.read_csv(OUTPUT_DIR / "topic_model_comparison.csv")
    metrics = [("npmi", "NPMI coherence"), ("diversity", "Topic diversity"), ("nmi_vs_tags", "Agreement with\ntag categories (NMI)")]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.36
    for i, (_, row) in enumerate(c.iterrows()):
        vals = [row[m] for m, _ in metrics]
        label = row["model"].split(" (")[0] + (f" ({int(row['n_topics'])} topics)")
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=SERIES[i], label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, color=TEXT, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.05)
    lda, bt = c.iloc[0], c.iloc[1]
    wins = sum(bt[m] > lda[m] for m, _ in metrics)
    ax.set_title(f"{COMMUNITY_NAME}: BERTopic beats LDA on {wins} of {len(metrics)} topic-quality metrics")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "03_topic_model_comparison")


def fig_topic_difficulty():
    t = pd.read_csv(OUTPUT_DIR / "topics.csv").sort_values("difficulty_index")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = [POS if v > 0 else NEG for v in t["difficulty_index"]]
    # Name the category only when most of the topic's tagged questions come from it.
    names = [f"{CATEGORY_SHORT.get(u, u) if isinstance(u, str) and pur >= 0.5 else 'Mixed'}: {lbl}  ({n} q)"
             for u, pur, lbl, n in zip(t["category"], t["category_purity"], t["label"], t["n_questions"])]
    ax.barh(names, t["difficulty_index"], color=colors, height=0.7)
    ax.axvline(0, color=TEXT_2, linewidth=1)
    fig.suptitle("Topic difficulty index\n(unanswered rate + time to answer + learner confusion)", x=0.02, ha="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("← easier than average      harder than average →")
    ax.grid(axis="y", visible=False)
    save(fig, "04_topic_difficulty")


def fig_topic_trends():
    tr = pd.read_csv(OUTPUT_DIR / "topic_trends.csv")
    tr["year"] = tr["period"].str[:4]
    y = tr.groupby(["topic", "year"])["n_questions"].sum().unstack(fill_value=0)
    y = y.loc[:, y.sum(axis=0) >= 100]  # years with too few questions give misleading shares
    share = y / y.sum(axis=0)
    names = topic_names()
    share = share.loc[share.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff"] + BLUE_RAMP)
    im = ax.imshow(share.values * 100, aspect="auto", cmap=cmap)
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([names.get(t, t) for t in share.index], fontsize=8.5)
    ax.set_xticks(range(len(share.columns)))
    ax.set_xticklabels(share.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("% of that year's questions")
    cb.outline.set_visible(False)
    ax.set_title("How the discussion mix changed over the years\n(years with 100+ questions)")
    save(fig, "05_topic_trends")


# ------------------------------------------------------------------ network

def fig_help_concentration():
    m = pd.read_csv(OUTPUT_DIR / "user_metrics.csv")
    h = np.sort(m["out_strength"].to_numpy())[::-1]
    cum = np.cumsum(h) / h.sum()
    users = np.arange(1, len(h) + 1) / len(h)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(users * 100, cum * 100, color=SERIES[0])
    ax.plot([0, 100], [0, 100], color=OTHER, linestyle="--", linewidth=1)
    for pct in [1, 10]:
        i = max(int(len(h) * pct / 100) - 1, 0)
        ax.plot(users[i] * 100, cum[i] * 100, "o", color=SERIES[0], markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"top {pct}% of users give {cum[i]:.0%} of help", (users[i] * 100, cum[i] * 100),
                    xytext=(10, -14), textcoords="offset points", fontsize=9, color=TEXT)
    ax.text(60, 52, "equal sharing", color=TEXT_2, fontsize=9, rotation=28)
    ax.set_xlabel("% of users (most helpful first)")
    ax.set_ylabel("% of all help given")
    ax.set_title("Help is concentrated in a small core of experts")
    save(fig, "06_help_concentration")


def fig_degree_distribution():
    m = pd.read_csv(OUTPUT_DIR / "user_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, title in [(axes[0], "out_degree", "Help given (people helped)"),
                           (axes[1], "in_degree", "Help received (helpers)")]:
        d = m[col][m[col] > 0].to_numpy()
        vals = np.sort(np.unique(d))
        ccdf = [(d >= v).mean() for v in vals]
        ax.loglog(vals, ccdf, "o", color=SERIES[0], markersize=4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Degree (log)")
        ax.set_ylabel("P(degree ≥ x)")
    fig.suptitle("Heavy-tailed degree distributions: a few hubs, many one-off learners",
                 x=0.02, ha="left", fontweight="bold", fontsize=12)
    save(fig, "07_degree_distribution")


def fig_network_graph():
    m = pd.read_csv(OUTPUT_DIR / "user_metrics.csv").set_index("user_id")
    e = pd.read_csv(OUTPUT_DIR / "help_edges.csv")
    top = m.nlargest(150, "out_strength").index
    sub = e[e.source_user.isin(top) | e.target_user.isin(top)]
    keep = set(top) | set(sub.target_user.value_counts().index[:250])
    sub = sub[sub.source_user.isin(keep) & sub.target_user.isin(keep)]
    g = nx.Graph()
    for r in sub.itertuples():
        g.add_edge(r.source_user, r.target_user, weight=r.weight)
    pos = nx.spring_layout(g, k=0.6, seed=42, weight="weight", iterations=150)
    roles = ["Core expert", "Bridge", "Helper"]
    color = {r: SERIES[i] for i, r in enumerate(roles)}
    nodes = list(g.nodes)
    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GRID, width=0.6, alpha=0.9)
    sizes = [15 + 18 * np.sqrt(m.at[n, "out_strength"]) if n in m.index else 15 for n in nodes]
    cols = [color.get(m.at[n, "role"], OTHER) if n in m.index else OTHER for n in nodes]
    order = np.argsort(sizes)  # draw big nodes last so they stay visible
    nx.draw_networkx_nodes(g, pos, nodelist=[nodes[i] for i in order], node_size=[sizes[i] for i in order],
                           node_color=[cols[i] for i in order], ax=ax, edgecolors=SURFACE, linewidths=1)
    for n in m.loc[m.index.isin(nodes)].nlargest(3, "out_strength").index:
        ax.annotate(m.at[n, "user_name"], pos[n], xytext=(14, 10), textcoords="offset points",
                    fontsize=9, color=TEXT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    counts = m["role"].value_counts()
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color[r], markersize=8,
                          label=f"{r} ({counts.get(r, 0)} users)") for r in roles]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", color=OTHER, markersize=8, label="Learner / other"))
    ax.legend(handles=handles, loc="lower left", fontsize=8.5)
    ax.set_title("Knowledge network core: top 150 helpers and the learners they help\n(node size = help given)")
    ax.axis("off")
    save(fig, "08_network_core")


def fig_network_over_time():
    t = pd.read_csv(OUTPUT_DIR / "network_over_time.csv")
    t = t[t["active_users"] >= 50]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, col, title, fmt in [(axes[0], "active_users", "Active users", "{:.0f}"),
                                (axes[1], "n_helpers", "Users who helped", "{:.0f}"),
                                (axes[2], "help_share_top10pct", "Help given by top 10%", "{:.0%}")]:
        ax.plot(t["year"], t[col], color=SERIES[0], marker="o", markersize=5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(t["year"])
        if "share" in col:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    peak = int(t.loc[t["active_users"].idxmax(), "year"]) if len(t) else None
    fig.suptitle(f"{COMMUNITY_NAME}: the network over time (most active year: {peak})", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)
    save(fig, "09_network_over_time")


# ------------------------------------------------------------------ diffusion

def fig_exposure():
    e = pd.read_csv(OUTPUT_DIR / "diffusion_exposure_test.csv").sort_values("observed_exposed_share")
    summ = json.load(open(OUTPUT_DIR / "diffusion_summary.json"))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(e))
    ax.hlines(y, e["null_mean"] * 100, e["observed_exposed_share"] * 100, color=GRID, linewidth=2)
    ax.plot(e["null_mean"] * 100, y, "o", color=OTHER, markersize=8, label="Expected by chance (shuffled times)")
    ax.plot(e["observed_exposed_share"] * 100, y, "o", color=SERIES[0], markersize=8, label="Observed")
    for yi, (obs, null, p) in enumerate(zip(e["observed_exposed_share"], e["null_mean"], e["p_value"])):
        ax.text(max(obs, null) * 100 + 0.8, yi, f"p = {p:.3f}", va="center", fontsize=8,
                color=TEXT if p < 0.05 else TEXT_2, fontweight="bold" if p < 0.05 else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(e["label"], fontsize=8.5)
    ax.set_xlabel("% of adopters who had contact with an earlier adopter before joining the topic")
    ax.set_title(f"Social exposure test: {int((e.p_value < 0.05).sum())} of {len(e)} topics significant\n"
                 f"(all topics combined: Stouffer z = {summ['exposure_stouffer_z']:.2f}, "
                 f"p = {summ['exposure_stouffer_p']:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "10_exposure_test")


def fig_learner_to_helper():
    s = json.load(open(OUTPUT_DIR / "diffusion_summary.json"))["learner_to_helper"]
    vals = [s["pct_helper_if_first_question_unanswered"], s["pct_helper_if_first_question_answered"]]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(["First question\nnot answered", "First question\nanswered"], [v * 100 for v in vals],
                  color=[OTHER, SERIES[0]], width=0.55)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=3, color=TEXT)
    ax.set_ylabel("% who later helped others")
    verdict = ("Learners whose first question was answered help others more often" if s["p_value"] < 0.05
               else "Getting an answer does not change who becomes a helper")
    ax.set_title(f"{verdict}\n(χ² p = {s['p_value']:.2f}, {s['n_became_helpers']} of {s['n_learners']} learners became helpers)",
                 fontsize=11)
    ax.grid(axis="x", visible=False)
    save(fig, "11_learner_to_helper")


def fig_influence():
    im = pd.read_csv(OUTPUT_DIR / "influence_maximization.csv")
    order = ["CELF greedy", "Top ExpertiseRank", "Top out-degree", "Top betweenness", "Top PageRank", "Random"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, model, title in [(axes[0], "IC", "Independent Cascade"), (axes[1], "LT", "Linear Threshold")]:
        d = im[im.model == model]
        for i, strat in enumerate(order):
            s = d[d.strategy == strat].sort_values("k")
            color = OTHER if strat == "Random" else SERIES[i]
            ax.plot(s["k"], s["spread_pct_of_users"], color=color, marker="o", markersize=4,
                    label=strat, linestyle="--" if strat == "Random" else "-")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Number of seed users (k)")
        ax.set_ylabel("% of users reached")
        ax.set_xticks(range(1, 11))
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle("Influence maximisation: which seed users spread knowledge furthest",
                 x=0.02, ha="left", fontweight="bold", fontsize=12)
    save(fig, "12_influence_maximization")


def fig_diffusion_validation():
    v = pd.read_csv(OUTPUT_DIR / "diffusion_validation.csv")
    rho = json.load(open(OUTPUT_DIR / "diffusion_summary.json"))["validation"]["simulated_ic_spread"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(v["simulated_ic_spread"], v["observed_reach"], s=36, color=SERIES[0],
               edgecolors=SURFACE, linewidths=1.5, alpha=0.85)
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlabel("Simulated IC spread from this user (users reached)")
    ax.set_ylabel("Observed reach within one year")
    ax.set_title(f"Simulated vs observed diffusion\n(Spearman ρ = {rho:.2f}, {len(v)} most active helpers)")
    save(fig, "13_diffusion_validation")


# ------------------------------------------------------------------ prediction & recommender

def fig_prediction():
    roc = pd.read_csv(OUTPUT_DIR / "prediction_roc.csv")
    met = pd.read_csv(OUTPUT_DIR / "prediction_metrics.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, model in enumerate([m for m in met.index if not m.startswith("Baseline")]):
        d = roc[roc.model == model]
        ax.plot(d["fpr"], d["tpr"], color=SERIES[i], label=f"{model} (AUC {met.at[model, 'roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], color=OTHER, linestyle="--", linewidth=1, label="Chance (AUC 0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Will a new question be answered? (future test set)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "14_prediction_roc")

    imp = pd.read_csv(OUTPUT_DIR / "prediction_importance.csv").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(imp["feature"].str.replace("_", " "), imp["importance_mean"], xerr=imp["importance_std"],
            color=SERIES[0], height=0.65, error_kw={"ecolor": TEXT_2, "elinewidth": 1})
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    fig.suptitle(f"What predicts an answer ({imp['model'].iloc[0]}, permutation importance)",
                 x=0.02, ha="left", fontweight="bold", fontsize=12)
    ax.grid(axis="y", visible=False)
    save(fig, "15_prediction_importance")


def fig_recommender():
    r = pd.read_csv(OUTPUT_DIR / "recommender_eval.csv")
    ks = ["hit@1", "hit@5", "hit@10"]
    x = np.arange(len(r))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, k in enumerate(ks):
        bars = ax.bar(x + (i - 1) * (w + 0.015), r[k] * 100, w, color=SERIES[i], label=k.replace("hit@", "Hit@"))
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(r["method"].str.replace(" (", "\n(", regex=False), fontsize=9)
    ax.set_ylabel("% of test questions")
    ax.set_ylim(0, 100)
    ax.set_title("Expert finder: is a real answerer among the top-k recommendations?")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    save(fig, "16_recommender_eval")


FIGURES = [fig_activity, fig_lda_selection, fig_model_comparison, fig_topic_difficulty, fig_topic_trends,
           fig_help_concentration, fig_degree_distribution, fig_network_graph, fig_network_over_time,
           fig_exposure, fig_learner_to_helper, fig_influence, fig_diffusion_validation,
           fig_prediction, fig_recommender]


def run():
    ok = 0
    for f in FIGURES:
        try:
            f()
            ok += 1
        except FileNotFoundError as exc:
            print(f"[figures] skipped {f.__name__}: missing {exc.filename}")
    print(f"[figures] {ok}/{len(FIGURES)} figure groups written to {FIGURE_DIR}")


if __name__ == "__main__":
    run()
