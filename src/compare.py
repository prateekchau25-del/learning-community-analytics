"""
Step 8: compare the learning communities (Data Science vs Artificial Intelligence).

Outputs in outputs/comparison/:
    overview.csv       one row per measure, one column per community
    tests.csv          statistical tests of the differences
    category_mix.csv   share of questions per subject category
    shared_users.csv   people active in both communities (linked by Stack Exchange account_id)
    figures/*.png      comparison figures
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from config import COMMUNITIES, COMPARISON_DIR, site_dirs
from data_loader import load_raw

SITES = [s for s in COMMUNITIES if (site_dirs(s)["outputs"] / "network_stats.json").exists()]


def site_summary(site: str) -> dict:
    d = site_dirs(site)
    q = pd.read_parquet(d["processed"] / "questions_clean.parquet")
    net = json.loads((d["outputs"] / "network_stats.json").read_text())
    dif = json.loads((d["outputs"] / "diffusion_summary.json").read_text())
    comp = pd.read_csv(d["outputs"] / "topic_model_comparison.csv")
    pred = pd.read_csv(d["outputs"] / "prediction_metrics.csv")
    rec = pd.read_csv(d["outputs"] / "recommender_eval.csv").set_index("method")
    im = pd.read_csv(d["outputs"] / "influence_maximization.csv")
    raw = load_raw(site)
    k10 = im[(im.k == im.k.max()) & (im.strategy == "CELF greedy")].set_index("model")["spread_pct_of_users"]
    best = pred.iloc[1:].sort_values("roc_auc").iloc[-1]
    return {
        "Questions": len(raw["questions"]), "Answers": len(raw["answers"]), "Comments": len(raw["comments"]),
        "Users": len(raw["users"]),
        "First question": str(q.created_at.min().date()), "Last question": str(q.created_at.max().date()),
        "Answered (%)": 100 * q.is_answered.mean(), "Accepted answer (%)": 100 * q.has_accepted.mean(),
        "Median hours to first answer": q.hours_to_first_answer.median(),
        "Questions with code (%)": 100 * q.has_code.mean(), "Questions with maths (%)": 100 * q.has_math.mean(),
        "Mean confusion score": q.confusion_score.mean(),
        "BERTopic topics": int(comp.iloc[1].n_topics), "BERTopic NPMI": comp.iloc[1].npmi,
        "BERTopic NMI vs tags": comp.iloc[1].nmi_vs_tags, "LDA NMI vs tags": comp.iloc[0].nmi_vs_tags,
        "Network users": net["n_users"], "Help links": net["n_edges"],
        "Users who help (%)": 100 * net["pct_users_who_help"],
        "Help from top 1% (%)": 100 * net["help_share_top1pct"],
        "Help from top 10% (%)": 100 * net["help_share_top10pct"], "Gini of help": net["gini_help_given"],
        "Reciprocity": net["reciprocity"], "Modularity": net["modularity"],
        "Communities (10+ users)": net["n_communities_10plus"],
        "Exposure effect z (Stouffer)": dif.get("exposure_stouffer_z"),
        "Exposure effect p": dif.get("exposure_stouffer_p"),
        "Topics with exposure effect": f"{dif['exposure_topics_significant']}/{dif['exposure_topics_tested']}",
        "Learners who became helpers (%)": 100 * dif["learner_to_helper"]["pct_became_helpers"],
        "Simulation vs reality (Spearman)": dif["validation"]["simulated_ic_spread"],
        "IC reach, 10 seeds (%)": k10.get("IC"), "LT reach, 10 seeds (%)": k10.get("LT"),
        "Best answer model": best.model, "Answer prediction ROC-AUC": best.roc_auc,
        "Expert finder Hit@10 (%)": 100 * rec.loc["Hybrid (embedding)", "hit@10"],
    }


def tests(sites):
    """Are the differences between the communities statistically significant?"""
    q = {s: pd.read_parquet(site_dirs(s)["processed"] / "questions_clean.parquet") for s in sites}
    a, b = sites
    rows = []
    table = [[q[s].is_answered.sum(), (~q[s].is_answered).sum()] for s in sites]
    chi2, p, _, _ = chi2_contingency(table)
    rows.append({"measure": "Answered rate", "test": "chi-square", "statistic": chi2, "p_value": p,
                 a: q[a].is_answered.mean(), b: q[b].is_answered.mean()})
    for col, label in [("hours_to_first_answer", "Hours to first answer"), ("confusion_score", "Confusion score"),
                       ("n_words", "Question length (words)")]:
        x, y = q[a][col].dropna(), q[b][col].dropna()
        stat, p = mannwhitneyu(x, y)
        rows.append({"measure": label, "test": "Mann-Whitney U", "statistic": stat, "p_value": p,
                     a: x.median(), b: y.median()})
    return pd.DataFrame(rows)


def category_mix(sites):
    frames = []
    for s in sites:
        qt = pd.read_parquet(site_dirs(s)["processed"] / "question_topics.parquet")
        share = qt["category"].fillna("Uncategorised").value_counts(normalize=True).rename(s)
        frames.append(share)
    return (pd.concat(frames, axis=1).fillna(0) * 100).reset_index().rename(columns={"index": "category"})


def shared_users(sites):
    """People with accounts in both communities, and what they do in each."""
    a, b = sites
    per_site = {}
    for s in sites:
        users = load_raw(s)["users"].dropna(subset=["account_id"])
        users = users[users["account_id"] > 0]  # -1 is the automated "Community" account
        m = pd.read_csv(site_dirs(s)["outputs"] / "user_metrics.csv")[["user_id", "role", "out_strength", "n_answers", "n_questions"]]
        per_site[s] = users.merge(m, on="user_id", how="left").fillna({"out_strength": 0, "n_answers": 0, "n_questions": 0})
    both = per_site[a].merge(per_site[b], on="account_id", suffixes=(f"_{a}", f"_{b}"))
    both["user_name"] = both[f"user_name_{a}"]
    cols = ["account_id", "user_name"] + [f"{c}_{s}" for s in sites for c in ("role", "out_strength", "n_answers", "n_questions")]
    both = both[cols].sort_values([f"out_strength_{a}", f"out_strength_{b}"], ascending=False)
    help_total = {s: pd.read_csv(site_dirs(s)["outputs"] / "user_metrics.csv")["out_strength"].sum() for s in sites}
    stats = {
        "shared_users": len(both),
        "helpers_in_both": int(((both[f"n_answers_{a}"] > 0) & (both[f"n_answers_{b}"] > 0)).sum()),
        **{f"share_of_help_{s}": float(both[f"out_strength_{s}"].sum() / help_total[s]) for s in sites},
    }
    return both, stats


def figures(overview, mix, sites):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures import SERIES, TEXT, style_axes
    fig_dir = COMPARISON_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [COMMUNITIES[s] for s in sites]

    # 1. key measures side by side (all in %, one axis)
    measures = ["Answered (%)", "Accepted answer (%)", "Users who help (%)", "Help from top 1% (%)",
                "Help from top 10% (%)", "Expert finder Hit@10 (%)"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(measures))
    for i, s in enumerate(sites):
        vals = [float(overview.loc[m, s]) for m in measures]
        bars = ax.bar(x + (i - 0.5) * 0.4, vals, 0.38, color=SERIES[i], label=names[i])
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (%)", "") for m in measures], fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Data Science vs Artificial Intelligence: key community measures")
    ax.legend(loc="upper left")
    style_axes(ax, grid="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "c1_key_measures.png", dpi=200)
    plt.close(fig)

    # 2. subject mix
    m = mix.set_index("category")
    m = m.loc[m.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(m))
    for i, s in enumerate(sites):
        ax.barh(y + (i - 0.5) * 0.4, m[s], 0.38, color=SERIES[i], label=names[i])
    ax.set_yticks(y)
    ax.set_yticklabels(m.index, fontsize=9)
    ax.set_xlabel("% of the community's questions")
    ax.set_title("What each community talks about (tag categories)")
    ax.legend(loc="lower right")
    style_axes(ax, grid="x")
    fig.tight_layout()
    fig.savefig(fig_dir / "c2_category_mix.png", dpi=200)
    plt.close(fig)


def run():
    if len(SITES) < 2:
        print("[compare] need results for two communities; run the pipeline for both first")
        return
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {s: site_summary(s) for s in SITES}
    overview = pd.DataFrame(summaries)
    overview.index.name = "measure"
    overview.reset_index().to_csv(COMPARISON_DIR / "overview.csv", index=False)
    tests(SITES).round(5).to_csv(COMPARISON_DIR / "tests.csv", index=False)
    mix = category_mix(SITES)
    mix.round(2).to_csv(COMPARISON_DIR / "category_mix.csv", index=False)
    both, stats = shared_users(SITES)
    both.to_csv(COMPARISON_DIR / "shared_users.csv", index=False)
    (COMPARISON_DIR / "shared_users_summary.json").write_text(json.dumps(stats, indent=2))
    figures(overview, mix, SITES)
    print(f"[compare] {stats['shared_users']} people are active in both communities "
          f"({stats['helpers_in_both']} of them answer in both)")


if __name__ == "__main__":
    run()
