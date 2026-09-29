"""
Step 4: knowledge diffusion — how topics and help spread through the community.

A. Observed diffusion
   - Topic adoption curves: when each user first takes part in a topic.
   - Social exposure test: were adopters more often in contact with earlier adopters than
     chance predicts? (time-shuffle test, Anagnostopoulos et al., 2008)
   - Learner -> helper transition: do learners who receive help go on to help others?
   - Thread cascades: how large the discussion grows around each question.
B. Simulated diffusion on the help network
   - Independent Cascade (IC, p = 1 - (1 - beta)^interactions) and Linear Threshold (LT)
     models (Kempe et al., 2003).
   - Influence maximisation with CELF greedy vs. centrality heuristics and random seeds.
   - Validation: does simulated influence match observed time-respecting reach?
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED
from data_loader import help_interactions, load_raw

N_SHUFFLES = 200
MC_RUNS_SELECT = 200      # Monte Carlo runs while choosing seeds
MC_RUNS_EVAL = 300        # Monte Carlo runs when reporting spread
N_CANDIDATES = 150        # seed candidates: most active helpers
MAX_SEEDS = 10
IC_BETA = 0.1             # IC: chance that one interaction passes the knowledge on

rng = np.random.default_rng(RANDOM_SEED)


# =========================================================================== A. observed

def participation_events(raw, qt):
    """(user, topic, time, kind): asking a question or answering one in a topic."""
    q = raw["questions"][["post_id", "user_id", "created_at"]].merge(qt[["post_id", "topic"]], on="post_id")
    a = raw["answers"][["question_id", "user_id", "created_at"]].merge(
        qt[["post_id", "topic"]], left_on="question_id", right_on="post_id")
    ev = pd.concat([q.assign(kind="ask")[["user_id", "topic", "created_at", "kind"]],
                    a.assign(kind="help")[["user_id", "topic", "created_at", "kind"]]])
    ev = ev.dropna(subset=["user_id"])
    ev["user_id"] = ev["user_id"].astype("int64")
    return ev


def contact_arrays(inter, user_index):
    """Every interaction in both directions as parallel arrays (u, v, time in days)."""
    s = inter["source_user"].map(user_index).to_numpy()
    t = inter["target_user"].map(user_index).to_numpy()
    ts = (inter["created_at"].astype("int64") // 10**9 / 86400).to_numpy()
    ok = ~(np.isnan(s.astype(float)) | np.isnan(t.astype(float)))
    s, t, ts = s[ok].astype(int), t[ok].astype(int), ts[ok]
    return np.concatenate([s, t]), np.concatenate([t, s]), np.concatenate([ts, ts])


def exposed_share(adopt_time, adopters, cu, cv, ct):
    """Share of adopters who, before adopting, had contact with someone who adopted earlier."""
    hit = (adopt_time[cv] < ct) & (ct < adopt_time[cu])
    exposed = np.intersect1d(np.unique(cu[hit]), adopters)
    first = adopters[np.argmin(adopt_time[adopters])]
    return len(np.setdiff1d(exposed, [first])) / max(len(adopters) - 1, 1)


def exposure_test(events, inter, topics):
    users = np.unique(np.concatenate([events["user_id"].unique(),
                                      inter["source_user"].unique(), inter["target_user"].unique()]))
    user_index = pd.Series(np.arange(len(users)), index=users)
    cu, cv, ct = contact_arrays(inter, user_index)
    first = events.groupby(["topic", "user_id"])["created_at"].min().reset_index()
    first["day"] = first["created_at"].astype("int64") // 10**9 / 86400

    rows = []
    for topic, grp in first.groupby("topic"):
        if len(grp) < 30:
            continue
        idx = user_index[grp["user_id"]].to_numpy()
        adopt = np.full(len(users), np.inf)
        adopt[idx] = grp["day"].to_numpy()
        observed = exposed_share(adopt, idx, cu, cv, ct)
        null = []
        for _ in range(N_SHUFFLES):
            shuffled = np.full(len(users), np.inf)
            shuffled[idx] = rng.permutation(grp["day"].to_numpy())
            null.append(exposed_share(shuffled, idx, cu, cv, ct))
        null = np.array(null)
        rows.append({
            "topic": topic, "n_adopters": len(grp), "observed_exposed_share": observed,
            "null_mean": null.mean(), "null_std": null.std(),
            "z_score": (observed - null.mean()) / (null.std() or np.nan),
            "p_value": (np.sum(null >= observed) + 1) / (N_SHUFFLES + 1),
        })
    out = pd.DataFrame(rows)
    return out.merge(topics[["topic", "label"]], on="topic", how="left") if len(out) else out


def combine_exposure(exposure: pd.DataFrame) -> dict:
    """Stouffer's method: one overall z-score from the per-topic tests."""
    from scipy.stats import norm
    z = exposure["z_score"].replace([np.inf, -np.inf], np.nan).dropna()
    z_all = float(z.sum() / np.sqrt(len(z)))
    return {"exposure_stouffer_z": z_all, "exposure_stouffer_p": float(norm.sf(z_all)),
            "exposure_significant_topics": exposure.loc[exposure["p_value"] < 0.05, "label"].tolist()}


def adoption_curves(events, communities):
    first = events.groupby(["topic", "user_id"])["created_at"].min().reset_index()
    first["period"] = first["created_at"].dt.to_period("Q").astype(str)
    first["community"] = first["user_id"].map(communities)
    curve = first.groupby(["topic", "period"]).agg(
        new_adopters=("user_id", "size"),
        communities_reached=("community", lambda s: s.dropna().nunique())).reset_index()
    curve["cumulative_adopters"] = curve.groupby("topic")["new_adopters"].cumsum()
    curve["share_of_final"] = curve["cumulative_adopters"] / curve.groupby("topic")["new_adopters"].transform("sum")
    return curve


def learner_to_helper(raw):
    """Do learners whose first question got answered later become helpers themselves?"""
    q = raw["questions"].dropna(subset=["user_id"]).sort_values("created_at")
    a = raw["answers"].dropna(subset=["user_id"])
    first_q = q.groupby("user_id").first()[["post_id", "created_at"]]
    answered_ids = set(a["question_id"])
    accepted_ids = set(q.loc[q["accepted_answer_id"].notna(), "post_id"])
    first_q["got_answer"] = first_q["post_id"].isin(answered_ids)
    first_q["got_accepted"] = first_q["post_id"].isin(accepted_ids)
    asker = q.set_index("post_id")["user_id"]
    a = a[a["question_id"].map(asker) != a["user_id"]]  # answering your own question is not helping others
    first_help = a.groupby("user_id")["created_at"].min()
    first_q["first_help_at"] = first_help.reindex(first_q.index)
    first_q["became_helper"] = first_q["first_help_at"] > first_q["created_at"]
    first_q["days_to_helper"] = (first_q["first_help_at"] - first_q["created_at"]).dt.days

    rate = first_q.groupby("got_answer")["became_helper"].mean()
    table = pd.crosstab(first_q["got_answer"], first_q["became_helper"])
    from scipy.stats import chi2_contingency
    chi2, p, _, _ = chi2_contingency(table) if table.shape == (2, 2) else (np.nan, np.nan, None, None)
    helpers = first_q[first_q["became_helper"]]
    return {
        "n_learners": int(len(first_q)),
        "n_became_helpers": int(first_q["became_helper"].sum()),
        "pct_became_helpers": float(first_q["became_helper"].mean()),
        "pct_helper_if_first_question_answered": float(rate.get(True, np.nan)),
        "pct_helper_if_first_question_unanswered": float(rate.get(False, np.nan)),
        "chi2": float(chi2), "p_value": float(p),
        "median_days_to_become_helper": float(helpers["days_to_helper"].median()) if len(helpers) else None,
    }


def thread_cascades(raw, qt):
    """Size of the discussion around each question (answers + comments on the thread)."""
    q, a, c = raw["questions"], raw["answers"], raw["comments"]
    post_to_q = pd.concat([q[["post_id"]].assign(question_id=q["post_id"]),
                           a[["post_id", "question_id"]]])
    cm = c.merge(post_to_q, on="post_id")
    size = (a.groupby("question_id").size().rename("n_answers").to_frame()
            .join(cm.groupby("question_id").size().rename("n_comments"), how="outer").fillna(0))
    people = pd.concat([a[["question_id", "user_id"]], cm[["question_id", "user_id"]]])
    size["n_participants"] = people.dropna().groupby("question_id")["user_id"].nunique()
    size = size.fillna(0)
    size["cascade_size"] = size["n_answers"] + size["n_comments"]
    size = size.reset_index().rename(columns={"index": "question_id"})
    size = q[["post_id"]].merge(size, left_on="post_id", right_on="question_id", how="left").fillna(0)
    size = size.merge(qt[["post_id", "topic"]], on="post_id", how="left")
    size = size.dropna(subset=["topic"]).astype({"topic": int})
    per_topic = size.groupby("topic").agg(
        n_threads=("post_id", "size"), mean_cascade_size=("cascade_size", "mean"),
        max_cascade_size=("cascade_size", "max"), mean_participants=("n_participants", "mean")).reset_index()
    distribution = size["cascade_size"].value_counts().sort_index().rename("n_threads").reset_index()
    return per_topic, distribution


# =========================================================================== B. simulated

class HelpNetwork:
    """Compact adjacency lists for fast Monte Carlo simulation (knowledge flows helper -> learner)."""

    def __init__(self, edges: pd.DataFrame):
        nodes = pd.Index(pd.unique(pd.concat([edges.source_user, edges.target_user])))
        self.nodes = nodes
        self.n = len(nodes)
        src = nodes.get_indexer(edges.source_user)
        dst = nodes.get_indexer(edges.target_user)
        w = edges["weight"].to_numpy(float)
        in_strength = np.bincount(dst, weights=w, minlength=self.n)
        # IC: each of the w interactions independently passes knowledge with probability beta.
        p_ic = 1 - (1 - IC_BETA) ** w
        # LT: influence weights w_uv / in_strength(v), which sum to 1 over v's helpers.
        b_lt = w / in_strength[dst]
        order = np.argsort(src, kind="stable")
        self.src, self.dst = src[order], dst[order]
        self.p_ic, self.b_lt = p_ic[order], b_lt[order]
        self.ptr = np.searchsorted(self.src, np.arange(self.n + 1))

    def out(self, u, weights):
        a, b = self.ptr[u], self.ptr[u + 1]
        return self.dst[a:b], weights[a:b]

    def ic(self, seeds, runs):
        total = 0
        for _ in range(runs):
            active = np.zeros(self.n, bool)
            active[seeds] = True
            frontier = list(seeds)
            while frontier:
                nxt = []
                for u in frontier:
                    v, p = self.out(u, self.p_ic)
                    hit = v[(~active[v]) & (rng.random(len(v)) < p)]
                    active[hit] = True
                    nxt.extend(hit.tolist())
                frontier = nxt
            total += active.sum()
        return total / runs

    def lt(self, seeds, runs):
        total = 0
        for _ in range(runs):
            threshold = rng.random(self.n)
            pressure = np.zeros(self.n)
            active = np.zeros(self.n, bool)
            active[seeds] = True
            frontier = list(seeds)
            while frontier:
                nxt = []
                for u in frontier:
                    v, p = self.out(u, self.b_lt)
                    pressure[v] += p
                    new = v[(~active[v]) & (pressure[v] >= threshold[v])]
                    active[new] = True
                    nxt.extend(new.tolist())
                frontier = nxt
            total += active.sum()
        return total / runs


def celf(net: HelpNetwork, candidates, k, model="ic"):
    """CELF lazy-greedy influence maximisation (Leskovec et al., 2007)."""
    sim = net.ic if model == "ic" else net.lt
    gains = sorted(((sim([c], MC_RUNS_SELECT), c) for c in candidates), reverse=True)
    seeds, spread = [gains[0][1]], gains[0][0]
    gains = gains[1:]
    while len(seeds) < k and gains:
        while True:
            _, c = gains.pop(0)
            gain = sim(seeds + [c], MC_RUNS_SELECT) - spread
            if not gains or gain >= gains[0][0]:
                seeds.append(c)
                spread += gain
                break
            gains.append((gain, c))
            gains.sort(reverse=True)
    return seeds


def observed_reach(inter, sources):
    """Users reachable from each source by time-respecting helper -> learner paths."""
    ev = inter.sort_values("created_at")
    s, t = ev["source_user"].to_numpy(), ev["target_user"].to_numpy()
    ts = ev["created_at"].astype("int64").to_numpy()
    reach = {}
    for src in sources:
        arrival = {src: -np.inf}
        for a, b, time in zip(s, t, ts):
            if a in arrival and arrival[a] <= time and b not in arrival:
                arrival[b] = time
        reach[src] = len(arrival) - 1
    return pd.Series(reach)


def run():
    raw = load_raw()
    qt = pd.read_parquet(PROCESSED_DIR / "question_topics.parquet")
    topics = pd.read_csv(OUTPUT_DIR / "topics.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "user_metrics.csv").set_index("user_id")
    edges = pd.read_csv(OUTPUT_DIR / "help_edges.csv")
    # Any contact (either direction) can expose a user to a topic; only help carries knowledge on.
    contacts = raw["interactions"].dropna(subset=["source_user", "target_user"]).copy()
    contacts["source_user"] = contacts["source_user"].astype("int64")
    contacts["target_user"] = contacts["target_user"].astype("int64")
    inter = help_interactions(raw)
    summary = {}

    # ---------------- A. observed
    events = participation_events(raw, qt)
    exposure = exposure_test(events, contacts, topics)
    exposure.round(4).to_csv(OUTPUT_DIR / "diffusion_exposure_test.csv", index=False)
    summary["exposure_topics_tested"] = int(len(exposure))
    summary["exposure_topics_significant"] = int((exposure["p_value"] < 0.05).sum())
    summary["exposure_mean_observed"] = float(exposure["observed_exposed_share"].mean())
    summary["exposure_mean_null"] = float(exposure["null_mean"].mean())
    summary.update(combine_exposure(exposure))
    print(f"[diffusion] exposure test: {summary['exposure_topics_significant']}/{len(exposure)} topics "
          f"show more contact with earlier adopters than chance")

    adoption_curves(events, metrics["community"]).to_csv(OUTPUT_DIR / "topic_adoption_curves.csv", index=False)
    summary["learner_to_helper"] = learner_to_helper(raw)
    per_topic, dist = thread_cascades(raw, qt)
    per_topic.round(3).to_csv(OUTPUT_DIR / "cascade_by_topic.csv", index=False)
    dist.to_csv(OUTPUT_DIR / "cascade_size_distribution.csv", index=False)

    # ---------------- B. simulated
    net = HelpNetwork(edges)
    idx = pd.Series(np.arange(net.n), index=net.nodes)
    in_net = metrics.index.intersection(net.nodes)
    candidates_ids = metrics.loc[in_net].sort_values("out_strength", ascending=False).index[:N_CANDIDATES]
    candidates = idx[candidates_ids].tolist()

    strategies = {}
    for name, col in [("Top out-degree", "out_strength"), ("Top ExpertiseRank", "expertise_rank"),
                      ("Top betweenness", "betweenness"), ("Top PageRank", "pagerank")]:
        strategies[name] = idx[metrics.loc[in_net].sort_values(col, ascending=False).index[:MAX_SEEDS]].tolist()

    rows, seed_sets = [], {}
    for model in ["ic", "lt"]:
        print(f"[diffusion] influence maximisation ({model.upper()}) ...")
        model_strategies = dict(strategies)
        model_strategies["CELF greedy"] = celf(net, candidates, MAX_SEEDS, model)
        sim = net.ic if model == "ic" else net.lt
        for name, seeds in model_strategies.items():
            for k in range(1, MAX_SEEDS + 1):
                rows.append({"model": model.upper(), "strategy": name, "k": k, "spread": sim(seeds[:k], MC_RUNS_EVAL)})
        random_spread = np.zeros(MAX_SEEDS)
        for _ in range(20):
            seeds = rng.choice(net.n, MAX_SEEDS, replace=False).tolist()
            random_spread += [sim(seeds[:k], MC_RUNS_EVAL // 10) for k in range(1, MAX_SEEDS + 1)]
        rows += [{"model": model.upper(), "strategy": "Random", "k": k + 1, "spread": s / 20}
                 for k, s in enumerate(random_spread)]
        seed_sets[model.upper()] = {name: [int(net.nodes[i]) for i in s] for name, s in model_strategies.items()}

    im = pd.DataFrame(rows)
    im["spread_pct_of_users"] = 100 * im["spread"] / net.n
    im.round(3).to_csv(OUTPUT_DIR / "influence_maximization.csv", index=False)
    with open(OUTPUT_DIR / "seed_sets.json", "w") as f:
        json.dump(seed_sets, f, indent=2)

    # ---------------- validation: simulated vs observed
    print("[diffusion] validating simulated influence against observed reach ...")
    sim_single = pd.Series({net.nodes[c]: net.ic([c], MC_RUNS_SELECT) for c in candidates})
    obs = observed_reach(inter, sim_single.index)
    valid = pd.DataFrame({"simulated_ic_spread": sim_single, "observed_reach": obs})
    valid = valid.join(metrics[["user_name", "out_strength", "expertise_rank", "betweenness"]])
    valid.index.name = "user_id"
    valid.reset_index().round(4).to_csv(OUTPUT_DIR / "diffusion_validation.csv", index=False)
    summary["validation"] = {
        col: float(spearmanr(valid[col], valid["observed_reach"])[0])
        for col in ["simulated_ic_spread", "out_strength", "expertise_rank", "betweenness"]
    }
    k10 = im[im.k == MAX_SEEDS].set_index(["model", "strategy"])["spread"]
    summary["spread_at_k10"] = {f"{m}|{s}": float(v) for (m, s), v in k10.items()}
    summary["network_users"] = int(net.n)

    with open(OUTPUT_DIR / "diffusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[diffusion] simulated influence vs observed reach: Spearman rho = "
          f"{summary['validation']['simulated_ic_spread']:.2f}")
    return summary


if __name__ == "__main__":
    run()
