"""
Learning Community Analytics — interactive dashboard.

Run from the project folder:
    streamlit run app/streamlit_app.py
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import OUTPUT_DIR, PROCESSED_DIR, TRAIN_FRACTION  # noqa: E402
from data_loader import data_source, get_db_engine, load_raw  # noqa: E402

st.set_page_config(page_title="Learning Community Analytics", page_icon="🎓", layout="wide")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b4b2aa"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# =========================================================================== data

@st.cache_data(show_spinner="Loading community data ...")
def raw_data():
    return load_raw()


@st.cache_data
def out(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path)


@st.cache_data
def processed(name):
    path = PROCESSED_DIR / name
    return pd.read_parquet(path) if path.exists() else None


def need(*frames):
    if any(f is None for f in frames):
        st.warning("Results are missing. Run `python src/run_pipeline.py` first.")
        st.stop()


def plot(fig, height=None):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="",
                      font=dict(size=13), hoverlabel=dict(font_size=13))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def topic_labels():
    t = out("topics.csv")
    return {} if t is None else dict(zip(t["topic"], t["label"]))


# =========================================================================== text helpers

@st.cache_resource(show_spinner="Loading language model ...")
def encoder():
    """Sentence-transformer if installed, else None (TF-IDF fallback)."""
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        return None


@st.cache_resource
def text_tools():
    from preprocess import DOMAIN_STOPWORDS, TOKEN_RE, _nltk_resources
    stop, lemmatizer, vader = _nltk_resources()
    stop = stop | DOMAIN_STOPWORDS

    def tokenize(text):
        toks = [lemmatizer.lemmatize(t) for t in TOKEN_RE.findall(text.lower())]
        return " ".join(t for t in toks if t not in stop and len(t) > 2)
    return tokenize, vader


@st.cache_resource(show_spinner="Building the expert finder ...")
def finder_and_vectors():
    from recommender import ExpertFinder
    from sklearn.feature_extraction.text import TfidfVectorizer
    q = processed("questions_clean.parquet")
    ids = np.load(PROCESSED_DIR / "embedding_post_ids.npy")
    q = q[q["post_id"].isin(ids)].reset_index(drop=True)
    if encoder() is not None:
        emb = np.load(PROCESSED_DIR / "embeddings.npy")
        idx = pd.Series(np.arange(len(ids)), index=ids)
        vectors, mode, tfidf = emb[idx[q["post_id"]].to_numpy()], "sentence embeddings", None
    else:
        tfidf = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        vectors, mode = tfidf.fit_transform(q["tokens"]).toarray().astype(np.float32), "TF-IDF"
    answers = raw_data()["answers"]
    return ExpertFinder(q, answers, vectors), q, vectors, mode, tfidf


def embed_query(text, tfidf):
    enc = encoder()
    if enc is not None:
        return enc.encode([text], normalize_embeddings=True)[0]
    tokenize, _ = text_tools()
    return tfidf.transform([tokenize(text)]).toarray()[0]


# =========================================================================== pages

def page_overview():
    raw = raw_data()
    q, a, u = raw["questions"], raw["answers"], raw["users"]
    st.title("🎓 Learning Community Analytics")
    st.caption("NLP topics and knowledge-diffusion networks in the CS50 Stack Exchange learning community")

    c = st.columns(5)
    c[0].metric("Questions", f"{len(q):,}")
    c[1].metric("Answers", f"{len(a):,}")
    c[2].metric("Comments", f"{len(raw['comments']):,}")
    c[3].metric("Users", f"{len(u):,}")
    c[4].metric("Time span", f"{q.created_at.min():%Y} – {q.created_at.max():%Y}")

    per = q.groupby(q["created_at"].dt.to_period("Q")).size()
    fig = go.Figure(go.Bar(x=per.index.astype(str), y=per.values, marker_color=SERIES[0],
                           hovertemplate="%{x}: %{y} questions<extra></extra>"))
    fig.update_layout(title="Questions per quarter (peak = COVID-19 lockdown, 2020)", yaxis_title="Questions")
    plot(fig, 360)

    st.subheader("Key findings")
    comp, stats, diff = out("topic_model_comparison.csv"), out("network_stats.json"), out("diffusion_summary.json")
    topics, pred, rec = out("topics.csv"), out("prediction_metrics.csv"), out("recommender_eval.csv")
    lines = []
    if comp is not None:
        b = comp.iloc[1]
        lines.append(f"**Topics:** BERTopic found {int(b.n_topics)} topics that match the course syllabus "
                     f"(NMI {b.nmi_vs_course_tags:.2f} vs {comp.iloc[0].nmi_vs_course_tags:.2f} for LDA).")
    if topics is not None:
        hard = topics.sort_values("difficulty_index", ascending=False).iloc[0]
        unit = str(hard.course_unit).replace(" (", ", ").rstrip(")")
        lines.append(f"**Hardest topic:** *{hard.label}* ({unit}): "
                     f"{hard.pct_unanswered:.0f}% of its questions stay unanswered.")
    if stats is not None:
        lines.append(f"**Network:** the top 1% of users give {stats['help_share_top1pct']:.0%} of all help; "
                     f"{stats['n_communities_10plus']} sub-communities (modularity {stats['modularity']:.2f}).")
    if diff is not None:
        l2h = diff["learner_to_helper"]
        lines.append(f"**Diffusion:** a modest social-influence effect overall (combined z = "
                     f"{diff['exposure_stouffer_z']:.2f}, p = {diff['exposure_stouffer_p']:.3f}), significant in "
                     f"{diff['exposure_topics_significant']} of {diff['exposure_topics_tested']} topics: "
                     f"{', '.join(diff['exposure_significant_topics'])}.")
        lines.append(f"**Simulation check:** simulated influence matches observed time-respecting reach "
                     f"(Spearman ρ = {diff['validation']['simulated_ic_spread']:.2f}).")
        lines.append(f"**Learners → helpers:** {l2h['pct_became_helpers']:.1%} of learners later helped others "
                     f"(median {l2h['median_days_to_become_helper']:.0f} days); getting an answer to the first "
                     f"question made {'a significant' if l2h['p_value'] < 0.05 else 'no significant'} difference "
                     f"(p = {l2h['p_value']:.2f}).")
    if pred is not None:
        best = pred.iloc[1:].sort_values("roc_auc").iloc[-1]
        lines.append(f"**Prediction:** {best.model} predicts whether a new question gets answered "
                     f"(ROC-AUC {best.roc_auc:.2f} on future questions).")
    if rec is not None:
        best = rec.sort_values("hit@10").iloc[-1]
        lines.append(f"**Expert finder:** a real answerer is in the top 10 for {best['hit@10']:.0%} of new questions.")
    for line in lines:
        st.markdown(f"- {line}")


def page_topics():
    st.title("🧠 Discussion topics")
    topics, comp, tmap, trends = out("topics.csv"), out("topic_model_comparison.csv"), out("topic_map.csv"), out("topic_trends.csv")
    need(topics, comp, tmap, trends)
    names = topic_labels()

    st.subheader("Model comparison: LDA vs BERTopic")
    st.dataframe(comp.rename(columns={"npmi": "NPMI coherence", "diversity": "Topic diversity",
                                      "nmi_vs_course_tags": "NMI vs course tags"}),
                 hide_index=True, width="stretch")

    st.subheader("Topic map")
    choice = st.selectbox("Highlight a topic", topics["topic"], format_func=lambda t: f"{t}: {names[t]}")
    tmap = tmap.assign(highlight=np.where(tmap["topic"] == choice, names[choice], "Other topics"))
    fig = px.scatter(tmap, x="x", y="y", color="highlight", hover_name="title",
                     color_discrete_map={names[choice]: SERIES[0], "Other topics": OTHER},
                     category_orders={"highlight": ["Other topics", names[choice]]})
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Every question, placed by meaning (UMAP of sentence embeddings)")
    plot(fig, 500)

    row = topics.set_index("topic").loc[choice]
    c = st.columns(4)
    c[0].metric("Questions", int(row.n_questions))
    c[1].metric("Unanswered", f"{row.pct_unanswered:.0f}%")
    c[2].metric("Median wait for answer", f"{row.median_hours_to_answer:.1f} h")
    # Name a course unit only when most of the topic's tagged questions come from it.
    c[3].metric("Course unit", str(row.course_unit) if row.unit_purity >= 0.5 else "Mixed")
    st.markdown(f"**Top words:** {row.top_words}")
    q = processed("questions_clean.parquet")
    qt = processed("question_topics.parquet")
    sample = q.merge(qt[qt.topic == choice][["post_id"]], on="post_id").nlargest(8, "view_count")
    st.markdown("**Most viewed questions in this topic**")
    for r in sample.itertuples():
        st.markdown(f"- [{r.title}](https://cs50.stackexchange.com/q/{r.post_id}) · {r.view_count:,} views")

    st.subheader("Topic difficulty")
    d = topics.sort_values("difficulty_index")
    fig = go.Figure(go.Bar(x=d["difficulty_index"], y=d["label"], orientation="h",
                           marker_color=[SERIES[7] if v > 0 else SERIES[0] for v in d["difficulty_index"]],
                           customdata=d[["pct_unanswered", "median_hours_to_answer", "n_questions"]],
                           hovertemplate="%{y}<br>difficulty %{x:.2f}<br>unanswered %{customdata[0]:.0f}%"
                                         "<br>median wait %{customdata[1]:.1f} h<br>%{customdata[2]} questions<extra></extra>"))
    fig.update_layout(title="Difficulty index (unanswered rate + wait time + confusion), higher = harder")
    plot(fig, 560)

    st.subheader("How topics changed over time")
    trends["year"] = trends["period"].str[:4]
    y = trends.groupby(["topic", "year"])["n_questions"].sum().reset_index()
    per_year = y.groupby("year")["n_questions"].transform("sum")
    y = y[per_year >= 100]  # years with very few questions give misleading shares
    y["share"] = 100 * y["n_questions"] / y.groupby("year")["n_questions"].transform("sum")
    y["label"] = y["topic"].map(names)
    heat = y.pivot(index="label", columns="year", values="share").fillna(0)
    fig = px.imshow(heat, color_continuous_scale=["#ffffff"] + BLUES, aspect="auto",
                    labels=dict(color="% of year's questions"))
    fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1f}% of questions<extra></extra>")
    st.caption("Only years with at least 100 questions are shown.")
    plot(fig, 560)


def network_html(metrics, edges, n_top):
    from pyvis.network import Network
    top = metrics.nlargest(n_top, "out_strength")
    ids = set(top["user_id"])
    sub = edges[edges.source_user.isin(ids) & edges.target_user.isin(ids)]
    comms = metrics["community"].value_counts().index[:3].tolist()
    color = {c: SERIES[i] for i, c in enumerate(comms)}
    net = Network(height="620px", width="100%", directed=True, bgcolor="#ffffff", font_color="#0b0b0b",
                  cdn_resources="in_line")
    net.barnes_hut(gravity=-6000, spring_length=140)
    for r in top.itertuples():
        net.add_node(int(r.user_id), label=str(r.user_name), size=8 + 3 * np.sqrt(r.out_strength),
                     color=color.get(r.community, OTHER),
                     title=f"{r.user_name}\nrole: {r.role}\nhelped: {r.out_strength}\ncommunity: {r.community}")
    for r in sub.itertuples():
        net.add_edge(int(r.source_user), int(r.target_user), value=int(r.weight), color="#cccccc")
    # generate_html avoids writing a file (pyvis would use the Windows code page and fail on names).
    return net.generate_html(notebook=False), comms


def page_network():
    st.title("🕸️ Knowledge network")
    stats, m, edges, over, val = (out("network_stats.json"), out("user_metrics.csv"), out("help_edges.csv"),
                                  out("network_over_time.csv"), out("network_validation.csv"))
    need(stats, m, edges, over, val)
    c = st.columns(5)
    c[0].metric("Users in network", f"{stats['n_users']:,}")
    c[1].metric("Helper → learner links", f"{stats['n_edges']:,}")
    c[2].metric("Help from top 1%", f"{stats['help_share_top1pct']:.0%}")
    c[3].metric("Communities (10+ users)", stats["n_communities_10plus"])
    c[4].metric("Modularity", f"{stats['modularity']:.2f}")

    st.subheader("Interactive network of the most active helpers")
    n_top = st.slider("Number of users to show", 30, 200, 80, step=10)
    html, comms = network_html(m, edges, n_top)
    st.caption("Drag nodes, scroll to zoom, hover for details. Colours = the three largest communities "
               f"({', '.join(map(str, comms))}); grey = other communities. Node size = help given.")
    st.components.v1.html(html, height=640, scrolling=False)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Top experts (ExpertiseRank)")
        names = topic_labels()
        show = m.head(20)[["user_name", "role", "out_strength", "n_answers", "n_accepted",
                           "expertise_rank", "betweenness", "community", "main_help_topic"]].copy()
        show["main_help_topic"] = show["main_help_topic"].map(names)
        st.dataframe(show, hide_index=True, width="stretch")
    with right:
        st.subheader("User roles")
        roles = m["role"].value_counts()
        fig = go.Figure(go.Bar(x=roles.values, y=roles.index, orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}: %{x} users<extra></extra>"))
        plot(fig, 330)

    st.subheader("The network over time")
    metric = st.radio("Measure", ["active_users", "n_helpers", "help_share_top10pct", "modularity", "reciprocity"],
                      horizontal=True, format_func=lambda s: s.replace("_", " "))
    fig = go.Figure(go.Scatter(x=over["year"], y=over[metric], mode="lines+markers", line=dict(color=SERIES[0], width=2),
                               marker=dict(size=8)))
    fig.update_layout(yaxis_title=metric.replace("_", " "))
    plot(fig, 330)

    st.subheader("Validation: do network scores agree with Stack Exchange reputation?")
    st.dataframe(val, hide_index=True, width="stretch")


def page_diffusion():
    st.title("🌊 Knowledge diffusion")
    summ, exp, im, curves, valid = (out("diffusion_summary.json"), out("diffusion_exposure_test.csv"),
                                    out("influence_maximization.csv"), out("topic_adoption_curves.csv"),
                                    out("diffusion_validation.csv"))
    need(summ, exp, im, curves, valid)
    l2h = summ["learner_to_helper"]
    c = st.columns(4)
    c[0].metric("Topics with social-influence signal", f"{summ['exposure_topics_significant']} / {summ['exposure_topics_tested']}",
                f"combined p = {summ['exposure_stouffer_p']:.3f}", delta_color="off")
    c[1].metric("Learners who became helpers", f"{l2h['pct_became_helpers']:.0%}")
    c[2].metric("…if first question answered", f"{l2h['pct_helper_if_first_question_answered']:.1%}",
                f"vs {l2h['pct_helper_if_first_question_unanswered']:.1%} unanswered", delta_color="off")
    c[3].metric("Simulation vs reality (ρ)", f"{summ['validation']['simulated_ic_spread']:.2f}")

    st.subheader("Social exposure test")
    st.caption("For each topic: the share of users who, before first joining the topic, had talked with someone "
               "who had already joined it — compared with 200 random re-orderings of the joining times.")
    e = exp.sort_values("observed_exposed_share")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=100 * e["null_mean"], y=e["label"], mode="markers", name="Expected by chance",
                             marker=dict(color=OTHER, size=10)))
    fig.add_trace(go.Scatter(x=100 * e["observed_exposed_share"], y=e["label"], mode="markers", name="Observed",
                             marker=dict(color=SERIES[0], size=10), customdata=e[["p_value"]],
                             hovertemplate="%{y}<br>observed %{x:.1f}%<br>p = %{customdata[0]:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="% of adopters exposed to an earlier adopter")
    plot(fig, 560)

    st.subheader("Topic adoption curves")
    names = topic_labels()
    pick = st.multiselect("Topics", sorted(curves["topic"].unique()), default=sorted(curves["topic"].unique())[:3],
                          format_func=lambda t: names.get(t, t), max_selections=8)
    fig = go.Figure()
    for i, t in enumerate(pick):
        d = curves[curves.topic == t]
        fig.add_trace(go.Scatter(x=d["period"], y=d["cumulative_adopters"], name=names.get(t, t),
                                 mode="lines", line=dict(color=SERIES[i], width=2)))
    fig.update_layout(yaxis_title="Users who have joined the topic", hovermode="x unified")
    plot(fig, 380)

    st.subheader("Influence maximisation (simulation)")
    model = st.radio("Diffusion model", ["IC", "LT"], horizontal=True,
                     format_func=lambda s: {"IC": "Independent Cascade", "LT": "Linear Threshold"}[s])
    d = im[im.model == model]
    order = ["CELF greedy", "Top ExpertiseRank", "Top out-degree", "Top betweenness", "Top PageRank", "Random"]
    fig = go.Figure()
    for i, s in enumerate(order):
        x = d[d.strategy == s].sort_values("k")
        fig.add_trace(go.Scatter(x=x["k"], y=x["spread_pct_of_users"], name=s, mode="lines+markers",
                                 line=dict(color=OTHER if s == "Random" else SERIES[i], width=2,
                                           dash="dash" if s == "Random" else "solid")))
    fig.update_layout(xaxis_title="Seed users", yaxis_title="% of users reached", hovermode="x unified")
    plot(fig, 400)
    seeds = json.loads((OUTPUT_DIR / "seed_sets.json").read_text())[model]["CELF greedy"]
    m = out("user_metrics.csv").set_index("user_id")
    st.markdown("**Best seed users (CELF):** " + ", ".join(str(m.at[s, "user_name"]) for s in seeds if s in m.index))

    st.subheader("Does the simulation match what really happened?")
    fig = px.scatter(valid, x="simulated_ic_spread", y="observed_reach", hover_name="user_name",
                     log_x=True, log_y=True, color_discrete_sequence=[SERIES[0]])
    fig.update_layout(xaxis_title="Simulated spread from user", yaxis_title="Observed time-respecting reach")
    plot(fig, 420)


@st.cache_resource(show_spinner="Training the answer-prediction model ...")
def answer_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from predict import CATEGORICAL, NUMERIC, TARGET, models
    df = processed("prediction_features.parquet")
    model = models()["Random Forest"]
    model.fit(df[NUMERIC + CATEGORICAL], df[TARGET].astype(int))
    return model, df


def page_prediction():
    st.title("🔮 Will this question get answered?")
    met, imp, roc = out("prediction_metrics.csv"), out("prediction_importance.csv"), out("prediction_roc.csv")
    need(met, imp, roc)

    st.subheader("Check your question")
    title = st.text_input("Title", "Segmentation fault in recover when reading the memory card")
    body = st.text_area("Question body", "I'm stuck on recover. My code compiles but I get a segmentation fault "
                        "when I run it. I use fread into a buffer of 512 bytes. What am I doing wrong?", height=120)
    tags = st.text_input("Tags (separated by spaces)", "pset4 recover segmentation-fault")
    has_code = st.checkbox("I include my code", value=True)
    if st.button("Predict", type="primary"):
        from predict import CATEGORICAL, NUMERIC
        from preprocess import CONFUSION_TERMS, ERROR_RE
        from topic_model import course_unit
        model, df = answer_model()
        tokenize, vader = text_tools()
        text = f"{title}. {body}"
        finder, q, vectors, mode, tfidf = finder_and_vectors()
        qt = processed("question_topics.parquet").set_index("post_id")
        rows, _ = finder.similar(embed_query(text, tfidf), n=15)
        topic = qt.loc[q.loc[rows, "post_id"], "topic"].mode().iloc[0]
        words = max(len(text.split()), 1)
        tag_str = "|".join(tags.split())
        topics = out("topics.csv").set_index("topic")
        recent = df.iloc[-200:]
        feat = pd.DataFrame([{
            "n_words": words, "title_words": len(title.split()), "n_code_lines": 20 if has_code else 0,
            "n_question_marks": text.count("?"), "n_tags": len(tags.split()),
            "sentiment": vader.polarity_scores(text)["compound"],
            "confusion_score": sum(text.lower().count(t) for t in CONFUSION_TERMS) / np.sqrt(words) * 10,
            "has_code": int(has_code), "has_error_msg": int(bool(ERROR_RE.search(text))),
            "has_check50_fail": int(":(" in text), "asker_prior_questions": 0, "asker_prior_answers": 0,
            "asker_is_new": 1, "asker_prior_answered_rate": -1,
            "questions_last_30d": recent["questions_last_30d"].median(),
            "helpers_last_30d": recent["helpers_last_30d"].median(),
            "topic_prior_answered_rate": 1 - topics.at[topic, "pct_unanswered"] / 100,
            "hour": datetime.now().hour, "weekday": datetime.now().weekday(),
            "topic": str(topic), "course_unit": course_unit(tag_str) or "Unknown",
        }])
        p = model.predict_proba(feat[NUMERIC + CATEGORICAL])[0, 1]
        c = st.columns(3)
        c[0].metric("Chance of a good answer", f"{p:.0%}")
        c[1].metric("Detected topic", topic_labels()[topic])
        c[2].metric("Course unit", course_unit(tag_str) or "Unknown")
        tips = []
        if not has_code:
            tips.append("Include the relevant part of your code: questions with code are answered more often.")
        if len(tags.split()) < 3:
            tips.append("Add more tags (problem set + concept), e.g. `pset4 recover malloc`.")
        if words < 40:
            tips.append("Describe what you expected, what happened, and what you already tried.")
        for t in tips:
            st.info(t)
        st.caption("Assumes a first-time asker posting now. Model: Random Forest trained on all questions.")

    st.subheader("Model evaluation on future questions")
    st.dataframe(met, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for i, mname in enumerate([m for m in met.model if not m.startswith("Baseline")]):
            d = roc[roc.model == mname]
            fig.add_trace(go.Scatter(x=d.fpr, y=d.tpr, name=mname, mode="lines", line=dict(color=SERIES[i], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", mode="lines", line=dict(color=OTHER, dash="dash")))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        plot(fig, 420)
    with right:
        d = imp.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=d.importance_mean, y=d.feature, orientation="h", marker_color=SERIES[0],
                               error_x=dict(type="data", array=d.importance_std, color="#52514e")))
        fig.update_layout(title="Feature importance (permutation)", xaxis_title="Drop in ROC-AUC")
        plot(fig, 420)


def page_experts():
    st.title("🧑‍🏫 Expert finder")
    finder, q, vectors, mode, tfidf = finder_and_vectors()
    st.caption(f"Matching by {mode}. Experts are scored on similar past questions they answered, "
               "how much they help overall and how recently they were active.")
    text = st.text_area("Describe your problem", "My hash table in speller leaks memory according to valgrind", height=100)
    if st.button("Find experts", type="primary") and text.strip():
        vec = embed_query(text, tfidf)
        ranking = finder.score(vec, method="hybrid").head(10)
        m = out("user_metrics.csv").set_index("user_id")
        names = topic_labels()
        experts = pd.DataFrame({"score": ranking.round(3)})
        experts = experts.join(m[["user_name", "role", "n_answers", "n_accepted", "main_help_topic"]], how="left")
        experts["main_help_topic"] = experts["main_help_topic"].map(names)
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Recommended experts")
            st.dataframe(experts.reset_index(drop=True), hide_index=True, width="stretch")
        with right:
            st.subheader("Similar past questions")
            rows, sims = finder.similar(vec, n=8)
            for r, s in zip(rows, sims):
                st.markdown(f"- [{q.at[r, 'title']}](https://cs50.stackexchange.com/q/{q.at[r, 'post_id']}) · {s:.2f}")

    ev = out("recommender_eval.csv")
    if ev is not None:
        st.subheader("How good are the recommendations? (future questions)")
        long = ev.melt(id_vars="method", value_vars=["hit@1", "hit@5", "hit@10"], var_name="metric", value_name="value")
        fig = px.bar(long, x="method", y="value", color="metric", barmode="group",
                     color_discrete_sequence=SERIES[:3])
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Real answerer in top k")
        plot(fig, 400)
        st.dataframe(ev, hide_index=True, width="stretch")


def sql_queries():
    text = (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=-- \d+\. )", text)
    out_ = {}
    for b in blocks:
        m = re.match(r"-- (\d+\. .+)\n", b)
        if m:
            out_[m.group(1)] = b[m.end():].split(";")[0].strip()
    return out_


def page_database():
    st.title("🗄️ PostgreSQL database")
    engine = get_db_engine()
    if engine is None:
        st.info("No PostgreSQL connection. The dashboard is using the CSV files.\n\n"
                "To connect: create the `learning_community` database in pgAdmin, fill in `.env`, then run "
                "`python src/load_to_postgres.py` and restart the dashboard.")
        return
    from sqlalchemy import inspect, text
    st.success("Connected to PostgreSQL")
    tables = inspect(engine).get_table_names()
    with engine.connect() as conn:
        counts = {t: conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar() for t in sorted(tables)}
    st.dataframe(pd.DataFrame({"table": counts.keys(), "rows": counts.values()}), hide_index=True)

    queries = sql_queries()
    name = st.selectbox("Ready-made analysis query", list(queries))
    sql = st.text_area("SQL (read-only)", queries[name], height=220)
    if st.button("Run query", type="primary"):
        if not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
            st.error("Only a single SELECT query is allowed here.")
        else:
            with engine.connect() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                st.dataframe(pd.read_sql(text(sql), conn), hide_index=True, width="stretch")


PAGES = {
    "Overview": page_overview,
    "Topics (NLP)": page_topics,
    "Knowledge network": page_network,
    "Knowledge diffusion": page_diffusion,
    "Answer prediction": page_prediction,
    "Expert finder": page_experts,
    "Database (SQL)": page_database,
}

with st.sidebar:
    st.header("Learning Community Analytics")
    # ?page=<name> in the URL opens that page directly (handy for links and screenshots).
    requested = st.query_params.get("page", "")
    names = list(PAGES)
    start = next((i for i, p in enumerate(names) if p.lower().startswith(requested.lower())), 0) if requested else 0
    page = st.radio("Page", names, index=start, label_visibility="collapsed")
    st.divider()
    st.caption(f"Data source: **{data_source()}**")
    st.caption("CS50 Stack Exchange · 2019–2026")

PAGES[page]()
