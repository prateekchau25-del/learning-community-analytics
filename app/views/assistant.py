from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from data import COMMUNITIES, answer_model, embed, expert_finder, out, processed, text_tools, topic_labels

EXAMPLES = {
    "datascience": ("Validation loss increases while training loss decreases",
                    "I'm training a CNN in Keras on 20k images. After epoch 5 the validation loss keeps going up while "
                    "the training loss goes down. I already added dropout. What am I doing wrong?",
                    "deep-learning keras overfitting cnn"),
    "ai": ("Why does PPO training become unstable after many episodes?",
           "My PPO agent learns well for 2,000 episodes, then the reward collapses. I use the default clip range "
           "and learning rate. Is this a known problem and how can I fix it?",
           "reinforcement-learning proximal-policy-optimization deep-rl"),
}


def features(site, title, body, tags, has_code):
    from preprocess import CONFUSION_TERMS, ERROR_RE, MATH_RE
    from topic_model import tag_category
    model, df, best = answer_model(site)
    _, vader = text_tools()
    text = f"{title}. {body}"
    finder, q, _, _ = expert_finder(site)
    qt = processed(site, "question_topics.parquet").set_index("post_id")
    rows, _ = finder.similar(embed(site, text), n=15)
    topic = qt.loc[q.loc[rows, "post_id"], "topic"].mode().iloc[0]
    words = max(len(text.split()), 1)
    tag_str = "|".join(tags.split())
    topics = out(site, "topics.csv").set_index("topic")
    recent = df.iloc[-200:]
    feat = pd.DataFrame([{
        "n_words": words, "title_words": len(title.split()), "n_code_lines": 20 if has_code else 0,
        "n_question_marks": text.count("?"), "n_tags": len(tags.split()),
        "sentiment": vader.polarity_scores(text)["compound"],
        "confusion_score": sum(text.lower().count(t) for t in CONFUSION_TERMS) / np.sqrt(words) * 10,
        "has_code": int(has_code), "has_error_msg": int(bool(ERROR_RE.search(text))),
        "has_math": int(bool(MATH_RE.search(text))), "asker_prior_questions": 0, "asker_prior_answers": 0,
        "asker_is_new": 1, "asker_prior_answered_rate": -1,
        "questions_last_30d": recent["questions_last_30d"].median(),
        "helpers_last_30d": recent["helpers_last_30d"].median(),
        "topic_prior_answered_rate": 1 - topics.at[topic, "pct_unanswered"] / 100,
        "hour": datetime.now().hour, "weekday": datetime.now().weekday(),
        "topic": str(topic), "category": tag_category(tag_str) or "Unknown",
    }])
    # The meaning of the text enters through the same sentence embedding used in training.
    vec = embed(site, text)
    if len(vec) != 384:  # TF-IDF fallback when sentence-transformers is not installed: use a neutral embedding
        vec = df[[f"e{i}" for i in range(384)]].mean().to_numpy()
    feat = pd.concat([feat, pd.DataFrame([vec], columns=[f"e{i}" for i in range(384)])], axis=1)
    return model, feat, topic, tag_category(tag_str), best


def gauge(p, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number=dict(suffix="%", font=dict(size=44)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=0.35),
                   bgcolor="#2A2340", borderwidth=0, steps=[dict(range=[0, 100], color="#2A2340")])))
    fig.update_layout(margin=dict(l=20, r=20, t=10, b=0))
    return fig


def render(site: str):
    ui.page_header("Question assistant",
                   f"Write a question for the {COMMUNITIES[site]} community: see its chance of an answer, "
                   "who can answer it, and similar solved questions")
    ex = EXAMPLES[site]
    with st.form("ask"):
        title = st.text_input("Title", ex[0])
        body = st.text_area("Question", ex[1], height=120)
        c1, c2 = st.columns([3, 1])
        tags = c1.text_input("Tags (separated by spaces)", ex[2])
        has_code = c2.checkbox("I include code", value=False)
        go_ = st.form_submit_button("Analyse my question", type="primary", width="stretch")

    if go_:
        from predict import FEATURES
        model, feat, topic, category, best = features(site, title, body, tags, has_code)
        p = model.predict_proba(feat[FEATURES])[0, 1]
        names = topic_labels(site)
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown("## Chance of a good answer")
            ui.chart(gauge(p, ui.COMMUNITY_COLOR[site]), 240)
            ui.kpis([("Detected topic", names.get(topic, topic), ""), ("Category", category or "Unknown", "from your tags")])
            tips = []
            if not has_code:
                tips.append("Include a minimal code example: questions with code get answered more often.")
            if len(tags.split()) < 3:
                tips.append("Add 3–5 specific tags (method + library + task).")
            if len(body.split()) < 40:
                tips.append("Explain what you expected, what happened and what you already tried.")
            for t in tips:
                st.info(t)
        with right:
            st.markdown("## Recommended experts")
            finder, q, mode, _ = expert_finder(site)
            vec = embed(site, f"{title}. {body}")
            ranking = finder.score(vec, method="hybrid").head(8)
            m = out(site, "user_metrics.csv").set_index("user_id")
            experts = pd.DataFrame({"match": ranking.values}, index=ranking.index)
            experts = experts.join(m[["user_name", "role", "n_answers", "n_accepted"]], how="left")
            st.dataframe(experts.reset_index(drop=True), hide_index=True, width="stretch", column_config={
                "match": st.column_config.ProgressColumn("Match", format="%.2f", min_value=0, max_value=1),
                "user_name": "Expert", "role": "Role", "n_answers": "Answers", "n_accepted": "Accepted"})
            st.markdown("## Similar solved questions")
            rows, sims = finder.similar(vec, n=6)
            for r, s in zip(rows, sims):
                st.markdown(f"- [{q.at[r, 'title']}](https://{site}.stackexchange.com/q/{q.at[r, 'post_id']}) · similarity {s:.2f}")
            st.caption(f"Matching by {mode}. Chance estimated by {best} (the best model in the evaluation) "
                       "for a first-time asker posting now.")

    with st.expander("How good are these models? (evaluated on future questions)"):
        met, rec = out(site, "prediction_metrics.csv"), out(site, "recommender_eval.csv")
        st.markdown("**Answer prediction** (trained on the oldest 80% of questions, tested on the newest 20%)")
        st.dataframe(met, hide_index=True, width="stretch")
        st.markdown("**Expert finder** (Hit@k = a real answerer is in the top k)")
        st.dataframe(rec, hide_index=True, width="stretch")
