"""
Step 2: discover discussion topics with LDA (baseline) and BERTopic (transformer based).

Both models are scored with the same metrics so they can be compared fairly:
    NPMI coherence   do a topic's top words really occur together? (-1..1, higher is better)
    Topic diversity  share of unique words across all topics' top-25 words (0..1)
    NMI vs tags      agreement with the course's own problem-set tags (0..1)

Outputs:
    data/processed/question_topics.parquet   topic of every question (both models)
    data/processed/embeddings.npy            sentence embeddings, reused by the recommender
    outputs/topics.csv                       BERTopic topics + difficulty metrics
    outputs/lda_topics.csv, outputs/lda_model_selection.csv
    outputs/topic_model_comparison.csv, outputs/topic_trends.csv, outputs/topic_map.csv
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score

from config import EMBEDDING_MODEL, OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED

warnings.filterwarnings("ignore", category=FutureWarning)

LDA_K_GRID = [6, 8, 10, 12, 14, 16, 18, 20]
TOP_N_COHERENCE = 10
TOP_N_DIVERSITY = 25

# Course-structure tags used to validate topics against the CS50 syllabus.
COURSE_TAGS = {
    "pset1": "Week 1 (C basics)", "mario": "Week 1 (C basics)", "cash": "Week 1 (C basics)",
    "credit": "Week 1 (C basics)", "pset2": "Week 2 (Arrays)", "caesar": "Week 2 (Arrays)",
    "readability": "Week 2 (Arrays)", "substitution": "Week 2 (Arrays)", "vigenere": "Week 2 (Arrays)",
    "scrabble": "Week 2 (Arrays)", "pset3": "Week 3 (Algorithms)", "plurality": "Week 3 (Algorithms)",
    "runoff": "Week 3 (Algorithms)", "tideman": "Week 3 (Algorithms)", "sort": "Week 3 (Algorithms)",
    "pset4": "Week 4 (Memory)", "recover": "Week 4 (Memory)", "filter": "Week 4 (Memory)",
    "volume": "Week 4 (Memory)", "blur": "Week 4 (Memory)", "pset4recover": "Week 4 (Memory)",
    "pset4filter": "Week 4 (Memory)", "pset5": "Week 5 (Data structures)",
    "speller": "Week 5 (Data structures)", "inheritance": "Week 5 (Data structures)",
    "pset6": "Week 6 (Python)", "dna": "Week 6 (Python)", "python": "Week 6 (Python)",
    "pset7": "Week 7 (SQL)", "sql": "Week 7 (SQL)", "movies": "Week 7 (SQL)", "fiftyville": "Week 7 (SQL)",
    "songs": "Week 7 (SQL)", "pset8": "Week 8-9 (Web / Flask)", "pset9": "Week 8-9 (Web / Flask)",
    "finance": "Week 8-9 (Web / Flask)", "flask": "Week 8-9 (Web / Flask)", "html": "Week 8-9 (Web / Flask)",
    "homepage": "Week 8-9 (Web / Flask)", "birthdays": "Week 8-9 (Web / Flask)",
    "javascript": "Week 8-9 (Web / Flask)", "cs50p": "CS50P (Python course)",
    "cs50w": "CS50W (Web course)", "cs50ai": "CS50AI (AI course)", "final-project": "Final project",
    "check50": "Tools (check50 / IDE)", "cs50-ide": "Tools (check50 / IDE)",
    "codespaces": "Tools (check50 / IDE)", "submit50": "Tools (check50 / IDE)",
}


def course_unit(tags: str) -> str | None:
    """Map a question's tags to its course unit (specific tags win over generic ones)."""
    units = [COURSE_TAGS[t] for t in str(tags).split("|") if t in COURSE_TAGS]
    specific = [u for u in units if not u.startswith("Tools")]
    return (specific or units or [None])[0]


# --------------------------------------------------------------------------- metrics

def npmi_coherence(topic_words: list[list[str]], doc_term: sparse.csr_matrix, vocab: dict) -> list[float]:
    """Average normalised PMI of all word pairs, using document co-occurrence."""
    binary = (doc_term > 0).astype(np.float32).tocsc()
    n_docs = binary.shape[0]
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()
    scores = []
    for words in topic_words:
        idx = [vocab[w] for w in words if w in vocab]
        pair_scores = []
        for i, j in itertools.combinations(idx, 2):
            co = binary[:, i].multiply(binary[:, j]).sum()
            if co == 0:
                pair_scores.append(-1.0)
                continue
            p_ij = co / n_docs
            pmi = np.log(p_ij / ((doc_freq[i] / n_docs) * (doc_freq[j] / n_docs)))
            pair_scores.append(pmi / -np.log(p_ij))
        scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
    return scores


def topic_diversity(topic_words: list[list[str]]) -> float:
    words = [w for topic in topic_words for w in topic[:TOP_N_DIVERSITY]]
    return len(set(words)) / max(len(words), 1)


def tag_nmi(topics: pd.Series, units: pd.Series) -> float:
    mask = units.notna() & (topics != -1)
    return float(normalized_mutual_info_score(units[mask], topics[mask]))


# --------------------------------------------------------------------------- LDA

def make_vectorizer(min_df=5):
    return CountVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=min_df,
                           max_df=0.4, max_features=6000)


def fit_lda(tokens: pd.Series, units: pd.Series):
    vectorizer = make_vectorizer()
    doc_term = vectorizer.fit_transform(tokens)
    vocab = vectorizer.vocabulary_
    words = vectorizer.get_feature_names_out()

    rows, best = [], None
    for k in LDA_K_GRID:
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=40,
                                        random_state=RANDOM_SEED, n_jobs=-1)
        doc_topic = lda.fit_transform(doc_term)
        top_words = [[words[i] for i in comp.argsort()[::-1][:TOP_N_DIVERSITY]] for comp in lda.components_]
        npmi = float(np.mean(npmi_coherence([w[:TOP_N_COHERENCE] for w in top_words], doc_term, vocab)))
        row = {"k": k, "npmi": npmi, "diversity": topic_diversity(top_words),
               "perplexity": float(lda.perplexity(doc_term)),
               "nmi_vs_tags": tag_nmi(pd.Series(doc_topic.argmax(1)), units)}
        rows.append(row)
        print(f"  LDA k={k:2d}  NPMI={npmi:.3f}  diversity={row['diversity']:.2f}")
        # Select on coherence x diversity so we don't pick many near-duplicate topics.
        score = npmi * row["diversity"]
        if best is None or score > best[0]:
            best = (score, k, lda, doc_topic, top_words)

    _, k, lda, doc_topic, top_words = best
    selection = pd.DataFrame(rows)
    topics = pd.DataFrame({
        "lda_topic": range(k),
        "top_words": [", ".join(w[:10]) for w in top_words],
        "npmi": npmi_coherence([w[:TOP_N_COHERENCE] for w in top_words], doc_term, vocab),
        "n_questions": np.bincount(doc_topic.argmax(1), minlength=k),
    })
    return selection, topics, doc_topic, doc_term, vocab


# --------------------------------------------------------------------------- BERTopic

def compute_embeddings(texts: list[str], post_ids: np.ndarray) -> np.ndarray:
    """Sentence embeddings, cached on disk and reused when the questions are unchanged."""
    path, ids_path = PROCESSED_DIR / "embeddings.npy", PROCESSED_DIR / "embedding_post_ids.npy"
    if path.exists() and ids_path.exists() and np.array_equal(np.load(ids_path), post_ids):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(path, emb.astype(np.float32))
    np.save(ids_path, post_ids)
    return emb


def fit_bertopic(docs: list[str], tokens: list[str], embeddings: np.ndarray):
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=RANDOM_SEED)
    hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=10, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    model = BERTopic(
        embedding_model=SentenceTransformer(EMBEDDING_MODEL),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=make_vectorizer(min_df=3),
        representation_model=MaximalMarginalRelevance(diversity=0.3),
        top_n_words=TOP_N_DIVERSITY,
        calculate_probabilities=False,
    )
    # Topic words come from the cleaned tokens; clustering uses the full-sentence embeddings.
    topics, _ = model.fit_transform(tokens, embeddings)
    outlier_share = float(np.mean(np.array(topics) == -1))
    # Give outlier questions the topic whose centre is closest in embedding space.
    topics = model.reduce_outliers(tokens, topics, strategy="embeddings", embeddings=embeddings)
    model.update_topics(tokens, topics=topics, vectorizer_model=make_vectorizer(min_df=3),
                        representation_model=MaximalMarginalRelevance(diversity=0.3),
                        top_n_words=TOP_N_DIVERSITY)
    return model, np.array(topics), outlier_share


# --------------------------------------------------------------------------- analysis

def topic_difficulty(q: pd.DataFrame) -> pd.DataFrame:
    """Per-topic signals of how hard learners find it."""
    g = q.groupby("topic")
    out = pd.DataFrame({
        "n_questions": g.size(),
        "pct_unanswered": 100 * (1 - g["is_answered"].mean()),
        "pct_accepted": 100 * g["has_accepted"].mean(),
        "median_hours_to_answer": g["hours_to_first_answer"].median(),
        "avg_confusion": g["confusion_score"].mean(),
        "avg_sentiment": g["sentiment"].mean(),
        "pct_with_error": 100 * g["has_error_msg"].mean(),
        "avg_views": g["view_count"].mean(),
    })
    z = lambda s: (s - s.mean()) / (s.std() or 1)
    out["difficulty_index"] = (z(out["pct_unanswered"]) + z(np.log1p(out["median_hours_to_answer"]))
                               + z(out["avg_confusion"])) / 3
    return out.round(3)


def run():
    q = pd.read_parquet(PROCESSED_DIR / "questions_clean.parquet")
    q = q[q["tokens"].str.split().str.len() >= 3].reset_index(drop=True)
    q["course_unit"] = q["tags"].map(course_unit)
    print(f"[topics] {len(q)} questions, {q.course_unit.notna().mean():.0%} mapped to a course unit")

    # ---- LDA baseline
    selection, lda_topics, doc_topic, doc_term, vocab = fit_lda(q["tokens"], q["course_unit"])
    q["lda_topic"] = doc_topic.argmax(1)
    q["lda_prob"] = doc_topic.max(1)
    selection.to_csv(OUTPUT_DIR / "lda_model_selection.csv", index=False)
    lda_topics.to_csv(OUTPUT_DIR / "lda_topics.csv", index=False)

    # ---- BERTopic
    embeddings = compute_embeddings(q["text_clean"].tolist(), q["post_id"].to_numpy())
    model, topics, outlier_share = fit_bertopic(q["text_clean"].tolist(), q["tokens"].tolist(), embeddings)
    q["topic"] = topics
    topic_ids = sorted(t for t in set(topics) if t != -1)
    top_words = {t: [w for w, _ in model.get_topic(t)][:TOP_N_DIVERSITY] for t in topic_ids}

    # ---- labels: dominant course unit + top words
    info = pd.DataFrame({"topic": topic_ids})
    info["top_words"] = info["topic"].map(lambda t: ", ".join(top_words[t][:10]))
    dominant = (q.dropna(subset=["course_unit"]).groupby("topic")["course_unit"]
                .agg(lambda s: s.value_counts().index[0]))
    purity = (q.dropna(subset=["course_unit"]).groupby("topic")["course_unit"]
              .agg(lambda s: s.value_counts(normalize=True).iloc[0]))
    info["course_unit"] = info["topic"].map(dominant)
    info["unit_purity"] = info["topic"].map(purity).round(3)
    info["label"] = info["topic"].map(lambda t: " / ".join(top_words[t][:3]))
    info["npmi"] = npmi_coherence([top_words[t][:TOP_N_COHERENCE] for t in topic_ids], doc_term, vocab)
    info = info.merge(topic_difficulty(q), left_on="topic", right_index=True)
    info = info.sort_values("n_questions", ascending=False)
    info.to_csv(OUTPUT_DIR / "topics.csv", index=False)
    q["topic_label"] = q["topic"].map(info.set_index("topic")["label"])

    # ---- model comparison
    best_lda = selection.loc[selection["k"] == len(lda_topics)].iloc[0]
    comparison = pd.DataFrame([
        {"model": f"LDA (k={len(lda_topics)})", "n_topics": len(lda_topics), "npmi": best_lda["npmi"],
         "diversity": best_lda["diversity"], "nmi_vs_course_tags": best_lda["nmi_vs_tags"],
         "outlier_share_before_reduction": 0.0},
        {"model": "BERTopic (MiniLM + UMAP + HDBSCAN)", "n_topics": len(topic_ids),
         "npmi": float(info["npmi"].mean()), "diversity": topic_diversity(list(top_words.values())),
         "nmi_vs_course_tags": tag_nmi(q["topic"], q["course_unit"]),
         "outlier_share_before_reduction": outlier_share},
    ]).round(4)
    comparison.to_csv(OUTPUT_DIR / "topic_model_comparison.csv", index=False)

    # ---- topic trends per quarter
    q["period"] = q["created_at"].dt.to_period("Q").astype(str)
    trends = q.groupby(["period", "topic"]).size().rename("n_questions").reset_index()
    trends["share"] = trends["n_questions"] / trends.groupby("period")["n_questions"].transform("sum")
    trends.to_csv(OUTPUT_DIR / "topic_trends.csv", index=False)

    # ---- 2-D map of all questions for the dashboard
    from umap import UMAP
    xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine",
              random_state=RANDOM_SEED).fit_transform(embeddings)
    pd.DataFrame({"post_id": q["post_id"], "x": xy[:, 0], "y": xy[:, 1], "topic": q["topic"],
                  "title": q["title"]}).to_csv(OUTPUT_DIR / "topic_map.csv", index=False)

    q[["post_id", "topic", "topic_label", "lda_topic", "lda_prob", "course_unit"]].to_parquet(
        PROCESSED_DIR / "question_topics.parquet", index=False)

    print(comparison.to_string(index=False))
    return info


if __name__ == "__main__":
    run()
