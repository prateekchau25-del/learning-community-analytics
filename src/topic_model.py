"""
Step 2: discover discussion topics with LDA (baseline) and BERTopic (transformer based).

Both models are scored with the same metrics so they can be compared fairly:
    NPMI coherence   do a topic's top words really occur together? (-1..1, higher is better)
    Topic diversity  share of unique words across all topics' top-25 words (0..1)
    NMI vs tags      agreement with the subject categories of the community's own tags (0..1)

Outputs:
    data/processed/<site>/question_topics.parquet   topic of every question (both models)
    data/processed/<site>/embeddings.npy           sentence embeddings, reused by the recommender
    outputs/<site>/topics.csv                      BERTopic topics + difficulty metrics
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
# 20 requested; one of them is the outlier topic (-1), so 19 real topics remain.
BERTOPIC_N_TOPICS = 20

# Tag -> subject category, used to validate topics against the communities' own tags.
# Categories are listed from most to least specific; a question with several tags
# gets the most specific category among them.
TAG_CATEGORIES = {
    "Generative AI & LLMs": ["large-language-models", "llm", "chatgpt", "gpt", "gpt-3", "gpt-4", "open-ai", "rag",
                             "generative-models", "generative-model", "image-generation", "prompt-engineering", "gan",
                             "generative-adversarial-networks", "diffusion-models", "stable-diffusion", "chat-bots",
                             "langchain", "llama", "fine-tuning", "text-generation", "variational-autoencoder"],
    "Natural Language Processing": ["nlp", "natural-language-processing", "text-classification", "word-embeddings",
                                    "embeddings", "bert", "word2vec", "machine-translation", "sentiment-analysis",
                                    "text-mining", "tokenization", "named-entity-recognition", "transformer",
                                    "attention", "information-retrieval", "topic-model", "language-model",
                                    "natural-language-understanding", "seq2seq"],
    "Computer Vision": ["computer-vision", "image-classification", "image-processing", "image-recognition",
                        "object-detection", "yolo", "image-preprocessing", "image-segmentation", "cnn",
                        "convolutional-neural-network", "convolutional-neural-networks", "opencv", "faster-rcnn",
                        "semantic-segmentation", "face-recognition"],
    "Reinforcement Learning": ["reinforcement-learning", "dqn", "deep-rl", "proximal-policy-optimization",
                               "q-learning", "policy-gradients", "stable-baselines", "markov-decision-process",
                               "reward-functions", "actor-critic-methods", "openai-gym", "gym",
                               "monte-carlo-methods", "multi-armed-bandits", "policies", "value-functions"],
    "Time Series & Sequences": ["time-series", "forecasting", "lstm", "long-short-term-memory", "arima", "sequence",
                                "rnn", "recurrent-neural-networks", "anomaly-detection", "sequence-modeling"],
    "Deep Learning": ["deep-learning", "neural-network", "neural-networks", "pytorch", "keras", "tensorflow",
                      "backpropagation", "activation-functions", "gradient-descent", "optimization", "loss-function",
                      "objective-functions", "gpu", "training", "batch-normalization", "dropout", "autoencoder",
                      "weights-initialization", "learning-rate", "architecture"],
    "Classical ML & Modelling": ["machine-learning", "machine-learning-model", "classification", "regression",
                                 "clustering", "decision-trees", "random-forest", "xgboost", "svm",
                                 "logistic-regression", "linear-regression", "feature-selection",
                                 "feature-engineering", "hyperparameter-tuning", "hyperparameter-optimization",
                                 "cross-validation", "class-imbalance", "overfitting", "model-evaluations", "metric",
                                 "predictive-modeling", "ensemble-modeling", "k-means", "scikit-learn", "data-leakage",
                                 "data-science-model", "supervised-learning", "unsupervised-learning", "boosting"],
    "Data Wrangling & Tools": ["python", "pandas", "r", "numpy", "dataset", "data", "data-cleaning", "preprocessing",
                               "visualization", "data-analysis", "sql", "matplotlib", "scipy", "excel", "dataframe",
                               "feature-scaling", "normalization", "jupyter", "logi-symphony", "data-mining",
                               "training-datasets", "datasets", "tools", "software-development", "bigdata"],
    "Statistics & Theory": ["statistics", "probability", "correlation", "sampling", "math", "mathematics",
                            "computational-learning-theory", "terminology", "definitions", "hypothesis-testing",
                            "distribution", "bayesian", "linear-algebra", "proofs", "theory"],
    "AI Philosophy & Ethics": ["agi", "artificial-consciousness", "ethics", "ai-safety", "philosophy", "social",
                               "reasoning", "history", "turing-test", "ai-design", "explainable-ai", "risk-management",
                               "legal", "neuroscience", "human-like"],
}
CATEGORY_OF_TAG = {tag: cat for cat, tags in TAG_CATEGORIES.items() for tag in tags}
CATEGORY_ORDER = list(TAG_CATEGORIES)


def tag_category(tags: str) -> str | None:
    """Map a question's tags to its most specific subject category."""
    cats = {CATEGORY_OF_TAG[t] for t in str(tags).split("|") if t in CATEGORY_OF_TAG}
    return min(cats, key=CATEGORY_ORDER.index) if cats else None


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


def ctfidf_vectorizer():
    # BERTopic fits this on one merged document per topic, so min_df counts topics, not
    # questions: keep it at 1 and filter to the shared reference vocabulary afterwards.
    return CountVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=1)


def fit_bertopic(tokens: list[str], embeddings: np.ndarray):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=RANDOM_SEED)
    # Scale the smallest topic with the corpus: about 30 questions per 4,000.
    min_size = int(np.clip(len(tokens) / 130, 10, 60))
    hdbscan_model = HDBSCAN(min_cluster_size=min_size, min_samples=10, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    model = BERTopic(
        embedding_model=None,  # embeddings are precomputed
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=ctfidf_vectorizer(),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        top_n_words=100,
        calculate_probabilities=False,
    )
    # Topic words come from the cleaned tokens; clustering uses the full-sentence embeddings.
    topics, _ = model.fit_transform(tokens, embeddings)
    outlier_share = float(np.mean(np.array(topics) == -1))
    # Merge near-duplicate clusters (e.g. several 'recover' clusters) to a size comparable with LDA.
    if len(set(topics)) > BERTOPIC_N_TOPICS:
        model.reduce_topics(tokens, nr_topics=BERTOPIC_N_TOPICS)
    # Give outlier questions the topic whose centre is closest in embedding space.
    topics = model.reduce_outliers(tokens, model.topics_, strategy="embeddings", embeddings=embeddings)
    model.update_topics(tokens, topics=topics, vectorizer_model=ctfidf_vectorizer(),
                        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True), top_n_words=100)
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
    q["category"] = q["tags"].map(tag_category)
    print(f"[topics] {len(q)} questions, {q.category.notna().mean():.0%} mapped to a tag category")

    # ---- LDA baseline
    selection, lda_topics, doc_topic, doc_term, vocab = fit_lda(q["tokens"], q["category"])
    q["lda_topic"] = doc_topic.argmax(1)
    q["lda_prob"] = doc_topic.max(1)
    selection.to_csv(OUTPUT_DIR / "lda_model_selection.csv", index=False)
    lda_topics.to_csv(OUTPUT_DIR / "lda_topics.csv", index=False)

    # ---- BERTopic
    embeddings = compute_embeddings(q["text_clean"].tolist(), q["post_id"].to_numpy())
    model, topics, outlier_share = fit_bertopic(q["tokens"].tolist(), embeddings)
    q["topic"] = topics
    topic_ids = sorted(t for t in set(topics) if t != -1)
    # Keep words from the shared reference vocabulary (>= 5 questions) so both models are scored alike.
    top_words = {t: [w for w, _ in model.get_topic(t) if w in vocab][:TOP_N_DIVERSITY] for t in topic_ids}

    # ---- labels: dominant tag category + top words
    info = pd.DataFrame({"topic": topic_ids})
    info["top_words"] = info["topic"].map(lambda t: ", ".join(top_words[t][:10]))
    dominant = (q.dropna(subset=["category"]).groupby("topic")["category"]
                .agg(lambda s: s.value_counts().index[0]))
    purity = (q.dropna(subset=["category"]).groupby("topic")["category"]
              .agg(lambda s: s.value_counts(normalize=True).iloc[0]))
    info["category"] = info["topic"].map(dominant)
    info["category_purity"] = info["topic"].map(purity).round(3)
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
         "diversity": best_lda["diversity"], "nmi_vs_tags": best_lda["nmi_vs_tags"],
         "outlier_share_before_reduction": 0.0},
        {"model": "BERTopic (MiniLM + UMAP + HDBSCAN)", "n_topics": len(topic_ids),
         "npmi": float(info["npmi"].mean()), "diversity": topic_diversity(list(top_words.values())),
         "nmi_vs_tags": tag_nmi(q["topic"], q["category"]),
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

    q[["post_id", "topic", "topic_label", "lda_topic", "lda_prob", "category"]].to_parquet(
        PROCESSED_DIR / "question_topics.parquet", index=False)

    print(comparison.to_string(index=False))
    return info


if __name__ == "__main__":
    run()
