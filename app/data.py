"""Cached data access for the dashboard. Every function takes the community (site)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import COMMUNITIES, COMPARISON_DIR, EMBEDDING_MODEL, site_dirs  # noqa: E402
from data_loader import data_source, get_db_engine, load_raw  # noqa: E402

__all__ = ["COMMUNITIES", "COMPARISON_DIR", "data_source", "get_db_engine", "available_sites", "raw", "out",
           "processed", "topic_labels", "comparison", "encoder", "embed", "expert_finder", "answer_model"]


def available_sites() -> list[str]:
    return [s for s in COMMUNITIES if (site_dirs(s)["outputs"] / "topics.csv").exists()]


@st.cache_data(show_spinner="Loading community data ...")
def raw(site: str):
    return load_raw(site)


@st.cache_data
def out(site: str, name: str):
    path = site_dirs(site)["outputs"] / name
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


@st.cache_data
def processed(site: str, name: str):
    path = site_dirs(site)["processed"] / name
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data
def comparison(name: str):
    path = COMPARISON_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


def topic_labels(site: str) -> dict:
    t = out(site, "topics.csv")
    return {} if t is None else dict(zip(t["topic"], t["label"]))


@st.cache_resource(show_spinner="Loading the language model ...")
def encoder():
    """Sentence-transformer if installed; None means the TF-IDF fallback is used."""
    try:
        from sentence_transformers import SentenceTransformer
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
def expert_finder(site: str):
    from recommender import ExpertFinder
    from sklearn.feature_extraction.text import TfidfVectorizer
    d = site_dirs(site)["processed"]
    q = pd.read_parquet(d / "questions_clean.parquet")
    ids = np.load(d / "embedding_post_ids.npy")
    q = q[q["post_id"].isin(ids)].reset_index(drop=True)
    tfidf = None
    if encoder() is not None:
        emb = np.load(d / "embeddings.npy")
        idx = pd.Series(np.arange(len(ids)), index=ids)
        vectors, mode = emb[idx[q["post_id"]].to_numpy()], "sentence embeddings"
    else:
        tfidf = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        vectors, mode = tfidf.fit_transform(q["tokens"]).toarray().astype(np.float32), "TF-IDF"
    return ExpertFinder(q, raw(site)["answers"], vectors), q, mode, tfidf


def embed(site: str, text: str):
    enc = encoder()
    if enc is not None:
        return enc.encode([text], normalize_embeddings=True)[0]
    tokenize, _ = text_tools()
    return expert_finder(site)[3].transform([tokenize(text)]).toarray()[0]


@st.cache_resource(show_spinner="Training the answer-prediction model ...")
def answer_model(site: str):
    """Train the best model from the evaluation (by ROC-AUC) on all questions of the community."""
    from predict import FEATURES, TARGET, models
    d = site_dirs(site)
    df = pd.read_parquet(d["processed"] / "prediction_features.parquet")
    emb = np.load(d["processed"] / "embeddings.npy")
    ids = np.load(d["processed"] / "embedding_post_ids.npy")
    df = df.join(pd.DataFrame(emb, index=ids, columns=[f"e{i}" for i in range(emb.shape[1])]), on="post_id")
    df = df.dropna(subset=["e0"])
    metrics = pd.read_csv(d["outputs"] / "prediction_metrics.csv")
    best = metrics.iloc[1:].sort_values("roc_auc").iloc[-1]["model"]
    model = models()[best]
    model.fit(df[FEATURES], df[TARGET].astype(int))
    return model, df, best
