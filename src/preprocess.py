"""
Step 1: clean the question text and build per-question features.

Output: data/processed/questions_clean.parquet
    text_clean   natural text (title + body without code blocks), for BERTopic and embeddings
    tokens       lower-cased lemmas without stopwords, for LDA and TF-IDF
    + structure, sentiment/confusion and outcome columns used by later steps
"""

import html
import re
import warnings

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from config import PROCESSED_DIR
from data_loader import load_raw

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")
ERROR_RE = re.compile(r"error|segmentation fault|traceback|exception|undefined reference|"
                      r"core dumped|valgrind|leak|warning:", re.I)
# check50 prints ":(" for each failed test.
CHECK50_FAIL_RE = re.compile(r":\(")

# Words that appear in almost every question and carry no topic information.
DOMAIN_STOPWORDS = {
    "cs50", "cs50x", "cs50p", "cs50w", "cs50ai", "pset", "problem", "set", "hi", "hello", "hey",
    "thanks", "thank", "please", "help", "question", "trying", "try", "tried", "get", "got", "know",
    "like", "want", "need", "would", "could", "work", "working", "works", "way", "anyone", "one",
    "also", "use", "using", "used", "make", "see", "seem", "seems", "im", "ive", "dont", "doesnt",
    "cant", "didnt", "isnt", "wont", "thats", "it", "code", "program", "run", "running", "fine",
    "able", "still", "time", "thing", "something", "anything", "everything", "think", "idea",
    "sure", "understand", "problem", "issue", "wrong", "right", "correct", "help", "appreciated",
    "advance", "much", "lot", "really", "just", "even", "though", "however", "first", "second",
    "new", "line", "lines", "every", "since", "keep", "keeps", "going", "look", "looks", "take",
    "said", "say", "says", "well", "good", "okay", "ok", "yes", "no", "without", "back", "made",
    "doe", "wa", "ha", "u", "le", "go", "come", "give", "gives", "given", "put", "let", "went",
    "getting", "following", "find", "found", "result", "correctly", "expected", "check", "error",
    "output", "print", "printing", "return", "returns", "function", "file", "value", "number",
    "actual", "instead", "part", "far", "done", "write", "wrote", "written", "change", "changed",
    "reason", "happen", "happens", "example", "may", "might", "must", "another", "anymore", "ever",
    "hope", "somebody", "someone", "maybe", "probably", "already", "yet", "able", "currently",
    "tell", "told", "stuck", "confused", "call", "called", "don", "doesn", "didn", "isn",
}

# Phrases that signal a confused or stuck learner.
CONFUSION_TERMS = [
    "confused", "confusing", "stuck", "don't understand", "dont understand", "do not understand",
    "not sure", "no idea", "lost", "struggling", "frustrated", "can't figure", "cant figure",
    "cannot figure", "doesn't work", "doesnt work", "not working", "what am i doing wrong",
    "what is wrong", "what's wrong", "why does", "why is", "hours", "days", "desperate", "hopeless",
]


def _nltk_resources():
    """Stopwords, lemmatizer and sentiment analyser, downloading NLTK data if needed."""
    import nltk
    for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4"), ("vader_lexicon", "sentiment/vader_lexicon")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words("english")), WordNetLemmatizer(), SentimentIntensityAnalyzer()


def split_html(body: str) -> tuple[str, int]:
    """Return (text without code blocks, number of code lines)."""
    soup = BeautifulSoup(body or "", "html.parser")
    code_lines = 0
    for pre in soup.find_all("pre"):
        code_lines += len([ln for ln in pre.get_text().splitlines() if ln.strip()])
        pre.decompose()
    text = soup.get_text(" ")
    text = URL_RE.sub(" ", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip(), code_lines


def clean_questions(questions: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    stop_words, lemmatizer, vader = _nltk_resources()
    stop_words |= DOMAIN_STOPWORDS

    df = questions.copy()
    df["title"] = df["title"].fillna("").map(html.unescape)
    parts = df["body"].fillna("").map(split_html)
    df["body_text"] = parts.str[0]
    df["n_code_lines"] = parts.str[1]
    df["text_clean"] = (df["title"] + ". " + df["body_text"]).str.strip()

    lemma_cache = {}

    def tokenize(text):
        out = []
        for tok in TOKEN_RE.findall(text.lower()):
            if tok not in lemma_cache:
                lemma_cache[tok] = lemmatizer.lemmatize(tok)
            lem = lemma_cache[tok]
            if lem not in stop_words and tok not in stop_words and len(lem) > 2:
                out.append(lem)
        return " ".join(out)

    df["tokens"] = df["text_clean"].map(tokenize)

    raw_text = df["body"].fillna("").map(lambda b: BeautifulSoup(b, "html.parser").get_text(" "))
    lower = df["text_clean"].str.lower()
    df["n_words"] = df["text_clean"].str.split().str.len()
    df["title_words"] = df["title"].str.split().str.len()
    df["has_code"] = df["n_code_lines"] > 0
    df["has_error_msg"] = raw_text.str.contains(ERROR_RE)
    df["has_check50_fail"] = raw_text.str.contains(CHECK50_FAIL_RE)
    df["n_question_marks"] = df["text_clean"].str.count(r"\?")
    df["n_tags"] = df["tags"].fillna("").str.split("|").map(lambda t: len([x for x in t if x]))
    df["sentiment"] = df["text_clean"].map(lambda t: vader.polarity_scores(t)["compound"])
    df["confusion_score"] = lower.map(lambda t: sum(t.count(term) for term in CONFUSION_TERMS))
    df["confusion_score"] = df["confusion_score"] / np.sqrt(df["n_words"].clip(lower=1)) * 10

    # Outcomes: first answer time and who gave it.
    first = (answers.sort_values("created_at")
             .groupby("question_id")
             .agg(first_answer_at=("created_at", "first"), first_answerer=("user_id", "first")))
    df = df.merge(first, left_on="post_id", right_index=True, how="left")
    df["hours_to_first_answer"] = (df["first_answer_at"] - df["created_at"]).dt.total_seconds() / 3600
    df["has_accepted"] = df["accepted_answer_id"].notna()
    df["is_answered"] = df["is_answered"].astype(bool)

    keep = ["post_id", "user_id", "created_at", "title", "tags", "text_clean", "tokens",
            "n_words", "title_words", "n_code_lines", "has_code", "has_error_msg", "has_check50_fail",
            "n_question_marks", "n_tags", "sentiment", "confusion_score", "score", "view_count",
            "answer_count", "is_answered", "has_accepted", "accepted_answer_id",
            "first_answer_at", "first_answerer", "hours_to_first_answer"]
    return df[keep].sort_values("created_at").reset_index(drop=True)


def run():
    raw = load_raw()
    df = clean_questions(raw["questions"], raw["answers"])
    df.to_parquet(PROCESSED_DIR / "questions_clean.parquet", index=False)
    print(f"[preprocess] {len(df)} questions cleaned "
          f"({df.has_code.mean():.0%} contain code, {df.has_error_msg.mean():.0%} mention an error)")
    return df


if __name__ == "__main__":
    run()
