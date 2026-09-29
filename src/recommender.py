"""
Step 6: expert finder — recommend who can answer a new question.

For a new question we find similar past questions (by meaning) and score each helper by
how well they answered those questions, then blend in how active and how recently active
the helper is.

Evaluation (temporal): experts are learned from the oldest 80% of questions; for each newer
question we check whether a person who really answered it appears in the top-k list.
Metrics: Hit@k (share of questions where a real answerer is in the top k) and MRR.

Outputs: outputs/recommender_eval.csv
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import EMBEDDING_MODEL, OUTPUT_DIR, PROCESSED_DIR, TRAIN_FRACTION
from data_loader import load_raw

N_SIMILAR = 50          # similar past questions considered per query
ACCEPTED_BONUS = 1.0    # an accepted answer counts double
RECENCY_DAYS = 365      # activity half-life style decay
HYBRID_WEIGHTS = {"content": 0.6, "authority": 0.2, "recency": 0.2}
KS = [1, 5, 10]


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0


class ExpertFinder:
    """Content + authority + recency expert recommender over a set of past Q&A."""

    def __init__(self, questions: pd.DataFrame, answers: pd.DataFrame, vectors: np.ndarray):
        # questions: post_id, user_id, title, created_at (aligned row-by-row with vectors)
        self.questions = questions.reset_index(drop=True)
        self.vectors = normalize(vectors)
        a = answers.dropna(subset=["user_id"]).copy()
        a["user_id"] = a["user_id"].astype("int64")
        asker = self.questions.set_index("post_id")["user_id"]
        a = a[a["question_id"].isin(asker.index)]
        a = a[a["question_id"].map(asker) != a["user_id"]]          # ignore self-answers
        a["credit"] = 1 + ACCEPTED_BONUS * a["is_accepted"].astype(float)
        self.answers = a
        row_of = pd.Series(np.arange(len(self.questions)), index=self.questions["post_id"])
        self.answer_rows = row_of[a["question_id"]].to_numpy()
        self.n_answers = a.groupby("user_id").size()
        self.last_active = a.groupby("user_id")["created_at"].max()

    def similar(self, query_vec: np.ndarray, n=N_SIMILAR):
        sims = self.vectors @ normalize(query_vec.reshape(1, -1)).ravel()
        top = np.argsort(sims)[::-1][:n]
        return top, sims[top]

    def score(self, query_vec, at=None, method="hybrid", exclude_user=None) -> pd.Series:
        if method == "popularity":
            scores = self.n_answers.astype(float)
        else:
            rows, sims = self.similar(query_vec)
            sim_of_row = pd.Series(sims, index=rows)
            hit = np.isin(self.answer_rows, rows)
            part = self.answers[hit]
            content = (part["credit"].to_numpy() * sim_of_row[self.answer_rows[hit]].to_numpy())
            scores = pd.Series(content, index=part["user_id"].to_numpy()).groupby(level=0).sum()
            if method == "hybrid":
                users = self.n_answers.index
                content = scores.reindex(users, fill_value=0)
                authority = np.log1p(self.n_answers)
                at = at if at is not None else self.last_active.max()
                days = (at - self.last_active).dt.days.clip(lower=0)
                recency = np.exp(-days / RECENCY_DAYS)
                scores = (HYBRID_WEIGHTS["content"] * _minmax(content)
                          + HYBRID_WEIGHTS["authority"] * _minmax(authority)
                          + HYBRID_WEIGHTS["recency"] * _minmax(recency))
        if exclude_user is not None:
            scores = scores.drop(exclude_user, errors="ignore")
        return scores.sort_values(ascending=False)


def load_vectors(kind: str, questions: pd.DataFrame, n_train=None):
    """Embeddings from the topic step, or TF-IDF vectors (fitted on the first n_train rows)."""
    if kind == "embedding":
        emb = np.load(PROCESSED_DIR / "embeddings.npy")
        ids = np.load(PROCESSED_DIR / "embedding_post_ids.npy")
        idx = pd.Series(np.arange(len(ids)), index=ids)
        return emb[idx[questions["post_id"]].to_numpy()], None
    vec = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec.fit(questions["tokens"].iloc[:n_train])
    return vec.transform(questions["tokens"]).toarray().astype(np.float32), vec


def evaluate():
    raw = load_raw()
    q = pd.read_parquet(PROCESSED_DIR / "questions_clean.parquet")
    ids = np.load(PROCESSED_DIR / "embedding_post_ids.npy")
    q = q[q["post_id"].isin(ids)].sort_values("created_at").reset_index(drop=True)
    cut = int(len(q) * TRAIN_FRACTION)
    train, test = q.iloc[:cut], q.iloc[cut:]
    answers = raw["answers"].dropna(subset=["user_id"]).copy()
    answers["user_id"] = answers["user_id"].astype("int64")

    truth = answers[answers["question_id"].isin(test["post_id"])].groupby("question_id")["user_id"].agg(set)
    known = set(answers.loc[answers["question_id"].isin(train["post_id"]), "user_id"])
    test = test[test["post_id"].isin(truth.index)].copy()
    test["truth"] = test["post_id"].map(truth).map(lambda s: s & known)
    reachable = test["truth"].str.len() > 0
    print(f"[recommender] {len(test)} answered test questions, "
          f"{reachable.mean():.0%} answered by someone already active before the cut-off")
    test = test[reachable]

    rows = []
    for kind in ["tfidf", "embedding"]:
        vectors, _ = load_vectors(kind, q, n_train=cut)
        finder = ExpertFinder(train, answers, vectors[:cut])
        methods = ["popularity", "content", "hybrid"] if kind == "embedding" else ["content", "hybrid"]
        for method in methods:
            ranks = []
            for i, row in test.iterrows():
                asker = int(row.user_id) if pd.notna(row.user_id) else None
                ranking = finder.score(vectors[i], at=row.created_at, method=method, exclude_user=asker)
                pos = [k for k, u in enumerate(ranking.index[:100], start=1) if u in row.truth]
                ranks.append(pos[0] if pos else np.inf)
            ranks = np.array(ranks)
            name = "Popularity (most answers)" if method == "popularity" else f"{method.title()} ({kind})"
            rec = {"method": name, "n_test_questions": len(ranks)}
            rec.update({f"hit@{k}": float(np.mean(ranks <= k)) for k in KS})
            rec["mrr"] = float(np.mean(np.where(np.isinf(ranks), 0, 1 / ranks)))
            rows.append(rec)
            print(f"  {name:28s} Hit@5 {rec['hit@5']:.3f}  Hit@10 {rec['hit@10']:.3f}  MRR {rec['mrr']:.3f}")
    out = pd.DataFrame(rows).drop_duplicates("method").round(4)
    out.to_csv(OUTPUT_DIR / "recommender_eval.csv", index=False)
    return out


def run():
    return evaluate()


if __name__ == "__main__":
    run()
