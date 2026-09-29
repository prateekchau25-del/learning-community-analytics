"""
Step 5: predict whether a new question will get a good answer (Stack Exchange's `is_answered`).

Only information available at the moment a question is posted is used (no scores, views
or answer counts). Besides structural and history features, the meaning of the question text
enters through its sentence embedding (compressed with PCA). The evaluation is temporal: train on the oldest 80% of questions,
test on the newest 20%, the way the model would be used in practice.

Outputs:
    outputs/prediction_metrics.csv      model comparison on the held-out future questions
    outputs/prediction_cv.csv           5-fold cross-validation on the training period
    outputs/prediction_importance.csv   permutation importance of each feature (the embedding counts as one)
    outputs/prediction_roc.csv          ROC curve points for every model
    data/processed/prediction_features.parquet
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import OUTPUT_DIR, PROCESSED_DIR, RANDOM_SEED, TRAIN_FRACTION
from data_loader import load_raw

TARGET = "is_answered"
NUMERIC = [
    "n_words", "title_words", "n_code_lines", "n_question_marks", "n_tags", "sentiment",
    "confusion_score", "has_code", "has_error_msg", "has_math",
    "asker_prior_questions", "asker_prior_answers", "asker_is_new", "asker_prior_answered_rate",
    "questions_last_30d", "helpers_last_30d", "topic_prior_answered_rate", "hour", "weekday",
]
CATEGORICAL = ["topic", "category"]
EMBEDDING = [f"e{i}" for i in range(384)]   # all-MiniLM-L6-v2 sentence embedding of title + body
FEATURES = NUMERIC + CATEGORICAL + EMBEDDING


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each question's sentence embedding (computed in the topic-modeling step)."""
    emb = np.load(PROCESSED_DIR / "embeddings.npy")
    ids = np.load(PROCESSED_DIR / "embedding_post_ids.npy")
    e = pd.DataFrame(emb, index=ids, columns=EMBEDDING)
    return df.join(e, on="post_id").dropna(subset=EMBEDDING[:1])


def count_before(times_by_user: dict, users, at) -> np.ndarray:
    """For each (user, time): how many of the user's events happened strictly before time."""
    out = np.zeros(len(users), dtype=int)
    for i, (u, t) in enumerate(zip(users, at)):
        arr = times_by_user.get(u)
        if arr is not None:
            out[i] = np.searchsorted(arr, t, side="left")
    return out


def build_features(q: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    q = q.sort_values("created_at").reset_index(drop=True).copy()
    t = q["created_at"].astype("int64").to_numpy()
    user = q["user_id"].fillna(-1).astype("int64").to_numpy()

    # Asker's history before this question.
    q["asker_prior_questions"] = q.groupby(user).cumcount()
    q.loc[user == -1, "asker_prior_questions"] = 0
    q["asker_is_new"] = (q["asker_prior_questions"] == 0).astype(int)
    ans = answers.dropna(subset=["user_id"]).sort_values("created_at")
    ans_times = {u: g["created_at"].astype("int64").to_numpy()
                 for u, g in ans.groupby(ans["user_id"].astype("int64"))}
    q["asker_prior_answers"] = count_before(ans_times, user, t)
    prior_rate = q.groupby(user)[TARGET].transform(lambda s: s.shift().expanding().mean())
    q["asker_prior_answered_rate"] = prior_rate.fillna(-1)  # -1 = no earlier question

    # How busy the community was in the 30 days before the question.
    month = np.int64(30 * 86400 * 10**9)
    q["questions_last_30d"] = np.searchsorted(t, t, side="left") - np.searchsorted(t, t - month, side="left")
    a_t = ans["created_at"].astype("int64").to_numpy()
    a_u = ans["user_id"].astype("int64").to_numpy()
    lo, hi = np.searchsorted(a_t, t - month), np.searchsorted(a_t, t)
    q["helpers_last_30d"] = [len(np.unique(a_u[i:j])) for i, j in zip(lo, hi)]

    # How often this topic's earlier questions got answered.
    q["topic_prior_answered_rate"] = (q.groupby("topic")[TARGET]
                                      .transform(lambda s: s.shift().expanding().mean()))
    q["topic_prior_answered_rate"] = q["topic_prior_answered_rate"].fillna(q[TARGET].mean())

    q["hour"] = q["created_at"].dt.hour
    q["weekday"] = q["created_at"].dt.weekday
    for col in ["has_code", "has_error_msg", "has_math"]:
        q[col] = q[col].astype(int)
    q["topic"] = q["topic"].astype(str)
    q["category"] = q["category"].fillna("Unknown")
    return q


def models():
    """The embedding is compressed with PCA inside each pipeline, so it is fitted on training data only."""
    def pre(scale, n_components):
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("text", PCA(n_components, random_state=RANDOM_SEED), EMBEDDING)], sparse_threshold=0)
    return {
        "Baseline (majority class)": Pipeline([("pre", pre(False, 2)), ("clf", DummyClassifier())]),
        "Logistic Regression": Pipeline([("pre", pre(True, 64)), ("clf", LogisticRegression(
            max_iter=3000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", pre(False, 32)), ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_SEED))]),
        "XGBoost": Pipeline([("pre", pre(False, 32)), ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1))]),
    }


def grouped_permutation_importance(model, X, y, groups: dict, repeats=10):
    """Drop in ROC-AUC when a group of columns is shuffled together (the 384 embedding columns form one group)."""
    rng = np.random.default_rng(RANDOM_SEED)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": name, "importance_mean": float(np.mean(drops)), "importance_std": float(np.std(drops))})
    return pd.DataFrame(rows)


def run():
    raw = load_raw()
    q = pd.read_parquet(PROCESSED_DIR / "questions_clean.parquet")
    qt = pd.read_parquet(PROCESSED_DIR / "question_topics.parquet")
    q = q.merge(qt[["post_id", "topic", "category"]], on="post_id", how="inner")
    df = build_features(q, raw["answers"])
    df.to_parquet(PROCESSED_DIR / "prediction_features.parquet", index=False)
    df = add_embeddings(df)

    cut = int(len(df) * TRAIN_FRACTION)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_train, y_train = train[FEATURES], train[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)
    print(f"[predict] train {len(train)} questions (until {train.created_at.max():%Y-%m}), "
          f"test {len(test)} (answered rate {y_test.mean():.0%})")

    rows, cv_rows, roc_rows, fitted = [], [], [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for name, model in models().items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_rows.append({"model": name, "cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std()})
        model.fit(X_train, y_train)
        fitted[name] = model
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, prob),
            "pr_auc": average_precision_score(y_test, prob),
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        })
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_rows += [{"model": name, "fpr": a, "tpr": b} for a, b in zip(fpr, tpr)]
        print(f"  {name:28s} test ROC-AUC {rows[-1]['roc_auc']:.3f}   CV {cv_auc.mean():.3f}")

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(OUTPUT_DIR / "prediction_metrics.csv", index=False)
    pd.DataFrame(cv_rows).round(4).to_csv(OUTPUT_DIR / "prediction_cv.csv", index=False)
    pd.DataFrame(roc_rows).round(4).to_csv(OUTPUT_DIR / "prediction_roc.csv", index=False)

    best = metrics.iloc[1:].sort_values("roc_auc", ascending=False).iloc[0]["model"]
    groups = {c: [c] for c in NUMERIC + CATEGORICAL}
    groups["text meaning (embedding)"] = EMBEDDING
    importance = grouped_permutation_importance(fitted[best], X_test, y_test, groups).assign(model=best)
    importance.sort_values("importance_mean", ascending=False).round(5).to_csv(
        OUTPUT_DIR / "prediction_importance.csv", index=False)
    print(f"[predict] best model: {best}")
    return metrics


if __name__ == "__main__":
    run()
