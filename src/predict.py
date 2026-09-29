"""
Step 5: predict whether a new question will get a good answer (Stack Exchange's `is_answered`).

Only information available at the moment a question is posted is used (no scores, views
or answer counts), and the evaluation is temporal: train on the oldest 80% of questions,
test on the newest 20%, the way the model would be used in practice.

Outputs:
    outputs/prediction_metrics.csv      model comparison on the held-out future questions
    outputs/prediction_cv.csv           5-fold cross-validation on the training period
    outputs/prediction_importance.csv   permutation importance of each feature
    outputs/prediction_roc.csv          ROC curve points for every model
    data/processed/prediction_features.parquet
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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
    "confusion_score", "has_code", "has_error_msg", "has_check50_fail",
    "asker_prior_questions", "asker_prior_answers", "asker_is_new", "asker_prior_answered_rate",
    "questions_last_30d", "helpers_last_30d", "topic_prior_answered_rate", "hour", "weekday",
]
CATEGORICAL = ["topic", "course_unit"]


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
    for col in ["has_code", "has_error_msg", "has_check50_fail"]:
        q[col] = q[col].astype(int)
    q["topic"] = q["topic"].astype(str)
    q["course_unit"] = q["course_unit"].fillna("Unknown")
    return q


def models():
    pre_scaled = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    pre_trees = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return {
        "Baseline (majority class)": Pipeline([("pre", pre_trees), ("clf", DummyClassifier())]),
        "Logistic Regression": Pipeline([("pre", pre_scaled), ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", pre_trees), ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_SEED))]),
        "XGBoost": Pipeline([("pre", pre_trees), ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1))]),
    }


def run():
    raw = load_raw()
    q = pd.read_parquet(PROCESSED_DIR / "questions_clean.parquet")
    qt = pd.read_parquet(PROCESSED_DIR / "question_topics.parquet")
    q = q.merge(qt[["post_id", "topic", "course_unit"]], on="post_id", how="inner")
    df = build_features(q, raw["answers"])
    df.to_parquet(PROCESSED_DIR / "prediction_features.parquet", index=False)

    cut = int(len(df) * TRAIN_FRACTION)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_train, y_train = train[NUMERIC + CATEGORICAL], train[TARGET].astype(int)
    X_test, y_test = test[NUMERIC + CATEGORICAL], test[TARGET].astype(int)
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
    imp = permutation_importance(fitted[best], X_test, y_test, scoring="roc_auc", n_repeats=20,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    importance = pd.DataFrame({"feature": NUMERIC + CATEGORICAL, "importance_mean": imp.importances_mean,
                               "importance_std": imp.importances_std, "model": best})
    importance.sort_values("importance_mean", ascending=False).round(5).to_csv(
        OUTPUT_DIR / "prediction_importance.csv", index=False)
    print(f"[predict] best model: {best}")
    return metrics


if __name__ == "__main__":
    run()
