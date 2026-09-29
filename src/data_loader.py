"""
Load the raw community tables from PostgreSQL when it is configured, and from
the CSV files otherwise, so the pipeline runs with or without a database.
"""

import os
from functools import lru_cache

import pandas as pd

from config import PROJECT_ROOT, RAW_DIR

RAW_TABLES = ["users", "questions", "answers", "comments", "interactions"]


@lru_cache(maxsize=1)
def get_db_engine():
    """Return a working SQLAlchemy engine, or None when no database is reachable."""
    if not (PROJECT_ROOT / ".env").exists():
        return None
    try:
        from sqlalchemy import text

        from db import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # no driver, wrong password, server down, ...
        if os.getenv("DEBUG_DB"):
            print(f"[data_loader] PostgreSQL not available: {exc}")
        return None


def data_source() -> str:
    return "PostgreSQL" if get_db_engine() is not None else "CSV files"


def _table_in_db(engine, name):
    from sqlalchemy import inspect
    return inspect(engine).has_table(name)


def load_table(name: str) -> pd.DataFrame:
    engine = get_db_engine()
    if engine is not None and _table_in_db(engine, name):
        df = pd.read_sql_table(name, engine)
        df = df.drop(columns=["interaction_id"], errors="ignore")
    else:
        df = pd.read_csv(RAW_DIR / f"{name}.csv")
        if "created_at" in df:
            df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
    for col in ("user_id", "source_user", "target_user", "reply_to_user_id", "accepted_answer_id"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


def load_raw() -> dict[str, pd.DataFrame]:
    return {name: load_table(name) for name in RAW_TABLES}


def help_interactions(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Interactions that pass knowledge from a helper to a learner: every answer, plus
    comments written by someone other than the person who asked the thread's question.
    (An asker replying "thanks, still broken" to a helper is not the asker helping.)
    """
    q, a = raw["questions"], raw["answers"]
    inter = raw["interactions"].dropna(subset=["source_user", "target_user"]).copy()
    post_to_question = pd.concat([
        pd.Series(q["post_id"].to_numpy(), index=q["post_id"]),
        pd.Series(a["question_id"].to_numpy(), index=a["post_id"]),
    ])
    asker = q.set_index("post_id")["user_id"]
    thread_asker = inter["parent_post_id"].map(post_to_question).map(asker)
    keep = (inter["type"] == "answer") | (inter["source_user"] != thread_asker)
    inter = inter[keep.fillna(True)]
    inter["source_user"] = inter["source_user"].astype("int64")
    inter["target_user"] = inter["target_user"].astype("int64")
    return inter
