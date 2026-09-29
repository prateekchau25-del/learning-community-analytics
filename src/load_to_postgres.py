"""
Load the raw CSV files (src/collect_data.py) and the analysis results (outputs/) into PostgreSQL.

Usage:
    python src/load_to_postgres.py --site cs50

Needs a .env file with your database details (copy .env.example).
The tables are recreated each run, so it is safe to run again.

Raw tables:     users, questions, answers, comments, interactions   (sql/schema.sql)
Result tables:  topics, question_topics, user_metrics, topic_trends, help_edges, and the
                model evaluation tables — created from the files in outputs/.
"""

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import PROJECT_ROOT, get_engine

SCHEMA_FILE = PROJECT_ROOT / "sql" / "schema.sql"
# Load order matters: users first, because other tables reference them.
TABLES = ["users", "questions", "answers", "comments", "interactions"]
ID_COLUMNS = ["user_id", "post_id", "question_id", "comment_id", "accepted_answer_id",
              "reply_to_user_id", "source_user", "target_user", "id", "parent_post_id",
              "user_reputation", "score", "view_count", "answer_count"]
PRIMARY_KEYS = {"users": "user_id", "questions": "post_id", "answers": "post_id", "comments": "comment_id"}

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# table name -> (file, primary key or None, extra SQL run after loading)
RESULT_TABLES = {
    "topics": (OUTPUT_DIR / "topics.csv", "topic", None),
    "question_topics": (PROCESSED_DIR / "question_topics.parquet", "post_id",
                        "ALTER TABLE question_topics ADD FOREIGN KEY (post_id) REFERENCES questions(post_id)"),
    "user_metrics": (OUTPUT_DIR / "user_metrics.csv", "user_id", None),
    "help_edges": (OUTPUT_DIR / "help_edges.csv", None, None),
    "topic_trends": (OUTPUT_DIR / "topic_trends.csv", None, None),
    "network_over_time": (OUTPUT_DIR / "network_over_time.csv", "year", None),
    "topic_model_comparison": (OUTPUT_DIR / "topic_model_comparison.csv", None, None),
    "diffusion_exposure_test": (OUTPUT_DIR / "diffusion_exposure_test.csv", "topic", None),
    "influence_maximization": (OUTPUT_DIR / "influence_maximization.csv", None, None),
    "prediction_metrics": (OUTPUT_DIR / "prediction_metrics.csv", "model", None),
    "recommender_eval": (OUTPUT_DIR / "recommender_eval.csv", "method", None),
}


def prepare_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the CSVs and convert types so they match sql/schema.sql."""
    tables = {}
    for name in TABLES:
        df = pd.read_csv(data_dir / f"{name}.csv")
        if name != "users":
            # User details live only in the users table.
            df = df.drop(columns=["user_name", "user_reputation"], errors="ignore")
        # IDs are read as floats when a column has blanks; make them whole numbers again.
        for col in df.columns.intersection(ID_COLUMNS):
            df[col] = df[col].astype("Int64")
        if "created_at" in df:
            df["created_at"] = pd.to_datetime(df["created_at"], unit="s")
        if name in PRIMARY_KEYS:
            # The API can return the same post twice while paging.
            df = df.drop_duplicates(PRIMARY_KEYS[name])
        tables[name] = df
    return tables


def load_results(conn):
    """Load analysis results when they exist (run src/run_pipeline.py to create them)."""
    present = {name: spec for name, spec in RESULT_TABLES.items() if spec[0].exists()}
    if not present:
        print("No analysis results found in outputs/ (run python src/run_pipeline.py first).")
        return
    print("Loading analysis results ...")
    for name, (path, key, extra) in present.items():
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        for col in df.columns.intersection(["first_at", "last_at", "first_seen", "last_seen"]):
            df[col] = pd.to_datetime(df[col])
        conn.execute(text(f"DROP TABLE IF EXISTS {name} CASCADE"))
        df.to_sql(name, conn, index=False, chunksize=1000)
        if key:
            conn.execute(text(f"ALTER TABLE {name} ADD PRIMARY KEY ({key})"))
        if extra:
            conn.execute(text(extra))
        print(f"  {name}: {len(df)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="cs50", help="Folder name under data/raw/")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "raw" / args.site
    if not data_dir.exists():
        raise SystemExit(f"No data in {data_dir}. Run: python src/collect_data.py --site {args.site}")

    tables = prepare_tables(data_dir)
    engine = get_engine()

    with engine.begin() as conn:
        print("Creating tables ...")
        conn.execute(text(SCHEMA_FILE.read_text()))
        for name in TABLES:
            tables[name].to_sql(name, conn, if_exists="append", index=False, chunksize=1000)
            print(f"  {name}: {len(tables[name])} rows")
        load_results(conn)

    print("\nDone. Open pgAdmin -> learning_community -> Schemas -> public -> Tables to see the data.")


if __name__ == "__main__":
    main()
