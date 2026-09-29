"""
Load the raw data and the analysis results of every community into PostgreSQL.

Usage:
    python src/load_to_postgres.py                 # every community found in data/raw/
    python src/load_to_postgres.py --site ai       # one community

Needs a .env file with your database details (copy .env.example).
The tables are recreated each run, so it is safe to run again.

Each community gets its own schema (datascience, ai) with:
    raw tables     users, questions, answers, comments, interactions   (sql/schema.sql)
    result tables  topics, question_topics, user_metrics, help_edges, topic_trends, ...
The comparison between communities goes to the schema "comparison".
"""

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import PROJECT_ROOT, get_engine

SCHEMA_FILE = PROJECT_ROOT / "sql" / "schema.sql"
SITES = ["datascience", "ai"]
# Load order matters: users first, because other tables reference them.
TABLES = ["users", "questions", "answers", "comments", "interactions"]
ID_COLUMNS = ["user_id", "account_id", "post_id", "question_id", "comment_id", "accepted_answer_id",
              "reply_to_user_id", "source_user", "target_user", "id", "parent_post_id",
              "user_reputation", "score", "view_count", "answer_count"]
PRIMARY_KEYS = {"users": "user_id", "questions": "post_id", "answers": "post_id", "comments": "comment_id"}
DATE_COLUMNS = ["first_at", "last_at", "first_seen", "last_seen"]

# table name -> (file relative to outputs/<site> or data/processed/<site>, primary key, extra SQL)
RESULT_TABLES = {
    "topics": ("outputs", "topics.csv", "topic", None),
    "question_topics": ("processed", "question_topics.parquet", "post_id",
                        "ALTER TABLE question_topics ADD FOREIGN KEY (post_id) REFERENCES questions(post_id)"),
    "user_metrics": ("outputs", "user_metrics.csv", "user_id", None),
    "help_edges": ("outputs", "help_edges.csv", None, None),
    "topic_trends": ("outputs", "topic_trends.csv", None, None),
    "network_over_time": ("outputs", "network_over_time.csv", "year", None),
    "topic_model_comparison": ("outputs", "topic_model_comparison.csv", None, None),
    "diffusion_exposure_test": ("outputs", "diffusion_exposure_test.csv", "topic", None),
    "influence_maximization": ("outputs", "influence_maximization.csv", None, None),
    "prediction_metrics": ("outputs", "prediction_metrics.csv", "model", None),
    "recommender_eval": ("outputs", "recommender_eval.csv", "method", None),
}


def prepare_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the CSVs and convert types so they match sql/schema.sql."""
    tables = {}
    for name in TABLES:
        df = pd.read_csv(data_dir / f"{name}.csv")
        if name != "users":
            # User details live only in the users table.
            df = df.drop(columns=["user_name", "user_reputation", "account_id"], errors="ignore")
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


def load_table(conn, df: pd.DataFrame, name: str, schema: str, key=None, extra=None):
    for col in df.columns.intersection(DATE_COLUMNS):
        df[col] = pd.to_datetime(df[col])
    conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{name}" CASCADE'))
    df.to_sql(name, conn, schema=schema, index=False, chunksize=1000)
    if key:
        conn.execute(text(f'ALTER TABLE "{schema}"."{name}" ADD PRIMARY KEY ({key})'))
    if extra:
        conn.execute(text(extra))
    print(f"  {schema}.{name}: {len(df)} rows")


def load_site(conn, site: str):
    raw_dir = PROJECT_ROOT / "data" / "raw" / site
    tables = prepare_tables(raw_dir)
    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{site}"'))
    conn.execute(text(f'SET search_path TO "{site}"'))
    print(f"Creating tables in schema {site} ...")
    conn.execute(text(SCHEMA_FILE.read_text()))
    for name in TABLES:
        tables[name].to_sql(name, conn, schema=site, if_exists="append", index=False, chunksize=1000)
        print(f"  {site}.{name}: {len(tables[name])} rows")

    dirs = {"outputs": PROJECT_ROOT / "outputs" / site, "processed": PROJECT_ROOT / "data" / "processed" / site}
    present = {n: spec for n, spec in RESULT_TABLES.items() if (dirs[spec[0]] / spec[1]).exists()}
    if not present:
        print(f"  (no analysis results for {site}; run python src/run_pipeline.py first)")
    for name, (where, file, key, extra) in present.items():
        path = dirs[where] / file
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        load_table(conn, df, name, site, key, extra)
    conn.execute(text("SET search_path TO public"))


# Tables that version 1 of this project (CS50 data) created in the public schema.
OLD_PUBLIC_TABLES = TABLES + list(RESULT_TABLES)


def drop_old_public_tables(conn):
    """Remove the version-1 tables from the public schema so pgAdmin shows only the current data."""
    from sqlalchemy import inspect
    old = [t for t in inspect(conn).get_table_names(schema="public") if t in OLD_PUBLIC_TABLES]
    for t in old:
        conn.execute(text(f'DROP TABLE IF EXISTS public."{t}" CASCADE'))
    if old:
        print(f"Removed {len(old)} old tables from the public schema (previous project version).")


def load_comparison(conn):
    folder = PROJECT_ROOT / "outputs" / "comparison"
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        return
    conn.execute(text('CREATE SCHEMA IF NOT EXISTS "comparison"'))
    print("Loading the cross-community comparison ...")
    for path in files:
        load_table(conn, pd.read_csv(path), path.stem, "comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", choices=SITES + ["all"], default="all")
    args = parser.parse_args()
    sites = SITES if args.site == "all" else [args.site]
    sites = [s for s in sites if (PROJECT_ROOT / "data" / "raw" / s / "questions.csv").exists()]
    if not sites:
        raise SystemExit("No data found. Run: python src/collect_data.py --site datascience")

    engine = get_engine()
    with engine.begin() as conn:
        drop_old_public_tables(conn)
        for site in sites:
            load_site(conn, site)
        load_comparison(conn)

    print("\nDone. In pgAdmin open learning_community -> Schemas -> "
          + ", ".join(sites) + " -> Tables to see the data.")


if __name__ == "__main__":
    main()
