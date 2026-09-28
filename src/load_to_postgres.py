"""
Load the CSV files from src/collect_data.py into PostgreSQL.

Usage:
    python src/load_to_postgres.py --site cs50

Needs a .env file with your database details (copy .env.example).
The tables are recreated each run, so it is safe to run again.
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

    print("\nDone. Open pgAdmin -> learning_community -> Schemas -> public -> Tables to see the data.")


if __name__ == "__main__":
    main()
