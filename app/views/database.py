import re

import pandas as pd
import streamlit as st

import ui
from data import COMMUNITIES, get_db_engine
from config import PROJECT_ROOT


def sql_queries():
    text = (PROJECT_ROOT / "sql" / "queries.sql").read_text(encoding="utf-8")
    out = {}
    for block in re.split(r"\n(?=-- \d+\. )", text):
        m = re.match(r"-- (\d+\. .+)\n", block)
        if m:
            out[m.group(1)] = block[m.end():].split(";")[0].strip()
    return out


def render(site: str):
    ui.page_header("Database", "The PostgreSQL database behind the dashboard: browse tables and run SQL")
    engine = get_db_engine()
    if engine is None:
        ui.callout("No PostgreSQL connection: the dashboard is reading the CSV files. To connect, create the "
                   "learning_community database in pgAdmin, fill in .env, run  python src/load_to_postgres.py  "
                   "and restart the dashboard.")
        return
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    schemas = [s for s in list(COMMUNITIES) + ["comparison"] if s in insp.get_schema_names()]
    schema = st.segmented_control("Schema", schemas, default=site if site in schemas else schemas[0],
                                  format_func=lambda s: COMMUNITIES.get(s, s.title())) or schemas[0]
    tables = sorted(insp.get_table_names(schema=schema))
    with engine.connect() as conn:
        counts = [(t, conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{t}"')).scalar()) for t in tables]
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown("## Tables")
        st.dataframe(pd.DataFrame(counts, columns=["table", "rows"]), hide_index=True, width="stretch", height=460)
    with right:
        st.markdown("## SQL console")
        queries = sql_queries()
        name = st.selectbox("Ready-made analysis query", list(queries))
        sql = st.text_area("SQL (read-only, runs in the selected schema)", queries[name], height=240)
        if st.button("Run query", type="primary"):
            if not re.match(r"^\s*(select|with)\b", sql, re.I) or ";" in sql.strip().rstrip(";"):
                st.error("Only a single SELECT query is allowed here.")
            else:
                with engine.connect() as conn:
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    conn.execute(text(f'SET search_path TO "{schema}", public'))
                    result = pd.read_sql(text(sql), conn)
                st.success(f"{len(result):,} rows")
                st.dataframe(result, hide_index=True, width="stretch")
                st.download_button("Download CSV", result.to_csv(index=False).encode(), f"{schema}_query.csv", "text/csv")
