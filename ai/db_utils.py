"""
Reusable MotherDuck / DuckDB connection and query utilities.

Provides centralized credential loading, connection management,
application-level read-only guards (NFR_3), and optional local DuckDB
fallback for offline unit/integration testing.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|COPY|"
    r"ATTACH|DETACH|INSTALL|LOAD|EXPORT|IMPORT|PRAGMA|CALL|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def get_motherduck_token() -> str:
    """Return the trimmed MOTHERDUCK_TOKEN from environment or .env."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    token = os.getenv("MOTHERDUCK_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "MOTHERDUCK_TOKEN is missing from the project .env file."
        )
    return token


@contextmanager
def get_motherduck_connection(
    read_only: bool = True,
    catalog_alias: str = "olist",
) -> Generator[Any, None, None]:
    """
    Context manager yielding an active DuckDB connection attached to
    MotherDuck database `md:olist_db` as `olist` (and `olist_db` if needed).

    If `OLIST_LOCAL_DB_PATH` is set (for local/offline testing), attaches the
    local DuckDB database file as `olist` instead.
    """
    import duckdb

    local_db_path = os.getenv("OLIST_LOCAL_DB_PATH", "").strip()
    con = duckdb.connect()
    try:
        if local_db_path:
            con.execute(f"ATTACH '{local_db_path}' AS {catalog_alias}")
            if catalog_alias != "olist_db":
                con.execute(f"ATTACH '{local_db_path}' AS olist_db")
        else:
            token = get_motherduck_token()
            os.environ["motherduck_token"] = token
            con.execute("LOAD motherduck")
            con.execute(f"ATTACH 'md:olist_db' AS {catalog_alias}")

        if read_only:
            # Application-level guard metadata; callers using run_query_df
            # also pass through read-only SQL verification.
            logger.debug("Opened MotherDuck connection in read-only guard mode.")

        yield con
    finally:
        try:
            con.close()
        except Exception:
            pass


def assert_read_only_sql(sql: str) -> None:
    """Ensure SQL string is a read-only SELECT/WITH/SHOW/DESCRIBE query."""
    cleaned = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        raise ValueError("Empty SQL query.")
    first_word = cleaned.split()[0].upper()
    if first_word not in {"SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"}:
        raise ValueError(
            f"Read-only guard blocked non-SELECT statement starting with {first_word!r}."
        )
    if FORBIDDEN_SQL_PATTERN.search(cleaned):
        match = FORBIDDEN_SQL_PATTERN.search(cleaned)
        keyword = match.group(1).upper() if match else "UNKNOWN"
        raise ValueError(
            f"Read-only guard blocked disallowed SQL keyword: {keyword}."
        )


def run_query_df(
    sql: str,
    params: Optional[Iterable[Any]] = None,
    read_only: bool = True,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Execute a SQL query against MotherDuck (`olist` catalog) and return a DataFrame.
    Enforces `assert_read_only_sql` when `read_only=True`.
    """
    if read_only:
        assert_read_only_sql(sql)

    with get_motherduck_connection(read_only=read_only) as con:
        cursor = con.execute(sql, list(params)) if params is not None else con.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        if max_rows is not None:
            rows = cursor.fetchmany(max_rows)
        else:
            rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)
