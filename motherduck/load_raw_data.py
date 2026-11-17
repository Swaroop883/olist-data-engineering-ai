"""
MotherDuck / DuckDB RAW Layer Loader (Module M2 - FR_2, NFR_1).

Supports loading the 9 Olist CSV datasets into MotherDuck (`md:olist_db.raw.raw_*`)
either directly from local CSV files via DuckDB (`read_csv_auto`) OR from AWS S3,
with idempotent `--skip-if-populated` support and row-count verification.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

DEFAULT_WINDOWS_CSV_DIR = Path(r"C:\Users\chsum\Desktop\DE_PROJECT\Olist_Dataset")
DEFAULT_WSL_CSV_DIR = Path("/mnt/c/Users/chsum/Desktop/DE_PROJECT/Olist_Dataset")

BUCKET = os.getenv("OLIST_S3_BUCKET", "olist-data-engineering-142366489643")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

FILES = {
    "raw_customers": "olist_customers_dataset.csv",
    "raw_geolocation": "olist_geolocation_dataset.csv",
    "raw_order_items": "olist_order_items_dataset.csv",
    "raw_order_payments": "olist_order_payments_dataset.csv",
    "raw_order_reviews": "olist_order_reviews_dataset.csv",
    "raw_orders": "olist_orders_dataset.csv",
    "raw_products": "olist_products_dataset.csv",
    "raw_sellers": "olist_sellers_dataset.csv",
    "raw_product_category_translation": "product_category_name_translation.csv",
}


def resolve_local_csv_dir(custom_dir: str | None = None) -> Path | None:
    """Return local CSV directory if available on WSL or Windows."""
    if custom_dir and Path(custom_dir).exists():
        return Path(custom_dir)
    env_dir = os.getenv("OLIST_DATASET_DIR", "").strip()
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    if DEFAULT_WSL_CSV_DIR.exists():
        return DEFAULT_WSL_CSV_DIR
    if DEFAULT_WINDOWS_CSV_DIR.exists():
        return DEFAULT_WINDOWS_CSV_DIR
    return None


def load_raw_tables(
    skip_if_populated: bool = False,
    local_csv_dir: str | None = None,
) -> dict[str, int]:
    """
    Load or verify all 9 raw Olist tables in MotherDuck `md.raw` using DuckDB.
    Automatically loads directly from local CSV files via DuckDB `read_csv_auto`
    when available, or falls back to S3 if configured.
    """
    token = os.getenv("MOTHERDUCK_TOKEN", "").strip()
    if not token:
        raise ValueError("MOTHERDUCK_TOKEN not found in .env")

    os.environ["motherduck_token"] = token

    con = duckdb.connect()
    row_counts: dict[str, int] = {}

    try:
        con.execute("LOAD motherduck")
        con.execute("ATTACH 'md:olist_db' AS md")
        con.execute("CREATE SCHEMA IF NOT EXISTS md.raw")

        resolved_local_dir = resolve_local_csv_dir(local_csv_dir)
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()

        if resolved_local_dir is None and aws_access_key and aws_secret_key:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
            con.execute(f"""
            CREATE OR REPLACE SECRET olist_s3 (
                TYPE S3,
                KEY_ID '{aws_access_key}',
                SECRET '{aws_secret_key}',
                REGION '{REGION}'
            )
            """)

        for table_name, file_name in FILES.items():
            if skip_if_populated:
                try:
                    existing_count = con.execute(
                        f"SELECT COUNT(*) FROM md.raw.{table_name}"
                    ).fetchone()[0]
                    if existing_count and existing_count > 0:
                        print(
                            f"Verified {table_name} in MotherDuck: {existing_count} rows (skipping reload)."
                        )
                        row_counts[table_name] = int(existing_count)
                        continue
                except Exception:
                    pass

            if resolved_local_dir is not None:
                source_path = str((resolved_local_dir / file_name).resolve()).replace("\\", "/")
            elif aws_access_key and aws_secret_key:
                source_path = f"s3://{BUCKET}/raw/olist/{file_name}"
            else:
                raise RuntimeError(
                    f"Table md.raw.{table_name} is not populated and no local CSV folder "
                    "was found. Set OLIST_DATASET_DIR in .env."
                )

            print(f"\nLoading {table_name} via DuckDB from {source_path}...")
            con.execute(f"""
                CREATE OR REPLACE TABLE md.raw.{table_name} AS
                SELECT *
                FROM read_csv_auto('{source_path}')
            """)

            count = con.execute(
                f"SELECT COUNT(*) FROM md.raw.{table_name}"
            ).fetchone()[0]
            row_counts[table_name] = int(count)
            print(f"Loaded {table_name}: {count} rows")

        print("\nAll 9 raw tables in MotherDuck verified successfully!")
        return row_counts
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load raw Olist CSVs into MotherDuck via DuckDB."
    )
    parser.add_argument(
        "--skip-if-populated",
        action="store_true",
        help="Skip reloading tables that already exist and have rows in md.raw",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Optional path to local Olist CSV files for direct DuckDB loading",
    )
    args = parser.parse_args()
    load_raw_tables(
        skip_if_populated=args.skip_if_populated,
        local_csv_dir=args.csv_dir,
    )