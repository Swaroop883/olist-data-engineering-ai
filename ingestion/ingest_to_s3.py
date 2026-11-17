"""
Olist Raw Dataset Ingestion to AWS S3 (Module M1 - FR_1, NFR_1).

Uploads the 9 Olist CSV files from the local dataset directory to the S3 raw
zone (`s3://<bucket>/raw/olist/`) with idempotent size/existence verification
and dual Windows/WSL path resolution.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest_to_s3")

DEFAULT_WINDOWS_PATH = Path(r"C:\Users\chsum\Desktop\DE_PROJECT\Olist_Dataset")
DEFAULT_WSL_PATH = Path("/mnt/c/Users/chsum/Desktop/DE_PROJECT/Olist_Dataset")
DEFAULT_BUCKET = os.getenv("OLIST_S3_BUCKET", "olist-data-engineering-142366489643")
S3_PREFIX = "raw/olist"

EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def resolve_dataset_dir(custom_dir: str | None = None) -> Path:
    """Resolve the local Olist dataset folder across WSL and Windows."""
    if custom_dir:
        return Path(custom_dir)
    env_dir = os.getenv("OLIST_DATASET_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    if DEFAULT_WSL_PATH.exists():
        return DEFAULT_WSL_PATH
    return DEFAULT_WINDOWS_PATH


def object_matches_local_size(s3_client, bucket: str, key: str, local_size: int) -> bool:
    """Return True if the S3 object already exists with the exact same byte size."""
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        return int(head.get("ContentLength", -1)) == int(local_size)
    except ClientError:
        return False


def ingest_csv_files(
    dataset_dir: Path | None = None,
    bucket_name: str = DEFAULT_BUCKET,
    force_upload: bool = False,
) -> dict[str, int]:
    """Upload Olist CSV files to S3 raw zone idempotently."""
    resolved_dir = resolve_dataset_dir(str(dataset_dir) if dataset_dir else None)
    if not resolved_dir.exists():
        logger.warning(
            "Local dataset folder %s not found on this machine. "
            "Skipping local-to-S3 upload (S3 raw zone already populated).",
            resolved_dir,
        )
        return {"uploaded": 0, "skipped": 0, "missing_dir": 1}

    s3 = boto3.client("s3")
    csv_files = sorted(resolved_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {resolved_dir}")

    uploaded = 0
    skipped = 0

    for file_path in csv_files:
        s3_key = f"{S3_PREFIX}/{file_path.name}"
        local_size = file_path.stat().st_size

        if not force_upload and object_matches_local_size(
            s3, bucket_name, s3_key, local_size
        ):
            logger.info(
                "%s already exists in s3://%s/%s (%d bytes) — skipping.",
                file_path.name,
                bucket_name,
                s3_key,
                local_size,
            )
            skipped += 1
            continue

        s3.upload_file(str(file_path), bucket_name, s3_key)
        logger.info("%s uploaded successfully to s3://%s/%s", file_path.name, bucket_name, s3_key)
        uploaded += 1

    logger.info(
        "S3 ingestion summary: uploaded=%d, skipped=%d, total_found=%d",
        uploaded,
        skipped,
        len(csv_files),
    )
    return {"uploaded": uploaded, "skipped": skipped, "missing_dir": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Olist CSV files to AWS S3.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to local Olist CSVs")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET, help="Target S3 bucket")
    parser.add_argument("--force", action="store_true", help="Re-upload even if size matches")
    args = parser.parse_args()

    ingest_csv_files(
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        bucket_name=args.bucket,
        force_upload=args.force,
    )