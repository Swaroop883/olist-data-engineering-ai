"""
Bulk / Multi-Batch AI Customer Review Enrichment Runner (Module M5 - FR_6, NFR_7).

Iterates over unenriched customer reviews in batches using the shared idempotent
enrichment logic in `ai/enrich_reviews.py`, with configurable `--max-batches`
so bulk backfills remain bounded and safe.
"""

from __future__ import annotations

import argparse
import time

from enrich_reviews import DEFAULT_BATCH_SIZE, run_incremental_enrichment

PAUSE_BETWEEN_BATCHES = 2


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-batch AI review enrichment with rate-limit safety."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Reviews per batch.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Maximum batches to process in one execution (0 for unlimited).",
    )
    args = parser.parse_args()

    total_processed = 0
    total_failed = 0
    batch_num = 0

    while True:
        batch_num += 1
        print(f"\n=== Starting Batch {batch_num} ===")
        stats = run_incremental_enrichment(batch_size=args.batch_size)
        total_processed += stats["processed"]
        total_failed += stats["failed"]

        if stats["processed"] == 0 and stats["failed"] == 0:
            print("All eligible reviews are already enriched.")
            break

        if args.max_batches > 0 and batch_num >= args.max_batches:
            print(f"Reached --max-batches={args.max_batches} limit.")
            break

        time.sleep(PAUSE_BETWEEN_BATCHES)

    print("\n========== FINAL SUMMARY ==========")
    print(f"Enriched this run: {total_processed}")
    print(f"Failed this run: {total_failed}")
    print("Destination: olist.ai.enriched_order_reviews")


if __name__ == "__main__":
    main()
