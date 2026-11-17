"""
Incremental & Idempotent AI Customer Review Enrichment (Module M5 - FR_6, NFR_1, NFR_7, NFR_8).

Enriches unenriched customer reviews in `olist.raw.raw_order_reviews` using Groq LLM
(`openai/gpt-oss-120b` by default) and stores structured sentiment, topic, summary,
issue_category, language, plus auditability metadata (`model_name`, `prompt_version`,
`enriched_at`) in `olist.ai.enriched_order_reviews`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODEL_NAME = os.getenv("GROQ_ENRICH_MODEL", "openai/gpt-oss-120b")
PROMPT_VERSION = "v2.0-contextual-audit"
DEFAULT_BATCH_SIZE = 5
MAX_RETRIES = 4


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing from .env")
    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=GROQ_API_KEY,
    )


# --------------------------------------------------
# 2. AI enrichment with retry handling
# --------------------------------------------------

def enrich_review(title: str | None, comment: str | None, score: int | None, llm=None) -> dict:
    """Enrich a single customer review into structured JSON fields."""
    if llm is None:
        llm = get_llm()

    review_text = (
        f"Title: {title or 'No title'}\n"
        f"Comment: {comment or 'No comment'}\n"
        f"Score: {score}/5"
    )

    prompt = f"""
Analyze this e-commerce customer review.

The review may be written in Portuguese or another language.

Return ONLY a valid JSON object with these keys:
- sentiment: positive, neutral, or negative
- topic: short category such as delivery, product_quality,
  packaging, price, or service
- summary: concise English summary of the review
- issue_category: a specific problem, or "none" if no issue
  is mentioned
- language: language of the review

Do not invent problems that the customer did not mention.
Return only JSON, without Markdown code fences.

Review:
{review_text}
"""

    response = llm.invoke(prompt)
    content = response.content

    if not isinstance(content, str):
        raise ValueError("Groq returned an unexpected response format")

    content = content.strip()

    # Handle JSON inside Markdown code fences
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0].strip()

    result = json.loads(content)

    required = {
        "sentiment",
        "topic",
        "summary",
        "issue_category",
        "language",
    }

    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError("Invalid enrichment JSON returned by Groq")

    if result["sentiment"] not in {"positive", "neutral", "negative"}:
        raise ValueError("Invalid sentiment returned by Groq")

    return result


def enrich_with_retries(title, comment, score, llm=None) -> dict:
    """Invoke enrich_review with exponential backoff for transient API rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            return enrich_review(title, comment, score, llm=llm)
        except Exception as exc:
            message = str(exc).lower()
            retryable = any(
                term in message
                for term in [
                    "429",
                    "rate limit",
                    "rate_limit",
                    "timeout",
                    "500",
                    "502",
                    "503",
                    "504",
                    "temporarily unavailable",
                ]
            )
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            wait_seconds = min(15 * (2**attempt), 120)
            print(
                f"Transient API error ({exc}). Waiting {wait_seconds}s "
                f"before retry {attempt + 1}/{MAX_RETRIES - 1}..."
            )
            time.sleep(wait_seconds)
    raise RuntimeError("Exhausted retries in enrich_with_retries")


# --------------------------------------------------
# 3. Create / migrate destination table (NFR_7 Auditability)
# --------------------------------------------------

def create_enrichment_table(con) -> None:
    """Create olist.ai.enriched_order_reviews and ensure audit metadata columns exist."""
    con.execute("CREATE SCHEMA IF NOT EXISTS olist.ai")

    con.execute("""
        CREATE TABLE IF NOT EXISTS olist.ai.enriched_order_reviews (
            review_id VARCHAR PRIMARY KEY,
            order_id VARCHAR,
            review_score INTEGER,
            review_comment_title VARCHAR,
            review_comment_message VARCHAR,
            sentiment VARCHAR,
            topic VARCHAR,
            summary VARCHAR,
            issue_category VARCHAR,
            language VARCHAR,
            enriched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name VARCHAR DEFAULT 'openai/gpt-oss-120b',
            prompt_version VARCHAR DEFAULT 'v2.0-contextual-audit'
        )
    """)

    # Non-destructive migration for existing table (preserves all 3,011 existing rows)
    con.execute("""
        ALTER TABLE olist.ai.enriched_order_reviews
        ADD COLUMN IF NOT EXISTS model_name VARCHAR DEFAULT 'openai/gpt-oss-120b'
    """)
    con.execute("""
        ALTER TABLE olist.ai.enriched_order_reviews
        ADD COLUMN IF NOT EXISTS prompt_version VARCHAR DEFAULT 'v2.0-contextual-audit'
    """)


# --------------------------------------------------
# 4. Process a bounded batch of new reviews
# --------------------------------------------------

def run_incremental_enrichment(batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, int]:
    """Enrich up to `batch_size` unenriched reviews idempotently."""
    if not MOTHERDUCK_TOKEN:
        raise RuntimeError("MOTHERDUCK_TOKEN is missing from .env")

    os.environ["motherduck_token"] = MOTHERDUCK_TOKEN
    con = duckdb.connect()

    processed = 0
    failed = 0

    try:
        con.execute("LOAD motherduck")
        con.execute("ATTACH 'md:olist_db' AS olist")

        create_enrichment_table(con)

        existing_count = con.execute(
            "SELECT COUNT(*) FROM olist.ai.enriched_order_reviews"
        ).fetchone()[0]
        print(f"Existing enriched reviews in MotherDuck: {existing_count}")

        if batch_size <= 0:
            print("Batch size is 0 (bounded cost mode). Skipping new API calls.")
            return {"processed": 0, "failed": 0, "existing": int(existing_count)}

        # Deduplicate raw reviews by review_id so duplicate review_ids across orders
        # do not trigger primary key collisions.
        reviews = con.execute("""
            WITH dedup_raw AS (
                SELECT
                    r.review_id,
                    r.order_id,
                    r.review_score,
                    r.review_comment_title,
                    r.review_comment_message,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.review_id
                        ORDER BY r.review_creation_date DESC NULLS LAST, r.order_id
                    ) AS rn
                FROM olist.raw.raw_order_reviews AS r
                WHERE COALESCE(TRIM(r.review_comment_message), '') <> ''
                  AND r.review_id IS NOT NULL
            )
            SELECT
                d.review_id,
                d.order_id,
                d.review_score,
                d.review_comment_title,
                d.review_comment_message
            FROM dedup_raw AS d
            LEFT JOIN olist.ai.enriched_order_reviews AS e
                ON d.review_id = e.review_id
            WHERE d.rn = 1
              AND e.review_id IS NULL
            ORDER BY d.review_id
            LIMIT ?
        """, [batch_size]).fetchall()

        if not reviews:
            print("No unenriched reviews with comments remain.")
            return {"processed": 0, "failed": 0, "existing": int(existing_count)}

        print(f"Found {len(reviews)} unenriched reviews to process (model={MODEL_NAME}).")
        llm = get_llm()

        for review in reviews:
            review_id, order_id, score, title, comment = review
            print(f"\nProcessing review: {review_id}")

            try:
                result = enrich_with_retries(title, comment, score, llm=llm)

                con.execute("""
                    INSERT OR IGNORE INTO olist.ai.enriched_order_reviews (
                        review_id,
                        order_id,
                        review_score,
                        review_comment_title,
                        review_comment_message,
                        sentiment,
                        topic,
                        summary,
                        issue_category,
                        language,
                        enriched_at,
                        model_name,
                        prompt_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                """, [
                    review_id,
                    order_id,
                    score,
                    title,
                    comment,
                    result["sentiment"],
                    result["topic"],
                    result["summary"],
                    result["issue_category"],
                    result["language"],
                    MODEL_NAME,
                    PROMPT_VERSION,
                ])

                processed += 1
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print("Saved successfully.")

            except Exception as exc:
                failed += 1
                print(f"Failed to process {review_id}: {exc}")

        print("\n---------- Batch Summary ----------")
        print(f"Successfully enriched: {processed}")
        print(f"Failed: {failed}")
        print(f"Batch size: {len(reviews)}")
        print(f"Model metadata: {MODEL_NAME} ({PROMPT_VERSION})")
        print("Destination: olist.ai.enriched_order_reviews")

        return {
            "processed": processed,
            "failed": failed,
            "existing": int(existing_count) + processed,
        }

    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="Incremental AI review enrichment via Groq.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("ENRICH_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        help="Maximum number of unenriched reviews to process in this run (0 to only verify table).",
    )
    args = parser.parse_args()
    run_incremental_enrichment(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
