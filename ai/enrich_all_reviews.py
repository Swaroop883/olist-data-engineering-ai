
import os
import json
import time
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

BATCH_SIZE = 5
PAUSE_BETWEEN_REVIEWS = 2
MAX_RETRIES = 5

if not MOTHERDUCK_TOKEN:
    raise RuntimeError("MOTHERDUCK_TOKEN is missing from .env")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


def enrich_review(title, comment, score):
    prompt = f"""
Analyze this e-commerce customer review. It may be in Portuguese
or another language.

Return ONLY a valid JSON object with these keys:
- sentiment: positive, neutral, or negative
- topic: short category such as delivery, product_quality,
  packaging, price, or service
- summary: concise English summary
- issue_category: specific problem, or "none" if no issue
- language: language of the review

Do not invent problems the customer did not mention.

Title: {title or 'No title'}
Comment: {comment or 'No comment'}
Score: {score}/5
"""

    response = llm.invoke(prompt)
    content = response.content

    if not isinstance(content, str):
        raise ValueError("Groq returned an unexpected response format")

    content = content.strip()

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


def create_table(con):
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
            enriched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def enrich_with_retries(title, comment, score):
    for attempt in range(MAX_RETRIES):
        try:
            return enrich_review(title, comment, score)
        except Exception as exc:
            message = str(exc).lower()

            retryable = any(term in message for term in [
                "429", "rate limit", "rate_limit", "timeout",
                "500", "502", "503", "504",
                "temporarily unavailable",
            ])

            if not retryable or attempt == MAX_RETRIES - 1:
                raise

            wait_seconds = min(30 * (2 ** attempt), 300)
            print(
                f"Temporary API error. Waiting {wait_seconds}s "
                f"before retry {attempt + 1}/{MAX_RETRIES - 1}."
            )
            time.sleep(wait_seconds)


def main():
    os.environ["motherduck_token"] = MOTHERDUCK_TOKEN
    con = duckdb.connect()

    processed = 0
    failed = 0

    try:
        con.execute("LOAD motherduck")
        con.execute("ATTACH 'md:olist_db' AS olist")
        create_table(con)

        while True:
            reviews = con.execute("""
                SELECT
                    r.review_id,
                    r.order_id,
                    r.review_score,
                    r.review_comment_title,
                    r.review_comment_message
                FROM olist.raw.raw_order_reviews AS r
                LEFT JOIN olist.ai.enriched_order_reviews AS e
                    ON r.review_id = e.review_id
                WHERE COALESCE(r.review_comment_message, '') <> ''
                  AND e.review_id IS NULL
                ORDER BY r.review_id
                LIMIT ?
            """, [BATCH_SIZE]).fetchall()

            if not reviews:
                print("\nNo unenriched reviews remain.")
                break

            print(f"\nProcessing batch of {len(reviews)} reviews.")

            for review in reviews:
                review_id, order_id, score, title, comment = review
                print(f"\nReview: {review_id}")

                try:
                    result = enrich_with_retries(title, comment, score)

                    con.execute("""
                        INSERT INTO olist.ai.enriched_order_reviews (
                            review_id, order_id, review_score,
                            review_comment_title, review_comment_message,
                            sentiment, topic, summary,
                            issue_category, language
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        review_id, order_id, score, title, comment,
                        result["sentiment"], result["topic"],
                        result["summary"], result["issue_category"],
                        result["language"],
                    ])

                    processed += 1
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    print("Saved successfully.")

                except Exception as exc:
                    failed += 1
                    print(f"Failed review {review_id}: {exc}")

                time.sleep(PAUSE_BETWEEN_REVIEWS)

            print(f"\nRun totals: saved={processed}, failed={failed}")

        print("\n========== FINAL SUMMARY ==========")
        print(f"Enriched this run: {processed}")
        print(f"Failed this run: {failed}")
        print("Destination: olist.ai.enriched_order_reviews")

    finally:
        con.close()


if __name__ == "__main__":
    main()
