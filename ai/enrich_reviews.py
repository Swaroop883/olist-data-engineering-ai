
import os
import json
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not MOTHERDUCK_TOKEN:
    raise RuntimeError("MOTHERDUCK_TOKEN is missing from .env")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

BATCH_SIZE = 5

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


# --------------------------------------------------
# 2. AI enrichment
# --------------------------------------------------

def enrich_review(title, comment, score):
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


# --------------------------------------------------
# 3. Create destination table
# --------------------------------------------------

def create_enrichment_table(con):
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


# --------------------------------------------------
# 4. Process a batch of new reviews
# --------------------------------------------------

def main():
    os.environ["motherduck_token"] = MOTHERDUCK_TOKEN
    con = duckdb.connect()

    processed = 0
    failed = 0

    try:
        con.execute("LOAD motherduck")
        con.execute("ATTACH 'md:olist_db' AS olist")

        create_enrichment_table(con)

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
            print("No unenriched reviews with comments remain.")
            return

        print(f"Found {len(reviews)} reviews to process.")

        for review in reviews:
            review_id, order_id, score, title, comment = review

            print(f"\nProcessing review: {review_id}")

            try:
                # Call Groq to transform the review
                result = enrich_review(title, comment, score)

                # Save each successful result separately
                con.execute("""
                    INSERT INTO olist.ai.enriched_order_reviews (
                        review_id,
                        order_id,
                        review_score,
                        review_comment_title,
                        review_comment_message,
                        sentiment,
                        topic,
                        summary,
                        issue_category,
                        language
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        print("Destination: olist.ai.enriched_order_reviews")

    finally:
        con.close()


if __name__ == "__main__":
    main()
