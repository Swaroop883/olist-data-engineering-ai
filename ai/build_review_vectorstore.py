"""
Contextual & Incremental ChromaDB Vector Store Builder (Module M6 - FR_7, NFR_1, Improvement 2).

Enriches review documents with full e-commerce order context (product category,
seller state, customer city/state, order status, and delivery delay status) as
promised in the Data Model & Technical Analysis documents, and supports
incremental delta indexing so Airflow runs only embed new reviews by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chromadb
import duckdb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "ai" / "chroma_db"
COLLECTION_NAME = "olist_enriched_reviews"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64

load_dotenv(PROJECT_ROOT / ".env", override=True)

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "").strip()


# --------------------------------------------------
# Load enriched reviews + e-commerce context from MotherDuck
# --------------------------------------------------

def load_reviews():
    """
    Fetch enriched reviews from `olist.marts.mart_enriched_review_analysis`
    joined with aggregated order-item product/seller context (`mart_sales`)
    and delivery delay status (`mart_delivery_performance`), deduplicated
    strictly to 1 row per `review_id`.
    """
    token = os.getenv("MOTHERDUCK_TOKEN", "").strip() or MOTHERDUCK_TOKEN
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN is missing from .env")

    os.environ["motherduck_token"] = token
    con = duckdb.connect()

    try:
        con.execute("LOAD motherduck")
        con.execute("ATTACH 'md:olist_db' AS olist")

        rows = con.execute("""
            WITH order_context AS (
                SELECT
                    order_id,
                    STRING_AGG(DISTINCT COALESCE(product_category_name, 'unknown'), ', ') AS product_categories,
                    STRING_AGG(DISTINCT COALESCE(seller_id, 'unknown'), ', ') AS seller_ids,
                    STRING_AGG(DISTINCT COALESCE(seller_state, 'unknown'), ', ') AS seller_states,
                    SUM(total_item_value) AS order_total_value,
                    COUNT(*) AS order_item_count
                FROM olist.marts.mart_sales
                GROUP BY order_id
            ),
            delivery_context AS (
                SELECT
                    order_id,
                    delivery_delay_days
                FROM olist.marts.mart_delivery_performance
            ),
            dedup_reviews AS (
                SELECT
                    r.review_id,
                    r.order_id,
                    r.review_score,
                    r.review_comment_title,
                    r.review_comment_message,
                    r.review_sentiment_category,
                    r.ai_sentiment,
                    r.ai_topic,
                    r.ai_summary,
                    r.ai_issue_category,
                    r.review_language,
                    r.order_status,
                    r.customer_city,
                    r.customer_state,
                    oc.product_categories,
                    oc.seller_ids,
                    oc.seller_states,
                    oc.order_total_value,
                    dc.delivery_delay_days,
                    ROW_NUMBER() OVER (
                        PARTITION BY r.review_id
                        ORDER BY r.ai_enriched_at DESC NULLS LAST, r.order_id
                    ) AS rn
                FROM olist.marts.mart_enriched_review_analysis AS r
                LEFT JOIN order_context AS oc
                    ON r.order_id = oc.order_id
                LEFT JOIN delivery_context AS dc
                    ON r.order_id = dc.order_id
                WHERE r.review_id IS NOT NULL
            )
            SELECT
                review_id,
                order_id,
                review_score,
                review_comment_title,
                review_comment_message,
                review_sentiment_category,
                ai_sentiment,
                ai_topic,
                ai_summary,
                ai_issue_category,
                review_language,
                order_status,
                customer_city,
                customer_state,
                product_categories,
                seller_ids,
                seller_states,
                order_total_value,
                delivery_delay_days
            FROM dedup_reviews
            WHERE rn = 1
            ORDER BY review_id
        """).fetchall()

        columns = [
            "review_id",
            "order_id",
            "review_score",
            "review_comment_title",
            "review_comment_message",
            "review_sentiment_category",
            "ai_sentiment",
            "ai_topic",
            "ai_summary",
            "ai_issue_category",
            "review_language",
            "order_status",
            "customer_city",
            "customer_state",
            "product_categories",
            "seller_ids",
            "seller_states",
            "order_total_value",
            "delivery_delay_days",
        ]

        return [dict(zip(columns, row)) for row in rows]

    finally:
        con.close()


# --------------------------------------------------
# Prepare review documents and metadata with e-commerce context
# --------------------------------------------------

def prepare_review(review: dict):
    review_id = str(review["review_id"])

    title = review.get("review_comment_title") or ""
    original_comment = review.get("review_comment_message") or ""
    summary = review.get("ai_summary") or ""
    product_category = review.get("product_categories") or "unknown"
    customer_state = review.get("customer_state") or "unknown"
    customer_city = review.get("customer_city") or "unknown"
    seller_state = review.get("seller_states") or "unknown"
    order_status = review.get("order_status") or "unknown"
    delay_days = review.get("delivery_delay_days")

    if delay_days is None:
        delivery_status = "Unknown / Not Delivered"
    elif int(delay_days) > 0:
        delivery_status = f"Delayed ({int(delay_days)} days late)"
    elif int(delay_days) < 0:
        delivery_status = f"Early ({abs(int(delay_days))} days early)"
    else:
        delivery_status = "On Time (0 days delay)"

    # Combine Portuguese source text, English AI summary, and structured Olist order context
    document = f"""
Review ID: {review_id}
Original title: {title}
Original customer comment: {original_comment}
English summary: {summary}
AI sentiment: {review.get("ai_sentiment") or ""}
AI topic: {review.get("ai_topic") or ""}
Issue category: {review.get("ai_issue_category") or ""}
Original language: {review.get("review_language") or ""}
Review score: {review.get("review_score")}
Product category: {product_category}
Customer location: {customer_city}, {customer_state}
Seller state: {seller_state}
Order status: {order_status}
Delivery status: {delivery_status}
""".strip()

    score = review.get("review_score")
    score = int(score) if score is not None else 0

    metadata = {
        "review_id": review_id,
        "order_id": str(review.get("order_id") or ""),
        "review_score": score,
        "score_sentiment": str(review.get("review_sentiment_category") or ""),
        "ai_sentiment": str(review.get("ai_sentiment") or ""),
        "ai_topic": str(review.get("ai_topic") or ""),
        "ai_issue_category": str(review.get("ai_issue_category") or ""),
        "review_language": str(review.get("review_language") or ""),
        "product_category": str(product_category),
        "customer_state": str(customer_state),
        "seller_state": str(seller_state),
        "order_status": str(order_status),
        "delivery_status": str(delivery_status),
    }

    return review_id, document, metadata


# --------------------------------------------------
# Build or incrementally refresh persistent Chroma vector index
# --------------------------------------------------

def build_or_update_vectorstore(full_refresh: bool = False) -> dict[str, int]:
    print("Loading enriched reviews from MotherDuck...")
    reviews = load_reviews()

    if not reviews:
        raise RuntimeError(
            "No reviews found in mart_enriched_review_analysis."
        )

    print(f"Loaded {len(reviews)} enriched review rows from MotherDuck.")

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    existing_ids: set[str] = set()
    if not full_refresh and collection.count() > 0:
        existing_records = collection.get(include=[])
        existing_ids = set(existing_records.get("ids") or [])
        print(f"Existing vectors in ChromaDB: {len(existing_ids)}")

    if not full_refresh:
        reviews_to_index = [
            r for r in reviews if str(r["review_id"]) not in existing_ids
        ]
    else:
        reviews_to_index = reviews

    if not reviews_to_index:
        print(
            "ChromaDB vector index is already up to date with MotherDuck "
            f"({len(existing_ids)} vectors). Skipping embedding generation."
        )
        return {
            "source_rows": len(reviews),
            "newly_indexed": 0,
            "total_vectors": collection.count(),
        }

    print(f"Preparing {len(reviews_to_index)} review documents for indexing...")
    prepared = [prepare_review(review) for review in reviews_to_index]

    ids = [item[0] for item in prepared]
    documents = [item[1] for item in prepared]
    metadatas = [item[2] for item in prepared]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate review IDs detected. Fix duplicates before indexing."
        )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings locally...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist(),
        )

        print(f"Indexed {min(end, len(ids))}/{len(ids)} delta reviews.")

    print("\n========== VECTOR INDEX SUMMARY ==========")
    print(f"Source rows in MotherDuck: {len(reviews)}")
    print(f"Newly indexed this run: {len(ids)}")
    print(f"Total vectors in collection: {collection.count()}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector database: {VECTOR_DB_PATH}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    return {
        "source_rows": len(reviews),
        "newly_indexed": len(ids),
        "total_vectors": collection.count(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build or incrementally update the Olist review ChromaDB index."
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Re-embed and upsert all reviews instead of only new/delta reviews.",
    )
    args = parser.parse_args()
    build_or_update_vectorstore(full_refresh=args.full_refresh)


if __name__ == "__main__":
    main()
