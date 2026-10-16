
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

load_dotenv(PROJECT_ROOT / ".env")

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN", "").strip()

if not MOTHERDUCK_TOKEN:
    raise RuntimeError("MOTHERDUCK_TOKEN is missing from .env")


# --------------------------------------------------
# Load enriched reviews from MotherDuck
# --------------------------------------------------

def load_reviews():
    os.environ["motherduck_token"] = MOTHERDUCK_TOKEN

    con = duckdb.connect()

    try:
        con.execute("LOAD motherduck")
        con.execute("ATTACH 'md:olist_db' AS olist")

        rows = con.execute("""
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
                review_language
            FROM olist.marts.mart_enriched_review_analysis
            WHERE review_id IS NOT NULL
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
        ]

        return [dict(zip(columns, row)) for row in rows]

    finally:
        con.close()


# --------------------------------------------------
# Prepare review documents and metadata
# --------------------------------------------------

def prepare_review(review):
    review_id = str(review["review_id"])

    title = review["review_comment_title"] or ""
    original_comment = review["review_comment_message"] or ""
    summary = review["ai_summary"] or ""

    # Combine the Portuguese source text with its AI summary.
    # This lets queries match either the original wording
    # or the English summary.
    document = f"""
Review ID: {review_id}
Original title: {title}
Original customer comment: {original_comment}
English summary: {summary}
AI sentiment: {review["ai_sentiment"] or ""}
AI topic: {review["ai_topic"] or ""}
Issue category: {review["ai_issue_category"] or ""}
Original language: {review["review_language"] or ""}
Review score: {review["review_score"]}
""".strip()

    score = review["review_score"]
    score = int(score) if score is not None else 0

    metadata = {
        "review_id": review_id,
        "order_id": str(review["order_id"] or ""),
        "review_score": score,
        "score_sentiment": str(
            review["review_sentiment_category"] or ""
        ),
        "ai_sentiment": str(review["ai_sentiment"] or ""),
        "ai_topic": str(review["ai_topic"] or ""),
        "ai_issue_category": str(review["ai_issue_category"] or ""),
        "review_language": str(review["review_language"] or ""),
    }

    return review_id, document, metadata


# --------------------------------------------------
# Build a persistent Chroma vector index
# --------------------------------------------------

def main():
    print("Loading enriched reviews from MotherDuck...")
    reviews = load_reviews()

    if not reviews:
        raise RuntimeError(
            "No reviews found in mart_enriched_review_analysis."
        )

    print(f"Loaded {len(reviews)} review rows.")

    prepared = [prepare_review(review) for review in reviews]

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

    print("Opening persistent Chroma database...")
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    # Upsert makes reruns safe: an existing review ID is updated
    # instead of being inserted as a duplicate.
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist(),
        )

        print(f"Indexed {min(end, len(ids))}/{len(ids)} reviews.")

    print("\n========== VECTOR INDEX SUMMARY ==========")
    print(f"Source rows: {len(reviews)}")
    print(f"Unique source review IDs: {len(set(ids))}")
    print(f"Vectors in collection: {collection.count()}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector database: {VECTOR_DB_PATH}")
    print(f"Embedding model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
