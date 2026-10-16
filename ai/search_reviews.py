
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "ai" / "chroma_db"
COLLECTION_NAME = "olist_enriched_reviews"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
TOP_K = 5


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Vector index contains {collection.count()} reviews.")

    while True:
        question = input(
            "\nAsk about customer reviews (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting semantic search.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # Convert the question into an embedding
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        # Retrieve the most semantically similar reviews
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print(f"\nTop {len(ids)} relevant reviews:\n")

        for rank, (review_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            print("=" * 65)
            print(f"Rank: {rank}")
            print(f"Source review ID: {review_id}")
            print(f"Similarity: {1 - distance:.4f}")
            print(f"Rating: {metadata.get('review_score')}")
            print(f"AI sentiment: {metadata.get('ai_sentiment')}")
            print(f"Topic: {metadata.get('ai_topic')}")
            print(f"Issue: {metadata.get('ai_issue_category')}")
            print("\nRetrieved evidence:")
            print(document)
            print()


if __name__ == "__main__":
    main()
