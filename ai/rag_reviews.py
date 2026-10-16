
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
AI_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHROMA_DIR = AI_DIR / "chroma_db"
COLLECTION_NAME = "olist_enriched_reviews"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


# --------------------------------------------------
# Initialize embedding model, vector database and LLM
# --------------------------------------------------

def initialize_rag():
    """
    Load the embedding model, existing Chroma collection,
    and Groq language model.

    Returns:
        embedding_model, collection, llm
    """

    load_dotenv(PROJECT_DIR / ".env")

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Check your project's .env file."
        )

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma database not found at: {CHROMA_DIR}"
        )

    print("Loading multilingual embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Connecting to the existing Chroma collection...")

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise ValueError(
            "The review vector collection is empty."
        )

    print(f"Vector collection loaded: {collection.count()} reviews")

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )

    print("RAG components initialized successfully.")

    return embedding_model, collection, llm


# --------------------------------------------------
# Retrieve relevant reviews
# --------------------------------------------------

def retrieve_reviews(
    question,
    embedding_model,
    collection,
    top_k=5,
):
    """
    Convert the user's question into an embedding and
    retrieve the most relevant review documents.

    Returns a list of dictionaries containing review IDs,
    documents, metadata and similarity distances.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a non-empty question.")

    total_reviews = collection.count()

    if total_reviews == 0:
        return []

    top_k = max(1, min(int(top_k), total_reviews))

    question_embedding = embedding_model.encode(
        question.strip(),
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_reviews = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    for index, document_id in enumerate(ids):
        metadata = (
            metadatas[index]
            if index < len(metadatas) and metadatas[index]
            else {}
        )

        document = (
            documents[index]
            if index < len(documents)
            else ""
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        review_id = str(
            metadata.get("review_id") or document_id
        )

        retrieved_reviews.append(
            {
                "review_id": review_id,
                "document": document or "",
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_reviews


# --------------------------------------------------
# Detect insufficient-evidence answers
# --------------------------------------------------
def is_insufficient_evidence_answer(answer):
    """
    Detect answers that explain why available evidence
    is insufficient. Such answers do not necessarily
    require individual review citations.
    """

    patterns = [
        # Different ways of expressing inability to answer
        r"\bunable to\b",
        r"\bcannot determine\b",
        r"\bcan't determine\b",
        r"\bcannot calculate\b",
        r"\bcan't calculate\b",
        r"\bcannot compute\b",
        r"\bcannot provide\b",
        r"\bcan't provide\b",
        r"\bcan't compute\b",
        r"\bnot possible to determine\b",
        r"\bnot possible to calculate\b",

        # Insufficient evidence or information
        r"\binsufficient evidence\b",
        r"\binsufficient information\b",
        r"\bnot enough evidence\b",
        r"\bnot enough information\b",
        r"\binadequate evidence\b",
        r"\bthere is no information about\b",

        # Statistics that cannot be calculated
        r"\bpercentage cannot be calculated\b",
        r"\bpercentage cannot be computed\b",
        r"\bcannot be calculated with the available data\b",
        r"\bthe available evidence does not\b",
        r"\bthe retrieved reviews do not\b",
    ]

    return any(
        re.search(pattern, answer, re.IGNORECASE)
        for pattern in patterns
    )


# --------------------------------------------------
# Generate an evidence-grounded answer
# --------------------------------------------------

def generate_answer(
    question,
    retrieved_reviews,
    llm,
):
    """
    Generate an answer using retrieved evidence.

    Citation rules:
    - Inline citations must use the format [review_id].
    - Every cited review ID must be present in the retrieved
      evidence.
    - Unsupported dataset-wide statistics must not be invented.
    - Valid insufficient-evidence responses do not receive an
      unnecessary citation warning.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a non-empty question.")

    if not retrieved_reviews:
        return (
            "I could not retrieve any reviews to support an "
            "answer. Please try a more specific question.",
            [],
        )

    valid_ids = {
        str(review["review_id"])
        for review in retrieved_reviews
        if review.get("review_id")
    }

    evidence_parts = []

    for review in retrieved_reviews:
        review_id = str(review.get("review_id", "unknown"))
        metadata = review.get("metadata") or {}
        document = review.get("document", "")

        evidence_parts.append(
            f"Review ID: [{review_id}]\n"
            f"Review score: {metadata.get('review_score', 'Not available')}\n"
            f"AI sentiment: {metadata.get('ai_sentiment', 'Not available')}\n"
            f"Topic: {metadata.get('ai_topic', 'Not available')}\n"
            f"Issue category: "
            f"{metadata.get('ai_issue_category', 'Not available')}\n"
            f"Review evidence:\n{document}"
        )

    evidence = "\n\n---\n\n".join(evidence_parts)

    prompt = f"""
You are an assistant that answers questions about customer reviews
from the Olist Brazilian E-Commerce dataset.

Follow these instructions carefully:

1. Answer using only the evidence provided below.
2. Do not invent review details, review IDs, counts, percentages,
   averages, or other statistics.
3. Cite factual claims about individual reviews using the exact
   inline format [review_id].
4. Only cite review IDs explicitly present in the retrieved evidence.
5. The retrieved reviews are a small subset of the full vector index.
   Do not treat them as a representative sample of all reviews.
6. If the question requires an exact dataset-wide count, percentage,
   average, or other statistic that cannot be calculated from the
   retrieved evidence, clearly explain this limitation.
7. Do not calculate a dataset-wide percentage without both an
   appropriate numerator and denominator.
8. When evidence is insufficient, say what information is missing.
   A clear explanation of this limitation does not require an inline
   review citation.
9. Answer in English, clearly and concisely.
10. Never claim that a statistic applies to all customers merely
    because it appears in a handful of retrieved reviews.

User question:
{question}

Retrieved evidence:
{evidence}

Write your answer now.
"""

    response = llm.invoke(prompt)

    if not isinstance(response.content, str):
        raise ValueError("Groq returned a non-text response.")

    answer = response.content.strip()

    if not answer:
        raise ValueError("Groq returned an empty answer.")

    # Recognize citations containing 32 hexadecimal characters.
    citations = re.findall(
        r"\[([0-9a-fA-F]{32})\]",
        answer,
    )

    # Reject citations that refer to reviews not retrieved.
    invalid_ids = set(citations) - valid_ids

    if invalid_ids:
        raise ValueError(
            "Citation validation failed. The answer cited "
            "IDs that were not retrieved: "
            + ", ".join(sorted(invalid_ids))
        )

    # Only warn about missing citations when the response does
    # not appear to be explaining an evidence limitation.
    if (
        not citations
        and not is_insufficient_evidence_answer(answer)
    ):
        answer += (
            "\n\nCitation warning: the model did not provide "
            "a verifiable inline review citation."
        )

    return answer, citations


# --------------------------------------------------
# Optional command-line interface
# --------------------------------------------------

def main():
    print("Olist Review RAG")
    print("=" * 40)

    embedding_model, collection, llm = initialize_rag()

    print(f"Reviews in vector index: {collection.count()}")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            reviews = retrieve_reviews(
                question,
                embedding_model,
                collection,
                top_k=5,
            )

            answer, citations = generate_answer(
                question,
                reviews,
                llm,
            )

            print("\nAnswer:")
            print(answer)

            print("\nRetrieved review IDs:")
            for review in reviews:
                print(f"- {review['review_id']}")

            print(f"\nValidated citations: {len(citations)}")

        except Exception as error:
            print(f"\nRequest failed: {error}")


if __name__ == "__main__":
    main()
