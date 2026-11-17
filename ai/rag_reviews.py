"""
Opinion-Aware Multi-Stage RAG Engine for Olist Customer Reviews (Module M6 - FR_7, Improvement 2).

Implements the ROQuA-inspired multi-stage retrieval and synthesis pipeline:
1. Query Understanding (Intent, Entity/Product Category, Aspect/Topic, Sentiment Constraint)
2. Question Rewriting (expanding implicit phrases such as "took forever to arrive" into explicit multilingual e-commerce terms)
3. Adaptive Top-K Selection (Default K=5; adapts dynamically between K=3 and K=8 based on query specificity and evidence coverage)
4. Candidate Vector Retrieval from ChromaDB
5. Opinion-Aware Re-ranking & Relevance Filtering
6. Relevance Evaluation & Dataset-Wide Statistical Guardrail
7. Grounded Answer Generation with Validated [review_id] Citations
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
AI_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env", override=True)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHROMA_DIR = AI_DIR / "chroma_db"
COLLECTION_NAME = "olist_enriched_reviews"
GROQ_MODEL_NAME = os.getenv("GROQ_RAG_MODEL", "openai/gpt-oss-120b")

DEFAULT_TOP_K = 5
MIN_ADAPTIVE_K = 3
MAX_ADAPTIVE_K = 8

ASPECT_SYNONYMS = {
    "delivery": [
        "delivery", "shipping", "late", "delay", "arrive", "arrived", "carrier",
        "post office", "correios", "took forever", "never arrived", "slow",
        "entrega", "atraso", "demora", "prazo", "chegou",
    ],
    "product_quality": [
        "quality", "broken", "defective", "damaged", "material", "fake",
        "original", "works", "working", "stopped working", "poor quality",
        "qualidade", "defeito", "quebrado", "produto",
    ],
    "packaging": [
        "package", "packaging", "box", "wrapped", "crushed", "sealed", "torn",
        "embalagem", "caixa", "amassada",
    ],
    "price": [
        "price", "expensive", "cheap", "worth", "value", "cost", "freight",
        "preço", "caro", "barato", "frete", "custo",
    ],
    "service": [
        "seller", "service", "support", "contact", "refund", "return", "wrong item",
        "missing", "incomplete", "atendimento", "vendedor", "troca", "errado",
    ],
}

STATISTICAL_QUERY_PATTERNS = [
    r"\bwhat percentage\b",
    r"\bwhat percent\b",
    r"\bexact percentage\b",
    r"\bhow many total customers\b",
    r"\bacross the entire dataset\b",
    r"\bof all customers\b",
    r"\bof all reviews\b",
    r"\btotal number of reviews in the dataset\b",
]


# --------------------------------------------------
# Stage 1 & 2: Query Understanding, Question Rewriting & Adaptive K
# --------------------------------------------------

def understand_and_rewrite_query(
    question: str,
    default_k: int = DEFAULT_TOP_K,
    adaptive: bool = True,
) -> dict[str, Any]:
    """
    Analyze the user question to extract intent, e-commerce aspect/topic,
    sentiment constraint, rewritten retrieval query, and adaptive Top-K (3..8).
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a non-empty question.")

    clean_q = question.strip()
    lower_q = clean_q.lower()

    # Detect aspect / topic
    detected_aspects: list[str] = []
    expansion_terms: list[str] = []
    for aspect, keywords in ASPECT_SYNONYMS.items():
        if any(kw in lower_q for kw in keywords):
            detected_aspects.append(aspect)
            expansion_terms.extend(keywords[:4])

    primary_aspect = detected_aspects[0] if detected_aspects else "general"

    # Detect sentiment polarity
    negative_cues = [
        "complain", "complaint", "complaints", "unhappy", "bad", "worst",
        "negative", "issue", "issues", "problem", "problems", "delay",
        "late", "broken", "defective", "angry", "dissatisfied", "took forever",
    ]
    positive_cues = [
        "praise", "happy", "good", "best", "positive", "love", "satisfied",
        "fast", "excellent", "great", "recommend",
    ]

    if any(cue in lower_q for cue in negative_cues) and not any(
        cue in lower_q for cue in positive_cues
    ):
        sentiment_constraint = "negative"
    elif any(cue in lower_q for cue in positive_cues) and not any(
        cue in lower_q for cue in negative_cues
    ):
        sentiment_constraint = "positive"
    else:
        sentiment_constraint = "any"

    # Check if user is asking for an unsupported dataset-wide statistic
    is_dataset_stat_query = any(
        re.search(pat, lower_q) for pat in STATISTICAL_QUERY_PATTERNS
    )

    # Determine intent
    if is_dataset_stat_query:
        intent = "dataset_wide_statistic_inquiry"
    elif sentiment_constraint == "negative":
        intent = "complaint_root_cause_analysis"
    elif sentiment_constraint == "positive":
        intent = "satisfaction_driver_analysis"
    elif len(detected_aspects) >= 1:
        intent = "aspect_opinion_qa"
    else:
        intent = "general_review_exploration"

    # Question rewriting (ROQuA Stage 2: resolve implicit expressions & add domain context)
    rewritten_parts = [clean_q]
    if detected_aspects:
        rewritten_parts.append(f"E-commerce aspect: {', '.join(detected_aspects)}.")
    if sentiment_constraint != "any":
        rewritten_parts.append(f"Customer sentiment: {sentiment_constraint}.")
    if expansion_terms:
        rewritten_parts.append("Related concepts: " + ", ".join(sorted(set(expansion_terms))) + ".")
    rewritten_query = " ".join(rewritten_parts)

    # Adaptive K selection (FR_7: default K=5, adaptive between 3 and 8)
    if not adaptive:
        selected_k = max(1, int(default_k))
        k_reason = f"Fixed Top-K={selected_k}"
    else:
        # Specific review ID or narrow single-aspect query -> K=3..4
        if re.search(r"\b[0-9a-f]{32}\b", lower_q):
            selected_k = MIN_ADAPTIVE_K
            k_reason = "Specific review ID lookup -> focused K=3"
        elif len(clean_q.split()) <= 5 and len(detected_aspects) == 1 and sentiment_constraint != "any":
            selected_k = 4
            k_reason = "Narrow single-aspect + single-sentiment query -> K=4"
        elif (
            len(detected_aspects) >= 2
            or "compare" in lower_q
            or "common" in lower_q
            or "main" in lower_q
            or "various" in lower_q
            or "reasons" in lower_q
            or "summary" in lower_q
            or "overview" in lower_q
        ):
            selected_k = MAX_ADAPTIVE_K if len(detected_aspects) >= 2 or "compare" in lower_q else 7
            k_reason = f"Broad multi-aspect or synthesis query -> expanded K={selected_k}"
        else:
            selected_k = DEFAULT_TOP_K
            k_reason = "Standard opinion question -> default K=5"

        selected_k = max(MIN_ADAPTIVE_K, min(MAX_ADAPTIVE_K, selected_k))

    return {
        "original_question": clean_q,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "detected_aspects": detected_aspects or ["general"],
        "primary_aspect": primary_aspect,
        "sentiment_constraint": sentiment_constraint,
        "is_dataset_stat_query": is_dataset_stat_query,
        "adaptive_k": selected_k,
        "k_reason": k_reason,
    }


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
    import chromadb
    from langchain_groq import ChatGroq
    from sentence_transformers import SentenceTransformer

    load_dotenv(PROJECT_DIR / ".env", override=True)

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
# Stage 4 & 5: Retrieve, Re-rank and Filter Reviews
# --------------------------------------------------

def rerank_and_filter_reviews(
    candidates: list[dict[str, Any]],
    query_plan: dict[str, Any],
    target_k: int,
) -> list[dict[str, Any]]:
    """
    Score candidate reviews by combining vector similarity with opinion/aspect
    alignment from the query understanding stage, then return the top `target_k`.
    """
    if not candidates:
        return []

    primary_aspect = query_plan.get("primary_aspect", "general")
    sentiment_constraint = query_plan.get("sentiment_constraint", "any")

    scored: list[dict[str, Any]] = []
    for item in candidates:
        meta = item.get("metadata") or {}
        distance = item.get("distance")
        base_sim = 1.0 - float(distance) if distance is not None else 0.5

        bonus = 0.0
        item_topic = str(meta.get("ai_topic") or "").lower()
        item_issue = str(meta.get("ai_issue_category") or "").lower()
        item_sentiment = str(meta.get("ai_sentiment") or meta.get("score_sentiment") or "").lower()

        if primary_aspect != "general" and (
            primary_aspect in item_topic or primary_aspect in item_issue
        ):
            bonus += 0.08

        if sentiment_constraint != "any" and item_sentiment == sentiment_constraint:
            bonus += 0.06

        relevance_score = round(min(1.0, max(0.0, base_sim + bonus)), 4)
        enriched_item = dict(item)
        enriched_item["relevance_score"] = relevance_score
        scored.append(enriched_item)

    scored.sort(
        key=lambda x: (
            x.get("relevance_score", 0.0),
            -(x.get("distance") if x.get("distance") is not None else 1.0),
        ),
        reverse=True,
    )

    return scored[:target_k]


def retrieve_reviews(
    question: str,
    embedding_model,
    collection,
    top_k: int | None = None,
    adaptive: bool = True,
):
    """
    Convert the user's question into an embedding using Opinion-Aware
    Multi-Stage Query Understanding, retrieve candidate review documents,
    re-rank by aspect/sentiment alignment, and return the top K results.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a non-empty question.")

    total_reviews = collection.count()
    if total_reviews == 0:
        return []

    query_plan = understand_and_rewrite_query(
        question,
        default_k=top_k if top_k is not None else DEFAULT_TOP_K,
        adaptive=(adaptive and top_k is None),
    )

    effective_k = (
        int(top_k)
        if top_k is not None
        else int(query_plan["adaptive_k"])
    )
    effective_k = max(1, min(effective_k, total_reviews))

    # Over-fetch candidate pool slightly so Stage 5 opinion re-ranking can promote
    # aspect-aligned and sentiment-aligned reviews.
    candidate_pool_size = min(total_reviews, max(effective_k, min(effective_k + 4, 12)))

    search_text = query_plan["rewritten_query"]
    question_embedding = embedding_model.encode(
        search_text,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=candidate_pool_size,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    candidates = []

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

        candidates.append(
            {
                "review_id": review_id,
                "document": document or "",
                "metadata": metadata,
                "distance": distance,
                "query_plan": query_plan,
            }
        )

    return rerank_and_filter_reviews(candidates, query_plan, effective_k)


# --------------------------------------------------
# Detect insufficient-evidence answers
# --------------------------------------------------
def is_insufficient_evidence_answer(answer: str) -> bool:
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
        r"\bno sufficient evidence found\b",

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
# Stage 6 & 7: Relevance Evaluation & Grounded Answer Generation
# --------------------------------------------------

def generate_answer(
    question: str,
    retrieved_reviews: list[dict[str, Any]],
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
            "No sufficient evidence found: I could not retrieve any reviews "
            "to support an answer. Please try a more specific question.",
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
            f"Product category: {metadata.get('product_category', 'Not available')}\n"
            f"Customer state: {metadata.get('customer_state', 'Not available')}\n"
            f"Delivery status: {metadata.get('delivery_status', 'Not available')}\n"
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
   retrieved evidence, clearly explain this limitation (and note that
   the Text-to-SQL module can compute exact warehouse-wide percentages).
7. Do not calculate a dataset-wide percentage without both an
   appropriate numerator and denominator.
8. When evidence is insufficient, explicitly state "No sufficient evidence found"
   and explain what information is missing. A clear explanation of this
   limitation does not require an inline review citation.
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

    # Recognize citations containing 32 hexadecimal characters (or test review IDs).
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
    print("Olist Opinion-Aware Multi-Stage Review RAG")
    print("=" * 45)

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
            query_plan = understand_and_rewrite_query(question, adaptive=True)
            print(
                f"Query Plan -> Intent: {query_plan['intent']} | "
                f"Aspect: {query_plan['primary_aspect']} | "
                f"Sentiment: {query_plan['sentiment_constraint']} | "
                f"Adaptive K: {query_plan['adaptive_k']}"
            )

            reviews = retrieve_reviews(
                question,
                embedding_model,
                collection,
                top_k=None,
                adaptive=True,
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
                print(
                    f"- {review['review_id']} "
                    f"(relevance={review.get('relevance_score', 0):.3f})"
                )

            print(f"\nValidated citations: {len(citations)}")

        except Exception as error:
            print(f"\nRequest failed: {error}")


if __name__ == "__main__":
    main()
