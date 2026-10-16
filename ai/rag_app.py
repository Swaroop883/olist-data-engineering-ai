
import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

AI_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

st.set_page_config(
    page_title="Olist Review Intelligence",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# Import RAG functions lazily
# Define these BEFORE the form can call them.
# --------------------------------------------------

def ensure_ai_path():
    ai_path = str(AI_DIR)

    if ai_path not in sys.path:
        sys.path.insert(0, ai_path)


@st.cache_resource(show_spinner=False)
def load_rag_components():
    ensure_ai_path()

    from rag_reviews import initialize_rag

    return initialize_rag()


def retrieve_reviews(question, embedding_model, collection):
    ensure_ai_path()

    from rag_reviews import retrieve_reviews as retrieve

    return retrieve(question, embedding_model, collection)


def generate_answer(question, retrieved_reviews, llm):
    ensure_ai_path()

    from rag_reviews import generate_answer as generate

    return generate(question, retrieved_reviews, llm)


# --------------------------------------------------
# Page interface
# --------------------------------------------------

st.title("🔎 Olist Review Intelligence")

st.caption(
    "Ask questions about customer reviews using semantic search "
    "and AI-generated answers grounded in retrieved evidence."
)

st.divider()

st.subheader("Ask a question about customer reviews")

st.markdown(
    "Try questions about delivery delays, product quality, "
    "customer complaints, or purchase experiences."
)

with st.form("rag_question_form"):
    question = st.text_input(
        "Your question",
        placeholder="What do customers say about late deliveries?",
    )

    submitted = st.form_submit_button(
        "Search reviews",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Search and answer
# --------------------------------------------------

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Loading the RAG system and searching reviews..."
            ):
                embedding_model, collection, llm = (
                    load_rag_components()
                )

                retrieved = retrieve_reviews(
                    question.strip(),
                    embedding_model,
                    collection,
                )

                generated = generate_answer(
                    question.strip(),
                    retrieved,
                    llm,
                )

                if isinstance(generated, tuple):
                    answer, citations = generated
                else:
                    answer = generated
                    citations = []

                st.session_state["rag_last_result"] = {
                    "question": question.strip(),
                    "answer": answer,
                    "citations": citations,
                    "retrieved": retrieved,
                    "review_count": collection.count(),
                }

        except Exception as exc:
            st.error(f"RAG request failed: {exc}")


# --------------------------------------------------
# Display the latest result
# --------------------------------------------------

result = st.session_state.get("rag_last_result")

if result:
    st.divider()

    st.subheader("Answer")
    st.markdown(result["answer"])

    retrieved = result["retrieved"]
    citations = set(result["citations"])

    st.subheader("Retrieved evidence")

    col1, col2 = st.columns(2)

    col1.metric("Reviews retrieved", len(retrieved))
    col2.metric(
        "Reviews in vector index",
        result["review_count"],
    )

    st.caption(
        "Similarity distance helps rank results. It is not a "
        "probability or a guarantee of factual correctness."
    )

    for rank, item in enumerate(retrieved, start=1):
        review_id = item["review_id"]
        metadata = item["metadata"]

        label = (
            "Cited source"
            if review_id in citations
            else "Retrieved source"
        )

        with st.expander(
            f"{rank}. {label} — Review {review_id}"
        ):
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Rating",
                metadata.get("review_score", "Unknown"),
            )

            col2.write("**AI sentiment**")
            col2.write(
                metadata.get("ai_sentiment") or "Unknown"
            )

            col3.write("**AI topic**")
            col3.write(
                metadata.get("ai_topic") or "Unknown"
            )

            st.write("**Issue category**")
            st.write(
                metadata.get("ai_issue_category") or "Unknown"
            )

            st.write("**Review evidence**")
            st.text(item["document"])

            st.caption(
                f"Review ID: {review_id} | "
                f"Distance: {item['distance']:.4f}"
            )

    st.info(
        "Answers use retrieved reviews. A handful of retrieved "
        "reviews cannot establish statistics for the entire dataset."
    )
