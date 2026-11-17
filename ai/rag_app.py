"""
Olist Opinion-Aware Multi-Stage RAG Streamlit Assistant (Module M6 - FR_7, Improvement 2).

Can be run standalone (`streamlit run ai/rag_app.py --server.port 8503`)
or embedded inside the unified platform (`app.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    st = None

# --------------------------------------------------
# Configuration
# --------------------------------------------------

AI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AI_DIR.parent
for _p in (str(PROJECT_ROOT), str(AI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from auto_visualizer import render_rag_evidence_charts
from copilot_orchestrator import detect_unsafe_user_prompt

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def ensure_ai_path():
    ai_path = str(AI_DIR)
    if ai_path not in sys.path:
        sys.path.insert(0, ai_path)


def _cache_resource_decorator(func):
    if st is not None:
        return st.cache_resource(show_spinner=False)(func)
    return func


@_cache_resource_decorator
def load_rag_components():
    ensure_ai_path()
    from rag_reviews import initialize_rag
    return initialize_rag()


def retrieve_reviews(question, embedding_model, collection, top_k=None, adaptive=True):
    ensure_ai_path()
    from rag_reviews import retrieve_reviews as retrieve
    return retrieve(
        question,
        embedding_model,
        collection,
        top_k=top_k,
        adaptive=adaptive,
    )


def generate_answer(question, retrieved_reviews, llm):
    ensure_ai_path()
    from rag_reviews import generate_answer as generate
    return generate(question, retrieved_reviews, llm)


def analyze_query(question, default_k=5, adaptive=True):
    ensure_ai_path()
    from rag_reviews import understand_and_rewrite_query
    return understand_and_rewrite_query(
        question,
        default_k=default_k,
        adaptive=adaptive,
    )


def render_rag_interface(embedded: bool = False) -> None:
    """Render the Opinion-Aware Multi-Stage RAG UI with Evidence Charts."""
    if not embedded:
        st.title("🔎 Olist Review Intelligence (Opinion-Aware RAG)")
    else:
        st.header("🔎 Customer Review Intelligence (Opinion-Aware Multi-Stage RAG)")

    st.caption(
        "ROQuA-inspired multi-stage RAG: Query Understanding → Question Rewriting → "
        "Adaptive Top-K Retrieval (K=3–8, default 5) → Opinion Re-ranking → "
        "Evidence-Grounded Synthesis with Validated `[review_id]` Citations & Visual Evidence Charts."
    )

    with st.sidebar:
        st.subheader("RAG Retrieval Settings")
        retrieval_mode = st.radio(
            "Retrieval Top-K Mode",
            options=["Adaptive K (3–8, default 5)", "Manual Top-K"],
            index=0,
            key="rag_k_mode",
        )
        manual_k = None
        if retrieval_mode == "Manual Top-K":
            manual_k = st.slider(
                "Number of reviews to retrieve (K)",
                min_value=3,
                max_value=8,
                value=5,
                key="rag_manual_k",
            )
        st.divider()
        st.markdown("**Sample Opinion Questions:**")
        sample_questions = [
            "What do customers complain about regarding delivery delays?",
            "Why are some customers unhappy with product quality or defective items?",
            "What positive experiences do customers mention about fast shipping?",
            "Do customers report issues with wrong or incomplete items from sellers?",
            "What percentage of all customers complained about delivery?",
        ]
        for q in sample_questions:
            st.caption(f"• {q}")

    st.subheader("Ask a question about customer reviews")
    st.markdown(
        "Try questions about delivery delays, product quality, packaging, "
        "seller service, or purchase experiences."
    )

    with st.form("rag_question_form"):
        question = st.text_input(
            "Your question",
            placeholder="What do customers say about late deliveries?",
        )

        submitted = st.form_submit_button(
            "Search reviews & visualize evidence",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            rejection = detect_unsafe_user_prompt(question.strip())
            if rejection:
                st.error(f"❌ {rejection}")
                st.session_state.pop("rag_last_result", None)
            else:
                try:
                    with st.spinner(
                        "Running Multi-Stage Query Understanding, ChromaDB Retrieval & Groq Synthesis..."
                    ):
                        adaptive_enabled = manual_k is None
                        query_plan = analyze_query(
                            question.strip(),
                            default_k=manual_k or 5,
                            adaptive=adaptive_enabled,
                        )

                        embedding_model, collection, llm = load_rag_components()

                        retrieved = retrieve_reviews(
                            question.strip(),
                            embedding_model,
                            collection,
                            top_k=manual_k,
                            adaptive=adaptive_enabled,
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
                            "query_plan": query_plan,
                            "answer": answer,
                            "citations": citations,
                            "retrieved": retrieved,
                            "review_count": collection.count(),
                        }

                except Exception as exc:
                    st.error(f"RAG request failed: {exc}")

    result = st.session_state.get("rag_last_result")

    if result:
        st.divider()

        query_plan = result.get("query_plan")
        if query_plan:
            with st.expander("🧠 Multi-Stage Query Understanding & Adaptive Retrieval Trace", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Detected Intent", query_plan["intent"])
                c2.metric("Primary Aspect", query_plan["primary_aspect"])
                c3.metric("Sentiment Filter", query_plan["sentiment_constraint"])
                c4.metric("Selected Top-K", query_plan["adaptive_k"])
                st.write(f"**Adaptive K Rationale:** {query_plan['k_reason']}")
                st.write(f"**Rewritten Retrieval Query:** `{query_plan['rewritten_query']}`")

        st.subheader("Grounded Answer")
        st.markdown(result["answer"])

        retrieved = result["retrieved"]
        citations = set(result["citations"])

        # Visual charts of retrieved review evidence
        render_rag_evidence_charts(retrieved, key_prefix="standalone_rag_viz")

        st.subheader("Retrieved Evidence & Citations")

        col1, col2, col3 = st.columns(3)
        col1.metric("Reviews retrieved (K)", len(retrieved))
        col2.metric("Verified inline citations", len(citations))
        col3.metric("Reviews in vector index", result["review_count"])

        st.caption(
            "Similarity distance and opinion-alignment relevance score help rank results. "
            "Every inline `[review_id]` citation is verified against the retrieved set."
        )

        for rank, item in enumerate(retrieved, start=1):
            review_id = item["review_id"]
            metadata = item["metadata"]

            label = (
                "✅ Cited source"
                if review_id in citations
                else "📄 Retrieved source"
            )

            with st.expander(f"{rank}. {label} — Review {review_id}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rating", metadata.get("review_score", "Unknown"))
                col2.write("**AI sentiment**")
                col2.write(metadata.get("ai_sentiment") or "Unknown")
                col3.write("**AI topic**")
                col3.write(metadata.get("ai_topic") or "Unknown")
                col4.write("**Relevance score**")
                col4.write(f"{item.get('relevance_score', 0.0):.4f}")

                c_meta1, c_meta2, c_meta3 = st.columns(3)
                c_meta1.write(f"**Issue category:** {metadata.get('ai_issue_category') or 'Unknown'}")
                c_meta2.write(f"**Product category:** {metadata.get('product_category') or 'N/A'}")
                c_meta3.write(f"**Delivery status:** {metadata.get('delivery_status') or 'N/A'}")

                st.write("**Review evidence**")
                st.text(item["document"])

                st.caption(
                    f"Review ID: {review_id} | "
                    f"Order ID: {metadata.get('order_id', 'N/A')} | "
                    f"Customer State: {metadata.get('customer_state', 'N/A')} | "
                    f"Cosine Distance: {item['distance']:.4f}"
                )

        st.info(
            "Answers use retrieved reviews. A handful of retrieved "
            "reviews cannot establish statistics for the entire dataset. "
            "For exact dataset-wide percentages or counts, use the Text-to-SQL Assistant."
        )


def main():
    st.set_page_config(
        page_title="Olist Review Intelligence",
        page_icon="🔎",
        layout="wide",
    )
    render_rag_interface(embedded=False)


if __name__ == "__main__":
    main()
