"""
Unified AI + BI Copilot Orchestrator (`ai/copilot_orchestrator.py`).

Implements the exact 7-Step Architecture Flow from the project architecture diagram:
1. User Query (Natural-language question via Streamlit or Web UI)
2. Query Understanding (LLM / Intent Classifier generates either a SQL query,
   a RAG Search query, or both)
3. Top Path: SQL & HNLP Validation Safety Layer (Allow only read-only SELECT, check approved
   tables/views, block INSERT/UPDATE/DELETE/DROP and natural-language / compound HNLP
   read+modify requests, prevent multiple statements, parse & validate SQL;
   if invalid/unsafe -> Query Rejected with safe message — NEVER falls back to a default query!)
   Bottom Path: RAG - Retrieve Relevant Reviews from ChromaDB vector store
4. Top Path: Analytical Database (MotherDuck / DuckDB with Read-Only guard)
   Bottom Path: Relevant Evidence (Top relevant reviews with metadata)
5. Results: Actual results from trusted MotherDuck data returned to the LLM
6. Answer Generation (LLM): Uses the actual SQL query results (with exact Python-precomputed
   percentages so the LLM never approximates mental math) and/or retrieved review evidence
   to generate a clear, accurate, well-explained answer with references
7. Final Answer (Streamlit): Answer with key insights, interactive Plotly visualisations,
   and source evidence shown to the user.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

AI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AI_DIR.parent
for _p in (str(PROJECT_ROOT), str(AI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(PROJECT_ROOT / ".env", override=True)

SQL_KEYWORDS = [
    "top", "highest", "lowest", "total", "sum", "average", "avg", "count",
    "how many", "revenue", "sales", "performing", "performance", "orders",
    "sellers", "state", "states", "category", "categories", "payment",
    "installments", "delay", "delays", "deliveries", "delivery doing",
    "trend", "monthly", "month", "repeat", "retention", "distribution",
    "percentage", "percent", "share", "rank", "ranking", "compare", "show",
]

RAG_OPINION_KEYWORDS = [
    "why", "complain", "complaint", "complaints", "say", "saying", "opinion",
    "opinions", "feedback", "review", "reviews", "unhappy", "happy",
    "dissatisfied", "satisfied", "broken", "defective", "quality",
    "late delivery", "packaging", "damaged", "wrong item", "experience",
    "experiences", "issue", "issues", "problem", "problems", "customers think",
    "what do customers", "what are customers saying", "reason", "reasons",
]

# Comprehensive Multi-Pattern Guardrail:
# Catches:
# 1. Compound HNLP read+write / read+modify queries (e.g., "read and modify teh some values",
#    "show top sales and update the values", "get orders then change the price", "fetch and edit...")
# 2. Direct natural-language mutation requests ("modify some values", "edit the sales table",
#    "update the review scores", "change the values in mart_sales", "set price to 0",
#    "insert new records", "add a row", "delete/drop/truncate/wipe/erase/overwrite...")
# 3. Raw SQL DDL/DML commands anywhere in the prompt (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, etc.)
UNSAFE_SQL_OR_NL_PATTERN = re.compile(
    r"("
    # (A) Compound HNLP Read + Modify/Write queries ("read and modify...", "show ... and update...", etc.)
    r"\b(?:read|show|get|fetch|select|view|display|find|list|query|check|retrieve|see|give|tell)\b.*"
    r"\b(?:and|then|also|plus|after\s+that|before|while|to)\b\s*"
    r"(?:please\s+|can\s+you\s+|could\s+you\s+|try\s+to\s+|also\s+|now\s+)*"
    r"\b(?:modify|modifying|modified|update|updating|updated|change|changing|edit|editing|"
    r"delete|deleting|drop|dropping|truncate|truncating|insert|inserting|add|adding|"
    r"append|appending|set|setting|replace|replacing|alter|altering|remove|removing|"
    r"overwrite|overwriting|write|writing|clear|clearing|wipe|wiping|erase|erasing|"
    r"reset|resetting|mutate|mutating|tamper|corrupt|manipulate)\b"
    r"|"
    # (B) Direct NL write/modify verbs followed (within 0..6 tokens, including typos like 'teh') by data/target nouns
    r"\b(?:modify|modifying|modified|edit|editing|edited|overwrite|overwriting|mutate|mutating|"
    r"tamper|tampering|corrupt|corrupting|manipulate|manipulating|delete|deleting|drop|dropping|"
    r"truncate|truncating|destroy|destroying|wipe|wiping|erase|erasing|remove|removing|"
    r"clear|clearing|reset|resetting|update|updating|alter|altering|insert|inserting|"
    r"replace|replacing)\b"
    r"(?:\s+[a-zA-Z0-9_]+){0,6}\s+"
    r"\b(?:value|values|val|vals|data|database|db|table|tables|record|records|row|rows|"
    r"column|columns|col|cols|field|fields|entry|entries|cell|cells|mart|marts|schema|schemas|"
    r"view|views|order|orders|review|reviews|sale|sales|price|prices|freight|payment|payments|"
    r"seller|sellers|customer|customers|product|products|category|categories|score|scores|"
    r"rating|ratings|status|revenue|delay|delays|sentiment|everything|all|it|them)\b"
    r"|"
    # (C) Contextual 'change' or 'set' targeting values/columns/rows (without blocking 'how did sales change over time')
    r"\b(?:change|changing|set|setting)\b\s+"
    r"(?:the|teh|some|all|any|these|those|existing|current|a|an|every|each|few|certain|specific)\s+"
    r"(?:[a-zA-Z0-9_]+\s+){0,3}"
    r"\b(?:value|values|val|vals|data|database|db|table|tables|record|records|row|rows|"
    r"column|columns|field|fields|entry|entries|price|prices|freight|score|scores|"
    r"rating|ratings|status|revenue|sales|orders|reviews|payment|payments|seller|sellers|"
    r"customer|customers|product|products|category|categories|mart|marts)\b"
    r"|"
    # (D) Assignment / mutation phrases like "set/change/update/modify/edit <col/value> to/= <val>"
    r"\b(?:change|changing|set|setting|update|updating|modify|modifying|edit|editing|overwrite)\b\s+"
    r"(?:the\s+|teh\s+|some\s+|all\s+)?[a-zA-Z0-9_.]+\s+(?:to|=)\s+[a-zA-Z0-9_.'\"-]+"
    r"|"
    # (E) Adding/inserting/appending/writing new rows, records, or values into tables/database
    r"\b(?:add|adding|append|appending|insert|inserting|write|writing)\b\s+"
    r"(?:a\s+|an\s+|the\s+|teh\s+|new\s+|some\s+|more\s+|extra\s+|fake\s+|dummy\s+)*"
    r"\b(?:row|rows|record|records|entry|entries|value|values|data|column|columns|table|tables)\b"
    r"|"
    # (F) Raw SQL DDL / DML statements or standalone modify/edit/delete/drop/update commands
    r"\b(?:delete\s+from|drop\s+table|drop\s+database|drop\s+schema|drop\s+view|truncate\s+table|"
    r"update\s+[a-zA-Z0-9_.]+\s+set|insert\s+into|replace\s+into|merge\s+into|alter\s+table|"
    r"create\s+table|create\s+or\s+replace|grant\s+|revoke\s+|copy\s+.*\s+(?:to|from)|"
    r"attach\s+|detach\s+)\b"
    r"|"
    r"^\s*(?:delete|drop|truncate|alter|insert|update|modify|edit|overwrite|wipe|erase|destroy)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

UNAUTHORIZED_SCHEMA_PATTERN = re.compile(
    r"\b(?:olist\.)?(?:raw|staging|core|information_schema|pg_catalog)\.[a-zA-Z0-9_]+|"
    r"\braw_(?:orders|customers|order_items|order_payments|order_reviews|products|sellers|geolocation)\b|"
    r"\bstg_(?:orders|customers|order_items|order_payments|order_reviews|products|sellers|geolocation)\b|"
    r"\bcore_(?:orders|order_items|order_payments|order_reviews|geolocation|product_categories)\b",
    re.IGNORECASE,
)


def detect_unsafe_user_prompt(prompt: str) -> str | None:
    """
    Step 3 Safety Layer Pre-Check:
    Detects:
      1. Compound HNLP read+modify / read+write queries (e.g., "read and modify some values")
      2. Natural-language requests to modify, edit, change, update, insert, delete, drop, or wipe data
      3. Raw SQL DDL/DML statements (DELETE, DROP, TRUNCATE, UPDATE, INSERT, ALTER, CREATE)
      4. Requests to query unapproved raw/staging/core/system schemas.
    Returns a descriptive rejection message if unsafe, or None if safe.
    """
    cleaned = (prompt or "").strip()
    if not cleaned:
        return None

    if UNSAFE_SQL_OR_NL_PATTERN.search(cleaned):
        return (
            "Query Rejected by Step 3 SQL & HNLP Safety Layer: Read-and-modify, write, or "
            "destructive operations (MODIFY, UPDATE, CHANGE, EDIT, INSERT, DELETE, DROP, "
            "TRUNCATE, ALTER) are strictly forbidden. Even when combined with a read request "
            "(e.g., 'read and modify values'), the entire query is blocked because the "
            "MotherDuck warehouse enforces a strict read-only SELECT guard."
        )

    if UNAUTHORIZED_SCHEMA_PATTERN.search(cleaned):
        return (
            "Query Rejected by Step 3 SQL Safety Layer: Access to unapproved tables or "
            "schemas (`raw`, `staging`, `core`, `system`) is blocked. Only approved "
            "analytical tables in `olist.marts` are permitted."
        )

    return None


def classify_prompt_routing(
    prompt: str,
    mode_override: str = "Auto (Smart Hybrid: SQL + RAG + Charts)",
) -> dict[str, Any]:
    """
    Step 2 — Query Understanding:
    Determine whether the user's question requires:
    - `sql_only` (Analytical SQL query over MotherDuck marts)
    - `rag_only` (Semantic search over ChromaDB review embeddings)
    - `hybrid` (Both SQL query and RAG review search)
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    norm_mode = (mode_override or "").strip().upper()
    if mode_override == "Force Hybrid (Always Run SQL + RAG + Charts)" or norm_mode in {"HYBRID", "FORCE_HYBRID"}:
        return {
            "route_type": "hybrid",
            "run_sql": True,
            "run_rag": True,
            "is_qualitative_primary": False,
            "reason": "Force Hybrid mode active -> generating both SQL query and RAG search query.",
        }
    if mode_override == "SQL Warehouse + Auto-Charts Only" or norm_mode in {"SQL", "SQL_ONLY"}:
        return {
            "route_type": "sql_only",
            "run_sql": True,
            "run_rag": False,
            "is_qualitative_primary": False,
            "reason": "SQL Warehouse mode active -> generating SQL query for MotherDuck.",
        }
    if mode_override == "Customer Review RAG + Evidence Charts Only" or norm_mode in {"RAG", "RAG_ONLY"}:
        return {
            "route_type": "rag_only",
            "run_sql": False,
            "run_rag": True,
            "is_qualitative_primary": True,
            "reason": "Customer Review RAG mode active -> generating semantic search query for ChromaDB.",
        }

    lower_p = prompt.lower()
    has_sql_signal = any(kw in lower_p for kw in SQL_KEYWORDS)
    has_rag_signal = any(kw in lower_p for kw in RAG_OPINION_KEYWORDS)

    if has_sql_signal and has_rag_signal:
        return {
            "route_type": "hybrid",
            "run_sql": True,
            "run_rag": True,
            "is_qualitative_primary": False,
            "reason": "Query Understanding detected both analytical metrics and customer review opinions -> running SQL Path + RAG Path.",
        }

    if has_rag_signal and not has_sql_signal:
        return {
            "route_type": "hybrid",
            "run_sql": True,
            "run_rag": True,
            "is_qualitative_primary": True,
            "reason": "Query Understanding detected a customer review/opinion question -> running RAG Review Search + Companion MotherDuck SQL for visual charts.",
        }

    return {
        "route_type": "sql_only",
        "run_sql": True,
        "run_rag": False,
        "is_qualitative_primary": False,
        "reason": "Query Understanding detected a numerical/analytical question -> running SQL Validation + MotherDuck Query + Auto-Visualisation.",
    }


def build_companion_sql_question(user_prompt: str) -> str:
    """
    Build a companion analytical SQL question ONLY when the user asks a genuine
    qualitative review question so Step 7 displays both review insights and warehouse graphs.
    """
    lower_p = user_prompt.lower()
    if any(w in lower_p for w in ["delivery", "deliveries", "late", "delay", "shipping", "arrive"]):
        return (
            "What is the average delivery delay by customer state in mart_delivery_performance "
            "(top 15 states ordered by avg delivery delay desc)?"
        )
    if any(w in lower_p for w in ["quality", "product", "defective", "broken", "category"]):
        return (
            "Show the top 15 product categories by total sales value and total items sold "
            "from mart_product_performance."
        )
    if any(w in lower_p for w in ["seller", "merchant", "wrong item", "service"]):
        return "Show the top 10 sellers by total product revenue from mart_seller_performance."
    if any(w in lower_p for w in ["payment", "installment", "credit", "boleto", "voucher"]):
        return "Show total orders and total payment value by payment_type from mart_payment_analysis."
    return (
        "Show the distribution of ai_sentiment and review_count in mart_enriched_review_analysis."
    )


def enrich_df_with_exact_percentages(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Precompute exact totals and exact percentage shares for numeric columns in `df`
    so the LLM in Step 6 NEVER makes mental-math rounding errors (e.g. reporting
    1,908 / 3,011 as 66% instead of the exact 63.37%).
    """
    if df is None or df.empty:
        return df, {}

    enriched = df.copy()
    totals: dict[str, float] = {}

    if len(enriched) > 1 and len(enriched) <= 30:
        for col in list(enriched.columns):
            if pd.api.types.is_numeric_dtype(enriched[col]) and not pd.api.types.is_bool_dtype(enriched[col]):
                col_lower = str(col).lower()
                if any(k in col_lower for k in ("count", "sales", "revenue", "value", "orders", "items", "reviews")):
                    col_sum = float(enriched[col].sum())
                    if col_sum > 0:
                        totals[col] = round(col_sum, 2)
                        pct_col = f"{col}_pct_of_total"
                        enriched[pct_col] = (100.0 * enriched[col] / col_sum).round(2)

    return enriched, totals


def generate_llm_sql_explanation(
    question: str,
    executed_sql: str,
    df: pd.DataFrame,
) -> str:
    """
    Step 6 — Answer Generation (LLM) for Analytical SQL Results:
    Takes the actual trusted results from MotherDuck (Step 5) enriched with exact
    Python-computed totals and percentages (`*_pct_of_total`) and generates a clear,
    accurate natural-language explanation.
    """
    if df is None or df.empty:
        return "The query executed successfully on MotherDuck but returned 0 rows."

    enriched_df, col_totals = enrich_df_with_exact_percentages(df)
    preview_rows = enriched_df.head(15).to_dict(orient="records")
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if api_key:
        try:
            from groq import Groq

            client = Groq(api_key=api_key)
            prompt = f"""
You are a senior e-commerce data analyst for the Olist Brazilian E-Commerce platform.
A user asked the following question:
"{question}"

The validated SQL query executed on MotherDuck (`olist.marts`) was:
{executed_sql}

Total rows returned: {len(df)}
Precomputed column totals (exact Python math): {col_totals}
Top result rows from MotherDuck (including exact `*_pct_of_total` shares where applicable):
{preview_rows}

Write a clear, accurate, and concise natural-language answer (3 to 5 bullet points)
explaining the key findings directly from these exact numbers.
Strict Rules:
- Use ONLY the exact numbers and exact `*_pct_of_total` percentages present above.
- NEVER do mental arithmetic or round `63.37%` to `66%` or `7.24%` to `3%`.
- Cite the exact figures and percentages to 2 decimal places when available.
"""
            resp = client.chat.completions.create(
                model=os.getenv("TEXT_TO_SQL_MODEL", "openai/gpt-oss-120b"),
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception:
            pass

    # Deterministic fallback when running offline/tests
    top_row = enriched_df.iloc[0].to_dict()
    top_str = ", ".join(
        f"**{k}**: `{v:,.2f}`" if isinstance(v, (float, np.floating)) else f"**{k}**: `{v}`"
        for k, v in list(top_row.items())[:5]
    )
    return (
        f"Based on the trusted MotherDuck query results (`{len(df)}` rows returned, "
        f"totals: `{col_totals}`), the leading record is {top_str}."
    )


def synthesize_unified_insight(
    prompt: str,
    sql_result: dict[str, Any] | None,
    rag_result: dict[str, Any] | None,
    llm_sql_narrative: str | None = None,
) -> str:
    """
    Step 6 & 7 — Combine the LLM-generated explanation of MotherDuck SQL results
    and the RAG-grounded review answer with `[review_id]` references.
    """
    parts: list[str] = []

    if sql_result and sql_result.get("df") is not None:
        df: pd.DataFrame = sql_result["df"]
        narrative = llm_sql_narrative or generate_llm_sql_explanation(
            prompt,
            sql_result.get("safe_sql", ""),
            df,
        )
        parts.append(
            f"**📊 Analytical Warehouse Insights (`{', '.join(sql_result.get('relevant_tables', ['olist.marts']))}` — "
            f"`{len(df)}` rows from MotherDuck):**\n\n{narrative}"
        )

    if rag_result and rag_result.get("answer"):
        parts.append(
            f"**🔎 Customer Review Evidence (Opinion-Aware RAG — "
            f"`K={len(rag_result.get('retrieved', []))}` reviews retrieved, "
            f"`{len(rag_result.get('citations', []))}` inline `[review_id]` citations):**\n\n"
            f"{rag_result['answer']}"
        )

    return "\n\n---\n\n".join(parts) if parts else "Query completed."


_CACHED_RAG_BUNDLE = None


def _get_or_init_rag_bundle():
    global _CACHED_RAG_BUNDLE
    if _CACHED_RAG_BUNDLE is None:
        if st is not None:
            try:
                from rag_app import load_rag_components
                _CACHED_RAG_BUNDLE = load_rag_components()
                return _CACHED_RAG_BUNDLE
            except Exception:
                pass
        from rag_reviews import initialize_rag
        _CACHED_RAG_BUNDLE = initialize_rag()
    return _CACHED_RAG_BUNDLE


def execute_connected_copilot_pipeline(
    user_query: str,
    routing_override: str = "AUTO",
) -> dict[str, Any]:
    """
    Executes the full 7-Step Connected Copilot Pipeline for `app.py` and programmatic callers,
    returning a structured dictionary with safety status, SQL DataFrame, explanations, and RAG docs.
    """
    cleaned_q = (user_query or "").strip()
    if not cleaned_q:
        return {
            "safety_blocked": True,
            "safety_error": "Please enter a non-empty question.",
            "route": "BLOCKED",
            "sql_executed": False,
            "sql_df": pd.DataFrame(),
            "sql_query": "",
            "sql_explanation": "",
            "rag_docs": [],
            "rag_answer": "",
            "rag_citations": [],
        }

    # Step 3 Pre-Check: Block any HNLP read+modify, write, DDL/DML, or unauthorized schema request
    rejection_msg = detect_unsafe_user_prompt(cleaned_q)
    if rejection_msg:
        return {
            "safety_blocked": True,
            "safety_error": rejection_msg,
            "route": "BLOCKED",
            "sql_executed": False,
            "sql_df": pd.DataFrame(),
            "sql_query": "",
            "sql_explanation": "",
            "rag_docs": [],
            "rag_answer": "",
            "rag_citations": [],
        }

    # Step 2: Query Understanding & Routing
    route_info = classify_prompt_routing(cleaned_q, mode_override=routing_override)
    route_label = str(route_info.get("route_type", "hybrid")).upper()

    sql_executed = False
    sql_df = pd.DataFrame()
    sql_query = ""
    sql_explanation = ""
    sql_result_bundle = None

    # Step 3 & 4 (Top Path): Guarded Text-to-SQL on MotherDuck
    if route_info.get("run_sql"):
        from text_to_sql_app import is_security_violation_error, run_agentic_text_to_sql

        try:
            if route_info.get("is_qualitative_primary"):
                target_sql_q = build_companion_sql_question(cleaned_q)
                out = run_agentic_text_to_sql(target_sql_q, max_Corrections=1)
            else:
                out = run_agentic_text_to_sql(cleaned_q, max_Corrections=2)
            sql_df = out["df"]
            sql_query = out["safe_sql"]
            sql_executed = True
            sql_explanation = generate_llm_sql_explanation(cleaned_q, sql_query, sql_df)
            sql_result_bundle = out
        except Exception as sql_exc:
            err_str = str(sql_exc)
            if is_security_violation_error(err_str):
                return {
                    "safety_blocked": True,
                    "safety_error": err_str,
                    "route": "BLOCKED",
                    "sql_executed": False,
                    "sql_df": pd.DataFrame(),
                    "sql_query": "",
                    "sql_explanation": "",
                    "rag_docs": [],
                    "rag_answer": "",
                    "rag_citations": [],
                }
            if route_info.get("is_qualitative_primary"):
                try:
                    comp_q = build_companion_sql_question(cleaned_q)
                    out = run_agentic_text_to_sql(comp_q, max_Corrections=1)
                    sql_df = out["df"]
                    sql_query = out["safe_sql"]
                    sql_executed = True
                    sql_explanation = generate_llm_sql_explanation(comp_q, sql_query, sql_df)
                    sql_result_bundle = out
                except Exception:
                    pass
            elif not route_info.get("run_rag"):
                return {
                    "safety_blocked": True,
                    "safety_error": err_str,
                    "route": route_label,
                    "sql_executed": False,
                    "sql_df": pd.DataFrame(),
                    "sql_query": "",
                    "sql_explanation": "",
                    "rag_docs": [],
                    "rag_answer": "",
                    "rag_citations": [],
                }

    # Step 3 & 4 (Bottom Path): Opinion-Aware Multi-Stage RAG on ChromaDB
    rag_docs: list[dict[str, Any]] = []
    rag_answer = ""
    rag_citations: list[str] = []
    rag_result_bundle = None

    if route_info.get("run_rag"):
        try:
            from rag_reviews import generate_answer, retrieve_reviews

            embedding_model, collection, llm = _get_or_init_rag_bundle()
            retrieved = retrieve_reviews(
                cleaned_q,
                embedding_model,
                collection,
                top_k=None,
                adaptive=True,
            )
            rag_answer, rag_citations = generate_answer(cleaned_q, retrieved, llm)
            rag_result_bundle = {
                "answer": rag_answer,
                "retrieved": retrieved,
                "citations": rag_citations,
            }
            for item in retrieved:
                meta = item.get("metadata") or {}
                rag_docs.append(
                    {
                        "review_id": item.get("review_id", ""),
                        "rating": meta.get("review_score", ""),
                        "ai_sentiment": meta.get("ai_sentiment", ""),
                        "ai_topic": meta.get("ai_topic", ""),
                        "ai_issue_category": meta.get("ai_issue_category", ""),
                        "product_category": meta.get("product_category", ""),
                        "customer_state": meta.get("customer_state", ""),
                        "delivery_status": meta.get("delivery_status", ""),
                        "relevance_score": item.get("relevance_score", 0.0),
                        "review_evidence": item.get("document", ""),
                    }
                )
        except Exception as rag_exc:
            rag_answer = f"RAG retrieval note: {rag_exc}"

    unified_insight = synthesize_unified_insight(
        prompt=cleaned_q,
        sql_result=sql_result_bundle,
        rag_result=rag_result_bundle,
        llm_sql_narrative=sql_explanation or None,
    )

    return {
        "safety_blocked": False,
        "safety_error": None,
        "route": route_label,
        "route_reason": route_info.get("reason", ""),
        "sql_executed": sql_executed,
        "sql_df": sql_df,
        "sql_query": sql_query,
        "sql_explanation": sql_explanation,
        "rag_docs": rag_docs,
        "rag_answer": rag_answer,
        "rag_citations": rag_citations,
        "unified_insight": unified_insight,
    }


def run_connected_copilot(
    user_query: str | None = None,
    routing_override: str = "AUTO",
) -> dict[str, Any] | None:
    """
    Dual-mode entry point:
    - When called with `user_query` (e.g., from `app.py`), executes the 7-step connected
      pipeline and returns the result dictionary.
    - When called without arguments (standalone UI), renders the interactive Streamlit chat.
    """
    if user_query is not None:
        return execute_connected_copilot_pipeline(
            user_query=user_query,
            routing_override=routing_override,
        )

    if st is None:
        return None

    st.header("Unified AI + BI Copilot Orchestrator")
    st.markdown("Ask natural language questions to generate SQL, search reviews, or both.")

    mode_override = st.radio(
        "Routing Mode:",
        options=[
            "Auto (Smart Hybrid: SQL + RAG + Charts)",
            "Force Hybrid (Always Run SQL + RAG + Charts)",
            "SQL Warehouse + Auto-Charts Only",
            "Customer Review RAG + Evidence Charts Only",
        ],
        horizontal=True,
    )

    prompt = st.chat_input(
        "E.g., What are the top product categories by sales, and what do customers complain about them?"
    )
    if prompt:
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            res = execute_connected_copilot_pipeline(prompt, routing_override=mode_override)
            if res.get("safety_blocked"):
                st.error(f"❌ {res.get('safety_error')}")
                return res
            st.info(f"**Execution Route ({res.get('route')}):** {res.get('route_reason')}")
            st.markdown(res.get("unified_insight", ""))
            return res
    return None