"""
Agentic & Schema-Aware Text-to-SQL Assistant for Olist MotherDuck Warehouse
(Module M7 - FR_8, NFR_3, Improvement 3).

Pipeline:
User Question -> HNLP & SQL Prompt Safety Guard -> Query Understanding & Schema Retrieval
-> Few-Shot Olist Examples -> Groq SQL Generation -> SQLGlot AST Validation & Read-Only Guard
-> MotherDuck Execution -> Self-Correction Retry Loop (Syntax/Schema only, never bypasses safety!)
-> Automatic Plotly Graph Visualization -> CSV Export.

Can be run standalone (`streamlit run ai/text_to_sql_app.py --server.port 8504`)
or embedded inside the unified platform (`app.py`).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # Allows importing SQL validation & benchmark functions in non-UI environments

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    import sqlglot
    from sqlglot import exp
    HAS_SQLGLOT = True
except ImportError:
    sqlglot = None
    exp = None
    HAS_SQLGLOT = False

AI_DIR = Path(__file__).resolve().parent
ROOT = AI_DIR.parent
for _p in (str(ROOT), str(AI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(ROOT / ".env", override=True)

from auto_visualizer import render_auto_visualizations
from copilot_orchestrator import detect_unsafe_user_prompt
from db_utils import assert_read_only_sql

# Approved analytical marts (9 core marts) + Dimensional Star Schema tables in `olist.marts`
APPROVED_MART_TABLES = {
    "mart_sales",
    "mart_customer_behavior",
    "mart_customer_reviews",
    "mart_delivery_performance",
    "mart_enriched_review_analysis",
    "mart_payment_analysis",
    "mart_product_performance",
    "mart_review_analysis",
    "mart_seller_performance",
}

APPROVED_STAR_TABLES = {
    "fact_order_items",
    "fact_sales",
    "fact_payments",
    "fact_reviews",
    "dim_customer",
    "dim_product",
    "dim_seller",
    "dim_date",
    "dim_location",
    "dim_payment",
    "dim_review",
    "olap_sales_cube",
}

APPROVED_TABLES = APPROVED_MART_TABLES | APPROVED_STAR_TABLES

TABLE_SCHEMAS: dict[str, str] = {
    "mart_sales": """mart_sales (Grain: 1 row per order_id + order_item_id; use COUNT(DISTINCT order_id) for order count):
order_id VARCHAR, order_item_id BIGINT, product_id VARCHAR,
seller_id VARCHAR, price DOUBLE, freight_value DOUBLE,
total_item_value DOUBLE, product_category_name VARCHAR,
seller_state VARCHAR, customer_id VARCHAR, customer_unique_id VARCHAR,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
customer_state VARCHAR""",
    "mart_customer_behavior": """mart_customer_behavior (Grain: 1 row per unique customer_unique_id):
customer_unique_id VARCHAR, total_orders BIGINT,
first_order_date TIMESTAMP, last_order_date TIMESTAMP""",
    "mart_customer_reviews": """mart_customer_reviews (Grain: 1 row per customer_unique_id):
customer_unique_id VARCHAR, total_reviews BIGINT,
average_review_score DOUBLE, positive_reviews HUGEINT,
neutral_reviews HUGEINT, negative_reviews HUGEINT""",
    "mart_delivery_performance": """mart_delivery_performance (Grain: 1 row per order_id):
order_id VARCHAR, customer_id VARCHAR, customer_unique_id VARCHAR,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
order_delivered_carrier_date TIMESTAMP,
order_delivered_customer_date TIMESTAMP,
order_estimated_delivery_date TIMESTAMP, customer_city VARCHAR,
customer_state VARCHAR, delivery_delay_days BIGINT""",
    "mart_enriched_review_analysis": """mart_enriched_review_analysis (Grain: AI-enriched reviews, 3,011 rows):
review_id VARCHAR, order_id VARCHAR, customer_id VARCHAR,
customer_unique_id VARCHAR, review_score BIGINT,
review_comment_title VARCHAR, review_comment_message VARCHAR,
review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
customer_city VARCHAR, customer_state VARCHAR,
review_sentiment_category VARCHAR, ai_sentiment VARCHAR,
ai_topic VARCHAR, ai_summary VARCHAR, ai_issue_category VARCHAR,
review_language VARCHAR, ai_enriched_at TIMESTAMP""",
    "mart_payment_analysis": """mart_payment_analysis (Grain: 1 row per payment_type):
payment_type VARCHAR, total_orders BIGINT,
total_payment_value DOUBLE, average_payment_value DOUBLE,
average_installments DOUBLE""",
    "mart_product_performance": """mart_product_performance (Grain: 1 row per product_id):
product_id VARCHAR, product_category_name VARCHAR,
total_orders BIGINT, total_items_sold BIGINT,
total_product_revenue DOUBLE, total_freight_value DOUBLE,
total_sales_value DOUBLE, average_item_price DOUBLE""",
    "mart_review_analysis": """mart_review_analysis (Grain: all order reviews):
review_id VARCHAR, order_id VARCHAR, customer_id VARCHAR,
customer_unique_id VARCHAR, review_score BIGINT,
review_comment_title VARCHAR, review_comment_message VARCHAR,
review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
customer_city VARCHAR, customer_state VARCHAR,
review_sentiment_category VARCHAR""",
    "mart_seller_performance": """mart_seller_performance (Grain: 1 row per seller_id):
seller_id VARCHAR, seller_state VARCHAR, total_orders BIGINT,
total_items_sold BIGINT, total_product_revenue DOUBLE,
total_freight_value DOUBLE, total_sales_value DOUBLE,
average_item_price DOUBLE""",
}

SCHEMA = (
    "Database: olist_db (attached as `olist`)\nSchema: marts\n\n"
    + "\n\n".join(TABLE_SCHEMAS.values())
)

FEW_SHOT_EXAMPLES = """
Few-Shot Domain Examples (Olist E-Commerce):

Question: "How are sales performing?"
SQL:
SELECT
    STRFTIME(order_purchase_timestamp, '%Y-%m') AS year_month,
    SUM(total_item_value) AS total_sales_value,
    COUNT(DISTINCT order_id) AS total_orders
FROM olist.marts.mart_sales
WHERE order_purchase_timestamp IS NOT NULL
GROUP BY year_month
ORDER BY year_month
LIMIT 200

Question: "How are deliveries doing?"
SQL:
SELECT
    customer_state,
    COUNT(*) AS total_orders,
    AVG(delivery_delay_days) AS avg_delivery_delay_days
FROM olist.marts.mart_delivery_performance
WHERE delivery_delay_days IS NOT NULL
GROUP BY customer_state
ORDER BY total_orders DESC
LIMIT 200

Question: "Which product categories have the highest sales value?"
SQL:
SELECT
    product_category_name,
    SUM(total_sales_value) AS category_sales_value
FROM olist.marts.mart_product_performance
GROUP BY product_category_name
ORDER BY category_sales_value DESC
LIMIT 200

Question: "What is the average delivery delay by customer state?"
SQL:
SELECT
    customer_state,
    AVG(delivery_delay_days) AS avg_delivery_delay_days
FROM olist.marts.mart_delivery_performance
WHERE delivery_delay_days IS NOT NULL
GROUP BY customer_state
ORDER BY avg_delivery_delay_days DESC
LIMIT 200

Question: "Show the top 10 sellers by total product revenue."
SQL:
SELECT
    seller_id,
    seller_state,
    total_product_revenue
FROM olist.marts.mart_seller_performance
ORDER BY total_product_revenue DESC
LIMIT 10

Question: "How many repeat customers have more than one order?"
SQL:
SELECT
    COUNT(*) AS repeat_customers
FROM olist.marts.mart_customer_behavior
WHERE total_orders > 1

Question: "Show the distribution of AI sentiment in enriched reviews."
SQL:
SELECT
    ai_sentiment,
    COUNT(*) AS review_count
FROM olist.marts.mart_enriched_review_analysis
GROUP BY ai_sentiment
ORDER BY review_count DESC
"""

SYSTEM_PROMPT = f"""
You are a careful analytics SQL assistant for Brazilian Olist e-commerce data.

Generate exactly one DuckDB SQL SELECT query for the user's question.
Use only tables and columns in this schema:

{SCHEMA}

{FEW_SHOT_EXAMPLES}

Rules:
- CRITICAL SAFETY RULE: If the user's question asks to modify, update, change, edit,
  insert, add, delete, drop, truncate, alter, or overwrite any data, values, rows, or tables
  (including compound requests like "read and modify some values" or "show sales and update values"),
  DO NOT generate a SELECT query for the read part. Instead, return ONLY the exact token:
  UNSAFE_MUTATION_REQUEST
- Only SELECT queries are allowed.
- Use only the listed tables in the marts schema.
- Fully qualify tables as olist.marts.table_name.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY,
  ATTACH, DETACH, INSTALL, LOAD, EXPORT, IMPORT, PRAGMA, or CALL.
- Never query raw, staging, core, system, or information_schema tables.
- Never guess column names.
- Pay strict attention to table grain: `mart_sales` has one row per order item,
  so use `COUNT(DISTINCT order_id)` when counting distinct orders in `mart_sales`.
- Use a reasonable LIMIT of at most 200 rows unless the query is
  a scalar aggregate.
- Return only SQL, without Markdown fences or explanations.
- For revenue, use the appropriate mart's documented revenue fields.
- Do not claim that results are causal or statistically representative.
"""

SECURITY_ERROR_MARKERS = (
    "safety layer",
    "disallowed sql operation",
    "only select statements are allowed",
    "only one sql statement is allowed",
    "table is not approved",
    "only the marts schema is allowed",
    "approved motherduck database/catalog",
    "unsafe_mutation_request",
    "read-only guard blocked",
    "strictly forbidden",
)


def is_security_violation_error(error_message: str) -> bool:
    """Return True if an exception message represents a security/guardrail rejection."""
    msg = (error_message or "").lower()
    return any(marker in msg for marker in SECURITY_ERROR_MARKERS)


def select_relevant_tables(question: str) -> list[str]:
    """
    Stage 1 (Query Understanding & Schema Retrieval):
    Identify the primary marts relevant to the user's question for transparency.
    """
    q = question.lower()
    matched: list[str] = []

    keyword_map = {
        "mart_product_performance": ["category", "categories", "product", "products"],
        "mart_seller_performance": ["seller", "sellers", "merchant"],
        "mart_delivery_performance": ["delivery", "deliveries", "delay", "delayed", "late", "shipping", "carrier", "estimated"],
        "mart_enriched_review_analysis": ["ai sentiment", "ai_sentiment", "topic", "issue", "enriched", "complaint", "language", "summary"],
        "mart_review_analysis": ["review", "reviews", "score", "rating", "satisfaction"],
        "mart_customer_reviews": ["customer review", "positive_reviews", "negative_reviews"],
        "mart_customer_behavior": ["repeat", "retention", "first_order", "last_order", "total_orders", "loyal"],
        "mart_payment_analysis": ["payment", "installment", "credit_card", "boleto", "voucher", "debit"],
        "mart_sales": ["sales", "revenue", "performing", "monthly", "trend", "month", "year", "freight", "order"],
    }

    for table_name, keywords in keyword_map.items():
        if any(kw in q for kw in keywords):
            matched.append(table_name)

    return matched or ["mart_sales", "mart_product_performance"]


def get_groq_client():
    from groq import Groq

    load_dotenv(ROOT / ".env", override=True)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing from the project .env file.")
    return Groq(api_key=key)


def clean_llm_sql(raw_sql: str) -> str:
    """Strip Markdown code fences and trailing semicolons from LLM output."""
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", (raw_sql or "").strip(), flags=re.I)
    return sql.strip().rstrip(";")


def generate_sql(
    question: str,
    previous_sql: str | None = None,
    error_message: str | None = None,
) -> str:
    """
    Generate SQL for `question`.
    First enforces `detect_unsafe_user_prompt(question)` so HNLP read+modify or
    destructive requests are blocked before calling the LLM.
    """
    rejection = detect_unsafe_user_prompt(question)
    if rejection:
        raise ValueError(rejection)

    client = get_groq_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if previous_sql and error_message:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": previous_sql})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"The generated SQL failed execution with column/syntax error:\n"
                    f"{error_message}\n\n"
                    f"Please fix the SQL query so it strictly uses only approved tables "
                    f"(`olist.marts.<table_name>`) and valid column names from the schema. "
                    f"Return ONLY the corrected SELECT query."
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=os.getenv("TEXT_TO_SQL_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        messages=messages,
    )
    sql = response.choices[0].message.content or ""
    cleaned_sql = clean_llm_sql(sql)
    if "UNSAFE_MUTATION_REQUEST" in cleaned_sql.upper():
        raise ValueError(
            "Query Rejected by Step 3 SQL & HNLP Safety Layer: Request attempts to modify "
            "or mutate warehouse data, which is strictly forbidden under the read-only guard."
        )
    return cleaned_sql


def _validate_sql_regex_fallback(sql: str) -> tuple[str, bool]:
    """
    Fallback validator when sqlglot is not installed in a minimal test environment.
    Enforces single statement, SELECT-only, approved `olist.marts.<table_name>`,
    and LIMIT <= 200.
    """
    cleaned = sql.strip().rstrip(";")
    if not cleaned:
        raise ValueError("The model returned empty SQL.")
    if "UNSAFE_MUTATION_REQUEST" in cleaned.upper():
        raise ValueError(
            "Query Rejected by Step 3 SQL & HNLP Safety Layer: Disallowed SQL mutation request."
        )
    if ";" in cleaned:
        raise ValueError("Only one SQL statement is allowed.")

    forbidden_kw = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH|"
        r"INSTALL|LOAD|EXPORT|IMPORT|PRAGMA|CALL|TRUNCATE|GRANT|REVOKE|REPLACE|MERGE)\b",
        re.IGNORECASE,
    )
    match = forbidden_kw.search(cleaned)
    if match:
        raise ValueError(f"Disallowed SQL operation: {match.group(1).upper()}.")

    first_token = cleaned.split()[0].upper()
    if first_token not in {"SELECT", "WITH"}:
        raise ValueError("Only SELECT statements are allowed.")

    table_refs = re.findall(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_.\"`]+)",
        cleaned,
        flags=re.IGNORECASE,
    )
    cte_names = {
        m.lower()
        for m in re.findall(r"\b([a-zA-Z0-9_]+)\s+AS\s*\(", cleaned, flags=re.IGNORECASE)
    }

    real_tables = [t.strip('"`') for t in table_refs if t.strip('"`').lower() not in cte_names]
    if not real_tables:
        raise ValueError("Query must reference at least one approved mart.")

    for ref in real_tables:
        parts = [p.lower() for p in ref.split(".")]
        if len(parts) != 3:
            raise ValueError(
                f"Table must be fully qualified as olist.marts.<table_name>: got {ref!r}."
            )
        catalog, schema, name = parts
        if name not in APPROVED_TABLES:
            raise ValueError(f"Table is not approved: {name}.")
        if schema != "marts":
            raise ValueError(f"Only the marts schema is allowed: {schema}.")
        if catalog not in {"olist", "olist_db"}:
            raise ValueError(
                f"Use the approved MotherDuck database/catalog, not {catalog!r}."
            )

    has_agg = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", cleaned, re.IGNORECASE))
    has_group_by = bool(re.search(r"\bGROUP\s+BY\b", cleaned, re.IGNORECASE))
    limit_match = re.search(r"\bLIMIT\s+(\S+)\s*$", cleaned, re.IGNORECASE)

    limit_applied = False
    if limit_match:
        try:
            limit_val = int(limit_match.group(1))
        except ValueError as exc:
            raise ValueError("Use a simple integer LIMIT.") from exc
        if limit_val > 200:
            cleaned = re.sub(r"\bLIMIT\s+\S+\s*$", "LIMIT 200", cleaned, flags=re.IGNORECASE)
            limit_applied = True
    elif not has_agg or has_group_by:
        cleaned = f"{cleaned} LIMIT 200"
        limit_applied = True

    return cleaned, limit_applied


def validate_sql_with_meta(sql: str) -> tuple[str, bool]:
    """
    Parse and allow exactly one SELECT statement over approved marts.
    Also rejects natural-language / HNLP mutation prompts if passed directly.
    Returns `(validated_sql, limit_was_modified)`.
    """
    if not sql or not sql.strip():
        raise ValueError("The model returned empty SQL.")

    if "UNSAFE_MUTATION_REQUEST" in sql.upper():
        raise ValueError(
            "Query Rejected by Step 3 SQL & HNLP Safety Layer: Disallowed mutation request."
        )

    nl_rejection = detect_unsafe_user_prompt(sql)
    if nl_rejection:
        raise ValueError(nl_rejection)

    if not HAS_SQLGLOT:
        return _validate_sql_regex_fallback(sql)

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as exc:
        raise ValueError(f"Could not parse the generated SQL: {exc}") from exc

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Only one SQL statement is allowed.")

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT statements are allowed.")

    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
        exp.Create, exp.Command, exp.Copy, exp.Transaction,
    )
    for node in tree.walk():
        if isinstance(node, forbidden):
            raise ValueError(f"Disallowed SQL operation: {type(node).__name__}.")

    cte_aliases = {
        (cte.alias or "").lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias
    }

    tables = [
        t for t in tree.find_all(exp.Table)
        if not ((t.name or "").lower() in cte_aliases and not t.db and not t.catalog)
    ]
    if not tables:
        raise ValueError("Query must reference at least one approved mart.")

    for table in tables:
        name = (table.name or "").lower()
        schema = (table.db or "").lower()
        catalog = (table.catalog or "").lower()

        if name not in APPROVED_TABLES:
            raise ValueError(f"Table is not approved: {name or '(unnamed)'}.")
        if schema != "marts":
            raise ValueError(f"Only the marts schema is allowed: {schema}.")
        if catalog not in {"olist", "olist_db"}:
            raise ValueError(
                f"Use the approved MotherDuck database/catalog, not {catalog!r}."
            )

    limit_modified = False
    is_scalar_agg = (
        any(isinstance(node, exp.AggFunc) for node in tree.walk())
        and tree.args.get("group") is None
    )

    limit = tree.args.get("limit")
    if limit is not None:
        try:
            limit_expr = limit.expression
            limit_value = int(limit_expr.this)
        except Exception as exc:
            raise ValueError("Use a simple integer LIMIT.") from exc
        if limit_value > 200:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(200)))
            limit_modified = True
    elif not is_scalar_agg:
        tree = tree.limit(200)
        limit_modified = True

    return tree.sql(dialect="duckdb"), limit_modified


def validate_sql(sql: str) -> str:
    """Backward-compatible validator returning the validated SQL string."""
    safe_sql, _ = validate_sql_with_meta(sql)
    return safe_sql


def execute_sql(sql: str) -> pd.DataFrame:
    """Execute a validated SELECT query against MotherDuck (or local test DB)."""
    import duckdb

    # Defense-in-depth: always verify read-only & AST safety before executing
    assert_read_only_sql(sql)
    safe_sql, _ = validate_sql_with_meta(sql)

    load_dotenv(ROOT / ".env", override=True)
    local_db_path = os.getenv("OLIST_LOCAL_DB_PATH", "").strip()
    token = os.getenv("MOTHERDUCK_TOKEN", "").strip()

    if not local_db_path and not token:
        raise RuntimeError("MOTHERDUCK_TOKEN is missing from .env.")

    con = duckdb.connect()
    try:
        if local_db_path:
            con.execute(f"ATTACH '{local_db_path}' AS olist (READ_ONLY)")
        else:
            os.environ["motherduck_token"] = token
            con.execute("LOAD motherduck")
            con.execute("ATTACH 'md:olist_db' AS olist")

        result = con.execute(safe_sql)
        columns = [description[0] for description in result.description]
        rows = result.fetchmany(201)

        if len(rows) > 200:
            rows = rows[:200]

        return pd.DataFrame(rows, columns=columns)
    finally:
        con.close()


def run_agentic_text_to_sql(
    question: str,
    max_Corrections: int = 2,
) -> dict[str, Any]:
    """
    Execute the full Agentic & Schema-Aware Text-to-SQL workflow.
    1. Immediately blocks unsafe natural-language / HNLP read+modify or DDL/DML queries.
    2. Never uses self-correction retries to convert a blocked security/mutation violation
       into a SELECT query; self-correction only retries genuine column/syntax errors.
    """
    start_time = time.perf_counter()

    # Pre-check natural language & HNLP prompt safety BEFORE calling LLM
    prompt_rejection = detect_unsafe_user_prompt(question)
    if prompt_rejection:
        raise ValueError(prompt_rejection)

    relevant_tables = select_relevant_tables(question)
    attempts: list[dict[str, Any]] = []

    current_sql = ""
    last_error = None

    for attempt_idx in range(1, max_Corrections + 2):
        try:
            current_sql = generate_sql(
                question,
                previous_sql=current_sql if attempt_idx > 1 else None,
                error_message=last_error if attempt_idx > 1 else None,
            )
            safe_sql, limit_modified = validate_sql_with_meta(current_sql)
            df = execute_sql(safe_sql)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

            attempts.append(
                {
                    "attempt": attempt_idx,
                    "generated_sql": current_sql,
                    "validated_sql": safe_sql,
                    "status": "success",
                    "error": None,
                }
            )
            return {
                "question": question,
                "relevant_tables": relevant_tables,
                "generated_sql": current_sql,
                "safe_sql": safe_sql,
                "limit_modified": limit_modified,
                "df": df,
                "attempts": attempts,
                "elapsed_ms": elapsed_ms,
            }
        except Exception as exc:
            last_error = str(exc)
            attempts.append(
                {
                    "attempt": attempt_idx,
                    "generated_sql": current_sql,
                    "validated_sql": None,
                    "status": "error",
                    "error": last_error,
                }
            )
            # CRITICAL: If the error is a security/guardrail block, abort immediately!
            if is_security_violation_error(last_error) or attempt_idx >= max_Corrections + 1:
                raise

    raise RuntimeError(last_error or "Unknown Text-to-SQL failure")


def render_text_to_sql_interface(embedded: bool = False) -> None:
    """Render the Agentic & Schema-Aware Text-to-SQL Streamlit UI with Auto-Graphs."""
    if not embedded:
        st.title("📊 Olist Agentic Text-to-SQL + Auto-Visualizer")
    else:
        st.header("📊 Agentic & Schema-Aware Text-to-SQL Assistant")

    st.caption(
        "Ask business questions in natural language. "
        "Groq (`openai/gpt-oss-120b`) generates SQL with few-shot Olist examples, "
        "HNLP + SQLGlot AST guardrails block read+modify / write attempts, DuckDB queries "
        "approved MotherDuck marts, and Plotly automatically visualizes the returned data."
    )

    with st.sidebar:
        st.subheader("Warehouse Guardrails")
        st.write("Database: `olist_db` (`olist`)")
        st.write("Schema: `marts`")
        st.write(f"Approved tables: **{len(APPROVED_TABLES)}** (9 Marts + Star Schema)")
        st.write("Access Policy: **SELECT-only + HNLP Write Block + Max 200 rows**")
        enable_self_correction = st.checkbox(
            "Enable Agentic Error Self-Correction",
            value=True,
            key="sql_self_correction",
        )
        st.divider()
        st.subheader("Example questions")
        examples = [
            "How are sales performing?",
            "How are deliveries doing?",
            "Which product categories have the highest sales value?",
            "What is the average delivery delay by customer state?",
            "Show the top 10 sellers by total product revenue.",
            "Show the distribution of AI sentiment in enriched reviews.",
            "How many repeat customers have more than one order?",
        ]
        for example in examples:
            st.caption(f"• {example}")

    tab_query, tab_schema, tab_eval = st.tabs(
        ["💬 Natural Language to SQL + Graphs", "🗂️ Approved Marts Schema", "🧪 Safety & Benchmark Suite"]
    )

    with tab_query:
        question = st.text_area(
            "Ask a question about Olist data",
            placeholder="e.g. How are sales performing?",
            height=100,
        )

        run = st.button("Generate SQL, Run Query & Plot Graphs", type="primary")

        if run:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    # Immediate HNLP & SQL Guardrail Check
                    nl_block = detect_unsafe_user_prompt(question.strip())
                    if nl_block:
                        raise ValueError(nl_block)

                    with st.spinner("Running HNLP Safety Check, Schema Retrieval, Groq SQL Generation & MotherDuck Query..."):
                        if enable_self_correction:
                            outcome = run_agentic_text_to_sql(question.strip(), max_Corrections=2)
                            generated_sql = outcome["generated_sql"]
                            safe_sql = outcome["safe_sql"]
                            limit_modified = outcome["limit_modified"]
                            df = outcome["df"]
                            relevant_tables = outcome["relevant_tables"]
                            attempts = outcome["attempts"]
                            elapsed_ms = outcome["elapsed_ms"]
                        else:
                            t0 = time.perf_counter()
                            relevant_tables = select_relevant_tables(question.strip())
                            generated_sql = generate_sql(question.strip())
                            safe_sql, limit_modified = validate_sql_with_meta(generated_sql)
                            df = execute_sql(safe_sql)
                            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                            attempts = [{"attempt": 1, "status": "success"}]

                    with st.expander("🔍 Schema-Aware Agent Trace", expanded=False):
                        st.write(f"**Relevant Marts Identified:** `{', '.join(relevant_tables)}`")
                        st.write(f"**Execution Latency:** `{elapsed_ms} ms` | **Attempts:** `{len(attempts)}`")
                        if len(attempts) > 1:
                            st.info(
                                f"Agent self-corrected SQL on attempt {len(attempts)} after initial error: "
                                f"`{attempts[0].get('error')}`"
                            )

                    st.success(f"Query completed in {elapsed_ms} ms. Returned {len(df)} rows.")

                    render_auto_visualizations(
                        df,
                        question=question.strip(),
                        key_prefix="t2sql_viz",
                    )

                    st.subheader("Generated SQL")
                    st.code(generated_sql, language="sql")

                    if limit_modified:
                        st.info("A safety limit of 200 rows was applied to the query.")
                        st.code(safe_sql, language="sql")

                    st.subheader("Query Result Table")
                    st.dataframe(df, width="stretch")

                    if not df.empty:
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results as CSV",
                            data=csv,
                            file_name="olist_text_to_sql_results.csv",
                            mime="text/csv",
                        )

                except Exception as exc:
                    st.error("❌ Query Rejected by Safety Layer / Execution Error")
                    st.code(str(exc))
                    st.caption(
                        "Only single-statement read-only SELECT queries on approved MotherDuck marts are permitted. "
                        "Natural-language or HNLP requests to modify, update, insert, or delete values are strictly blocked."
                    )

    with tab_schema:
        st.subheader("Approved Analytical Marts & Star Schema")
        for tbl_name, tbl_doc in TABLE_SCHEMAS.items():
            with st.expander(f"📋 olist.marts.{tbl_name}", expanded=False):
                st.code(tbl_doc, language="sql")

    with tab_eval:
        st.subheader("SQL & HNLP Guardrail Safety Verification")
        st.markdown(
            "Verify that `validate_sql()` and `detect_unsafe_user_prompt()` block unauthorized tables, "
            "DDL/DML mutations, compound HNLP read-and-modify queries, and enforce the 200-row cap."
        )
        if st.button("Run Built-in SQL & HNLP Safety Check Suite"):
            safety_cases = [
                (
                    "Valid aggregate query",
                    "SELECT product_category_name, SUM(total_sales_value) AS sales FROM olist.marts.mart_product_performance GROUP BY product_category_name ORDER BY sales DESC LIMIT 10",
                    True,
                ),
                (
                    "Block HNLP compound read + modify query",
                    "read and modify teh some values in olist.marts.mart_sales",
                    False,
                ),
                (
                    "Block HNLP show + update values query",
                    "show top 10 product categories and update their total sales values to 0",
                    False,
                ),
                (
                    "Block unauthorized raw schema access",
                    "SELECT * FROM olist.raw.raw_customers LIMIT 10",
                    False,
                ),
                (
                    "Block DELETE statement",
                    "DELETE FROM olist.marts.mart_sales WHERE price < 10",
                    False,
                ),
                (
                    "Block DROP TABLE statement",
                    "DROP TABLE olist.marts.mart_sales",
                    False,
                ),
                (
                    "Cap excessive LIMIT 5000 -> LIMIT 200",
                    "SELECT order_id, price FROM olist.marts.mart_sales LIMIT 5000",
                    True,
                ),
            ]
            rows = []
            for label, test_sql, should_pass in safety_cases:
                try:
                    out_sql, lim_mod = validate_sql_with_meta(test_sql)
                    passed = should_pass is True
                    detail = f"Allowed -> {out_sql} (limit_capped={lim_mod})"
                except Exception as err:
                    passed = should_pass is False
                    detail = f"Blocked as expected -> {err}"
                rows.append(
                    {
                        "Test Case": label,
                        "Expected": "Allow" if should_pass else "Block",
                        "Status": "✅ PASS" if passed else "❌ FAIL",
                        "Details": detail,
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")


def main():
    st.set_page_config(
        page_title="Olist Text-to-SQL",
        page_icon="📊",
        layout="wide",
    )
    render_text_to_sql_interface(embedded=False)


if __name__ == "__main__":
    main()
