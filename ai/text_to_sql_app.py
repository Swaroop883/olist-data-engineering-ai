
import os
import re
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import sqlglot
from dotenv import load_dotenv
from groq import Groq
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

st.set_page_config(
    page_title="Olist Text-to-SQL",
    page_icon="📊",
    layout="wide",
)

# The app can query only these approved marts.
APPROVED_TABLES = {
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

SCHEMA = """
Database: olist_db
Schema: marts

mart_sales:
order_id VARCHAR, order_item_id BIGINT, product_id VARCHAR,
seller_id VARCHAR, price DOUBLE, freight_value DOUBLE,
total_item_value DOUBLE, product_category_name VARCHAR,
seller_state VARCHAR, customer_id VARCHAR, customer_unique_id VARCHAR,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
customer_state VARCHAR

mart_customer_behavior:
customer_unique_id VARCHAR, total_orders BIGINT,
first_order_date TIMESTAMP, last_order_date TIMESTAMP

mart_customer_reviews:
customer_unique_id VARCHAR, total_reviews BIGINT,
average_review_score DOUBLE, positive_reviews HUGEINT,
neutral_reviews HUGEINT, negative_reviews HUGEINT

mart_delivery_performance:
order_id VARCHAR, customer_id VARCHAR, customer_unique_id VARCHAR,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
order_delivered_carrier_date TIMESTAMP,
order_delivered_customer_date TIMESTAMP,
order_estimated_delivery_date TIMESTAMP, customer_city VARCHAR,
customer_state VARCHAR, delivery_delay_days BIGINT

mart_enriched_review_analysis:
review_id VARCHAR, order_id VARCHAR, customer_id VARCHAR,
customer_unique_id VARCHAR, review_score BIGINT,
review_comment_title VARCHAR, review_comment_message VARCHAR,
review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
customer_city VARCHAR, customer_state VARCHAR,
review_sentiment_category VARCHAR, ai_sentiment VARCHAR,
ai_topic VARCHAR, ai_summary VARCHAR, ai_issue_category VARCHAR,
review_language VARCHAR, ai_enriched_at TIMESTAMP

mart_payment_analysis:
payment_type VARCHAR, total_orders BIGINT,
total_payment_value DOUBLE, average_payment_value DOUBLE,
average_installments DOUBLE

mart_product_performance:
product_id VARCHAR, product_category_name VARCHAR,
total_orders BIGINT, total_items_sold BIGINT,
total_product_revenue DOUBLE, total_freight_value DOUBLE,
total_sales_value DOUBLE, average_item_price DOUBLE

mart_review_analysis:
review_id VARCHAR, order_id VARCHAR, customer_id VARCHAR,
customer_unique_id VARCHAR, review_score BIGINT,
review_comment_title VARCHAR, review_comment_message VARCHAR,
review_creation_date TIMESTAMP, review_answer_timestamp TIMESTAMP,
order_status VARCHAR, order_purchase_timestamp TIMESTAMP,
customer_city VARCHAR, customer_state VARCHAR,
review_sentiment_category VARCHAR

mart_seller_performance:
seller_id VARCHAR, seller_state VARCHAR, total_orders BIGINT,
total_items_sold BIGINT, total_product_revenue DOUBLE,
total_freight_value DOUBLE, total_sales_value DOUBLE,
average_item_price DOUBLE
"""

SYSTEM_PROMPT = f"""
You are a careful analytics SQL assistant for Brazilian Olist e-commerce data.

Generate exactly one DuckDB SQL SELECT query for the user's question.
Use only tables and columns in this schema:

{SCHEMA}

Rules:
- Only SELECT queries are allowed.
- Use only the nine listed tables in the marts schema.
- Fully qualify tables as olist.marts.table_name.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY,
  ATTACH, DETACH, INSTALL, LOAD, EXPORT, IMPORT, PRAGMA, or CALL.
- Never query raw, staging, core, system, or information_schema tables.
- Never guess column names.
- Use a reasonable LIMIT of at most 200 rows unless the query is
  a scalar aggregate.
- Return only SQL, without Markdown fences or explanations.
- For revenue, use the appropriate mart's documented revenue fields.
- Do not claim that results are causal or statistically representative.
"""


def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing from the project .env file.")
    return Groq(api_key=key)


def generate_sql(question: str) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=os.getenv("TEXT_TO_SQL_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    sql = response.choices[0].message.content or ""
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql.strip(), flags=re.I)
    return sql.strip().rstrip(";")


def validate_sql(sql: str) -> str:
    """Parse and allow exactly one SELECT statement over approved marts."""
    if not sql.strip():
        raise ValueError("The model returned empty SQL.")

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as exc:
        raise ValueError(f"Could not parse the generated SQL: {exc}") from exc

    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Only one SQL statement is allowed.")

    tree = statements[0]

    # Require a SELECT (including SELECT ... WITH ...).
    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT statements are allowed.")

    # Reject nested write/administrative commands and unsafe constructs.
    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
        exp.Create, exp.Command, exp.Copy, exp.Transaction,
    )
    for node in tree.walk():
        if isinstance(node, forbidden):
            raise ValueError(f"Disallowed SQL operation: {type(node).__name__}.")

    # Every referenced table must be explicitly allowlisted.
    tables = list(tree.find_all(exp.Table))
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

    # Add a row cap for non-aggregate result sets.
    # This is a safety limit, not a replacement for SQL validation.
    if not any(isinstance(node, exp.AggFunc) for node in tree.walk()):
        limit = tree.args.get("limit")
        if limit is None:
            tree = tree.limit(200)
        else:
            try:
                limit_expr = limit.expression
                limit_value = int(limit_expr.this)
            except Exception:
                raise ValueError("Use a simple integer LIMIT.")
            if limit_value > 200:
                tree.set("limit", exp.Limit(expression=exp.Literal.number(200)))

    return tree.sql(dialect="duckdb")


def execute_sql(sql: str) -> pd.DataFrame:
    token = os.getenv("MOTHERDUCK_TOKEN", "").strip()
    if not token:
        raise RuntimeError("MOTHERDUCK_TOKEN is missing from .env.")

    # DuckDB's MotherDuck extension reads this environment variable.
    os.environ["motherduck_token"] = token

    con = duckdb.connect()
    try:
        con.execute("LOAD motherduck")
        con.execute("ATTACH 'md:olist_db' AS olist")
        result = con.execute(sql)
        columns = [description[0] for description in result.description]
        rows = result.fetchmany(201)

        if len(rows) > 200:
            rows = rows[:200]

        return pd.DataFrame(rows, columns=columns)
    finally:
        con.close()


st.title("📊 Olist Text-to-SQL")
st.caption(
    "Ask business questions in natural language. "
    "Groq generates SQL; DuckDB queries approved MotherDuck marts."
)

with st.sidebar:
    st.subheader("Available data")
    st.write("Database: `olist_db`")
    st.write("Schema: `marts`")
    st.write(f"Approved marts: **{len(APPROVED_TABLES)}**")
    st.divider()
    st.subheader("Example questions")
    examples = [
        "Which product categories have the highest sales value?",
        "What is the average delivery delay by customer state?",
        "Show the top 10 sellers by total product revenue.",
        "How are review scores distributed by AI sentiment?",
        "Which payment types have the highest average payment value?",
        "How many repeat customers have more than one order?",
    ]
    for example in examples:
        st.write(f"• {example}")

question = st.text_area(
    "Ask a question about Olist data",
    placeholder="e.g. Which states have the highest average delivery delay?",
    height=100,
)

run = st.button("Generate SQL and run query", type="primary")

if run:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Generating SQL with Groq..."):
                generated_sql = generate_sql(question)

            st.subheader("Generated SQL")
            st.code(generated_sql, language="sql")

            with st.spinner("Validating SQL and querying MotherDuck..."):
                safe_sql = validate_sql(generated_sql)

                if (
    "LIMIT 200" in safe_sql.upper()
    and "LIMIT 200" not in generated_sql.upper()
):
                    st.info("A safety limit was applied to the query.")
                    st.code(safe_sql, language="sql")

                df = execute_sql(safe_sql)

            st.success(f"Query completed. Returned {len(df)} rows.")
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
            st.error("The query could not be completed.")
            st.code(str(exc))
            st.caption(
                "Review the generated SQL and try a more specific question. "
                "Only SELECT queries on approved marts are permitted."
            )
