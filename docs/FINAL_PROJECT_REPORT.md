# Final Project Report: Olist End-to-End Cloud Data Engineering & AI Analytics Platform

## 1. Executive Summary

This report documents the architecture, implementation, engineering decisions, and validation of the **Olist Data Engineering + AI Analytics Platform**. Built on the public Brazilian Olist E-Commerce dataset (9 CSV tables, 99,441 orders, 112,650 order items, and 99,224 reviews), the system delivers a complete cloud data pipeline integrated with three production-ready AI and analytics workflows:
1. **Groq LLM Review Enrichment (`olist.ai.enriched_order_reviews`)**: Converts raw Portuguese customer reviews into structured analytical fields (`sentiment`, `topic`, `summary`, `issue_category`, `language`) with full model/prompt audit metadata (`model_name`, `prompt_version`, `enriched_at`).
2. **Opinion-Aware Multi-Stage RAG (`ai/rag_reviews.py`, `ai/rag_app.py`)**: Implements a 7-stage ROQuA-inspired retrieval-augmented generation pipeline with query understanding, implicit opinion rewriting, adaptive Top-K selection (`K=5` default, `3–8` adaptive), e-commerce order context enrichment, opinion re-ranking, inline `[review_id]` citation verification, and dataset-wide percentage refusal guardrails.
3. **Agentic & Schema-Aware Text-to-SQL (`ai/text_to_sql_app.py`)**: Combines dynamic schema retrieval, domain few-shot prompting, SQLGlot AST validation (SELECT-only, table allowlist, 200-row safety cap), and an agentic error self-correction loop over MotherDuck analytical marts.
4. **Interactive BI Dashboard & Unified Serving Portal (`ai/bi_dashboard.py`, `app.py`)**: Serves 8 comprehensive analytical sections with global date, state, and order status filters alongside the RAG and Text-to-SQL assistants in a single Streamlit application.

---

## 2. Requirements Traceability Matrix (`FR_0`–`FR_10` & `NFR_1`–`NFR_8`)

| Requirement Code | Requirement Description | Implemented Artifact(s) | Verification Status |
|---|---|---|---|
| **FR_0 (SETUP)** | GitHub repository & dataset structure understanding | Git repo (`feature/ai-review-enrichment`), `README.md` | Completed & Verified |
| **FR_1 (INGEST)** | Ingest 9 Olist CSV files into AWS S3 raw zone (`raw/olist/`) | `ingestion/ingest_to_s3.py` | Completed & Idempotent |
| **FR_2 (RAW LOAD)** | Load raw Olist data from S3 into MotherDuck `raw` schema | `motherduck/load_raw_data.py`, `motherduck/verify_raw_data.py` | Completed (112,650 item rows verified) |
| **FR_3 (STAGING)** | dbt staging models to clean, cast timestamps, and standardize columns | `dbt/models/staging/stg_*.sql` (9 models) | Completed & Tested |
| **FR_4 (MARTS)** | Core relational models, 9 analytical marts, and Star Schema (`fact_*`, `dim_*`) | `dbt/models/core/core_*.sql` (6), `dbt/models/marts/mart_*.sql` (9), `dbt/models/marts/fact_*.sql` (4), `dbt/models/marts/dim_*.sql` (7) | Completed (20 mart/star models) |
| **FR_5 (ORCHESTRATE)** | Orchestrate ingestion, transformation, AI enrichment, and vector refresh via Airflow | `airflow/dags/olist_pipeline.py` | Completed (6-task DAG) |
| **FR_6 (ENRICH)** | Enrich customer reviews via Groq LLM with model/prompt metadata | `ai/enrich_reviews.py`, `ai/enrich_all_reviews.py` | Completed (3,011+ reviews enriched) |
| **FR_7 (RAG QA)** | Opinion-Aware Multi-Stage RAG with adaptive K (`5` default, `3–8` range) & citations | `ai/rag_reviews.py`, `ai/rag_app.py`, `ai/build_review_vectorstore.py` | Completed & Tested |
| **FR_8 (TEXT-SQL)** | Guarded Text-to-SQL interface over approved marts with SELECT-only guard | `ai/text_to_sql_app.py`, `ai/evaluate_text_to_sql.py` | Completed & 100% Safety Pass |
| **FR_9 (DASHBOARD)** | Interactive Streamlit BI dashboard covering sales, customers, products, sellers, delivery, payments, and reviews | `ai/bi_dashboard.py`, `app.py` | Completed (8 sections + filters) |
| **FR_10 (QUALITY)** | Run data-quality checks for uniqueness, nulls, relationships, and accepted values | `dbt/models/schema.yml`, `dbt/tests/*.sql`, `tests/test_end_to_end.py` | Completed & Verified |
| **NFR_1 (IDEMPOTENCY)** | Repeatable pipeline loads without duplicate business records | S3 byte-size check, `INSERT OR IGNORE`, ChromaDB delta `upsert` | Completed |
| **NFR_2 (QUALITY GATE)** | Data-quality checks gate downstream vector store refresh | `dbt_test >> update_review_vectorstore` in Airflow DAG | Completed |
| **NFR_3 (SECURITY)** | Least-privilege / SELECT-only SQL guard & allowlisted tables | `validate_sql_with_meta()` in `ai/text_to_sql_app.py` & `ai/db_utils.py` | Completed |
| **NFR_4 (MAINTAINABILITY)** | Modular Python/SQL structure, documentation, and version control | `ai/db_utils.py`, modular `render_*` functions, PEP 8 | Completed |
| **NFR_5 (PERFORMANCE)** | Precomputed marts & `@st.cache_data` caching for fast dashboard queries | `ai/bi_dashboard.py` + `olist.marts.*` | Completed |
| **NFR_6 (SCALABILITY)** | Layered architecture (`raw -> staging -> core -> marts`) | MotherDuck + dbt + incremental AI processing | Completed |
| **NFR_7 (AUDITABILITY)** | Store `model_name`, `prompt_version`, `enriched_at` + `[review_id]` source citations | `ai/enrich_reviews.py`, `ai/rag_reviews.py` | Completed |
| **NFR_8 (COST)** | Bounded AI batch size & incremental ChromaDB delta embedding | `--batch-size` flag & `existing_ids` check in `build_review_vectorstore.py` | Completed |

---

## 3. Implementation of the Three Research-Backed Improvements

### 3.1 Improvement 1 — Dimensional Star Schema Gold Layer (`dbt/models/marts/`)
Why we chose a layered **Star Schema + Curated Marts** design instead of a single flat table or purely 3NF normalized operational tables:
- A single flat table repeats customer, order, and payment attributes across multi-item orders (e.g., an order with 3 items duplicates order-level fields 3 times, risking inflated order counts and payment sums).
- Purely normalized 3NF tables require 4–6 table joins for common questions, increasing LLM hallucination risk in Text-to-SQL.
- Our hybrid Gold layer preserves all **9 pre-aggregated analytical marts (`mart_*`)** while providing **4 Fact tables (`fact_order_items`, `fact_sales`, `fact_payments`, `fact_reviews`)** and **7 Dimension tables (`dim_customer`, `dim_product`, `dim_seller`, `dim_date`, `dim_location`, `dim_payment`, `dim_review`)**.

### 3.2 Improvement 2 — Opinion-Aware Multi-Stage RAG (`ai/rag_reviews.py`)
Inspired by **ROQuA (Zhou et al., 2026)**, our RAG pipeline addresses implicit e-commerce language (e.g., *"It took forever to arrive"* implies a delivery delay even without the word *"delivery"*):
1. **Stage 1 (Query Understanding)**: Extracts user intent, primary aspect (`delivery`, `product_quality`, `packaging`, `price`, `service`), and sentiment polarity (`positive`, `negative`, `any`), and detects unsupported dataset-wide statistical queries.
2. **Stage 2 (Question Rewriting)**: Expands implicit phrases with multilingual domain terms (`entrega`, `atraso`, `demora`, `qualidade`, `defeito`) before embedding.
3. **Stage 3 (Adaptive Top-K Selection)**: Defaults to `K=5`, narrowing to `K=3–4` for specific single-review or narrow single-aspect lookups, and expanding to `K=7–8` for broad multi-aspect syntheses.
4. **Stage 4 & 5 (Contextual Retrieval & Opinion Re-ranking)**: Over-fetches candidate vectors from ChromaDB (where each review is enriched with Olist product category, customer state, seller state, and delivery delay status) and re-ranks candidates by combining cosine similarity with aspect and sentiment alignment.
5. **Stage 6 & 7 (Evidence Evaluation & Cited Generation)**: Generates grounded English answers with regex-validated `[review_id]` citations and refuses dataset-wide percentage calculations that cannot be computed from a Top-K retrieval sample.

### 3.3 Improvement 3 — Agentic & Schema-Aware Text-to-SQL (`ai/text_to_sql_app.py`)
Inspired by **Ojuri et al. (2025)**:
1. **Dynamic Schema Identification**: Identifies relevant marts for the user's question and supplies explicit table grain annotations (highlighting that `mart_sales` is at the order-item grain so distinct orders require `COUNT(DISTINCT order_id)`).
2. **Few-Shot Domain Prompting**: Provides verified Olist SQL examples inside the system prompt.
3. **SQLGlot AST Guardrail**: Parses the generated SQL into an AST, rejects any non-`SELECT` or administrative statement, verifies that every referenced physical table belongs to `olist.marts.<approved_table>`, and caps non-aggregate result sets at `LIMIT 200` (displaying the limit notice only when the AST limit was actually injected or reduced).
4. **Agentic Error Self-Correction Loop**: If validation or DuckDB execution raises an error, the agent feeds the exact error trace back to the LLM to self-correct the query automatically.
