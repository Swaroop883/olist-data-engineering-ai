# Olist End-to-End Cloud Data Engineering, OLAP Warehouse & Connected AI Analytics Platform

An enterprise-grade cloud data engineering, multi-dimensional OLAP warehousing, and compound-safe AI analytics platform built on the **Brazilian E-Commerce Public Dataset by Olist** (`99,441` orders, `112,650` order items, `99,224` Portuguese customer reviews across `9` interconnected relational CSV datasets).

## 1. Problem Statement & Architectural Motivation

Modern e-commerce marketplaces generate two complementary yet traditionally disconnected data modalities:

1. **Siloed Structured & Unstructured Data**: Relational SQL warehouses track quantitative KPIs (`$13,221,498.11` delivered revenue, `7.49%` late deliveries across `96,476` delivered orders) but cannot explain *why* customers leave 1-star ratings buried inside `99,224` unstructured Portuguese review comments.
2. **Multi-Fact Fan-Trap vs. Multi-Join Latency**: Joining `1:N` order items (`112,650` rows), `1:N` payment installments, and `1:N` customer reviews into a naive flat table (OBT) causes a **302.23% revenue fan-trap error**, whereas a normalized 3NF schema requires **8 runtime joins** (`24.844 ms` geometric mean latency) and degrades LLM Text-to-SQL accuracy.
3. **Multilingual Semantics & LLM Arithmetic Hallucinations**: Standard top-K vector search misses implicit Portuguese idioms (e.g., *"não recebi"* for missing items, *"atraso"* for late shipping) and hallucinates unverified dataset percentages when asked quantitative questions over qualitative text chunks.
4. **Compound HNLP Read+Modify & Destructive SQL Vulnerabilities**: Direct LLM Text-to-SQL wrappers misinterpret compound **Hybrid Natural Language Queries (HNLP)** such as *"read and modify some values"* or *"show top categories and update their sales"* as benign `SELECT` queries, creating severe data-mutation and schema-exposure risks.

This platform solves all four problems through a **6-Layer Cloud ELT & AI Architecture**, a **Hybrid Kimball Galaxy Schema + 9 Precomputed Marts**, an **Online 7-Step Connected AI + BI Copilot**, a **7-Stage ROQuA Opinion-Aware RAG Engine**, and a **3-Layer HNLP Defense-in-Depth Security Architecture**.

## 2. End-to-End System Architecture

### 2.1 Offline 6-Layer Cloud ELT & AI Enrichment Pipeline

Orchestrated end-to-end by **Apache Airflow** (`airflow/dags/olist_pipeline.py`):

1. **Layer 1 — AWS S3 Raw Data Lake (`ingestion/ingest_to_s3.py`)**: Validates schemas and uploads all 9 raw Olist CSV files to `s3://<bucket>/raw/olist/` using `boto3` with remote byte-size idempotency checks.
2. **Layer 2 — MotherDuck Serverless Cloud Warehouse (`motherduck/load_raw_data.py`)**: Streams raw CSVs from S3 into `md:olist_db` (`olist.raw` schema) with primary-key constraints and `INSERT OR IGNORE` deduplication.
3. **Layer 3A — Groq LLM Review Enrichment (`ai/enrich_reviews.py`)**: Incrementally enriches Portuguese customer reviews (`LEFT JOIN ... WHERE r.review_id IS NULL`, 100-row checkpoint batches) using **Groq (`openai/gpt-oss-120b`, temperature `0.1`)** into `olist.raw.raw_review_enrichment` (`3,011` enriched reviews with English translation, sentiment polarity, 9 issue categories, and audit metadata `model_name`, `prompt_version='v1.0'`, `enriched_at`).
4. **Layer 3B — 3-Tier dbt Medallion & Kimball Galaxy Transformations (`dbt/models/`)**:
   - **Bronze (`dbt/models/staging/`)**: `9` `stg_*` views enforcing `TRIM()`, explicit type casting, and null primary-key filtering.
   - **Silver (`dbt/models/core/` & `dbt/snapshots/`)**: `6` `core_*` intermediate tables computing `delivery_days`, `estimated_delivery_days`, `is_late`, order-grain payment rollups, and 71 Portuguese-to-English category mappings, plus an **SCD Type 2 snapshot (`snp_sellers`)** tracking historical seller city/state relocations (`dbt_valid_from`, `dbt_valid_to`).
   - **Gold (`dbt/models/marts/`)**: `4` multi-grain Fact tables (`fact_order_items`, `fact_sales`, `fact_payments`, `fact_reviews`), `7` shared Conformed Dimensions (`dim_customer`, `dim_product`, `dim_seller`, `dim_date`, `dim_location`, `dim_payment`, `dim_review`), and `9` precomputed analytical marts (`mart_*`).
5. **Layer 4 — Two-Tier Automated Data Quality Gates (`dbt/models/schema.yml` & `dbt/tests/`)**: Runs declarative schema tests (`not_null`, `unique`, `relationships`, `accepted_values`) plus 3 singular SQL business assertions (`assert_positive_sales_values.sql`, `assert_non_negative_payments.sql`, `assert_valid_review_scores.sql`).
6. **Layer 5 — Multilingual Vector Indexing (`ai/build_review_vectorstore.py`)**: Embeds enriched reviews combined with Olist order/product/delivery context using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`384`-dim) into persistent **ChromaDB** (`ai/chroma_db`) via deterministic `review_id` delta upserts.

### 2.2 Online 7-Step Connected AI + BI Copilot Architecture (`ai/copilot_orchestrator.py`)

1. **Step 1 — User Natural-Language Entry**: Accepts free-form queries via the Unified Streamlit Workbench (`app.py`, port `8501`) or standalone REST API (`web_ui/server.py`, port `8080`).
2. **Step 2 — Pre-LLM Safety Screening & Intent Routing (`classify_prompt_routing`)**: Screens the raw user prompt for destructive or compound HNLP read+modify intent first; if safe, classifies the query into `SQL_ONLY`, `RAG_ONLY`, or `HYBRID` (automatically generating a companion SQL query when qualitative questions require warehouse KPIs).
3. **Step 3 — Dual Execution Paths**:
   - **Top Path (Guarded Text-to-SQL)**: Dynamic Top-K table selection (`select_relevant_tables`), grain-aware prompt construction, and 3-layer HNLP + SQLGlot AST validation.
   - **Bottom Path (7-Stage ROQuA RAG)**: Cross-lingual Portuguese/English query expansion, adaptive Top-K (`K=3..8`, default `5`), candidate over-fetching (`max(k*4, 20)`), and opinion-aware re-ranking (`+0.08` topic match, `+0.06` sentiment match).
4. **Step 4 — Read-Only MotherDuck Execution**: Executes validated `SELECT` queries against `olist.marts` with an automatic `LIMIT 200` safety cap.
5. **Step 5 — Exact Python Percentage Precomputation**: Computes exact `*_pct_of_total` share columns in Pandas prior to LLM synthesis so the model never guesses arithmetic.
6. **Step 6 — Grounded Narrative Synthesis**: Combines exact MotherDuck SQL metrics with regex-verified `[review_id]` review citations (stripping or rejecting any hallucinated citation IDs).
7. **Step 7 — Automatic Chart Recommendation Engine (`ai/auto_visualizer.py`)**: Inspects result dataframe cardinality and column types to render interactive Plotly KPI cards, time-series area/line charts, grouped bars, donut charts, or treemaps.

## 3. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| Cloud Data Lake | AWS S3 (`boto3`) | Immutable raw CSV landing zone (`s3://.../raw/olist/`) |
| Cloud Data Warehouse | MotherDuck / DuckDB (`duckdb==1.5.6`) | Serverless analytical cloud database (`md:olist_db`) |
| Transformation & Testing | dbt (`dbt-duckdb`) | Bronze/Silver/Gold Medallion + Kimball Galaxy modeling & quality tests |
| Workflow Orchestration | Apache Airflow | Automated 5-stage pipeline DAG (`airflow/dags/olist_pipeline.py`) |
| LLM Inference | Groq (`openai/gpt-oss-120b`) | Batch review enrichment, RAG synthesis, and agentic Text-to-SQL |
| Multilingual Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | 384-dimensional Portuguese/English semantic representations |
| Vector Database | ChromaDB (`ai/chroma_db`) | Persistent HNSW vector indexing and metadata filtering |
| SQL & HNLP Security | Regex Guard + SQLGlot (`duckdb` dialect) | Pre-LLM HNLP read+modify block, `SELECT`-only AST check, 21-table allowlist |
| BI, OLAP & Web Serving | Streamlit (`8501`), FastAPI/HTTP (`8080`), Plotly | 5-tab analytics workbench, Universal $2^N$ OLAP Cube Studio & LAN server |

## 4. Warehouse Schema Catalog (`olist.marts`)

### 4.1 Nine Precomputed Analytical Marts

| Mart Name | Grain | Rows | Primary Analytical Scope |
| :--- | :--- | :--- | :--- |
| `mart_sales` | `order_id, order_item_id` | `112,650` | Item price, freight value, `total_item_value`, English category, customer/seller state |
| `mart_delivery_performance` | `order_id` | `96,476` | Carrier/customer delivery timestamps, estimated date, `delivery_delay_days`, late flag |
| `mart_customer_behavior` | `customer_unique_id` | `96,096` | `total_orders`, repeat purchase behavior, first/last order timestamps, lifetime spend |
| `mart_review_analysis` | `review_id` | `99,224` | Full review dataset with rule-based sentiment categorization and review scores |
| `mart_product_performance` | `product_id` | `32,951` | Total product revenue, units sold, freight cost, average item price, category |
| `mart_seller_performance` | `seller_id` | `3,095` | Seller revenue ranking, items fulfilled, freight value, seller city/state |
| `mart_enriched_review_analysis` | `review_id` | `3,011` | Groq AI-enriched reviews with `ai_sentiment`, `ai_topic`, `ai_summary`, `ai_issue_category` |
| `mart_customer_reviews` | `customer_unique_id` | `96,096` | Customer-level review counts, average review score, positive/neutral/negative split |
| `mart_payment_analysis` | `payment_type` | `5` | Order count, total payment value, average payment value, average installments |

### 4.2 Kimball Galaxy Schema (Fact Constellation) & Universal OLAP Cube

- **4 Multi-Grain Fact Tables**: `fact_order_items` (`112,650` rows), `fact_sales` (`99,441` orders), `fact_payments`, and `fact_reviews` (`99,224` reviews).
- **7 Conformed Dimension Tables**: `dim_customer`, `dim_product`, `dim_seller`, `dim_date`, `dim_location`, `dim_payment`, and `dim_review`.
- **Universal Multi-Mart OLAP Cube Studio (`ai/bi_dashboard.py`)**: Supports **Roll-Up**, **Drill-Down**, **Slice**, **Dice**, **2D Pivot Heatmap Matrix**, and **ANSI SQL `GROUP BY CUBE (D1, D2)`** (computing the full $2^N$ cuboid lattice with `GROUPING_ID(D1, D2)` bitmask identification across any mart or fact table).

## 5. Empirical Evaluation & Benchmark Results (`analysis/`)

All benchmark suites are reproducible via `python analysis/run_all_analysis.py` (results persisted in `analysis/latest_analysis_report.json`):

### 5.1 Warehouse Architecture Benchmark (`TPC-H & Star Schema Benchmark Q1.1–Q4.1`)

| Warehouse Architecture | Runtime Joins (Q4.1) | Avg Latency (ms) | GeoMean Latency (ms) | Multi-Fact Fan-Trap Error | Aggregation Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hybrid Galaxy + Kimball Marts (Ours)** | **0** | **41.385 ms** | **19.972 ms** | **0.0%** | **100.0%** (`2.538 ms` / `394 QPS` on Q1.1) |
| Pure Star Schema (SSB Fact + 4 Dims) | 3 | 38.342 ms | 19.530 ms | 0.0% | 100.0% |
| 3NF / Snowflake (TPC-H 7-Table Schema) | 8 | 42.973 ms | 24.844 ms | 0.0% | 100.0% |
| Naive Single Flat Table (OBT) | 0 | 38.812 ms | 18.806 ms | **302.23%** | **1.0%** (Cartesian fan-trap inflation) |

### 5.2 ROQuA Opinion-Aware RAG Evaluation (`RAGAS Benchmark`)

- **Mean RAGAS Composite Score**: `90.45%` (`0.9045`) across `RAG_Q1`–`RAG_Q5`
- **Mean Context Recall@K**: `100.0%` (`1.000`)
- **Mean Faithfulness (Verified `[review_id]` Citations)**: `100.0%` (`1.000`)
- **Mean Context Precision@K**: `87.30%` (`0.873`)
- **Mean Reciprocal Rank (MRR)**: `86.67%` (`0.8667`)
- **Warm Retrieval Latency**: `67.73 ms` – `89.15 ms`

### 5.3 Agentic Text-to-SQL & 3-Layer HNLP Security Evaluation (`BIRD-SQL & Yale Spider`)

- **BIRD-SQL Execution Accuracy (EX)**: `100.0%` (`6/6` complex analytical queries `SQL_B1`–`SQL_B6`)
- **Mean Valid Efficiency Score (VES)**: `1.1597`
- **Yale Spider AST Component Match (F1)**: `1.000` (`100.0%` across `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, and `KEYWORDS`)
- **Adversarial & Compound HNLP Block Rate**: `100.0%` (`11/11` safety checks in `ai/evaluate_text_to_sql.py` + `8/8` adversarial suite in `analysis/text_to_sql_bird_spider_analysis.py`, blocking `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `COPY`, `ATTACH`, `PRAGMA`, unauthorized `olist.raw` access, and compound HNLP read+modify prompts).

## 6. Repository Structure

```text
olist-data-engineering-ai/
├── .streamlit/
│   └── config.toml                          # Headless LAN server binding (0.0.0.0:8501)
├── ai/
│   ├── auto_visualizer.py                   # Automatic Plotly chart recommendation engine
│   ├── bi_dashboard.py                      # 8-Section BI Suite & Universal 2^N OLAP Cube Studio
│   ├── build_review_vectorstore.py          # Incremental ChromaDB vector indexer
│   ├── copilot_orchestrator.py              # Online 7-Step Connected AI + BI Copilot orchestrator
│   ├── db_utils.py                          # MotherDuck connection & read-only SQL execution guard
│   ├── enrich_all_reviews.py                # Multi-batch Groq review enrichment runner
│   ├── enrich_reviews.py                    # Idempotent Groq (gpt-oss-120b) review enrichment
│   ├── evaluate_text_to_sql.py              # 11-check HNLP/SQL safety & MotherDuck benchmark suite
│   ├── rag_app.py                           # Opinion-Aware RAG Streamlit interface
│   ├── rag_reviews.py                       # 7-Stage ROQuA retrieval, rewriting & re-ranking engine
│   ├── search_reviews.py                    # CLI semantic review search utility
│   ├── text_to_sql_app.py                   # Agentic Text-to-SQL engine & 3-layer HNLP guardrails
│   └── unified_app.py                       # Unified app entrypoint wrapper
├── airflow/
│   └── dags/
│       └── olist_pipeline.py                # 5-stage Apache Airflow DAG
├── analysis/
│   ├── __init__.py
│   ├── latest_analysis_report.json          # Persisted benchmark scorecard results
│   ├── olap_warehouse_analysis.py           # TPC-H & Star Schema Benchmark (SSB Q1.1–Q4.1)
│   ├── rag_ragas_analysis.py                # RAGAS retrieval & faithfulness evaluation
│   ├── run_all_analysis.py                  # Unified benchmark runner
│   └── text_to_sql_bird_spider_analysis.py  # BIRD-SQL (EX, VES), Spider AST F1 & HNLP safety audit
├── dbt/
│   ├── dbt_project.yml                      # dbt project configuration
│   ├── macros/
│   │   └── generate_schema_name.sql         # Custom target schema macro
│   ├── models/
│   │   ├── sources.yml                      # Raw & AI-enriched MotherDuck sources
│   │   ├── schema.yml                       # Declarative data quality tests
│   │   ├── staging/                         # 9 Bronze stg_* cleaning views
│   │   ├── core/                            # 6 Silver core_* intermediate tables
│   │   └── marts/                           # 9 Gold mart_* tables + 4 fact_* + 7 dim_* models
│   ├── snapshots/                           # SCD Type 2 seller tracking (snp_sellers)
│   └── tests/                               # 3 singular SQL business assertion tests
├── docs/
│   └── FINAL_PROJECT_REPORT.md              # Comprehensive technical architecture & evaluation report
├── ingestion/
│   └── ingest_to_s3.py                      # Idempotent AWS S3 CSV uploader
├── motherduck/
│   ├── load_raw_data.py                     # S3 -> MotherDuck raw table loader
│   ├── test_connection.py                   # MotherDuck connectivity smoke test
│   ├── test_s3_access.py                    # DuckDB httpfs S3 access test
│   └── verify_raw_data.py                   # Raw table row-count verifier
├── tests/
│   └── test_end_to_end.py                   # 8-suite automated end-to-end verification tests
├── web_ui/
│   ├── index.html                           # Standalone dark-glassmorphism HTML5/Tailwind UI
│   └── server.py                            # Zero-dependency REST API & web server (port 8080)
├── app.py                                   # Main 5-Tab Unified Streamlit Workbench (port 8501)
├── requirements.txt                         # Locked Python dependencies (duckdb==1.5.6)
├── run_lan_server.sh                        # One-command Linux LAN server & firewall launcher
└── README.md                                # Project documentation
```

## 7. Setup, Local Execution & Multi-PC LAN Demo Guide

### 7.1 Environment Variables (`.env`)

Create a `.env` file in the project root (never committed to Git):

```ini
MOTHERDUCK_TOKEN=your_motherduck_token_here
GROQ_API_KEY=your_groq_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=eu-north-1
```

### 7.2 Activate Virtual Environment & Run Verification Tests

```bash
cd /media/sai/OS/acads/de_project/olist-data-engineering-ai
source rag_venv/bin/activate

# 1. Run all 8 end-to-end automated unit & integration tests
python3 tests/test_end_to_end.py

# 2. Run the 11-case HNLP Read+Modify & SQLGlot AST Security Evaluation
python3 ai/evaluate_text_to_sql.py
```

### 7.3 Launch for Local PC & Friend's PC on the Same Wi-Fi / LAN Router

```bash
chmod +x run_lan_server.sh
./run_lan_server.sh
```

- **On Your Linux Host PC**: Open `http://localhost:8501`
- **On Your Friend's PC (Connected to the Same Router/Hotspot)**: Open `http://<YOUR_HOST_LAN_IP>:8501` (printed automatically in the terminal banner and in the Streamlit sidebar)
- **Optional Standalone HTML5/Tailwind Web UI (Port 8080)**:
  ```bash
  python3 web_ui/server.py --host 0.0.0.0 --port 8080
  ```
