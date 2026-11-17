"""
Apache Airflow End-to-End Orchestration DAG (`olist_pipeline`)
(Module M4 - TASK 2, FR_5, NFR_1, NFR_2, NFR_8).

Matches the Offline / Scheduled Data Pipeline in the Architecture Diagram:
1. Ingest Raw Data via DuckDB into MotherDuck (`load_raw_data.py --skip-if-populated`)
2. Incremental AI Review Understanding / Enrichment (`ai/enrich_reviews.py`)
3. Transform with dbt (`dbt run` -> staging, core, 9 marts, Star Schema)
4. Validate with dbt quality gates (`dbt test`)
5. Update Vector Store / RAG (`ai/build_review_vectorstore.py`)
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from dotenv import load_dotenv

DEFAULT_WSL_PROJECT_DIR = (
    "/mnt/c/Users/chsum/Desktop/B-Tech Acads/Sem 5/DE_PROJECT/olist-data-engineering-ai"
)
RESOLVED_PROJECT_DIR = str(Path(__file__).resolve().parents[2])
PROJECT_DIR = (
    DEFAULT_WSL_PROJECT_DIR
    if os.path.exists(DEFAULT_WSL_PROJECT_DIR)
    else RESOLVED_PROJECT_DIR
)

# Always load the current credentials from .env
load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=True)

# Airflow environment Python & dbt binary
PYTHON_BIN = sys.executable
DBT_BIN = os.path.join(os.path.dirname(PYTHON_BIN), "dbt")

# Prefer rag_venv for AI & ChromaDB tasks if available in the project root
RAG_VENV_PYTHON = os.path.join(PROJECT_DIR, "rag_venv", "bin", "python")
AI_PYTHON_BIN = RAG_VENV_PYTHON if os.path.exists(RAG_VENV_PYTHON) else PYTHON_BIN


def run_raw_loading():
    """Step 1 (Data Pipeline): Ingest/verify raw Olist data in MotherDuck via DuckDB."""
    subprocess.run(
        [
            PYTHON_BIN,
            os.path.join(PROJECT_DIR, "motherduck", "load_raw_data.py"),
            "--skip-if-populated",
        ],
        cwd=PROJECT_DIR,
        env=os.environ.copy(),
        check=True,
    )


def run_ai_review_enrichment():
    """
    Step 2 (Supporting AI Component): Incrementally enrich any unenriched customer
    reviews into `olist.ai.enriched_order_reviews`.
    """
    batch_size = os.getenv("AIRFLOW_ENRICH_BATCH_SIZE", "5")
    subprocess.run(
        [
            AI_PYTHON_BIN,
            os.path.join(PROJECT_DIR, "ai", "enrich_reviews.py"),
            "--batch-size",
            str(batch_size),
        ],
        cwd=PROJECT_DIR,
        env=os.environ.copy(),
        check=True,
    )


def run_dbt_run():
    """Step 3 (Transform): Run all dbt staging, core, analytical marts, and Star Schema models."""
    subprocess.run(
        [DBT_BIN, "run"],
        cwd=os.path.join(PROJECT_DIR, "dbt"),
        env=os.environ.copy(),
        check=True,
    )


def run_dbt_test():
    """Step 4 (Quality Gate): Run dbt data-quality tests on MotherDuck models."""
    subprocess.run(
        [DBT_BIN, "test"],
        cwd=os.path.join(PROJECT_DIR, "dbt"),
        env=os.environ.copy(),
        check=True,
    )


def run_vectorstore_refresh():
    """
    Step 5 (Vector Store / RAG): Incrementally embed any newly enriched reviews
    into ChromaDB (`ai/chroma_db`).
    """
    subprocess.run(
        [
            AI_PYTHON_BIN,
            os.path.join(PROJECT_DIR, "ai", "build_review_vectorstore.py"),
        ],
        cwd=PROJECT_DIR,
        env=os.environ.copy(),
        check=True,
    )


with DAG(
    dag_id="olist_pipeline",
    start_date=datetime(2026, 9, 30),
    schedule=timedelta(days=15),
    catchup=False,
    tags=["olist", "data-engineering", "duckdb", "motherduck", "dbt", "ai", "rag"],
) as dag:

    raw_loading = PythonOperator(
        task_id="load_raw_data",
        python_callable=run_raw_loading,
        retries=2,
        retry_delay=timedelta(minutes=5),
    )

    ai_enrichment = PythonOperator(
        task_id="enrich_incremental_reviews",
        python_callable=run_ai_review_enrichment,
        retries=2,
        retry_delay=timedelta(minutes=5),
    )

    dbt_run = PythonOperator(
        task_id="dbt_run",
        python_callable=run_dbt_run,
        retries=1,
        retry_delay=timedelta(minutes=3),
    )

    dbt_test = PythonOperator(
        task_id="dbt_test",
        python_callable=run_dbt_test,
        retries=1,
        retry_delay=timedelta(minutes=2),
    )

    vectorstore_refresh = PythonOperator(
        task_id="update_review_vectorstore",
        python_callable=run_vectorstore_refresh,
        retries=1,
        retry_delay=timedelta(minutes=3),
    )

    (
        raw_loading
        >> ai_enrichment
        >> dbt_run
        >> dbt_test
        >> vectorstore_refresh
    )