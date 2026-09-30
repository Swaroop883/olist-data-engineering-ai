from datetime import datetime, timedelta
import os
import subprocess
import sys

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from dotenv import load_dotenv


PROJECT_DIR = "/mnt/c/Users/chsum/Desktop/B-Tech Acads/Sem 5/DE_PROJECT/olist-data-engineering-ai"

# Always load the current credentials from .env
load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=True)

# Use the same Python environment in which Airflow is running
PYTHON_BIN = sys.executable
DBT_BIN = os.path.join(os.path.dirname(PYTHON_BIN), "dbt")


def run_raw_loading():
    """Load raw Olist CSV files from S3 into MotherDuck."""
    subprocess.run(
        [
            PYTHON_BIN,
            os.path.join(PROJECT_DIR, "motherduck", "load_raw_data.py"),
        ],
        cwd=PROJECT_DIR,
        env=os.environ.copy(),
        check=True,
    )


def run_dbt_run():
    """Run all dbt staging, core and mart models."""
    subprocess.run(
        [DBT_BIN, "run"],
        cwd=os.path.join(PROJECT_DIR, "dbt"),
        env=os.environ.copy(),
        check=True,
    )


def run_dbt_test():
    """Run dbt data-quality tests."""
    subprocess.run(
        [DBT_BIN, "test"],
        cwd=os.path.join(PROJECT_DIR, "dbt"),
        env=os.environ.copy(),
        check=True,
    )


with DAG(
    dag_id="olist_pipeline",
    start_date=datetime(2026, 9, 30),
    schedule=timedelta(days=15),
    catchup=False,
    tags=["olist", "data-engineering"],
) as dag:

    raw_loading = PythonOperator(
        task_id="load_raw_data",
        python_callable=run_raw_loading,
        retries=2,
        retry_delay=timedelta(minutes=5),
    )

    dbt_run = PythonOperator(
        task_id="dbt_run",
        python_callable=run_dbt_run,
    )

    dbt_test = PythonOperator(
        task_id="dbt_test",
        python_callable=run_dbt_test,
    )

    raw_loading >> dbt_run >> dbt_test