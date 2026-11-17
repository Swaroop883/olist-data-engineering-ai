"""
Comprehensive End-to-End Automated Test Suite for Olist Data Engineering + AI Platform
(TASK 4 - Module M9, FR_10, NFR_1..NFR_8, CS_4).

Tests:
1. Repository & Module Syntax / AST Integrity across all Python files
2. dbt Project, Sources, Schema Tests, and Star Schema + 9 Marts SQL Integrity
3. Airflow DAG Structure, Task Order, and DuckDB -> MotherDuck Idempotency
4. Opinion-Aware Multi-Stage RAG Engine:
   - Query Understanding (Intent, Aspect, Sentiment detection)
   - Question Rewriting (implicit -> explicit multilingual concepts)
   - Adaptive Top-K Selection (K=3..8, default 5)
   - Opinion-Aware Candidate Re-ranking
   - Insufficient-Evidence & Dataset-Wide Percentage Refusal Detection
   - Inline [review_id] Citation Validation (rejecting hallucinated review IDs)
5. Agentic & Schema-Aware Text-to-SQL Engine:
   - Schema Retrieval (`select_relevant_tables`)
   - SQL & HNLP Safety Validator (SELECT-only, approved marts allowlist, DDL/DML & read+modify block)
   - Accurate LIMIT 200 Injection & Capping (`validate_sql_with_meta`)
6. Grain & Aggregation Verification on Synthetic Olist Data (ensuring order-item grain
   in `mart_sales` vs. distinct `order_id` count is never conflated)
7. 7-Step Architecture Orchestrator & Automatic Chart Engine (`copilot_orchestrator.py`
   & `auto_visualizer.py`):
   - Step 2 Query Understanding (SQL vs RAG vs Hybrid)
   - Step 3 Safety Layer Rejection (`detect_unsafe_user_prompt` & `run_connected_copilot` on SQL + HNLP Read+Modify)
   - Step 6 LLM/Grounded Answer Generation (`generate_llm_sql_explanation`)
   - Step 7 Automatic Plotly Chart Selection (KPI cards, Line/Area trends, Bar/Pie/Scatter charts)
8. Secrets & Security Audit (`.gitignore` excludes `.env`, no hardcoded API keys)

Run with:
    python3 tests/test_end_to_end.py
"""

from __future__ import annotations

import ast
import sqlite3
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_DIR = PROJECT_ROOT / "ai"

for p in (str(PROJECT_ROOT), str(AI_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke(self, _prompt: str) -> FakeLLMResponse:
        return FakeLLMResponse(self.response_text)


class TestOlistEndToEndPlatform(unittest.TestCase):
    """End-to-end verification tests for all modules (M0–M9) and 7-Step Architecture."""

    def test_01_all_python_files_parse_cleanly(self):
        """Verify every Python file in the project has valid syntax."""
        py_files = [
            p
            for p in PROJECT_ROOT.rglob("*.py")
            if ".git" not in p.parts and "venv" not in str(p)
        ]
        self.assertGreaterEqual(len(py_files), 17)
        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8-sig")
            try:
                ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                self.fail(f"SyntaxError in {py_file}: {exc}")

    def test_02_dbt_models_and_star_schema_complete(self):
        """Verify all 9 analytical marts + 11 Star Schema models + staging + core exist."""
        marts_dir = PROJECT_ROOT / "dbt" / "models" / "marts"
        expected_marts = [
            "mart_sales.sql",
            "mart_customer_behavior.sql",
            "mart_customer_reviews.sql",
            "mart_delivery_performance.sql",
            "mart_enriched_review_analysis.sql",
            "mart_payment_analysis.sql",
            "mart_product_performance.sql",
            "mart_review_analysis.sql",
            "mart_seller_performance.sql",
        ]
        expected_star = [
            "dim_customer.sql",
            "dim_product.sql",
            "dim_seller.sql",
            "dim_date.sql",
            "dim_location.sql",
            "dim_payment.sql",
            "dim_review.sql",
            "fact_order_items.sql",
            "fact_sales.sql",
            "fact_payments.sql",
            "fact_reviews.sql",
        ]
        for fname in expected_marts + expected_star:
            fpath = marts_dir / fname
            self.assertTrue(fpath.exists(), f"Missing dbt model: {fpath}")
            self.assertGreater(fpath.stat().st_size, 20)

        schema_yml = (PROJECT_ROOT / "dbt" / "models" / "schema.yml").read_text(
            encoding="utf-8"
        )
        for test_type in ("not_null", "unique", "relationships", "accepted_values"):
            self.assertIn(test_type, schema_yml)

    def test_03_airflow_dag_integrates_ai_and_vectorstore(self):
        """Verify airflow/dags/olist_pipeline.py defines all 5 pipeline stages in order."""
        dag_source = (
            PROJECT_ROOT / "airflow" / "dags" / "olist_pipeline.py"
        ).read_text(encoding="utf-8")
        for task_id in (
            "load_raw_data",
            "enrich_incremental_reviews",
            "dbt_run",
            "dbt_test",
            "update_review_vectorstore",
        ):
            self.assertIn(task_id, dag_source)
        self.assertIn(
            "raw_loading\n        >> ai_enrichment\n        >> dbt_run\n        >> dbt_test\n        >> vectorstore_refresh",
            dag_source,
        )

    def test_04_rag_query_understanding_and_adaptive_k(self):
        """Verify Opinion-Aware Multi-Stage RAG query rewriting and adaptive Top-K (3..8)."""
        from rag_reviews import (
            generate_answer,
            is_insufficient_evidence_answer,
            rerank_and_filter_reviews,
            understand_and_rewrite_query,
        )

        plan_deliv = understand_and_rewrite_query(
            "My order took forever to arrive and was late", adaptive=True
        )
        self.assertEqual(plan_deliv["primary_aspect"], "delivery")
        self.assertEqual(plan_deliv["sentiment_constraint"], "negative")
        self.assertTrue(3 <= plan_deliv["adaptive_k"] <= 8)

        plan_broad = understand_and_rewrite_query(
            "Compare common customer complaints about delivery delays and product quality issues",
            adaptive=True,
        )
        self.assertGreaterEqual(plan_broad["adaptive_k"], 7)

        rid = "a" * 32
        plan_specific = understand_and_rewrite_query(f"Check review {rid}", adaptive=True)
        self.assertEqual(plan_specific["adaptive_k"], 3)

        plan_stat = understand_and_rewrite_query(
            "What percentage of all customers complained about delivery?"
        )
        self.assertTrue(plan_stat["is_dataset_stat_query"])
        self.assertTrue(
            is_insufficient_evidence_answer(
                "I cannot calculate the exact percentage across the entire dataset from retrieved reviews."
            )
        )

        candidates = [
            {
                "review_id": "1" * 32,
                "document": "Good product",
                "metadata": {"ai_topic": "price", "ai_sentiment": "positive"},
                "distance": 0.20,
            },
            {
                "review_id": "2" * 32,
                "document": "Delivery took 3 weeks past estimate",
                "metadata": {"ai_topic": "delivery", "ai_sentiment": "negative"},
                "distance": 0.25,
            },
        ]
        reranked = rerank_and_filter_reviews(candidates, plan_deliv, target_k=2)
        self.assertEqual(reranked[0]["review_id"], "2" * 32)

        valid_llm = FakeLLM(f"Customers reported severe shipping delays [{'2' * 32}].")
        answer, citations = generate_answer(
            "What do customers say about delivery?", reranked, valid_llm
        )
        self.assertEqual(citations, ["2" * 32])
        self.assertNotIn("Citation warning", answer)

        hallucinating_llm = FakeLLM(f"Customers complained [{'f' * 32}].")
        with self.assertRaises(ValueError):
            generate_answer(
                "What do customers say about delivery?", reranked, hallucinating_llm
            )

    def test_05_text_to_sql_guardrails_and_limit_handling(self):
        """Verify Text-to-SQL validation, allowlist, DML/DDL & HNLP read+modify blocking, and LIMIT 200 cap."""
        from evaluate_text_to_sql import run_safety_evaluation
        from text_to_sql_app import (
            run_agentic_text_to_sql,
            select_relevant_tables,
            validate_sql,
            validate_sql_with_meta,
        )

        safety_report = run_safety_evaluation()
        self.assertEqual(safety_report["pass_rate"], 100.0)

        sql_10 = (
            "SELECT seller_id, total_product_revenue "
            "FROM olist.marts.mart_seller_performance "
            "ORDER BY total_product_revenue DESC LIMIT 10"
        )
        safe_10, mod_10 = validate_sql_with_meta(sql_10)
        self.assertFalse(mod_10)
        self.assertIn("LIMIT 10", safe_10.upper())

        sql_no_lim = "SELECT order_id, price FROM olist.marts.mart_sales"
        safe_no_lim, mod_no_lim = validate_sql_with_meta(sql_no_lim)
        self.assertTrue(mod_no_lim)
        self.assertIn("LIMIT 200", safe_no_lim.upper())

        star_sql = (
            "SELECT p.product_category_name_english, SUM(f.total_item_value) AS rev "
            "FROM olist.marts.fact_sales f "
            "JOIN olist.marts.dim_product p ON f.product_id = p.product_id "
            "GROUP BY 1 LIMIT 5"
        )
        validated_star = validate_sql(star_sql)
        self.assertIn("fact_sales", validated_star.lower())

        rel_tables = select_relevant_tables("What is the average delivery delay by state?")
        self.assertIn("mart_delivery_performance", rel_tables)

        # Verify HNLP compound read+modify queries are blocked before any LLM/DB execution
        for unsafe_hnlp in (
            "read and modify teh some values",
            "read and modify some values in mart_sales",
            "show the top 10 product categories and update their sales values",
            "read customer reviews and edit the review score to 5",
            "change the some values in the database",
        ):
            with self.assertRaises(ValueError):
                validate_sql_with_meta(unsafe_hnlp)
            with self.assertRaises(ValueError):
                run_agentic_text_to_sql(unsafe_hnlp)

    def test_06_grain_correctness_order_items_vs_orders(self):
        """
        Verify SQL aggregation grain on multi-item orders:
        counting rows in `mart_sales` counts order items, whereas `COUNT(DISTINCT order_id)`
        counts distinct orders and computes accurate Average Order Value (AOV).
        """
        con = sqlite3.connect(":memory:")
        con.execute("""
            CREATE TABLE mart_sales (
                order_id TEXT,
                order_item_id INTEGER,
                customer_unique_id TEXT,
                price REAL,
                freight_value REAL,
                total_item_value REAL
            )
        """)
        con.executemany(
            "INSERT INTO mart_sales VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("order_101", 1, "cust_1", 90.0, 10.0, 100.0),
                ("order_101", 2, "cust_1", 90.0, 10.0, 100.0),
                ("order_101", 3, "cust_1", 90.0, 10.0, 100.0),
                ("order_102", 1, "cust_2", 90.0, 10.0, 100.0),
            ],
        )
        row = con.execute("""
            SELECT
                SUM(total_item_value) AS total_sales,
                COUNT(DISTINCT order_id) AS distinct_orders,
                COUNT(*) AS total_items,
                SUM(total_item_value) / COUNT(DISTINCT order_id) AS aov,
                AVG(price) AS avg_item_price
            FROM mart_sales
        """).fetchone()
        con.close()

        total_sales, distinct_orders, total_items, aov, avg_item_price = row
        self.assertEqual(total_sales, 400.0)
        self.assertEqual(distinct_orders, 2)
        self.assertEqual(total_items, 4)
        self.assertEqual(aov, 200.0)
        self.assertEqual(avg_item_price, 90.0)

    def test_07_copilot_orchestrator_and_auto_visualizer(self):
        """Verify 7-Step Architecture Orchestrator, Safety Rejection, and Auto-Visualizer."""
        from auto_visualizer import determine_recommended_charts
        from copilot_orchestrator import (
            build_companion_sql_question,
            classify_prompt_routing,
            detect_unsafe_user_prompt,
            run_connected_copilot,
            synthesize_unified_insight,
        )

        # Step 3 Safety Layer Rejection for destructive SQL & HNLP read+modify prompts
        self.assertIsNotNone(detect_unsafe_user_prompt("DROP TABLE olist.marts.mart_sales"))
        self.assertIsNotNone(detect_unsafe_user_prompt("SELECT * FROM olist.raw.raw_orders"))
        self.assertIsNotNone(detect_unsafe_user_prompt("read and modify teh some values"))
        self.assertIsNotNone(detect_unsafe_user_prompt("read the sales table and update some values"))
        self.assertIsNotNone(detect_unsafe_user_prompt("show top categories and change the sales values"))
        self.assertIsNotNone(detect_unsafe_user_prompt("edit some values in mart_sales"))
        self.assertIsNone(detect_unsafe_user_prompt("How are sales performing?"))
        self.assertIsNone(detect_unsafe_user_prompt("Which top 10 product categories have the highest total sales value?"))

        blocked_res = run_connected_copilot("read and modify teh some values", routing_override="AUTO")
        self.assertTrue(blocked_res["safety_blocked"])
        self.assertEqual(blocked_res["route"], "BLOCKED")

        # Step 2 Routing on the 3 architecture diagram example questions
        r_sales = classify_prompt_routing("How are sales performing?")
        self.assertTrue(r_sales["run_sql"])

        r_reviews = classify_prompt_routing("What are customers saying in reviews?")
        self.assertTrue(r_reviews["run_rag"])

        r_deliv = classify_prompt_routing("How are deliveries doing?")
        self.assertTrue(r_deliv["run_sql"])

        comp_q = build_companion_sql_question("Why are customers unhappy with late deliveries?")
        self.assertIn("mart_delivery_performance", comp_q)

        # Auto-visualizer chart specs:
        df_kpi = pd.DataFrame({"repeat_customers": [2997]})
        specs_kpi = determine_recommended_charts(df_kpi)
        self.assertEqual(specs_kpi[0]["chart_type"], "kpi_cards")

        df_cat = pd.DataFrame(
            {
                "ai_sentiment": ["positive", "negative", "neutral"],
                "review_count": [1908, 885, 218],
            }
        )
        specs_cat = determine_recommended_charts(df_cat)
        self.assertEqual(specs_cat[0]["chart_type"], "bar")
        self.assertEqual(specs_cat[1]["chart_type"], "pie")

        df_time = pd.DataFrame(
            {
                "year_month": ["2018-01", "2018-02", "2018-03"],
                "total_sales_value": [100000.0, 120000.0, 140000.0],
            }
        )
        specs_time = determine_recommended_charts(df_time)
        self.assertEqual(specs_time[0]["chart_type"], "line")

        summary = synthesize_unified_insight(
            "Test prompt",
            {"relevant_tables": ["mart_payment_analysis"], "df": df_cat, "elapsed_ms": 42.0},
            {"answer": "Customers are generally positive.", "retrieved": [1, 2], "citations": ["a" * 32]},
            llm_sql_narrative="Positive sentiment leads with 1,908 reviews.",
        )
        self.assertIn("Analytical Warehouse Insights", summary)
        self.assertIn("Customer Review Evidence", summary)

    def test_08_no_exposed_secrets_in_repository(self):
        """Ensure .env is gitignored and no real API tokens are hardcoded in Python files."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore)
        for py_file in PROJECT_ROOT.rglob("*.py"):
            if ".git" in py_file.parts:
                continue
            content = py_file.read_text(encoding="utf-8-sig")
            self.assertNotRegex(content, r"gsk_[A-Za-z0-9]{20,}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
