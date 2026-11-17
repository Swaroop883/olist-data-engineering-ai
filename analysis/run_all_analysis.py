import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from datetime import datetime, timezone
from analysis.olap_warehouse_analysis import run_tpch_ssb_benchmark
from analysis.rag_ragas_analysis import run_ragas_evaluation
from analysis.text_to_sql_bird_spider_analysis import run_bird_spider_evaluation

REPORT_PATH = Path(__file__).resolve().parent / "latest_analysis_report.json"


def run_complete_system_analysis(use_live_llm: bool = False) -> dict:
    print("\n1. Running TPC-H / Star Schema Benchmark (SSB Q1.1-Q4.1)...")
    olap_rep = run_tpch_ssb_benchmark(scale_factor=0.02)
    print(pd.DataFrame(olap_rep["summary_table"]).to_string(index=False))

    print("\n2. Running RAGAS Customer Review RAG Evaluation...")
    rag_rep = run_ragas_evaluation(run_llm_generation=use_live_llm)
    print(pd.DataFrame(rag_rep["query_results"]).drop(columns=["Evaluation Question"]).to_string(index=False))

    print("\n3. Running BIRD-SQL & Yale Spider Text-to-SQL Evaluation...")
    sql_rep = run_bird_spider_evaluation(use_live_llm=use_live_llm)
    print(pd.DataFrame(sql_rep["query_results"]).drop(columns=["Validated SQL", "Natural Language Question"]).to_string(index=False))

    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "olap_tpch_ssb": olap_rep,
        "rag_ragas": rag_rep,
        "text_to_sql_bird_spider": sql_rep,
    }
    REPORT_PATH.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\n✅ All 3 industry-standard analysis suites completed! Report saved to {REPORT_PATH}")
    return bundle


if __name__ == "__main__":
    run_complete_system_analysis(use_live_llm=False)
