"""
Standalone Custom HTML5 / Tailwind / Plotly.js Web Dashboard Server (`web_ui/server.py`).

Provides a sleek, non-Streamlit modern dark-glassmorphism web dashboard on Port 8080
with zero extra pip dependencies (uses Python's built-in `http.server` + JSON REST APIs)
connected directly to:
1. Unified NLP Prompt Copilot (`/api/copilot` -> SQL + RAG + Auto-Charts)
2. 8-Section BI & OLAP Dashboard (`/api/bi_overview`)
3. Opinion-Aware Multi-Stage RAG (`/api/rag`)
4. Agentic Guarded Text-to-SQL (`/api/sql`)
5. Architecture Benchmarks & Scorecard (`/api/benchmarks` from `analysis/`)

Run on Linux / LAN:
    python web_ui/server.py --port 8080
Then open:
    http://localhost:8080 (or http://<YOUR_LAN_IP>:8080 from a friend's PC on the same router)
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
AI_DIR = PROJECT_ROOT / "ai"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

for p in (str(PROJECT_ROOT), str(AI_DIR), str(ANALYSIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from auto_visualizer import determine_recommended_charts
from copilot_orchestrator import (
    build_companion_sql_question,
    classify_prompt_routing,
    detect_unsafe_user_prompt,
    enrich_df_with_exact_percentages,
    generate_llm_sql_explanation,
)
from rag_reviews import (
    generate_answer,
    initialize_rag,
    retrieve_reviews,
    understand_and_rewrite_query,
)
from analysis.olap_warehouse_analysis import run_tpch_ssb_benchmark
from text_to_sql_app import is_security_violation_error, run_agentic_text_to_sql

_RAG_CACHE = None


def _get_rag():
    global _RAG_CACHE
    if _RAG_CACHE is None:
        _RAG_CACHE = initialize_rag()
    return _RAG_CACHE


def _get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _build_benchmarks_payload() -> dict:
    olap_rep = run_tpch_ssb_benchmark(scale_factor=0.02)
    archs = []
    for row in olap_rep.get("summary_table", []):
        archs.append(
            {
                "Architecture": row.get("Architecture"),
                "Overall_Score_100": row.get("Aggregation Accuracy (%)", 100.0),
                "Avg_Query_Latency_ms": row.get("GeoMean Latency (ms)", 1.0),
            }
        )
    return {
        "1_schema_architecture_experiments": {
            "architectures": archs,
            "summary_table": olap_rep.get("summary_table", []),
        }
    }


class OlistWebDashboardHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            html_bytes = (WEB_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        if parsed.path == "/api/benchmarks":
            try:
                data = _build_benchmarks_payload()
                self._send_json({"ok": True, "data": data})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw_body or "{}")

        if parsed.path == "/api/copilot":
            prompt = (body.get("prompt") or "").strip()
            mode = body.get("mode") or "Auto (Smart Hybrid: SQL + RAG + Charts)"
            if not prompt:
                self._send_json({"ok": False, "error": "Please enter a prompt."}, status=400)
                return

            rejection = detect_unsafe_user_prompt(prompt)
            if rejection:
                self._send_json({"ok": False, "rejected": True, "error": rejection})
                return

            routing = classify_prompt_routing(prompt, mode_override=mode)
            sql_data = None
            rag_data = None
            llm_narrative = None

            if routing["run_sql"]:
                try:
                    if routing.get("is_qualitative_primary"):
                        comp_q = build_companion_sql_question(prompt)
                        out = run_agentic_text_to_sql(comp_q, max_Corrections=1)
                    else:
                        out = run_agentic_text_to_sql(prompt, max_Corrections=2)
                except Exception as exc:
                    err_str = str(exc)
                    if is_security_violation_error(err_str) or not routing.get("is_qualitative_primary"):
                        self._send_json({"ok": False, "rejected": True, "error": err_str})
                        return
                    comp_q = build_companion_sql_question(prompt)
                    out = run_agentic_text_to_sql(comp_q, max_Corrections=1)

                df = out["df"]
                enriched_df, totals = enrich_df_with_exact_percentages(df)
                llm_narrative = generate_llm_sql_explanation(prompt, out["safe_sql"], df)
                sql_data = {
                    "safe_sql": out["safe_sql"],
                    "relevant_tables": out["relevant_tables"],
                    "elapsed_ms": out["elapsed_ms"],
                    "columns": list(enriched_df.columns),
                    "rows": enriched_df.to_dict(orient="records"),
                    "chart_specs": determine_recommended_charts(df),
                    "totals": totals,
                }

            if routing["run_rag"]:
                try:
                    q_plan = understand_and_rewrite_query(prompt, adaptive=True)
                    emb_model, collection, llm = _get_rag()
                    retrieved = retrieve_reviews(prompt, emb_model, collection, adaptive=True)
                    ans, citations = generate_answer(prompt, retrieved, llm)
                    rag_data = {
                        "query_plan": q_plan,
                        "answer": ans,
                        "citations": citations,
                        "retrieved": retrieved,
                    }
                except Exception as r_exc:
                    rag_data = {"error": str(r_exc), "answer": "", "retrieved": [], "citations": []}

            self._send_json(
                {
                    "ok": True,
                    "routing": routing,
                    "llm_narrative": llm_narrative,
                    "sql": sql_data,
                    "rag": rag_data,
                }
            )
            return

        self._send_json({"ok": False, "error": "Unknown endpoint"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Olist Custom HTML5/Tailwind/Plotly Web UI.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    args = parser.parse_args()

    lan_ip = _get_lan_ip()
    server = HTTPServer(("0.0.0.0", args.port), OlistWebDashboardHandler)
    print(f"🚀 Olist Custom Web UI running on Host:      http://localhost:{args.port}")
    print(f"🌐 Share with Friend's PC (Same Router/LAN): http://{lan_ip}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()