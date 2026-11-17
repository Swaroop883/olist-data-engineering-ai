import socket
import pandas as pd
import plotly.express as px
import streamlit as st

from ai.bi_dashboard import render_bi_dashboard
from ai.rag_app import render_rag_interface
from ai.text_to_sql_app import render_text_to_sql_interface
from ai.copilot_orchestrator import run_connected_copilot
from ai.auto_visualizer import render_auto_visualizations, render_rag_evidence_charts
from analysis.olap_warehouse_analysis import run_tpch_ssb_benchmark
from analysis.rag_ragas_analysis import run_ragas_evaluation
from analysis.text_to_sql_bird_spider_analysis import run_bird_spider_evaluation


def get_local_lan_ip() -> str:
    """Detect the Linux PC's local Wi-Fi / router LAN IPv4 address for sharing with other PCs."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    st.set_page_config(
        page_title="Olist Data Engineering & AI Analytics Platform",
        page_icon="🚀",
        layout="wide",
    )

    st.sidebar.markdown("## 🚀 Olist DE + AI Platform")
    view = st.sidebar.radio(
        "Navigation",
        [
            "🤖 1. Unified NLP Copilot & Visual Answer",
            "📈 2. 8-Section BI & Universal OLAP Cube",
            "🔎 3. RAG Customer Review Assistant",
            "📊 4. Guarded Text-to-SQL Workbench",
            "🔬 5. Analysis (TPC-H/SSB, RAGAS & BIRD-SQL)",
        ],
    )

    lan_ip = get_local_lan_ip()
    with st.sidebar.expander("🌐 LAN Demo Server (Friend's PC)", expanded=True):
        st.caption("Connected to the same Wi-Fi / router? Open this URL on your friend's PC:")
        st.code(f"http://{lan_ip}:8501", language="text")
        st.caption(f"Host (this Linux PC): `http://localhost:8501`")

    if view == "🤖 1. Unified NLP Copilot & Visual Answer":
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#0f172a,#1e3a8a);padding:20px 24px;border-radius:14px;border:1px solid #3b82f6;margin-bottom:16px;">
                <h2 style="margin:0;color:#ffffff;">🤖 Unified NLP Copilot & Visual Answer</h2>
                <p style="margin:6px 0 0 0;color:#cbd5e1;">
                    Ask any business or customer review question. The platform routes your query across <b>Guarded Text-to-SQL (MotherDuck)</b> and <b>Opinion-Aware RAG (ChromaDB)</b>, blocks any <b>SQL or HNLP Read+Modify / Write attempt</b>, verifies mathematical accuracy, and renders interactive visual analytics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_q1, c_q2, c_q3, c_q4, c_q5 = st.columns(5)
        sample_prompt = None
        if c_q1.button("📊 Sales by Category", use_container_width=True):
            sample_prompt = "Which top 10 product categories have the highest total sales value?"
        if c_q2.button("🚚 Delivery Delays", use_container_width=True):
            sample_prompt = "What is the average delivery delay in days for each customer state?"
        if c_q3.button("💬 Customer Reviews RAG", use_container_width=True):
            sample_prompt = "What do customers complain about regarding late deliveries and product quality in reviews?"
        if c_q4.button("🛡️ Test HNLP Read+Modify Block", use_container_width=True):
            sample_prompt = "read and modify teh some values in mart_sales"
        if c_q5.button("🛡️ Test SQL DELETE Block", use_container_width=True):
            sample_prompt = "DELETE FROM olist.marts.mart_sales WHERE price > 0"

        if sample_prompt:
            st.session_state["copilot_q"] = sample_prompt

        default_text = st.session_state.get(
            "copilot_q",
            "Which top 10 product categories have the highest total sales value?",
        )

        col_inp, col_mode = st.columns([3.0, 1.3])
        with col_inp:
            user_query = st.text_input(
                "1️⃣ Ask a natural-language question:",
                value=default_text,
                key="copilot_input",
            )
        with col_mode:
            routing_mode = st.selectbox(
                "2️⃣ Routing Mode:",
                options=[
                    "Auto (Smart Hybrid: SQL + RAG + Charts)",
                    "Force Hybrid (Always Run SQL + RAG + Charts)",
                    "SQL Warehouse + Auto-Charts Only",
                    "Customer Review RAG + Evidence Charts Only",
                ],
                index=0,
                key="copilot_routing_mode",
            )

        if st.button("🚀 Run Connected Pipeline", type="primary") or sample_prompt:
            active_query = sample_prompt if sample_prompt else user_query
            st.session_state["copilot_q"] = active_query
            with st.spinner("Executing Query Understanding → HNLP & SQL Safety Guard → MotherDuck / ChromaDB → Visual Analytics..."):
                res = run_connected_copilot(active_query, routing_override=routing_mode)

            if res.get("safety_blocked"):
                st.error(f"❌ {res.get('safety_error')}")
            else:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Routing Mode", str(res.get("route", "HYBRID")))
                s2.metric("SQL Safety Guard", "✅ Validated SELECT" if res.get("sql_executed") else "N/A (RAG Only)")
                df_res = res.get("sql_df")
                s3.metric("MotherDuck Rows", f"{len(df_res):,}" if isinstance(df_res, pd.DataFrame) else "0")
                s4.metric("ChromaDB Citations", f"{len(res.get('rag_docs', []))} Reviews")

                st.markdown("### 💡 Executive Answer & Grounded Insights")
                if res.get("sql_explanation"):
                    st.info(res["sql_explanation"])
                if res.get("rag_answer"):
                    st.success(res["rag_answer"])

                if isinstance(df_res, pd.DataFrame) and not df_res.empty:
                    st.markdown("### 📊 Visual Analytics & Interactive Data Explorer")
                    render_auto_visualizations(df_res, question=active_query, key_prefix="copilot_viz")
                    with st.expander("🛡️ View Executed MotherDuck SQL"):
                        st.code(res.get("sql_query", ""), language="sql")

                rag_docs = res.get("rag_docs", [])
                if rag_docs:
                    render_rag_evidence_charts(rag_docs, key_prefix="copilot_rag_viz")
                    with st.expander(f"🔎 View {len(rag_docs)} Retrieved Customer Reviews from ChromaDB", expanded=True):
                        st.dataframe(pd.DataFrame(rag_docs), use_container_width=True)

    elif view == "📈 2. 8-Section BI & Universal OLAP Cube":
        render_bi_dashboard(embedded=True)

    elif view == "🔎 3. RAG Customer Review Assistant":
        render_rag_interface(embedded=True)

    elif view == "📊 4. Guarded Text-to-SQL Workbench":
        render_text_to_sql_interface(embedded=True)

    elif view == "🔬 5. Analysis (TPC-H/SSB, RAGAS & BIRD-SQL)":
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#0f172a,#1e293b);padding:20px 24px;border-radius:14px;border:1px solid #334155;margin-bottom:16px;">
                <h2 style="margin:0;color:#f8fafc;">🔬 System Analysis & Industry Standard Evaluation Suite</h2>
                <p style="margin:6px 0 0 0;color:#94a3b8;">
                    Live, reproducible evaluation across three industry-standard frameworks:
                    <b>1. TPC-H & Star Schema Benchmark (SSB Q1.1–Q4.1)</b> for Warehouse Architectures,
                    <b>2. RAGAS Framework</b> for RAG Retrieval & Faithfulness, and
                    <b>3. BIRD-SQL & Yale Spider</b> for Text-to-SQL Execution Accuracy, Efficiency & HNLP/SQL Guardrails.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_olap, tab_rag, tab_sql = st.tabs([
            "🏛️ 1. Warehouse Architecture Analysis (TPC-H & SSB)",
            "🔎 2. RAG System Analysis (RAGAS Framework)",
            "📊 3. Text-to-SQL Analysis (BIRD-SQL & Spider)",
        ])

        with tab_olap:
            st.markdown("#### 🏛️ Live TPC-H & Star Schema Benchmark (SSB Q1.1–Q4.1) Evaluation")
            st.caption(
                "Executes the canonical Star Schema Benchmark (O'Neil et al.) derived from TPC-H live in DuckDB across "
                "3NF/Snowflake (7-table TPC-H), Pure Star Schema (SSB), Our Hybrid Galaxy + Materialized Marts, and a Flat Wide Table."
            )
            if st.button("▶️ Run Live TPC-H / SSB Architecture Benchmark", key="run_ssb_btn", type="primary") or "ssb_res" not in st.session_state:
                with st.spinner("Generating TPC-H / SSB tables and executing Query Flights Q1.1–Q4.1 live..."):
                    st.session_state["ssb_res"] = run_tpch_ssb_benchmark(scale_factor=0.02)

            ssb_res = st.session_state["ssb_res"]
            df_ssb = pd.DataFrame(ssb_res["summary_table"])
            st.dataframe(df_ssb, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_lat = px.bar(
                    df_ssb,
                    x="Architecture",
                    y="GeoMean Latency (ms)",
                    color="Architecture",
                    text_auto=".2f",
                    title="SSB Query Flights Geometric Mean Latency (ms — Lower is Faster)",
                )
                fig_lat.update_layout(height=380, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig_lat, use_container_width=True)
            with c2:
                fig_acc = px.bar(
                    df_ssb,
                    x="Architecture",
                    y="Aggregation Accuracy (%)",
                    color="Multi-Fact Fan-Trap Error (%)",
                    text_auto=".1f",
                    title="Multi-Fact Aggregation Accuracy (%) vs. Fan-Trap Error",
                )
                fig_acc.update_layout(height=380, template="plotly_dark")
                st.plotly_chart(fig_acc, use_container_width=True)

            with st.expander("🔍 Inspect Individual SSB Query Flights (Q1.1, Q2.1, Q3.1, Q4.1) & Live SQL", expanded=False):
                st.dataframe(pd.DataFrame(ssb_res["per_query_flights"]), use_container_width=True)

            if ssb_res.get("olist_live_comparison"):
                st.markdown("##### 🇧🇷 Live MotherDuck Olist Schema Comparison")
                st.dataframe(pd.DataFrame(ssb_res["olist_live_comparison"]), use_container_width=True)

        with tab_rag:
            st.markdown("#### 🔎 RAG System Evaluation (RAGAS & Information Retrieval Metrics)")
            st.caption(
                "Computes Context Precision@K, Mean Reciprocal Rank (MRR), Context Recall@K, Faithfulness (Citation Grounding), and Answer Relevancy."
            )
            run_live_rag_llm = st.checkbox("Include Live Groq LLM Generation in Faithfulness Check", value=True, key="rag_live_chk")
            if st.button("▶️ Run Live RAGAS Evaluation on ChromaDB", key="run_ragas_btn", type="primary") or "ragas_res" not in st.session_state:
                with st.spinner("Evaluating RAG pipeline on ChromaDB review corpus..."):
                    st.session_state["ragas_res"] = run_ragas_evaluation(run_llm_generation=run_live_rag_llm)

            rag_rep = st.session_state["ragas_res"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Context Precision@K", f"{rag_rep['mean_context_precision']:.2%}")
            m2.metric("Mean Reciprocal Rank (MRR)", f"{rag_rep['mean_mrr']:.2%}")
            m3.metric("Context Recall@K", f"{rag_rep['mean_context_recall']:.2%}")
            m4.metric("Faithfulness (Grounded)", f"{rag_rep['mean_faithfulness']:.2%}")
            m5.metric("Answer Relevancy", f"{rag_rep['mean_answer_relevancy']:.2%}")

            df_rag = pd.DataFrame(rag_rep["query_results"])
            st.dataframe(df_rag, use_container_width=True)

            fig_rag = px.bar(
                df_rag,
                x="Query ID",
                y=["Context Precision@K", "Context Recall@K", "Faithfulness", "Answer Relevancy"],
                barmode="group",
                title="RAGAS Component Metrics Breakdown per Query",
            )
            fig_rag.update_layout(height=380, template="plotly_dark")
            st.plotly_chart(fig_rag, use_container_width=True)

        with tab_sql:
            st.markdown("#### 📊 Text-to-SQL System Evaluation (BIRD-SQL & Yale Spider Standards)")
            st.caption(
                "Computes Execution Accuracy (EX), Valid Efficiency Score (VES), Spider AST Component Match (F1), and SQLGlot + HNLP Adversarial Guardrail Block Rate."
            )
            run_live_sql_llm = st.checkbox("Generate Fresh SQL via Live Groq LLM", value=True, key="sql_live_chk")
            if st.button("▶️ Run Live BIRD-SQL & Spider Evaluation", key="run_bird_btn", type="primary") or "bird_res" not in st.session_state:
                with st.spinner("Running Gold SQL vs. Predicted SQL on MotherDuck..."):
                    st.session_state["bird_res"] = run_bird_spider_evaluation(use_live_llm=run_live_sql_llm)

            sql_rep = st.session_state["bird_res"]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Execution Accuracy (EX)", f"{sql_rep['execution_accuracy_pct']:.1f}%")
            b2.metric("Valid Efficiency Score (VES)", f"{sql_rep['mean_ves']:.3f}")
            b3.metric("Spider AST Match (F1)", f"{sql_rep['mean_spider_ast_f1']:.2%}")
            b4.metric("Adversarial Block Rate", f"{sql_rep['adversarial_block_rate_pct']:.1f}%")

            st.markdown("##### 📋 BIRD-SQL & Spider Query-Level Results")
            st.dataframe(pd.DataFrame(sql_rep["query_results"]), use_container_width=True)

            st.markdown("##### 🛡️ Adversarial Security Guardrail Verification (DML / DDL / HNLP Read+Modify / Raw Schema Attacks)")
            st.dataframe(pd.DataFrame(sql_rep["safety_results"]), use_container_width=True)


if __name__ == "__main__":
    main()
