from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai.db_utils import run_query_df


MART_CATALOG: Dict[str, str] = {
    "mart_sales": "Order-Item Sales & Revenue Mart (112,650 rows)",
    "mart_product_performance": "Product & Category Performance Mart (32,951 rows)",
    "mart_seller_performance": "Seller Revenue & Freight Performance Mart (3,095 rows)",
    "mart_delivery_performance": "Order Logistics & Delivery Delay Mart (96,476 rows)",
    "mart_payment_analysis": "Payment Methods & Installments Mart",
    "mart_customer_behavior": "Customer Repeat Purchase & Frequency Mart (96,096 rows)",
    "mart_customer_reviews": "Customer Review Satisfaction Roll-Up Mart",
    "mart_review_analysis": "Full Customer Reviews Mart (99,224 rows)",
    "mart_enriched_review_analysis": "Groq AI-Enriched Customer Reviews Mart (3,011 rows)",
    "fact_order_items": "Galaxy Schema — Fact Order Items",
    "fact_payments": "Galaxy Schema — Fact Payments",
    "fact_reviews": "Galaxy Schema — Fact Reviews",
    "olap_sales_cube": "Pre-Computed 8-Cuboid Sales OLAP Lattice",
}


@st.cache_data(ttl=300, show_spinner=False)
def q_df(sql: str) -> pd.DataFrame:
    return run_query_df(sql, read_only=True)


@st.cache_data(ttl=600, show_spinner=False)
def get_available_marts() -> List[str]:
    try:
        df = q_df("SHOW TABLES FROM olist.marts")
        col = df.columns[0]
        found = [str(x) for x in df[col].tolist()]
        ordered = [m for m in MART_CATALOG if m in found]
        extras = [m for m in found if m not in ordered]
        return ordered + extras
    except Exception:
        return list(MART_CATALOG.keys())


@st.cache_data(ttl=600, show_spinner=False)
def get_table_schema_metadata(table_name: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Inspects columns of `olist.marts.<table_name>` and classifies them into:
      - dimensions: categorical/text/boolean/id columns + virtual date hierarchy columns
      - measures: numeric columns
      - date_cols: timestamp/date columns
    """
    df_desc = q_df(f"DESCRIBE olist.marts.{table_name}")
    dimensions: List[str] = []
    measures: List[str] = []
    date_cols: List[str] = []

    for _, row in df_desc.iterrows():
        col_name = str(row["column_name"])
        col_type = str(row["column_type"]).upper()

        if any(t in col_type for t in ["TIMESTAMP", "DATE"]):
            date_cols.append(col_name)
            dimensions.extend([
                f"YEAR({col_name})",
                f"QUARTER({col_name})",
                f"YEAR_MONTH({col_name})",
            ])
        elif any(t in col_type for t in ["DOUBLE", "FLOAT", "DECIMAL", "INT", "BIGINT", "HUGEINT"]):
            if col_name.endswith("_id") or col_name in ("date_key", "zip_code_prefix", "cuboid_id"):
                dimensions.append(col_name)
            else:
                measures.append(col_name)
                if col_name in ("review_score", "total_orders", "payment_installments", "order_year"):
                    dimensions.append(col_name)
        else:
            dimensions.append(col_name)

    if not measures:
        measures = ["*"]
    return dimensions, measures, date_cols


def _dim_sql_expr(dim_token: str) -> Tuple[str, str]:
    """Returns (SQL expression, clean alias) for a selected dimension token."""
    if dim_token.startswith("YEAR(") and dim_token.endswith(")"):
        col = dim_token[5:-1]
        return f"CAST(EXTRACT(YEAR FROM {col}) AS VARCHAR)", f"{col}_year"
    if dim_token.startswith("QUARTER(") and dim_token.endswith(")"):
        col = dim_token[8:-1]
        return f"STRFTIME({col}, '%Y') || '-Q' || CAST(EXTRACT(QUARTER FROM {col}) AS VARCHAR)", f"{col}_quarter"
    if dim_token.startswith("YEAR_MONTH(") and dim_token.endswith(")"):
        col = dim_token[11:-1]
        return f"STRFTIME({col}, '%Y-%m')", f"{col}_year_month"
    return f"CAST({dim_token} AS VARCHAR)", dim_token


def render_universal_olap_cube_explorer() -> None:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a,#1e293b);padding:18px 22px;border-radius:14px;border:1px solid #334155;margin-bottom:16px;">
            <h3 style="margin:0;color:#f8fafc;font-size:1.2rem;">🧊 Universal Multi-Mart Interactive OLAP Cube Studio</h3>
            <p style="margin:6px 0 0 0;color:#94a3b8;font-size:0.9rem;">
                Select <b>any Data Mart or Fact Table</b> in <code>olist.marts</code>, pick your <b>Dimensions</b>, <b>Measures</b>, <b>Aggregation</b>, and <b>OLAP Operation</b> (<i>Roll-Up, Drill-Down, Slice, Dice, Pivot Matrix, or ANSI SQL <code>GROUP BY CUBE</code></i>) to explore the warehouse with full freedom.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    available_marts = get_available_marts()
    c_mart, c_op, c_chart = st.columns([1.4, 1.3, 1.1])
    with c_mart:
        selected_mart = st.selectbox(
            "1️⃣ Select Any Mart or Fact Table (`olist.marts`)",
            options=available_marts,
            format_func=lambda m: f"{m} — {MART_CATALOG.get(m, 'Analytical Table')}",
            key="univ_olap_mart",
        )
    with c_op:
        olap_op = st.selectbox(
            "2️⃣ Select OLAP Cube Operation",
            options=[
                "Roll-Up / Multi-Dim Group By",
                "Slice (Single Dimension Filter)",
                "Dice (Multi-Dimension Sub-Cube)",
                "Drill-Down (2-Level Hierarchy)",
                "Pivot (2D Cross-Tab Matrix)",
                "ANSI SQL GROUP BY CUBE (Full 2^N Lattice)",
            ],
            key="univ_olap_op",
        )
    with c_chart:
        chart_type = st.selectbox(
            "3️⃣ Visualization Type",
            options=[
                "Bar Chart",
                "Horizontal Bar",
                "Grouped / Stacked Bar",
                "Line / Trend Chart",
                "Area Chart",
                "Donut / Share Chart",
                "Pivot Heatmap Matrix",
                "Treemap Hierarchy",
            ],
            key="univ_olap_chart",
        )

    dimensions, measures, _ = get_table_schema_metadata(selected_mart)
    if not dimensions:
        st.warning(f"Could not load columns for `olist.marts.{selected_mart}`.")
        return

    # Sensible default selections per mart
    default_dim1 = next(
        (d for d in ["customer_state", "product_category_name", "seller_state", "payment_type", "ai_sentiment", "review_sentiment_category", "YEAR_MONTH(order_purchase_timestamp)"] if d in dimensions),
        dimensions[0],
    )
    default_dim2 = next(
        (d for d in ["YEAR(order_purchase_timestamp)", "order_status", "ai_topic", "seller_state", "customer_state"] if d in dimensions and d != default_dim1),
        "(None)",
    )
    default_measure = next(
        (m for m in ["total_item_value", "total_sales_value", "total_product_revenue", "total_payment_value", "delivery_delay_days", "review_score", "total_orders", "price"] if m in measures),
        measures[0],
    )

    col_d1, col_d2, col_m, col_agg, col_lim = st.columns([1.2, 1.2, 1.1, 0.9, 0.7])
    with col_d1:
        dim1 = st.selectbox(
            "Primary Dimension (Axis / Row)",
            options=dimensions,
            index=dimensions.index(default_dim1) if default_dim1 in dimensions else 0,
            key=f"olap_d1_{selected_mart}",
        )
    with col_d2:
        dim2_opts = ["(None)"] + [d for d in dimensions if d != dim1]
        dim2 = st.selectbox(
            "Secondary Dimension (Color / Column / Drill)",
            options=dim2_opts,
            index=dim2_opts.index(default_dim2) if default_dim2 in dim2_opts else 0,
            key=f"olap_d2_{selected_mart}",
        )
    with col_m:
        measure_col = st.selectbox(
            "Measure Column",
            options=measures + ["order_id", "customer_unique_id", "*"],
            index=measures.index(default_measure) if default_measure in measures else 0,
            key=f"olap_m_{selected_mart}",
        )
    with col_agg:
        default_agg = "AVG" if "delay" in measure_col or "score" in measure_col or "average" in measure_col else "SUM"
        if measure_col in ("*", "order_id", "customer_unique_id"):
            default_agg = "COUNT DISTINCT" if measure_col != "*" else "COUNT"
        agg_opts = ["SUM", "AVG", "COUNT", "COUNT DISTINCT", "MAX", "MIN"]
        agg_func = st.selectbox(
            "Aggregation",
            options=agg_opts,
            index=agg_opts.index(default_agg),
            key=f"olap_agg_{selected_mart}",
        )
    with col_lim:
        top_n = st.number_input("Max Rows", min_value=5, max_value=500, value=50, step=5, key="olap_topn")

    dim1_expr, dim1_alias = _dim_sql_expr(dim1)
    has_dim2 = dim2 != "(None)" or olap_op in ("Drill-Down (2-Level Hierarchy)", "Pivot (2D Cross-Tab Matrix)", "ANSI SQL GROUP BY CUBE (Full 2^N Lattice)")
    if has_dim2 and dim2 == "(None)":
        fallback_d2 = next((d for d in dimensions if d != dim1), dim1)
        dim2_expr, dim2_alias = _dim_sql_expr(fallback_d2)
    elif dim2 != "(None)":
        dim2_expr, dim2_alias = _dim_sql_expr(dim2)
    else:
        dim2_expr, dim2_alias = "", ""

    # Optional Slice / Dice Filter Controls
    where_clauses = [f"{dim1_expr} IS NOT NULL"]
    if has_dim2 and dim2_expr:
        where_clauses.append(f"{dim2_expr} IS NOT NULL")

    if olap_op in ("Slice (Single Dimension Filter)", "Dice (Multi-Dimension Sub-Cube)"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            try:
                vals_df = q_df(f"SELECT DISTINCT {dim1_expr} AS v FROM olist.marts.{selected_mart} WHERE {dim1_expr} IS NOT NULL ORDER BY 1 LIMIT 100")
                distinct_v1 = [str(x) for x in vals_df["v"].tolist()]
            except Exception:
                distinct_v1 = []
            chosen_v1 = st.multiselect(
                f"Slice / Filter `{dim1_alias}` values (leave empty for Top values)",
                options=distinct_v1,
                default=distinct_v1[:3] if olap_op == "Slice (Single Dimension Filter)" and len(distinct_v1) >= 3 else [],
                key=f"slice_v1_{selected_mart}_{dim1_alias}",
            )
            if chosen_v1:
                safe_list = ", ".join("'" + v.replace("'", "''") + "'" for v in chosen_v1)
                where_clauses.append(f"{dim1_expr} IN ({safe_list})")
        if olap_op == "Dice (Multi-Dimension Sub-Cube)" and has_dim2 and dim2_expr:
            with f_col2:
                try:
                    vals_df2 = q_df(f"SELECT DISTINCT {dim2_expr} AS v FROM olist.marts.{selected_mart} WHERE {dim2_expr} IS NOT NULL ORDER BY 1 LIMIT 100")
                    distinct_v2 = [str(x) for x in vals_df2["v"].tolist()]
                except Exception:
                    distinct_v2 = []
                chosen_v2 = st.multiselect(
                    f"Dice Filter `{dim2_alias}` values",
                    options=distinct_v2,
                    default=distinct_v2[:3] if len(distinct_v2) >= 3 else [],
                    key=f"dice_v2_{selected_mart}_{dim2_alias}",
                )
                if chosen_v2:
                    safe_list2 = ", ".join("'" + v.replace("'", "''") + "'" for v in chosen_v2)
                    where_clauses.append(f"{dim2_expr} IN ({safe_list2})")

    # Build Measure SQL Expression
    if agg_func == "COUNT" or measure_col == "*":
        metric_expr = "COUNT(*)"
        metric_alias = "row_count"
    elif agg_func == "COUNT DISTINCT":
        metric_expr = f"COUNT(DISTINCT {measure_col})"
        metric_alias = f"distinct_{measure_col}"
    else:
        metric_expr = f"ROUND({agg_func}({measure_col}), 2)"
        metric_alias = f"{agg_func.lower()}_{measure_col}"

    where_sql = " AND ".join(where_clauses)

    if olap_op == "ANSI SQL GROUP BY CUBE (Full 2^N Lattice)" and has_dim2 and dim2_expr:
        cube_sql = f"""
            SELECT
                GROUPING_ID({dim1_expr}, {dim2_expr}) AS cuboid_id,
                COALESCE({dim1_expr}, 'ALL') AS {dim1_alias},
                COALESCE({dim2_expr}, 'ALL') AS {dim2_alias},
                {metric_expr} AS {metric_alias},
                COUNT(*) AS records_in_cell
            FROM olist.marts.{selected_mart}
            WHERE {where_sql}
            GROUP BY CUBE ({dim1_expr}, {dim2_expr})
            ORDER BY cuboid_id DESC, {metric_alias} DESC
            LIMIT {int(top_n)}
        """
    elif has_dim2 and dim2_expr:
        cube_sql = f"""
            SELECT
                {dim1_expr} AS {dim1_alias},
                {dim2_expr} AS {dim2_alias},
                {metric_expr} AS {metric_alias},
                COUNT(*) AS records_in_cell
            FROM olist.marts.{selected_mart}
            WHERE {where_sql}
            GROUP BY 1, 2
            ORDER BY {metric_alias} DESC
            LIMIT {int(top_n)}
        """
    else:
        cube_sql = f"""
            SELECT
                {dim1_expr} AS {dim1_alias},
                {metric_expr} AS {metric_alias},
                COUNT(*) AS records_in_cell
            FROM olist.marts.{selected_mart}
            WHERE {where_sql}
            GROUP BY 1
            ORDER BY {metric_alias} DESC
            LIMIT {int(top_n)}
        """

    try:
        df_cube = q_df(cube_sql)
    except Exception as exc:
        st.error(f"Could not execute OLAP query on `{selected_mart}`: {exc}")
        st.code(cube_sql, language="sql")
        return

    if df_cube.empty:
        st.info("No rows matched the current OLAP filter criteria.")
        return

    # KPI Summary Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Mart / Table", selected_mart)
    k2.metric("Cuboid Cells Returned", f"{len(df_cube):,}")
    k3.metric(f"Total / Sum `{metric_alias}`", f"{df_cube[metric_alias].sum():,.2f}")
    k4.metric(f"Peak `{metric_alias}`", f"{df_cube[metric_alias].max():,.2f}")

    # Render Visualization
    if (olap_op == "Pivot (2D Cross-Tab Matrix)" or chart_type == "Pivot Heatmap Matrix") and has_dim2 and dim2_alias in df_cube.columns:
        piv = df_cube.pivot_table(index=dim1_alias, columns=dim2_alias, values=metric_alias, aggfunc="sum", fill_value=0)
        fig = px.imshow(
            piv,
            text_auto=".2s",
            aspect="auto",
            color_continuous_scale="Blues",
            title=f"OLAP Pivot Matrix: {metric_alias} by {dim1_alias} × {dim2_alias} ({selected_mart})",
        )
        fig.update_layout(height=460, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Treemap Hierarchy" and has_dim2 and dim2_alias in df_cube.columns:
        fig = px.treemap(
            df_cube,
            path=[dim1_alias, dim2_alias],
            values=metric_alias,
            color=metric_alias,
            color_continuous_scale="Tealgrn",
            title=f"OLAP Drill-Down Treemap: {dim1_alias} → {dim2_alias} ({selected_mart})",
        )
        fig.update_layout(height=460, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Donut / Share Chart":
        fig = px.pie(
            df_cube.head(15),
            names=dim1_alias,
            values=metric_alias,
            hole=0.45,
            title=f"Share of {metric_alias} by {dim1_alias} ({selected_mart})",
        )
        fig.update_layout(height=430, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type in ("Line / Trend Chart", "Area Chart"):
        df_sorted = df_cube.sort_values(by=dim1_alias)
        fn = px.area if chart_type == "Area Chart" else px.line
        fig = fn(
            df_sorted,
            x=dim1_alias,
            y=metric_alias,
            color=dim2_alias if (has_dim2 and dim2_alias in df_cube.columns) else None,
            markers=True,
            title=f"{metric_alias} Trend Across {dim1_alias} ({selected_mart})",
        )
        fig.update_layout(height=430, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Horizontal Bar":
        fig = px.bar(
            df_cube.head(25).iloc[::-1],
            x=metric_alias,
            y=dim1_alias,
            color=dim2_alias if (has_dim2 and dim2_alias in df_cube.columns) else metric_alias,
            orientation="h",
            title=f"{metric_alias} by {dim1_alias} ({selected_mart})",
        )
        fig.update_layout(height=450, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.bar(
            df_cube.head(30),
            x=dim1_alias,
            y=metric_alias,
            color=dim2_alias if (has_dim2 and dim2_alias in df_cube.columns) else metric_alias,
            barmode="group" if chart_type == "Grouped / Stacked Bar" else "relative",
            title=f"{olap_op}: {metric_alias} by {dim1_alias}" + (f" × {dim2_alias}" if has_dim2 and dim2_alias in df_cube.columns else ""),
        )
        fig.update_layout(height=430, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View OLAP Result Table, Generated SQL & Download CSV", expanded=False):
        st.code(" ".join(cube_sql.split()), language="sql")
        st.dataframe(df_cube, use_container_width=True)
        st.download_button(
            "⬇️ Download OLAP Cube CSV",
            data=df_cube.to_csv(index=False).encode("utf-8"),
            file_name=f"olap_{selected_mart}_{metric_alias}.csv",
            mime="text/csv",
            key=f"dl_olap_{selected_mart}",
        )


def render_bi_dashboard(
    embedded: bool = False,
    date_range: Optional[Tuple[Any, Any]] = None,
    selected_states: Optional[List[str]] = None,
    selected_statuses: Optional[List[str]] = None,
) -> None:
    if not embedded:
        st.set_page_config(page_title="Olist 8-Section BI & Universal OLAP Cube", layout="wide")
        st.title("📈 Olist 8-Section BI & Universal OLAP Cube Dashboard")

    # 1. Universal Multi-Mart OLAP Cube Explorer at the top
    render_universal_olap_cube_explorer()

    st.markdown("### 📊 Complete 8-Section Departmental Business Intelligence Suite")

    where_parts = ["1=1"]
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        where_parts.append(
            f"CAST(order_purchase_timestamp AS DATE) BETWEEN DATE '{date_range[0]}' AND DATE '{date_range[1]}'"
        )
    if selected_states:
        st_list = ", ".join("'" + s.replace("'", "''") + "'" for s in selected_states)
        where_parts.append(f"customer_state IN ({st_list})")
    if selected_statuses:
        status_list = ", ".join("'" + s.replace("'", "''") + "'" for s in selected_statuses)
        where_parts.append(f"order_status IN ({status_list})")
    sales_where = " AND ".join(where_parts)

    tabs = st.tabs([
        "1️⃣ Executive Overview",
        "2️⃣ Sales Analytics",
        "3️⃣ Customer Analytics",
        "4️⃣ Product Analytics",
        "5️⃣ Seller Analytics",
        "6️⃣ Delivery Performance",
        "7️⃣ Payment Analytics",
        "8️⃣ Reviews & AI Sentiment",
    ])

    # Tab 1: Executive Overview
    with tabs[0]:
        kpi_df = q_df(f"""
            SELECT
                ROUND(SUM(total_item_value), 2) AS total_gmv,
                COUNT(DISTINCT order_id) AS total_orders,
                COUNT(DISTINCT customer_unique_id) AS total_customers,
                COUNT(*) AS total_items,
                ROUND(SUM(total_item_value) / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS aov
            FROM olist.marts.mart_sales
            WHERE {sales_where}
        """)
        rev_kpi = q_df("SELECT ROUND(AVG(review_score), 2) AS avg_score FROM olist.marts.mart_review_analysis")
        del_kpi = q_df("SELECT ROUND(AVG(delivery_delay_days), 2) AS avg_delay FROM olist.marts.mart_delivery_performance")

        r = kpi_df.iloc[0]
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Sales (GMV)", f"R$ {float(r['total_gmv'] or 0):,.2f}")
        c2.metric("Distinct Orders", f"{int(r['total_orders'] or 0):,}")
        c3.metric("Unique Customers", f"{int(r['total_customers'] or 0):,}")
        c4.metric("Order Average (AOV)", f"R$ {float(r['aov'] or 0):,.2f}")
        c5.metric("Avg Review Score", f"{float(rev_kpi.iloc[0]['avg_score'] or 0):.2f} / 5.0")
        c6.metric("Avg Delivery Delay", f"{float(del_kpi.iloc[0]['avg_delay'] or 0):.2f} days")

        monthly_df = q_df(f"""
            SELECT STRFTIME(order_purchase_timestamp, '%Y-%m') AS year_month,
                   ROUND(SUM(total_item_value), 2) AS monthly_gmv,
                   COUNT(DISTINCT order_id) AS monthly_orders
            FROM olist.marts.mart_sales
            WHERE {sales_where} AND order_purchase_timestamp IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """)
        fig_m = px.area(monthly_df, x="year_month", y="monthly_gmv", markers=True, title="Monthly Gross Merchandise Value (GMV) Trend")
        fig_m.update_layout(height=360, template="plotly_dark")
        st.plotly_chart(fig_m, use_container_width=True)

    # Tab 2: Sales Analytics
    with tabs[1]:
        st_sales = q_df(f"""
            SELECT customer_state, ROUND(SUM(total_item_value), 2) AS state_sales, COUNT(DISTINCT order_id) AS orders
            FROM olist.marts.mart_sales
            WHERE {sales_where}
            GROUP BY 1 ORDER BY 2 DESC LIMIT 15
        """)
        fig_s = px.bar(st_sales, x="customer_state", y="state_sales", color="orders", title="Top 15 Customer States by GMV (`mart_sales`)")
        fig_s.update_layout(height=380, template="plotly_dark")
        st.plotly_chart(fig_s, use_container_width=True)

    # Tab 3: Customer Analytics
    with tabs[2]:
        cust_df = q_df("""
            SELECT
                CASE WHEN total_orders > 1 THEN 'Repeat Customer (>1 Order)' ELSE 'One-Time Customer (1 Order)' END AS customer_type,
                COUNT(*) AS customer_count
            FROM olist.marts.mart_customer_behavior
            GROUP BY 1
        """)
        fig_c = px.pie(cust_df, names="customer_type", values="customer_count", hole=0.45, title="Customer Retention Split (`mart_customer_behavior`)")
        fig_c.update_layout(height=380, template="plotly_dark")
        st.plotly_chart(fig_c, use_container_width=True)

    # Tab 4: Product Analytics
    with tabs[3]:
        prod_df = q_df("""
            SELECT COALESCE(product_category_name, 'unknown') AS category,
                   ROUND(SUM(total_sales_value), 2) AS total_sales,
                   SUM(total_items_sold) AS items_sold
            FROM olist.marts.mart_product_performance
            GROUP BY 1 ORDER BY 2 DESC LIMIT 15
        """)
        fig_p = px.bar(prod_df.iloc[::-1], x="total_sales", y="category", orientation="h", color="items_sold", title="Top 15 Product Categories by Revenue (`mart_product_performance`)")
        fig_p.update_layout(height=420, template="plotly_dark")
        st.plotly_chart(fig_p, use_container_width=True)

    # Tab 5: Seller Analytics
    with tabs[4]:
        sell_df = q_df("""
            SELECT seller_id, seller_state, ROUND(total_product_revenue, 2) AS product_revenue, total_items_sold
            FROM olist.marts.mart_seller_performance
            ORDER BY total_product_revenue DESC LIMIT 10
        """)
        fig_sel = px.bar(sell_df, x="seller_id", y="product_revenue", color="seller_state", title="Top 10 Sellers by Product Revenue (`mart_seller_performance`)")
        fig_sel.update_layout(height=380, template="plotly_dark")
        st.plotly_chart(fig_sel, use_container_width=True)

    # Tab 6: Delivery Performance
    with tabs[5]:
        del_df = q_df("""
            SELECT customer_state,
                   ROUND(AVG(delivery_delay_days), 2) AS avg_delay_days,
                   ROUND(100.0 * SUM(CASE WHEN delivery_delay_days > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_order_pct
            FROM olist.marts.mart_delivery_performance
            GROUP BY 1 ORDER BY late_order_pct DESC LIMIT 20
        """)
        fig_del = px.bar(del_df, x="customer_state", y="late_order_pct", color="avg_delay_days", title="Late Delivery Percentage & Avg Delay Days by State (`mart_delivery_performance`)")
        fig_del.update_layout(height=380, template="plotly_dark")
        st.plotly_chart(fig_del, use_container_width=True)

    # Tab 7: Payment Analytics
    with tabs[6]:
        pay_df = q_df("""
            SELECT payment_type, total_orders, ROUND(total_payment_value, 2) AS total_payment_value, ROUND(average_installments, 2) AS avg_installments
            FROM olist.marts.mart_payment_analysis
            ORDER BY total_payment_value DESC
        """)
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            fig_pay1 = px.pie(pay_df, names="payment_type", values="total_payment_value", hole=0.45, title="Payment Value Share (`mart_payment_analysis`)")
            fig_pay1.update_layout(height=360, template="plotly_dark")
            st.plotly_chart(fig_pay1, use_container_width=True)
        with c_p2:
            fig_pay2 = px.bar(pay_df, x="payment_type", y="avg_installments", color="payment_type", title="Average Installments by Payment Method")
            fig_pay2.update_layout(height=360, template="plotly_dark")
            st.plotly_chart(fig_pay2, use_container_width=True)

    # Tab 8: Customer Reviews & AI Sentiment
    with tabs[7]:
        ai_sent = q_df("""
            SELECT ai_sentiment, COUNT(*) AS review_count
            FROM olist.marts.mart_enriched_review_analysis
            GROUP BY 1 ORDER BY 2 DESC
        """)
        ai_topic = q_df("""
            SELECT ai_topic, COUNT(*) AS count
            FROM olist.marts.mart_enriched_review_analysis
            GROUP BY 1 ORDER BY 2 DESC LIMIT 10
        """)
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            fig_rs = px.pie(ai_sent, names="ai_sentiment", values="review_count", hole=0.45, title="Groq LLM Sentiment Distribution (`mart_enriched_review_analysis`)")
            fig_rs.update_layout(height=360, template="plotly_dark")
            st.plotly_chart(fig_rs, use_container_width=True)
        with r_c2:
            fig_rt = px.bar(ai_topic, x="ai_topic", y="count", color="ai_topic", title="Top AI Review Topics (`mart_enriched_review_analysis`)")
            fig_rt.update_layout(height=360, template="plotly_dark")
            st.plotly_chart(fig_rt, use_container_width=True)


if __name__ == "__main__":
    render_bi_dashboard(embedded=False)
