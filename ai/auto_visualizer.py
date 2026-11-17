from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

try:
    import plotly.express as px
except ImportError:
    px = None

try:
    import streamlit as st
except ImportError:
    st = None


def _is_temporal_column(col_name: str) -> bool:
    lower_c = str(col_name).lower()
    return any(
        k in lower_c
        for k in ("date", "month", "year", "timestamp", "quarter")
    )


def determine_recommended_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Inspect a result DataFrame and return recommended chart specifications
    for automated visualization (used by unit tests, Copilot, and Web UI).
    """
    if df is None or df.empty:
        return []

    num_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if not str(c).endswith("_pct_of_total")
    ]
    cat_cols = [c for c in df.columns if c not in df.select_dtypes(include=["number"]).columns]

    if len(df) == 1 and num_cols:
        return [
            {
                "chart_type": "kpi_cards",
                "metrics": num_cols[:4],
                "title": "Scalar KPI Summary",
            }
        ]

    if not num_cols:
        return [{"chart_type": "table", "columns": list(df.columns), "title": "Result Table"}]

    primary_num = num_cols[0]
    primary_cat = cat_cols[0] if cat_cols else df.columns[0]
    is_time = _is_temporal_column(primary_cat)

    if is_time:
        return [
            {
                "chart_type": "line",
                "x": primary_cat,
                "y": primary_num,
                "title": f"Trend of {primary_num} over {primary_cat}",
            },
            {
                "chart_type": "bar",
                "x": primary_cat,
                "y": primary_num,
                "title": f"{primary_num} by {primary_cat}",
            },
        ]

    specs: List[Dict[str, Any]] = [
        {
            "chart_type": "bar",
            "x": primary_cat,
            "y": primary_num,
            "title": f"{primary_num} by {primary_cat}",
        }
    ]
    if len(df) <= 12:
        specs.append(
            {
                "chart_type": "pie",
                "names": primary_cat,
                "values": primary_num,
                "title": f"Share of {primary_num} by {primary_cat}",
            }
        )
    else:
        specs.append(
            {
                "chart_type": "treemap",
                "path": [primary_cat],
                "values": primary_num,
                "title": f"Distribution of {primary_num} by {primary_cat}",
            }
        )
    return specs


def render_auto_visualizations(
    df: pd.DataFrame,
    question: str = "",
    prompt_context: str = "",
    key_prefix: str = "viz",
) -> None:
    if st is None or px is None:
        return

    _ = question or prompt_context
    if df is None or df.empty:
        st.info("Query returned 0 rows.")
        return

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Case 1: Single scalar KPI result (1 row)
    if len(df) == 1 and num_cols:
        cols = st.columns(min(4, len(num_cols)))
        for idx, nc in enumerate(num_cols[:4]):
            val = float(df[nc].iloc[0] or 0)
            cols[idx].metric(
                nc.replace("_", " ").title(),
                f"{val:,.2f}" if val % 1 != 0 else f"{int(val):,}",
            )
        st.dataframe(df, use_container_width=True)
        return

    if not num_cols:
        st.dataframe(df, use_container_width=True)
        return

    primary_num = next(
        (c for c in num_cols if not c.endswith("_pct") and not c.endswith("_pct_of_total")),
        num_cols[0],
    )
    primary_cat = cat_cols[0] if cat_cols else df.columns[0]

    # KPI Summary Strip for multi-row result
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows Returned", f"{len(df):,}")
    k2.metric(f"Sum ({primary_num})", f"{float(df[primary_num].sum()):,.2f}")
    k3.metric(f"Average ({primary_num})", f"{float(df[primary_num].mean()):,.2f}")
    k4.metric(f"Peak ({primary_num})", f"{float(df[primary_num].max()):,.2f}")

    # Side-by-side Primary Visual + Share/Distribution Visual
    is_time = _is_temporal_column(primary_cat)
    c_main, c_side = st.columns([1.45, 1.0])

    with c_main:
        if is_time:
            df_sorted = df.sort_values(by=primary_cat)
            fig1 = px.area(
                df_sorted,
                x=primary_cat,
                y=primary_num,
                markers=True,
                title=f"Trend of {primary_num} over {primary_cat}",
            )
        elif len(df) > 10:
            fig1 = px.bar(
                df.head(20).iloc[::-1],
                x=primary_num,
                y=primary_cat,
                orientation="h",
                color=primary_num,
                color_continuous_scale="Cyan",
                title=f"Top {min(20, len(df))} {primary_cat} by {primary_num}",
            )
        else:
            fig1 = px.bar(
                df,
                x=primary_cat,
                y=primary_num,
                color=primary_cat,
                text_auto=".2s",
                title=f"{primary_num} by {primary_cat}",
            )
        fig1.update_layout(height=400, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True, key=f"{key_prefix}_main_chart")

    with c_side:
        if len(df) <= 12 and (df[primary_num] >= 0).all():
            fig2 = px.pie(
                df,
                names=primary_cat,
                values=primary_num,
                hole=0.48,
                title=f"Proportional Share of {primary_num}",
            )
        else:
            fig2 = px.treemap(
                df.head(25),
                path=[primary_cat],
                values=primary_num if (df[primary_num] > 0).all() else None,
                color=primary_num,
                color_continuous_scale="Blues",
                title=f"Hierarchy / Distribution Map ({primary_num})",
            )
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_side_chart")

    # Interactive Data Table with Visual Progress Bars on Numeric Columns
    col_cfg = {}
    max_v = float(df[primary_num].max() or 1.0)
    min_v = float(df[primary_num].min() or 0.0)
    if min_v >= 0 and max_v > 0:
        col_cfg[primary_num] = st.column_config.ProgressColumn(
            label=f"{primary_num} (Visual Scale)",
            format="%.2f",
            min_value=0.0,
            max_value=max_v,
        )

    st.markdown("##### 📊 Interactive Result Explorer (Sortable with Inline Visual Bars)")
    st.dataframe(df, column_config=col_cfg, use_container_width=True)

    with st.expander("🎨 Customize Chart Type & Axes", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        custom_type = cc1.selectbox(
            "Chart Style",
            ["Bar", "Horizontal Bar", "Line", "Area", "Scatter", "Pie / Donut"],
            key=f"{key_prefix}_ctype",
        )
        x_col = cc2.selectbox(
            "X Axis",
            df.columns.tolist(),
            index=df.columns.get_loc(primary_cat),
            key=f"{key_prefix}_x",
        )
        y_col = cc3.selectbox(
            "Y Axis",
            num_cols,
            index=num_cols.index(primary_num),
            key=f"{key_prefix}_y",
        )
        if custom_type == "Bar":
            cfig = px.bar(df.head(40), x=x_col, y=y_col, color=x_col)
        elif custom_type == "Horizontal Bar":
            cfig = px.bar(df.head(40), x=y_col, y=x_col, orientation="h", color=y_col)
        elif custom_type == "Line":
            cfig = px.line(df, x=x_col, y=y_col, markers=True)
        elif custom_type == "Area":
            cfig = px.area(df, x=x_col, y=y_col)
        elif custom_type == "Scatter":
            cfig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                size=y_col if (df[y_col] > 0).all() else None,
                color=x_col,
            )
        else:
            cfig = px.pie(df.head(15), names=x_col, values=y_col, hole=0.4)
        cfig.update_layout(height=380, template="plotly_dark")
        st.plotly_chart(cfig, use_container_width=True, key=f"{key_prefix}_custom_fig")


def render_rag_evidence_charts(retrieved_reviews, key_prefix: str = "rag_viz") -> None:
    """Visualize sentiment, topic, rating, and relevance in retrieved RAG evidence."""
    if st is None or px is None:
        return

    if not retrieved_reviews:
        st.info("No retrieved review evidence is available to visualize.")
        return

    rows = []
    for item in retrieved_reviews:
        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        rows.append(
            {
                "review_id": str(item.get("review_id", "Unknown")),
                "rating": pd.to_numeric(
                    metadata.get("review_score", item.get("rating")), errors="coerce"
                ),
                "sentiment": metadata.get("ai_sentiment")
                or item.get("ai_sentiment")
                or "Unknown",
                "topic": metadata.get("ai_topic") or item.get("ai_topic") or "Unknown",
                "issue_category": metadata.get("ai_issue_category")
                or item.get("ai_issue_category")
                or "Unknown",
                "relevance_score": pd.to_numeric(
                    item.get("relevance_score"), errors="coerce"
                ),
                "distance": pd.to_numeric(item.get("distance"), errors="coerce"),
            }
        )

    evidence_df = pd.DataFrame(rows)
    if evidence_df.empty:
        st.info("Retrieved reviews do not contain usable evidence fields.")
        return

    st.subheader("📊 Visual Analysis of Retrieved Review Evidence")

    chart_cols = st.columns(2)

    sentiment_df = (
        evidence_df["sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )
    with chart_cols[0]:
        fig_sentiment = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Reviews",
            color="Sentiment",
            title="Retrieved Reviews by AI Sentiment",
            text="Reviews",
        )
        fig_sentiment.update_layout(height=350, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_sentiment, use_container_width=True, key=f"{key_prefix}_sentiment")

    topic_df = (
        evidence_df["topic"]
        .value_counts()
        .head(10)
        .rename_axis("Topic")
        .reset_index(name="Reviews")
    )
    with chart_cols[1]:
        fig_topic = px.bar(
            topic_df,
            x="Reviews",
            y="Topic",
            orientation="h",
            color="Reviews",
            title="Retrieved Reviews by AI Topic",
            text="Reviews",
        )
        fig_topic.update_layout(height=350, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_topic, use_container_width=True, key=f"{key_prefix}_topic")

    rating_df = evidence_df.dropna(subset=["rating"])
    relevance_df = evidence_df.dropna(subset=["relevance_score"])

    extra_cols = st.columns(2)
    with extra_cols[0]:
        if not rating_df.empty:
            rating_counts = (
                rating_df["rating"]
                .value_counts()
                .sort_index()
                .rename_axis("Rating")
                .reset_index(name="Reviews")
            )
            fig_rating = px.bar(
                rating_counts,
                x="Rating",
                y="Reviews",
                title="Retrieved Reviews by Star Rating",
                text="Reviews",
            )
            fig_rating.update_layout(height=330, template="plotly_dark")
            st.plotly_chart(fig_rating, use_container_width=True, key=f"{key_prefix}_rating")
        else:
            st.caption("No numeric review ratings were found in the retrieved evidence.")

    with extra_cols[1]:
        if not relevance_df.empty:
            fig_relevance = px.bar(
                relevance_df.sort_values("relevance_score", ascending=True),
                x="relevance_score",
                y="review_id",
                orientation="h",
                title="Opinion/Relevance Score by Retrieved Review",
                labels={"relevance_score": "Relevance score", "review_id": "Review ID"},
            )
            fig_relevance.update_layout(height=330, template="plotly_dark")
            st.plotly_chart(fig_relevance, use_container_width=True, key=f"{key_prefix}_relevance")
        else:
            st.caption("No numeric relevance scores were found in the retrieved evidence.")

    with st.expander("View chart source data", expanded=False):
        st.dataframe(evidence_df, use_container_width=True)
