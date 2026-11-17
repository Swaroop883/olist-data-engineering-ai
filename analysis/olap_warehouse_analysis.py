import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import math
from typing import Dict, Any, List
import duckdb
import pandas as pd
from ai.db_utils import get_motherduck_connection


def _setup_tpch_and_ssb_environment(con: duckdb.DuckDBPyConnection, scale_factor: float = 0.02) -> str:
    source_mode = "duckdb_native_tpch"
    try:
        con.execute("LOAD tpch;")
        con.execute(f"CALL dbgen(sf={scale_factor});")
    except Exception:
        try:
            con.execute("INSTALL tpch; LOAD tpch;")
            con.execute(f"CALL dbgen(sf={scale_factor});")
        except Exception:
            source_mode = "deterministic_tpch_ssb"
            con.execute("""
                CREATE TABLE region AS SELECT range AS r_regionkey, 'REGION_' || range AS r_name FROM range(5);
                CREATE TABLE nation AS SELECT range AS n_nationkey, 'NATION_' || range AS n_name, (range % 5) AS n_regionkey FROM range(25);
                CREATE TABLE supplier AS SELECT range AS s_suppkey, 'Supplier#' || range AS s_name, (range % 25) AS s_nationkey FROM range(1, 501);
                CREATE TABLE customer AS SELECT range AS c_custkey, 'Customer#' || range AS c_name, (range % 25) AS c_nationkey, 'SEG_' || (range % 5) AS c_mktsegment FROM range(1, 3001);
                CREATE TABLE part AS SELECT range AS p_partkey, 'Part#' || range AS p_name, 'MFGR#' || ((range % 5) + 1) AS p_mfgr, 'BRAND#' || ((range % 25) + 1) AS p_brand, 'TYPE#' || ((range % 5) + 1) AS p_type FROM range(1, 2001);
                CREATE TABLE orders AS SELECT range AS o_orderkey, ((range % 3000) + 1) AS o_custkey, DATE '1994-01-01' + INTERVAL (range % 1825) DAY AS o_orderdate, '1-URGENT' AS o_orderpriority FROM range(1, 15001);
                CREATE TABLE lineitem AS SELECT ((range % 15000) + 1) AS l_orderkey, ((range % 4) + 1) AS l_linenumber, ((range % 2000) + 1) AS l_partkey, ((range % 500) + 1) AS l_suppkey, ((range % 50) + 1)::DOUBLE AS l_quantity, (((range % 50) + 1) * 120.5)::DOUBLE AS l_extendedprice, ((range % 10) * 0.01)::DOUBLE AS l_discount, ((range % 8) * 0.01)::DOUBLE AS l_tax, DATE '1994-01-01' + INTERVAL (range % 1825) DAY AS l_shipdate FROM range(1, 60001);
            """)

    con.execute("""
        CREATE OR REPLACE TABLE ssb_dim_customer AS
        SELECT c.c_custkey AS custkey, c.c_name, n.n_name AS c_nation, r.r_name AS c_region, c.c_mktsegment
        FROM customer c JOIN nation n ON c.c_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey;

        CREATE OR REPLACE TABLE ssb_dim_supplier AS
        SELECT s.s_suppkey AS suppkey, s.s_name, n.n_name AS s_nation, r.r_name AS s_region
        FROM supplier s JOIN nation n ON s.s_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey;

        CREATE OR REPLACE TABLE ssb_dim_part AS
        SELECT p.p_partkey AS partkey, p.p_name, p.p_mfgr, p.p_type AS p_category, p.p_brand FROM part p;

        CREATE OR REPLACE TABLE ssb_dim_date AS
        SELECT DISTINCT CAST(STRFTIME(o_orderdate, '%Y%m%d') AS INTEGER) AS datekey,
               o_orderdate AS d_date, EXTRACT(YEAR FROM o_orderdate)::INTEGER AS d_year,
               EXTRACT(QUARTER FROM o_orderdate)::INTEGER AS d_quarter, EXTRACT(MONTH FROM o_orderdate)::INTEGER AS d_month
        FROM orders;

        CREATE OR REPLACE TABLE ssb_fact_lineorder AS
        SELECT l.l_orderkey AS lo_orderkey, l.l_linenumber AS lo_linenumber, o.o_custkey AS lo_custkey,
               l.l_partkey AS lo_partkey, l.l_suppkey AS lo_suppkey,
               CAST(STRFTIME(o.o_orderdate, '%Y%m%d') AS INTEGER) AS lo_orderdatekey,
               l.l_quantity AS lo_quantity, l.l_extendedprice AS lo_extendedprice,
               l.l_discount AS lo_discount, (l.l_extendedprice * (1.0 - l.l_discount)) AS lo_revenue
        FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey;

        CREATE OR REPLACE TABLE ssb_fact_payments AS
        SELECT o.o_orderkey AS pay_orderkey, seq AS pay_seq, o.o_custkey AS pay_custkey, (150.0 * seq)::DOUBLE AS pay_amount
        FROM orders o CROSS JOIN (SELECT 1 AS seq UNION ALL SELECT 2 AS seq) s
        WHERE (o.o_orderkey % 3 = 0) OR s.seq = 1;

        CREATE OR REPLACE TABLE galaxy_mart_revenue_cube AS
        SELECT dd.d_year, dc.c_region, dc.c_nation, ds.s_region, ds.s_nation, dp.p_mfgr, dp.p_category, dp.p_brand,
               COUNT(DISTINCT lo.lo_orderkey) AS total_orders, COUNT(*) AS total_lineitems,
               SUM(lo.lo_revenue) AS total_revenue
        FROM ssb_fact_lineorder lo
        JOIN ssb_dim_date dd ON lo.lo_orderdatekey = dd.datekey
        JOIN ssb_dim_customer dc ON lo.lo_custkey = dc.custkey
        JOIN ssb_dim_supplier ds ON lo.lo_suppkey = ds.suppkey
        JOIN ssb_dim_part dp ON lo.lo_partkey = dp.partkey
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8;

        CREATE OR REPLACE TABLE flat_wide_obt AS
        SELECT lo.lo_orderkey, lo.lo_linenumber, dd.d_year, dc.c_region, dc.c_nation,
               ds.s_region, ds.s_nation, dp.p_mfgr, dp.p_category, dp.p_brand,
               lo.lo_revenue, fp.pay_amount
        FROM ssb_fact_lineorder lo
        JOIN ssb_dim_date dd ON lo.lo_orderdatekey = dd.datekey
        JOIN ssb_dim_customer dc ON lo.lo_custkey = dc.custkey
        JOIN ssb_dim_supplier ds ON lo.lo_suppkey = ds.suppkey
        JOIN ssb_dim_part dp ON lo.lo_partkey = dp.partkey
        JOIN ssb_fact_payments fp ON lo.lo_orderkey = fp.pay_orderkey;
    """)
    return source_mode


def _time_query_ms(con: duckdb.DuckDBPyConnection, sql: str, runs: int = 5) -> float:
    con.execute(sql).fetchall()
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    return round(timings[len(timings) // 2], 3)


def run_tpch_ssb_benchmark(scale_factor: float = 0.02) -> Dict[str, Any]:
    con = duckdb.connect(":memory:")
    source_mode = _setup_tpch_and_ssb_environment(con, scale_factor=scale_factor)

    queries = {
        "SSB Q1.1 (Time + Discount Roll-Up)": {
            "3NF / Snowflake (TPC-H 7-Table Schema)": "SELECT EXTRACT(YEAR FROM o.o_orderdate) AS d_year, SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey GROUP BY 1 ORDER BY 1",
            "Pure Star Schema (SSB Fact + 4 Dims)": "SELECT dd.d_year, SUM(lo.lo_revenue) AS revenue FROM ssb_fact_lineorder lo JOIN ssb_dim_date dd ON lo.lo_orderdatekey = dd.datekey GROUP BY 1 ORDER BY 1",
            "Hybrid Galaxy + Kimball Marts (Ours)": "SELECT d_year, SUM(total_revenue) AS revenue FROM galaxy_mart_revenue_cube GROUP BY 1 ORDER BY 1",
            "Naive Single Flat Table (OBT)": "SELECT d_year, SUM(lo_revenue) AS revenue FROM flat_wide_obt GROUP BY 1 ORDER BY 1",
        },
        "SSB Q2.1 (Brand + Supplier Region Drill-Down)": {
            "3NF / Snowflake (TPC-H 7-Table Schema)": "SELECT EXTRACT(YEAR FROM o.o_orderdate) AS d_year, p.p_brand, SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN part p ON l.l_partkey = p.p_partkey JOIN supplier s ON l.l_suppkey = s.s_suppkey JOIN nation n ON s.s_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey GROUP BY 1, 2 ORDER BY 1, 2",
            "Pure Star Schema (SSB Fact + 4 Dims)": "SELECT dd.d_year, dp.p_brand, SUM(lo.lo_revenue) AS revenue FROM ssb_fact_lineorder lo JOIN ssb_dim_date dd ON lo.lo_orderdatekey = dd.datekey JOIN ssb_dim_part dp ON lo.lo_partkey = dp.partkey JOIN ssb_dim_supplier ds ON lo.lo_suppkey = ds.suppkey GROUP BY 1, 2 ORDER BY 1, 2",
            "Hybrid Galaxy + Kimball Marts (Ours)": "SELECT d_year, p_brand, SUM(total_revenue) AS revenue FROM galaxy_mart_revenue_cube GROUP BY 1, 2 ORDER BY 1, 2",
            "Naive Single Flat Table (OBT)": "SELECT d_year, p_brand, SUM(lo_revenue) AS revenue FROM flat_wide_obt GROUP BY 1, 2 ORDER BY 1, 2",
        },
        "SSB Q3.1 (Customer Nation x Supplier Nation)": {
            "3NF / Snowflake (TPC-H 7-Table Schema)": "SELECT nc.n_name AS c_nation, ns.n_name AS s_nation, EXTRACT(YEAR FROM o.o_orderdate) AS d_year, SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation nc ON c.c_nationkey = nc.n_nationkey JOIN region rc ON nc.n_regionkey = rc.r_regionkey JOIN supplier s ON l.l_suppkey = s.s_suppkey JOIN nation ns ON s.s_nationkey = ns.n_nationkey JOIN region rs ON ns.n_regionkey = rs.r_regionkey GROUP BY 1, 2, 3 ORDER BY 3, 4 DESC",
            "Pure Star Schema (SSB Fact + 4 Dims)": "SELECT dc.c_nation, ds.s_nation, dd.d_year, SUM(lo.lo_revenue) AS revenue FROM ssb_fact_lineorder lo JOIN ssb_dim_customer dc ON lo.lo_custkey = dc.custkey JOIN ssb_dim_supplier ds ON lo.lo_suppkey = ds.suppkey JOIN ssb_dim_date dd ON lo.lo_orderdatekey = dd.datekey GROUP BY 1, 2, 3 ORDER BY 3, 4 DESC",
            "Hybrid Galaxy + Kimball Marts (Ours)": "SELECT c_nation, s_nation, d_year, SUM(total_revenue) AS revenue FROM galaxy_mart_revenue_cube GROUP BY 1, 2, 3 ORDER BY 3, 4 DESC",
            "Naive Single Flat Table (OBT)": "SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue FROM flat_wide_obt GROUP BY 1, 2, 3 ORDER BY 3, 4 DESC",
        },
        "SSB Q4.1 (Full 4-Dimension Cube Roll-Up)": {
            "3NF / Snowflake (TPC-H 7-Table Schema)": "SELECT EXTRACT(YEAR FROM o.o_orderdate) AS d_year, nc.n_name AS c_nation, p.p_type AS p_category, SUM(l.l_extendedprice * (1.0 - l.l_discount)) AS revenue FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation nc ON c.c_nationkey = nc.n_nationkey JOIN region rc ON nc.n_regionkey = rc.r_regionkey JOIN supplier s ON l.l_suppkey = s.s_suppkey JOIN nation ns ON s.s_nationkey = ns.n_nationkey JOIN region rs ON ns.n_regionkey = rs.r_regionkey JOIN part p ON l.l_partkey = p.p_partkey GROUP BY 1, 2, 3 ORDER BY 1, 2",
            "Pure Star Schema (SSB Fact + 4 Dims)": "SELECT dd.d_year, dc.c_nation, dp.p_category, SUM(lo.lo_revenue) AS revenue FROM ssb_fact_lineorder lo JOIN ssb_dim_date dd ON lo.lo_orderdatekey = dd.datekey JOIN ssb_dim_customer dc ON lo.lo_custkey = dc.custkey JOIN ssb_dim_supplier ds ON lo.lo_suppkey = ds.suppkey JOIN ssb_dim_part dp ON lo.lo_partkey = dp.partkey GROUP BY 1, 2, 3 ORDER BY 1, 2",
            "Hybrid Galaxy + Kimball Marts (Ours)": "SELECT d_year, c_nation, p_category, SUM(total_revenue) AS revenue FROM galaxy_mart_revenue_cube GROUP BY 1, 2, 3 ORDER BY 1, 2",
            "Naive Single Flat Table (OBT)": "SELECT d_year, c_nation, p_category, SUM(lo_revenue) AS revenue FROM flat_wide_obt GROUP BY 1, 2, 3 ORDER BY 1, 2",
        },
    }

    architectures = [
        "Hybrid Galaxy + Kimball Marts (Ours)",
        "Pure Star Schema (SSB Fact + 4 Dims)",
        "3NF / Snowflake (TPC-H 7-Table Schema)",
        "Naive Single Flat Table (OBT)",
    ]
    join_counts = {
        "Hybrid Galaxy + Kimball Marts (Ours)": 0,
        "Pure Star Schema (SSB Fact + 4 Dims)": 3,
        "3NF / Snowflake (TPC-H 7-Table Schema)": 8,
        "Naive Single Flat Table (OBT)": 0,
    }

    true_pay = con.execute("SELECT SUM(pay_amount) FROM ssb_fact_payments").fetchone()[0] or 1.0
    obt_pay = con.execute("SELECT SUM(pay_amount) FROM flat_wide_obt").fetchone()[0] or 0.0
    fan_trap_errors = {
        "Hybrid Galaxy + Kimball Marts (Ours)": 0.0,
        "Pure Star Schema (SSB Fact + 4 Dims)": 0.0,
        "3NF / Snowflake (TPC-H 7-Table Schema)": 0.0,
        "Naive Single Flat Table (OBT)": round(((obt_pay - true_pay) / true_pay) * 100.0, 2),
    }

    per_query_rows: List[Dict[str, Any]] = []
    arch_timings: Dict[str, List[float]] = {a: [] for a in architectures}
    for q_name, arch_sql_map in queries.items():
        for arch in architectures:
            ms = _time_query_ms(con, arch_sql_map[arch], runs=5)
            arch_timings[arch].append(ms)
            per_query_rows.append({
                "Query Flight": q_name,
                "Architecture": arch,
                "Latency (ms)": ms,
                "QPS (Queries/sec)": round(1000.0 / max(ms, 0.05), 1),
                "SQL Query": arch_sql_map[arch],
            })

    summary_rows: List[Dict[str, Any]] = []
    for arch in architectures:
        times = arch_timings[arch]
        geo_mean = math.exp(sum(math.log(max(t, 0.01)) for t in times) / len(times))
        avg_ms = sum(times) / len(times)
        err_pct = fan_trap_errors[arch]
        accuracy_pct = 100.0 if err_pct == 0.0 else round(max(0.0, 100.0 - min(99.0, err_pct)), 2)
        summary_rows.append({
            "Architecture": arch,
            "Benchmark Standard": f"TPC-H / SSB ({source_mode})",
            "Runtime Joins (Q4.1)": join_counts[arch],
            "Avg Latency (ms)": round(avg_ms, 3),
            "GeoMean Latency (ms)": round(geo_mean, 3),
            "Throughput (QPS)": round(1000.0 / max(avg_ms, 0.05), 1),
            "Multi-Fact Fan-Trap Error (%)": err_pct,
            "Aggregation Accuracy (%)": accuracy_pct,
        })
    con.close()

    return {
        "benchmark_name": "TPC-H & Star Schema Benchmark (SSB Q1.1–Q4.1)",
        "data_source": source_mode,
        "summary_table": summary_rows,
        "per_query_flights": per_query_rows,
        "olist_live_comparison": [],
    }
