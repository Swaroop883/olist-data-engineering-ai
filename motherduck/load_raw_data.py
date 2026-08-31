import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("MOTHERDUCK_TOKEN")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not token:
    raise ValueError("MOTHERDUCK_TOKEN not found")

if not aws_access_key or not aws_secret_key:
    raise ValueError("AWS credentials not found")

os.environ["motherduck_token"] = token

bucket = "olist-data-engineering-142366489643"
region = "eu-north-1"

files = {
    "raw_customers": "olist_customers_dataset.csv",
    "raw_geolocation": "olist_geolocation_dataset.csv",
    "raw_order_items": "olist_order_items_dataset.csv",
    "raw_order_payments": "olist_order_payments_dataset.csv",
    "raw_order_reviews": "olist_order_reviews_dataset.csv",
    "raw_orders": "olist_orders_dataset.csv",
    "raw_products": "olist_products_dataset.csv",
    "raw_sellers": "olist_sellers_dataset.csv",
    "raw_product_category_translation": "product_category_name_translation.csv"
}

con = duckdb.connect()

con.execute("LOAD motherduck")

con.execute("ATTACH 'md:olist_db' AS md")

con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

con.execute(f"""
CREATE OR REPLACE SECRET olist_s3 (
    TYPE S3,
    KEY_ID '{aws_access_key}',
    SECRET '{aws_secret_key}',
    REGION '{region}'
)
""")

con.execute("CREATE SCHEMA IF NOT EXISTS md.raw")

for table_name, file_name in files.items():

    s3_path = f"s3://{bucket}/raw/olist/{file_name}"

    print(f"\nLoading {table_name}...")

    con.execute(f"""
        CREATE OR REPLACE TABLE md.raw.{table_name} AS
        SELECT *
        FROM read_csv_auto('{s3_path}')
    """)

    count = con.execute(
        f"SELECT COUNT(*) FROM md.raw.{table_name}"
    ).fetchone()[0]

    print(f"Loaded {table_name}: {count} rows")

print("\nAll raw tables loaded successfully!")

con.close()