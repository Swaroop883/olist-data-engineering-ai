import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    raise ValueError("MOTHERDUCK_TOKEN not found")

os.environ["motherduck_token"] = token

con = duckdb.connect()

con.execute("""
SET custom_extension_repository = 'https://extensions.duckdb.org'
""")

con.execute("LOAD motherduck")

con.execute("ATTACH 'md:olist_db' AS md")

tables = [
    "raw_customers",
    "raw_geolocation",
    "raw_order_items",
    "raw_order_payments",
    "raw_order_reviews",
    "raw_orders",
    "raw_products",
    "raw_sellers",
    "raw_product_category_translation"
]

print("\nMOTHERDUCK RAW TABLE VERIFICATION")
print("-" * 45)

for table in tables:

    count = con.execute(
        f"SELECT COUNT(*) FROM md.raw.{table}"
    ).fetchone()[0]

    print(f"{table}: {count} rows")

print("-" * 45)
print("All raw tables verified successfully!")

con.close()