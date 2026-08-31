import duckdb
import os
from dotenv import load_dotenv

load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = "eu-north-1"

con = duckdb.connect()

con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

con.execute(f"""
CREATE SECRET (
    TYPE S3,
    KEY_ID '{aws_access_key}',
    SECRET '{aws_secret_key}',
    REGION '{aws_region}'
)
""")

result = con.execute("""
SELECT COUNT(*)
FROM read_csv_auto(
    's3://olist-data-engineering-142366489643/raw/olist/olist_customers_dataset.csv'
)
""").fetchone()

print("Customers rows:", result[0])

con.close()