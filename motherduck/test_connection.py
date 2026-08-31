import duckdb
import os
from dotenv import load_dotenv
import traceback

print("DuckDB version:", duckdb.__version__)

load_dotenv()
token = os.getenv("MOTHERDUCK_TOKEN", "").strip()

if not token:
    print("TOKEN NOT FOUND — check your .env file has MOTHERDUCK_TOKEN=... with no quotes")
    raise SystemExit(1)

print("Token found, length:", len(token))

os.environ["motherduck_token"] = token

try:
    con = duckdb.connect(config={"allow_unsigned_extensions": "false"})
    con.execute("LOAD motherduck")
    con.execute("ATTACH 'md:olist_db' AS md")
    result = con.execute("SELECT current_database()").fetchall()
    print("SUCCESS:", result)
    con.close()
except Exception:
    print("FULL ERROR:")
    traceback.print_exc()