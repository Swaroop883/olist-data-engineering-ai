"""
Entry-point alias in `ai/unified_app.py` that delegates to the root `app.py`
so users can run either `streamlit run app.py` or `streamlit run ai/unified_app.py`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import main

if __name__ == "__main__":
    main()
