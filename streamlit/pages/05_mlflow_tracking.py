from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


bootstrap_streamlit(__file__)

from src.config import Config
from src.mlflow_utils import normalize_mlflow_tracking_uri


st.set_page_config(page_title="MLflow", layout="wide")

st.title("MLflow Tracking")
st.caption("View the tracking backend used by the chatbot and how to launch MLflow UI.")


def _sqlite_db_path(uri: str) -> Path | None:
	prefix = "sqlite:///"
	if not uri.startswith(prefix):
		return None
	return Path(uri[len(prefix) :])


RAW_TRACKING_URI = str(Config.MLFLOW_TRACKING_URI)
CHATBOT_TRACKING_URI = normalize_mlflow_tracking_uri(RAW_TRACKING_URI)
db_path = _sqlite_db_path(CHATBOT_TRACKING_URI)

st.markdown("**Configured tracking location**")
st.code(RAW_TRACKING_URI, language="text")

st.markdown("**Effective chatbot tracking URI**")
st.code(CHATBOT_TRACKING_URI, language="text")

if db_path is not None:
	st.markdown("**SQLite database**")
	st.code(str(db_path), language="text")

if db_path is not None and not db_path.exists():
	st.info("No MLflow SQLite database found yet. Runs will create it automatically.")

st.divider()
st.markdown("**Launch MLflow UI**")
st.code(
	f'mlflow ui --backend-store-uri "{CHATBOT_TRACKING_URI}" --port 5001',
	language="bash",
)
st.caption("Then open: http://localhost:5001")

st.divider()
st.markdown("**Embedded MLflow UI**")
mlflow_ui_url = st.text_input(
	"MLflow UI URL",
	value="http://localhost:5001",
	help="URL that your browser can access for MLflow UI.",
)

if mlflow_ui_url:
	st.markdown(f"[Open MLflow UI in a new tab]({mlflow_ui_url})")
	components.iframe(src=mlflow_ui_url, height=900, scrolling=True)
else:
	st.warning("Please provide a valid MLflow UI URL to embed.")

