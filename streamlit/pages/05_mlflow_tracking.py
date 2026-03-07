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

st.set_page_config(page_title="MLflow", layout="wide")

st.title("MLflow Tracking")

st.divider()
mlflow_ui_url = st.text_input(
	"MLflow URL",
	value="http://localhost:5001",
	help="URL that your browser can access for MLflow UI.",
)
st.markdown(f"[Open MLflow UI in a new tab]({mlflow_ui_url})")


