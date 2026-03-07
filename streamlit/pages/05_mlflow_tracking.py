from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


bootstrap_streamlit(__file__)

from src.config import Config


st.set_page_config(page_title="MLflow", layout="wide")

st.title("MLflow Tracking")
st.caption("View where MLflow runs are stored and how to launch the MLflow UI.")


MLRUNS_DIR = Path(Config.MLFLOW_TRACKING_URI)

st.markdown("**Tracking directory**")
st.code(str(MLRUNS_DIR), language="text")

if not MLRUNS_DIR.exists():
	st.warning("`artifacts/mlruns/` was not found. Run a training notebook/pipeline to generate MLflow runs.")
	st.stop()

exp_dirs = [p for p in MLRUNS_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
st.markdown(f"Found **{len(exp_dirs)}** experiment folders.")

st.divider()
st.markdown("**Launch MLflow UI**")
st.code(
	f"mlflow ui --backend-store-uri {MLRUNS_DIR.as_posix()} --port 5000",
	language="bash",
)
st.caption("Then open: http://localhost:5000")

