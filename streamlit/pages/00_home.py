from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


bootstrap_streamlit(__file__)


st.title("Usecase Demo")
st.caption("Use the sidebar to navigate between pages.")

st.markdown(
	"""
This Streamlit app is organized as a multi-page application.

Page navigation (order/names/visibility) is managed in `streamlit/page_registry.py`.
Pages are implemented as scripts under `streamlit/pages/`.

Included pages:
- Architecture
- Regression Dashboard
- Deep Learning Dashboard
- Chatbot
- MLflow Tracking
	""".strip()
)

st.divider()
st.markdown("**Quick start**")
st.code("streamlit run streamlit/app.py", language="bash")
