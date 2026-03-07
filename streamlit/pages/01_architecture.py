from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.config import Config
from src.utils import bootstrap_streamlit


BOOT = bootstrap_streamlit(__file__)
REPO_ROOT = BOOT.repo_root


_repo = Path(Config.PROJECT_ROOT)
_vector = Path(Config.VECTOR_DB_PATH)
try:
	_vector_display = _vector.resolve().relative_to(_repo.resolve()).as_posix()
except Exception:
	_vector_display = str(_vector)


st.set_page_config(page_title="Architecture", layout="wide")

st.title("Architecture")
st.markdown(
	"""
This repo is a small, multi-usecase demo workspace. The Streamlit UI lives in `streamlit/` and is split into pages.

Key folders:
- `src/`: Python modules
- `notebooks/`: development notebooks for each use case
- `artifacts/`: saved models, vector stores, MLflow runs
- `data/`: datasets (CSV, text, images)
""".strip()
)

st.divider()
st.markdown("**Where things live**")

st.markdown(
	(
		"""
- Chatbot code: `src/chatbot/`
- Streamlit pages: `streamlit/pages/`
		- Persisted vector stores: `{vector_dir}/`
		""".strip()
	).format(vector_dir=_vector_display)
)

