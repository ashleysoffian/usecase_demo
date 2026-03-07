from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


bootstrap_streamlit(__file__)

from src.config import Config


st.set_page_config(page_title="Regression", layout="wide")

st.title("Regression Dashboard")
st.caption("Upload a CSV and run predictions using the saved regression pipeline.")


MODEL_PATH = Path(Config.MODEL_PATH) / Config.BEST_REG_PIPE_FILE


@st.cache_resource
def _load_model(model_path: Path):
	import joblib

	return joblib.load(model_path)


def _get_expected_columns(model) -> Optional[list[str]]:
	cols = getattr(model, "feature_names_in_", None)
	if cols is None:
		return None
	try:
		return list(cols)
	except Exception:
		return None


if not MODEL_PATH.exists():
	st.error(f"Missing model artifact: {MODEL_PATH}")
	st.stop()

model = _load_model(MODEL_PATH)
expected_cols = _get_expected_columns(model)
if expected_cols:
	st.caption(f"Expected columns: {', '.join(expected_cols)}")

uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
if uploaded is None:
	st.info("Upload a CSV file to begin.")
	st.stop()

try:
	df = pd.read_csv(uploaded)
except Exception as e:
	st.error(f"Failed to read CSV: {e}")
	st.stop()

st.subheader("Input preview")
st.dataframe(df.head(50), use_container_width=True)

predict_clicked = st.button("Predict", type="primary")
if not predict_clicked:
	st.stop()

try:
	preds = model.predict(df)
except Exception as e:
	st.error(
		"Prediction failed. Ensure your CSV columns match the training schema.\n\n"
		f"Error: {e}"
	)
	st.stop()

out = df.copy()
out["prediction"] = preds

st.subheader("Predictions")
st.dataframe(out.head(50), use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
	"Download predictions CSV",
	data=csv_bytes,
	file_name="predictions.csv",
	mime="text/csv",
)

