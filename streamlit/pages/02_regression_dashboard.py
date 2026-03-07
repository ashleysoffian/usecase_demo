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
from src.regression.predict import load_regression_pipeline, predict_regression


st.set_page_config(page_title="Regression", layout="wide")

st.title("Used Car Price Prediction")
st.markdown(
	"""
Objective:

- To predict the fair market price of used vehicle based on vehicle attributes, usage history, market demand, and condition signals

Business Use Cases:

- Risk control for under or overpricing
- Trade-in valuation
- Marketplace listing recommendations
- Dealer pricing optimization

ML Framing:

- Type: Supervised Regression
- Target: Price
- Prediction: At listing time
"""
)
st.divider()


MODEL_PATH = Path(Config.MODEL_PATH) / Config.BEST_REG_PIPE_FILE
DATASET_PATH = Path(Config.DATA_PATH) / "merc.csv"


@st.cache_resource
def _load_model(model_path: Path):
	return load_regression_pipeline(model_path)


@st.cache_data(show_spinner=False)
def _load_model_options(dataset_path: Path) -> list[str]:
	if not dataset_path.exists():
		return []

	try:
		df = pd.read_csv(dataset_path, usecols=["model"])
	except Exception:
		return []

	values = df["model"].dropna().astype(str).str.strip()
	options = sorted({v for v in values if v})
	return options


def _get_expected_columns(model) -> Optional[list[str]]:
	cols = getattr(model, "feature_names_in_", None)
	if cols is None:
		return None
	try:
		return list(cols)
	except Exception:
		return None


def _is_numeric_feature(name: str) -> bool:
	low = name.lower()
	numeric_keys = (
		"mileage",
		"tax",
		"mpg",
		"engine_size",
		"enginesize",
		"car_age",
		"year",
	)
	return any(key in low for key in numeric_keys)


def _default_value(name: str):
	defaults: dict[str, object] = {
		"mileage": 45000.0,
		"tax": 150.0,
		"mpg": 45.0,
		"engine_size": 1.8,
		"enginesize": 1.8,
		"car_age": 4.0,
		"year": 2022.0,
		"model": "A Class",
		"transmission": "Automatic",
		"fuel_type": "Petrol",
	}
	return defaults.get(name.lower(), 0.0 if _is_numeric_feature(name) else "")


def _render_input_fields(expected_cols: list[str], *, model_options: list[str]) -> dict[str, object]:
	inputs: dict[str, object] = {}
	col_left, col_right = st.columns(2)

	for index, col_name in enumerate(expected_cols):
		target_col = col_left if index % 2 == 0 else col_right
		with target_col:
			label = col_name.replace("_", " ").title()
			low = col_name.lower()

			if low in {"transmission"}:
				inputs[col_name] = st.selectbox(
					label,
					options=["Automatic", "Manual", "Semi-Auto"],
					index=0,
					key=f"input_{col_name}",
				)
			elif low in {"fuel_type", "fueltype"}:
				inputs[col_name] = st.selectbox(
					label,
					options=["Petrol", "Diesel", "Hybrid"],
					index=0,
					key=f"input_{col_name}",
				)
			elif low == "model" and model_options:
				default_model = str(_default_value(col_name))
				default_idx = model_options.index(default_model) if default_model in model_options else 0
				inputs[col_name] = st.selectbox(
					label,
					options=model_options,
					index=default_idx,
					key=f"input_{col_name}",
				)
			elif _is_numeric_feature(col_name):
				inputs[col_name] = st.number_input(
					label,
					value=float(_default_value(col_name)),
					step=1.0,
					key=f"input_{col_name}",
				)
			else:
				inputs[col_name] = st.text_input(
					label,
					value=str(_default_value(col_name)),
					key=f"input_{col_name}",
				)

	return inputs


if not MODEL_PATH.exists():
	st.error(f"Missing model artifact: {MODEL_PATH}")
	st.stop()

model = _load_model(MODEL_PATH)
expected_cols = _get_expected_columns(model)
model_options = _load_model_options(DATASET_PATH)

with st.expander("Model details", expanded=False):
	st.write(f"Model artifact: {MODEL_PATH}")
	if expected_cols:
		st.write("Expected feature columns:")
		st.code(", ".join(expected_cols), language="text")
	else:
		st.write("Expected feature columns are not available in this model metadata.")

if not expected_cols:
	st.error("Unable to render input fields because model feature metadata is unavailable.")
	st.stop()

st.subheader("Input Features")
input_data = _render_input_fields(expected_cols, model_options=model_options)

predict_clicked = st.button("Predict", type="primary")
if predict_clicked:
	try:
		result = predict_regression(input_data, model=model)
	except Exception as e:
		st.error(
			"Prediction failed. Ensure input values are valid for the trained schema.\n\n"
			f"Error: {e}"
		)
		st.stop()

	out = pd.DataFrame([input_data])
	out["prediction"] = result.predictions

	pred_value = float(result.predictions[0])
	st.metric("Predicted price", f"{pred_value:,.2f}")

	st.subheader("Prediction Result")
	st.dataframe(out, use_container_width=True)

