from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


bootstrap_streamlit(__file__)

from src.classification.predict import load_classification_pipeline, predict_classification
from src.config import Config


st.set_page_config(page_title="Classification", layout="wide")

st.title("Machine Failure Prediction")
st.markdown(
	"""
Objective:

- To predict whether a machine is likely to fail based on operational conditions, usage patterns, and wear indicators.

Business Use Cases:

- Predictive maintenance planning
- Operational risk control
- Maintenance resource optimization

ML Framing:

- Type: Binary Supervised Classification
- Target: Machine Failure
"""
)
st.divider()

st.markdown(
	"""
	<style>
	div[data-testid="stMetricValue"] > div {
		font-size: 1.5rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


MODEL_PATH = Path(Config.MODEL_PATH) / Config.BEST_CLASSIFICATION_PIPE_FILE
TYPE_OPTIONS = ["L", "M", "H"]

FALLBACK_INPUT_COLUMNS = [
	"Type",
	"Air temperature [K]",
	"Process temperature [K]",
	"Rotational speed [rpm]",
	"Torque [Nm]",
	"Tool wear [min]",
]


@st.cache_resource
def _load_model(model_path: Path):
	return load_classification_pipeline(model_path)


def _get_expected_columns(model) -> list[str]:
	cols = getattr(model, "feature_names_in_", None)
	if cols is not None:
		try:
			resolved = [str(c) for c in list(cols)]
			if resolved:
				return resolved
		except Exception:
			pass

	# Direct fallback that always works with notebook-style input.
	return FALLBACK_INPUT_COLUMNS.copy()


def _default_value(name: str):
	defaults: dict[str, object] = {
		"type": "M",
		"air temperature [k]": 300.0,
		"process temperature [k]": 310.0,
		"rotational speed [rpm]": 1500.0,
		"torque [nm]": 40.0,
		"tool wear [min]": 10.0,
		"air_temp": 300.0,
		"process_temp": 310.0,
		"rotational_speed": 1500.0,
		"torque": 40.0,
		"tool_wear": 10.0,
	}
	if name.strip().lower() == "type":
		return str(defaults.get("type", "M"))
	return float(defaults.get(name.lower(), 0.0))


def _render_input_fields(expected_cols: list[str], *, type_options: list[str]) -> dict[str, object]:
	inputs: dict[str, object] = {}
	col_left, col_right = st.columns(2)

	for index, col_name in enumerate(expected_cols):
		target_col = col_left if index % 2 == 0 else col_right
		with target_col:
			label = col_name.replace("_", " ").title()
			low = col_name.strip().lower()

			if low == "type":
				default_value = str(_default_value(col_name))
				default_idx = type_options.index(default_value) if default_value in type_options else 0
				inputs[col_name] = st.selectbox(
					label,
					options=type_options,
					index=default_idx,
					key=f"input_{col_name}",
				)
			else:
				inputs[col_name] = st.number_input(
					label,
					value=float(_default_value(col_name)),
					step=1.0,
					key=f"input_{col_name}",
				)

	return inputs


def _format_predicted_class(value: object) -> str:
	"""Return user-friendly class text for binary machine failure output."""
	try:
		cls = int(value)
	except (TypeError, ValueError):
		return str(value)

	if cls == 0:
		return "0: not failure"
	if cls == 1:
		return "1: failure"
	return str(value)


if not MODEL_PATH.exists():
	st.error(f"Missing model artifact: {MODEL_PATH}")
	st.stop()

model = _load_model(MODEL_PATH)
expected_cols = _get_expected_columns(model)
type_options = TYPE_OPTIONS

with st.expander("Model details", expanded=False):
	st.write(f"Model artifact: {MODEL_PATH}")
	st.write("Expected feature columns:")
	st.code(", ".join(expected_cols), language="text")

st.subheader("Input Features")
input_data = _render_input_fields(expected_cols, type_options=type_options)

predict_clicked = st.button("Predict", type="primary")
if predict_clicked:
	try:
		result = predict_classification(input_data, model=model)
	except Exception as e:
		st.error(
			"Prediction failed. Ensure input values are valid for the trained schema.\n\n"
			f"Error: {e}"
		)
		st.stop()

	out = pd.DataFrame([input_data])
	out["predicted_class"] = [_format_predicted_class(v) for v in result.predicted_classes]

	predicted_class = result.predicted_classes[0]
	st.metric("Predicted class", _format_predicted_class(predicted_class))

	if result.positive_probability is not None:
		failure_prob = float(result.positive_probability[0])
		st.metric("Predicted failure probability", f"{failure_prob:.2%}")
		out["predicted_failure_probability"] = result.positive_probability

	st.subheader("Prediction Result")
	st.dataframe(out, use_container_width=True)

