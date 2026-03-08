from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

from src.classification.model import ClassificationModelConfig, prepare_inference_dataframe
from src.config import Config
from src.functions.model_pipeline import ModelPipeline


@dataclass
class ClassificationPredictionResult:
	"""Prediction outputs for classification inference."""

	predicted_classes: np.ndarray
	probabilities: np.ndarray | None
	positive_probability: np.ndarray | None
	features: pd.DataFrame
	output: pd.DataFrame


def _ensure_legacy_functions_aliases() -> None:
	"""Provide module aliases so joblib can load legacy `functions.*` artifacts."""
	if "functions" in sys.modules:
		return

	try:
		import src.functions as src_functions
		import src.functions.classification_pipeline as classification_pipeline
		import src.functions.feature_eng as feature_eng
		import src.functions.model as model
		import src.functions.model_pipeline as model_pipeline
	except Exception:
		return

	sys.modules.setdefault("functions", src_functions)
	sys.modules.setdefault("functions.classification_pipeline", classification_pipeline)
	sys.modules.setdefault("functions.feature_eng", feature_eng)
	sys.modules.setdefault("functions.model", model)
	sys.modules.setdefault("functions.model_pipeline", model_pipeline)


def _expected_columns_from_model(model: Any) -> list[str] | None:
	cols = getattr(model, "feature_names_in_", None)
	if cols is None:
		return None
	try:
		return [str(c) for c in list(cols)]
	except Exception:
		return None


def _positive_class_index(model: Any, *, positive_label: Any = 1) -> int | None:
	classes = getattr(model, "classes_", None)
	if classes is None:
		return None
	classes_list = list(classes)
	if positive_label in classes_list:
		return classes_list.index(positive_label)
	if len(classes_list) == 2:
		return 1
	return None


def load_classification_pipeline(model_path: str | Path | None = None):
	"""Load saved classification pipeline artifact from disk."""
	_ensure_legacy_functions_aliases()
	default_path = Path(Config.MODEL_PATH) / Config.BEST_CLASSIFICATION_PIPE_FILE
	path = Path(model_path) if model_path is not None else default_path
	if not path.exists():
		raise FileNotFoundError(f"Classification pipeline artifact not found: {path}")
	return ModelPipeline.load_pipeline(str(path))


def predict_classification(
	data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
	*,
	model: Any | None = None,
	model_path: str | Path | None = None,
	config: ClassificationModelConfig | None = None,
	prediction_column: str = "predicted_class",
	probability_column: str = "predicted_failure_probability",
	positive_label: Any = 1,
) -> ClassificationPredictionResult:
	"""Run classification predictions on raw or normalized input payloads."""
	cfg = config or ClassificationModelConfig()
	pipe = model if model is not None else load_classification_pipeline(model_path or cfg.model_path)

	expected_columns = _expected_columns_from_model(pipe)
	features = prepare_inference_dataframe(
		data,
		config=cfg,
		expected_columns=expected_columns,
		strict=False,
	)

	pred_classes = np.asarray(pipe.predict(features))
	pred_proba: np.ndarray | None = None
	positive_probability: np.ndarray | None = None

	if hasattr(pipe, "predict_proba"):
		try:
			pred_proba = np.asarray(pipe.predict_proba(features), dtype=float)
			pos_idx = _positive_class_index(pipe, positive_label=positive_label)
			if pos_idx is not None and pred_proba.ndim == 2 and pred_proba.shape[1] > pos_idx:
				positive_probability = pred_proba[:, pos_idx]
		except Exception:
			pred_proba = None
			positive_probability = None

	out = features.copy()
	out[prediction_column] = pred_classes
	if positive_probability is not None:
		out[probability_column] = positive_probability

	return ClassificationPredictionResult(
		predicted_classes=pred_classes,
		probabilities=pred_proba,
		positive_probability=positive_probability,
		features=features,
		output=out,
	)


def predict_from_csv(
	input_csv_path: str | Path,
	*,
	output_csv_path: str | Path | None = None,
	model_path: str | Path | None = None,
	config: ClassificationModelConfig | None = None,
	prediction_column: str = "predicted_class",
	probability_column: str = "predicted_failure_probability",
	positive_label: Any = 1,
) -> ClassificationPredictionResult:
	"""Run classification predictions from CSV input and optionally save output CSV."""
	input_path = Path(input_csv_path)
	if not input_path.exists():
		raise FileNotFoundError(f"Prediction input CSV not found: {input_path}")

	df = pd.read_csv(input_path)
	result = predict_classification(
		df,
		model_path=model_path,
		config=config,
		prediction_column=prediction_column,
		probability_column=probability_column,
		positive_label=positive_label,
	)

	if output_csv_path is not None:
		output_path = Path(output_csv_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		result.output.to_csv(output_path, index=False)

	return result


if __name__ == "__main__":
	example = pd.DataFrame(
		[
			{
				"Type": "M",
				"Air temperature [K]": 300.0,
				"Process temperature [K]": 310.0,
				"Rotational speed [rpm]": 1500.0,
				"Torque [Nm]": 400.0,
				"Tool wear [min]": 10.0,
			}
		]
	)

	res = predict_classification(example)
	print(res.output)

