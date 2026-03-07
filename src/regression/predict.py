from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.functions.model_pipeline import ModelPipeline
from src.regression.model import RegressionPipelineConfig, prepare_inference_dataframe


@dataclass
class RegressionPredictionResult:
	"""Prediction outputs for regression inference."""

	predictions: np.ndarray
	features: pd.DataFrame
	output: pd.DataFrame


def _expected_columns_from_model(model: Any) -> list[str] | None:
	cols = getattr(model, "feature_names_in_", None)
	if cols is None:
		return None
	try:
		return [str(c) for c in list(cols)]
	except Exception:
		return None


def load_regression_pipeline(model_path: str | Path | None = None):
	"""Load saved regression pipeline artifact from disk."""
	default_path = Path(Config.MODEL_PATH) / Config.BEST_REG_PIPE_FILE
	path = Path(model_path) if model_path is not None else default_path
	if not path.exists():
		raise FileNotFoundError(f"Regression pipeline artifact not found: {path}")
	return ModelPipeline.load_pipeline(str(path))


def predict_regression(
	data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
	*,
	model: Any | None = None,
	model_path: str | Path | None = None,
	config: RegressionPipelineConfig | None = None,
	prediction_column: str = "prediction",
) -> RegressionPredictionResult:
	"""Run regression predictions on raw input data.

	Input data can be raw notebook-style schema (`fuelType`, `engineSize`, `year`) or
	already normalized schema (`fuel_type`, `engine_size`, `car_age`).
	"""
	cfg = config or RegressionPipelineConfig()
	pipe = model if model is not None else load_regression_pipeline(model_path or cfg.model_path)

	expected_columns = _expected_columns_from_model(pipe)
	features = prepare_inference_dataframe(data, config=cfg, expected_columns=expected_columns)
	preds = np.asarray(pipe.predict(features), dtype=float)

	out = features.copy()
	out[prediction_column] = preds

	return RegressionPredictionResult(predictions=preds, features=features, output=out)


def predict_from_csv(
	input_csv_path: str | Path,
	*,
	output_csv_path: str | Path | None = None,
	model_path: str | Path | None = None,
	config: RegressionPipelineConfig | None = None,
	prediction_column: str = "prediction",
) -> RegressionPredictionResult:
	"""Run regression predictions from CSV input and optionally save output CSV."""
	input_path = Path(input_csv_path)
	if not input_path.exists():
		raise FileNotFoundError(f"Prediction input CSV not found: {input_path}")

	df = pd.read_csv(input_path)
	result = predict_regression(
		df,
		model_path=model_path,
		config=config,
		prediction_column=prediction_column,
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
				"mileage": 45000,
				"car_age": 4,
				"engine_size": 1.8,
				"tax": 150,
				"mpg": 45,
				"fuel_type": "Petrol",
				"transmission": "Automatic",
				"model": "A Class",
			}
		]
	)
	res = predict_regression(example)
	print(res.output)

