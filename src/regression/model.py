from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import Config
from src.functions.model import ModelTrainer
from src.functions.model_pipeline import ModelPipeline, PipelineTrainResult


DEFAULT_NUMERIC_FEATURES: tuple[str, ...] = ("mileage", "tax", "mpg", "engine_size", "car_age")
DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = ("model", "transmission", "fuel_type")
DEFAULT_LOG_FEATURES: tuple[str, ...] = ("mileage", "mpg", "engine_size", "car_age")
DEFAULT_MODEL_TYPES: tuple[str, ...] = ("linear", "random_forest", "xgboost")


@dataclass(frozen=True)
class RegressionFeatureSchema:
	"""Resolved training schema for a specific dataset."""

	target_column: str
	feature_columns: tuple[str, ...]
	numeric_features: tuple[str, ...]
	categorical_features: tuple[str, ...]
	log_columns: tuple[str, ...]


@dataclass(frozen=True)
class RegressionPipelineConfig:
	"""Configuration for regression pipeline training and inference."""

	dataset_path: Path = field(default_factory=lambda: Path(Config.DATA_PATH) / "merc.csv")
	model_path: Path = field(default_factory=lambda: Path(Config.MODEL_PATH) / Config.BEST_REG_PIPE_FILE)
	target_column: str = "price"
	current_year: int = 2026

	rename_columns: Mapping[str, str] = field(
		default_factory=lambda: {
			"fuelType": "fuel_type",
			"engineSize": "engine_size",
		}
	)
	year_column: str = "year"
	car_age_column: str = "car_age"
	derive_car_age: bool = True
	drop_year_column: bool = True

	numeric_features: tuple[str, ...] | None = DEFAULT_NUMERIC_FEATURES
	categorical_features: tuple[str, ...] | None = DEFAULT_CATEGORICAL_FEATURES
	log_columns: tuple[str, ...] | None = DEFAULT_LOG_FEATURES
	auto_infer_features: bool = True
	auto_infer_log_columns: bool = False
	model_types: tuple[str, ...] = DEFAULT_MODEL_TYPES

	test_size: float = 0.2
	random_state: int = 42
	cv: int = 2
	scoring: str | None = "neg_root_mean_squared_error"
	target_transform: str = "log1p"
	encoding: str = "onehot"
	scaler: str = "standard"
	skew_threshold: float = 1.5
	n_jobs: int = -1
	verbose: int = 0
	log_to_mlflow: bool = True
	run_name: str | None = None


def feature_columns(config: RegressionPipelineConfig | None = None) -> list[str]:
	"""Return configured feature columns (if explicitly provided)."""
	cfg = config or RegressionPipelineConfig()
	return [*(cfg.numeric_features or ()), *(cfg.categorical_features or ())]


def load_regression_dataframe(csv_path: str | Path | None = None) -> pd.DataFrame:
	"""Load the regression dataset from CSV."""
	path = Path(csv_path) if csv_path is not None else Path(Config.DATA_PATH) / "merc.csv"
	if not path.exists():
		raise FileNotFoundError(f"Regression dataset not found: {path}")
	return pd.read_csv(path)


def prepare_regression_dataframe(
	df: pd.DataFrame,
	*,
	config: RegressionPipelineConfig | None = None,
	current_year: int | None = None,
) -> pd.DataFrame:
	"""Apply dataset normalization before training/prediction.

	Default behavior (for `merc.csv`) also supports:
	- `fuelType` -> `fuel_type`
	- `engineSize` -> `engine_size`
	- derive `car_age` from `year`
	"""
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame")

	cfg = config or RegressionPipelineConfig()
	year_ref = int(current_year if current_year is not None else cfg.current_year)

	out = df.copy()
	if cfg.rename_columns:
		out = out.rename(columns=dict(cfg.rename_columns))

	if cfg.derive_car_age and cfg.year_column in out.columns and cfg.car_age_column not in out.columns:
		years = pd.to_numeric(out[cfg.year_column], errors="coerce")
		out[cfg.car_age_column] = year_ref - years

	if cfg.drop_year_column and cfg.year_column in out.columns:
		out = out.drop(columns=[cfg.year_column])

	return out


def _require_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns for {context}: {missing}")


def _infer_feature_lists(df: pd.DataFrame, *, target_column: str) -> tuple[list[str], list[str]]:
	candidates = [c for c in df.columns if c != target_column]
	numeric = [c for c in candidates if is_numeric_dtype(df[c])]
	categorical = [c for c in candidates if c not in numeric]
	return numeric, categorical


def resolve_feature_schema(
	df: pd.DataFrame,
	*,
	config: RegressionPipelineConfig | None = None,
) -> RegressionFeatureSchema:
	"""Resolve train-time feature schema from config + available columns.

	Rules:
	- Use configured numeric/categorical columns when present.
	- If columns are missing (or none configured), infer from dataset dtypes.
	- Optional auto-append of inferred columns via `auto_infer_features`.
	"""
	cfg = config or RegressionPipelineConfig()
	_require_columns(df, [cfg.target_column], context="schema resolution")

	inferred_num, inferred_cat = _infer_feature_lists(df, target_column=cfg.target_column)

	configured_num = [c for c in (cfg.numeric_features or ()) if c in df.columns and c != cfg.target_column]
	configured_cat = [
		c
		for c in (cfg.categorical_features or ())
		if c in df.columns and c != cfg.target_column and c not in configured_num
	]

	if configured_num or configured_cat:
		numeric = list(configured_num)
		categorical = list(configured_cat)
		if cfg.auto_infer_features:
			used = set(numeric) | set(categorical)
			for col in inferred_num:
				if col not in used:
					numeric.append(col)
					used.add(col)
			for col in inferred_cat:
				if col not in used:
					categorical.append(col)
					used.add(col)
	else:
		numeric = list(inferred_num)
		categorical = list(inferred_cat)

	if not numeric and not categorical:
		raise ValueError("No usable feature columns were found for training.")

	if cfg.log_columns:
		log_columns = [c for c in cfg.log_columns if c in numeric]
	elif cfg.auto_infer_log_columns and numeric:
		skewness = df[numeric].skew(numeric_only=True)
		log_columns = [
			c
			for c in numeric
			if pd.notna(skewness.get(c)) and abs(float(skewness[c])) >= float(cfg.skew_threshold)
		]
	else:
		log_columns = []

	ordered_features = [*numeric, *categorical]

	return RegressionFeatureSchema(
		target_column=cfg.target_column,
		feature_columns=tuple(ordered_features),
		numeric_features=tuple(numeric),
		categorical_features=tuple(categorical),
		log_columns=tuple(log_columns),
	)


def build_training_matrices(
	df: pd.DataFrame,
	*,
	config: RegressionPipelineConfig | None = None,
	return_schema: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, RegressionFeatureSchema]:
	"""Prepare X/y matrices from raw dataset for pipeline training."""
	cfg = config or RegressionPipelineConfig()
	prepared = prepare_regression_dataframe(df, config=cfg)

	schema = resolve_feature_schema(prepared, config=cfg)
	_require_columns(prepared, [*schema.feature_columns, schema.target_column], context="training")

	X = prepared[list(schema.feature_columns)].copy()
	y = pd.to_numeric(prepared[schema.target_column], errors="coerce")
	if y.isna().any():
		n_bad = int(y.isna().sum())
		raise ValueError(
			f"Target column '{schema.target_column}' contains {n_bad} non-numeric/NaN value(s)."
		)

	if return_schema:
		return X, y, schema
	return X, y


def split_train_test(
	X: pd.DataFrame,
	y: pd.Series,
	*,
	config: RegressionPipelineConfig | None = None,
):
	"""Split train/test exactly as notebook workflow (non-stratified)."""
	cfg = config or RegressionPipelineConfig()
	return ModelTrainer.split_train_test(
		X,
		y,
		test_size=cfg.test_size,
		random_state=cfg.random_state,
		stratify=False,
	)


def train_regression_candidates(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	*,
	schema: RegressionFeatureSchema | None = None,
	config: RegressionPipelineConfig | None = None,
) -> dict[str, PipelineTrainResult]:
	"""Train all configured regression models using preprocessing pipeline."""
	cfg = config or RegressionPipelineConfig()

	if schema is None:
		inferred_num = [c for c in X_train.columns if is_numeric_dtype(X_train[c])]
		inferred_cat = [c for c in X_train.columns if c not in inferred_num]
		log_columns = [c for c in (cfg.log_columns or ()) if c in inferred_num]
		schema = RegressionFeatureSchema(
			target_column=cfg.target_column,
			feature_columns=tuple(X_train.columns.tolist()),
			numeric_features=tuple(inferred_num),
			categorical_features=tuple(inferred_cat),
			log_columns=tuple(log_columns),
		)

	results = ModelPipeline.train_regression_all(
		X_train=X_train,
		y_train=y_train,
		model_types=cfg.model_types,
		numeric_features=list(schema.numeric_features),
		categorical_features=list(schema.categorical_features),
		log_columns=list(schema.log_columns),
		skew_threshold=cfg.skew_threshold,
		target_transform=cfg.target_transform,
		encoding=cfg.encoding,
		scaler=cfg.scaler,
		cv=cfg.cv,
		scoring=cfg.scoring,
		random_state=cfg.random_state,
		n_jobs=cfg.n_jobs,
		verbose=cfg.verbose,
	)
	return results


def select_best_candidate(results: dict[str, PipelineTrainResult]) -> PipelineTrainResult:
	"""Return best candidate by CV score."""
	return ModelPipeline.select_best(results)


def evaluate_regression_candidates(
	results: dict[str, PipelineTrainResult],
	*,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	decimals: int = 2,
	sort_by: str | None = "test_rmse",
) -> pd.DataFrame:
	"""Build a combined evaluation table for all trained regression pipelines."""
	if not results:
		raise ValueError("results is empty")

	frames: list[pd.DataFrame] = []
	for model_name, result in results.items():
		metrics = ModelPipeline.evaluate_regression_pipe(
			result,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			decimals=decimals,
		)
		metrics = metrics.copy()
		metrics.insert(0, "model", model_name)
		metrics.insert(1, "split", metrics.index)
		frames.append(metrics.set_index(["model", "split"]))

	out = pd.concat(frames).sort_index()

	if sort_by is not None:
		test_rows = out.xs("test", level="split")
		if sort_by == "test_rmse":
			order = test_rows["rmse"].sort_values(ascending=True).index
		elif sort_by == "test_mae":
			order = test_rows["mae"].sort_values(ascending=True).index
		elif sort_by == "test_mape":
			order = test_rows["mape"].sort_values(ascending=True).index
		elif sort_by == "test_r2":
			order = test_rows["r2"].sort_values(ascending=False).index
		else:
			raise ValueError("Invalid sort_by value")
		out = out.loc[pd.IndexSlice[order, :], :]

	return out


def prepare_inference_dataframe(
	data: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]],
	*,
	config: RegressionPipelineConfig | None = None,
	expected_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
	"""Prepare raw inference payload into model-ready feature frame."""
	cfg = config or RegressionPipelineConfig()

	if isinstance(data, pd.DataFrame):
		df = data.copy()
	elif isinstance(data, Mapping):
		df = pd.DataFrame([dict(data)])
	else:
		df = pd.DataFrame(data)

	prepared = prepare_regression_dataframe(df, config=cfg)
	if cfg.target_column in prepared.columns:
		prepared = prepared.drop(columns=[cfg.target_column])

	if expected_columns is not None:
		cols = list(expected_columns)
	else:
		configured = [c for c in feature_columns(cfg) if c in prepared.columns]
		if configured:
			cols = configured
		else:
			cols = [c for c in prepared.columns if c != cfg.target_column]

	if not cols:
		raise ValueError("No feature columns available for prediction.")

	_require_columns(prepared, cols, context="prediction")
	return prepared[cols].copy()


__all__ = [
	"RegressionFeatureSchema",
	"RegressionPipelineConfig",
	"build_training_matrices",
	"evaluate_regression_candidates",
	"feature_columns",
	"load_regression_dataframe",
	"prepare_inference_dataframe",
	"prepare_regression_dataframe",
	"resolve_feature_schema",
	"select_best_candidate",
	"split_train_test",
	"train_regression_candidates",
]

