from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config
from src.functions.classification_pipeline import (
	ClassificationPipelineConfig as NotebookClassificationPipelineConfig,
	MachineFailureFeatureEngineer,
)


DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = ("type",)
DEFAULT_PIPELINE_MODEL_TYPES: tuple[str, ...] = ("logistic", "random_forest", "xgboost")
DEFAULT_METRICS: tuple[str, ...] = ("recall", "f1", "roc_auc")


@dataclass(frozen=True)
class ClassificationModelConfig:
	"""Configuration for notebook-aligned classification training + inference."""

	dataset_path: Path = field(default_factory=lambda: Path(Config.DATA_PATH) / "machine_failure.csv")
	model_path: Path = field(
		default_factory=lambda: Path(Config.MODEL_PATH) / Config.BEST_CLASSIFICATION_PIPE_FILE
	)
	target_column: str = "machine_failure"

	# Notebook-equivalent preprocessing/training settings
	auto_rename_columns: bool = True
	add_engineered_features: bool = True
	log_rotational_speed: bool = True
	categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES
	encoding: str = "onehot"
	pipeline_encoding: str = "onehot"
	scaler: str = "standard"
	test_size: float = 0.2
	stratify: bool = True
	random_state: int = 42
	cv: int = 2
	scoring: str = "recall"
	stratified_cv: bool = True
	refit: bool = True
	metric_set: str = "imbalanced"
	metrics: tuple[str, ...] = DEFAULT_METRICS
	pipeline_model_types: tuple[str, ...] = DEFAULT_PIPELINE_MODEL_TYPES
	return_best_pipeline: bool = True

	# Evaluation labels
	pos_label: int = 1
	fit_metric: str = "recall"
	overfit_gap: float = 0.10
	overfit_train_min: float = 0.85
	underfit_max: float = 0.60

	# Optional plotting
	make_plots: bool = False
	plot_select_by: str = "roc_auc"
	confusion_normalize: str | None = None

	# Tracking
	log_to_mlflow: bool = True
	mlflow_experiment: str = "classification_experiment"
	run_name: str | None = None
	show_checkpoints: bool = True


def load_classification_dataframe(csv_path: str | Path | None = None) -> pd.DataFrame:
	"""Load machine-failure classification dataset from CSV."""
	path = Path(csv_path) if csv_path is not None else Path(Config.DATA_PATH) / "machine_failure.csv"
	if not path.exists():
		raise FileNotFoundError(f"Classification dataset not found: {path}")
	return pd.read_csv(path)


def prepare_classification_dataframe(
	df: pd.DataFrame,
	*,
	config: ClassificationModelConfig | None = None,
) -> pd.DataFrame:
	"""Normalize training dataframe to notebook-compatible schema."""
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame")

	cfg = config or ClassificationModelConfig()
	out = df.copy()
	out.columns = out.columns.astype(str).str.lower()

	if cfg.auto_rename_columns:
		out = out.rename(columns=MachineFailureFeatureEngineer.DEFAULT_RENAME)

	if cfg.target_column not in out.columns:
		raise ValueError(
			f"Target column '{cfg.target_column}' not found after normalization. "
			"Check source CSV columns or set auto_rename_columns accordingly."
		)

	return out


def build_notebook_classification_config(
	config: ClassificationModelConfig | None = None,
) -> NotebookClassificationPipelineConfig:
	"""Map local config to `ClassificationPipelineConfig` used in notebook workflow."""
	cfg = config or ClassificationModelConfig()
	return NotebookClassificationPipelineConfig(
		target=cfg.target_column,
		auto_rename_columns=cfg.auto_rename_columns,
		add_engineered_features=cfg.add_engineered_features,
		log_rotational_speed=cfg.log_rotational_speed,
		categorical_features=list(cfg.categorical_features),
		encoding=cfg.encoding,
		scaler=cfg.scaler,
		test_size=cfg.test_size,
		stratify=cfg.stratify,
		cv=cfg.cv,
		scoring=cfg.scoring,
		stratified_cv=cfg.stratified_cv,
		refit=cfg.refit,
		metric_set=cfg.metric_set,
		metrics=list(cfg.metrics),
		pipeline_model_types=tuple(cfg.pipeline_model_types),
		random_state=cfg.random_state,
		return_best_pipeline=cfg.return_best_pipeline,
		pos_label=cfg.pos_label,
		fit_metric=cfg.fit_metric,
		overfit_gap=cfg.overfit_gap,
		overfit_train_min=cfg.overfit_train_min,
		underfit_max=cfg.underfit_max,
		make_plots=cfg.make_plots,
		plot_select_by=cfg.plot_select_by,
		confusion_normalize=cfg.confusion_normalize,
	)


def prepare_inference_dataframe(
	data: pd.DataFrame | Mapping[str, Any] | list[Mapping[str, Any]],
	*,
	config: ClassificationModelConfig | None = None,
	expected_columns: list[str] | tuple[str, ...] | None = None,
	strict: bool = False,
) -> pd.DataFrame:
	"""Prepare inference payload for classification pipeline prediction.

	Supports either notebook-style raw keys (e.g. `Type`, `Air temperature [K]`) or
	already normalized keys (`type`, `air_temp`, ...).
	"""
	cfg = config or ClassificationModelConfig()

	if isinstance(data, pd.DataFrame):
		df = data.copy()
	elif isinstance(data, Mapping):
		df = pd.DataFrame([dict(data)])
	else:
		df = pd.DataFrame(data)

	# Drop target column if it is accidentally included.
	target_aliases = {cfg.target_column.lower()}
	if cfg.auto_rename_columns:
		for raw_name, normalized_name in MachineFailureFeatureEngineer.DEFAULT_RENAME.items():
			if normalized_name == cfg.target_column:
				target_aliases.add(str(raw_name).lower())

	drop_cols = [c for c in df.columns if str(c).lower() in target_aliases]
	if drop_cols:
		df = df.drop(columns=drop_cols)

	if expected_columns:
		expected = list(expected_columns)
		missing = [c for c in expected if c not in df.columns]
		if not missing:
			return df[expected].copy()
		if strict:
			raise ValueError(f"Missing expected columns for prediction: {missing}")

	if df.empty and len(df.columns) == 0:
		raise ValueError("No usable feature columns found for prediction input.")

	return df.copy()


__all__ = [
	"ClassificationModelConfig",
	"build_notebook_classification_config",
	"load_classification_dataframe",
	"prepare_classification_dataframe",
	"prepare_inference_dataframe",
]

