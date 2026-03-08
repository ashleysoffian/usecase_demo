from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import mlflow
import numpy as np
import pandas as pd

from src.functions.model_pipeline import ModelPipeline, PipelineTrainResult
from src.mlflow_utils import configure_regression_mlflow
from src.regression.model import (
	RegressionFeatureSchema,
	RegressionPipelineConfig,
	build_training_matrices,
	evaluate_regression_candidates,
	load_regression_dataframe,
	select_best_candidate,
	split_train_test,
	train_regression_candidates,
)


@dataclass
class RegressionTrainingResult:
	"""Outputs from a full regression training run."""

	results: dict[str, PipelineTrainResult]
	schema: RegressionFeatureSchema
	evaluation_table: pd.DataFrame
	best_model_name: str
	best_params: dict[str, Any]
	best_score: float
	model_path: Path
	run_id: str | None = None


class RegressionTrainingPipeline:
	"""Notebook-equivalent regression pipeline with training, evaluation, and export."""

	def __init__(self, config: RegressionPipelineConfig | None = None):
		self.config = config or RegressionPipelineConfig()
		self._run_started_at: float | None = None

	def _checkpoint(self, message: str) -> None:
		if not self.config.show_checkpoints:
			return
		if self._run_started_at is None:
			elapsed = 0.0
		else:
			elapsed = time.perf_counter() - self._run_started_at
		print(f"[regression-train][{elapsed:7.1f}s] {message}", flush=True)

	@staticmethod
	def _metric_key(name: str) -> str:
		return name.strip().lower().replace(" ", "_").replace("-", "_")

	def _save_best_pipeline(self, *, best_result: PipelineTrainResult) -> Path:
		model_path = Path(self.config.model_path)
		model_path.parent.mkdir(parents=True, exist_ok=True)
		ModelPipeline.save(best_result.best_estimator, str(model_path))
		self._checkpoint(f"Model artifact saved to: {model_path}")
		return model_path

	def _log_pipeline_run(
		self,
		*,
		schema: RegressionFeatureSchema,
		results: Mapping[str, PipelineTrainResult],
		evaluation_table: pd.DataFrame,
		best_result: PipelineTrainResult,
		model_path: Path,
	) -> str:
		self._checkpoint("Configuring MLflow")
		configure_regression_mlflow()

		params = {
			"dataset_path": str(self.config.dataset_path),
			"target_column": schema.target_column,
			"feature_columns": ",".join(schema.feature_columns),
			"numeric_features": ",".join(schema.numeric_features),
			"categorical_features": ",".join(schema.categorical_features),
			"log_columns": ",".join(schema.log_columns),
			"model_types": ",".join(self.config.model_types),
			"test_size": self.config.test_size,
			"random_state": self.config.random_state,
			"cv": self.config.cv,
			"scoring": self.config.scoring,
			"target_transform": self.config.target_transform,
			"encoding": self.config.encoding,
			"scaler": self.config.scaler,
			"skew_threshold": self.config.skew_threshold,
			"n_jobs": self.config.n_jobs,
		}

		self._checkpoint("Logging run artifacts and metrics to MLflow")
		with mlflow.start_run(run_name=self.config.run_name or "regression_pipeline") as run:
			mlflow.log_params(params)
			mlflow.log_param("best_model_name", best_result.name)

			mlflow.log_metric("best_cv_score", float(best_result.best_score))
			for model_name, result in results.items():
				mlflow.log_metric(f"{self._metric_key(model_name)}_cv_score", float(result.best_score))

			for model_name in evaluation_table.index.get_level_values("model").unique():
				test_metrics = evaluation_table.loc[(model_name, "test")]
				for metric_name, metric_value in test_metrics.items():
					value = float(metric_value)
					if np.isnan(value):
						continue
					mlflow.log_metric(
						f"{self._metric_key(model_name)}_test_{self._metric_key(str(metric_name))}",
						value,
					)

			mlflow.log_artifact(str(model_path), artifact_path="models")

			evaluation_export = model_path.parent / "regression_evaluation.csv"
			evaluation_table.reset_index().to_csv(evaluation_export, index=False)
			mlflow.log_artifact(str(evaluation_export), artifact_path="reports")
			self._checkpoint(f"MLflow logging complete. run_id={run.info.run_id}")

			return run.info.run_id

	def run(self) -> RegressionTrainingResult:
		"""Run full regression training and export the best model pipeline."""
		self._run_started_at = time.perf_counter()
		self._checkpoint("Start regression training")
		self._checkpoint(f"Loading dataset: {self.config.dataset_path}")
		raw_df = load_regression_dataframe(self.config.dataset_path)
		self._checkpoint(f"Dataset loaded ({raw_df.shape[0]} rows, {raw_df.shape[1]} cols)")
		self._checkpoint("Preparing training matrices")
		X, y, schema = build_training_matrices(raw_df, config=self.config, return_schema=True)

		self._checkpoint("Splitting train/test")
		X_train, X_test, y_train, y_test = split_train_test(X, y, config=self.config)

		self._checkpoint("Training candidate models (this may take a while)")
		results = train_regression_candidates(X_train, y_train, schema=schema, config=self.config)
		self._checkpoint("Model training complete")
		self._checkpoint("Selecting best model")
		best_result = select_best_candidate(results)
		self._checkpoint(f"Best model selected: {best_result.name} (cv={float(best_result.best_score):.4f})")
		self._checkpoint("Evaluating trained models")
		evaluation_table = evaluate_regression_candidates(
			results,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			decimals=2,
			sort_by="test_rmse",
		)

		self._checkpoint("Saving best model artifact")
		model_path = self._save_best_pipeline(best_result=best_result)

		run_id: str | None = None
		if self.config.log_to_mlflow:
			self._checkpoint("MLflow logging enabled")
			run_id = self._log_pipeline_run(
				schema=schema,
				results=results,
				evaluation_table=evaluation_table,
				best_result=best_result,
				model_path=model_path,
			)
		else:
			self._checkpoint("MLflow logging skipped")

		self._checkpoint("Training pipeline finished")

		return RegressionTrainingResult(
			results=results,
			schema=schema,
			evaluation_table=evaluation_table,
			best_model_name=best_result.name,
			best_params=dict(best_result.best_params),
			best_score=float(best_result.best_score),
			model_path=model_path,
			run_id=run_id,
		)


def start_regression_run(*, run_name: str | None = None):
	"""Start an MLflow run for regression training."""
	configure_regression_mlflow()
	return mlflow.start_run(run_name=run_name)


def log_regression_training(
	*,
	run_name: str | None = None,
	model_name: str | None = None,
	params: Mapping[str, Any] | None = None,
	metrics: Mapping[str, float] | None = None,
) -> str:
	"""Log a regression training run to MLflow and return run_id."""
	configure_regression_mlflow()
	with mlflow.start_run(run_name=run_name) as run:
		if model_name:
			mlflow.log_param("model_name", model_name)

		if params:
			for key, value in params.items():
				mlflow.log_param(key, value)

		if metrics:
			for key, value in metrics.items():
				mlflow.log_metric(key, float(value))

		return run.info.run_id


def run_regression_pipeline(config: RegressionPipelineConfig | None = None) -> RegressionTrainingResult:
	"""Convenience entrypoint to run regression training pipeline."""
	return RegressionTrainingPipeline(config=config).run()


if __name__ == "__main__":
	result = run_regression_pipeline()
	print(result.evaluation_table)
	print(f"Best model: {result.best_model_name}")
	print(f"Best CV score: {result.best_score:.4f}")
	print(f"Saved pipeline: {result.model_path}")

