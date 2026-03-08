from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import mlflow
import numpy as np
import pandas as pd

from src.classification.model import (
	ClassificationModelConfig,
	build_notebook_classification_config,
	load_classification_dataframe,
	prepare_classification_dataframe,
)
from src.config import Config
from src.functions.classification_pipeline import ClassificationPipeline, ClassificationPipelineResult
from src.functions.model_pipeline import ModelPipeline
from src.mlflow_utils import configure_mlflow


@dataclass
class ClassificationTrainingResult:
	"""Outputs from a full classification training run."""

	run_result: ClassificationPipelineResult
	evaluation_table: pd.DataFrame
	best_model_name: str
	best_params: dict[str, Any]
	best_score: float | None
	model_path: Path
	run_id: str | None = None


class ClassificationTrainingPipeline:
	"""Notebook-equivalent classification pipeline with export + optional MLflow."""

	def __init__(self, config: ClassificationModelConfig | None = None):
		self.config = config or ClassificationModelConfig()
		self._run_started_at: float | None = None

	def _checkpoint(self, message: str) -> None:
		if not self.config.show_checkpoints:
			return
		if self._run_started_at is None:
			elapsed = 0.0
		else:
			elapsed = time.perf_counter() - self._run_started_at
		print(f"[classification-train][{elapsed:7.1f}s] {message}", flush=True)

	@staticmethod
	def _metric_key(name: str) -> str:
		return name.strip().lower().replace(" ", "_").replace("-", "_")

	@staticmethod
	def _safe_float(value: Any) -> float | None:
		try:
			f = float(value)
		except Exception:
			return None
		if np.isnan(f) or np.isinf(f):
			return None
		return f

	def _resolve_best_details(
		self,
		*,
		run_result: ClassificationPipelineResult,
	) -> tuple[str, dict[str, Any], float | None]:
		best_model = run_result.best_model
		name = getattr(best_model, "name", None)
		params = dict(getattr(best_model, "best_params", {}) or {})
		score = self._safe_float(getattr(best_model, "best_score", None))

		if not name:
			name = getattr(getattr(best_model, "__class__", None), "__name__", "best_pipeline")

		return str(name), params, score

	def _save_best_pipeline(self, *, run_result: ClassificationPipelineResult) -> Path:
		pipeline_obj = run_result.best_pipeline
		if pipeline_obj is None:
			pipeline_obj = getattr(run_result.best_model, "best_estimator", None)
		if pipeline_obj is None:
			raise ValueError("No exportable pipeline found. Enable return_best_pipeline in config.")

		model_path = Path(self.config.model_path)
		model_path.parent.mkdir(parents=True, exist_ok=True)
		ModelPipeline.save(pipeline_obj, str(model_path))
		self._checkpoint(f"Model artifact saved to: {model_path}")
		return model_path

	def _log_pipeline_run(
		self,
		*,
		run_result: ClassificationPipelineResult,
		model_path: Path,
		best_model_name: str,
		best_score: float | None,
	) -> str:
		self._checkpoint("Configuring MLflow")
		configure_mlflow(
			experiment_name=self.config.mlflow_experiment,
			tracking_uri=str(Config.MLFLOW_TRACKING_URI),
		)

		params = {
			"dataset_path": str(self.config.dataset_path),
			"target_column": self.config.target_column,
			"categorical_features": ",".join(self.config.categorical_features),
			"pipeline_model_types": ",".join(self.config.pipeline_model_types),
			"encoding": self.config.encoding,
			"pipeline_encoding": self.config.pipeline_encoding,
			"scaler": self.config.scaler,
			"test_size": self.config.test_size,
			"stratify": self.config.stratify,
			"cv": self.config.cv,
			"scoring": self.config.scoring,
			"stratified_cv": self.config.stratified_cv,
			"metric_set": self.config.metric_set,
			"metrics": ",".join(self.config.metrics),
			"random_state": self.config.random_state,
			"auto_rename_columns": self.config.auto_rename_columns,
			"add_engineered_features": self.config.add_engineered_features,
			"log_rotational_speed": self.config.log_rotational_speed,
		}

		evaluation_table = run_result.metrics_table.copy()

		self._checkpoint("Logging run artifacts and metrics to MLflow")
		with mlflow.start_run(run_name=self.config.run_name or "classification_pipeline") as run:
			mlflow.log_params(params)
			mlflow.log_param("best_model_name", best_model_name)
			if best_score is not None:
				mlflow.log_metric("best_cv_score", best_score)

			if isinstance(evaluation_table, pd.DataFrame) and not evaluation_table.empty:
				if "model" in evaluation_table.columns:
					for _, row in evaluation_table.iterrows():
						model_name = self._metric_key(str(row.get("model", "model")))
						for col in evaluation_table.columns:
							if col == "model":
								continue
							value = self._safe_float(row[col])
							if value is None:
								continue
							mlflow.log_metric(f"{model_name}_{self._metric_key(col)}", value)
				else:
					for row_idx, row in evaluation_table.iterrows():
						row_key = self._metric_key(str(row_idx))
						for col in evaluation_table.columns:
							value = self._safe_float(row[col])
							if value is None:
								continue
							mlflow.log_metric(f"{row_key}_{self._metric_key(col)}", value)

			evaluation_export = model_path.parent / "classification_evaluation.csv"
			evaluation_table.to_csv(evaluation_export, index=False)

			mlflow.log_artifact(str(model_path), artifact_path="models")
			mlflow.log_artifact(str(evaluation_export), artifact_path="reports")
			self._checkpoint(f"MLflow logging complete. run_id={run.info.run_id}")
			return run.info.run_id

	def run(self) -> ClassificationTrainingResult:
		"""Run full classification training and save the best pipeline artifact."""
		self._run_started_at = time.perf_counter()
		self._checkpoint("Start classification training")
		self._checkpoint(f"Loading dataset: {self.config.dataset_path}")
		raw_df = load_classification_dataframe(self.config.dataset_path)
		self._checkpoint(f"Dataset loaded ({raw_df.shape[0]} rows, {raw_df.shape[1]} cols)")
		self._checkpoint("Preparing dataframe schema")
		prepared_df = prepare_classification_dataframe(raw_df, config=self.config)

		self._checkpoint("Building classification pipeline config")
		notebook_cfg = build_notebook_classification_config(self.config)
		self._checkpoint("Training candidate models (this may take a while)")
		run_result = ClassificationPipeline.run(prepared_df, config=notebook_cfg)
		self._checkpoint("Model training complete")

		self._checkpoint("Selecting best model")
		best_model_name, best_params, best_score = self._resolve_best_details(run_result=run_result)
		if best_score is None:
			self._checkpoint(f"Best model selected: {best_model_name}")
		else:
			self._checkpoint(f"Best model selected: {best_model_name} (cv={best_score:.4f})")

		self._checkpoint("Saving best model artifact")
		model_path = self._save_best_pipeline(run_result=run_result)

		run_id: str | None = None
		if self.config.log_to_mlflow:
			self._checkpoint("MLflow logging enabled")
			run_id = self._log_pipeline_run(
				run_result=run_result,
				model_path=model_path,
				best_model_name=best_model_name,
				best_score=best_score,
			)
		else:
			self._checkpoint("MLflow logging skipped")

		self._checkpoint("Training pipeline finished")

		return ClassificationTrainingResult(
			run_result=run_result,
			evaluation_table=run_result.metrics_table,
			best_model_name=best_model_name,
			best_params=best_params,
			best_score=best_score,
			model_path=model_path,
			run_id=run_id,
		)


def run_classification_pipeline(
	config: ClassificationModelConfig | None = None,
) -> ClassificationTrainingResult:
	"""Convenience entrypoint to run classification training pipeline."""
	return ClassificationTrainingPipeline(config=config).run()


if __name__ == "__main__":
	result = run_classification_pipeline()
	print(result.evaluation_table)
	print(f"Best model: {result.best_model_name}")
	if result.best_score is not None:
		print(f"Best CV score: {result.best_score:.4f}")
	print(f"Saved pipeline: {result.model_path}")
	if result.run_id:
		print(f"MLflow run id: {result.run_id}")

