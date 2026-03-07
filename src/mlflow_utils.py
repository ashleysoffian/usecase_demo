from __future__ import annotations

from pathlib import Path

import mlflow

from src.config import Config


def normalize_mlflow_tracking_uri(uri: str | Path) -> str:
	"""Normalize MLflow tracking URI.

	- If a URI scheme exists (e.g., http://, sqlite:///), return as-is.
	- If a local path is provided, use a SQLite MLflow backend at `<path>/mlflow.db`.
	"""
	uri_str = str(uri)
	if "://" in uri_str:
		return uri_str

	path = Path(uri_str).expanduser().resolve()
	path.mkdir(parents=True, exist_ok=True)
	return f"sqlite:///{path.as_posix()}/mlflow.db"


def configure_mlflow(*, experiment_name: str, tracking_uri: str | Path) -> str:
	"""Configure MLflow tracking + experiment and return effective tracking URI."""
	effective_uri = normalize_mlflow_tracking_uri(tracking_uri)
	mlflow.set_tracking_uri(effective_uri)
	mlflow.set_experiment(experiment_name)
	return effective_uri


def configure_chatbot_mlflow(*, experiment_name: str = "chatbot_rag") -> str:
	"""Configure MLflow for chatbot workloads."""
	return configure_mlflow(
		experiment_name=experiment_name,
		tracking_uri=str(Config.MLFLOW_TRACKING_URI),
	)


def configure_regression_mlflow() -> str:
	"""Configure MLflow for regression training workloads."""
	return configure_mlflow(
		experiment_name=Config.MLFLOW_EXPERIMENT_REGRESSION,
		tracking_uri=str(Config.MLFLOW_TRACKING_URI),
	)


def configure_deeplearning_mlflow() -> str:
	"""Configure MLflow for deep-learning training workloads."""
	return configure_mlflow(
		experiment_name=Config.MLFLOW_EXPERIMENT_DL,
		tracking_uri=str(Config.MLFLOW_TRACKING_URI),
	)
