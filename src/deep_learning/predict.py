from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
)


def labels_from_generator(generator, fallback: list[str] | tuple[str, ...] | None = None) -> list[str]:
	"""Get labels from generator."""
	if hasattr(generator, "class_indices") and isinstance(generator.class_indices, dict):
		inv = {v: k for k, v in generator.class_indices.items()}
		labels = [inv[i] for i in range(len(inv))]
		friendly = {
			"class_0_nonagri": "non-agri",
			"class_0_non_agri": "non-agri",
			"class_0_non-agri": "non-agri",
			"class_1_agri": "agri",
		}
		return [friendly.get(x, x) for x in labels]

	if fallback is None:
		return ["class_0", "class_1"]
	return list(fallback)


def collect_predictions(model, generator, *, verbose: int = 0):
	"""Collect y_true/y_pred/probabilities from model and generator."""
	if hasattr(generator, "reset"):
		generator.reset()

	y_true = np.asarray(getattr(generator, "classes"))
	if y_true.ndim > 1:
		y_true = np.argmax(y_true, axis=1)

	y_prob = np.asarray(model.predict(generator, verbose=verbose))
	if y_prob.ndim == 1:
		y_prob = y_prob.reshape(-1, 1)

	if y_prob.shape[1] == 1:
		y_prob2 = np.hstack([1 - y_prob, y_prob])
	else:
		y_prob2 = y_prob

	y_pred = np.argmax(y_prob2, axis=1)
	return y_true.astype(int), y_pred.astype(int), y_prob2.astype(float)


def compute_classification_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_prob2: np.ndarray,
) -> dict[str, float]:
	"""Compute core binary-classification metrics used in the notebook."""
	pos_prob = np.clip(y_prob2[:, 1], 1e-7, 1.0 - 1e-7)
	roc_auc = float(roc_auc_score(y_true, pos_prob))
	loss = float(log_loss(y_true, pos_prob, labels=[0, 1]))

	return {
		"Accuracy": float(accuracy_score(y_true, y_pred)),
		"Precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"Recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"F1": float(f1_score(y_true, y_pred, zero_division=0)),
		"ROC-AUC": roc_auc,
		"Loss": loss,
	}


def evaluate_generator(
	*,
	model_name: str,
	model,
	generator,
	class_labels: list[str] | tuple[str, ...] | None = None,
	predict_verbose: int = 0,
) -> dict[str, Any]:
	"""Evaluate a model against a generator and return metrics + artifacts."""
	labels = labels_from_generator(generator, fallback=class_labels)
	y_true, y_pred, y_prob2 = collect_predictions(model, generator, verbose=predict_verbose)

	metrics = compute_classification_metrics(y_true, y_pred, y_prob2)
	cm = confusion_matrix(y_true, y_pred)
	report = classification_report(
		y_true,
		y_pred,
		target_names=labels,
		zero_division=0,
		output_dict=True,
	)

	return {
		"model_name": model_name,
		"labels": labels,
		"metrics": metrics,
		"y_true": y_true,
		"y_pred": y_pred,
		"y_prob2": y_prob2,
		"confusion_matrix": cm,
		"classification_report": report,
	}


def results_to_dataframe(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
	"""Convert evaluation result dictionary to a compact metrics table."""
	rows: dict[str, dict[str, float]] = {}
	for model_name, payload in results.items():
		metrics = payload.get("metrics", {})
		rows[model_name] = {k: float(v) for k, v in metrics.items()}

	if not rows:
		return pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Loss"])

	df = pd.DataFrame(rows).T
	ordered = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Loss"]
	ordered_present = [c for c in ordered if c in df.columns]
	if ordered_present:
		df = df[ordered_present]
	return df

