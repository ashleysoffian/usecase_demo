from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	ConfusionMatrixDisplay,
	f1_score,
	mean_absolute_error,
	mean_squared_error,
	precision_score,
	r2_score,
	recall_score,
	roc_auc_score,
	RocCurveDisplay,
)


try:
	import xgboost  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
	xgboost = None  # type: ignore[assignment]


RegressionScoring = Literal[
	"r2",
	"neg_mean_squared_error",
	"neg_root_mean_squared_error",
	"neg_mean_absolute_error",
]
ClassificationScoring = Literal[
	"accuracy",
	"f1",
	"f1_macro",
	"recall",
	"roc_auc",
]

ScalerType = Literal["standard", "minmax", "robust", "none"]


@dataclass
class TrainResult:
	name: str
	best_estimator: Any
	best_params: dict[str, Any]
	best_score: float
	grid_search: GridSearchCV


class Model_Training:
	"""Model training utilities with CV and GridSearchCV."""

	# Hyperparameter grids (kept intentionally compact for reasonable runtime)
	REGRESSION_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
		"linear_regression": {
			"fit_intercept": [True, False],
			"positive": [False, True],
		},
		"random_forest": {
			"n_estimators": [200, 500],
			"max_depth": [None, 5, 10, 20],
			"min_samples_split": [2, 5, 10],
			"min_samples_leaf": [1, 2, 4],
			"max_features": ["sqrt", "log2", None],
		},
		"xgboost": {
			"n_estimators": [200, 500],
			"learning_rate": [0.01, 0.05, 0.1],
			"max_depth": [3, 5, 7],
			"subsample": [0.8, 1.0],
			"colsample_bytree": [0.8, 1.0],
			"reg_lambda": [1.0, 10.0],
			"min_child_weight": [1, 5],
		},
	}

	CLASSIFICATION_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
		"logistic_regression": {
			"C": [0.01, 0.1, 1.0, 10.0],
			"penalty": ["l2"],
			"solver": ["lbfgs"],
			"class_weight": [None, "balanced"],
			"max_iter": [1000],
		},
		"random_forest": {
			"n_estimators": [200, 500],
			"max_depth": [None, 5, 10, 20],
			"min_samples_split": [2, 5, 10],
			"min_samples_leaf": [1, 2, 4],
			"max_features": ["sqrt", "log2", None],
			"class_weight": [None, "balanced"],
		},
		"xgboost": {
			"n_estimators": [200, 500],
			"learning_rate": [0.01, 0.05, 0.1],
			"max_depth": [3, 5, 7],
			"subsample": [0.8, 1.0],
			"colsample_bytree": [0.8, 1.0],
			"reg_lambda": [1.0, 10.0],
			"min_child_weight": [1, 5],
			"scale_pos_weight": [1.0, 5.0, 10.0],
		},
	}

	@staticmethod
	def _make_cv(*, cv: int, random_state: int, stratified: bool):
		if stratified:
			return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
		return KFold(n_splits=cv, shuffle=True, random_state=random_state)

	@staticmethod
	def _get_xgboost_regressor(*, random_state: int):
		try:
			from xgboost import XGBRegressor  # type: ignore[import-not-found]
		except Exception as exc:  # pragma: no cover
			raise ImportError(
				"XGBoost is required for XGBRegressor. Install with `pip install xgboost`."
			) from exc
		return XGBRegressor(
			random_state=random_state,
			objective="reg:squarederror",
			n_jobs=-1,
		)

	@staticmethod
	def _get_xgboost_classifier(*, random_state: int):
		try:
			from xgboost import XGBClassifier  # type: ignore[import-not-found]
		except Exception as exc:  # pragma: no cover
			raise ImportError(
				"XGBoost is required for XGBClassifier. Install with `pip install xgboost`."
			) from exc
		return XGBClassifier(
			random_state=random_state,
			eval_metric="logloss",
			n_jobs=-1,
			use_label_encoder=False,
		)

	@staticmethod
	def _grid_search(
		*,
		name: str,
		estimator: Any,
		param_grid: dict[str, list[Any]],
		X: Any,
		y: Any,
		cv_obj: Any,
		scoring: str,
		n_jobs: int,
		verbose: int,
		refit: bool,
	):
		gs = GridSearchCV(
			estimator=estimator,
			param_grid=param_grid,
			scoring=scoring,
			cv=cv_obj,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)
		gs.fit(X, y)
		return TrainResult(
			name=name,
			best_estimator=gs.best_estimator_,
			best_params=dict(gs.best_params_ or {}),
			best_score=float(gs.best_score_),
			grid_search=gs,
		)

	@staticmethod
	def train_regression(
		X: Any,
		y: Any,
		*,
		cv: int = 5,
		scoring: RegressionScoring = "neg_root_mean_squared_error",
		random_state: int = 42,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
	):
		"""Train regression models with CV using GridSearchCV.

		Models:
			- LinearRegression
			- RandomForestRegressor
			- XGBRegressor

		Returns:
			Dict of model_name -> TrainResult
		"""
		cv_obj = Model_Training._make_cv(cv=cv, random_state=random_state, stratified=False)

		results: dict[str, TrainResult] = {}
		results["linear_regression"] = Model_Training._grid_search(
			name="linear_regression",
			estimator=LinearRegression(),
			param_grid=Model_Training.REGRESSION_PARAM_GRIDS["linear_regression"],
			X=X,
			y=y,
			cv_obj=cv_obj,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)

		rf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
		results["random_forest"] = Model_Training._grid_search(
			name="random_forest",
			estimator=rf,
			param_grid=Model_Training.REGRESSION_PARAM_GRIDS["random_forest"],
			X=X,
			y=y,
			cv_obj=cv_obj,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)

		xgb = Model_Training._get_xgboost_regressor(random_state=random_state)
		results["xgboost"] = Model_Training._grid_search(
			name="xgboost",
			estimator=xgb,
			param_grid=Model_Training.REGRESSION_PARAM_GRIDS["xgboost"],
			X=X,
			y=y,
			cv_obj=cv_obj,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)

		return results

	@staticmethod
	def train_classification(
		X: Any,
		y: Any,
		*,
		cv: int = 5,
		scoring: ClassificationScoring = "accuracy",
		random_state: int = 42,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
		stratified_cv: bool = True,
	):
		"""Train classification models with CV using GridSearchCV.

		Models:
			- LogisticRegression
			- RandomForestClassifier
			- XGBClassifier

		Returns:
			Dict of model_name -> TrainResult
		"""
		cv_obj = Model_Training._make_cv(cv=cv, random_state=random_state, stratified=stratified_cv)

		results: dict[str, TrainResult] = {}
		logreg = LogisticRegression()
		results["logistic_regression"] = Model_Training._grid_search(
			name="logistic_regression",
			estimator=logreg,
			param_grid=Model_Training.CLASSIFICATION_PARAM_GRIDS["logistic_regression"],
			X=X,
			y=y,
			cv_obj=cv_obj,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)

		rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
		results["random_forest"] = Model_Training._grid_search(
			name="random_forest",
			estimator=rf,
			param_grid=Model_Training.CLASSIFICATION_PARAM_GRIDS["random_forest"],
			X=X,
			y=y,
			cv_obj=cv_obj,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)

		xgb = Model_Training._get_xgboost_classifier(random_state=random_state)
		results["xgboost"] = Model_Training._grid_search(
			name="xgboost",
			estimator=xgb,
			param_grid=Model_Training.CLASSIFICATION_PARAM_GRIDS["xgboost"],
			X=X,
			y=y,
			cv_obj=cv_obj,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)

		return results


class ModelTrainer:
	"""API for splitting, scaling, and training."""

	@staticmethod
	def _unwrap_estimator(model_or_result: Any) -> Any:
		"""Accept either an estimator or a TrainResult and return the estimator."""
		if isinstance(model_or_result, TrainResult):
			return model_or_result.best_estimator
		return model_or_result

	@staticmethod
	def split_train_test(
		X: Any,
		y: Any,
		*,
		test_size: float = 0.2,
		random_state: int = 42,
		stratify: bool = False,
	):
		"""Split into train/test sets."""
		return train_test_split(
			X,
			y,
			test_size=test_size,
			random_state=random_state,
			stratify=y if stratify else None,
		)

	@staticmethod
	def scale_after_split(
		X_train: Any,
		X_test: Any,
		*,
		scaler: ScalerType = "standard",
		columns: list[str] | None = None,
		return_scaler: bool = False,
	):
		"""Fit scaler on X_train, transform X_train and X_test (no leakage)."""
		scaler = str(scaler).lower().strip()
		if scaler == "none":
			if return_scaler:
				return X_train, X_test, None
			return X_train, X_test

		if scaler == "standard":
			scaler_obj = StandardScaler()
		elif scaler == "minmax":
			scaler_obj = MinMaxScaler()
		elif scaler == "robust":
			scaler_obj = RobustScaler()
		else:
			raise ValueError("scaler must be one of: 'standard', 'minmax', 'robust', 'none'")

		# DataFrame path (preserve columns/index)
		if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
			if columns is None:
				scale_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
			else:
				scale_cols = columns
			missing = [
				c for c in scale_cols if c not in X_train.columns or c not in X_test.columns
			]
			if missing:
				raise ValueError(f"columns not found in X_train/X_test: {missing}")

			X_train_out = X_train.copy()
			X_test_out = X_test.copy()

			scaler_obj.fit(X_train_out[scale_cols])
			X_train_out[scale_cols] = scaler_obj.transform(X_train_out[scale_cols])
			X_test_out[scale_cols] = scaler_obj.transform(X_test_out[scale_cols])

			if return_scaler:
				return X_train_out, X_test_out, scaler_obj
			return X_train_out, X_test_out

		# ndarray/array-like path
		X_train_arr = np.asarray(X_train)
		X_test_arr = np.asarray(X_test)
		scaler_obj.fit(X_train_arr)
		X_train_scaled = scaler_obj.transform(X_train_arr)
		X_test_scaled = scaler_obj.transform(X_test_arr)
		if return_scaler:
			return X_train_scaled, X_test_scaled, scaler_obj
		return X_train_scaled, X_test_scaled

	@staticmethod
	def train_regression(
		X: Any,
		y: Any,
		*,
		cv: int = 5,
		scoring: RegressionScoring = "neg_root_mean_squared_error",
		random_state: int = 42,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
	):
		"""Train regression models via GridSearchCV."""
		return Model_Training.train_regression(
			X,
			y,
			cv=cv,
			scoring=scoring,
			random_state=random_state,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)

	@staticmethod
	def train_classification(
		X: Any,
		y: Any,
		*,
		cv: int = 5,
		scoring: ClassificationScoring = "accuracy",
		random_state: int = 42,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
		stratified_cv: bool = True,
	):
		"""Train classification models via GridSearchCV."""
		return Model_Training.train_classification(
			X,
			y,
			cv=cv,
			scoring=scoring,
			random_state=random_state,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
			stratified_cv=stratified_cv,
		)

	@staticmethod
	def select_best(results: dict[str, TrainResult]) -> TrainResult:
		"""Select best model by CV score (higher is better in sklearn)."""
		if not results:
			raise ValueError("results is empty")
		return max(results.values(), key=lambda r: r.best_score)

	@staticmethod
	def evaluate_regression(
		estimator: Any,
		X_train: Any,
		y_train: Any,
		X_test: Any,
		y_test: Any,
		*,
		decimals: int = 2,
	) -> pd.DataFrame:
		"""Create a train/test evaluation table for regression."""
		estimator = ModelTrainer._unwrap_estimator(estimator)
		if not hasattr(estimator, "predict"):
			raise TypeError("estimator must implement predict() (or be a TrainResult)")
		yhat_train = estimator.predict(X_train)
		yhat_test = estimator.predict(X_test)

		def _rmse(y_true, y_pred) -> float:
			return float(np.sqrt(mean_squared_error(y_true, y_pred)))

		def _mape(y_true, y_pred) -> float:
			"""Mean Absolute Percentage Error (in %), ignoring zero targets."""
			y_true_arr = np.asarray(y_true, dtype=float)
			y_pred_arr = np.asarray(y_pred, dtype=float)
			denom = np.abs(y_true_arr)
			mask = denom > 0
			if not np.any(mask):
				return float("nan")
			return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / denom[mask])) * 100.0)

		out = pd.DataFrame(
			{
				"rmse": [_rmse(y_train, yhat_train), _rmse(y_test, yhat_test)],
				"mae": [
					float(mean_absolute_error(y_train, yhat_train)),
					float(mean_absolute_error(y_test, yhat_test)),
				],
				"mape": [_mape(y_train, yhat_train), _mape(y_test, yhat_test)],
				"r2": [float(r2_score(y_train, yhat_train)), float(r2_score(y_test, yhat_test))],
			},
			index=["train", "test"],
		)
		if decimals is not None:
			out = out.round(decimals)
		return out

	@staticmethod
	def evaluate_regression_all(
		results: dict[str, TrainResult] | dict[str, Any],
		X_train: Any,
		y_train: Any,
		X_test: Any,
		y_test: Any,
		*,
		sort_by: Literal["test_rmse", "test_mae", "test_mape", "test_r2"] | None = "test_rmse",
		decimals: int = 2,
	) -> pd.DataFrame:
		"""Evaluate every model returned by `train_regression`.

		Returns a single DataFrame with a MultiIndex (model, split).
		"""
		if not results:
			raise ValueError("results is empty")

		frames: list[pd.DataFrame] = []
		for name, res in results.items():
			metrics = ModelTrainer.evaluate_regression(
				estimator=res,
				X_train=X_train,
				y_train=y_train,
				X_test=X_test,
				y_test=y_test,
				decimals=decimals,
			)
			metrics = metrics.copy()
			metrics.insert(0, "model", name)
			metrics.insert(1, "split", metrics.index)
			metrics = metrics.set_index(["model", "split"])
			frames.append(metrics)

		out = pd.concat(frames).sort_index()

		if sort_by is not None:
			if sort_by == "test_rmse":
				order = out.xs("test", level="split")["rmse"].sort_values(ascending=True).index
			elif sort_by == "test_mae":
				order = out.xs("test", level="split")["mae"].sort_values(ascending=True).index
			elif sort_by == "test_mape":
				order = out.xs("test", level="split")["mape"].sort_values(ascending=True).index
			elif sort_by == "test_r2":
				order = out.xs("test", level="split")["r2"].sort_values(ascending=False).index
			else:
				raise ValueError("Invalid sort_by value")
			out = out.loc[pd.IndexSlice[order, :], :]

		return out

	@staticmethod
	def plot_actual_vs_predicted(
		y_true: Any,
		y_pred: Any,
		*,
		title: str = "Actual vs Predicted",
		xlabel: str = "Actual",
		ylabel: str = "Predicted",
		add_diagonal: bool = True,
		use_plotly: bool = True,
	):
		"""Plot actual vs predicted values for regression.

		Parameters
		----------
		y_true : array-like
			True target values.
		y_pred : array-like
			Predicted values.
		title : str, default="Actual vs Predicted"
			Plot title.
		xlabel : str, default="Actual"
			X-axis label.
		ylabel : str, default="Predicted"
			Y-axis label.
		add_diagonal : bool, default=True
			Whether to add a diagonal reference line (perfect predictions).
		use_plotly : bool, default=True
			If True, use Plotly for interactive plot. If False, use Matplotlib.

		Returns
		-------
		Figure object (plotly.graph_objs.Figure or matplotlib.figure.Figure)
		"""
		y_true_arr = np.asarray(y_true)
		y_pred_arr = np.asarray(y_pred)

		if y_true_arr.shape != y_pred_arr.shape:
			raise ValueError("y_true and y_pred must have the same shape")

		if use_plotly:
			try:
				import plotly.express as px
				import plotly.graph_objects as go
			except Exception as exc:  # pragma: no cover
				raise ImportError("Plotly is required for interactive plots. Install with `pip install plotly`.") from exc

			# Create DataFrame for Plotly
			plot_df = pd.DataFrame({
				xlabel: y_true_arr,
				ylabel: y_pred_arr
			})

			fig = px.scatter(
				plot_df,
				x=xlabel,
				y=ylabel,
				title=title,
				labels={xlabel: xlabel, ylabel: ylabel}
			)

			# Add diagonal line if requested
			if add_diagonal:
				min_val = min(y_true_arr.min(), y_pred_arr.min())
				max_val = max(y_true_arr.max(), y_pred_arr.max())
				fig.add_shape(
					type='line',
					x0=min_val, y0=min_val,
					x1=max_val, y1=max_val,
					line=dict(color='red', dash='dash'),
					name='Perfect Prediction'
				)

			fig.update_layout(
				showlegend=False,
				hovermode='closest'
			)

			return fig

		else:
			try:
				import matplotlib.pyplot as plt
			except Exception as exc:  # pragma: no cover
				raise ImportError("Matplotlib is required for static plots. Install with `pip install matplotlib`.") from exc

			fig, ax = plt.subplots(figsize=(8, 6))
			ax.scatter(y_true_arr, y_pred_arr, alpha=0.5, edgecolors='k', linewidth=0.5)

			# Add diagonal line if requested
			if add_diagonal:
				min_val = min(y_true_arr.min(), y_pred_arr.min())
				max_val = max(y_true_arr.max(), y_pred_arr.max())
				ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
				ax.legend()

			ax.set_xlabel(xlabel)
			ax.set_ylabel(ylabel)
			ax.set_title(title)
			ax.grid(True, alpha=0.3)
			fig.tight_layout()

			return fig

	@staticmethod
	def plot_actual_vs_predicted_all(
		results: dict[str, TrainResult] | dict[str, Any],
		X_test: Any,
		y_test: Any,
		*,
		title: str | None = None,
		xlabel: str = "Actual",
		ylabel: str = "Predicted",
		add_diagonal: bool = True,
		use_plotly: bool = True,
		figsize: tuple[int, int] = (15, 5),
	):
		"""Plot actual vs predicted for all regression models.

		Parameters
		----------
		results : dict
			Dictionary of model results (e.g., from train_regression).
		X_test : array-like
			Test features.
		y_test : array-like
			Test target values.
		title : str, optional
			Overall plot title. If None, uses default.
		xlabel : str, default="Actual"
			X-axis label.
		ylabel : str, default="Predicted"
			Y-axis label.
		add_diagonal : bool, default=True
			Whether to add a diagonal reference line.
		use_plotly : bool, default=True
			If True, use Plotly subplots. If False, use Matplotlib.
		figsize : tuple, default=(15, 5)
			Figure size for matplotlib (width, height).

		Returns
		-------
		Figure object (plotly.graph_objs.Figure or matplotlib.figure.Figure)
		"""
		if not results:
			raise ValueError("results is empty")

		y_test_arr = np.asarray(y_test)
		n_models = len(results)

		if use_plotly:
			try:
				from plotly.subplots import make_subplots
				import plotly.graph_objects as go
			except Exception as exc:  # pragma: no cover
				raise ImportError("Plotly is required for interactive plots. Install with `pip install plotly`.") from exc

			# Create subplots
			fig = make_subplots(
				rows=1,
				cols=n_models,
				subplot_titles=[name.replace('_', ' ').title() for name in results.keys()],
				horizontal_spacing=0.1
			)

			min_val = float('inf')
			max_val = float('-inf')

			# Collect all predictions and find global min/max for consistent diagonal lines
			predictions = {}
			for name, res in results.items():
				estimator = ModelTrainer._unwrap_estimator(res)
				y_pred = estimator.predict(X_test)
				predictions[name] = y_pred
				min_val = min(min_val, y_test_arr.min(), y_pred.min())
				max_val = max(max_val, y_test_arr.max(), y_pred.max())

			# Add scatter plots
			for idx, (name, y_pred) in enumerate(predictions.items(), 1):
				fig.add_trace(
					go.Scatter(
						x=y_test_arr,
						y=y_pred,
						mode='markers',
						marker=dict(size=6, opacity=0.6),
						name=name.replace('_', ' ').title(),
						showlegend=False
					),
					row=1, col=idx
				)

				# Add diagonal line
				if add_diagonal:
					fig.add_shape(
						type='line',
						x0=min_val, y0=min_val,
						x1=max_val, y1=max_val,
						line=dict(color='red', dash='dash', width=2),
						row=1, col=idx
					)

				# Update axes labels
				fig.update_xaxes(title_text=xlabel, row=1, col=idx)
				if idx == 1:
					fig.update_yaxes(title_text=ylabel, row=1, col=idx)

			# Update layout
			if title is None:
				title = "Actual vs Predicted - All Models"
			fig.update_layout(
				title_text=title,
				height=500,
				showlegend=False,
				hovermode='closest'
			)

			return fig

		else:
			try:
				import matplotlib.pyplot as plt
			except Exception as exc:  # pragma: no cover
				raise ImportError("Matplotlib is required for static plots. Install with `pip install matplotlib`.") from exc

			fig, axes = plt.subplots(1, n_models, figsize=figsize)
			if n_models == 1:
				axes = [axes]

			min_val = float('inf')
			max_val = float('-inf')

			# Collect all predictions and find global min/max
			predictions = {}
			for name, res in results.items():
				estimator = ModelTrainer._unwrap_estimator(res)
				y_pred = estimator.predict(X_test)
				predictions[name] = y_pred
				min_val = min(min_val, y_test_arr.min(), y_pred.min())
				max_val = max(max_val, y_test_arr.max(), y_pred.max())

			# Create plots
			for ax, (name, y_pred) in zip(axes, predictions.items()):
				ax.scatter(y_test_arr, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)

				# Add diagonal line
				if add_diagonal:
					ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)

				ax.set_xlabel(xlabel)
				ax.set_ylabel(ylabel)
				ax.set_title(name.replace('_', ' ').title())
				ax.grid(True, alpha=0.3)

			if title is None:
				title = "Actual vs Predicted - All Models"
			fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
			fig.tight_layout()

			return fig

	@staticmethod
	def evaluate_classification(
		estimator: Any,
		X_train: Any,
		y_train: Any,
		X_test: Any,
		y_test: Any,
		*,
		average: Literal["binary", "macro", "weighted"] | None = None,
		metric_set: Literal["default", "imbalanced", "balanced", "all"] = "default",
		metrics: list[str] | tuple[str, ...] | None = None,
		pos_label: Any = 1,
		decimals: int = 2,
	) -> pd.DataFrame:
		"""Create a train/test evaluation table for classification.

		Supports selectable metric sets to make the function usable for both
		imbalanced and balanced datasets.

		Metric selection:
		- metric_set='default': accuracy, precision, recall, f1 (+ roc_auc when available)
		- metric_set='imbalanced': recall, f1, roc_auc, confusion_matrix
		- metric_set='balanced': accuracy, precision, recall, f1, roc_auc, confusion_matrix
		- metric_set='all': accuracy, precision, recall, f1, roc_auc, confusion_matrix
		- metrics: explicit list/tuple of metric names overrides metric_set

		Confusion matrix output:
		- Binary: adds columns tn, fp, fn, tp
		- Multi-class: adds a 'confusion_matrix' column with list-of-lists

		ROC-AUC is included only when `predict_proba` is available.
		"""
		estimator = ModelTrainer._unwrap_estimator(estimator)
		if not hasattr(estimator, "predict"):
			raise TypeError("estimator must implement predict() (or be a TrainResult)")

		y_train_arr = np.asarray(y_train)
		classes = np.unique(y_train_arr)
		is_binary = len(classes) == 2

		if average is None:
			average = "binary" if is_binary else "weighted"

		yhat_train = estimator.predict(X_train)
		yhat_test = estimator.predict(X_test)

		if metrics is not None:
			if not isinstance(metrics, (list, tuple)):
				raise TypeError("metrics must be a list/tuple of metric names or None")
			selected = [str(m).lower().strip() for m in metrics]
		else:
			metric_set = str(metric_set).lower().strip()
			if metric_set == "default":
				selected = ["accuracy", "precision", "recall", "f1"]
			elif metric_set == "imbalanced":
				selected = ["recall", "f1", "roc_auc", "confusion_matrix"]
			elif metric_set in ("balanced", "all"):
				selected = ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]
			else:
				raise ValueError("metric_set must be one of: 'default', 'imbalanced', 'balanced', 'all'")

		# Pre-compute probabilities if ROC-AUC requested.
		need_proba = any(m in ("roc_auc", "roc_auc_ovr_weighted") for m in selected)
		p_train = p_test = None
		if need_proba and hasattr(estimator, "predict_proba"):
			try:
				p_train = estimator.predict_proba(X_train)
				p_test = estimator.predict_proba(X_test)
			except Exception:
				p_train = p_test = None

		def _score_kwargs():
			# Only pass pos_label when using binary averaging.
			if average == "binary":
				return {"average": average, "zero_division": 0, "pos_label": pos_label}
			return {"average": average, "zero_division": 0}

		cols: dict[str, list[Any]] = {}
		for m in selected:
			if m == "accuracy":
				cols["accuracy"] = [
					float(accuracy_score(y_train, yhat_train)),
					float(accuracy_score(y_test, yhat_test)),
				]
			elif m == "precision":
				cols["precision"] = [
					float(precision_score(y_train, yhat_train, **_score_kwargs())),
					float(precision_score(y_test, yhat_test, **_score_kwargs())),
				]
			elif m == "recall":
				cols["recall"] = [
					float(recall_score(y_train, yhat_train, **_score_kwargs())),
					float(recall_score(y_test, yhat_test, **_score_kwargs())),
				]
			elif m == "f1":
				cols["f1"] = [
					float(f1_score(y_train, yhat_train, **_score_kwargs())),
					float(f1_score(y_test, yhat_test, **_score_kwargs())),
				]
			elif m == "roc_auc":
				# Include only when proba available.
				if p_train is None or p_test is None:
					continue
				try:
					if is_binary:
						# Choose the column that corresponds to pos_label when possible.
						pos_idx = 1
						classes_ = getattr(estimator, "classes_", None)
						if classes_ is not None:
							classes_list = list(classes_)
							if pos_label in classes_list:
								pos_idx = classes_list.index(pos_label)
						cols["roc_auc"] = [
							float(roc_auc_score(y_train, p_train[:, pos_idx])),
							float(roc_auc_score(y_test, p_test[:, pos_idx])),
						]
					else:
						cols["roc_auc_ovr_weighted"] = [
							float(roc_auc_score(y_train, p_train, multi_class="ovr", average="weighted")),
							float(roc_auc_score(y_test, p_test, multi_class="ovr", average="weighted")),
						]
				except Exception:
					# If ROC-AUC can't be computed (label format, proba shape, etc.), omit it.
					pass
			elif m == "confusion_matrix":
				try:
					cm_train = confusion_matrix(y_train, yhat_train)
					cm_test = confusion_matrix(y_test, yhat_test)
					if is_binary and cm_train.shape == (2, 2) and cm_test.shape == (2, 2):
						tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel().tolist()
						tn_te, fp_te, fn_te, tp_te = cm_test.ravel().tolist()
						cols["tn"] = [int(tn_tr), int(tn_te)]
						cols["fp"] = [int(fp_tr), int(fp_te)]
						cols["fn"] = [int(fn_tr), int(fn_te)]
						cols["tp"] = [int(tp_tr), int(tp_te)]
					else:
						cols["confusion_matrix"] = [cm_train.tolist(), cm_test.tolist()]
				except Exception:
					pass
			else:
				raise ValueError(
					f"Unknown metric {m!r}. Supported: accuracy, precision, recall, f1, roc_auc, confusion_matrix"
				)

		out = pd.DataFrame(cols, index=["train", "test"])

		if decimals is not None:
			out = out.round(decimals)
		return out

	@staticmethod
	def evaluate_classification_all(
		results: dict[str, Any],
		X_train: Any,
		y_train: Any,
		X_test: Any,
		y_test: Any,
		*,
		average: Literal["binary", "macro", "weighted"] | None = None,
		metric_set: Literal["default", "imbalanced", "balanced", "all"] = "default",
		metrics: list[str] | tuple[str, ...] | None = None,
		pos_label: Any = 1,
		decimals: int = 3,
		# Fit-status labeling
		fit_metric: str = "recall",
		overfit_gap: float = 0.10,
		overfit_train_min: float = 0.85,
		underfit_max: float = 0.60,
		add_gap_column: bool = True,
		sort_by: str | None = None,
	) -> pd.DataFrame:
		"""Evaluate multiple classification models and return a single summary table.

		This calls `evaluate_classification(...)` for each model, then flattens the
		(train/test) rows into `train_<metric>` / `test_<metric>` columns.

		Adds:
		- model: model name
		- fit_status: one of {overfit, underfit, ideal, unknown}
		- optionally: gap_<fit_metric> = train - test

		Fit-status heuristic (on `fit_metric`):
		- overfit: (train - test) >= overfit_gap AND train >= overfit_train_min
		- underfit: train <= underfit_max AND test <= underfit_max
		- ideal: otherwise

		Args:
			results: dict[name -> TrainResult or fitted estimator]
			fit_metric: metric to judge fit quality (e.g., recall, f1, accuracy, roc_auc)
			sort_by: column name to sort by (e.g., "test_recall"); defaults to "test_<fit_metric>" when present.
		"""
		results = ModelTrainer._as_estimator_dict(results)
		rows: list[dict[str, Any]] = []
		fit_metric = str(fit_metric).lower().strip()

		def _flatten_metrics(df: pd.DataFrame) -> dict[str, Any]:
			out: dict[str, Any] = {}
			for col in df.columns:
				out[f"train_{col}"] = df.loc["train", col]
				out[f"test_{col}"] = df.loc["test", col]
			return out

		for name, res in results.items():
			eval_df = ModelTrainer.evaluate_classification(
				res,
				X_train,
				y_train,
				X_test,
				y_test,
				average=average,
				metric_set=metric_set,
				metrics=metrics,
				pos_label=pos_label,
				decimals=decimals,
			)
			flat = _flatten_metrics(eval_df)
			flat["model"] = name

			train_key = f"train_{fit_metric}"
			test_key = f"test_{fit_metric}"
			train_val = flat.get(train_key, np.nan)
			test_val = flat.get(test_key, np.nan)
			try:
				train_f = float(train_val)
				test_f = float(test_val)
			except Exception:
				train_f = float("nan")
				test_f = float("nan")

			gap = train_f - test_f if (np.isfinite(train_f) and np.isfinite(test_f)) else float("nan")
			if add_gap_column:
				flat[f"gap_{fit_metric}"] = gap

			if not (np.isfinite(train_f) and np.isfinite(test_f)):
				flat["fit_status"] = "unknown"
			elif gap >= overfit_gap and train_f >= overfit_train_min:
				flat["fit_status"] = "overfit"
			elif train_f <= underfit_max and test_f <= underfit_max:
				flat["fit_status"] = "underfit"
			else:
				flat["fit_status"] = "ideal"

			rows.append(flat)

		out = pd.DataFrame(rows)
		# Put key identifiers first.
		front = [c for c in ["model", "fit_status"] if c in out.columns]
		rest = [c for c in out.columns if c not in front]
		out = out[front + rest]

		if sort_by is None:
			candidate = f"test_{fit_metric}"
			sort_by = candidate if candidate in out.columns else None
		if sort_by is not None and sort_by in out.columns:
			out = out.sort_values(by=sort_by, ascending=False, na_position="last").reset_index(drop=True)

		return out

	@staticmethod
	def plot_metrics_table(
		metrics: pd.DataFrame,
		*,
		title: str | None = None,
		fmt: str = "{:.3f}",
		figsize: tuple[float, float] | None = None,
	):
		"""Render a metrics DataFrame as a matplotlib table figure."""
		try:
			import matplotlib.pyplot as plt
		except Exception as exc:  # pragma: no cover
			raise ImportError("Plotting requires matplotlib.") from exc

		if not isinstance(metrics, pd.DataFrame):
			raise TypeError("metrics must be a pandas DataFrame")

		if figsize is None:
			fig_w = max(6.0, 1.1 * (metrics.shape[1] + 1))
			fig_h = max(2.0, 0.6 * (metrics.shape[0] + 2))
			figsize = (fig_w, fig_h)

		fig, ax = plt.subplots(figsize=figsize)
		ax.axis("off")

		cell_text: list[list[str]] = []
		for _, row in metrics.iterrows():
			row_text: list[str] = []
			for v in row.values:
				if v is None or (isinstance(v, float) and np.isnan(v)):
					row_text.append("")
				elif isinstance(v, (int, float, np.floating)):
					row_text.append(fmt.format(float(v)))
				else:
					row_text.append(str(v))
			cell_text.append(row_text)

		table = ax.table(
			cellText=cell_text,
			rowLabels=metrics.index.tolist(),
			colLabels=metrics.columns.tolist(),
			cellLoc="center",
			loc="center",
		)
		table.auto_set_font_size(False)
		table.set_fontsize(10)
		table.scale(1, 1.4)

		if title:
			ax.set_title(title, pad=12, fontweight="bold")

		fig.tight_layout()
		return fig

	@staticmethod
	def plot_roc_curve(
		estimator: Any,
		X: Any,
		y: Any,
		*,
		title: str = "ROC Curve",
	):
		"""Plot ROC curve (binary classification only)."""
		try:
			import matplotlib.pyplot as plt
		except Exception as exc:  # pragma: no cover
			raise ImportError("Plotting requires matplotlib.") from exc

		estimator = ModelTrainer._unwrap_estimator(estimator)
		if not hasattr(estimator, "predict"):
			raise TypeError("estimator must implement predict() (or be a TrainResult)")

		y_arr = np.asarray(y)
		if len(np.unique(y_arr)) != 2:
			raise ValueError("plot_roc_curve supports binary classification only")

		fig, ax = plt.subplots(figsize=(6, 5))
		RocCurveDisplay.from_estimator(estimator, X, y, ax=ax)
		ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
		ax.set_title(title)
		fig.tight_layout()
		return fig

	@staticmethod
	def plot_confusion_matrix(
		estimator: Any,
		X: Any,
		y: Any,
		*,
		labels: list[Any] | None = None,
		normalize: Literal["true", "pred", "all"] | None = None,
		cmap: str = "Blues",
		title: str = "Confusion Matrix",
	):
		"""Plot a confusion matrix (binary or multiclass).

		Args:
			estimator: Fitted estimator or TrainResult.
			X: Features (typically test split).
			y: True labels.
			labels: Optional list of labels to index the matrix.
			normalize: Normalize confusion matrix over the true (rows), predicted (cols), or all.
			cmap: Matplotlib colormap name.
			title: Plot title.

		Returns:
			Matplotlib Figure.
		"""
		try:
			import matplotlib.pyplot as plt
		except Exception as exc:  # pragma: no cover
			raise ImportError("Plotting requires matplotlib.") from exc

		estimator = ModelTrainer._unwrap_estimator(estimator)
		if not hasattr(estimator, "predict"):
			raise TypeError("estimator must implement predict() (or be a TrainResult)")

		y_pred = estimator.predict(X)
		fig, ax = plt.subplots(figsize=(6, 5))
		ConfusionMatrixDisplay.from_predictions(
			y_true=y,
			y_pred=y_pred,
			labels=labels,
			normalize=normalize,
			cmap=cmap,
			ax=ax,
			colorbar=True,
		)
		# sklearn adds minor ticks + minor grid to draw cell borders; remove them
		# to avoid the extra inner square lines.
		ax.grid(False, which="both")
		ax.set_xticks([], minor=True)
		ax.set_yticks([], minor=True)
		ax.tick_params(which="minor", bottom=False, left=False)
		ax.set_title(title)
		fig.tight_layout()
		return fig

	@staticmethod
	def _as_estimator_dict(models: Any) -> dict[str, Any]:
		"""Normalize input to a dict[name] -> estimator/TrainResult."""
		if isinstance(models, dict):
			return models
		raise TypeError("models must be a dict[name -> estimator or TrainResult]")

	@staticmethod
	def _score_single_model(
		estimator: Any,
		X: Any,
		y: Any,
		*,
		metric: Literal["roc_auc", "recall", "f1", "accuracy"] = "roc_auc",
		average: Literal["binary", "macro", "weighted"] | None = None,
		pos_label: Any = 1,
	) -> float:
		"""Compute a selection score for choosing the best model."""
		est = ModelTrainer._unwrap_estimator(estimator)
		y_arr = np.asarray(y)
		classes = np.unique(y_arr)
		is_binary = len(classes) == 2
		if average is None:
			average = "binary" if is_binary else "weighted"

		metric = str(metric).lower().strip()  # type: ignore[assignment]
		if metric == "accuracy":
			y_hat = est.predict(X)
			return float(accuracy_score(y, y_hat))
		if metric == "recall":
			y_hat = est.predict(X)
			kwargs = {"average": average, "zero_division": 0}
			if average == "binary":
				kwargs["pos_label"] = pos_label
			return float(recall_score(y, y_hat, **kwargs))
		if metric == "f1":
			y_hat = est.predict(X)
			kwargs = {"average": average, "zero_division": 0}
			if average == "binary":
				kwargs["pos_label"] = pos_label
			return float(f1_score(y, y_hat, **kwargs))
		if metric == "roc_auc":
			if not hasattr(est, "predict_proba"):
				return float("nan")
			try:
				p = est.predict_proba(X)
			except Exception:
				return float("nan")
			try:
				if is_binary:
					pos_idx = 1
					classes_ = getattr(est, "classes_", None)
					if classes_ is not None:
						classes_list = list(classes_)
						if pos_label in classes_list:
							pos_idx = classes_list.index(pos_label)
					return float(roc_auc_score(y, p[:, pos_idx]))
				return float(roc_auc_score(y, p, multi_class="ovr", average="weighted"))
			except Exception:
				return float("nan")

		raise ValueError("metric must be one of: 'roc_auc', 'recall', 'f1', 'accuracy'")

	@staticmethod
	def plot_roc_curves(
		models: dict[str, Any],
		X: Any,
		y: Any,
		*,
		title: str = "ROC Curves",
		best_only: bool = False,
		select_by: Literal["roc_auc", "recall", "f1", "accuracy"] = "roc_auc",
		average: Literal["binary", "macro", "weighted"] | None = None,
		pos_label: Any = 1,
	):
		"""Plot ROC curves for multiple models on a single chart.

		Args:
			models: dict of name -> fitted estimator or TrainResult.
			X, y: Evaluation data (typically test split).
			title: Figure title.
			best_only: If True, plots only the best model (by select_by on (X, y)).
			select_by: Metric to choose the best model.
			average: Used only for select_by when metric is recall/f1.
			pos_label: Positive label used for binary select_by and ROC-AUC selection.

		Returns:
			Matplotlib Figure.
		"""
		try:
			import matplotlib.pyplot as plt
		except Exception as exc:  # pragma: no cover
			raise ImportError("Plotting requires matplotlib.") from exc

		models = ModelTrainer._as_estimator_dict(models)
		y_arr = np.asarray(y)
		if len(np.unique(y_arr)) != 2:
			raise ValueError("plot_roc_curves supports binary classification only")

		items = list(models.items())
		if best_only:
			scored = []
			for name, est in items:
				s = ModelTrainer._score_single_model(
					est,
					X,
					y,
					metric=select_by,
					average=average,
					pos_label=pos_label,
				)
				scored.append((name, est, s))
			# Prefer finite scores; otherwise fall back to first.
			finite = [(n, e, s) for (n, e, s) in scored if np.isfinite(s)]
			best_name, best_est, _ = max(finite, key=lambda t: t[2]) if finite else scored[0]
			items = [(best_name, best_est)]
			title = f"{title} ({'best ' + str(select_by)})"

		fig, ax = plt.subplots(figsize=(7, 5))
		for name, est in items:
			estimator = ModelTrainer._unwrap_estimator(est)
			RocCurveDisplay.from_estimator(estimator, X, y, ax=ax, name=name)
		ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
		ax.set_title(title)
		fig.tight_layout()
		return fig

	@staticmethod
	def plot_confusion_matrices(
		models: dict[str, Any],
		X: Any,
		y: Any,
		*,
		title: str = "Confusion Matrices",
		best_only: bool = False,
		select_by: Literal["roc_auc", "recall", "f1", "accuracy"] = "roc_auc",
		average: Literal["binary", "macro", "weighted"] | None = None,
		pos_label: Any = 1,
		labels: list[Any] | None = None,
		normalize: Literal["true", "pred", "all"] | None = None,
		cmap: str = "Blues",
	):
		"""Plot confusion matrices for multiple models.

		Args:
			models: dict of name -> fitted estimator or TrainResult.
			X, y: Evaluation data (typically test split).
			title: Figure title.
			best_only: If True, plots only the best model (by select_by on (X, y)).
			select_by: Metric to choose the best model.
			average: Used only for select_by when metric is recall/f1.
			pos_label: Positive label used for binary select_by.
			labels: Optional list of labels to index the matrix.
			normalize: Normalize confusion matrix over the true (rows), predicted (cols), or all.
			cmap: Matplotlib colormap name.

		Returns:
			Matplotlib Figure.
		"""
		try:
			import matplotlib.pyplot as plt
		except Exception as exc:  # pragma: no cover
			raise ImportError("Plotting requires matplotlib.") from exc

		models = ModelTrainer._as_estimator_dict(models)
		items = list(models.items())
		if not items:
			raise ValueError("models is empty")

		if best_only:
			scored = []
			for name, est in items:
				s = ModelTrainer._score_single_model(
					est,
					X,
					y,
					metric=select_by,
					average=average,
					pos_label=pos_label,
				)
				scored.append((name, est, s))
			finite = [(n, e, s) for (n, e, s) in scored if np.isfinite(s)]
			best_name, best_est, _ = max(finite, key=lambda t: t[2]) if finite else scored[0]
			items = [(best_name, best_est)]
			title = f"{title} ({'best ' + str(select_by)})"

		n = len(items)
		fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
		if n == 1:
			axes = [axes]

		for ax, (name, est) in zip(axes, items):
			estimator = ModelTrainer._unwrap_estimator(est)
			y_pred = estimator.predict(X)
			ConfusionMatrixDisplay.from_predictions(
				y_true=y,
				y_pred=y_pred,
				labels=labels,
				normalize=normalize,
				cmap=cmap,
				ax=ax,
				colorbar=False,
			)
			# Remove inner cell grid lines (minor grid/ticks) for cleaner boxes.
			ax.grid(False, which="both")
			ax.set_xticks([], minor=True)
			ax.set_yticks([], minor=True)
			ax.tick_params(which="minor", bottom=False, left=False)
			ax.set_title(name)

		fig.suptitle(title, y=1.02)
		fig.tight_layout()
		return fig

