
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from src.functions.model import ModelTrainer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost


TaskType = Literal["regression", "classification"]
ModelType = Literal["linear", "logistic", "random_forest", "xgboost"]
ScalerType = Literal["standard", "minmax", "robust", "none"]
EncodingType = Literal["onehot", "ordinal"]
TargetTransformType = Literal["none", "log1p"]


class SkewLogTransformer(BaseEstimator, TransformerMixin):
	"""Log1p-transform skewed numeric features (fit decides which to transform).

	Purpose: include your feature_eng.py log-transform behavior inside an sklearn Pipeline.
	Learns skewness on the training set only (prevents leakage).

	Notes:
	- Uses `np.log1p`.
	- If a feature has negative values, it learns a per-feature shift so min becomes 0.
	"""

	def __init__(
		self,
		*,
		feature_names: list[str] | None = None,
		columns: list[str] | None = None,
		skew_threshold: float = 1.0,
	):
		self.feature_names = feature_names
		self.columns = columns
		self.skew_threshold = float(skew_threshold)

	def fit(self, X: Any, y: Any = None):
		X_df = self._to_df(X)

		cols = self.columns if self.columns is not None else list(X_df.columns)
		missing = [c for c in cols if c not in X_df.columns]
		if missing:
			raise ValueError(f"SkewLogTransformer: columns not in input: {missing}")

		skews = X_df[cols].apply(lambda s: pd.Series(s).skew(skipna=True))
		self.transformed_columns_ = [
			c for c in cols if pd.notna(skews.get(c)) and abs(float(skews[c])) >= self.skew_threshold
		]

		# shift so min is >= 0 before log1p
		shifts: dict[str, float] = {}
		for c in self.transformed_columns_:
			min_val = pd.to_numeric(X_df[c], errors="coerce").min(skipna=True)
			if pd.isna(min_val):
				shifts[c] = 0.0
			else:
				min_val_f = float(min_val)
				shifts[c] = float(-min_val_f) if min_val_f < 0 else 0.0
		self.shifts_ = shifts
		return self

	def transform(self, X: Any):
		X_df = self._to_df(X).copy()

		for c in getattr(self, "transformed_columns_", []):
			shift = float(self.shifts_.get(c, 0.0))
			vals = pd.to_numeric(X_df[c], errors="coerce")
			X_df[c] = np.log1p(vals + shift)

		return X_df.to_numpy()

	def _to_df(self, X: Any) -> pd.DataFrame:
		if isinstance(X, pd.DataFrame):
			return X
		X_arr = np.asarray(X)
		if X_arr.ndim != 2:
			raise ValueError("SkewLogTransformer expects 2D input")
		cols = self.feature_names if self.feature_names is not None else [f"f{i}" for i in range(X_arr.shape[1])]
		return pd.DataFrame(X_arr, columns=cols)


class MappingOrdinalEncoder(BaseEstimator, TransformerMixin):
	"""Ordinal-encode categorical columns using explicit mapping dicts.

	Supports:
	- single-column mapping: {"Diesel": 2, "Petrol": 1}
	- multi-column mapping: {"fuel_type": {"Diesel": 2, ...}, "transmission": {...}}
	"""

	def __init__(
		self,
		*,
		columns: list[str],
		mapping: dict[str, int] | dict[str, dict[str, int]],
		unknown_value: int = -1,
	):
		self.columns = columns
		self.mapping = mapping
		self.unknown_value = int(unknown_value)

	def fit(self, X: Any, y: Any = None):
		if not self.columns:
			raise ValueError("MappingOrdinalEncoder: columns must be provided")

		# normalize mapping to dict[col] -> dict[value] -> int
		if self._is_single_mapping(self.mapping):
			if len(self.columns) != 1:
				raise ValueError("MappingOrdinalEncoder: single mapping provided but columns has != 1 column")
			self.mapping_ = {self.columns[0]: self.mapping}  # type: ignore[assignment]
		else:
			self.mapping_ = self.mapping  # type: ignore[assignment]

		for c in self.columns:
			if c not in self.mapping_:
				raise ValueError(f"MappingOrdinalEncoder: missing mapping for column '{c}'")
		return self

	def transform(self, X: Any):
		X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=self.columns)
		out = pd.DataFrame(index=X_df.index)

		for c in self.columns:
			m = self.mapping_[c]
			encoded = X_df[c].map(m)
			out[c] = encoded.fillna(self.unknown_value).astype(int)

		return out.to_numpy()

	@staticmethod
	def _is_single_mapping(obj: Any) -> bool:
		return isinstance(obj, dict) and all(not isinstance(v, dict) for v in obj.values())


@dataclass(frozen=True)
class PipelineBundle:
	"""Wrapper for saving a pipeline + optional metadata."""

	pipeline: Pipeline
	feature_columns: list[str] | None = None


@dataclass
class PipelineTrainResult:
	"""Training result for a full preprocessing+model pipeline."""

	name: str
	best_estimator: Pipeline
	best_params: dict[str, Any]
	best_score: float
	grid_search: GridSearchCV


class ModelPipeline:
	"""Build preprocessing+model pipelines (incl. skew-log transform) and save/load with joblib."""

	@staticmethod
	def _unwrap_pipeline(pipeline_or_result: Any) -> Pipeline:
		"""Accept Pipeline, PipelineTrainResult, or GridSearchCV and return a fitted Pipeline."""
		if isinstance(pipeline_or_result, PipelineTrainResult):
			return pipeline_or_result.best_estimator
		if isinstance(pipeline_or_result, GridSearchCV):
			return pipeline_or_result.best_estimator_
		if isinstance(pipeline_or_result, Pipeline):
			return pipeline_or_result
		# allow callers to pass bundles or objects with `.pipeline`
		if hasattr(pipeline_or_result, "pipeline"):
			return pipeline_or_result.pipeline  # type: ignore[return-value]
		raise TypeError("pipeline_or_result must be a Pipeline, PipelineTrainResult, GridSearchCV, or PipelineBundle")

	@staticmethod
	def evaluate_regression_pipe(
		pipeline_or_result: PipelineTrainResult | GridSearchCV | PipelineBundle | Pipeline,
		X_train: Any,
		y_train: Any,
		X_test: Any,
		y_test: Any,
		*,
		decimals: int = 2,
	) -> pd.DataFrame:
		"""Create a train/test evaluation table for a fitted regression pipeline.

		Mirrors `ModelTrainer.evaluate_regression` but works directly with the
		preprocessing+model pipeline objects in this module.
		"""
		pipe = ModelPipeline._unwrap_pipeline(pipeline_or_result)
		if not hasattr(pipe, "predict"):
			raise TypeError("pipeline must implement predict()")

		yhat_train = pipe.predict(X_train)
		yhat_test = pipe.predict(X_test)

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
	def evaluate_regression_all_pipe(
		results: dict[str, PipelineTrainResult] | dict[str, Any],
		X_train: Any,
		y_train: Any,
		X_test: Any,
		y_test: Any,
		*,
		sort_by: Literal["test_rmse", "test_mae", "test_mape", "test_r2"] | None = "test_rmse",
		decimals: int = 2,
	) -> pd.DataFrame:
		"""Evaluate every pipeline returned by `train_regression_all`.

		Returns a single DataFrame with a MultiIndex (model, split).
		"""
		if not results:
			raise ValueError("results is empty")

	@staticmethod
	def evaluate_classification_pipe(
		pipeline_or_result: PipelineTrainResult | GridSearchCV | PipelineBundle | Pipeline,
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
	) -> pd.DataFrame:
		"""Create a train/test evaluation table for a fitted classification pipeline."""
		pipe = ModelPipeline._unwrap_pipeline(pipeline_or_result)
		return ModelTrainer.evaluate_classification(
			pipe,
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

	@staticmethod
	def evaluate_classification_all_pipe(
		results: dict[str, PipelineTrainResult] | dict[str, Any],
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
		"""Evaluate every classification pipeline returned by `train_classification_all`.

		Accepts either:
		- dict[name -> PipelineTrainResult]
		- dict[name -> fitted Pipeline]
		"""
		if not results:
			raise ValueError("results is empty")

		pipes: dict[str, Any] = {}
		for name, obj in results.items():
			if isinstance(obj, PipelineTrainResult):
				pipes[name] = obj.best_estimator
			elif isinstance(obj, GridSearchCV):
				pipes[name] = obj.best_estimator_
			elif isinstance(obj, Pipeline):
				pipes[name] = obj
			elif hasattr(obj, "best_estimator"):
				pipes[name] = obj.best_estimator  # type: ignore[assignment]
			else:
				pipes[name] = obj

		return ModelTrainer.evaluate_classification_all(
			results=pipes,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			average=average,
			metric_set=metric_set,
			metrics=metrics,
			pos_label=pos_label,
			decimals=decimals,
			fit_metric=fit_metric,
			overfit_gap=overfit_gap,
			overfit_train_min=overfit_train_min,
			underfit_max=underfit_max,
			add_gap_column=add_gap_column,
			sort_by=sort_by,
		)

		frames: list[pd.DataFrame] = []
		for name, res in results.items():
			metrics = ModelPipeline.evaluate_regression_pipe(
				pipeline_or_result=res,
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
	def plot_actual_vs_predicted_all(
		results: dict[str, PipelineTrainResult] | dict[str, Any],
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
		"""Plot actual vs predicted for all regression pipelines.

		This mirrors `ModelTrainer.plot_actual_vs_predicted_all`, but uses the fitted
		preprocessing+model pipelines from this module.
		
		Returns:
			- Plotly Figure if use_plotly=True
			- Matplotlib Figure if use_plotly=False
		"""
		if not results:
			raise ValueError("results is empty")

		y_test_arr = np.asarray(y_test)
		n_models = len(results)

		# Collect predictions and global min/max for consistent axes/diagonal
		predictions: dict[str, np.ndarray] = {}
		min_val = float("inf")
		max_val = float("-inf")
		for name, res in results.items():
			pipe = ModelPipeline._unwrap_pipeline(res)
			y_pred = np.asarray(pipe.predict(X_test))
			predictions[str(name)] = y_pred
			min_val = min(min_val, float(np.nanmin(y_test_arr)), float(np.nanmin(y_pred)))
			max_val = max(max_val, float(np.nanmax(y_test_arr)), float(np.nanmax(y_pred)))

		if title is None:
			title = "Actual vs Predicted - All Models"

		if use_plotly:
			from plotly.subplots import make_subplots
			import plotly.graph_objects as go

			fig = make_subplots(
				rows=1,
				cols=n_models,
				subplot_titles=[k.replace("_", " ").title() for k in predictions.keys()],
				horizontal_spacing=0.08,
			)

			for idx, (name, y_pred) in enumerate(predictions.items(), start=1):
				fig.add_trace(
					go.Scatter(
						x=y_test_arr,
						y=y_pred,
						mode="markers",
						marker=dict(size=6, opacity=0.6),
						showlegend=False,
					),
					row=1,
					col=idx,
				)

				if add_diagonal:
					fig.add_shape(
						type="line",
						x0=min_val,
						y0=min_val,
						x1=max_val,
						y1=max_val,
						line=dict(color="red", dash="dash", width=2),
						row=1,
						col=idx,
					)

				fig.update_xaxes(title_text=xlabel, row=1, col=idx)
				if idx == 1:
					fig.update_yaxes(title_text=ylabel, row=1, col=idx)

				# keep axis ranges aligned across subplots
				fig.update_xaxes(range=[min_val, max_val], row=1, col=idx)
				fig.update_yaxes(range=[min_val, max_val], row=1, col=idx)

			fig.update_layout(
				title_text=title,
				height=500,
				hovermode="closest",
				showlegend=False,
			)
			return fig

		import matplotlib.pyplot as plt

		fig, axes = plt.subplots(1, n_models, figsize=figsize)
		if n_models == 1:
			axes = [axes]

		for ax, (name, y_pred) in zip(axes, predictions.items()):
			ax.scatter(y_test_arr, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
			if add_diagonal:
				ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.7)
			ax.set_xlabel(xlabel)
			ax.set_ylabel(ylabel)
			ax.set_title(name.replace("_", " ").title())
			ax.grid(True, alpha=0.3)
			ax.set_xlim(min_val, max_val)
			ax.set_ylim(min_val, max_val)

		fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
		fig.tight_layout()
		return fig

	@staticmethod
	def _default_model_kwargs(*, model_type: ModelType, random_state: int, n_jobs: int) -> dict[str, Any]:
		# Only include kwargs supported by each estimator.
		if model_type in ("random_forest", "xgboost"):
			return {"random_state": random_state, "n_jobs": n_jobs}
		return {}

	@staticmethod
	def _make_cv(*, cv: int, random_state: int, stratified: bool):
		"""Create a CV splitter.

		For classification, stratified folds help preserve class balance.
		"""
		if stratified:
			return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
		return KFold(n_splits=cv, shuffle=True, random_state=random_state)

	@staticmethod
	def _grid_search_pipeline(
		*,
		name: str,
		pipeline: Pipeline,
		param_grid: dict[str, list[Any]],
		X_train: Any,
		y_train: Any,
		cv: Any,
		scoring: str | None,
		n_jobs: int,
		verbose: int,
		refit: bool,
	) -> PipelineTrainResult:
		gs = GridSearchCV(
			estimator=pipeline,
			param_grid=param_grid or {},
			cv=cv,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)
		gs.fit(X_train, y_train)
		return PipelineTrainResult(
			name=name,
			best_estimator=gs.best_estimator_,
			best_params=dict(gs.best_params_ or {}),
			best_score=float(gs.best_score_),
			grid_search=gs,
		)

	@staticmethod
	def select_best(results: dict[str, PipelineTrainResult]) -> PipelineTrainResult:
		"""Select best pipeline by CV score (higher is better in sklearn)."""
		if not results:
			raise ValueError("results is empty")
		return max(results.values(), key=lambda r: r.best_score)

	@staticmethod
	def train_regression_all(
		*,
		X_train: Any,
		y_train: Any,
		feature_engineer: TransformerMixin | None = None,
		model_types: tuple[ModelType, ...] = ("linear", "random_forest", "xgboost"),
		numeric_features: list[str] | None = None,
		categorical_features: list[str] | None = None,
		log_columns: list[str] | None = None,
		skew_threshold: float = 1.0,
		target_transform: TargetTransformType = "none",
		encoding: EncodingType = "onehot",
		ordinal_mapping: dict[str, int] | dict[str, dict[str, int]] | None = None,
		impute_numeric: Literal["median", "mean"] = "median",
		impute_categorical: Literal["most_frequent"] = "most_frequent",
		scaler: ScalerType = "standard",
		cv: int = 5,
		scoring: str | None = "neg_root_mean_squared_error",
		random_state: int = 42,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
		model_kwargs: dict[str, Any] | None = None,
	) -> dict[str, PipelineTrainResult]:
		"""Train multiple regression pipelines (like `model.py` train_regression).

		Uses the same preprocessing config for each model, so results are comparable.
		Returns: dict[model_type] -> PipelineTrainResult
		"""
		results: dict[str, PipelineTrainResult] = {}
		for mt in model_types:
			kwargs = dict(ModelPipeline._default_model_kwargs(model_type=mt, random_state=random_state, n_jobs=n_jobs))
			if model_kwargs:
				kwargs.update(model_kwargs)

			pipe = ModelPipeline.build(
				task="regression",
				model_type=mt,
				feature_engineer=feature_engineer,
				numeric_features=numeric_features,
				categorical_features=categorical_features,
				log_columns=log_columns,
				skew_threshold=skew_threshold,
				target_transform=target_transform,
				encoding=encoding,
				ordinal_mapping=ordinal_mapping,
				impute_numeric=impute_numeric,
				impute_categorical=impute_categorical,
				scaler=scaler,
				model_kwargs=kwargs,
			)

			grid = ModelPipeline.default_param_grid(
				task="regression",
				model_type=mt,
				target_transform=target_transform,
			)

			results[mt] = ModelPipeline._grid_search_pipeline(
				name=mt,
				pipeline=pipe,
				param_grid=grid,
				X_train=X_train,
				y_train=y_train,
				cv=cv,
				scoring=scoring,
				n_jobs=n_jobs,
				verbose=verbose,
				refit=refit,
			)

		return results

	@staticmethod
	def train_classification_all(
		*,
		X_train: Any,
		y_train: Any,
		feature_engineer: TransformerMixin | None = None,
		model_types: tuple[ModelType, ...] = ("logistic", "random_forest", "xgboost"),
		numeric_features: list[str] | None = None,
		categorical_features: list[str] | None = None,
		log_columns: list[str] | None = None,
		skew_threshold: float = 1.0,
		encoding: EncodingType = "onehot",
		ordinal_mapping: dict[str, int] | dict[str, dict[str, int]] | None = None,
		impute_numeric: Literal["median", "mean"] = "median",
		impute_categorical: Literal["most_frequent"] = "most_frequent",
		scaler: ScalerType = "standard",
		cv: int = 5,
		scoring: str | None = "accuracy",
		random_state: int = 42,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
		stratified_cv: bool = True,
		model_kwargs: dict[str, Any] | None = None,
	) -> dict[str, PipelineTrainResult]:
		"""Train multiple classification pipelines (like `model.py` train_classification).

		If `stratified_cv=True`, uses StratifiedKFold for GridSearchCV.
		If False, uses plain KFold.
		"""
		cv_obj = ModelPipeline._make_cv(cv=cv, random_state=random_state, stratified=stratified_cv)
		results: dict[str, PipelineTrainResult] = {}
		for mt in model_types:
			kwargs = dict(ModelPipeline._default_model_kwargs(model_type=mt, random_state=random_state, n_jobs=n_jobs))
			if model_kwargs:
				kwargs.update(model_kwargs)

			pipe = ModelPipeline.build(
				task="classification",
				model_type=mt,
				feature_engineer=feature_engineer,
				numeric_features=numeric_features,
				categorical_features=categorical_features,
				log_columns=log_columns,
				skew_threshold=skew_threshold,
				encoding=encoding,
				ordinal_mapping=ordinal_mapping,
				impute_numeric=impute_numeric,
				impute_categorical=impute_categorical,
				scaler=scaler,
				model_kwargs=kwargs,
			)

			grid = ModelPipeline.default_param_grid(
				task="classification",
				model_type=mt,
				# classification doesn't use target_transform
				target_transform="none",
			)

			results[mt] = ModelPipeline._grid_search_pipeline(
				name=mt,
				pipeline=pipe,
				param_grid=grid,
				X_train=X_train,
				y_train=y_train,
				cv=cv_obj,
				scoring=scoring,
				n_jobs=n_jobs,
				verbose=verbose,
				refit=refit,
			)

		return results

	@staticmethod
	def build(
		*,
		task: TaskType,
		model_type: ModelType,
		feature_engineer: TransformerMixin | None = None,
		numeric_features: list[str] | None = None,
		categorical_features: list[str] | None = None,
		# log transform
		log_columns: list[str] | None = None,
		skew_threshold: float = 1.0,
		# target transform (regression only)
		target_transform: TargetTransformType = "none",
		# categorical encoding
		encoding: EncodingType = "onehot",
		ordinal_mapping: dict[str, int] | dict[str, dict[str, int]] | None = None,
		# imputers + scaling
		impute_numeric: Literal["median", "mean"] = "median",
		impute_categorical: Literal["most_frequent"] = "most_frequent",
		scaler: ScalerType = "standard",
		# model kwargs
		model_kwargs: dict[str, Any] | None = None,
	) -> Pipeline:
		"""Create a Pipeline with preprocessing + model.

		You must pass `numeric_features` / `categorical_features` explicitly.
		(This keeps behavior predictable and avoids guessing dtypes.)
		"""
		model_kwargs = model_kwargs or {}

		# scaler
		if scaler == "standard":
			scaler_step: Any = StandardScaler()
		elif scaler == "minmax":
			scaler_step = MinMaxScaler()
		elif scaler == "robust":
			scaler_step = RobustScaler()
		elif scaler == "none":
			scaler_step = "passthrough"
		else:
			raise ValueError(f"Unknown scaler: {scaler}")

		# OneHotEncoder sklearn compatibility
		try:
			ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
		except TypeError:  # pragma: no cover
			ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

		if numeric_features is None:
			numeric_features = []
		if categorical_features is None:
			categorical_features = []

		num_pipe = Pipeline(
			steps=[
				(
					"log",
					SkewLogTransformer(
						feature_names=numeric_features,
						columns=log_columns,
						skew_threshold=skew_threshold,
					),
				),
				("imputer", SimpleImputer(strategy=impute_numeric)),
				("scaler", scaler_step),
			]
		)

		if encoding == "onehot":
			cat_pipe = Pipeline(
				steps=[
					("imputer", SimpleImputer(strategy=impute_categorical)),
					("encoder", ohe),
				]
			)
		elif encoding == "ordinal":
			if ordinal_mapping is None:
				raise ValueError("encoding='ordinal' requires ordinal_mapping")
			if not categorical_features:
				raise ValueError("encoding='ordinal' requires categorical_features")
			cat_pipe = Pipeline(
				steps=[
					("imputer", SimpleImputer(strategy=impute_categorical)),
					(
						"encoder",
						MappingOrdinalEncoder(columns=categorical_features, mapping=ordinal_mapping),
					),
				]
			)
		else:
			raise ValueError(f"Unknown encoding: {encoding}")

		preprocess = ColumnTransformer(
			transformers=[
				("num", num_pipe, numeric_features),
				("cat", cat_pipe, categorical_features),
			],
			remainder="drop",
		)

		base_model = ModelPipeline._make_model(task=task, model_type=model_type, model_kwargs=model_kwargs)

		# Target transform (regression only)
		if target_transform != "none":
			if task != "regression":
				raise ValueError("target_transform is supported for regression only")
			if target_transform == "log1p":
				# Requires y >= 0. If y can be negative, handle shifting before training.
				model: Any = TransformedTargetRegressor(
					regressor=base_model,
					func=np.log1p,
					inverse_func=np.expm1,
				)
			else:
				raise ValueError(f"Unknown target_transform: {target_transform}")
		else:
			model = base_model

		steps: list[tuple[str, Any]] = []
		if feature_engineer is not None:
			steps.append(("feature_engineer", feature_engineer))
		steps.append(("preprocess", preprocess))
		steps.append(("model", model))
		return Pipeline(steps=steps)

	@staticmethod
	def default_param_grid(
		*,
		task: TaskType,
		model_type: ModelType,
		target_transform: TargetTransformType = "none",
	) -> dict[str, list[Any]]:
		"""Default GridSearchCV grids (names match pipeline step names).

		If `target_transform != "none"` for regression, the underlying estimator parameters live under:
		- `model__regressor__<param>`
		Otherwise:
		- `model__<param>`
		"""
		prefix = "model__regressor__" if (task == "regression" and target_transform != "none") else "model__"
		# Match the hyperparameter ranges defined in `src/functions/model.py`.
		if task == "regression" and model_type == "linear":
			return {
				f"{prefix}fit_intercept": [True, False],
				f"{prefix}positive": [False, True],
			}

		if task == "classification" and model_type == "logistic":
			return {
				f"{prefix}C": [0.01, 0.1, 1.0, 10.0],
				f"{prefix}penalty": ["l2"],
				f"{prefix}solver": ["lbfgs"],
				f"{prefix}class_weight": [None, "balanced"],
				f"{prefix}max_iter": [1000],
			}

		if model_type == "random_forest" and task == "regression":
			return {
				f"{prefix}n_estimators": [200, 500],
				f"{prefix}max_depth": [None, 5, 10, 20],
				f"{prefix}min_samples_split": [2, 5, 10],
				f"{prefix}min_samples_leaf": [1, 2, 4],
				f"{prefix}max_features": ["sqrt", "log2", None],
			}

		if model_type == "random_forest" and task == "classification":
			return {
				f"{prefix}n_estimators": [200, 500],
				f"{prefix}max_depth": [None, 5, 10, 20],
				f"{prefix}min_samples_split": [2, 5, 10],
				f"{prefix}min_samples_leaf": [1, 2, 4],
				f"{prefix}max_features": ["sqrt", "log2", None],
				f"{prefix}class_weight": [None, "balanced"],
			}

		if model_type == "xgboost" and task == "regression":
			return {
				f"{prefix}n_estimators": [200, 500],
				f"{prefix}learning_rate": [0.01, 0.05, 0.1],
				f"{prefix}max_depth": [3, 5, 7],
				f"{prefix}subsample": [0.8, 1.0],
				f"{prefix}colsample_bytree": [0.8, 1.0],
				f"{prefix}reg_lambda": [1.0, 10.0],
				f"{prefix}min_child_weight": [1, 5],
			}

		if model_type == "xgboost" and task == "classification":
			return {
				f"{prefix}n_estimators": [200, 500],
				f"{prefix}learning_rate": [0.01, 0.05, 0.1],
				f"{prefix}max_depth": [3, 5, 7],
				f"{prefix}subsample": [0.8, 1.0],
				f"{prefix}colsample_bytree": [0.8, 1.0],
				f"{prefix}reg_lambda": [1.0, 10.0],
				f"{prefix}min_child_weight": [1, 5],
				f"{prefix}scale_pos_weight": [1.0, 5.0, 10.0],
			}

		return {}

	@staticmethod
	def train_with_gridsearch(
		*,
		pipeline: Pipeline,
		X_train: Any,
		y_train: Any,
		param_grid: dict[str, list[Any]] | None = None,
		cv: int = 5,
		scoring: str | None = None,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
	) -> GridSearchCV:
		"""Fit preprocessing+model pipeline with GridSearchCV."""
		gs = GridSearchCV(
			estimator=pipeline,
			param_grid=param_grid or {},
			cv=cv,
			scoring=scoring,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
		)
		gs.fit(X_train, y_train)
		return gs

	@staticmethod
	def save(bundle_or_pipeline: PipelineBundle | Pipeline, path: str) -> None:
		"""Save pipeline (and optional metadata) to disk."""
		joblib.dump(bundle_or_pipeline, path)

	@staticmethod
	def load(path: str) -> PipelineBundle | Pipeline:
		"""Load a saved pipeline (or PipelineBundle)."""
		return joblib.load(path)

	@staticmethod
	def load_pipeline(path: str) -> Pipeline:
		"""Load a saved pipeline (works for both Pipeline and PipelineBundle)."""
		obj = ModelPipeline.load(path)
		return obj.pipeline if hasattr(obj, "pipeline") else obj

	@staticmethod
	def predict(pipeline_or_bundle: PipelineBundle | Pipeline, X: Any) -> np.ndarray:
		"""Predict using a fitted pipeline.

		This is intended for regression (returns numeric predictions). For classification
		probabilities, call `pipeline.predict_proba(X)` directly.
		"""
		pipe = pipeline_or_bundle.pipeline if hasattr(pipeline_or_bundle, "pipeline") else pipeline_or_bundle
		return pipe.predict(X)

	@staticmethod
	def predict_from_path(path: str, X: Any) -> np.ndarray:
		"""Load pipeline from disk and predict in one step."""
		pipe = ModelPipeline.load_pipeline(path)
		return pipe.predict(X)

	@staticmethod
	def _make_model(*, task: TaskType, model_type: ModelType, model_kwargs: dict[str, Any]) -> Any:
		if task == "regression":
			if model_type == "linear":
				return LinearRegression(**model_kwargs)
			if model_type == "random_forest":
				return RandomForestRegressor(random_state=model_kwargs.pop("random_state", 42), **model_kwargs)
			if model_type == "xgboost":
				return xgboost.XGBRegressor(
					random_state=model_kwargs.pop("random_state", 42),
					n_jobs=model_kwargs.pop("n_jobs", -1),
					**model_kwargs,
				)
			raise ValueError(f"Unknown regression model_type: {model_type}")

		if task == "classification":
			if model_type == "logistic":
				return LogisticRegression(max_iter=model_kwargs.pop("max_iter", 2000), **model_kwargs)
			if model_type == "random_forest":
				return RandomForestClassifier(random_state=model_kwargs.pop("random_state", 42), **model_kwargs)
			if model_type == "xgboost":
				return xgboost.XGBClassifier(
					random_state=model_kwargs.pop("random_state", 42),
					n_jobs=model_kwargs.pop("n_jobs", -1),
					eval_metric=model_kwargs.pop("eval_metric", "logloss"),
					**model_kwargs,
				)
			raise ValueError(f"Unknown classification model_type: {model_type}")

		raise ValueError(f"Unknown task: {task}")

