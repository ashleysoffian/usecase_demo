
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from functions.feature_eng import Feature_Engineering as FE
from functions.model import ModelTrainer as MT
from functions.model_pipeline import ModelPipeline


ScalerType = Literal["standard", "minmax", "robust", "none"]


@dataclass(frozen=True)
class ClassificationPipelineConfig:
	"""Configuration for an end-to-end classification workflow.

	This is intentionally aligned with the manual workflow used in
	`classification.ipynb` (encode -> split -> scale -> train -> evaluate).
	"""

	target: str
	drop_columns: list[str] | None = None
	categorical_features: list[str] | None = None
	encoding: FE.EncodingType = "onehot"
	scaler: ScalerType = "standard"

	test_size: float = 0.2
	stratify: bool = True
	random_state: int = 42

	# Training
	cv: int = 5
	scoring: str = "recall"
	stratified_cv: bool = True
	n_jobs: int = -1
	verbose: int = 0
	refit: bool = True

	# Evaluation
	metric_set: Literal["default", "imbalanced", "balanced", "all"] = "imbalanced"
	metrics: list[str] | tuple[str, ...] | None = None
	pos_label: Any = 1
	decimals: int = 3

	# Fit-status labeling
	fit_metric: str = "recall"
	overfit_gap: float = 0.10
	overfit_train_min: float = 0.85
	underfit_max: float = 0.60

	# Raw-data pipeline options (for demo/prediction on raw inputs)
	use_raw_pipeline: bool = False
	# If True, also trains and returns fitted sklearn Pipelines (best_pipeline/pipelines)
	# even when use_raw_pipeline=False. Note: this adds extra training time.
	return_best_pipeline: bool = False
	# if True, lowercases column names and applies the default renaming used in your notebook
	auto_rename_columns: bool = True
	# if True, creates engineered features: mechanical_power, wear_stress, temp_delta
	add_engineered_features: bool = True
	# if True, applies log1p transform to rotational_speed before building interactions
	log_rotational_speed: bool = True
	# columns to drop if present (e.g., leakage features)
	extra_drop_columns: list[str] | None = None

	# Features used by the preprocessing pipeline (after feature engineering step)
	# If None, defaults to the machine-failure feature set.
	pipeline_numeric_features: list[str] | None = None
	pipeline_categorical_features: list[str] | None = None

	# Pipeline training models (sklearn Pipeline + GridSearchCV)
	pipeline_model_types: tuple[Literal["logistic", "random_forest", "xgboost"], ...] = (
		"logistic",
		"random_forest",
	)
	pipeline_encoding: FE.EncodingType = "onehot"
	pipeline_skew_threshold: float = 1.0

	# Plotting
	make_plots: bool = True
	plot_select_by: Literal["roc_auc", "recall", "f1", "accuracy"] = "roc_auc"
	confusion_normalize: Literal["true", "pred", "all"] | None = None



@dataclass
class ClassificationPipelineResult:
	"""Outputs from running the classification pipeline."""

	config: ClassificationPipelineConfig
	X: pd.DataFrame
	y: pd.Series
	X_train: pd.DataFrame
	X_test: pd.DataFrame
	y_train: pd.Series
	y_test: pd.Series
	X_train_s: pd.DataFrame
	X_test_s: pd.DataFrame
	models: dict[str, Any]
	best_model: Any
	metrics_table: pd.DataFrame
	# If use_raw_pipeline=True, these are fitted sklearn Pipelines you can save/use later.
	pipelines: dict[str, Any] | None = None
	best_pipeline: Any | None = None
	# If use_raw_pipeline=True, these are the feature-engineered versions (for inspection/debugging).
	X_fe: pd.DataFrame | None = None
	X_train_fe: pd.DataFrame | None = None
	X_test_fe: pd.DataFrame | None = None

	# Plots (generated when config.make_plots=True)
	roc_curves_all: Any | None = None
	roc_curves_best: Any | None = None
	confusion_matrices_all: Any | None = None
	confusion_matrices_best: Any | None = None


class MachineFailureFeatureEngineer(BaseEstimator, TransformerMixin):
	"""Feature engineering for the machine failure dataset.

	Designed so the trained sklearn Pipeline can accept *raw* data at prediction time.
	"""

	DEFAULT_RENAME = {
		"product id": "product_id",
		"air temperature [k]": "air_temp",
		"process temperature [k]": "process_temp",
		"rotational speed [rpm]": "rotational_speed",
		"torque [nm]": "torque",
		"tool wear [min]": "tool_wear",
		"machine failure": "machine_failure",
	}

	DEFAULT_DROP = ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]

	def __init__(
		self,
		*,
		auto_rename_columns: bool = True,
		add_engineered_features: bool = True,
		log_rotational_speed: bool = True,
		extra_drop_columns: list[str] | None = None,
	):
		self.auto_rename_columns = bool(auto_rename_columns)
		self.add_engineered_features = bool(add_engineered_features)
		self.log_rotational_speed = bool(log_rotational_speed)
		self.extra_drop_columns = extra_drop_columns

	def fit(self, X: Any, y: Any = None):
		return self

	def transform(self, X: Any):
		if not isinstance(X, pd.DataFrame):
			raise TypeError("MachineFailureFeatureEngineer expects a pandas DataFrame")

		out = X.copy()
		# Normalize columns to match notebook behavior
		out.columns = out.columns.astype(str)
		out.columns = out.columns.str.lower()
		if self.auto_rename_columns:
			# mapping keys are expected lower-case
			out = out.rename(columns=self.DEFAULT_RENAME)

		# Drop leakage + IDs if present
		drop_cols = list(self.DEFAULT_DROP)
		if self.extra_drop_columns:
			drop_cols.extend(list(self.extra_drop_columns))
		drop_cols = [c for c in drop_cols if c in out.columns]
		if drop_cols:
			out = out.drop(columns=drop_cols)

		# Log transform rotational_speed (optional)
		if self.log_rotational_speed and "rotational_speed" in out.columns:
			s = pd.to_numeric(out["rotational_speed"], errors="coerce")
			min_val = s.min(skipna=True)
			shift = float(-min_val) if pd.notna(min_val) and float(min_val) < 0 else 0.0
			out["rotational_speed"] = np.log1p(s + shift)

		if self.add_engineered_features:
			# mechanical_power = torque * rotational_speed
			if "torque" in out.columns and "rotational_speed" in out.columns:
				out["mechanical_power"] = pd.to_numeric(out["torque"], errors="coerce") * pd.to_numeric(
					out["rotational_speed"], errors="coerce"
				)
			# wear_stress = torque * tool_wear
			if "torque" in out.columns and "tool_wear" in out.columns:
				out["wear_stress"] = pd.to_numeric(out["torque"], errors="coerce") * pd.to_numeric(
					out["tool_wear"], errors="coerce"
				)
			# temp_delta = process_temp - air_temp
			if "process_temp" in out.columns and "air_temp" in out.columns:
				out["temp_delta"] = pd.to_numeric(out["process_temp"], errors="coerce") - pd.to_numeric(
					out["air_temp"], errors="coerce"
				)

		return out


class ClassificationPipeline:
	"""High-level classification pipeline.

	Use when you want the same steps as in `classification.ipynb` packaged into a
	single call.
	"""

	@staticmethod
	def prepare_xy(
		df: pd.DataFrame,
		*,
		target: str,
		drop_columns: list[str] | None = None,
		categorical_features: list[str] | None = None,
		encoding: FE.EncodingType = "onehot",
	) -> tuple[pd.DataFrame, pd.Series]:
		if not isinstance(df, pd.DataFrame):
			raise TypeError("df must be a pandas DataFrame")
		if target not in df.columns:
			raise ValueError(f"target column not found: {target!r}")

		work = df.copy()
		if drop_columns:
			missing = [c for c in drop_columns if c not in work.columns]
			if missing:
				raise ValueError(f"drop_columns not found in df: {missing}")
			work = work.drop(columns=drop_columns)

		y = work[target]
		X = work.drop(columns=[target])

		if categorical_features:
			X = FE.encode_categorical_features(X, columns=categorical_features, encoding=encoding)

		if not isinstance(X, pd.DataFrame):
			raise TypeError("Encoded X must be a DataFrame")
		return X, y

	@staticmethod
	def run(
		df: pd.DataFrame,
		*,
		config: ClassificationPipelineConfig,
	) -> ClassificationPipelineResult:
		"""Run the full workflow and return a bundle of results."""
		if config.use_raw_pipeline:
			return ClassificationPipeline.run_raw_pipeline(df, config=config)

		if not isinstance(df, pd.DataFrame):
			raise TypeError("df must be a pandas DataFrame")
		if config.target not in df.columns:
			raise ValueError(f"target column not found: {config.target!r}")

		work = df.copy()
		if config.drop_columns:
			missing = [c for c in config.drop_columns if c not in work.columns]
			if missing:
				raise ValueError(f"drop_columns not found in df: {missing}")
			work = work.drop(columns=config.drop_columns)

		y = work[config.target]
		X_raw = work.drop(columns=[config.target])

		# Apply feature engineering BEFORE encoding/scaling so the model actually learns from:
		# - log1p(rotational_speed)
		# - mechanical_power, wear_stress, temp_delta
		fe_step = MachineFailureFeatureEngineer(
			auto_rename_columns=config.auto_rename_columns,
			add_engineered_features=config.add_engineered_features,
			log_rotational_speed=config.log_rotational_speed,
			extra_drop_columns=config.extra_drop_columns,
		)
		X_fe = fe_step.transform(X_raw)

		# Encode categorical features (after feature engineering)
		X_model = X_fe
		if config.categorical_features:
			X_model = FE.encode_categorical_features(
				X_model,
				columns=config.categorical_features,
				encoding=config.encoding,
			)
		if not isinstance(X_model, pd.DataFrame):
			raise TypeError("Encoded X must be a DataFrame")

		X_train, X_test, y_train, y_test = MT.split_train_test(
			X_model,
			y,
			test_size=config.test_size,
			stratify=config.stratify,
			random_state=config.random_state,
		)

		X_train_s, X_test_s = MT.scale_after_split(X_train, X_test, scaler=config.scaler)

		# Keep engineered (pre-encoding) splits for inspection/debugging.
		X_train_fe = X_fe.loc[X_train.index]
		X_test_fe = X_fe.loc[X_test.index]

		models = MT.train_classification(
			X_train_s,
			y_train,
			cv=config.cv,
			scoring=config.scoring,
			stratified_cv=config.stratified_cv,
			random_state=config.random_state,
			n_jobs=config.n_jobs,
			verbose=config.verbose,
			refit=config.refit,
		)

		best_model = MT.select_best(models)

		metrics_table = MT.evaluate_classification_all(
			results=models,
			X_train=X_train_s,
			y_train=y_train,
			X_test=X_test_s,
			y_test=y_test,
			average=None,
			metric_set=config.metric_set,
			metrics=config.metrics,
			pos_label=config.pos_label,
			decimals=config.decimals,
			fit_metric=config.fit_metric,
			overfit_gap=config.overfit_gap,
			overfit_train_min=config.overfit_train_min,
			underfit_max=config.underfit_max,
		)

		pipelines = best_pipeline = None
		if config.return_best_pipeline:
			# Build fitted sklearn Pipelines for saving/reuse.
			# This trains a second time (pipeline-based) because the non-raw path
			# trains on already-encoded/scaled matrices and does not retain fitted
			# encoder/scaler objects.
			try:
				# Defaults aligned to your notebook
				if config.pipeline_numeric_features is None:
					num = [
						"air_temp",
						"process_temp",
						"rotational_speed",
						"torque",
						"tool_wear",
					]
					if config.add_engineered_features:
						num.extend(["mechanical_power", "wear_stress", "temp_delta"])
					pipeline_numeric_features = num
				else:
					pipeline_numeric_features = list(config.pipeline_numeric_features)

				pipeline_categorical_features = (
					list(config.pipeline_categorical_features)
					if config.pipeline_categorical_features is not None
					else (list(config.categorical_features) if config.categorical_features is not None else [])
				)

				# Use the SAME split indices as the non-raw path.
				X_train_raw = X_raw.loc[X_train.index]
				X_test_raw = X_raw.loc[X_test.index]

				pipe_results = ModelPipeline.train_classification_all(
					X_train=X_train_raw,
					y_train=y_train,
					feature_engineer=fe_step,
					model_types=config.pipeline_model_types,
					numeric_features=pipeline_numeric_features,
					categorical_features=pipeline_categorical_features,
					encoding=config.pipeline_encoding,
					scaler=config.scaler,
					skew_threshold=config.pipeline_skew_threshold,
					cv=config.cv,
					scoring=config.scoring,
					random_state=config.random_state,
					n_jobs=config.n_jobs,
					verbose=config.verbose,
					refit=config.refit,
					stratified_cv=config.stratified_cv,
				)

				best_pipe = ModelPipeline.select_best(pipe_results)
				pipelines = {k: v.best_estimator for k, v in pipe_results.items()}
				best_pipeline = best_pipe.best_estimator
			except Exception:
				pipelines = best_pipeline = None

		roc_all = roc_best = cm_all = cm_best = None
		if config.make_plots:
			# In this path, models were trained on scaled+encoded matrices.
			try:
				roc_all = MT.plot_roc_curves(
					models,
					X_test_s,
					y_test,
					title="ROC Curves (All Models)",
					best_only=False,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
				)
				roc_best = MT.plot_roc_curves(
					models,
					X_test_s,
					y_test,
					title="ROC Curves",
					best_only=True,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
				)
			except Exception:
				roc_all = roc_best = None

			# In notebooks, matplotlib may auto-display any created figures at the end
			# of the cell (inline backend). Close them immediately so they are only
			# rendered when the user explicitly displays/prints them later.
			try:
				import matplotlib.pyplot as plt
				for _fig in (roc_all, roc_best):
					if _fig is not None:
						plt.close(_fig)
			except Exception:
				pass

			try:
				cm_all = MT.plot_confusion_matrices(
					models,
					X_test_s,
					y_test,
					title="Confusion Matrices (All Models)",
					best_only=False,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
					normalize=config.confusion_normalize,
				)
				cm_best = MT.plot_confusion_matrices(
					models,
					X_test_s,
					y_test,
					title="Confusion Matrix",
					best_only=True,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
					normalize=config.confusion_normalize,
				)
			except Exception:
				cm_all = cm_best = None

			try:
				import matplotlib.pyplot as plt
				for _fig in (cm_all, cm_best):
					if _fig is not None:
						plt.close(_fig)
			except Exception:
				pass

		return ClassificationPipelineResult(
			config=config,
			X=X_model,
			y=y,
			X_train=X_train,
			X_test=X_test,
			y_train=y_train,
			y_test=y_test,
			X_train_s=X_train_s,
			X_test_s=X_test_s,
			models=models,
			best_model=best_model,
			metrics_table=metrics_table,
			pipelines=pipelines,
			best_pipeline=best_pipeline,
			X_fe=X_fe,
			X_train_fe=X_train_fe,
			X_test_fe=X_test_fe,
			roc_curves_all=roc_all,
			roc_curves_best=roc_best,
			confusion_matrices_all=cm_all,
			confusion_matrices_best=cm_best,
		)
	@staticmethod
	def run_raw_pipeline(
		df: pd.DataFrame,
		*,
		config: ClassificationPipelineConfig,
	) -> ClassificationPipelineResult:
		"""Train full sklearn Pipelines that can accept raw inputs for prediction demos.

		This uses `ModelPipeline.train_classification_all` under the hood and inserts
		`MachineFailureFeatureEngineer` as the first step.
		"""
		if not isinstance(df, pd.DataFrame):
			raise TypeError("df must be a pandas DataFrame")
		if config.target not in df.columns and config.auto_rename_columns:
			# allow raw column naming; feature engineer will rename, so we delay target extraction
			pass

		fe_step = MachineFailureFeatureEngineer(
			auto_rename_columns=config.auto_rename_columns,
			add_engineered_features=config.add_engineered_features,
			log_rotational_speed=config.log_rotational_speed,
			extra_drop_columns=config.extra_drop_columns,
		)
		df_fe = fe_step.transform(df)
		if config.target not in df_fe.columns:
			raise ValueError(
				f"target column not found after feature engineering: {config.target!r}. "
				"If you are passing raw CSV columns, set auto_rename_columns=True."
			)

		y = df_fe[config.target]

		# Keep X as *raw* so the fitted pipelines can be used later on raw inputs.
		df_raw = df.copy()
		df_raw.columns = df_raw.columns.astype(str)
		lower_to_actual = {c.lower(): c for c in df_raw.columns}
		target_candidates = [config.target.lower()]
		if config.auto_rename_columns:
			target_candidates.extend(
				[k for k, v in MachineFailureFeatureEngineer.DEFAULT_RENAME.items() if v == config.target]
			)
		actual_target_col = next((lower_to_actual.get(c) for c in target_candidates if c in lower_to_actual), None)
		if actual_target_col is None:
			raise ValueError(
				f"Could not find target column in raw df: {config.target!r}. "
				"If your CSV uses the original naming, set auto_rename_columns=True."
			)
		X = df_raw.drop(columns=[actual_target_col])

		# Defaults aligned to your notebook
		if config.pipeline_numeric_features is None:
			num = [
				"air_temp",
				"process_temp",
				"rotational_speed",
				"torque",
				"tool_wear",
			]
			if config.add_engineered_features:
				num.extend(["mechanical_power", "wear_stress", "temp_delta"])
			pipeline_numeric_features = num
		else:
			pipeline_numeric_features = list(config.pipeline_numeric_features)

		pipeline_categorical_features = (
			list(config.pipeline_categorical_features)
			if config.pipeline_categorical_features is not None
			else (["type"] if any(str(c).lower() == "type" for c in X.columns) else [])
		)

		# Split on raw X; y is aligned by index.
		X_train, X_test, y_train, y_test = MT.split_train_test(
			X,
			y,
			test_size=config.test_size,
			stratify=config.stratify,
			random_state=config.random_state,
		)

		# Also compute engineered versions for inspection (log transforms + new features).
		X_fe = fe_step.transform(X)
		X_train_fe = fe_step.transform(X_train)
		X_test_fe = fe_step.transform(X_test)

		pipe_results = ModelPipeline.train_classification_all(
			X_train=X_train,
			y_train=y_train,
			feature_engineer=fe_step,
			model_types=config.pipeline_model_types,
			numeric_features=pipeline_numeric_features,
			categorical_features=pipeline_categorical_features,
			encoding=config.pipeline_encoding,
			scaler=config.scaler,
			skew_threshold=config.pipeline_skew_threshold,
			cv=config.cv,
			scoring=config.scoring,
			random_state=config.random_state,
			n_jobs=config.n_jobs,
			verbose=config.verbose,
			refit=config.refit,
			stratified_cv=config.stratified_cv,
		)

		best_pipe = ModelPipeline.select_best(pipe_results)
		pipelines = {k: v.best_estimator for k, v in pipe_results.items()}

		metrics_table = MT.evaluate_classification_all(
			results=pipelines,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			metric_set=config.metric_set,
			metrics=config.metrics,
			pos_label=config.pos_label,
			decimals=config.decimals,
			fit_metric=config.fit_metric,
			overfit_gap=config.overfit_gap,
			overfit_train_min=config.overfit_train_min,
			underfit_max=config.underfit_max,
		)

		roc_all = roc_best = cm_all = cm_best = None
		if config.make_plots:
			# In this path, pipelines accept raw inputs; evaluate/plot on raw X_test.
			try:
				roc_all = MT.plot_roc_curves(
					pipelines,
					X_test,
					y_test,
					title="ROC Curves (All Models)",
					best_only=False,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
				)
				roc_best = MT.plot_roc_curves(
					pipelines,
					X_test,
					y_test,
					title="ROC Curves",
					best_only=True,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
				)
			except Exception:
				roc_all = roc_best = None

			try:
				import matplotlib.pyplot as plt
				for _fig in (roc_all, roc_best):
					if _fig is not None:
						plt.close(_fig)
			except Exception:
				pass

			try:
				cm_all = MT.plot_confusion_matrices(
					pipelines,
					X_test,
					y_test,
					title="Confusion Matrices (All Models)",
					best_only=False,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
					normalize=config.confusion_normalize,
				)
				cm_best = MT.plot_confusion_matrices(
					pipelines,
					X_test,
					y_test,
					title="Confusion Matrix",
					best_only=True,
					select_by=config.plot_select_by,
					pos_label=config.pos_label,
					normalize=config.confusion_normalize,
				)
			except Exception:
				cm_all = cm_best = None

			try:
				import matplotlib.pyplot as plt
				for _fig in (cm_all, cm_best):
					if _fig is not None:
						plt.close(_fig)
			except Exception:
				pass

		# Return bundle; pipelines accept raw inputs and perform feature engineering internally.
		return ClassificationPipelineResult(
			config=config,
			X=X,
			y=y,
			X_train=X_train,
			X_test=X_test,
			y_train=y_train,
			y_test=y_test,
			X_train_s=X_train,
			X_test_s=X_test,
			models=pipelines,
			best_model=best_pipe.best_estimator,
			metrics_table=metrics_table,
			pipelines=pipelines,
			best_pipeline=best_pipe.best_estimator,
			X_fe=X_fe,
			X_train_fe=X_train_fe,
			X_test_fe=X_test_fe,
			roc_curves_all=roc_all,
			roc_curves_best=roc_best,
			confusion_matrices_all=cm_all,
			confusion_matrices_best=cm_best,
		)

	@staticmethod
	def train_classification_pipelines(
		*,
		X_train: pd.DataFrame,
		y_train: pd.Series,
		numeric_features: list[str],
		categorical_features: list[str] | None = None,
		feature_engineer: TransformerMixin | None = None,
		model_types: tuple[Literal["logistic", "random_forest", "xgboost"], ...] = ("logistic", "random_forest"),
		encoding: FE.EncodingType = "onehot",
		scaler: ScalerType = "standard",
		skew_threshold: float = 1.0,
		cv: int = 5,
		scoring: str = "recall",
		random_state: int = 42,
		n_jobs: int = -1,
		verbose: int = 0,
		refit: bool = True,
		stratified_cv: bool = True,
	) -> dict[str, Any]:
		"""Train sklearn Pipelines directly from `X_train, y_train`.

		This is the X/y-split-friendly API (mirrors how you use ModelPipeline in regression.ipynb).
		"""
		return ModelPipeline.train_classification_all(
			X_train=X_train,
			y_train=y_train,
			feature_engineer=feature_engineer,
			model_types=model_types,
			numeric_features=list(numeric_features),
			categorical_features=list(categorical_features) if categorical_features is not None else None,
			encoding=encoding,
			scaler=scaler,
			skew_threshold=skew_threshold,
			cv=cv,
			scoring=scoring,
			random_state=random_state,
			n_jobs=n_jobs,
			verbose=verbose,
			refit=refit,
			stratified_cv=stratified_cv,
		)

	@staticmethod
	def evaluate_classification_all_pipe(
		*,
		results: dict[str, Any],
		X_train: Any,
		y_train: Any,
		X_test: Any,
		y_test: Any,
		metric_set: Literal["default", "imbalanced", "balanced", "all"] = "imbalanced",
		metrics: list[str] | tuple[str, ...] | None = None,
		pos_label: Any = 1,
		decimals: int = 3,
		fit_metric: str = "recall",
		overfit_gap: float = 0.10,
		overfit_train_min: float = 0.85,
		underfit_max: float = 0.60,
		sort_by: str | None = None,
	) -> pd.DataFrame:
		"""Evaluate many fitted sklearn Pipelines (classification) in one table."""
		return ModelPipeline.evaluate_classification_all_pipe(
			results=results,
			X_train=X_train,
			y_train=y_train,
			X_test=X_test,
			y_test=y_test,
			metric_set=metric_set,
			metrics=metrics,
			pos_label=pos_label,
			decimals=decimals,
			fit_metric=fit_metric,
			overfit_gap=overfit_gap,
			overfit_train_min=overfit_train_min,
			underfit_max=underfit_max,
			sort_by=sort_by,
		)

