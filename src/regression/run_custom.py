from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.regression.model import RegressionPipelineConfig
from src.regression.train import run_regression_pipeline


def _csv_to_tuple(raw: str | None) -> tuple[str, ...] | None:
	if raw is None:
		return None
	values = tuple(part.strip() for part in raw.split(",") if part.strip())
	return values or None


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m src.regression.run_custom",
		description="Train the regression pipeline on any CSV dataset.",
	)

	parser.add_argument("--dataset", required=True, help="Path to input CSV dataset.")
	parser.add_argument("--target", required=True, help="Target column name.")
	parser.add_argument("--model-path", default=None, help="Output path for saved model pipeline (.joblib).")

	parser.add_argument(
		"--model-types",
		default="linear,random_forest,xgboost",
		help="Comma-separated models. Example: linear,random_forest,xgboost",
	)
	parser.add_argument("--numeric-cols", default=None, help="Comma-separated numeric feature columns.")
	parser.add_argument("--categorical-cols", default=None, help="Comma-separated categorical feature columns.")
	parser.add_argument("--log-cols", default=None, help="Comma-separated numeric columns for skew log transform.")

	parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
	parser.add_argument("--cv", type=int, default=2, help="GridSearchCV folds.")
	parser.add_argument("--scoring", default="neg_root_mean_squared_error", help="GridSearchCV scoring metric.")
	parser.add_argument(
		"--target-transform",
		choices=["none", "log1p"],
		default="log1p",
		help="Target transform during training.",
	)
	parser.add_argument(
		"--scaler",
		choices=["standard", "minmax", "robust", "none"],
		default="standard",
		help="Numeric scaling method.",
	)
	parser.add_argument(
		"--encoding",
		choices=["onehot", "ordinal"],
		default="onehot",
		help="Categorical encoding strategy.",
	)
	parser.add_argument("--skew-threshold", type=float, default=1.5, help="Skew threshold for log transform.")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
	parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for model fitting.")
	parser.add_argument("--verbose", type=int, default=0, help="Training verbosity.")

	parser.add_argument(
		"--no-auto-infer-features",
		action="store_true",
		help="Disable automatic feature inference for missing/unspecified columns.",
	)
	parser.add_argument(
		"--auto-infer-log-columns",
		action="store_true",
		help="Auto-select log-transform columns by skewness when --log-cols is not provided.",
	)
	parser.add_argument(
		"--disable-merc-transforms",
		action="store_true",
		help="Disable merc.csv-specific normalization (rename fuelType/engineSize and year->car_age).",
	)

	parser.add_argument("--run-name", default=None, help="Optional MLflow run name.")
	parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")

	return parser


def main(argv: Sequence[str] | None = None) -> int:
	parser = _build_parser()
	args = parser.parse_args(argv)

	cfg_kwargs: dict[str, object] = {
		"dataset_path": Path(args.dataset),
		"target_column": args.target,
		"model_types": _csv_to_tuple(args.model_types) or ("linear", "random_forest", "xgboost"),
		"numeric_features": _csv_to_tuple(args.numeric_cols),
		"categorical_features": _csv_to_tuple(args.categorical_cols),
		"log_columns": _csv_to_tuple(args.log_cols),
		"test_size": float(args.test_size),
		"cv": int(args.cv),
		"scoring": args.scoring,
		"target_transform": args.target_transform,
		"scaler": args.scaler,
		"encoding": args.encoding,
		"skew_threshold": float(args.skew_threshold),
		"random_state": int(args.random_state),
		"n_jobs": int(args.n_jobs),
		"verbose": int(args.verbose),
		"auto_infer_features": not bool(args.no_auto_infer_features),
		"auto_infer_log_columns": bool(args.auto_infer_log_columns),
		"run_name": args.run_name,
		"log_to_mlflow": not bool(args.no_mlflow),
	}

	if args.model_path:
		cfg_kwargs["model_path"] = Path(args.model_path)

	if args.disable_merc_transforms:
		cfg_kwargs["rename_columns"] = {}
		cfg_kwargs["derive_car_age"] = False
		cfg_kwargs["drop_year_column"] = False

	config = RegressionPipelineConfig(**cfg_kwargs)
	result = run_regression_pipeline(config=config)

	print("Regression training complete.")
	print(f"Saved model: {result.model_path}")
	print(f"Best model: {result.best_model_name}")
	print(f"Best CV score: {result.best_score:.6f}")
	if result.run_id:
		print(f"MLflow run id: {result.run_id}")

	print("\nResolved schema")
	print(f"  target: {result.schema.target_column}")
	print(f"  features: {list(result.schema.feature_columns)}")
	print(f"  numeric: {list(result.schema.numeric_features)}")
	print(f"  categorical: {list(result.schema.categorical_features)}")
	print(f"  log_columns: {list(result.schema.log_columns)}")

	print("\nEvaluation")
	print(result.evaluation_table)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
