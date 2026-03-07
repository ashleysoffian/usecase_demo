from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import splitfolders

from src.config import Config
from src.deep_learning.model import (
	build_cnn_base_model,
	build_cnn_vit_hybrid,
	ensure_model_called,
	get_custom_objects,
	pick_feature_layer_name,
	set_global_seed,
)
from src.deep_learning.predict import evaluate_generator, results_to_dataframe
from src.mlflow_utils import configure_deeplearning_mlflow


model1_cnn_base = "model1_cnn_base"
model2_cnn_vit_frozen = "model2_cnn_vit_frozen"
model3_cnn_vit_tuned = "model3_cnn_vit_tuned"


@dataclass(frozen=True)
class DeepLearningPipelineConfig:
	"""Configuration for converting the deep-learning notebook into a pipeline."""

	dataset_dir: Path = field(default_factory=lambda: Path(Config.DATA_PATH) / "images_dataSAT")
	split_dir: Path = field(default_factory=lambda: Path(Config.DATA_PATH) / "images_split")
	model_dir: Path = field(default_factory=lambda: Path(Config.MODEL_PATH))

	seed: int = 42
	image_width: int = 64
	image_height: int = 64
	n_channels: int = 3

	batch_size: int = 32
	model2_batch_size: int | None = None
	model3_batch_size: int | None = None
	epochs: int = 3

	train_ratio: float = 0.7
	val_ratio: float = 0.15
	test_ratio: float = 0.15
	force_resplit: bool = False

	learning_rate_model1: float = 1e-3
	learning_rate_model2: float = 1e-4
	learning_rate_model3: float = 1e-4

	transformer_layers: int = 4
	transformer_heads: int = 8
	transformer_mlp_dim: int = 2048
	feature_layer_name: str | None = "batch_normalization_5"

	model1_filename: str = "model1_best.model.keras"
	model2_filename: str = "model2_cnn_vit_best.model.keras"
	model3_filename: str = "model3_cnn_vit_tuned_best.model.keras"
	export_best_filename: str = Config.DL_MODEL_FILE

	class_labels: tuple[str, str] = ("non-agri", "agri")
	log_to_mlflow: bool = True
	run_name: str | None = None
	skip_training: bool = False


@dataclass
class DeepLearningPipelineResult:
	"""Outputs from a full pipeline run."""

	model_paths: dict[str, Path]
	histories: dict[str, dict[str, list[float]]]
	evaluation: dict[str, dict[str, Any]]
	metrics_table: pd.DataFrame
	best_model_name: str
	best_model_path: Path
	exported_model_path: Path
	run_id: str | None = None


class DeepLearningPipeline:
	"""Notebook-equivalent deep-learning pipeline (3 models + evaluation)."""

	def __init__(self, config: DeepLearningPipelineConfig | None = None):
		self.config = config or DeepLearningPipelineConfig()
		self._validate_config()

	def _validate_config(self) -> None:
		ratio_sum = float(self.config.train_ratio + self.config.val_ratio + self.config.test_ratio)
		if not np.isclose(ratio_sum, 1.0):
			raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.4f}")

		if not self.config.dataset_dir.exists():
			raise FileNotFoundError(f"Dataset folder not found: {self.config.dataset_dir}")

		class_dirs = [p for p in self.config.dataset_dir.iterdir() if p.is_dir()]
		if len(class_dirs) < 2:
			raise ValueError(
				f"Expected at least two class folders under {self.config.dataset_dir}, found {len(class_dirs)}"
			)

		if self.config.batch_size <= 0:
			raise ValueError("batch_size must be > 0")

	def _split_ready(self) -> bool:
		return all((self.config.split_dir / split).exists() for split in ("train", "val", "test"))

	def _prepare_split(self) -> None:
		if self.config.force_resplit and self.config.split_dir.exists():
			shutil.rmtree(self.config.split_dir)

		if self._split_ready():
			return

		self.config.split_dir.mkdir(parents=True, exist_ok=True)
		splitfolders.ratio(
			str(self.config.dataset_dir),
			output=str(self.config.split_dir),
			seed=int(self.config.seed),
			ratio=(self.config.train_ratio, self.config.val_ratio, self.config.test_ratio),
		)

	def _make_generators(self, *, batch_size: int):
		import tensorflow as tf
		ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

		img_size = (int(self.config.image_width), int(self.config.image_height))

		datagen_train = ImageDataGenerator(
			rescale=1.0 / 255.0,
			rotation_range=40,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode="nearest",
		)
		datagen_eval = ImageDataGenerator(rescale=1.0 / 255.0)

		train_gen = datagen_train.flow_from_directory(
			str(self.config.split_dir / "train"),
			target_size=img_size,
			batch_size=int(batch_size),
			class_mode="binary",
			shuffle=True,
		)
		val_gen = datagen_eval.flow_from_directory(
			str(self.config.split_dir / "val"),
			target_size=img_size,
			batch_size=int(batch_size),
			class_mode="binary",
			shuffle=False,
		)
		test_gen = datagen_eval.flow_from_directory(
			str(self.config.split_dir / "test"),
			target_size=img_size,
			batch_size=int(batch_size),
			class_mode="binary",
			shuffle=False,
		)
		return train_gen, val_gen, test_gen

	def _checkpoint_path(self, filename: str) -> Path:
		self.config.model_dir.mkdir(parents=True, exist_ok=True)
		return self.config.model_dir / filename

	def _train_model(
		self,
		*,
		model,
		train_gen,
		val_gen,
		checkpoint_path: Path,
		epochs: int,
		monitor: str,
		mode: str,
	) -> dict[str, list[float]]:
		import tensorflow as tf
		ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

		checkpoint = ModelCheckpoint(
			filepath=str(checkpoint_path),
			monitor=monitor,
			mode=mode,
			save_best_only=True,
			verbose=1,
		)

		history = model.fit(
			train_gen,
			epochs=int(epochs),
			validation_data=val_gen,
			callbacks=[checkpoint],
			verbose=1,
		)

		return {k: [float(x) for x in values] for k, values in history.history.items()}

	def _load_saved_model(self, model_path: Path):
		import tensorflow as tf
		return tf.keras.models.load_model(
			str(model_path),
			custom_objects=get_custom_objects(),
			compile=False,
		)

	@staticmethod
	def _clean_metric_name(name: str) -> str:
		cleaned = re.sub(r"[^0-9A-Za-z_\-\. /]", "_", str(name))
		cleaned = cleaned.replace(" ", "_").replace("/", "_").lower()
		cleaned = re.sub(r"_+", "_", cleaned).strip("_.-")
		return cleaned or "metric"

	def _log_pipeline_run(
		self,
		*,
		model_paths: Mapping[str, Path],
		histories: Mapping[str, Mapping[str, list[float]]],
		evaluation: Mapping[str, Mapping[str, Any]],
		best_model_name: str,
		exported_model_path: Path,
	) -> str:
		configure_deeplearning_mlflow()

		params = {
			"image_width": self.config.image_width,
			"image_height": self.config.image_height,
			"batch_size": self.config.batch_size,
			"epochs": self.config.epochs,
			"skip_training": int(self.config.skip_training),
			"learning_rate_model1": self.config.learning_rate_model1,
			"learning_rate_model2": self.config.learning_rate_model2,
			"learning_rate_model3": self.config.learning_rate_model3,
			"transformer_layers": self.config.transformer_layers,
			"transformer_heads": self.config.transformer_heads,
			"transformer_mlp_dim": self.config.transformer_mlp_dim,
		}

		with mlflow.start_run(run_name=self.config.run_name or "deeplearning_pipeline") as run:
			mlflow.log_params(params)
			mlflow.log_param("best_model_name", best_model_name)

			for model_name, payload in evaluation.items():
				for metric_name, metric_value in payload.get("metrics", {}).items():
					value = float(metric_value)
					if np.isnan(value):
						continue
					mlflow.log_metric(
						f"{self._clean_metric_name(model_name)}_{self._clean_metric_name(metric_name)}",
						value,
					)

			for model_name, history in histories.items():
				for hist_name, series in history.items():
					if not series:
						continue
					mlflow.log_metric(
						f"{self._clean_metric_name(model_name)}_{self._clean_metric_name(hist_name)}_last",
						float(series[-1]),
					)

			for model_name, model_path in model_paths.items():
				if model_path.exists():
					mlflow.log_artifact(str(model_path), artifact_path=f"checkpoints/{self._clean_metric_name(model_name)}")

			if exported_model_path.exists():
				mlflow.log_artifact(str(exported_model_path), artifact_path="exported")

			return run.info.run_id

	def run(self) -> DeepLearningPipelineResult:
		"""Run notebook-equivalent deep-learning training and evaluation pipeline."""
		set_global_seed(self.config.seed)
		self._prepare_split()
		self.config.model_dir.mkdir(parents=True, exist_ok=True)

		path_model1 = self._checkpoint_path(self.config.model1_filename)
		path_model2 = self._checkpoint_path(self.config.model2_filename)
		path_model3 = self._checkpoint_path(self.config.model3_filename)
		batch2 = int(self.config.model2_batch_size or self.config.batch_size)
		batch3 = int(self.config.model3_batch_size or self.config.batch_size)

		if self.config.skip_training:
			missing = [p for p in (path_model1, path_model2, path_model3) if not p.exists()]
			if missing:
				missing_str = ", ".join(str(p) for p in missing)
				raise FileNotFoundError(
					"skip_training=True requires existing checkpoint files. "
					f"Missing: {missing_str}"
				)

			_, val1, test1 = self._make_generators(batch_size=self.config.batch_size)
			_, val2, test2 = self._make_generators(batch_size=batch2)
			_, val3, test3 = self._make_generators(batch_size=batch3)

			history1: dict[str, list[float]] = {}
			history2: dict[str, list[float]] = {}
			history3: dict[str, list[float]] = {}
		else:
			import tensorflow as tf
			Adam = tf.keras.optimizers.Adam

			# Model 1: CNN base
			train1, val1, test1 = self._make_generators(batch_size=self.config.batch_size)
			model1 = build_cnn_base_model(
				image_width=self.config.image_width,
				image_height=self.config.image_height,
				n_channels=self.config.n_channels,
			)
			model1.compile(
				optimizer=Adam(learning_rate=float(self.config.learning_rate_model1)),
				loss="binary_crossentropy",
				metrics=["accuracy"],
			)
			history1 = self._train_model(
				model=model1,
				train_gen=train1,
				val_gen=val1,
				checkpoint_path=path_model1,
				epochs=self.config.epochs,
				monitor="val_accuracy",
				mode="max",
			)

			# Model 2: CNN -> ViT hybrid (frozen CNN)
			train2, val2, test2 = self._make_generators(batch_size=batch2)
			cnn_backbone = self._load_saved_model(path_model1)
			cnn_backbone = ensure_model_called(
				cnn_backbone,
				input_shape=(self.config.image_width, self.config.image_height, self.config.n_channels),
			)
			feature_layer_2 = pick_feature_layer_name(cnn_backbone, self.config.feature_layer_name)
			model2 = build_cnn_vit_hybrid(
				cnn_backbone,
				feature_layer_name=feature_layer_2,
				num_transformer_layers=self.config.transformer_layers,
				num_heads=self.config.transformer_heads,
				mlp_dim=self.config.transformer_mlp_dim,
				num_classes=1,
				fine_tune_cnn=False,
			)
			model2.compile(
				optimizer=Adam(learning_rate=float(self.config.learning_rate_model2)),
				loss="binary_crossentropy",
				metrics=["accuracy"],
			)
			history2 = self._train_model(
				model=model2,
				train_gen=train2,
				val_gen=val2,
				checkpoint_path=path_model2,
				epochs=self.config.epochs,
				monitor="val_loss",
				mode="min",
			)

			# Model 3: CNN -> ViT hybrid (fine-tuned CNN)
			train3, val3, test3 = self._make_generators(batch_size=batch3)
			cnn_backbone_ft = self._load_saved_model(path_model1)
			cnn_backbone_ft = ensure_model_called(
				cnn_backbone_ft,
				input_shape=(self.config.image_width, self.config.image_height, self.config.n_channels),
			)
			feature_layer_3 = pick_feature_layer_name(cnn_backbone_ft, self.config.feature_layer_name)
			model3 = build_cnn_vit_hybrid(
				cnn_backbone_ft,
				feature_layer_name=feature_layer_3,
				num_transformer_layers=self.config.transformer_layers,
				num_heads=self.config.transformer_heads,
				mlp_dim=self.config.transformer_mlp_dim,
				num_classes=1,
				fine_tune_cnn=True,
			)
			model3.compile(
				optimizer=Adam(learning_rate=float(self.config.learning_rate_model3)),
				loss="binary_crossentropy",
				metrics=["accuracy"],
			)
			history3 = self._train_model(
				model=model3,
				train_gen=train3,
				val_gen=val3,
				checkpoint_path=path_model3,
				epochs=self.config.epochs,
				monitor="val_loss",
				mode="min",
			)

		# Load best checkpoints for evaluation
		best1 = self._load_saved_model(path_model1)
		best2 = self._load_saved_model(path_model2)
		best3 = self._load_saved_model(path_model3)

		validation_evaluation = {
			model1_cnn_base: evaluate_generator(
				model_name=model1_cnn_base,
				model=best1,
				generator=val1,
				class_labels=self.config.class_labels,
				predict_verbose=0,
			),
			model2_cnn_vit_frozen: evaluate_generator(
				model_name=model2_cnn_vit_frozen,
				model=best2,
				generator=val2,
				class_labels=self.config.class_labels,
				predict_verbose=0,
			),
			model3_cnn_vit_tuned: evaluate_generator(
				model_name=model3_cnn_vit_tuned,
				model=best3,
				generator=val3,
				class_labels=self.config.class_labels,
				predict_verbose=0,
			),
		}

		validation_table = results_to_dataframe(validation_evaluation).round(4)

		metric_for_best = "ROC-AUC" if "ROC-AUC" in validation_table.columns else "Accuracy"
		best_scores = pd.to_numeric(validation_table[metric_for_best], errors="coerce")
		if best_scores.isna().all() and "Accuracy" in validation_table.columns:
			best_scores = pd.to_numeric(validation_table["Accuracy"], errors="coerce")
		best_model_name = str(best_scores.idxmax())

		evaluation = {
			model1_cnn_base: evaluate_generator(
				model_name=model1_cnn_base,
				model=best1,
				generator=test1,
				class_labels=self.config.class_labels,
				predict_verbose=1,
			),
			model2_cnn_vit_frozen: evaluate_generator(
				model_name=model2_cnn_vit_frozen,
				model=best2,
				generator=test2,
				class_labels=self.config.class_labels,
				predict_verbose=1,
			),
			model3_cnn_vit_tuned: evaluate_generator(
				model_name=model3_cnn_vit_tuned,
				model=best3,
				generator=test3,
				class_labels=self.config.class_labels,
				predict_verbose=1,
			),
		}

		metrics_table = results_to_dataframe(evaluation).round(4)

		name_to_path = {
			model1_cnn_base: path_model1,
			model2_cnn_vit_frozen: path_model2,
			model3_cnn_vit_tuned: path_model3,
		}
		best_model_path = name_to_path[best_model_name]

		name_to_export_filename = {
			model1_cnn_base: self.config.model1_filename,
			model2_cnn_vit_frozen: self.config.model2_filename,
			model3_cnn_vit_tuned: self.config.model3_filename,
		}
		export_filename = name_to_export_filename.get(best_model_name, self.config.export_best_filename)
		exported_model_path = self.config.model_dir / export_filename
		if exported_model_path.suffix.lower() not in {".keras", ".h5"}:
			exported_model_path = exported_model_path.with_suffix(".keras")

		best_model = self._load_saved_model(best_model_path)
		best_model.save(str(exported_model_path), include_optimizer=False)

		model_paths = {
			model1_cnn_base: path_model1,
			model2_cnn_vit_frozen: path_model2,
			model3_cnn_vit_tuned: path_model3,
		}
		histories = {
			model1_cnn_base: history1,
			model2_cnn_vit_frozen: history2,
			model3_cnn_vit_tuned: history3,
		}

		run_id: str | None = None
		if self.config.log_to_mlflow:
			run_id = self._log_pipeline_run(
				model_paths=model_paths,
				histories=histories,
				evaluation=evaluation,
				best_model_name=best_model_name,
				exported_model_path=exported_model_path,
			)

		return DeepLearningPipelineResult(
			model_paths=model_paths,
			histories=histories,
			evaluation=evaluation,
			metrics_table=metrics_table,
			best_model_name=best_model_name,
			best_model_path=best_model_path,
			exported_model_path=exported_model_path,
			run_id=run_id,
		)


def start_deeplearning_run(*, run_name: str | None = None):
	"""Start an MLflow run for deep-learning training."""
	configure_deeplearning_mlflow()
	return mlflow.start_run(run_name=run_name)


def log_deeplearning_training(
	*,
	run_name: str | None = None,
	model_name: str | None = None,
	params: Mapping[str, Any] | None = None,
	metrics: Mapping[str, float] | None = None,
) -> str:
	"""Log a deep-learning training run to MLflow and return run_id."""
	configure_deeplearning_mlflow()
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


def run_deeplearning_pipeline(
	config: DeepLearningPipelineConfig | None = None,
) -> DeepLearningPipelineResult:
	"""Convenience entrypoint to run the deep-learning pipeline."""
	return DeepLearningPipeline(config=config).run()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run deep-learning training/evaluation pipeline.")
	parser.add_argument(
		"--skip-training",
		action="store_true",
		help="Skip model fitting and only evaluate/log using existing checkpoint files.",
	)
	parser.add_argument(
		"--no-mlflow",
		action="store_true",
		help="Disable MLflow logging for this run.",
	)
	args = parser.parse_args()

	config = DeepLearningPipelineConfig(
		skip_training=bool(args.skip_training),
		log_to_mlflow=not bool(args.no_mlflow),
	)

	result = run_deeplearning_pipeline(config=config)
	print(result.metrics_table)
	print(f"Best model: {result.best_model_name}")
	print(f"Exported model: {result.exported_model_path}")
	if result.run_id:
		print(f"MLflow run_id: {result.run_id}")
