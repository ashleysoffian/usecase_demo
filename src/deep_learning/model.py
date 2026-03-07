from __future__ import annotations

import importlib
import random
from typing import Any

import numpy as np

tf = importlib.import_module("tensorflow")
layers = tf.keras.layers
HeUniform = tf.keras.initializers.HeUniform
Model = tf.keras.models.Model
Sequential = tf.keras.models.Sequential


def set_global_seed(seed: int = 42) -> None:
	"""Set Python/NumPy/TensorFlow seeds for reproducible runs."""
	random.seed(int(seed))
	np.random.seed(int(seed))
	tf.random.set_seed(int(seed))


@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(layers.Layer):
	"""Learned positional embeddings for sequence tokens."""

	def __init__(self, num_patches: int, embed_dim: int, **kwargs):
		super().__init__(**kwargs)
		self.num_patches = int(num_patches)
		self.embed_dim = int(embed_dim)
		self.pos = None

	def build(self, input_shape):
		if self.pos is None:
			self.pos = self.add_weight(
				name="pos_embedding",
				shape=(1, self.num_patches, self.embed_dim),
				initializer="random_normal",
				trainable=True,
			)
		super().build(input_shape)

	def call(self, tokens):
		return tokens + self.pos

	def get_config(self):
		config = super().get_config()
		config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
		return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
	"""Minimal Transformer encoder block."""

	def __init__(
		self,
		embed_dim: int,
		num_heads: int = 8,
		mlp_dim: int = 2048,
		dropout: float = 0.1,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.embed_dim = int(embed_dim)
		self.num_heads = int(num_heads)
		self.mlp_dim = int(mlp_dim)
		self.dropout = float(dropout)

		head_dim = max(1, self.embed_dim // max(1, self.num_heads))
		self.mha = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=head_dim)
		self.norm1 = layers.LayerNormalization(epsilon=1e-6)
		self.norm2 = layers.LayerNormalization(epsilon=1e-6)
		self.mlp = tf.keras.Sequential(
			[
				layers.Dense(self.mlp_dim, activation="gelu"),
				layers.Dropout(self.dropout),
				layers.Dense(self.embed_dim),
				layers.Dropout(self.dropout),
			]
		)

	def build(self, input_shape):
		shape = tf.TensorShape(input_shape)
		seq_len = shape[1] if shape.rank and shape.rank > 1 else None
		feat_dim = shape[-1]

		seq_len = int(seq_len) if seq_len is not None else 1
		feat_dim = int(feat_dim) if feat_dim is not None else self.embed_dim

		tensor_shape = tf.TensorShape((None, seq_len, feat_dim))
		self.mha.build(tensor_shape, tensor_shape, tensor_shape)
		self.norm1.build(tensor_shape)
		self.mlp.build(tensor_shape)
		self.norm2.build(tensor_shape)

		super().build(input_shape)

	def call(self, x, training=None):
		x = self.norm1(x + self.mha(x, x, training=training))
		return self.norm2(x + self.mlp(x, training=training))

	def get_config(self):
		config = super().get_config()
		config.update(
			{
				"embed_dim": self.embed_dim,
				"num_heads": self.num_heads,
				"mlp_dim": self.mlp_dim,
				"dropout": self.dropout,
			}
		)
		return config


def ensure_model_called(model, *, input_shape: tuple[int, int, int] = (64, 64, 3)):
	"""Ensure Keras model has a built input graph after loading from disk."""
	try:
		_ = model.input
		return model
	except Exception:
		pass

	dummy = tf.zeros((1,) + tuple(input_shape), dtype=tf.float32)
	_ = model(dummy, training=False)
	return model


def pick_feature_layer_name(cnn_model, preferred_name: str | None = "batch_normalization_5") -> str:
	"""Pick a valid 4D feature-map layer from a CNN model."""

	if preferred_name and any(layer.name == preferred_name for layer in cnn_model.layers):
		return preferred_name

	for layer in reversed(cnn_model.layers):
		shape = getattr(layer.output, "shape", None)
		if shape is None:
			continue
		if len(shape) == 4 and shape[-1] is not None:
			return layer.name

	raise ValueError("Could not find a 4D feature-map layer in the CNN model.")


def build_cnn_base_model(
	*,
	image_width: int = 64,
	image_height: int = 64,
	n_channels: int = 3,
	dropout_rate: float = 0.4,
):
	"""Build the base CNN model from the deep-learning notebook."""
	return Sequential(
		[
			layers.Input(shape=(int(image_width), int(image_height), int(n_channels))),
			layers.Conv2D(
				32,
				(5, 5),
				activation="relu",
				padding="same",
				strides=(1, 1),
				kernel_initializer=HeUniform(),
			),
			layers.MaxPooling2D(2, 2),
			layers.BatchNormalization(),
			layers.Conv2D(
				64,
				(5, 5),
				activation="relu",
				padding="same",
				strides=(1, 1),
				kernel_initializer=HeUniform(),
			),
			layers.MaxPooling2D(2, 2),
			layers.BatchNormalization(),
			layers.Conv2D(
				128,
				(5, 5),
				activation="relu",
				padding="same",
				strides=(1, 1),
				kernel_initializer=HeUniform(),
			),
			layers.MaxPooling2D(2, 2),
			layers.BatchNormalization(),
			layers.Conv2D(
				256,
				(5, 5),
				activation="relu",
				padding="same",
				strides=(1, 1),
				kernel_initializer=HeUniform(),
			),
			layers.MaxPooling2D(2, 2),
			layers.BatchNormalization(),
			layers.Conv2D(
				512,
				(5, 5),
				activation="relu",
				padding="same",
				strides=(1, 1),
				kernel_initializer=HeUniform(),
			),
			layers.MaxPooling2D(2, 2),
			layers.BatchNormalization(),
			layers.Conv2D(
				1024,
				(5, 5),
				activation="relu",
				padding="same",
				strides=(1, 1),
				kernel_initializer=HeUniform(),
			),
			layers.MaxPooling2D(2, 2),
			layers.BatchNormalization(),
			layers.GlobalAveragePooling2D(),
			layers.Dense(64, activation="relu", kernel_initializer=HeUniform()),
			layers.BatchNormalization(),
			layers.Dropout(dropout_rate),
			layers.Dense(128, activation="relu", kernel_initializer=HeUniform()),
			layers.BatchNormalization(),
			layers.Dropout(dropout_rate),
			layers.Dense(256, activation="relu", kernel_initializer=HeUniform()),
			layers.BatchNormalization(),
			layers.Dropout(dropout_rate),
			layers.Dense(512, activation="relu", kernel_initializer=HeUniform()),
			layers.BatchNormalization(),
			layers.Dropout(dropout_rate),
			layers.Dense(1024, activation="relu", kernel_initializer=HeUniform()),
			layers.BatchNormalization(),
			layers.Dropout(dropout_rate),
			layers.Dense(2048, activation="relu", kernel_initializer=HeUniform()),
			layers.BatchNormalization(),
			layers.Dropout(dropout_rate),
			layers.Dense(1, activation="sigmoid"),
		]
	)


def build_cnn_vit_hybrid(
	cnn_model,
	*,
	feature_layer_name: str,
	num_transformer_layers: int = 4,
	num_heads: int = 8,
	mlp_dim: int = 2048,
	num_classes: int = 1,
	fine_tune_cnn: bool = False,
):
	"""Build CNN -> Transformer hybrid model used for Model 2/Model 3."""
	cnn_model = ensure_model_called(cnn_model)
	cnn_model.trainable = bool(fine_tune_cnn)

	features = cnn_model.get_layer(feature_layer_name).output
	H, W, C = features.shape[1], features.shape[2], features.shape[3]
	if None in (H, W, C):
		raise ValueError(
			f"Backbone feature map has unknown shape: {features.shape}. "
			"Make sure CNN input shape is fixed."
		)

	H, W, C = int(H), int(W), int(C)
	x = layers.Reshape((H * W, C))(features)
	x = AddPositionEmbedding(H * W, C)(x)

	for _ in range(int(num_transformer_layers)):
		x = TransformerBlock(C, num_heads=num_heads, mlp_dim=mlp_dim)(x)

	x = layers.GlobalAveragePooling1D()(x)
	out_units = int(num_classes)
	out_activation = "sigmoid" if out_units == 1 else "softmax"
	outputs = layers.Dense(out_units, activation=out_activation)(x)

	return Model(cnn_model.inputs, outputs, name="CNN_ViT_hybrid")


def get_custom_objects() -> dict[str, Any]:
	"""Custom Keras objects required to reload hybrid models."""
	return {
		"AddPositionEmbedding": AddPositionEmbedding,
		"TransformerBlock": TransformerBlock,
	}

