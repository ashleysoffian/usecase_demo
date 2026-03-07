from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


bootstrap_streamlit(__file__)

from src.config import Config
from src.deep_learning.model import get_custom_objects


st.set_page_config(page_title="Deep Learning", layout="wide")

st.title("Deep Learning Dashboard")
st.caption("Run image prediction using the trained model and preview the dataset.")


DATA_DIR = Path(Config.DATA_PATH) / "images_dataSAT"
MODEL_PATH = Path(Config.MODEL_PATH) / Config.DL_MODEL_FILE

TARGET_IMAGE_SIZE = (64, 64)
PREDICTION_PREVIEW_WIDTH = 200


def _iter_images(folder: Path):
	for suffix in ("*.jpg", "*.jpeg", "*.png"):
		yield from folder.glob(suffix)


def _class_labels_from_data(class_dirs: list[Path]) -> list[str]:
	if class_dirs:
		return [p.name.replace("_", " ") for p in class_dirs]
	return ["non-agri", "agri"]


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str):
	import tensorflow as tf

	custom_objects = get_custom_objects()
	try:
		return tf.keras.models.load_model(
			model_path,
			custom_objects=custom_objects,
			compile=False,
		)
	except Exception as first_error:
		try:
			return tf.keras.models.load_model(
				model_path,
				custom_objects=custom_objects,
				compile=False,
				safe_mode=False,
			)
		except TypeError:
			raise first_error


def _prepare_image(uploaded_file, *, target_size: tuple[int, int]):
	image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
	resized = image.resize(target_size)
	arr = np.asarray(resized, dtype=np.float32) / 255.0
	arr = np.expand_dims(arr, axis=0)
	return image, arr


def _predict_probabilities(model, image_batch: np.ndarray) -> np.ndarray:
	raw = np.asarray(model.predict(image_batch, verbose=0))
	if raw.ndim == 0:
		raw = np.asarray([[float(raw)]], dtype=np.float32)
	elif raw.ndim == 1:
		raw = raw.reshape(1, -1)

	if raw.shape[1] == 1:
		positive = float(raw[0, 0])
		return np.asarray([1.0 - positive, positive], dtype=np.float32)

	probs = raw[0].astype(np.float32)
	total = float(probs.sum())
	if total > 0:
		probs = probs / total
	return probs


def _load_prediction_model() -> tuple[Any, Path]:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Configured model artifact not found: {MODEL_PATH}")
	return _load_model(str(MODEL_PATH.resolve())), MODEL_PATH


class_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()]) if DATA_DIR.exists() else []
class_labels = _class_labels_from_data(class_dirs)
model_available = MODEL_PATH.exists()

st.subheader("Prediction")

if not model_available:
	st.info(
		f"Configured model artifact was not found: `{MODEL_PATH}`. "
		"Train the model first, then upload an image for inference."
	)
else:
	st.caption(f"Model selected: {MODEL_PATH}")
	uploaded_image = st.file_uploader(
		"Upload satellite image",
		type=["jpg", "jpeg", "png"],
		accept_multiple_files=False,
	)

	if uploaded_image is not None:
		preview_image, image_batch = _prepare_image(uploaded_image, target_size=TARGET_IMAGE_SIZE)
		try:
			model, _ = _load_prediction_model()
		except Exception as exc:
			st.error("Unable to load the deep learning model for inference.")
			st.exception(exc)
			st.stop()
		st.caption("Image uploaded:")
		probabilities = _predict_probabilities(model, image_batch)

		if len(class_labels) < len(probabilities):
			labels = [f"class_{i}" for i in range(len(probabilities))]
		else:
			labels = class_labels[: len(probabilities)]

		pred_idx = int(np.argmax(probabilities))
		pred_label = labels[pred_idx]
		pred_conf = float(probabilities[pred_idx])

		left, right = st.columns([1, 1])
		with left:
			st.image(preview_image, caption=uploaded_image.name, width=PREDICTION_PREVIEW_WIDTH)

		with right:
			content_col, _ = st.columns([3, 2])
			with content_col:
				st.success(f"Predicted class: {pred_label}")
				st.metric("Confidence", f"{pred_conf:.2%}")
				prob_df = pd.DataFrame(
					{
						"Class": labels,
						"Probability": [float(v) for v in probabilities],
					}
				)
				st.dataframe(prob_df, use_container_width=True, hide_index=True)
				st.bar_chart(prob_df.set_index("Class"))


st.divider()

st.subheader("Sample images")

if not DATA_DIR.exists():
	st.warning(f"Dataset folder not found: {DATA_DIR}")
elif not class_dirs:
	st.warning("No class folders found under `data/images_dataSAT/`.")
else:
	for class_dir in class_dirs:
		st.markdown(f"**{class_dir.name}**")
		imgs = []
		for p in _iter_images(class_dir):
			imgs.append(p)
			if len(imgs) >= 6:
				break

		if not imgs:
			st.caption("(No images found)")
			continue

		cols = st.columns(min(3, len(imgs)))
		for i, img_path in enumerate(imgs):
			cols[i % len(cols)].image(str(img_path), caption=img_path.name, width=PREDICTION_PREVIEW_WIDTH)

