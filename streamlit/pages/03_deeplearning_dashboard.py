from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from src.utils import bootstrap_streamlit


bootstrap_streamlit(__file__)

from src.config import Config


st.set_page_config(page_title="Deep Learning", layout="wide")

st.title("Deep Learning Dashboard")
st.caption("Dataset preview for the satellite image classification use case.")


DATA_DIR = Path(Config.DATA_PATH) / "images_dataSAT"
MODEL_CANDIDATE = Path(Config.MODEL_PATH) / Config.DL_MODEL_FILE

if not DATA_DIR.exists():
	st.error(f"Missing dataset folder: {DATA_DIR}")
	st.stop()

if not MODEL_CANDIDATE.exists():
	st.info(
		"No exported deep learning model artifact was found in `artifacts/models/`. "
		"This page currently previews the dataset only."
	)


def _iter_images(folder: Path):
	for suffix in ("*.jpg", "*.jpeg", "*.png"):
		yield from folder.glob(suffix)


class_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
if not class_dirs:
	st.warning("No class folders found under `data/images_dataSAT/`.")
	st.stop()

st.subheader("Sample images")

for class_dir in sorted(class_dirs):
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
		cols[i % len(cols)].image(str(img_path), caption=img_path.name, use_container_width=True)

