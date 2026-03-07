from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import joblib

from dotenv import load_dotenv

from src.config import Config


def ensure_on_syspath(path: str | Path) -> Path:
	"""Ensure `path` is present on `sys.path` (at highest priority).

	Returns the resolved Path.
	"""
	p = Path(path).resolve()
	if str(p) not in sys.path:
		sys.path.insert(0, str(p))
	return p


def load_dotenv_files(
	paths: Iterable[str | Path],
	*,
	override: bool = False,
) -> None:
	"""Load environment variables from a list of .env files.

	Missing files are ignored.
	"""
	for p in paths:
		pp = Path(p)
		if pp.exists() and pp.is_file():
			load_dotenv(pp, override=override)


@dataclass(frozen=True)
class StreamlitBootstrap:
	repo_root: Path
	streamlit_dir: Path


def bootstrap_streamlit(
	current_file: str | Path,
	*,
	override_env: bool = False,
) -> StreamlitBootstrap:
	"""Bootstrap imports + env for Streamlit scripts/pages.

	- Detects `streamlit/` directory based on file location
	- Ensures repo root is on `sys.path` so `import src...` works
	- Loads `streamlit/.env` then `<repo>/.env` (no override by default)

	Returns:
		StreamlitBootstrap(repo_root, streamlit_dir)
	"""
	file_path = Path(current_file).resolve()
	file_dir = file_path.parent
	streamlit_dir = file_dir.parent if file_dir.name == "pages" else file_dir
	repo_root = streamlit_dir.parent

	ensure_on_syspath(repo_root)
	load_dotenv_files(
		[streamlit_dir / ".env", repo_root / ".env"],
		override=override_env,
	)

	return StreamlitBootstrap(repo_root=repo_root, streamlit_dir=streamlit_dir)

# File Utilities

def ensure_directory(path: str | Path) -> Path:
	"""Create a directory (including parents) if it doesn't exist.

	Returns the resolved directory path.
	"""
	p = Path(path).expanduser().resolve()
	p.mkdir(parents=True, exist_ok=True)
	return p

# Data Utilities

def load_data(filename: str, *, encoding: str | None = None) -> pd.DataFrame:
	"""Load a CSV from the data folder."""
	base = Path(Config.DATA_PATH)
	path = base / filename
	return pd.read_csv(path, encoding=encoding) if encoding else pd.read_csv(path)

# Model Utilities

def save_model(model, filename: str, *, model_dir: str | Path | None = None) -> Path:
	"""Save a trained model artifact with joblib.

	By default saves into Config.MODEL_PATH.
	"""
	base = Path(model_dir) if model_dir is not None else Path(Config.MODEL_PATH)
	ensure_directory(base)
	out_path = base / filename
	joblib.dump(model, out_path)
	return out_path


def load_model(filename: str, *, model_dir: str | Path | None = None) -> Any:
	"""Load a saved model artifact with joblib.

	By default loads from Config.MODEL_PATH.
	"""
	base = Path(model_dir) if model_dir is not None else Path(Config.MODEL_PATH)
	path = base / filename
	return joblib.load(path)

# Logging Utility

def print_section(title: str) -> None:
	print("\n" + "=" * 50)
	print(title)
	print("=" * 50)
