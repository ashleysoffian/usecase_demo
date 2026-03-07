from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma

from src.chatbot.rag_pipeline import (
	PolicyQAConfig,
	SUPPORTED_IMAGE_SUFFIXES,
	answer_question,
	default_persist_base,
	get_embeddings,
	load_or_build_vectordb_for_upload,
	persist_dir_for,
)


def ui_summarize(
	uploaded_doc_id: str,
	file_path: str | None,
	*,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
) -> str:
	"""Generate a summary of the uploaded file using retrieved context."""

	if not uploaded_doc_id:
		return "Please upload a file and click **Index** first."

	pdir = persist_dir_for(uploaded_doc_id, persist_base=persist_base)
	if not (pdir.exists() and any(pdir.iterdir())):
		if file_path:
			_, vectordb = load_or_build_vectordb_for_upload(
				Path(file_path),
				doc_id=uploaded_doc_id,
				persist_base=persist_base,
				config=config,
			)
		else:
			return "Vector store cache not found. Re-upload the file and re-index."
	else:
		vectordb = Chroma(
			persist_directory=str(pdir),
			embedding_function=get_embeddings(config=config),
		)

	out = answer_question(
		vectordb,
		"Summarize the uploaded document or image. Provide a concise, structured summary with citations.",
		k=max(config.top_k, 12),
		config=config,
	)
	return out["answer"]


def ui_index_uploaded(
	file_path: str | None,
	*,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
) -> tuple[str, str, str]:
	"""Index uploaded file (PDF or image) and return (status, doc_id, summary)."""

	if not file_path:
		return "Please upload a file first.", "", ""
	path = Path(file_path)
	if not path.exists():
		return "Uploaded file path not found.", "", ""

	try:
		doc_id, _ = load_or_build_vectordb_for_upload(
			path, persist_base=persist_base, config=config
		)
	except (ValueError, RuntimeError) as e:
		return str(e), "", ""

	# Generate summary immediately after indexing
	try:
		summary = ui_summarize(
			doc_id,
			str(path),
			persist_base=persist_base,
			config=config,
		)
	except Exception as e:
		summary = f"Summary failed: {e}"

	return f"Indexed: {path.name} (cache key: {doc_id[:12]})", doc_id, summary


def ui_answer(
	question: str,
	uploaded_doc_id: str,
	file_path: str | None,
	*,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
) -> str:
	question = (question or "").strip()
	if not question:
		return "Please enter a question."

	if not uploaded_doc_id:
		return "Please upload a file and click **Index** first."

	pdir = persist_dir_for(uploaded_doc_id, persist_base=persist_base)
	if not (pdir.exists() and any(pdir.iterdir())):
		if file_path:
			try:
				_, vectordb = load_or_build_vectordb_for_upload(
					Path(file_path),
					doc_id=uploaded_doc_id,
					persist_base=persist_base,
					config=config,
				)
			except Exception as e:
				return str(e)
		else:
			return "Vector store cache not found. Re-upload the file and re-index."
	else:
		vectordb = Chroma(
			persist_directory=str(pdir),
			embedding_function=get_embeddings(config=config),
		)

	out = answer_question(vectordb, question, config=config)
	return out["answer"]


def run_streamlit_app(
	*,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
	"""Streamlit UI: upload file -> index -> summary + QA."""

	import streamlit as st

	st.set_page_config(page_title="Summarizer + QA", layout="centered")
	st.title("Document/Image Summarizer + QA")
	st.caption("Upload a PDF/JPG/JPEG/PNG, index it, then ask questions.")

	if "file_path" not in st.session_state:
		st.session_state.file_path = None
	if "uploaded_doc_id" not in st.session_state:
		st.session_state.uploaded_doc_id = ""
	if "summary" not in st.session_state:
		st.session_state.summary = ""
	if "answer" not in st.session_state:
		st.session_state.answer = ""
	if "status" not in st.session_state:
		st.session_state.status = ""

	uploaded = st.file_uploader(
		"Upload file",
		type=["pdf", "jpg", "jpeg", "png"],
		accept_multiple_files=False,
	)

	if uploaded is not None:
		name = uploaded.name or "upload"
		suffix = Path(name).suffix.lower()
		if suffix not in {".pdf", *SUPPORTED_IMAGE_SUFFIXES}:
			st.error("Unsupported file type. Please upload PDF/JPG/JPEG/PNG.")
		else:
			data = uploaded.getvalue()
			doc_id = hashlib.sha256(data).hexdigest()
			base = (persist_base or default_persist_base()).resolve()
			upload_dir = base / "_uploads"
			upload_dir.mkdir(parents=True, exist_ok=True)
			upload_path = upload_dir / f"{doc_id[:16]}{suffix}"
			if not upload_path.exists():
				# Write once; subsequent reruns reuse the same file
				upload_path.write_bytes(data)
			st.session_state.file_path = str(upload_path)
			st.session_state.status = f"Selected: {name}"
			st.session_state.answer = ""

	cols = st.columns([1, 1])
	with cols[0]:
		index_clicked = st.button("Index", type="primary")
	with cols[1]:
		clear_clicked = st.button("Clear")

	if clear_clicked:
		st.session_state.file_path = None
		st.session_state.uploaded_doc_id = ""
		st.session_state.summary = ""
		st.session_state.answer = ""
		st.session_state.status = ""
		st.rerun()

	if index_clicked:
		_require_openai_api_key()
		with st.spinner("Indexing and generating summary..."):
			status, doc_id, summary = ui_index_uploaded(
				st.session_state.file_path,
				persist_base=persist_base,
				config=config,
			)
			st.session_state.status = status
			st.session_state.uploaded_doc_id = doc_id
			st.session_state.summary = summary
			st.session_state.answer = ""

	if st.session_state.status:
		st.info(st.session_state.status)

	if st.session_state.summary:
		st.subheader("Summary")
		st.markdown(st.session_state.summary)

	st.divider()
	st.subheader("Q&A")
	question = st.text_area(
		"Ask a question",
		placeholder="e.g., What is the document about?",
		height=80,
	)
	ask_clicked = st.button("Ask")
	if ask_clicked:
		_require_openai_api_key()
		with st.spinner("Answering..."):
			st.session_state.answer = ui_answer(
				question,
				st.session_state.uploaded_doc_id,
				st.session_state.file_path,
				persist_base=persist_base,
				config=config,
			)

	if st.session_state.answer:
		st.subheader("Answer")
		st.markdown(st.session_state.answer)


def _require_openai_api_key() -> None:
	if not os.getenv("OPENAI_API_KEY"):
		raise RuntimeError(
			"Missing OPENAI_API_KEY env var. Set it (and restart your shell/kernel) before running."
		)


def main() -> int:
	# Streamlit must be launched with `streamlit run ...`, so this CLI just prints help.
	print("This chatbot runs via Streamlit.")
	print("Run: streamlit run streamlit/pages/04_chatbot.py")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

