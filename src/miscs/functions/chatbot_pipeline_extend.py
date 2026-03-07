from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import easyocr


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


_EASYOCR_READER = None


@dataclass(frozen=True)
class PolicyQAConfig:
	chunk_size: int = 900
	chunk_overlap: int = 150
	top_k: int = 4
	llm_model: str = "gpt-4o-mini"
	emb_model: str = "text-embedding-3-small"
	llm_temperature: float = 0.2


SYSTEM_RULES = """
You are a document summarizer and chatbot assistant.

Rules:
1) Answer ONLY using the provided context.
2) If the answer is not in the context, say: "I can't find that in the provided documents."
3) When you answer, include citations in the format [source p.X] for each key claim.
4) Keep answers concise and practical.
""".strip()


PROMPT = ChatPromptTemplate.from_messages(
	[
		("system", SYSTEM_RULES),
		("human", "Question: {question}\n\nContext:\n{context}"),
	]
)


def _find_repo_root(start: Path):
	"""Best-effort repo root discovery (looks for README.md)."""
	start = start.resolve()
	for p in [start, *start.parents]:
		if (p / "README.md").exists():
			return p
	return start.parent


def default_persist_base():
	"""Choose a stable default persist directory for Chroma."""
	env = os.getenv("POLICY_QA_PERSIST_DIR")
	if env:
		return Path(env)

	# On Azure App Service (Linux), /home is the persisted, writable location.
	# Using the repo directory (wwwroot) can be read-only depending on deployment mode.
	if os.getenv("WEBSITE_SITE_NAME") or os.getenv("WEBSITE_INSTANCE_ID"):
		home = os.getenv("HOME") or "/home"
		return Path(home) / "chroma_store"

	repo_root = _find_repo_root(Path(__file__))
	return repo_root / "chroma_store"


def persist_dir_for(doc_id: str, *, persist_base: Optional[Path] = None):
	persist_base = (persist_base or default_persist_base()).resolve()
	return persist_base / doc_id[:16]


def sha256_file(path: str | Path, block_size: int = 1 << 20):
	path = Path(path)
	h = hashlib.sha256()
	with path.open("rb") as f:
		while True:
			chunk = f.read(block_size)
			if not chunk:
				break
			h.update(chunk)
	return h.hexdigest()


def load_pdf(path: str | Path):
	path = Path(path)
	loader = PyPDFLoader(str(path))
	docs = loader.load()
	for d in docs:
		d.metadata.setdefault("source", str(path.name))
	return docs


def ocr_image(path: str | Path):
	"""Extract text from a single image via OCR.

	Uses easyocr (CPU).
	"""
	path = Path(path)

	global _EASYOCR_READER
	if _EASYOCR_READER is None:
		# Keep CPU by default for predictable Windows installs
		_EASYOCR_READER = easyocr.Reader(["en"], gpu=False)

	# paragraph=True groups text blocks into more natural lines
	pieces = _EASYOCR_READER.readtext(str(path), detail=0, paragraph=True)
	text = "\n".join(pieces) if pieces else ""

	return (text or "").strip()


def load_uploaded_file(path: str | Path):
	"""Load an uploaded file into LangChain Documents.

	- PDFs: uses the existing PyPDFLoader path (page metadata preserved)
	- Images (.jpg/.jpeg/.png): OCR into a single Document
	"""
	path = Path(path)
	suffix = path.suffix.lower()
	if suffix == ".pdf":
		return load_pdf(path)
	if suffix in SUPPORTED_IMAGE_SUFFIXES:
		text = ocr_image(path)
		return [
			Document(
				page_content=text,
				metadata={"source": str(path.name), "page": 0, "filetype": suffix.lstrip(".")},
			)
		]
	raise ValueError(
		f"Unsupported upload type: {suffix}. Supported: .pdf, {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
	)


def load_or_build_vectordb_for_upload(
	file_path: str | Path,
	*,
	doc_id: Optional[str] = None,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
	"""Index an uploaded PDF/image and return (doc_id, vectordb)."""
	file_path = Path(file_path)
	if doc_id is None:
		doc_id = sha256_file(file_path)

	pdir = persist_dir_for(doc_id, persist_base=persist_base)
	pdir.mkdir(parents=True, exist_ok=True)
	embeddings = get_embeddings(config=config)

	# If directory exists and looks non-empty, load it
	if any(pdir.iterdir()):
		vectordb = Chroma(persist_directory=str(pdir), embedding_function=embeddings)
		return doc_id, vectordb

	docs = load_uploaded_file(file_path)
	chunks = split_docs(docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
	vectordb = Chroma.from_documents(
		documents=chunks,
		embedding=embeddings,
		persist_directory=str(pdir),
	)
	try:
		vectordb.persist()
	except Exception:
		pass
	return doc_id, vectordb


def split_docs(
	docs: List[Document], *, chunk_size: int, chunk_overlap: int
):
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],
	)
	return splitter.split_documents(docs)


def get_embeddings(*, config: PolicyQAConfig = PolicyQAConfig()):
	return OpenAIEmbeddings(model=config.emb_model)


def get_llm(*, config: PolicyQAConfig = PolicyQAConfig()):
	return ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)


def load_or_build_vectordb(
	pdf_path: str | Path,
	*,
	doc_id: Optional[str] = None,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
	pdf_path = Path(pdf_path)
	if doc_id is None:
		doc_id = sha256_file(pdf_path)

	pdir = persist_dir_for(doc_id, persist_base=persist_base)
	pdir.mkdir(parents=True, exist_ok=True)
	embeddings = get_embeddings(config=config)

	# If directory exists and looks non-empty, load it
	if any(pdir.iterdir()):
		vectordb = Chroma(persist_directory=str(pdir), embedding_function=embeddings)
		return doc_id, vectordb

	# Otherwise build it
	docs = load_pdf(pdf_path)
	chunks = split_docs(docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
	vectordb = Chroma.from_documents(
		documents=chunks,
		embedding=embeddings,
		persist_directory=str(pdir),
	)
	# Persist for older Chroma interfaces
	try:
		vectordb.persist()
	except Exception:
		pass
	return doc_id, vectordb


def format_context(docs: List[Document]):
	lines: List[str] = []
	for i, d in enumerate(docs, start=1):
		src = d.metadata.get("source", "unknown")
		page = d.metadata.get("page", None)
		page_str = f"p.{page + 1}" if isinstance(page, int) else "p.?"
		lines.append(f"--- Snippet {i} ({src} {page_str}) ---\n{d.page_content}")
	return "\n\n".join(lines)


def answer_question(
	vectordb: Chroma,
	question: str,
	*,
	k: Optional[int] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
	prompt: ChatPromptTemplate = PROMPT,
):
	retriever = vectordb.as_retriever(search_kwargs={"k": k or config.top_k})
	docs = retriever.invoke(question)

	llm = get_llm(config=config)
	context = format_context(docs)
	msg = prompt.format_messages(question=question, context=context)
	resp = llm.invoke(msg)

	return {
		"answer": resp.content,
		"sources": docs,
	}


def ui_summarize(
	uploaded_doc_id: str,
	file_path: str | None,
	*,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
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
):
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
	except ValueError as e:
		return str(e), "", ""
	except RuntimeError as e:
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
):
	question = (question or "").strip()
	if not question:
		return "Please enter a question."

	# Upload-any mode
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
	"""Streamlit UI: upload file -> index -> summary + QA.

	Run with:
		streamlit run streamlit_app_extend.py
	"""
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


def _require_openai_api_key():
	if not os.getenv("OPENAI_API_KEY"):
		raise RuntimeError(
			"Missing OPENAI_API_KEY env var. Set it (and restart your shell/kernel) before running."
		)


def main(argv: Optional[List[str]] = None):
	# Streamlit must be launched with `streamlit run ...`, so this CLI just prints help.
	print("This module now uses Streamlit (not Gradio).")
	print("Run: streamlit run streamlit_app_extend.py")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

