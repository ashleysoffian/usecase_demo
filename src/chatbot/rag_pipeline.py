from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import easyocr
import mlflow
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chatbot.prompt_templates import PROMPT, SYSTEM_RULES
from src.config import Config
from src.mlflow_utils import configure_chatbot_mlflow


configure_chatbot_mlflow()


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
	show_checkpoints: bool = True


def _checkpoint(config: PolicyQAConfig, message: str, *, started_at: float | None = None) -> None:
	if not config.show_checkpoints:
		return
	if started_at is None:
		elapsed = 0.0
	else:
		elapsed = time.perf_counter() - started_at
	print(f"[chatbot-rag][{elapsed:7.1f}s] {message}", flush=True)

def default_persist_base() -> Path:
	"""Choose a stable default persist directory for Chroma."""
	env = os.getenv("POLICY_QA_PERSIST_DIR")
	if env:
		return Path(env)

	# On Azure App Service (Linux), /home is the persisted, writable location.
	# Using the repo directory (wwwroot) can be read-only depending on deployment mode.
	if os.getenv("WEBSITE_SITE_NAME") or os.getenv("WEBSITE_INSTANCE_ID"):
		home = os.getenv("HOME") or "/home"
		return Path(home) / "chroma_store"

	return Path(Config.VECTOR_DB_PATH)


def persist_dir_for(doc_id: str, *, persist_base: Optional[Path] = None) -> Path:
	persist_base = (persist_base or default_persist_base()).resolve()
	return persist_base / doc_id[:16]


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
	path = Path(path)
	h = hashlib.sha256()
	with path.open("rb") as f:
		while True:
			chunk = f.read(block_size)
			if not chunk:
				break
			h.update(chunk)
	return h.hexdigest()


def load_pdf(path: str | Path) -> List[Document]:
	path = Path(path)
	loader = PyPDFLoader(str(path))
	docs = loader.load()
	for d in docs:
		d.metadata.setdefault("source", str(path.name))
	return docs


def ocr_image(path: str | Path) -> str:
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


def load_uploaded_file(path: str | Path) -> List[Document]:
	"""Load an uploaded file into LangChain Documents.

	- PDFs: uses PyPDFLoader (page metadata preserved)
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
				metadata={
					"source": str(path.name),
					"page": 0,
					"filetype": suffix.lstrip("."),
				},
			)
		]
	raise ValueError(
		f"Unsupported upload type: {suffix}. Supported: .pdf, {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
	)


def split_docs(docs: List[Document], *, chunk_size: int, chunk_overlap: int) -> List[Document]:
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


def generate_answer(
	question: str,
	context: str,
	llm: ChatOpenAI,
	*,
	prompt: ChatPromptTemplate = PROMPT,
	llm_model: str,
	retrieval_k: int,
) -> str:
	msg = prompt.format_messages(question=question, context=context)
	response = llm.invoke(msg)
	answer = str(response.content)

	try:
		with mlflow.start_run():
			mlflow.log_param("llm_model", llm_model)
			mlflow.log_param("retrieval_k", retrieval_k)
			mlflow.log_metric("response_length", len(answer))
	except Exception:
		pass

	return answer


def load_or_build_vectordb_for_upload(
	file_path: str | Path,
	*,
	doc_id: Optional[str] = None,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
) -> tuple[str, Chroma]:
	"""Index an uploaded PDF/image and return (doc_id, vectordb)."""
	started_at = time.perf_counter()
	_checkpoint(config, "Start upload indexing", started_at=started_at)

	file_path = Path(file_path)
	if doc_id is None:
		doc_id = sha256_file(file_path)

	pdir = persist_dir_for(doc_id, persist_base=persist_base)
	pdir.mkdir(parents=True, exist_ok=True)
	embeddings = get_embeddings(config=config)

	# If directory exists and looks non-empty, load it
	if any(pdir.iterdir()):
		_checkpoint(config, f"Using cached vector store: {pdir}", started_at=started_at)
		vectordb = Chroma(persist_directory=str(pdir), embedding_function=embeddings)
		return doc_id, vectordb

	_checkpoint(config, f"Loading file: {file_path.name}", started_at=started_at)
	docs = load_uploaded_file(file_path)
	_checkpoint(config, f"Loaded {len(docs)} document(s)", started_at=started_at)
	_checkpoint(config, "Splitting into chunks", started_at=started_at)
	chunks = split_docs(docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
	_checkpoint(config, f"Generated {len(chunks)} chunk(s)", started_at=started_at)
	_checkpoint(config, "Building Chroma vector store", started_at=started_at)
	vectordb = Chroma.from_documents(
		documents=chunks,
		embedding=embeddings,
		persist_directory=str(pdir),
	)
	try:
		_checkpoint(config, "Persisting vector store", started_at=started_at)
		vectordb.persist()
	except Exception:
		pass
	_checkpoint(config, "Upload indexing complete", started_at=started_at)
	return doc_id, vectordb


def load_or_build_vectordb(
	pdf_path: str | Path,
	*,
	doc_id: Optional[str] = None,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
) -> tuple[str, Chroma]:
	started_at = time.perf_counter()
	_checkpoint(config, "Start PDF indexing", started_at=started_at)
	pdf_path = Path(pdf_path)
	if doc_id is None:
		doc_id = sha256_file(pdf_path)

	pdir = persist_dir_for(doc_id, persist_base=persist_base)
	pdir.mkdir(parents=True, exist_ok=True)
	embeddings = get_embeddings(config=config)

	# If directory exists and looks non-empty, load it
	if any(pdir.iterdir()):
		_checkpoint(config, f"Using cached vector store: {pdir}", started_at=started_at)
		vectordb = Chroma(persist_directory=str(pdir), embedding_function=embeddings)
		return doc_id, vectordb

	# Otherwise build it
	_checkpoint(config, f"Loading PDF: {pdf_path.name}", started_at=started_at)
	docs = load_pdf(pdf_path)
	_checkpoint(config, f"Loaded {len(docs)} page document(s)", started_at=started_at)
	_checkpoint(config, "Splitting into chunks", started_at=started_at)
	chunks = split_docs(docs, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
	_checkpoint(config, f"Generated {len(chunks)} chunk(s)", started_at=started_at)
	_checkpoint(config, "Building Chroma vector store", started_at=started_at)
	vectordb = Chroma.from_documents(
		documents=chunks,
		embedding=embeddings,
		persist_directory=str(pdir),
	)
	# Persist for older Chroma interfaces
	try:
		_checkpoint(config, "Persisting vector store", started_at=started_at)
		vectordb.persist()
	except Exception:
		pass
	_checkpoint(config, "PDF indexing complete", started_at=started_at)
	return doc_id, vectordb


def format_context(docs: List[Document]) -> str:
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
) -> dict:
	started_at = time.perf_counter()
	retrieval_k = k or config.top_k
	_checkpoint(config, f"Retrieving top-{retrieval_k} context chunks", started_at=started_at)
	retriever = vectordb.as_retriever(search_kwargs={"k": retrieval_k})
	docs = retriever.invoke(question)
	_checkpoint(config, f"Retrieved {len(docs)} chunk(s)", started_at=started_at)

	_checkpoint(config, f"Generating response via {config.llm_model}", started_at=started_at)
	llm = get_llm(config=config)
	context = format_context(docs)
	answer = generate_answer(
		question,
		context,
		llm,
		prompt=prompt,
		llm_model=config.llm_model,
		retrieval_k=retrieval_k,
	)
	_checkpoint(config, "Answer generation complete", started_at=started_at)

	return {
		"answer": answer,
		"sources": docs,
	}

