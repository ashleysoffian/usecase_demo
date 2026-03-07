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



@dataclass(frozen=True)
class PolicyQAConfig:
	chunk_size: int = 900
	chunk_overlap: int = 150
	top_k: int = 4
	llm_model: str = "gpt-4o-mini"
	emb_model: str = "text-embedding-3-small"
	llm_temperature: float = 0.2


SYSTEM_RULES = """
You are a company policy assistant.

Rules:
1) Answer ONLY using the provided policy context.
2) If the answer is not in the context, say: "I can't find that in the provided policy documents."
3) When you answer, include citations in the format [source p.X] for each key claim.
4) Keep answers concise and practical.
""".strip()


PROMPT = ChatPromptTemplate.from_messages(
	[
		("system", SYSTEM_RULES),
		("human", "Question: {question}\n\nPolicy context:\n{context}"),
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

	repo_root = _find_repo_root(Path(__file__))
	preferred = repo_root / "chroma_store"
	if preferred.exists():
		return preferred
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


def retrieve_docs(retriever, question: str):
	"""Retrieve documents using the modern LangChain retriever API."""
	if not hasattr(retriever, "invoke"):
		raise AttributeError("Retriever does not support invoke(question).")
	return retriever.invoke(question)


def answer_question(
	vectordb: Chroma,
	question: str,
	*,
	k: Optional[int] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
	prompt: ChatPromptTemplate = PROMPT,
):
	retriever = vectordb.as_retriever(search_kwargs={"k": k or config.top_k})
	docs = retrieve_docs(retriever, question)

	llm = get_llm(config=config)
	context = format_context(docs)
	msg = prompt.format_messages(question=question, context=context)
	resp = llm.invoke(msg)

	return {
		"answer": resp.content,
		"sources": docs,
	}


def init_fixed_policy(
	fixed_policy_pdf: str | Path,
	*,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
	pdf_path = Path(fixed_policy_pdf)
	if not pdf_path.exists():
		return None, None
	doc_id, vectordb = load_or_build_vectordb(
		pdf_path, persist_base=persist_base, config=config
	)
	return doc_id, vectordb


def ui_index_uploaded(
	pdf_file_path: str | None,
	*,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
	"""Index uploaded PDF and return (status, doc_id)."""
	if not pdf_file_path:
		return "Please upload a PDF first.", ""
	pdf_path = Path(pdf_file_path)
	if not pdf_path.exists():
		return "Uploaded file path not found.", ""

	doc_id, _ = load_or_build_vectordb(pdf_path, persist_base=persist_base, config=config)
	return f"Indexed: {pdf_path.name} (cache key: {doc_id[:12]})", doc_id


def ui_answer(
	mode: str,
	question: str,
	uploaded_doc_id: str,
	pdf_file_path: str | None,
	*,
	fixed_vectordb: Optional[Chroma],
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
	question = (question or "").strip()
	if not question:
		return "Please enter a question."

	# Fixed policy mode
	if mode == "Fixed policy":
		if fixed_vectordb is None:
			return "Fixed policy is not configured. Upload a PDF and index it instead."
		out = answer_question(fixed_vectordb, question, config=config)
		return out["answer"]

	# Upload-any mode
	if not uploaded_doc_id:
		return "Please upload a PDF and click **Index** first."

	pdir = persist_dir_for(uploaded_doc_id, persist_base=persist_base)
	if not (pdir.exists() and any(pdir.iterdir())):
		if pdf_file_path:
			_, vectordb = load_or_build_vectordb(
				Path(pdf_file_path),
				doc_id=uploaded_doc_id,
				persist_base=persist_base,
				config=config,
			)
		else:
			return "Vector store cache not found. Re-upload the PDF and re-index."
	else:
		vectordb = Chroma(
			persist_directory=str(pdir),
			embedding_function=get_embeddings(config=config),
		)

	out = answer_question(vectordb, question, config=config)
	return out["answer"]


def build_demo(
	*,
	fixed_policy_pdf: Optional[str | Path] = None,
	persist_base: Optional[Path] = None,
	config: PolicyQAConfig = PolicyQAConfig(),
):
	"""Return a Gradio Blocks demo matching the notebook flow."""
	import gradio as gr

	if fixed_policy_pdf is None:
		# Match the notebook default, but resolve relative to repo root when possible
		repo_root = _find_repo_root(Path(__file__))
		candidate = repo_root / "qabot" / "2025-Vistra-Code-of-Conduct.pdf"
		fixed_policy_pdf = candidate if candidate.exists() else Path("2025-Vistra-Code-of-Conduct.pdf")

	_, fixed_vectordb = init_fixed_policy(
		fixed_policy_pdf, persist_base=persist_base, config=config
	)

	with gr.Blocks() as demo:
		gr.Markdown(
			"# Policy QA Chatbot (Hybrid Demo)\nFixed policy for instant demo + Upload-any-PDF with caching"
		)

		mode = gr.Radio(
			choices=["Fixed policy", "Upload any PDF"],
			value="Fixed policy" if fixed_vectordb is not None else "Upload any PDF",
			label="Mode",
		)

		with gr.Row():
			pdf = gr.File(
				label="Upload PDF (for Upload-any mode)",
				file_types=[".pdf"],
				type="filepath",
			)
			index_btn = gr.Button("Index")

		status = gr.Markdown("")
		uploaded_doc_id = gr.Textbox(label="Uploaded doc cache key", interactive=False)

		question = gr.Textbox(
			label="Ask a question",
			placeholder="e.g., What is the reimbursement policy for travel?",
			lines=2,
		)
		ask_btn = gr.Button("Ask")
		answer = gr.Markdown("")

		index_btn.click(
			fn=lambda p: ui_index_uploaded(p, persist_base=persist_base, config=config),
			inputs=[pdf],
			outputs=[status, uploaded_doc_id],
		)

		ask_btn.click(
			fn=lambda m, q, did, p: ui_answer(
				m,
				q,
				did,
				p,
				fixed_vectordb=fixed_vectordb,
				persist_base=persist_base,
				config=config,
			),
			inputs=[mode, question, uploaded_doc_id, pdf],
			outputs=[answer],
		)

	return demo


def _require_openai_api_key():
	if not os.getenv("OPENAI_API_KEY"):
		raise RuntimeError(
			"Missing OPENAI_API_KEY env var. Set it (and restart your shell/kernel) before running."
		)


def main(argv: Optional[List[str]] = None):
	parser = argparse.ArgumentParser(description="Policy QA Chatbot (Hybrid RAG) demo")
	parser.add_argument(
		"--fixed-policy-pdf",
		default=None,
		help="Path to a fixed policy PDF for instant demo.",
	)
	parser.add_argument(
		"--persist-dir",
		default=None,
		help="Base directory for persisted Chroma stores.",
	)
	args = parser.parse_args(argv)

	_require_openai_api_key()

	persist_base = Path(args.persist_dir) if args.persist_dir else None
	demo = build_demo(
		fixed_policy_pdf=args.fixed_policy_pdf,
		persist_base=persist_base,
	)
	demo.launch()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

