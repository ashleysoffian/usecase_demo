"""Compatibility wrapper for the chatbot pipeline.

The extended chatbot implementation was moved into:
- src/chatbot/rag_pipeline.py (indexing + retrieval)
- src/chatbot/prompt_templates.py (prompt + rules)
- src/chatbot/chatbot.py (Streamlit UI helpers)

This module is kept to avoid breaking older imports.
"""

from __future__ import annotations

from src.chatbot.chatbot import run_streamlit_app, ui_answer, ui_index_uploaded, ui_summarize
from src.chatbot.rag_pipeline import (
    PROMPT,
    SYSTEM_RULES,
    PolicyQAConfig,
    SUPPORTED_IMAGE_SUFFIXES,
    answer_question,
    default_persist_base,
    format_context,
    get_embeddings,
    get_llm,
    load_or_build_vectordb,
    load_or_build_vectordb_for_upload,
    load_pdf,
    load_uploaded_file,
    ocr_image,
    persist_dir_for,
    sha256_file,
    split_docs,
)


def main() -> int:
    print("This chatbot runs via Streamlit.")
    print("Run: streamlit run streamlit/pages/04_chatbot.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
