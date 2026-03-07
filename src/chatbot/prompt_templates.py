from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


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
