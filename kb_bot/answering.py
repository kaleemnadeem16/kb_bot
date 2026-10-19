"""Turn retrieved chunks into an answer: OpenAI chat, or an extractive fallback."""

from __future__ import annotations

from typing import Protocol

from .chunking import Chunk
from .embeddings import with_retries

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "If the passages do not contain the answer, say you could not find it in the documents. "
    "Be concise and cite passages like [1] or [2]."
)


class Answerer(Protocol):
    def answer(self, question: str, chunks: list[Chunk]) -> str: ...


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{i}] ({c.citation})\n{c.text}" for i, c in enumerate(chunks, 1))
    return f"Context:\n{context}\n\nQuestion: {question}"


class OpenAIAnswerer:
    def __init__(self, client, model: str, temperature: float = 0.1, max_tokens: int = 500):
        self.client, self.model = client, model
        self.temperature, self.max_tokens = temperature, max_tokens

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        response = with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, chunks)},
                ],
            )
        )
        return (response.choices[0].message.content or "").strip()


class ExtractiveAnswerer:
    """No LLM: show the best-matching passages. Used when no API key is configured."""

    max_passages = 2  # the bot only reports these as sources, so sources match what is shown

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks[: self.max_passages], 1))
        return f"(offline mode, no LLM: best matching passages)\n\n{passages}"
