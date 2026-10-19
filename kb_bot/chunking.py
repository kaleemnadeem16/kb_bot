"""Split documents into overlapping chunks, preferring paragraph and sentence boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import Document

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    file_type: str
    chunk_id: int
    page: int | None = None

    @property
    def citation(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page else self.source


def clean(text: str) -> str:
    """Normalise whitespace but keep paragraph breaks (blank lines)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paragraphs if p)


def _break_point(text: str, start: int, end: int) -> int:
    """Best place to cut in text[start:end]: paragraph break, then sentence end, then space."""
    floor = start + (end - start) // 2  # never cut in the first half of the window
    window = text[floor:end]
    para = window.rfind("\n\n")
    if para != -1:
        return floor + para + 2
    sentences = list(_SENTENCE_END.finditer(window))
    if sentences:
        return floor + sentences[-1].end()
    space = window.rfind(" ")
    return floor + space + 1 if space != -1 else end


def split_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    text = clean(text)
    if len(text) <= size:
        return [text] if text else []
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = _break_point(text, start, end)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        new_start = max(end - overlap, start + 1)  # always move forward
        while new_start < end and not text[new_start - 1].isspace():
            new_start += 1  # don't start the next chunk in the middle of a word
        start = new_start
    return pieces


def chunk_documents(docs: list[Document], size: int = 500, overlap: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for i, piece in enumerate(split_text(doc.text, size, overlap)):
            chunks.append(Chunk(piece, doc.source, doc.file_type, i, doc.page))
    return chunks
