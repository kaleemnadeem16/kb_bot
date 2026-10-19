"""Load TXT, MD, CSV and PDF files from a folder into plain-text documents."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".txt", ".md", ".csv", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str  # file name, shown in citations
    text: str
    file_type: str
    page: int | None = None  # 1-based, PDFs only


def load_documents(folder: Path) -> list[Document]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Data folder not found: {folder}")
    docs: list[Document] = []
    for path in sorted(folder.iterdir()):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in SUPPORTED:
            continue
        try:
            loaded = _LOADERS[suffix](path)
        except Exception as exc:  # one bad file should not stop the rest
            log.warning("Skipping %s: %s", path.name, exc)
            continue
        loaded = [d for d in loaded if d.text.strip()]
        if not loaded:
            log.warning("Skipping %s: no text found", path.name)
        docs.extend(loaded)
    return docs


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def _load_text(path: Path) -> list[Document]:
    return [Document(path.name, _read_text(path), path.suffix.lower())]


def _load_csv(path: Path) -> list[Document]:
    """One line per row as `column: value | column: value`, so rows stay searchable."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    header = f"Table from {path.name}. Columns: {', '.join(rows[0])}."
    lines = [
        f"Row {i}: " + " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
        for i, row in enumerate(rows, 1)
    ]
    return [Document(path.name, "\n".join([header, *lines]), ".csv")]


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    docs = []
    for number, page in enumerate(PdfReader(str(path)).pages, 1):
        docs.append(Document(path.name, page.extract_text() or "", ".pdf", number))
    return docs


_LOADERS = {".txt": _load_text, ".md": _load_text, ".csv": _load_csv, ".pdf": _load_pdf}
