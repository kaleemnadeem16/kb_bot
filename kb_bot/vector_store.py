"""FAISS cosine-similarity index, saved as `index.faiss` + `meta.json` (no pickle)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


def fingerprint(chunks: list[Chunk], embedder_name: str, settings_tag: str = "") -> str:
    """Changes whenever the text, the chunk settings or the embedding model changes."""
    h = hashlib.sha256(f"{embedder_name}\0{settings_tag}".encode())
    for c in chunks:
        h.update(f"{c.source}\0{c.page}\0{c.chunk_id}\0{c.text}\0".encode())
    return h.hexdigest()


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], key: str):
        self.index, self.chunks, self.key = index, chunks, key

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[Chunk], key: str) -> "VectorStore":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vectors) != len(chunks):
            raise ValueError("need exactly one vector per chunk")
        faiss.normalize_L2(vectors)  # inner product on unit vectors == cosine similarity
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, key)

    def search(self, vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)
        scores, ids = self.index.search(query, min(k, self.index.ntotal))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        meta = {"key": self.key, "dim": self.index.d, "chunks": [asdict(c) for c in self.chunks]}
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, expected_key: str) -> "VectorStore | None":
        """Return the saved index only if it matches `expected_key`; otherwise None."""
        folder = Path(folder)
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            if meta["key"] != expected_key:
                return None
            index = faiss.read_index(str(folder / "index.faiss"))
            chunks = [Chunk(**c) for c in meta["chunks"]]
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return None
        if index.ntotal != len(chunks) or index.d != meta.get("dim"):
            return None
        return cls(index, chunks, meta["key"])
