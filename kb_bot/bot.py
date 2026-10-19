"""KnowledgeBase: load, chunk, embed, index, retrieve, answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from .answering import Answerer, ExtractiveAnswerer, OpenAIAnswerer
from .chunking import Chunk, chunk_documents
from .config import Settings
from .documents import load_documents
from .embeddings import Embedder, OfflineEmbedder, OpenAIEmbedder
from .vector_store import VectorStore, fingerprint


@dataclass
class Source:
    citation: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: Embedder | None = None,
                 answerer: Answerer | None = None):
        self.settings = settings.validate()
        if embedder is None or answerer is None:
            default_embedder, default_answerer = self._default_backends(settings)
            embedder, answerer = embedder or default_embedder, answerer or default_answerer
        self.embedder, self.answerer = embedder, answerer
        self.store: VectorStore | None = None
        self.chunks: list[Chunk] = []
        self.rebuilt = False

    @staticmethod
    def _default_backends(s: Settings) -> tuple[Embedder, Answerer]:
        if s.offline:
            return OfflineEmbedder(), ExtractiveAnswerer()
        from openai import OpenAI

        client = OpenAI(api_key=s.api_key)
        return (OpenAIEmbedder(client, s.embedding_model),
                OpenAIAnswerer(client, s.chat_model, s.temperature))

    @property
    def mode(self) -> str:
        return "offline" if isinstance(self.answerer, ExtractiveAnswerer) else "openai"

    def load(self, force_rebuild: bool = False) -> int:
        """Read the data folder and make the index current. Returns the chunk count."""
        s = self.settings
        docs = load_documents(s.data_dir)
        self.chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
        if not self.chunks:
            raise ValueError(f"No readable text found in {s.data_dir}/ (supported: txt, md, csv, pdf)")
        key = fingerprint(self.chunks, self.embedder.name, f"{s.chunk_size}/{s.chunk_overlap}")
        self.store = None if force_rebuild else VectorStore.load(s.index_dir, key)
        self.rebuilt = self.store is None
        if self.store is None:
            vectors = self.embedder.embed([c.text for c in self.chunks])
            self.store = VectorStore.build(vectors, self.chunks, key)
            self.store.save(s.index_dir)
        return len(self.chunks)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        if self.store is None:
            raise RuntimeError("Call load() before ask().")
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        hits = self.store.search(self.embedder.embed([question])[0], top_k or self.settings.top_k)
        floor = max(self.settings.min_score, getattr(self.embedder, "min_score", 0.0))
        hits = [(c, score) for c, score in hits if score > 0 and score >= floor]
        hits = hits[: getattr(self.answerer, "max_passages", len(hits))]
        if not hits:
            return Answer(question, "I could not find anything relevant in the documents.")
        text = self.answerer.answer(question, [c for c, _ in hits])
        return Answer(question, text, [Source(c.citation, score, c.text) for c, score in hits])

    def stats(self) -> dict:
        return {"mode": self.mode, "files": sorted({c.source for c in self.chunks}),
                "chunks": len(self.chunks), "index_rebuilt_this_run": self.rebuilt}
