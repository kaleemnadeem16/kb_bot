"""Runtime settings, read from environment variables (and `.env` if present)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    chat_model: str = "gpt-4.1-nano"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    min_score: float = 0.0
    temperature: float = 0.1
    data_dir: Path = Path("data")
    index_dir: Path = Path("vector_store")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            chat_model=os.getenv("OPENAI_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            top_k=int(os.getenv("TOP_K", cls.top_k)),
            min_score=float(os.getenv("MIN_SCORE", cls.min_score)),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            index_dir=Path(os.getenv("INDEX_DIR", "vector_store")),
        )

    @property
    def offline(self) -> bool:
        """True when no usable API key is set: the bot runs without calling OpenAI."""
        return not self.api_key or self.api_key.startswith("your_")

    def with_(self, **changes) -> "Settings":
        return replace(self, **changes)

    def validate(self) -> "Settings":
        """Raise ValueError with a readable message for settings that cannot work."""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be at least 0 and smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be greater than 0")
        if not math.isfinite(self.min_score):
            raise ValueError("MIN_SCORE must be a finite number")
        if not (math.isfinite(self.temperature) and 0 <= self.temperature <= 2):
            raise ValueError("TEMPERATURE must be between 0 and 2")
        return self
