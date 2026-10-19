"""Embedding back ends: OpenAI, or an offline hashing embedder that needs no API key."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Callable, Protocol

import numpy as np

OFFLINE_DIM = 8192  # wide enough that hash collisions rarely create false matches
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was "
    "what when where which who why with you your do does can".split()
)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OfflineEmbedder:
    """Hashed bag-of-words vectors. Keyword-level matching only, but deterministic and free.

    Lets you try the whole pipeline (and run the tests) without an API key.
    """

    name = "offline-hash-v1"
    min_score = 0.15  # below this, a match is probably noise

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), OFFLINE_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                if token in _STOPWORDS:
                    continue
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                out[row, int.from_bytes(digest, "big") % OFFLINE_DIM] += 1.0
        np.log1p(out, out=out)  # dampen repeated words
        return out


def with_retries(call: Callable, attempts: int = 4, base_delay: float = 1.0):
    """Run `call`, retrying rate-limit and transient API errors with exponential backoff."""
    import openai

    retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                 openai.InternalServerError)
    for attempt in range(attempts):
        try:
            return call()
        except retryable:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * 2**attempt)


class OpenAIEmbedder:
    def __init__(self, client, model: str, batch_size: int = 100):
        self.client, self.name, self.batch_size = client, model, batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = with_retries(lambda: self.client.embeddings.create(model=self.name, input=batch))
            vectors.extend(item.embedding for item in response.data)
        return np.asarray(vectors, dtype=np.float32)
