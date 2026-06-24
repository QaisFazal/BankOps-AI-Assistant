"""Embedding provider abstractions for local retrieval.

The default provider is deterministic and local so tests do not need API keys.
Later, this interface can be implemented with OpenAI embeddings or
sentence-transformers without changing the retriever.
"""

import hashlib
import math
import re
from typing import Protocol


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class EmbeddingProvider(Protocol):
    """Minimal async interface for dense embedding providers."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


def tokenize(text: str) -> list[str]:
    """Tokenize text for local retrieval."""

    return TOKEN_PATTERN.findall(text.lower())


class HashEmbeddingProvider:
    """Small deterministic embedding provider for local development.

    This is not semantically rich like a model embedding, but it gives the app a
    dense-vector scoring path now and keeps the implementation offline.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text into normalized hashed bag-of-words vectors."""

        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector

        return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two normalized vectors."""

    if not left or not right:
        return 0.0

    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
