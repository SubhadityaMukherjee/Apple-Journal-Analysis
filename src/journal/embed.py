from typing import Protocol

import ollama

from .config import EMBED_MODEL


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(self, model: str = EMBED_MODEL, client=None):
        self.model = model
        self._client = client or ollama.Client()

    def embed(self, text: str) -> list[float]:
        resp = self._client.embed(model=self.model, input=text)
        return resp["embeddings"][0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embed(model=self.model, input=texts)
        return resp["embeddings"]
