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
        # Batch embedding to avoid context length limits
        # nomic-embed-text has 8192 token context. Use conservative batch size
        # to handle long entries (we've seen entries up to 21K chars!)
        BATCH_SIZE = 10
        MAX_CHARS = 5000  # Truncate extremely long entries

        # Truncate texts to avoid context overflow
        truncated_texts = [t[:MAX_CHARS] for t in texts]

        if len(truncated_texts) <= BATCH_SIZE:
            resp = self._client.embed(model=self.model, input=truncated_texts)
            return resp["embeddings"]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(truncated_texts), BATCH_SIZE):
            batch = truncated_texts[i:i+BATCH_SIZE]
            resp = self._client.embed(model=self.model, input=batch)
            all_embeddings.extend(resp["embeddings"])
        return all_embeddings
