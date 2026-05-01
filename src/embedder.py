import hashlib
import logging
import re
from typing import List

import numpy as np


logger = logging.getLogger(__name__)


class LocalHashEmbeddings:
    """Offline-safe fallback embedder for local development."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _embed_text(self, text: str) -> List[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = self._tokenize(text)

        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm

        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedder(
    provider: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs
):
    """
    Get an embedding model.

    Args:
        provider: "huggingface", "openai", "ollama", or "local"
        model_name: Name of the embedding model
    """
    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"}
            )
        except Exception as exc:
            logger.warning(
                "Hugging Face embeddings unavailable, falling back to local hash embeddings: %s",
                exc,
            )
            return LocalHashEmbeddings()
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_name)
    elif provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(model=model_name)
    elif provider == "local":
        return LocalHashEmbeddings()
    else:
        raise ValueError(f"Unknown provider: {provider}")


def embed_documents(embedder, documents: List) -> List[List[float]]:
    texts = [doc.page_content for doc in documents]
    return embedder.embed_documents(texts)


def embed_query(embedder, query: str) -> List[float]:
    return embedder.embed_query(query)


if __name__ == "__main__":
    embedder = get_embedder()
    test_text = "What is Retrieval-Augmented Generation?"
    embedding = embed_query(embedder, test_text)
    print(f"Embedding dimension: {len(embedding)}")
