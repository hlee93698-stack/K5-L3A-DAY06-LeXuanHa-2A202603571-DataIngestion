"""Real local embedding model (Sentence Transformers). No mock fallback: any load or
inference error is raised to the caller."""
from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, question: str) -> list[float]: ...


class E5Embedder:
    """intfloat/multilingual-e5-* adapter.

    Per the model card, documents are encoded as "passage: <text>" and queries
    as "query: <text>"; vectors are L2-normalized so cosine similarity applies.
    """

    def __init__(self, model_name: str, query_prefix: str = "query: ",
                 passage_prefix: str = "passage: ", batch_size: int = 16,
                 revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import kept local

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu", revision=revision)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        dim = get_dim()
        if not dim:
            raise RuntimeError(f"Could not determine embedding dimension for {model_name}")
        self.dimensions = int(dim)
        self.max_seq_length = int(self.model.max_seq_length)

    def count_tokens(self, texts: list[str], with_prefix: bool = True) -> list[int]:
        tok = self.model.tokenizer
        prefix = self.passage_prefix if with_prefix else ""
        return [len(tok(prefix + t, add_special_tokens=True)["input_ids"]) for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode([self.passage_prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True)
        if len(vecs) != len(texts):
            raise RuntimeError("embedding count mismatch")
        return vecs.tolist()

    def embed_query(self, question: str) -> list[float]:
        vec = self.model.encode([self.query_prefix + question], normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)[0]
        return vec.tolist()
