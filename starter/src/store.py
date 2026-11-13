"""Persistent vector store: Chroma (PersistentClient, on disk), cosine distance.

One collection per strategy. Upserts are idempotent: chunk IDs are derived from
content + position, so re-ingesting the same document version adds nothing, and
chunks that disappear when a document changes are deleted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk

SCALAR = (str, int, float, bool)


class ChromaStore:
    def __init__(self, path: Path, dimensions: int, prefix: str = "lab06") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.dimensions = dimensions
        self.prefix = prefix

    def collection(self, strategy: str, model_name: str | None = None):
        meta = {"hnsw:space": "cosine", "strategy": strategy}
        if model_name:
            meta["embedding_model"] = model_name
        return self.client.get_or_create_collection(name=f"{self.prefix}_{strategy}", metadata=meta,
                                                    embedding_function=None)

    def existing_ids(self, strategy: str, document_id: str) -> set[str]:
        got = self.collection(strategy).get(where={"document_id": document_id}, include=[])
        return set(got["ids"])

    def count(self, strategy: str) -> int:
        return self.collection(strategy).count()

    def sync_document(self, strategy: str, document_id: str, chunks: list[Chunk],
                      embed_fn, model_name: str, batch: int = 64) -> dict[str, int]:
        """Make the collection hold exactly `chunks` for this document.

        Only chunks whose ID is new are embedded and written; obsolete IDs of the
        same document are deleted. Returns counts for the reload log.
        """
        col = self.collection(strategy, model_name)
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError(f"duplicate chunk ids in {document_id}/{strategy}")
        old = self.existing_ids(strategy, document_id)
        new_ids = {c.chunk_id for c in chunks}
        obsolete = sorted(old - new_ids)
        if obsolete:
            col.delete(ids=obsolete)
        todo = [c for c in chunks if c.chunk_id not in old]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            vectors = embed_fn([c.text for c in part])
            self.upsert(strategy, part, vectors, model_name)
        return {"existing_before": len(old), "added": len(todo), "deleted": len(obsolete),
                "unchanged": len(new_ids & old), "after": len(self.existing_ids(strategy, document_id))}

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]],
               model_name: str | None = None) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"vector dim {len(v)} != {self.dimensions}")
        col = self.collection(strategy, model_name)
        col.upsert(ids=[c.chunk_id for c in chunks], embeddings=vectors,
                   documents=[c.text for c in chunks],
                   metadatas=[self._meta(c) for c in chunks])

    def copy_collection(self, src: str, dst: str, model_name: str) -> dict[str, int]:
        """Mirror src → dst (same IDs, vectors, texts, metadata); used to build C from B."""
        s = self.collection(src).get(include=["embeddings", "documents", "metadatas"])
        d = self.collection(dst, model_name)
        old = set(d.get(include=[])["ids"])
        obsolete = sorted(old - set(s["ids"]))
        if obsolete:
            d.delete(ids=obsolete)
        todo = [i for i, cid in enumerate(s["ids"]) if cid not in old]
        for k in range(0, len(todo), 256):
            idx = todo[k:k + 256]
            d.upsert(ids=[s["ids"][i] for i in idx],
                     embeddings=[list(s["embeddings"][i]) for i in idx],
                     documents=[s["documents"][i] for i in idx],
                     metadatas=[s["metadatas"][i] for i in idx])
        return {"existing_before": len(old), "added": len(todo), "deleted": len(obsolete),
                "after": d.count()}

    @staticmethod
    def _meta(c: Chunk) -> dict[str, Any]:
        meta = {"document_id": c.document_id, "source": c.source, "page": c.page,
                "page_end": c.page_end}
        for k, v in c.metadata.items():
            if v is None:
                continue
            meta[k] = v if isinstance(v, SCALAR) else str(v)
        return meta

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector has wrong dimension")
        where = None
        if filters:
            clauses = [{k: {"$eq": v}} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self.collection(strategy).query(query_embeddings=[query_vector], n_results=top_k,
                                              where=where,
                                              include=["documents", "metadatas", "distances"])
        out = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                          res["metadatas"][0], res["distances"][0]), 1):
            out.append({"rank": rank, "chunk_id": cid, "text": doc, "metadata": meta,
                        "distance": float(dist), "score": 1.0 - float(dist)})
        return out
