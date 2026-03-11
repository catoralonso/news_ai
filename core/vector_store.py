"""
core/vector_store.py
────────────────────
Capa de abstracción sobre ChromaDB.
Local por defecto → swap sencillo a Vertex AI Vector Search en producción.

RAM estimada: ~200 MB para 10 000 chunks de 512 tokens.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/embeddings")


@dataclass
class SearchResult:
    text: str
    score: float          # distancia coseno (menor = más similar)
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Wrapper minimalista sobre ChromaDB.

    Uso:
        vs = VectorStore(collection_name="news_research")
        vs.upsert(texts, metadatas)
        results = vs.query("alcaldía nueva obra pública", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "news_research",
        persist_dir: str = CHROMA_PERSIST_DIR,
        embedding_fn=None,          # None → ChromaDB usa su modelo interno (all-MiniLM-L6-v2)
    ):
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fn,  # inyectable → fácil swap
        )

    # ── Escritura ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Agrega o actualiza documentos en la colección."""
        if not texts:
            return

        ids = ids or [_stable_id(t) for t in texts]
        metadatas = metadatas or [{} for _ in texts]

        self._collection.upsert(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def delete_collection(self) -> None:
        """Borra la colección entera (útil en tests)."""
        self._client.delete_collection(self._collection.name)

    # ── Lectura ───────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: dict | None = None,          # filtros de metadata
    ) -> list[SearchResult]:
        """Búsqueda semántica. Devuelve top_k resultados ordenados por similitud."""
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results: list[SearchResult] = []
        for doc, dist, meta, doc_id in zip(
            raw["documents"][0],
            raw["distances"][0],
            raw["metadatas"][0],
            raw["ids"][0],
        ):
            results.append(
                SearchResult(text=doc, score=dist, metadata=meta, doc_id=doc_id)
            )
        return results

    def count(self) -> int:
        return self._collection.count()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stable_id(text: str) -> str:
    """ID determinista a partir del contenido (evita duplicados)."""
    return hashlib.md5(text.encode()).hexdigest()
