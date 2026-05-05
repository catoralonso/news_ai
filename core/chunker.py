"""
core/chunker.py
───────────────
Divide textos largos en chunks con overlap.
Usado por el RAG pipeline antes de embeddear.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Divide `text` en chunks de `chunk_size` palabras con `overlap` palabras
    de solapamiento entre chunks contiguos.

    Parámetros elegidos para equilibrio entre contexto y uso de RAM:
    - chunk_size=512 → ~400 tokens, cabe bien en embeddings de 768 dims
    - overlap=64     → ~12 % de solapamiento, evita cortes de ideas
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap

    return chunks


def chunk_document(
    doc: dict,
    text_key: str = "content",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[dict]:
    """
    Chunkea un documento estructurado y propaga sus metadatos a cada chunk.

    Entrada esperada:
        {
          "title": "...",
          "date": "...",
          "category": "...",
          "content": "texto largo..."
        }

    Salida:
        [
          {"text": "...", "title": "...", "date": "...", "category": "...", "chunk_idx": 0},
          ...
        ]
    """
    text = doc.get(text_key, "")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    metadata_base = {k: v for k, v in doc.items() if k != text_key}

    return [
        {**metadata_base, "text": chunk, "chunk_idx": i}
        for i, chunk in enumerate(chunks)
    ]
