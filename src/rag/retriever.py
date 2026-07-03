from __future__ import annotations

import re

from rag.chunking import Chunk
from rag.store import VectorIndex


def keyword_rerank(query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5) -> list[tuple[Chunk, float]]:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    rescored: list[tuple[Chunk, float]] = []
    for chunk, base_score in candidates:
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.text.lower()))
        overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        combined = (0.6 * base_score) + (0.4 * overlap)
        rescored.append((chunk, combined))
    rescored.sort(key=lambda item: item[1], reverse=True)
    return rescored[:top_k]


def retrieve(index: VectorIndex, query: str, *, top_k: int = 5, rerank: bool = False) -> list[Chunk]:
    initial_k = top_k * 3 if rerank else top_k
    candidates = index.search(query, top_k=initial_k)
    if rerank:
        candidates = keyword_rerank(query, candidates, top_k=top_k)
    return [chunk for chunk, _ in candidates[:top_k]]
