from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunking import Chunk


class VectorIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [chunk.text for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[index], float(scores[index])) for index in ranked]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vocabulary": self.vectorizer.vocabulary_,
            "idf": self.vectorizer.idf_.tolist(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
