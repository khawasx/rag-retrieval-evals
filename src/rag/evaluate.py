from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag.chunking import load_documents, naive_chunk, semantic_chunk
from rag.retriever import retrieve
from rag.store import VectorIndex


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return len(set(top) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def build_index(strategy: str, corpus: Path) -> VectorIndex:
    documents = load_documents(corpus)
    if strategy == "naive":
        chunks = naive_chunk(documents)
    elif strategy == "semantic":
        chunks = semantic_chunk(documents)
    elif strategy == "naive_rerank":
        chunks = naive_chunk(documents)
    elif strategy == "semantic_rerank":
        chunks = semantic_chunk(documents)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return VectorIndex(chunks)


def chunk_is_relevant(chunk, question: dict) -> bool:
    relevant_ids = set(question.get("relevant_chunk_ids", []))
    if chunk.chunk_id in relevant_ids:
        return True
    terms = question.get("answer_terms", [])
    if not terms:
        return False
    text = chunk.text.lower()
    return all(term.lower() in text for term in terms)


def evaluate_strategy(strategy: str, corpus: Path, questions: list[dict], k: int = 5) -> dict[str, float]:
    index = build_index(strategy, corpus)
    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []

    for question in questions:
        rerank = strategy.endswith("_rerank")
        chunks = retrieve(index, question["question"], top_k=k, rerank=rerank)
        retrieved_ids = [chunk.chunk_id for chunk in chunks]
        relevant_ids = {chunk.chunk_id for chunk in index.chunks if chunk_is_relevant(chunk, question)}

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        mrrs.append(mrr(retrieved_ids, relevant_ids))

    return {
        "precision_at_k": sum(precisions) / len(precisions),
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
    }


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
