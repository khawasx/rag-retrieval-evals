# RAG Retrieval Evals

**Most RAG demos measure nothing. This one measures retrieval quality
directly, before the LLM ever gets involved.**

## Why this exists

A chatbot that "seems to answer okay" tells you almost nothing about whether
your retrieval is any good, because a capable model can paper over mediocre
retrieval. This repo isolates and measures the retrieval step on its own
terms: given a query, did we fetch the right chunks, and how much did our
chunking/embedding choices matter?

## The question

For a fixed document set (synthetic company policy corpus in `docs/corpus/`)
and 20 labelled questions with known answer passages, how do these compare:

| Strategy | Precision@5 | Recall@5 | MRR |
|---|---|---|---|
| Naive fixed-size chunking | 24.0% | 95.0% | 0.858 |
| Semantic/structure-aware chunking | 17.0% | 80.0% | 0.704 |
| Naive chunking + reranker | 25.0% | 96.7% | 0.858 |
| Semantic chunking + reranker | 18.0% | 82.5% | 0.702 |

Full results: [`docs/results.md`](docs/results.md)

## Method

- **Document set**: four internal policy documents (remote work, expenses,
  incident response, data retention) with clear `##` section structure.
- **Question set**: 20 hand-labelled questions in `evals/questions.jsonl`, each
  with semantic chunk ids and `answer_terms` for cross-strategy evaluation.
- **Embedding model**: scikit-learn TF-IDF (lightweight, reproducible, no API keys).
- **Chunking strategies**: naive 280-char windows with overlap; semantic split on
  markdown headings with paragraph-aware sub-chunking.
- **Metrics**: precision@5, recall@5, MRR against labelled relevant passages.

## Results

On this small corpus, **naive chunking with overlap achieved higher recall**
(95% vs 80%) because overlapping windows surface the right terms more often,
at the cost of noisier top-5 results (precision 24% vs 17%). **Reranking**
added a small lift (+1–2.5pp precision, +1.7–2.5pp recall) via keyword overlap
re-scoring — useful but not transformative at this scale.

**Chunking strategy mattered more than reranking** here: semantic chunks are
cleaner for human review, but naive overlap wins raw recall on a 4-document
corpus. I would expect semantic chunking to close the gap as document count
and section length grow.

## What I'd do differently at scale

Add hybrid BM25 + dense retrieval, hard negative mining in the question set,
and cross-encoder reranking. Evaluate on a harder adversarial question split
(paraphrased queries, multi-hop questions).

## Reproduce it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python -m rag.index docs/corpus --strategy semantic
PYTHONPATH=src python -m rag.run_eval evals/questions.jsonl
pytest
```

## Repo structure

```
src/rag/       indexing, chunking strategies, retrieval, evaluation
docs/corpus/   source policy documents
evals/         labelled question set
docs/          benchmark results
```
