from __future__ import annotations

import json
from pathlib import Path

import typer
from tabulate import tabulate

from rag.evaluate import evaluate_strategy, load_questions

app = typer.Typer(add_completion=False)


@app.command()
def main(
    questions: Path = typer.Argument(..., help="Path to questions JSONL"),
    corpus: Path = typer.Option(Path("docs/corpus"), "--corpus"),
    k: int = typer.Option(5, "--k"),
    output: Path = typer.Option(Path("docs/results.md"), "--output"),
) -> None:
    rows = load_questions(questions)
    strategies = ["naive", "semantic", "naive_rerank", "semantic_rerank"]
    summary = {}
    table_rows = []

    for strategy in strategies:
        metrics = evaluate_strategy(strategy, corpus, rows, k=k)
        summary[strategy] = metrics
        table_rows.append(
            [
                strategy,
                f"{metrics['precision_at_k'] * 100:.1f}%",
                f"{metrics['recall_at_k'] * 100:.1f}%",
                f"{metrics['mrr']:.3f}",
            ]
        )

    markdown = "# RAG retrieval eval results\n\n"
    markdown += tabulate(
        table_rows,
        headers=["Strategy", f"Precision@{k}", f"Recall@{k}", "MRR"],
        tablefmt="github",
    )
    markdown += "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    (output.parent / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(markdown)


if __name__ == "__main__":
    app()
