from __future__ import annotations

from pathlib import Path

import typer

from rag.chunking import load_documents, naive_chunk, semantic_chunk
from rag.store import VectorIndex

app = typer.Typer(add_completion=False)


@app.command()
def main(
    corpus: Path = typer.Argument(..., help="Corpus directory"),
    strategy: str = typer.Option("semantic", "--strategy", help="naive or semantic"),
    output: Path = typer.Option(Path("indexes"), "--output"),
) -> None:
    documents = load_documents(corpus)
    if strategy == "naive":
        chunks = naive_chunk(documents)
    elif strategy == "semantic":
        chunks = semantic_chunk(documents)
    else:
        raise typer.BadParameter("strategy must be naive or semantic")

    index = VectorIndex(chunks)
    index_path = output / f"{strategy}.json"
    index.save(index_path)
    typer.echo(f"Indexed {len(chunks)} chunks using {strategy} -> {index_path}")


if __name__ == "__main__":
    app()
