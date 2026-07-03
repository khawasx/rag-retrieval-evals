from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    heading: str | None = None


def load_documents(corpus_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "section"


def naive_chunk(
    documents: list[tuple[str, str]],
    chunk_size: int = 280,
    overlap: int = 60,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    for source, text in documents:
        normalized = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            chunk_text = normalized[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{source}::naive::{counter}",
                    source=source,
                    text=chunk_text,
                )
            )
            counter += 1
            if end == len(normalized):
                break
            start = max(end - overlap, start + 1)
    return chunks


def semantic_chunk(documents: list[tuple[str, str]], max_chars: int = 420) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    for source, text in documents:
        sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading_match = re.match(r"^## (.+)", section)
            heading = heading_match.group(1).strip() if heading_match else None
            body = section
            if heading:
                body = re.sub(r"^## .+\n?", "", section).strip()

            if len(body) <= max_chars:
                chunk_id = f"{source}::{_slug(heading or 'intro')}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=source,
                        text=body or section,
                        heading=heading,
                    )
                )
                counter += 1
                continue

            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            buffer = ""
            for paragraph in paragraphs:
                candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(
                            Chunk(
                                chunk_id=f"{source}::{_slug(heading or 'intro')}_{counter}",
                                source=source,
                                text=buffer,
                                heading=heading,
                            )
                        )
                        counter += 1
                    buffer = paragraph
            if buffer:
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}::{_slug(heading or 'intro')}_{counter}",
                        source=source,
                        text=buffer,
                        heading=heading,
                    )
                )
                counter += 1
    return chunks
