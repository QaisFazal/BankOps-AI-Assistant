"""Document ingestion service for local markdown files.

The first storage target is JSONL, not Pinecone. Each line in the output file is
one chunk with the metadata needed for later filtering and citation display.
"""

import hashlib
import json
from pathlib import Path

import yaml

from app.models.documents import DocumentChunk, DocumentMetadata


REQUIRED_METADATA_FIELDS = {
    "department",
    "document_type",
    "access_level",
    "created_date",
    "title",
    "source_file",
}


def discover_documents(directory: Path) -> list[Path]:
    """Find markdown documents ready for ingestion."""

    if not directory.exists():
        return []

    return sorted(path for path in directory.rglob("*.md") if path.is_file())


def parse_markdown_with_metadata(path: Path) -> tuple[DocumentMetadata, str]:
    """Parse YAML frontmatter and markdown body from one source document."""

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path} is missing YAML frontmatter.")

    try:
        _, raw_metadata, body = text.split("---", 2)
    except ValueError as exc:
        raise ValueError(f"{path} has invalid YAML frontmatter.") from exc

    metadata_dict = yaml.safe_load(raw_metadata) or {}
    missing_fields = REQUIRED_METADATA_FIELDS.difference(metadata_dict)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"{path} is missing metadata field(s): {missing}.")

    metadata_dict = {key: str(value) for key, value in metadata_dict.items()}
    metadata = DocumentMetadata(**metadata_dict)
    return metadata, body.strip()


def split_text_into_chunks(text: str, max_chars: int = 900) -> list[str]:
    """Split markdown text into readable chunks without embeddings yet.

    The splitter groups paragraphs until a chunk would exceed max_chars. Very
    long paragraphs are sliced so one huge section cannot break ingestion.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_length = 0

            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start : start + max_chars].strip())
            continue

        next_length = current_length + len(paragraph) + (2 if current_parts else 0)
        if current_parts and next_length > max_chars:
            chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]
            current_length = len(paragraph)
        else:
            current_parts.append(paragraph)
            current_length = next_length

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def build_document_id(source_file: str) -> str:
    """Create a stable id from the source file path."""

    return hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:12]


def chunk_markdown_document(path: Path, max_chars: int = 900) -> list[DocumentChunk]:
    """Parse and split one markdown file into JSONL-ready chunks."""

    metadata, body = parse_markdown_with_metadata(path)
    document_id = build_document_id(metadata.source_file)
    text_chunks = split_text_into_chunks(body, max_chars=max_chars)

    return [
        DocumentChunk(
            chunk_id=f"{document_id}-{index:04d}",
            document_id=document_id,
            chunk_index=index,
            text=chunk_text,
            source_file=metadata.source_file,
            title=metadata.title,
            department=metadata.department,
            document_type=metadata.document_type,
            access_level=metadata.access_level,
            created_date=metadata.created_date,
        )
        for index, chunk_text in enumerate(text_chunks)
    ]


def ingest_documents_to_jsonl(
    source_directory: Path,
    output_path: Path,
    max_chars: int = 900,
) -> list[DocumentChunk]:
    """Read markdown docs, chunk them, and store chunks in a JSONL file."""

    chunks: list[DocumentChunk] = []
    for document_path in discover_documents(source_directory):
        chunks.extend(chunk_markdown_document(document_path, max_chars=max_chars))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

    return chunks
