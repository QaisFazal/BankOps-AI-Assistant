"""Document ingestion service placeholder."""

from pathlib import Path


def discover_documents(directory: Path) -> list[Path]:
    """Find text-like documents ready for ingestion."""

    if not directory.exists():
        return []

    return sorted(path for path in directory.iterdir() if path.is_file())
