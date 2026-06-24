"""Document models used by retrieval and ingestion."""

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """A small piece of context returned from retrieval."""

    id: str
    chunk_id: str | None = None
    document_id: str | None = None
    title: str
    snippet: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    source_file: str = ""
    department: str = ""
    document_type: str = ""
    access_level: str = ""
    created_date: str = ""


class DocumentMetadata(BaseModel):
    """Metadata required on every source markdown document."""

    department: str
    document_type: str
    access_level: str
    created_date: str
    title: str
    source_file: str


class DocumentChunk(BaseModel):
    """A chunk ready to be written to local JSONL storage."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str = Field(..., min_length=1)
    source_file: str
    title: str
    department: str
    document_type: str
    access_level: str
    created_date: str
