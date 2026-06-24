"""Document models used by retrieval and ingestion."""

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    """A small piece of context returned from retrieval."""

    id: str
    title: str
    snippet: str
    score: float
