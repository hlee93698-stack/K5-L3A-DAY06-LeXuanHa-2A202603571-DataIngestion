"""Shared data contracts; keep traceability from chunk back to source."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentPage:
    document_id: str
    source: str
    page: int                      # physical PDF page, 1-based
    text: str                      # normalized text of this page
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int                      # page where the chunk body starts
    text: str                      # text that is embedded and returned
    metadata: dict[str, Any] = field(default_factory=dict)
    page_end: int | None = None    # last page covered (structure chunks may span pages)

    def __post_init__(self) -> None:
        if self.page_end is None:
            self.page_end = self.page
