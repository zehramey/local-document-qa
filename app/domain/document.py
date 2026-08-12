from dataclasses import dataclass
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class Document:
    document_id: str
    filename: str
    document_type: DocumentType
    size_bytes: int


@dataclass(frozen=True)
class ExtractionResult:
    document: Document
    pages: list[ExtractedPage]
