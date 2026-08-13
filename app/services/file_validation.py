from dataclasses import dataclass, field
from pathlib import Path

from app.domain.document import DocumentType
from app.domain.errors import DocumentErrorCode, DocumentProcessingError

_PDF_SIGNATURE = b"%PDF-"


@dataclass(frozen=True)
class FileValidationConfig:
    max_size_bytes: int
    allowed_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".pdf", ".txt"})
    )


class FileValidator:
    """Validates uploaded file metadata and content before extraction runs."""

    def __init__(self, config: FileValidationConfig) -> None:
        self._config = config

    def validate(self, filename: str, content: bytes) -> DocumentType:
        extension = Path(filename).suffix.lower()
        if extension not in self._config.allowed_extensions:
            raise DocumentProcessingError(
                DocumentErrorCode.UNSUPPORTED_FILE_TYPE,
                f"Unsupported file extension: '{extension or filename}'. "
                f"Supported extensions: {sorted(self._config.allowed_extensions)}",
            )

        if len(content) == 0:
            raise DocumentProcessingError(
                DocumentErrorCode.EMPTY_FILE, f"File is empty: '{filename}'"
            )

        if len(content) > self._config.max_size_bytes:
            raise DocumentProcessingError(
                DocumentErrorCode.FILE_TOO_LARGE,
                f"File size exceeds the limit: {len(content)} bytes > allowed "
                f"{self._config.max_size_bytes} bytes",
            )

        document_type = DocumentType.PDF if extension == ".pdf" else DocumentType.TXT
        if document_type == DocumentType.PDF and not content.startswith(_PDF_SIGNATURE):
            raise DocumentProcessingError(
                DocumentErrorCode.CORRUPTED_FILE,
                f"'{filename}' does not have a valid PDF signature.",
            )

        return document_type
