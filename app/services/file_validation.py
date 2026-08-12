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
                f"Desteklenmeyen dosya uzantısı: '{extension or filename}'. "
                f"Desteklenen uzantılar: {sorted(self._config.allowed_extensions)}",
            )

        if len(content) == 0:
            raise DocumentProcessingError(
                DocumentErrorCode.EMPTY_FILE, f"Dosya boş: '{filename}'"
            )

        if len(content) > self._config.max_size_bytes:
            raise DocumentProcessingError(
                DocumentErrorCode.FILE_TOO_LARGE,
                f"Dosya boyutu sınırı aşıyor: {len(content)} bayt > izin verilen "
                f"{self._config.max_size_bytes} bayt",
            )

        document_type = DocumentType.PDF if extension == ".pdf" else DocumentType.TXT
        if document_type == DocumentType.PDF and not content.startswith(_PDF_SIGNATURE):
            raise DocumentProcessingError(
                DocumentErrorCode.CORRUPTED_FILE,
                f"'{filename}' geçerli bir PDF imzasına sahip değil.",
            )

        return document_type
