from enum import Enum


class DocumentErrorCode(str, Enum):
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    CORRUPTED_FILE = "corrupted_file"
    ENCRYPTED_PDF = "encrypted_pdf"
    SCANNED_PDF_NO_TEXT = "scanned_pdf_no_text"
    TEXT_DECODING_ERROR = "text_decoding_error"


class DocumentProcessingError(Exception):
    """Raised when an uploaded document fails validation or text extraction."""

    def __init__(self, code: DocumentErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
