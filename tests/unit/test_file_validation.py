import pytest
from app.domain.document import DocumentType
from app.domain.errors import DocumentErrorCode, DocumentProcessingError
from app.services.file_validation import FileValidationConfig, FileValidator


def test_valid_txt_is_accepted() -> None:
    validator = FileValidator(FileValidationConfig(max_size_bytes=1024))

    document_type = validator.validate("notes.txt", b"hello")

    assert document_type == DocumentType.TXT


def test_valid_pdf_signature_is_accepted() -> None:
    validator = FileValidator(FileValidationConfig(max_size_bytes=1024))

    document_type = validator.validate("doc.pdf", b"%PDF-1.4\n...")

    assert document_type == DocumentType.PDF


def test_unsupported_extension_is_rejected() -> None:
    validator = FileValidator(FileValidationConfig(max_size_bytes=1024))

    with pytest.raises(DocumentProcessingError) as exc_info:
        validator.validate("data.docx", b"some bytes")

    assert exc_info.value.code == DocumentErrorCode.UNSUPPORTED_FILE_TYPE


def test_empty_file_is_rejected() -> None:
    validator = FileValidator(FileValidationConfig(max_size_bytes=1024))

    with pytest.raises(DocumentProcessingError) as exc_info:
        validator.validate("empty.txt", b"")

    assert exc_info.value.code == DocumentErrorCode.EMPTY_FILE


def test_file_larger_than_configured_limit_is_rejected() -> None:
    validator = FileValidator(FileValidationConfig(max_size_bytes=10))

    with pytest.raises(DocumentProcessingError) as exc_info:
        validator.validate("big.txt", b"x" * 11)

    assert exc_info.value.code == DocumentErrorCode.FILE_TOO_LARGE


def test_pdf_extension_without_pdf_signature_is_rejected() -> None:
    validator = FileValidator(FileValidationConfig(max_size_bytes=1024))

    with pytest.raises(DocumentProcessingError) as exc_info:
        validator.validate("fake.pdf", b"this is not a pdf")

    assert exc_info.value.code == DocumentErrorCode.CORRUPTED_FILE
