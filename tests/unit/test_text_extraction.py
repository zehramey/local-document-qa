import pytest
from app.domain.document import DocumentType
from app.domain.errors import DocumentErrorCode, DocumentProcessingError
from app.services.file_validation import FileValidationConfig, FileValidator
from app.services.text_extraction import TextExtractionService

from tests.unit.pdf_builders import (
    build_blank_pdf_bytes,
    build_encrypted_pdf_bytes,
    build_pdf_bytes,
)


@pytest.fixture
def service() -> TextExtractionService:
    validator = FileValidator(FileValidationConfig(max_size_bytes=10 * 1024 * 1024))
    return TextExtractionService(validator)


def test_valid_txt_extracts_single_page(service: TextExtractionService) -> None:
    result = service.extract("notes.txt", b"hello world")

    assert result.document.document_type == DocumentType.TXT
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert result.pages[0].text == "hello world"


def test_valid_multi_page_pdf_extracts_all_pages(service: TextExtractionService) -> None:
    pdf_bytes = build_pdf_bytes(["first page text", "second page text", "third page text"])

    result = service.extract("doc.pdf", pdf_bytes)

    assert result.document.document_type == DocumentType.PDF
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert "first page text" in result.pages[0].text
    assert "second page text" in result.pages[1].text
    assert "third page text" in result.pages[2].text


def test_empty_file_is_rejected(service: TextExtractionService) -> None:
    with pytest.raises(DocumentProcessingError) as exc_info:
        service.extract("empty.txt", b"")

    assert exc_info.value.code == DocumentErrorCode.EMPTY_FILE


def test_unsupported_extension_is_rejected(service: TextExtractionService) -> None:
    with pytest.raises(DocumentProcessingError) as exc_info:
        service.extract("data.docx", b"some bytes")

    assert exc_info.value.code == DocumentErrorCode.UNSUPPORTED_FILE_TYPE


def test_corrupted_pdf_is_rejected_with_clear_message(service: TextExtractionService) -> None:
    corrupted = b"%PDF-1.4\n" + b"not a real pdf body" * 10

    with pytest.raises(DocumentProcessingError) as exc_info:
        service.extract("broken.pdf", corrupted)

    assert exc_info.value.code == DocumentErrorCode.CORRUPTED_FILE
    assert exc_info.value.message


def test_encrypted_pdf_is_detected(service: TextExtractionService) -> None:
    encrypted = build_encrypted_pdf_bytes()

    with pytest.raises(DocumentProcessingError) as exc_info:
        service.extract("secure.pdf", encrypted)

    assert exc_info.value.code == DocumentErrorCode.ENCRYPTED_PDF


def test_scanned_pdf_without_text_is_detected(service: TextExtractionService) -> None:
    blank_pdf = build_blank_pdf_bytes(page_count=1)

    with pytest.raises(DocumentProcessingError) as exc_info:
        service.extract("scanned.pdf", blank_pdf)

    assert exc_info.value.code == DocumentErrorCode.SCANNED_PDF_NO_TEXT


def test_non_utf8_txt_is_rejected(service: TextExtractionService) -> None:
    non_utf8 = "café".encode("latin-1")  # invalid utf-8 byte sequence for "é"

    with pytest.raises(DocumentProcessingError) as exc_info:
        service.extract("notes.txt", non_utf8)

    assert exc_info.value.code == DocumentErrorCode.TEXT_DECODING_ERROR


def test_document_id_is_stable_for_identical_content(service: TextExtractionService) -> None:
    content = b"identical content"

    first = service.extract("a.txt", content)
    second = service.extract("b.txt", content)

    assert first.document.document_id == second.document.document_id


def test_document_id_differs_for_different_content(service: TextExtractionService) -> None:
    first = service.extract("a.txt", b"content one")
    second = service.extract("a.txt", b"content two")

    assert first.document.document_id != second.document.document_id


def test_page_metadata_is_preserved_across_pages(service: TextExtractionService) -> None:
    pdf_bytes = build_pdf_bytes(["page one", "page two"])

    result = service.extract("doc.pdf", pdf_bytes)

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
