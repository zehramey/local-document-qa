import fitz  # PyMuPDF

from app.domain.document import ExtractedPage
from app.domain.errors import DocumentErrorCode, DocumentProcessingError


class PdfTextExtractor:
    """Extracts per-page text from a PDF using PyMuPDF.

    Does not perform OCR: a PDF with no extractable text (e.g. a scanned
    image-only document) is rejected with a clear error instead of silently
    returning empty pages.
    """

    def extract(self, content: bytes) -> list[ExtractedPage]:
        try:
            document = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise DocumentProcessingError(
                DocumentErrorCode.CORRUPTED_FILE, f"PDF açılamadı veya bozuk: {exc}"
            ) from exc

        try:
            if document.is_encrypted or document.needs_pass:
                raise DocumentProcessingError(
                    DocumentErrorCode.ENCRYPTED_PDF,
                    "PDF şifreli/parola korumalı; bu doküman işlenemez.",
                )

            pages = [
                ExtractedPage(
                    page_number=index + 1,
                    text=document.load_page(index).get_text("text"),
                )
                for index in range(document.page_count)
            ]
        finally:
            document.close()

        if not any(page.text.strip() for page in pages):
            raise DocumentProcessingError(
                DocumentErrorCode.SCANNED_PDF_NO_TEXT,
                "PDF hiç metin içermiyor; taranmış (görüntü tabanlı) bir doküman "
                "olabilir. Bu fazda OCR desteklenmiyor.",
            )

        return pages
