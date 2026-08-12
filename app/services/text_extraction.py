from app.domain.document import Document, DocumentType, ExtractionResult
from app.services.document_id import compute_document_id
from app.services.file_validation import FileValidator
from app.services.pdf_extraction import PdfTextExtractor
from app.services.txt_extraction import TxtTextExtractor


class TextExtractionService:
    """Validates an uploaded file and extracts its raw, per-page text.

    Text cleaning (header/footer removal, whitespace normalization) is a
    separate step — see TextCleaningService.
    """

    def __init__(
        self,
        validator: FileValidator,
        pdf_extractor: PdfTextExtractor | None = None,
        txt_extractor: TxtTextExtractor | None = None,
    ) -> None:
        self._validator = validator
        self._pdf_extractor = pdf_extractor or PdfTextExtractor()
        self._txt_extractor = txt_extractor or TxtTextExtractor()

    def extract(self, filename: str, content: bytes) -> ExtractionResult:
        document_type = self._validator.validate(filename, content)

        pages = (
            self._pdf_extractor.extract(content)
            if document_type == DocumentType.PDF
            else self._txt_extractor.extract(content)
        )

        document = Document(
            document_id=compute_document_id(content),
            filename=filename,
            document_type=document_type,
            size_bytes=len(content),
        )
        return ExtractionResult(document=document, pages=pages)
