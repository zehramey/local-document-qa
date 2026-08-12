from app.domain.document import ExtractedPage
from app.domain.errors import DocumentErrorCode, DocumentProcessingError


class TxtTextExtractor:
    """Decodes a plain-text file as strict UTF-8.

    Files in other encodings are rejected with a clear error rather than
    silently mangled via a lossy decode.
    """

    def extract(self, content: bytes) -> list[ExtractedPage]:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError(
                DocumentErrorCode.TEXT_DECODING_ERROR,
                f"TXT dosyası UTF-8 olarak decode edilemedi: {exc}",
            ) from exc

        return [ExtractedPage(page_number=1, text=text)]
