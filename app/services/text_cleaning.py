import re

from app.domain.document import ExtractedPage

_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")
_REPEATED_SPACES = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class TextCleaningService:
    """Removes repeated headers/footers and normalizes whitespace, page by page.

    Header/footer detection is frequency-based: a line is treated as a
    header/footer if it recurs on at least `header_footer_min_page_ratio` of
    the pages, and is only attempted when there are at least
    `header_footer_min_pages` pages (a single page gives no repetition signal).
    """

    def __init__(
        self,
        header_footer_min_pages: int = 3,
        header_footer_min_page_ratio: float = 0.6,
    ) -> None:
        self._min_pages = header_footer_min_pages
        self._min_ratio = header_footer_min_page_ratio

    def clean(self, pages: list[ExtractedPage]) -> list[ExtractedPage]:
        header_footer_lines = self._detect_header_footer_lines(pages)

        cleaned = []
        for page in pages:
            text = self._strip_header_footer_lines(page.text, header_footer_lines)
            text = self._dehyphenate(text)
            text = self._normalize_whitespace(text)
            cleaned.append(ExtractedPage(page_number=page.page_number, text=text))
        return cleaned

    def _detect_header_footer_lines(self, pages: list[ExtractedPage]) -> set[str]:
        if len(pages) < self._min_pages:
            return set()

        line_page_counts: dict[str, int] = {}
        for page in pages:
            distinct_lines = {line.strip() for line in page.text.splitlines() if line.strip()}
            for line in distinct_lines:
                line_page_counts[line] = line_page_counts.get(line, 0) + 1

        threshold = len(pages) * self._min_ratio
        return {line for line, count in line_page_counts.items() if count >= threshold}

    @staticmethod
    def _strip_header_footer_lines(text: str, header_footer_lines: set[str]) -> str:
        if not header_footer_lines:
            return text
        kept_lines = [line for line in text.splitlines() if line.strip() not in header_footer_lines]
        return "\n".join(kept_lines)

    @staticmethod
    def _dehyphenate(text: str) -> str:
        return _HYPHEN_LINE_BREAK.sub(r"\1\2", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = _REPEATED_SPACES.sub(" ", text)
        text = _EXCESS_BLANK_LINES.sub("\n\n", text)
        return text.strip()
