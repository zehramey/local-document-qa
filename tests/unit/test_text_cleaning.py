from app.domain.document import ExtractedPage
from app.services.text_cleaning import TextCleaningService


def test_dehyphenates_line_break_split_words() -> None:
    service = TextCleaningService()
    pages = [ExtractedPage(page_number=1, text="This is infor-\nmation split across a line.")]

    cleaned = service.clean(pages)

    assert "infor-\nmation" not in cleaned[0].text
    assert "information" in cleaned[0].text


def test_collapses_excess_whitespace_and_blank_lines() -> None:
    service = TextCleaningService()
    pages = [ExtractedPage(page_number=1, text="too    many   spaces\n\n\n\nand blank lines")]

    cleaned = service.clean(pages)

    assert "too many spaces" in cleaned[0].text
    assert "\n\n\n" not in cleaned[0].text


def test_removes_repeated_header_and_footer_lines() -> None:
    service = TextCleaningService(header_footer_min_pages=3, header_footer_min_page_ratio=0.6)
    pages = [
        ExtractedPage(page_number=1, text="Company Confidential\nBody text page one\nFooter"),
        ExtractedPage(page_number=2, text="Company Confidential\nBody text page two\nFooter"),
        ExtractedPage(page_number=3, text="Company Confidential\nBody text page three\nFooter"),
    ]

    cleaned = service.clean(pages)

    for page in cleaned:
        assert "Company Confidential" not in page.text
        assert "Footer" not in page.text
    assert "Body text page one" in cleaned[0].text
    assert "Body text page two" in cleaned[1].text
    assert "Body text page three" in cleaned[2].text


def test_does_not_remove_header_footer_below_page_threshold() -> None:
    service = TextCleaningService(header_footer_min_pages=3, header_footer_min_page_ratio=0.6)
    pages = [
        ExtractedPage(page_number=1, text="Repeated Line\nUnique body one"),
        ExtractedPage(page_number=2, text="Repeated Line\nUnique body two"),
    ]

    cleaned = service.clean(pages)

    assert "Repeated Line" in cleaned[0].text
    assert "Repeated Line" in cleaned[1].text


def test_preserves_page_numbers_while_cleaning() -> None:
    service = TextCleaningService()
    pages = [
        ExtractedPage(page_number=1, text="first"),
        ExtractedPage(page_number=2, text="second"),
    ]

    cleaned = service.clean(pages)

    assert [page.page_number for page in cleaned] == [1, 2]
