from app.domain.chunk import ChunkingConfig
from app.domain.document import Document, DocumentType, ExtractedPage
from app.services.chunking import ChunkingService
from app.services.tokenization import ApproximateTokenizer, tail_by_token_count

_TOKENIZER = ApproximateTokenizer()


def _make_document(document_id: str = "doc-id", filename: str = "test.txt") -> Document:
    return Document(
        document_id=document_id, filename=filename, document_type=DocumentType.TXT, size_bytes=100
    )


def test_max_token_limit_is_respected_by_every_chunk() -> None:
    words = " ".join(f"word{i}" for i in range(300))
    page = ExtractedPage(page_number=1, text=words)
    config = ChunkingConfig(max_tokens=50, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page], config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= config.max_tokens


def test_overlap_prepends_trailing_tokens_of_previous_chunk() -> None:
    para1 = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet."
    para2 = "Kilo lima mike november oscar papa quebec romeo sierra tango."
    page = ExtractedPage(page_number=1, text=f"{para1}\n\n{para2}")
    max_tokens = _TOKENIZER.count_tokens(para1)
    config = ChunkingConfig(max_tokens=max_tokens, overlap_tokens=3, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page], config)

    assert len(chunks) == 2
    expected_overlap = tail_by_token_count(para1, 3, _TOKENIZER)
    assert chunks[1].text.startswith(expected_overlap)


def test_paragraph_boundary_is_preferred_over_mid_paragraph_split() -> None:
    para1 = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet."
    para2 = "Kilo lima mike november oscar papa quebec romeo sierra tango."
    page = ExtractedPage(page_number=1, text=f"{para1}\n\n{para2}")
    max_tokens = _TOKENIZER.count_tokens(para1)
    config = ChunkingConfig(max_tokens=max_tokens, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page], config)

    assert len(chunks) == 2
    assert chunks[0].text == para1
    assert chunks[1].text == para2


def test_sentence_boundary_is_used_when_paragraph_exceeds_max_tokens() -> None:
    sentences = [
        "Bir cumle burada.",
        "Iki cumle burada.",
        "Uc cumle burada.",
        "Dort cumle burada.",
    ]
    paragraph_text = " ".join(sentences)
    page = ExtractedPage(page_number=1, text=paragraph_text)
    sentence_tokens = _TOKENIZER.count_tokens(sentences[0])
    config = ChunkingConfig(max_tokens=sentence_tokens * 2, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page], config)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.text.endswith(".")
    combined = " ".join(chunk.text.replace("\n\n", " ") for chunk in chunks)
    for sentence in sentences:
        assert sentence in combined


def test_very_short_paragraph_is_merged_with_neighboring_chunk() -> None:
    long_paragraph = " ".join(f"filler{i}" for i in range(60)) + "."
    text = f"Hi.\n\n# Big Heading\n\n{long_paragraph}"
    page = ExtractedPage(page_number=1, text=text)
    config = ChunkingConfig(max_tokens=500, overlap_tokens=0, min_tokens=10)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page], config)

    assert len(chunks) == 1
    assert "Hi." in chunks[0].text
    assert "Big Heading" in chunks[0].text
    assert chunks[0].section_title is not None
    assert "Big Heading" in chunks[0].section_title


def test_long_single_paragraph_is_split_at_token_boundary() -> None:
    words = [f"word{i}" for i in range(500)]
    paragraph_text = " ".join(words) + "."
    page = ExtractedPage(page_number=1, text=paragraph_text)
    config = ChunkingConfig(max_tokens=50, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page], config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= config.max_tokens
    assert "word0" in chunks[0].text
    all_text = " ".join(chunk.text for chunk in chunks)
    assert "word499" in all_text


def test_chunk_can_span_multiple_pages() -> None:
    page1 = ExtractedPage(page_number=1, text="Short text on page one.")
    page2 = ExtractedPage(page_number=2, text="Short text on page two.")
    config = ChunkingConfig(max_tokens=500, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page1, page2], config)

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_list_items_are_kept_together_in_one_chunk() -> None:
    text = "Intro paragraph before the list.\n\n" + "\n".join(f"- item {i}" for i in range(5))
    page = ExtractedPage(page_number=1, text=text)
    config = ChunkingConfig(max_tokens=500, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page], config)

    assert len(chunks) == 1
    for i in range(5):
        assert f"item {i}" in chunks[0].text


def test_empty_page_is_skipped_without_crashing() -> None:
    page1 = ExtractedPage(page_number=1, text="Content on page one.")
    page2 = ExtractedPage(page_number=2, text="")
    page3 = ExtractedPage(page_number=3, text="Content on page three.")
    config = ChunkingConfig(max_tokens=500, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), [page1, page2, page3], config)

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 3


def test_all_empty_pages_produce_no_chunks() -> None:
    pages = [ExtractedPage(page_number=1, text=""), ExtractedPage(page_number=2, text="   ")]
    config = ChunkingConfig(max_tokens=500, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()

    chunks = service.chunk_document(_make_document(), pages, config)

    assert chunks == []


def test_chunk_id_is_stable_across_runs() -> None:
    page = ExtractedPage(page_number=1, text="Some repeatable content for id stability testing.")
    config = ChunkingConfig(max_tokens=10, overlap_tokens=2, min_tokens=0)
    service = ChunkingService()
    document = _make_document()

    first_run = service.chunk_document(document, [page], config)
    second_run = service.chunk_document(document, [page], config)

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]


def test_chunk_metadata_is_preserved() -> None:
    page = ExtractedPage(page_number=1, text="Metadata preservation content for the test.")
    config = ChunkingConfig(max_tokens=500, overlap_tokens=0, min_tokens=0)
    service = ChunkingService()
    document = _make_document(document_id="abc123", filename="report.pdf")

    chunks = service.chunk_document(document, [page], config)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "abc123"
    assert chunk.filename == "report.pdf"
    assert chunk.chunk_index == 0
    assert chunk.chunking_strategy == config.strategy_name
    assert chunk.chunking_version == config.version


def test_identical_input_produces_identical_output() -> None:
    page = ExtractedPage(page_number=1, text="Deterministic output content for the test run.")
    config = ChunkingConfig(max_tokens=8, overlap_tokens=2, min_tokens=0)
    service = ChunkingService()
    document = _make_document()

    first_run = service.chunk_document(document, [page], config)
    second_run = service.chunk_document(document, [page], config)

    assert first_run == second_run
