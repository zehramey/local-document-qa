"""Structural chunking service.

Boundary preference, in order: heading > paragraph > sentence > hard token
cut. Bullet lists and (heuristically detected) tables are kept whole unless
they themselves exceed max_tokens. See DocumentBlockParser for the block
detection heuristics and ChunkingService for chunk assembly.
"""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum

from app.domain.chunk import Chunk, ChunkingConfig
from app.domain.document import Document, ExtractedPage
from app.services.tokenization import (
    ApproximateTokenizer,
    Tokenizer,
    cut_text_at_token_limit,
    split_sentences,
    tail_by_token_count,
)

_HEADING_MARKDOWN = re.compile(r"^#{1,6}\s+\S.*$")
_HEADING_NUMBERED = re.compile(r"^\d+(\.\d+)*\.?\s+\S.{0,100}$")
_LIST_ITEM = re.compile(r"^([-*•◦‣]|\d+[.)])\s+\S")
_TABLE_ROW = re.compile(r"\S(\s{2,})\S+(\s{2,})\S+")
_SENTENCE_END = (".", "!", "?", '"', "'")


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"


@dataclass(frozen=True)
class Block:
    block_type: BlockType
    text: str
    page_number: int


@dataclass(frozen=True)
class _Unit:
    """A piece of text guaranteed to fit within max_tokens on its own."""

    text: str
    page_number: int
    token_count: int
    is_heading: bool


@dataclass
class _RawChunk:
    texts: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    section_title: str | None = None
    token_count: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(self.texts)


def _is_heading(line: str) -> bool:
    if _HEADING_MARKDOWN.match(line):
        return True
    if len(line) <= 100 and not line.endswith(_SENTENCE_END) and _HEADING_NUMBERED.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and line.isupper() and len(line) <= 100


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM.match(line))


def _is_table_row(line: str) -> bool:
    return line.count("|") >= 2 or bool(_TABLE_ROW.search(line))


class DocumentBlockParser:
    """Splits each page's cleaned text into heading/paragraph/list/table blocks.

    Heuristic, line-based: table detection in particular is best-effort
    (multi-space-separated columns or pipe-delimited rows) since extraction
    yields plain text, not structured table data.
    """

    def parse(self, pages: list[ExtractedPage]) -> list[Block]:
        blocks: list[Block] = []
        for page in pages:
            blocks.extend(self._parse_page(page))
        return blocks

    def _parse_page(self, page: ExtractedPage) -> list[Block]:
        blocks: list[Block] = []
        paragraph_lines: list[str] = []
        list_lines: list[str] = []
        table_lines: list[str] = []

        def flush_paragraph() -> None:
            if paragraph_lines:
                text = " ".join(paragraph_lines)
                blocks.append(Block(BlockType.PARAGRAPH, text, page.page_number))
                paragraph_lines.clear()

        def flush_list() -> None:
            if list_lines:
                blocks.append(Block(BlockType.LIST, "\n".join(list_lines), page.page_number))
                list_lines.clear()

        def flush_table() -> None:
            if table_lines:
                blocks.append(Block(BlockType.TABLE, "\n".join(table_lines), page.page_number))
                table_lines.clear()

        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                flush_list()
                flush_table()
                continue
            if _is_heading(line):
                flush_paragraph()
                flush_list()
                flush_table()
                blocks.append(Block(BlockType.HEADING, line, page.page_number))
            elif _is_list_item(line):
                flush_paragraph()
                flush_table()
                list_lines.append(line)
            elif _is_table_row(line):
                flush_paragraph()
                flush_list()
                table_lines.append(line)
            else:
                flush_list()
                flush_table()
                paragraph_lines.append(line)

        flush_paragraph()
        flush_list()
        flush_table()
        return blocks


class ChunkingService:
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer or ApproximateTokenizer()
        self._parser = DocumentBlockParser()

    def chunk_document(
        self, document: Document, pages: list[ExtractedPage], config: ChunkingConfig
    ) -> list[Chunk]:
        blocks = self._parser.parse(pages)
        units = self._blocks_to_units(blocks, config.max_tokens)
        raw_chunks = self._pack_units(units, config.max_tokens)
        raw_chunks = self._merge_tiny_chunks(raw_chunks, config.min_tokens, config.max_tokens)
        raw_chunks = self._apply_overlap(raw_chunks, config.overlap_tokens)
        return self._finalize(document, raw_chunks, config)

    # -- block -> unit -------------------------------------------------

    def _blocks_to_units(self, blocks: list[Block], max_tokens: int) -> list[_Unit]:
        units: list[_Unit] = []
        for block in blocks:
            units.extend(self._block_to_units(block, max_tokens))
        return units

    def _block_to_units(self, block: Block, max_tokens: int) -> list[_Unit]:
        token_count = self._tokenizer.count_tokens(block.text)
        if token_count <= max_tokens:
            return [
                _Unit(
                    text=block.text,
                    page_number=block.page_number,
                    token_count=token_count,
                    is_heading=block.block_type == BlockType.HEADING,
                )
            ]

        if block.block_type == BlockType.PARAGRAPH:
            pieces = split_sentences(block.text) or [block.text]
        else:
            pieces = block.text.split("\n")

        units: list[_Unit] = []
        for piece in pieces:
            units.extend(self._piece_to_units(piece, block.page_number, max_tokens))
        return units

    def _piece_to_units(self, piece: str, page_number: int, max_tokens: int) -> list[_Unit]:
        token_count = self._tokenizer.count_tokens(piece)
        if token_count <= max_tokens:
            return [
                _Unit(
                    text=piece,
                    page_number=page_number,
                    token_count=token_count,
                    is_heading=False,
                )
            ]

        units: list[_Unit] = []
        remainder = piece
        while remainder:
            head, remainder = cut_text_at_token_limit(remainder, max_tokens, self._tokenizer)
            units.append(
                _Unit(
                    text=head,
                    page_number=page_number,
                    token_count=self._tokenizer.count_tokens(head),
                    is_heading=False,
                )
            )
        return units

    # -- unit -> raw chunk -----------------------------------------------

    def _pack_units(self, units: list[_Unit], max_tokens: int) -> list[_RawChunk]:
        raw_chunks: list[_RawChunk] = []
        current: _RawChunk | None = None
        current_section_title: str | None = None

        def flush() -> None:
            nonlocal current
            if current is not None and current.texts:
                raw_chunks.append(current)
            current = None

        for unit in units:
            if unit.is_heading:
                current_section_title = unit.text
                if current is not None and current.texts:
                    flush()

            if current is None:
                current = _RawChunk(page_start=unit.page_number, page_end=unit.page_number)
            elif current.token_count + unit.token_count > max_tokens and current.texts:
                flush()
                current = _RawChunk(page_start=unit.page_number, page_end=unit.page_number)

            current.texts.append(unit.text)
            current.token_count += unit.token_count
            current.page_end = max(current.page_end, unit.page_number)
            current.section_title = current_section_title

        flush()
        return raw_chunks

    # -- merge tiny chunks -------------------------------------------------

    def _merge_tiny_chunks(
        self, raw_chunks: list[_RawChunk], min_tokens: int, max_tokens: int
    ) -> list[_RawChunk]:
        if len(raw_chunks) <= 1:
            return raw_chunks

        merged: list[_RawChunk] = []
        for chunk in raw_chunks:
            if (
                chunk.token_count < min_tokens
                and merged
                and merged[-1].token_count + chunk.token_count <= max_tokens
            ):
                previous = merged[-1]
                previous.texts.extend(chunk.texts)
                previous.token_count = self._tokenizer.count_tokens(previous.text)
                previous.page_end = max(previous.page_end, chunk.page_end)
                previous.section_title = previous.section_title or chunk.section_title
            else:
                merged.append(chunk)

        if len(merged) >= 2 and merged[0].token_count < min_tokens:
            first, second = merged[0], merged[1]
            if first.token_count + second.token_count <= max_tokens:
                second.texts = first.texts + second.texts
                second.token_count = self._tokenizer.count_tokens(second.text)
                second.page_start = min(first.page_start, second.page_start)
                second.section_title = first.section_title or second.section_title
                merged.pop(0)

        return merged

    # -- overlap -------------------------------------------------------

    def _apply_overlap(self, raw_chunks: list[_RawChunk], overlap_tokens: int) -> list[_RawChunk]:
        if overlap_tokens <= 0 or len(raw_chunks) <= 1:
            return raw_chunks

        original_texts = [chunk.text for chunk in raw_chunks]
        for index in range(1, len(raw_chunks)):
            previous = raw_chunks[index - 1]
            current = raw_chunks[index]
            overlap_text = tail_by_token_count(
                original_texts[index - 1], overlap_tokens, self._tokenizer
            )
            if not overlap_text:
                continue
            current.texts = [overlap_text, *current.texts]
            current.token_count = self._tokenizer.count_tokens(current.text)
            current.page_start = min(current.page_start, previous.page_end)

        return raw_chunks

    # -- finalize -------------------------------------------------------

    def _finalize(
        self, document: Document, raw_chunks: list[_RawChunk], config: ChunkingConfig
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, raw in enumerate(raw_chunks):
            text = raw.text
            chunk_id = self._compute_chunk_id(document.document_id, config, index, text)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    filename=document.filename,
                    page_start=raw.page_start,
                    page_end=raw.page_end,
                    chunk_index=index,
                    section_title=raw.section_title,
                    token_count=self._tokenizer.count_tokens(text),
                    text=text,
                    chunking_strategy=config.strategy_name,
                    chunking_version=config.version,
                )
            )
        return chunks

    @staticmethod
    def _compute_chunk_id(document_id: str, config: ChunkingConfig, index: int, text: str) -> str:
        payload = (
            f"{document_id}:{config.strategy_name}:{config.version}:"
            f"{config.max_tokens}:{config.overlap_tokens}:{index}:{text}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
