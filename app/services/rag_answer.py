"""Turns retrieved chunks + a question into a validated RagAnswer.

If retrieval found nothing (RetrievalResult.has_results is False), the LLM
is never called — the fixed not-found phrase is returned directly, which
is both cheaper and more honest than asking a model to reason over an
empty context.

The model may only *report* which chunk_ids it used; CitationValidator is
the actual authority on whether each cited id was really retrieved for
this document/query. Fabricated or out-of-scope ids are silently dropped
from `citations` and reported in `rejected_citation_ids` instead of being
shown to the user as if they were real sources.
"""

import json
import re

from app.domain.citation import Citation, CitationError
from app.domain.llm import GenerationConfig
from app.domain.rag_answer import RagAnswer, RagAnswerError, RagAnswerErrorCode
from app.domain.retrieval import RetrievalResult, RetrievedChunk
from app.services.citation import CitationValidator
from app.services.llm_provider import LlmProvider
from app.services.rag_prompt import NOT_FOUND_PHRASE, build_rag_prompt
from app.services.tokenization import ApproximateTokenizer, Tokenizer

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class RagAnswerService:
    def __init__(self, llm: LlmProvider, tokenizer: Tokenizer | None = None) -> None:
        self._llm = llm
        self._tokenizer = tokenizer or ApproximateTokenizer()

    def answer(
        self, question: str, retrieval_result: RetrievalResult, config: GenerationConfig
    ) -> RagAnswer:
        if not retrieval_result.has_results:
            return RagAnswer(
                answer=NOT_FOUND_PHRASE,
                answerable=False,
                citations=[],
                rejected_citation_ids=[],
                metrics=None,
            )

        context_chunks = self._fit_to_context_limit(
            retrieval_result.chunks, config.context_token_limit
        )
        prompt = build_rag_prompt(question, context_chunks)
        generation = self._llm.generate(prompt, config)

        parsed = self._parse_json(generation.text)
        citations, rejected = self._validate_citations(
            parsed["cited_chunk_ids"], context_chunks, retrieval_result.document_id
        )

        return RagAnswer(
            answer=str(parsed["answer"]),
            answerable=bool(parsed["answerable"]),
            citations=citations,
            rejected_citation_ids=rejected,
            metrics=generation.metrics,
        )

    def _fit_to_context_limit(
        self, chunks: list[RetrievedChunk], context_token_limit: int
    ) -> list[RetrievedChunk]:
        """Keeps chunks, in order, until the token budget is used up.

        Always keeps at least the first chunk even if it alone exceeds the
        budget — returning zero context over a strict limit would be worse.
        """
        kept: list[RetrievedChunk] = []
        total_tokens = 0
        for chunk in chunks:
            chunk_tokens = self._tokenizer.count_tokens(chunk.text)
            if kept and total_tokens + chunk_tokens > context_token_limit:
                break
            kept.append(chunk)
            total_tokens += chunk_tokens
        return kept

    @staticmethod
    def _validate_citations(
        cited_chunk_ids: object, context_chunks: list[RetrievedChunk], document_id: str
    ) -> tuple[list[Citation], list[str]]:
        validator = CitationValidator(context_chunks, expected_document_id=document_id)
        citations: list[Citation] = []
        rejected: list[str] = []
        for chunk_id in cited_chunk_ids if isinstance(cited_chunk_ids, list) else []:
            try:
                citations.append(validator.validate(str(chunk_id)))
            except CitationError:
                rejected.append(str(chunk_id))
        return citations, rejected

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, object]:
        match = _JSON_BLOCK.search(raw_text)
        candidate = match.group(0) if match else raw_text
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise RagAnswerError(
                RagAnswerErrorCode.MALFORMED_JSON,
                f"LLM çıktısı geçerli JSON değil: {exc}",
                raw_text=raw_text,
            ) from exc

        if not isinstance(parsed, dict):
            raise RagAnswerError(
                RagAnswerErrorCode.MALFORMED_JSON,
                "LLM çıktısı bir JSON nesnesi (object) değil.",
                raw_text=raw_text,
            )

        for field in ("answer", "answerable", "cited_chunk_ids"):
            if field not in parsed:
                raise RagAnswerError(
                    RagAnswerErrorCode.MISSING_REQUIRED_FIELD,
                    f"LLM çıktısında '{field}' alanı eksik.",
                    raw_text=raw_text,
                )
        return parsed
