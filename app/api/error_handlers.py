"""Maps domain errors to clean JSON responses.

Every handler returns {error_code, message} built from the exception's own
`.code`/`.message` — never a Python traceback or `str(exc)` of an
unexpected exception, which could leak internal paths/implementation
details to the client.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.domain.citation import CitationError
from app.domain.embedding import EmbeddingError, EmbeddingErrorCode
from app.domain.errors import DocumentProcessingError
from app.domain.indexing import IndexingError, IndexingErrorCode
from app.domain.llm import LlmError, LlmErrorCode
from app.domain.rag_answer import RagAnswerError
from app.domain.reranking import RerankerError, RerankerErrorCode

logger = logging.getLogger(__name__)

_INDEXING_STATUS = {
    IndexingErrorCode.QDRANT_UNAVAILABLE: 503,
    IndexingErrorCode.VECTOR_DIMENSION_MISMATCH: 500,
    IndexingErrorCode.UPSERT_FAILED: 502,
    IndexingErrorCode.DELETE_FAILED: 502,
}
_LLM_STATUS = {
    LlmErrorCode.SERVER_UNAVAILABLE: 503,
    LlmErrorCode.TIMEOUT: 504,
    LlmErrorCode.MALFORMED_RESPONSE: 502,
}
_EMBEDDING_STATUS = {
    EmbeddingErrorCode.EMPTY_TEXT: 400,
    EmbeddingErrorCode.MODEL_LOAD_FAILED: 503,
}
_RERANKER_STATUS = {
    RerankerErrorCode.DEVICE_UNAVAILABLE: 503,
    RerankerErrorCode.MODEL_LOAD_FAILED: 503,
}


def _error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error_code": error_code, "message": message}
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocumentProcessingError)
    async def _document_processing_error(_: Request, exc: DocumentProcessingError) -> JSONResponse:
        return _error_response(400, exc.code.value, exc.message)

    @app.exception_handler(EmbeddingError)
    async def _embedding_error(_: Request, exc: EmbeddingError) -> JSONResponse:
        return _error_response(_EMBEDDING_STATUS.get(exc.code, 500), exc.code.value, exc.message)

    @app.exception_handler(IndexingError)
    async def _indexing_error(_: Request, exc: IndexingError) -> JSONResponse:
        return _error_response(_INDEXING_STATUS.get(exc.code, 500), exc.code.value, exc.message)

    @app.exception_handler(LlmError)
    async def _llm_error(_: Request, exc: LlmError) -> JSONResponse:
        return _error_response(_LLM_STATUS.get(exc.code, 500), exc.code.value, exc.message)

    @app.exception_handler(RerankerError)
    async def _reranker_error(_: Request, exc: RerankerError) -> JSONResponse:
        return _error_response(_RERANKER_STATUS.get(exc.code, 500), exc.code.value, exc.message)

    @app.exception_handler(RagAnswerError)
    async def _rag_answer_error(_: Request, exc: RagAnswerError) -> JSONResponse:
        return _error_response(502, exc.code.value, exc.message)

    @app.exception_handler(CitationError)
    async def _citation_error(_: Request, exc: CitationError) -> JSONResponse:
        return _error_response(500, exc.code.value, exc.message)

    @app.exception_handler(Exception)
    async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error handling request")
        return _error_response(500, "internal_error", "İç sunucu hatası oluştu.")
