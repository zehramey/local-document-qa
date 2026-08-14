from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_active_collection_name,
    get_chunk_repository,
    get_default_rag_answer_service,
    get_generation_config,
    get_llm_provider,
    get_retrieval_config,
    get_retrieval_service,
)
from app.api.schemas import (
    CitationResponse,
    GenerationMetricsResponse,
    QuestionRequest,
    QuestionResponse,
    RetrievedChunkResponse,
)
from app.domain.llm import GenerationConfig
from app.domain.retrieval import RetrievalConfig
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.rag_answer import RagAnswerService
from app.services.retrieval import RetrievalService

router = APIRouter(tags=["questions"])


@router.post("/questions", response_model=QuestionResponse)
def ask_question(
    request: QuestionRequest,
    repository: QdrantChunkRepository = Depends(get_chunk_repository),
    collection_name: str = Depends(get_active_collection_name),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    rag_service: RagAnswerService = Depends(get_default_rag_answer_service),
    retrieval_config: RetrievalConfig = Depends(get_retrieval_config),
    generation_config: GenerationConfig = Depends(get_generation_config),
) -> QuestionResponse:
    if repository.get_document(collection_name, request.document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    # A non-default model_id bypasses the (test-overridable) default
    # rag_service dependency and builds a fresh one for that model —
    # production-only path, not exercised by the default-model API tests.
    if request.model_id:
        rag_service = RagAnswerService(get_llm_provider(request.model_id))

    retrieval_result = retrieval_service.retrieve(
        request.question, request.document_id, retrieval_config
    )
    llm_chunks = retrieval_service.context_for_llm(retrieval_result, retrieval_config)
    rag_answer = rag_service.answer(request.question, retrieval_result, generation_config)

    metrics = rag_answer.metrics
    return QuestionResponse(
        answer=rag_answer.answer,
        answerable=rag_answer.answerable,
        citations=[
            CitationResponse(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                page_start=c.page_start,
                page_end=c.page_end,
            )
            for c in rag_answer.citations
        ],
        rejected_citation_ids=rag_answer.rejected_citation_ids,
        retrieved_chunks=[
            RetrievedChunkResponse(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                retrieval_score=chunk.retrieval_score,
                reranker_score=chunk.reranker_score,
                final_rank=chunk.final_rank,
            )
            for chunk in llm_chunks
        ],
        metrics=(
            GenerationMetricsResponse(
                model_id=metrics.model_id,
                quantization=metrics.quantization,
                prompt_tokens=metrics.prompt_tokens,
                output_tokens=metrics.output_tokens,
                time_to_first_token_seconds=metrics.time_to_first_token_seconds,
                total_duration_seconds=metrics.total_duration_seconds,
                tokens_per_second=metrics.tokens_per_second,
            )
            if metrics is not None
            else None
        ),
    )
