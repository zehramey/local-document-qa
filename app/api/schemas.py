from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class DocumentSummaryResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    page_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummaryResponse]


class QuestionRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1)
    model_id: str | None = None


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    text: str
    retrieval_score: float
    reranker_score: float | None
    final_rank: int


class GenerationMetricsResponse(BaseModel):
    model_id: str
    quantization: str | None
    prompt_tokens: int
    output_tokens: int
    time_to_first_token_seconds: float | None
    total_duration_seconds: float
    tokens_per_second: float | None


class QuestionResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[CitationResponse]
    rejected_citation_ids: list[str]
    retrieved_chunks: list[RetrievedChunkResponse]
    metrics: GenerationMetricsResponse | None


class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    quantization: str | None
    state: str | None


class ModelListResponse(BaseModel):
    models: list[ModelInfoResponse]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
