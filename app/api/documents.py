from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.dependencies import (
    get_active_collection_name,
    get_chunk_repository,
    get_document_ingestion_service,
    get_indexing_service,
)
from app.api.schemas import DocumentListResponse, DocumentSummaryResponse, DocumentUploadResponse
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.document_pipeline import DocumentIngestionService
from app.services.indexing import IndexingService

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    ingestion_service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentUploadResponse:
    content = await file.read()
    result = ingestion_service.ingest(file.filename or "file", content)
    return DocumentUploadResponse(
        document_id=result.document_id, filename=result.filename, chunk_count=result.chunk_count
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    repository: QdrantChunkRepository = Depends(get_chunk_repository),
    collection_name: str = Depends(get_active_collection_name),
) -> DocumentListResponse:
    summaries = repository.list_documents(collection_name)
    return DocumentListResponse(
        documents=[
            DocumentSummaryResponse(
                document_id=s.document_id,
                filename=s.filename,
                chunk_count=s.chunk_count,
                page_count=s.page_count,
            )
            for s in summaries
        ]
    )


@router.get("/documents/{document_id}", response_model=DocumentSummaryResponse)
def get_document(
    document_id: str,
    repository: QdrantChunkRepository = Depends(get_chunk_repository),
    collection_name: str = Depends(get_active_collection_name),
) -> DocumentSummaryResponse:
    summary = repository.get_document(collection_name, document_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentSummaryResponse(
        document_id=summary.document_id,
        filename=summary.filename,
        chunk_count=summary.chunk_count,
        page_count=summary.page_count,
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    repository: QdrantChunkRepository = Depends(get_chunk_repository),
    collection_name: str = Depends(get_active_collection_name),
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> None:
    if repository.get_document(collection_name, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    indexing_service.delete_document(document_id)
