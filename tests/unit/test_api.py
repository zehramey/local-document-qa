import pytest
from app.api import dependencies as deps
from app.domain.chunk import ChunkingConfig
from app.domain.llm import LlmError, LlmErrorCode
from app.main import app
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.document_pipeline import DocumentIngestionService
from app.services.embedding_provider import FakeEmbeddingProvider
from app.services.file_validation import FileValidationConfig, FileValidator
from app.services.indexing import IndexingService
from app.services.llm_provider import FakeLlmProvider
from app.services.rag_answer import RagAnswerService
from app.services.retrieval import RetrievalService
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

_ANSWER_JSON = '{"answer": "test answer", "answerable": true, "cited_chunk_ids": []}'


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(vector_dimension=8)


@pytest.fixture
def repository() -> QdrantChunkRepository:
    return QdrantChunkRepository(QdrantClient(location=":memory:"))


@pytest.fixture
def client(fake_provider: FakeEmbeddingProvider, repository: QdrantChunkRepository) -> TestClient:
    indexing_service = IndexingService(fake_provider, repository)
    ingestion_service = DocumentIngestionService(
        validator=FileValidator(FileValidationConfig(max_size_bytes=1024 * 1024)),
        indexing_service=indexing_service,
        chunking_config=ChunkingConfig(max_tokens=100, overlap_tokens=10),
    )
    retrieval_service = RetrievalService(fake_provider, repository)
    rag_service = RagAnswerService(FakeLlmProvider(response_text=_ANSWER_JSON))

    app.dependency_overrides[deps.get_embedding_provider] = lambda: fake_provider
    app.dependency_overrides[deps.get_chunk_repository] = lambda: repository
    app.dependency_overrides[deps.get_document_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[deps.get_indexing_service] = lambda: indexing_service
    app.dependency_overrides[deps.get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[deps.get_default_rag_answer_service] = lambda: rag_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _upload_sample_document(client: TestClient) -> str:
    response = client.post(
        "/documents",
        files={
            "file": (
                "notes.txt",
                b"This is a test document. It provides information about the content.",
                "text/plain",
            )
        },
    )
    assert response.status_code == 201
    return response.json()["document_id"]


def test_api_health_check(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_document_upload_succeeds(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"This is a test document.", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"]
    assert body["filename"] == "notes.txt"
    assert body["chunk_count"] > 0


def test_unsupported_file_type_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("notes.docx", b"some bytes", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "unsupported_file_type"


def test_question_endpoint_returns_answer(client: TestClient) -> None:
    document_id = _upload_sample_document(client)

    response = client.post(
        "/questions", json={"document_id": document_id, "question": "What is this document about?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "test answer"
    assert body["answerable"] is True


def test_question_for_unknown_document_returns_404(client: TestClient) -> None:
    response = client.post(
        "/questions", json={"document_id": "f" * 64, "question": "question"}
    )

    assert response.status_code == 404


def test_get_unknown_document_returns_404(client: TestClient) -> None:
    response = client.get(f"/documents/{'f' * 64}")

    assert response.status_code == 404


def test_document_delete_removes_document(client: TestClient) -> None:
    document_id = _upload_sample_document(client)

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/documents/{document_id}")
    assert get_response.status_code == 404


def test_model_unavailable_returns_service_unavailable(client: TestClient) -> None:
    document_id = _upload_sample_document(client)
    broken_rag_service = RagAnswerService(
        FakeLlmProvider(
            raises=LlmError(LlmErrorCode.SERVER_UNAVAILABLE, "could not reach the LLM server")
        )
    )
    app.dependency_overrides[deps.get_default_rag_answer_service] = lambda: broken_rag_service

    response = client.post(
        "/questions", json={"document_id": document_id, "question": "question"}
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "server_unavailable"


def test_qdrant_unavailable_returns_service_unavailable() -> None:
    unreachable_repository = QdrantChunkRepository(
        QdrantClient(url="http://127.0.0.1:1", timeout=1)
    )
    fake_provider = FakeEmbeddingProvider(vector_dimension=8)

    app.dependency_overrides[deps.get_embedding_provider] = lambda: fake_provider
    app.dependency_overrides[deps.get_chunk_repository] = lambda: unreachable_repository

    with TestClient(app) as test_client:
        response = test_client.get("/documents")

    app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error_code"] == "qdrant_unavailable"
