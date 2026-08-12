"""End-to-end RAG demonstration with real models: BGE-M3 embeddings, an
embedded (on-disk, no server) Qdrant instance, and a real local LLM served
by LM Studio (see app.services.lm_studio_provider).

Excluded from the default test run (see pyproject `addopts`) because it
needs `pip install -e ".[embeddings]"` and a running LM Studio server with
the configured model loaded (`Settings.llm_model_id`, default
"qwen/qwen3.5-4b") at `Settings.llm_base_url`.

Run explicitly with: pytest -m integration -s
(the -s is needed to see the printed answer/citations)

Runs three real questions against a small sample document: one directly
answerable, one a paraphrase of the same fact, and one about something the
document never mentions.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from app.domain.chunk import ChunkingConfig
from app.domain.llm import GenerationConfig
from app.domain.retrieval import RetrievalConfig
from app.repositories.qdrant_chunk_repository import QdrantChunkRepository
from app.services.chunking import ChunkingService
from app.services.embedding_models import BGE_M3
from app.services.file_validation import FileValidationConfig, FileValidator
from app.services.indexing import IndexingService
from app.services.lm_studio_provider import LMStudioProvider
from app.services.rag_answer import RagAnswerService
from app.services.retrieval import RetrievalService
from app.services.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
from app.services.text_extraction import TextExtractionService
from qdrant_client import QdrantClient

pytestmark = pytest.mark.integration

_SAMPLE_DOCUMENT = """Giriş

Local Document QA, kullanıcıların kendi PDF ve TXT dosyalarına soru \
sorabildiği, tamamen yerel çalışan bir sistemdir.

Metin Çıkarma

Sistem, PDF dosyalarından metni PyMuPDF kütüphanesi kullanarak sayfa \
sayfa çıkarır. Her sayfanın numarası korunur.

Parçalama

Çıkarılan metin, başlık ve paragraf sınırları önceliklendirilerek token \
tabanlı parçalara (chunk) bölünür.

Embedding ve Arama

Her chunk, BAAI/bge-m3 modeliyle vektöre çevrilir ve Qdrant içinde \
saklanır. Kullanıcı sorusu geldiğinde, cosine similarity ile en alakalı \
chunk'lar bulunur.

Sınırlamalar

Bu sistemde OCR desteği yoktur; taranmış (görüntü tabanlı) PDF'ler \
işlenemez.
"""

_QUESTIONS = [
    ("dogrudan", "Sistem PDF dosyalarından metni hangi kütüphaneyle çıkarır?"),
    ("paraphrase", "Doküman içeriğini okumak için hangi teknoloji kullanılıyor?"),
    ("dokumanda_yok", "Sistemin fiyatlandırma modeli nedir?"),
]


def test_three_real_questions_against_real_models() -> None:
    embedding_provider = SentenceTransformerEmbeddingProvider(
        model_id=BGE_M3.model_id, device="cpu"
    )
    llm_provider = LMStudioProvider(model_id="qwen/qwen3.5-4b")
    assert llm_provider.health_check(), "LM Studio sunucusu veya model yüklü değil"

    storage_dir = Path(tempfile.mkdtemp(prefix="rag_pipeline_integration_"))
    try:
        repository = QdrantChunkRepository(QdrantClient(path=str(storage_dir)))
        indexing_service = IndexingService(embedding_provider, repository)
        retrieval_service = RetrievalService(embedding_provider, repository)
        rag_service = RagAnswerService(llm_provider)

        validator = FileValidator(FileValidationConfig(max_size_bytes=1024 * 1024))
        extraction_result = TextExtractionService(validator).extract(
            "local_document_qa_intro.txt", _SAMPLE_DOCUMENT.encode("utf-8")
        )
        chunks = ChunkingService().chunk_document(
            extraction_result.document,
            extraction_result.pages,
            ChunkingConfig(max_tokens=120, overlap_tokens=20),
        )
        indexing_service.index_document(extraction_result.document, chunks)

        retrieval_config = RetrievalConfig(top_k=5, llm_top_k=3)
        generation_config = GenerationConfig(temperature=0.0, top_p=1.0, max_new_tokens=512)

        for label, question in _QUESTIONS:
            retrieval_result = retrieval_service.retrieve(
                question, extraction_result.document.document_id, retrieval_config
            )
            rag_answer = rag_service.answer(question, retrieval_result, generation_config)

            print(f"\n=== [{label}] Soru: {question} ===")
            print(f"Answerable: {rag_answer.answerable}")
            print(f"Cevap: {rag_answer.answer}")
            print(f"Retrieved chunk'lar ({len(retrieval_result.chunks)}):")
            for chunk in retrieval_result.chunks:
                preview = chunk.text[:80].replace("\n", " ")
                print(
                    f"  - chunk_id={chunk.chunk_id[:12]}... sayfa={chunk.page_start}-"
                    f"{chunk.page_end} score={chunk.retrieval_score:.3f} text='{preview}...'"
                )
            print(f"Citations ({len(rag_answer.citations)}):")
            for citation in rag_answer.citations:
                print(
                    f"  - chunk_id={citation.chunk_id[:12]}... "
                    f"dosya={citation.filename} sayfa={citation.page_start}-{citation.page_end}"
                )
            if rag_answer.rejected_citation_ids:
                print(f"Reddedilen citation'lar: {rag_answer.rejected_citation_ids}")
            metrics = rag_answer.metrics
            if metrics is not None:
                print(
                    f"Metrics: model={metrics.model_id} quant={metrics.quantization} "
                    f"prompt_tokens={metrics.prompt_tokens} output_tokens={metrics.output_tokens} "
                    f"ttft={metrics.time_to_first_token_seconds} "
                    f"total_s={metrics.total_duration_seconds:.2f} "
                    f"tok/s={metrics.tokens_per_second}"
                )

            if label == "dokumanda_yok":
                assert rag_answer.answerable is False
            else:
                assert rag_answer.answerable is True
                assert len(rag_answer.citations) > 0
    finally:
        shutil.rmtree(storage_dir, ignore_errors=True)
