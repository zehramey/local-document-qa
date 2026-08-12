"""Real embedding provider, backed by sentence-transformers.

The model is loaded once in `__init__` and reused for every call — it is
never reloaded per request. `sentence-transformers`/`torch` are optional
dependencies (see pyproject `[project.optional-dependencies].embeddings`);
the import happens lazily here so the rest of the app works without them
installed. This class is exercised only by the marked integration test
(tests/integration) — unit tests use FakeEmbeddingProvider instead.
"""

from app.domain.embedding import EmbeddingModelInfo
from app.services.embedding_provider import ensure_non_empty_texts


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        revision: str | None = None,
        query_instruction: str = "",
        passage_instruction: str = "",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers kurulu değil. Gerçek embedding modeli için "
                '`pip install -e ".[embeddings]"` çalıştırın.'
            ) from exc

        self._model = SentenceTransformer(model_id, revision=revision, device=device)
        self._query_instruction = query_instruction
        self._passage_instruction = passage_instruction
        self._normalize = normalize
        self._batch_size = batch_size
        dimension = int(self._model.get_sentence_embedding_dimension())
        self._info = EmbeddingModelInfo(
            model_id=model_id,
            revision=revision or "main",
            vector_dimension=dimension,
            normalized=normalize,
        )

    @property
    def model_info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ensure_non_empty_texts(texts)
        prefixed = [f"{self._passage_instruction}{text}" for text in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        ensure_non_empty_texts([text])
        prefixed = f"{self._query_instruction}{text}"
        vector = self._model.encode(
            prefixed, normalize_embeddings=self._normalize, convert_to_numpy=True
        )
        return [float(value) for value in vector]
