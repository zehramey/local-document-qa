"""Real, multilingual reranker backed by a sentence-transformers CrossEncoder.

The model is loaded once in `__init__` and reused for every call. Hardware
compatibility is validated up front: requesting a CUDA device on a machine
without a working CUDA-enabled torch build raises a clear `RerankerError`
instead of failing deep inside the model call. `sentence-transformers`/
`torch` are optional (see pyproject `[project.optional-dependencies].embeddings`);
imported lazily so the base install doesn't require them. Exercised only by
the marked integration test — unit tests use FakeReranker instead.
"""

from dataclasses import replace

from app.domain.reranking import RerankerError, RerankerErrorCode, RerankerModelInfo
from app.domain.retrieval import RetrievedChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        revision: str | None = None,
        batch_size: int = 16,
    ) -> None:
        if device.startswith("cuda"):
            self._ensure_cuda_available(device)

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. For the real reranker model, run "
                '`pip install -e ".[embeddings]"`.'
            ) from exc

        try:
            self._model = CrossEncoder(model_id, revision=revision, device=device)
        except Exception as exc:
            raise RerankerError(
                RerankerErrorCode.MODEL_LOAD_FAILED, f"Reranker model could not be loaded: {exc}"
            ) from exc

        self._info = RerankerModelInfo(model_id=model_id, revision=revision or "main")
        self._batch_size = batch_size

    @staticmethod
    def _ensure_cuda_available(device: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RerankerError(
                RerankerErrorCode.DEVICE_UNAVAILABLE,
                "torch is not installed; a CUDA device cannot be used.",
            ) from exc
        if not torch.cuda.is_available():
            raise RerankerError(
                RerankerErrorCode.DEVICE_UNAVAILABLE,
                f"'{device}' was requested but CUDA is not available on this machine.",
            )

    @property
    def model_info(self) -> RerankerModelInfo:
        return self._info

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs, batch_size=self._batch_size)
        rescored = [
            replace(chunk, reranker_score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        return sorted(rescored, key=lambda chunk: chunk.reranker_score or 0.0, reverse=True)
