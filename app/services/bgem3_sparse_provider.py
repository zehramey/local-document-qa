"""Real sparse embedding provider, backed by BAAI/bge-m3's lexical output.

`sentence-transformers` (used for the dense path, see
sentence_transformer_provider.py) only exposes bge-m3's dense vector — its
unified dense+sparse+multi-vector output is specific to BAAI's own
FlagEmbedding library (BGEM3FlagModel). This is therefore a genuinely
separate optional dependency (pyproject `[project.optional-dependencies].hybrid`)
and a genuinely separate model load from SentenceTransformerEmbeddingProvider,
even though both ultimately wrap the same "BAAI/bge-m3" weights — running
hybrid search currently means bge-m3 is loaded into memory twice (once per
library). Imported lazily so the base install doesn't require FlagEmbedding.
"""

from app.domain.embedding import SparseVector
from app.services.embedding_provider import ensure_non_empty_texts


class Bgem3SparseEmbeddingProvider:
    def __init__(self, model_id: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is not installed. For hybrid (dense+sparse) search, run "
                '`pip install -e ".[hybrid]"`.'
            ) from exc

        self._model = BGEM3FlagModel(model_id, device=device, use_fp16=False)

    def embed_documents_sparse(self, texts: list[str]) -> list[SparseVector]:
        ensure_non_empty_texts(texts)
        output = self._model.encode(
            texts, return_dense=False, return_sparse=True, return_colbert_vecs=False
        )
        return [self._to_sparse_vector(weights) for weights in output["lexical_weights"]]

    def embed_query_sparse(self, text: str) -> SparseVector:
        return self.embed_documents_sparse([text])[0]

    @staticmethod
    def _to_sparse_vector(lexical_weights: dict[str, float]) -> SparseVector:
        """lexical_weights maps a token id (as a string key) to its weight —
        BGEM3FlagModel's own output shape, not a Qdrant convention."""
        items = sorted(
            (int(token_id), float(weight)) for token_id, weight in lexical_weights.items()
        )
        return SparseVector(indices=[i for i, _ in items], values=[w for _, w in items])
