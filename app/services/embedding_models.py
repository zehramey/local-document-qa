"""Per-model query/passage instruction conventions.

These are facts taken from each model's published model card, not our own
benchmarked findings — kept separate from any in-house experiment results
per project policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelPreset:
    model_id: str
    query_instruction: str
    passage_instruction: str


# Per the BAAI/bge-m3 model card: general-purpose retrieval use does not
# require a special query or passage instruction prefix (unlike the base
# intfloat/e5 family, which does).
BGE_M3 = EmbeddingModelPreset(model_id="BAAI/bge-m3", query_instruction="", passage_instruction="")

# intfloat/multilingual-e5-large-instruct is deferred to the benchmark phase
# (Faz 4 scope: baseline only). Its instruction convention will be added
# here once that model is actually integrated.
