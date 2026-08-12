"""Reranker model presets.

Model choice is a model-card fact (multilingual support), not our own
benchmarked finding — see project rule separating model-card claims from
in-house experiment results.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerModelPreset:
    model_id: str


# Per its model card, BAAI/bge-reranker-v2-m3 is a multilingual cross-encoder
# reranker built on the same backbone family as bge-m3.
BGE_RERANKER_V2_M3 = RerankerModelPreset(model_id="BAAI/bge-reranker-v2-m3")
