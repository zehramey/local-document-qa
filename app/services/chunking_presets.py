from app.domain.chunk import ChunkingConfig

CHUNKING_PRESETS: dict[str, ChunkingConfig] = {
    "300_50": ChunkingConfig(max_tokens=300, overlap_tokens=50),
    "500_75": ChunkingConfig(max_tokens=500, overlap_tokens=75),
    "800_100": ChunkingConfig(max_tokens=800, overlap_tokens=100),
}


def get_chunking_config(name: str) -> ChunkingConfig:
    """Looks up a named preset, for benchmark configs selected by name (e.g. from a CLI flag)."""
    try:
        return CHUNKING_PRESETS[name]
    except KeyError:
        available = sorted(CHUNKING_PRESETS)
        raise ValueError(f"Bilinmeyen chunking preset: '{name}'. Mevcut: {available}") from None
