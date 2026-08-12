import pytest
from app.services.chunking_presets import CHUNKING_PRESETS, get_chunking_config


@pytest.mark.parametrize(
    ("name", "max_tokens", "overlap_tokens"),
    [("300_50", 300, 50), ("500_75", 500, 75), ("800_100", 800, 100)],
)
def test_presets_have_expected_values(name: str, max_tokens: int, overlap_tokens: int) -> None:
    config = get_chunking_config(name)

    assert config.max_tokens == max_tokens
    assert config.overlap_tokens == overlap_tokens


def test_all_presets_are_registered() -> None:
    assert set(CHUNKING_PRESETS) == {"300_50", "500_75", "800_100"}


def test_unknown_preset_name_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen chunking preset"):
        get_chunking_config("does-not-exist")
