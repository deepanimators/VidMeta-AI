import pytest
from vidmeta.video.transcription import _merge_adjacent_turns


def test_merge_adjacent_turns_merges_close_same_speaker():
    turns = [
        (0.0, 1.0, "spk_0"),
        (1.2, 2.0, "spk_0"),
        (3.0, 4.0, "spk_1"),
        (4.2, 5.0, "spk_1"),
    ]
    merged = _merge_adjacent_turns(turns, gap_threshold=0.5)
    assert merged[0][0] == 0.0
    assert merged[0][1] == pytest.approx(2.0)
    assert merged[1][0] == 3.0
    assert merged[1][1] == pytest.approx(5.0)


def test_merge_adjacent_turns_separates_distant_segments():
    turns = [
        (0.0, 1.0, "spk_0"),
        (2.0, 3.0, "spk_0"),
    ]
    merged = _merge_adjacent_turns(turns, gap_threshold=0.5)
    assert len(merged) == 2
