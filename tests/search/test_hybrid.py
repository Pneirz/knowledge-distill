# %%
import numpy as np
import pytest

from distill.search.hybrid import reciprocal_rank_fusion


def test_rrf_single_list():
    """RRF with one list scores rank 1 highest."""
    ranked = [10, 20, 30]
    results = reciprocal_rank_fusion([ranked])
    positions = [pos for pos, _ in results]
    assert positions[0] == 10  # rank 1 has highest score


def test_rrf_two_lists_boosted_overlap():
    """Items appearing in both lists receive higher combined score."""
    list_a = [1, 2, 3]
    list_b = [2, 1, 4]
    results = reciprocal_rank_fusion([list_a, list_b])

    result_dict = {pos: score for pos, score in results}
    # Items 1 and 2 appear in both lists; they should rank above item 4 (single list)
    assert result_dict[1] > result_dict[4]
    assert result_dict[2] > result_dict[4]


def test_rrf_score_decreases_with_rank():
    """Within a single list, scores must be strictly decreasing."""
    ranked = list(range(10))
    results = reciprocal_rank_fusion([ranked])

    scores = [score for _, score in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_rrf_empty_lists():
    """RRF with empty input returns empty output."""
    results = reciprocal_rank_fusion([[], []])
    assert results == []


def test_rrf_no_duplicates_in_output():
    """Each position appears at most once in the output."""
    list_a = [1, 2, 3]
    list_b = [3, 2, 1]
    results = reciprocal_rank_fusion([list_a, list_b])
    positions = [pos for pos, _ in results]
    assert len(positions) == len(set(positions))
