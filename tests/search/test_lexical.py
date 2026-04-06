# %%
import pytest

from distill.search.lexical import build_bm25_index, search_bm25, tokenize


def test_tokenize_lowercases():
    assert tokenize("Transformer Model") == ["transformer", "model"]


def test_tokenize_removes_punctuation():
    tokens = tokenize("attention: the key mechanism!")
    assert "attention" in tokens
    assert ":" not in tokens
    assert "!" not in tokens


def test_bm25_returns_relevant_result():
    """The document most relevant to the query should rank first."""
    corpus = [
        "The Transformer model uses self-attention mechanisms.",
        "Convolutional networks process images efficiently.",
        "Recurrent networks model sequences with hidden states.",
    ]
    index = build_bm25_index(corpus)
    results = search_bm25(index, "transformer attention", top_k=3)

    assert len(results) > 0
    top_position, top_score = results[0]
    assert top_position == 0  # First document is most relevant


def test_bm25_normalized_scores_in_range():
    """All returned scores must be in [0, 1]."""
    corpus = ["alpha beta", "gamma delta", "alpha gamma"]
    index = build_bm25_index(corpus)
    results = search_bm25(index, "alpha", top_k=3)
    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_bm25_empty_query_returns_empty():
    """A query with no matching tokens returns empty results."""
    corpus = ["alpha beta", "gamma delta"]
    index = build_bm25_index(corpus)
    results = search_bm25(index, "zzz_no_match_token", top_k=5)
    assert results == []
