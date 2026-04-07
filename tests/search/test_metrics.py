# %%
from distill.search.metrics import mean_reciprocal_rank, recall_at_k


def test_recall_at_k():
    """Recall@k counts the share of relevant ids retrieved in the top-k."""
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert recall_at_k(ranked, relevant, k=2) == 0.5
    assert recall_at_k(ranked, relevant, k=4) == 1.0


def test_mean_reciprocal_rank():
    """MRR averages the first-hit reciprocal rank across queries."""
    ranked_lists = [["a", "b", "c"], ["x", "y", "z"]]
    relevant_sets = [{"b"}, {"z"}]
    assert mean_reciprocal_rank(ranked_lists, relevant_sets) == (0.5 + (1 / 3)) / 2
