def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute recall@k for a ranked result list."""
    if not relevant_ids:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(
    ranked_lists: list[list[str]],
    relevant_sets: list[set[str]],
) -> float:
    """Compute mean reciprocal rank across multiple queries."""
    if not ranked_lists or not relevant_sets or len(ranked_lists) != len(relevant_sets):
        return 0.0

    reciprocal_ranks: list[float] = []
    for ranked_ids, relevant_ids in zip(ranked_lists, relevant_sets, strict=True):
        rr = 0.0
        for rank, item_id in enumerate(ranked_ids, start=1):
            if item_id in relevant_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
