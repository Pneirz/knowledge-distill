import pickle
import re
import string
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for BM25 indexing.

    Lowercases, removes punctuation, splits on whitespace.
    No stopword removal: short ML terms like 'is', 'at', 'on' can matter
    in technical queries.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """Build a BM25Okapi index from a list of text strings."""
    tokenized = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def save_bm25(index: BM25Okapi, path: Path) -> None:
    """Serialize BM25 index to disk with pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(index, fh)


def load_bm25(path: Path) -> BM25Okapi:
    """Deserialize BM25 index from disk."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def search_bm25(
    index: BM25Okapi,
    query: str,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Query the BM25 index and return (position, normalized_score) pairs.

    Scores are normalized to [0, 1] by dividing by the max score.
    Returns at most top_k results sorted by score descending.
    """
    tokens = tokenize(query)
    raw_scores = index.get_scores(tokens)

    max_score = float(np.max(raw_scores)) if len(raw_scores) > 0 else 1.0
    if max_score == 0.0:
        return []

    normalized = raw_scores / max_score

    # Get top-k positions
    top_positions = np.argsort(normalized)[::-1][:top_k]
    return [
        (int(pos), float(normalized[pos]))
        for pos in top_positions
        if normalized[pos] > 0.0
    ]
