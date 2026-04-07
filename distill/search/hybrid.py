from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

import faiss

from distill.search.embeddings import encode_texts, search_index
from distill.search.lexical import search_bm25


def reciprocal_rank_fusion(
    result_lists: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over lists of 1 / (k + rank(d)).
    k=60 is the standard constant from the original RRF paper.

    This is more robust than score-blending because it does not require
    normalizing scores from incompatible scales (BM25 vs cosine).
    """
    scores: dict[int, float] = {}
    for ranked_list in result_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by descending RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    encoder: Any,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    top_k: int = 10,
) -> list[dict]:
    """Perform hybrid semantic + lexical search with RRF fusion.

    1. Encode query and search FAISS for semantic matches.
    2. Search BM25 for lexical matches.
    3. Fuse both ranked lists with RRF.
    4. Return top_k results as dicts with chunk_id, score, and rank.
    """
    n_candidates = min(top_k * 3, len(chunk_ids))

    # Semantic search
    query_embedding = encode_texts(encoder, [query])[0]
    sem_scores, sem_positions = search_index(faiss_index, query_embedding, top_k=n_candidates)
    sem_ranked = [int(p) for p in sem_positions if p >= 0]

    # Lexical search
    lex_results = search_bm25(bm25_index, query, top_k=n_candidates)
    lex_ranked = [pos for pos, _ in lex_results]

    # RRF fusion
    fused = reciprocal_rank_fusion([sem_ranked, lex_ranked])

    # Build output with chunk_ids
    results = []
    for rank, (position, rrf_score) in enumerate(fused[:top_k], start=1):
        if 0 <= position < len(chunk_ids):
            results.append({
                "chunk_id": chunk_ids[position],
                "position": position,
                "score": rrf_score,
                "rank": rank,
            })

    return results
