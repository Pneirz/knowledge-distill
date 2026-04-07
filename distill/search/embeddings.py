from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def load_encoder(model_name: str = "BAAI/bge-small-en-v1.5") -> Any:
    """Load the sentence embedding model.

    BAAI/bge-small-en-v1.5 is 33M parameters, CPU-friendly, and performs
    well on BEIR benchmarks for technical English text.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_texts(
    encoder: Any,
    texts: list[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of texts into a float32 embedding matrix (N, D).

    Normalizes to unit length by default so cosine similarity reduces to
    dot product, which is what FAISS IndexFlatIP computes efficiently.
    """
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an IndexFlatIP (inner product) FAISS index.

    Because embeddings are L2-normalized, inner product equals cosine
    similarity. IndexFlatIP is exact (no approximation) and appropriate
    for corpora up to ~100k chunks.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """Persist a FAISS index to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    return faiss.read_index(str(path))


def save_chunk_ids(chunk_ids: list[str], path: Path) -> None:
    """Save the ordered list of chunk IDs that maps FAISS positions to DB IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(chunk_ids, fh)


def load_chunk_ids(path: Path) -> list[str]:
    """Load the chunk ID list corresponding to a FAISS index."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def search_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the FAISS index.

    Returns (scores, positions) arrays of shape (1, top_k).
    Scores are cosine similarities in [-1, 1] (higher is better).
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    scores, positions = index.search(query_embedding, top_k)
    return scores[0], positions[0]
