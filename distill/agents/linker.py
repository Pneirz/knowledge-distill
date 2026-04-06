import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from distill.config import Config
from distill.db.models import EvidenceLink
from distill.db.repository import (
    get_claims_by_doc,
    get_document,
    insert_evidence_link,
)
from distill.llm.client import BaseLLMClient
from distill.llm.prompts import contradiction_detection_prompt

# Similarity threshold above which two claims are compared for contradiction
SIMILARITY_THRESHOLD = 0.85


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    return float(np.dot(vec_a, vec_b))


def detect_relation(
    client: BaseLLMClient,
    claim_a_text: str,
    claim_b_text: str,
) -> tuple[str, float]:
    """Ask Claude to classify the relation between two similar claims.

    Returns (relation, confidence). The relation is one of:
    contradicts | supports | refines | unrelated.
    """
    prompt = contradiction_detection_prompt(claim_a_text, claim_b_text)
    result = client.complete_json(
        system="You classify relationships between scientific claims. Return only valid JSON.",
        user=prompt,
        max_tokens=256,
    )
    relation = result.get("relation", "unrelated")
    confidence = float(result.get("confidence", 0.5))
    return relation, confidence


def run_linker(
    client: BaseLLMClient,
    conn: sqlite3.Connection,
    doc_id: str,
    faiss_index,
    chunk_ids: list[str],
    embeddings: np.ndarray,
) -> list[str]:
    """Create EvidenceLinks between claims of this document and existing claims.

    Algorithm:
    1. For each claim in the document, find its chunk's embedding.
    2. Search for similar chunks in the FAISS index (similarity > threshold).
    3. For each similar chunk from a different document, compare claims using Claude.
    4. Insert an EvidenceLink for relations that are not 'unrelated'.

    Returns list of created link_ids.
    """
    doc = get_document(conn, doc_id)
    if doc is None:
        raise ValueError(f"Document not found: {doc_id}")

    claims = get_claims_by_doc(conn, doc_id)
    if not claims:
        return []

    from distill.search.embeddings import search_index
    from distill.db.repository import get_chunk, get_claims_by_doc as get_claims

    created_links: list[str] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for claim in claims:
        chunk = get_chunk(conn, claim.chunk_id)
        if chunk is None or chunk.embedding_id is None:
            continue

        # Find the position of this chunk in the FAISS index
        try:
            position = chunk_ids.index(chunk.chunk_id)
        except ValueError:
            continue

        chunk_embedding = embeddings[position].reshape(1, -1)
        scores, positions = search_index(faiss_index, chunk_embedding[0], top_k=10)

        for sim_score, sim_position in zip(scores, positions):
            if sim_score < SIMILARITY_THRESHOLD:
                break
            if sim_position < 0 or sim_position >= len(chunk_ids):
                continue

            similar_chunk_id = chunk_ids[int(sim_position)]
            if similar_chunk_id == chunk.chunk_id:
                continue  # Skip self

            similar_chunk = get_chunk(conn, similar_chunk_id)
            if similar_chunk is None or similar_chunk.doc_id == doc_id:
                continue  # Skip same document

            # Compare claims from the similar chunk
            similar_claims = [
                c for c in get_claims(conn, similar_chunk.doc_id)
                if c.chunk_id == similar_chunk_id
            ]
            for similar_claim in similar_claims:
                relation, confidence = detect_relation(
                    client, claim.claim_text, similar_claim.claim_text
                )
                if relation == "unrelated":
                    continue

                link_id = str(uuid.uuid4())
                link = EvidenceLink(
                    link_id=link_id,
                    from_type="claim",
                    from_id=claim.claim_id,
                    to_type="claim",
                    to_id=similar_claim.claim_id,
                    relation=relation,
                    confidence=confidence,
                    created_at=now,
                )
                insert_evidence_link(conn, link)
                created_links.append(link_id)

                # Mark contradicted claims
                if relation == "contradicts":
                    from distill.db.repository import update_claim_verification
                    # The older claim (lower year) gets marked as contradicted
                    doc_a = get_document(conn, claim.doc_id)
                    doc_b = get_document(conn, similar_claim.doc_id)
                    if doc_a and doc_b and doc_a.year and doc_b.year:
                        older_claim_id = (
                            claim.claim_id
                            if doc_a.year < doc_b.year
                            else similar_claim.claim_id
                        )
                        update_claim_verification(
                            conn, older_claim_id, -1, now.isoformat()
                        )

    return created_links
