# %%
from datetime import datetime

import numpy as np

from distill.db.models import Chunk, Claim, Document
from distill.db.repository import (
    get_claims_by_doc,
    get_links_from,
    insert_chunks,
    insert_claims,
    insert_document,
    update_chunk_embedding_id,
)


def test_detect_relation_returns_llm_classification(mock_llm):
    """detect_relation should return the relation and confidence from the LLM JSON."""
    from distill.agents.linker import detect_relation

    mock_llm.complete_json.return_value = {
        "relation": "supports",
        "confidence": 0.91,
    }

    relation, confidence = detect_relation(
        mock_llm,
        "Transformers use self-attention.",
        "Self-attention is the core Transformer mechanism.",
    )

    assert relation == "supports"
    assert confidence == 0.91


def test_run_linker_creates_contradiction_link_and_marks_older_claim(
    db, mock_llm, monkeypatch
):
    """run_linker should create contradiction links and mark older contradicted claims."""
    from distill.agents.linker import run_linker

    older_doc = Document(
        doc_id="doc-old",
        title="Older Paper",
        source_type="pdf",
        content_hash="hash-old",
        status="verified",
        ingested_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
        year=2018,
    )
    newer_doc = Document(
        doc_id="doc-new",
        title="Newer Paper",
        source_type="pdf",
        content_hash="hash-new",
        status="verified",
        ingested_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
        year=2020,
    )
    insert_document(db, older_doc)
    insert_document(db, newer_doc)

    older_chunk = Chunk(
        chunk_id="chunk-old",
        doc_id=older_doc.doc_id,
        text="Attention does not scale efficiently.",
        chunk_index=0,
    )
    newer_chunk = Chunk(
        chunk_id="chunk-new",
        doc_id=newer_doc.doc_id,
        text="Sparse attention scales efficiently.",
        chunk_index=0,
    )
    insert_chunks(db, [older_chunk, newer_chunk])
    update_chunk_embedding_id(db, older_chunk.chunk_id, "0")
    update_chunk_embedding_id(db, newer_chunk.chunk_id, "1")

    older_claim = Claim(
        claim_id="claim-old",
        doc_id=older_doc.doc_id,
        chunk_id=older_chunk.chunk_id,
        claim_text="Attention does not scale efficiently.",
        claim_type="limitation",
        verified=1,
        raw_quote="Attention does not scale efficiently.",
    )
    newer_claim = Claim(
        claim_id="claim-new",
        doc_id=newer_doc.doc_id,
        chunk_id=newer_chunk.chunk_id,
        claim_text="Sparse attention scales efficiently.",
        claim_type="method",
        verified=1,
        raw_quote="Sparse attention scales efficiently.",
    )
    insert_claims(db, [older_claim, newer_claim])

    mock_llm.complete_json.return_value = {
        "relation": "contradicts",
        "confidence": 0.84,
    }

    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    def fake_search_index(index, query_embedding, top_k=10):
        return np.array([0.99, 0.98]), np.array([0, 1])

    monkeypatch.setattr("distill.search.embeddings.search_index", fake_search_index)
    created = run_linker(
        mock_llm,
        db,
        older_doc.doc_id,
        faiss_index=object(),
        chunk_ids=[older_chunk.chunk_id, newer_chunk.chunk_id],
        embeddings=embeddings,
    )

    assert len(created) == 1
    links = get_links_from(db, "claim", older_claim.claim_id)
    assert len(links) == 1
    assert links[0].relation == "contradicts"

    updated_older_claim = get_claims_by_doc(db, older_doc.doc_id)[0]
    assert updated_older_claim.verified == -1
