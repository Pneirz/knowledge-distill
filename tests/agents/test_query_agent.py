# %%
from datetime import datetime

from distill.db.models import Chunk, Claim, Concept
from distill.db.repository import (
    insert_chunks,
    insert_claim_concept,
    insert_claims,
    insert_document,
    upsert_concept,
    update_claim_lifecycle,
)


def _seed_query_data(db, sample_document):
    insert_document(db, sample_document)
    active_chunk = Chunk(
        chunk_id="chunk-active",
        doc_id=sample_document.doc_id,
        text="Transformers use self-attention to model token relationships.",
        chunk_index=0,
        section="Methods",
    )
    superseded_chunk = Chunk(
        chunk_id="chunk-superseded",
        doc_id=sample_document.doc_id,
        text="Earlier attention variants were less efficient.",
        chunk_index=1,
        section="Background",
    )
    contested_chunk = Chunk(
        chunk_id="chunk-contested",
        doc_id=sample_document.doc_id,
        text="Some studies report attention is not efficient at long range.",
        chunk_index=2,
        section="Discussion",
    )
    insert_chunks(db, [active_chunk, superseded_chunk, contested_chunk])

    claims = [
        Claim(
            claim_id="claim-active",
            doc_id=sample_document.doc_id,
            chunk_id=active_chunk.chunk_id,
            claim_text="Self-attention models token relationships effectively.",
            claim_type="method",
            verified=1,
            raw_quote="self-attention to model token relationships",
        ),
        Claim(
            claim_id="claim-superseded",
            doc_id=sample_document.doc_id,
            chunk_id=superseded_chunk.chunk_id,
            claim_text="Earlier attention variants were less efficient.",
            claim_type="limitation",
            verified=1,
            raw_quote="Earlier attention variants were less efficient.",
        ),
        Claim(
            claim_id="claim-contested",
            doc_id=sample_document.doc_id,
            chunk_id=contested_chunk.chunk_id,
            claim_text="Attention is not efficient at long range.",
            claim_type="limitation",
            verified=1,
            raw_quote="attention is not efficient at long range",
        ),
    ]
    insert_claims(db, claims)
    update_claim_lifecycle(db, "claim-superseded", "superseded", superseded_by_claim_id="claim-active")
    update_claim_lifecycle(db, "claim-contested", "contested")

    concept = Concept(
        concept_id="concept-attention",
        name="attention",
        created_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
    )
    upsert_concept(db, concept)
    for claim_id in ["claim-active", "claim-superseded", "claim-contested"]:
        insert_claim_concept(db, claim_id, concept.concept_id)


def test_build_evidence_candidates_excludes_superseded_by_default(db, sample_document):
    """Progressive disclosure excludes superseded claims from primary candidates by default."""
    from distill.agents.query_agent import build_evidence_candidates

    _seed_query_data(db, sample_document)
    search_results = [
        {"chunk_id": "chunk-active", "score": 0.9, "rank": 1},
        {"chunk_id": "chunk-superseded", "score": 0.8, "rank": 2},
    ]

    candidates = build_evidence_candidates(db, search_results, include_superseded=False)
    chunk_ids = [candidate["chunk_id"] for candidate in candidates]
    assert "chunk-active" in chunk_ids
    assert "chunk-superseded" not in chunk_ids


def test_build_evidence_candidates_can_include_superseded(db, sample_document):
    """Superseded claims reappear as secondary evidence when explicitly requested."""
    from distill.agents.query_agent import build_evidence_candidates

    _seed_query_data(db, sample_document)
    search_results = [
        {"chunk_id": "chunk-active", "score": 0.9, "rank": 1},
        {"chunk_id": "chunk-superseded", "score": 0.8, "rank": 2},
    ]

    candidates = build_evidence_candidates(db, search_results, include_superseded=True)
    superseded = next(candidate for candidate in candidates if candidate["chunk_id"] == "chunk-superseded")
    assert superseded["lifecycle_status"] == "superseded"
    assert superseded["is_primary_evidence"] is False


def test_run_query_marks_contested_uncertainty(db, sample_document, mock_llm, monkeypatch):
    """Contested claims should surface as uncertainty rather than primary support."""
    from distill.agents.query_agent import run_query

    _seed_query_data(db, sample_document)

    def fake_hybrid_search(*args, **kwargs):
        return [
            {"chunk_id": "chunk-active", "score": 0.9, "rank": 1},
            {"chunk_id": "chunk-contested", "score": 0.85, "rank": 2},
        ]

    monkeypatch.setattr("distill.agents.query_agent.hybrid_search", fake_hybrid_search)
    mock_llm.complete_json.return_value = {
        "answer": "Transformers use self-attention effectively.",
        "sources": [
            {"doc_id": sample_document.doc_id[:8], "chunk_id": "chunk-active", "quote": "self-attention"},
            {"doc_id": sample_document.doc_id[:8], "chunk_id": "chunk-cont", "quote": "not efficient"},
        ],
        "confidence": 0.8,
        "uncertainty": "",
    }

    result = run_query(
        mock_llm,
        db,
        "How does attention work?",
        encoder=None,
        faiss_index=None,
        bm25_index=None,
        chunk_ids=[],
        top_k=2,
    )

    assert "contested" in result["uncertainty"].lower()
    assert any(src["lifecycle_status"] == "contested" for src in result["sources"])
