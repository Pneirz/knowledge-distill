# %%
from datetime import datetime, UTC

import pytest

from distill.db.models import AuditEvent, Chunk, Claim, Concept, EvidenceLink
from distill.db.repository import (
    get_claim_ids_for_concept,
    get_claims_by_doc,
    get_claims_by_chunk,
    get_concept_by_name,
    get_document,
    get_document_by_hash,
    get_links_from,
    get_unverified_claims,
    insert_chunks,
    insert_claim_concept,
    insert_claims,
    insert_document,
    insert_evidence_link,
    insert_audit_event,
    list_audit_events,
    list_concepts,
    list_documents,
    update_claim_lifecycle,
    update_claim_verification,
    update_document_status,
    upsert_concept,
)


# ---------------------------------------------------------------------------
# Document tests
# ---------------------------------------------------------------------------

def test_insert_and_get_document(db, sample_document):
    """Inserted document can be retrieved by primary key."""
    insert_document(db, sample_document)
    result = get_document(db, sample_document.doc_id)
    assert result is not None
    assert result.doc_id == sample_document.doc_id
    assert result.title == sample_document.title
    assert result.authors == sample_document.authors


def test_get_document_not_found(db):
    """get_document returns None for an unknown doc_id."""
    assert get_document(db, "nonexistent") is None


def test_get_document_by_hash(db, sample_document):
    """Document can be retrieved by content hash for duplicate detection."""
    insert_document(db, sample_document)
    result = get_document_by_hash(db, sample_document.content_hash)
    assert result is not None
    assert result.doc_id == sample_document.doc_id


def test_duplicate_hash_raises(db, sample_document):
    """Inserting two documents with the same content hash must raise."""
    import sqlite3 as _sqlite3

    insert_document(db, sample_document)
    duplicate = sample_document
    duplicate.doc_id = "other-id"
    with pytest.raises(_sqlite3.IntegrityError):
        insert_document(db, duplicate)


def test_update_document_status(db, sample_document):
    """Status and extra path fields can be updated after insertion."""
    insert_document(db, sample_document)
    update_document_status(db, sample_document.doc_id, "parsed", parsed_path="data/02_parsed/doc.json")
    updated = get_document(db, sample_document.doc_id)
    assert updated.status == "parsed"
    assert updated.parsed_path == "data/02_parsed/doc.json"


def test_list_documents_filtered_by_status(db, sample_document):
    """list_documents with status filter returns only matching documents."""
    insert_document(db, sample_document)
    ingested = list_documents(db, status="ingested")
    parsed = list_documents(db, status="parsed")
    assert len(ingested) == 1
    assert len(parsed) == 0


# ---------------------------------------------------------------------------
# Chunk tests
# ---------------------------------------------------------------------------

def _make_chunk(doc_id: str, idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{doc_id}-{idx}",
        doc_id=doc_id,
        text=f"This is chunk number {idx}.",
        chunk_index=idx,
        section="Introduction",
        page_start=1,
        page_end=1,
        token_count=10,
    )


def test_insert_and_get_chunks(db, sample_document):
    """Chunks inserted for a document can be retrieved in order."""
    insert_document(db, sample_document)
    chunks = [_make_chunk(sample_document.doc_id, i) for i in range(3)]
    insert_chunks(db, chunks)

    from distill.db.repository import get_chunks_by_doc
    result = get_chunks_by_doc(db, sample_document.doc_id)
    assert len(result) == 3
    assert result[0].chunk_index == 0
    assert result[2].chunk_index == 2


# ---------------------------------------------------------------------------
# Claim tests
# ---------------------------------------------------------------------------

def _make_claim(doc_id: str, chunk_id: str, idx: int) -> Claim:
    return Claim(
        claim_id=f"claim-{idx}",
        doc_id=doc_id,
        chunk_id=chunk_id,
        claim_text=f"Claim number {idx}.",
        claim_type="finding",
        raw_quote=f"Claim number {idx}.",
    )


def test_insert_and_get_claims(db, sample_document):
    """Claims inserted for a document can be retrieved."""
    insert_document(db, sample_document)
    chunk = _make_chunk(sample_document.doc_id, 0)
    insert_chunks(db, [chunk])
    claims = [_make_claim(sample_document.doc_id, chunk.chunk_id, i) for i in range(2)]
    insert_claims(db, claims)

    result = get_claims_by_doc(db, sample_document.doc_id)
    assert len(result) == 2
    by_chunk = get_claims_by_chunk(db, chunk.chunk_id)
    assert len(by_chunk) == 2


def test_get_unverified_claims(db, sample_document):
    """get_unverified_claims returns only claims with verified=0."""
    insert_document(db, sample_document)
    chunk = _make_chunk(sample_document.doc_id, 0)
    insert_chunks(db, [chunk])
    claims = [_make_claim(sample_document.doc_id, chunk.chunk_id, i) for i in range(3)]
    insert_claims(db, claims)

    update_claim_verification(db, claims[0].claim_id, 1, datetime.now(UTC).isoformat())
    unverified = get_unverified_claims(db)
    assert len(unverified) == 2


def test_update_claim_lifecycle(db, sample_document):
    """Lifecycle updates are stored independently from traceability."""
    insert_document(db, sample_document)
    chunk = _make_chunk(sample_document.doc_id, 0)
    insert_chunks(db, [chunk])
    claim = _make_claim(sample_document.doc_id, chunk.chunk_id, 0)
    newer_claim = _make_claim(sample_document.doc_id, chunk.chunk_id, 1)
    insert_claims(db, [claim, newer_claim])

    update_claim_lifecycle(
        db,
        claim.claim_id,
        "superseded",
        superseded_by_claim_id=newer_claim.claim_id,
    )

    updated = get_claims_by_doc(db, sample_document.doc_id)[0]
    assert updated.verified == 0
    assert updated.lifecycle_status == "superseded"
    assert updated.superseded_by_claim_id == newer_claim.claim_id


# ---------------------------------------------------------------------------
# Concept tests
# ---------------------------------------------------------------------------

def test_upsert_concept_insert(db):
    """A new concept can be inserted via upsert."""
    concept = Concept(
        concept_id="concept-001",
        name="transformer",
        created_at=datetime(2026, 4, 6),
        updated_at=datetime(2026, 4, 6),
        definition="Attention-based neural network architecture.",
        domain="architecture",
    )
    upsert_concept(db, concept)
    result = get_concept_by_name(db, "transformer")
    assert result is not None
    assert result.definition == "Attention-based neural network architecture."


def test_upsert_concept_update(db):
    """Upserting an existing concept by name updates its fields."""
    concept = Concept(
        concept_id="concept-001",
        name="transformer",
        created_at=datetime(2026, 4, 6),
        updated_at=datetime(2026, 4, 6),
        definition="Old definition.",
    )
    upsert_concept(db, concept)

    updated = Concept(
        concept_id="concept-001",
        name="transformer",
        created_at=datetime(2026, 4, 6),
        updated_at=datetime(2026, 4, 6, 13, 0),
        definition="Updated definition.",
    )
    upsert_concept(db, updated)

    result = get_concept_by_name(db, "transformer")
    assert result.definition == "Updated definition."
    assert len(list_concepts(db)) == 1


# ---------------------------------------------------------------------------
# EvidenceLink tests
# ---------------------------------------------------------------------------

def test_insert_and_get_evidence_link(db):
    """Evidence links can be inserted and retrieved by source object."""
    link = EvidenceLink(
        link_id="link-001",
        from_type="claim",
        from_id="claim-1",
        to_type="claim",
        to_id="claim-2",
        relation="supports",
        confidence=0.9,
        created_at=datetime(2026, 4, 6),
    )
    insert_evidence_link(db, link)
    results = get_links_from(db, "claim", "claim-1")
    assert len(results) == 1
    assert results[0].relation == "supports"
    audit = list_audit_events(db, entity_type="evidence_link")
    assert len(audit) == 1
    assert audit[0].action == "created"


# ---------------------------------------------------------------------------
# ClaimConcept junction tests
# ---------------------------------------------------------------------------

def test_insert_claim_concept_idempotent(db, sample_document):
    """Inserting the same claim-concept link twice must not raise."""
    insert_document(db, sample_document)
    chunk = _make_chunk(sample_document.doc_id, 0)
    insert_chunks(db, [chunk])
    claim = _make_claim(sample_document.doc_id, chunk.chunk_id, 0)
    insert_claims(db, [claim])

    concept = Concept(
        concept_id="concept-001",
        name="attention",
        created_at=datetime(2026, 4, 6),
        updated_at=datetime(2026, 4, 6),
    )
    upsert_concept(db, concept)

    insert_claim_concept(db, claim.claim_id, concept.concept_id, role="subject")
    insert_claim_concept(db, claim.claim_id, concept.concept_id, role="subject")

    result = get_claim_ids_for_concept(db, concept.concept_id)
    assert result == [claim.claim_id]


def test_insert_and_list_audit_events(db):
    """Audit events can be stored and filtered by entity."""
    event = AuditEvent(
        event_id="evt-001",
        entity_type="claim",
        entity_id="claim-123",
        action="verification_updated",
        details_json='{"verified": 1}',
        created_at=datetime(2026, 4, 6, 12, 0, 0),
    )
    insert_audit_event(db, event)

    result = list_audit_events(db, entity_type="claim", entity_id="claim-123")
    assert len(result) == 1
    assert result[0].event_id == "evt-001"
