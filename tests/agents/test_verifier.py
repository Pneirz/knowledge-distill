# %%
from datetime import datetime

from distill.agents.verifier import verify_claim_traceability
from distill.db.models import Chunk, Claim
from distill.db.repository import (
    get_claims_by_doc,
    insert_chunks,
    insert_claims,
    insert_document,
    list_audit_events,
)


def test_verify_traceability_exact_match():
    """A raw_quote that is an exact substring of the chunk text returns True."""
    chunk_text = "The Transformer model achieves state-of-the-art performance on translation tasks."
    raw_quote = "state-of-the-art performance on translation tasks"
    assert verify_claim_traceability("some claim", raw_quote, chunk_text) is True


def test_verify_traceability_fuzzy_match():
    """A raw_quote with minor whitespace differences still passes fuzzy matching."""
    chunk_text = "The  model  achieves   state-of-the-art  performance."
    raw_quote = "model achieves state-of-the-art performance"
    assert verify_claim_traceability("some claim", raw_quote, chunk_text) is True


def test_verify_traceability_fails_missing_quote():
    """A claim with no raw_quote fails traceability."""
    assert verify_claim_traceability("some claim", None, "any text") is False


def test_verify_traceability_fails_wrong_quote():
    """A raw_quote from a different text fails fuzzy matching."""
    chunk_text = "This is completely unrelated text about something else."
    raw_quote = "The Transformer achieves state-of-the-art results on WMT 2014."
    assert verify_claim_traceability("some claim", raw_quote, chunk_text) is False


def test_run_verifier_marks_verified_claims(db, tmp_data_root, sample_document):
    """run_verifier marks claims with valid raw_quote as verified=1."""
    from distill.agents.verifier import run_verifier
    from distill.config import Config

    cfg = Config(data_root=tmp_data_root)
    insert_document(db, sample_document)

    chunk = Chunk(
        chunk_id="chunk-001",
        doc_id=sample_document.doc_id,
        text="The Transformer achieves state-of-the-art results.",
        chunk_index=0,
        section="Results",
    )
    insert_chunks(db, [chunk])

    claim = Claim(
        claim_id="claim-001",
        doc_id=sample_document.doc_id,
        chunk_id="chunk-001",
        claim_text="Transformer achieves SOTA results.",
        claim_type="finding",
        raw_quote="Transformer achieves state-of-the-art results",
    )
    insert_claims(db, [claim])

    # Update status to 'compiled' so verifier accepts it
    from distill.db.repository import update_document_status
    update_document_status(db, sample_document.doc_id, "compiled")

    report = run_verifier(db, sample_document.doc_id, cfg)
    assert report["verified"] == 1
    assert report["failed"] == 0
    events = list_audit_events(db, entity_type="claim")
    assert len(events) == 1
    assert events[0].action == "verification_updated"


def test_run_verifier_marks_untraceable_claims(db, tmp_data_root, sample_document):
    """run_verifier marks claims with missing raw_quote as verified=-1."""
    from distill.agents.verifier import run_verifier
    from distill.config import Config

    cfg = Config(data_root=tmp_data_root)
    insert_document(db, sample_document)

    chunk = Chunk(
        chunk_id="chunk-001",
        doc_id=sample_document.doc_id,
        text="Some chunk text.",
        chunk_index=0,
    )
    insert_chunks(db, [chunk])

    claim = Claim(
        claim_id="claim-001",
        doc_id=sample_document.doc_id,
        chunk_id="chunk-001",
        claim_text="Some claim without a quote.",
        claim_type="finding",
        raw_quote=None,  # Missing quote
    )
    insert_claims(db, [claim])

    from distill.db.repository import update_document_status
    update_document_status(db, sample_document.doc_id, "compiled")

    report = run_verifier(db, sample_document.doc_id, cfg)
    assert report["failed"] == 1
    assert report["verified"] == 0


def test_generate_verification_report_separates_traceability_and_lifecycle(db, sample_document):
    """Verification report keeps lifecycle counts separate from traceability."""
    from distill.agents.verifier import generate_verification_report
    from distill.db.repository import update_claim_lifecycle

    insert_document(db, sample_document)
    chunk = Chunk(
        chunk_id="chunk-001",
        doc_id=sample_document.doc_id,
        text="Transformer performs well.",
        chunk_index=0,
    )
    insert_chunks(db, [chunk])
    claim = Claim(
        claim_id="claim-001",
        doc_id=sample_document.doc_id,
        chunk_id="chunk-001",
        claim_text="Transformer performs well.",
        claim_type="finding",
        verified=1,
        raw_quote="Transformer performs well.",
    )
    newer_claim = Claim(
        claim_id="claim-002",
        doc_id=sample_document.doc_id,
        chunk_id="chunk-001",
        claim_text="Transformer performs even better with scaling.",
        claim_type="finding",
        verified=1,
        raw_quote="Transformer performs even better with scaling.",
    )
    insert_claims(db, [claim, newer_claim])
    update_claim_lifecycle(
        db,
        claim.claim_id,
        "superseded",
        superseded_by_claim_id=newer_claim.claim_id,
    )

    report = generate_verification_report(db, sample_document.doc_id)
    doc_report = report["documents"][0]
    assert doc_report["traceability"]["verified"] == 2
    assert doc_report["lifecycle"]["superseded"] == 1


def test_check_concept_obsolescence_marks_superseded_without_changing_verified(
    db, sample_document, mock_llm
):
    """Obsolescence review updates lifecycle fields instead of verification."""
    from datetime import datetime

    from distill.agents.verifier import check_concept_obsolescence
    from distill.db.models import Concept, Document
    from distill.db.repository import insert_claim_concept, upsert_concept

    old_doc = sample_document
    new_doc = Document(
        doc_id="test-doc-002",
        title="Newer Transformer Paper",
        source_type="pdf",
        content_hash="hash-002",
        status="compiled",
        ingested_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
        authors=["Doe, J."],
        year=2020,
    )
    old_doc.status = "compiled"
    insert_document(db, old_doc)
    insert_document(db, new_doc)

    old_chunk = Chunk(chunk_id="chunk-old", doc_id=old_doc.doc_id, text="Old chunk", chunk_index=0)
    new_chunk = Chunk(chunk_id="chunk-new", doc_id=new_doc.doc_id, text="New chunk", chunk_index=0)
    insert_chunks(db, [old_chunk, new_chunk])

    old_claim = Claim(
        claim_id="claim-old",
        doc_id=old_doc.doc_id,
        chunk_id=old_chunk.chunk_id,
        claim_text="Attention does not scale efficiently.",
        claim_type="limitation",
        verified=1,
        raw_quote="Attention does not scale efficiently.",
    )
    new_claim = Claim(
        claim_id="claim-new",
        doc_id=new_doc.doc_id,
        chunk_id=new_chunk.chunk_id,
        claim_text="Sparse attention scales efficiently.",
        claim_type="method",
        verified=1,
        raw_quote="Sparse attention scales efficiently.",
    )
    insert_claims(db, [old_claim, new_claim])

    concept = Concept(
        concept_id="concept-001",
        name="attention",
        created_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
    )
    upsert_concept(db, concept)
    insert_claim_concept(db, old_claim.claim_id, concept.concept_id)
    insert_claim_concept(db, new_claim.claim_id, concept.concept_id)

    mock_llm.complete_json.return_value = {
        "obsolete_claim_indices": [0],
        "reasons": {"0": "Newer evidence supersedes the earlier limitation."},
    }
    result = check_concept_obsolescence(mock_llm, db, "attention")

    updated_claims = {claim.claim_id: claim for claim in get_claims_by_doc(db, old_doc.doc_id)}
    assert result["obsolete_claims"] == ["claim-old"]
    assert updated_claims["claim-old"].verified == 1
    assert updated_claims["claim-old"].lifecycle_status == "superseded"
