# %%
from datetime import datetime

import pytest

from distill.agents.verifier import verify_claim_traceability
from distill.db.models import Chunk, Claim, Document
from distill.db.repository import insert_chunks, insert_claims, insert_document


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
