# %%
from datetime import datetime

from distill.db.models import Chunk, Claim, Concept, Document, EvidenceLink
from distill.db.repository import (
    insert_chunks,
    insert_claims,
    insert_document,
    insert_evidence_link,
    upsert_concept,
    update_claim_lifecycle,
)


def _seed_output_doc(db):
    doc = Document(
        doc_id="doc-001",
        title="Attention Paper",
        source_type="pdf",
        content_hash="hash-001",
        status="compiled",
        ingested_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
        year=2017,
    )
    insert_document(db, doc)
    chunk = Chunk(
        chunk_id="chunk-001",
        doc_id=doc.doc_id,
        text="Self-attention models token relationships.",
        chunk_index=0,
        section="Method",
    )
    insert_chunks(db, [chunk])
    active_claim = Claim(
        claim_id="claim-active",
        doc_id=doc.doc_id,
        chunk_id=chunk.chunk_id,
        claim_text="Self-attention models token relationships.",
        claim_type="method",
        verified=1,
        raw_quote="Self-attention models token relationships.",
    )
    superseded_claim = Claim(
        claim_id="claim-old",
        doc_id=doc.doc_id,
        chunk_id=chunk.chunk_id,
        claim_text="Older attention variants were less efficient.",
        claim_type="limitation",
        verified=1,
        raw_quote="Older attention variants were less efficient.",
    )
    insert_claims(db, [active_claim, superseded_claim])
    update_claim_lifecycle(
        db,
        superseded_claim.claim_id,
        "superseded",
        superseded_by_claim_id=active_claim.claim_id,
    )
    return doc, active_claim, superseded_claim


def test_generate_brief_excludes_superseded_claims(db, tmp_path, mock_llm):
    """Brief generation should only include verified active claims in the prompt context."""
    from distill.agents.output_agent import generate_brief

    doc, _, _ = _seed_output_doc(db)
    mock_llm.complete.return_value = "Brief output"

    dest = generate_brief(mock_llm, db, "attention", [doc.doc_id], tmp_path)

    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "Brief output"
    _, called_user_prompt = mock_llm.complete.call_args.args[:2]
    assert "Self-attention models token relationships." in called_user_prompt
    assert "Older attention variants were less efficient." not in called_user_prompt


def test_generate_comparison_table_uses_active_verified_claims_only(db, tmp_path):
    """Comparison tables should skip superseded claims when selecting dimension values."""
    from distill.agents.output_agent import generate_comparison_table

    doc, active_claim, _ = _seed_output_doc(db)

    dest = generate_comparison_table(db, [doc.doc_id], ["method", "limitation"], tmp_path)
    table = dest.read_text(encoding="utf-8")

    assert active_claim.claim_text[:40] in table
    assert "| — |" in table or "â€”" in table


def test_generate_concept_map_renders_concept_links(db, tmp_path):
    """Concept maps should render Mermaid edges between linked concepts."""
    from distill.agents.output_agent import generate_concept_map

    concept_a = Concept(
        concept_id="concept-a",
        name="self-attention",
        created_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
    )
    concept_b = Concept(
        concept_id="concept-b",
        name="transformer",
        created_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
    )
    upsert_concept(db, concept_a)
    upsert_concept(db, concept_b)
    insert_evidence_link(
        db,
        EvidenceLink(
            link_id="link-concepts",
            from_type="concept",
            from_id=concept_a.concept_id,
            to_type="concept",
            to_id=concept_b.concept_id,
            relation="uses",
            confidence=0.9,
            created_at=datetime(2026, 4, 6, 12, 0, 0),
        ),
    )

    dest = generate_concept_map(db, ["self-attention", "transformer"], tmp_path)
    mermaid = dest.read_text(encoding="utf-8")

    assert "graph TD" in mermaid
    assert "self_attention" in mermaid
    assert "transformer" in mermaid
    assert "uses" in mermaid
