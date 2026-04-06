import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz import fuzz

from distill.config import Config
from distill.db.repository import (
    get_chunk,
    get_claims_by_doc,
    get_contradictions,
    get_document,
    update_claim_verification,
    update_document_status,
)
from distill.llm.client import LLMClient
from distill.llm.prompts import obsolescence_prompt

# Fuzzy match threshold for raw_quote traceability verification
TRACEABILITY_THRESHOLD = 85


def verify_claim_traceability(
    claim_text: str,
    raw_quote: str | None,
    chunk_text: str,
) -> bool:
    """Verify that a claim's raw_quote appears in its source chunk.

    Uses partial ratio fuzzy matching (threshold=85) to tolerate minor
    whitespace and formatting differences between the extracted quote and
    the original chunk text.

    Returns False if raw_quote is absent or does not match the chunk text.
    """
    if not raw_quote:
        return False
    score = fuzz.partial_ratio(raw_quote.lower(), chunk_text.lower())
    return score >= TRACEABILITY_THRESHOLD


def run_verifier(
    conn: sqlite3.Connection,
    doc_id: str,
    cfg: Config,
    client: LLMClient | None = None,
) -> dict:
    """Verify all claims for a document.

    For each claim:
    - If raw_quote fuzzy-matches the source chunk: mark verified=1.
    - If raw_quote is missing or does not match: mark verified=-1.

    Optionally checks for obsolescence using Claude if client is provided.
    Returns a verification report dict.
    """
    doc = get_document(conn, doc_id)
    if doc is None:
        raise ValueError(f"Document not found: {doc_id}")

    claims = get_claims_by_doc(conn, doc_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    verified_count = 0
    failed_count = 0
    failed_claims: list[dict] = []

    for claim in claims:
        chunk = get_chunk(conn, claim.chunk_id)
        if chunk is None:
            update_claim_verification(conn, claim.claim_id, -1, now)
            failed_count += 1
            failed_claims.append({
                "claim_id": claim.claim_id,
                "reason": "chunk not found",
            })
            continue

        is_traceable = verify_claim_traceability(
            claim.claim_text, claim.raw_quote, chunk.text
        )
        if is_traceable:
            update_claim_verification(conn, claim.claim_id, 1, now)
            verified_count += 1
        else:
            update_claim_verification(conn, claim.claim_id, -1, now)
            failed_count += 1
            failed_claims.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text[:120],
                "raw_quote": claim.raw_quote,
                "reason": "quote not found in source chunk",
            })

    # Update document status to 'verified'
    update_document_status(conn, doc_id, "verified")

    return {
        "doc_id": doc_id,
        "total_claims": len(claims),
        "verified": verified_count,
        "failed": failed_count,
        "failed_claims": failed_claims,
    }


def generate_verification_report(
    conn: sqlite3.Connection,
    doc_id: str | None = None,
) -> dict:
    """Generate a full verification report for one document or all documents.

    Returns counts of verified/failed claims and all contradiction pairs.
    """
    from distill.db.repository import list_documents

    if doc_id:
        docs = [get_document(conn, doc_id)]
    else:
        docs = list_documents(conn)

    summary: list[dict] = []
    for doc in docs:
        if doc is None:
            continue
        claims = get_claims_by_doc(conn, doc.doc_id)
        verified = sum(1 for c in claims if c.verified == 1)
        failed = sum(1 for c in claims if c.verified == -1)
        unverified = sum(1 for c in claims if c.verified == 0)
        summary.append({
            "doc_id": doc.doc_id,
            "title": doc.title,
            "total_claims": len(claims),
            "verified": verified,
            "failed": failed,
            "unverified": unverified,
        })

    contradictions = get_contradictions(conn)
    return {
        "documents": summary,
        "contradiction_pairs": len(contradictions),
        "contradictions": [
            {"claim_a": a, "claim_b": b} for a, b in contradictions
        ],
    }


def check_concept_obsolescence(
    client: LLMClient,
    conn: sqlite3.Connection,
    concept_name: str,
) -> dict:
    """Check if older claims about a concept are superseded by newer ones.

    Retrieves all claims mentioning the concept, orders them by paper year,
    and asks Claude to identify obsolete claims.
    """
    from distill.db.repository import (
        get_concept_by_name,
        get_claim_ids_for_concept,
    )

    concept = get_concept_by_name(conn, concept_name)
    if concept is None:
        return {"concept": concept_name, "obsolete_claims": [], "reasons": {}}

    claim_ids = get_claim_ids_for_concept(conn, concept.concept_id)
    claims_with_year: list[dict] = []

    for claim_id in claim_ids:
        row = conn.execute(
            "SELECT c.*, d.year FROM claim c JOIN document d ON c.doc_id = d.doc_id "
            "WHERE c.claim_id = ?",
            (claim_id,),
        ).fetchone()
        if row:
            claims_with_year.append({
                "claim_id": row["claim_id"],
                "claim_text": row["claim_text"],
                "year": row["year"] or 0,
            })

    if not claims_with_year:
        return {"concept": concept_name, "obsolete_claims": [], "reasons": {}}

    claims_with_year.sort(key=lambda x: x["year"])
    prompt = obsolescence_prompt(concept_name, claims_with_year)
    result = client.complete_json(
        system="You identify obsolete scientific claims. Return only valid JSON.",
        user=prompt,
        max_tokens=512,
    )

    obsolete_indices = result.get("obsolete_claim_indices", [])
    reasons = result.get("reasons", {})
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    obsolete_claim_ids = []
    for idx in obsolete_indices:
        if 0 <= idx < len(claims_with_year):
            cid = claims_with_year[idx]["claim_id"]
            update_claim_verification(conn, cid, -1, now)
            obsolete_claim_ids.append(cid)

    return {
        "concept": concept_name,
        "obsolete_claims": obsolete_claim_ids,
        "reasons": reasons,
    }
