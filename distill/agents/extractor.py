import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from distill.config import Config
from distill.db.models import Claim, Concept
from distill.db.repository import (
    get_chunks_by_doc,
    get_document,
    insert_claim_concept,
    insert_claims,
    update_document_status,
    upsert_concept,
)
from distill.llm.client import BaseLLMClient
from distill.llm.prompts import extraction_system_prompt


def _build_extraction_prompt(chunk_text: str, doc_title: str, year: int | None) -> str:
    """Construct the user prompt for the extraction call."""
    year_str = f" ({year})" if year else ""
    return (
        f"Paper: {doc_title}{year_str}\n\n"
        f"Text chunk:\n{chunk_text}\n\n"
        "Extract all claims and concepts from this chunk following the instructions."
    )


def extract_from_chunk(
    client: BaseLLMClient,
    chunk_text: str,
    doc_title: str,
    year: int | None,
) -> dict:
    """Call Claude to extract claims and concepts from a single chunk.

    Returns the raw parsed JSON dict from the LLM response.
    Expected keys: 'claims', 'concepts'.
    """
    system = extraction_system_prompt()
    user = _build_extraction_prompt(chunk_text, doc_title, year)
    return client.complete_json(system, user, max_tokens=2048)


def process_chunk(
    client: BaseLLMClient,
    conn: sqlite3.Connection,
    chunk_id: str,
    doc_id: str,
    doc_title: str,
    doc_year: int | None,
) -> tuple[list[str], list[str]]:
    """Extract and persist claims and concepts for a single chunk.

    Returns (claim_ids, concept_ids).
    """
    from distill.db.repository import get_chunk

    chunk = get_chunk(conn, chunk_id)
    if chunk is None:
        raise ValueError(f"Chunk not found: {chunk_id}")

    result = extract_from_chunk(client, chunk.text, doc_title, doc_year)

    raw_claims = result.get("claims", [])
    raw_concepts = result.get("concepts", [])
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Build and insert Concept records first (claims reference them)
    concept_name_to_id: dict[str, str] = {}
    for rc in raw_concepts:
        name = rc.get("name", "").strip().lower()
        if not name:
            continue
        from distill.db.repository import get_concept_by_name
        existing = get_concept_by_name(conn, name)
        concept_id = existing.concept_id if existing else str(uuid.uuid4())
        concept = Concept(
            concept_id=concept_id,
            name=name,
            created_at=now,
            updated_at=now,
            aliases=[a.lower() for a in rc.get("aliases", [])],
            definition=rc.get("definition"),
            domain=rc.get("domain"),
        )
        upsert_concept(conn, concept)
        concept_name_to_id[name] = concept_id

    # Build and insert Claim records
    claims: list[Claim] = []
    claim_concept_links: list[tuple[str, str]] = []

    for rc in raw_claims:
        claim_text = rc.get("claim_text", "").strip()
        if not claim_text:
            continue

        claim_id = str(uuid.uuid4())
        claim = Claim(
            claim_id=claim_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            claim_text=claim_text,
            claim_type=rc.get("claim_type", "finding"),
            confidence=float(rc.get("confidence", 1.0)),
            raw_quote=rc.get("raw_quote"),
            page_ref=chunk.page_start,
        )
        claims.append(claim)

        # Record concept associations for this claim
        for concept_name in rc.get("concepts", []):
            normalized = concept_name.strip().lower()
            if normalized in concept_name_to_id:
                claim_concept_links.append((claim_id, concept_name_to_id[normalized]))

    if claims:
        insert_claims(conn, claims)

    for claim_id, concept_id in claim_concept_links:
        insert_claim_concept(conn, claim_id, concept_id)

    return [c.claim_id for c in claims], list(concept_name_to_id.values())


def run_extractor(
    client: BaseLLMClient,
    conn: sqlite3.Connection,
    doc_id: str,
    cfg: Config,
) -> None:
    """Run extraction for all chunks of a document.

    Saves extraction summary JSON to 03_extracted/ and updates status to 'extracted'.
    """
    doc = get_document(conn, doc_id)
    if doc is None:
        raise ValueError(f"Document not found: {doc_id}")
    if doc.status != "parsed":
        raise ValueError(f"Document {doc_id} has status '{doc.status}', expected 'parsed'")

    chunks = get_chunks_by_doc(conn, doc_id)
    all_claim_ids: list[str] = []
    all_concept_ids: list[str] = []

    for chunk in chunks:
        claim_ids, concept_ids = process_chunk(
            client, conn, chunk.chunk_id, doc_id, doc.title, doc.year
        )
        all_claim_ids.extend(claim_ids)
        all_concept_ids.extend(concept_ids)

    # Save extraction summary
    summary = {
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "claim_count": len(all_claim_ids),
        "concept_count": len(set(all_concept_ids)),
        "claim_ids": all_claim_ids,
    }
    dest = cfg.extracted_path / f"{doc_id}.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    update_document_status(
        conn, doc_id, "extracted",
        extracted_path=str(dest.relative_to(cfg.data_root))
    )
