import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from distill.db.models import AuditEvent, Chunk, Claim, Concept, Document, EvidenceLink


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_list(value: list) -> str:
    return json.dumps(value)


def _deserialize_list(value: Optional[str]) -> list:
    if value is None:
        return []
    return json.loads(value)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def insert_document(conn: sqlite3.Connection, doc: Document) -> None:
    """Insert a new document record. Raises IntegrityError on duplicate hash."""
    conn.execute(
        """
        INSERT INTO document (
            doc_id, title, source_type, authors, year, url,
            raw_path, parsed_path, extracted_path, wiki_path,
            content_hash, status, ingested_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc.doc_id,
            doc.title,
            doc.source_type,
            _serialize_list(doc.authors),
            doc.year,
            doc.url,
            doc.raw_path,
            doc.parsed_path,
            doc.extracted_path,
            doc.wiki_path,
            doc.content_hash,
            doc.status,
            doc.ingested_at.isoformat(),
            doc.updated_at.isoformat(),
        ),
    )
    conn.commit()


def get_document(conn: sqlite3.Connection, doc_id: str) -> Optional[Document]:
    """Return a Document by primary key, or None if not found."""
    row = conn.execute(
        "SELECT * FROM document WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return _row_to_document(row) if row else None


def get_document_by_hash(conn: sqlite3.Connection, content_hash: str) -> Optional[Document]:
    """Return a Document by content hash (used for duplicate detection)."""
    row = conn.execute(
        "SELECT * FROM document WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return _row_to_document(row) if row else None


def update_document_status(
    conn: sqlite3.Connection,
    doc_id: str,
    status: str,
    **fields,
) -> None:
    """Update document status and any additional fields supplied as kwargs.

    Supported extra fields: raw_path, parsed_path, extracted_path, wiki_path.
    """
    allowed = {"raw_path", "parsed_path", "extracted_path", "wiki_path"}
    extra = {k: v for k, v in fields.items() if k in allowed}
    extra["status"] = status
    extra["updated_at"] = _now()

    set_clause = ", ".join(f"{k} = ?" for k in extra)
    values = list(extra.values()) + [doc_id]
    conn.execute(f"UPDATE document SET {set_clause} WHERE doc_id = ?", values)
    conn.commit()


def list_documents(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
) -> list[Document]:
    """Return all documents, optionally filtered by status."""
    if status:
        rows = conn.execute(
            "SELECT * FROM document WHERE status = ? ORDER BY ingested_at", (status,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM document ORDER BY ingested_at"
        ).fetchall()
    return [_row_to_document(r) for r in rows]


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        title=row["title"],
        source_type=row["source_type"],
        content_hash=row["content_hash"],
        status=row["status"],
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        authors=_deserialize_list(row["authors"]),
        year=row["year"],
        url=row["url"],
        raw_path=row["raw_path"],
        parsed_path=row["parsed_path"],
        extracted_path=row["extracted_path"],
        wiki_path=row["wiki_path"],
    )


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

def insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Bulk insert chunks for a document."""
    conn.executemany(
        """
        INSERT INTO chunk (
            chunk_id, doc_id, section, text,
            page_start, page_end, chunk_index, token_count, embedding_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                c.chunk_id, c.doc_id, c.section, c.text,
                c.page_start, c.page_end, c.chunk_index,
                c.token_count, c.embedding_id,
            )
            for c in chunks
        ],
    )
    conn.commit()


def get_chunks_by_doc(conn: sqlite3.Connection, doc_id: str) -> list[Chunk]:
    """Return all chunks for a document ordered by chunk_index."""
    rows = conn.execute(
        "SELECT * FROM chunk WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
    ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Optional[Chunk]:
    """Return a single chunk by primary key."""
    row = conn.execute(
        "SELECT * FROM chunk WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return _row_to_chunk(row) if row else None


def get_all_chunks(conn: sqlite3.Connection) -> list[Chunk]:
    """Return all chunks across all documents ordered by doc and index."""
    rows = conn.execute(
        "SELECT * FROM chunk ORDER BY doc_id, chunk_index"
    ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def update_chunk_embedding_id(
    conn: sqlite3.Connection,
    chunk_id: str,
    embedding_id: str,
) -> None:
    """Set the FAISS embedding_id for a chunk after indexing."""
    conn.execute(
        "UPDATE chunk SET embedding_id = ? WHERE chunk_id = ?",
        (embedding_id, chunk_id),
    )
    conn.commit()


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        text=row["text"],
        chunk_index=row["chunk_index"],
        section=row["section"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        token_count=row["token_count"],
        embedding_id=row["embedding_id"],
    )


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

def insert_claims(conn: sqlite3.Connection, claims: list[Claim]) -> None:
    """Bulk insert claims extracted from a document."""
    conn.executemany(
        """
        INSERT INTO claim (
            claim_id, doc_id, chunk_id, claim_text, claim_type,
            confidence, verified, verified_at, page_ref, raw_quote,
            lifecycle_status, superseded_by_claim_id, lifecycle_updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                c.claim_id, c.doc_id, c.chunk_id, c.claim_text, c.claim_type,
                c.confidence, c.verified,
                c.verified_at.isoformat() if c.verified_at else None,
                c.page_ref, c.raw_quote,
                c.lifecycle_status, c.superseded_by_claim_id,
                c.lifecycle_updated_at.isoformat() if c.lifecycle_updated_at else None,
            )
            for c in claims
        ],
    )
    conn.commit()


def get_claims_by_doc(conn: sqlite3.Connection, doc_id: str) -> list[Claim]:
    """Return all claims for a document."""
    rows = conn.execute(
        "SELECT * FROM claim WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    return [_row_to_claim(r) for r in rows]


def get_claims_by_chunk(conn: sqlite3.Connection, chunk_id: str) -> list[Claim]:
    """Return all claims extracted from a specific chunk."""
    rows = conn.execute(
        "SELECT * FROM claim WHERE chunk_id = ? ORDER BY claim_id", (chunk_id,)
    ).fetchall()
    return [_row_to_claim(r) for r in rows]


def get_unverified_claims(conn: sqlite3.Connection, limit: int = 50) -> list[Claim]:
    """Return claims not yet verified (verified = 0)."""
    rows = conn.execute(
        "SELECT * FROM claim WHERE verified = 0 LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_claim(r) for r in rows]


def update_claim_verification(
    conn: sqlite3.Connection,
    claim_id: str,
    verified: int,
    verified_at: str,
) -> None:
    """Set the verification status of a claim."""
    conn.execute(
        "UPDATE claim SET verified = ?, verified_at = ? WHERE claim_id = ?",
        (verified, verified_at, claim_id),
    )
    conn.commit()


def update_claim_lifecycle(
    conn: sqlite3.Connection,
    claim_id: str,
    lifecycle_status: str,
    superseded_by_claim_id: Optional[str] = None,
    lifecycle_updated_at: Optional[str] = None,
) -> None:
    """Update lifecycle metadata for a claim."""
    timestamp = lifecycle_updated_at or _now()
    conn.execute(
        """
        UPDATE claim
        SET lifecycle_status = ?, superseded_by_claim_id = ?, lifecycle_updated_at = ?
        WHERE claim_id = ?
        """,
        (lifecycle_status, superseded_by_claim_id, timestamp, claim_id),
    )
    conn.commit()


def _row_to_claim(row: sqlite3.Row) -> Claim:
    return Claim(
        claim_id=row["claim_id"],
        doc_id=row["doc_id"],
        chunk_id=row["chunk_id"],
        claim_text=row["claim_text"],
        claim_type=row["claim_type"],
        confidence=row["confidence"],
        verified=row["verified"],
        verified_at=_parse_dt(row["verified_at"]),
        page_ref=row["page_ref"],
        raw_quote=row["raw_quote"],
        lifecycle_status=row["lifecycle_status"] or "active",
        superseded_by_claim_id=row["superseded_by_claim_id"],
        lifecycle_updated_at=_parse_dt(row["lifecycle_updated_at"]),
    )


# ---------------------------------------------------------------------------
# Concept
# ---------------------------------------------------------------------------

def upsert_concept(conn: sqlite3.Connection, concept: Concept) -> None:
    """Insert or update a concept by name (unique constraint)."""
    conn.execute(
        """
        INSERT INTO concept (
            concept_id, name, aliases, definition, wiki_path, domain,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            aliases    = excluded.aliases,
            definition = excluded.definition,
            wiki_path  = excluded.wiki_path,
            domain     = excluded.domain,
            updated_at = excluded.updated_at
        """,
        (
            concept.concept_id,
            concept.name,
            _serialize_list(concept.aliases),
            concept.definition,
            concept.wiki_path,
            concept.domain,
            concept.created_at.isoformat(),
            concept.updated_at.isoformat(),
        ),
    )
    conn.commit()


def get_concept_by_name(conn: sqlite3.Connection, name: str) -> Optional[Concept]:
    """Return a concept by its canonical name."""
    row = conn.execute(
        "SELECT * FROM concept WHERE name = ?", (name,)
    ).fetchone()
    return _row_to_concept(row) if row else None


def get_concept(conn: sqlite3.Connection, concept_id: str) -> Optional[Concept]:
    """Return a concept by primary key."""
    row = conn.execute(
        "SELECT * FROM concept WHERE concept_id = ?", (concept_id,)
    ).fetchone()
    return _row_to_concept(row) if row else None


def list_concepts(conn: sqlite3.Connection) -> list[Concept]:
    """Return all concepts ordered by name."""
    rows = conn.execute("SELECT * FROM concept ORDER BY name").fetchall()
    return [_row_to_concept(r) for r in rows]


def _row_to_concept(row: sqlite3.Row) -> Concept:
    return Concept(
        concept_id=row["concept_id"],
        name=row["name"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        aliases=_deserialize_list(row["aliases"]),
        definition=row["definition"],
        wiki_path=row["wiki_path"],
        domain=row["domain"],
    )


# ---------------------------------------------------------------------------
# EvidenceLink
# ---------------------------------------------------------------------------

def insert_evidence_link(conn: sqlite3.Connection, link: EvidenceLink) -> None:
    """Insert a new evidence link between two knowledge objects."""
    conn.execute(
        """
        INSERT INTO evidence_link (
            link_id, from_type, from_id, to_type, to_id,
            relation, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            link.link_id,
            link.from_type,
            link.from_id,
            link.to_type,
            link.to_id,
            link.relation,
            link.confidence,
            link.created_at.isoformat(),
        ),
    )
    conn.commit()

    insert_audit_event(
        conn,
        AuditEvent(
            event_id=str(uuid4()),
            entity_type="evidence_link",
            entity_id=link.link_id,
            action="created",
            details_json=json.dumps(
                {
                    "from_type": link.from_type,
                    "from_id": link.from_id,
                    "to_type": link.to_type,
                    "to_id": link.to_id,
                    "relation": link.relation,
                    "confidence": link.confidence,
                }
            ),
        ),
    )


def get_links_from(
    conn: sqlite3.Connection,
    from_type: str,
    from_id: str,
) -> list[EvidenceLink]:
    """Return all links originating from a given object."""
    rows = conn.execute(
        "SELECT * FROM evidence_link WHERE from_type = ? AND from_id = ?",
        (from_type, from_id),
    ).fetchall()
    return [_row_to_link(r) for r in rows]


def get_contradictions(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return pairs of (from_id, to_id) connected by a 'contradicts' relation."""
    rows = conn.execute(
        "SELECT from_id, to_id FROM evidence_link WHERE relation = 'contradicts'"
    ).fetchall()
    return [(r["from_id"], r["to_id"]) for r in rows]


def _row_to_link(row: sqlite3.Row) -> EvidenceLink:
    return EvidenceLink(
        link_id=row["link_id"],
        from_type=row["from_type"],
        from_id=row["from_id"],
        to_type=row["to_type"],
        to_id=row["to_id"],
        relation=row["relation"],
        confidence=row["confidence"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ---------------------------------------------------------------------------
# ClaimConcept junction
# ---------------------------------------------------------------------------

def insert_claim_concept(
    conn: sqlite3.Connection,
    claim_id: str,
    concept_id: str,
    role: Optional[str] = None,
) -> None:
    """Link a claim to a concept. Idempotent (INSERT OR IGNORE)."""
    conn.execute(
        "INSERT OR IGNORE INTO claim_concept (claim_id, concept_id, role) VALUES (?, ?, ?)",
        (claim_id, concept_id, role),
    )
    conn.commit()


def get_concept_ids_for_claim(
    conn: sqlite3.Connection,
    claim_id: str,
) -> list[str]:
    """Return all concept_ids linked to a claim."""
    rows = conn.execute(
        "SELECT concept_id FROM claim_concept WHERE claim_id = ?", (claim_id,)
    ).fetchall()
    return [r["concept_id"] for r in rows]


def get_claim_ids_for_concept(
    conn: sqlite3.Connection,
    concept_id: str,
) -> list[str]:
    """Return all claim_ids linked to a concept."""
    rows = conn.execute(
        "SELECT claim_id FROM claim_concept WHERE concept_id = ?", (concept_id,)
    ).fetchall()
    return [r["claim_id"] for r in rows]


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

def insert_audit_event(conn: sqlite3.Connection, event: AuditEvent) -> None:
    """Persist an audit trail event."""
    conn.execute(
        """
        INSERT INTO audit_log (
            event_id, entity_type, entity_id, action, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.entity_type,
            event.entity_id,
            event.action,
            event.details_json,
            event.created_at.isoformat(),
        ),
    )
    conn.commit()


def list_audit_events(
    conn: sqlite3.Connection,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> list[AuditEvent]:
    """Return audit events ordered chronologically."""
    query = "SELECT * FROM audit_log"
    clauses = []
    params: list[str] = []

    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if entity_id:
        clauses.append("entity_id = ?")
        params.append(entity_id)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at"

    rows = conn.execute(query, params).fetchall()
    return [_row_to_audit_event(r) for r in rows]


def _row_to_audit_event(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        event_id=row["event_id"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        action=row["action"],
        details_json=row["details_json"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
