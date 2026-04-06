from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

# Type aliases for constrained string fields
SourceType = Literal["pdf", "html", "github"]
ClaimType = Literal["finding", "method", "limitation", "comparison", "definition", "hypothesis"]
DocStatus = Literal["ingested", "parsed", "extracted", "compiled", "verified"]
Relation = Literal["supports", "contradicts", "refines", "defines", "uses", "extends", "cites"]


@dataclass
class Document:
    """Primary source unit. Immutable after ingestion except for status updates."""

    doc_id: str
    title: str
    source_type: SourceType
    content_hash: str
    status: DocStatus
    ingested_at: datetime
    updated_at: datetime
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    url: Optional[str] = None
    raw_path: Optional[str] = None
    parsed_path: Optional[str] = None
    extracted_path: Optional[str] = None
    wiki_path: Optional[str] = None


@dataclass
class Chunk:
    """Indexable text fragment with provenance back to its parent document."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    token_count: Optional[int] = None
    embedding_id: Optional[str] = None


@dataclass
class Claim:
    """Atomic knowledge unit extracted from a chunk.

    raw_quote is mandatory for traceability: it must be a verbatim excerpt
    from the source chunk that supports the claim.
    """

    claim_id: str
    doc_id: str
    chunk_id: str
    claim_text: str
    claim_type: ClaimType
    confidence: float = 1.0
    # 0 = unverified, 1 = verified, -1 = contradicted / untraceable
    verified: int = 0
    verified_at: Optional[datetime] = None
    page_ref: Optional[int] = None
    raw_quote: Optional[str] = None


@dataclass
class Concept:
    """Canonical knowledge entity that aggregates claims across documents."""

    concept_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    aliases: list[str] = field(default_factory=list)
    definition: Optional[str] = None
    wiki_path: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class EvidenceLink:
    """Typed directed relation between two objects in the knowledge graph."""

    link_id: str
    from_type: str  # 'claim' | 'concept' | 'document'
    from_id: str
    to_type: str
    to_id: str
    relation: Relation
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
