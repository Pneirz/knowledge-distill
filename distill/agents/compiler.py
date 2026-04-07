import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from distill.config import Config
from distill.db.models import AuditEvent
from distill.db.repository import (
    get_claims_by_doc,
    get_document,
    insert_audit_event,
    update_document_status,
)
from distill.llm.client import BaseLLMClient
from distill.llm.prompts import compilation_system_prompt


def _slugify(text: str) -> str:
    """Convert a title or name to a filesystem-safe slug."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:80]


def _generate_summary(client: BaseLLMClient, doc_title: str, claims_text: str) -> str:
    """Ask Claude to write a 150-250 word summary grounded in extracted claims."""
    system = compilation_system_prompt()
    user = (
        f"Paper: {doc_title}\n\n"
        f"Extracted claims:\n{claims_text}\n\n"
        "Write a concise summary (150-250 words) of this paper based on these claims."
    )
    return client.complete(system, user, max_tokens=512, temperature=0.2)


def generate_paper_note(
    client: BaseLLMClient,
    conn: sqlite3.Connection,
    doc_id: str,
    wiki_root: Path,
) -> Path:
    """Generate an Obsidian-compatible Markdown note for a paper.

    The note uses YAML frontmatter with Obsidian aliases and tags,
    and Obsidian wiki-links ([[concept]]) for concept references.
    """
    doc = get_document(conn, doc_id)
    if doc is None:
        raise ValueError(f"Document not found: {doc_id}")

    claims = get_claims_by_doc(conn, doc_id)

    # Build claims text for LLM summary
    claims_text = "\n".join(
        f"[{c.claim_type}] {c.claim_text}" for c in claims[:40]
    )
    summary = _generate_summary(client, doc.title, claims_text) if claims else ""

    # Collect unique concept names from claims
    concept_names: list[str] = []
    from distill.db.repository import get_concept_ids_for_claim, get_concept
    seen_concepts: set[str] = set()
    for claim in claims:
        for concept_id in get_concept_ids_for_claim(conn, claim.claim_id):
            concept = get_concept(conn, concept_id)
            if concept and concept.name not in seen_concepts:
                seen_concepts.add(concept.name)
                concept_names.append(concept.name)

    # Group claims by type for note sections
    claims_by_type: dict[str, list] = {}
    for claim in claims:
        claims_by_type.setdefault(claim.claim_type, []).append(claim)

    # Build frontmatter
    tags = ["ml"]
    if doc.year:
        tags.append(f"year-{doc.year}")
    if doc.source_type:
        tags.append(doc.source_type)

    frontmatter = {
        "doc_id": doc_id,
        "note_type": "paper",
        "title": doc.title,
        "authors": doc.authors,
        "year": doc.year,
        "url": doc.url,
        "source_type": doc.source_type,
        "status": doc.status,
        "tags": tags,
        "aliases": [],
        "concepts": concept_names[:20],
        "claim_count": len(claims),
        "verified_claims": sum(1 for c in claims if c.verified == 1),
        "compiled_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }

    # Build note body
    lines = [
        "---",
        yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False).strip(),
        "---",
        "",
        f"## Summary",
        "",
        summary.strip() if summary else "_Summary pending extraction._",
        "",
    ]

    # Key Claims section grouped by type
    lines.append("## Key Claims")
    lines.append("")
    for claim_type, type_claims in claims_by_type.items():
        lines.append(f"### {claim_type.capitalize()}")
        for claim in type_claims:
            quote = f'  > *"{claim.raw_quote}"*' if claim.raw_quote else ""
            page = f"  — page {claim.page_ref}" if claim.page_ref else ""
            lines.append(f"- {claim.claim_text}")
            if quote:
                lines.append(quote + page)
        lines.append("")

    # Related Concepts with Obsidian wiki-links
    if concept_names:
        lines.append("## Related Concepts")
        lines.append("")
        links = " · ".join(f"[[{name}]]" for name in concept_names[:15])
        lines.append(links)
        lines.append("")

    lines.append("## Open Questions")
    lines.append("")
    lines.append("_Add questions as you read._")
    lines.append("")

    if doc.url:
        lines.append("## Sources")
        lines.append("")
        lines.append(f"- [{doc.title}]({doc.url})")

    note_content = "\n".join(lines)

    # Write to 04_compiled_wiki/papers/
    dest = wiki_root / "papers" / f"{_slugify(doc.title)}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(note_content, encoding="utf-8")

    return dest


def generate_concept_note(
    conn: sqlite3.Connection,
    concept_name: str,
    wiki_root: Path,
) -> Path:
    """Generate an Obsidian note for a concept, linking to all papers that mention it."""
    from distill.db.repository import get_concept_by_name, get_claim_ids_for_concept, get_claims_by_doc

    concept = get_concept_by_name(conn, concept_name)
    if concept is None:
        raise ValueError(f"Concept not found: {concept_name}")

    claim_ids = get_claim_ids_for_concept(conn, concept.concept_id)

    # Find all documents that mention this concept
    doc_ids: set[str] = set()
    relevant_claims: list = []
    from distill.db.repository import get_chunk
    from distill.db.repository import _row_to_claim

    for claim_id in claim_ids:
        import sqlite3 as _sq
        row = conn.execute(
            "SELECT * FROM claim WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        if row:
            from distill.db.repository import _row_to_claim
            claim = _row_to_claim(row)
            doc_ids.add(claim.doc_id)
            relevant_claims.append(claim)

    # Collect document titles
    doc_titles: dict[str, str] = {}
    for doc_id in doc_ids:
        doc = get_document(conn, doc_id)
        if doc:
            doc_titles[doc_id] = doc.title

    frontmatter = {
        "concept_id": concept.concept_id,
        "note_type": "concept",
        "name": concept.name,
        "aliases": concept.aliases,
        "domain": concept.domain,
        "definition": concept.definition,
        "tags": ["ml", "concept"] + ([concept.domain] if concept.domain else []),
        "linked_docs": list(doc_ids),
        "claim_count": len(claim_ids),
        "updated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z",
    }

    lines = [
        "---",
        yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False).strip(),
        "---",
        "",
        "## Definition",
        "",
        concept.definition or "_Definition pending._",
        "",
        "## Papers that use this concept",
        "",
    ]

    for doc_id, title in doc_titles.items():
        lines.append(f"- [[{_slugify(title)}]] ({title})")
    lines.append("")

    lines.append("## Key Claims")
    lines.append("")
    for claim in relevant_claims[:10]:
        title = doc_titles.get(claim.doc_id, claim.doc_id)
        lines.append(f"- [{title}] {claim.claim_text}")
    lines.append("")

    note_content = "\n".join(lines)

    dest = wiki_root / "concepts" / f"{_slugify(concept.name)}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(note_content, encoding="utf-8")

    return dest


def run_compiler(
    client: BaseLLMClient,
    conn: sqlite3.Connection,
    doc_id: str,
    cfg: Config,
) -> None:
    """Compile paper and concept notes for a document.

    Updates document status to 'compiled'.
    """
    doc = get_document(conn, doc_id)
    if doc is None:
        raise ValueError(f"Document not found: {doc_id}")
    if doc.status != "extracted":
        raise ValueError(f"Document {doc_id} has status '{doc.status}', expected 'extracted'")

    paper_note_path = generate_paper_note(client, conn, doc_id, cfg.wiki_path)

    # Generate concept notes for all concepts linked to this document's claims
    from distill.db.repository import get_concept_ids_for_claim
    from distill.db.repository import get_concept
    claims = get_claims_by_doc(conn, doc_id)
    compiled_concepts: set[str] = set()
    for claim in claims:
        for concept_id in get_concept_ids_for_claim(conn, claim.claim_id):
            concept = get_concept(conn, concept_id)
            if concept and concept.name not in compiled_concepts:
                generate_concept_note(conn, concept.name, cfg.wiki_path)
                compiled_concepts.add(concept.name)

    update_document_status(
        conn, doc_id, "compiled",
        wiki_path=str(paper_note_path.relative_to(cfg.data_root))
    )
    insert_audit_event(
        conn,
        AuditEvent(
            event_id=str(uuid4()),
            entity_type="document",
            entity_id=doc_id,
            action="compiled",
            details_json=json.dumps(
                {
                    "wiki_path": str(paper_note_path.relative_to(cfg.data_root)),
                    "concept_notes": sorted(compiled_concepts),
                }
            ),
        ),
    )
