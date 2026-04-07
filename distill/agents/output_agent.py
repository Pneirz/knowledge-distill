import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from distill.db.repository import get_claims_by_doc, get_document, list_documents
from distill.llm.client import BaseLLMClient
from distill.llm.prompts import brief_system_prompt


def generate_brief(
    client: BaseLLMClient,
    conn: sqlite3.Connection,
    topic: str,
    doc_ids: list[str],
    output_path: Path,
) -> Path:
    """Generate a Markdown executive brief on a topic from selected documents.

    Gathers all claims from the specified documents and asks Claude to
    synthesize a structured brief. The output is saved to 06_outputs/.
    """
    # Collect claims and context from all selected documents
    sections: list[str] = []
    for doc_id in doc_ids:
        doc = get_document(conn, doc_id)
        if doc is None:
            continue
        claims = get_claims_by_doc(conn, doc_id)
        if not claims:
            continue

        claims_text = "\n".join(
            f"[{c.claim_type}] {c.claim_text}"
            for c in claims
            if c.verified == 1 and c.lifecycle_status != "superseded"
        )
        sections.append(f"## {doc.title} ({doc.year})\n{claims_text}")

    if not sections:
        raise ValueError("No verified claims found for the selected documents.")

    context = "\n\n".join(sections)
    system = brief_system_prompt()
    user = (
        f"Topic: {topic}\n\n"
        f"Claims from relevant papers:\n{context}\n\n"
        "Write a structured executive brief covering: state of the art, "
        "key findings, methods, open questions, and contradictions."
    )

    brief_text = client.complete(system, user, max_tokens=2048, temperature=0.2)

    # Write to output file
    dest = output_path / f"brief_{_slugify(topic)}_{_datestamp()}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(brief_text, encoding="utf-8")
    return dest


def generate_comparison_table(
    conn: sqlite3.Connection,
    doc_ids: list[str],
    dimensions: list[str],
    output_path: Path,
) -> Path:
    """Generate a Markdown comparison table across papers and dimensions.

    Dimensions are claim types or specific fields like 'architecture', 'dataset'.
    """
    rows: list[dict] = []
    for doc_id in doc_ids:
        doc = get_document(conn, doc_id)
        if doc is None:
            continue
        claims = get_claims_by_doc(conn, doc_id)
        row = {"title": doc.title, "year": doc.year or ""}
        for dim in dimensions:
            # Find the first claim whose type matches the dimension
            matching = [
                c for c in claims
                if c.claim_type == dim and c.verified == 1 and c.lifecycle_status != "superseded"
            ]
            row[dim] = matching[0].claim_text[:80] if matching else "—"
        rows.append(row)

    if not rows:
        raise ValueError("No documents found for comparison.")

    # Build Markdown table
    headers = ["Title", "Year"] + dimensions
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for row in rows:
        values = [row["title"], str(row["year"])] + [row.get(d, "—") for d in dimensions]
        data_lines.append("| " + " | ".join(values) + " |")

    table = "\n".join([header_line, separator] + data_lines)

    dest = output_path / f"comparison_{_datestamp()}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(table, encoding="utf-8")
    return dest


def generate_concept_map(
    conn: sqlite3.Connection,
    concept_names: list[str],
    output_path: Path,
) -> Path:
    """Generate a Mermaid concept map showing relations between concepts.

    Uses evidence_links to populate edges.
    """
    from distill.db.repository import get_concept_by_name, get_links_from

    lines = ["```mermaid", "graph TD"]

    for name in concept_names:
        concept = get_concept_by_name(conn, name)
        if concept is None:
            continue
        node_id = _slugify(name).replace("-", "_")
        lines.append(f'    {node_id}["{name}"]')

        # Add edges from evidence links
        links = get_links_from(conn, "concept", concept.concept_id)
        for link in links:
            from distill.db.repository import get_concept
            target = get_concept(conn, link.to_id)
            if target and target.name in concept_names:
                target_id = _slugify(target.name).replace("-", "_")
                lines.append(f'    {node_id} -->|{link.relation}| {target_id}')

    lines.append("```")
    mermaid = "\n".join(lines)

    dest = output_path / f"concept_map_{_datestamp()}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(mermaid, encoding="utf-8")
    return dest


def _slugify(text: str) -> str:
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)[:60]


def _datestamp() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y%m%d")
