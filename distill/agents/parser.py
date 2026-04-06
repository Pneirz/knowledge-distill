import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import tiktoken

from distill.config import Config
from distill.db.models import Chunk
from distill.db.repository import get_document, insert_chunks, update_document_status

# Tokenizer for counting chunk sizes (model-agnostic cl100k_base encoding)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------

def parse_pdf(raw_path: Path) -> dict[str, Any]:
    """Extract structured content from a PDF using PyMuPDF.

    Returns:
        {
          'full_text': str,
          'sections': [{'title': str, 'text': str, 'page_start': int, 'page_end': int}],
          'references': [str],
        }
    """
    import fitz  # pymupdf

    doc = fitz.open(str(raw_path))
    sections: list[dict] = []
    current_section_title = "Introduction"
    current_text_parts: list[str] = []
    current_page_start = 1
    full_text_parts: list[str] = []

    # Heuristic: lines in ALL CAPS or Title Case with short length are section headers
    header_pattern = re.compile(r"^[A-Z][A-Z\s\d]{2,50}$|^\d+\.\s+[A-Z][a-zA-Z\s]+$")

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        full_text_parts.append(page_text)

        for line in page_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if header_pattern.match(stripped) and len(stripped) < 80:
                # Save current section
                if current_text_parts:
                    sections.append({
                        "title": current_section_title,
                        "text": " ".join(current_text_parts).strip(),
                        "page_start": current_page_start,
                        "page_end": page_num,
                    })
                current_section_title = stripped
                current_text_parts = []
                current_page_start = page_num
            else:
                current_text_parts.append(stripped)

    # Flush last section
    if current_text_parts:
        sections.append({
            "title": current_section_title,
            "text": " ".join(current_text_parts).strip(),
            "page_start": current_page_start,
            "page_end": len(doc),
        })

    # Extract references section if present
    references: list[str] = []
    for section in sections:
        if "reference" in section["title"].lower():
            references = [
                line.strip()
                for line in section["text"].split("\n")
                if line.strip()
            ]

    doc.close()
    return {
        "full_text": "\n".join(full_text_parts),
        "sections": sections,
        "references": references,
    }


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def parse_html(raw_path: Path) -> dict[str, Any]:
    """Extract clean article text from a saved HTML file using BeautifulSoup.

    Strips navigation, ads, footers. Returns the same structure as parse_pdf.
    """
    from bs4 import BeautifulSoup

    content = raw_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(content, "lxml")

    # Remove noise elements
    for tag in soup(["nav", "header", "footer", "aside", "script", "style", "figure"]):
        tag.decompose()

    # Find main content container
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"post|article|content|body", re.I))
        or soup.find("body")
    )

    sections: list[dict] = []
    current_title = "Introduction"
    current_parts: list[str] = []

    for element in main.descendants if main else []:
        if not hasattr(element, "name"):
            continue
        if element.name in {"h1", "h2", "h3"}:
            if current_parts:
                sections.append({
                    "title": current_title,
                    "text": " ".join(current_parts).strip(),
                    "page_start": None,
                    "page_end": None,
                })
            current_title = element.get_text(strip=True)
            current_parts = []
        elif element.name in {"p", "li", "blockquote"}:
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_parts.append(text)

    if current_parts:
        sections.append({
            "title": current_title,
            "text": " ".join(current_parts).strip(),
            "page_start": None,
            "page_end": None,
        })

    full_text = " ".join(s["text"] for s in sections)
    return {"full_text": full_text, "sections": sections, "references": []}


# ---------------------------------------------------------------------------
# GitHub/Markdown parser
# ---------------------------------------------------------------------------

def parse_github_repo(raw_path: Path) -> dict[str, Any]:
    """Parse a Markdown/README file as a structured document.

    Splits on H2/H3 headers to create sections.
    """
    content = raw_path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    sections: list[dict] = []
    current_title = "Overview"
    current_parts: list[str] = []

    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if current_parts:
                sections.append({
                    "title": current_title,
                    "text": "\n".join(current_parts).strip(),
                    "page_start": None,
                    "page_end": None,
                })
            current_title = line.lstrip("#").strip()
            current_parts = []
        else:
            current_parts.append(line)

    if current_parts:
        sections.append({
            "title": current_title,
            "text": "\n".join(current_parts).strip(),
            "page_start": None,
            "page_end": None,
        })

    full_text = "\n".join(s["text"] for s in sections)
    return {"full_text": full_text, "sections": sections, "references": []}


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_sections(
    sections: list[dict],
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[dict]:
    """Split sections into token-bounded chunks with overlap.

    Splits on sentence boundaries ('. ', '! ', '? ') to avoid cutting
    mid-sentence. Each chunk carries the section title and page references
    from its parent section.

    Returns a list of dicts:
        {'text': str, 'section': str, 'page_start': int|None, 'page_end': int|None}
    """
    chunks: list[dict] = []

    for section in sections:
        text = section["text"]
        if not text.strip():
            continue

        # Split into sentences as atomic units
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_sentences: list[str] = []
        current_tokens = 0
        overlap_sentences: list[str] = []

        for sentence in sentences:
            sentence_tokens = _count_tokens(sentence)
            if current_tokens + sentence_tokens > max_tokens and current_sentences:
                chunks.append({
                    "text": " ".join(current_sentences),
                    "section": section["title"],
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                })
                # Carry overlap sentences into the next chunk
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    t = _count_tokens(s)
                    if overlap_count + t > overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_count += t
                current_sentences = overlap_sentences + [sentence]
                current_tokens = sum(_count_tokens(s) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        if current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "section": section["title"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
            })

    return chunks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def parse_document(
    conn: sqlite3.Connection,
    doc_id: str,
    cfg: Config,
) -> list[str]:
    """Parse a document from 01_raw/ and store chunks in the DB.

    Dispatch is based on document.source_type.
    Saves parsed JSON to 02_parsed/ and updates document status to 'parsed'.
    Returns list of inserted chunk_ids.
    """
    doc = get_document(conn, doc_id)
    if doc is None:
        raise ValueError(f"Document not found: {doc_id}")
    if doc.status != "ingested":
        raise ValueError(f"Document {doc_id} has status '{doc.status}', expected 'ingested'")

    raw_path = cfg.data_root / doc.raw_path

    # Dispatch to the correct parser based on source type
    if doc.source_type == "pdf":
        parsed = parse_pdf(raw_path)
    elif doc.source_type == "html":
        parsed = parse_html(raw_path)
    elif doc.source_type == "github":
        parsed = parse_github_repo(raw_path)
    else:
        raise ValueError(f"Unknown source_type: {doc.source_type}")

    # Chunk sections with configured parameters
    raw_chunks = chunk_sections(
        parsed["sections"],
        max_tokens=cfg.chunk_max_tokens,
        overlap_tokens=cfg.chunk_overlap,
    )

    # Build Chunk dataclass instances with UUIDs
    chunks: list[Chunk] = []
    for idx, c in enumerate(raw_chunks):
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=c["text"],
            chunk_index=idx,
            section=c["section"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            token_count=_count_tokens(c["text"]),
        )
        chunks.append(chunk)

    # Persist chunks
    insert_chunks(conn, chunks)

    # Save parsed JSON to 02_parsed/
    parsed_dest = cfg.parsed_path / f"{doc_id}.json"
    parsed_data = {
        "doc_id": doc_id,
        "sections": parsed["sections"],
        "references": parsed["references"],
        "chunk_count": len(chunks),
    }
    parsed_dest.write_text(json.dumps(parsed_data, indent=2, ensure_ascii=False), encoding="utf-8")

    update_document_status(
        conn, doc_id, "parsed", parsed_path=str(parsed_dest.relative_to(cfg.data_root))
    )

    return [c.chunk_id for c in chunks]


def run_parser(
    conn: sqlite3.Connection,
    cfg: Config,
    doc_ids: list[str] | None = None,
) -> dict[str, list[str]]:
    """Parse one or more documents.

    If doc_ids is None, parses all documents with status='ingested'.
    Returns a mapping of {doc_id: [chunk_ids]}.
    """
    from distill.db.repository import list_documents

    if doc_ids is None:
        docs = list_documents(conn, status="ingested")
        doc_ids = [d.doc_id for d in docs]

    results: dict[str, list[str]] = {}
    for doc_id in doc_ids:
        results[doc_id] = parse_document(conn, doc_id, cfg)
    return results
