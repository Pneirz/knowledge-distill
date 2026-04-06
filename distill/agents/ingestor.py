import hashlib
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from distill.config import Config
from distill.db.models import Document
from distill.db.repository import get_document_by_hash, insert_document


def detect_source_type(file_path: Path) -> str:
    """Infer source type from file extension.

    Returns 'pdf', 'html', or 'github'.
    Raises ValueError for unsupported extensions.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm", ".mhtml"}:
        return "html"
    if suffix in {".md", ".txt"} or file_path.name == "README":
        return "github"
    raise ValueError(f"Unsupported file type: {suffix} ({file_path.name})")


def compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file contents for duplicate detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def extract_metadata_from_filename(file_path: Path) -> dict:
    """Parse title, authors, and year from the filename.

    Expected format (all optional): "Author et al - Year - Title.pdf"
    Falls back to using the stem as the title if parsing fails.
    """
    stem = file_path.stem
    # Pattern: "Author et al - 2017 - Attention Is All You Need"
    pattern = r"^(.+?)\s*-\s*(\d{4})\s*-\s*(.+)$"
    match = re.match(pattern, stem)
    if match:
        author_part, year_str, title = match.groups()
        authors = [a.strip() for a in re.split(r",|and|&", author_part) if a.strip()]
        return {"title": title.strip(), "authors": authors, "year": int(year_str)}
    return {"title": stem, "authors": [], "year": None}


def detect_new_files(inbox_path: Path) -> list[Path]:
    """Return all files in inbox that have a supported extension."""
    supported = {".pdf", ".html", ".htm", ".mhtml", ".md", ".txt"}
    return [
        f for f in inbox_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported
    ]


def ingest_file(
    conn: sqlite3.Connection,
    file_path: Path,
    cfg: Config,
) -> str | None:
    """Ingest a single file into the knowledge base.

    Pipeline:
    1. Compute SHA-256 hash and check for duplicates.
    2. Copy file to 01_raw/ with a doc_id-based name.
    3. Insert Document record in DB with status='ingested'.
    4. Return the new doc_id, or None if the file was already ingested.
    """
    content_hash = compute_hash(file_path)

    # Duplicate detection: skip files already in the DB
    existing = get_document_by_hash(conn, content_hash)
    if existing is not None:
        return None

    doc_id = str(uuid.uuid4())
    source_type = detect_source_type(file_path)
    metadata = extract_metadata_from_filename(file_path)

    # Copy to 01_raw/ preserving the original extension
    raw_dest = cfg.raw_path / f"{doc_id}{file_path.suffix}"
    shutil.copy2(file_path, raw_dest)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    doc = Document(
        doc_id=doc_id,
        title=metadata["title"],
        source_type=source_type,
        content_hash=content_hash,
        status="ingested",
        ingested_at=now,
        updated_at=now,
        authors=metadata["authors"],
        year=metadata["year"],
        raw_path=str(raw_dest.relative_to(cfg.data_root)),
    )
    insert_document(conn, doc)
    return doc_id


def run_ingestor(conn: sqlite3.Connection, cfg: Config) -> list[str]:
    """Process all files in the inbox directory.

    Returns a list of newly ingested doc_ids (skips duplicates silently).
    """
    new_files = detect_new_files(cfg.inbox_path)
    ingested = []
    for file_path in new_files:
        doc_id = ingest_file(conn, file_path, cfg)
        if doc_id is not None:
            ingested.append(doc_id)
    return ingested
