# %%
import shutil
from pathlib import Path

import pytest

from distill.agents.ingestor import (
    compute_hash,
    detect_new_files,
    detect_source_type,
    extract_metadata_from_filename,
    ingest_file,
    run_ingestor,
)
from distill.config import Config
from distill.db.repository import get_document, get_document_by_hash


@pytest.fixture
def sample_pdf(tmp_data_root: Path) -> Path:
    """Create a minimal fake PDF file in 00_inbox/ for ingestor tests."""
    inbox = tmp_data_root / "00_inbox"
    dest = inbox / "Vaswani et al - 2017 - Attention Is All You Need.pdf"
    # Write binary content that looks like a PDF header
    dest.write_bytes(b"%PDF-1.4 fake content for testing purposes")
    return dest


@pytest.fixture
def sample_html(tmp_data_root: Path) -> Path:
    """Create a minimal HTML file in 00_inbox/."""
    inbox = tmp_data_root / "00_inbox"
    dest = inbox / "test-article.html"
    dest.write_text("<html><body><p>Test article content.</p></body></html>")
    return dest


def test_detect_source_type_pdf():
    assert detect_source_type(Path("paper.pdf")) == "pdf"


def test_detect_source_type_html():
    assert detect_source_type(Path("article.html")) == "html"
    assert detect_source_type(Path("article.htm")) == "html"


def test_detect_source_type_github():
    assert detect_source_type(Path("README.md")) == "github"


def test_detect_source_type_unsupported():
    with pytest.raises(ValueError, match="Unsupported"):
        detect_source_type(Path("file.docx"))


def test_compute_hash_deterministic(sample_pdf: Path):
    """Same file produces identical hash on repeated calls."""
    hash1 = compute_hash(sample_pdf)
    hash2 = compute_hash(sample_pdf)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex


def test_compute_hash_different_files(sample_pdf: Path, sample_html: Path):
    """Different files produce different hashes."""
    assert compute_hash(sample_pdf) != compute_hash(sample_html)


def test_extract_metadata_with_full_pattern():
    """Parses 'Author et al - Year - Title' pattern correctly."""
    meta = extract_metadata_from_filename(
        Path("Vaswani et al - 2017 - Attention Is All You Need.pdf")
    )
    assert meta["year"] == 2017
    assert "Attention Is All You Need" in meta["title"]


def test_extract_metadata_fallback():
    """Falls back to stem as title when filename does not match pattern."""
    meta = extract_metadata_from_filename(Path("random_name.pdf"))
    assert meta["title"] == "random_name"
    assert meta["year"] is None


def test_ingest_file_registers_in_db(db, tmp_data_root: Path, sample_pdf: Path):
    """Ingesting a file creates a Document record in the database."""
    cfg = Config(data_root=tmp_data_root)
    doc_id = ingest_file(db, sample_pdf, cfg)

    assert doc_id is not None
    doc = get_document(db, doc_id)
    assert doc is not None
    assert doc.status == "ingested"
    assert doc.source_type == "pdf"
    assert doc.year == 2017


def test_ingest_file_copies_to_raw(db, tmp_data_root: Path, sample_pdf: Path):
    """Ingesting a file copies it to 01_raw/."""
    cfg = Config(data_root=tmp_data_root)
    doc_id = ingest_file(db, sample_pdf, cfg)

    raw_files = list((tmp_data_root / "01_raw").iterdir())
    assert len(raw_files) == 1
    assert raw_files[0].suffix == ".pdf"


def test_ingest_duplicate_returns_none(db, tmp_data_root: Path, sample_pdf: Path):
    """Ingesting the same file twice skips the duplicate and returns None."""
    cfg = Config(data_root=tmp_data_root)
    doc_id_1 = ingest_file(db, sample_pdf, cfg)
    doc_id_2 = ingest_file(db, sample_pdf, cfg)

    assert doc_id_1 is not None
    assert doc_id_2 is None


def test_run_ingestor_processes_inbox(db, tmp_data_root: Path, sample_pdf: Path, sample_html: Path):
    """run_ingestor returns all newly ingested doc_ids."""
    cfg = Config(data_root=tmp_data_root)
    doc_ids = run_ingestor(db, cfg)

    assert len(doc_ids) == 2


def test_detect_new_files_excludes_unsupported(tmp_data_root: Path):
    """detect_new_files ignores files with unsupported extensions."""
    inbox = tmp_data_root / "00_inbox"
    (inbox / "document.docx").write_bytes(b"fake docx")
    (inbox / "paper.pdf").write_bytes(b"%PDF fake")

    found = detect_new_files(inbox)
    names = [f.name for f in found]
    assert "paper.pdf" in names
    assert "document.docx" not in names
