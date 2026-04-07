# %%
import sqlite3

from distill.db.schema import initialize_db


def test_initialize_db_creates_all_tables(db):
    """All expected tables must exist after initialization."""
    expected_tables = {
        "document", "chunk", "claim", "concept",
        "evidence_link", "claim_concept", "audit_log",
    }
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    actual = {r["name"] for r in rows}
    assert expected_tables.issubset(actual)


def test_initialize_db_idempotent():
    """Calling initialize_db twice on the same connection must not raise."""
    conn = sqlite3.connect(":memory:")
    initialize_db(conn)
    initialize_db(conn)  # second call: IF NOT EXISTS prevents errors
    conn.close()


def test_foreign_keys_are_enforced(db):
    """Inserting a chunk referencing a non-existent document must fail."""
    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO chunk (chunk_id, doc_id, text, chunk_index)
            VALUES ('c1', 'nonexistent', 'text', 0)
            """
        )


def test_document_status_check_constraint(db, sample_document):
    """Inserting a document with an invalid status must fail."""
    import pytest
    from distill.db.repository import insert_document

    insert_document(db, sample_document)

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "UPDATE document SET status = 'invalid_status' WHERE doc_id = ?",
            (sample_document.doc_id,),
        )


def test_claim_lifecycle_columns_exist(db):
    """Claim table includes lifecycle fields introduced for temporal validity."""
    rows = db.execute("PRAGMA table_info(claim)").fetchall()
    columns = {row["name"] for row in rows}
    assert {"lifecycle_status", "superseded_by_claim_id", "lifecycle_updated_at"}.issubset(columns)
