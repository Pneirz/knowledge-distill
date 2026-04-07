import sqlite3

# Full DDL for the knowledge base schema.
# All tables use TEXT primary keys (UUID v4) for portability.
# JSON arrays are serialized as TEXT columns.
DDL = """
CREATE TABLE IF NOT EXISTS document (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source_type     TEXT NOT NULL
                    CHECK(source_type IN ('pdf', 'html', 'github')),
    authors         TEXT,
    year            INTEGER,
    url             TEXT,
    raw_path        TEXT,
    parsed_path     TEXT,
    extracted_path  TEXT,
    wiki_path       TEXT,
    content_hash    TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'ingested'
                    CHECK(status IN ('ingested', 'parsed', 'extracted', 'compiled', 'verified')),
    ingested_at     TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    section         TEXT,
    text            TEXT NOT NULL,
    page_start      INTEGER,
    page_end        INTEGER,
    chunk_index     INTEGER NOT NULL,
    token_count     INTEGER,
    embedding_id    TEXT
);

CREATE TABLE IF NOT EXISTS claim (
    claim_id        TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    chunk_id        TEXT NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    claim_text      TEXT NOT NULL,
    claim_type      TEXT NOT NULL
                    CHECK(claim_type IN (
                        'finding', 'method', 'limitation',
                        'comparison', 'definition', 'hypothesis'
                    )),
    confidence      REAL NOT NULL DEFAULT 1.0
                    CHECK(confidence BETWEEN 0.0 AND 1.0),
    verified        INTEGER NOT NULL DEFAULT 0,
    verified_at     TEXT,
    page_ref        INTEGER,
    raw_quote       TEXT,
    lifecycle_status TEXT NOT NULL DEFAULT 'active'
                    CHECK(lifecycle_status IN ('active', 'superseded', 'contested')),
    superseded_by_claim_id TEXT REFERENCES claim(claim_id),
    lifecycle_updated_at TEXT
);

CREATE TABLE IF NOT EXISTS concept (
    concept_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    aliases         TEXT,
    definition      TEXT,
    wiki_path       TEXT,
    domain          TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_link (
    link_id         TEXT PRIMARY KEY,
    from_type       TEXT NOT NULL,
    from_id         TEXT NOT NULL,
    to_type         TEXT NOT NULL,
    to_id           TEXT NOT NULL,
    relation        TEXT NOT NULL
                    CHECK(relation IN (
                        'supports', 'contradicts', 'refines',
                        'defines', 'uses', 'extends', 'cites'
                    )),
    confidence      REAL NOT NULL DEFAULT 1.0,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_concept (
    claim_id        TEXT NOT NULL REFERENCES claim(claim_id) ON DELETE CASCADE,
    concept_id      TEXT NOT NULL REFERENCES concept(concept_id) ON DELETE CASCADE,
    role            TEXT,
    PRIMARY KEY (claim_id, concept_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    event_id        TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    action          TEXT NOT NULL,
    details_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_doc      ON chunk(doc_id);
CREATE INDEX IF NOT EXISTS idx_claim_doc      ON claim(doc_id);
CREATE INDEX IF NOT EXISTS idx_claim_chunk    ON claim(chunk_id);
CREATE INDEX IF NOT EXISTS idx_claim_type     ON claim(claim_type);
CREATE INDEX IF NOT EXISTS idx_claim_lifecycle ON claim(lifecycle_status);
CREATE INDEX IF NOT EXISTS idx_evlink_from    ON evidence_link(from_type, from_id);
CREATE INDEX IF NOT EXISTS idx_evlink_to      ON evidence_link(to_type, to_id);
CREATE INDEX IF NOT EXISTS idx_concept_name   ON concept(name);
CREATE INDEX IF NOT EXISTS idx_audit_entity   ON audit_log(entity_type, entity_id);
"""


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    ddl: str,
) -> None:
    columns = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _ensure_schema_compat(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema for existing databases."""
    _ensure_column(
        conn,
        "claim",
        "lifecycle_status",
        "lifecycle_status TEXT NOT NULL DEFAULT 'active' CHECK(lifecycle_status IN ('active', 'superseded', 'contested'))",
    )
    _ensure_column(
        conn,
        "claim",
        "superseded_by_claim_id",
        "superseded_by_claim_id TEXT",
    )
    _ensure_column(
        conn,
        "claim",
        "lifecycle_updated_at",
        "lifecycle_updated_at TEXT",
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audit_log ("
        "event_id TEXT PRIMARY KEY, "
        "entity_type TEXT NOT NULL, "
        "entity_id TEXT NOT NULL, "
        "action TEXT NOT NULL, "
        "details_json TEXT NOT NULL, "
        "created_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_claim_lifecycle ON claim(lifecycle_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id)")


def initialize_db(conn: sqlite3.Connection) -> None:
    """Apply the full schema DDL to the given connection.

    Safe to call on an existing database (all statements use IF NOT EXISTS).
    """
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL)
    _ensure_schema_compat(conn)
    conn.commit()


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and Row factory set."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
