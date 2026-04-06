# %%
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from distill.config import Config
from distill.db.schema import initialize_db


@pytest.fixture
def db() -> sqlite3.Connection:
    """In-memory SQLite database with full schema applied.

    Using :memory: means no files are created and no teardown is needed.
    Pass this connection directly to repository functions in tests.
    """
    conn = sqlite3.connect(":memory:")
    initialize_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """Temporary directory tree matching the 8-layer data structure."""
    cfg = Config(data_root=tmp_path)
    cfg.ensure_dirs()
    return tmp_path


@pytest.fixture
def mock_llm():
    """LLMClient mock that returns empty extraction results by default.

    Override mock_llm.complete_json.return_value in individual tests to
    simulate specific LLM responses without making real API calls.
    """
    from distill.llm.client import LLMClient

    client = MagicMock(spec=LLMClient)
    client.complete.return_value = "Mock LLM response."
    client.complete_json.return_value = {"claims": [], "concepts": []}
    client.get_usage_stats.return_value = {
        "input_tokens": 0,
        "output_tokens": 0,
    }
    return client


@pytest.fixture
def sample_document():
    """Minimal Document dataclass instance for use in repository tests."""
    from distill.db.models import Document

    return Document(
        doc_id="test-doc-001",
        title="Attention Is All You Need",
        source_type="pdf",
        content_hash="abc123def456",
        status="ingested",
        ingested_at=datetime(2026, 4, 6, 12, 0, 0),
        updated_at=datetime(2026, 4, 6, 12, 0, 0),
        authors=["Vaswani, A.", "Shazeer, N."],
        year=2017,
        url="https://arxiv.org/abs/1706.03762",
    )
