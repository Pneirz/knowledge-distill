# %%
import numpy as np
from click.testing import CliRunner

from distill.cli.main import cli


class _StubClient:
    def __init__(self) -> None:
        self._json_calls = 0
        self.query_chunk_id = "chunk"
        self.query_doc_id = "doc"

    def complete(self, system: str, user: str, max_tokens: int = 0, temperature: float = 0.0):
        return "Short grounded summary."

    def complete_json(self, system: str, user: str, max_tokens: int = 0):
        self._json_calls += 1
        if "Extract all claims and concepts" in user:
            return {
                "claims": [
                    {
                        "claim_text": "Self-attention models token relationships.",
                        "claim_type": "method",
                        "confidence": 0.95,
                        "raw_quote": "self-attention models token relationships",
                        "concepts": ["self-attention"],
                    }
                ],
                "concepts": [
                    {
                        "name": "self-attention",
                        "definition": "Mechanism relating tokens to one another.",
                        "aliases": ["attention"],
                        "domain": "architecture",
                    }
                ],
            }
        if "identify obsolete scientific claims" in system.lower():
            return {"obsolete_claim_indices": [], "reasons": {}}
        return {
            "answer": "Self-attention models token relationships. [doc, chunk]",
            "sources": [
                {
                    "doc_id": self.query_doc_id[:8],
                    "chunk_id": self.query_chunk_id[:8],
                    "quote": "self-attention models token relationships",
                    "relevance": 0.95,
                }
            ],
            "confidence": 0.88,
            "uncertainty": "",
        }

    def get_usage_stats(self):
        return {"input_tokens": 0, "output_tokens": 0}


def test_cli_pipeline_smoke(monkeypatch, tmp_data_root):
    """Simple smoke test for init -> ingest -> parse -> extract -> compile -> verify -> query."""
    runner = CliRunner()
    source_file = tmp_data_root / "mini-paper.md"
    source_file.write_text(
        "# Tiny Paper\n\n"
        "## Method\n\n"
        "This paper explains how self-attention models token relationships.\n",
        encoding="utf-8",
    )

    stub_client = _StubClient()
    monkeypatch.setattr("distill.cli.main._get_client", lambda ctx: stub_client)
    monkeypatch.setattr("distill.search.embeddings.load_encoder", lambda model_name: object())
    monkeypatch.setattr(
        "distill.search.embeddings.encode_texts",
        lambda encoder, texts, batch_size=64, normalize=True: np.array(
            [[float(i + 1), float((i + 1) * 2), 1.0] for i, _ in enumerate(texts)],
            dtype=np.float32,
        ),
    )

    query_chunk_id: dict[str, str] = {}

    def fake_load_indices(ctx):
        conn = ctx.obj["conn"]
        row = conn.execute(
            """
            SELECT c.chunk_id, c.doc_id
            FROM chunk c
            ORDER BY c.chunk_index
            LIMIT 1
            """
        ).fetchone()
        query_chunk_id["value"] = row["chunk_id"]
        stub_client.query_chunk_id = row["chunk_id"]
        stub_client.query_doc_id = row["doc_id"]
        return object(), object(), [row["chunk_id"]], object()

    def fake_hybrid_search(query, encoder, faiss_index, bm25_index, chunk_ids, top_k=10):
        return [{"chunk_id": chunk_ids[0], "score": 0.9, "rank": 1}]

    monkeypatch.setattr("distill.cli.main._load_search_indices", fake_load_indices)
    monkeypatch.setattr("distill.agents.query_agent.hybrid_search", fake_hybrid_search)

    result = runner.invoke(cli, ["--data-root", str(tmp_data_root), "init"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        cli,
        ["--data-root", str(tmp_data_root), "ingest", str(source_file)],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli, ["--data-root", str(tmp_data_root), "parse", "--all"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli, ["--data-root", str(tmp_data_root), "extract", "--all"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(cli, ["--data-root", str(tmp_data_root), "compile", "--all"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        cli,
        ["--data-root", str(tmp_data_root), "verify", "--all", "--report"],
    )
    assert result.exit_code == 0, result.output
    assert "traceability" in result.output

    result = runner.invoke(cli, ["--data-root", str(tmp_data_root), "reindex"])
    assert result.exit_code == 0, result.output
    assert (tmp_data_root / "05_search_index" / "chunks.faiss").exists()
    assert (tmp_data_root / "05_search_index" / "bm25.pkl").exists()

    result = runner.invoke(
        cli,
        ["--data-root", str(tmp_data_root), "review-lifecycle", "self-attention"],
    )
    assert result.exit_code == 0, result.output
    assert "self-attention" in result.output

    result = runner.invoke(
        cli,
        [
            "--data-root",
            str(tmp_data_root),
            "query",
            "How does self-attention work?",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"lifecycle_status": "active"' in result.output
    assert query_chunk_id["value"][:8] in result.output
