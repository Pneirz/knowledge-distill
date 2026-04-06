# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Initialize knowledge base (creates directory structure + SQLite DB)
distill init

# Run all tests
poetry run pytest tests/ -v

# Run a specific test file
poetry run pytest tests/agents/test_parser.py -v

# Lint
poetry run ruff check .

# Format
poetry run ruff format .
```

## Pipeline commands

All pipeline steps are idempotent — re-running with `--all` skips already-processed documents. Each command accepts either `--all` or a single `DOC_ID`.

```bash
distill ingest                    # register new docs from data/00_inbox/
distill parse --all               # extract text, detect sections, create chunks
distill extract --all             # call Claude to extract claims and concepts per chunk
distill compile --all             # generate Obsidian notes (papers/ and concepts/)
distill verify --all              # verify each claim's raw_quote against its source chunk
distill reindex                   # build FAISS and BM25 indices from all chunks

distill query "your question"     # hybrid search + evidence-grounded answer
distill output "topic" --type brief|table|concept-map
distill status                    # table of all documents with pipeline stage
```

Optional filename convention for automatic metadata extraction when ingesting:
```
Author et al - YEAR - Title.pdf
```

## Architecture

**distill** is a CLI-based knowledge base that extracts claims from research documents (PDF, HTML, Markdown) and builds a traceable, searchable store. Every claim must include a `raw_quote` (verbatim excerpt from source) verified via fuzzy matching (85% threshold).

### Processing pipeline

Documents flow through 5 sequential stages, each mapped to a directory under `data/`:

```
00_inbox/ → ingest → 01_raw/ → parse → 02_parsed/
→ extract → 03_extracted/ → compile → 04_compiled_wiki/
→ verify → (status updated in DB)
```

Each stage has a corresponding agent in `distill/agents/` and a CLI command in `distill/cli/main.py`. Document status is tracked in SQLite: `ingested → parsed → extracted → compiled → verified`.

### Key modules

- **`distill/agents/`** — One agent per pipeline stage. `extractor.py` calls the configured LLM backend; `verifier.py` runs fuzzy matching; `linker.py` detects cross-document claim relations (supports/contradicts/refines) using FAISS similarity + the configured backend for classification; `query_agent.py` and `output_agent.py` handle search/QA/generation.
- **`distill/db/`** — Raw SQL (no ORM). `schema.py` has DDL; `repository.py` has CRUD; `models.py` has dataclasses. Uses in-memory SQLite (`:memory:`) in tests.
- **`distill/llm/`** — Anthropic client with exponential backoff (`client.py`) and structured system prompts (`prompts.py`).
- **`distill/search/`** — Hybrid search: FAISS (semantic) + BM25 (lexical), fused with Reciprocal Rank Fusion in `hybrid.py`.
- **`distill/config.py`** — Single `Config` class reading from environment variables. All path resolution lives here.

### Data model

```
document (1) → (*) chunk (1) → (*) claim
concept (*) ↔ (*) claim  [via claim_concept junction]
evidence_link  [generic from_id/to_id relation: supports|contradicts|refines|...]
```

`Claim.raw_quote` is the epistemological anchor: every claim must point back to a verbatim chunk excerpt, and `verifier.py` enforces this. Claims failing the check are marked `verified=-1`. The linker marks the older claim (by document year) as contradicted when a `contradicts` relation is found.

## Environment variables

See `.env.example`. Use `LLM_BACKEND=claude-code`, `codex`, or `anthropic`. `ANTHROPIC_API_KEY` is required only for `anthropic`. Key defaults: `LLM_MODEL=claude-sonnet-4-6`, `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, `CHUNK_MAX_TOKENS=512`, `CHUNK_OVERLAP=64`, `KNOWLEDGE_DATA_ROOT=data`.

## Testing conventions

- Tests live in `tests/<module_name>/test_<something>.py`
- All test files start with `# %%`
- LLM calls are mocked — no real API calls in tests
- DB fixtures use in-memory SQLite via `conftest.py`
