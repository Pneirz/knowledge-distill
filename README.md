# distill

Personal knowledge base for ML/AI research with epistemological rigor.

Every claim in this system is traceable to a verbatim excerpt of its source document.
No assertion can exist without a chain back to the original text that supports it.

Inspired by [Andrej Karpathy's approach to LLM knowledge bases](https://x.com/karpathy).

---

## What it does

Takes PDFs, web articles, and GitHub READMEs as input and produces:

- A SQLite database of documents, chunks, claims, concepts, and their relations
- An Obsidian vault with one note per paper and one per concept, linked via `[[wiki-links]]`
- Hybrid semantic + lexical search (FAISS + BM25) with Reciprocal Rank Fusion
- Lifecycle-aware claims with `active`, `superseded`, and `contested` states
- Contradiction detection between claims from different papers
- Audit trail for verification, lifecycle review, contradiction links, and compilation
- Evidence-grounded answers to free-text questions

## Stack

| Component | Choice |
|---|---|
| LLM | Claude Code CLI, Codex CLI, or Anthropic API |
| Database | SQLite (no ORM) |
| Semantic search | FAISS + `sentence-transformers` (BAAI/bge-small-en-v1.5) |
| Lexical search | BM25 (`rank-bm25`) |
| Search fusion | Reciprocal Rank Fusion (RRF) |
| Wiki frontend | Obsidian (Markdown + YAML frontmatter) |
| PDF parsing | PyMuPDF |
| HTML parsing | BeautifulSoup4 |
| CLI | Click + Rich |

---

## Setup

Requires Python 3.12 and [Poetry](https://python-poetry.org/).

```bash
git clone <repo>
cd distill

poetry install

cp .env.example .env
# Choose LLM_BACKEND=claude-code, codex, or anthropic
# Only set ANTHROPIC_API_KEY if using anthropic

distill init
```

`distill init` creates the full directory structure, initializes the SQLite database,
and sets up an Obsidian vault at `data/04_compiled_wiki/`.

Backend modes:

- `LLM_BACKEND=claude-code`: uses the local `claude` CLI session, no API key required
- `LLM_BACKEND=codex`: uses the local `codex` CLI session, no API key required
- `LLM_BACKEND=anthropic`: uses the Anthropic API and requires `ANTHROPIC_API_KEY`

Obsidian setup: open Obsidian, choose "Open folder as vault", and select
`data/04_compiled_wiki/`. Install the Dataview community plugin to enable the
queries in `00_INDEX.md`.

---

## Usage

### Ingest a document

Place a PDF or saved HTML article in `data/00_inbox/`.

Rename it using the optional convention for automatic metadata extraction:

```text
Author et al - YEAR - Title.pdf
```

Then run:

```bash
distill ingest
```

### Run the pipeline

```bash
distill parse --all
distill extract --all
distill compile --all
distill verify --all
distill review-lifecycle --all
distill reindex
```

Each step is idempotent. Re-running with `--all` skips already-processed documents.
You can process a single document by passing its `doc_id` instead of `--all`.

### Query the knowledge base

```bash
distill query "what attention mechanism does the Transformer use?"
distill query "what are the limitations?" --top-k 5 --format markdown
distill query "how did attention methods evolve?" --include-superseded
```

Every answer includes cited sources, a confidence level, and an explicit uncertainty
statement. By default, query prefers `active` and `verified` claims as primary
evidence. Use `--include-superseded` to bring historical claims back as secondary
evidence.

### Review lifecycle

```bash
distill review-lifecycle "self-attention"
distill review-lifecycle --all
```

Lifecycle review updates temporal validity without changing traceability:

- `active`: default evidence for query and output
- `superseded`: preserved for history, excluded from primary evidence by default
- `contested`: contradicted by another active claim and surfaced as uncertainty

### Generate outputs

```bash
distill output "attention mechanisms" --type brief
distill output "comparison" --type table --dimensions finding,method,limitation
distill output "transformer,self-attention,positional-encoding" --type concept-map
```

### Inspect status

```bash
distill status
```

---

## Directory structure

```text
data/
|-- 00_inbox/
|-- 01_raw/
|-- 02_parsed/
|-- 03_extracted/
|-- 04_compiled_wiki/
|   |-- .obsidian/
|   |-- 00_INDEX.md
|   |-- papers/
|   |-- concepts/
|   |-- methods/
|   `-- contradictions/
|-- 05_search_index/
|-- 06_outputs/
`-- 07_registry/
    `-- distill.db
```

---

## Epistemological guarantees

**Traceability:** every `Claim` record has a `raw_quote` field, a verbatim excerpt
from its source chunk. The `verify` command uses fuzzy matching (threshold 85) to
confirm the quote exists in the chunk. Claims that fail this check are marked
`verified=-1`.

The audit chain is:

```text
Claim.raw_quote -> Chunk.text -> Chunk.page_ref -> Document.raw_path
```

**Contradiction detection:** when two claims from different papers have embedding
similarity above 0.85, Claude classifies their relation as one of
`supports | contradicts | refines | extends | unrelated`.
Contradictions are stored as `EvidenceLink(relation='contradicts')` and can mark
claims as `contested`.

**Temporal validity:** lifecycle review is separate from traceability. A claim can be
fully verified and still become `superseded` when newer evidence replaces it, or
`contested` when another active claim contradicts it.

**Auditability:** important state changes are recorded in `audit_log`, including
claim verification, lifecycle transitions, contradiction links, and document compilation.

---

## Data model

```text
Document   -> has many Chunks
Chunk      -> has many Claims
Claim      -> linked to many Concepts (via claim_concept)
Claim      -> has traceability (verified) and lifecycle (active|superseded|contested)
EvidenceLink -> typed directed edge between any two objects
                (supports, contradicts, refines, defines, uses, extends, cites)
```

Document pipeline states: `ingested -> parsed -> extracted -> compiled -> verified`

---

## Sources accepted

| Type | Format | Notes |
|---|---|---|
| Academic papers | PDF | arXiv, journals |
| Web articles | HTML | Saved with browser |
| Repositories | Markdown | README and docs |

Sources should be peer-reviewed or from verifiable authors.
Non-peer-reviewed articles are accepted but treated as lower-confidence sources.

---

## Development

```bash
python -m pytest tests -v
python -m pytest tests/cli/test_pipeline_smoke.py -q
python -m ruff check .
```

Tests use an in-memory SQLite database and mock the LLM client.
No real API calls are made during testing.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `claude-code` | `claude-code`, `codex`, or `anthropic` |
| `ANTHROPIC_API_KEY` | - | Required only when `LLM_BACKEND=anthropic` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model name for the selected backend |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence transformer for FAISS |
| `KNOWLEDGE_DATA_ROOT` | `data` | Root directory for all data layers |
| `CHUNK_MAX_TOKENS` | `512` | Maximum tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between consecutive chunks |
