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
- Contradiction detection between claims from different papers
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

**Obsidian setup:** open Obsidian → "Open folder as vault" → select `data/04_compiled_wiki/`.
Install the **Dataview** community plugin to enable the index queries in `00_INDEX.md`.

---

## Usage

### Ingest a document

Place a PDF or saved HTML article in `data/00_inbox/`.

Rename it using the optional convention for automatic metadata extraction:

```
Author et al - YEAR - Title.pdf
```

Then run:

```bash
distill ingest
```

### Run the pipeline

```bash
distill parse --all      # extract text, detect sections, create chunks
distill extract --all    # call the configured LLM backend to extract claims and concepts per chunk
distill compile --all    # generate Obsidian notes (papers/ and concepts/)
distill verify --all     # verify each claim's raw_quote against its source chunk
distill reindex          # build FAISS and BM25 indices
```

Each step is idempotent. Re-running with `--all` skips already-processed documents.
You can process a single document by passing its `doc_id` instead of `--all`.

### Query the knowledge base

```bash
distill query "what attention mechanism does the Transformer use?"
distill query "what are the limitations?" --top-k 5 --format markdown
```

Every answer includes cited sources, a confidence level, and an explicit uncertainty statement.

### Generate outputs

```bash
# Executive brief on a topic
distill output "attention mechanisms" --type brief

# Comparison table across papers (by claim type)
distill output "comparison" --type table --dimensions finding,method,limitation

# Concept map in Mermaid format
distill output "transformer,self-attention,positional-encoding" --type concept-map
```

### Inspect status

```bash
distill status    # table of all documents with pipeline stage
```

---

## Directory structure

```
data/
├── 00_inbox/           # drop new documents here
├── 01_raw/             # immutable copy of originals
├── 02_parsed/          # JSON: sections and chunks per document
├── 03_extracted/       # JSON: claims and concepts per document
├── 04_compiled_wiki/   # Obsidian vault
│   ├── .obsidian/
│   ├── 00_INDEX.md     # Dataview index of all papers and concepts
│   ├── papers/
│   ├── concepts/
│   ├── methods/
│   └── contradictions/
├── 05_search_index/    # serialized FAISS and BM25 indices
├── 06_outputs/         # generated briefs, tables, concept maps
└── 07_registry/
    └── distill.db      # SQLite: documents, chunks, claims, concepts, links
```

---

## Epistemological guarantees

**Traceability:** every `Claim` record has a `raw_quote` field — a verbatim excerpt from
its source chunk. The `verify` command uses fuzzy matching (threshold 85) to confirm
the quote exists in the chunk. Claims that fail this check are marked `verified=-1`.

The audit chain is:

```
Claim.raw_quote → Chunk.text → Chunk.page_ref → Document.raw_path
```

**Contradiction detection:** when two claims from different papers have embedding
similarity above 0.85, Claude classifies their relation as one of
`supports | contradicts | refines | extends | unrelated`.
Contradictions are stored as `EvidenceLink(relation='contradicts')` and
generate a note in `contradictions/`.

**Obsolescence:** the `verify` command can flag claims about a concept that are
superseded by newer claims from more recent papers.

---

## Data model

```
Document   → has many Chunks
Chunk      → has many Claims
Claim      → linked to many Concepts (via claim_concept)
EvidenceLink → typed directed edge between any two objects
               (supports, contradicts, refines, defines, uses, extends, cites)
```

Document pipeline states: `ingested → parsed → extracted → compiled → verified`

---

## Sources accepted

| Type | Format | Notes |
|---|---|---|
| Academic papers | PDF | arXiv, journals |
| Web articles | HTML | Saved with browser (Medium, Towards Data Science, LinkedIn) |
| Repositories | Markdown | README and docs |

Sources should be peer-reviewed or from verifiable authors.
Non-peer-reviewed articles (Medium, LinkedIn) are accepted but treated as lower-confidence sources.

---

## Development

```bash
poetry run pytest tests/ -v    # run all tests (58 tests, no API calls)
poetry run ruff check .        # lint
```

Tests use an in-memory SQLite database and mock the LLM client.
No real API calls are made during testing.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `claude-code` | `claude-code`, `codex`, or `anthropic` |
| `ANTHROPIC_API_KEY` | — | Required only when `LLM_BACKEND=anthropic` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model name for the selected backend |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence transformer for FAISS |
| `KNOWLEDGE_DATA_ROOT` | `data` | Root directory for all data layers |
| `CHUNK_MAX_TOKENS` | `512` | Maximum tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between consecutive chunks |
