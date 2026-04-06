import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from distill.config import Config
from distill.db.schema import get_connection, initialize_db

console = Console()


def _get_config(ctx: click.Context) -> Config:
    return ctx.obj["config"]


def _get_conn(ctx: click.Context):
    return ctx.obj["conn"]


def _get_client(ctx: click.Context):
    """Lazily initialize LLMClient on first use (avoids import cost for non-LLM commands)."""
    if "client" not in ctx.obj:
        from distill.llm.client import LLMClient

        cfg = _get_config(ctx)
        if not cfg.anthropic_api_key:
            console.print("[red]ANTHROPIC_API_KEY is not set.[/red]")
            sys.exit(1)
        ctx.obj["client"] = LLMClient(cfg.anthropic_api_key, cfg.llm_model)
    return ctx.obj["client"]


def _load_search_indices(ctx: click.Context):
    """Load FAISS and BM25 indices from disk. Raises if indices do not exist."""
    import pickle

    import faiss
    from sentence_transformers import SentenceTransformer

    cfg = _get_config(ctx)
    faiss_path = cfg.index_path / "chunks.faiss"
    ids_path = cfg.index_path / "chunk_ids.pkl"
    bm25_path = cfg.index_path / "bm25.pkl"

    for path in [faiss_path, ids_path, bm25_path]:
        if not path.exists():
            console.print(f"[red]Search index not found: {path}[/red]")
            console.print("Run [bold]distill reindex[/bold] first.")
            sys.exit(1)

    faiss_index = faiss.read_index(str(faiss_path))
    with open(ids_path, "rb") as fh:
        chunk_ids = pickle.load(fh)
    with open(bm25_path, "rb") as fh:
        bm25_index = pickle.load(fh)

    encoder = SentenceTransformer(cfg.embedding_model)
    return faiss_index, bm25_index, chunk_ids, encoder


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--data-root",
    envvar="KNOWLEDGE_DATA_ROOT",
    default="data",
    show_default=True,
    help="Root directory for all data layers.",
)
@click.pass_context
def cli(ctx: click.Context, data_root: str) -> None:
    """Personal ML/AI knowledge base CLI."""
    ctx.ensure_object(dict)
    cfg = Config(data_root=Path(data_root))
    ctx.obj["config"] = cfg
    # Only open DB if it exists (init command creates it)
    if cfg.db_path.exists():
        conn = get_connection(str(cfg.db_path))
        ctx.obj["conn"] = conn
    else:
        ctx.obj["conn"] = None


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize the knowledge base directory structure and database."""
    cfg = _get_config(ctx)
    cfg.ensure_dirs()

    # Initialize SQLite database
    conn = get_connection(str(cfg.db_path))
    initialize_db(conn)
    ctx.obj["conn"] = conn
    conn.close()

    # Set up Obsidian vault config
    _setup_obsidian_vault(cfg.wiki_path)

    console.print(f"[green]Knowledge base initialized at {cfg.data_root}[/green]")
    console.print(f"Obsidian vault: [bold]{cfg.wiki_path}[/bold]")
    console.print("Open Obsidian and select that folder as your vault.")


def _setup_obsidian_vault(wiki_path: Path) -> None:
    """Create minimal Obsidian config and index note in the vault."""
    obsidian_dir = wiki_path / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)

    # Basic app.json config
    app_config = {
        "useMarkdownLinks": False,
        "newLinkFormat": "shortest",
        "defaultViewMode": "source",
    }
    (obsidian_dir / "app.json").write_text(json.dumps(app_config, indent=2))

    # Community plugins list (user must install manually)
    plugins_config = {"enabledPlugins": ["dataview"]}
    (obsidian_dir / "community-plugins.json").write_text(
        json.dumps(plugins_config, indent=2)
    )

    # Index note with Dataview query
    index_note = wiki_path / "00_INDEX.md"
    if not index_note.exists():
        index_note.write_text(
            "# Knowledge Base Index\n\n"
            "## Papers\n\n"
            "```dataview\n"
            "TABLE title, year, claim_count, verified_claims, compiled_at\n"
            "FROM \"papers\"\n"
            "WHERE note_type = \"paper\"\n"
            "SORT year DESC\n"
            "```\n\n"
            "## Concepts\n\n"
            "```dataview\n"
            "TABLE name, domain, claim_count\n"
            "FROM \"concepts\"\n"
            "WHERE note_type = \"concept\"\n"
            "SORT name ASC\n"
            "```\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", required=False, type=click.Path(exists=True))
@click.pass_context
def ingest(ctx: click.Context, path: str | None) -> None:
    """Detect and register new documents from the inbox (or a specific file)."""
    from distill.agents.ingestor import ingest_file, run_ingestor

    cfg = _get_config(ctx)
    conn = _get_conn(ctx)
    if conn is None:
        console.print("[red]Run 'distill init' first.[/red]")
        sys.exit(1)

    if path:
        doc_id = ingest_file(conn, Path(path), cfg)
        if doc_id:
            console.print(f"[green]Ingested:[/green] {doc_id}")
        else:
            console.print("[yellow]File already in knowledge base (duplicate hash).[/yellow]")
    else:
        doc_ids = run_ingestor(conn, cfg)
        console.print(f"[green]Ingested {len(doc_ids)} new document(s).[/green]")
        for doc_id in doc_ids:
            console.print(f"  {doc_id}")


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("doc_id", required=False)
@click.option("--all", "process_all", is_flag=True, help="Parse all ingested documents.")
@click.pass_context
def parse(ctx: click.Context, doc_id: str | None, process_all: bool) -> None:
    """Extract text and create chunks from raw documents."""
    from distill.agents.parser import parse_document, run_parser

    cfg = _get_config(ctx)
    conn = _get_conn(ctx)
    if conn is None:
        console.print("[red]Run 'distill init' first.[/red]")
        sys.exit(1)

    if process_all:
        results = run_parser(conn, cfg)
        total_chunks = sum(len(v) for v in results.values())
        console.print(f"[green]Parsed {len(results)} document(s), {total_chunks} chunks.[/green]")
    elif doc_id:
        chunk_ids = parse_document(conn, doc_id, cfg)
        console.print(f"[green]Parsed {doc_id}: {len(chunk_ids)} chunks.[/green]")
    else:
        console.print("[red]Provide a DOC_ID or --all.[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("doc_id", required=False)
@click.option("--all", "process_all", is_flag=True)
@click.pass_context
def extract(ctx: click.Context, doc_id: str | None, process_all: bool) -> None:
    """Extract claims and concepts via Claude API."""
    from distill.agents.extractor import run_extractor
    from distill.db.repository import list_documents

    cfg = _get_config(ctx)
    conn = _get_conn(ctx)
    client = _get_client(ctx)

    if process_all:
        docs = list_documents(conn, status="parsed")
        for doc in docs:
            console.print(f"Extracting [bold]{doc.title}[/bold]...")
            run_extractor(client, conn, doc.doc_id, cfg)
        console.print(f"[green]Extraction complete for {len(docs)} document(s).[/green]")
    elif doc_id:
        run_extractor(client, conn, doc_id, cfg)
        console.print(f"[green]Extraction complete for {doc_id}.[/green]")
    else:
        console.print("[red]Provide a DOC_ID or --all.[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("doc_id", required=False)
@click.option("--all", "process_all", is_flag=True)
@click.pass_context
def compile(ctx: click.Context, doc_id: str | None, process_all: bool) -> None:
    """Generate Obsidian wiki notes for compiled documents."""
    from distill.agents.compiler import run_compiler
    from distill.db.repository import list_documents

    cfg = _get_config(ctx)
    conn = _get_conn(ctx)
    client = _get_client(ctx)

    if process_all:
        docs = list_documents(conn, status="extracted")
        for doc in docs:
            console.print(f"Compiling [bold]{doc.title}[/bold]...")
            run_compiler(client, conn, doc.doc_id, cfg)
        console.print(f"[green]Compiled {len(docs)} document(s).[/green]")
    elif doc_id:
        run_compiler(client, conn, doc_id, cfg)
        console.print(f"[green]Compiled {doc_id}.[/green]")
    else:
        console.print("[red]Provide a DOC_ID or --all.[/red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("doc_id", required=False)
@click.option("--all", "process_all", is_flag=True)
@click.option("--report", is_flag=True, help="Print full verification report.")
@click.pass_context
def verify(ctx: click.Context, doc_id: str | None, process_all: bool, report: bool) -> None:
    """Verify claim traceability and detect contradictions."""
    from distill.agents.verifier import generate_verification_report, run_verifier
    from distill.db.repository import list_documents

    cfg = _get_config(ctx)
    conn = _get_conn(ctx)

    if process_all:
        docs = list_documents(conn, status="compiled")
        for doc in docs:
            result = run_verifier(conn, doc.doc_id, cfg)
            console.print(
                f"{doc.title}: {result['verified']}/{result['total_claims']} verified"
            )
    elif doc_id:
        result = run_verifier(conn, doc_id, cfg)
        console.print(
            f"[green]{result['verified']}/{result['total_claims']} claims verified.[/green]"
        )
        if result["failed"] > 0:
            console.print(f"[yellow]{result['failed']} claim(s) failed traceability check.[/yellow]")
    else:
        console.print("[red]Provide a DOC_ID or --all.[/red]")
        sys.exit(1)

    if report:
        full = generate_verification_report(conn, doc_id)
        console.print_json(json.dumps(full, indent=2))


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query_text")
@click.option("--top-k", default=10, show_default=True)
@click.option(
    "--format", "fmt",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    show_default=True,
)
@click.pass_context
def query(ctx: click.Context, query_text: str, top_k: int, fmt: str) -> None:
    """Search the knowledge base and answer a question with evidence."""
    from distill.agents.query_agent import run_query

    conn = _get_conn(ctx)
    client = _get_client(ctx)
    faiss_index, bm25_index, chunk_ids, encoder = _load_search_indices(ctx)

    result = run_query(
        client, conn, query_text, encoder, faiss_index, bm25_index, chunk_ids, top_k=top_k
    )

    if fmt == "json":
        click.echo(json.dumps(result, indent=2))
    elif fmt == "markdown":
        _print_query_markdown(result)
    else:
        _print_query_text(result)


def _print_query_text(result: dict) -> None:
    console.print("\n[bold]Answer:[/bold]")
    console.print(result["answer"])
    console.print(f"\n[dim]Confidence: {result['confidence']:.0%}[/dim]")
    if result.get("uncertainty"):
        console.print(f"[yellow]Uncertainty: {result['uncertainty']}[/yellow]")
    console.print(f"\n[bold]Sources ({len(result['sources'])}):[/bold]")
    for src in result["sources"]:
        console.print(f"  - {src.get('title', src.get('doc_id', '?'))}: {src.get('quote', '')[:80]}")


def _print_query_markdown(result: dict) -> None:
    lines = [
        f"## Answer\n\n{result['answer']}\n",
        f"**Confidence:** {result['confidence']:.0%}\n",
    ]
    if result.get("uncertainty"):
        lines.append(f"**Uncertainty:** {result['uncertainty']}\n")
    lines.append("## Sources\n")
    for src in result["sources"]:
        title = src.get("title", src.get("doc_id", "?"))
        quote = src.get("quote", "")[:100]
        lines.append(f'- **{title}**: *"{quote}"*')
    click.echo("\n".join(lines))


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("topic")
@click.option("--doc-ids", multiple=True, help="Document IDs to include.")
@click.option(
    "--type", "output_type",
    type=click.Choice(["brief", "table", "concept-map"]),
    default="brief",
    show_default=True,
)
@click.option(
    "--dimensions",
    default="finding,method,limitation",
    help="Comma-separated claim types for table output.",
)
@click.pass_context
def output(
    ctx: click.Context,
    topic: str,
    doc_ids: tuple[str, ...],
    output_type: str,
    dimensions: str,
) -> None:
    """Generate a brief, comparison table, or concept map."""
    from distill.agents.output_agent import (
        generate_brief,
        generate_comparison_table,
        generate_concept_map,
    )
    from distill.db.repository import list_documents

    cfg = _get_config(ctx)
    conn = _get_conn(ctx)
    client = _get_client(ctx)

    # Resolve doc_ids: use all compiled docs if none specified
    resolved_ids = list(doc_ids)
    if not resolved_ids:
        docs = list_documents(conn, status="compiled")
        resolved_ids = [d.doc_id for d in docs]

    if not resolved_ids:
        console.print("[yellow]No compiled documents found. Run 'distill compile --all' first.[/yellow]")
        sys.exit(1)

    if output_type == "brief":
        dest = generate_brief(client, conn, topic, resolved_ids, cfg.outputs_path)
    elif output_type == "table":
        dim_list = [d.strip() for d in dimensions.split(",")]
        dest = generate_comparison_table(conn, resolved_ids, dim_list, cfg.outputs_path)
    elif output_type == "concept-map":
        # topic is treated as comma-separated concept names for concept-map
        concept_names = [c.strip() for c in topic.split(",")]
        dest = generate_concept_map(conn, concept_names, cfg.outputs_path)

    console.print(f"[green]Output written to:[/green] {dest}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show the status of all documents in the knowledge base."""
    from distill.db.repository import list_documents

    conn = _get_conn(ctx)
    if conn is None:
        console.print("[red]Run 'distill init' first.[/red]")
        sys.exit(1)

    docs = list_documents(conn)
    if not docs:
        console.print("[yellow]No documents in the knowledge base yet.[/yellow]")
        return

    table = Table(title="Knowledge Base Status")
    table.add_column("Doc ID", style="dim", width=12)
    table.add_column("Title", max_width=40)
    table.add_column("Year", width=6)
    table.add_column("Type", width=8)
    table.add_column("Status", width=12)

    status_colors = {
        "ingested": "cyan",
        "parsed": "blue",
        "extracted": "yellow",
        "compiled": "green",
        "verified": "bold green",
    }
    for doc in docs:
        color = status_colors.get(doc.status, "white")
        table.add_row(
            doc.doc_id[:8],
            doc.title[:40],
            str(doc.year or ""),
            doc.source_type,
            f"[{color}]{doc.status}[/{color}]",
        )

    console.print(table)

    if ctx.obj.get("client"):
        stats = ctx.obj["client"].get_usage_stats()
        console.print(
            f"\nAPI usage this session: "
            f"{stats['input_tokens']} input / {stats['output_tokens']} output tokens"
        )


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def reindex(ctx: click.Context) -> None:
    """Rebuild FAISS and BM25 search indices from all chunks in the database."""
    import pickle

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from distill.db.repository import get_all_chunks
    from distill.search.embeddings import (
        build_faiss_index,
        encode_texts,
        load_encoder,
        save_chunk_ids,
        save_index,
    )
    from distill.search.lexical import build_bm25_index, save_bm25

    cfg = _get_config(ctx)
    conn = _get_conn(ctx)

    chunks = get_all_chunks(conn)
    if not chunks:
        console.print("[yellow]No chunks found. Run 'distill parse --all' first.[/yellow]")
        return

    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    console.print(f"Indexing {len(texts)} chunks...")

    # Semantic index
    encoder = load_encoder(cfg.embedding_model)
    embeddings = encode_texts(encoder, texts)
    faiss_index = build_faiss_index(embeddings)
    save_index(faiss_index, cfg.index_path / "chunks.faiss")
    save_chunk_ids(chunk_ids, cfg.index_path / "chunk_ids.pkl")

    # Lexical index
    bm25 = build_bm25_index(texts)
    save_bm25(bm25, cfg.index_path / "bm25.pkl")

    # Update chunk embedding IDs
    from distill.db.repository import update_chunk_embedding_id
    for idx, chunk in enumerate(chunks):
        update_chunk_embedding_id(conn, chunk.chunk_id, str(idx))

    console.print(f"[green]Reindex complete. {len(chunks)} chunks indexed.[/green]")
