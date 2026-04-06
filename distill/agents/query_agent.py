import sqlite3

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from distill.db.repository import get_chunk, get_document
from distill.llm.client import LLMClient
from distill.llm.prompts import query_system_prompt
from distill.search.hybrid import hybrid_search


def build_context(
    conn: sqlite3.Connection,
    search_results: list[dict],
    max_tokens: int = 6000,
) -> tuple[str, list[dict]]:
    """Build a context string from ranked search results for the LLM.

    Returns (context_string, source_metadata_list).
    Respects max_tokens limit by stopping early when the limit is reached.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    context_parts: list[str] = []
    sources: list[dict] = []
    token_count = 0

    for result in search_results:
        chunk = get_chunk(conn, result["chunk_id"])
        if chunk is None:
            continue
        doc = get_document(conn, chunk.doc_id)
        if doc is None:
            continue

        # Format as a labeled passage
        label = f"[{doc.doc_id[:8]}, {chunk.chunk_id[:8]}]"
        title_info = f"{doc.title} ({doc.year})" if doc.year else doc.title
        passage = f"{label} {title_info} — {chunk.section or 'General'}\n{chunk.text}"

        passage_tokens = len(enc.encode(passage))
        if token_count + passage_tokens > max_tokens:
            break

        context_parts.append(passage)
        token_count += passage_tokens
        sources.append({
            "doc_id": doc.doc_id,
            "chunk_id": chunk.chunk_id,
            "title": doc.title,
            "year": doc.year,
            "section": chunk.section,
            "rank": result["rank"],
        })

    return "\n\n---\n\n".join(context_parts), sources


def run_query(
    client: LLMClient,
    conn: sqlite3.Connection,
    query: str,
    encoder: SentenceTransformer,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    top_k: int = 10,
) -> dict:
    """Answer a query with evidence from the knowledge base.

    1. Perform hybrid search (FAISS + BM25 with RRF fusion).
    2. Build context from top-k chunks.
    3. Ask Claude to answer using only the context.
    4. Return structured response with sources and confidence.
    """
    # Retrieve relevant chunks
    search_results = hybrid_search(
        query, encoder, faiss_index, bm25_index, chunk_ids, top_k=top_k
    )

    if not search_results:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "sources": [],
            "confidence": 0.0,
            "uncertainty": "Knowledge base may be empty or the query has no matches.",
        }

    context, source_metadata = build_context(conn, search_results)

    system = query_system_prompt()
    user = f"Question: {query}\n\nContext:\n{context}"

    result = client.complete_json(system, user, max_tokens=2048)

    # Enrich sources with metadata from DB
    answer_sources = result.get("sources", [])
    for src in answer_sources:
        for meta in source_metadata:
            if src.get("chunk_id", "").startswith(meta["chunk_id"][:8]):
                src.update(meta)
                break

    return {
        "answer": result.get("answer", ""),
        "sources": answer_sources,
        "confidence": result.get("confidence", 0.5),
        "uncertainty": result.get("uncertainty", ""),
        "chunks_retrieved": len(search_results),
    }
