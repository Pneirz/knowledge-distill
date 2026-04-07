from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any

import faiss
from rank_bm25 import BM25Okapi

from distill.db.repository import (
    get_chunk,
    get_claims_by_chunk,
    get_concept_ids_for_claim,
    get_document,
)
from distill.llm.client import BaseLLMClient
from distill.llm.prompts import query_system_prompt
from distill.search.hybrid import hybrid_search


def build_context(
    conn: sqlite3.Connection,
    search_results: list[dict],
    max_tokens: int = 6000,
    include_superseded: bool = False,
) -> tuple[str, list[dict]]:
    """Build a progressive-disclosure context from ranked search results."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    candidates = build_evidence_candidates(
        conn, search_results, include_superseded=include_superseded
    )

    if not candidates:
        return "", []

    primary = [c for c in candidates if c["is_primary_evidence"]]
    secondary = [c for c in candidates if not c["is_primary_evidence"]]
    ordered = primary + secondary

    short_lines = ["Evidence summary:"]
    token_count = len(enc.encode("\n".join(short_lines)))
    selected: list[dict] = []

    for candidate in ordered:
        summary_line = (
            f"- [{candidate['doc_id'][:8]}, {candidate['chunk_id'][:8]}] "
            f"{candidate['title']} | {candidate['section']} | "
            f"claim: {candidate['claim_text']} | verified={candidate['verified']} | "
            f"lifecycle={candidate['lifecycle_status']}"
        )
        line_tokens = len(enc.encode(summary_line))
        if token_count + line_tokens > max_tokens:
            break
        short_lines.append(summary_line)
        token_count += line_tokens
        selected.append(candidate)

    context_parts = ["\n".join(short_lines), "", "Expanded passages:"]
    token_count = len(enc.encode("\n\n".join(context_parts)))
    expanded_sources: list[dict] = []

    for candidate in selected:
        label = f"[{candidate['doc_id'][:8]}, {candidate['chunk_id'][:8]}]"
        quote_prefix = f"Representative quote: {candidate['raw_quote']}\n" if candidate["raw_quote"] else ""
        note_prefix = f"Note: {candidate['note']}\n" if candidate["note"] else ""
        passage = (
            f"{label} {candidate['title']} ({candidate['year']}) - {candidate['section']}\n"
            f"Claim: {candidate['claim_text']}\n"
            f"Verified: {candidate['verified']} | Lifecycle: {candidate['lifecycle_status']}\n"
            f"{note_prefix}{quote_prefix}{candidate['chunk_text']}"
        )
        passage_tokens = len(enc.encode(passage))
        if token_count + passage_tokens > max_tokens:
            continue

        context_parts.append(passage)
        token_count += passage_tokens
        expanded_sources.append(candidate)

    return "\n\n---\n\n".join(context_parts), expanded_sources


def build_evidence_candidates(
    conn: sqlite3.Connection,
    search_results: list[dict],
    include_superseded: bool = False,
) -> list[dict]:
    """Rank chunk-level evidence candidates using lifecycle-aware signals."""
    candidates: list[dict] = []
    concept_counter: Counter[str] = Counter()
    chunk_claims: dict[str, list] = {}

    for result in search_results:
        claims = get_claims_by_chunk(conn, result["chunk_id"])
        filtered_claims = [
            claim for claim in claims
            if include_superseded or claim.lifecycle_status != "superseded"
        ]
        chunk_claims[result["chunk_id"]] = filtered_claims
        for claim in filtered_claims:
            for concept_id in get_concept_ids_for_claim(conn, claim.claim_id):
                concept_counter[concept_id] += 1

    for result in search_results:
        chunk = get_chunk(conn, result["chunk_id"])
        if chunk is None:
            continue
        doc = get_document(conn, chunk.doc_id)
        if doc is None:
            continue

        claims = chunk_claims.get(chunk.chunk_id, [])
        if not claims and not include_superseded:
            continue
        if not claims:
            claims = get_claims_by_chunk(conn, chunk.chunk_id)

        primary_claim = select_primary_claim(claims, include_superseded=include_superseded)
        if primary_claim is None:
            continue

        repeated_concept_boost = 0
        for concept_id in get_concept_ids_for_claim(conn, primary_claim.claim_id):
            repeated_concept_boost += max(concept_counter[concept_id] - 1, 0)

        superseded_count = sum(1 for claim in claims if claim.lifecycle_status == "superseded")
        verified_count = sum(1 for claim in claims if claim.verified == 1)
        score = (
            result["score"]
            + verified_count * 0.05
            + repeated_concept_boost * 0.03
            - superseded_count * 0.04
        )

        note = None
        is_primary = primary_claim.verified == 1 and primary_claim.lifecycle_status == "active"
        if primary_claim.lifecycle_status == "contested":
            note = "This claim is contested by another active claim."
            is_primary = False
        elif primary_claim.lifecycle_status == "superseded":
            note = "This claim has been superseded by newer evidence."
            is_primary = False
        elif primary_claim.verified != 1:
            note = "This claim failed traceability verification."
            is_primary = False

        candidates.append({
            "doc_id": doc.doc_id,
            "chunk_id": chunk.chunk_id,
            "title": doc.title,
            "year": doc.year,
            "section": chunk.section or "General",
            "chunk_text": chunk.text,
            "claim_id": primary_claim.claim_id,
            "claim_text": primary_claim.claim_text,
            "raw_quote": primary_claim.raw_quote,
            "verified": primary_claim.verified,
            "lifecycle_status": primary_claim.lifecycle_status,
            "superseded_by_claim_id": primary_claim.superseded_by_claim_id,
            "is_primary_evidence": is_primary,
            "note": note,
            "rank": result["rank"],
            "score": score,
        })

    return sorted(
        candidates,
        key=lambda c: (
            not c["is_primary_evidence"],
            c["lifecycle_status"] == "superseded",
            c["verified"] != 1,
            -c["score"],
        ),
    )


def select_primary_claim(claims: list, include_superseded: bool = False):
    """Choose the best representative claim for a chunk."""
    if not claims:
        return None

    def priority(claim) -> tuple[int, int, int]:
        lifecycle_rank = {
            "active": 0,
            "contested": 1,
            "superseded": 1 if include_superseded else 2,
        }
        return (
            0 if claim.verified == 1 else 1,
            lifecycle_rank.get(claim.lifecycle_status, 2),
            0 if claim.raw_quote else 1,
        )

    return sorted(claims, key=priority)[0]


def run_query(
    client: BaseLLMClient,
    conn: sqlite3.Connection,
    query: str,
    encoder: Any,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    chunk_ids: list[str],
    top_k: int = 10,
    include_superseded: bool = False,
) -> dict:
    """Answer a query with lifecycle-aware evidence from the knowledge base."""
    search_results = hybrid_search(
        query, encoder, faiss_index, bm25_index, chunk_ids, top_k=top_k
    )

    if not search_results:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "sources": [],
            "confidence": 0.0,
            "uncertainty": "Knowledge base may be empty or the query has no matches.",
            "chunks_retrieved": 0,
        }

    context, source_metadata = build_context(
        conn,
        search_results,
        include_superseded=include_superseded,
    )
    if not source_metadata:
        return {
            "answer": "No eligible evidence matched the query after lifecycle filtering.",
            "sources": [],
            "confidence": 0.0,
            "uncertainty": "Relevant results may exist only as superseded or unverified claims.",
            "chunks_retrieved": len(search_results),
        }

    system = query_system_prompt()
    user = f"Question: {query}\n\nContext:\n{context}"

    result = client.complete_json(system, user, max_tokens=2048)

    answer_sources = result.get("sources", [])
    for src in answer_sources:
        for meta in source_metadata:
            if src.get("chunk_id", "").startswith(meta["chunk_id"][:8]):
                src.update({
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "title": meta["title"],
                    "year": meta["year"],
                    "section": meta["section"],
                    "rank": meta["rank"],
                    "verified": meta["verified"],
                    "lifecycle_status": meta["lifecycle_status"],
                    "is_primary_evidence": meta["is_primary_evidence"],
                    "note": meta["note"],
                })
                break

    uncertainty = result.get("uncertainty", "")
    contested_sources = [src for src in source_metadata if src["lifecycle_status"] == "contested"]
    if contested_sources:
        warning = "Some relevant evidence is contested by contradictory active claims."
        uncertainty = f"{uncertainty} {warning}".strip()

    return {
        "answer": result.get("answer", ""),
        "sources": answer_sources,
        "confidence": result.get("confidence", 0.5),
        "uncertainty": uncertainty,
        "chunks_retrieved": len(search_results),
    }
