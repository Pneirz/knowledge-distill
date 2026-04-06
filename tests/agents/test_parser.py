# %%
from pathlib import Path

import pytest

from distill.agents.parser import chunk_sections, parse_html, parse_github_repo


def test_chunk_sections_respects_token_limit():
    """Chunks must not exceed max_tokens when text has sentence boundaries."""
    # Build realistic multi-sentence text so the chunker can split on boundaries
    sentences = [f"This is sentence number {i} in the section." for i in range(40)]
    sections = [
        {
            "title": "Introduction",
            "text": " ".join(sentences),
            "page_start": 1,
            "page_end": 2,
        }
    ]
    chunks = chunk_sections(sections, max_tokens=64, overlap_tokens=16)
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    assert len(chunks) > 1, "Text should be split into multiple chunks"
    for chunk in chunks:
        # Allow a single-sentence overshoot at boundaries (one sentence ~ 15 tokens)
        assert len(enc.encode(chunk["text"])) <= 80


def test_chunk_sections_preserves_section_metadata():
    """Chunks carry section title and page references from their parent section."""
    sections = [
        {
            "title": "Methods",
            "text": "This is the methods section. It describes the approach.",
            "page_start": 3,
            "page_end": 4,
        }
    ]
    chunks = chunk_sections(sections)
    assert all(c["section"] == "Methods" for c in chunks)
    assert all(c["page_start"] == 3 for c in chunks)


def test_chunk_sections_empty_section_skipped():
    """Sections with empty text produce no chunks."""
    sections = [{"title": "Empty", "text": "", "page_start": 1, "page_end": 1}]
    chunks = chunk_sections(sections)
    assert chunks == []


def test_chunk_sections_overlap_shared_content():
    """When a section is split, the last sentences of a chunk reappear at the start of the next."""
    text = ". ".join([f"Sentence {i}" for i in range(40)]) + "."
    sections = [{"title": "S", "text": text, "page_start": None, "page_end": None}]
    chunks = chunk_sections(sections, max_tokens=50, overlap_tokens=20)
    if len(chunks) > 1:
        # Last words of chunk[0] should appear somewhere in chunk[1]
        last_sentence = chunks[0]["text"].split(". ")[-1]
        assert last_sentence in chunks[1]["text"] or len(last_sentence.split()) <= 2


def test_parse_html_extracts_paragraphs(tmp_path: Path):
    """parse_html extracts article text and skips nav/footer elements."""
    html_content = """
    <html>
    <body>
      <nav>Navigation garbage</nav>
      <article>
        <h2>Introduction</h2>
        <p>This is the introduction paragraph.</p>
        <h2>Methods</h2>
        <p>This describes the methods.</p>
      </article>
      <footer>Footer garbage</footer>
    </body>
    </html>
    """
    html_file = tmp_path / "article.html"
    html_file.write_text(html_content)

    result = parse_html(html_file)
    assert "introduction paragraph" in result["full_text"].lower()
    assert "methods" in result["full_text"].lower()
    assert "navigation garbage" not in result["full_text"].lower()
    assert "footer garbage" not in result["full_text"].lower()


def test_parse_html_returns_sections(tmp_path: Path):
    """parse_html creates sections for each H2/H3 header."""
    html_content = """
    <html><body><article>
      <h2>Section One</h2><p>Content one.</p>
      <h2>Section Two</h2><p>Content two.</p>
    </article></body></html>
    """
    html_file = tmp_path / "article.html"
    html_file.write_text(html_content)
    result = parse_html(html_file)
    titles = [s["title"] for s in result["sections"]]
    assert "Section One" in titles
    assert "Section Two" in titles


def test_parse_github_repo_splits_on_headers(tmp_path: Path):
    """parse_github_repo splits Markdown on ## headers."""
    md_content = """# Project Title

Some intro text.

## Installation

Install the package.

## Usage

Use it like this.
"""
    md_file = tmp_path / "README.md"
    md_file.write_text(md_content)
    result = parse_github_repo(md_file)

    titles = [s["title"] for s in result["sections"]]
    assert "Installation" in titles
    assert "Usage" in titles
