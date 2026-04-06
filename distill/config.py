import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the knowledge base system.

    Reads paths and parameters from environment variables with sensible
    defaults. The data_root parameter overrides KNOWLEDGE_DATA_ROOT when
    provided explicitly (useful for tests with temporary directories).
    """

    def __init__(self, data_root: Path | None = None) -> None:
        # Resolve data root from argument or environment
        self.data_root = data_root or Path(
            os.environ.get("KNOWLEDGE_DATA_ROOT", "data")
        )

        # Data layer directories (8-layer architecture)
        self.inbox_path = self.data_root / "00_inbox"
        self.raw_path = self.data_root / "01_raw"
        self.parsed_path = self.data_root / "02_parsed"
        self.extracted_path = self.data_root / "03_extracted"
        self.wiki_path = self.data_root / "04_compiled_wiki"
        self.index_path = self.data_root / "05_search_index"
        self.outputs_path = self.data_root / "06_outputs"
        self.registry_path = self.data_root / "07_registry"
        self.db_path = self.registry_path / "distill.db"

        # LLM settings
        self.anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
        self.llm_model: str = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        self.embedding_model: str = os.environ.get(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )

        # Chunking parameters
        self.chunk_max_tokens: int = int(os.environ.get("CHUNK_MAX_TOKENS", "512"))
        self.chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "64"))

    def ensure_dirs(self) -> None:
        """Create the full directory structure if it does not exist."""
        dirs = [
            self.inbox_path,
            self.raw_path,
            self.parsed_path,
            self.extracted_path,
            self.wiki_path / "papers",
            self.wiki_path / "concepts",
            self.wiki_path / "methods",
            self.wiki_path / "contradictions",
            self.index_path,
            self.outputs_path,
            self.registry_path,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
