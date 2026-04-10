"""
Standalone configuration for FTTF Holographic RAG.

All values can be overridden by passing a HoloConfig instance to HoloRAG.
No external dependency on AI_Command_Center config.py.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class HoloConfig:
    """Configuration for the holographic RAG system.

    Override any field at construction time:
        cfg = HoloConfig(holo_dim=5000, ollama_url="http://gpu-box:11434/api/embeddings")
    """

    # --- Embedding service ---
    ollama_url: str = "http://localhost:11434/api/embeddings"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    # --- ChromaDB ---
    chroma_path: str = "./chroma_db_storage"
    chroma_collection: str = "my_codebase"

    # --- Holographic memory ---
    holo_dim: int = 10_000
    holo_chunk_size: int = 25
    holo_flat_limit: int = 60
    holo_cache_path: str = "./chroma_db_storage/holo_cache.pkl"

    # Signal processing
    holo_sharpen_p: float = 0.7
    holo_corvacs_a: float = 1.5

    # Multi-bank binding
    holo_n_banks: int = 4

    # Key orthogonalization
    holo_orth_enabled: bool = True
    holo_orth_iters: int = 2
    holo_orth_step: float = 0.3

    # Hit tracking & promotion
    holo_hits_path: str = "./chroma_db_storage/holo_hits.json"
    holo_promote_threshold: int = 3

    # Snippet control
    max_snippet_chars: int = 1500

    # Project root directories for project-name extraction
    allowed_project_roots: List[str] = field(default_factory=list)

    # Ollama health check
    ollama_wait_timeout: int = 60
    ollama_poll_interval: int = 5

    # Embedding batch size
    embed_batch_size: int = 200

    # Maximum documents to fetch from ChromaDB
    max_docs: int = 60_000
