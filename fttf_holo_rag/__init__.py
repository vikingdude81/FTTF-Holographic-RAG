"""
FTTF Holographic RAG — Fourier Transform holographic memory for local RAG.

Public API:
    from fttf_holo_rag import HoloRAG, HoloConfig

    rag = HoloRAG(HoloConfig(chroma_path="./my_db"))
    results = rag.search("websocket authentication")
    rag.build_cache()

Ingestion API:
    from fttf_holo_rag import ingest_directories, chunk_text

    ingest_directories(["./my-project"], chroma_path="./chroma_db_storage")
"""

from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.search import HoloRAG
from fttf_holo_rag.engine import (
    HolographicMemory,
    MultiBankMemory,
    DimensionProjector,
    HitTracker,
)
from fttf_holo_rag.ingest import ingest_directories
from fttf_holo_rag.chunker import chunk_text

__version__ = "0.1.0"

__all__ = [
    "HoloRAG",
    "HoloConfig",
    "HolographicMemory",
    "MultiBankMemory",
    "DimensionProjector",
    "HitTracker",
    "ingest_directories",
    "chunk_text",
]
