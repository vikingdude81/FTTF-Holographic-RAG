"""
FTTF Holographic RAG — Fourier Transform holographic memory for local RAG.

Public API:
    from fttf_holo_rag import HoloRAG, HoloConfig

    rag = HoloRAG(HoloConfig(chroma_path="./my_db"))
    results = rag.search("websocket authentication")
    rag.build_cache()
"""

from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.search import HoloRAG
from fttf_holo_rag.engine import (
    HolographicMemory,
    MultiBankMemory,
    DimensionProjector,
    HitTracker,
    sharpen,
    corvacs_lite,
    decode_signal,
    orthogonalize_keys,
    get_embedding,
)

__version__ = "0.1.0"

__all__ = [
    "HoloRAG",
    "HoloConfig",
    "HolographicMemory",
    "MultiBankMemory",
    "DimensionProjector",
    "HitTracker",
    "sharpen",
    "corvacs_lite",
    "decode_signal",
    "orthogonalize_keys",
    "get_embedding",
]
