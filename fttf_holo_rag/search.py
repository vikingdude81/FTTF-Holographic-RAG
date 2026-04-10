"""
search.py — High-level search API for FTTF Holographic RAG.

HoloRAG is the main entry point: standalone usage or plugin integration.
Decoupled from qwen-agent — no BaseTool dependency.
"""

import logging
from dataclasses import dataclass, field

import chromadb
import numpy as np
from numpy.linalg import norm

from fttf_holo_rag.cache import HoloCache, SKIP_DIRS
from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.engine import (
    DimensionProjector,
    HitTracker,
    get_embedding,
)

log = logging.getLogger(__name__)


@dataclass
class FileResult:
    """A single file match within a project."""
    filepath: str
    relevance: float  # 0.0 – 1.0


@dataclass
class ProjectResult:
    """A project match with optional file-level results."""
    name: str
    similarity: float  # 0.0 – 1.0
    raw_similarity: float
    file_count: int
    hits: int = 0
    top_files: list[FileResult] = field(default_factory=list)


@dataclass
class SearchResult:
    """Complete search result from a holographic query."""
    query: str
    stage: str  # "centroid" or "HRR-fallback"
    projects: list[ProjectResult] = field(default_factory=list)
    promotable: list[tuple[str, int, int]] = field(default_factory=list)

    def to_text(self) -> str:
        """Format as human-readable text (compatible with LLM tool output)."""
        result = f"HOLOGRAPHIC SEARCH v3 (10000D, {self.stage} ranking + 768D file recovery):\n\n"
        for p in self.projects:
            pct = round(p.similarity * 100, 1)
            hit_str = f", recalled {p.hits}x" if p.hits > 0 else ""
            result += f"-> {p.name} (similarity: {pct}%, {p.file_count} files{hit_str})\n"
            for f in p.top_files:
                result += f"   {f.filepath} (file relevance: {f.relevance*100:.1f}%)\n"
            result += "\n"

        if self.promotable:
            result += "PROMOTED (frequently recalled):\n"
            for key, hits, sessions in self.promotable[:5]:
                result += f"  * {key}: {hits} hits across {sessions} sessions\n"

        return result


class HoloRAG:
    """Main holographic RAG interface.

    Usage:
        rag = HoloRAG(HoloConfig(chroma_path="./my_db"))
        result = rag.search("websocket authentication")
        print(result.to_text())

    Plugin integration:
        # Pass to any framework that accepts a callable
        tool_fn = rag.search
    """

    def __init__(self, config: HoloConfig | None = None):
        self.cfg = config or HoloConfig()
        self._cache = HoloCache(self.cfg)
        self._tracker: HitTracker | None = None
        self._ollama_verified = False
        self._projector = DimensionProjector(
            self.cfg.embed_dim, self.cfg.holo_dim, seed=42
        )

    @property
    def cache(self) -> HoloCache:
        return self._cache

    def _get_tracker(self) -> HitTracker:
        if self._tracker is None:
            self._tracker = HitTracker(path=self.cfg.holo_hits_path)
        return self._tracker

    def _ensure_ollama(self) -> bool:
        if self._ollama_verified:
            return True
        from fttf_holo_rag.cache import wait_for_ollama
        if wait_for_ollama(self.cfg):
            self._ollama_verified = True
            return True
        return False

    def build_cache(self, force: bool = False):
        """Build or refresh the holographic cache."""
        self._cache.build(force_rebuild=force)

    def search(self, query: str, top_k_projects: int = 5,
               top_k_files: int = 3, session_id: str = '') -> SearchResult:
        """Search across all projects.

        Stage 1: Project ranking via centroid similarity (10kD).
        Stage 2: File ranking within top projects via raw 768D cosine.
        """
        log.info("[HOLO SEARCH v3] Query: '%s'", query)
        cfg = self.cfg

        self._cache.build()

        if not self._ensure_ollama():
            raise ConnectionError("Ollama embedding service is unreachable.")

        # Embed and project query
        query_emb_raw = get_embedding(query, cfg.ollama_url, cfg.embed_model)
        query_emb_10k = self._projector.project(query_emb_raw)

        tracker = self._get_tracker()
        boosts = tracker.get_boost_scores()

        # Stage 1: Project ranking
        cache = self._cache.data
        centroids = cache.get('centroids')
        use_centroids = centroids is not None and len(centroids) > 0
        search_vecs = centroids if use_centroids else cache['vectors']

        scores = []
        for proj_name, proj_vec in search_vecs.items():
            n_p, n_q = norm(proj_vec), norm(query_emb_10k)
            sim = float(np.dot(query_emb_10k, proj_vec) / (n_q * n_p)) if n_p > 1e-12 and n_q > 1e-12 else 0.0
            boost = boosts.get(proj_name, 0.0) * 0.1
            scores.append((proj_name, min(sim + boost, 1.0), sim))
        scores.sort(key=lambda x: x[1], reverse=True)

        # Track hits
        for proj, _, _ in scores[:3]:
            tracker.record_hit(proj, session_id=session_id)

        # Stage 2: File ranking
        stage = "centroid" if use_centroids else "HRR-fallback"
        projects = []
        for rank, (proj_name, boosted, raw) in enumerate(scores[:top_k_projects]):
            files = cache['files'].get(proj_name, [])
            hits = tracker.get_hits(proj_name)
            top_files = []

            emb_map = (cache.get('embeddings') or {}).get(proj_name)
            if emb_map is not None and rank < 3:
                file_scores = []
                for fp in files:
                    fp_emb = emb_map.get(fp)
                    if fp_emb is None:
                        continue
                    fp_vec = np.array(fp_emb, dtype=np.float32)
                    q_vec = np.array(query_emb_raw, dtype=np.float32)
                    n_f, n_q2 = norm(fp_vec), norm(q_vec)
                    sim_file = float(np.dot(q_vec, fp_vec) / (n_q2 * n_f)) if n_f > 1e-12 and n_q2 > 1e-12 else 0.0
                    file_scores.append((sim_file, fp))
                file_scores.sort(reverse=True)
                top_files = [
                    FileResult(filepath=fp, relevance=sim)
                    for sim, fp in file_scores[:top_k_files]
                ]

            projects.append(ProjectResult(
                name=proj_name,
                similarity=boosted,
                raw_similarity=raw,
                file_count=len(files),
                hits=hits,
                top_files=top_files,
            ))

        promotable = tracker.get_promotable(threshold=cfg.holo_promote_threshold)

        result = SearchResult(
            query=query,
            stage=stage,
            projects=projects,
            promotable=promotable[:5],
        )
        log.info('[HOLO SEARCH v3 COMPLETE]')
        return result

    def search_scoped(self, query: str, project_name: str,
                      top_k: int = 5) -> list[FileResult]:
        """Search within a specific project by 768D cosine similarity."""
        self._cache.build()

        if not self._ensure_ollama():
            raise ConnectionError("Ollama embedding service is unreachable.")

        emb_map = (self._cache.data.get('embeddings') or {}).get(project_name)
        if not emb_map:
            available = sorted((self._cache.data.get('embeddings') or {}).keys())[:20]
            raise KeyError(
                f"Project '{project_name}' not found. "
                f"Available: {', '.join(available)}"
            )

        query_emb = get_embedding(query, self.cfg.ollama_url, self.cfg.embed_model)

        scores = []
        for fp, emb in emb_map.items():
            fp_norm = fp.replace('\\', '/')
            if any(d in fp_norm for d in SKIP_DIRS):
                continue
            n = norm(emb)
            sim = float(np.dot(query_emb, emb) / (norm(query_emb) * n)) if n > 0 else 0.0
            scores.append((sim, fp))
        scores.sort(reverse=True)

        return [FileResult(filepath=fp, relevance=sim) for sim, fp in scores[:top_k]]

    def list_projects(self) -> list[str]:
        """List all project names in the ChromaDB collection."""
        db_client = chromadb.PersistentClient(path=self.cfg.chroma_path)
        collection = db_client.get_collection(name=self.cfg.chroma_collection)
        data = collection.get(include=['metadatas'], limit=self.cfg.max_docs)
        projects = set()
        for m in data['metadatas']:
            fp = m.get('filepath', '')
            projects.add(extract_project_name(fp, self.cfg.allowed_project_roots))
        return sorted(projects)

    def get_file_content(self, filepath: str) -> str | None:
        """Retrieve a file's content from ChromaDB by its filepath (doc ID)."""
        try:
            db_client = chromadb.PersistentClient(path=self.cfg.chroma_path)
            collection = db_client.get_collection(name=self.cfg.chroma_collection)
            result = collection.get(ids=[filepath], include=['documents'])
            if result['documents']:
                return result['documents'][0]
        except Exception as exc:
            log.warning('[HOLO] get_file_content failed for %r: %s', filepath, exc)
        return None
