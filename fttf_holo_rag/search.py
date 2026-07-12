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
from fttf_holo_rag.keyword import KeywordIndex
from fttf_holo_rag.utils import extract_project_name

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
class KeywordResult:
    """A single keyword search hit."""
    doc_id: str
    filepath: str
    project: str
    score: float       # BM25 score (normalized 0–1)
    snippet: str = ''  # first N chars of document text


@dataclass
class KeywordSearchResult:
    """Complete keyword search result."""
    query: str
    results: list[KeywordResult] = field(default_factory=list)
    total_indexed: int = 0

    def to_text(self) -> str:
        out = f"KEYWORD SEARCH (BM25, {self.total_indexed} docs indexed):\n\n"
        if not self.results:
            out += "  (no matches)\n"
            return out
        for r in self.results:
            pct = round(r.score * 100, 1)
            out += f"-> [{r.project}] {r.filepath} (relevance: {pct}%)\n"
            if r.snippet:
                out += f"   {r.snippet}\n"
        return out


@dataclass
class HybridResult:
    """A fused result combining vector and keyword scores."""
    doc_id: str
    filepath: str
    project: str
    vector_score: float
    keyword_score: float
    combined_score: float
    snippet: str = ''


@dataclass
class HybridSearchResult:
    """Complete hybrid search result."""
    query: str
    alpha: float  # weight given to vector score (1-alpha = keyword weight)
    results: list[HybridResult] = field(default_factory=list)

    def to_text(self) -> str:
        out = f"HYBRID SEARCH (alpha={self.alpha:.2f}, vec+keyword fusion):\n\n"
        if not self.results:
            out += "  (no matches)\n"
            return out
        for r in self.results:
            pct = round(r.combined_score * 100, 1)
            out += (f"-> [{r.project}] {r.filepath} "
                    f"(combined: {pct}%, vec: {r.vector_score*100:.1f}%, "
                    f"kw: {r.keyword_score*100:.1f}%)\n")
            if r.snippet:
                out += f"   {r.snippet}\n"
        return out


@dataclass
class SearchResult:
    """Complete search result from a holographic query."""
    query: str
    stage: str  # "centroid" or "HRR-fallback"
    projects: list[ProjectResult] = field(default_factory=list)
    promotable: list[tuple[str, int, int]] = field(default_factory=list)
    holo_dim: int = 10_000

    def to_text(self) -> str:
        """Format as human-readable text (compatible with LLM tool output)."""
        result = f"HOLOGRAPHIC SEARCH v3 ({self.holo_dim}D, {self.stage} ranking + 768D file recovery):\n\n"
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
        self._keyword_index: KeywordIndex | None = None

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

    def _get_collection(self):
        """Lazily create the ChromaDB client and return the configured collection."""
        if not hasattr(self, '_db_client') or self._db_client is None:
            self._db_client = chromadb.PersistentClient(path=self.cfg.chroma_path)
        return self._db_client.get_collection(name=self.cfg.chroma_collection)

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
        if not query or not query.strip():
            raise ValueError("Search query must be a non-empty string.")
        cfg = self.cfg

        self._cache.build()

        if not self._cache.loaded:
            raise RuntimeError(
                "Holographic cache is empty — ensure ChromaDB is populated and accessible "
                f"at '{cfg.chroma_path}' with collection '{cfg.chroma_collection}'."
            )

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
        tracker.flush()

        # Stage 2: File ranking
        stage = "centroid" if use_centroids else "HRR-fallback"
        q_vec = np.array(query_emb_raw, dtype=np.float32)
        n_q2 = norm(q_vec)
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
                    n_f = norm(fp_vec)
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
            holo_dim=cfg.holo_dim,
        )
        log.info('[HOLO SEARCH v3 COMPLETE]')
        return result

    def search_scoped(self, query: str, project_name: str,
                      top_k: int = 5) -> list[FileResult]:
        """Search within a specific project by 768D cosine similarity."""
        if not query or not query.strip():
            raise ValueError("Search query must be a non-empty string.")
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
        """List all project names in the ChromaDB collection (batched to avoid SQLite variable limits)."""
        collection = self._get_collection()
        projects: set[str] = set()
        offset = 0
        batch_size = 500
        while True:
            data = collection.get(include=['metadatas'], limit=batch_size, offset=offset)
            ids = data.get('ids', [])
            if not ids:
                break
            for m in data['metadatas']:
                fp = m.get('filepath', '')
                projects.add(extract_project_name(fp, self.cfg.allowed_project_roots))
            if len(ids) < batch_size:
                break
            offset += len(ids)
        return sorted(projects)

    def get_file_content(self, filepath: str) -> str | None:
        """Retrieve a file's content from ChromaDB by its filepath (doc ID)."""
        try:
            collection = self._get_collection()
            result = collection.get(ids=[filepath], include=['documents'])
            if result['documents']:
                return result['documents'][0]
        except Exception as exc:
            log.warning('[HOLO] get_file_content failed for %r: %s', filepath, exc)
        return None

    # ─────────────────────────────────────────────────────────
    # Keyword (BM25) Search
    # ─────────────────────────────────────────────────────────

    def _ensure_keyword_index(self) -> KeywordIndex:
        """Build or return the cached BM25 keyword index."""
        if self._keyword_index is not None and self._keyword_index.doc_count > 0:
            return self._keyword_index
        collection = self._get_collection()
        idx = KeywordIndex()
        idx.build_from_chromadb(collection, max_docs=self.cfg.max_docs)
        self._keyword_index = idx
        return idx

    def keyword_search(self, query: str, top_k: int = 10,
                       snippet_chars: int = 120) -> KeywordSearchResult:
        """BM25 keyword search across all documents in ChromaDB.

        Unlike the holographic vector search, this matches exact words
        and is useful for known identifiers, error messages, or terms
        that don't embed well.
        """
        if not query or not query.strip():
            raise ValueError("Search query must be a non-empty string.")

        idx = self._ensure_keyword_index()
        hits = idx.search_normalized(query, top_k=top_k)

        # Fetch metadata + snippets for the hits
        if not hits:
            return KeywordSearchResult(query=query, total_indexed=idx.doc_count)

        doc_ids = [doc_id for doc_id, _ in hits]
        collection = self._get_collection()
        fetched = collection.get(
            ids=doc_ids,
            include=['documents', 'metadatas'],
        )
        meta_map = {}
        doc_map = {}
        for did, meta, doc in zip(
            fetched['ids'], fetched['metadatas'], fetched['documents']
        ):
            meta_map[did] = meta
            doc_map[did] = doc

        results = []
        for doc_id, score in hits:
            meta = meta_map.get(doc_id, {})
            fp = meta.get('filepath', doc_id)
            proj = extract_project_name(fp, self.cfg.allowed_project_roots)
            text = doc_map.get(doc_id, '')
            snippet = (text[:snippet_chars] + '...') if text and len(text) > snippet_chars else (text or '')
            results.append(KeywordResult(
                doc_id=doc_id,
                filepath=fp,
                project=proj,
                score=score,
                snippet=snippet,
            ))

        return KeywordSearchResult(
            query=query, results=results, total_indexed=idx.doc_count,
        )

    # ─────────────────────────────────────────────────────────
    # Hybrid Search (Vector + Keyword Fusion)
    # ─────────────────────────────────────────────────────────

    def hybrid_search(self, query: str, top_k: int = 10,
                      alpha: float = 0.5,
                      snippet_chars: int = 120) -> HybridSearchResult:
        """Fuse holographic vector similarity with BM25 keyword scores.

        alpha controls the blend:
            1.0 = pure vector search
            0.0 = pure keyword search
            0.5 = equal weight (default)

        The fusion uses min-max normalized scores from both sources,
        then combines: score = alpha * vec_score + (1-alpha) * kw_score.
        """
        if not query or not query.strip():
            raise ValueError("Search query must be a non-empty string.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        # --- Keyword scores ---
        idx = self._ensure_keyword_index()
        kw_all = idx.search_normalized(query, top_k=top_k * 3)
        kw_scores = {doc_id: score for doc_id, score in kw_all}

        # --- Vector scores (file-level 768D cosine) ---
        self._cache.build()
        if not self._cache.loaded:
            raise RuntimeError("Holographic cache is empty.")
        if not self._ensure_ollama():
            raise ConnectionError("Ollama embedding service is unreachable.")

        query_emb = get_embedding(query, self.cfg.ollama_url, self.cfg.embed_model)
        q_vec = np.array(query_emb, dtype=np.float32)
        n_q = norm(q_vec)

        vec_scores: dict[str, tuple[float, str]] = {}  # doc_id -> (sim, project)
        embeddings = self._cache.data.get('embeddings') or {}
        for proj_name, emb_map in embeddings.items():
            for fp, emb in emb_map.items():
                fp_vec = np.array(emb, dtype=np.float32)
                n_f = norm(fp_vec)
                sim = float(np.dot(q_vec, fp_vec) / (n_q * n_f)) if n_f > 1e-12 and n_q > 1e-12 else 0.0
                vec_scores[fp] = (sim, proj_name)

        # Normalize vector scores to 0-1
        if vec_scores:
            max_vec = max(s for s, _ in vec_scores.values())
            min_vec = min(s for s, _ in vec_scores.values())
            span = max_vec - min_vec if max_vec > min_vec else 1.0
            vec_norm = {did: ((s - min_vec) / span, proj) for did, (s, proj) in vec_scores.items()}
        else:
            vec_norm = {}

        # --- Fuse ---
        all_doc_ids = set(kw_scores.keys()) | set(vec_norm.keys())
        fused: list[tuple[str, float, float, float, str]] = []
        for doc_id in all_doc_ids:
            v_score, proj = vec_norm.get(doc_id, (0.0, ''))
            k_score = kw_scores.get(doc_id, 0.0)
            # Resolve project from keyword side if missing
            if not proj and doc_id in kw_scores:
                proj = ''
            combined = alpha * v_score + (1 - alpha) * k_score
            fused.append((doc_id, combined, v_score, k_score, proj))

        fused.sort(key=lambda x: x[1], reverse=True)
        top = fused[:top_k]

        # Fetch metadata + snippets
        if top:
            doc_ids = [d[0] for d in top]
            collection = self._get_collection()
            fetched = collection.get(ids=doc_ids, include=['documents', 'metadatas'])
            meta_map = {did: m for did, m in zip(fetched['ids'], fetched['metadatas'])}
            doc_map = {did: d for did, d in zip(fetched['ids'], fetched['documents'])}
        else:
            meta_map, doc_map = {}, {}

        results = []
        for doc_id, combined, v_score, k_score, proj in top:
            meta = meta_map.get(doc_id, {})
            fp = meta.get('filepath', doc_id)
            if not proj:
                proj = extract_project_name(fp, self.cfg.allowed_project_roots)
            text = doc_map.get(doc_id, '')
            snippet = (text[:snippet_chars] + '...') if text and len(text) > snippet_chars else (text or '')
            results.append(HybridResult(
                doc_id=doc_id,
                filepath=fp,
                project=proj,
                vector_score=v_score,
                keyword_score=k_score,
                combined_score=combined,
                snippet=snippet,
            ))

        return HybridSearchResult(query=query, alpha=alpha, results=results)
