"""
cache.py — Holographic cache building, HMAC signing, and persistence.

Manages the on-disk cache of holographic project vectors, centroids,
file embeddings, and keys.  All ChromaDB interaction lives here.
"""

import getpass
import hashlib
import hmac
import logging
import os
import pickle  # used only for dumps() in save_signed_cache; loads() gated by HMAC verification
import time

import chromadb
import numpy as np
import requests
from chromadb.utils import embedding_functions
from numpy.linalg import norm

from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.engine import (
    HolographicMemory,
    DimensionProjector,
    orthogonalize_keys,
)
from fttf_holo_rag.utils import extract_project_name

log = logging.getLogger(__name__)

SKIP_DIRS = {
    '/target/', '/node_modules/', '/.git/', '/__pycache__/', '/dist/',
    '/build/', '/.local/', '/.venv/', '/venv/', '/.tox/',
    '/.mypy_cache/', '/site-packages/',
}


# ─────────────────────────────────────────────────────────────
# HMAC Cache Integrity
# ─────────────────────────────────────────────────────────────

def _cache_hmac_key(cache_path: str) -> bytes:
    """Derive a machine-local HMAC key for cache integrity."""
    secret = f"holo-cache-{os.path.abspath(cache_path)}-{getpass.getuser()}"
    return hashlib.sha256(secret.encode()).digest()


def _sign_cache(data_bytes: bytes, cache_path: str) -> bytes:
    return hmac.new(_cache_hmac_key(cache_path), data_bytes, hashlib.sha256).digest()


def save_signed_cache(cache_dict: dict, path: str):
    """Pickle + HMAC sign and write to disk."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = pickle.dumps(cache_dict)
    sig = _sign_cache(data, path)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(sig)   # 32 bytes
        f.write(data)
    os.replace(tmp, path)


def load_signed_cache(path: str) -> dict:
    """Load pickle cache with HMAC verification. Raises ValueError on tamper."""
    with open(path, 'rb') as f:
        sig = f.read(32)
        data = f.read()
    expected = _sign_cache(data, path)
    if not hmac.compare_digest(sig, expected):
        raise ValueError(
            "Cache integrity check failed — file may be tampered or from a different machine"
        )
    return pickle.loads(data)


# ─────────────────────────────────────────────────────────────
# Ollama Health Check
# ─────────────────────────────────────────────────────────────

def wait_for_ollama(cfg: HoloConfig) -> bool:
    """Block until Ollama responds on its API port, or timeout."""
    base = cfg.ollama_url.rsplit('/api/', 1)[0]
    health_url = f"{base}/api/tags"
    deadline = time.time() + cfg.ollama_wait_timeout
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=3)
            if r.status_code == 200:
                log.info('[OLLAMA] Service is reachable.')
                return True
        except Exception:
            pass
        remaining = int(deadline - time.time())
        log.warning('[OLLAMA] Not reachable, retrying in %ds (%ds left)',
                    cfg.ollama_poll_interval, remaining)
        time.sleep(cfg.ollama_poll_interval)
    log.error('[OLLAMA] Timed out after %ds.', cfg.ollama_wait_timeout)
    return False


# ─────────────────────────────────────────────────────────────
# Mtime-Based Invalidation
# ─────────────────────────────────────────────────────────────

def _get_all(coll, include, max_docs: int = 60_000, batch: int = 1000) -> dict:
    """Batched coll.get() up to max_docs.

    A single get() with a large limit fails on big collections
    ("too many SQL variables" from SQLite), which previously crashed the
    rebuild path and silently downgraded mtime_hash to a bare count.
    """
    out = {'ids': [], 'metadatas': [], 'embeddings': [], 'documents': []}
    offset = 0
    while offset < max_docs:
        data = coll.get(include=include, limit=min(batch, max_docs - offset), offset=offset)
        got = data.get('ids') or []
        if not got:
            break
        out['ids'] += got
        for key in include:
            if key == 'ids':
                continue
            vals = data.get(key)
            if vals is not None:
                out[key].extend(list(vals))
        offset += len(got)
    return out


def mtime_hash(coll, max_docs: int = 60_000) -> str:
    """Build a fingerprint of the collection using metadata timestamps (batched)."""
    try:
        data = _get_all(coll, ['metadatas'], max_docs)
        mtimes = sorted(
            str((m or {}).get('mtime', (m or {}).get('last_modified', '')))
            for m in data['metadatas']
        )
        combined = f"{len(data['ids'])}|{'|'.join(mtimes)}"
        return hashlib.sha256(combined.encode()).hexdigest()
    except Exception:
        return str(coll.count())


# ─────────────────────────────────────────────────────────────
# Build Project Vector (v2 pipeline)
# ─────────────────────────────────────────────────────────────

def build_project_vector(hrr: HolographicMemory, embs_10k, cfg: HoloConfig):
    """Build a superposition vector using v2 pipeline:
      - Orthogonalize keys to reduce cross-talk
      - Multi-bank binding to reduce interference
    """
    all_keys = []
    n = len(embs_10k)

    raw_keys = [hrr.generate_key() for _ in range(n)]
    if cfg.holo_orth_enabled and n >= 2:
        orth_chunk = min(n, 200)
        orth_keys = []
        for ci in range(0, n, orth_chunk):
            chunk_keys = raw_keys[ci:ci + orth_chunk]
            orth_keys.extend(orthogonalize_keys(
                chunk_keys, iters=cfg.holo_orth_iters, step=cfg.holo_orth_step,
            ))
        raw_keys = orth_keys

    if n <= cfg.holo_flat_limit:
        banks = [np.zeros(cfg.holo_dim) for _ in range(cfg.holo_n_banks)]
        for idx, (key, emb) in enumerate(zip(raw_keys, embs_10k)):
            all_keys.append(key)
            b = idx % cfg.holo_n_banks
            banks[b] += hrr.bind(key, emb)
        return np.sum(banks, axis=0), all_keys

    # Hierarchical with multi-bank
    banks = [np.zeros(cfg.holo_dim) for _ in range(cfg.holo_n_banks)]
    for i in range(0, n, cfg.holo_chunk_size):
        chunk_embs = embs_10k[i:i + cfg.holo_chunk_size]
        chunk_keys_slice = raw_keys[i:i + cfg.holo_chunk_size]
        csup = np.zeros(cfg.holo_dim)
        for key, emb in zip(chunk_keys_slice, chunk_embs):
            all_keys.append(key)
            csup += hrr.bind(key, emb)
        b = (i // cfg.holo_chunk_size) % cfg.holo_n_banks
        banks[b] += csup
    return np.sum(banks, axis=0), all_keys


# ─────────────────────────────────────────────────────────────
# Full Cache Builder
# ─────────────────────────────────────────────────────────────

class HoloCache:
    """In-memory holographic cache with disk persistence."""

    def __init__(self, cfg: HoloConfig):
        self.cfg = cfg
        self._data = {
            'vectors': None,
            'centroids': None,
            'files': None,
            'keys': None,
            'doc_count': None,
            'mtime_hash': None,
            'embeddings': None,
        }
        self._ollama_verified = False

    @property
    def loaded(self) -> bool:
        return self._data['vectors'] is not None

    @property
    def data(self) -> dict:
        return self._data

    def _ensure_ollama(self) -> bool:
        if self._ollama_verified:
            return True
        if wait_for_ollama(self.cfg):
            self._ollama_verified = True
            return True
        return False

    def build(self, force_rebuild: bool = False):
        """Build or load the holographic cache. Uses mtime-based invalidation."""
        cfg = self.cfg

        if self.loaded and not force_rebuild:
            return

        # Try loading from disk
        if not force_rebuild and os.path.exists(cfg.holo_cache_path):
            try:
                try:
                    cached = load_signed_cache(cfg.holo_cache_path)
                except ValueError:
                    log.warning('[HOLO] Unsigned/tampered cache detected. Rebuilding...')
                    cached = None
                if cached is not None:
                    db_client = chromadb.PersistentClient(path=cfg.chroma_path)
                    coll = db_client.get_collection(name=cfg.chroma_collection)
                    current_hash = mtime_hash(coll, cfg.max_docs)
                    if cached.get('mtime_hash') == current_hash:
                        self._data.update(cached)
                        log.info('[HOLO] Loaded cache: %d projects, %d docs.',
                                 len(self._data['vectors']), self._data['doc_count'])
                        return
                    log.info('[HOLO] Cache stale. Rebuilding...')
            except Exception as e:
                log.warning('[HOLO] Cache load failed (%s). Rebuilding...', e)

        log.info('[HOLO] Building holographic vectors (%dD)...', cfg.holo_dim)
        db_client = chromadb.PersistentClient(path=cfg.chroma_path)
        coll = db_client.get_collection(name=cfg.chroma_collection)
        current_hash = mtime_hash(coll, cfg.max_docs)

        raw_embs = None
        try:
            all_data = _get_all(coll, ['metadatas', 'embeddings'], cfg.max_docs)
            raw_embs = all_data.get('embeddings') or []
            if not raw_embs or all(e is None for e in raw_embs):
                raw_embs = None
        except Exception:
            raw_embs = None

        if raw_embs is None:
            if not self._ensure_ollama():
                log.error('[HOLO] Skipping — Ollama unreachable.')
                return
            all_data = _get_all(coll, ['metadatas', 'documents'], cfg.max_docs)
            log.info('[HOLO] Re-embedding %d documents via Ollama...', len(all_data['ids']))
            ollama_ef = embedding_functions.OllamaEmbeddingFunction(
                url=cfg.ollama_url, model_name=cfg.embed_model,
            )
            raw_embs = []
            docs = all_data['documents']
            consecutive_failures = 0
            for i in range(0, len(docs), cfg.embed_batch_size):
                batch = [d[:2000] if d and d.strip() else "empty"
                         for d in docs[i:i + cfg.embed_batch_size]]
                try:
                    raw_embs.extend(ollama_ef(batch))
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    log.warning('[HOLO] Batch %d failed: %s', i, e)
                    raw_embs.extend([None] * len(batch))
                    if consecutive_failures >= 3:
                        remaining = len(docs) - (i + cfg.embed_batch_size)
                        if remaining > 0:
                            log.error('[HOLO] 3 consecutive failures — aborting. %d docs skipped.', remaining)
                            raw_embs.extend([None] * remaining)
                        break
                if (i // cfg.embed_batch_size) % 10 == 0:
                    log.info('  [%d/%d]', i, len(docs))
            log.info('  [%d/%d] done.', len(docs), len(docs))

        doc_count = len(all_data['ids'])
        skipped = sum(1 for e in raw_embs if e is None)
        if skipped:
            log.warning('[HOLO] %d/%d documents skipped (embedding failures).', skipped, doc_count)

        project_map = {}
        for doc_id, meta, emb in zip(all_data['ids'], all_data['metadatas'], raw_embs):
            if emb is None:
                continue
            fp = meta.get('filepath', doc_id)
            proj = extract_project_name(fp, cfg.allowed_project_roots)
            project_map.setdefault(proj, []).append((fp, np.array(emb, dtype=np.float32)))

        projector = DimensionProjector(cfg.embed_dim, cfg.holo_dim, seed=42)
        hrr = HolographicMemory(cfg.holo_dim)

        project_vectors = {}
        project_centroids = {}
        project_files = {}
        project_keys = {}
        project_embeddings = {}

        for proj_name, pairs in project_map.items():
            embs_10k = [projector.project(emb) for _, emb in pairs]
            sup, keys = build_project_vector(hrr, embs_10k, cfg)
            project_vectors[proj_name] = sup
            project_centroids[proj_name] = np.mean(embs_10k, axis=0)
            project_files[proj_name] = [fp for fp, _ in pairs]
            project_keys[proj_name] = keys
            project_embeddings[proj_name] = {fp: emb for fp, emb in pairs}

        self._data.update({
            'vectors': project_vectors,
            'centroids': project_centroids,
            'files': project_files,
            'keys': project_keys,
            'doc_count': doc_count,
            'mtime_hash': current_hash,
            'embeddings': project_embeddings,
        })

        try:
            cache_dict = dict(self._data, version=3)
            save_signed_cache(cache_dict, cfg.holo_cache_path)
            log.info('[HOLO] Cache saved (signed).')
        except Exception as e:
            log.warning('[HOLO] Cache save failed: %s', e)

        log.info('[HOLO] Built %d project vectors from %d files.',
                 len(project_vectors), doc_count)
