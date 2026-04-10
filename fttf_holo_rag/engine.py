"""
engine.py — Core Holographic Memory primitives (FTTF v2).

Pure-math module: depends only on numpy + requests.
No coupling to ChromaDB, config, or any framework.

Upgrades over v1:
  1. Signal processing: sharpen() + corvacs_lite() for cleaner decode
  2. Orthogonalization: decorrelate keys to reduce cross-talk
  3. Multi-bank binding: spread items across N banks, ensemble retrieve
  4. Hit tracking: record recall frequency for importance-based promotion
"""

import json
import os

import numpy as np
from numpy.linalg import norm
import requests


# ─────────────────────────────────────────────────────────────
# Signal Processing
# ─────────────────────────────────────────────────────────────

def sharpen(vec, p=2.0, eps=1e-12):
    """Element-wise magnitude sharpening for real-valued vectors.

    Amplifies strong components, suppresses weak ones.
    sign(x) * (|x| + eps)^p
    p > 1  → contrast-increasing  (default 2.0)
    p == 1 → identity
    p < 1  → softening
    """
    if p == 1.0:
        return vec.copy()
    abs_vec = np.abs(vec) + eps
    return np.sign(vec) * (abs_vec ** p)


def corvacs_lite(vec, a=3.0):
    """Gentle magnitude limiter (CORVACS-lite) — element-wise.

    Applies tanh saturation to prevent any single dimension from dominating.
    tanh(a * x) / a
    a == 0 → identity
    a > 0  → soft saturation (default 3.0)
    """
    if a <= 0:
        return vec.copy()
    return np.tanh(a * vec) / a


def decode_signal(vec, sharpen_p=0.7, corvacs_a=1.5):
    """Full signal processing decode pipeline.

    1. Normalize to unit norm
    2. Sharpen: compress dynamic range (p < 1 amplifies weak signals)
    3. CORVACS-lite: prevent outlier dimensions from dominating

    Call this on unbind results before computing cosine similarity.
    """
    n = norm(vec)
    if n < 1e-12:
        return vec
    out = vec / n
    out = sharpen(out, p=sharpen_p)
    out = corvacs_lite(out, a=corvacs_a)
    return out


# ─────────────────────────────────────────────────────────────
# Key Orthogonalization
# ─────────────────────────────────────────────────────────────

def orthogonalize_keys(keys, iters=2, step=0.3):
    """Gram-Schmidt-like decorrelation for real-valued key vectors.

    Iteratively subtracts correlated components and re-normalizes.
    Reduces cross-talk between bindings in superposition.
    """
    if iters <= 0 or len(keys) < 2:
        return keys

    V = len(keys)
    D = len(keys[0])
    K = np.array(keys, dtype=np.float64)

    for _ in range(iters):
        G = K @ K.T
        np.fill_diagonal(G, 0.0)
        correction = G @ K
        K -= (step / D) * correction
        norms = np.linalg.norm(K, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        K /= norms

    return [K[i] for i in range(V)]


# ─────────────────────────────────────────────────────────────
# Core HRR Primitives
# ─────────────────────────────────────────────────────────────

class HolographicMemory:
    def __init__(self, dim):
        self.dim = dim

    def generate_key(self):
        v = np.random.normal(0, 1.0, self.dim)
        n = norm(v)
        return v / n if n > 0 else v

    def bind(self, key, value):
        return np.real(np.fft.ifft(np.fft.fft(key) * np.fft.fft(value)))

    def unbind(self, memory, key):
        return np.real(np.fft.ifft(np.fft.fft(memory) * np.conj(np.fft.fft(key))))


# ─────────────────────────────────────────────────────────────
# Multi-Bank Holographic Memory
# ─────────────────────────────────────────────────────────────

class MultiBankMemory:
    """Spreads bindings across N banks to reduce interference.

    Items are assigned round-robin. On retrieve, each bank is probed
    and similarities are summed (ensemble) for the final score.
    """

    def __init__(self, dim, n_banks=4, sharpen_p=0.7, corvacs_a=1.5,
                 orthogonalize=True, orth_iters=2, orth_step=0.3):
        self.dim = dim
        self.n_banks = n_banks
        self.sharpen_p = sharpen_p
        self.corvacs_a = corvacs_a
        self.orthogonalize = orthogonalize
        self.orth_iters = orth_iters
        self.orth_step = orth_step
        self.hrr = HolographicMemory(dim)
        self.banks = [np.zeros(dim, dtype=np.float64) for _ in range(n_banks)]
        self.keys = []
        self.bank_assignments = []
        self._item_count = 0

    def bind_item(self, key, value, bank_idx=None):
        if bank_idx is None:
            bank_idx = self._item_count % self.n_banks
        bound = self.hrr.bind(key, value)
        self.banks[bank_idx] += bound
        self.keys.append(key)
        self.bank_assignments.append(bank_idx)
        self._item_count += 1
        return bank_idx

    def unbind_item(self, query_key, bank_idx=None):
        if bank_idx is not None:
            return self.hrr.unbind(self.banks[bank_idx], query_key)
        recovered = np.zeros(self.dim, dtype=np.float64)
        for bank in self.banks:
            recovered += self.hrr.unbind(bank, query_key)
        return recovered

    def build_from_embeddings(self, embeddings, keys=None):
        n = len(embeddings)
        if keys is None:
            keys = [self.hrr.generate_key() for _ in range(n)]
        if self.orthogonalize and n >= 2:
            keys = orthogonalize_keys(keys, self.orth_iters, self.orth_step)
        self.banks = [np.zeros(self.dim, dtype=np.float64) for _ in range(self.n_banks)]
        self.keys = []
        self.bank_assignments = []
        self._item_count = 0
        for key, emb in zip(keys, embeddings):
            self.bind_item(key, emb)
        return keys, self.bank_assignments

    def similarity(self, query_vec, proj_vec=None):
        if proj_vec is not None:
            target = proj_vec
        else:
            target = np.sum(self.banks, axis=0)
        n_query = norm(query_vec)
        n_target = norm(target)
        if n_query < 1e-12 or n_target < 1e-12:
            return 0.0
        return float(np.dot(query_vec, target) / (n_query * n_target))


# ─────────────────────────────────────────────────────────────
# Hit Tracking & Promotion
# ─────────────────────────────────────────────────────────────

class HitTracker:
    """Tracks recall frequency per project/key for importance-based promotion."""

    def __init__(self, path=None):
        self.path = path
        self.hits = {}
        self.sessions = {}
        if path and os.path.exists(path):
            self._load()

    def record_hit(self, key, session_id=None):
        self.hits[key] = self.hits.get(key, 0) + 1
        if session_id:
            if key not in self.sessions:
                self.sessions[key] = set()
            self.sessions[key].add(session_id)
        self._save()

    def get_hits(self, key):
        return self.hits.get(key, 0)

    def get_session_count(self, key):
        return len(self.sessions.get(key, set()))

    def get_promotable(self, threshold=3):
        return [
            (key, self.hits[key], len(self.sessions.get(key, set())))
            for key in self.hits
            if len(self.sessions.get(key, set())) >= threshold
        ]

    def get_boost_scores(self):
        if not self.hits:
            return {}
        max_hits = max(self.hits.values())
        if max_hits == 0:
            return {}
        return {k: v / max_hits for k, v in self.hits.items()}

    def _save(self):
        if not self.path:
            return
        data = {
            'hits': self.hits,
            'sessions': {k: list(v) for k, v in self.sessions.items()},
        }
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def _load(self):
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            self.hits = data.get('hits', {})
            self.sessions = {k: set(v) for k, v in data.get('sessions', {}).items()}
        except (json.JSONDecodeError, KeyError):
            self.hits = {}
            self.sessions = {}


# ─────────────────────────────────────────────────────────────
# Dimension Projection
# ─────────────────────────────────────────────────────────────

class DimensionProjector:
    """Johnson-Lindenstrauss random projection: source_dim → target_dim.

    Uses a fixed seed so the same projection matrix is always produced,
    keeping holographic vectors consistent across sessions.
    """

    def __init__(self, source_dim=768, target_dim=10000, seed=42):
        self.source_dim = source_dim
        self.target_dim = target_dim
        rng = np.random.RandomState(seed)
        self.matrix = rng.normal(0, 1.0 / np.sqrt(target_dim), (source_dim, target_dim))

    def project(self, vec):
        projected = vec @ self.matrix
        n = norm(projected)
        return projected / n if n > 0 else projected


# ─────────────────────────────────────────────────────────────
# Embedding Utility
# ─────────────────────────────────────────────────────────────

def get_embedding(text, ollama_url, model="nomic-embed-text"):
    """Embed text via Ollama and return a normalized numpy vector."""
    payload = {"model": model, "prompt": text[:2000]}
    resp = requests.post(ollama_url, json=payload, timeout=30)
    resp.raise_for_status()
    vec = np.array(resp.json()["embedding"])
    n = norm(vec)
    return vec / n if n > 0 else vec
