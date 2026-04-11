"""
keyword.py — BM25 keyword search for FTTF Holographic RAG.

Provides lexical (word-match) search over ChromaDB documents using
Okapi BM25 scoring.  No extra dependencies beyond the standard library.

Usage:
    index = KeywordIndex()
    index.add("file1.py", "def hello world")
    index.add("file2.py", "async websocket handler")
    results = index.search("hello", top_k=5)
    # -> [("file1.py", 0.83)]
"""

import math
import re
import logging
from collections import Counter
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r'[a-z0-9]+', re.IGNORECASE)

# Common English stop-words kept minimal to avoid filtering useful code identifiers
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'it', 'of', 'in', 'to', 'and', 'or', 'for',
    'on', 'at', 'by', 'be', 'as', 'do', 'if', 'so', 'no', 'not', 'but',
    'are', 'was', 'has', 'had', 'its', 'this', 'that', 'with', 'from',
    'will', 'can', 'all', 'been', 'have', 'were', 'they', 'than', 'each',
})


def tokenize(text: str) -> list[str]:
    """Lowercase alpha-numeric tokenization with camelCase/snake_case splitting."""
    # Split camelCase BEFORE lowercasing so the boundary is visible
    expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    raw = _SPLIT_RE.findall(expanded.lower())
    tokens: list[str] = []
    for tok in raw:
        if tok and tok not in _STOP_WORDS:
            tokens.append(tok)
    return tokens


# ─────────────────────────────────────────────────────────────
# BM25 Index
# ─────────────────────────────────────────────────────────────

@dataclass
class KeywordIndex:
    """Okapi BM25 inverted index over text documents.

    Parameters:
        k1: Term-frequency saturation (default 1.5).
        b:  Length normalization (0=none, 1=full; default 0.75).
    """
    k1: float = 1.5
    b: float = 0.75
    _docs: dict[str, list[str]] = field(default_factory=dict)
    _doc_lens: dict[str, int] = field(default_factory=dict)
    _avgdl: float = 0.0
    _df: dict[str, int] = field(default_factory=dict)      # document frequency per term
    _tf: dict[str, Counter] = field(default_factory=dict)   # term frequency per doc
    _n: int = 0

    # ── Build ──

    def add(self, doc_id: str, text: str) -> None:
        """Index a single document (idempotent on doc_id)."""
        if doc_id in self._docs:
            return
        tokens = tokenize(text)
        self._docs[doc_id] = tokens
        self._doc_lens[doc_id] = len(tokens)
        self._tf[doc_id] = Counter(tokens)
        for term in set(tokens):
            self._df[term] = self._df.get(term, 0) + 1
        self._n += 1
        self._avgdl = sum(self._doc_lens.values()) / self._n if self._n else 0.0

    def build_from_chromadb(self, collection, max_docs: int = 60_000,
                            project_field: str = 'filepath') -> None:
        """Bulk-load documents from a ChromaDB collection."""
        data = collection.get(include=['documents', 'metadatas'], limit=max_docs)
        for doc_id, doc_text, meta in zip(
            data['ids'], data['documents'], data['metadatas']
        ):
            if not doc_text or not doc_text.strip():
                continue
            self.add(doc_id, doc_text)
        log.info('[KEYWORD] Indexed %d documents.', self._n)

    # ── Search ──

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return (doc_id, bm25_score) pairs sorted highest-first."""
        if self._n == 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: dict[str, float] = {}
        for term in q_tokens:
            if term not in self._df:
                continue
            df = self._df[term]
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf_counter in self._tf.items():
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lens[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search_normalized(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Like search() but scores are min-max normalized to 0.0–1.0."""
        raw = self.search(query, top_k=top_k)
        if not raw:
            return []
        max_score = raw[0][1]
        if max_score <= 0:
            return [(doc_id, 0.0) for doc_id, _ in raw]
        return [(doc_id, score / max_score) for doc_id, score in raw]

    @property
    def doc_count(self) -> int:
        return self._n

    @property
    def vocab_size(self) -> int:
        return len(self._df)
