# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-11

### Added

- **BM25 keyword search**: `HoloRAG.keyword_search()` for lexical word matching across all documents
- **Hybrid search**: `HoloRAG.hybrid_search()` fusing vector + keyword scores with tunable alpha blending
- `KeywordIndex` class with Okapi BM25 scoring, camelCase/snake_case-aware tokenization, and stop-word filtering
- CLI commands: `fttf-holo keyword` and `fttf-holo hybrid` with `--json` output support
- `--alpha` flag for hybrid search to control vector vs keyword weighting
- `KeywordIndex` exported from top-level package
- Tests for tokenizer, BM25 index, keyword search results, and hybrid search results

## [0.1.0] - 2026-04-10

### Added

- 2-stage holographic search: centroid project ranking (10,000D) followed by 768D file recovery
- Multi-bank HRR binding with key orthogonalization to reduce cross-talk interference
- Signal processing pipeline: sharpening and CORVACS-lite amplitude correction
- HMAC-signed cache with mtime-based invalidation and atomic writes
- Hit tracking and promotion for frequently recalled projects
- CLI interface (`fttf-holo search`, `fttf-holo build`, `fttf-holo projects`)
- LangChain / plugin integration support via callable `rag.search`
- `HoloConfig` dataclass for centralized, override-friendly configuration
- PEP 561 `py.typed` marker for typed codebase consumers
