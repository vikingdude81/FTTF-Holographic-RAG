# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-04-10

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
