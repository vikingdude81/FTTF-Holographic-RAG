"""
tests/test_search.py — Tests for HoloRAG search API and config.
"""

import unittest
from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.search import HoloRAG, SearchResult, ProjectResult, FileResult


class TestHoloConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = HoloConfig()
        self.assertEqual(cfg.holo_dim, 10_000)
        self.assertEqual(cfg.embed_dim, 768)
        self.assertEqual(cfg.holo_n_banks, 4)
        self.assertEqual(cfg.embed_model, 'nomic-embed-text')
        self.assertIsInstance(cfg.allowed_project_roots, list)

    def test_override(self):
        cfg = HoloConfig(holo_dim=5000, embed_model='bge-small')
        self.assertEqual(cfg.holo_dim, 5000)
        self.assertEqual(cfg.embed_model, 'bge-small')

    def test_custom_paths(self):
        cfg = HoloConfig(
            chroma_path='/data/mydb',
            holo_cache_path='/data/cache.pkl',
        )
        self.assertEqual(cfg.chroma_path, '/data/mydb')
        self.assertEqual(cfg.holo_cache_path, '/data/cache.pkl')


class TestSearchResult(unittest.TestCase):
    def test_to_text(self):
        result = SearchResult(
            query='test query',
            stage='centroid',
            projects=[
                ProjectResult(
                    name='my-project',
                    similarity=0.65,
                    raw_similarity=0.62,
                    file_count=50,
                    hits=2,
                    top_files=[
                        FileResult(filepath='/path/to/file.py', relevance=0.78),
                    ],
                ),
            ],
        )
        text = result.to_text()
        self.assertIn('my-project', text)
        self.assertIn('65.0%', text)
        self.assertIn('file.py', text)
        self.assertIn('78.0%', text)
        self.assertIn('recalled 2x', text)

    def test_empty_result(self):
        result = SearchResult(query='nothing', stage='centroid')
        text = result.to_text()
        self.assertIn('HOLOGRAPHIC SEARCH v3', text)


class TestHoloRAGInit(unittest.TestCase):
    def test_default_config(self):
        rag = HoloRAG()
        self.assertIsInstance(rag.cfg, HoloConfig)
        self.assertEqual(rag.cfg.holo_dim, 10_000)

    def test_custom_config(self):
        cfg = HoloConfig(holo_dim=5000)
        rag = HoloRAG(cfg)
        self.assertEqual(rag.cfg.holo_dim, 5000)

    def test_cache_not_loaded_initially(self):
        rag = HoloRAG()
        self.assertFalse(rag.cache.loaded)


class TestImports(unittest.TestCase):
    """Verify the public API is importable."""

    def test_top_level_imports(self):
        from fttf_holo_rag import (
            HoloRAG, HoloConfig,
            HolographicMemory, MultiBankMemory, DimensionProjector,
            HitTracker,
        )
        self.assertIsNotNone(HoloRAG)
        self.assertIsNotNone(HoloConfig)
        self.assertIsNotNone(HolographicMemory)

    def test_engine_imports_still_available(self):
        """Internal helpers remain importable directly from fttf_holo_rag.engine."""
        from fttf_holo_rag.engine import (
            sharpen, corvacs_lite, decode_signal,
            orthogonalize_keys, get_embedding,
        )
        self.assertIsNotNone(sharpen)

    def test_version(self):
        import fttf_holo_rag
        self.assertEqual(fttf_holo_rag.__version__, '0.1.0')


if __name__ == '__main__':
    unittest.main(verbosity=2)
