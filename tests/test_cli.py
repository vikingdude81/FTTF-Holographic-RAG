"""
tests/test_cli.py — Tests for the CLI entry point and config loading.
"""

import sys
import unittest
from unittest.mock import patch

from fttf_holo_rag.cli import _load_config, main
from fttf_holo_rag.config import HoloConfig


class _FakeArgs:
    """Minimal namespace matching the CLI argument structure."""
    chroma_path = None
    collection = None
    ollama_url = None
    cache_path = None
    dim = None


class TestLoadConfig(unittest.TestCase):
    def test_defaults_when_no_args(self):
        cfg = _load_config(_FakeArgs())
        self.assertIsInstance(cfg, HoloConfig)
        self.assertEqual(cfg.holo_dim, 10_000)
        self.assertEqual(cfg.chroma_path, './chroma_db_storage')

    def test_chroma_path_override(self):
        args = _FakeArgs()
        args.chroma_path = '/tmp/testdb'
        cfg = _load_config(args)
        self.assertEqual(cfg.chroma_path, '/tmp/testdb')

    def test_collection_override(self):
        args = _FakeArgs()
        args.collection = 'my_coll'
        cfg = _load_config(args)
        self.assertEqual(cfg.chroma_collection, 'my_coll')

    def test_ollama_url_override(self):
        args = _FakeArgs()
        args.ollama_url = 'http://gpu-box:11434/api/embeddings'
        cfg = _load_config(args)
        self.assertEqual(cfg.ollama_url, 'http://gpu-box:11434/api/embeddings')

    def test_cache_path_override(self):
        args = _FakeArgs()
        args.cache_path = '/tmp/cache.pkl'
        cfg = _load_config(args)
        self.assertEqual(cfg.holo_cache_path, '/tmp/cache.pkl')

    def test_dim_override(self):
        args = _FakeArgs()
        args.dim = 5000
        cfg = _load_config(args)
        self.assertEqual(cfg.holo_dim, 5000)


class TestMainHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['fttf-holo', '--help']):
                main()
        self.assertEqual(ctx.exception.code, 0)

    def test_no_subcommand_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['fttf-holo']):
                main()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_unknown_subcommand_exits_nonzero(self):
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['fttf-holo', 'nonexistent']):
                main()
        self.assertNotEqual(ctx.exception.code, 0)

    def test_search_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['fttf-holo', 'search', '--help']):
                main()
        self.assertEqual(ctx.exception.code, 0)

    def test_build_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['fttf-holo', 'build', '--help']):
                main()
        self.assertEqual(ctx.exception.code, 0)

    def test_projects_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            with patch('sys.argv', ['fttf-holo', 'projects', '--help']):
                main()
        self.assertEqual(ctx.exception.code, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
