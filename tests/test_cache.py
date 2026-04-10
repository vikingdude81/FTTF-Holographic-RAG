"""
tests/test_cache.py — Tests for cache building, HMAC signing, and utilities.
"""

import hashlib
import hmac
import os
import pickle
import shutil
import tempfile

import numpy as np
from numpy.linalg import norm
import unittest

from fttf_holo_rag.cache import (
    _cache_hmac_key,
    save_signed_cache,
    load_signed_cache,
    build_project_vector,
    SKIP_DIRS,
)
from fttf_holo_rag.config import HoloConfig
from fttf_holo_rag.engine import HolographicMemory
from fttf_holo_rag.utils import extract_project_name


class TestHMACSigning(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.tmpdir, 'test_cache.pkl')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_roundtrip(self):
        data = {'vectors': {'proj_a': np.array([1.0, 2.0, 3.0])}, 'version': 3}
        save_signed_cache(data, self.cache_path)
        loaded = load_signed_cache(self.cache_path)
        self.assertEqual(loaded['version'], 3)
        np.testing.assert_array_equal(loaded['vectors']['proj_a'], data['vectors']['proj_a'])

    def test_tampered_data_raises(self):
        data = {'test': 'value'}
        save_signed_cache(data, self.cache_path)
        # Tamper with the file
        with open(self.cache_path, 'r+b') as f:
            f.seek(40)
            f.write(b'TAMPERED')
        with self.assertRaises((ValueError, Exception)):
            load_signed_cache(self.cache_path)

    def test_atomic_write(self):
        """save_signed_cache uses atomic replace (tmp -> final)."""
        data = {'test': 'initial'}
        save_signed_cache(data, self.cache_path)
        self.assertTrue(os.path.exists(self.cache_path))
        self.assertFalse(os.path.exists(self.cache_path + '.tmp'))


class TestBuildProjectVector(unittest.TestCase):
    def setUp(self):
        self.cfg = HoloConfig(holo_dim=5000, holo_flat_limit=60,
                              holo_n_banks=4, holo_chunk_size=25,
                              holo_orth_enabled=True, holo_orth_iters=2,
                              holo_orth_step=0.3)

    def test_flat_binding(self):
        np.random.seed(42)
        hrr = HolographicMemory(5000)
        embs = [np.random.randn(5000) for _ in range(30)]
        for i in range(30):
            embs[i] /= norm(embs[i])
        sup, keys = build_project_vector(hrr, embs, self.cfg)
        self.assertEqual(len(keys), 30)
        self.assertEqual(sup.shape, (5000,))
        self.assertGreater(norm(sup), 0)

    def test_hierarchical_binding(self):
        np.random.seed(42)
        hrr = HolographicMemory(5000)
        # 100 items > flat_limit=60 → triggers hierarchical
        embs = [np.random.randn(5000) for _ in range(100)]
        for i in range(100):
            embs[i] /= norm(embs[i])
        sup, keys = build_project_vector(hrr, embs, self.cfg)
        self.assertEqual(len(keys), 100)
        self.assertEqual(sup.shape, (5000,))

    def test_no_orth(self):
        cfg = HoloConfig(holo_dim=5000, holo_orth_enabled=False, holo_n_banks=4)
        np.random.seed(42)
        hrr = HolographicMemory(5000)
        embs = [np.random.randn(5000) for _ in range(10)]
        sup, keys = build_project_vector(hrr, embs, cfg)
        self.assertEqual(len(keys), 10)


class TestExtractProjectName(unittest.TestCase):
    def test_windows_projects_path(self):
        self.assertEqual(
            extract_project_name(r'C:\Users\bob\Projects\myapp\src\main.py'),
            'myapp'
        )

    def test_windows_github_path(self):
        self.assertEqual(
            extract_project_name(r'C:\Users\bob\OneDrive\Documents\GitHub\coolrepo\test.py'),
            'coolrepo'
        )

    def test_drive_root(self):
        self.assertEqual(
            extract_project_name(r'E:\my-project\src\lib.rs'),
            'my-project'
        )

    def test_allowed_roots(self):
        self.assertEqual(
            extract_project_name(r'E:\my-project\src\lib.rs', allowed_roots=[r'E:\\']),
            'my-project'
        )

    def test_empty_path(self):
        self.assertEqual(extract_project_name(''), 'root')

    def test_slash_handling(self):
        self.assertEqual(
            extract_project_name('E:/projects/foo/bar.py', allowed_roots=['E:/projects']),
            'foo'
        )


class TestSkipDirs(unittest.TestCase):
    def test_known_dirs(self):
        for d in ['/node_modules/', '/.git/', '/__pycache__/', '/.venv/', '/site-packages/']:
            self.assertIn(d, SKIP_DIRS)


if __name__ == '__main__':
    unittest.main(verbosity=2)
