"""
tests/test_engine.py — Comprehensive tests for FTTF v2 holographic engine.

Tests:
  1. Signal processing (sharpen, corvacs_lite, decode_signal)
  2. Key orthogonalization
  3. Multi-bank binding & ensemble retrieval
  4. Hit tracking & promotion
  5. End-to-end capacity: v1 vs v2 comparison
  6. Integration: full pipeline with DimensionProjector

Run: python -m pytest tests/test_engine.py -v
"""

import json
import os
import shutil
import tempfile

import numpy as np
from numpy.linalg import norm
import unittest

from fttf_holo_rag.engine import (
    HolographicMemory,
    MultiBankMemory,
    DimensionProjector,
    HitTracker,
    sharpen,
    corvacs_lite,
    decode_signal,
    orthogonalize_keys,
)


# ─────────────────────────────────────────────────────────────
# 1. Signal Processing Tests
# ─────────────────────────────────────────────────────────────

class TestSharpen(unittest.TestCase):
    def test_identity_at_p1(self):
        vec = np.random.randn(1000)
        result = sharpen(vec, p=1.0)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_amplifies_strong_components(self):
        np.random.seed(42)
        vec = np.random.randn(1000)
        compressed = sharpen(vec, p=0.5)
        self.assertLess(np.std(compressed), np.std(vec))
        np.testing.assert_array_equal(np.sign(vec), np.sign(compressed))

    def test_output_shape_preserved(self):
        vec = np.random.randn(500)
        result = sharpen(vec, p=3.0)
        self.assertEqual(result.shape, vec.shape)

    def test_real_output(self):
        vec = np.random.randn(1000)
        result = sharpen(vec, p=2.0)
        self.assertTrue(np.isrealobj(result))


class TestCorvacsLite(unittest.TestCase):
    def test_identity_at_a0(self):
        vec = np.random.randn(1000)
        result = corvacs_lite(vec, a=0)
        np.testing.assert_allclose(result, vec, atol=1e-10)

    def test_reduces_peak_magnitude(self):
        np.random.seed(42)
        vec = np.random.randn(1000)
        saturated = corvacs_lite(vec, a=3.0)
        peak_orig = np.max(np.abs(np.fft.fft(vec)))
        peak_sat = np.max(np.abs(np.fft.fft(saturated)))
        self.assertLess(peak_sat, peak_orig)

    def test_output_shape_preserved(self):
        vec = np.random.randn(500)
        result = corvacs_lite(vec, a=5.0)
        self.assertEqual(result.shape, vec.shape)


class TestDecodeSignal(unittest.TestCase):
    def test_identity_passthrough(self):
        vec = np.random.randn(1000)
        result = decode_signal(vec, sharpen_p=1.0, corvacs_a=0)
        sim = np.dot(result, vec) / (norm(result) * norm(vec))
        self.assertGreater(sim, 0.999)

    def test_pipeline_runs(self):
        vec = np.random.randn(1000)
        result = decode_signal(vec, sharpen_p=2.0, corvacs_a=3.0)
        self.assertEqual(result.shape, vec.shape)
        self.assertTrue(np.isfinite(result).all())


# ─────────────────────────────────────────────────────────────
# 2. Orthogonalization Tests
# ─────────────────────────────────────────────────────────────

class TestOrthogonalizeKeys(unittest.TestCase):
    def test_reduces_correlation(self):
        np.random.seed(42)
        D, N = 2000, 20
        keys = [np.random.randn(D) for _ in range(N)]
        for i in range(N):
            keys[i] /= norm(keys[i])
        G_before = np.array(keys) @ np.array(keys).T
        np.fill_diagonal(G_before, 0)
        max_corr_before = np.max(np.abs(G_before))
        orth_keys = orthogonalize_keys(keys, iters=3, step=0.3)
        G_after = np.array(orth_keys) @ np.array(orth_keys).T
        np.fill_diagonal(G_after, 0)
        max_corr_after = np.max(np.abs(G_after))
        self.assertLess(max_corr_after, max_corr_before)

    def test_preserves_unit_norm(self):
        np.random.seed(42)
        keys = [np.random.randn(1000) for _ in range(10)]
        for i in range(10):
            keys[i] /= norm(keys[i])
        orth_keys = orthogonalize_keys(keys, iters=2, step=0.3)
        for k in orth_keys:
            self.assertAlmostEqual(norm(k), 1.0, places=2)

    def test_passthrough_single_key(self):
        key = np.random.randn(500)
        key /= norm(key)
        result = orthogonalize_keys([key], iters=2)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0], key, atol=1e-10)

    def test_passthrough_zero_iters(self):
        keys = [np.random.randn(500) for _ in range(5)]
        result = orthogonalize_keys(keys, iters=0)
        for orig, res in zip(keys, result):
            np.testing.assert_array_equal(orig, res)


# ─────────────────────────────────────────────────────────────
# 3. Multi-Bank Memory Tests
# ─────────────────────────────────────────────────────────────

class TestMultiBankMemory(unittest.TestCase):
    def setUp(self):
        self.dim = 5000
        self.mb = MultiBankMemory(
            dim=self.dim, n_banks=4,
            sharpen_p=1.0, corvacs_a=0, orthogonalize=False,
        )
        self.hrr = HolographicMemory(self.dim)

    def test_round_robin_assignment(self):
        np.random.seed(42)
        keys = [self.hrr.generate_key() for _ in range(8)]
        values = [self.hrr.generate_key() for _ in range(8)]
        for i, (k, v) in enumerate(zip(keys, values)):
            b = self.mb.bind_item(k, v)
            self.assertEqual(b, i % 4)

    def test_single_item_recovery(self):
        np.random.seed(42)
        key = self.hrr.generate_key()
        value = self.hrr.generate_key()
        self.mb.bind_item(key, value)
        recovered = self.mb.unbind_item(key, bank_idx=0)
        sim = np.dot(recovered, value) / (norm(recovered) * norm(value))
        self.assertGreater(sim, 0.3)

    def test_multi_item_recovery(self):
        np.random.seed(42)
        N = 20
        keys = [self.hrr.generate_key() for _ in range(N)]
        values = [self.hrr.generate_key() for _ in range(N)]
        for k, v in zip(keys, values):
            self.mb.bind_item(k, v)
        recovered_count = 0
        for i, (k, v) in enumerate(zip(keys, values)):
            recovered = self.mb.unbind_item(k, bank_idx=i % 4)
            sim = np.dot(recovered, v) / (norm(recovered) * norm(v))
            if sim > 0.3:
                recovered_count += 1
        self.assertGreater(recovered_count / N, 0.7)

    def test_build_from_embeddings(self):
        np.random.seed(42)
        embs = [np.random.randn(self.dim) for _ in range(12)]
        for i in range(12):
            embs[i] /= norm(embs[i])
        keys, assigns = self.mb.build_from_embeddings(embs)
        self.assertEqual(len(keys), 12)
        self.assertEqual(len(assigns), 12)
        for bank in self.mb.banks:
            self.assertGreater(norm(bank), 0)

    def test_banks_vs_single_performance(self):
        np.random.seed(42)
        DIM, N = 5000, 40
        values = [np.random.randn(DIM) / norm(np.random.randn(DIM)) for _ in range(N)]
        hrr = HolographicMemory(DIM)

        np.random.seed(42)
        single_keys = [hrr.generate_key() for _ in range(N)]
        single_sup = np.zeros(DIM)
        for k, v in zip(single_keys, values):
            single_sup += hrr.bind(k, v)
        single_recovered = sum(
            1 for k, v in zip(single_keys, values)
            if np.dot(hrr.unbind(single_sup, k), v) / (norm(hrr.unbind(single_sup, k)) * norm(v)) > 0.3
        )

        np.random.seed(42)
        mb = MultiBankMemory(DIM, n_banks=4, sharpen_p=1.0, corvacs_a=0, orthogonalize=False)
        mb_keys = [hrr.generate_key() for _ in range(N)]
        for k, v in zip(mb_keys, values):
            mb.bind_item(k, v)
        mb_recovered = sum(
            1 for i, (k, v) in enumerate(zip(mb_keys, values))
            if np.dot(mb.unbind_item(k, bank_idx=i % 4), v) / (norm(mb.unbind_item(k, bank_idx=i % 4)) * norm(v)) > 0.3
        )
        self.assertGreaterEqual(mb_recovered, single_recovered)


class TestMultiBankWithOrth(unittest.TestCase):
    def test_orth_improves_recovery(self):
        np.random.seed(42)
        DIM, N = 5000, 40
        hrr = HolographicMemory(DIM)
        values = [hrr.generate_key() for _ in range(N)]

        np.random.seed(42)
        mb_no = MultiBankMemory(DIM, n_banks=4, sharpen_p=1.0, corvacs_a=0, orthogonalize=False)
        keys_no, _ = mb_no.build_from_embeddings(values)
        no_count = sum(
            1 for i, (k, v) in enumerate(zip(keys_no, values))
            if np.dot(mb_no.unbind_item(k, bank_idx=i % 4), v) / (norm(mb_no.unbind_item(k, bank_idx=i % 4)) * norm(v) + 1e-12) > 0.3
        )

        np.random.seed(42)
        mb_orth = MultiBankMemory(DIM, n_banks=4, sharpen_p=1.0, corvacs_a=0,
                                  orthogonalize=True, orth_iters=2, orth_step=0.3)
        keys_orth, _ = mb_orth.build_from_embeddings(values)
        orth_count = sum(
            1 for i, (k, v) in enumerate(zip(keys_orth, values))
            if np.dot(mb_orth.unbind_item(k, bank_idx=i % 4), v) / (norm(mb_orth.unbind_item(k, bank_idx=i % 4)) * norm(v) + 1e-12) > 0.3
        )
        self.assertGreaterEqual(orth_count, no_count)


# ─────────────────────────────────────────────────────────────
# 4. Hit Tracking Tests
# ─────────────────────────────────────────────────────────────

class TestHitTracker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, 'hits.json')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_basic_hit_counting(self):
        tracker = HitTracker(path=self.path)
        tracker.record_hit('project_a')
        tracker.record_hit('project_a')
        tracker.record_hit('project_a')
        self.assertEqual(tracker.get_hits('project_a'), 3)
        self.assertEqual(tracker.get_hits('project_b'), 0)

    def test_session_tracking(self):
        tracker = HitTracker(path=self.path)
        tracker.record_hit('proj', session_id='s1')
        tracker.record_hit('proj', session_id='s1')
        tracker.record_hit('proj', session_id='s2')
        tracker.record_hit('proj', session_id='s3')
        self.assertEqual(tracker.get_hits('proj'), 4)
        self.assertEqual(tracker.get_session_count('proj'), 3)

    def test_promotable(self):
        tracker = HitTracker(path=self.path)
        tracker.record_hit('hot_project', session_id='s1')
        tracker.record_hit('hot_project', session_id='s2')
        tracker.record_hit('hot_project', session_id='s3')
        tracker.record_hit('cold_project', session_id='s1')
        promotable = tracker.get_promotable(threshold=3)
        keys = [k for k, _, _ in promotable]
        self.assertIn('hot_project', keys)
        self.assertNotIn('cold_project', keys)

    def test_boost_scores(self):
        tracker = HitTracker(path=self.path)
        tracker.record_hit('a')
        tracker.record_hit('a')
        tracker.record_hit('b')
        boosts = tracker.get_boost_scores()
        self.assertAlmostEqual(boosts['a'], 1.0)
        self.assertAlmostEqual(boosts['b'], 0.5)

    def test_persistence(self):
        tracker1 = HitTracker(path=self.path)
        tracker1.record_hit('proj', session_id='s1')
        tracker1.record_hit('proj', session_id='s2')
        tracker1.flush()
        tracker2 = HitTracker(path=self.path)
        self.assertEqual(tracker2.get_hits('proj'), 2)
        self.assertEqual(tracker2.get_session_count('proj'), 2)

    def test_corrupt_file_recovery(self):
        with open(self.path, 'w') as f:
            f.write('NOT VALID JSON{{{')
        tracker = HitTracker(path=self.path)
        self.assertEqual(tracker.get_hits('anything'), 0)

    def test_no_path_no_save(self):
        tracker = HitTracker(path=None)
        tracker.record_hit('x')
        self.assertEqual(tracker.get_hits('x'), 1)


# ─────────────────────────────────────────────────────────────
# 5. Capacity Tests
# ─────────────────────────────────────────────────────────────

class TestCapacity(unittest.TestCase):
    def test_v2_beats_v1_at_scale(self):
        np.random.seed(42)
        DIM, N = 5000, 60
        hrr = HolographicMemory(DIM)
        values = [hrr.generate_key() for _ in range(N)]

        np.random.seed(42)
        keys_v1 = [hrr.generate_key() for _ in range(N)]
        sup_v1 = np.zeros(DIM)
        for k, v in zip(keys_v1, values):
            sup_v1 += hrr.bind(k, v)
        v1 = sum(1 for k, v in zip(keys_v1, values) if np.dot(hrr.unbind(sup_v1, k), v) / (norm(hrr.unbind(sup_v1, k)) * norm(v) + 1e-12) > 0.3)

        np.random.seed(42)
        mb = MultiBankMemory(DIM, n_banks=4, sharpen_p=2.0, corvacs_a=3.0, orthogonalize=True)
        keys_v2, _ = mb.build_from_embeddings(values)
        v2 = sum(1 for i, (k, v) in enumerate(zip(keys_v2, values)) if np.dot(mb.unbind_item(k, bank_idx=i % 4), v) / (norm(mb.unbind_item(k, bank_idx=i % 4)) * norm(v) + 1e-12) > 0.3)
        self.assertGreaterEqual(v2, v1)

    def test_v2_at_high_load(self):
        np.random.seed(42)
        DIM, N = 10000, 100
        hrr = HolographicMemory(DIM)
        values = [hrr.generate_key() for _ in range(N)]

        np.random.seed(42)
        keys_v1 = [hrr.generate_key() for _ in range(N)]
        sup_v1 = np.zeros(DIM)
        for k, v in zip(keys_v1, values):
            sup_v1 += hrr.bind(k, v)
        v1 = sum(1 for k, v in zip(keys_v1, values) if np.dot(hrr.unbind(sup_v1, k), v) / (norm(hrr.unbind(sup_v1, k)) * norm(v) + 1e-12) > 0.1)

        np.random.seed(42)
        mb = MultiBankMemory(DIM, n_banks=4, orthogonalize=True, orth_iters=2, orth_step=0.3)
        keys_v2, _ = mb.build_from_embeddings(values)
        v2 = sum(1 for i, (k, v) in enumerate(zip(keys_v2, values)) if np.dot(mb.unbind_item(k, bank_idx=i % 4), v) / (norm(mb.unbind_item(k, bank_idx=i % 4)) * norm(v) + 1e-12) > 0.1)
        self.assertGreaterEqual(v2, v1)
        self.assertGreater(v2 / N * 100, 50)


# ─────────────────────────────────────────────────────────────
# 6. Integration Tests
# ─────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        SOURCE_DIM, TARGET_DIM = 768, 10000
        N_PROJECTS, FILES_PER = 5, 30
        projector = DimensionProjector(SOURCE_DIM, TARGET_DIM, seed=42)
        project_vecs = {}
        project_centroids_hd = {}

        for p in range(N_PROJECTS):
            rng = np.random.RandomState(p * 7777 + 13)
            centroid = rng.randn(SOURCE_DIM)
            centroid /= norm(centroid)
            file_embs = [centroid + rng.randn(SOURCE_DIM) * 0.15 for _ in range(FILES_PER)]
            for i in range(FILES_PER):
                file_embs[i] /= norm(file_embs[i])
            embs_hd = [projector.project(e) for e in file_embs]
            project_centroids_hd[f'project_{p}'] = projector.project(centroid)
            mb = MultiBankMemory(TARGET_DIM, n_banks=4, sharpen_p=1.0, corvacs_a=0, orthogonalize=True)
            mb.build_from_embeddings(embs_hd)
            project_vecs[f'project_{p}'] = np.sum(mb.banks, axis=0)

        query_hd = project_centroids_hd['project_2']
        scores = {name: np.dot(query_hd, pvec) / (norm(query_hd) * norm(pvec) + 1e-12) for name, pvec in project_vecs.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.assertEqual(ranked[0][0], 'project_2')


class TestBackwardCompatibility(unittest.TestCase):
    def test_holographic_memory_unchanged(self):
        np.random.seed(42)
        hrr = HolographicMemory(1000)
        key = hrr.generate_key()
        value = hrr.generate_key()
        bound = hrr.bind(key, value)
        recovered = hrr.unbind(bound, key)
        sim = np.dot(recovered, value) / (norm(recovered) * norm(value))
        self.assertGreater(sim, 0.5)

    def test_projector_unchanged(self):
        p1 = DimensionProjector(768, 10000, seed=42)
        p2 = DimensionProjector(768, 10000, seed=42)
        vec = np.random.randn(768)
        np.testing.assert_array_equal(p1.project(vec), p2.project(vec))


# ─────────────────────────────────────────────────────────────
# 7. Stress Tests
# ─────────────────────────────────────────────────────────────

class TestStress(unittest.TestCase):
    def test_200_items_multibank(self):
        np.random.seed(42)
        DIM, N = 10000, 200
        hrr = HolographicMemory(DIM)
        values = [hrr.generate_key() for _ in range(N)]

        np.random.seed(42)
        mb = MultiBankMemory(DIM, n_banks=4, orthogonalize=True, orth_iters=2, orth_step=0.3)
        keys_v2, _ = mb.build_from_embeddings(values)
        v2 = sum(
            1 for i, (k, v) in enumerate(zip(keys_v2, values))
            if np.dot(mb.unbind_item(k, bank_idx=i % 4), v) / (norm(mb.unbind_item(k, bank_idx=i % 4)) * norm(v) + 1e-12) > 0.05
        )
        self.assertGreater(v2 / N * 100, 50)

    def test_orthogonalize_large_batch(self):
        np.random.seed(42)
        keys = [np.random.randn(10000) for _ in range(200)]
        for i in range(200):
            keys[i] /= norm(keys[i])
        orth_keys = orthogonalize_keys(keys, iters=2, step=0.3)
        self.assertEqual(len(orth_keys), 200)
        for k in orth_keys:
            self.assertTrue(np.isfinite(k).all())
            self.assertAlmostEqual(norm(k), 1.0, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
