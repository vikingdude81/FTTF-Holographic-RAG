"""
tests/test_ingest.py — Tests for ingest.py

Uses unittest.mock to avoid requiring a live ChromaDB or Ollama instance.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

from fttf_holo_rag.ingest import IngestResult, ingest_directories


def _make_collection(existing_ids: list[str] | None = None):
    """Return a mock ChromaDB collection."""
    col = MagicMock()
    col.get.return_value = {"ids": existing_ids or []}
    col.upsert = MagicMock()
    return col


def _make_chroma_client(collection):
    """Return a mock chromadb.PersistentClient whose get_or_create_collection returns *collection*."""
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


class TestIngestDirectoriesBasic(unittest.TestCase):
    """Verify basic crawl and upsert behaviour with a temp directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Write a couple of small Python files
        with open(os.path.join(self.tmpdir, "a.py"), "w") as fh:
            fh.write("x = 1\n")
        with open(os.path.join(self.tmpdir, "b.md"), "w") as fh:
            fh.write("# Hello\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_returns_ingest_result(self, mock_ef, mock_chroma):
        col = _make_collection()
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        result = ingest_directories([self.tmpdir])
        self.assertIsInstance(result, IngestResult)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_adds_new_files(self, mock_ef, mock_chroma):
        col = _make_collection()
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        result = ingest_directories([self.tmpdir])
        self.assertEqual(result.added, 2)
        self.assertEqual(result.failed, 0)
        # upsert should have been called
        self.assertTrue(col.upsert.called)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_dedup_skips_existing(self, mock_ef, mock_chroma):
        # Pre-populate collection with one file's absolute path
        existing = [os.path.abspath(os.path.join(self.tmpdir, "a.py"))]
        col = _make_collection(existing_ids=existing)
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        result = ingest_directories([self.tmpdir])
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.added, 1)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_nonexistent_directory_skipped(self, mock_ef, mock_chroma):
        col = _make_collection()
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        result = ingest_directories(["/this/does/not/exist"])
        self.assertEqual(result.added, 0)
        self.assertFalse(col.upsert.called)


class TestIngestBatchLogic(unittest.TestCase):
    """Verify batch splitting and one-by-one fallback."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Write 5 small files so we can test batching
        for i in range(5):
            with open(os.path.join(self.tmpdir, f"f{i}.py"), "w") as fh:
                fh.write(f"# file {i}\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_batch_size_respected(self, mock_ef, mock_chroma):
        col = _make_collection()
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        ingest_directories([self.tmpdir], batch_size=2)
        # With 5 files and batch_size=2 we expect ceil(5/2) = 3 upsert calls
        self.assertEqual(col.upsert.call_count, 3)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_fallback_on_batch_failure(self, mock_ef, mock_chroma):
        col = _make_collection()
        # First batch upsert fails, subsequent single upserts succeed
        fail_once = [True]

        def _upsert(**kwargs):
            if fail_once[0] and len(kwargs.get("documents", [])) > 1:
                fail_once[0] = False
                raise RuntimeError("simulated batch failure")

        col.upsert.side_effect = _upsert
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        result = ingest_directories([self.tmpdir], batch_size=5)
        # All 5 should end up added (via fallback)
        self.assertEqual(result.added, 5)
        self.assertEqual(result.failed, 0)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_failed_count_on_single_failure(self, mock_ef, mock_chroma):
        col = _make_collection()
        # Every upsert raises regardless of batch size
        col.upsert.side_effect = RuntimeError("always fails")
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        result = ingest_directories([self.tmpdir], batch_size=5)
        self.assertEqual(result.added, 0)
        self.assertEqual(result.failed, 5)


class TestIngestCacheInvalidation(unittest.TestCase):
    """Verify holographic cache deletion after ingestion."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "src.py"), "w") as fh:
            fh.write("pass\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_cache_file_removed_after_ingest(self, mock_ef, mock_chroma):
        col = _make_collection()
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            cache_path = tf.name

        try:
            self.assertTrue(os.path.exists(cache_path))
            ingest_directories([self.tmpdir], holo_cache_path=cache_path)
            self.assertFalse(os.path.exists(cache_path))
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_cache_not_required(self, mock_ef, mock_chroma):
        col = _make_collection()
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        # Should not raise even when holo_cache_path is None or missing
        result = ingest_directories([self.tmpdir], holo_cache_path=None)
        self.assertIsInstance(result, IngestResult)

        result2 = ingest_directories([self.tmpdir], holo_cache_path="/no/such/file.pkl")
        self.assertIsInstance(result2, IngestResult)


class TestIngestMaxChars(unittest.TestCase):
    """Verify that content is truncated to max_chars."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "big.py"), "w") as fh:
            fh.write("x = 1\n" * 1000)  # ~6000 chars

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("fttf_holo_rag.ingest.chromadb")
    @patch("fttf_holo_rag.ingest.embedding_functions")
    def test_content_truncated(self, mock_ef, mock_chroma):
        col = _make_collection()
        mock_chroma.PersistentClient.return_value = _make_chroma_client(col)

        ingest_directories([self.tmpdir], max_chars=100)
        # Retrieve the document passed to upsert
        call_kwargs = col.upsert.call_args
        docs = call_kwargs.kwargs["documents"]
        self.assertLessEqual(len(docs[0]), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
