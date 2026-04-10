"""
tests/test_chunker.py — Tests for chunker.py
"""

import unittest

from fttf_holo_rag.chunker import TextChunk, chunk_text, prep_directory


class TestChunkText(unittest.TestCase):
    def test_empty_string(self):
        result = chunk_text("")
        self.assertEqual(result, [])

    def test_short_text_single_chunk(self):
        text = "Hello, world!"
        chunks = chunk_text(text, chunk_size=25_000, overlap=500)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[0].end, len(text))
        self.assertEqual(chunks[0].index, 0)

    def test_returns_text_chunk_instances(self):
        chunks = chunk_text("abc", chunk_size=10, overlap=2)
        for c in chunks:
            self.assertIsInstance(c, TextChunk)

    def test_multiple_chunks(self):
        # 100 chars; chunk_size=30, overlap=5 → should produce multiple chunks
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        self.assertGreater(len(chunks), 1)

    def test_chunk_indices_sequential(self):
        text = "x" * 200
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        for i, c in enumerate(chunks):
            self.assertEqual(c.index, i)

    def test_overlap_respected(self):
        text = "a" * 100
        overlap = 10
        chunk_size = 40
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1].end
            curr_start = chunks[i].start
            # The new chunk starts (chunk_size - overlap) chars after the previous
            self.assertLessEqual(curr_start, prev_end)
            self.assertGreaterEqual(prev_end - curr_start, 0)

    def test_covers_full_text(self):
        text = "hello world " * 50  # 600 chars
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # First chunk starts at 0; last chunk ends at len(text)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, len(text))

    def test_paragraph_boundary_detection(self):
        # Build a text where chunks should ideally break at \n\n
        para = "word " * 300  # ~1500 chars per paragraph
        text = para + "\n\n" + para + "\n\n" + para
        chunks = chunk_text(text, chunk_size=2000, overlap=100)
        # At least 2 chunks should be produced
        self.assertGreaterEqual(len(chunks), 2)
        # The first chunk should end at or near a \n\n boundary (not mid-word)
        first_end = chunks[0].end
        # Either the boundary is at the split point or we just have a single long chunk
        self.assertTrue(
            first_end == len(text) or text[first_end - 2: first_end] == "\n\n"
            or "\n\n" in text[max(0, first_end - 200): first_end]
        )

    def test_no_data_lost(self):
        """Concatenating chunks (without overlap) must contain every character."""
        text = "abcdefghij" * 40  # 400 chars
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # Build a coverage set of characters
        covered: set[int] = set()
        for c in chunks:
            for pos in range(c.start, c.end):
                covered.add(pos)
        self.assertEqual(covered, set(range(len(text))))

    def test_exact_chunk_size_no_paragraph(self):
        # Text with no \n\n — boundary detection falls back to hard cut
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].end - chunks[0].start, 200)

    def test_single_char_text(self):
        chunks = chunk_text("Z", chunk_size=10, overlap=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Z")


class TestPrepDirectory(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile

        self.tmpdir = tempfile.mkdtemp()
        self.outdir = tempfile.mkdtemp()
        # Write a couple of small text files
        with open(os.path.join(self.tmpdir, "file_a.py"), "w") as fh:
            fh.write("print('hello')\n" * 5)
        with open(os.path.join(self.tmpdir, "file_b.md"), "w") as fh:
            fh.write("# Title\n\nSome content.\n")
        # A file with unsupported extension (should be ignored)
        with open(os.path.join(self.tmpdir, "ignore.bin"), "w") as fh:
            fh.write("binary stuff")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(self.outdir, ignore_errors=True)

    def test_creates_manifest(self):
        import os

        manifest = prep_directory(self.tmpdir, self.outdir)
        self.assertTrue(os.path.exists(os.path.join(self.outdir, "manifest.json")))
        self.assertEqual(manifest.input_dir, os.path.abspath(self.tmpdir))
        self.assertEqual(manifest.output_dir, os.path.abspath(self.outdir))

    def test_chunks_created(self):
        manifest = prep_directory(self.tmpdir, self.outdir)
        self.assertGreater(len(manifest.chunks), 0)

    def test_unsupported_extension_ignored(self):
        import os

        manifest = prep_directory(self.tmpdir, self.outdir)
        sources = {c["source"] for c in manifest.chunks}
        self.assertFalse(any("ignore.bin" in s for s in sources))

    def test_chunk_files_have_headers(self):
        import os

        prep_directory(self.tmpdir, self.outdir)
        chunk_files = [
            f for f in os.listdir(self.outdir) if f.endswith(".chunk.txt")
        ]
        self.assertGreater(len(chunk_files), 0)
        with open(os.path.join(self.outdir, chunk_files[0])) as fh:
            content = fh.read()
        self.assertIn("# source:", content)

    def test_manifest_to_dict(self):
        import os

        manifest = prep_directory(self.tmpdir, self.outdir)
        d = manifest.to_dict()
        self.assertIn("input_dir", d)
        self.assertIn("output_dir", d)
        self.assertIn("chunks", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
