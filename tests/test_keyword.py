"""
tests/test_keyword.py — Tests for BM25 keyword search and hybrid search.
"""

import unittest
from fttf_holo_rag.keyword import KeywordIndex, tokenize
from fttf_holo_rag.search import (
    KeywordResult, KeywordSearchResult,
    HybridResult, HybridSearchResult,
)


class TestTokenize(unittest.TestCase):
    def test_basic(self):
        tokens = tokenize("hello world")
        self.assertEqual(tokens, ["hello", "world"])

    def test_stop_words_removed(self):
        tokens = tokenize("this is a test of the system")
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("a", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("of", tokens)
        self.assertIn("test", tokens)
        self.assertIn("system", tokens)

    def test_camel_case_split(self):
        tokens = tokenize("getData")
        self.assertIn("get", tokens)
        self.assertIn("data", tokens)

    def test_snake_case_split(self):
        tokens = tokenize("get_data_from_api")
        self.assertIn("get", tokens)
        self.assertIn("data", tokens)
        self.assertIn("api", tokens)

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_numbers_preserved(self):
        tokens = tokenize("error 404 not found")
        self.assertIn("404", tokens)
        self.assertIn("error", tokens)
        self.assertIn("found", tokens)

    def test_mixed_code(self):
        tokens = tokenize("def handleWebSocketAuth(request):")
        self.assertIn("def", tokens)
        self.assertIn("handle", tokens)
        self.assertIn("web", tokens)
        self.assertIn("socket", tokens)
        self.assertIn("auth", tokens)
        self.assertIn("request", tokens)


class TestKeywordIndex(unittest.TestCase):
    def setUp(self):
        self.idx = KeywordIndex()
        self.idx.add("doc1", "websocket authentication handler for secure connections")
        self.idx.add("doc2", "database connection pool management")
        self.idx.add("doc3", "authentication middleware with JWT token validation")
        self.idx.add("doc4", "REST API endpoint for user login authentication")
        self.idx.add("doc5", "logging framework with structured output")

    def test_doc_count(self):
        self.assertEqual(self.idx.doc_count, 5)

    def test_vocab_size(self):
        self.assertGreater(self.idx.vocab_size, 0)

    def test_search_single_term(self):
        results = self.idx.search("authentication")
        self.assertGreater(len(results), 0)
        doc_ids = [r[0] for r in results]
        self.assertIn("doc1", doc_ids)
        self.assertIn("doc3", doc_ids)
        self.assertIn("doc4", doc_ids)

    def test_search_multiple_terms(self):
        results = self.idx.search("websocket authentication")
        self.assertGreater(len(results), 0)
        # doc1 has both terms, should rank highest
        self.assertEqual(results[0][0], "doc1")

    def test_search_no_match(self):
        results = self.idx.search("xyzzyplugh")
        self.assertEqual(len(results), 0)

    def test_search_empty_query(self):
        results = self.idx.search("")
        self.assertEqual(len(results), 0)

    def test_search_normalized(self):
        results = self.idx.search_normalized("authentication")
        self.assertGreater(len(results), 0)
        # Top result should have score 1.0
        self.assertAlmostEqual(results[0][1], 1.0)
        # All scores should be between 0 and 1
        for _, score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_top_k_limits(self):
        results = self.idx.search("authentication", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_idempotent_add(self):
        self.idx.add("doc1", "totally different content")
        # Should still have 5 docs (doc1 not re-added)
        self.assertEqual(self.idx.doc_count, 5)

    def test_scores_are_positive(self):
        results = self.idx.search("authentication")
        for _, score in results:
            self.assertGreater(score, 0.0)


class TestKeywordSearchResult(unittest.TestCase):
    def test_to_text_with_results(self):
        result = KeywordSearchResult(
            query="auth",
            total_indexed=100,
            results=[
                KeywordResult(
                    doc_id="d1",
                    filepath="/proj/auth.py",
                    project="proj",
                    score=0.92,
                    snippet="def authenticate(user)...",
                ),
            ],
        )
        text = result.to_text()
        self.assertIn("KEYWORD SEARCH", text)
        self.assertIn("100 docs", text)
        self.assertIn("auth.py", text)
        self.assertIn("92.0%", text)

    def test_to_text_empty(self):
        result = KeywordSearchResult(query="nope", total_indexed=50)
        text = result.to_text()
        self.assertIn("no matches", text)


class TestHybridSearchResult(unittest.TestCase):
    def test_to_text(self):
        result = HybridSearchResult(
            query="websocket",
            alpha=0.6,
            results=[
                HybridResult(
                    doc_id="f1",
                    filepath="/proj/ws.py",
                    project="proj",
                    vector_score=0.8,
                    keyword_score=0.6,
                    combined_score=0.72,
                    snippet="class WebSocketHandler...",
                ),
            ],
        )
        text = result.to_text()
        self.assertIn("HYBRID SEARCH", text)
        self.assertIn("alpha=0.60", text)
        self.assertIn("ws.py", text)
        self.assertIn("72.0%", text)

    def test_to_text_empty(self):
        result = HybridSearchResult(query="nothing", alpha=0.5)
        text = result.to_text()
        self.assertIn("no matches", text)


class TestKeywordIndexImport(unittest.TestCase):
    def test_import_from_top_level(self):
        from fttf_holo_rag import KeywordIndex
        self.assertIsNotNone(KeywordIndex)


if __name__ == '__main__':
    unittest.main()
