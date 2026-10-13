import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from vectorstore import VectorStore, get_shared_embeddings

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.fake_embeddings = FakeEmbeddings(size=64)
        self.vs = VectorStore(embedding_model=self.fake_embeddings)

    def test_empty_vectorstore_search(self):
        results = self.vs.search("Any query", k=3)
        self.assertEqual(results, [])

    def test_add_and_search_documents(self):
        docs = [
            Document(page_content="The Supplier shall be liable for all direct damages.", metadata={"page_number": 1}),
            Document(page_content="The agreement terminates upon 30 days written notice.", metadata={"page_number": 2}),
        ]
        self.vs.add_documents(docs)
        self.assertEqual(len(self.vs.get_all_documents()), 2)

        results = self.vs.search("termination notice", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("terminates", results[0].page_content)
        self.assertEqual(results[0].metadata.get("page_number"), 2)

    def test_clear_vectorstore(self):
        docs = [Document(page_content="Confidential information clause.", metadata={"page_number": 1})]
        self.vs.add_documents(docs)
        self.assertEqual(len(self.vs.get_all_documents()), 1)

        self.vs.clear()
        self.assertIsNone(self.vs.vectorstore)
        self.assertEqual(len(self.vs.get_all_documents()), 0)
        self.assertEqual(self.vs.search("Confidential"), [])

    def test_embedding_cache_by_model_name(self):
        # Verify get_shared_embeddings caches separately by model name
        import vectorstore
        vectorstore._global_embeddings_cache.clear()
        
        # Inject fake embeddings into cache to verify lookup
        dummy1 = FakeEmbeddings(size=32)
        dummy2 = FakeEmbeddings(size=64)
        vectorstore._global_embeddings_cache["model_a"] = dummy1
        vectorstore._global_embeddings_cache["model_b"] = dummy2

        self.assertIs(vectorstore.get_shared_embeddings("model_a"), dummy1)
        self.assertIs(vectorstore.get_shared_embeddings("model_b"), dummy2)

if __name__ == "__main__":
    unittest.main()
