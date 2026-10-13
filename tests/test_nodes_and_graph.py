import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from schema import QueryOutput
from langchain_core.documents import Document
from nodes.rag_node import rag_answer
from nodes.risk_node import analyze_risk
from nodes.summary_node import summarize_contract
from nodes.supervisor_node import supervisor_node

class DummyVS:
    def __init__(self, docs=None):
        self._docs = docs or []

    def search(self, q, k=4):
        return self._docs[:k]

    def get_all_documents(self):
        return list(self._docs)

class TestNodesAndGraph(unittest.TestCase):
    def test_rag_missing_context(self):
        result = rag_answer("What is the clause?", context=None)
        self.assertIsInstance(result, QueryOutput)
        self.assertIn("No document has been uploaded", result.answer)
        self.assertEqual(result.page_numbers, [])

    def test_rag_empty_search(self):
        empty_vs = DummyVS([])
        result = rag_answer("What is the clause?", context=empty_vs)
        self.assertIsInstance(result, QueryOutput)
        self.assertIn("could not find any relevant sections", result.answer)

    @patch("nodes.rag_node.ChatGoogleGenerativeAI")
    def test_rag_grounded_citations(self, mock_llm_cls):
        mock_instance = MagicMock()
        mock_llm_cls.return_value = mock_instance

        # LLM tries to return page 99 (hallucinated) and page 2 (real)
        fake_output = QueryOutput(
            answer="The governing law is New York.",
            page_numbers=[2, 99]
        )
        with patch("nodes.rag_node.rag_prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = fake_output
            mock_prompt.__or__.return_value = mock_chain

            vs = DummyVS([
                Document(page_content="Governing law is NY.", metadata={"page_number": 2}),
                Document(page_content="Notice clause.", metadata={"page_number": 3})
            ])
            result = rag_answer("Governing law?", context=vs)
            # Page 99 must be filtered out! Only page 2 is valid evidence
            self.assertEqual(result.page_numbers, [2])

    def test_risk_missing_context(self):
        result = analyze_risk("Check risks", vectorstore=None)
        self.assertIsInstance(result, QueryOutput)
        self.assertEqual(result.risk_level, "none")

    def test_summary_missing_context(self):
        result = summarize_contract(vectorstore=None)
        self.assertIsInstance(result, QueryOutput)
        self.assertIn("No document has been uploaded", result.answer)

    def test_supervisor_single_node_passthrough(self):
        # When only one node ran, supervisor should pass it through without an extra LLM call
        rag_out = QueryOutput(answer="Direct RAG answer", page_numbers=[1], risk_level="low")
        state = {
            "query": "What is term?",
            "rag_answer_node": rag_out,
            "summary_node": None,
            "risk_analysis_node": None
        }
        res = supervisor_node(state)
        self.assertEqual(res["final_answer"].answer, "Direct RAG answer")
        self.assertEqual(res["final_answer"].risk_level, "low")
        self.assertEqual(res["final_answer"].page_numbers, [1])

    def test_supervisor_empty_state_fallback(self):
        state = {
            "query": "hello",
            "rag_answer_node": None,
            "summary_node": None,
            "risk_analysis_node": None
        }
        res = supervisor_node(state)
        self.assertIsInstance(res["final_answer"], QueryOutput)
        self.assertIn("No analysis could be completed", res["final_answer"].answer)

    @patch("nodes.supervisor_node.ChatGoogleGenerativeAI")
    def test_supervisor_multi_node_synthesis(self, mock_llm_cls):
        mock_instance = MagicMock()
        mock_llm_cls.return_value = mock_instance

        synthesized_out = QueryOutput(
            answer="Synthesized summary and risk analysis.",
            page_numbers=[1]
        )
        with patch("nodes.supervisor_node.ChatPromptTemplate") as mock_tmpl_cls:
            mock_tmpl = MagicMock()
            mock_tmpl_cls.from_template.return_value = mock_tmpl
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = synthesized_out
            mock_tmpl.__or__.return_value = mock_chain

            sum_out = QueryOutput(answer="Summary text", page_numbers=[1])
            risk_out = QueryOutput(answer="Risk text", page_numbers=[3], risk_level="high")
            state = {
                "query": "Summarize and check risks",
                "rag_answer_node": None,
                "summary_node": sum_out,
                "risk_analysis_node": risk_out
            }
            res = supervisor_node(state)
            self.assertEqual(res["final_answer"].risk_level, "high")
            # Combined pages from workers (1 and 3) must be merged
            self.assertIn(1, res["final_answer"].page_numbers)
            self.assertIn(3, res["final_answer"].page_numbers)

if __name__ == "__main__":
    unittest.main()
