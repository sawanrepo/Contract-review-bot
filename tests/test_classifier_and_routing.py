import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from schema import IntentOutput
from nodes.classifier_node import classify_query
from langgraph_flow import route_intents

class TestClassifierAndRouting(unittest.TestCase):
    @patch("nodes.classifier_node.ChatGoogleGenerativeAI")
    def test_classify_query_success(self, mock_llm_cls):
        mock_instance = MagicMock()
        mock_llm_cls.return_value = mock_instance
        
        mock_structured = MagicMock()
        mock_instance.with_structured_output.return_value = mock_structured
        
        # When chain.invoke is called, return IntentOutput
        mock_structured.__or__ = MagicMock()
        
        with patch("nodes.classifier_node.template") as mock_template:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = IntentOutput(intent=["risk_analysis"])
            mock_template.__or__.return_value = mock_chain

            result = classify_query("What are the liability risks?")
            self.assertEqual(result.intent, ["risk_analysis"])

    @patch("nodes.classifier_node.ChatGoogleGenerativeAI")
    def test_classify_query_fallback_on_error(self, mock_llm_cls):
        mock_llm_cls.side_effect = Exception("API connection timed out")
        result = classify_query("Any question")
        self.assertEqual(result.intent, ["query_answer"])

    def test_route_intents_single(self):
        self.assertEqual(route_intents({"intents": ["query_answer"]}), ["rag"])
        self.assertEqual(route_intents({"intents": ["summary"]}), ["summary"])
        self.assertEqual(route_intents({"intents": ["risk_analysis"]}), ["risk"])

    def test_route_intents_multiple(self):
        branches = route_intents({"intents": ["summary", "risk_analysis"]})
        self.assertIn("summary", branches)
        self.assertIn("risk", branches)
        self.assertEqual(len(branches), 2)

    def test_route_intents_empty_fallback(self):
        # Must never return [] to avoid graph stall
        self.assertEqual(route_intents({"intents": []}), ["rag"])
        self.assertEqual(route_intents({}), ["rag"])

    def test_route_intents_unknown_intent_fallback(self):
        self.assertEqual(route_intents({"intents": ["unrecognized_intent"]}), ["rag"])

if __name__ == "__main__":
    unittest.main()
