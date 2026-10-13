import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from schema import ComplianceOutput
import tools

class TestTools(unittest.TestCase):
    @patch("tools.check_compliance")
    def test_compliance_checker_success(self, mock_check):
        mock_check.return_value = ComplianceOutput(answer="All terms adhere to GDPR standards.")
        res = tools.compliance_checker("Data processing clause")
        self.assertEqual(res, "All terms adhere to GDPR standards.")

    @patch("tools.check_compliance")
    def test_compliance_checker_error_handling(self, mock_check):
        mock_check.side_effect = RuntimeError("Compliance model failed")
        res = tools.compliance_checker("Sample text")
        self.assertIn("Compliance check error", res)

    def test_legal_search_missing_key(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}):
            res = tools.legal_search("california non-compete laws")
            self.assertIn("Tavily API key is not configured", res)

    @patch("tools.TavilySearchResults")
    def test_legal_search_success(self, mock_tavily_cls):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "valid-test-key"}):
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = [{"url": "https://example.com", "content": "Law precedent"}]
            mock_tavily_cls.return_value = mock_instance

            res = tools.legal_search("contract law")
            self.assertIn("Law precedent", res)

    @patch("tools.TavilySearchResults")
    def test_legal_search_api_failure(self, mock_tavily_cls):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "valid-test-key"}):
            mock_instance = MagicMock()
            mock_instance.invoke.side_effect = Exception("Rate limit on Tavily")
            mock_tavily_cls.return_value = mock_instance

            res = tools.legal_search("contract law")
            self.assertIn("Legal search encountered an error", res)

if __name__ == "__main__":
    unittest.main()
