from langchain.tools import StructuredTool
from nodes.compliance_node import check_compliance
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

def compliance_checker(context: str) -> str:
    try:
        result = check_compliance(context)
        return result.answer if hasattr(result, "answer") else "Compliance check completed."
    except Exception as e:
        return f"Compliance check error: {str(e)}"

def legal_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or "your_key" in api_key:
        return "Tavily API key is not configured. External legal search is unavailable."
    try:
        tavily = TavilySearchResults(max_results=3)
        return str(tavily.invoke({"query": query}))
    except Exception as e:
        return f"Legal search encountered an error: {str(e)}"

compliance_tool = StructuredTool.from_function(
    name="ComplianceChecker",
    func=compliance_checker,
    description="Pass contract excerpts and user query as a string to check compliance under GDPR, labor laws, and financial regulations."
)

legal_search_tool = StructuredTool.from_function(
    name="LegalSearch",
    func=legal_search,
    description="Use this tool to search for external legal standards, case law, or regulations online."
)

TOOLS = [compliance_tool, legal_search_tool]