from langchain.tools import StructuredTool
from nodes.compliance_node import check_compliance
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

def compliance_checker(context: str):
    result = check_compliance(context)
    return result.answer if hasattr(result, "answer") else "Compliance check failed."

def legal_search(query: str):
    tavily = TavilySearchAPIWrapper()
    result = tavily.search(query)
    return result 

compliance_tool = StructuredTool.from_function(
    name="ComplianceChecker",
    func=compliance_checker,
    description="Pass a single string combining contract excerpts and user query to check compliance under GDPR, labor laws, and financial regulations."
)

legal_search_tool = StructuredTool.from_function(
    name="LegalSearch",
    func=legal_search,
    description="Use this tool to search for legal information or precedents online."
)

TOOLS = [compliance_tool, legal_search_tool]