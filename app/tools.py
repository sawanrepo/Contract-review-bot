from langchain.tools import StructuredTool
from nodes.compliance_node import check_compliance
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

def compliance_checker(query: str, memory: list = None):
    result = check_compliance(query, memory=memory)
    return result.answer if hasattr(result, "answer") else "Compliance check failed."

def legal_search(query: str):
    tavily = TavilySearchAPIWrapper()
    result = tavily.search(query)
    return result 

compliance_tool = StructuredTool.from_function(
    name="ComplianceChecker",
    func=compliance_checker,
    description="Use this tool to check if a clause or document section complies with GDPR, employment law, or financial regulations."
)

legal_search_tool = StructuredTool.from_function(
    name="LegalSearch",
    func=legal_search,
    description="Use this tool to search for legal information or precedents online."
)

TOOLS = [compliance_tool, legal_search_tool]