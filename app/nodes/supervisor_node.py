from schema import GraphState, QueryOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os

llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
structured_llm = llm.with_structured_output(QueryOutput)

supervisor_prompt_template = """
You are a legal assistant.

Given the following findings, generate a clear, helpful final response for the user’s legal query.

User Query:
{query}

{summary_section}
{risk_section}
{rag_section}

Only return this JSON format:

{{
  "answer": "your response here",
  "page_numbers": []
}}
"""

def prepare_supervisor_prompt_inputs(query, summary=None, risk_analysis=None, rag_answer=None):
    summary_section = f"Summary of the contract:\n{summary}\n" if summary else ""
    risk_section = f"Identified risks:\n{risk_analysis}\n" if risk_analysis else ""
    rag_section = f"Answer to the user’s question:\n{rag_answer}\n" if rag_answer else ""

    return {
        "query": query,
        "summary_section": summary_section,
        "risk_section": risk_section,
        "rag_section": rag_section,
    }

supervisor_prompt = ChatPromptTemplate.from_template(supervisor_prompt_template)

supervisor_chain = supervisor_prompt | structured_llm

def extract_page_numbers(obj):
    try:
        return sorted(set(obj.page_numbers)) if hasattr(obj, "page_numbers") and obj.page_numbers else []
    except:
        return []

def supervisor_node(state: GraphState):
    query = state["query"]
    summary = getattr(state.get("summary_node"), "answer", None) or state.get("summary_node")
    rag_answer = getattr(state.get("rag_answer_node"), "answer", None) or state.get("rag_answer_node")
    risk_analysis = getattr(state.get("risk_analysis_node"), "answer", None) or state.get("risk_analysis_node")

    input_data = prepare_supervisor_prompt_inputs(query, summary, risk_analysis, rag_answer)
    response = supervisor_chain.invoke(input_data)

    combined_pages = (
        extract_page_numbers(state.get("summary_node")) +
        extract_page_numbers(state.get("rag_answer_node")) +
        extract_page_numbers(state.get("risk_analysis_node"))
    )
    response.page_numbers = sorted(set(combined_pages))

    return {
        "final_answer": response,
        **state
    }