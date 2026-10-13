from schema import GraphState, QueryOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

supervisor_prompt_template = """You are a senior lead legal counsel reviewing findings from your analysis team.

Synthesize the following findings into a clear, unified, and highly structured final advisory response for the user.

User Query:
{query}

{summary_section}
{risk_section}
{rag_section}
"""

def extract_field(obj, field_name, default=None):
    if obj is None:
        return default
    if hasattr(obj, field_name):
        val = getattr(obj, field_name)
        return val if val is not None else default
    if isinstance(obj, dict):
        val = obj.get(field_name)
        return val if val is not None else default
    return default

def extract_pages(obj) -> list:
    pages = extract_field(obj, "page_numbers", [])
    if isinstance(pages, list):
        return [p for p in pages if isinstance(p, int)]
    return []

def supervisor_node(state: GraphState):
    query = state.get("query", "")
    summary_obj = state.get("summary_node")
    rag_obj = state.get("rag_answer_node")
    risk_obj = state.get("risk_analysis_node")

    active_nodes = [obj for obj in (summary_obj, rag_obj, risk_obj) if obj is not None]

    # Fast-path: If only one node produced an output, pass it through directly to save quota and latency
    if len(active_nodes) == 1:
        single_obj = active_nodes[0]
        if isinstance(single_obj, QueryOutput):
            return {**state, "final_answer": single_obj}
        elif isinstance(single_obj, dict) and "answer" in single_obj:
            output = QueryOutput(
                answer=single_obj.get("answer", ""),
                page_numbers=single_obj.get("page_numbers", []),
                risk_level=single_obj.get("risk_level"),
                key_clauses=single_obj.get("key_clauses")
            )
            return {**state, "final_answer": output}

    # If no node ran, return a safe fallback
    if len(active_nodes) == 0:
        return {
            **state,
            "final_answer": QueryOutput(
                answer="No analysis could be completed. Please check your query and contract upload.",
                page_numbers=[]
            )
        }

    # Multiple nodes ran: synthesize their findings
    summary_text = extract_field(summary_obj, "answer", str(summary_obj) if summary_obj else "")
    risk_text = extract_field(risk_obj, "answer", str(risk_obj) if risk_obj else "")
    rag_text = extract_field(rag_obj, "answer", str(rag_obj) if rag_obj else "")

    summary_section = f"### Contract Summary Findings:\n{summary_text}\n" if summary_text else ""
    risk_section = f"### Risk Analysis Findings:\n{risk_text}\n" if risk_text else ""
    rag_section = f"### Specific Q&A Findings:\n{rag_text}\n" if rag_text else ""

    combined_pages = sorted(set(
        extract_pages(summary_obj) +
        extract_pages(rag_obj) +
        extract_pages(risk_obj)
    ))

    # Highest risk level among workers
    risk_levels = [
        extract_field(obj, "risk_level")
        for obj in (summary_obj, rag_obj, risk_obj)
        if extract_field(obj, "risk_level") is not None
    ]
    overall_risk = None
    if "high" in risk_levels:
        overall_risk = "high"
    elif "medium" in risk_levels:
        overall_risk = "medium"
    elif "low" in risk_levels:
        overall_risk = "low"

    try:
        model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
        structured_llm = llm.with_structured_output(QueryOutput)
        prompt = ChatPromptTemplate.from_template(supervisor_prompt_template)
        chain = prompt | structured_llm

        response: QueryOutput = chain.invoke({
            "query": query,
            "summary_section": summary_section,
            "risk_section": risk_section,
            "rag_section": rag_section,
        })

        # Merge extracted pages
        merged_pages = sorted(set(response.page_numbers + combined_pages))
        response.page_numbers = merged_pages
        if overall_risk and not response.risk_level:
            response.risk_level = overall_risk

        return {**state, "final_answer": response}
    except Exception as e:
        print(f"Supervisor node synthesis error: {e}")
        # Fallback to concatenating the text
        fallback_answer = f"{summary_section}\n{risk_section}\n{rag_section}".strip()
        fallback_output = QueryOutput(
            answer=fallback_answer or f"Error in supervisor: {e}",
            page_numbers=combined_pages,
            risk_level=overall_risk
        )
        return {**state, "final_answer": fallback_output}