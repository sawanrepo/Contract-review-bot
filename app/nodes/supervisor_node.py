# will be updated to llm later now just a simple function to return the final answer.

from schema import GraphState

def supervisor_node(state: GraphState):
    query = state["query"]
    summary = state.get("summary_node")
    rag_answer = state.get("rag_answer_node")
    risk_analysis = state.get("risk_analysis_node")

    parts = []

    if summary:
        parts.append("Here is a brief summary of the document:")
        parts.append(summary.answer if hasattr(summary, 'answer') else str(summary)) 

    if risk_analysis:
        parts.append("Identified potential risks:")
        parts.append(risk_analysis.answer if hasattr(risk_analysis, 'answer') else str(risk_analysis)) 

    if rag_answer:
        parts.append("Response to your query:")
        parts.append(rag_answer.answer if hasattr(rag_answer, 'answer') else str(rag_answer))  

    refined = f"Dear user, based on your legal inquiry — \"{query}\" — please find the following:"
    refined += "\n\n" + "\n\n".join(parts)
    refined += "\n\nIf you have any further concerns or need clarification on a specific clause, feel free to ask."

    return {"final_answer": refined.strip(), **state} 