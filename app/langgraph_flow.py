from langgraph.graph import StateGraph, END
from schema import GraphState
from nodes.classifier_node import classify_query
from nodes.rag_node import rag_answer
from nodes.summary_node import summarize_contract
from nodes.risk_node import analyze_risk
from nodes.supervisor_node import supervisor_node

def classifier_node(state: GraphState):
    classification = classify_query(state.get("query", ""))
    intents = classification.intent if classification and classification.intent else ["query_answer"]
    return {"intents": intents}

def rag_node(state: GraphState):
    return {
        "rag_answer_node": rag_answer(
            query=state.get("query", ""),
            context=state.get("context"),
            memory=state.get("memory", [])
        )
    }

def summary_node(state: GraphState):
    return {
        "summary_node": summarize_contract(
            vectorstore=state.get("context"),
            memory=state.get("memory", [])
        )
    }

def risk_node(state: GraphState):
    return {
        "risk_analysis_node": analyze_risk(
            query=state.get("query", ""),
            vectorstore=state.get("context"),
            memory=state.get("memory", [])
        )
    }

def route_intents(state: GraphState):
    branches = []
    for intent in state.get("intents", []):
        if intent == "query_answer" and "rag" not in branches:
            branches.append("rag")
        elif intent == "summary" and "summary" not in branches:
            branches.append("summary")
        elif intent == "risk_analysis" and "risk" not in branches:
            branches.append("risk")

    # Safe fallback if classifier produced nothing matched
    if not branches:
        branches.append("rag")

    return branches

graph = StateGraph(GraphState)

graph.add_node("classify", classifier_node)
graph.add_node("rag", rag_node)
graph.add_node("summary", summary_node)
graph.add_node("risk", risk_node)
graph.add_node("supervisor", supervisor_node)

graph.add_conditional_edges("classify", route_intents, {
    "rag": "rag",
    "summary": "summary",
    "risk": "risk",
})

graph.add_edge("rag", "supervisor")
graph.add_edge("summary", "supervisor")
graph.add_edge("risk", "supervisor")
graph.add_edge("supervisor", END)

graph.set_entry_point("classify")
contract_graph = graph.compile()