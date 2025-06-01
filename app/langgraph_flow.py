from langgraph.graph import StateGraph, END
from schema import GraphState
from nodes.classifier_node import classify_query
from nodes.rag_node import rag_answer
from nodes.summary_node import summarize_contract
from nodes.risk_node import analyze_risk
from nodes.supervisor_node import supervisor_node

def classifier_node(state: GraphState):
    intents = classify_query(state["query"]).intent
    return {"intents": intents or []}

def rag_node(state: GraphState):
    return {"rag_answer_node": rag_answer(state["query"], state["context"], memory=state["memory"])}

def summary_node(state: GraphState):
    return {"summary_node": summarize_contract(state["context"], memory=state["memory"])}

def risk_node(state: GraphState):
    return {"risk_analysis_node": analyze_risk(state["query"], state["context"], memory=state["memory"])}

# def output_node(state: GraphState):
#     return state["final_answer"]

def route_intents(state: GraphState):
    branches = []
    for intent in state["intents"]:
        if intent == "query_answer":
            branches.append("rag")
        elif intent == "summary":
            branches.append("summary")
        elif intent == "risk_analysis":
            branches.append("risk")
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
if __name__ == "__main__":
    graph.visualize("contract_graph.png", format="png", show=True)