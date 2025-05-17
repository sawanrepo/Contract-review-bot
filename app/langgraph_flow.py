from nodes.classifier_node import classify_query
from nodes.rag_node import rag_answer
from nodes.risk_node import analyze_risk
from nodes.summary_node import summarize_contract

def langgraph_flow(query, context):
    intent = classify_query(query)

    if intent == "risk_analysis":
        return analyze_risk(context, query)
    elif intent == "summary":
        return summarize_contract(context)
    elif intent == "query_answer":
        return rag_answer(context, query)
    else:
        return {"error": "Intent not recognized."}