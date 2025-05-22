from nodes.classifier_node import classify_query
from nodes.rag_node import rag_answer
from nodes.risk_node import analyze_risk
from nodes.summary_node import summarize_contract

def langgraph_flow(query, context):
    intent_obj = classify_query(query)
    intents = intent_obj.intent if hasattr(intent_obj, "intent") else []

    result = {}

    if "risk_analysis" in intents:
        result["risk_analysis"] = analyze_risk(query, context)
    
    if "summary" in intents:
        result["summary"] = summarize_contract(context)
    
    if "query_answer" in intents:
        result["query_answer"] = rag_answer(query, context)

    if not result:
        result = {"error": "Intent not recognized."}

    return result