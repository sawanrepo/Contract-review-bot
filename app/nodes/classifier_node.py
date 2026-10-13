from schema import IntentOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

template = PromptTemplate(
    input_variables=["input"],
    template="""You are an intent classifier for a contract review assistant.

Classify the user's request into one or more of these intents:
- "risk_analysis": when the user wants to identify risks, liabilities, red flags, or dangerous clauses.
- "summary": when the user wants a summary, overview, key points, or high-level walkthrough.
- "query_answer": when the user asks a specific question about a clause, general contract terms, greeting, or any other query.

If uncertain, default to ["query_answer"].

User Request: {input}
"""
)

def classify_query(query: str) -> IntentOutput:
    try:
        model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.0)
        structured_model = llm.with_structured_output(IntentOutput)
        chain = template | structured_model
        response = chain.invoke({"input": query})
        if not response or not response.intent:
            return IntentOutput(intent=["query_answer"])
        return response
    except Exception as e:
        print(f"Classifier error (falling back to query_answer): {e}")
        return IntentOutput(intent=["query_answer"])