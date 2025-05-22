from schema import SubqueriesOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
structured_model = llm.with_structured_output(SubqueriesOutput)

subquery_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a legal document assistant.

Given the user's question:
"{query}"

Generate 2 to 3 focused sub-questions that can help retrieve relevant clauses from a legal contract.

Respond strictly in this JSON format:

{{
  "subqueries": [
    "...",
    "...",
    "..."
  ]
}}
"""
)

def generate_subqueries(query: str) -> SubqueriesOutput:
    chain = subquery_prompt | structured_model
    return chain.invoke({"query": query}).subqueries

def get_multiquery_docs(query: str, vectorstore, k: int = 3):
    subqueries = generate_subqueries(query)
    all_docs = []
    seen = set()
    
    for subquery in subqueries:
        docs = vectorstore.search(subquery, k=k)
        for doc in docs:
            key = (doc.metadata.get('page_number'), doc.page_content)
            if key not in seen:
                all_docs.append(doc)
                seen.add(key)
    
    return all_docs


if __name__ == "__main__":
    query = "What are the penalties for breach of contract?"
    subqueries = generate_subqueries(query)
    print(subqueries)