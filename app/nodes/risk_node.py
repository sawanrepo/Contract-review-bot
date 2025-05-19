from app.schema import QueryOutput
from app.utils import get_multiquery_docs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from app.vectorstore import VectorStore
import os

load_dotenv()
vs = VectorStore()
llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
structured_model = llm.with_structured_output(QueryOutput)
template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a legal risk analysis expert.

Based on the following contract excerpts, identify **any potential risks** (legal, financial, operational, etc.) present in the contract. Be honest, cautious, and prioritize user safety.

Contract Excerpts:
{context}

User Request:
{query}

Provide your analysis and **mention the list of page numbers** where the risky content was found, in this exact JSON format:

{{
  "answer": "...", 
  "page_numbers": [...]
}}
"""
)
def analyze_risk(query: str, vectorstore=vs, k: int = 3) -> QueryOutput:
    docs = get_multiquery_docs(query, vectorstore, k=k)
    if not docs:
        return QueryOutput(answer="No relevant contract excerpts found.", page_numbers=[])
        #later we will use to do web search to give reply based on Country law and other legal aspects mentioning query wasnt found in doc but according to....
    context = "\n\n".join(
        f"Page {doc.metadata.get('page_number', 'unknown')}:\n{doc.page_content}"
        for doc in docs
    )
    chain = template | structured_model
    return chain.invoke({"context": context, "query": query})