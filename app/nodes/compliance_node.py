from schema import ComplianceOutput
from utils import get_multiquery_docs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from vectorstore import VectorStore
import os

load_dotenv()
vs = VectorStore()
llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
structured_model = llm.with_structured_output(ComplianceOutput)

template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a compliance officer.

Check the following contract excerpts for legal compliance under general international standards:
- GDPR or similar privacy regulations
- Basic labor law (e.g., working hours, termination)
- Common financial rules (e.g., late payment clauses)
- Regulatory obligations (e.g., disclosures, anti-bribery)

Mark anything missing, misleading, or illegal.

{context}

User Request:
{query}

Respond in this format:

{{
  "answer": "...",
}}
"""
)

def check_compliance(query: str, vectorstore=vs, memory: list = None) -> ComplianceOutput:
    docs = get_multiquery_docs(query, vectorstore, k=3)
    if not docs:
        return ComplianceOutput(answer="No relevant contract excerpts found.")

    context = "\n\n".join(
        f"Page {doc.metadata.get('page_number', 'unknown')}:\n{doc.page_content}"
        for doc in docs
    )
    
    memory_context = ""
    if memory:
        memory_context = "\n\nChat History:\n" + "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in memory[-5:]
        )
    full_context = f"\nMemory Context:\n{memory_context}\n\nContract Excerpts:\n{context}"
    chain = template | structured_model
    return chain.invoke({"context": full_context, "query": query})