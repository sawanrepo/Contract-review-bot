from schema import ComplianceOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model, temperature=0.1)

template = PromptTemplate(
    input_variables=["context"],
    template="""
You are a compliance officer.

Check the following contract excerpts for legal compliance under general international standards:
- GDPR or similar privacy regulations
- Basic labor law (e.g., working hours, termination)
- Common financial rules (e.g., late payment clauses)
- Regulatory obligations (e.g., disclosures, anti-bribery)

Mark anything missing, misleading, or illegal.

{context}

Respond in this format:

{{
  "answer": "...",
}}
"""
)

def check_compliance(context: str) -> ComplianceOutput:
    llm = get_llm()
    structured_model = llm.with_structured_output(ComplianceOutput)
    chain = template | structured_model
    return chain.invoke({"context": context})