from tools import TOOLS
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from utils import get_multiquery_docs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from vectorstore import VectorStore
import os

load_dotenv()
vs = VectorStore()
llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a legal risk analysis expert.

Based on the following contract excerpts, identify **any potential risks** (legal, financial, operational, etc.) present in the contract. Be honest, cautious, and prioritize user safety.
If you're unsure or it requires compliance check or external legal info, call the appropriate tool.
while calling compliance_tool, pass the context Excerpts and user query as a single string.

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
def analyze_risk(query: str, vectorstore=vs, memory:list = None):
    docs = get_multiquery_docs(query, vectorstore, 3)
    context = "\n\n".join(
        f"Page {doc.metadata.get('page_number', 'unknown')}:\n{doc.page_content}"
        for doc in docs
    )
    memory_context = ""
    if memory:
        memory_context = "\n\nChat History:\n" + "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in memory[-5:]  # last 5 messages as memory context.
        )
    full_context = f"{memory_context}\n\nContract Excerpts:\n{context}"
    prompt_text = template.format(context=full_context, query=query)
    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent.invoke({"input": prompt_text})