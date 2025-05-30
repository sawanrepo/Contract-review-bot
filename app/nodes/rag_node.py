from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os
from tools import TOOLS

load_dotenv()

llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))

template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a legal document assistant.

Based on the following contract excerpts, answer the user's query truthfully.

{context}

User Query:
{query}

Please provide the answer and specify the list of page numbers of the document excerpts where you found the answer, in this JSON format:
If you're unsure or it requires compliance check or external legal info, call the appropriate tool.

{{
  "answer": "...",
  "page_numbers": [...]
}}
"""
)

def rag_answer(query: str, context, memory: list = None):
    docs = context.search(query, k=3)
    context_text = "\n\n".join(
        f"Page {doc.metadata.get('page_number', 'unknown')}:\n{doc.page_content}"
        for doc in docs
    )

    memory_context = ""
    if memory:
        memory_context = "\n\nChat History:\n" + "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in memory[-5:]
        )

    full_context = f"{memory_context}\n\nContract Excerpts:\n{context_text}"
    prompt_text = template.format(context=full_context, query=query)

    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent.invoke({"input": prompt_text})