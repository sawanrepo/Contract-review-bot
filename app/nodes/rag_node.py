from schema import QueryOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from vectorstore import VectorStore
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model =os.getenv("MODEL_NAME"))
structured_query_model = llm.with_structured_output(QueryOutput)

template = PromptTemplate(
    input_variables=["context", "query"],
    template="""You are a legal document assistant.

Based on the following contract excerpts, answer the user's query truthfully.

Contract Excerpts:
{context}

User Query:
{query}

Please provide the answer and specify the list of page numbers of the document excerpts where you found the answer, in this JSON format:

{{
  "answer": "...",
  "page_numbers": [...]
}}
"""

)

def rag_answer(query: str, context: VectorStore,memory: list = None) -> QueryOutput:
    docs = context.search(query, k=3)
    context_text = "\n\n".join(
        f"Page {doc.metadata.get('page_number', 'unknown')}:\n{doc.page_content}"
        for doc in docs
    )
    memory_context = ""
    if memory:
        memory_context = "\n\nChat History:\n" + "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in memory[-5:] #last 5 messages as memory context.
        )
    full_context = f"{memory_context}\n\nContract Excerpts:\n{context_text}"
    chain = template | structured_query_model
    return chain.invoke({"context": full_context, "query": query})