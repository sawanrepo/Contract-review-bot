from schema import QueryOutput
from vectorstore import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME"))
structured_model = llm.with_structured_output(QueryOutput)
vs = VectorStore()

template = PromptTemplate(
    input_variables=["context"],
    template="""
You are a legal contract summarization expert.

Given the following contract excerpts, generate a concise but complete summary. Focus on important legal, financial, and operational terms. Maintain neutrality and clarity.

Contract Excerpts:
{context}

Respond with a well-structured summary and list of page numbers where you found the key information in this JSON format:

{{
  "answer": "...",
  "page_numbers": [...]
}}
"""
)

def summarize_contract(vectorstore=vs,memory:list = None ) -> QueryOutput:
    docs = vectorstore.search("summarize the contract", k=10)  # Fetching more documents for a comprehensive summary
    if not docs:
        return QueryOutput(answer="No relevant content found for summarization.", page_numbers=[])

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

    chain = template | structured_model
    return chain.invoke({"context": full_context})