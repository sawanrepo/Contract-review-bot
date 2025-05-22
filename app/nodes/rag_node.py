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

def rag_answer(query: str) -> QueryOutput:
    vs = VectorStore()
    docs = vs.search(query, k= 3)
    context = "\n\n".join(
    f"Page {doc.metadata.get('page_number', 'unknown')}:\n{doc.page_content}" 
    for doc in docs
    )
    chain = template | structured_query_model
    return chain.invoke({"context": context, "query": query}) 