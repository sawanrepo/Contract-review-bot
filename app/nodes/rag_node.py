from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from schema import QueryOutput
import os

load_dotenv()

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise, professional legal contract assistant.
Your job is to answer the user's question based strictly on the provided contract excerpts and conversation context.

Guidelines:
1. State the answer clearly and objectively. If the contract explicitly addresses the issue, reference the specifics.
2. If the contract does not mention the requested topic, clearly state that the information was not found in the uploaded document.
3. In `page_numbers`, list all 1-based page numbers where you found the relevant clauses.
4. In `key_clauses`, cite brief exact quotes or clause identifiers (e.g., "Section 12.1", "Indemnity clause") that support your answer."""),
    ("human", """Conversation History:
{chat_history}

Contract Excerpts:
{context}

User Query:
{query}
""")
])

def rag_answer(query: str, context, memory: list = None) -> QueryOutput:
    if context is None:
        return QueryOutput(
            answer="No document has been uploaded yet. Please upload a contract first.",
            page_numbers=[]
        )

    docs = context.search(query, k=4)
    if not docs:
        return QueryOutput(
            answer="I could not find any relevant sections in the uploaded contract matching your query.",
            page_numbers=[]
        )

    retrieved_pages = sorted({
        doc.metadata.get("page_number")
        for doc in docs
        if doc.metadata.get("page_number") is not None
    })

    context_text = "\n\n".join(
        f"[Page {doc.metadata.get('page_number', 'unknown')}]:\n{doc.page_content.strip()}"
        for doc in docs
    )

    history_text = "No previous history."
    if memory:
        recent = memory[-5:]
        history_text = "\n".join(
            f"{m.get('role', 'User').capitalize()}: {m.get('content', '')}"
            for m in recent if isinstance(m, dict)
        )

    try:
        model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
        structured_model = llm.with_structured_output(QueryOutput)
        chain = rag_prompt | structured_model

        result: QueryOutput = chain.invoke({
            "chat_history": history_text,
            "context": context_text,
            "query": query
        })

        # Ensure page numbers are populated
        if not result.page_numbers and retrieved_pages:
            result.page_numbers = retrieved_pages

        return result
    except Exception as e:
        print(f"RAG node error: {e}")
        return QueryOutput(
            answer=f"An error occurred while analyzing the contract: {str(e)}",
            page_numbers=retrieved_pages
        )