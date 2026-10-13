from schema import SubqueriesOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model, temperature=0.1)

subquery_prompt = PromptTemplate(
    input_variables=["query"],
    template="""You are a legal document retrieval assistant.

Given the user's question:
"{query}"

Generate 2 to 3 concise, targeted sub-queries to retrieve the most relevant clauses, definitions, and exceptions from a contract.
"""
)

def generate_subqueries(query: str) -> list:
    try:
        llm = get_llm()
        structured_model = llm.with_structured_output(SubqueriesOutput)
        chain = subquery_prompt | structured_model
        res = chain.invoke({"query": query})
        return res.subqueries if hasattr(res, "subqueries") and res.subqueries else []
    except Exception as e:
        print(f"Notice: subquery generation skipped: {e}")
        return []

def get_multiquery_docs(query: str, vectorstore, k: int = 3):
    if vectorstore is None:
        return []

    # Search original query first
    queries_to_search = [query]
    generated = generate_subqueries(query)
    queries_to_search.extend(generated)

    all_docs = []
    seen = set()

    for q in queries_to_search:
        try:
            docs = vectorstore.search(q, k=k)
            for doc in docs:
                page = doc.metadata.get('page_number')
                key = (page, doc.page_content[:100])
                if key not in seen:
                    all_docs.append(doc)
                    seen.add(key)
        except Exception as e:
            print(f"Search warning for query '{q}': {e}")

    return all_docs