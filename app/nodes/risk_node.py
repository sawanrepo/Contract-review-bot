from utils import get_multiquery_docs
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from schema import QueryOutput
import os

load_dotenv()

risk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior legal risk analyst.
Your task is to identify and assess potential legal, financial, operational, and regulatory risks in the provided contract excerpts.

Assessment Guidelines:
1. **Identify Red Flags**: Unlimited liability, one-sided indemnity, unilateral termination, missing dispute resolution, onerous warranties, IP assignment traps, or non-compete restrictions.
2. **Assign Severity**:
   - "high": Fatal or existential risks (e.g. uncapped liability, waiver of fundamental rights, immediate termination without cause).
   - "medium": Asymmetric obligations, ambiguity in milestones/payment, harsh penalty interest, strict indemnities.
   - "low": Minor ambiguities or missing standard operational clauses.
   - "none": No notable risks detected.
3. **Structured Recommendations**: Clearly state the risk, its impact, and recommended negotiation revisions.
4. **Citations**: List 1-based page numbers in `page_numbers` and exact problematic clause quotes in `key_clauses`."""),
    ("human", """Conversation Context:
{chat_history}

Contract Excerpts:
{context}

User Query / Focus Area:
{query}
""")
])

def analyze_risk(query: str, vectorstore=None, memory: list = None) -> QueryOutput:
    if vectorstore is None:
        return QueryOutput(
            answer="No document has been uploaded yet. Please upload a contract first.",
            page_numbers=[],
            risk_level="none"
        )

    docs = get_multiquery_docs(query, vectorstore, k=3)
    if not docs:
        # Fallback to general vector search
        docs = vectorstore.search(query, k=4)

    if not docs:
        return QueryOutput(
            answer="Could not retrieve relevant clauses from the uploaded document to perform risk analysis.",
            page_numbers=[],
            risk_level="none"
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
        chain = risk_prompt | structured_model

        result: QueryOutput = chain.invoke({
            "chat_history": history_text,
            "context": context_text,
            "query": query
        })

        # Ground page citations against retrieved evidence to eliminate hallucinated page numbers
        if result.page_numbers and retrieved_pages:
            valid_pages = [p for p in result.page_numbers if p in retrieved_pages]
            result.page_numbers = sorted(set(valid_pages if valid_pages else retrieved_pages))
        elif not result.page_numbers and retrieved_pages:
            result.page_numbers = retrieved_pages

        return result
    except Exception as e:
        print(f"Risk analysis node error: {e}")
        return QueryOutput(
            answer=f"An error occurred during risk analysis: {str(e)}",
            page_numbers=retrieved_pages,
            risk_level="medium"
        )