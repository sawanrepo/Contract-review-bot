from schema import QueryOutput
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert legal counsel specialized in contract abstraction and executive summarization.

Analyze the provided contract excerpts and deliver a structured executive summary with the following sections:
1. **Contract Overview & Parties**: Who is entering into the agreement, the effective date, and primary purpose.
2. **Key Obligations & Scope**: Core deliverables, milestones, or services provided.
3. **Financial Terms**: Fees, payment schedules, late penalties, or price adjustment mechanisms.
4. **Term & Termination**: Duration, renewal terms, termination for convenience vs cause, and notice periods.
5. **Liability & Governing Law**: Liability caps, indemnities, dispute resolution venue, and governing jurisdiction.

Cite exact 1-based page numbers in `page_numbers` for the clauses referenced."""),
    ("human", """Contract Excerpts:
{context}

Please provide the executive summary.
""")
])

def summarize_contract(vectorstore=None, memory: list = None) -> QueryOutput:
    if vectorstore is None:
        return QueryOutput(
            answer="No document has been uploaded yet. Please upload a contract first.",
            page_numbers=[]
        )

    # Use all chunks if available, or sample representative chunks
    all_docs = vectorstore.get_all_documents() if hasattr(vectorstore, "get_all_documents") else []
    if not all_docs:
        all_docs = vectorstore.search("agreement terms parties liability termination", k=10)

    if not all_docs:
        return QueryOutput(answer="No relevant content found for summarization.", page_numbers=[])

    # If document has many chunks, sample representative chunks across the document
    if len(all_docs) > 16:
        # Preamble/first chunks
        selected_docs = all_docs[:6]
        # Middle chunks
        step = max(1, (len(all_docs) - 8) // 5)
        for i in range(6, len(all_docs) - 3, step):
            selected_docs.append(all_docs[i])
        # Trailing chunks
        selected_docs.extend(all_docs[-3:])
    else:
        selected_docs = all_docs

    context_text = "\n\n".join(
        f"[Page {doc.metadata.get('page_number', 'unknown')}]:\n{doc.page_content.strip()}"
        for doc in selected_docs
    )

    page_numbers = sorted({
        doc.metadata.get("page_number")
        for doc in selected_docs
        if doc.metadata.get("page_number") is not None
    })

    try:
        model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
        structured_model = llm.with_structured_output(QueryOutput)
        chain = summary_prompt | structured_model

        result: QueryOutput = chain.invoke({"context": context_text})
        if not result.page_numbers and page_numbers:
            result.page_numbers = page_numbers
        return result
    except Exception as e:
        print(f"Summary node error: {e}")
        return QueryOutput(
            answer=f"An error occurred while generating the summary: {str(e)}",
            page_numbers=page_numbers
        )