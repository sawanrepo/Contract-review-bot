import streamlit as st
import os

# Suppress PyTorch/Streamlit watcher conflict on torch.classes
try:
    import torch
    if hasattr(torch, "classes"):
        torch.classes.__path__ = []
except Exception:
    pass


st.set_page_config(
    page_title="Contract Review Bot 🤖",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from vectorstore import VectorStore, get_shared_embeddings
from document_loader import load_documents
from langchain_core.documents import Document
from langgraph_flow import contract_graph
from memory import ChatMemory
from schema import QueryOutput

@st.cache_resource(show_spinner="🧠 Loading embedding model...")
def get_cached_embeddings():
    return get_shared_embeddings("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_cached_embeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore(embedding_model=embedding_model)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "last_uploaded_file_name" not in st.session_state:
    st.session_state.last_uploaded_file_name = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"name": "", "chunks": 0, "pages": 0}
if "ChatMemory" not in st.session_state:
    st.session_state.ChatMemory = ChatMemory()

# --- Sidebar ---
with st.sidebar:
    st.title("⚖️ Contract Review Bot")
    st.caption("AI-powered legal risk analysis, clause Q&A, and summarization.")

    st.markdown("---")
    uploaded_file = st.file_uploader("📎 Upload Contract (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.last_uploaded_file_name:
            with st.spinner("📑 Parsing and indexing document..."):
                try:
                    chunks = load_documents(uploaded_file)
                    st.session_state.vectorstore.clear()
                    st.session_state.vectorstore.add_documents(chunks)

                    pages = sorted({
                        doc.metadata.get("page_number", 1)
                        for doc in chunks
                    })

                    st.session_state.last_uploaded_file_name = uploaded_file.name
                    st.session_state.document_loaded = True
                    st.session_state.doc_stats = {
                        "name": uploaded_file.name,
                        "chunks": len(chunks),
                        "pages": max(pages) if pages else 1
                    }
                    st.session_state.messages = []
                    st.session_state.ChatMemory.clear()
                    st.success("✅ Contract successfully indexed!")
                except Exception as e:
                    st.error(f"Error parsing document: {e}")

    if st.session_state.document_loaded:
        st.markdown("### 📄 Active Document")
        st.info(
            f"**File:** `{st.session_state.doc_stats['name']}`\n\n"
            f"**Pages:** {st.session_state.doc_stats['pages']} | **Chunks:** {st.session_state.doc_stats['chunks']}"
        )

        st.markdown("### ⚡ Quick Actions")
        quick_query = None
        if st.button("📋 Summarize Contract", use_container_width=True):
            quick_query = "Please provide a comprehensive executive summary of this contract."
        if st.button("⚠️ Identify Key Risks", use_container_width=True):
            quick_query = "Analyze this contract and highlight all major legal, financial, and operational risks."
        if st.button("🚪 Termination & Liability", use_container_width=True):
            quick_query = "What are the termination conditions, liability caps, and indemnification obligations in this agreement?"
        if st.button("💳 Financial & Payment Terms", use_container_width=True):
            quick_query = "What are the payment terms, billing schedules, and penalty clauses?"

        st.markdown("---")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.ChatMemory.clear()
            st.rerun()

        # Export Report Button
        if st.session_state.messages:
            report_lines = [
                f"# Contract Review Report: {st.session_state.doc_stats['name']}",
                "",
                "---",
                ""
            ]
            for m in st.session_state.messages:
                role = "User" if m["role"] == "user" else "Assistant"
                report_lines.append(f"### {role}:")
                report_lines.append(m["content"])
                if m.get("page_numbers"):
                    report_lines.append(f"\n*Referenced Pages: {m['page_numbers']}*")
                report_lines.append("\n---\n")

            st.download_button(
                label="📥 Export Review Report",
                data="\n".join(report_lines),
                file_name=f"contract_review_{st.session_state.doc_stats['name']}.md",
                mime="text/markdown",
                use_container_width=True
            )
    else:
        st.info("Upload a contract above to enable analysis tools.")

# --- Main Interface ---
st.title("🤖 Legal Contract Intelligence")

if not st.session_state.document_loaded:
    st.info("👋 Welcome! Please upload a PDF or TXT contract in the sidebar to start reviewing.")
    st.markdown("""
    ### How it works:
    - **Page-Aware Ingestion**: Chunks your contract while preserving exact page numbers.
    - **Multi-Node LangGraph**:
      - 🎯 **Classifier**: Detects whether you need an executive summary, risk analysis, or answers to clause questions.
      - 🔍 **RAG Engine**: Retrieves relevant clauses with citations.
      - ⚠️ **Risk Assessment**: Categorizes clauses by severity (High / Medium / Low).
      - 👔 **Supervisor**: Synthesizes cross-clause findings into cohesive legal advice.
    """)
else:
    # Display message history
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "⚖️"
        with st.chat_message(message["role"], avatar=avatar):
            if message.get("risk_level") == "high":
                st.error("🔴 **HIGH RISK CLAUSES IDENTIFIED**")
            elif message.get("risk_level") == "medium":
                st.warning("🟡 **MEDIUM RISK CLAUSES IDENTIFIED**")
            elif message.get("risk_level") == "low":
                st.info("🟢 **LOW RISK / INFORMATIONAL**")

            st.markdown(message["content"])

            if message.get("page_numbers"):
                pages_str = ", ".join(f"Page {p}" for p in message["page_numbers"])
                st.caption(f"📑 Citations: **{pages_str}**")

            if message.get("key_clauses"):
                with st.expander("📌 Referenced Clauses & Quotes"):
                    for clause in message["key_clauses"]:
                        st.markdown(f"- *\"{clause}\"*")

    # Handle Input from chat or quick action buttons
    user_input = st.chat_input("Ask a question about the contract, check risks, or request a summary...")
    prompt_to_run = user_input or (quick_query if 'quick_query' in locals() and quick_query else None)

    if prompt_to_run:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        st.session_state.ChatMemory.add_message("user", prompt_to_run)

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("🤖 Reviewing contract clauses..."):
                try:
                    result = contract_graph.invoke({
                        "query": prompt_to_run,
                        "context": st.session_state.vectorstore,
                        "memory": st.session_state.ChatMemory.get_messages()
                    })

                    bot_obj = result.get("final_answer")
                    if isinstance(bot_obj, QueryOutput):
                        answer_text = bot_obj.answer
                        page_nums = bot_obj.page_numbers or []
                        risk_lvl = bot_obj.risk_level
                        key_clauses = bot_obj.key_clauses or []
                    elif isinstance(bot_obj, dict):
                        answer_text = bot_obj.get("answer", "Analysis completed.")
                        page_nums = bot_obj.get("page_numbers", [])
                        risk_lvl = bot_obj.get("risk_level")
                        key_clauses = bot_obj.get("key_clauses", [])
                    else:
                        answer_text = str(bot_obj) if bot_obj else "Unable to generate response."
                        page_nums = []
                        risk_lvl = None
                        key_clauses = []

                    if risk_lvl == "high":
                        st.error("🔴 **HIGH RISK CLAUSES IDENTIFIED**")
                    elif risk_lvl == "medium":
                        st.warning("🟡 **MEDIUM RISK CLAUSES IDENTIFIED**")
                    elif risk_lvl == "low":
                        st.info("🟢 **LOW RISK / INFORMATIONAL**")

                    st.markdown(answer_text)

                    if page_nums:
                        pages_str = ", ".join(f"Page {p}" for p in sorted(set(page_nums)))
                        st.caption(f"📑 Citations: **{pages_str}**")

                    if key_clauses:
                        with st.expander("📌 Referenced Clauses & Quotes"):
                            for clause in key_clauses:
                                st.markdown(f"- *\"{clause}\"*")

                    # Record assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "page_numbers": page_nums,
                        "risk_level": risk_lvl,
                        "key_clauses": key_clauses
                    })
                    st.session_state.ChatMemory.add_message("assistant", answer_text)

                except Exception as e:
                    err_msg = f"❌ Error during contract analysis: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    st.session_state.ChatMemory.add_message("assistant", err_msg)