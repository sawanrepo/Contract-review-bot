import streamlit as st
from vectorstore import VectorStore
from document_loader import load_documents
from langchain.schema import Document
from langgraph_flow import langgraph_flow

st.set_page_config(page_title="Contract Review Bot 🤖", layout="wide")
st.title("📄 Contract Review Bot")
vs = VectorStore()

uploaded_file = st.file_uploader("Upload a contract (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("Processing document..."):
        chunks = load_documents(uploaded_file)
        if chunks and not isinstance(chunks[0], Document):
            chunks = [Document(page_content=getattr(doc, 'page_content', str(doc))) for doc in chunks]

        vs.add_documents(chunks)
        st.success("✅ Document processed and embedded!")

    st.markdown("---")
    
    query = st.text_input("🔎 Ask a question, or request a summary/risk analysis of any clause.")

    if st.button("🚀 Run Analysis"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Thinking..."):
                result = langgraph_flow(query=query, context=vs)

                if "error" in result:
                    st.error(result["error"])
                else:
                    if "summary" in result:
                        st.markdown("### 🧾 Summary")
                        st.write(result["summary"].answer)
                        st.markdown(f"📄 Pages Referenced: {result['summary'].page_numbers}")

                    if "risk_analysis" in result:
                        st.markdown("### ⚠️ Risk Analysis")
                        st.write(result["risk_analysis"].answer)
                        st.markdown(f"📄 Risk Pages: {result['risk_analysis'].page_numbers}")

                    if "query_answer" in result:
                        st.markdown("### 🤖 Answer to Your Query")
                        st.write(result["query_answer"].answer)
                        st.markdown(f"📄 Pages: {result['query_answer'].page_numbers}")