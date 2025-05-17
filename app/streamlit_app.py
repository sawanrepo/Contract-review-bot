'''
Just made basic ui with few things will be re done later with all functionality and proper working ,.... 
Ignore this file for now .
'''
import streamlit as st
from document_loader import load_documents  
from langchain.schema import Document
from vectorstore import VectorStore

st.set_page_config(page_title="Contract Review Bot 🤖", layout="wide")
st.title("📄 Contract Review Bot")

vs = VectorStore()

uploaded_file = st.file_uploader("Upload a contract (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("Processing document..."):
        chunks = load_documents(uploaded_file)
        if chunks and not isinstance(chunks[0], Document):
            chunks = [Document(page_content=txt) for txt in chunks]
        vs.add_documents(chunks)
        st.success("Document processed and embedded!")

    query = st.text_input("Ask a question about the contract")

    if query:
        with st.spinner("Thinking..."):
            results = vs.search(query, k=3)
            if results:
                for i, res in enumerate(results, 1):
                    st.markdown(f"**Result {i}:**")
                    st.write(res.page_content)
            else:
                st.write("No relevant results found.") 




