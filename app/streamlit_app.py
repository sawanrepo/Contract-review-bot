import streamlit as st
from vectorstore import VectorStore
from document_loader import load_documents
from langchain.schema import Document
from langgraph_flow import langgraph_flow
from memory import ChatMemory

st.set_page_config(page_title="Contract Review Bot 🤖", layout="wide")
st.title("💬 Contract Review Chatbot")

vs = VectorStore()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "last_uploaded_file_name" not in st.session_state:
    st.session_state.last_uploaded_file_name = ""
if "ChatMemory" not in st.session_state:
    st.session_state.ChatMemory = ChatMemory()

uploaded_file = st.file_uploader("📎 Upload a contract (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.name != st.session_state.last_uploaded_file_name:
        with st.spinner("📚 Processing document..."):
            chunks = load_documents(uploaded_file)
            if chunks and not isinstance(chunks[0], Document):
                chunks = [Document(page_content=getattr(doc, 'page_content', str(doc))) for doc in chunks]

            vs.clear()
            vs.add_documents(chunks)
            st.session_state.last_uploaded_file_name = uploaded_file.name
            st.session_state.document_loaded = True
            st.session_state.messages = []
            st.session_state.ChatMemory.clear()

        st.success("✅ Document processed and embedded!")

st.markdown("---")
if st.session_state.document_loaded:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.ChatMemory.clear()
        st.success("🧹 Chat cleared!")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

if st.session_state.document_loaded:
    user_input = st.chat_input("Type your query or request a summary/risk analysis...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤖 Thinking..."):
                result = langgraph_flow(query=user_input, context=vs,memory=st.session_state.ChatMemory.get_messages())

                if "error" in result:
                    bot_response = f"❌ Error: {result['error']}"
                else:
                    bot_response = ""
                    if "summary" in result:
                        bot_response += f"### 🧾 Summary\n{result['summary'].answer}\n📄 **Pages:** {result['summary'].page_numbers}\n\n"
                    if "risk_analysis" in result:
                        bot_response += f"### ⚠️ Risk Analysis\n{result['risk_analysis'].answer}\n📄 **Pages:** {result['risk_analysis'].page_numbers}\n\n"
                    if "query_answer" in result:
                        bot_response += f"### 🤖 Answer\n{result['query_answer'].answer}\n📄 **Pages:** {result['query_answer'].page_numbers}\n\n"

                bot_response = bot_response.strip()
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response}) #to display previous messages in ui.
                st.session_state.ChatMemory.add_message("assistant", bot_response) 
else:
    st.info("📄 Please upload a contract to begin the chat.")