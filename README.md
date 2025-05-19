
# 🤖 Contract Review Bot (LangGraph + Streamlit)

An AI-powered contract analysis assistant that allows users to upload legal documents (PDF/TXT), analyze them with structured outputs, and interact through a Streamlit interface. The bot uses **LangChain**, **LangGraph**, **RAG**, and **CHROMA**, with **schema-based reasoning** and memory.

---

## 🧠 Features

✅ Upload PDF or TXT contract files  
✅ Page-wise chunking and semantic embedding using FAISS  
✅ LangGraph-based multi-node reasoning:  
  • Clause-level risk analysis  
  • Semantic Q&A via RAG  
  • Contract summarization  
✅ Session memory for contextual conversation  
✅ Structured responses via `Pydantic` schema  
✅ Streamlit interface with clean UI and response highlights  

---

## 📁 Project Structure

```
contract-review-bot/
│
├── app/
│   ├── streamlit_app.py           # Main Streamlit frontend
│   ├── document_loader.py         # PDF/TXT reading, page splitting
│   ├── vectorstore.py             # Embedding + FAISS storage/search
│   ├── langgraph_flow.py          # LangGraph flow and execution
│   ├── memory.py                  # LangChain memory (per session)
│   ├── schema.py                  # Pydantic schemas for structured output
│   ├── utils.py                   # Parsers, highlight tools, etc.
│   └── nodes/                     # LangGraph nodes
│       ├── classifier_node.py     # Route query to the right node
│       ├── rag_node.py            # Retrieve + LLM answer
│       ├── risk_node.py           # Risk scoring of clauses
│       ├── summary_node.py        # Contract summarization
│       └── compliance_node.py     # Rule-based compliance checks (optional)
│
├── temp_files/                    # User-uploaded contract PDFs/TXTs
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## ⚙️ How It Works

### 🔄 1. Upload & Chunk
- Upload PDF or TXT via Streamlit
- Pages are split and chunked into logical text blocks
- Chunks are embedded using Sentence Transformers (or OpenAI) and stored in FAISS

### 🧠 2. LangGraph Reasoning
LangGraph is used to intelligently route queries:
```
Start → Classifier → 
   ├─→ RAG (query answering)
   ├─→ Risk Analysis (clause-level)
   ├─→ Summarization
   └─→ Compliance Check
```

### 🧾 3. Structured Output
Each node returns structured responses like:
```json
{
  "type": "risk_analysis",
  "clauses": [
    {
      "clause_type": "Termination",
      "risk_score": 8.5,
      "comment": "Unilateral termination clause found.",
      "excerpt": "The lessor may terminate the contract at any time."
    }
  ]
}
```

### 💬 4. Streamlit UI
- Show chat-style interactions
- Highlight clause excerpts
- Display risk scores and summaries
- Maintain memory across interactions

---

## 🔧 Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/contract-review-bot.git
cd contract-review-bot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.template .env # On Windows: copy .env.template .env
```
#### Then open .env and fill in the required credentials or keys.


### 5. Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 🛠 Technologies Used

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| `Streamlit`      | UI for uploading, chat, results  |
| `LangChain`      | Memory, tools, chains            |
| `LangGraph`      | State-machine style reasoning    |
| `Chroma`          | Embedding vector search          |
| `PyPDF2`, `fitz` | PDF text extraction              |
| `Pydantic`       | Structured schema for outputs    |

---

## 💡 Future Enhancements

- Authentication and multi-user session support  
- Export structured reports as PDF  
- Clause highlighting inside rendered PDF  
- Fine-tuned clause-type classification  
- Custom contract template comparison  

---
