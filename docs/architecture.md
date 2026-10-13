# System Architecture - Contract Review Bot

This document details the end-to-end system architecture of the **Contract Review Bot**, an AI-powered legal document analysis system combining **Streamlit**, **FAISS**, **LangGraph**, and **Google Gemini (gemini-2.5-flash)**.

---

## 1. High-Level Architecture Diagram

The system follows a modular pipeline spanning Document Ingestion, Semantic Indexing, State-Machine Orchestration, Multi-Node Reasoning, and Presentation.

```mermaid
flowchart TD
    %% User & Presentation
    subgraph UI ["Client & Presentation Layer (Streamlit)"]
        User(["👤 User"])
        StreamlitApp["app/streamlit_app.py\n• File Uploader (.pdf / .txt)\n• Interactive Chat History\n• Risk Severity Indicators\n• Source Page Citation Badges\n• Markdown Audit Report Export"]
        SessionState[("st.session_state\n• VectorStore (In-Memory)\n• ChatMemory (Turn Buffer)\n• Document Metadata")]
    end

    %% Ingestion Pipeline
    subgraph Ingestion ["Ingestion & Transformation Pipeline"]
        DocLoader["app/document_loader.py\n• PyPDFLoader / TextLoader\n• 1-based page_number Normalization\n• RecursiveCharacterTextSplitter\n  (chunk_size=900, overlap=150)\n• Secure Temp File Auto-Cleanup"]
    end

    %% Storage & Indexing
    subgraph Storage ["Vector Storage & Retrieval"]
        EmbeddingEngine["app/vectorstore.py\n• HuggingFaceEmbeddings\n  (all-MiniLM-L6-v2)\n• Model-Keyed Global Cache"]
        FAISSStore["FAISS In-Memory Vector Store\n• Similarity Search (Cosine / L2)\n• get_all_documents() Extraction"]
    end

    %% LangGraph Orchestration
    subgraph Orchestration ["Orchestration & Reasoning Layer (LangGraph)"]
        GraphEntry["Graph Entry: classifier_node\n(app/nodes/classifier_node.py)\nStructured Intent Classification"]
        Router{"Intent Router\n(app/langgraph_flow.py)\nConditional Fan-Out"}

        RAGNode["RAG Node\n(app/nodes/rag_node.py)\n• Context Search (k=4)\n• Grounded Page Citation\n• Clause Excerpt Extraction"]
        RiskNode["Risk Analysis Node\n(app/nodes/risk_node.py)\n• Multi-Query Expansion (utils.py)\n• Severity: High/Med/Low\n• Red Flag Identification"]
        SummaryNode["Summary Node\n(app/nodes/summary_node.py)\n• Representative Document Sampling\n• 5-Section Legal Overview"]
        
        SupervisorNode["Supervisor Node\n(app/nodes/supervisor_node.py)\n• Fast-path passthrough (1 worker)\n• Synthesis LLM chain (>1 workers)\n• Deduplicated Page Merging"]
    end

    %% External Services
    subgraph ExternalServices ["External Foundation Models & APIs"]
        GeminiAPI["Google Gemini 2.5 Flash\n(via langchain-google-genai)\nStructured Output (Pydantic)"]
        TavilyAPI["Tavily Search API (Optional)\n(app/tools.py - LegalSearch)"]
    end

    %% Standalone / Planned Tooling
    subgraph ToolsModule ["Tool Registry (app/tools.py)"]
        ComplianceModule["app/nodes/compliance_node.py\n• check_compliance()\n• GDPR / Labor / Finance Rules"]
        TavilyTool["app/tools.py\n• legal_search() via TavilySearchResults"]
    end

    %% Data Flow Connections
    User -->|1. Upload Contract| StreamlitApp
    StreamlitApp -->|2. Buffer upload| DocLoader
    DocLoader -->|3. Normalized chunks| Storage
    EmbeddingEngine -->|Embed vectors| FAISSStore
    Storage -.->|Store in session| SessionState

    User -->|4. Submit Query / Quick Action| StreamlitApp
    StreamlitApp -->|5. Invoke Graph with VectorStore & Memory| GraphEntry
    GraphEntry -->|Classify intent| GeminiAPI
    GraphEntry -->|Return intents| Router

    Router -->|intent == 'query_answer'| RAGNode
    Router -->|intent == 'risk_analysis'| RiskNode
    Router -->|intent == 'summary'| SummaryNode
    Router -.->|Fallback if unrecognized| RAGNode

    RAGNode -->|Semantic search| FAISSStore
    RAGNode -->|Structured QueryOutput| GeminiAPI
    RiskNode -->|Multi-query search| FAISSStore
    RiskNode -->|Structured QueryOutput| GeminiAPI
    SummaryNode -->|Sample all/key chunks| FAISSStore
    SummaryNode -->|Structured QueryOutput| GeminiAPI

    RAGNode -->|Branch result| SupervisorNode
    RiskNode -->|Branch result| SupervisorNode
    SummaryNode -->|Branch result| SupervisorNode

    SupervisorNode -.->|Multi-branch synthesis if needed| GeminiAPI
    SupervisorNode -->|6. Unified QueryOutput| StreamlitApp
    StreamlitApp -->|7. Render response & update memory| User

    ToolsModule -.->|Standalone utility tools / Planned agent node| ExternalServices

    classDef primary fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#ffffff;
    classDef storage fill:#059669,stroke:#047857,stroke-width:2px,color:#ffffff;
    classDef router fill:#d97706,stroke:#b45309,stroke-width:2px,color:#ffffff;
    classDef external fill:#7c3aed,stroke:#6d28d9,stroke-width:2px,color:#ffffff;
    classDef planned fill:#475569,stroke:#334155,stroke-width:2px,stroke-dasharray: 5 5,color:#ffffff;

    class StreamlitApp,DocLoader primary;
    class FAISSStore,EmbeddingEngine,SessionState storage;
    class Router,GraphEntry router;
    class GeminiAPI,TavilyAPI external;
    class ToolsModule planned;
```

---

## 2. Component Responsibilities

### A. Presentation & Session Layer (`app/streamlit_app.py`)
- **State Isolation**: Maintains conversation messages, active document statistics, and user query buffers in `st.session_state`.
- **Resource Caching**: Utilizes `@st.cache_resource` for the `sentence-transformers/all-MiniLM-L6-v2` embedding model, ensuring that heavy PyTorch model weights are loaded only once per server runtime.
- **Client Feedback**: Renders risk status badges (🔴 High, 🟡 Medium, 🟢 Low), citation chips (`Page 1, 4`), collapsible clause quote expanders, and report exports.

### B. Document Ingestion Pipeline (`app/document_loader.py`)
- **Format Normalization**: Handles both `.pdf` and `.txt` files.
- **Page Metadata Continuity**: Normalizes `PyPDFLoader`'s 0-indexed page metadata into consistent 1-indexed integers (`metadata["page_number"]`). Assigns page `1` to plain text files.
- **Recursive Text Splitting**: Chunks content using `RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)` with paragraph, newline, and word boundaries.
- **Disk Cleanliness**: Employs unique process-isolated temporary file paths (`upload_{pid}_{filename}`) and cleans up disk files in a `finally` block immediately after extraction.

### C. Vector Engine & Embeddings (`app/vectorstore.py`)
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) on CPU.
- **Model Cache**: Implements an in-memory dictionary `_global_embeddings_cache` to store embedding model instances keyed by model name.
- **Session-Scoped Index**: In the web application, `VectorStore` instances remain in-memory per user session, eliminating file locking conflicts and cross-session contamination.
- **Document Access**: Exposes `get_all_documents()` alongside `search(query, k)` to allow representative sampling for contract-level summarization.

### D. LangGraph State Machine (`app/langgraph_flow.py`)
- **Typed State**: Uses `GraphState(TypedDict)` containing `query`, `memory`, `context`, `intents`, and intermediate node results.
- **Intent Router**: Routes to `rag`, `summary`, or `risk` nodes. Crucially, if classification returns empty or unrecognized intents, it enforces a fallback to `rag` to prevent execution deadlock.
- **Fan-Out Execution**: Supports single-intent routing or multi-intent branching (e.g., when a user asks to summarize and flag risks simultaneously).
- **Supervisor Join**: Consolidates results from active branches before reaching the `END` state.

### E. Reasoning Nodes (`app/nodes/`)
- **Classifier Node (`classifier_node.py`)**: Uses `ChatGoogleGenerativeAI.with_structured_output(IntentOutput)` to categorize user queries without fragile regex parsing.
- **RAG Node (`rag_node.py`)**: Retrieves top-4 semantic chunks, formats excerpts with page markers, injects recent conversation history, and invokes the LLM using `QueryOutput` schema. Filters returned citations against actual retrieved evidence.
- **Risk Node (`risk_node.py`)**: Executes multi-query retrieval (`utils.get_multiquery_docs`), scores overall risk severity (`high`, `medium`, `low`), and flags specific clause risks.
- **Summary Node (`summary_node.py`)**: Replaces flawed similarity search (`"summarize the contract"`) with representative document sampling across document sections, generating structured summaries covering Parties, Scope, Financials, Term, and Governing Law.
- **Supervisor Node (`supervisor_node.py`)**:
  - *Fast Path*: When only one analysis node was invoked, passes through the `QueryOutput` directly with zero added latency and zero additional token usage.
  - *Synthesis Path*: When multiple nodes were invoked, synthesizes cross-clause findings, merges page citations, and elevates the highest risk severity.

### F. Tools & Compliance Module (`app/tools.py` & `app/nodes/compliance_node.py`)
- **Compliance Checker**: Evaluates excerpts against international standards (GDPR, labor law, financial regulations).
- **Legal Search**: Connects to `TavilySearchResults` for external legal precedents when configured.
- *Status Note*: `tools.py` defines `TOOLS = [compliance_tool, legal_search_tool]`. These tools are currently standalone utility tools. They are not bound into LangGraph's active worker nodes, preventing runaway ReAct loops and staying within free-tier quota limits.

---

## 3. Technology Stack & Component Mapping

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit 1.45+ | Reactive web interface, session management, file upload, chat UI |
| **Orchestration** | LangGraph 0.4.5 | Stateful graph execution, conditional intent routing, supervisor aggregation |
| **LLM Provider** | Google Gemini 2.5 Flash | Structured reasoning, classification, summarization, risk scoring |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | 384-dimensional dense semantic vector representation |
| **Vector Index** | FAISS (`faiss-cpu`) | High-performance in-memory similarity search |
| **Parsing & Chunking** | LangChain Core / Community, PyPDF | PDF/TXT extraction, page metadata tracking, recursive chunking |
| **Data Validation** | Pydantic v2 | Strict JSON schema enforcement for model responses |
