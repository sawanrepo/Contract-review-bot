# System Design Decisions & Architecture Trade-Offs

This document captures the engineering decisions, design trade-offs, and scalability implications of the **Contract Review Bot**. Each section evaluates why a particular technology or pattern was chosen, its limitations, viable alternatives, and what architectural changes would be required at **1,000** and **10,000 concurrent users**.

---

## 1. LangGraph State Machine vs. Monolithic LLM Prompting

### 1. Current Implementation
The application uses **LangGraph (`StateGraph`)** with a typed state (`GraphState`). User requests enter an intent classifier (`classifier_node`), route conditionally to specialized worker nodes (`rag_node`, `risk_node`, `summary_node`), and converge at a `supervisor_node` for synthesis and citation aggregation.

### 2. Reason for the Choice
- **Separation of Concerns**: Contract analysis entails distinct cognitive tasks. Answering a specific query requires tight semantic search; risk analysis requires multi-query expansion and aggressive boundary-testing; summarization requires broad representative coverage across all sections.
- **Token & Cost Efficiency**: Rather than passing the full contract text into a massive system prompt on every turn, LangGraph selectively invokes only the specialized sub-graph needed.
- **Deterministic Orchestration**: State transitions, error fallbacks, and multi-branch joins are controlled in code rather than relying entirely on LLM reasoning to follow multi-step instructions.

### 3. Trade-Offs & Limitations
- **Added Graph Latency**: An extra classification step occurs before routing to worker nodes (~500ms to 1s).
- **Complexity**: Debugging state transitions and edge predicates requires tracing state dictionaries rather than reading a single prompt response.

### 4. Alternative Approaches
- **Single Mega-Prompt**: Dump all contract text and a prompt like *"Summarize, find risks, and answer questions."* (Prone to instruction drift, lost-in-the-middle phenomenon, and excessive token burn).
- **Linear Sequential Chains (LCEL)**: Run classify $\rightarrow$ retrieve $\rightarrow$ risk $\rightarrow$ summary sequentially on every query. (Wastes 3x the LLM calls when a user only wants a simple clause lookup).

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- **Current Bottleneck**: Graph execution is in-process and synchronous per request.
- **1,000 Users**: Decouple graph execution into an asynchronous worker pool (Celery / Temporal / AWS SQS + FastAPI). Nodes execute as independent micro-tasks.
- **10,000 Users**: Implement distributed workflow orchestration (Temporal or custom event-driven state machines with Redis persistence for checkpoints). Classify queries using a lightweight fine-tuned classifier (e.g. SetFit / DistilBERT) in <20ms on an edge gateway instead of calling an external LLM API for intent routing.

---

## 2. In-Memory FAISS vs. Managed Vector Databases

### 1. Current Implementation
The application uses **FAISS (`faiss-cpu`)** initialized in-memory per user session. Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.

### 2. Reason for the Choice
- **Zero External Infrastructure**: FAISS runs as an in-memory C++ library directly inside the application process without requiring Docker containers, cloud accounts, or external network calls.
- **Session Privacy**: When a contract is analyzed, the vectors reside in transient memory and are discarded when the session ends or resets. No client legal text is stored in a shared external database.
- **Sub-Millisecond Search**: For a 50-page contract yielding ~200 chunks, in-memory FAISS similarity search completes in <2 milliseconds.

### 3. Trade-Offs & Limitations
- **Process Memory Consumption**: If 100 users upload large contracts simultaneously in the same Streamlit server, memory consumption scales linearly ($O(N)$ with document size and user count).
- **Ephemeral Lifecycle**: Reloading the page or server restart requires re-uploading and re-embedding the contract.

### 4. Alternative Approaches
- **ChromaDB / SQLite**: Provides local disk persistence but introduces SQLite file lock contention on Windows and concurrency bottlenecks.
- **Managed Vector DBs (Pinecone, Qdrant, Milvus, pgvector)**: Centralized, scalable, and persistent, but requires network I/O, API keys, external costs, and strict multi-tenant isolation.

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- **1,000 Users**: Migrate to a distributed vector database such as **Qdrant** or **pgvector (PostgreSQL)** with tenant-isolated namespaces (`tenant_id = user_id`, `collection = contract_id`).
- **10,000 Users**: Sharded vector clusters (e.g. Milvus or Qdrant cluster on Kubernetes) with tiered caching:
  - Hot tier: In-memory HNSW index on dedicated vector nodes.
  - Cold tier: Quantized embeddings (IVF-PQ) in object storage (S3/GCS).
  - Embeddings computed asynchronously via a distributed GPU embedding service (Triton Inference Server / vLLM).

---

## 3. Chunk Size (900 chars) & Overlap (150 chars) Strategy

### 1. Current Implementation
Text is segmented using `RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)` with hierarchical separators: `["\n\n", "\n", " ", ""]`.

### 2. Reason for the Choice
- **Embedding Window Alignment**: `sentence-transformers/all-MiniLM-L6-v2` has an optimal token window of 256 tokens (~1,000 characters). A chunk size of 900 characters ensures that chunks are **never truncated** by the embedding model.
- **Legal Clause Integrity**: Legal contracts are heavily structured into sections, subsections, and clauses. Chunks smaller than 400 characters fragment conditional sentences (*"provided that..."*). Chunks larger than 1,500 characters dilute semantic density.
- **Overlap for Semantic Cohesion**: A 150-character overlap prevents sentences and legal caveats spanning a boundary from being cut in half.

### 3. Trade-Offs & Limitations
- Fixed character-count splitting does not guarantee that a chunk starts and ends on a true legal clause boundary.
- Tables, exhibits, and multi-column contract formatting can be segmented awkwardly.

### 4. Alternative Approaches
- **Semantic Chunking**: Split by embedding similarity between consecutive sentences. (Higher compute overhead during ingestion).
- **Clause-Based Parsing (AST / Regex)**: Split strictly by numbered legal headers (`Section 1.1`, `Article IV`). (Fragile when contracts use non-standard formatting).

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- Introduce a two-tier hybrid chunking pipeline:
  1. Structural parsing (PDF layout extraction via `pdfplumber` or `marker` to extract genuine headings, paragraphs, and tables).
  2. Late chunking / parent-child retrieval: Store small chunks (300 chars) for high-precision vector search, but retrieve the parent section (2,000 chars) for LLM context.

---

## 4. Multi-Query Retrieval & Expansion

### 1. Current Implementation
In `app/utils.py`, `get_multiquery_docs` takes the user query, uses Gemini to generate 2–3 focused legal sub-queries, executes vector searches for **both** the original query and the sub-queries, and deduplicates the retrieved chunks.

### 2. Reason for the Choice
- **Vocabulary Mismatch**: Users often ask conversational questions (*"Can they fire me anytime?"*) while contracts use formal legal language (*"Termination for convenience without cause upon notice"*).
- **Cross-Clause Dependencies**: An indemnification question often touches limitation of liability, insurance requirements, and notice periods. Sub-queries uncover these interrelated clauses.

### 3. Trade-Offs & Limitations
- Adds an additional LLM call before vector search in the risk node.
- Increases the number of retrieved chunks sent to the reasoning LLM.

### 4. Alternative Approaches
- **Single Dense Retrieval**: Search only the raw user query. (Misses clauses using legalese synonyms).
- **BM25 + Dense Hybrid Search**: Combine keyword matching with vector search via Reciprocal Rank Fusion (RRF). (Faster than multi-query expansion and requires no extra LLM call).

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- Replace LLM-based query expansion with:
  1. A local legal domain synonym expander / rewriter.
  2. **Hybrid Search (Dense + Sparse)**: Combine dense vectors (bge-small or Cohere) with BM25 / SPLADE on Elasticsearch / Qdrant.
  3. Re-ranking: Retrieve top 25 chunks via hybrid search and re-rank with a cross-encoder model (e.g. `bge-reranker-large`) in <15ms.

---

## 5. Structured Outputs (Pydantic v2) vs. Freeform Text

### 1. Current Implementation
All nodes enforce strict Pydantic schemas via `llm.with_structured_output()`:
- `IntentOutput`: `intent: List[Literal["risk_analysis", "summary", "query_answer"]]`
- `QueryOutput`: `answer: str`, `page_numbers: List[int]`, `risk_level: Optional[str]`, `key_clauses: Optional[List[str]]`
- `SubqueriesOutput`: `subqueries: List[str]`

### 2. Reason for the Choice
- **Elimination of Parsing Failures**: Legacy ReAct string parsers rely on regex matching (`"Thought: ... Action: ..."`). Models frequently output slight formatting variations that throw `OutputParserException`.
- **Type-Safe Contract**: Downstream components (Streamlit badges, supervisor aggregators, citation expanders) can safely access typed attributes (`result.page_numbers`, `result.risk_level`) without parsing markdown.
- **Native JSON Schema Mode**: Modern Gemini models support constrained grammar decoding (JSON schema constraint at the decoding layer), guaranteeing valid schemas without retry loops.

### 3. Trade-Offs & Limitations
- Schema constraints require model support for tool calling or structured outputs.
- Rigid schemas can slightly constrain open-ended conversational replies.

### 4. Alternative Approaches
- Outputting raw Markdown and using regex to extract `Page:` or `Risk:`. (Highly fragile; breaks on novel model outputs).
- LangChain `JsonOutputParser` with repair prompts. (Adds latency and extra LLM calls on parse failures).

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- Maintain Pydantic schemas as the internal API standard across all services.
- Define schemas via Protocol Buffers (gRPC) or OpenAPI specs for microservice communication between the ingestion engine, retrieval engine, and reasoning workers.

---

## 6. Model Weight Caching (`@st.cache_resource`)

### 1. Current Implementation
- `sentence-transformers/all-MiniLM-L6-v2` is wrapped in `@st.cache_resource` in `streamlit_app.py`.
- `vectorstore.py` maintains an internal dictionary `_global_embeddings_cache` keyed by model name.

### 2. Reason for the Choice
- **Streamlit Execution Model**: In Streamlit, every user keystroke, button click, or widget change re-executes the entire script from top to bottom.
- **Preventing OOM & Latency**: Loading PyTorch and Sentence-Transformers on CPU takes ~3–5 seconds and consumes ~400 MB of RAM. Without caching, a user interacting with the app 10 times would repeatedly re-allocate gigabytes of RAM and experience painful latency.

### 3. Trade-Offs & Limitations
- Model weights remain resident in server RAM for the lifetime of the process.

### 4. Alternative Approaches
- External Embedding API (e.g. Google `text-embedding-004` or OpenAI `text-embedding-3-small`). (Eliminates local RAM consumption but adds external API costs and per-chunk network latency).

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- Remove local embedding model inference from web worker processes completely.
- Offload embedding generation to a dedicated, autoscaled embedding microservice (Triton / Text Embeddings Inference by HuggingFace) running on GPU clusters, accessed via gRPC.

---

## 7. Strict Source-Page Traceability & Citation Grounding

### 1. Current Implementation
1. Document ingestion tracks 1-based page numbers in `doc.metadata["page_number"]`.
2. Retrieval injects `[Page X]:` headers directly into the LLM context.
3. RAG and Risk nodes post-process the LLM's returned `page_numbers`, validating them against the set of pages actually retrieved (`retrieved_pages`). Hallucinated page numbers are stripped.
4. Streamlit renders verified citations as clickable chips and quote expanders.

### 2. Reason for the Choice
- **Legal Compliance & Trust**: In legal review, an ungrounded claim is dangerous. Lawyers and users must verify every risk or summary statement against the exact page of the contract.
- **Anti-Hallucination Guardrail**: LLMs frequently hallucinate plausible-sounding page numbers (e.g. citing Page 99 in a 4-page contract). Programmatic intersection with retrieved evidence prevents this failure mode.

### 3. Trade-Offs & Limitations
- Chunks spanning page boundaries might attribute text slightly to the boundary page.
- Text files default to Page 1 because raw `.txt` files lack native page boundaries.

### 4. Alternative Approaches
- Highlighting bounding boxes inside rendered PDF canvases via pdf2image and OCR coordinates. (Significantly richer UX, but requires PDF rendering infrastructure and layout coordinates).

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- Store character offset spans `(start_char, end_char)` and bounding box coordinates `(x0, y0, x1, y1, page)` alongside text chunks in vector metadata.
- Return character spans with responses to enable instant PDF viewer highlighting in frontend web apps.

---

## 8. Multi-Turn Conversation Memory

### 1. Current Implementation
`ChatMemory` ([`app/memory.py`](file:///d:/Contract-review-bot/app/memory.py)) maintains an in-memory chronological list of dictionaries: `[{"role": "user"|"assistant", "content": "..."}]`. Up to the last 5 conversation turns are injected into the LLM context.

### 2. Reason for the Choice
- **Pronoun & Context Resolution**: Enables follow-up questions like *"What did you say about the liability clause earlier?"* or *"Does that apply to termination too?"*.
- **No Deprecation Warnings**: Replaced legacy `ConversationBufferMemory` with a lightweight, serializable data structure.

### 3. Trade-Offs & Limitations
- Sliding window of 5 turns discards context from older conversation turns in long review sessions.
- In-memory storage is lost if the user refreshes the browser page.

### 4. Alternative Approaches
- **Summary Memory**: Summarize previous turns using an LLM. (Consumes extra tokens on every turn).
- **Persistent Storage**: Save conversation history to PostgreSQL / DynamoDB with session tokens.

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- Store conversation state in a managed session store (Redis / DynamoDB) keyed by `session_id` with an explicit TTL (e.g. 24 hours).
- Use LangGraph Checkpointers (`SqliteSaver` or `PostgresSaver`) to enable durable state resumption, time-travel debugging, and multi-device session continuity.

---

## 9. Streamlit Frontend Suitability

### 1. Current Implementation
The application interface is built entirely with **Streamlit** ([`app/streamlit_app.py`](file:///d:/Contract-review-bot/app/streamlit_app.py)).

### 2. Reason for the Choice
- **Rapid Prototyping & Demonstration**: Allows an AI engineer to build a complete interactive legal dashboard (file uploader, chat streams, expanders, report downloads) in ~200 lines of pure Python without frontend/backend separation overhead.
- **Python-Native State**: `st.session_state` connects directly to Python vector stores and LangGraph instances without serializing objects over REST APIs.

### 3. Trade-Offs & Limitations
- **Script Rerun Paradigm**: Every widget interaction reruns the Python script, requiring careful caching to avoid performance degradation.
- **Limited UI Customization**: Complex side-by-side PDF viewers with real-time text highlight overlays are difficult to build natively in Streamlit compared to React / Vue.
- **Single Process Web Server**: Not designed for massive horizontal scaling or enterprise multi-tenancy.

### 4. Alternative Approaches
- **Decoupled Architecture**: FastAPI backend + React / Next.js frontend with PDF.js viewer. (Standard enterprise architecture, but requires maintaining separate repositories, build systems, and API contracts).

### 5. Scaling to 1,000 – 10,000 Concurrent Users
- Migrate to:
  - **Frontend**: React / Next.js deployed on Vercel/Cloudflare CDN with an interactive PDF viewer.
  - **API Gateway**: Envoy / Kong handling rate limiting, authentication, and JWT verification.
  - **Backend API**: FastAPI / Go service exposing Server-Sent Events (SSE) or WebSockets for streaming token responses.
