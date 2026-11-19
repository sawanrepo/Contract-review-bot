# 📜 Version Change Report (v1.0 Baseline vs. v2.0 Production)

This document summarizes the architectural upgrades, configuration parameters, and core metric comparisons between **v1.0 (Baseline)** and **v2.0 (Dual-Query Hybrid Retrieval)** of the Contract Review Bot.

---

## 1. What's New in v2.0

### A. Architectural & Algorithmic Upgrades
1. **Integrated BM25 Lexical Engine ([`app/bm25.py`](app/bm25.py))**:
   - Zero-dependency, pure-Python implementation of BM25Okapi ($k_1=1.5, b=0.75$).
   - Tokenizes and indexes contract clause text to capture exact legal terms, clause numbers, and statutory definitions that dense models often miss.

2. **Dual-Query Hybrid Fusion ([`app/vectorstore.py`](app/vectorstore.py))**:
   - Replaced dense-only FAISS with a calibrated hybrid search engine combining dense semantic similarity and sparse BM25 saturation:
     $$\text{Sim}_{\text{dense}} = \max\left(0.0, 1.0 - \frac{d_{\text{FAISS FlatL2}}}{2.0}\right)$$
     $$\text{Sim}_{\text{sparse}} = \frac{S}{S + 10.0}$$
     $$\text{Score} = 0.6 \cdot \text{Sim}_{\text{dense}} + 0.4 \cdot \text{Sim}_{\text{sparse}}$$
   - Scale-invariant linear fusion with exact candidate deduplication.

3. **Dual-Query Routing**:
   - **Dense Query**: Prefixed with BGE retrieval prompt (`"Represent this sentence for searching relevant passages: "`).
   - **Sparse Query**: Routes clean category keywords (or user query) directly to BM25 to avoid noise from conversational syntax.

4. **Optimized Document Chunking ([`app/document_loader.py`](app/document_loader.py))**:
   - Upgraded from 900 to 1200 characters (200 overlap), keeping complete multi-clause legal sentences together in single chunks.

5. **Retrieval Depth Expansion ([`app/nodes/rag_node.py`](app/nodes/rag_node.py))**:
   - Upgraded default retrieval depth from $K=4$ to $K=6$ to capture distributed multi-paragraph evidence without blowing up LLM prompt context.

6. **Expanded Test Suite**:
   - Expanded unit tests from 28 to 41 tests passing offline in ~0.1s.

---

## 2. Configuration Comparison Matrix

| Parameter / Setting | Previous Version (v1.0) | Current Version (v2.0) | Reason for Change |
|---|---|---|---|
| **Retrieval Method** | Dense FAISS Vector Search | Dual-Query Hybrid (FAISS + BM25Okapi) | Overcomes dense blind spots for exact legal keywords |
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) | `BAAI/bge-base-en-v1.5` (768-dim, normalized) | Higher semantic capacity and legal domain representation |
| **Chunk Size** | 900 characters | 1200 characters | Preserves complete multi-line clauses and statutory text |
| **Chunk Overlap** | 150 characters | 200 characters | Prevents boundary-split evidence between adjacent chunks |
| **Retrieval Depth ($K$)** | **$K = 4$** | **$K = 6$** | Captures multi-paragraph definitions and exceptions |
| **Fusion Weights** | N/A (Dense only) | $\alpha = 0.6$ (Dense), $0.4$ (Sparse) | Balanced semantic understanding with lexical precision |
| **BM25 Saturation ($k_{\text{bm25}}$)** | N/A | $10.0$ | Calibrates unbounded BM25 scores into $[0, 1]$ cosine scale |
| **Query Conditioning** | Raw user string | Dual-Query (BGE prompt + category routing) | Optimizes representations for both dense and sparse models |

---

## 3. Core Performance Metrics Comparison

Evaluated on the official CUAD dataset benchmark across 10 contracts (410 queries, 129 answerable queries, seed 42) at each version's operational retrieval depth:

| Core Metric | Previous Version (v1.0, $K=4$) | Current Version (v2.0, $K=6$) | Absolute Improvement | Relative Improvement |
|---|---:|---:|---:|---:|
| **Hit Rate@K** | 48.84% | **70.54%** | **+21.70%** | **+44.4%** |
| **Recall@K** | 38.90% | **63.78%** | **+24.88%** | **+63.9%** |
| **Evidence Coverage@K** | 41.75% | **66.09%** | **+24.34%** | **+58.3%** |
| **Precision@K** | 15.89% | **17.89%** | **+2.00%** | **+12.6%** |
| **Mean Reciprocal Rank (MRR)** | 0.3783 | **0.5521** | **+0.1738** | **+45.9%** |
| **Top-1 Recall (Recall@1)** | 17.69% | **35.05%** | **+17.36%** | **+98.1%** |
| **Steady-State Query Latency** | 13.63 ms | **65.96 ms** | +52.33 ms | Sub-70 ms on CPU |

---

## 4. Key Metric Takeaways

- **Hit Rate Broke 70%**: The system now finds relevant contract evidence in **70.54%** of queries (up from 48.84%).
- **Top-1 Quality Nearly Doubled**: Recall@1 jumped from 17.69% to **35.05%**, meaning the very first retrieved chunk answers the query over a third of the time.
- **Evidence Span Integrity**: Character-level evidence coverage grew from 41.75% to **66.09%**, minimizing partial-clause hallucinations.
- **Latency Remains Interactive**: At **65.96 ms** per query on CPU, retrieval adds virtually no noticeable delay to the end-to-end user experience.
