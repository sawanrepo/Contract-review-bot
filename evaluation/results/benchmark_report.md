# CUAD RAG Retrieval Benchmark Report - Contract Review Bot

## 1. Executive Summary & Production Upgrade (F6)

This evaluation benchmarks the upgraded production retrieval architecture (**F6 Dual-Query Hybrid Retrieval**) against the original baseline on the official CUAD dataset under identical, controlled evaluation conditions.

### Key Upgrades Implemented:
- **Embedding Model**: Upgraded from `sentence-transformers/all-MiniLM-L6-v2` (384-dim) to `BAAI/bge-base-en-v1.5` (768-dim, normalized).
- **Document Chunking**: Upgraded from chunk size 900 / overlap 150 to chunk size 1200 / overlap 200 (RecursiveCharacterTextSplitter).
- **Lexical Search Engine**: Integrated zero-dependency `BM25Okapi` with calibrated query saturation ($k_{\text{bm25}} = 10.0$).
- **Hybrid Fusion**: Exact Cosine + Calibrated BM25 Saturation ($\alpha = 0.6$ dense, $0.4$ sparse) with exact deduplication.
- **Query Routing**: Dual-Query Routing (Dense query prefixed with BGE retrieval instruction; Sparse query powered by category keywords).
- **Retrieval Depth**: Upgraded default retrieval depth from $K=4$ to $K=6$.

## 2. Benchmark Evaluation Conditions

- **Contracts Evaluated**: 10 matched contracts
- **Total Queries**: 410 queries
- **Answerable Queries**: 129 queries (129 fully mapped)
- **Unanswerable (Negative) Queries**: 281 queries
- **Random Seed**: 42
- **Chunk Size / Overlap**: 1200 / 200
- **Primary Embedding Model**: `BAAI/bge-base-en-v1.5`

## 3. Performance Benchmark Comparison Matrix

| Metric | Baseline (MiniLM-L6, K=4) | Production F6 (BGE-Base + BM25, K=6) | Absolute Gain | Relative Improvement |
|---|---:|---:|---:|---:|
| Recall@1 | 17.69% | **35.05%** | +17.36% | +98.1% |
| Recall@3 | 33.41% | **49.96%** | +16.55% | +49.5% |
| Recall@4 | 38.90% | **54.47%** | +15.57% | +40.0% |
| Recall@5 | 41.80% | **56.29%** | +14.49% | +34.7% |
| Recall@6 | 43.85% | **63.78%** | +19.93% | +45.4% |
| Recall@10 | 57.52% | **72.05%** | +14.53% | +25.3% |
| Precision@4 | 15.89% | **21.71%** | +5.82% | +36.6% |
| Precision@6 | 13.47% | **17.89%** | +4.42% | +32.8% |
| Hit Rate@4 | 48.84% | **64.34%** | +15.50% | +31.7% |
| Hit Rate@6 | 53.49% | **70.54%** | +17.05% | +31.9% |
| Evidence Coverage@4 | 41.75% | **56.72%** | +14.97% | +35.9% |
| Evidence Coverage@6 | 46.80% | **66.09%** | +19.29% | +41.2% |
| MRR (Mean Reciprocal Rank) | 0.3783 | **0.5521** | +0.1738 | +45.9% |
| Mean Retrieval Latency | 13.63 ms | **65.96 ms** | +52.33 ms | CPU sub-120ms |

## 4. Latency and Compute Profile

- **Model Initialization / Warmup**: 31.88s
- **Average Indexing Time Per Contract**: 6.943s
- **Mean Retrieval Latency**: 65.96 ms
- **P50 Retrieval Latency**: 65.43 ms
- **P95 Retrieval Latency**: 78.64 ms

## 5. Artifacts and Visualization Files

- `recall_at_k.png`: Recall progression from K=1 to K=10
- `precision_at_k.png`: Precision curve across retrieval depths
- `hit_rate_at_k.png`: Hit rate curve across retrieval depths
- `evidence_coverage_at_k.png`: Character-level evidence span coverage
- `category_recall_at_6.png`: Recall@6 breakdown across CUAD legal categories
- `category_hit_rate_at_6.png`: Hit Rate@6 breakdown across CUAD legal categories
- `retrieval_latency_distribution.png`: Latency histogram and percentiles
- `query_results.csv`: Complete query-by-query retrieval log and metrics
- `category_metrics.csv`: Aggregate legal category performance
- `failure_analysis.csv`: Detailed failure log for missed queries
- `retrieved_chunks.json`: Full chunk text and relevance scores
