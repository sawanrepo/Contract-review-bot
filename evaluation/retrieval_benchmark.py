import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.document_loader import load_documents
from app.vectorstore import VectorStore, get_shared_embeddings
from evaluation.config import EvaluationConfig
from evaluation.contract_selector import MatchedContract
from evaluation.ground_truth_builder import (
    align_chunks_to_context,
    map_evidence_to_chunks,
    QuestionGroundTruth,
    SpanMappingResult,
)
from evaluation.metrics import (
    compute_chunk_precision_at_k,
    compute_chunk_recall_at_k,
    compute_evidence_coverage_intervals,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_per_span_coverage_details,
    compute_reciprocal_rank,
    QueryMetricResult,
    SpanCoverageDetail,
)

@dataclass
class RetrievedChunkInfo:
    chunk_id: str
    rank: int
    score: float  # FAISS IndexFlatL2 squared Euclidean distance (L2^2)
    page_number: int
    start_char: int
    end_char: int
    content: str

@dataclass
class QueryExecutionResult:
    contract_title: str
    question_id: str
    question_text: str
    category: str
    is_impossible: bool
    ground_truth: QuestionGroundTruth
    metrics: QueryMetricResult
    retrieved_chunks: List[RetrievedChunkInfo] = field(default_factory=list)

@dataclass
class ContractBenchmarkResult:
    contract_title: str
    file_path: str
    file_type: str
    num_chunks: int
    indexing_time_sec: float
    query_results: List[QueryExecutionResult] = field(default_factory=list)

@dataclass
class BenchmarkSuiteResult:
    config: EvaluationConfig
    contract_results: List[ContractBenchmarkResult] = field(default_factory=list)
    model_load_time_sec: float = 0.0

def warmup_embedding_model(embedding_model_name: str) -> float:
    """
    Initializes and warms up the embedding model once,
    returning the initial load time so it is separated from steady-state query latency.
    """
    t0 = time.perf_counter()
    shared_emb = get_shared_embeddings(embedding_model_name)
    # Warm up forward pass
    _ = shared_emb.embed_query("warmup contract review clause query")
    load_time = time.perf_counter() - t0
    return load_time

def evaluate_single_contract(
    matched: MatchedContract,
    config: EvaluationConfig,
    embedding_model: Any,
) -> ContractBenchmarkResult:
    """
    Evaluates a single contract in complete isolation:
    1. Loads documents using app.document_loader.
    2. Tags chunks with deterministic chunk IDs.
    3. Builds ground truth mappings (exact_offset, exact_substr, norm_substr, fuzzy_candidate).
    4. Creates an isolated in-memory FAISS VectorStore.
    5. Runs all 41 queries and records metrics, scores, and latencies.
    6. Destroys the FAISS index.
    """
    contract = matched.contract
    file_path = matched.file_path

    # Step 1: Load and chunk document
    raw_chunks = load_documents(file_path, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    # Step 2: Assign deterministic chunk metadata
    for idx, c in enumerate(raw_chunks):
        c.metadata["chunk_id"] = f"{contract.title}_chunk_{idx}"
        c.metadata["chunk_index"] = idx

    # Step 3: Align chunks to context and build ground truth
    chunk_spans = align_chunks_to_context(raw_chunks, contract.context)
    chunk_offsets_map: Dict[str, Tuple[int, int]] = {
        cs.chunk_id: (cs.start_char, cs.end_char) for cs in chunk_spans
    }

    ground_truths: Dict[str, QuestionGroundTruth] = {}
    for q in contract.questions:
        gt = map_evidence_to_chunks(q, chunk_spans, contract.context)
        ground_truths[q.id] = gt

    # Step 4: Build isolated FAISS index and measure indexing time
    vs = VectorStore(embedding_model=embedding_model, persist_directory=None)
    t_index_start = time.perf_counter()
    vs.add_documents(raw_chunks, batch_size=16)
    indexing_time_sec = time.perf_counter() - t_index_start

    # Step 5: Execute queries
    query_results: List[QueryExecutionResult] = []

    try:
        for q in contract.questions:
            gt = ground_truths[q.id]
            query_str = q.question

            # Measure steady-state retrieval latency
            t_query_start = time.perf_counter()
            search_results = vs.search_with_score(query_str, k=config.top_k, sparse_query=q.category)
            retrieval_latency_ms = (time.perf_counter() - t_query_start) * 1000.0

            retrieved_chunk_infos: List[RetrievedChunkInfo] = []
            retrieved_ids: List[str] = []
            retrieved_texts: List[str] = []
            scores: List[float] = []

            for rank, (doc, score) in enumerate(search_results, start=1):
                cid = doc.metadata.get("chunk_id", f"unknown_chunk_{rank}")
                c_start, c_end = chunk_offsets_map.get(cid, (-1, -1))
                retrieved_ids.append(cid)
                retrieved_texts.append(doc.page_content)
                scores.append(float(score))
                retrieved_chunk_infos.append(
                    RetrievedChunkInfo(
                        chunk_id=cid,
                        rank=rank,
                        score=float(score),
                        page_number=doc.metadata.get("page_number", 1),
                        start_char=c_start,
                        end_char=c_end,
                        content=doc.page_content,
                    )
                )

            # Compute metrics
            mrr, first_rank = compute_reciprocal_rank(
                gt.validated_chunk_ids, retrieved_ids, max_k=config.top_k
            )

            recalls: Dict[int, float] = {}
            precisions: Dict[int, float] = {}
            hit_rates: Dict[int, float] = {}
            ndcgs: Dict[int, float] = {}
            coverages: Dict[int, float] = {}
            span_details: Dict[int, List[SpanCoverageDetail]] = {}

            if gt.is_impossible:
                # Negative query: do not compute recall, hit rate, or coverage
                pass
            else:
                # Evidence spans from validated ground truth
                evidence_spans = [
                    (s.span_start, s.span_end) for s in gt.span_results if s.validated
                ]
                for k in config.k_values:
                    recalls[k] = compute_chunk_recall_at_k(gt.validated_chunk_ids, retrieved_ids, k)
                    precisions[k] = compute_chunk_precision_at_k(gt.validated_chunk_ids, retrieved_ids, k)
                    hit_rates[k] = compute_hit_rate_at_k(gt.validated_chunk_ids, retrieved_ids, k)
                    ndcgs[k] = compute_ndcg_at_k(gt.validated_chunk_ids, retrieved_ids, k)

                    # Interval-based Evidence Coverage@K using document character offsets
                    chunk_intervals_k = [
                        (rc.start_char, rc.end_char) for rc in retrieved_chunk_infos[:k]
                    ]
                    coverages[k] = compute_evidence_coverage_intervals(evidence_spans, chunk_intervals_k)
                    span_details[k] = compute_per_span_coverage_details(evidence_spans, chunk_intervals_k)

            top1_dist = scores[0] if scores else None
            mean_dist = sum(scores) / len(scores) if scores else None

            metric_res = QueryMetricResult(
                question_id=q.id,
                category=q.category,
                is_impossible=q.is_impossible,
                mapping_status=gt.mapping_status,
                total_ground_truth_chunks=len(gt.validated_chunk_ids),
                num_retrieved=len(retrieved_ids),
                first_relevant_rank=first_rank,
                recalls=recalls,
                precisions=precisions,
                hit_rates=hit_rates,
                ndcgs=ndcgs,
                evidence_coverages=coverages,
                span_coverage_details=span_details,
                mrr=mrr,
                top1_distance=top1_dist,
                mean_topk_distance=mean_dist,
                retrieval_latency_ms=retrieval_latency_ms,
            )

            query_results.append(
                QueryExecutionResult(
                    contract_title=contract.title,
                    question_id=q.id,
                    question_text=q.question,
                    category=q.category,
                    is_impossible=q.is_impossible,
                    ground_truth=gt,
                    metrics=metric_res,
                    retrieved_chunks=retrieved_chunk_infos,
                )
            )
    finally:
        # Step 6: Ensure isolated index is destroyed to prevent memory leaks or leakage
        vs.clear()
        del vs
        del raw_chunks
        gc.collect()

    return ContractBenchmarkResult(
        contract_title=contract.title,
        file_path=file_path,
        file_type=matched.file_type,
        num_chunks=len(chunk_spans),
        indexing_time_sec=indexing_time_sec,
        query_results=query_results,
    )

def run_retrieval_benchmark(
    matched_contracts: List[MatchedContract],
    config: EvaluationConfig,
) -> BenchmarkSuiteResult:
    """
    Runs the complete isolated benchmark across all selected contracts.
    """
    # Exclude initial model loading time from steady-state query latencies
    model_load_time_sec = warmup_embedding_model(config.embedding_model)
    embedding_model = get_shared_embeddings(config.embedding_model)

    suite_result = BenchmarkSuiteResult(
        config=config,
        model_load_time_sec=model_load_time_sec,
    )

    total = len(matched_contracts)
    print(f"\n=======================================================")
    print(f"Starting CUAD Retrieval Benchmark on {total} contract(s)")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Model Init Time: {model_load_time_sec:.2f}s")
    print(f"Top-K: {config.top_k}, Chunk Size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    print(f"=======================================================\n")

    for idx, matched in enumerate(matched_contracts, start=1):
        t0 = time.time()
        print(f"[{idx}/{total}] Processing contract: {matched.contract.title[:60]}... ({matched.file_type.upper()})")
        res = evaluate_single_contract(matched, config, embedding_model)
        suite_result.contract_results.append(res)
        elapsed = time.time() - t0
        print(f"   Indexed {res.num_chunks} chunks in {res.indexing_time_sec:.2f}s. Evaluated {len(res.query_results)} queries in {elapsed:.2f}s total.")

    return suite_result
