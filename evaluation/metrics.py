import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

@dataclass
class SpanCoverageDetail:
    span_index: int
    span_start: int
    span_end: int
    span_length: int
    covered_chars: int
    coverage_percentage: float
    is_single_chunk_complete: bool  # Covered 100% by a single retrieved chunk
    is_multi_chunk_complete: bool   # Covered 100% collectively across multiple retrieved chunks

@dataclass
class QueryMetricResult:
    question_id: str
    category: str
    is_impossible: bool
    mapping_status: str
    total_ground_truth_chunks: int
    num_retrieved: int
    first_relevant_rank: Optional[int]
    recalls: Dict[int, float] = field(default_factory=dict)
    precisions: Dict[int, float] = field(default_factory=dict)
    hit_rates: Dict[int, float] = field(default_factory=dict)
    ndcgs: Dict[int, float] = field(default_factory=dict)
    evidence_coverages: Dict[int, float] = field(default_factory=dict)
    span_coverage_details: Dict[int, List[SpanCoverageDetail]] = field(default_factory=dict)
    mrr: float = 0.0
    top1_distance: Optional[float] = None       # FAISS IndexFlatL2 squared Euclidean distance (L2^2)
    mean_topk_distance: Optional[float] = None   # Mean squared Euclidean distance across top-K
    retrieval_latency_ms: float = 0.0

    # Backward-compatible property aliases
    @property
    def top1_score(self) -> Optional[float]:
        return self.top1_distance

    @property
    def mean_topk_score(self) -> Optional[float]:
        return self.mean_topk_distance

def compute_chunk_recall_at_k(ground_truth_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    r"""
    Recall@K = |G \cap R_K| / |G|
    """
    if not ground_truth_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_retrieved = ground_truth_ids & top_k_retrieved
    return len(relevant_retrieved) / len(ground_truth_ids)

def compute_chunk_precision_at_k(ground_truth_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    r"""
    Precision@K = |G \cap R_K| / min(K, |R|)
    Documented convention: denominator is actual number of chunks returned up to K.
    If 0 chunks returned, returns 0.0.
    """
    actual_k = min(k, len(retrieved_ids))
    if actual_k == 0:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:actual_k])
    relevant_retrieved = ground_truth_ids & top_k_retrieved
    return len(relevant_retrieved) / actual_k

def compute_hit_rate_at_k(ground_truth_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    r"""
    HitRate@K = 1.0 if |G \cap R_K| > 0 else 0.0
    """
    if not ground_truth_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    return 1.0 if len(ground_truth_ids & top_k_retrieved) > 0 else 0.0

def compute_reciprocal_rank(ground_truth_ids: Set[str], retrieved_ids: List[str], max_k: int = 10) -> Tuple[float, Optional[int]]:
    """
    MRR component: 1 / rank of first relevant chunk in retrieved_ids[:max_k].
    Returns (reciprocal_rank, first_rank). If none found, returns (0.0, None).
    """
    if not ground_truth_ids:
        return 0.0, None

    for rank, chunk_id in enumerate(retrieved_ids[:max_k], start=1):
        if chunk_id in ground_truth_ids:
            return 1.0 / rank, rank
    return 0.0, None

def compute_ndcg_at_k(ground_truth_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """
    nDCG@K using binary relevance:
    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(K, |G|)} 1 / log2(i + 1)
    """
    if not ground_truth_ids:
        return 0.0

    actual_k = min(k, len(retrieved_ids))
    dcg = 0.0
    for i in range(actual_k):
        rel = 1.0 if retrieved_ids[i] in ground_truth_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i=0 corresponds to rank 1, log2(2)=1

    ideal_k = min(k, len(ground_truth_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

# ---------------------------------------------------------------------------
# Character-Interval Evidence Coverage Implementation
# ---------------------------------------------------------------------------

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merges overlapping or adjacent [start, end) character intervals into disjoint intervals.
    Filters out invalid intervals where end <= start or start < 0.
    """
    valid = [(int(s), int(e)) for s, e in intervals if s is not None and e is not None and e > s and s >= 0]
    if not valid:
        return []
    valid.sort(key=lambda x: (x[0], x[1]))
    merged = [valid[0]]
    for cur_s, cur_e in valid[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))
    return merged

def interval_intersection_length(
    intervals_a: List[Tuple[int, int]],
    intervals_b: List[Tuple[int, int]],
) -> int:
    """
    Computes total unique character length of the intersection between two interval sets.
    Pre-merges both sets so no character is ever double-counted.
    """
    merged_a = merge_intervals(intervals_a)
    merged_b = merge_intervals(intervals_b)
    total_overlap = 0
    for a_s, a_e in merged_a:
        for b_s, b_e in merged_b:
            overlap = max(0, min(a_e, b_e) - max(a_s, b_s))
            total_overlap += overlap
    return total_overlap

def compute_evidence_coverage_intervals(
    evidence_span_intervals: List[Tuple[int, int]],
    retrieved_chunk_intervals: List[Tuple[int, int]],
) -> float:
    """
    Evidence Coverage@K:
    Measures how much of the annotated evidence text character intervals
    is covered by the union of intervals covered by retrieved chunks.
    Avoids double-counting overlapping evidence characters in both numerator and denominator.
    """
    merged_evidence = merge_intervals(evidence_span_intervals)
    total_evidence_chars = sum(e - s for s, e in merged_evidence)
    if total_evidence_chars == 0:
        return 0.0

    covered_chars = interval_intersection_length(merged_evidence, retrieved_chunk_intervals)
    return covered_chars / total_evidence_chars

def compute_per_span_coverage_details(
    evidence_spans: List[Tuple[int, int]],
    retrieved_chunk_intervals: List[Tuple[int, int]],
) -> List[SpanCoverageDetail]:
    """
    Computes granular coverage metrics for each annotated evidence span independently:
    - Coverage percentage
    - is_single_chunk_complete: True if at least one single retrieved chunk completely covers the span.
    - is_multi_chunk_complete: True if covered 100% collectively across multiple chunks, but not by a single chunk.
    """
    valid_chunks = [(int(s), int(e)) for s, e in retrieved_chunk_intervals if s is not None and e is not None and e > s and s >= 0]
    results: List[SpanCoverageDetail] = []

    for idx, (s_start, s_end) in enumerate(evidence_spans):
        s_len = max(0, s_end - s_start)
        if s_len == 0:
            results.append(
                SpanCoverageDetail(
                    span_index=idx,
                    span_start=s_start,
                    span_end=s_end,
                    span_length=0,
                    covered_chars=0,
                    coverage_percentage=0.0,
                    is_single_chunk_complete=False,
                    is_multi_chunk_complete=False,
                )
            )
            continue

        covered = interval_intersection_length([(s_start, s_end)], valid_chunks)
        coverage_pct = min(1.0, covered / s_len)

        # Single chunk containment: a single chunk spans from <= s_start to >= s_end
        single_complete = any(c_s <= s_start and c_e >= s_end for c_s, c_e in valid_chunks)
        multi_complete = (covered >= s_len) and not single_complete

        results.append(
            SpanCoverageDetail(
                span_index=idx,
                span_start=s_start,
                span_end=s_end,
                span_length=s_len,
                covered_chars=covered,
                coverage_percentage=coverage_pct,
                is_single_chunk_complete=single_complete,
                is_multi_chunk_complete=multi_complete,
            )
        )

    return results

def compute_percentiles(values: List[float], percentiles: List[int] = [50, 95]) -> Dict[str, float]:
    if not values:
        return {f"P{p}": 0.0 for p in percentiles}
    arr = np.array(values)
    return {f"P{p}": float(np.percentile(arr, p)) for p in percentiles}
