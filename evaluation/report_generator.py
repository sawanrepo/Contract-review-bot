import csv
import json
import os
import sys
from typing import Any, Dict, List

_conda_site = r"C:\ProgramData\anaconda3\Lib\site-packages"
if os.path.exists(_conda_site) and _conda_site not in sys.path:
    sys.path.append(_conda_site)

import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import compute_percentiles, SpanCoverageDetail
from evaluation.retrieval_benchmark import BenchmarkSuiteResult, QueryExecutionResult

def format_span_coverage_details(details: List[SpanCoverageDetail]) -> str:
    if not details:
        return ""
    items = []
    for d in details:
        if d.is_single_chunk_complete:
            mode = "single_chunk_complete"
        elif d.is_multi_chunk_complete:
            mode = "multi_chunk_complete"
        elif d.coverage_percentage > 0.0:
            mode = "partial"
        else:
            mode = "uncovered"
        items.append(f"span_{d.span_index}:{d.coverage_percentage * 100:.1f}%({mode})")
    return ";".join(items)

def generate_reports(suite_result: BenchmarkSuiteResult, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    config = suite_result.config
    contract_results = suite_result.contract_results

    all_query_results: List[QueryExecutionResult] = []
    for cr in contract_results:
        all_query_results.extend(cr.query_results)

    # 1. Save benchmark_config.json
    config_dict = config.to_dict()
    config_dict["selected_contracts"] = [
        {
            "title": cr.contract_title,
            "file_path": cr.file_path,
            "file_type": cr.file_type,
            "num_chunks": cr.num_chunks,
            "indexing_time_sec": round(cr.indexing_time_sec, 3),
        }
        for cr in contract_results
    ]
    with open(os.path.join(output_dir, "benchmark_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # 2. Generate query_results.csv
    query_csv_path = os.path.join(output_dir, "query_results.csv")
    with open(query_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "contract_title",
            "question_id",
            "category",
            "is_impossible",
            "mapping_status",
            "expected_evidence_spans_count",
            "ground_truth_chunk_ids",
            "retrieved_chunk_ids",
            "squared_l2_distances",
            "first_relevant_rank",
            "mrr",
            "recall_at_1",
            "recall_at_3",
            "recall_at_4",
            "recall_at_5",
            "recall_at_10",
            "precision_at_1",
            "precision_at_3",
            "precision_at_4",
            "precision_at_5",
            "precision_at_10",
            "hit_rate_at_1",
            "hit_rate_at_3",
            "hit_rate_at_4",
            "hit_rate_at_5",
            "hit_rate_at_10",
            "evidence_coverage_at_1",
            "evidence_coverage_at_3",
            "evidence_coverage_at_4",
            "evidence_coverage_at_5",
            "evidence_coverage_at_10",
            "span_coverage_details_at_4",
            "span_coverage_details_at_10",
            "retrieval_latency_ms",
            "question_text",
        ])

        for qr in all_query_results:
            gt = qr.ground_truth
            m = qr.metrics
            retrieved_ids = [rc.chunk_id for rc in qr.retrieved_chunks]
            scores = [round(rc.score, 4) for rc in qr.retrieved_chunks]

            span_cov_4_str = format_span_coverage_details(m.span_coverage_details.get(4, []))
            span_cov_10_str = format_span_coverage_details(m.span_coverage_details.get(10, []))

            writer.writerow([
                qr.contract_title,
                qr.question_id,
                qr.category,
                qr.is_impossible,
                gt.mapping_status,
                len(gt.span_results),
                ";".join(sorted(gt.validated_chunk_ids)),
                ";".join(retrieved_ids),
                ";".join(map(str, scores)),
                m.first_relevant_rank if m.first_relevant_rank is not None else "None",
                round(m.mrr, 4) if not qr.is_impossible else "",
                round(m.recalls.get(1, 0.0), 4) if not qr.is_impossible else "",
                round(m.recalls.get(3, 0.0), 4) if not qr.is_impossible else "",
                round(m.recalls.get(4, 0.0), 4) if not qr.is_impossible else "",
                round(m.recalls.get(5, 0.0), 4) if not qr.is_impossible else "",
                round(m.recalls.get(10, 0.0), 4) if not qr.is_impossible else "",
                round(m.precisions.get(1, 0.0), 4) if not qr.is_impossible else "",
                round(m.precisions.get(3, 0.0), 4) if not qr.is_impossible else "",
                round(m.precisions.get(4, 0.0), 4) if not qr.is_impossible else "",
                round(m.precisions.get(5, 0.0), 4) if not qr.is_impossible else "",
                round(m.precisions.get(10, 0.0), 4) if not qr.is_impossible else "",
                round(m.hit_rates.get(1, 0.0), 4) if not qr.is_impossible else "",
                round(m.hit_rates.get(3, 0.0), 4) if not qr.is_impossible else "",
                round(m.hit_rates.get(4, 0.0), 4) if not qr.is_impossible else "",
                round(m.hit_rates.get(5, 0.0), 4) if not qr.is_impossible else "",
                round(m.hit_rates.get(10, 0.0), 4) if not qr.is_impossible else "",
                round(m.evidence_coverages.get(1, 0.0), 4) if not qr.is_impossible else "",
                round(m.evidence_coverages.get(3, 0.0), 4) if not qr.is_impossible else "",
                round(m.evidence_coverages.get(4, 0.0), 4) if not qr.is_impossible else "",
                round(m.evidence_coverages.get(5, 0.0), 4) if not qr.is_impossible else "",
                round(m.evidence_coverages.get(10, 0.0), 4) if not qr.is_impossible else "",
                span_cov_4_str if not qr.is_impossible else "",
                span_cov_10_str if not qr.is_impossible else "",
                round(m.retrieval_latency_ms, 2),
                qr.question_text,
            ])

    # 3. Generate failure_analysis.csv
    failure_csv_path = os.path.join(output_dir, "failure_analysis.csv")
    with open(failure_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "contract_title",
            "question_id",
            "category",
            "mapping_status",
            "failure_reason",
            "expected_evidence_text",
            "ground_truth_chunk_ids",
            "first_relevant_rank",
            "top1_squared_l2_distance",
            "retrieved_top4_chunks_preview",
            "question_text",
        ])

        for qr in all_query_results:
            if qr.is_impossible:
                continue

            gt = qr.ground_truth
            m = qr.metrics
            is_unmapped = gt.mapping_status == "unmapped"
            is_miss = m.hit_rates.get(4, 0.0) == 0.0

            if is_unmapped or is_miss:
                reason = "UNMAPPED_EVIDENCE" if is_unmapped else "RECALL_MISS_AT_K4"
                top4_previews = " | ".join(
                    f"[{rc.chunk_id} (dist: {round(rc.score, 3)})]: {rc.content[:80]}..."
                    for rc in qr.retrieved_chunks[:4]
                )
                writer.writerow([
                    qr.contract_title,
                    qr.question_id,
                    qr.category,
                    gt.mapping_status,
                    reason,
                    gt.all_evidence_text[:300].replace("\n", " "),
                    ";".join(sorted(gt.validated_chunk_ids)),
                    m.first_relevant_rank if m.first_relevant_rank is not None else "Not in Top 10",
                    round(m.top1_distance, 4) if m.top1_distance is not None else "",
                    top4_previews,
                    qr.question_text,
                ])

    # 4. Generate fuzzy_candidates.csv (Audit file)
    fuzzy_csv_path = os.path.join(output_dir, "fuzzy_candidates.csv")
    with open(fuzzy_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "contract_title",
            "question_id",
            "span_index",
            "fuzzy_similarity",
            "candidate_chunk_ids",
            "span_text",
        ])
        for qr in all_query_results:
            for sr in qr.ground_truth.span_results:
                if sr.status == "fuzzy_candidate":
                    writer.writerow([
                        qr.contract_title,
                        qr.question_id,
                        sr.span_index,
                        round(sr.fuzzy_similarity, 4),
                        ";".join(sr.candidate_chunk_ids),
                        sr.span_text[:200].replace("\n", " "),
                    ])

    # 5. Generate retrieved_chunks.json
    chunks_dict = {}
    for qr in all_query_results:
        c_key = f"{qr.contract_title}::{qr.question_id}"
        chunks_dict[c_key] = [
            {
                "chunk_id": rc.chunk_id,
                "rank": rc.rank,
                "squared_l2_distance": round(rc.score, 4),
                "page_number": rc.page_number,
                "start_char": rc.start_char,
                "end_char": rc.end_char,
                "content": rc.content,
            }
            for rc in qr.retrieved_chunks
        ]
    with open(os.path.join(output_dir, "retrieved_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, indent=2)

    # 6. Aggregate calculations
    answerable_all = [qr for qr in all_query_results if not qr.is_impossible]
    unanswerable_all = [qr for qr in all_query_results if qr.is_impossible]

    fully_mapped_queries = [qr for qr in answerable_all if qr.ground_truth.is_fully_mapped]
    partially_mapped_queries = [qr for qr in answerable_all if qr.ground_truth.is_partially_mapped]
    unmapped_queries = [qr for qr in answerable_all if qr.ground_truth.is_unmapped]

    def calc_mean_metrics(query_list: List[QueryExecutionResult]) -> Dict[str, Any]:
        if not query_list:
            return {}
        n = len(query_list)
        res = {
            "query_count": n,
            "mrr": float(np.mean([q.metrics.mrr for q in query_list])),
        }
        for k in config.k_values:
            res[f"recall_at_{k}"] = float(np.mean([q.metrics.recalls.get(k, 0.0) for q in query_list]))
            res[f"precision_at_{k}"] = float(np.mean([q.metrics.precisions.get(k, 0.0) for q in query_list]))
            res[f"hit_rate_at_{k}"] = float(np.mean([q.metrics.hit_rates.get(k, 0.0) for q in query_list]))
            res[f"ndcg_at_{k}"] = float(np.mean([q.metrics.ndcgs.get(k, 0.0) for q in query_list]))
            res[f"evidence_coverage_at_{k}"] = float(np.mean([q.metrics.evidence_coverages.get(k, 0.0) for q in query_list]))
        return res

    primary_metrics = calc_mean_metrics(fully_mapped_queries)
    combined_metrics = calc_mean_metrics(fully_mapped_queries + partially_mapped_queries)

    # Negative query distances (FAISS IndexFlatL2 squared Euclidean distance)
    neg_top1_dists = [qr.metrics.top1_distance for qr in unanswerable_all if qr.metrics.top1_distance is not None]
    neg_mean_dists = [qr.metrics.mean_topk_distance for qr in unanswerable_all if qr.metrics.mean_topk_distance is not None]

    negative_analysis = {
        "count": len(unanswerable_all),
        "metric_type": "squared_euclidean_distance (L2^2)",
        "mean_top1_squared_l2_distance": float(np.mean(neg_top1_dists)) if neg_top1_dists else None,
        "p50_top1_squared_l2_distance": float(np.percentile(neg_top1_dists, 50)) if neg_top1_dists else None,
        "p90_top1_squared_l2_distance": float(np.percentile(neg_top1_dists, 90)) if neg_top1_dists else None,
        "mean_top10_squared_l2_distance": float(np.mean(neg_mean_dists)) if neg_mean_dists else None,
        "note": "FAISS IndexFlatL2 computes squared Euclidean distance (sum of squared vector differences, ||q - v||_2^2). Lower values indicate closer proximity in embedding space."
    }

    # Latencies
    all_latencies = [qr.metrics.retrieval_latency_ms for qr in all_query_results]
    latency_stats = {
        "mean_retrieval_latency_ms": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "p50_retrieval_latency_ms": float(np.percentile(all_latencies, 50)) if all_latencies else 0.0,
        "p95_retrieval_latency_ms": float(np.percentile(all_latencies, 95)) if all_latencies else 0.0,
        "mean_indexing_time_sec_per_contract": float(np.mean([cr.indexing_time_sec for cr in contract_results])),
        "model_warmup_load_time_sec": suite_result.model_load_time_sec,
    }

    aggregate_data = {
        "benchmark_summary": {
            "selected_contracts": len(contract_results),
            "total_questions": len(all_query_results),
            "answerable_questions": len(answerable_all),
            "unanswerable_questions": len(unanswerable_all),
        },
        "mapping_coverage": {
            "fully_mapped_questions": len(fully_mapped_queries),
            "partially_mapped_questions": len(partially_mapped_queries),
            "unmapped_questions": len(unmapped_queries),
            "full_mapping_rate": len(fully_mapped_queries) / len(answerable_all) if answerable_all else 0.0,
            "overall_valid_mapping_rate": (len(fully_mapped_queries) + len(partially_mapped_queries)) / len(answerable_all) if answerable_all else 0.0,
        },
        "primary_retrieval_metrics_fully_mapped": primary_metrics,
        "sensitivity_retrieval_metrics_fully_and_partially_mapped": combined_metrics,
        "negative_query_distance_analysis": negative_analysis,
        "latency_metrics": latency_stats,
    }

    with open(os.path.join(output_dir, "aggregate_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate_data, f, indent=2)

    # 7. Generate category_metrics.csv (across 41 categories)
    category_map: Dict[str, List[QueryExecutionResult]] = {}
    for qr in all_query_results:
        category_map.setdefault(qr.category, []).append(qr)

    category_csv_path = os.path.join(output_dir, "category_metrics.csv")
    with open(category_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category",
            "total_questions",
            "answerable_questions",
            "fully_mapped_questions",
            "recall_at_4",
            "recall_at_5",
            "precision_at_4",
            "precision_at_5",
            "hit_rate_at_4",
            "hit_rate_at_5",
            "evidence_coverage_at_4",
            "evidence_coverage_at_5",
            "mrr",
        ])

        for cat, q_list in sorted(category_map.items()):
            ans_list = [q for q in q_list if not q.is_impossible]
            fully_mapped_cat = [q for q in ans_list if q.ground_truth.is_fully_mapped]

            if fully_mapped_cat:
                r4 = np.mean([q.metrics.recalls.get(4, 0.0) for q in fully_mapped_cat])
                r5 = np.mean([q.metrics.recalls.get(5, 0.0) for q in fully_mapped_cat])
                p4 = np.mean([q.metrics.precisions.get(4, 0.0) for q in fully_mapped_cat])
                p5 = np.mean([q.metrics.precisions.get(5, 0.0) for q in fully_mapped_cat])
                hr4 = np.mean([q.metrics.hit_rates.get(4, 0.0) for q in fully_mapped_cat])
                hr5 = np.mean([q.metrics.hit_rates.get(5, 0.0) for q in fully_mapped_cat])
                cov4 = np.mean([q.metrics.evidence_coverages.get(4, 0.0) for q in fully_mapped_cat])
                cov5 = np.mean([q.metrics.evidence_coverages.get(5, 0.0) for q in fully_mapped_cat])
                mrr = np.mean([q.metrics.mrr for q in fully_mapped_cat])
            else:
                r4 = r5 = p4 = p5 = hr4 = hr5 = cov4 = cov5 = mrr = 0.0

            writer.writerow([
                cat,
                len(q_list),
                len(ans_list),
                len(fully_mapped_cat),
                round(r4, 4),
                round(r5, 4),
                round(p4, 4),
                round(p5, 4),
                round(hr4, 4),
                round(hr5, 4),
                round(cov4, 4),
                round(cov5, 4),
                round(mrr, 4),
            ])

    # 8. Generate Visual Charts
    generate_visual_plots(suite_result, aggregate_data, category_map, output_dir)
    print(f"\nAll benchmark reports and visual charts successfully generated in: {output_dir}")

def generate_visual_plots(
    suite_result: BenchmarkSuiteResult,
    aggregate_data: Dict[str, Any],
    category_map: Dict[str, List[QueryExecutionResult]],
    output_dir: str,
):
    primary = aggregate_data.get("primary_retrieval_metrics_fully_mapped", {})
    k_vals = suite_result.config.k_values

    # Plot 1: Recall@K
    recalls = [primary.get(f"recall_at_{k}", 0.0) for k in k_vals]
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_vals, recalls, marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    plt.title("Chunk Recall@K on CUAD (Fully Mapped Answers)", fontsize=13, fontweight="bold")
    plt.xlabel("Top-K Retrieved Chunks", fontsize=11)
    plt.ylabel("Recall@K", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 1.05)
    for x, y in zip(k_vals, recalls):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recall_at_k.png"), dpi=200)
    plt.close()

    # Plot 2: Precision@K
    precisions = [primary.get(f"precision_at_{k}", 0.0) for k in k_vals]
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_vals, precisions, marker="s", color="#ff7f0e", linewidth=2.5, markersize=8)
    plt.title("Chunk Precision@K on CUAD", fontsize=13, fontweight="bold")
    plt.xlabel("Top-K Retrieved Chunks", fontsize=11)
    plt.ylabel("Precision@K (Relevant / Actual Returned)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 1.05)
    for x, y in zip(k_vals, precisions):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_at_k.png"), dpi=200)
    plt.close()

    # Plot 3: Hit Rate@K
    hit_rates = [primary.get(f"hit_rate_at_{k}", 0.0) for k in k_vals]
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_vals, hit_rates, marker="^", color="#2ca02c", linewidth=2.5, markersize=8)
    plt.title("Hit Rate@K on CUAD", fontsize=13, fontweight="bold")
    plt.xlabel("Top-K Retrieved Chunks", fontsize=11)
    plt.ylabel("Hit Rate@K", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 1.05)
    for x, y in zip(k_vals, hit_rates):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hit_rate_at_k.png"), dpi=200)
    plt.close()

    # Plot 4: Evidence Coverage@K (Interval-based)
    coverages = [primary.get(f"evidence_coverage_at_{k}", 0.0) for k in k_vals]
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_vals, coverages, marker="D", color="#8e44ad", linewidth=2.5, markersize=8)
    plt.title("Interval Evidence Coverage@K on CUAD", fontsize=13, fontweight="bold")
    plt.xlabel("Top-K Retrieved Chunks", fontsize=11)
    plt.ylabel("Evidence Coverage@K (Unique Chars Covered / Total Chars)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 1.05)
    for x, y in zip(k_vals, coverages):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evidence_coverage_at_k.png"), dpi=200)
    plt.close()

    # Plot 5 & 6: Category-wise Recall@5 & Hit Rate@5 (Top categories sorted)
    cat_stats = []
    for cat, q_list in category_map.items():
        mapped = [q for q in q_list if not q.is_impossible and q.ground_truth.is_fully_mapped]
        if mapped:
            r5 = float(np.mean([q.metrics.recalls.get(5, 0.0) for q in mapped]))
            hr5 = float(np.mean([q.metrics.hit_rates.get(5, 0.0) for q in mapped]))
            cat_stats.append((cat, r5, hr5, len(mapped)))

    if cat_stats:
        cat_stats_sorted = sorted(cat_stats, key=lambda x: x[1], reverse=True)
        top_cats = cat_stats_sorted[:20]
        c_names = [c[0] for c in top_cats]
        c_r5 = [c[1] for c in top_cats]
        c_hr5 = [c[2] for c in top_cats]

        # Category Recall@5
        plt.figure(figsize=(10, 7))
        y_pos = np.arange(len(c_names))
        plt.barh(y_pos, c_r5, align="center", color="#4a90e2")
        plt.yticks(y_pos, c_names, fontsize=9)
        plt.gca().invert_yaxis()
        plt.xlabel("Recall@5", fontsize=11)
        plt.title("Category-wise Recall@5 (Top Active Categories)", fontsize=13, fontweight="bold")
        plt.xlim(0, 1.05)
        plt.grid(True, axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "category_recall_at_5.png"), dpi=200)
        plt.close()

        # Category Hit Rate@5
        plt.figure(figsize=(10, 7))
        plt.barh(y_pos, c_hr5, align="center", color="#50e3c2")
        plt.yticks(y_pos, c_names, fontsize=9)
        plt.gca().invert_yaxis()
        plt.xlabel("Hit Rate@5", fontsize=11)
        plt.title("Category-wise Hit Rate@5 (Top Active Categories)", fontsize=13, fontweight="bold")
        plt.xlim(0, 1.05)
        plt.grid(True, axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "category_hit_rate_at_5.png"), dpi=200)
        plt.close()

    # Plot 7: Retrieval Latency Distribution
    all_latencies = []
    for cr in suite_result.contract_results:
        all_latencies.extend([qr.metrics.retrieval_latency_ms for qr in cr.query_results])

    if all_latencies:
        plt.figure(figsize=(7, 4.5))
        plt.hist(all_latencies, bins=25, color="#e67e22", edgecolor="black", alpha=0.75)
        p50 = np.percentile(all_latencies, 50)
        p95 = np.percentile(all_latencies, 95)
        plt.axvline(p50, color="green", linestyle="dashed", linewidth=2, label=f"P50: {p50:.1f}ms")
        plt.axvline(p95, color="red", linestyle="dashed", linewidth=2, label=f"P95: {p95:.1f}ms")
        plt.title("FAISS Steady-State Query Retrieval Latency", fontsize=13, fontweight="bold")
        plt.xlabel("Latency (ms)", fontsize=11)
        plt.ylabel("Query Frequency", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "retrieval_latency_distribution.png"), dpi=200)
        plt.close()
