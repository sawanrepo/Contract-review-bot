import os
import sys
import unittest
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dataset_parser import CUADAnswer, CUADContract, CUADQuestion
from evaluation.contract_selector import resolve_contract_file
from evaluation.ground_truth_builder import (
    align_chunks_to_context,
    map_evidence_to_chunks,
    normalize_text,
    ChunkSpan,
)
from evaluation.metrics import (
    compute_chunk_precision_at_k,
    compute_chunk_recall_at_k,
    compute_evidence_coverage_intervals,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_per_span_coverage_details,
    compute_reciprocal_rank,
    merge_intervals,
)

class TestEvaluationPipeline(unittest.TestCase):

    def test_text_normalization(self):
        raw = "  This   is a  “Test”  — with ‘Smart’   Quotes… \n\n and   spaces.  "
        normalized = normalize_text(raw)
        self.assertEqual(normalized, 'this is a "test" - with \'smart\' quotes… and spaces.')

    def test_recall_at_k(self):
        gt = {"chunk_1", "chunk_2", "chunk_3"}
        retrieved = ["chunk_1", "chunk_4", "chunk_5", "chunk_2"]
        # Top 1: chunk_1 (1 / 3)
        self.assertAlmostEqual(compute_chunk_recall_at_k(gt, retrieved, 1), 1 / 3)
        # Top 3: chunk_1, chunk_4, chunk_5 (1 / 3)
        self.assertAlmostEqual(compute_chunk_recall_at_k(gt, retrieved, 3), 1 / 3)
        # Top 4: chunk_1, chunk_4, chunk_5, chunk_2 (2 / 3)
        self.assertAlmostEqual(compute_chunk_recall_at_k(gt, retrieved, 4), 2 / 3)
        # Empty ground truth returns 0.0
        self.assertEqual(compute_chunk_recall_at_k(set(), retrieved, 4), 0.0)

    def test_precision_at_k_convention(self):
        gt = {"chunk_1", "chunk_2"}
        retrieved = ["chunk_1", "chunk_3"]
        # K=2: 1 relevant out of 2 -> 0.5
        self.assertAlmostEqual(compute_chunk_precision_at_k(gt, retrieved, 2), 0.5)
        # K=5, but only 2 chunks actually returned: denominator is min(5, 2) = 2 -> 1 / 2 = 0.5
        self.assertAlmostEqual(compute_chunk_precision_at_k(gt, retrieved, 5), 0.5)
        # 0 chunks returned
        self.assertEqual(compute_chunk_precision_at_k(gt, [], 5), 0.0)

    def test_hit_rate_at_k(self):
        gt = {"chunk_2"}
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        self.assertEqual(compute_hit_rate_at_k(gt, retrieved, 1), 0.0)
        self.assertEqual(compute_hit_rate_at_k(gt, retrieved, 2), 1.0)
        self.assertEqual(compute_hit_rate_at_k(gt, retrieved, 3), 1.0)

    def test_reciprocal_rank_and_mrr(self):
        gt = {"chunk_3"}
        retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
        mrr, rank = compute_reciprocal_rank(gt, retrieved, max_k=4)
        self.assertEqual(rank, 3)
        self.assertAlmostEqual(mrr, 1.0 / 3.0)

        # No match in top 4
        mrr_miss, rank_miss = compute_reciprocal_rank({"chunk_99"}, retrieved, max_k=4)
        self.assertEqual(mrr_miss, 0.0)
        self.assertIsNone(rank_miss)

    def test_ndcg_at_k(self):
        gt = {"chunk_1", "chunk_3"}
        retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
        # Rank 1: rel=1 -> 1 / log2(2) = 1.0
        # Rank 2: rel=0
        # Rank 3: rel=1 -> 1 / log2(4) = 0.5
        # DCG@3 = 1.5
        # Ideal: rank 1 and 2 both relevant -> 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
        # nDCG@3 = 1.5 / 1.63092975 = 0.91972
        ndcg = compute_ndcg_at_k(gt, retrieved, 3)
        self.assertGreater(ndcg, 0.91)
        self.assertLess(ndcg, 0.93)

    def test_interval_evidence_coverage_and_deduplication(self):
        # Case 1: Partial interval coverage
        # Evidence: [100, 200) (length 100)
        # Retrieved chunk: [50, 150) -> overlap is [100, 150) = 50 chars -> 50%
        spans = [(100, 200)]
        chunks = [(50, 150)]
        cov = compute_evidence_coverage_intervals(spans, chunks)
        self.assertAlmostEqual(cov, 0.5)

        # Case 2: Overlapping evidence spans should NOT double-count characters
        # Evidence 1: [100, 200), Evidence 2: [150, 250) -> Union is [100, 250) (length 150)
        # Retrieved chunk: [100, 250) -> covers 150 chars -> 100%
        overlapping_spans = [(100, 200), (150, 250)]
        retrieved_chunk = [(100, 250)]
        cov_overlap = compute_evidence_coverage_intervals(overlapping_spans, retrieved_chunk)
        self.assertAlmostEqual(cov_overlap, 1.0)

        # Case 3: Per-span coverage details (single vs multi-chunk complete)
        # Span 0: [100, 200) -> covered completely by single chunk [80, 220)
        # Span 1: [300, 500) -> split across two chunks: [280, 410) and [400, 520) (collective 100%)
        # Span 2: [600, 700) -> partial coverage by [550, 650) (50%)
        multi_spans = [(100, 200), (300, 500), (600, 700)]
        all_chunks = [(80, 220), (280, 410), (400, 520), (550, 650)]
        details = compute_per_span_coverage_details(multi_spans, all_chunks)
        self.assertEqual(len(details), 3)

        # Span 0: single chunk complete
        self.assertAlmostEqual(details[0].coverage_percentage, 1.0)
        self.assertTrue(details[0].is_single_chunk_complete)
        self.assertFalse(details[0].is_multi_chunk_complete)

        # Span 1: multi-chunk complete
        self.assertAlmostEqual(details[1].coverage_percentage, 1.0)
        self.assertFalse(details[1].is_single_chunk_complete)
        self.assertTrue(details[1].is_multi_chunk_complete)

        # Span 2: partial
        self.assertAlmostEqual(details[2].coverage_percentage, 0.5)
        self.assertFalse(details[2].is_single_chunk_complete)
        self.assertFalse(details[2].is_multi_chunk_complete)

    def test_ground_truth_exact_offset_matching(self):
        context = "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789"
        # Chunks: [0..20], [15..35], [30..50]
        chunks = [
            ChunkSpan("c0", 0, 0, 20, context[0:20]),
            ChunkSpan("c1", 1, 15, 35, context[15:35]),
            ChunkSpan("c2", 2, 30, 50, context[30:50]),
        ]
        # Evidence at offset 18..25 ("STUVWXYZ" part)
        evidence_text = context[18:25]
        q = CUADQuestion(
            id="test__Category",
            question="Where is the text?",
            category="Category",
            is_impossible=False,
            answers=[CUADAnswer(text=evidence_text, answer_start=18)],
        )
        gt = map_evidence_to_chunks(q, chunks, context)
        self.assertEqual(gt.mapping_status, "fully_mapped")
        # 18..25 overlaps c0 (0..20) and c1 (15..35)
        self.assertEqual(gt.validated_chunk_ids, {"c0", "c1"})
        self.assertEqual(gt.span_results[0].status, "exact_offset")
        self.assertTrue(gt.span_results[0].validated)

    def test_ground_truth_fuzzy_candidate_not_validated(self):
        # 19 matching words out of 21 unique words -> Jaccard = 19/21 = 0.9048 (> 0.85)
        span_text = "this agreement shall be governed construed and enforced in accordance with the laws of the state of new york applicable to contracts"
        chunk_text = "this agreement shall be governed construed and interpreted in accordance with the laws of the state of new york applicable to contracts"
        chunks = [
            ChunkSpan("c0", 0, -1, -1, chunk_text),
        ]
        q = CUADQuestion(
            id="test__GoverningLaw",
            question="Find governing law",
            category="GoverningLaw",
            is_impossible=False,
            answers=[CUADAnswer(text=span_text, answer_start=0)],
        )
        gt = map_evidence_to_chunks(q, chunks, "completely different context")
        # Critical methodological rule: fuzzy matching must be recorded as candidate and NOT validated!
        self.assertEqual(gt.mapping_status, "unmapped")
        self.assertEqual(len(gt.validated_chunk_ids), 0)
        self.assertFalse(gt.span_results[0].validated)
        self.assertEqual(gt.span_results[0].status, "fuzzy_candidate")
        self.assertIn("c0", gt.span_results[0].candidate_chunk_ids)

    def test_multiple_evidence_spans_partial_mapping(self):
        context = "Clause A exists here. Clause B is elsewhere."
        chunks = [
            ChunkSpan("c0", 0, 0, 21, "Clause A exists here."),
        ]
        q = CUADQuestion(
            id="test__Multi",
            question="Find both",
            category="Multi",
            is_impossible=False,
            answers=[
                CUADAnswer(text="Clause A exists here.", answer_start=0),
                CUADAnswer(text="Clause B is elsewhere.", answer_start=22),
            ],
        )
        gt = map_evidence_to_chunks(q, chunks, context)
        # Span 1 is mapped to c0, Span 2 is unmapped -> question is partially_mapped
        self.assertEqual(gt.mapping_status, "partially_mapped")
        self.assertEqual(gt.validated_chunk_ids, {"c0"})
        self.assertTrue(gt.span_results[0].validated)
        self.assertFalse(gt.span_results[1].validated)

    def test_contract_file_matching_unicode_nfc(self):
        title = "LECLANCHÉ S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT"
        txt_index = {
            "LECLANCHÉ S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT.txt": "/path/to/contract.txt"
        }
        res = resolve_contract_file(title, txt_index, {}, prefer_txt=True)
        self.assertIsNotNone(res)
        path, ftype, method = res
        self.assertEqual(path, "/path/to/contract.txt")
        self.assertEqual(ftype, "txt")
        self.assertEqual(method, "nfc_txt")

    def test_negative_query_handling(self):
        q = CUADQuestion(
            id="test__Impossible",
            question="Find something that does not exist",
            category="NonExistent",
            is_impossible=True,
            answers=[],
        )
        gt = map_evidence_to_chunks(q, [], "contract context")
        self.assertTrue(gt.is_impossible)
        self.assertEqual(gt.mapping_status, "impossible")
        self.assertEqual(len(gt.validated_chunk_ids), 0)

    def test_isolated_vectorstore_no_cross_contamination(self):
        from app.vectorstore import VectorStore
        from langchain_core.embeddings import FakeEmbeddings

        # Fake deterministic embeddings
        emb = FakeEmbeddings(size=16)

        vs1 = VectorStore(embedding_model=emb)
        vs1.add_documents([Document(page_content="Contract A secret clause", metadata={"chunk_id": "A_1"})])

        vs2 = VectorStore(embedding_model=emb)
        vs2.add_documents([Document(page_content="Contract B separate clause", metadata={"chunk_id": "B_1"})])

        res1 = vs1.search("secret", k=5)
        res2 = vs2.search("secret", k=5)

        # vs1 only contains A_1, vs2 only contains B_1
        self.assertEqual([d.metadata["chunk_id"] for d in res1], ["A_1"])
        self.assertEqual([d.metadata["chunk_id"] for d in res2], ["B_1"])

        vs1.clear()
        vs2.clear()

if __name__ == "__main__":
    unittest.main()
