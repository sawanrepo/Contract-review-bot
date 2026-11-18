import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from langchain_core.documents import Document
from evaluation.dataset_parser import CUADAnswer, CUADQuestion

def normalize_text(text: str) -> str:
    """
    Standardizes whitespace, unicode characters, quotes and dashes
    without destroying meaningful legal text.
    """
    if not text:
        return ""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Normalize curly quotes and special dashes
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    # Collapse multiple whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

@dataclass
class ChunkSpan:
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    page_content: str

@dataclass
class SpanMappingResult:
    span_index: int
    span_text: str
    span_start: int
    span_end: int
    status: str  # "exact_offset", "exact_substring", "normalized_substring", "fuzzy_candidate", "unmapped"
    validated: bool  # True ONLY for exact_offset, exact_substring, normalized_substring
    matched_chunk_ids: List[str] = field(default_factory=list)
    candidate_chunk_ids: List[str] = field(default_factory=list)  # for fuzzy_candidate
    fuzzy_similarity: float = 0.0

@dataclass
class QuestionGroundTruth:
    question_id: str
    category: str
    is_impossible: bool
    mapping_status: str  # "fully_mapped", "partially_mapped", "unmapped", "impossible"
    span_results: List[SpanMappingResult] = field(default_factory=list)
    validated_chunk_ids: Set[str] = field(default_factory=set)
    all_evidence_text: str = ""  # Combined text of all validated ground-truth spans

    @property
    def is_fully_mapped(self) -> bool:
        return self.mapping_status == "fully_mapped"

    @property
    def is_partially_mapped(self) -> bool:
        return self.mapping_status == "partially_mapped"

    @property
    def is_unmapped(self) -> bool:
        return self.mapping_status == "unmapped"

def align_chunks_to_context(
    chunks: List[Document],
    context: str,
    overlap_buffer: int = 300,
) -> List[ChunkSpan]:
    """
    Finds the character offsets of each chunk in the original context text.
    """
    chunk_spans: List[ChunkSpan] = []
    cursor = 0

    for i, c in enumerate(chunks):
        content = c.page_content
        chunk_id = c.metadata.get("chunk_id", f"chunk_{i}")
        
        # Search starting slightly before the cursor to accommodate chunk overlap
        search_start = max(0, cursor - overlap_buffer)
        pos = context.find(content, search_start)
        if pos == -1:
            # Fallback search from the very beginning if text jumped
            pos = context.find(content, 0)

        if pos != -1:
            start_char = pos
            end_char = pos + len(content)
            cursor = pos
        else:
            start_char = -1
            end_char = -1

        chunk_spans.append(
            ChunkSpan(
                chunk_id=chunk_id,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                page_content=content,
            )
        )

    return chunk_spans

def token_jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(normalize_text(text_a).split())
    tokens_b = set(normalize_text(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

def map_evidence_to_chunks(
    question: CUADQuestion,
    chunk_spans: List[ChunkSpan],
    context: str,
) -> QuestionGroundTruth:
    """
    Maps each annotated evidence span to document chunks using strict validated hierarchy:
    Tier 1: Exact offset
    Tier 2: Exact substring
    Tier 3: Normalized substring
    Tier 4: Fuzzy candidate (> 0.85 Jaccard overlap, excluded from primary metrics)
    """
    if question.is_impossible or not question.answers:
        return QuestionGroundTruth(
            question_id=question.id,
            category=question.category,
            is_impossible=True,
            mapping_status="impossible",
            span_results=[],
            validated_chunk_ids=set(),
            all_evidence_text="",
        )

    span_results: List[SpanMappingResult] = []
    validated_chunk_ids: Set[str] = set()
    validated_evidence_texts: List[str] = []

    for idx, ans in enumerate(question.answers):
        span_text = ans.text
        span_start = ans.answer_start
        span_end = span_start + len(span_text)

        matched_chunk_ids: List[str] = []
        status = "unmapped"
        validated = False

        # Tier 1: Exact Offset Match (only if context offset matches span_text)
        if (
            0 <= span_start < len(context)
            and context[span_start:span_end] == span_text
        ):
            offset_matches = [
                cs.chunk_id
                for cs in chunk_spans
                if cs.start_char >= 0
                and cs.start_char < span_end
                and cs.end_char > span_start
            ]
            if offset_matches:
                matched_chunk_ids = offset_matches
                status = "exact_offset"
                validated = True

        # Tier 2: Exact Substring Containment
        if not validated:
            substr_matches = [
                cs.chunk_id for cs in chunk_spans if span_text in cs.page_content
            ]
            if substr_matches:
                matched_chunk_ids = substr_matches
                status = "exact_substring"
                validated = True

        # Tier 3: Normalized Substring
        if not validated:
            norm_span = normalize_text(span_text)
            norm_matches = [
                cs.chunk_id
                for cs in chunk_spans
                if norm_span in normalize_text(cs.page_content)
            ]
            if norm_matches:
                matched_chunk_ids = norm_matches
                status = "normalized_substring"
                validated = True

        # Tier 4: Fuzzy Candidate Check (Never mark as validated!)
        candidate_chunk_ids: List[str] = []
        fuzzy_sim = 0.0
        if not validated:
            # Check for candidate chunks with high token overlap
            best_sim = 0.0
            best_chunk_id = None
            for cs in chunk_spans:
                sim = token_jaccard_similarity(span_text, cs.page_content)
                if sim > best_sim:
                    best_sim = sim
                    best_chunk_id = cs.chunk_id

            if best_sim >= 0.85 and best_chunk_id is not None:
                status = "fuzzy_candidate"
                candidate_chunk_ids = [best_chunk_id]
                fuzzy_sim = best_sim

        if validated:
            validated_chunk_ids.update(matched_chunk_ids)
            validated_evidence_texts.append(span_text)

        span_results.append(
            SpanMappingResult(
                span_index=idx,
                span_text=span_text,
                span_start=span_start,
                span_end=span_end,
                status=status,
                validated=validated,
                matched_chunk_ids=matched_chunk_ids,
                candidate_chunk_ids=candidate_chunk_ids,
                fuzzy_similarity=fuzzy_sim,
            )
        )

    # Determine overall Question Mapping Status
    total_spans = len(span_results)
    validated_spans = sum(1 for s in span_results if s.validated)

    if validated_spans == total_spans:
        q_status = "fully_mapped"
    elif validated_spans > 0:
        q_status = "partially_mapped"
    else:
        q_status = "unmapped"

    return QuestionGroundTruth(
        question_id=question.id,
        category=question.category,
        is_impossible=False,
        mapping_status=q_status,
        span_results=span_results,
        validated_chunk_ids=validated_chunk_ids,
        all_evidence_text=" \n ".join(validated_evidence_texts),
    )
