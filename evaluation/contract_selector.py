import os
import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from evaluation.dataset_parser import CUADContract

@dataclass
class MatchedContract:
    contract: CUADContract
    file_path: str
    file_type: str  # "txt" or "pdf"
    match_method: str

def build_file_index(directory: str) -> Dict[str, str]:
    """
    Builds a map from normalized basename to full file path,
    recursing into subdirectories if present (e.g. for full_contract_pdf/Part_I).
    """
    index: Dict[str, str] = {}
    if not os.path.exists(directory):
        return index

    for root, _, files in os.walk(directory):
        for f in files:
            norm_name = unicodedata.normalize("NFC", f)
            index[norm_name] = os.path.join(root, f)
            index[f] = os.path.join(root, f)
    return index

def resolve_contract_file(
    title: str,
    txt_index: Dict[str, str],
    pdf_index: Dict[str, str],
    prefer_txt: bool = True,
) -> Optional[Tuple[str, str, str]]:
    """
    Attempts to locate the matching contract file.
    Returns (file_path, file_type, match_method) or None.
    """
    norm_title = unicodedata.normalize("NFC", title)
    txt_key = norm_title + ".txt"
    pdf_key = norm_title + ".pdf"

    if prefer_txt:
        if txt_key in txt_index:
            return txt_index[txt_key], "txt", "nfc_txt" if txt_key != (title + ".txt") else "exact_txt"
        if (title + ".txt") in txt_index:
            return txt_index[title + ".txt"], "txt", "exact_txt"
        if pdf_key in pdf_index:
            return pdf_index[pdf_key], "pdf", "nfc_pdf" if pdf_key != (title + ".pdf") else "exact_pdf"
        if (title + ".pdf") in pdf_index:
            return pdf_index[title + ".pdf"], "pdf", "exact_pdf"
    else:
        if pdf_key in pdf_index:
            return pdf_index[pdf_key], "pdf", "nfc_pdf" if pdf_key != (title + ".pdf") else "exact_pdf"
        if (title + ".pdf") in pdf_index:
            return pdf_index[title + ".pdf"], "pdf", "exact_pdf"
        if txt_key in txt_index:
            return txt_index[txt_key], "txt", "nfc_txt" if txt_key != (title + ".txt") else "exact_txt"
        if (title + ".txt") in txt_index:
            return txt_index[title + ".txt"], "txt", "exact_txt"

    return None

def select_contracts(
    all_contracts: List[CUADContract],
    num_contracts: Optional[int] = 10,
    seed: int = 42,
    txt_dir: str = "",
    pdf_dir: str = "",
    prefer_txt: bool = True,
) -> Tuple[List[MatchedContract], List[str]]:
    """
    Selects contracts deterministically and matches them against disk files.
    Returns (matched_contracts, unmatched_titles).
    """
    txt_index = build_file_index(txt_dir)
    pdf_index = build_file_index(pdf_dir)

    # Sort all contracts by title for deterministic initial ordering
    sorted_contracts = sorted(all_contracts, key=lambda c: c.title)

    rng = random.Random(seed)
    # Shuffle contracts deterministically
    shuffled = list(sorted_contracts)
    rng.shuffle(shuffled)

    target_count = len(shuffled) if num_contracts is None else min(num_contracts, len(shuffled))
    candidates = shuffled[:target_count]

    matched: List[MatchedContract] = []
    unmatched_titles: List[str] = []

    for c in candidates:
        resolution = resolve_contract_file(c.title, txt_index, pdf_index, prefer_txt=prefer_txt)
        if resolution is not None:
            path, ftype, method = resolution
            matched.append(
                MatchedContract(
                    contract=c,
                    file_path=path,
                    file_type=ftype,
                    match_method=method,
                )
            )
        else:
            unmatched_titles.append(c.title)

    return matched, unmatched_titles
