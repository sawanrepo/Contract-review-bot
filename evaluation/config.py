import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"

@dataclass
class EvaluationConfig:
    dataset_path: str = os.path.join("CUAD_v1", "CUAD_v1", "CUAD_v1.json")
    txt_dir: str = os.path.join("CUAD_v1", "CUAD_v1", "full_contract_txt")
    pdf_dir: str = os.path.join("CUAD_v1", "CUAD_v1", "full_contract_pdf")
    output_dir: str = os.path.join("evaluation", "results")
    num_contracts: Optional[int] = 10  # None means evaluate all contracts
    seed: int = 42
    top_k: int = 10
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_model: str = DEFAULT_MODEL_NAME
    k_values: Tuple[int, ...] = (1, 3, 4, 5, 6, 10)
    prefer_txt: bool = True

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "txt_dir": self.txt_dir,
            "pdf_dir": self.pdf_dir,
            "output_dir": self.output_dir,
            "num_contracts": self.num_contracts if self.num_contracts is not None else "all",
            "seed": self.seed,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "k_values": list(self.k_values),
            "prefer_txt": self.prefer_txt,
        }

def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description="CUAD RAG Retrieval Evaluation Framework for Contract Review Bot"
    )
    parser.add_argument(
        "--num-contracts",
        type=str,
        default="10",
        help="Number of contracts to evaluate, e.g. 10, 20, 50, or 'all' (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible contract selection (default: 42)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.path.join("CUAD_v1", "CUAD_v1", "CUAD_v1.json"),
        help="Path to CUAD_v1.json",
    )
    parser.add_argument(
        "--txt-dir",
        type=str,
        default=os.path.join("CUAD_v1", "CUAD_v1", "full_contract_txt"),
        help="Path to full_contract_txt folder",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=os.path.join("CUAD_v1", "CUAD_v1", "full_contract_pdf"),
        help="Path to full_contract_pdf folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("evaluation", "results"),
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Max top-K chunks to retrieve (default: 10)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size for document splitter (default: 1200)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for document splitter (default: 200)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace embedding model name (default: {DEFAULT_MODEL_NAME})",
    )

    args = parser.parse_args()

    num_contracts_val: Optional[int]
    if str(args.num_contracts).lower() in ("all", "-1"):
        num_contracts_val = None
    else:
        num_contracts_val = int(args.num_contracts)

    return EvaluationConfig(
        dataset_path=args.dataset_path,
        txt_dir=args.txt_dir,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        num_contracts=num_contracts_val,
        seed=args.seed,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )
