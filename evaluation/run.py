import os
import sys
import time

# Ensure offline operation and avoid HF network timeouts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.config import parse_args
from evaluation.contract_selector import select_contracts
from evaluation.dataset_parser import load_cuad_dataset
from evaluation.report_generator import generate_reports
from evaluation.retrieval_benchmark import run_retrieval_benchmark

def main():
    config = parse_args()

    print("=================================================================")
    print(" CUAD RAG Retrieval Evaluation Framework - Contract Review Bot ")
    print("=================================================================")
    print(f"Dataset path:      {config.dataset_path}")
    print(f"TXT directory:     {config.txt_dir}")
    print(f"PDF directory:     {config.pdf_dir}")
    print(f"Output directory:  {config.output_dir}")
    print(f"Target contracts:  {config.num_contracts if config.num_contracts is not None else 'ALL (510)'}")
    print(f"Random seed:       {config.seed}")
    print(f"Top-K:             {config.top_k}")
    print(f"Embedding model:   {config.embedding_model}")
    print("-----------------------------------------------------------------")

    # 1. Load dataset
    t0 = time.time()
    print("Loading CUAD v1 dataset JSON...")
    all_contracts = load_cuad_dataset(config.dataset_path)
    print(f"Loaded {len(all_contracts)} total contracts from CUAD JSON in {time.time() - t0:.2f}s.")

    # 2. Select contracts
    matched, unmatched = select_contracts(
        all_contracts=all_contracts,
        num_contracts=config.num_contracts,
        seed=config.seed,
        txt_dir=config.txt_dir,
        pdf_dir=config.pdf_dir,
        prefer_txt=config.prefer_txt,
    )

    print(f"Selected {len(matched)} contracts successfully matched with local contract files.")
    if unmatched:
        print(f"WARNING: {len(unmatched)} contract(s) could not be matched with local files:")
        for u in unmatched:
            print(f"   - {u}")
        print("These unmatched contracts have been excluded from this run.")

    if not matched:
        print("ERROR: No matched contracts available for evaluation. Exiting.")
        sys.exit(1)

    # 3. Run Benchmark
    benchmark_start = time.time()
    suite_result = run_retrieval_benchmark(matched, config)
    total_benchmark_time = time.time() - benchmark_start

    # 4. Generate Reports & Visual Charts
    print(f"\nGenerating reports and visual charts into: {config.output_dir} ...")
    generate_reports(suite_result, config.output_dir)

    print("\n=================================================================")
    print(f" Benchmark Completed Successfully in {total_benchmark_time:.2f}s")
    print(f" Reports saved to: {os.path.abspath(config.output_dir)}")
    print("=================================================================")

if __name__ == "__main__":
    main()
