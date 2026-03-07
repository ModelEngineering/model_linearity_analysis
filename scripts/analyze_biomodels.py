"""
Process all BioModels SBML files and write Jacobian CV results to a CSV.

Usage:
    source activate.sh
    python scripts/analyze_biomodels.py [--directory DIR] [--output FILE]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

EXCLUDED_MODELS = [
    "BIOMD0000000014",
    "BIOMD0000000019",
    "BIOMD0000000088",
    "BIOMD0000000205",
    "BIOMD0000000235",
    "BIOMD0000000255",
    "BIOMD0000000470",  # 2500 reactions
    "BIOMD0000000471",  # 2100 reactions
    "BIOMD0000000472",  # >2000 reactions
    "BIOMD0000000473",  # >2000reactions
]

import src.constants as cn  # noqa: E402
from linear_analyzer import LinearAnalyzer  # type: ignore

N_CLUSTER = 1
BIOMODELS_DIR = "/Users/jlheller/home/Technical/repos/temp-biomodels/final"
OUTPUT_FILENAME = f"{N_CLUSTER}_model_linearity_analysis_data.csv"
OUTPUT_PTH = os.path.join(cn.DATA_DIR, OUTPUT_FILENAME)


def main() -> None:
    """Parse arguments and run processBioModelsCVs over the BioModels directory."""
    parser = argparse.ArgumentParser(
        description="Compute Jacobian CV for all BioModel SBML files."
    )
    parser.add_argument(
        "--directory",
        default=BIOMODELS_DIR,
        help="Directory containing BioModel subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PTH,
        help="Path for the output CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--n_cluster",
        default=1,
        help="Number of clusters for k-means clustering (default: %(default)s)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Processing models in: {args.directory}")
    print(f"Output file: {args.output}")

    result_dct = LinearAnalyzer.processBioModelsCVs(
        directory=args.directory,
        output_data_file=args.output,
        excluded_models=EXCLUDED_MODELS,
        n_cluster=args.n_cluster,
    )

    n_success = len(result_dct)
    print(f"\nDone. Successfully processed {n_success} model(s).")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
