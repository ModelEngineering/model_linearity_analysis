"""
Process all BioModels SBML files and write Jacobian CV results to a CSV.

Usage:
    source activate.sh
    python scripts/analyze_biomodels.py [--directory DIR] [--output FILE]
"""
import src.constants as cn  # noqa: E402

import argparse
import os


EXCLUDED_MODELS = [
    "BIOMD0000000014",
    "BIOMD0000000019",
    "BIOMD0000000088",
    "BIOMD0000000205",
    "BIOMD0000000235",
    "BIOMD0000000255",
    "BIOMD0000000293",
    "BIOMD0000000469",  # 2500 reactions
    "BIOMD0000000470",  # 2500 reactions
    "BIOMD0000000471",  # 2100 reactions
    "BIOMD0000000472",  # >2000 reactions
    "BIOMD0000000473",  # >2000reactions
    "BIOMD0000000574",
]

import src.constants as cn  # noqa: E402
from linear_analyzer import LinearAnalyzer  # type: ignore

N_CLUSTER = 5
OUTPUT_FILENAME = f"{N_CLUSTER}_model_linearity_analysis_data.csv"
OUTPUT_PTH = os.path.join(cn.DATA_DIR, OUTPUT_FILENAME)
NO_PATH = "no_path"


def main() -> None:
    """Parse arguments and run processBioModelsCVs over the BioModels directory."""
    parser = argparse.ArgumentParser(
        description="Compute Jacobian CV for all BioModel SBML files."
    )
    parser.add_argument(
        "--directory",
        default=cn.BIOMODELS_DIR,
        help="Directory containing BioModel subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=NO_PATH,
        help="Path for the output CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--n_cluster",
        default=N_CLUSTER,
        type=int,
        help="Number of clusters for k-means clustering (default: %(default)s)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Use sequential (contiguous) partitioning instead of k-means (default: False)",
    )
    args = parser.parse_args()

    if args.output == NO_PATH:
        if args.sequential:
            output_filename = f"{args.n_cluster}s_model_linearity_analysis_data_sequential.csv"
        else:
            output_filename = f"{args.n_cluster}_model_linearity_analysis_data.csv"
        output_path = os.path.join(cn.DATA_DIR, output_filename)
    else:
        output_path = args.output

    print("\n**********************************************************")
    print(f"***Processing models in: {args.directory}")
    print(f"***Output path: {output_path}")
    print(f"***Number of clusters: {args.n_cluster}")
    print(f"***Using sequential partitioning: {args.sequential}")
    print("**********************************************************\n")

    result_dct = LinearAnalyzer.partitionBiomodelsJacobians(
        directory=args.directory,
        output_data_file=output_path,
        excluded_models=EXCLUDED_MODELS,
        n_cluster=args.n_cluster,
        is_sequential_partition=args.sequential,
    )

    n_success = len(result_dct)
    print(f"\nDone. Successfully processed {n_success} model(s).")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
