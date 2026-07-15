"""Run repeated stratified CV across processed datasets."""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from defectinsight.model_registry import default_fast_model_names
from defectinsight.multidataset import run_cv


def main():
    parser = argparse.ArgumentParser(description="Run repeated stratified CV across datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Processed .pkl/.csv datasets or glob patterns.")
    parser.add_argument("--models", nargs="+", default=default_fast_model_names())
    parser.add_argument("--output", default="results/cv_results/")
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    result = run_cv(
        args.datasets,
        model_names=args.models,
        out_dir=args.output,
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        n_jobs=args.n_jobs,
    )
    print(f"Fold metrics: {Path(args.output) / 'fold_metrics.csv'}")
    print(f"Summary: {Path(args.output) / 'summary.csv'}")
    if not result["warnings"].empty:
        print(f"Warnings: {Path(args.output) / 'warnings.csv'}")


if __name__ == "__main__":
    main()
