"""Run global statistical tests across datasets."""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from defectinsight.model_registry import default_fast_model_names
from defectinsight.multidataset import run_cv, run_global_stats


def main():
    parser = argparse.ArgumentParser(description="Compare classifiers across datasets.")
    parser.add_argument("--datasets", nargs="*", default=[], help="Processed datasets used if CV results are missing.")
    parser.add_argument("--models", nargs="+", default=default_fast_model_names())
    parser.add_argument("--cv-results", default="results/cv_results/fold_metrics.csv")
    parser.add_argument("--cv-output", default="results/cv_results/")
    parser.add_argument("--output", default="results/cv_stats/")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=3)
    args = parser.parse_args()

    cv_results = Path(args.cv_results)
    if not cv_results.exists():
        if not args.datasets:
            raise SystemExit("CV results are missing. Pass --datasets or run experiments/run_cv.py first.")
        run_cv(
            args.datasets,
            model_names=args.models,
            out_dir=args.cv_output,
            n_splits=args.cv_splits,
            n_repeats=args.cv_repeats,
        )
        cv_results = Path(args.cv_output) / "fold_metrics.csv"

    run_global_stats(cv_results, out_dir=args.output, metric=args.metric)
    print(f"Global summary: {Path(args.output) / 'global_summary.txt'}")
    print(f"Pairwise tests: {Path(args.output) / 'global_pairwise_tests.csv'}")


if __name__ == "__main__":
    main()
