"""
Multi-dataset DefectInsight orchestration.

Example:
    python run_all.py --datasets data/processed/*.pkl --models random_forest svm logistic_regression xgboost
"""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from defectinsight.model_registry import default_fast_model_names
from defectinsight.multidataset import (
    expand_paths,
    run_cross_dataset_generalization,
    run_cv,
    run_global_stats,
    run_learning_curves,
)
from defectinsight.reports import generate_reports


def main():
    parser = argparse.ArgumentParser(description="Run DefectInsight across multiple public datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Processed .pkl/.csv datasets or glob patterns.")
    parser.add_argument("--models", nargs="+", default=default_fast_model_names())
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-feature-importance", action="store_true")
    parser.add_argument("--skip-confusion", action="store_true")
    parser.add_argument("--skip-cross-dataset", action="store_true")
    parser.add_argument("--learning-curves", action="store_true")
    parser.add_argument("--importance-max-rows", type=int, default=5000)
    args = parser.parse_args()

    dataset_paths = expand_paths(args.datasets)
    if not dataset_paths:
        raise SystemExit("No datasets matched.")

    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Running repeated stratified CV...")
    run_cv(
        [str(path) for path in dataset_paths],
        model_names=args.models,
        out_dir=results_dir / "cv_results",
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        n_jobs=args.n_jobs,
    )

    print("Running global statistical tests...")
    run_global_stats(results_dir / "cv_results" / "fold_metrics.csv", out_dir=results_dir / "cv_stats")

    if not args.skip_feature_importance:
        print("Generating feature-importance artifacts...")
        from experiments.run_feature_importance import main as feature_main

        sys.argv = [
            "experiments/run_feature_importance.py",
            "--datasets",
            *[str(path) for path in dataset_paths],
            "--models",
            *args.models,
            "--output",
            str(results_dir / "feature_importance"),
            "--max-rows",
            str(args.importance_max_rows),
        ]
        feature_main()

    if not args.skip_confusion:
        print("Generating per-fold confusion matrices...")
        from experiments.run_confusion import main as confusion_main

        sys.argv = [
            "experiments/run_confusion.py",
            "--datasets",
            *[str(path) for path in dataset_paths],
            "--models",
            *args.models,
            "--output",
            str(results_dir / "confusion_matrices"),
            "--cv-splits",
            str(args.cv_splits),
            "--cv-repeats",
            str(args.cv_repeats),
        ]
        confusion_main()

    if not args.skip_cross_dataset and len(dataset_paths) >= 2:
        print("Running cross-dataset generalization...")
        run_cross_dataset_generalization(
            [str(path) for path in dataset_paths],
            model_names=args.models,
            out_dir=results_dir / "cross_dataset",
        )

    if args.learning_curves:
        print("Generating learning curves...")
        run_learning_curves(
            [str(path) for path in dataset_paths],
            model_names=args.models,
            out_dir=results_dir / "learning_curves",
        )

    print("Generating advanced reports...")
    reports = generate_reports(results_dir=results_dir, reports_dir=reports_dir)
    print("Done.")
    print(f"  CV results: {results_dir / 'cv_results'}")
    print(f"  Statistical tests: {results_dir / 'cv_stats'}")
    print(f"  Reports: {reports_dir}")
    for name, path in reports.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
