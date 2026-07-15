"""
Run the full DefectInsight experiment workflow.

Examples:
    python experiments/run_all.py --repo /path/to/git/repo --fast
    python experiments/run_all.py --features-csv data/features.csv --fast
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "defectinsight_matplotlib")
)

from defectinsight import DefectInsightPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DefectInsight preprocessing, training, CV statistics, and explainability."
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--repo", help="Local Git repository to analyze.")
    source.add_argument("--features-csv", help="Existing normalized feature CSV to analyze.")
    parser.add_argument("--branch", default="HEAD")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="Skip model grid search.")
    parser.add_argument("--data-dir", default=str(BASE / "data"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--results-dir", default=str(BASE / "results"))
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--task-type", default="traditional_file")
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--cv-scoring", default="f1")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-explainability", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir) if args.model_dir else data_dir / "models"

    config = {
        "base_dir": BASE,
        "repo": args.repo,
        "branch": args.branch,
        "max_files": args.max_files,
        "fast": args.fast,
        "data_dir": data_dir,
        "model_dir": model_dir,
        "results_dir": Path(args.results_dir),
        "dataset_name": args.dataset_name,
        "task_type": args.task_type,
        "cv_splits": args.cv_splits,
        "cv_repeats": args.cv_repeats,
        "cv_scoring": args.cv_scoring,
        "skip_cv": args.skip_cv,
        "skip_explainability": args.skip_explainability,
    }

    pipeline = DefectInsightPipeline(config)
    artifacts = pipeline.run(data=args.features_csv)

    print("\nDefectInsight experiment complete")
    print(f"  Best model          : {artifacts['best_model']}")
    print(f"  Data directory      : {artifacts['data_dir']}")
    print(f"  Model directory     : {artifacts['model_dir']}")
    print(f"  Results directory   : {artifacts['results_dir']}")
    if "cv_stats" in artifacts:
        print(f"  CV stats            : {Path(artifacts['results_dir']) / 'cv_stats'}")
    if "feature_importance" in artifacts:
        print(f"  Feature importance  : {Path(artifacts['results_dir']) / 'feature_importance'}")
    if "confusion_matrices" in artifacts:
        print(f"  Confusion matrices  : {Path(artifacts['results_dir']) / 'confusion_matrices'}")


if __name__ == "__main__":
    main()
