"""Framework-level orchestration for DefectInsight."""

import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "defectinsight_matplotlib")
)


class DefectInsightPipeline:
    """Run preprocessing, training, evaluation, CV, and explainability artifacts."""

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.models = {}

    def run(self, data=None):
        base_dir = Path(self.config.get("base_dir", Path(__file__).resolve().parents[1]))
        data_dir = Path(self.config.get("data_dir", base_dir / "data"))
        model_dir = Path(self.config.get("model_dir", data_dir / "models"))
        results_dir = Path(self.config.get("results_dir", base_dir / "results"))
        branch = self.config.get("branch", "HEAD")
        max_files = self.config.get("max_files")
        fast = bool(self.config.get("fast", False))
        repo = self.config.get("repo")
        dataset_name = self.config.get("dataset_name")
        task_type = self.config.get("task_type", "traditional_file")

        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        features_csv = self._prepare_features(data, repo, branch, max_files, data_dir)

        from preprocessing.preprocess import run as preprocess
        from models.train_compare import run as train_models
        from evaluation.evaluate import run as evaluate

        preprocess(str(features_csv), str(data_dir))
        trained, metrics, best_name = train_models(str(data_dir), str(model_dir), fast)
        self.models = trained

        evaluate(
            str(data_dir),
            str(model_dir),
            repo_path=repo,
            dataset_name=dataset_name,
            task_type=task_type,
        )

        artifacts = {
            "features_csv": str(features_csv),
            "data_dir": str(data_dir),
            "model_dir": str(model_dir),
            "results_dir": str(results_dir),
            "best_model": best_name,
            "metrics": metrics,
        }

        if not self.config.get("skip_cv", False):
            from defectinsight.cv_stats import run_repeated_cv

            artifacts["cv_stats"] = run_repeated_cv(
                str(features_csv),
                results_dir / "cv_stats",
                n_splits=int(self.config.get("cv_splits", 10)),
                n_repeats=int(self.config.get("cv_repeats", 3)),
                scoring=self.config.get("cv_scoring", "f1"),
            )

        if not self.config.get("skip_explainability", False):
            test = pd.read_csv(data_dir / "test.csv")
            feature_cols = [c for c in test.columns if c != "is_defect_prone"]
            X_test = test[feature_cols]
            y_test = test["is_defect_prone"]

            from defectinsight.explainability import (
                save_confusion_matrices,
                save_feature_importance_artifacts,
            )

            artifacts["confusion_matrices"] = save_confusion_matrices(
                trained, X_test, y_test, results_dir / "confusion_matrices"
            )
            artifacts["feature_importance"] = save_feature_importance_artifacts(
                trained, X_test, y_test, results_dir / "feature_importance"
            )

        return artifacts

    def _prepare_features(self, data, repo, branch, max_files, data_dir):
        features_csv = Path(self.config.get("features_csv", data_dir / "features.csv"))

        if data is not None:
            if isinstance(data, pd.DataFrame):
                features_csv.parent.mkdir(parents=True, exist_ok=True)
                data.to_csv(features_csv, index=False)
                return features_csv

            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Feature data not found: {data_path}")
            features_csv.parent.mkdir(parents=True, exist_ok=True)
            if data_path.resolve() != features_csv.resolve():
                shutil.copyfile(data_path, features_csv)
            return features_csv

        if self.config.get("features_csv"):
            if not features_csv.exists():
                raise FileNotFoundError(f"Feature data not found: {features_csv}")
            return features_csv

        if repo:
            from collector.git_extractor import run as extract_git
            from features.feature_builder import run as build_features

            raw_csv = data_dir / "raw.csv"
            extract_git(repo, str(raw_csv), branch, max_files)
            build_features(str(raw_csv), str(features_csv), repo)
            return features_csv

        if features_csv.exists():
            return features_csv

        raise ValueError("Provide --repo, --features-csv, or an existing data/features.csv.")
