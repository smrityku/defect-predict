"""
Research data store for paper-ready experiment comparison.

The store is intentionally file-based so it works in free/demo environments and
can be committed or exported with the thesis replication package.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_json(path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def register_experiment(data_dir, model_dir, repo_path=None, dataset_name=None, task_type="traditional_file"):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    store_dir = data_dir / "research"
    results_path = model_dir / "results.json"
    predictions_path = data_dir / "predictions.csv"

    if not results_path.exists() or not predictions_path.exists():
        return None

    results = _load_json(results_path, {})
    pred = pd.read_csv(predictions_path)
    best = results.get("_best", "")
    best_metrics = results.get(best, {}) if best else {}
    risk_counts = pred["risk_level"].value_counts().to_dict() if "risk_level" in pred.columns else {}

    project_name = dataset_name
    if not project_name and repo_path:
        project_name = Path(repo_path).name
    if not project_name:
        project_name = "unknown_project"

    record = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "repo_path": str(repo_path or ""),
        "dataset_name": dataset_name or project_name,
        "task_type": task_type,
        "files_or_records": int(len(pred)),
        "best_model": best,
        "best_metrics": {
            key: best_metrics.get(key)
            for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
            if key in best_metrics
        },
        "risk_counts": risk_counts,
        "all_model_metrics": {
            name: {
                key: metrics.get(key)
                for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
                if key in metrics
            }
            for name, metrics in results.items()
            if not name.startswith("_")
        },
    }

    experiments_path = store_dir / "experiments.json"
    experiments = _load_json(experiments_path, [])
    experiments.append(record)
    _write_json(experiments_path, experiments)

    rows = []
    for exp in experiments:
        row = {
            "run_id": exp["run_id"],
            "created_at": exp["created_at"],
            "project_name": exp["project_name"],
            "dataset_name": exp["dataset_name"],
            "task_type": exp["task_type"],
            "files_or_records": exp["files_or_records"],
            "best_model": exp["best_model"],
        }
        row.update({f"best_{k}": v for k, v in exp["best_metrics"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(store_dir / "experiments_summary.csv", index=False)
    return record

