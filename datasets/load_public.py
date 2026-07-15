"""
Public dataset loader for thesis experiments.

Normalizes PROMISE/NASA-style traditional datasets and JIT commit datasets into
the same feature schema used by the existing preprocessing/training pipeline.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


TARGET_ALIASES = [
    "is_defect_prone", "defective", "bug", "bugs", "defects", "fault",
    "faults", "label", "target", "defect_label", "buggy", "contains_bug",
    "is_buggy", "problems", "c",
]

META_ALIASES = {
    "filepath": ["filepath", "file", "file_name", "filename", "path", "module", "class"],
    "language": ["language", "lang"],
    "commit_hash": ["commit_hash", "hash", "revision", "commit"],
    "project": ["project", "project_name", "repo", "repository"],
}


def _find_col(columns, aliases):
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _binary_target(series):
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) > 0).astype(int)
    values = series.astype(str).str.lower().str.strip()
    return values.isin(["1", "true", "yes", "y", "bug", "buggy", "defect", "defective", "faulty"]).astype(int)


def normalize_frame(df, dataset_name=None, task_type="traditional_file", target_col=None):
    target_col = target_col or _find_col(df.columns, TARGET_ALIASES)
    if not target_col:
        raise ValueError("Could not find target column. Pass --target-col explicitly.")

    out = pd.DataFrame()
    out["is_defect_prone"] = _binary_target(df[target_col])

    for canonical, aliases in META_ALIASES.items():
        col = _find_col(df.columns, aliases)
        if col:
            out[canonical] = df[col]

    if "filepath" not in out.columns:
        out["filepath"] = [f"record_{i}" for i in range(len(df))]
    if "filename" not in out.columns:
        out["filename"] = out["filepath"].astype(str).map(lambda p: Path(p).name)
    if "directory" not in out.columns:
        out["directory"] = out["filepath"].astype(str).map(lambda p: str(Path(p).parent))
    if "file_ext" not in out.columns:
        out["file_ext"] = out["filepath"].astype(str).map(lambda p: Path(p).suffix.lower())
    if "language" not in out.columns:
        out["language"] = "Unknown"

    ignore = {target_col}
    ignore.update(c for c in df.columns if c.lower() in {alias.lower() for alias in TARGET_ALIASES})
    ignore.update(c for aliases in META_ALIASES.values() for c in aliases)
    numeric_cols = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in numeric_cols:
        out[col] = df[col].fillna(df[col].median())

    # Fill expected feature names when public datasets use different metric sets.
    defaults = {
        "total_commits": 1, "total_authors": 1, "total_churn": 0,
        "avg_churn": 0, "max_churn": 0, "lines_added": 0, "lines_deleted": 0,
        "avg_loc": 0, "max_loc": 0, "min_loc": 0, "loc_var": 0,
        "avg_cc": 0, "sum_cc": 0, "churn_per_loc": 0, "add_del_ratio": 1,
        "loc_range": 0, "cyclomatic_complexity": 1, "num_functions": 0,
        "num_classes": 0, "comment_lines": 0, "blank_lines": 0,
        "code_lines": 0, "comment_ratio": 0, "avg_func_complexity": 0,
        "max_nesting": 0, "long_methods": 0, "complexity_per_loc": 0,
        "churn_density": 0, "author_diversity": 1, "commit_frequency": 0,
        "defect_commits": 0, "first_seen": "", "last_seen": "",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value

    out["dataset_name"] = dataset_name or "public_dataset"
    out["task_type"] = task_type

    return out


def normalize(input_csv, output_csv, dataset_name=None, task_type="traditional_file", target_col=None):
    df = pd.read_csv(input_csv)
    out = normalize_frame(
        df,
        dataset_name=dataset_name or Path(input_csv).stem,
        task_type=task_type,
        target_col=target_col,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Loaded {len(out)} records from {input_csv}")
    print(f"Defect-prone: {int(out['is_defect_prone'].sum())} ({out['is_defect_prone'].mean()*100:.1f}%)")
    print(f"Output: {output_csv}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="../data/features.csv")
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--task-type", default="traditional_file", choices=["traditional_file", "jit_commit"])
    ap.add_argument("--target-col", default=None)
    args = ap.parse_args()
    normalize(args.input, args.output, args.dataset_name, args.task_type, args.target_col)
