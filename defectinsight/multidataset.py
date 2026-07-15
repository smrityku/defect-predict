"""Multi-dataset experiment utilities for DefectInsight."""

import json
import math
import os
import tempfile
import time
import glob
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, t, wilcoxon
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import warnings

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "defectinsight_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing.preprocess import NUMERIC_FEATURES, TARGET
from .model_registry import resolve_model_specs, slugify_model_name

warnings.filterwarnings("ignore", category=ConvergenceWarning)


META_COLUMNS = {
    TARGET,
    "filepath",
    "file_name",
    "filename",
    "directory",
    "file_ext",
    "language",
    "first_seen",
    "last_seen",
    "dataset_name",
    "task_type",
    "project",
    "project_name",
    "commit_hash",
    "hash",
    "revision",
    "commit",
}

LEAKAGE_COLUMNS = {
    "bug",
    "bugs",
    "buggy",
    "defect",
    "defects",
    "defective",
    "defect_label",
    "defect_commits",
    "defect_rate",
    "defect_density",
    "fault",
    "faults",
    "label",
    "target",
}

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def dataset_name_from_path(path):
    return Path(path).stem


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
        matches = [Path(match) for match in matches]
        paths.extend(matches)
    unique = []
    seen = set()
    for path in paths:
        resolved = str(Path(path))
        if resolved not in seen:
            unique.append(Path(path))
            seen.add(resolved)
    return unique


def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        payload = pd.read_pickle(path)
        if isinstance(payload, dict) and "data" in payload:
            df = payload["data"]
            name = payload.get("dataset_name", path.stem)
        else:
            df = payload
            name = path.stem
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        name = path.stem
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

    if TARGET not in df.columns:
        raise ValueError(f"{path} is missing target column '{TARGET}'")

    if "dataset_name" in df.columns and df["dataset_name"].notna().any():
        name = str(df["dataset_name"].dropna().iloc[0])
    return name, df.copy()


def feature_columns(df):
    lang_cols = [c for c in df.columns if c.startswith("lang_")]
    known = [c for c in NUMERIC_FEATURES if c in df.columns]
    numeric = [
        c
        for c in df.columns
        if c not in known
        and c not in META_COLUMNS
        and c.lower() not in LEAKAGE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    cols = known + lang_cols + numeric
    seen = set()
    ordered = []
    for col in cols:
        if col not in seen and col in df.columns and col.lower() not in LEAKAGE_COLUMNS:
            ordered.append(col)
            seen.add(col)
    if not ordered:
        raise ValueError("No usable feature columns were found.")
    return ordered


def prepare_xy(df, selected_features=None):
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    cols = list(selected_features) if selected_features is not None else feature_columns(df)
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]
    return X, y, cols


def build_estimator(estimator):
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scale", RobustScaler()),
            ("model", clone(estimator)),
        ]
    )


def effective_cv(y, n_splits, n_repeats, random_state=42):
    counts = y.value_counts()
    if len(counts) < 2:
        return None, {"status": "skipped", "reason": "requires at least two target classes"}
    splits = min(int(n_splits), int(counts.min()))
    if splits < 2:
        return None, {"status": "skipped", "reason": "least populated class has fewer than two rows"}
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=int(n_repeats), random_state=random_state)
    return cv, {"status": "ready", "effective_splits": splits, "n_repeats": int(n_repeats)}


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            return proba[:, 0]
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
        return proba[:, positive_index]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=float)
        return (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    return model.predict(X)


def metric_payload(y_true, y_pred, y_score):
    has_two_classes = pd.Series(y_true).nunique() > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_two_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_two_classes else np.nan,
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_two_classes else 0.0,
    }


def confidence_interval(values, confidence=0.95):
    values = pd.Series(values).dropna().astype(float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if n == 1:
        return mean, mean, mean
    sem = float(values.std(ddof=1) / math.sqrt(n))
    margin = float(t.ppf((1 + confidence) / 2.0, n - 1) * sem)
    return mean, mean - margin, mean + margin


def summarize_fold_metrics(fold_df):
    rows = []
    for (dataset, model), group in fold_df.groupby(["dataset", "model"], sort=True):
        row = {"dataset": dataset, "model": model, "n_folds": int(len(group))}
        for metric in METRICS:
            mean, low, high = confidence_interval(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{metric}_ci95_low"] = max(0.0, low) if not np.isnan(low) else low
            row[f"{metric}_ci95_high"] = min(1.0, high) if not np.isnan(high) else high
        row["fit_seconds_mean"] = float(group["fit_seconds"].mean())
        row["predict_seconds_mean"] = float(group["predict_seconds"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "f1_mean"], ascending=[True, False])


def run_cv_for_dataset(path, model_specs, n_splits=10, n_repeats=3, random_state=42):
    dataset, df = load_dataset(path)
    X, y, cols = prepare_xy(df)
    cv, meta = effective_cv(y, n_splits, n_repeats, random_state=random_state)
    if cv is None:
        return [], [], [{"dataset": dataset, **meta}], []

    fold_rows = []
    prediction_rows = []
    warning_rows = []

    for model_name, spec in model_specs.items():
        for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            repeat = ((split_idx - 1) // meta["effective_splits"]) + 1
            fold = ((split_idx - 1) % meta["effective_splits"]) + 1
            estimator = build_estimator(spec["est"])

            fit_start = time.perf_counter()
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = time.perf_counter() - fit_start

            pred_start = time.perf_counter()
            X_test = X.iloc[test_idx]
            y_true = y.iloc[test_idx]
            y_pred = estimator.predict(X_test)
            y_score = positive_scores(estimator, X_test)
            predict_seconds = time.perf_counter() - pred_start

            metrics = metric_payload(y_true, y_pred, y_score)
            fold_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "split_index": split_idx,
                    "repeat": repeat,
                    "fold": fold,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    **metrics,
                }
            )
            for row_idx, yt, yp, ys in zip(test_idx, y_true, y_pred, y_score):
                prediction_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "split_index": split_idx,
                        "row_index": int(row_idx),
                        "actual": int(yt),
                        "predicted": int(yp),
                        "score": float(ys),
                    }
                )

    meta_rows = [
        {
            "dataset": dataset,
            "path": str(path),
            "rows": int(len(df)),
            "features": int(len(cols)),
            "class_0": int((y == 0).sum()),
            "class_1": int((y == 1).sum()),
            **meta,
        }
    ]
    return fold_rows, prediction_rows, meta_rows, warning_rows


def run_cv(datasets, model_names=None, out_dir="results/cv_results", n_splits=10, n_repeats=3, n_jobs=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = expand_paths(datasets)
    model_specs, skipped = resolve_model_specs(model_names)

    if int(n_jobs) == 1:
        outputs = [
            run_cv_for_dataset(path, model_specs, n_splits=n_splits, n_repeats=n_repeats)
            for path in paths
        ]
    else:
        tasks = [
            joblib.delayed(run_cv_for_dataset)(path, model_specs, n_splits=n_splits, n_repeats=n_repeats)
            for path in paths
        ]
        outputs = joblib.Parallel(n_jobs=int(n_jobs))(tasks)

    fold_rows, prediction_rows, meta_rows, warning_rows = [], [], [], []
    for folds, preds, meta, warnings_ in outputs:
        fold_rows.extend(folds)
        prediction_rows.extend(preds)
        meta_rows.extend(meta)
        warning_rows.extend(warnings_)
    warning_rows.extend(skipped)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)
    meta_df = pd.DataFrame(meta_rows)
    warning_df = pd.DataFrame(warning_rows)
    summary_df = summarize_fold_metrics(fold_df) if not fold_df.empty else pd.DataFrame()

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    pred_df.to_csv(out_dir / "fold_predictions.csv", index=False)
    meta_df.to_csv(out_dir / "dataset_metadata.csv", index=False)
    warning_df.to_csv(out_dir / "warnings.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    write_json(out_dir / "summary.json", summary_df.to_dict(orient="records"))
    return {"fold_metrics": fold_df, "summary": summary_df, "metadata": meta_df, "warnings": warning_df}


def cohen_d(a, b):
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd and not np.isnan(sd) else 0.0


def cliffs_delta(a, b):
    a = pd.Series(a).dropna().astype(float).to_numpy()
    b = pd.Series(b).dropna().astype(float).to_numpy()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    greater = sum(x > y for x in a for y in b)
    lesser = sum(x < y for x in a for y in b)
    return float((greater - lesser) / (len(a) * len(b)))


def run_global_stats(cv_results_csv, out_dir="results/cv_stats", metric="f1"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.read_csv(cv_results_csv)
    per_dataset = (
        fold_df.groupby(["dataset", "model"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "score"})
    )
    matrix = per_dataset.pivot(index="dataset", columns="model", values="score").dropna(axis=1, how="any")
    matrix.to_csv(out_dir / "global_model_scores.csv")

    model_summary = (
        per_dataset.groupby("model")["score"]
        .agg(["mean", "std", "count", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    model_summary.to_csv(out_dir / "global_model_summary.csv", index=False)

    pairwise = []
    for left, right in combinations(matrix.columns, 2):
        left_scores = matrix[left]
        right_scores = matrix[right]
        try:
            stat, p_value = wilcoxon(left_scores, right_scores, zero_method="zsplit")
            note = ""
        except ValueError as exc:
            stat, p_value = np.nan, np.nan
            note = str(exc)
        pairwise.append(
            {
                "model_a": left,
                "model_b": right,
                "metric": metric,
                "wilcoxon_statistic": None if np.isnan(stat) else float(stat),
                "p_value": None if np.isnan(p_value) else float(p_value),
                "cohen_d_paired": cohen_d(left_scores, right_scores),
                "cliffs_delta": cliffs_delta(left_scores, right_scores),
                "mean_a": float(left_scores.mean()),
                "mean_b": float(right_scores.mean()),
                "note": note,
            }
        )
    pairwise_df = pd.DataFrame(pairwise)
    pairwise_df.to_csv(out_dir / "global_pairwise_tests.csv", index=False)

    stats_payload = {"metric": metric, "datasets": list(matrix.index), "models": list(matrix.columns), "friedman": None}
    if matrix.shape[1] >= 3 and matrix.shape[0] >= 2:
        stat, p_value = friedmanchisquare(*[matrix[col].values for col in matrix.columns])
        stats_payload["friedman"] = {"statistic": float(stat), "p_value": float(p_value)}
        try:
            import scikit_posthocs as sp

            nemenyi = sp.posthoc_nemenyi_friedman(matrix.values)
            nemenyi.index = matrix.columns
            nemenyi.columns = matrix.columns
            nemenyi.to_csv(out_dir / "nemenyi_global.csv")
            stats_payload["nemenyi"] = "nemenyi_global.csv"
        except ImportError:
            stats_payload["nemenyi"] = "skipped: scikit-posthocs is not installed"
    else:
        stats_payload["note"] = "Friedman/Nemenyi require at least two datasets and three complete models."

    write_json(out_dir / "global_stats.json", stats_payload)
    write_global_summary(out_dir / "global_summary.txt", model_summary, pairwise_df, stats_payload)
    return {"summary": model_summary, "pairwise": pairwise_df, "stats": stats_payload}


def write_global_summary(path, summary_df, pairwise_df, stats_payload):
    lines = [
        "DefectInsight Global Statistical Summary",
        "=" * 44,
        f"Metric: {stats_payload.get('metric', 'f1')}",
        "",
        "Model ranking:",
    ]
    for _, row in summary_df.iterrows():
        lines.append(f"- {row['model']}: mean={row['mean']:.4f}, std={row['std']:.4f}, datasets={int(row['count'])}")
    friedman = stats_payload.get("friedman")
    lines.append("")
    if friedman:
        lines.append(f"Friedman: statistic={friedman['statistic']:.4f}, p={friedman['p_value']:.6f}")
    else:
        lines.append(stats_payload.get("note", "Friedman test unavailable."))
    lines.append("")
    lines.append("Pairwise Wilcoxon and effect sizes are saved in global_pairwise_tests.csv.")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_cross_dataset_generalization(datasets, model_names=None, out_dir="results/cross_dataset"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = [load_dataset(path) for path in expand_paths(datasets)]
    model_specs, skipped = resolve_model_specs(model_names)
    rows = []
    warning_rows = list(skipped)

    for (train_name, train_df), (test_name, test_df) in combinations(loaded, 2):
        for source_name, source_df, target_name, target_df in [
            (train_name, train_df, test_name, test_df),
            (test_name, test_df, train_name, train_df),
        ]:
            common = sorted(set(feature_columns(source_df)) & set(feature_columns(target_df)))
            if not common:
                continue
            X_train, y_train, _ = prepare_xy(source_df, common)
            X_test, y_test, _ = prepare_xy(target_df, common)
            for model_name, spec in model_specs.items():
                estimator = build_estimator(spec["est"])
                started = time.perf_counter()
                try:
                    estimator.fit(X_train, y_train)
                    y_pred = estimator.predict(X_test)
                    y_score = positive_scores(estimator, X_test)
                except ValueError as exc:
                    warning_rows.append(
                        {
                            "train_dataset": source_name,
                            "test_dataset": target_name,
                            "model": model_name,
                            "reason": str(exc),
                        }
                    )
                    continue
                metrics = metric_payload(y_test, y_pred, y_score)
                rows.append(
                    {
                        "train_dataset": source_name,
                        "test_dataset": target_name,
                        "model": model_name,
                        "n_features": int(len(common)),
                        "train_size": int(len(X_train)),
                        "test_size": int(len(X_test)),
                        "seconds": float(time.perf_counter() - started),
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "generalization.csv", index=False)
    pd.DataFrame(warning_rows).to_csv(out_dir / "warnings.csv", index=False)
    return df


def run_learning_curves(datasets, model_names=None, out_dir="results/learning_curves", train_sizes=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_sizes = train_sizes or [0.1, 0.25, 0.5, 0.75, 1.0]
    model_specs, skipped = resolve_model_specs(model_names)
    rows = []

    for path in expand_paths(datasets):
        dataset, df = load_dataset(path)
        X, y, _ = prepare_xy(df)
        if y.nunique() < 2 or y.value_counts().min() < 2:
            continue
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y))
        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        for model_name, spec in model_specs.items():
            for train_size in train_sizes:
                n = max(2, int(len(X_train_full) * float(train_size)))
                n = min(n, len(X_train_full))
                sample = y_train_full.sample(n=n, random_state=int(float(train_size) * 1000) + 42)
                if sample.nunique() < 2:
                    sample = y_train_full
                X_train = X_train_full.loc[sample.index]
                y_train = y_train_full.loc[sample.index]
                estimator = build_estimator(spec["est"])
                started = time.perf_counter()
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                y_score = positive_scores(estimator, X_test)
                metrics = metric_payload(y_test, y_pred, y_score)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "train_fraction": float(train_size),
                        "train_size": int(len(X_train)),
                        "test_size": int(len(X_test)),
                        "seconds": float(time.perf_counter() - started),
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "learning_curve.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "warnings.csv", index=False)
    return df
