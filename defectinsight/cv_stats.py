"""Repeated cross-validation and statistical tests for DefectInsight."""

import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from models.train_compare import MODELS
from preprocessing.preprocess import NUMERIC_FEATURES, TARGET

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _model_slug(name):
    return name.lower().replace(" ", "_")


def load_feature_matrix(features_csv):
    """Load a publication-safe feature matrix from the normalized feature CSV."""
    df = pd.read_csv(features_csv)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    lang_cols = [c for c in df.columns if c.startswith("lang_")]
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + lang_cols
    if not feature_cols:
        raise ValueError("No usable numeric or language feature columns were found.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]
    return X, y, feature_cols


def make_cv(n_splits, n_repeats, y, random_state=42):
    counts = y.value_counts()
    if len(counts) < 2:
        return None, {
            "status": "skipped",
            "reason": "Repeated stratified CV requires at least two target classes.",
            "class_counts": counts.to_dict(),
        }

    effective_splits = min(int(n_splits), int(counts.min()))
    if effective_splits < 2:
        return None, {
            "status": "skipped",
            "reason": "Each class must have at least two samples for stratified CV.",
            "class_counts": counts.to_dict(),
        }

    cv = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=int(n_repeats),
        random_state=random_state,
    )
    return cv, {
        "status": "ready",
        "requested_splits": int(n_splits),
        "effective_splits": int(effective_splits),
        "n_repeats": int(n_repeats),
        "class_counts": counts.to_dict(),
    }


def build_cv_estimator(estimator):
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scale", RobustScaler()),
            ("model", clone(estimator)),
        ]
    )


def run_repeated_cv(
    features_csv,
    out_dir,
    n_splits=10,
    n_repeats=3,
    scoring="f1",
    random_state=42,
):
    """Run repeated stratified CV and save scores plus statistical tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = load_feature_matrix(features_csv)
    cv, cv_meta = make_cv(n_splits, n_repeats, y, random_state=random_state)
    cv_meta["features_csv"] = str(features_csv)
    cv_meta["n_rows"] = int(len(X))
    cv_meta["n_features"] = int(len(feature_cols))
    _write_json(out_dir / "cv_metadata.json", cv_meta)

    if cv is None:
        pd.DataFrame().to_csv(out_dir / "cv_scores.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "cv_summary.csv", index=False)
        _write_json(out_dir / "statistical_tests.json", {"status": "skipped", **cv_meta})
        return {"status": "skipped", "metadata": cv_meta}

    scorer = scoring
    if scoring == "f1":
        scorer = make_scorer(f1_score, zero_division=0)

    scores_by_model = {}
    score_rows = []
    summary_rows = []

    for name, spec in MODELS.items():
        estimator = build_cv_estimator(spec["est"])
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scorer, n_jobs=1)
        scores_by_model[name] = scores
        summary_rows.append(
            {
                "model": name,
                "scoring": scoring,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "n_scores": int(len(scores)),
            }
        )
        for fold_idx, score in enumerate(scores, start=1):
            score_rows.append(
                {
                    "model": name,
                    "fold_index": fold_idx,
                    "scoring": scoring,
                    "score": float(score),
                }
            )

    scores_df = pd.DataFrame(score_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("mean", ascending=False)
    scores_df.to_csv(out_dir / "cv_scores.csv", index=False)
    summary_df.to_csv(out_dir / "cv_summary.csv", index=False)
    _write_json(out_dir / "cv_summary.json", summary_df.to_dict(orient="records"))

    test_payload = save_statistical_tests(scores_by_model, out_dir)
    return {
        "status": "ok",
        "metadata": cv_meta,
        "summary": summary_df.to_dict(orient="records"),
        "tests": test_payload,
    }


def save_statistical_tests(scores_by_model, out_dir):
    out_dir = Path(out_dir)
    names = list(scores_by_model.keys())
    wilcoxon_rows = []

    for left, right in combinations(names, 2):
        left_scores = np.asarray(scores_by_model[left])
        right_scores = np.asarray(scores_by_model[right])
        try:
            stat, p_value = wilcoxon(left_scores, right_scores, zero_method="zsplit")
            note = ""
        except ValueError as exc:
            stat, p_value = np.nan, np.nan
            note = str(exc)
        wilcoxon_rows.append(
            {
                "model_a": left,
                "model_b": right,
                "statistic": None if np.isnan(stat) else float(stat),
                "p_value": None if np.isnan(p_value) else float(p_value),
                "mean_a": float(np.mean(left_scores)),
                "mean_b": float(np.mean(right_scores)),
                "mean_difference_a_minus_b": float(np.mean(left_scores - right_scores)),
                "note": note,
            }
        )

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_df.to_csv(out_dir / "wilcoxon_pairwise.csv", index=False)

    payload = {
        "status": "ok",
        "wilcoxon_pairwise": wilcoxon_rows,
        "friedman": None,
        "nemenyi": None,
        "notes": [],
    }

    if len(names) >= 3:
        arrays = [np.asarray(scores_by_model[name]) for name in names]
        try:
            stat, p_value = friedmanchisquare(*arrays)
            payload["friedman"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "models": names,
            }
        except ValueError as exc:
            payload["friedman"] = {"error": str(exc), "models": names}

        try:
            import scikit_posthocs as sp

            matrix = pd.DataFrame({name: scores_by_model[name] for name in names})
            nemenyi = sp.posthoc_nemenyi_friedman(matrix.values)
            nemenyi.index = names
            nemenyi.columns = names
            nemenyi.to_csv(out_dir / "nemenyi_posthoc.csv")
            payload["nemenyi"] = str(out_dir / "nemenyi_posthoc.csv")
        except ImportError:
            payload["notes"].append(
                "scikit-posthocs is not installed; Nemenyi post-hoc table was skipped."
            )
        except Exception as exc:
            payload["notes"].append(f"Nemenyi post-hoc table failed: {exc}")
    else:
        payload["notes"].append("Friedman/Nemenyi tests require at least three models.")

    _write_json(out_dir / "statistical_tests.json", payload)
    return payload
