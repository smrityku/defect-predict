"""Explainability and diagnostic plot artifacts for DefectInsight."""

import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "defectinsight_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

warnings.filterwarnings("ignore")


def model_slug(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_confusion_matrices(models, X, y, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    payload = {}

    for name, model in models.items():
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        payload[name] = cm.tolist()
        rows.append(
            {
                "model": name,
                "true_clean_pred_clean": int(cm[0][0]),
                "true_clean_pred_defect": int(cm[0][1]),
                "true_defect_pred_clean": int(cm[1][0]),
                "true_defect_pred_defect": int(cm[1][1]),
            }
        )

        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Clean", "Defect"],
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        fig.savefig(out_dir / f"{model_slug(name)}.png", dpi=150)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(out_dir / "confusion_matrices.csv", index=False)
    _write_json(out_dir / "confusion_matrices.json", payload)
    return payload


def _importance_from_model(model, feature_names):
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        source = "feature_importances"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        source = "linear_coefficients"
    else:
        return None, None

    if len(values) != len(feature_names):
        return None, None

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "source": source,
        }
    ).sort_values("importance", ascending=False)
    return frame, source


def _plot_importance(frame, name, out_path, max_features=20):
    top = frame.head(max_features).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#2563EB")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_permutation_importance(model, X, y, name, out_dir, max_features=20):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=1)
    frame = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": result.importances_mean,
            "std": result.importances_std,
            "source": "permutation",
        }
    ).sort_values("importance", ascending=False)
    frame.to_csv(out_dir / f"permutation_{model_slug(name)}.csv", index=False)

    top = frame.head(max_features).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top["feature"],
        top["importance"],
        xerr=top["std"],
        color="#0F766E",
        ecolor="#134E4A",
    )
    ax.set_xlabel("Mean decrease in score")
    ax.set_title(f"Permutation Importance - {name}")
    plt.tight_layout()
    fig.savefig(out_dir / f"permutation_{model_slug(name)}.png", dpi=150)
    plt.close(fig)
    return frame


def _try_shap_summary(model, X, name, out_dir, max_samples=200):
    try:
        import shap
    except ImportError:
        return "shap is not installed."

    try:
        sample = X.sample(min(len(X), max_samples), random_state=42) if len(X) else X
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        if getattr(shap_values, "ndim", 0) == 3:
            shap_values = shap_values[:, :, -1]
        shap.summary_plot(shap_values, sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_{model_slug(name)}.png", dpi=150, bbox_inches="tight")
        plt.close()
        return None
    except Exception as exc:
        plt.close()
        return str(exc)


def save_feature_importance_artifacts(
    models,
    X,
    y,
    out_dir,
    max_features=20,
    include_permutation_for_unsupported=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = {}
    saved = {}
    feature_names = list(X.columns)

    for name, model in models.items():
        frame, source = _importance_from_model(model, feature_names)
        slug = model_slug(name)
        notes[name] = []

        if frame is not None:
            csv_path = out_dir / f"{slug}.csv"
            png_path = out_dir / f"{slug}.png"
            frame.to_csv(csv_path, index=False)
            _plot_importance(frame, name, png_path, max_features=max_features)
            saved[name] = {"importance": str(csv_path), "plot": str(png_path), "source": source}
        elif include_permutation_for_unsupported:
            try:
                frame = _plot_permutation_importance(model, X, y, name, out_dir, max_features)
                saved[name] = {
                    "importance": str(out_dir / f"permutation_{slug}.csv"),
                    "plot": str(out_dir / f"permutation_{slug}.png"),
                    "source": "permutation",
                }
            except Exception as exc:
                notes[name].append(f"permutation importance failed: {exc}")
        else:
            notes[name].append("model does not expose feature_importances_ or coef_.")

        if hasattr(model, "feature_importances_"):
            shap_error = _try_shap_summary(model, X, name, out_dir)
            if shap_error:
                notes[name].append(f"SHAP skipped: {shap_error}")
            else:
                saved.setdefault(name, {})["shap_plot"] = str(out_dir / f"shap_{slug}.png")

    _write_json(out_dir / "feature_importance_artifacts.json", saved)
    _write_json(out_dir / "explainability_notes.json", notes)
    return {"saved": saved, "notes": notes}
