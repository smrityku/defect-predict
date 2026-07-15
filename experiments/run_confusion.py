"""Generate confusion matrices for every repeated-CV fold."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from defectinsight.model_registry import default_fast_model_names, resolve_model_specs, slugify_model_name
from defectinsight.multidataset import build_estimator, effective_cv, expand_paths, load_dataset, prepare_xy


def main():
    parser = argparse.ArgumentParser(description="Generate per-fold confusion matrices.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["random_forest"])
    parser.add_argument("--output", default="results/confusion_matrices/")
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--cv-repeats", type=int, default=3)
    args = parser.parse_args()

    model_specs, skipped = resolve_model_specs(args.models)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in expand_paths(args.datasets):
        dataset, df = load_dataset(path)
        X, y, _ = prepare_xy(df)
        cv, meta = effective_cv(y, args.cv_splits, args.cv_repeats)
        if cv is None:
            rows.append({"dataset": dataset, "note": meta["reason"]})
            continue

        for model_name, spec in model_specs.items():
            model_slug = slugify_model_name(model_name)
            model_out = output / dataset / model_slug
            model_out.mkdir(parents=True, exist_ok=True)
            for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                repeat = ((split_idx - 1) // meta["effective_splits"]) + 1
                fold = ((split_idx - 1) % meta["effective_splits"]) + 1
                estimator = build_estimator(spec["est"])
                estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_true = y.iloc[test_idx]
                y_pred = estimator.predict(X.iloc[test_idx])
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                fig, ax = plt.subplots(figsize=(4.5, 4))
                disp = ConfusionMatrixDisplay(cm, display_labels=["Clean", "Defect"])
                disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
                ax.set_title(f"{dataset} - {model_name} R{repeat} F{fold}")
                plt.tight_layout()
                plot_path = model_out / f"repeat_{repeat:02d}_fold_{fold:02d}.png"
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)

                rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "true_clean_pred_clean": int(cm[0][0]),
                        "true_clean_pred_defect": int(cm[0][1]),
                        "true_defect_pred_clean": int(cm[1][0]),
                        "true_defect_pred_defect": int(cm[1][1]),
                        "plot": str(plot_path),
                    }
                )

    pd.DataFrame(rows).to_csv(output / "fold_confusion_matrices.csv", index=False)
    pd.DataFrame(skipped).to_csv(output / "warnings.csv", index=False)
    print(f"Confusion matrix summary: {output / 'fold_confusion_matrices.csv'}")


if __name__ == "__main__":
    main()
