"""Generate feature-importance artifacts for each processed dataset."""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import pandas as pd

from defectinsight.explainability import save_feature_importance_artifacts
from defectinsight.model_registry import default_fast_model_names, resolve_model_specs, slugify_model_name
from defectinsight.multidataset import build_estimator, expand_paths, load_dataset, prepare_xy


def transformed_frame(pipeline, X, original_cols):
    variance = pipeline.named_steps["variance"]
    mask = variance.get_support()
    cols = [col for col, keep in zip(original_cols, mask) if keep]
    values = pipeline[:-1].transform(X)
    return pd.DataFrame(values, columns=cols)


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP/coefficients/importance artifacts by dataset.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["random_forest", "logistic_regression"])
    parser.add_argument("--output", default="results/feature_importance/")
    parser.add_argument("--max-rows", type=int, default=5000, help="Sample rows per dataset for expensive explainability.")
    args = parser.parse_args()

    model_specs, skipped = resolve_model_specs(args.models)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in expand_paths(args.datasets):
        dataset, df = load_dataset(path)
        if args.max_rows and len(df) > args.max_rows:
            df = df.sample(n=args.max_rows, random_state=42)
        X, y, cols = prepare_xy(df)
        dataset_out = output / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        for model_name, spec in model_specs.items():
            pipeline = build_estimator(spec["est"])
            pipeline.fit(X, y)
            final_model = pipeline.named_steps["model"]
            X_model = transformed_frame(pipeline, X, cols)
            result = save_feature_importance_artifacts(
                {model_name: final_model},
                X_model,
                y.reset_index(drop=True),
                dataset_out / slugify_model_name(model_name),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "output": str(dataset_out / slugify_model_name(model_name)),
                    "notes": "; ".join(result["notes"].get(model_name, [])),
                }
            )

    pd.DataFrame(rows).to_csv(output / "feature_importance_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(output / "warnings.csv", index=False)
    print(f"Feature importance summary: {output / 'feature_importance_summary.csv'}")


if __name__ == "__main__":
    main()
