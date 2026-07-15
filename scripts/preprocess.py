"""
Batch-normalize public defect datasets into processed pickle files.

Example:
    python scripts/preprocess.py --input data/public/*.csv --output data/processed/
"""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import pandas as pd

from datasets.load_public import normalize_frame
from defectinsight.multidataset import expand_paths


def read_csv(path, backend="pandas"):
    if backend == "dask":
        try:
            import dask.dataframe as dd
        except ImportError as exc:
            raise ImportError("Dask backend requested but dask is not installed.") from exc
        return dd.read_csv(path).compute()
    return pd.read_csv(path)


def preprocess_one(path, output_dir, backend="pandas", target_col=None, task_type="traditional_file", write_csv=False):
    path = Path(path)
    dataset_name = path.stem
    df = read_csv(path, backend=backend)
    normalized = normalize_frame(
        df,
        dataset_name=dataset_name,
        task_type=task_type,
        target_col=target_col,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / f"{dataset_name}.pkl"
    normalized.to_pickle(pkl_path)
    csv_path = None
    if write_csv:
        csv_path = output_dir / f"{dataset_name}.csv"
        normalized.to_csv(csv_path, index=False)

    return {
        "dataset": dataset_name,
        "input": str(path),
        "output": str(pkl_path),
        "csv_output": str(csv_path) if csv_path else "",
        "rows": int(len(normalized)),
        "features": int(len(normalized.columns)),
        "defect_rate": float(normalized["is_defect_prone"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize multiple public defect datasets.")
    parser.add_argument("--input", nargs="+", required=True, help="CSV files or glob patterns.")
    parser.add_argument("--output", default="data/processed/", help="Processed dataset output directory.")
    parser.add_argument("--backend", choices=["pandas", "dask"], default="pandas")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--task-type", default="traditional_file", choices=["traditional_file", "jit_commit"])
    parser.add_argument("--write-csv", action="store_true", help="Also write normalized CSV files.")
    args = parser.parse_args()

    paths = expand_paths(args.input)
    if not paths:
        raise SystemExit("No input datasets matched.")

    rows = []
    for path in paths:
        row = preprocess_one(
            path,
            args.output,
            backend=args.backend,
            target_col=args.target_col,
            task_type=args.task_type,
            write_csv=args.write_csv,
        )
        rows.append(row)
        print(f"{row['dataset']}: {row['rows']} rows -> {row['output']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(Path(args.output) / "preprocess_summary.csv", index=False)
    print(f"Summary: {Path(args.output) / 'preprocess_summary.csv'}")


if __name__ == "__main__":
    main()
