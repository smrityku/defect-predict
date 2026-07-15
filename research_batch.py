"""
Batch project analysis for thesis/research comparison.

Runs the existing extraction -> feature -> preprocessing -> training ->
evaluation pipeline for many local repositories and produces aggregate CSV,
Markdown, HTML, and chart outputs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent.resolve()


def slugify(path):
    name = Path(path).name or "project"
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name).strip("-").lower()
    parent = re.sub(r"[^a-zA-Z0-9_.-]+", "-", Path(path).parent.name).strip("-").lower()
    return slug or parent or "project"


def run_cmd(cmd, cwd, timeout, env):
    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_sec": round(time.time() - started, 2),
    }


def is_git_repo(path):
    if not Path(path).is_dir():
        return False
    result = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=path,
                            capture_output=True, text=True)
    return result.returncode == 0


def count_source_files(path):
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0
    source_exts = {
        ".py", ".rb", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".cs",
        ".cpp", ".c", ".php", ".swift", ".kt", ".rs", ".scala", ".ex",
        ".exs", ".vue", ".html", ".css", ".scss", ".sass", ".sql", ".sh",
        ".erb", ".haml", ".slim", ".dart",
    }
    return sum(1 for line in result.stdout.splitlines()
               if Path(line.strip()).suffix.lower() in source_exts)


def analyze_project(project_path, out_root, max_files, timeout, fast):
    project_path = str(Path(project_path).expanduser())
    slug = slugify(project_path)
    project_out = out_root / slug
    data_dir = project_out / "data"
    model_dir = data_dir / "models"
    logs_dir = project_out / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "project": Path(project_path).name,
        "path": project_path,
        "slug": slug,
        "status": "pending",
        "source_files_at_head": 0,
        "max_files": max_files,
        "duration_sec": 0.0,
    }

    if not Path(project_path).exists():
        record.update(status="skipped", reason="path does not exist")
        return record
    if not Path(project_path).is_dir():
        record.update(status="skipped", reason="not a directory")
        return record
    if not is_git_repo(project_path):
        record.update(status="skipped", reason="not a git repository")
        return record

    record["source_files_at_head"] = count_source_files(project_path)
    if record["source_files_at_head"] == 0:
        record.update(status="skipped", reason="no supported source files")
        return record

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-defect")
    env.setdefault("LOKY_MAX_CPU_COUNT", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")

    steps = [
        ("extract", [
            sys.executable, str(BASE / "collector" / "git_extractor.py"),
            "--repo", project_path,
            "--output", str(data_dir / "raw.csv"),
            "--max-files", str(max_files),
        ]),
        ("features", [
            sys.executable, str(BASE / "features" / "feature_builder.py"),
            "--input", str(data_dir / "raw.csv"),
            "--output", str(data_dir / "features.csv"),
            "--repo", project_path,
        ]),
        ("preprocess", [
            sys.executable, str(BASE / "preprocessing" / "preprocess.py"),
            "--input", str(data_dir / "features.csv"),
            "--output-dir", str(data_dir),
        ]),
        ("train", [
            sys.executable, str(BASE / "models" / "train_compare.py"),
            "--data-dir", str(data_dir),
            "--model-dir", str(model_dir),
        ] + (["--fast"] if fast else [])),
        ("evaluate", [
            sys.executable, str(BASE / "evaluation" / "evaluate.py"),
            "--data-dir", str(data_dir),
            "--model-dir", str(model_dir),
            "--repo", project_path,
            "--dataset-name", Path(project_path).name,
            "--task-type", "traditional_file",
        ]),
    ]

    started = time.time()
    for step_name, cmd in steps:
        try:
            result = run_cmd(cmd, BASE, timeout, env)
        except subprocess.TimeoutExpired as exc:
            record.update(status="failed", reason=f"{step_name} timed out after {timeout}s")
            (logs_dir / f"{step_name}.stderr.txt").write_text(str(exc), encoding="utf-8")
            record["duration_sec"] = round(time.time() - started, 2)
            return record
        (logs_dir / f"{step_name}.stdout.txt").write_text(result["stdout"], encoding="utf-8")
        (logs_dir / f"{step_name}.stderr.txt").write_text(result["stderr"], encoding="utf-8")
        if result["returncode"] != 0:
            record.update(
                status="failed",
                reason=f"{step_name} failed",
                error_tail=(result["stderr"] or result["stdout"])[-600:],
            )
            record["duration_sec"] = round(time.time() - started, 2)
            return record

    record["duration_sec"] = round(time.time() - started, 2)
    record["status"] = "analyzed"

    try:
        results = json.loads((model_dir / "results.json").read_text(encoding="utf-8"))
        best = results.get("_best", "")
        metrics = results.get(best, {})
        pred = pd.read_csv(data_dir / "predictions.csv")
        raw = pd.read_csv(data_dir / "raw.csv")
        record.update({
            "files_analyzed": int(len(raw)),
            "prediction_rows": int(len(pred)),
            "defect_prone_files": int(raw["is_defect_prone"].sum()) if "is_defect_prone" in raw else None,
            "defect_ratio": round(float(raw["is_defect_prone"].mean()), 4) if "is_defect_prone" in raw and len(raw) else None,
            "best_model": best,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "mcc": metrics.get("mcc"),
            "critical_files": int((pred.get("risk_level", pd.Series(dtype=str)) == "Critical").sum()),
            "high_files": int((pred.get("risk_level", pd.Series(dtype=str)) == "High").sum()),
            "medium_files": int((pred.get("risk_level", pd.Series(dtype=str)) == "Medium").sum()),
            "low_files": int((pred.get("risk_level", pd.Series(dtype=str)) == "Low").sum()),
            "project_output": str(project_out),
        })
    except Exception as exc:
        record.update(status="failed", reason="could not summarize outputs", error_tail=str(exc))
    return record


def write_reports(records, out_root, max_files):
    df = pd.DataFrame(records)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "project_comparison.csv"
    json_path = out_root / "project_comparison.json"
    md_path = out_root / "project_comparison_report.md"
    html_path = out_root / "project_comparison_report.html"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    analyzed = df[df["status"] == "analyzed"].copy()
    skipped = df[df["status"] != "analyzed"].copy()
    lines = [
        "# Defect Prediction Project Comparison Report",
        "",
        f"Projects requested: {len(df)}",
        f"Projects analyzed: {len(analyzed)}",
        f"Projects skipped/failed: {len(skipped)}",
        f"Per-project source-file cap: {max_files}",
        "",
        "## Method Summary",
        "",
        "Each project was analyzed with the same defect-predictor pipeline: Git history extraction, static/process feature building, preprocessing, multiple model comparison, evaluation, line-risk explanation generation, and research-run storage. The current batch uses a file cap so all repositories can be compared consistently in a local run; full-paper final numbers should be regenerated without the cap for selected benchmark projects.",
        "",
    ]

    def md_table(frame):
        if frame.empty:
            return "_No rows._"
        safe = frame.fillna("")
        headers = list(safe.columns)
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in safe.iterrows():
            rows.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in headers) + " |")
        return "\n".join(rows)

    if not analyzed.empty:
        top = analyzed.sort_values(["f1", "roc_auc"], ascending=False).head(20)
        cols = ["project", "files_analyzed", "defect_ratio", "best_model", "f1", "roc_auc", "pr_auc", "mcc", "medium_files", "high_files", "critical_files"]
        lines.extend(["## Top Model Results", "", md_table(top[cols]), ""])

        risk = analyzed.sort_values(["critical_files", "high_files", "medium_files"], ascending=False).head(20)
        risk_cols = ["project", "prediction_rows", "critical_files", "high_files", "medium_files", "low_files", "defect_ratio"]
        lines.extend(["## Highest Risk Project Snapshots", "", md_table(risk[risk_cols]), ""])

    if not skipped.empty:
        cols = [c for c in ["project", "path", "status", "reason"] if c in skipped.columns]
        lines.extend(["## Skipped Or Failed Projects", "", md_table(skipped[cols].fillna("")), ""])

    lines.extend([
        "## Research Notes",
        "",
        "- `defect_ratio` is derived from commit-message keyword labelling in the local Git histories, so it is useful for project comparison but weaker than curated SZZ labels.",
        "- `f1`, `ROC-AUC`, `PR-AUC`, and `MCC` are better paper metrics than accuracy because defect data is imbalanced.",
        "- Line-level suggestions are explanation support over the file-level model, not independently supervised line-defect labels.",
        "- For publication, use this batch as a project-screening stage, then run full uncapped analysis and public datasets for the final experiment tables.",
        "",
        f"Raw CSV: `{csv_path}`",
        f"Raw JSON: `{json_path}`",
    ])
    md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")

    try:
        html = df.to_html(index=False, classes="table table-striped", border=0)
        html_path.write_text(
            "<html><head><title>Defect Prediction Comparison</title>"
            "<style>body{font-family:Arial,sans-serif;padding:24px;max-width:1400px;margin:auto}"
            "table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left}"
            "th{background:#f3f4f6}</style></head><body>"
            + md.replace("\n", "<br>")
            + "<h2>Full Raw Table</h2>"
            + html
            + "</body></html>",
            encoding="utf-8",
        )
    except Exception:
        pass
    return {"csv": str(csv_path), "json": str(json_path), "md": str(md_path), "html": str(html_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", required=True)
    parser.add_argument("--out-root", default=str(BASE / "data" / "batch_research"))
    parser.add_argument("--max-files", type=int, default=250)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    paths = [line.strip() for line in Path(args.paths_file).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    out_root = Path(args.out_root).resolve()
    records = []
    for idx, path in enumerate(paths, start=1):
        print(f"[{idx}/{len(paths)}] {path}", flush=True)
        record = analyze_project(path, out_root, args.max_files, args.timeout, args.fast)
        print(f"  -> {record['status']}: {record.get('reason', record.get('best_model', ''))}", flush=True)
        records.append(record)
        write_reports(records, out_root, args.max_files)
    report_paths = write_reports(records, out_root, args.max_files)
    print(json.dumps(report_paths, indent=2))


if __name__ == "__main__":
    main()
