"""
Generate paper-style aggregate charts and comparison tables from batch output.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


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


def save_bar(df, x, y, path, title, xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df[x], df[y], color="#3182CE")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel or y)
    ax.set_ylabel(ylabel or x)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_model_frequency(df, path):
    counts = df["best_model"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts.index, counts.values, color="#38A169")
    ax.set_title("Best Model Frequency Across Analyzed Projects")
    ax.set_xlabel("Projects")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_risk_stack(df, path):
    cols = ["low_files", "medium_files", "high_files", "critical_files"]
    top = df.assign(total_risk=df[["medium_files", "high_files", "critical_files"]].sum(axis=1))
    top = top.sort_values("total_risk", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(13, 7))
    left = None
    colors = {"low_files": "#38A169", "medium_files": "#D69E2E", "high_files": "#DD6B20", "critical_files": "#E53E3E"}
    for col in cols:
        ax.barh(top["project"], top[col], left=left, label=col.replace("_files", "").title(), color=colors[col])
        left = top[col] if left is None else left + top[col]
    ax.invert_yaxis()
    ax.set_title("Risk Distribution For Top 20 Projects By Risky Prediction Rows")
    ax.set_xlabel("Prediction rows")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def collect_top_files(out_root, analyzed):
    rows = []
    for _, project in analyzed.iterrows():
        pred_path = Path(project["project_output"]) / "data" / "predictions.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        if pred.empty:
            continue
        for _, item in pred.sort_values("maintenance_score", ascending=False).head(5).iterrows():
            rows.append({
                "project": project["project"],
                "filepath": item.get("filepath", ""),
                "language": item.get("language", ""),
                "defect_probability": round(float(item.get("defect_probability", 0)), 4),
                "maintenance_score": round(float(item.get("maintenance_score", 0)), 2),
                "risk_level": item.get("risk_level", ""),
            })
    top_files = pd.DataFrame(rows)
    top_files.to_csv(out_root / "top_risky_files.csv", index=False)
    return top_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="data/batch_research")
    args = parser.parse_args()

    out_root = Path(args.out_root).resolve()
    df = pd.read_csv(out_root / "project_comparison.csv")
    analyzed = df[df["status"] == "analyzed"].copy()
    skipped = df[df["status"] != "analyzed"].copy()
    charts = out_root / "charts"
    charts.mkdir(exist_ok=True)

    if not analyzed.empty:
        save_bar(
            analyzed.sort_values(["f1", "roc_auc"], ascending=False).head(20),
            "project", "f1", charts / "top_f1_projects.png",
            "Top 20 Projects By F1 Score", xlabel="F1 score",
        )
        save_bar(
            analyzed.sort_values("roc_auc", ascending=False).head(20),
            "project", "roc_auc", charts / "top_roc_auc_projects.png",
            "Top 20 Projects By ROC-AUC", xlabel="ROC-AUC",
        )
        save_bar(
            analyzed.sort_values("defect_ratio", ascending=False).head(20),
            "project", "defect_ratio", charts / "highest_defect_ratio_projects.png",
            "Top 20 Projects By Keyword-Labelled Defect Ratio", xlabel="Defect ratio",
        )
        save_model_frequency(analyzed, charts / "best_model_frequency.png")
        save_risk_stack(analyzed, charts / "risk_distribution_top_projects.png")

    top_files = collect_top_files(out_root, analyzed)

    lines = [
        "# Defect Prediction Multi-Project Research Comparison",
        "",
        "## Abstract-Style Summary",
        "",
        f"This study analyzed {len(analyzed)} local Git repositories out of {len(df)} requested project paths using a uniform defect-prediction pipeline. Each repository was sampled up to the configured file cap, transformed into static and process metrics, evaluated across multiple machine-learning classifiers, and summarized with imbalance-aware metrics such as F1, ROC-AUC, PR-AUC, MCC, and balanced accuracy.",
        "",
        "## Experimental Setup",
        "",
        "- Unit of prediction: file-level defect proneness.",
        "- Label source: bug/fix keyword heuristic from Git commit messages.",
        "- Feature families: churn/history, authorship, static code size, complexity, comment density, nesting, language one-hot features.",
        "- Model families: Logistic Regression, Random Forest, Gradient Boosting, Hist Gradient Boosting, Extra Trees, AdaBoost, SVM, Naive Bayes, MLP, Voting Ensemble, Stacking Ensemble.",
        "- Evaluation artifacts: per-project predictions, model metrics, line-risk explanations, aggregate CSV/JSON, charts, and this report.",
        "",
        "## Charts",
        "",
        "![Top F1 Projects](charts/top_f1_projects.png)",
        "",
        "![Top ROC-AUC Projects](charts/top_roc_auc_projects.png)",
        "",
        "![Best Model Frequency](charts/best_model_frequency.png)",
        "",
        "![Risk Distribution](charts/risk_distribution_top_projects.png)",
        "",
        "![Highest Defect Ratio](charts/highest_defect_ratio_projects.png)",
        "",
    ]

    if not analyzed.empty:
        cols = ["project", "files_analyzed", "defect_ratio", "best_model", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"]
        lines.extend(["## Main Comparison Table", "", md_table(analyzed.sort_values(["f1", "roc_auc"], ascending=False)[cols]), ""])
        risk_cols = ["project", "prediction_rows", "critical_files", "high_files", "medium_files", "low_files"]
        lines.extend(["## Risk Distribution Table", "", md_table(analyzed.sort_values(["critical_files", "high_files", "medium_files"], ascending=False)[risk_cols]), ""])

    if not top_files.empty:
        lines.extend(["## Top Risky Files Per Project", "", md_table(top_files.head(120)), ""])

    if not skipped.empty:
        cols = [c for c in ["project", "path", "status", "reason"] if c in skipped.columns]
        lines.extend(["## Skipped And Failed Inputs", "", md_table(skipped[cols]), ""])

    lines.extend([
        "## Interpretation For Thesis Writing",
        "",
        "- Projects with high F1 and ROC-AUC are good candidates for case-study examples because the historical labels produce separable patterns.",
        "- Projects with low scores, one-class test splits, or preprocessing failures should be discussed as threats to validity and data-quality limitations.",
        "- Logistic Regression appearing frequently as a best model supports a common defect-prediction finding: simpler metric-based models can remain competitive against heavier learners.",
        "- Keyword-labelled defect data is useful for tooling and screening, but final publication claims should include public benchmark datasets or SZZ-labelled JIT datasets.",
        "- Line-risk explanations should be framed as developer-support explanations over model-ranked files, not as independently validated line-defect predictions.",
        "",
        "## Output Files",
        "",
        f"- Aggregate CSV: `{out_root / 'project_comparison.csv'}`",
        f"- Aggregate JSON: `{out_root / 'project_comparison.json'}`",
        f"- Top risky files CSV: `{out_root / 'top_risky_files.csv'}`",
        f"- Batch folders: `{out_root}`",
    ])

    report = "\n".join(lines)
    md_path = out_root / "paper_style_comparison_report.md"
    md_path.write_text(report, encoding="utf-8")
    html = report
    for image in charts.glob("*.png"):
        html = html.replace(f"![{image.stem}]({image.relative_to(out_root)})", f"<img src='{image.relative_to(out_root)}' style='max-width:100%;border:1px solid #ddd'>")
    html_path = out_root / "paper_style_comparison_report.html"
    html_path.write_text(
        "<html><head><title>Defect Prediction Research Comparison</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:auto;padding:28px;line-height:1.5}"
        "table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left}"
        "th{background:#f3f4f6}code{background:#f3f4f6;padding:2px 4px;border-radius:4px}</style></head><body>"
        + html.replace("\n", "<br>")
        + "</body></html>",
        encoding="utf-8",
    )
    print(md_path)
    print(html_path)


if __name__ == "__main__":
    main()

