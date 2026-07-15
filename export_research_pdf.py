"""
Export the batch research comparison into a shareable multipage PDF.
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


BASE = Path(__file__).parent.resolve()
OUT_ROOT = BASE / "data" / "batch_research"
PDF_PATH = OUT_ROOT / "defect_prediction_project_comparison_report.pdf"


def add_text_page(pdf, title, paragraphs, footer=None):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 0.94
    title_lines = textwrap.wrap(title, width=36)
    for title_line in title_lines:
        ax.text(0.08, y, title_line, fontsize=18, fontweight="bold", va="top", color="#1A202C")
        y -= 0.035
    y -= 0.025
    for para in paragraphs:
        if para.startswith("## "):
            y -= 0.015
            ax.text(0.08, y, para[3:], fontsize=14, fontweight="bold", va="top", color="#2D3748")
            y -= 0.035
            continue
        wrapped = textwrap.wrap(para, width=92) or [""]
        for line in wrapped:
            ax.text(0.08, y, line, fontsize=10.5, va="top", color="#2D3748")
            y -= 0.022
        y -= 0.012
        if y < 0.08:
            break
    if footer:
        ax.text(0.08, 0.04, footer, fontsize=8.5, color="#718096")
    pdf.savefig(fig)
    plt.close(fig)


def add_chart_page(pdf, image_path, title):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
    ax.axis("off")
    fig.text(0.04, 0.96, title, fontsize=17, fontweight="bold", va="top", color="#1A202C")
    img = plt.imread(image_path)
    ax.imshow(img)
    pdf.savefig(fig)
    plt.close(fig)


def add_table_pages(pdf, df, title, columns, rows_per_page=22, font_size=7.5):
    data = df[columns].copy().fillna("")
    for col in data.columns:
        if data[col].dtype == "float64":
            data[col] = data[col].map(lambda v: f"{v:.4g}" if v != "" else "")
        else:
            limits = {
                "project": 24,
                "filepath": 58,
                "path": 58,
                "best_model": 24,
                "reason": 34,
            }
            limit = limits.get(col)
            if limit:
                data[col] = data[col].astype(str).map(
                    lambda v: v if len(v) <= limit else v[: max(limit - 1, 1)] + "..."
                )
    total_pages = max(1, (len(data) + rows_per_page - 1) // rows_per_page)
    for page_idx, start in enumerate(range(0, len(data), rows_per_page), start=1):
        chunk = data.iloc[start:start + rows_per_page]
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.84])
        ax.axis("off")
        fig.text(0.03, 0.96, f"{title} ({page_idx}/{total_pages})", fontsize=15, fontweight="bold", va="top")
        weights = []
        for col in chunk.columns:
            if col == "filepath":
                weights.append(3.8)
            elif col in ("project", "path"):
                weights.append(2.3)
            elif col == "best_model":
                weights.append(2.0)
            elif col in ("reason", "status", "risk_level", "language"):
                weights.append(1.4)
            else:
                weights.append(1.05)
        total_weight = sum(weights)
        col_widths = [w / total_weight for w in weights]
        table = ax.table(
            cellText=chunk.astype(str).values,
            colLabels=list(chunk.columns),
            loc="upper left",
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0, 1, 1],
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#CBD5E0")
            cell.set_linewidth(0.4)
            if row == 0:
                cell.set_facecolor("#E2E8F0")
                cell.set_text_props(fontweight="bold", color="#2D3748")
            elif row % 2 == 0:
                cell.set_facecolor("#F7FAFC")
        pdf.savefig(fig)
        plt.close(fig)


def main():
    comparison = pd.read_csv(OUT_ROOT / "project_comparison.csv")
    risky_files = pd.read_csv(OUT_ROOT / "top_risky_files.csv")
    analyzed = comparison[comparison["status"] == "analyzed"].copy()
    skipped = comparison[comparison["status"] != "analyzed"].copy()

    charts = [
        ("charts/top_f1_projects.png", "Top 20 Projects by F1 Score"),
        ("charts/top_roc_auc_projects.png", "Top 20 Projects by ROC-AUC"),
        ("charts/best_model_frequency.png", "Best Model Frequency Across Projects"),
        ("charts/risk_distribution_top_projects.png", "Risk Distribution for Top Projects"),
        ("charts/highest_defect_ratio_projects.png", "Highest Keyword-Labelled Defect Ratio"),
    ]

    with PdfPages(PDF_PATH) as pdf:
        add_text_page(pdf, "Defect Prediction Multi-Project Research Comparison", [
            "## Abstract-Style Summary",
            f"This report analyzes {len(analyzed)} local Git repositories out of {len(comparison)} requested project paths using a uniform defect-prediction pipeline. Each repository was sampled up to 250 source files, transformed into static and process metrics, evaluated across multiple machine-learning classifiers, and summarized with imbalance-aware metrics.",
            "## Experimental Setup",
            "Prediction unit: file-level defect proneness. Label source: bug/fix keyword heuristic from Git commit messages. Feature families: churn/history, authorship, static size, complexity, comment density, nesting, and language features.",
            "Model families: Logistic Regression, Random Forest, Gradient Boosting, Hist Gradient Boosting, Extra Trees, AdaBoost, SVM, Naive Bayes, MLP, Voting Ensemble, and Stacking Ensemble.",
            "Primary metrics: F1, ROC-AUC, PR-AUC, MCC, balanced accuracy, precision, and recall. Line-risk explanations are developer-support explanations over model-ranked files, not independently validated line-defect labels.",
            "## Key Batch Counts",
            f"Analyzed: {len(analyzed)} projects. Skipped or failed: {len(skipped)} projects. Generated outputs include aggregate CSV/JSON, top risky files, charts, and per-project artifacts.",
        ], footer=f"Generated from {OUT_ROOT}")

        for rel, title in charts:
            path = OUT_ROOT / rel
            if path.exists():
                add_chart_page(pdf, path, title)

        top_cols = ["project", "files_analyzed", "defect_ratio", "best_model", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"]
        top = analyzed.sort_values(["f1", "roc_auc"], ascending=False)
        add_table_pages(pdf, top, "Main Project Comparison", top_cols, rows_per_page=24, font_size=7.2)

        risk_cols = ["project", "prediction_rows", "critical_files", "high_files", "medium_files", "low_files", "defect_ratio"]
        risk = analyzed.sort_values(["critical_files", "high_files", "medium_files"], ascending=False)
        add_table_pages(pdf, risk, "Risk Distribution by Project", risk_cols, rows_per_page=24, font_size=7.5)

        file_cols = ["project", "filepath", "language", "defect_probability", "maintenance_score", "risk_level"]
        top_files = risky_files.sort_values(["project", "maintenance_score"], ascending=[True, False])
        add_table_pages(pdf, top_files.head(160), "Top Risky Files Per Project", file_cols, rows_per_page=18, font_size=6.0)

        if not skipped.empty:
            skip_cols = [c for c in ["project", "status", "reason", "path"] if c in skipped.columns]
            add_table_pages(pdf, skipped, "Skipped and Failed Inputs", skip_cols, rows_per_page=24, font_size=6.8)

        add_text_page(pdf, "Interpretation and Threats to Validity", [
            "## Interpretation",
            "Projects with high F1 and ROC-AUC are useful candidates for case-study examples because their historical labels produce separable patterns. Logistic Regression appearing frequently as a best model supports a common defect-prediction finding: simpler metric-based models can remain competitive against heavier learners.",
            "## Threats to Validity",
            "The current labels are generated from commit-message keywords, which are weaker than curated SZZ labels. The 250-file cap makes the batch feasible and comparable, but final publication-quality experiments should rerun selected repositories uncapped and include public benchmark datasets.",
            "Line-risk suggestions should be presented as developer-support explanations, not as independently supervised line-level defect predictions unless line-labelled data is added.",
            "## Recommended Next Step",
            "Use this PDF as a screening/reporting artifact. For thesis tables, select representative projects from this batch, rerun them without a cap, and add public PROMISE or ApacheJIT-style datasets.",
        ])

    print(PDF_PATH)


if __name__ == "__main__":
    main()
