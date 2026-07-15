"""Advanced report generation for multi-dataset DefectInsight runs."""

import html
import os
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "defectinsight_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _read_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _text_page(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.06, 0.95, title, fontsize=18, weight="bold", va="top")
    y = 0.9
    for line in lines:
        ax.text(0.06, y, line, fontsize=10, va="top", wrap=True)
        y -= 0.032
        if y < 0.06:
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def performance_summary_pdf(cv_summary_csv, output_pdf, cross_dataset_csv=None, learning_curve_csv=None):
    df = _read_csv(cv_summary_csv)
    cross = _read_csv(cross_dataset_csv) if cross_dataset_csv else pd.DataFrame()
    learning = _read_csv(learning_curve_csv) if learning_curve_csv else pd.DataFrame()
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        if df.empty:
            _text_page(pdf, "DefectInsight Performance Summary", ["No CV summary data was found."])
            return

        top = df.sort_values("f1_mean", ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        labels = top["dataset"] + " / " + top["model"]
        ax.barh(labels.iloc[::-1], top["f1_mean"].iloc[::-1], color="#2563EB")
        ax.set_xlabel("Mean F1")
        ax.set_title("Top Dataset/Model Combinations")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        lines = [
            f"Datasets: {df['dataset'].nunique()}",
            f"Models: {df['model'].nunique()}",
            f"Rows in summary: {len(df)}",
            "",
            "This report includes mean, standard deviation, and 95% confidence interval fields in results/cv_results/summary.csv.",
        ]
        _text_page(pdf, "Performance Notes", lines)

        if not cross.empty:
            top_cross = cross.sort_values("f1", ascending=False).head(20)
            lines = [
                f"{row.train_dataset} -> {row.test_dataset} / {row.model}: F1={row.f1:.4f}, MCC={row.mcc:.4f}"
                for row in top_cross.itertuples()
            ]
            _text_page(pdf, "Cross-dataset Generalization", lines)

        if not learning.empty:
            best_learning = learning.sort_values(["dataset", "model", "train_fraction"])
            lines = [
                f"{row.dataset} / {row.model} / train={row.train_fraction:.2f}: F1={row.f1:.4f}"
                for row in best_learning.head(60).itertuples()
            ]
            _text_page(pdf, "Learning Curves", lines)


def statistical_tests_pdf(stats_dir, output_pdf):
    stats_dir = Path(stats_dir)
    summary = (stats_dir / "global_summary.txt").read_text(encoding="utf-8") if (stats_dir / "global_summary.txt").exists() else ""
    pairwise = _read_csv(stats_dir / "global_pairwise_tests.csv")
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        _text_page(pdf, "DefectInsight Statistical Tests", summary.splitlines() or ["No statistical summary was found."])
        if not pairwise.empty:
            top = pairwise.sort_values("p_value", na_position="last").head(20)
            lines = [
                f"{row.model_a} vs {row.model_b}: p={row.p_value}, Cohen d={row.cohen_d_paired:.3f}, Cliff delta={row.cliffs_delta:.3f}"
                for row in top.itertuples()
            ]
            _text_page(pdf, "Lowest Pairwise p-values", lines)


def scalability_analysis_pdf(cv_summary_csv, fold_metrics_csv, output_pdf):
    summary = _read_csv(cv_summary_csv)
    folds = _read_csv(fold_metrics_csv)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        if folds.empty:
            _text_page(pdf, "DefectInsight Scalability Analysis", ["No runtime data was found."])
            return
        runtime = folds.groupby(["dataset", "model"], as_index=False)[["fit_seconds", "predict_seconds"]].mean()
        runtime["total_seconds"] = runtime["fit_seconds"] + runtime["predict_seconds"]
        top = runtime.sort_values("total_seconds", ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        labels = top["dataset"] + " / " + top["model"]
        ax.barh(labels.iloc[::-1], top["total_seconds"].iloc[::-1], color="#0F766E")
        ax.set_xlabel("Mean seconds per fold")
        ax.set_title("Slowest Dataset/Model Fold Runtimes")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        lines = [
            f"Total CV folds: {len(folds)}",
            f"Total fit time recorded: {folds['fit_seconds'].sum():.2f}s",
            f"Total prediction time recorded: {folds['predict_seconds'].sum():.2f}s",
        ]
        if not summary.empty:
            lines.append(f"Best average F1: {summary['f1_mean'].max():.4f}")
        _text_page(pdf, "Runtime Notes", lines)


def explainability_dashboard_html(feature_root, confusion_root, output_html):
    feature_root = Path(feature_root)
    confusion_root = Path(confusion_root)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    datasets = set()
    for root in [feature_root, confusion_root]:
        if root.exists():
            for img in root.rglob("*.png"):
                rel = img.relative_to(root)
                if rel.parts:
                    datasets.add(rel.parts[0])

    sections = []
    for dataset in sorted(datasets):
        images = []
        for root in [feature_root / dataset, confusion_root / dataset]:
            if root.exists():
                for img in sorted(root.rglob("*.png"))[:30]:
                    rel = os.path.relpath(img, output_html.parent)
                    images.append(f"<figure><img src='{html.escape(rel)}'><figcaption>{html.escape(img.name)}</figcaption></figure>")
        if images:
            sections.append(f"<section><h2>{html.escape(dataset)}</h2><div class='grid'>{''.join(images)}</div></section>")

    body = "\n".join(sections) or "<p>No explainability images were found.</p>"
    output_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DefectInsight Explainability Dashboard</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#172033;background:#f8fafc}}
h1{{margin-bottom:4px}} h2{{margin-top:28px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
figure{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin:0}}
img{{width:100%;height:auto}} figcaption{{font-size:12px;color:#475569;margin-top:8px}}
</style>
</head>
<body>
<h1>DefectInsight Explainability Dashboard</h1>
<p>SHAP, feature ranking, permutation importance, and confusion-matrix artifacts by dataset.</p>
{body}
</body></html>
""",
        encoding="utf-8",
    )


def generate_reports(results_dir="results", reports_dir="reports"):
    results_dir = Path(results_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    performance_summary_pdf(
        results_dir / "cv_results" / "summary.csv",
        reports_dir / "performance_summary.pdf",
        cross_dataset_csv=results_dir / "cross_dataset" / "generalization.csv",
        learning_curve_csv=results_dir / "learning_curves" / "learning_curve.csv",
    )
    statistical_tests_pdf(results_dir / "cv_stats", reports_dir / "statistical_tests.pdf")
    scalability_analysis_pdf(
        results_dir / "cv_results" / "summary.csv",
        results_dir / "cv_results" / "fold_metrics.csv",
        reports_dir / "scalability_analysis.pdf",
    )
    explainability_dashboard_html(
        results_dir / "feature_importance",
        results_dir / "confusion_matrices",
        reports_dir / "explainability_dashboard.html",
    )
    return {
        "performance_summary": str(reports_dir / "performance_summary.pdf"),
        "statistical_tests": str(reports_dir / "statistical_tests.pdf"),
        "scalability_analysis": str(reports_dir / "scalability_analysis.pdf"),
        "explainability_dashboard": str(reports_dir / "explainability_dashboard.html"),
    }
