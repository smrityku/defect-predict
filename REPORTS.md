# Public Dataset Reports and Results Guide

This guide is only for public benchmark datasets such as PROMISE, NASA MDP,
Jureczko-style datasets, SEACRAFT-style curated datasets, and other CSV defect
prediction datasets.

It does not cover local Git repository reports or private-project batch reports.

## Public Dataset Folder Layout

Place raw public CSV datasets here:

```text
data/public/eclipse.csv
data/public/mozilla.csv
data/public/apache.csv
data/public/kc1.csv
data/public/pc1.csv
```

The public dataset normalizer accepts flexible column names:

| Expected meaning | Accepted examples |
| --- | --- |
| File/module name | `file_name`, `filepath`, `filename`, `file`, `module`, `class` |
| Defect target | `defect_label`, `defects`, `bug`, `bugs`, `label`, `target`, `faults` |
| Metrics | Any numeric metric columns, such as LOC, complexity, churn, coupling, Halstead metrics |

## Step 1: Preprocess Public Datasets

Normalize every public CSV into processed `.pkl` files:

```bash
python scripts/preprocess.py --input data/public/*.csv --output data/processed/ --write-csv
```

Generated outputs:

| Path | What it contains |
| --- | --- |
| `data/processed/{dataset}.pkl` | Normalized public dataset used by experiments. |
| `data/processed/{dataset}.csv` | Optional normalized CSV copy, created by `--write-csv`. |
| `data/processed/preprocess_summary.csv` | Row count, feature count, defect rate, and output path per dataset. |

For very large public CSV files:

```bash
python scripts/preprocess.py --input data/public/huge_dataset.csv \
  --output data/processed/ --backend dask
```

## Step 2: Run Public Dataset CV Experiments

Run repeated stratified cross-validation:

```bash
python experiments/run_cv.py --datasets data/processed/*.pkl \
  --models random_forest svm logistic_regression \
  --cv-splits 10 --cv-repeats 3
```

Generated outputs:

| Path | What it contains |
| --- | --- |
| `results/cv_results/fold_metrics.csv` | Fold-level metrics for every dataset, model, repeat, and fold. |
| `results/cv_results/fold_predictions.csv` | Row-level predictions for every CV fold. |
| `results/cv_results/summary.csv` | Mean, standard deviation, and 95% confidence intervals per dataset/model. |
| `results/cv_results/dataset_metadata.csv` | Dataset rows, feature count, class counts, and effective CV settings. |
| `results/cv_results/warnings.csv` | Skipped datasets/models or dependency notes. |

## Step 3: Run Statistical Tests Across Public Datasets

Compare classifiers across all processed public datasets:

```bash
python experiments/run_stats.py --datasets data/processed/*.pkl
```

Generated outputs:

| Path | What it contains |
| --- | --- |
| `results/cv_stats/global_summary.txt` | Human-readable model ranking and test summary. |
| `results/cv_stats/global_model_summary.csv` | Mean/std/min/max score per model across datasets. |
| `results/cv_stats/global_model_scores.csv` | Dataset-by-model score matrix. |
| `results/cv_stats/global_pairwise_tests.csv` | Wilcoxon tests, paired Cohen's d, and Cliff's delta. |
| `results/cv_stats/global_stats.json` | Machine-readable global statistical-test summary. |
| `results/cv_stats/nemenyi_global.csv` | Nemenyi post-hoc table when `scikit-posthocs` is installed. |

## Step 4: Generate Public Dataset Explainability Reports

Generate feature-importance artifacts:

```bash
python experiments/run_feature_importance.py --datasets data/processed/*.pkl \
  --models random_forest logistic_regression
```

Generated outputs:

| Path | What it contains |
| --- | --- |
| `results/feature_importance/feature_importance_summary.csv` | Dataset/model output directory and notes. |
| `results/feature_importance/{dataset}/{model}/*.csv` | Feature ranking tables. |
| `results/feature_importance/{dataset}/{model}/*.png` | Feature-importance plots. |
| `results/feature_importance/{dataset}/{model}/explainability_notes.json` | SHAP or permutation notes. |

SHAP plots are generated when `shap` is installed and supported by the model.

## Step 5: Generate Public Dataset Confusion Matrices

Generate per-fold confusion matrices:

```bash
python experiments/run_confusion.py --datasets data/processed/*.pkl \
  --models random_forest
```

Generated outputs:

| Path | What it contains |
| --- | --- |
| `results/confusion_matrices/fold_confusion_matrices.csv` | Numeric confusion-matrix counts for every dataset/model/fold. |
| `results/confusion_matrices/{dataset}/{model}/*.png` | Per-fold confusion matrix plots. |

## Step 6: One Command for All Public Dataset Reports

After preprocessing, run the full public dataset report pipeline:

```bash
python run_all.py --datasets data/processed/*.pkl \
  --models random_forest svm logistic_regression
```

Optional expanded run with learning curves:

```bash
python run_all.py --datasets data/processed/*.pkl \
  --models random_forest svm logistic_regression \
  --learning-curves
```

Optional XGBoost run:

```bash
python run_all.py --datasets data/processed/*.pkl \
  --models random_forest svm logistic_regression xgboost
```

`xgboost` requires the optional `xgboost` dependency.

## Final Public Dataset Reports

These are the main final reports to open after `python run_all.py ...`:

| Path | What it contains |
| --- | --- |
| `reports/performance_summary.pdf` | Consolidated public-dataset performance summary, including top dataset/model combinations and cross-dataset notes when available. |
| `reports/statistical_tests.pdf` | Public-dataset Wilcoxon/Friedman/effect-size report. |
| `reports/scalability_analysis.pdf` | Runtime and scalability report across public datasets and models. |
| `reports/explainability_dashboard.html` | HTML dashboard linking public-dataset feature-importance and confusion-matrix plots. |

Supporting public-dataset result folders:

```text
results/cv_results/
results/cv_stats/
results/feature_importance/
results/confusion_matrices/
results/cross_dataset/
results/learning_curves/
```

## Fast Public Dataset Checklist

Run these two commands:

```bash
python scripts/preprocess.py --input data/public/*.csv --output data/processed/ --write-csv
python run_all.py --datasets data/processed/*.pkl --models random_forest svm logistic_regression
```

Then open:

```text
reports/performance_summary.pdf
reports/statistical_tests.pdf
reports/scalability_analysis.pdf
reports/explainability_dashboard.html
```
