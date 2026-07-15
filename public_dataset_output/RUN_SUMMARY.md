# Public Dataset Run Summary

All public dataset artifacts for this run are in this folder:

```text
public_dataset_output/
```

## Dataset Set

Fetched 13 public OpenML / PROMISE NASA MDP-style defect datasets:

```text
ar1, ar3, ar4, ar5, ar6, kc1, kc2, kc3, mc2, mw1, pc1, pc3, pc4
```

Total rows across datasets: **8,211**.

Dataset source files and OpenML IDs:

```text
public_dataset_output/dataset_manifest.csv
public_dataset_output/dataset_sources.json
```

Raw and converted data:

```text
public_dataset_output/raw_arff/
public_dataset_output/raw_csv/
public_dataset_output/processed/
```

## Commands Run

ARFF files were fetched from OpenML download URLs and converted to CSV.

Preprocessing:

```bash
python3 scripts/preprocess.py --input public_dataset_output/raw_csv/*.csv \
  --output public_dataset_output/processed --write-csv
```

Full public dataset report run:

```bash
python3 run_all.py --datasets "public_dataset_output/processed/*.pkl" \
  --models random_forest svm logistic_regression \
  --results-dir public_dataset_output/results \
  --reports-dir public_dataset_output/reports \
  --cv-splits 5 --cv-repeats 2 \
  --importance-max-rows 3000
```

Cross-dataset generalization was rerun after adding a guard for pairings with no
usable non-constant common features, then reports were regenerated.

## Final Reports

Open these first:

```text
public_dataset_output/reports/performance_summary.pdf
public_dataset_output/reports/statistical_tests.pdf
public_dataset_output/reports/scalability_analysis.pdf
public_dataset_output/reports/explainability_dashboard.html
```

## Main Result Files

```text
public_dataset_output/results/cv_results/summary.csv
public_dataset_output/results/cv_results/fold_metrics.csv
public_dataset_output/results/cv_stats/global_summary.txt
public_dataset_output/results/cv_stats/global_pairwise_tests.csv
public_dataset_output/results/feature_importance/feature_importance_summary.csv
public_dataset_output/results/confusion_matrices/fold_confusion_matrices.csv
public_dataset_output/results/cross_dataset/generalization.csv
```

## Statistical Summary

Metric: **F1**

Model ranking across 13 datasets:

| Rank | Model | Mean F1 | Std |
| --- | --- | ---: | ---: |
| 1 | Logistic Regression | 0.3759 | 0.1401 |
| 2 | Random Forest | 0.3651 | 0.1618 |
| 3 | SVM | 0.1987 | 0.1617 |

Friedman test:

```text
statistic = 19.5385
p-value   = 0.000057
```

Nemenyi post-hoc output was skipped because `scikit-posthocs` is not installed
in the local environment.

## Top CV Rows by Mean F1

| Dataset | Model | Mean F1 | Mean MCC | Mean ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| ar5 | Logistic Regression | 0.5700 | 0.5078 | 0.9100 |
| ar5 | Random Forest | 0.5500 | 0.4823 | 0.9317 |
| kc2 | Random Forest | 0.5437 | 0.4595 | 0.8174 |
| ar3 | Random Forest | 0.5433 | 0.5320 | 0.8159 |
| pc4 | Logistic Regression | 0.5280 | 0.5103 | 0.9115 |
| ar4 | Random Forest | 0.5062 | 0.4763 | 0.8257 |
| kc2 | Logistic Regression | 0.4938 | 0.4299 | 0.8314 |
| ar4 | Logistic Regression | 0.4909 | 0.4235 | 0.7940 |
| mc2 | Logistic Regression | 0.4792 | 0.3153 | 0.7299 |
| pc4 | Random Forest | 0.4681 | 0.4755 | 0.9425 |

## Notes

- CV setting used: **5 folds x 2 repeats**.
- Models used: **Random Forest, SVM, Logistic Regression**.
- Cross-dataset generalization produced `258` completed train/test/model rows.
- Cross-dataset warnings are saved in:

```text
public_dataset_output/results/cross_dataset/warnings.csv
```

Most warnings are dataset-pair/model combinations where the common feature set
had no feature surviving the variance threshold.
