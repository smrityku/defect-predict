# DefectInsight: An Explainable Industrial Software Defect Prediction Framework

DefectInsight is an explainable file-level software defect prediction framework
for industrial Git repositories. It extracts repository history, builds static
and process metrics, compares multiple classifiers, runs repeated stratified
cross-validation with statistical tests, and generates developer-facing risk and
explainability reports.

## Quick start

```bash
# Install dependencies
python3 -m pip install -r requirements.txt

# Run the publication-oriented experiment workflow
python3 experiments/run_all.py --repo /path/to/your/repo --fast

# Or run the legacy operational pipeline
python run_pipeline.py --repo /path/to/your/repo

# Open the dashboard
# → http://localhost:5000
```

## What it does

| Phase | Script | Output |
|-------|--------|--------|
| 1 – Git extraction | `collector/git_extractor.py` | `data/raw.csv` |
| 2 – Feature building | `features/feature_builder.py` | `data/features.csv` |
| 3 – Preprocessing | `preprocessing/preprocess.py` | `data/train.csv`, `data/test.csv` |
| 4 – Model training | `models/train_compare.py` | `data/models/*.pkl`, `data/models/results.json` |
| 5 – Evaluation | `evaluation/evaluate.py` | `data/predictions.csv`, `data/report.html` |
| 6 – Research experiments | `experiments/run_all.py` | `results/cv_stats/`, `results/feature_importance/`, `results/confusion_matrices/` |
| 7 – Dashboard | `dashboard/app.py` | http://localhost:5000 |

## Dashboard views

- **Overview**: project-level summary, risk distribution, language mix, and top risky files.
- **File Risk Table**: sortable/filterable file list with defect probability, risky-code proportion, top risky line range, maintenance score, and risk level.
- **Inspect Risky Code**: opened from the File Risk Table via **Inspect**. Shows line/range risk, developer-facing explanations, suggested refactors, RuboCop-style hints, and local commands such as focused `rubocop` checks.
- **Model Comparison**: model metrics and generated model plots.
- **Research Runs**: stored experiment summaries for thesis tables.
- **Run Pipeline**: run analysis against a local Git repository path.

## CLI options

```
python run_pipeline.py
  --repo         /path/to/git/repo       (required)
  --branch       HEAD                    (default: HEAD)
  --max-files    500                     (optional, for quick test)
  --fast                                 (skip grid search, faster)
  --no-dashboard                         (skip launching Flask)
```

Publication-oriented experiment runner:

```bash
python3 experiments/run_all.py
  --repo          /path/to/git/repo       (analyze a repository)
  --features-csv  data/features.csv       (or reuse a normalized dataset)
  --fast                                  (skip grid search)
  --cv-splits     10                      (default)
  --cv-repeats    3                       (default)
  --results-dir   results                 (default)
```

## Run phases individually

```bash
cd collector
python git_extractor.py --repo /path/to/repo --output ../data/raw.csv

cd ../features
python feature_builder.py --input ../data/raw.csv --output ../data/features.csv --repo /path/to/repo

cd ../preprocessing
python preprocess.py --input ../data/features.csv --output-dir ../data/

cd ../models
python train_compare.py --data-dir ../data/ --model-dir ../data/models/ --fast

cd ../evaluation
python evaluate.py --data-dir ../data/ --model-dir ../data/models/

cd ../dashboard
python app.py
```

## Dataset preprocessing

For a local Git repository, DefectInsight creates a normalized feature table:

```bash
python run_pipeline.py --repo /path/to/repo --fast --no-dashboard
```

For PROMISE/NASA-style public datasets, normalize the CSV into the local schema:

```bash
python datasets/load_public.py --input /path/to/dataset.csv \
  --output data/features.csv --dataset-name KC1 --target-col defects
```

Then run the experiment workflow:

```bash
python3 experiments/run_all.py --features-csv data/features.csv --fast
```

## Multi-dataset public benchmark runs

Collect public CSV datasets under `data/public/` using consistent names:

```text
data/public/eclipse.csv
data/public/mozilla.csv
data/public/apache.csv
data/public/kc1.csv
data/public/pc1.csv
```

Recommended dataset families:

- PROMISE/Jureczko-style class or file metrics: http://promise.site.uottawa.ca/SERepository/
- NASA MDP/PROMISE datasets discussed in defect-prediction quality studies: https://arxiv.org/abs/1805.10787
- Continuous Defect Prediction large-scale data direction: https://arxiv.org/abs/1703.04142
- Defectors large Python defect dataset: https://arxiv.org/abs/2303.04738

Expected raw columns are flexible. The batch normalizer recognizes common names
such as `file_name`, `filepath`, `defect_label`, `defects`, `bug`, `bugs`,
`label`, and numeric metric columns.

Normalize every public CSV into processed pickle files:

```bash
python scripts/preprocess.py --input data/public/*.csv --output data/processed/ --write-csv
```

For large CSV files, use the Dask-backed reader:

```bash
python scripts/preprocess.py --input data/public/huge_dataset.csv \
  --output data/processed/ --backend dask
```

Run repeated CV at scale:

```bash
python experiments/run_cv.py --datasets data/processed/*.pkl \
  --models random_forest svm logistic_regression --cv-splits 10 --cv-repeats 3
```

Run global statistical tests across datasets:

```bash
python experiments/run_stats.py --datasets data/processed/*.pkl
```

Generate explainability and fold-level confusion matrices:

```bash
python experiments/run_feature_importance.py --datasets data/processed/*.pkl \
  --models random_forest logistic_regression

python experiments/run_confusion.py --datasets data/processed/*.pkl \
  --models random_forest
```

Single multi-dataset orchestration entry point:

```bash
python run_all.py --datasets data/processed/*.pkl \
  --models random_forest svm logistic_regression xgboost
```

For an expanded run with learning curves:

```bash
python run_all.py --datasets data/processed/*.pkl \
  --models random_forest svm logistic_regression --learning-curves
```

## Outputs

- `data/predictions.csv` — every file with defect probability, maintenance score, risk level
- `data/line_risks.json` — risky line ranges, code proportion, and explanation reasons per file
- `data/research/experiments.json` — stored project/model runs for thesis comparison
- `data/research/experiments_summary.csv` — paper-friendly experiment summary table
- `data/report.html` — full HTML report with all charts
- `data/models/results.json` — model comparison metrics (F1, AUC, etc.)
- `data/models/best_model.pkl` — best-performing serialised model
- `results/cv_stats/cv_scores.csv` — repeated stratified fold-level scores
- `results/cv_stats/cv_summary.csv` — mean/std CV summary per classifier
- `results/cv_stats/wilcoxon_pairwise.csv` — pairwise Wilcoxon signed-rank tests
- `results/cv_stats/nemenyi_posthoc.csv` — Friedman/Nemenyi post-hoc table when `scikit-posthocs` is installed
- `results/feature_importance/` — tree importances, linear coefficients, permutation importances, and optional SHAP plots
- `results/confusion_matrices/` — confusion-matrix plots and CSV/JSON counts
- `results/cv_results/fold_metrics.csv` — multi-dataset fold-level metrics with runtime fields
- `results/cv_results/summary.csv` — mean, std, and 95% confidence intervals per dataset/model
- `results/cv_stats/global_summary.txt` — cross-dataset Wilcoxon/Friedman/effect-size summary
- `results/cross_dataset/generalization.csv` — train-one-dataset/test-another results
- `results/learning_curves/learning_curve.csv` — optional performance vs. training-size table
- `reports/performance_summary.pdf` — consolidated performance report
- `reports/statistical_tests.pdf` — statistical testing report
- `reports/scalability_analysis.pdf` — runtime and scalability report
- `reports/explainability_dashboard.html` — dashboard linking feature-importance and confusion plots

## Batch research reports

```bash
# Analyze many local projects listed in a text file
python research_batch.py --paths-file data/local_project_paths.txt \
  --out-root data/batch_research --max-files 250 --timeout 420 --fast

# Generate paper-style Markdown/HTML reports and charts
python generate_research_report.py --out-root data/batch_research

# Export a shareable PDF
python export_research_pdf.py
```

Main batch outputs:

- `data/batch_research/project_comparison.csv`
- `data/batch_research/paper_style_comparison_report.md`
- `data/batch_research/paper_style_comparison_report.html`
- `data/batch_research/defect_prediction_project_comparison_report.pdf`
- `data/batch_research/charts/*.png`

## Research workflows

```bash
# Normalize a PROMISE/NASA-style CSV dataset into the local feature schema
python datasets/load_public.py --input /path/to/dataset.csv \
  --output data/features.csv --dataset-name KC1 --target-col defects

# Extract commit-level JIT rows from a Git repository
python collector/jit_extractor.py --repo /path/to/repo \
  --output data/jit_features.csv --max-commits 1000

# Train/evaluate a normalized public or JIT dataset
python preprocessing/preprocess.py --input data/features.csv --output-dir data/
python models/train_compare.py --data-dir data/ --model-dir data/models/ --fast
python evaluation/evaluate.py --data-dir data/ --model-dir data/models/ \
  --dataset-name KC1 --task-type traditional_file
```

## Reproducing figures and statistical tests

```bash
python3 experiments/run_all.py --features-csv data/features.csv --fast \
  --cv-splits 10 --cv-repeats 3
```

This regenerates:

- model comparison and operational report artifacts under `data/`
- cross-validation summaries and Wilcoxon/Friedman/Nemenyi outputs under `results/cv_stats/`
- feature-importance and optional SHAP artifacts under `results/feature_importance/`
- confusion-matrix diagnostics under `results/confusion_matrices/`

Install `shap` and `scikit-posthocs` from `requirements.txt` to enable SHAP
summary plots and Nemenyi post-hoc tables.

## Framework API

```python
from defectinsight import DefectInsightPipeline

pipeline = DefectInsightPipeline({
    "repo": "/path/to/git/repo",
    "fast": True,
    "cv_splits": 10,
    "cv_repeats": 3,
})
artifacts = pipeline.run()
```

Line-level explanations are heuristic localizations over the file-level model
score. They identify suspicious code regions and explain why those lines were
flagged; they should be reported as explanation support, not as independently
supervised line-defect labels unless a line-labelled dataset is added.

## Supported languages

Python, Ruby, JavaScript, TypeScript, Java, Go, C#, C++, C, PHP,
Swift, Kotlin, Rust, Scala, Elixir, Vue, HTML, CSS, SCSS, SQL, Shell, ERB, HAML

## Defect labelling

Files are labelled *defect-prone* when at least one commit that touched
them contains a bug-keyword in its message (fix, bug, error, crash,
patch, regression, …). This follows the approach of Kamei et al. (2013).

## Maintenance Effort Score

```
Score = 0.40 × defect_probability
      + 0.20 × normalised_LOC
      + 0.20 × normalised_churn
      + 0.20 × normalised_cyclomatic_complexity
```

Scores are in [0, 100]. Risk levels: Critical >=70, High >=50,
Medium >=30, Low <30.
