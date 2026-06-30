# DefectPredict

ML-based defect prediction and maintenance effort estimation for any Git codebase.

## Quick start

```bash
# Install dependencies
pip install pandas numpy scikit-learn scipy matplotlib seaborn flask joblib

# Run the full pipeline
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
| 4 – Model training | `models/train_compare.py` | `data/models/*.pkl` |
| 5 – Evaluation | `evaluation/evaluate.py` | `data/predictions.csv`, `data/report.html` |
| 6 – Dashboard | `dashboard/app.py` | http://localhost:5000 |

## CLI options

```
python run_pipeline.py
  --repo         /path/to/git/repo       (required)
  --branch       HEAD                    (default: HEAD)
  --max-commits  500                     (optional, for quick test)
  --fast                                 (skip grid search, faster)
  --no-dashboard                         (skip launching Flask)
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

## Outputs

- `data/predictions.csv` — every file with defect probability, maintenance score, risk level
- `data/line_risks.json` — risky line ranges, code proportion, and explanation reasons per file
- `data/research/experiments.json` — stored project/model runs for thesis comparison
- `data/research/experiments_summary.csv` — paper-friendly experiment summary table
- `data/report.html` — full HTML report with all charts
- `data/models/results.json` — model comparison metrics (F1, AUC, etc.)
- `data/models/best_model.pkl` — best-performing serialised model

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

Scores are in [0, 100].  Risk levels: Critical ≥70 · High ≥50 · Medium ≥30 · Low <30.
# defect-predict
