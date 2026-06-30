# Advanced Defect Prediction System - Research Upgrade Plan

## Current Baseline

The current app is a local Git-based file-level defect predictor. It extracts
repository history, builds static and process metrics, trains several classical
machine learning models, evaluates them, generates reports, and serves a Flask
dashboard.

Important correction: the earlier feature set included `defect_rate` and
`defect_density`, which are derived from defect-labelled commits. These leak the
target into training and can create unrealistically perfect results. They have
been removed from the training feature list.

## Reference Paper Findings

| Paper | Main lesson for this app |
|---|---|
| Madeyski & Kawalerowicz, 2017 - Continuous Defect Prediction | Move beyond one-time file scoring toward continuous feedback using CI/build history, process metrics, and file-change records. |
| Hasanpour et al., 2020 - Deep Learning Models | Add neural/deep baselines, but compare them carefully against strong classical models because defect datasets are small and imbalanced. |
| Khakhar & Dubey, 2020 - Imbalance and OS-ELM | Treat imbalance as a first-class research problem. Report balanced accuracy/recall and compare resampling/cost-sensitive methods. |
| Shah & Pujara, 2020 - ML on NASA PROMISE | Include public benchmark datasets, not only one local Git repository, so results are comparable with prior work. |
| Pornprasit & Tantithamthavorn, 2021 - JITLine | Add just-in-time commit-level prediction and effort-aware metrics. Avoid transductive training that uses test data. |
| Keshavarz & Nagappan, 2022 - ApacheJIT | Use large JIT datasets such as ApacheJIT for robust experiments and cross-project validation. |
| Bludau & Pretschner, 2022 - Feature Sets in JIT Defect Prediction | Add richer feature families: workflow features, AST-change features, and effort-aware evaluation such as defects found in 20% LOC. |
| Haldar & Capretz, 2024 - Traditional and JIT Feature Importance | Combine traditional file/module prediction with JIT commit prediction and explain feature importance with SHAP/permutation/integrated gradients. |

## Proposed Research Contribution

Build a dual-level, explainable defect prediction system:

1. Traditional file-level defect prediction for release/test planning.
2. Just-in-time commit-level defect prediction for code review and CI support.
3. Hybrid feature engineering combining static metrics, process metrics,
   developer/workflow metrics, and optional AST-change metrics.
4. Broad model comparison across classical ML, ensemble learning, and neural
   baselines.
5. Publication-grade validation with benchmark datasets, time-aware splits,
   cross-project splits, imbalance-aware metrics, and effort-aware metrics.
6. Practical web dashboard for ranked files, risky commits, model comparison,
   feature explanations, and downloadable reports.

## Advanced System Architecture

### Data Layer

- Local Git repository extractor.
- Public dataset loader for PROMISE/NASA-style module datasets.
- JIT dataset loader for ApacheJIT/OpenStack/Qt-style commit datasets.
- Optional CI/build result importer from GitHub Actions, Travis, Jenkins, or CSV.

### Feature Layer

- File/module features: LOC, complexity, functions/classes, comment ratio,
  nesting, churn, author count, revision count.
- JIT change features: lines added/deleted, files touched, subsystems touched,
  entropy, age, recent developer experience, prior defects, time since last
  change.
- Workflow features: weekday/time, pull request size, review count, CI status,
  branch/build context when available.
- AST-change features: changed node types, max changed depth, method-level
  structural changes. Start with Python/JavaScript/TypeScript via Tree-sitter
  when dependencies are available.

### Model Layer

Already added in code:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Hist Gradient Boosting
- Extra Trees
- AdaBoost
- SVM
- Naive Bayes
- MLP neural classifier
- Voting Ensemble
- Stacking Ensemble

Next optional models:

- XGBoost/LightGBM/CatBoost when dependency installation is allowed.
- PyTorch tabular neural network/autoencoder if the thesis needs a deep learning
  chapter.
- Text/code embedding model for changed lines or commit messages, only if enough
  labelled data is available.

### Evaluation Layer

Report these as primary metrics:

- F1
- MCC
- Balanced accuracy
- Recall
- Precision
- ROC-AUC
- PR-AUC
- Confusion matrix

Report these for practical effort-awareness:

- Recall@20%LOC
- PCI@20%LOC
- Popt/effort-aware ranking score
- Top-k risky files or commits

Validation protocols:

- Single-project holdout for demo only.
- Time-aware split for realistic JIT prediction.
- Cross-project validation for publication strength.
- Repeated stratified cross-validation for traditional public datasets.
- Ablation study by feature family.
- Statistical comparison, e.g. Wilcoxon signed-rank and effect size, if enough
  project-level results are available.

## Paper Direction

Suggested title:

**An Explainable Hybrid System for Traditional and Just-In-Time Software Defect Prediction**

Possible research questions:

- RQ1: How do traditional file-level and JIT commit-level defect prediction
  models compare across public and project-specific datasets?
- RQ2: Which feature families contribute most to defect prediction performance?
- RQ3: Do ensemble and neural models improve over classical baselines under
  imbalance-aware and effort-aware metrics?
- RQ4: Can explanations help developers understand why files or commits are
  ranked as risky?

Minimum publication-ready experiment:

1. Evaluate at least one public traditional dataset family.
2. Evaluate at least one public JIT dataset family.
3. Run at least 8 to 10 models.
4. Include feature-family ablation.
5. Include SHAP or permutation importance.
6. Include time-aware validation for JIT.
7. Include a deployed demo and reproducible GitHub repository.

## Deployment Recommendation

Vercel is good for a frontend dashboard, but not ideal for full model training:
serverless functions have execution and storage limits. A better free deployment
path is:

- Frontend: Vercel.
- ML API/demo: Hugging Face Spaces or Streamlit Community Cloud.
- Lightweight API alternative: Render free tier or Railway-style free credits if
  available.
- Model artifacts: committed small demo model, GitHub Release artifact, or
  Hugging Face dataset/model storage.

For thesis/demo reliability, the strongest free setup is a Streamlit or Hugging
Face Spaces app that loads a prepared model and sample datasets. Use local CLI
training for heavy experiments, then deploy prediction/reporting only.

## Implementation Roadmap

### Level 1 - Credible Baseline

- Remove target leakage from training features.
- Expand model family and metrics.
- Regenerate results on a repository with both clean and defect-prone files.
- Update the report and dashboard to show MCC, balanced accuracy, and PR-AUC.

### Level 2 - Dataset Expansion

- Add dataset loaders for CSV-based module datasets and JIT commit datasets.
- Standardize schemas: `traditional_file`, `jit_commit`, and `prediction`.
- Save experiment metadata, dataset name, split strategy, and model parameters.
- Store `data/research/experiments.json` and `experiments_summary.csv` after
  evaluation so paper tables can compare projects, datasets, models, and metrics.

### Level 3 - JIT Prediction

- Add commit-level extractor.
- Add SZZ-inspired labelling or import labels from public JIT datasets.
- Add time-aware split and effort-aware metrics.
- Add line/range-level explanation output for each risky file. The first version
  is heuristic and explainable; a later publication extension can replace or
  validate it with supervised line-labelled data.

### Level 4 - Explainability

- Add permutation importance as default.
- Add optional SHAP if installed.
- Add per-file/per-commit explanation payloads for the dashboard.

### Level 5 - Deployment Demo

- Create a lightweight web demo that loads sample data and model artifacts.
- Keep heavy training local/offline.
- Deploy the demo on Hugging Face Spaces or Streamlit Community Cloud.
- Optionally deploy a separate frontend on Vercel.
