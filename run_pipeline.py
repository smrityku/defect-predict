"""
run_pipeline.py - Master orchestration script
=============================================
Runs all 6 phases end-to-end against any local Git repository.

Usage:
    python run_pipeline.py --repo /path/to/repo
    python run_pipeline.py --repo /path/to/repo --fast --no-dashboard
    python run_pipeline.py --repo /path/to/repo --max-files 200
"""

import argparse, os, sys
from pathlib import Path

BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data"
MODEL_DIR = DATA_DIR / "models"

sys.path.insert(0, str(BASE))

def banner(msg):
    print(f"\n{'='*58}\n  {msg}\n{'='*58}")

def run(repo, branch, max_files, fast, no_dashboard):
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    banner("DefectPredict - Full Pipeline")
    print(f"  Repo      : {repo}")
    print(f"  Branch    : {branch}")
    print(f"  Max files : {max_files or 'all'}")

    banner("Phase 1 - Git Extraction")
    from collector.git_extractor import run as p1
    p1(repo, str(DATA_DIR / "raw.csv"), branch, max_files)

    banner("Phase 2 - Feature Building")
    from features.feature_builder import run as p2
    p2(str(DATA_DIR / "raw.csv"), str(DATA_DIR / "features.csv"), repo)

    banner("Phase 3 - Preprocessing")
    from preprocessing.preprocess import run as p3
    p3(str(DATA_DIR / "features.csv"), str(DATA_DIR))

    banner("Phase 4 - Model Training")
    from models.train_compare import run as p4
    p4(str(DATA_DIR), str(MODEL_DIR), fast)

    banner("Phase 5 - Evaluation & Maintenance Scoring")
    from evaluation.evaluate import run as p5
    p5(str(DATA_DIR), str(MODEL_DIR), repo_path=repo)

    banner("Pipeline Complete!")
    print(f"  Predictions : {DATA_DIR / 'predictions.csv'}")
    print(f"  HTML report : {DATA_DIR / 'report.html'}")
    print(f"  Models dir  : {MODEL_DIR}")

    if not no_dashboard:
        banner("Phase 6 - Dashboard")
        print("  Open http://localhost:5000\n  Ctrl-C to stop")
        sys.path.insert(0, str(BASE / "dashboard"))
        os.chdir(BASE / "dashboard")
        from app import app
        app.run(debug=False, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the full defect prediction pipeline")
    ap.add_argument("--repo",         required=True)
    ap.add_argument("--branch",       default="HEAD")
    ap.add_argument("--max-files",    type=int, default=None,
                    help="Limit number of files (for quick testing)")
    ap.add_argument("--fast",         action="store_true",
                    help="Skip grid search for faster training")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="Skip launching the web dashboard")
    a = ap.parse_args()

    if not os.path.isdir(a.repo):
        print(f"ERROR: repo path not found: {a.repo}")
        sys.exit(1)

    run(a.repo, a.branch, a.max_files, a.fast, a.no_dashboard)
