"""
Phase 6 – Flask Dashboard
=========================
Run:  python app.py
Open: http://localhost:5000
"""

import json, os, subprocess, sys, threading
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE      = Path(__file__).parent.parent
DATA_DIR  = BASE / "data"
MODEL_DIR = DATA_DIR / "models"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_pipe = {"running": False, "step": "", "log": [], "done": False, "error": ""}

# ── helpers ─────────────────────────────────────────────────────────────────
def load_df(name):
    p = DATA_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def load_results():
    p = MODEL_DIR / "results.json"
    return json.load(open(p)) if p.exists() else {}

# ── pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

# ── API ──────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def status():
    return jsonify({
        "predictions_ready": (DATA_DIR/"predictions.csv").exists(),
        "model_ready":       (MODEL_DIR/"best_model.pkl").exists(),
        "pipeline": _pipe,
    })

@app.route("/api/predictions")
def predictions():
    df = load_df("predictions.csv")
    if df.empty: return jsonify({"files":[],"total":0})
    lang   = request.args.get("language","all")
    risk   = request.args.get("risk","all")
    search = request.args.get("search","").lower()
    if lang != "all" and "language" in df.columns:
        df = df[df["language"]==lang]
    if risk != "all" and "risk_level" in df.columns:
        df = df[df["risk_level"]==risk]
    if search and "filepath" in df.columns:
        df = df[df["filepath"].str.lower().str.contains(search,na=False)]
    df = df.sort_values("maintenance_score", ascending=False)
    cols = ["filepath","language","defect_probability",
            "maintenance_score","risk_level","predicted_defect","actual_defect"]
    cols = [c for c in cols if c in df.columns]
    return jsonify({"files": df[cols].fillna("").to_dict(orient="records"),
                    "total": len(df)})

@app.route("/api/summary")
def summary():
    df  = load_df("predictions.csv")
    res = load_results()
    if df.empty: return jsonify({})
    risk_c = df["risk_level"].value_counts().to_dict() if "risk_level" in df else {}
    lang_c = df["language"].value_counts().to_dict()   if "language"   in df else {}
    dc     = "actual_defect" if "actual_defect" in df.columns else "predicted_defect"
    dr     = float(df[dc].mean()) if dc in df.columns else 0
    best   = res.get("_best","")
    bm     = res.get(best,{})
    top5   = (df.sort_values("maintenance_score",ascending=False)
              .head(5)[["filepath","maintenance_score","risk_level"]]
              .fillna("").to_dict(orient="records")) if not df.empty else []
    return jsonify({
        "total_files": len(df),
        "defect_rate": round(dr*100,1),
        "risk_counts": risk_c,
        "language_counts": lang_c,
        "best_model": best,
        "best_metrics": {k:v for k,v in bm.items()
                         if k in ("accuracy","precision","recall","f1","roc_auc")},
        "top_files": top5,
        "model_comparison": {
            n:{k:v for k,v in m.items()
               if k in ("accuracy","precision","recall","f1","roc_auc")}
            for n,m in res.items() if not n.startswith("_")
        },
    })

@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline():
    data      = request.json or {}
    repo_path = data.get("repo_path","").strip()
    if not repo_path or not os.path.isdir(repo_path):
        return jsonify({"error": f"Invalid path: {repo_path}"}), 400
    if _pipe["running"]:
        return jsonify({"error": "Already running"}), 409

    def go():
        _pipe.update({"running":True,"log":[],"done":False,"error":""})
        DATA_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
        steps = [
            ("Extracting git history", [
                sys.executable, str(BASE/"collector"/"git_extractor.py"),
                "--repo", repo_path, "--output", str(DATA_DIR/"raw.csv")]),
            ("Building features", [
                sys.executable, str(BASE/"features"/"feature_builder.py"),
                "--input", str(DATA_DIR/"raw.csv"),
                "--output", str(DATA_DIR/"features.csv"),
                "--repo", repo_path]),
            ("Preprocessing", [
                sys.executable, str(BASE/"preprocessing"/"preprocess.py"),
                "--input", str(DATA_DIR/"features.csv"),
                "--output-dir", str(DATA_DIR)]),
            ("Training models", [
                sys.executable, str(BASE/"models"/"train_compare.py"),
                "--data-dir", str(DATA_DIR),
                "--model-dir", str(MODEL_DIR), "--fast"]),
            ("Evaluating", [
                sys.executable, str(BASE/"evaluation"/"evaluate.py"),
                "--data-dir", str(DATA_DIR),
                "--model-dir", str(MODEL_DIR)]),
        ]
        for name, cmd in steps:
            _pipe["step"] = name
            _pipe["log"].append(f"▶ {name}...")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if r.returncode != 0:
                    _pipe["error"] = r.stderr[-400:]
                    _pipe["log"].append(f"✗ {name} failed")
                    break
                _pipe["log"].append(f"✓ {name} done")
            except Exception as e:
                _pipe["error"] = str(e); break
        else:
            _pipe["log"].append("✓ All done!")
        _pipe.update({"running":False,"done":True})

    threading.Thread(target=go, daemon=True).start()
    return jsonify({"message":"started"})

@app.route("/api/pipeline-log")
def pipeline_log(): return jsonify(_pipe)

@app.route("/images/<path:fn>")
def img(fn): return send_from_directory(str(DATA_DIR), fn)

@app.route("/model-images/<path:fn>")
def model_img(fn): return send_from_directory(str(MODEL_DIR), fn)

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    print("\n  Defect Predictor Dashboard → http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
