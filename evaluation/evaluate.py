"""
Phase 5 – Evaluation & Maintenance Effort Scoring
===================================================
Loads the best model, computes deep evaluation metrics, permutation
feature importance, precision-recall curve, maintenance effort scores,
and generates a full HTML report.

Usage:
    python evaluate.py --data-dir ../data/ --model-dir ../data/models/
"""

import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                              precision_recall_curve, average_precision_score)
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explainability.line_risk import generate_line_risk_report
from research_store import register_experiment

# ── Maintenance scoring ─────────────────────────────────────────────────────
def maintenance_score(row):
    p   = float(row.get("defect_probability", 0))
    loc = min(float(row.get("avg_loc",0))   / 1000.0, 1.0)
    ch  = min(float(row.get("total_churn",0))/ 5000.0, 1.0)
    cc  = min(float(row.get("cyclomatic_complexity",1))/50.0,1.0)
    return round((0.40*p + 0.20*loc + 0.20*ch + 0.20*cc)*100, 2)

def risk_label(s):
    if s>=70: return "Critical"
    if s>=50: return "High"
    if s>=30: return "Medium"
    return "Low"

# ── Plots ───────────────────────────────────────────────────────────────────
def plot_pr(model, X_te, y_te, out):
    proba = model.predict_proba(X_te)
    positive_index = list(model.classes_).index(1) if 1 in model.classes_ and proba.shape[1] > 1 else 0
    yb = proba[:, positive_index]
    if y_te.nunique() < 2:
        fig,ax=plt.subplots(figsize=(6,5))
        ax.text(0.5, 0.5, "PR curve unavailable: test set has one class",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        plt.tight_layout(); plt.savefig(f"{out}/precision_recall.png",dpi=120); plt.close()
        return
    prec,rec,_ = precision_recall_curve(y_te,yb)
    ap = average_precision_score(y_te,yb)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(rec,prec,color="#3182CE",lw=2,label=f"AP={ap:.3f}")
    ax.fill_between(rec,prec,alpha=.1,color="#3182CE")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out}/precision_recall.png",dpi=120); plt.close()

def plot_perm_importance(model, X_te, y_te, out):
    print("  Computing permutation importance...")
    pi = permutation_importance(model,X_te,y_te,n_repeats=10,random_state=42,n_jobs=1)
    idx = np.argsort(pi.importances_mean)[-20:]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.barh([X_te.columns[i] for i in idx], pi.importances_mean[idx],
            xerr=pi.importances_std[idx], color="#3182CE", ecolor="#1A5276")
    ax.set_xlabel("Mean decrease in score")
    ax.set_title("Feature Importance (Permutation)")
    plt.tight_layout(); plt.savefig(f"{out}/feature_importance.png",dpi=120); plt.close()

def plot_maintenance_dist(df, out):
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
    a1.hist(df["maintenance_score"],bins=20,color="#3182CE",edgecolor="white")
    a1.set_xlabel("Maintenance Score"); a1.set_ylabel("Files")
    a1.set_title("Maintenance Score Distribution"); a1.grid(axis="y",alpha=.3)
    counts = df["risk_level"].value_counts()
    cols   = {"Critical":"#E53E3E","High":"#DD6B20","Medium":"#D69E2E","Low":"#38A169"}
    a2.pie(counts.values, labels=counts.index,
           colors=[cols.get(l,"#A0AEC0") for l in counts.index],
           autopct="%1.1f%%",startangle=90)
    a2.set_title("Risk Distribution")
    plt.tight_layout(); plt.savefig(f"{out}/maintenance_dist.png",dpi=120); plt.close()

# ── HTML report ─────────────────────────────────────────────────────────────
def html_report(results, pred_df, out):
    best  = results.get("_best","")
    bm    = results.get(best,{})
    top20 = (pred_df.sort_values("maintenance_score",ascending=False)
             .head(20)[["filepath","language","defect_probability",
                         "maintenance_score","risk_level"]]
             .fillna("").to_html(index=False,classes="tbl",
                                 float_format="{:.3f}".format,border=0))
    rows_list = []
    for n, m in results.items():
        if n.startswith("_"):
            continue
        row_class = " class=\"best-row\"" if n == best else ""
        star      = "&#9733;" if n == best else ""
        rows_list.append(
            "<tr" + row_class + ">"
            "<td><b>" + n + "</b>" + star + "</td>"
            "<td>" + f"{m.get('accuracy',0):.4f}" + "</td>"
            "<td>" + f"{m.get('balanced_accuracy',0):.4f}" + "</td>"
            "<td>" + f"{m.get('precision',0):.4f}" + "</td>"
            "<td>" + f"{m.get('recall',0):.4f}" + "</td>"
            "<td>" + f"{m.get('f1',0):.4f}" + "</td>"
            "<td>" + f"{m.get('roc_auc',0):.4f}" + "</td>"
            "<td>" + f"{m.get('pr_auc',0):.4f}" + "</td>"
            "<td>" + f"{m.get('mcc',0):.4f}" + "</td></tr>"
        )
    rows = "".join(rows_list)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>DefectInsight Report</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:1100px;margin:0 auto;padding:24px;color:#1A202C}}
  h1{{color:#2C3E50;border-bottom:3px solid #3182CE;padding-bottom:8px}}
  h2{{color:#2D3748;margin-top:28px}}
  .summary{{background:#EBF8FF;border-left:4px solid #3182CE;padding:14px;border-radius:4px}}
  table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}}
  th{{background:#3182CE;color:#fff;padding:9px 12px;text-align:left}}
  td{{padding:7px 12px;border-bottom:1px solid #EDF2F7}}
  tr:nth-child(even){{background:#F7FAFC}}
  .best-row td{{background:#BEE3F8!important;font-weight:600}}
  .tbl th{{background:#2C3E50}}
  .imgs{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}}
  .imgs img{{width:100%;border:1px solid #E2E8F0;border-radius:8px}}
</style></head><body>
<h1>DefectInsight Report</h1>
<div class="summary">
  <b>Best model:</b> {best} &nbsp;|&nbsp;
  <b>F1:</b> {bm.get('f1',0):.4f} &nbsp;|&nbsp;
  <b>ROC-AUC:</b> {bm.get('roc_auc',0):.4f} &nbsp;|&nbsp;
  <b>MCC:</b> {bm.get('mcc',0):.4f} &nbsp;|&nbsp;
  <b>Files analysed:</b> {len(pred_df)}
</div>
<h2>Model Comparison</h2>
<table><thead><tr><th>Model</th><th>Accuracy</th><th>Balanced Acc.</th><th>Precision</th>
<th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>MCC</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Visualisations</h2>
<div class="imgs">
  <img src="models/model_comparison.png">
  <img src="models/roc_curves.png">
  <img src="feature_importance.png">
  <img src="precision_recall.png">
  <img src="maintenance_dist.png">
</div>
<h2>Top 20 High-Risk Files</h2>{top20}
</body></html>"""
    path = f"{out}/report.html"
    with open(path,"w") as f: f.write(html)
    print(f"  HTML report     : {path}")

def run(data_dir, model_dir, repo_path=None, dataset_name=None, task_type="traditional_file"):
    print(f"\n[Phase 5] Evaluation  data_dir={data_dir}")
    print("-"*55)

    te   = pd.read_csv(f"{data_dir}/test.csv")
    meta = pd.read_csv(f"{data_dir}/test_meta.csv") \
           if os.path.isfile(f"{data_dir}/test_meta.csv") else pd.DataFrame()

    feat_cols = [c for c in te.columns if c != "is_defect_prone"]
    X_te = te[feat_cols]; y_te = te["is_defect_prone"]

    model = joblib.load(f"{model_dir}/best_model.pkl")
    with open(f"{model_dir}/results.json") as f:
        results = json.load(f)

    proba = model.predict_proba(X_te)
    if proba.shape[1] == 1:
        positive_index = 0
    else:
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
    yb = proba[:, positive_index]
    yp = model.predict(X_te)

    print(f"  Best model      : {results.get('_best','?')}")
    print(f"  Test F1         : {f1_score(y_te,yp,zero_division=0):.4f}")
    if y_te.nunique() > 1:
        print(f"  ROC-AUC         : {roc_auc_score(y_te,yb):.4f}")
    else:
        print("  ROC-AUC         : n/a (test set has one class)")
    print(classification_report(y_te,yp,labels=[0,1],target_names=["Clean","Defect"],zero_division=0))

    # Build predictions dataframe
    pred = meta.copy() if not meta.empty else pd.DataFrame()
    pred["defect_probability"] = yb
    pred["predicted_defect"]   = yp
    pred["actual_defect"]      = y_te.values
    if "filepath" not in pred.columns:
        pred["filepath"] = [f"file_{i}" for i in range(len(pred))]
    if "language" not in pred.columns:
        pred["language"] = "Unknown"

    # Attach metric columns needed for scoring
    for col in ["avg_loc","total_churn","cyclomatic_complexity"]:
        pred[col] = te[col].values if col in te.columns else 0

    pred["maintenance_score"] = pred.apply(maintenance_score, axis=1)
    pred["risk_level"]        = pred["maintenance_score"].apply(risk_label)

    # Plots
    plot_pr(model, X_te, y_te, data_dir)
    plot_perm_importance(model, X_te, y_te, data_dir)
    plot_maintenance_dist(pred, data_dir)

    pred.to_csv(f"{data_dir}/predictions.csv", index=False)
    print(f"  Predictions CSV : {data_dir}/predictions.csv")

    line_risk_path = f"{data_dir}/line_risks.json"
    generate_line_risk_report(f"{data_dir}/predictions.csv", line_risk_path, repo_path=repo_path)
    print(f"  Line risk JSON  : {line_risk_path}")

    # Risk summary
    rc = pred["risk_level"].value_counts()
    for lv in ["Critical","High","Medium","Low"]:
        print(f"    {lv:10s}: {rc.get(lv,0)} files")

    html_report(results, pred, data_dir)
    exp = register_experiment(data_dir, model_dir, repo_path=repo_path,
                              dataset_name=dataset_name, task_type=task_type)
    if exp:
        print(f"  Research run    : {exp['run_id']} ({exp['project_name']})")
    print("[Phase 5] Done.\n")
    return pred

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="../data/")
    ap.add_argument("--model-dir", default="../data/models/")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--task-type", default="traditional_file")
    a = ap.parse_args()
    run(a.data_dir, a.model_dir, a.repo, a.dataset_name, a.task_type)
