"""
Phase 4 – Model Training & Comparison
======================================
Trains Logistic Regression, Random Forest, Gradient Boosting, SVM.
Cross-validates, optionally grid-searches, plots confusion matrices,
ROC curves, feature importance and saves all models + results.json.

Usage:
    python train_compare.py --data-dir ../data/ --model-dir ../data/models/
    python train_compare.py --data-dir ../data/ --model-dir ../data/models/ --fast
"""

import argparse, json, os, sys, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier, GradientBoostingClassifier,
                                     ExtraTreesClassifier, AdaBoostClassifier,
                                     HistGradientBoostingClassifier, VotingClassifier,
                                     StackingClassifier)
from sklearn.svm             import SVC
from sklearn.dummy           import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     classification_report, roc_curve,
                                     balanced_accuracy_score, matthews_corrcoef,
                                     average_precision_score)
from sklearn.neural_network  import MLPClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.exceptions      import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODELS = {
    "Logistic Regression": {
        "est": LogisticRegression(max_iter=1000, random_state=42),
        "grid": {"C":[0.01,0.1,1,10], "class_weight":[None,"balanced"]},
    },
    "Random Forest": {
        "est": RandomForestClassifier(random_state=42, n_jobs=1),
        "grid": {"n_estimators":[100,200],"max_depth":[None,10,20],
                 "class_weight":[None,"balanced"]},
    },
    "Gradient Boosting": {
        "est": GradientBoostingClassifier(random_state=42),
        "grid": {"n_estimators":[100,200],"learning_rate":[0.05,0.1,0.2],
                 "max_depth":[3,5]},
    },
    "Hist Gradient Boosting": {
        "est": HistGradientBoostingClassifier(random_state=42),
        "grid": {"max_iter":[100,200], "learning_rate":[0.05,0.1],
                 "max_leaf_nodes":[15,31]},
    },
    "Extra Trees": {
        "est": ExtraTreesClassifier(random_state=42, n_jobs=1),
        "grid": {"n_estimators":[100,250], "max_depth":[None,10,20],
                 "class_weight":[None,"balanced"]},
    },
    "AdaBoost": {
        "est": AdaBoostClassifier(random_state=42),
        "grid": {"n_estimators":[50,100,200], "learning_rate":[0.05,0.1,1.0]},
    },
    "SVM": {
        "est": SVC(probability=True, random_state=42),
        "grid": {"C":[0.1,1,10],"kernel":["rbf","linear"],
                 "class_weight":[None,"balanced"]},
    },
    "Naive Bayes": {
        "est": GaussianNB(),
        "grid": {},
    },
    "MLP": {
        "est": MLPClassifier(hidden_layer_sizes=(32,), max_iter=300,
                             solver="lbfgs", random_state=42),
        "grid": {},
    },
}

COLORS = ["#3182CE","#38A169","#DD6B20","#805AD5",
          "#E53E3E","#319795","#718096","#D69E2E"]

def eval_model(model, X_te, y_te):
    yp = model.predict(X_te)
    proba = model.predict_proba(X_te)
    if proba.shape[1] == 1:
        positive_index = 0
    else:
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
    yb = proba[:, positive_index]
    roc_auc = roc_auc_score(y_te, yb) if y_te.nunique() > 1 else 0.0
    pr_auc = average_precision_score(y_te, yb) if y_te.nunique() > 1 else 0.0
    return {
        "accuracy":  round(accuracy_score(y_te,yp),4),
        "balanced_accuracy": round(balanced_accuracy_score(y_te,yp),4),
        "precision": round(precision_score(y_te,yp,zero_division=0),4),
        "recall":    round(recall_score(y_te,yp,zero_division=0),4),
        "f1":        round(f1_score(y_te,yp,zero_division=0),4),
        "roc_auc":   round(roc_auc,4),
        "pr_auc":    round(pr_auc,4),
        "mcc":       round(matthews_corrcoef(y_te,yp),4) if y_te.nunique() > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_te,yp,labels=[0,1]).tolist(),
        "report": classification_report(y_te,yp,labels=[0,1],zero_division=0),
    }

def plot_cm(cm, name, out):
    fig,ax=plt.subplots(figsize=(4,3.5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=["Clean","Defect"],yticklabels=["Clean","Defect"],ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"{out}/cm_{name.lower().replace(' ','_')}.png",dpi=120)
    plt.close()

def plot_roc(models, X_te, y_te, out):
    fig,ax=plt.subplots(figsize=(7,5))
    if y_te.nunique() < 2:
        ax.text(0.5, 0.5, "ROC unavailable: test set has one class",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f"{out}/roc_curves.png",dpi=120); plt.close()
        return
    for (name,m),col in zip(models.items(), cycle(COLORS)):
        proba = m.predict_proba(X_te)
        positive_index = list(m.classes_).index(1) if 1 in m.classes_ and proba.shape[1] > 1 else 0
        yb = proba[:, positive_index]
        fpr,tpr,_=roc_curve(y_te,yb)
        auc=roc_auc_score(y_te,yb)
        ax.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})",color=col,lw=2)
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{out}/roc_curves.png",dpi=120); plt.close()

def plot_comparison(results, out):
    names   = [n for n in results if not n.startswith("_")]
    metrics = ["balanced_accuracy","precision","recall","f1","roc_auc","pr_auc","mcc"]
    x  = np.arange(len(names)); w = 0.15
    fig,ax = plt.subplots(figsize=(12,5))
    clrs = ["#3182CE","#38A169","#DD6B20","#E53E3E","#805AD5","#319795","#718096"]
    for i,met in enumerate(metrics):
        vals = [results[n][met] for n in names]
        bars = ax.bar(x+i*w, vals, w, label=met.upper(), color=clrs[i])
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.005,
                    f"{b.get_height():.2f}", ha="center",va="bottom",fontsize=7)
    ax.set_xticks(x+w*2); ax.set_xticklabels(names,fontsize=11)
    ax.set_ylim(0,1.18); ax.set_ylabel("Score")
    ax.set_title("Model Comparison"); ax.legend(loc="upper right",fontsize=8)
    ax.grid(axis="y",alpha=.3); plt.tight_layout()
    plt.savefig(f"{out}/model_comparison.png",dpi=120); plt.close()

def plot_importance(model, feat_names, name, out):
    if not hasattr(model,"feature_importances_"): return
    imp = model.feature_importances_
    idx = np.argsort(imp)[-20:]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.barh([feat_names[i] for i in idx], imp[idx], color="#3182CE")
    ax.set_xlabel("Importance"); ax.set_title(f"Feature Importance – {name}")
    plt.tight_layout()
    plt.savefig(f"{out}/importance_{name.lower().replace(' ','_')}.png",dpi=120)
    plt.close()

def run(data_dir, model_dir, fast=False):
    print(f"\n[Phase 4] Model Training  fast={fast}")
    print("-"*55)
    os.makedirs(model_dir, exist_ok=True)

    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    feat_cols = [c for c in tr.columns if c != "is_defect_prone"]
    X_tr = tr[feat_cols]; y_tr = tr["is_defect_prone"]
    X_te = te[feat_cols]; y_te = te["is_defect_prone"]
    print(f"  Train {X_tr.shape}  Test {X_te.shape}")

    class_counts = y_tr.value_counts()
    min_class_count = class_counts.min() if len(class_counts) else 0
    cv_splits = min(5, min_class_count)
    trained, results = {}, {}

    for name, spec in MODELS.items():
        print(f"  ▶ {name}...", end=" ", flush=True)
        if y_tr.nunique() < 2:
            m = DummyClassifier(strategy="most_frequent")
            m.fit(X_tr, y_tr)
        elif fast or cv_splits < 2 or not spec["grid"]:
            m = spec["est"]
            m.fit(X_tr, y_tr)
        else:
            cv = StratifiedKFold(n_splits=cv_splits,shuffle=True,random_state=42)
            gs = GridSearchCV(spec["est"], spec["grid"],
                              cv=cv, scoring="f1", n_jobs=1)
            gs.fit(X_tr, y_tr)
            m = gs.best_estimator_

        trained[name] = m
        results[name] = eval_model(m, X_te, y_te)
        print(f"F1={results[name]['f1']:.3f}  AUC={results[name]['roc_auc']:.3f}")
        plot_cm(results[name]["confusion_matrix"], name, model_dir)
        plot_importance(m, list(X_tr.columns), name, model_dir)
        joblib.dump(m, f"{model_dir}/{name.lower().replace(' ','_')}.pkl")

    if y_tr.nunique() >= 2 and cv_splits >= 2:
        ensemble_specs = [
            ("Voting Ensemble", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  random_state=42, n_jobs=1)),
                    ("et", ExtraTreesClassifier(n_estimators=200, class_weight="balanced",
                                                random_state=42, n_jobs=1)),
                    ("hgb", HistGradientBoostingClassifier(random_state=42)),
                ],
                voting="soft"
            )),
            ("Stacking Ensemble", StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                                  random_state=42, n_jobs=1)),
                    ("gb", GradientBoostingClassifier(random_state=42)),
                ],
                final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                cv=cv_splits,
                n_jobs=1
            )),
        ]
        for name, m in ensemble_specs:
            print(f"  ▶ {name}...", end=" ", flush=True)
            m.fit(X_tr, y_tr)
            trained[name] = m
            results[name] = eval_model(m, X_te, y_te)
            print(f"F1={results[name]['f1']:.3f}  AUC={results[name]['roc_auc']:.3f}")
            plot_cm(results[name]["confusion_matrix"], name, model_dir)
            joblib.dump(m, f"{model_dir}/{name.lower().replace(' ','_')}.pkl")

    plot_roc(trained, X_te, y_te, model_dir)
    plot_comparison(results, model_dir)

    best_name = max(results, key=lambda n: results[n]["f1"])
    joblib.dump(trained[best_name], f"{model_dir}/best_model.pkl")
    print(f"\n  Best model      : {best_name}")
    print(f"  F1 / AUC        : {results[best_name]['f1']} / {results[best_name]['roc_auc']}")

    clean = {n:{k:v for k,v in r.items() if k!="report"}
             for n,r in results.items()}
    clean["_best"] = best_name
    with open(f"{model_dir}/results.json","w") as f:
        json.dump(clean, f, indent=2)

    print("[Phase 4] Done.\n")
    return trained, results, best_name

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="../data/")
    ap.add_argument("--model-dir", default="../data/models/")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    run(a.data_dir, a.model_dir, a.fast)
