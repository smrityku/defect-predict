"""
Phase 3 – Preprocessing
========================
Handles class imbalance (oversampling), normalises features with
RobustScaler, selects top features via mutual information + ANOVA,
and writes train/test CSVs.

Usage:
    python preprocess.py --input ../data/features.csv --output-dir ../data/
"""

import argparse, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils import resample

TARGET = "is_defect_prone"

NUMERIC_FEATURES = [
    "total_commits","total_authors","total_churn","avg_churn","max_churn",
    "lines_added","lines_deleted","avg_loc","max_loc","min_loc","loc_var",
    "avg_cc","sum_cc","defect_rate","churn_per_loc","add_del_ratio",
    "loc_range","cyclomatic_complexity","num_functions","num_classes",
    "comment_lines","blank_lines","code_lines","comment_ratio",
    "avg_func_complexity","max_nesting","long_methods",
    "complexity_per_loc","churn_density","author_diversity",
    "defect_density","commit_frequency",
]

META_COLS = ["filepath","filename","directory","file_ext","language",
             "first_seen","last_seen","defect_commits"]

def oversample(X, y, rs=42):
    df = X.copy(); df[TARGET] = y.values
    maj = df[df[TARGET]==0]; mn = df[df[TARGET]==1]
    if len(mn)==0 or len(maj)==0 or len(maj)/max(len(mn),1) < 1.5:
        return X, y
    mn_up = resample(mn, replace=True, n_samples=len(maj), random_state=rs)
    bal   = pd.concat([maj, mn_up]).sample(frac=1, random_state=rs).reset_index(drop=True)
    return bal.drop(columns=[TARGET]), bal[TARGET]

def select_features(X_tr, y_tr, k=20):
    k = min(k, X_tr.shape[1])
    mi = set(X_tr.columns[SelectKBest(mutual_info_classif,k=k).fit(X_tr,y_tr).get_support()])
    fk = set(X_tr.columns[SelectKBest(f_classif,          k=k).fit(X_tr,y_tr).get_support()])
    return list(mi | fk)

def run(input_path, output_dir, test_size=0.2, k=20, rs=42):
    print(f"\n[Phase 3] Preprocessing  input={input_path}")
    print("-"*55)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    if TARGET not in df.columns:
        print(f"ERROR: '{TARGET}' column missing"); sys.exit(1)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    lang_cols = [c for c in df.columns if c.startswith("lang_")]
    feat_cols  = [c for c in NUMERIC_FEATURES if c in df.columns] + lang_cols

    X    = df[feat_cols].copy()
    y    = df[TARGET].copy()
    meta = df[[c for c in META_COLS if c in df.columns]].copy()

    # Fill NaN with median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Drop zero-variance
    zv = X.columns[X.std()==0].tolist()
    if zv: print(f"  Dropping zero-variance: {zv}"); X = X.drop(columns=zv)

    print(f"  Rows / features : {X.shape}")
    print(f"  Class counts    : {y.value_counts().to_dict()}")

    # Stratified split
    X_tr,X_te,y_tr,y_te,m_tr,m_te = train_test_split(
        X,y,meta, test_size=test_size, stratify=y, random_state=rs)

    # Oversample training set only
    ratio = y_tr.sum()/len(y_tr)
    print(f"  Defect ratio    : {ratio:.1%}")
    if ratio < 0.35:
        print("  Applying oversampling...")
        X_tr, y_tr = oversample(X_tr, y_tr, rs)
        print(f"  After oversample: {y_tr.value_counts().to_dict()}")

    # Scale
    scaler = RobustScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_te_s = pd.DataFrame(scaler.transform(X_te),     columns=X_te.columns)

    # Feature selection
    sel = select_features(X_tr_s, y_tr, k)
    print(f"  Selected {len(sel)} features")
    X_tr_f = X_tr_s[sel]
    X_te_f = X_te_s[sel]

    # Save
    pd.concat([X_tr_f, y_tr.reset_index(drop=True)],  axis=1).to_csv(f"{output_dir}/train.csv",     index=False)
    pd.concat([X_te_f, y_te.reset_index(drop=True)],  axis=1).to_csv(f"{output_dir}/test.csv",      index=False)
    m_te.reset_index(drop=True).to_csv(f"{output_dir}/test_meta.csv", index=False)
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    with open(f"{output_dir}/selected_features.txt","w") as f:
        f.write("\n".join(sel))

    print(f"  Train rows      : {len(X_tr_f)}")
    print(f"  Test rows       : {len(X_te_f)}")
    print(f"  Saved to        : {output_dir}")
    print("[Phase 3] Done.\n")
    return X_tr_f, X_te_f, y_tr, y_te, sel

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output-dir", default="../data/")
    ap.add_argument("--test-size",  type=float, default=0.2)
    ap.add_argument("--k-features", type=int,   default=20)
    a = ap.parse_args()
    run(a.input, a.output_dir, a.test_size, a.k_features)
