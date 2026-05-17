"""
Phase 2 – Feature Builder
=========================
Enriches the raw CSV from Phase 1 with static code metrics computed
directly from file content (no external tools required).

Usage:
    python feature_builder.py --input ../data/raw.csv
                               --output ../data/features.csv
                               --repo /path/to/repo
"""

import argparse, os, re, sys
from pathlib import Path
import pandas as pd
import numpy as np

# ── Per-language regex patterns ─────────────────────────────────────────────
_IF   = r"\b(if|elif|elsif|unless|else\s+if)\b"
_LOOP = r"\b(for|foreach|while|until|loop)\b"
_BOOL = r"(&&|\|\||and\b|or\b)"
_CASE = r"\b(case|when|switch)\b"
_ERR  = r"\b(catch|rescue|except|finally|ensure)\b"

DECISION_RE = re.compile("|".join([_IF,_LOOP,_BOOL,_CASE,_ERR]), re.I)

FUNC_RE = {
    "Python":     re.compile(r"^\s*def\s+\w+",re.M),
    "Ruby":       re.compile(r"^\s*def\s+\w+",re.M),
    "JavaScript": re.compile(r"(function\s+\w+|\w+\s*=\s*(?:async\s*)?\(.*?\)\s*=>)",re.M),
    "TypeScript": re.compile(r"(function\s+\w+|\w+\s*=\s*(?:async\s*)?\(.*?\)\s*=>)",re.M),
    "Java":       re.compile(r"(public|private|protected|static)\s+[\w<>\[\]]+\s+\w+\s*\(",re.M),
    "Go":         re.compile(r"^\s*func\s+\w+",re.M),
    "PHP":        re.compile(r"function\s+\w+",re.M),
    "default":    re.compile(r"\b(def|func|function)\s+\w+",re.M),
}
CLASS_RE = re.compile(r"^\s*class\s+\w+",re.M)
COMMENT_RE = {
    "Python":     re.compile(r"(#[^\n]*|'{3}[\s\S]*?'{3}|\"{3}[\s\S]*?\"{3})"),
    "Ruby":       re.compile(r"(#[^\n]*|=begin[\s\S]*?=end)"),
    "default":    re.compile(r"(//[^\n]*|/\*[\s\S]*?\*/|#[^\n]*)"),
}

def read_file(path):
    try:
        with open(path,"r",encoding="utf-8",errors="replace") as f:
            return f.read()
    except Exception:
        return ""

def static_metrics(filepath, language, repo_path=None):
    m = dict(cyclomatic_complexity=1, num_functions=0, num_classes=0,
             comment_lines=0, blank_lines=0, code_lines=0,
             comment_ratio=0.0, avg_func_complexity=0.0,
             max_nesting=0, long_methods=0)

    full = os.path.join(repo_path, filepath) if repo_path else filepath
    if not os.path.isfile(full):
        return m

    content = read_file(full)
    lines   = content.splitlines()
    total   = len(lines) or 1

    blank   = sum(1 for l in lines if not l.strip())
    m["blank_lines"] = blank

    cre     = COMMENT_RE.get(language, COMMENT_RE["default"])
    cm      = sum(len(x.splitlines()) for x in cre.findall(content))
    m["comment_lines"] = cm
    m["code_lines"]    = max(0, total - blank - cm)
    m["comment_ratio"] = round(cm / total, 4)

    decisions = len(DECISION_RE.findall(content))
    m["cyclomatic_complexity"] = decisions + 1

    fre  = FUNC_RE.get(language, FUNC_RE["default"])
    funcs = fre.findall(content)
    m["num_functions"] = len(funcs)
    m["avg_func_complexity"] = round(decisions / max(len(funcs),1), 2)
    m["num_classes"] = len(CLASS_RE.findall(content))

    # Nesting depth via indentation
    depths = []
    for line in lines:
        s = line.lstrip()
        if s:
            indent = len(line)-len(s)
            depths.append(indent // 4)
    m["max_nesting"] = max(depths) if depths else 0

    # Long methods (rough: func blocks > 30 lines)
    blocks = re.split(r"\n\s*(def |func |function )", content)
    m["long_methods"] = sum(1 for b in blocks if len(b.splitlines()) > 30)

    return m

def enrich(df, repo_path=None):
    metrics_list = []
    for _, row in df.iterrows():
        mm = static_metrics(row["filepath"], row.get("language","Unknown"), repo_path)
        metrics_list.append(mm)
    mdf = pd.DataFrame(metrics_list)
    df  = pd.concat([df.reset_index(drop=True), mdf.reset_index(drop=True)], axis=1)

    # Derived composite features
    df["complexity_per_loc"] = df["cyclomatic_complexity"] / df["avg_loc"].clip(lower=1)
    df["churn_density"]      = df["total_churn"] / df["total_commits"].clip(lower=1)
    df["author_diversity"]   = df["total_authors"] / df["total_commits"].clip(lower=1)
    df["defect_density"]     = df["defect_commits"] / df["avg_loc"].clip(lower=1)
    df["commit_frequency"]   = df["total_commits"] / 365.0

    # One-hot language dummies (for tree models)
    lang_dummies = pd.get_dummies(df["language"], prefix="lang")
    df = pd.concat([df, lang_dummies], axis=1)
    return df

def run(input_path, output_path, repo_path=None):
    print(f"\n[Phase 2] Feature Builder  input={input_path}")
    print("-"*55)
    if not os.path.isfile(input_path):
        print(f"ERROR: {input_path} not found"); sys.exit(1)
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} file records")
    df = enrich(df, repo_path)
    df.to_csv(output_path, index=False)
    print(f"  Feature columns : {len(df.columns)}")
    print(f"  Output          : {output_path}")
    print("[Phase 2] Done.\n")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", default="../data/features.csv")
    ap.add_argument("--repo",   default=None)
    a = ap.parse_args()
    run(a.input, a.output, a.repo)
