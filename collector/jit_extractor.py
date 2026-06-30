"""
Commit-level Just-In-Time defect prediction extractor.

This creates one row per commit using process/change metrics. Labels are based
on bug-fix keywords as a lightweight baseline; public JIT datasets with curated
SZZ labels should be preferred for final paper experiments.
"""

import argparse
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFECT_RE = re.compile(
    r"\b(fix(e[sd])?|bug|defect|error|crash|fault|issue|patch|problem|repair|hotfix|regression|broken|fail(ed|ure)?)\b",
    re.I,
)


def git(args, cwd, timeout=120):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                            text=True, errors="replace", timeout=timeout)
    return result.stdout


def _entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    return round(-sum((c / total) * math.log2(c / total) for c in counts if c), 4)


def _commit_rows(repo, max_commits=None):
    fmt = "%H%x1f%ae%x1f%aI%x1f%s"
    args = ["log", f"--format={fmt}", "--numstat"]
    if max_commits:
        args.insert(1, f"-n{max_commits}")
    raw = git(args, repo)

    rows = []
    current = None
    file_churn = []
    for line in raw.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            if current:
                current.update(_summarize_files(file_churn))
                rows.append(current)
            commit_hash, author, date_raw, message = parts
            try:
                dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now()
            current = {
                "commit_hash": commit_hash[:12],
                "author": author,
                "commit_date": dt.isoformat(),
                "commit_hour": dt.hour,
                "commit_weekday": dt.weekday(),
                "message_length": len(message),
                "is_defect_prone": 1 if DEFECT_RE.search(message or "") else 0,
                "filepath": commit_hash[:12],
                "filename": commit_hash[:12],
                "directory": ".",
                "file_ext": "",
                "language": "Commit",
                "first_seen": dt.date().isoformat(),
                "last_seen": dt.date().isoformat(),
                "defect_commits": 1 if DEFECT_RE.search(message or "") else 0,
            }
            file_churn = []
            continue

        if current and line.strip():
            stat = line.split("\t")
            if len(stat) >= 3:
                try:
                    added = int(stat[0])
                    deleted = int(stat[1])
                except ValueError:
                    added = deleted = 0
                file_churn.append((stat[2], added, deleted))

    if current:
        current.update(_summarize_files(file_churn))
        rows.append(current)
    return rows


def _summarize_files(file_churn):
    files = [item[0] for item in file_churn]
    added = [item[1] for item in file_churn]
    deleted = [item[2] for item in file_churn]
    churn = [a + d for a, d in zip(added, deleted)]
    dirs = {str(Path(f).parent) for f in files}
    exts = {Path(f).suffix.lower() for f in files if Path(f).suffix}

    total_churn = sum(churn)
    files_touched = len(files)
    return {
        "files_touched": files_touched,
        "directories_touched": len(dirs),
        "extensions_touched": len(exts),
        "lines_added": sum(added),
        "lines_deleted": sum(deleted),
        "total_churn": total_churn,
        "avg_churn": round(total_churn / max(files_touched, 1), 4),
        "max_churn": max(churn) if churn else 0,
        "change_entropy": _entropy(churn),
        "total_commits": 1,
        "total_authors": 1,
        "avg_loc": total_churn,
        "max_loc": total_churn,
        "min_loc": total_churn,
        "loc_var": float(pd.Series(churn).var()) if len(churn) > 1 else 0.0,
        "avg_cc": 0,
        "sum_cc": 0,
        "churn_per_loc": round(total_churn / max(total_churn, 1), 4),
        "add_del_ratio": round(sum(added) / max(sum(deleted), 1), 4),
        "loc_range": max(churn) - min(churn) if churn else 0,
        "cyclomatic_complexity": 1,
        "num_functions": 0,
        "num_classes": 0,
        "comment_lines": 0,
        "blank_lines": 0,
        "code_lines": total_churn,
        "comment_ratio": 0,
        "avg_func_complexity": 0,
        "max_nesting": 0,
        "long_methods": 0,
        "complexity_per_loc": 0,
        "churn_density": round(total_churn / max(files_touched, 1), 4),
        "author_diversity": 1,
        "commit_frequency": 1,
        "task_type": "jit_commit",
    }


def run(repo, output, max_commits=None):
    if not os.path.isdir(repo):
        raise ValueError(f"Repo not found: {repo}")
    rows = _commit_rows(repo, max_commits=max_commits)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Commits analysed: {len(df)}")
    print(f"Bug/fix labelled: {int(df['is_defect_prone'].sum())} ({df['is_defect_prone'].mean()*100:.1f}%)")
    print(f"Output: {output}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--output", default="../data/jit_features.csv")
    ap.add_argument("--max-commits", type=int, default=None)
    args = ap.parse_args()
    run(args.repo, args.output, args.max_commits)

