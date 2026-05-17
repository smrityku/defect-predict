"""
Phase 1 - Git Extractor (v2 - full repo scan)
==============================================
1. git ls-tree  -> enumerate ALL tracked source files
2. git log --follow --numstat -> full commit history per file
3. Aggregate into one feature row per file

Usage:
    python git_extractor.py --repo /path/to/repo --output ../data/raw.csv
"""

import argparse, os, re, sys, subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

DEFECT_RE = re.compile(
    r"\b(fix(e[sd])?|bug|defect|error|crash|fault|issue|patch|problem"
    r"|repair|resolv(e[sd]?)?|workaround|hotfix|regression|typo"
    r"|broken|incorrect|invalid|wrong|fail(ed|ure)?)\b",
    re.IGNORECASE,
)

LANG_MAP = {
    ".py":"Python",    ".rb":"Ruby",        ".js":"JavaScript",
    ".ts":"TypeScript",".java":"Java",      ".go":"Go",
    ".cs":"CSharp",    ".cpp":"C++",        ".c":"C",
    ".php":"PHP",      ".swift":"Swift",    ".kt":"Kotlin",
    ".rs":"Rust",      ".scala":"Scala",    ".ex":"Elixir",
    ".exs":"Elixir",   ".jsx":"JavaScript", ".tsx":"TypeScript",
    ".vue":"Vue",      ".html":"HTML",      ".css":"CSS",
    ".scss":"SCSS",    ".sass":"SCSS",      ".sql":"SQL",
    ".sh":"Shell",     ".bash":"Shell",     ".zsh":"Shell",
    ".erb":"Ruby",     ".haml":"Ruby",      ".slim":"Ruby",
    ".tf":"HCL",       ".yaml":"YAML",      ".yml":"YAML",
    ".json":"JSON",    ".graphql":"GraphQL",".gql":"GraphQL",
    ".dart":"Dart",    ".r":"R",            ".m":"Objective-C",
}
SOURCE_EXTS = set(LANG_MAP)

def git(args, cwd, timeout=120):
    r = subprocess.run(["git"]+args, cwd=cwd,
                       capture_output=True, text=True,
                       errors="replace", timeout=timeout)
    return r.stdout

def is_source(path):
    return Path(path).suffix.lower() in SOURCE_EXTS

def lang(path):
    return LANG_MAP.get(Path(path).suffix.lower(), "Unknown")

def is_defect(msg):
    return 1 if DEFECT_RE.search(msg or "") else 0

def list_all_files(repo):
    """All source files tracked at HEAD via git ls-tree."""
    out = git(["ls-tree", "-r", "HEAD", "--name-only"], repo)
    return [l.strip() for l in out.strip().splitlines()
            if l.strip() and is_source(l.strip())]

def file_history(repo, filepath):
    """
    Full commit history for one file.
    Uses plain text COMMIT_SEP marker (no null bytes).
    """
    MARK = "COMMITSEP"
    fmt  = MARK + "%H" + chr(31) + "%ae" + chr(31) + "%aI" + chr(31) + "%s"
    out  = git(["log", "--follow",
                "--format=" + fmt,
                "--numstat", "--", filepath], repo)

    commits, current = [], None
    for line in out.splitlines():
        if line.startswith(MARK):
            if current:
                commits.append(current)
            body = line[len(MARK):]
            parts = body.split(chr(31), 3)
            if len(parts) < 4:
                current = None
                continue
            h, email, ds, msg = parts
            try:
                dt = datetime.fromisoformat(ds.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now()
            current = {
                "hash":    h.strip()[:10],
                "author":  email.strip(),
                "date":    dt.strftime("%Y-%m-%d"),
                "year":    dt.year,
                "month":   dt.month,
                "msg":     msg.strip(),
                "added":   0,
                "deleted": 0,
            }
        elif current and line.strip():
            p = line.split("\t", 2)
            if len(p) >= 2:
                try:
                    current["added"]   += int(p[0])
                    current["deleted"] += int(p[1])
                except ValueError:
                    pass  # binary file "-"
    if current:
        commits.append(current)
    return commits

def file_static(repo, filepath):
    """Read HEAD content; return (loc, blank_lines, cyclomatic_complexity)."""
    content = git(["show", "HEAD:" + filepath], repo, timeout=15)
    lines   = content.splitlines()
    loc     = len(lines)
    blank   = sum(1 for l in lines if not l.strip())
    kw      = re.compile(
        r"\b(if |elif |else\s*:|elsif |unless |for |foreach |while |until "
        r"|switch |case |catch |rescue |except )\b|(&&|\|\|)",
        re.I,
    )
    cc = len(kw.findall(content))
    return loc, blank, cc

def build_row(repo, filepath):
    history        = file_history(repo, filepath)
    loc, blank, cc = file_static(repo, filepath)

    if not history:
        return {
            "filepath": filepath, "language": lang(filepath),
            "filename": Path(filepath).name,
            "directory": str(Path(filepath).parent),
            "file_ext": Path(filepath).suffix.lower(),
            "total_commits": 1, "total_authors": 1,
            "total_churn": 0, "avg_churn": 0.0, "max_churn": 0,
            "lines_added": loc, "lines_deleted": 0,
            "avg_loc": float(loc), "max_loc": loc, "min_loc": loc,
            "loc_var": 0.0, "avg_cc": float(cc), "sum_cc": cc,
            "defect_commits": 0, "defect_rate": 0.0,
            "churn_per_loc": 0.0, "add_del_ratio": 1.0,
            "loc_range": 0, "blank_lines": blank,
            "first_seen": "", "last_seen": "", "is_defect_prone": 0,
        }

    total_commits  = len(set(h["hash"] for h in history))
    total_authors  = len(set(h["author"] for h in history))
    churns         = [h["added"] + h["deleted"] for h in history]
    total_churn    = sum(churns)
    lines_added    = sum(h["added"]   for h in history)
    lines_deleted  = sum(h["deleted"] for h in history)
    defect_commits = sum(is_defect(h["msg"]) for h in history)
    dates          = [h["date"] for h in history]
    loc_var        = float(pd.Series(churns).var()) if len(churns) > 1 else 0.0

    return {
        "filepath":      filepath,
        "language":      lang(filepath),
        "filename":      Path(filepath).name,
        "directory":     str(Path(filepath).parent),
        "file_ext":      Path(filepath).suffix.lower(),
        "total_commits": total_commits,
        "total_authors": total_authors,
        "total_churn":   total_churn,
        "avg_churn":     round(total_churn / max(total_commits, 1), 2),
        "max_churn":     max(churns) if churns else 0,
        "lines_added":   lines_added,
        "lines_deleted": lines_deleted,
        "avg_loc":       float(loc),
        "max_loc":       loc,
        "min_loc":       loc,
        "loc_var":       round(loc_var, 2),
        "avg_cc":        float(cc),
        "sum_cc":        cc * total_commits,
        "defect_commits": defect_commits,
        "defect_rate":   round(defect_commits / max(total_commits, 1), 4),
        "churn_per_loc": round(total_churn / max(loc, 1), 4),
        "add_del_ratio": round(lines_added / max(lines_deleted, 1), 4),
        "loc_range":     0,
        "blank_lines":   blank,
        "first_seen":    min(dates),
        "last_seen":     max(dates),
        "is_defect_prone": 1 if defect_commits > 0 else 0,
    }

def run(repo, output, branch="HEAD", limit=None):
    print(f"\n[Phase 1] Git Extractor  repo={repo}")
    print("-" * 55)

    if not os.path.isdir(repo):
        print(f"ERROR: {repo} not found"); sys.exit(1)
    if subprocess.run(["git", "rev-parse", "--git-dir"],
                      cwd=repo, capture_output=True).returncode != 0:
        print("ERROR: not a git repo"); sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    all_files = list_all_files(repo)
    if limit:
        all_files = all_files[:limit]
    print(f"  Tracked source files : {len(all_files)}")

    rows = []
    for i, fp in enumerate(all_files):
        print(f"  [{i+1}/{len(all_files)}] {fp[:70]:<70}", end="\r", flush=True)
        try:
            rows.append(build_row(repo, fp))
        except Exception as e:
            print(f"\n  Warning: skipped {fp} ({e})")

    print()  # newline after progress
    if not rows:
        print("ERROR: no rows built — check repo path and git history")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    dp = int(df["is_defect_prone"].sum())
    total = len(df)
    print(f"  Files analysed       : {total}")
    print(f"  Defect-prone         : {dp} ({dp/max(total,1)*100:.1f}%)")
    print(f"  Languages            : {', '.join(sorted(df['language'].unique()))}")
    print(f"  Output               : {output}")
    print("[Phase 1] Done.\n")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",      required=True)
    ap.add_argument("--output",    default="../data/raw.csv")
    ap.add_argument("--branch",    default="HEAD")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap number of files (useful for quick tests)")
    a = ap.parse_args()
    run(a.repo, a.output, a.branch, a.max_files)
