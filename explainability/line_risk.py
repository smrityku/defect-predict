"""
Line-level risk localization.

This module turns a file-level defect probability into inspectable line ranges.
It is intentionally heuristic: the supervised model predicts the file risk, and
these rules identify code sections that contain common maintainability and
fault-proneness signals.
"""

import json
import math
import os
import re
from pathlib import Path

import pandas as pd

DECISION_RE = re.compile(r"\b(if|elif|elsif|unless|for|while|case|when|switch|catch|rescue|except)\b|&&|\|\|", re.I)
ERROR_RE = re.compile(r"\b(raise|throw|except|catch|rescue|finally|ensure|panic|fatal|error)\b", re.I)
STATE_RE = re.compile(r"\b(global|static|mutable|cache|memo|session|token|password|secret)\b", re.I)
TODO_RE = re.compile(r"\b(todo|fixme|hack|temporary|workaround|bug)\b", re.I)
NULL_RE = re.compile(r"\b(null|nil|none|undefined|NaN)\b")

ADVICE = {
    "branching or loop logic": {
        "impact": "More branches mean more execution paths to test and more chances for missed edge cases.",
        "fix": "Split compound conditions into named predicate methods and add unit tests for each branch.",
        "rubocop": "Metrics/CyclomaticComplexity, Metrics/PerceivedComplexity",
    },
    "error handling path": {
        "impact": "Error paths often run less frequently and can hide broad rescues, swallowed exceptions, or inconsistent responses.",
        "fix": "Rescue specific exception classes, keep rescue blocks short, and test failure paths explicitly.",
        "rubocop": "Lint/RescueException, Style/RescueStandardError, Lint/SuppressedException",
    },
    "state or sensitive value": {
        "impact": "Mutable state and sensitive values can create order-dependent bugs or security mistakes.",
        "fix": "Keep state local where possible, avoid caching request-specific values, and never log secrets/tokens.",
        "rubocop": "Security/Eval, Rails/SaveBang, Rails/OutputSafety",
    },
    "TODO/FIXME style marker": {
        "impact": "Temporary code usually marks known incomplete behavior or a workaround that may fail in edge cases.",
        "fix": "Convert the TODO/FIXME into a tracked issue and either implement the missing behavior or isolate the workaround.",
        "rubocop": "Lint/RedundantCopDisableDirective",
    },
    "null/empty value handling": {
        "impact": "Nil/blank handling is a common source of runtime errors and inconsistent business behavior.",
        "fix": "Use guard clauses, explicit defaults, and tests for nil, blank, and missing parameters.",
        "rubocop": "Style/SafeNavigation, Rails/Blank",
    },
    "long dense line": {
        "impact": "Dense lines hide multiple responsibilities and make review mistakes more likely.",
        "fix": "Extract intermediate variables or helper methods and keep one idea per line.",
        "rubocop": "Layout/LineLength, Metrics/AbcSize",
    },
    "deep nesting": {
        "impact": "Deep nesting makes control flow hard to reason about and increases the chance of incorrect early returns.",
        "fix": "Replace nested conditionals with guard clauses, service objects, or small private methods.",
        "rubocop": "Metrics/BlockNesting, Metrics/MethodLength",
    },
    "nested block entry": {
        "impact": "Nested blocks make local variables and exit behavior harder to track.",
        "fix": "Extract the block body into a named method or use early returns to flatten the code.",
        "rubocop": "Metrics/BlockLength, Metrics/BlockNesting",
    },
    "many operators": {
        "impact": "Many operators on one line usually means hidden business logic that is difficult to verify.",
        "fix": "Break the expression into named variables and test each calculation or boolean condition.",
        "rubocop": "Metrics/AbcSize, Metrics/CyclomaticComplexity",
    },
    "inherits high file-level model risk": {
        "impact": "The file-level model ranks this file as risky based on churn, complexity, history, or size.",
        "fix": "Review this section first, then add focused tests around the behavior changed most often.",
        "rubocop": "Use RuboCop metrics plus project tests to confirm the hotspot.",
    },
}


def _advice_for(reason):
    return ADVICE.get(reason, {
        "impact": "This pattern correlates with maintainability or defect risk.",
        "fix": "Review the surrounding behavior and add focused regression tests.",
        "rubocop": "Project-specific linting recommended.",
    })


def _line_reasons(line, prev_indent=0):
    stripped = line.strip()
    reasons = []
    score = 0.0

    if not stripped:
        return 0.0, reasons

    indent = len(line) - len(line.lstrip(" "))
    nesting = indent // 4

    if DECISION_RE.search(line):
        score += 0.22
        reasons.append("branching or loop logic")
    if ERROR_RE.search(line):
        score += 0.18
        reasons.append("error handling path")
    if STATE_RE.search(line):
        score += 0.12
        reasons.append("state or sensitive value")
    if TODO_RE.search(line):
        score += 0.20
        reasons.append("TODO/FIXME style marker")
    if NULL_RE.search(line):
        score += 0.10
        reasons.append("null/empty value handling")
    if len(line) > 120:
        score += 0.12
        reasons.append("long dense line")
    if nesting >= 4:
        score += min(0.20, nesting * 0.04)
        reasons.append("deep nesting")
    if indent > prev_indent and nesting >= 3:
        score += 0.05
        reasons.append("nested block entry")

    operators = sum(line.count(op) for op in ["&&", "||", "==", "!=", ">=", "<=", "+", "-", "*", "/"])
    if operators >= 4:
        score += 0.08
        reasons.append("many operators")

    return min(score, 1.0), reasons


def _merge_segments(line_scores, threshold=0.35, max_gap=1):
    selected = [item for item in line_scores if item["line_score"] >= threshold]
    if not selected:
        selected = line_scores[:5]

    selected = sorted(selected, key=lambda x: x["line"])
    segments = []
    current = None
    for item in selected:
        if current and item["line"] <= current["end_line"] + max_gap + 1:
            current["end_line"] = item["line"]
            current["max_score"] = max(current["max_score"], item["line_score"])
            current["reasons"].update(item["reasons"])
        else:
            if current:
                current["reasons"] = sorted(current["reasons"])
                segments.append(current)
            current = {
                "start_line": item["line"],
                "end_line": item["line"],
                "max_score": item["line_score"],
                "reasons": set(item["reasons"]),
            }
    if current:
        current["reasons"] = sorted(current["reasons"])
        segments.append(current)

    for seg in segments:
        seg["max_score"] = round(float(seg["max_score"]), 4)
        seg["line_range"] = f"{seg['start_line']}-{seg['end_line']}" if seg["start_line"] != seg["end_line"] else str(seg["start_line"])
        seg["recommendations"] = [
            {
                "reason": reason,
                "impact": _advice_for(reason)["impact"],
                "suggested_change": _advice_for(reason)["fix"],
                "rubocop_hint": _advice_for(reason)["rubocop"],
            }
            for reason in seg["reasons"]
        ]
    return sorted(segments, key=lambda x: x["max_score"], reverse=True)[:10]


def analyze_file(repo_path, filepath, defect_probability=0.0, maintenance_score=0.0):
    if not repo_path:
        return {
            "filepath": filepath,
            "available": False,
            "error": "source file not available",
            "line_proportion": 0.0,
            "segments": [],
            "top_lines": [],
        }

    full_path = Path(repo_path) / filepath
    if not full_path.is_file():
        return {
            "filepath": filepath,
            "available": False,
            "error": "source file not available",
            "line_proportion": 0.0,
            "segments": [],
            "top_lines": [],
        }

    content = full_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    code_lines = [line for line in lines if line.strip()]
    if not code_lines:
        return {
            "filepath": filepath,
            "available": True,
            "line_proportion": 0.0,
            "segments": [],
            "top_lines": [],
        }

    base = min(max(float(defect_probability), 0.0), 1.0) * 0.35
    effort = min(max(float(maintenance_score), 0.0), 100.0) / 100.0 * 0.15
    prev_indent = 0
    line_scores = []

    for idx, line in enumerate(lines, start=1):
        local, reasons = _line_reasons(line, prev_indent)
        prev_indent = len(line) - len(line.lstrip(" "))
        if not line.strip():
            continue

        combined = min(1.0, base + effort + local)
        if reasons or combined >= 0.35:
            issue_details = [
                {"reason": reason, **_advice_for(reason)}
                for reason in (reasons or ["inherits high file-level model risk"])
            ]
            line_scores.append({
                "line": idx,
                "line_score": round(float(combined), 4),
                "risk_percent": round(float(combined * 100), 1),
                "reasons": reasons or ["inherits high file-level model risk"],
                "issue_details": issue_details,
                "primary_advice": issue_details[0],
                "code": line.strip()[:180],
            })

    line_scores = sorted(line_scores, key=lambda x: x["line_score"], reverse=True)
    risky_count = sum(1 for item in line_scores if item["line_score"] >= 0.50)
    line_proportion = risky_count / max(len(code_lines), 1)

    return {
        "filepath": filepath,
        "available": True,
        "total_lines": len(lines),
        "code_lines": len(code_lines),
        "risky_lines": risky_count,
        "line_proportion": round(float(line_proportion), 4),
        "line_proportion_percent": round(float(line_proportion * 100), 1),
        "segments": _merge_segments(line_scores),
        "recommendations": build_recommendations(line_scores),
        "suggested_commands": build_suggested_commands(filepath, line_scores),
        "top_lines": sorted(line_scores[:25], key=lambda x: x["line"]),
    }


def build_recommendations(line_scores):
    reason_counts = {}
    for item in line_scores:
        for reason in item.get("reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    recommendations = []
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:6]:
        advice = _advice_for(reason)
        recommendations.append({
            "reason": reason,
            "occurrences": count,
            "impact": advice["impact"],
            "suggested_change": advice["fix"],
            "rubocop_hint": advice["rubocop"],
        })
    return recommendations


def build_suggested_commands(filepath, line_scores):
    reasons = []
    for item in line_scores:
        reasons.extend(item.get("reasons", []))
    cops = []
    for reason in sorted(set(reasons)):
        cops.extend(_advice_for(reason)["rubocop"].split(", "))
    cops = sorted({cop for cop in cops if cop and "/" in cop})

    commands = []
    if filepath.endswith(".rb") and cops:
        commands.append({
            "label": "Run focused RuboCop checks",
            "command": f"bundle exec rubocop {filepath} --only {','.join(cops[:8])}",
        })
        commands.append({
            "label": "Run safe autocorrect first",
            "command": f"bundle exec rubocop {filepath} -a",
        })
    commands.append({
        "label": "Review tests around this file",
        "command": f"rg \"{Path(filepath).stem}\" spec test",
    })
    return commands


def generate_line_risk_report(predictions_csv, output_json, repo_path=None):
    pred = pd.read_csv(predictions_csv)
    reports = []
    for _, row in pred.iterrows():
        reports.append(analyze_file(
            repo_path=repo_path,
            filepath=row.get("filepath", ""),
            defect_probability=row.get("defect_probability", 0.0),
            maintenance_score=row.get("maintenance_score", 0.0),
        ))

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    return reports
