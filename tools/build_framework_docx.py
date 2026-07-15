#!/usr/bin/env python3
import csv
import math
import zipfile
from collections import Counter
from datetime import date
from html import escape
from pathlib import Path

from build_ieee_docx import (
    BASE,
    ROOT,
    content_types,
    fnum,
    heading,
    numbering_xml,
    para,
    pct,
    read_rows,
    rels_xml,
    root_rels,
    section_break_two_col,
    styles_xml,
    table,
    text_run,
)


OUT = BASE / "defectinsight_framework_journal_review_draft.docx"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def callout(title, body):
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="6" w:color="A6A6A6"/><w:left w:val="single" w:sz="6" w:color="A6A6A6"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="A6A6A6"/><w:right w:val="single" w:sz="6" w:color="A6A6A6"/></w:tblBorders></w:tblPr>'
        '<w:tblGrid><w:gridCol w:w="9000"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="9000" w:type="dxa"/>'
        '<w:shd w:fill="F4F6F9"/><w:tcMar><w:top w:w="120" w:type="dxa"/><w:left w:w="160" w:type="dxa"/>'
        '<w:bottom w:w="120" w:type="dxa"/><w:right w:w="160" w:type="dxa"/></w:tcMar></w:tcPr>'
        + para("", before=0, after=60, runs=[text_run(title, bold=True, size=20), text_run(" " + body, size=20)])
        + "</w:tc></w:tr></w:tbl>"
    )


def numbered(text):
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
        '<w:spacing w:after="80" w:line="276" w:lineRule="auto"/></w:pPr>'
        + text_run(text, size=20)
        + "</w:p>"
    )


def numbering_with_decimal():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
</w:numbering>'''


def final_section_one_col():
    return (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:num="1" w:space="360"/></w:sectPr>'
    )


def main():
    rows = [r for r in read_rows(BASE / "project_comparison.csv") if r["status"] == "analyzed"]
    risky = read_rows(BASE / "top_risky_files.csv")
    n_projects = len(rows)
    n_files = int(sum(float(r["files_analyzed"]) for r in rows))
    mean_defect = sum(float(r["defect_ratio"]) for r in rows) / n_projects
    model_counts = Counter(r["best_model"] for r in rows)
    top_f1 = sorted(rows, key=lambda r: float(r["f1"]), reverse=True)[:6]
    top_risky = sorted(risky, key=lambda r: float(r["maintenance_score"]), reverse=True)[:6]

    doc = []
    doc.append(para("DefectInsight: A Confidentiality-Aware and Explainable Defect Prediction Framework for Industrial Software Repositories", style="Title"))
    doc.append(para("Framework-positioned journal review draft | Generated from local private-repository artifacts | " + date.today().isoformat(), align="center", after=200, runs=[text_run("Framework-positioned journal review draft | Generated from local private-repository artifacts | " + date.today().isoformat(), italic=True, size=18, color="555555")]))
    doc.append(heading("Abstract", 1))
    doc.append(para(
        f"Software defect prediction is most valuable in practice when it can operate inside private industrial repositories, explain why files are risky, and translate model outputs into developer-facing maintenance priorities. This paper reframes the current work as DefectInsight, an end-to-end, confidentiality-aware defect prediction framework for heterogeneous software projects. DefectInsight integrates Git mining, static and process metric extraction, keyword-based defect labeling, multi-model learning, risk ranking, line-region explanation, and dashboard-oriented developer support. The current empirical artifact analyzes {n_projects} available local Git repositories and {n_files:,} sampled source-file records using 11 classifier configurations and imbalance-aware metrics including F1, ROC-AUC, PR-AUC, Matthews correlation coefficient, and balanced accuracy. Results from the current batch indicate that Logistic Regression and Random Forest are frequently competitive, while risk distributions vary substantially across projects. The stronger contribution, however, is not the classifier comparison alone. It is the framework design for private-repository defect prediction and the research protocol needed to validate it. The draft therefore identifies additional experiments required for a journal-ready study, including public benchmark validation, feature importance, ablation studies, statistical testing, and deployment evaluation."
    ))
    doc.append(para("Index Terms--defect prediction, explainable software analytics, private repositories, process metrics, industrial software engineering, reproducibility.", runs=[text_run("Index Terms--", bold=True, size=20), text_run("defect prediction, explainable software analytics, private repositories, process metrics, industrial software engineering, reproducibility.", size=20)]))
    doc.append(section_break_two_col())

    doc.append(heading("I. Introduction", 1))
    doc.append(para("The current study should be positioned as more than a comparison of machine-learning classifiers. Classifier comparison is a crowded area in software defect prediction. A stronger journal contribution is an industrial framework that addresses the practical constraints of private repositories: limited source-code shareability, heterogeneous languages, noisy historical labels, developer-facing explanations, and deployment into quality-assurance workflows."))
    doc.append(callout("Central claim:", "DefectInsight is an explainable, reproducible, and confidentiality-aware defect prediction framework for industrial repositories. It combines predictive modeling with risk localization and developer-oriented prioritization rather than reporting model scores in isolation."))
    doc.append(para("The current empirical batch provides a useful foundation: 42 available local repositories, a unified pipeline, 11 classifiers, file-level predictions, line-region explanations, and dashboard-ready outputs. To become a journal-level contribution, the paper should now emphasize framework design and add stronger validation."))
    doc.append(para("The proposed research questions are:", keep_next=True))
    for rq in [
        "RQ1: Can a unified pipeline predict defect-prone files across heterogeneous industrial repositories?",
        "RQ2: Which classifier families provide consistently robust performance under repository heterogeneity?",
        "RQ3: Which static, process, repository, and temporal metrics contribute most to defect prediction?",
        "RQ4: How stable are predictions across programming languages and project types?",
        "RQ5: Can explainable risk rankings improve developer prioritization during review and maintenance?",
    ]:
        doc.append(numbered(rq))

    doc.append(heading("II. Background", 1))
    doc.append(heading("A. Software Defect Prediction", 2))
    doc.append(para("Software defect prediction estimates which files, modules, or changes are likely to contain defects. Traditional studies use static metrics such as size and complexity, process metrics such as churn and ownership, or hybrid feature sets that combine code structure with development history."))
    doc.append(heading("B. Explainable AI in Software Engineering", 2))
    doc.append(para("A defect predictor is more useful when it explains its ranking. Developers need to know whether a file is flagged because of high churn, high complexity, recent bug-fix history, dense branching logic, or risky localized code regions. Explainability therefore becomes part of the engineering artifact, not an optional visualization."))
    doc.append(heading("C. Industrial Defect Prediction", 2))
    doc.append(para("Industrial repositories are often private, domain-specific, and heterogeneous. Raw source code may not be publishable, commit messages may be inconsistent, and language ecosystems may differ across projects. These constraints motivate a framework that separates raw-source analysis from shareable derived metrics and reproducible scripts."))
    doc.append(heading("D. Research Gap", 2))
    doc.append(para("Existing studies often emphasize model comparison or public benchmark performance. The gap addressed here is an end-to-end framework for private-repository defect prediction that includes metric extraction, model selection, explainable risk localization, developer dashboarding, and confidentiality-aware reproducibility."))

    doc.append(heading("III. Proposed Framework", 1))
    doc.append(para("DefectInsight is defined as an end-to-end framework with eight major stages. The framework can run locally against authorized repositories and export only non-sensitive derived artifacts for research reporting."))
    doc.append(table([
        ["Stage", "Input", "Output"],
        ["1. Repository ingestion", "Authorized Git repository", "Commit and file inventory"],
        ["2. Feature extraction", "Source files and Git history", "Static, process, repository, and temporal metrics"],
        ["3. Label generation", "Commit messages", "Keyword-derived defect-prone labels"],
        ["4. Model training", "Feature matrix and labels", "Trained classifier set"],
        ["5. Model selection", "Validation metrics", "Best model per project/task"],
        ["6. Risk scoring", "Predicted probabilities", "Ranked file-level risk list"],
        ["7. Explainability", "Model score and code structure", "Risk reasons and line-region hints"],
        ["8. Developer delivery", "Risk artifacts", "Dashboard, CSV/JSON outputs, and CI/CD annotations"],
    ], [1400, 2500, 3100], "Table I. DefectInsight framework stages.", font_size=13))
    doc.append(para("The framework workflow is: Git repository -> feature extraction -> commit history mining -> label generation -> feature engineering -> model training -> model selection -> probability estimation -> risk ranking -> explainability -> developer dashboard."))

    doc.append(heading("IV. Dataset", 1))
    doc.append(para(f"The current private-repository batch includes {n_projects} available local Git repositories and {n_files:,} sampled source-file records. Each repository was sampled up to 250 source files to keep the first comparative batch computationally bounded. The mean keyword-labeled defect ratio across analyzed repositories is {mean_defect:.3f}."))
    doc.append(para("Repositories that were unavailable, were not Git repositories, or had no supported source files are excluded from manuscript-level claims. For public publication, repository names should be anonymized unless explicit disclosure permission is available."))
    doc.append(table([
        ["Dataset property", "Current value", "Journal-ready improvement"],
        ["Private repositories", str(n_projects), "Retain as industrial case-study dataset"],
        ["Sampled file records", f"{n_files:,}", "Run uncapped analysis for final selected repositories"],
        ["Label source", "Bug/fix commit keywords", "Add SZZ or issue-linked validation where possible"],
        ["Public validation", "Not yet included", "Add PROMISE, NASA MDP, Apache/Eclipse/Mozilla datasets"],
    ], [2200, 1900, 3100], "Table II. Current dataset status and required strengthening.", font_size=13))

    doc.append(heading("V. Feature Engineering", 1))
    doc.append(para("A journal version should expand feature engineering substantially. The current implementation already combines static and process signals; the revised paper should define every feature category and justify its defect-prediction relevance."))
    doc.append(table([
        ["Feature group", "Examples", "Expected contribution"],
        ["Static metrics", "LOC, comments, blank lines, complexity, nesting, function count, class count", "Captures structural complexity and review difficulty"],
        ["Process metrics", "Commit count, churn, authorship, bug-fix history, recent edits", "Captures change instability and socio-technical risk"],
        ["Repository metrics", "Language, directory depth, module size, dependency indicators", "Captures project and module context"],
        ["Temporal metrics", "Time since last change, weekly commit rate, release interval", "Captures recency and development rhythm"],
    ], [1600, 3300, 2700], "Table III. Feature groups proposed for the expanded framework paper.", font_size=13))

    doc.append(heading("VI. Machine Learning Pipeline", 1))
    doc.append(para("The current pipeline evaluates Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, AdaBoost, Support Vector Machine, Naive Bayes, Multi-Layer Perceptron, Voting Ensemble, and Stacking Ensemble. The revised contribution should present these as model families inside the framework rather than as the main story."))
    doc.append(table(
        [["Model", "Best-model count"]] + [[m, c] for m, c in model_counts.most_common()],
        [3600, 1800],
        "Table IV. Best-model frequency in the current private-repository batch.",
        font_size=13,
    ))
    doc.append(para("For journal readiness, the experimental protocol should use stratified repeated cross-validation, nested cross-validation for hyperparameter tuning where feasible, and consistent handling of one-class or near-one-class splits."))

    doc.append(heading("VII. Explainability Layer", 1))
    doc.append(para("The explainability layer is one of the strongest differentiators. DefectInsight should report not only that a file is risky, but why it is risky and where a developer should inspect first. Current explanations include branching or loop logic, deep nesting, error-handling paths, long dense lines, nested block entry, and inherited high file-level model risk."))
    doc.append(table(
        [["Project", "File", "Language", "Defect probability", "Maintenance score"]]
        + [[r["project"], r["filepath"], r["language"], fnum(r["defect_probability"], 2), fnum(r["maintenance_score"], 1)] for r in top_risky],
        [1300, 3500, 700, 850, 850],
        "Table V. Example risky files from the current framework output.",
        font_size=12,
    ))
    doc.append(para("The journal version should add SHAP values, permutation importance, or model-specific feature importance to connect file-level risk with measurable explanatory factors."))

    doc.append(heading("VIII. Experimental Design", 1))
    doc.append(para("The current batch is suitable as an initial industrial evaluation. A journal-ready design should include three layers: private-repository evaluation, public benchmark replication, and developer-oriented usefulness evaluation."))
    doc.append(table([
        ["Experiment", "Purpose", "Status"],
        ["Private repository batch", "Evaluate industrial feasibility across heterogeneous projects", "Current results available"],
        ["Public benchmark validation", "Strengthen external validity and reproducibility", "Required next"],
        ["Ablation study", "Measure contribution of static, process, language, and temporal features", "Required next"],
        ["Statistical testing", "Assess whether model differences are significant", "Required next"],
        ["Developer study or proxy evaluation", "Assess usefulness of explainable risk rankings", "Recommended next"],
    ], [2100, 3300, 1800], "Table VI. Experimental design for a stronger journal submission.", font_size=13))

    doc.append(heading("IX. Results", 1))
    doc.append(para("Current results support the feasibility of the framework. Logistic Regression appears as the best model in 18 repositories, Random Forest in 6 repositories, and several ensemble or tree-based learners in smaller numbers. This distribution suggests that no single classifier should be treated as universally dominant."))
    doc.append(table(
        [["Project", "Files", "Defect ratio", "Best model", "F1", "ROC-AUC"]]
        + [[r["project"], str(int(float(r["files_analyzed"]))), pct(r["defect_ratio"]), r["best_model"], fnum(r["f1"]), fnum(r["roc_auc"])] for r in top_f1],
        [1600, 650, 900, 1600, 600, 750],
        "Table VII. Top current private-repository results by F1 score.",
        font_size=12,
    ))
    doc.append(para("These results should be framed cautiously. Perfect or near-perfect scores in some repositories may reflect highly separable keyword-derived labels, small test splits, or class-distribution artifacts. The final manuscript should report confidence intervals and statistical comparisons."))

    doc.append(heading("X. Statistical Analysis", 1))
    doc.append(para("The current artifact does not yet include formal statistical testing. The journal version should add Friedman tests across classifiers, Wilcoxon signed-rank tests for pairwise comparisons, Nemenyi post-hoc analysis where appropriate, confidence intervals, and effect sizes."))
    doc.append(table([
        ["Analysis", "Question Answered", "Recommended Output"],
        ["Friedman test", "Do classifiers differ overall?", "p-value and average ranks"],
        ["Nemenyi post-hoc", "Which classifiers differ after ranking?", "Critical difference diagram"],
        ["Wilcoxon signed-rank", "Are two paired classifiers different?", "p-value with correction"],
        ["Effect size", "Is the difference practically meaningful?", "Cliff's delta or rank-biserial correlation"],
        ["Confidence interval", "How uncertain is the estimate?", "95% CI for F1, ROC-AUC, PR-AUC, MCC"],
    ], [1800, 2900, 2500], "Table VIII. Statistical analysis plan for journal-level validation.", font_size=13))

    doc.append(heading("XI. Discussion", 1))
    doc.append(para("The main interpretation should shift from algorithm ranking to framework value. DefectInsight addresses a practical gap: organizations need defect prediction that runs locally on private repositories, produces actionable developer outputs, and shares only derived artifacts. The classifier comparison remains useful, but it becomes one validation component inside a broader system."))
    doc.append(para("The strongest discussion angle is that confidentiality-aware reproducibility is not a contradiction. Raw source code can remain private while anonymized metrics, labels, predictions, scripts, schema definitions, and public benchmark replications are shared."))

    doc.append(heading("XII. Threats to Validity", 1))
    doc.append(para("Construct validity is affected by keyword-derived labels and heuristic line-risk explanations. Internal validity is affected by sampling limits and potential class imbalance. External validity is limited until public benchmark datasets are added. Conclusion validity is limited until repeated validation, confidence intervals, and statistical tests are completed."))

    doc.append(heading("XIII. Reproducibility", 1))
    doc.append(para("Raw private repositories should not be uploaded to a public archive. The reproducibility package should include the pipeline code, configuration files, anonymized derived metrics, keyword-derived labels, predictions, charts, a data dictionary, and instructions for rerunning the pipeline on authorized repositories. Public benchmark experiments should be included to support open replication."))

    doc.append(heading("XIV. Conclusion", 1))
    doc.append(para("The paper has stronger potential as a framework contribution than as a typical classifier comparison. DefectInsight should be presented as an industrial, explainable, and confidentiality-aware defect prediction framework. The current private-repository results provide an empirical starting point; the next step is to add public datasets, feature importance, ablation studies, statistical testing, and deployment evaluation. With those additions, the work can become a more substantial empirical software engineering contribution."))

    doc.append(heading("References", 1))
    refs = [
        'T. J. McCabe, "A complexity measure," IEEE Transactions on Software Engineering, vol. SE-2, no. 4, pp. 308-320, Dec. 1976.',
        'N. E. Fenton and M. Neil, "A critique of software defect prediction models," IEEE Transactions on Software Engineering, vol. 25, no. 5, pp. 675-689, Sep. 1999.',
        'T. Menzies, J. Greenwald, and A. Frank, "Data mining static code attributes to learn defect predictors," IEEE Transactions on Software Engineering, vol. 33, no. 1, pp. 2-13, Jan. 2007.',
        'S. Kim, E. J. Whitehead Jr., and Y. Zhang, "Classifying software changes: Clean or buggy?" in Proc. IEEE/ACM ASE, 2008, pp. 318-327.',
        'T. Hall, S. Beecham, D. Bowes, D. Gray, and D. Counsell, "A systematic literature review on fault prediction performance in software engineering," IEEE Transactions on Software Engineering, vol. 38, no. 6, pp. 1276-1304, Nov.-Dec. 2012.',
        'A. Ghotra, S. McIntosh, and A. E. Hassan, "Revisiting the impact of classification techniques on the performance of defect prediction models," in Proc. IEEE/ACM ICSE, 2015, pp. 789-800.',
        'Y. Kamei et al., "A large-scale empirical study of just-in-time defect prediction," IEEE Transactions on Software Engineering, vol. 39, no. 6, pp. 757-773, Jun. 2013.',
        'M. Tantithamthavorn, S. McIntosh, A. E. Hassan, and K. Matsumoto, "The impact of automated parameter optimization on defect prediction models," IEEE Transactions on Software Engineering, vol. 44, no. 7, pp. 736-756, Jul. 2018.',
        'M. Shepperd, Q. Song, Z. Sun, and C. Mair, "Data quality: Some comments on the NASA software defect datasets," IEEE Transactions on Software Engineering, vol. 39, no. 9, pp. 1208-1215, Sep. 2013.',
        'M. D\\\'Ambros, M. Lanza, and R. Robbes, "Evaluating defect prediction approaches: A benchmark and an extensive comparison," Empirical Software Engineering, vol. 17, no. 4-5, pp. 531-577, Aug. 2012.',
    ]
    for i, ref in enumerate(refs, 1):
        doc.append(para(f"[{i}] {ref}", before=0, after=44, runs=[text_run(f"[{i}] {ref}", size=16)]))

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="' + NS["w"] + '" xmlns:r="' + NS["r"] + '" xmlns:wp="' + NS["wp"] + '" xmlns:a="' + NS["a"] + '" xmlns:pic="' + NS["pic"] + '"><w:body>'
        + "".join(doc)
        + final_section_one_col()
        + "</w:body></w:document>"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types([]))
        z.writestr("_rels/.rels", root_rels())
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/numbering.xml", numbering_with_decimal())
        z.writestr("word/_rels/document.xml.rels", rels_xml([]))
    print(OUT)


if __name__ == "__main__":
    main()
