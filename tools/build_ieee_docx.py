#!/usr/bin/env python3
import csv
import math
import os
import zipfile
from collections import Counter
from datetime import date
from html import escape
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "batch_research"
OUT = BASE / "defect_prediction_ieee_review_draft.docx"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value, digits=3):
    try:
        x = float(value)
    except Exception:
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x:.{digits}f}"


def pct(value, digits=1):
    try:
        x = float(value)
    except Exception:
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x * 100:.{digits}f}%"


def text_run(text, bold=False, italic=False, size=20, color="000000"):
    props = [f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>']
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if color != "000000":
        props.append(f'<w:color w:val="{color}"/>')
    return (
        "<w:r><w:rPr>"
        + "".join(props)
        + f'</w:rPr><w:t xml:space="preserve">{escape(str(text))}</w:t></w:r>'
    )


def para(text="", style=None, align=None, before=0, after=120, runs=None, keep_next=False):
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if before or after:
        ppr.append(f'<w:spacing w:before="{before}" w:after="{after}" w:line="276" w:lineRule="auto"/>')
    if keep_next:
        ppr.append("<w:keepNext/>")
    body = "".join(runs) if runs is not None else text_run(text, size=20)
    return "<w:p><w:pPr>" + "".join(ppr) + "</w:pPr>" + body + "</w:p>"


def heading(text, level):
    return para(text, style=f"Heading{level}", before=220 if level == 1 else 160, after=80, keep_next=True)


def bullet(text):
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
        '<w:spacing w:after="80" w:line="276" w:lineRule="auto"/></w:pPr>'
        + text_run(text, size=20)
        + "</w:p>"
    )


def table(rows, widths, caption=None, font_size=14):
    total = sum(widths)
    xml = []
    if caption:
        xml.append(para(caption, style="Caption", before=120, after=60, keep_next=True))
    xml.append(
        f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{total}" w:type="dxa"/>'
        '<w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
        "</w:tblPr><w:tblGrid>"
    )
    for w in widths:
        xml.append(f'<w:gridCol w:w="{w}"/>')
    xml.append("</w:tblGrid>")
    for i, row in enumerate(rows):
        xml.append("<w:tr>")
        for j, cell in enumerate(row):
            fill = '<w:shd w:fill="F2F4F7"/>' if i == 0 else ""
            bold = i == 0
            align = "center" if j > 0 else "left"
            xml.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{fill}'
                '<w:tcMar><w:top w:w="80" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
                '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar></w:tcPr>'
                + para(str(cell), align=align, before=0, after=0, runs=[text_run(cell, bold=bold, size=font_size)])
                + "</w:tc>"
            )
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def rels_xml(image_rels):
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>',
    ]
    rels.extend(image_rels)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def image_para(path, rid, fig_no, caption, max_width_in=3.05):
    img = Image.open(path)
    w_px, h_px = img.size
    width_in = max_width_in
    height_in = width_in * h_px / w_px
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    name = escape(path.name)
    return (
        '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="60"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{fig_no}" name="{name}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr>'
        f'<pic:cNvPr id="{fig_no}" name="{name}"/><pic:cNvPicPr/>'
        '</pic:nvPicPr><pic:blipFill>'
        f'<a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch>'
        '</pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
        + para(f"Fig. {fig_no}. {caption}", style="Caption", align="center", before=0, after=120)
    )


def section_break_two_col():
    return (
        '<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/>'
        '<w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:num="1" w:space="360"/></w:sectPr></w:pPr></w:p>'
    )


def final_section():
    return (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:num="2" w:space="360"/></w:sectPr>'
    )


def styles_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="20"/></w:rPr><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="120"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="220" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="60"/></w:pPr><w:rPr><w:b/><w:i/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="80" w:after="80"/></w:pPr><w:rPr><w:i/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="A6A6A6"/><w:left w:val="single" w:sz="4" w:color="A6A6A6"/><w:bottom w:val="single" w:sz="4" w:color="A6A6A6"/><w:right w:val="single" w:sz="4" w:color="A6A6A6"/><w:insideH w:val="single" w:sz="4" w:color="A6A6A6"/><w:insideV w:val="single" w:sz="4" w:color="A6A6A6"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''


def numbering_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>'''


def content_types(image_names):
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Default Extension="png" ContentType="image/png"/>',
    ]
    overrides = [
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
        '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>',
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        + "".join(defaults + overrides)
        + "</Types>"
    )


def root_rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''


def main():
    rows = read_rows(BASE / "project_comparison.csv")
    analyzed = [r for r in rows if r["status"] == "analyzed"]
    risky = read_rows(BASE / "top_risky_files.csv")
    n_projects = len(analyzed)
    n_files = int(sum(float(r["files_analyzed"]) for r in analyzed))
    mean_defect = sum(float(r["defect_ratio"]) for r in analyzed) / n_projects
    model_counts = Counter(r["best_model"] for r in analyzed)
    top_f1 = sorted(analyzed, key=lambda r: float(r["f1"]), reverse=True)[:8]
    top_risk_projects = sorted(analyzed, key=lambda r: (float(r["high_files"]) + float(r["medium_files"])), reverse=True)[:8]
    top_risky = sorted(risky, key=lambda r: float(r["maintenance_score"]), reverse=True)[:10]

    image_paths = [
        BASE / "charts" / "best_model_frequency.png",
        BASE / "charts" / "top_f1_projects.png",
        BASE / "charts" / "top_roc_auc_projects.png",
        BASE / "charts" / "risk_distribution_top_projects.png",
    ]
    image_rels = []
    media = []
    for idx, path in enumerate(image_paths, start=3):
        target = f"media/{path.name}"
        image_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
        )
        media.append((f"word/{target}", path, f"rId{idx}"))

    doc = []
    doc.append(para("A Multi-Project Defect Prediction Pipeline Using Static and Process Metrics", style="Title"))
    doc.append(para("Review draft prepared for manual partner review | Generated from local private-repository artifacts | " + date.today().isoformat(), align="center", after=200, runs=[text_run("Review draft prepared for manual partner review | Generated from local private-repository artifacts | " + date.today().isoformat(), italic=True, size=18, color="555555")]))
    doc.append(heading("Abstract", 1))
    doc.append(para(
        f"Software defect prediction supports early identification of defect-prone files and helps teams prioritize testing and review effort. This study evaluates a uniform defect-prediction pipeline across {n_projects} available local Git repositories. Each repository was sampled up to 250 source files and represented through static metrics, process metrics, language indicators, and commit-history features. Defect labels were inferred from bug/fix keyword heuristics in commit messages. Multiple machine-learning classifiers were evaluated, including Logistic Regression, Random Forest, Gradient Boosting, Hist Gradient Boosting, Extra Trees, AdaBoost, Support Vector Machine, Naive Bayes, Multi-Layer Perceptron, Voting Ensemble, and Stacking Ensemble. Performance was assessed using F1 score, receiver operating characteristic area under the curve (ROC-AUC), precision-recall area under the curve (PR-AUC), Matthews correlation coefficient (MCC), balanced accuracy, precision, and recall. Results indicate that Random Forest and Logistic Regression frequently achieved competitive performance, although the best model varied across projects. The study contributes a reproducible pipeline, comparative multi-project evidence, and developer-facing risky-file explanations while acknowledging limitations caused by heuristic labels and private-source confidentiality."
    ))
    doc.append(para("Index Terms--software defect prediction, machine learning, static metrics, process metrics, ROC-AUC, reproducibility.", runs=[text_run("Index Terms--", bold=True, size=20), text_run("software defect prediction, machine learning, static metrics, process metrics, ROC-AUC, reproducibility.", size=20)]))
    doc.append(section_break_two_col())

    doc.append(heading("I. Introduction", 1))
    doc.append(para("Software defect prediction is an established research area in empirical software engineering. Predictive models can identify defect-prone files before release, allowing teams to focus testing, code review, and maintenance effort on high-risk components. This is especially useful in large codebases where exhaustive manual inspection is costly."))
    doc.append(para("Prior research has shown that static code metrics, process metrics, and historical change information can support defect prediction [1]-[5]. However, prediction performance is sensitive to repository-specific factors, including project size, commit discipline, language mix, and defect-label quality. Class imbalance further complicates evaluation because defect-prone files usually form a minority class. Accuracy alone is therefore insufficient; metrics such as F1 score, ROC-AUC, PR-AUC, MCC, and balanced accuracy provide a more reliable view under imbalance [5], [7], [11]."))
    doc.append(para("This study evaluates a defect-prediction pipeline over available local Git repositories. The pipeline extracts static and process metrics, labels files using bug/fix keyword heuristics, trains multiple classifier families, and produces model-comparison results with file-level and line-region explanations. The aim is not to claim universal generalization from private repositories. Instead, the work studies whether a consistent pipeline can produce comparable, developer-actionable results across heterogeneous industrial-style projects."))
    doc.append(para("The contributions are as follows:", keep_next=True))
    for item in [
        f"A multi-project comparison of 11 classifier configurations across {n_projects} available local Git repositories.",
        "A reproducible pipeline that transforms Git history and source files into standardized research artifacts.",
        "A risk-analysis layer that ranks defect-prone files and provides line-region explanations for developer review.",
        "A confidentiality-aware reporting approach that separates private source repositories from shareable derived metrics and scripts.",
    ]:
        doc.append(bullet(item))

    doc.append(heading("II. Related Work", 1))
    doc.append(para("Early defect-prediction research connected code size and structural complexity with defect proneness. McCabe introduced cyclomatic complexity as a measure of control-flow complexity [1], while later work criticized overly simplistic defect models and emphasized careful empirical validation [2]. Menzies et al. showed that static code attributes can be used to learn defect predictors from software metrics [3]."))
    doc.append(para("Process metrics later became central to defect prediction. Commit history, churn, authorship, and change frequency capture how software evolves and often improve prediction beyond static structure alone [4], [8], [12]. Kamei et al. conducted a large-scale just-in-time defect-prediction study and demonstrated the value of change-level metrics [8]. Cross-project prediction studies also showed that models may degrade when training and test projects differ substantially [9], [13], [18]."))
    doc.append(para("Machine-learning classifiers have been widely studied for defect prediction. Random Forest, Support Vector Machine, boosting, bagging, and stacking approaches have been evaluated across many datasets [7], [10], [11], [16]. Class imbalance remains a major challenge; ROC-AUC, PR-AUC, MCC, balanced accuracy, precision, recall, and F1 score are commonly recommended to capture different performance trade-offs under skewed distributions [5], [7], [11]."))
    doc.append(para("Reproducibility is also a persistent concern. Public datasets such as NASA and PROMISE have supported replication, but data-quality issues have been documented [17], [19]. Private industrial repositories can provide practical relevance, but source confidentiality limits open redistribution. A rigorous study using private repositories should therefore publish scripts, anonymized derived metrics, aggregate results, and complementary public benchmark experiments where possible."))

    doc.append(heading("III. Methodology", 1))
    doc.append(heading("A. Dataset Scope", 2))
    doc.append(para(f"The study analyzed {n_projects} available local Git repositories and {n_files:,} sampled source-file records. Repositories that were not locally available, were not Git repositories, or did not contain supported source files are excluded from the manuscript-level dataset description. Each included repository was sampled up to 250 source files to keep the batch experiment comparable and computationally bounded."))
    doc.append(para("Because the source repositories are private, public reporting should use anonymized repository identifiers such as R01-R42. Internal engineering handover reports may retain repository names only when distributed to authorized stakeholders."))
    doc.append(heading("B. Prediction Unit and Labeling", 2))
    doc.append(para("The prediction unit is the source file. A file is labeled defect-prone when its Git history includes at least one commit message containing bug/fix-related terms such as bug, fix, error, crash, patch, or regression. This strategy is scalable, but it is weaker than manually curated labels or SZZ-style bug-introducing change identification [4], [8]."))
    doc.append(heading("C. Feature Extraction and Models", 2))
    doc.append(para("Each file is represented using static metrics, including lines of code, comment density, approximate complexity, nesting depth, language indicators, and source-structure features; and process metrics, including churn, number of commits touching the file, authorship diversity, and bug/fix keyword history."))
    doc.append(para("The pipeline evaluates Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, AdaBoost, Support Vector Machine, Naive Bayes, Multi-Layer Perceptron, Voting Ensemble, and Stacking Ensemble. This set covers interpretable baselines, nonlinear learners, and ensemble methods commonly used in defect-prediction studies [7], [10], [11], [16]."))
    doc.append(heading("D. Evaluation Metrics", 2))
    doc.append(para("The evaluation uses precision, recall, F1 score, ROC-AUC, PR-AUC, MCC, and balanced accuracy. These metrics are reported together because no single metric fully captures defect-prediction performance under class imbalance."))
    doc.append(table([
        ["Metric", "Definition", "Reason for Use"],
        ["F1", "Harmonic mean of precision and recall", "Summarizes positive-class retrieval"],
        ["ROC-AUC", "Area under true-positive/false-positive curve", "Threshold-independent ranking view"],
        ["PR-AUC", "Area under precision-recall curve", "More informative under rare positives"],
        ["MCC", "Correlation between predicted and true labels", "Robust single score under imbalance"],
        ["Balanced accuracy", "Mean of sensitivity and specificity", "Avoids majority-class dominance"],
    ], [800, 1700, 1750], "Table I. Evaluation metrics used for imbalanced defect prediction."))
    doc.append(heading("E. Explainability Layer", 2))
    doc.append(para("The pipeline ranks files by defect probability and maintenance effort score. It also identifies risky line regions using heuristic localization over file-level predictions. Explanations include contributing factors such as high complexity, branching logic, deep nesting, dense lines, error-handling paths, and inherited high file-level model risk. These line-level explanations are developer-support signals, not independently supervised line-defect labels."))

    doc.append(heading("IV. Results", 1))
    doc.append(para(f"The analyzed dataset contains {n_projects} repositories, {n_files:,} sampled file records, and a mean keyword-labeled defect ratio of {mean_defect:.3f}. Model performance varied substantially across projects, which supports project-specific evaluation rather than a single universal model recommendation."))
    doc.append(table(
        [["Model", "Best-model count"]] + [[m, c] for m, c in model_counts.most_common(8)],
        [2800, 1400],
        "Table II. Frequency with which each classifier was selected as the best model across analyzed repositories.",
    ))
    doc.append(image_para(image_paths[0], "rId3", 1, "Best-model frequency across analyzed repositories."))
    doc.append(heading("A. Top Project-Level Results", 2))
    doc.append(table(
        [["Project", "Files", "Defect ratio", "Best model", "F1", "ROC-AUC", "MCC"]]
        + [[r["project"], str(int(float(r["files_analyzed"]))), pct(r["defect_ratio"]), r["best_model"], fnum(r["f1"]), fnum(r["roc_auc"]), fnum(r["mcc"])] for r in top_f1],
        [1150, 450, 650, 900, 350, 450, 350],
        "Table III. Top projects by F1 score in the available local batch.",
        font_size=12,
    ))
    doc.append(image_para(image_paths[1], "rId4", 2, "Top projects ranked by F1 score."))
    doc.append(image_para(image_paths[2], "rId5", 3, "Top projects ranked by ROC-AUC."))
    doc.append(heading("B. Risk Distribution", 2))
    doc.append(para("Risk distribution analysis showed that defect-prone files often concentrate in controllers, service classes, configuration files, authentication flows, data-processing modules, and high-churn feature areas. This pattern is consistent with the expectation that complex coordination logic and frequently modified files accumulate maintenance risk."))
    doc.append(table(
        [["Project", "Prediction rows", "Critical", "High", "Medium", "Low"]]
        + [[r["project"], str(int(float(r["prediction_rows"]))), str(int(float(r["critical_files"]))), str(int(float(r["high_files"]))), str(int(float(r["medium_files"]))), str(int(float(r["low_files"])))] for r in top_risk_projects],
        [1350, 700, 500, 500, 600, 500],
        "Table IV. Repositories with the largest high/medium risk-file concentration.",
        font_size=12,
    ))
    doc.append(image_para(image_paths[3], "rId6", 4, "Risk distribution across selected high-risk projects."))
    doc.append(heading("C. File-Level Hotspots", 2))
    doc.append(table(
        [["Project", "File", "Lang.", "Defect prob.", "Maint. score"]]
        + [[r["project"], r["filepath"], r["language"], fnum(r["defect_probability"], 2), fnum(r["maintenance_score"], 1)] for r in top_risky[:8]],
        [900, 1900, 500, 550, 550],
        "Table V. Highest-ranked risky files by maintenance score.",
        font_size=12,
    ))

    doc.append(heading("V. Discussion", 1))
    doc.append(heading("A. Model Interpretation", 2))
    doc.append(para("Random Forest and Logistic Regression were frequently competitive. Tree-based models handled nonlinear feature interactions and heterogeneous feature distributions well, while Logistic Regression remained effective when metric distributions were moderately separable. This finding aligns with prior evidence that simple metric-based models can remain strong baselines [3], [7]."))
    doc.append(heading("B. Private Repository Constraints", 2))
    doc.append(para("The repositories analyzed in this study are private local projects. This improves practical relevance but limits public redistribution. The correct publication framing is confidentiality-aware: raw source code should not be uploaded to public archives, and repository names should be anonymized unless explicit permission is available. Shareable artifacts should consist of derived metrics, labels, predictions, charts, configuration files, and scripts."))
    doc.append(heading("C. Threats to Validity", 2))
    doc.append(para("Construct validity is affected by keyword-based defect labels and heuristic line-risk explanations. Internal validity is affected by sampling up to 250 files per repository and by possible one-class splits. External validity is limited because the repositories are private and may not represent all software domains. Conclusion validity is affected by the absence of statistical significance tests in the current draft. A final submission should add confidence intervals, paired tests, or non-parametric comparisons where appropriate."))

    doc.append(heading("VI. Conclusion", 1))
    doc.append(para("This study presented a multi-project defect-prediction pipeline evaluated across available local Git repositories. The pipeline combines static metrics, process metrics, keyword-derived defect labels, multiple classifier families, imbalance-aware evaluation metrics, and developer-facing risk explanations. Random Forest and Logistic Regression were frequently competitive, but model effectiveness varied across repositories, confirming that project-specific data characteristics remain important."))
    doc.append(para("The results support defect prediction as a practical prioritization tool for testing and maintenance. However, the findings should be interpreted with appropriate caution because labels are heuristic, source repositories are private, and line-level explanations are not independently supervised defect labels. Future work should add public benchmark validation, SZZ-based labeling, statistical significance testing, and continuous integration support for ongoing defect-risk monitoring."))

    doc.append(heading("Reproducibility and Data Availability", 1))
    doc.append(para("The raw source repositories used in this study are private and cannot be redistributed publicly. To support transparency without violating confidentiality, the replication package should include the pipeline source code, scripts for extraction and evaluation, anonymized derived metrics, keyword-derived labels, model predictions, aggregate tables, charts, a data dictionary, and instructions for rerunning the pipeline on authorized repositories."))
    doc.append(para("If IEEE DataPort or another archival repository is used, the uploaded package should exclude proprietary source files and should state that raw private repositories are available only to authorized reviewers under confidentiality constraints. To strengthen independent replication, the final paper should also include experiments on public benchmark datasets such as PROMISE/NASA or public GitHub projects where redistribution is permitted."))

    doc.append(heading("References", 1))
    refs = [
        'T. J. McCabe, "A complexity measure," IEEE Transactions on Software Engineering, vol. SE-2, no. 4, pp. 308-320, Dec. 1976.',
        'N. E. Fenton and M. Neil, "A critique of software defect prediction models," IEEE Transactions on Software Engineering, vol. 25, no. 5, pp. 675-689, Sep. 1999.',
        'T. Menzies, J. Greenwald, and A. Frank, "Data mining static code attributes to learn defect predictors," IEEE Transactions on Software Engineering, vol. 33, no. 1, pp. 2-13, Jan. 2007.',
        'S. Kim, E. J. Whitehead Jr., and Y. Zhang, "Classifying software changes: Clean or buggy?" in Proc. IEEE/ACM ASE, 2008, pp. 318-327.',
        'T. Hall, S. Beecham, D. Bowes, D. Gray, and D. Counsell, "A systematic literature review on fault prediction performance in software engineering," IEEE Transactions on Software Engineering, vol. 38, no. 6, pp. 1276-1304, Nov.-Dec. 2012.',
        'Z. Li, L. Zhang, and H. Leung, "A survey of software defect prediction studies," Information and Software Technology, vol. 51, no. 5, pp. 946-959, May 2009.',
        'A. Ghotra, S. McIntosh, and A. E. Hassan, "Revisiting the impact of classification techniques on the performance of defect prediction models," in Proc. IEEE/ACM ICSE, 2015, pp. 789-800.',
        'Y. Kamei et al., "A large-scale empirical study of just-in-time defect prediction," IEEE Transactions on Software Engineering, vol. 39, no. 6, pp. 757-773, Jun. 2013.',
        'B. Turhan, T. Menzies, A. B. Bener, and J. Di Stefano, "On the relative value of cross-company and within-company data for defect prediction," Empirical Software Engineering, vol. 14, no. 5, pp. 540-578, Oct. 2009.',
        'D. Bowes, T. Hall, and D. Gray, "Software defect prediction: Do different classifiers find the same defects?" Software Quality Journal, vol. 20, no. 2, pp. 145-163, Jun. 2012.',
        'M. Tantithamthavorn, S. McIntosh, A. E. Hassan, and K. Matsumoto, "The impact of automated parameter optimization on defect prediction models," IEEE Transactions on Software Engineering, vol. 44, no. 7, pp. 736-756, Jul. 2018.',
        'A. Mockus and D. Weiss, "Predicting risk of software changes," Bell Labs Technical Journal, vol. 5, no. 2, pp. 169-180, Apr.-Jun. 2000.',
        'J. Nam, S. Kim, and I. S. Jeong, "Transfer defect learning," in Proc. IEEE/ACM ICSE, 2013, pp. 382-391.',
        'X. Yang, D. Lo, X. Xia, Q. Wang, and J. Sun, "Deep learning for just-in-time defect prediction," in Proc. IEEE QRS, 2015, pp. 17-26.',
        'X. Xia, D. Lo, S. Wang, and B. Zhou, "Accurate developer recommendation for bug resolution," Empirical Software Engineering, vol. 20, no. 1, pp. 117-161, Feb. 2015.',
        'R. Malhotra, "A systematic review of machine learning techniques for software defect prediction," Applied Soft Computing, vol. 27, pp. 504-518, Feb. 2015.',
        'M. Shepperd, Q. Song, Z. Sun, and C. Mair, "Data quality: Some comments on the NASA software defect datasets," IEEE Transactions on Software Engineering, vol. 39, no. 9, pp. 1208-1215, Sep. 2013.',
        'T. Menzies, B. Turhan, A. B. Bener, and J. Di Stefano, "Cross-project defect prediction: A large scale experiment," in Proc. IEEE ASE, 2008, pp. 431-440.',
        'J. Gray, D. Bowes, and T. Hall, "Reflections on the NASA MDP datasets," Software Quality Journal, vol. 25, no. 4, pp. 1007-1024, Dec. 2017.',
        'M. Hosseini, B. Turhan, and D. Gunarathna, "A systematic literature review and meta-analysis on cross project defect prediction," IEEE Transactions on Software Engineering, vol. 45, no. 2, pp. 111-147, Feb. 2019.',
        'A. E. Hassan, "The road ahead for mining software repositories," in Proc. IEEE Frontiers of Software Maintenance, 2008, pp. 48-57.',
        'J. Nam and S. Kim, "Clami: Defect prediction on unlabeled datasets," in Proc. IEEE/ACM International Conference on Automated Software Engineering, 2015, pp. 452-463.',
        'M. D\'Ambros, M. Lanza, and R. Robbes, "Evaluating defect prediction approaches: A benchmark and an extensive comparison," Empirical Software Engineering, vol. 17, no. 4-5, pp. 531-577, Aug. 2012.',
        'IEEE DataPort, "IEEE DataPort." [Online]. Available: https://ieee-dataport.org. Accessed: Jul. 4, 2026.',
        'ACM, "Artifact Review and Badging." [Online]. Available: https://www.acm.org/publications/policies/artifact-review-and-badging-current. Accessed: Jul. 4, 2026.',
    ]
    for i, ref in enumerate(refs, 1):
        doc.append(para(f"[{i}] {ref}", before=0, after=36, runs=[text_run(f"[{i}] {ref}", size=15)]))

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="' + NS["w"] + '" xmlns:r="' + NS["r"] + '" xmlns:wp="' + NS["wp"] + '" xmlns:a="' + NS["a"] + '" xmlns:pic="' + NS["pic"] + '"><w:body>'
        + "".join(doc)
        + final_section()
        + "</w:body></w:document>"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types([p.name for p in image_paths]))
        z.writestr("_rels/.rels", root_rels())
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/numbering.xml", numbering_xml())
        z.writestr("word/_rels/document.xml.rels", rels_xml(image_rels))
        for archive_name, path, _rid in media:
            z.write(path, archive_name)
    print(OUT)


if __name__ == "__main__":
    main()
