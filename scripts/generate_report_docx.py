#!/usr/bin/env python3
"""Generate a Word report for Task 2 without external docx dependencies."""

from __future__ import annotations

import html
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "task2_act_calvin_report.docx"
PLOT = ROOT / "plots" / "task2_loss_curve.png"


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def p(text: str = "", style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{esc(text)}</w:t></w:r></w:p>"


def bullet(text: str) -> str:
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
        f"<w:r><w:t>{esc(text)}</w:t></w:r></w:p>"
    )


def heading(text: str, level: int = 1) -> str:
    return p(text, f"Heading{level}")


def table(rows: list[list[object]]) -> str:
    cells = []
    for row in rows:
        cell_xml = "".join(
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"{p(str(cell))}</w:tc>"
            for cell in row
        )
        cells.append(f"<w:tr>{cell_xml}</w:tr>")
    return (
        "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblLook w:val=\"04A0\"/></w:tblPr>"
        + "".join(cells)
        + "</w:tbl>"
    )


def image() -> str:
    # 5.7 x 3.5 inch in EMUs.
    cx, cy = 5212080, 3200400
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0"
        xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="1" name="Task2 loss curve"/>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="task2_loss_curve.png"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="rIdImage1"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:rPr><w:b/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
    <w:rPr><w:b/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph">
    <w:name w:val="List Paragraph"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:ind w:left="720"/></w:pPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>
    </w:tblBorders></w:tblPr>
  </w:style>
</w:styles>
"""


def numbering_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0">
    <w:lvl w:ilvl="0">
      <w:start w:val="1"/>
      <w:numFmt w:val="bullet"/>
      <w:lvlText w:val="•"/>
      <w:lvlJc w:val="left"/>
      <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>
    </w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>
"""


def document_xml() -> str:
    body: list[str] = []
    body.append(p("Task 2: ACT Policy Generalization on CALVIN with LeRobot", "Title"))
    body.append(p("GitHub repository: TODO: paste public repository link"))
    body.append(p("Model weights: TODO: paste cloud storage link for exports/task2_act_weights.tar.gz"))
    body.append(p("Generated artifact: exports/task2_act_weights.tar.gz"))

    body.append(heading("1. Task Background"))
    body.append(p(
        "This experiment studies cross-environment generalization for embodied visuomotor policy learning. "
        "The policy is trained with Action Chunking Transformer (ACT) on CALVIN environment A and on mixed "
        "environments A+B+C, then evaluated zero-shot on unseen environment D."
    ))

    body.append(heading("2. Dataset"))
    body.append(p(
        "The training data uses the CALVIN ABC-D split in LeRobot v3.0 format. Environment labels A, B and C "
        "are recovered from the official CALVIN task_ABC_D metadata. The D test subset starts at episode 17870 "
        "in the ABC-D LeRobot dataset, immediately after the 17870 ABC training episodes."
    ))
    body.append(table([
        ["Split", "Episode selection", "Episodes", "Purpose"],
        ["A train", "Environment A", "512", "single-environment policy training"],
        ["ABC train", "512 from A, 512 from B, 512 from C", "1536", "multi-environment policy training"],
        ["A validation", "held-out A episodes", "64", "in-domain validation"],
        ["ABC validation", "held-out A+B+C episodes", "64", "mixed-domain validation"],
        ["D test", "unseen D episodes", "64", "zero-shot cross-environment evaluation"],
    ]))

    body.append(heading("3. Method"))
    body.append(p(
        "Both models use the LeRobot ACTPolicy implementation with the same architecture and hyperparameters. "
        "The input observation contains the top camera image and the 15-dimensional robot state. The target is "
        "the 7-dimensional end-effector action. ACT predicts a chunk of future actions and is trained with "
        "action L1 loss over the chunk."
    ))
    body.append(table([
        ["Item", "Value"],
        ["Policy", "LeRobot ACTPolicy"],
        ["Visual input", "observation.images.top, 200 x 200 RGB"],
        ["State input", "observation.state, 15 dimensions"],
        ["Action output", "action, 7 dimensions"],
        ["Chunk size", "100"],
        ["n_action_steps", "100"],
        ["Optimizer", "AdamW"],
        ["Learning rate", "1e-5"],
        ["Weight decay", "1e-4"],
        ["Batch size", "4"],
        ["Training steps", "5000"],
        ["Checkpoints", "2500 and 5000 steps; final checkpoint is checkpoints/last"],
    ]))

    body.append(heading("4. Training Results"))
    body.append(p(
        "Both models converged during training. The A-only model reduced total loss from 71.38 to 0.477, "
        "while the ABC model reduced total loss from 74.24 to 0.454. Final logged Action L1 losses were "
        "0.1106 for A-only and 0.1002 for ABC."
    ))
    body.append(table([
        ["Run", "Train episodes", "Step 1 loss", "Step 5000 loss", "Step 5000 Action L1"],
        ["ACT trained on A", "512", "71.3845", "0.4770", "0.1106"],
        ["ACT trained on A+B+C", "1536", "74.2411", "0.4539", "0.1002"],
    ]))
    body.append(p("Figure 1. Training Action L1 loss curves."))
    body.append(image())

    body.append(heading("5. Evaluation Results"))
    body.append(p(
        "The evaluation metric is chunk-level mean Action L1 error. A success proxy is also reported as the "
        "fraction of samples with mean L1 below 0.05."
    ))
    body.append(table([
        ["Run", "Split", "Mean L1", "Median L1", "Std L1", "Success proxy", "Samples", "Episodes"],
        ["ACT trained on A", "A val", "0.3595", "0.2988", "0.2734", "0.0505", "3845", "64"],
        ["ACT trained on A", "D test", "0.8978", "0.8921", "0.0676", "0.0000", "3796", "64"],
        ["ACT trained on A+B+C", "ABC val", "0.3385", "0.2664", "0.2479", "0.0083", "3723", "64"],
        ["ACT trained on A+B+C", "D test", "0.9090", "0.9008", "0.0738", "0.0000", "3796", "64"],
    ]))

    body.append(heading("6. Analysis"))
    body.append(p(
        "The mixed A+B+C model achieved lower validation error on the mixed validation split than the A-only "
        "model achieved on A validation, indicating that multi-environment training did not hurt optimization "
        "under the same ACT architecture and hyperparameters. However, the zero-shot D result did not improve: "
        "the A-only model reached 0.8978 mean L1 on D, while the ABC model reached 0.9090."
    ))
    body.append(p(
        "This suggests that visual distribution shift from the unseen D environment remains the dominant factor. "
        "ACT action chunking helps maintain temporally smooth predictions by producing a sequence of future "
        "actions at each observation, but it does not fully solve visual mismatch when the policy receives "
        "observations outside the training distribution. Once the first observation is encoded with biased visual "
        "features, the entire predicted chunk can inherit that error. This explains why the ABC model improves "
        "mixed-domain validation but does not show a clear zero-shot advantage on D in this run."
    ))
    body.append(p(
        "Overall, the experiment verifies the required A-only versus A+B+C comparison under identical ACT settings. "
        "The results show that broader training data can improve validation behavior on seen environments, but "
        "additional domain randomization, stronger visual augmentation, or more diverse camera/background coverage "
        "would likely be needed to improve robust zero-shot transfer to D."
    ))

    body.append(heading("7. Reproducibility"))
    body.append(table([
        ["Command", "Purpose"],
        ["python scripts/train_act_calvin.py --config configs/task2_act_calvin.yaml --run-kind a", "Train ACT on environment A"],
        ["python scripts/train_act_calvin.py --config configs/task2_act_calvin.yaml --run-kind abc", "Train ACT on mixed A+B+C"],
        ["bash scripts/task2_eval.sh configs/task2_act_calvin.yaml outputs/task2_act_calvin results/final", "Evaluate both policies"],
    ]))

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {''.join(body)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def write_docx() -> None:
    if not PLOT.exists():
        raise FileNotFoundError(PLOT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>
""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
""",
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rIdNumbering" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
  <Relationship Id="rIdImage1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/task2_loss_curve.png"/>
</Relationships>
""",
        )
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/numbering.xml", numbering_xml())
        zf.writestr("word/document.xml", document_xml())
        zf.write(PLOT, "word/media/task2_loss_curve.png")

    # Keep a plain copy of the plot near the report for manual editing if needed.
    shutil.copy2(PLOT, OUT.parent / "task2_loss_curve.png")
    print(OUT)


if __name__ == "__main__":
    write_docx()
