# -*- coding: utf-8 -*-
"""Converts the final-report markdown draft into a conference-paper-style
.docx, matching the project's established house style (Calibri body,
compact margins) but sized for an 8-12 page document: headings, inline
bold/italic/code, bullet lists, tables, figures with captions, and a
hanging-indent reference list.
"""
import re
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PROJECT_ROOT = Path(__file__).resolve().parent
DRAFT_PATH = PROJECT_ROOT / "report" / "final_report_draft.md"
OUT_PATH = PROJECT_ROOT / "Final_Report.docx"

NAVY = RGBColor(0x1F, 0x2A, 0x44)
GRAY = RGBColor(0x55, 0x55, 0x55)
FIG_CAPTION_GRAY = RGBColor(0x40, 0x40, 0x40)

INLINE_PATTERN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")


def add_inline_runs(paragraph, text, base_size=10.5, base_color=None):
    """Parses **bold**, *italic*, `code` spans within a line of text and
    adds corresponding runs to `paragraph`."""
    parts = INLINE_PATTERN.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*") and part.endswith("*") and not part.startswith("**"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(base_size - 0.5)
        else:
            run = paragraph.add_run(part)
        run.font.size = Pt(base_size)
        if base_color:
            run.font.color.rgb = base_color


def set_cell_shading(cell, hex_color):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shd)


def set_header_row_repeat(row):
    """Marks a table row as a header row that repeats on each page the
    table spans (prevents an awkward page break separating the header
    from its data, as happened with Table 1)."""
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def parse_table(lines, start_idx):
    """Parses a markdown pipe table starting at lines[start_idx]; returns
    (rows, next_idx)."""
    rows = []
    i = start_idx
    while i < len(lines) and lines[i].strip().startswith("|"):
        row_line = lines[i].strip()
        if re.match(r"^\|[\s:-]+\|$", row_line.replace("|", "|").replace(" ", "")) or set(row_line.replace("|", "").replace(" ", "")) <= set("-:"):
            i += 1
            continue
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def build():
    doc = Document()

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.0

    lines = DRAFT_PATH.read_text(encoding="utf-8").split("\n")

    i = 0
    first_heading_seen = False
    in_references = False

    # Title (line 0) and author line (lines 2-3)
    title_text = lines[0].lstrip("# ").strip()
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run(title_text)
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = NAVY
    title_p.paragraph_format.space_after = Pt(4)

    author_p = doc.add_paragraph()
    author_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = author_p.add_run(lines[2].strip())
    run.font.size = Pt(11.5)
    author_p.paragraph_format.space_after = Pt(0)

    affil_p = doc.add_paragraph()
    affil_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = affil_p.add_run(lines[3].strip())
    run.font.size = Pt(10)
    run.font.color.rgb = GRAY
    affil_p.paragraph_format.space_after = Pt(16)

    i = 5  # skip past title/author block and blank line

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Figure placeholder: [FIGURE N: path — caption]
        fig_match = re.match(r"\[FIGURE\s+(\d+):\s*(.+?)\s*[\u2014-]\s*(.+)\]", stripped)
        if fig_match:
            fig_num, fig_path, caption = fig_match.groups()
            img_path = PROJECT_ROOT / fig_path
            if img_path.exists():
                doc.add_picture(str(img_path), width=Inches(5.8))
                last_p = doc.paragraphs[-1]
                last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_p = doc.add_paragraph()
            cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cap_p.add_run(f"Figure {fig_num}. {caption}")
            run.italic = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = FIG_CAPTION_GRAY
            cap_p.paragraph_format.space_after = Pt(10)
            i += 1
            continue

        # Table
        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            if rows:
                ncols = len(rows[0])
                table = doc.add_table(rows=len(rows), cols=ncols)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.style = "Table Grid"
                for r, row in enumerate(rows):
                    for c, cell_text in enumerate(row):
                        if c >= ncols:
                            continue
                        cell = table.cell(r, c)
                        cell.text = ""
                        p = cell.paragraphs[0]
                        add_inline_runs(p, cell_text, base_size=9.5)
                        if r == 0:
                            set_cell_shading(cell, "1F2A44")
                            for run in p.runs:
                                run.bold = True
                                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                if len(rows) > 1:
                    set_header_row_repeat(table.rows[0])
                doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        # Table caption/label line (bold, starts with "**Table")
        if stripped.startswith("**Table"):
            p = doc.add_paragraph()
            add_inline_runs(p, stripped, base_size=10)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # Italic footnote-style line under a table, e.g. "*Exploitation column..."
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(stripped.strip("*"))
            run.italic = True
            run.font.size = Pt(9)
            run.font.color.rgb = GRAY
            i += 1
            continue

        # Headings
        if stripped.startswith("## "):
            text = stripped[3:].strip()
            if text.lower().startswith("references"):
                in_references = True
            elif text.lower().startswith("appendix"):
                in_references = False
            p = doc.add_heading(level=1)
            run = p.add_run(text)
            run.font.size = Pt(14)
            run.font.color.rgb = NAVY
            run.font.name = "Calibri"
            p.paragraph_format.space_before = Pt(12) if first_heading_seen else Pt(2)
            p.paragraph_format.space_after = Pt(4)
            first_heading_seen = True
            i += 1
            continue

        if stripped.startswith("### "):
            text = stripped[4:].strip()
            p = doc.add_heading(level=2)
            run = p.add_run(text)
            run.font.size = Pt(12)
            run.font.color.rgb = NAVY
            run.font.name = "Calibri"
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            i += 1
            continue

        # Bullet list item
        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline_runs(p, stripped[2:].strip(), base_size=10.5)
            p.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # Reference entry (plain paragraph, hanging indent, inside References section)
        if in_references:
            p = doc.add_paragraph()
            add_inline_runs(p, stripped, base_size=10)
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(6)
            i += 1
            continue

        # Abstract label handling: "## Abstract" already caught by H1 branch above.
        # Ordinary paragraph
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_inline_runs(p, stripped, base_size=10.5)
        i += 1

    doc.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
