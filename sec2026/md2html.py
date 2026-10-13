"""Tiny Markdown -> HTML for the report (headings, paragraphs, lists, tables, bold, code, images).

    python sec2026/md2html.py submission_v2/REPORT.md submission_v2/REPORT.html
then print to PDF with a browser (Edge: msedge --headless --print-to-pdf=REPORT.pdf REPORT.html).
"""
import html
import re
import sys
from pathlib import Path

CSS = """body{font-family:Segoe UI,Arial,sans-serif;font-size:9.6pt;line-height:1.35;margin:0}
h1{font-size:14pt;margin:0 0 4pt}h2{font-size:11pt;margin:8pt 0 3pt;border-bottom:1px solid #ccc}
table{border-collapse:collapse;margin:4pt 0;font-size:8.8pt}td,th{border:1px solid #bbb;padding:1.5pt 5pt}
th{background:#eee}img{width:100%;margin:2pt 0}p,li{margin:2pt 0}ol,ul{margin:2pt 0 2pt 16pt;padding:0}
code{font-family:Consolas,monospace;font-size:8.5pt}@page{size:A4;margin:12mm}"""


def inline(s):
    s = html.escape(s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)


def convert(md):
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            n = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{n}>{inline(ln[n:].strip())}</h{n}>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s\-|:]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append("<table>" + "".join(
                "<tr>" + "".join(f"<{'th' if r == 0 else 'td'}>{inline(c)}</{'th' if r == 0 else 'td'}>" for c in row) + "</tr>"
                for r, row in enumerate(rows)) + "</table>")
            continue
        elif re.match(r"^\d+\. ", ln):
            items = []
            while i < len(lines) and (re.match(r"^\d+\. ", lines[i]) or lines[i].startswith("   ")):
                if re.match(r"^\d+\. ", lines[i]):
                    items.append(re.sub(r"^\d+\. ", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ol>" + "".join(f"<li>{inline(t)}</li>" for t in items) + "</ol>")
            continue
        elif ln.strip():
            para = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||\d+\. )", lines[i + 1]):
                i += 1
                para.append(lines[i])
            out.append(f"<p>{inline(' '.join(para))}</p>")
        i += 1
    return f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>" + "\n".join(out)


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(convert(Path(sys.argv[1]).read_text(encoding="utf-8")), encoding="utf-8")
