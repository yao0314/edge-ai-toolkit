"""Render one of this project's Markdown deliverables to PDF, with nothing to install.

pandoc and weasyprint are not here and adding them means a download. PyMuPDF is already a
dependency and its Story engine flows HTML+CSS into pages, which is what a two-page report needs
and what matplotlib is bad at.

The Markdown subset is the one these reports actually use: ATX headings, paragraphs, blockquotes,
fenced code, tables, images, and inline bold / italic / code. Anything else passes through as text
rather than being silently dropped.

  python scripts/md_to_pdf.py submission/REPORT.md --out submission/REPORT.pdf
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pymupdf

CSS = """
body { font-family: sans-serif; font-size: 9.0pt; line-height: 1.38; color: #1c1c1c; }
h1 { font-size: 15.5pt; font-weight: bold; margin-top: 0; margin-bottom: 5pt; color: #10212e; }
h2 { font-size: 10.8pt; font-weight: bold; margin-top: 10pt; margin-bottom: 3pt; color: #0072b2; }
p  { margin-top: 0; margin-bottom: 5pt; text-align: justify; }
blockquote { font-size: 9.8pt; color: #10212e; margin-top: 4pt; margin-bottom: 9pt;
             margin-left: 0; padding-left: 10pt; border-left: 3px solid #0072b2; }
code { font-family: monospace; font-size: 8.6pt; color: #0b4f72; }
pre  { font-family: monospace; font-size: 7.9pt; color: #233; background-color: #f2f4f5;
       padding: 6pt; margin-top: 4pt; margin-bottom: 7pt; line-height: 1.3; }
table { font-size: 7.9pt; margin-top: 3pt; margin-bottom: 6pt; width: 100%; }
th { font-weight: bold; text-align: left; color: #10212e; border-bottom: 1px solid #c8cfd3;
     padding-bottom: 2pt; }
td { padding-top: 1.6pt; padding-bottom: 1.6pt; border-bottom: 1px solid #eceff1; }
img { width: 410; }
"""


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    return s


def convert(md: str, base: Path) -> str:
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):                                   # fenced code
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            out.append("<pre>" + "<br/>".join(buf) + "</pre>")
        elif ln.startswith("!["):                                   # image
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln)
            if m:
                # the path is resolved INSIDE the Archive, so it stays relative to the md file
                out.append(f'<p><img src="{m.group(1)}"/></p>')
        elif ln.startswith("|"):                                    # table
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            i -= 1
            body = ["<tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr>"]
            for r in rows[2:]:                                      # row 1 is the --- separator
                body.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("<table>" + "".join(body) + "</table>")
        elif ln.startswith("> "):                                   # blockquote, possibly wrapped
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            i -= 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
        elif ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{min(lvl, 3)}>{inline(ln.lstrip('# ').strip())}</h{min(lvl, 3)}>")
        elif ln.strip():                                            # paragraph, joined to its wraps
            buf = []
            while i < len(lines) and lines[i].strip() and not lines[i][:1] in "#>|!`":
                buf.append(lines[i].strip())
                i += 1
            i -= 1
            out.append("<p>" + inline(" ".join(buf)) + "</p>")
        i += 1
    return "<html><body>" + "".join(out) + "</body></html>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("md")
    ap.add_argument("--out")
    ap.add_argument("--margin", type=float, default=40.0, help="page margin in points")
    a = ap.parse_args()

    src = Path(a.md)
    out = Path(a.out or src.with_suffix(".pdf"))
    # Story resolves <img src> through an Archive, not the filesystem: without one the picture is
    # silently replaced by its alt text, which is how a report ships with [image] where a figure was
    story = pymupdf.Story(html=convert(src.read_text(encoding="utf-8"), src.parent), user_css=CSS,
                          archive=pymupdf.Archive(src.parent))
    page = pymupdf.paper_rect("a4")
    frame = page + (a.margin, a.margin, -a.margin, -a.margin)
    writer = pymupdf.DocumentWriter(str(out))
    more, n = 1, 0
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(frame)
        story.draw(dev)
        writer.end_page()
        n += 1
    writer.close()
    print(f"wrote {out}  ({n} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
