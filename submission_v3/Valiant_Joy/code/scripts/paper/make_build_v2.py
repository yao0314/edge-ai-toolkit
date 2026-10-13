"""Derive scripts/paper/build_tex_v2.py from build_tex.py for the rewritten paper (paper/v2).

The converter, checks and data-driven tables are reused unchanged; what changes is where the source
lives, where each float sits, the figure captions (rewritten for a reader who skims), the table
labels and numbering (placement order changed), and one new summary table of the hypotheses of
Section V, whose values are the registered results in results/paper/FINDINGS.md.

    python scripts/paper/make_build_v2.py && python scripts/paper/build_tex_v2.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "build_tex.py").read_text(encoding="utf-8")
s = src


def sub(old: str, new: str, count: int = 1) -> None:
    global s
    if old not in s:
        raise SystemExit(f"pattern not found: {old[:70]!r}")
    s = s.replace(old, new, count)


# ---- paths
sub('SRC = ROOT / "paper" / "manuscript.md"', 'SRC = ROOT / "paper" / "v2" / "manuscript.md"')
sub('OUTDIR = ROOT / "paper" / "tex"', 'OUTDIR = ROOT / "paper" / "v2" / "tex"')

# ---- float plan: the order of the story, not the order of the experiments
i = s.index("PLAN = [")
j = s.index("]\n", i) + 2
s = s[:i] + '''PLAN = [
    ("I", "", 1, "fig", 1),        # the case, shown on page one: the eleven-release sweep
    ("III", "B", 1, "tab", 1),     # the six models and their files
    ("IV", "", 1, "tab", 2),       # the absolute latencies behind Fig. 1
    ("V", "", 1, "tab", 3),        # five explanations, four rejected
    ("V", "", 2, "fig", 2),        # the localization
    ("VI", "", 1, "fig", 3),       # who the release affects
    ("VI", "", 3, "tab", 4),       # what INT8 buys at a fixed release (floats to the top of Sec. VII)
    ("VII", "A", 1, "fig", 4),     # which method wins, by model class
    ("VII", "D", 1, "fig", 5),     # the default activation type
]
''' + s[j:]

# ---- no paragraph is loosened in the new text
i = s.index("LOOSEN = (")
j = s.index(")\n", i) + 2
s = s[:i] + "LOOSEN: tuple[str, ...] = ()\n" + s[j:]

# ---- table captions: section references and labels follow the new numbering
sub("the deployment-resolution file of Section~\\\\ref{secIVA}.}\\\\label{tab1}",
    "the deployment-resolution file of Section~\\\\ref{secVIIA}.}\\\\label{tab1}")
sub("blocks come from the two protocols of Section~\\\\ref{secIIIC}.}\\\\label{tab2}",
    "blocks come from the two kinds of run of Section~\\\\ref{secIIIC}.}\\\\label{tab4}")
sub('"\\\\caption{Absolute latencies behind the MobileViT-XXS ratios of Fig.~\\\\ref{fig2}: "',
    '"\\\\caption{Absolute latencies behind the MobileViT-XXS ratios of Fig.~\\\\ref{fig1}: "')
sub('f"median of all {sess} sessions in ms under the version protocol, the same three "',
    'f"median of all {sess} sessions in ms, the same three "')
sub('"rule marks the 1.24.4 step.}\\\\label{tab3}\\n"', '"rule marks the 1.24.4 step.}\\\\label{tab2}\\n"')

# ---- the availability note points at the new section numbers
sub('("of the MobileViT-XXS files of Section~\\\\ref{secVA}:", [',
    '("of the files of Section~\\\\ref{secIV}: MobileViT-XXS", [')
sub('("MobileViT-XXS 256\\\\,px (Section~\\\\ref{secVD})", [',
    '("MobileViT-XXS 256\\\\,px (Section~\\\\ref{secVI})", [')

# ---- the hypotheses of Section V, and the table map
HYP = r'''

def table_hypotheses() -> str:
    """Table III: the five explanations of Section V. Its values are registered results, copied
    from results/paper/FINDINGS.md (spin: Exp. 25; MatMul-only: coverage run; profile: Exp. 23c;
    probe: Exp. 27b); the audit checks them there."""
    rows = [
        ("The executed graph changed", "Dump the optimized graph each release builds",
         "Identical at 1.23.2, 1.24.4 and 1.25.1: 698 nodes, same order and attributes"),
        ("The thread pool's spin policy", "Disable intra-op spinning; 27 sessions each way, randomized",
         "The step persists: +24\\%, disjoint sessions"),
        ("A slower operator mix", "Per-operator profile at four threads",
         "Transpose carries 62\\% of the added time, the rest +9\\% together"),
        ("The quantized MatMuls", "A static file quantizing only the weight MatMuls",
         "No step ($-$1\\%); the full static file +35\\% in the same run"),
        ("The eight-bit Transposes", "Time them alone, net of a Transpose-free control",
         "87\\% of the step at four threads; no step at one"),
    ]
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    return ("\\begin{table}[tb]\n\\centering\n"
            "\\caption{Five explanations of the 1.24.4 step, tested in the order given, each "
            "against a result written down before the measurement. The first four are rejected "
            "or fall short; the fifth locates the step (Fig.~\\ref{fig2}).}\\label{tab3}\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}\n"
            "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.8cm}"
            ">{\\raggedright\\arraybackslash}p{4.1cm}>{\\raggedright\\arraybackslash}p{4.5cm}@{}}\n"
            "\\toprule\nExplanation & Test & Result \\\\\n\\midrule\n" + body +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


TABLES = {1: table_settings, 2: table_versions, 3: table_hypotheses, 4: table_results}
'''
sub("TABLES = {1: table_settings, 2: table_results, 3: table_versions, 4: table_scope}", HYP.strip("\n"))

# ---- figure captions, written for a reader who looks at the figures first
i = s.index("    return {\n        1: (\"fig3_default_config.pdf\",")
j = s.index("    }\n", i) + len("    }\n")
s = s[:i] + r'''    return {
        1: ("fig4_version_sweep.pdf",
            "One set of byte-identical INT8 files on eleven ONNX Runtime releases: INT8 speedup "
            "over the FP32 file of the same release, three sessions per point, whiskers over the "
            "INT8 sessions. At 1.24.4 (shaded from there on) the static MobileViT-XXS file falls "
            "at four threads while its one-thread speedup and the dynamic file hold; at 1.25.1 it "
            "falls again because its FP32 baseline sped up (Table~\\ref{tab2})."),
        2: ("fig5_localization.pdf",
            "Where the step lives. Added latency at 1.24.4 against 1.23.2 for the full static "
            "file (black), for a probe holding only its eight-bit Transposes (blue) and for "
            "three controls (orange, hatched). Four Transposes whose moved block is two bytes wide carry "
            "the step; the other 52, the same shapes in float32 and one thread do not. Probe "
            "values are medians over the sessions that pass its registered core-delivery check."),
        3: ("fig6_scope.pdf",
            "Who the release affects. Change in four-thread latency from 1.23.2 to 1.24.4 for "
            "each model's static INT8 file (bars) and its FP32 sibling (circles). The gray band "
            "is the 7\\% floor; a bar is colored only when it leaves the band with disjoint "
            "sessions. Only the hybrid model steps up, and YOLOv8n steps down."),
        4: ("fig1_method_by_class.pdf",
            "Which INT8 method wins, by model, on release 1.30.0: speedup over FP32 at one thread "
            "(left) and four (right), log axis. Circles are static INT8, squares dynamic INT8. On "
            "the convolutional models (unshaded) static wins by an order of magnitude; on the "
            f"attention and hybrid models (shaded) dynamic wins. Values in Table~\\ref{{tab4}} "
            f"(YOLOv8n at {yolo_px}\\,px)."),
        5: ("fig3_default_config.pdf",
            "The quantizer's default activation type is slower than FP32. Static INT8 speedup "
            "over FP32 at four threads on 1.30.0 for two formats, QDQ and QOperator (quantized operators "
            "stored directly), and two "
            "activation/weight types; U8S8 is the fast path, S8S8 the documented default. A run "
            "separate from Table~\\ref{tab4}, whose QDQ U8S8 cells "
            f"({sig3(tab_u8[0])}$\\times$, {sig3(tab_u8[1])}$\\times$) differ from this run's "
            f"({sig3(run_u8[0])}$\\times$, {sig3(run_u8[1])}$\\times$)."),
    }
''' + s[j:]

# ---- the Table IV block headings use the paper's names for the two kinds of run
sub("Harness protocol: static and dynamic from the same run",
    "Within one release: static and dynamic from the same run")
sub("Version protocol: median of three isolated sessions",
    "Release comparisons: median of three isolated sessions")

# ---- layout fixes from the v2 audit
sub('out.append(f"\\n\\\\subsection{{{tex_escape(text)}}}"',
    'out.append(f"\\n\\\\subsection{{{typeset_numbers(tex_escape(text))}}}"')
sub("\\hyphenation{QuantizeLinear DequantizeLinear ConvInteger MobileViT}",
    "\\hyphenation{QuantizeLinear DequantizeLinear ConvInteger MobileViT MobileNetV ImageNet}")
sub('f"\\\\times 10^{{-{m[2]}}}$"', 'f"\\\\,{{\\\\times}}\\\\,10^{{-{m[2]}}}$"')
sub("(including \"\n                \"the 396-row sweep)", "(including \"\n                \"the 396 measurements of the release sweep)")
sub("last accessed 18-19 Sep 2026.", "last accessed 18--19 Sep 2026.")
sub("% Generated by scripts/paper/build_tex.py from paper/manuscript.md",
    "% Generated by scripts/paper/build_tex_v2.py from paper/v2/manuscript.md")
sub("% the four figure PDFs next to it.", "% the five figure PDFs next to it.")

# ---- the public artifact repository
sub('AVAILABILITY = ("The quantized ONNX files, measurement scripts, raw measurement CSVs (including "\n'
    '                "the 396 measurements of the release sweep) and dated decision log are available from the authors on "\n'
    '                "request.',
    'AVAILABILITY = ("The model files listed below by SHA-256 prefix, the measurement and analysis scripts, "\n'
    '                "the raw measurement CSVs (including the 396 measurements of the release sweep) and the dated "\n'
    '                "decision log are available at \\\\url{https://github.com/Pric0123/taai2026-int8-cpu-artifact}; "\n'
    '                "the other model files are available from the authors on request.')

out = HERE / "build_tex_v2.py"
out.write_text(s, encoding="utf-8")
print("wrote", out)
