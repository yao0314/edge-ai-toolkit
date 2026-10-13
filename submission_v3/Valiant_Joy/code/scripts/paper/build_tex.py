"""Convert paper/manuscript.md into paper/tex/main.tex for Springer's llncs class.

This exists because the Word route was the wrong route. Reproducing LNCS by hand in .docx meant
approximating the layout, and every layout defect in that approximation had to be found by eye -
a clipped figure, a figure pushed past the reference list, a page count that could only be
estimated. llncs.cls does the layout correctly by construction, and the figures are vector PDFs,
which LaTeX places natively.

What this script owns, because the manuscript cannot express it: the four tables (every value is
read from a CSV or a run's summary.json at build time, never typed in), the figure captions, and
where each float is anchored. What it checks, and refuses to build past: every [n] citation has a
reference and every reference is cited; every Fig./Table/Section reference names something that
exists; every float in PLAN is placed, in numbering order.

Output: paper/tex/main.tex plus the figure PDFs copied alongside it, ready to drop into the
Overleaf LNCS template (compiled locally with llncs.cls v2.25, copied into paper/tex from Springer's
LNCS kit; it provides the credits environment). Run make_figures.py first: Table II and Figs 1-2
share paper/figs/_fig_data.csv.

    python scripts/paper/build_tex.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

SRC = ROOT / "paper" / "manuscript.md"
FIGS = ROOT / "paper" / "figs"
OUTDIR = ROOT / "paper" / "tex"
FIG_DATA = FIGS / "_fig_data.csv"                       # written by make_figures.collect()
BISECT = ROOT / "results" / "paper" / "ort_version_bisect.csv"
FACTORIAL = ROOT / "results" / "paper" / "quant_factorial_classification.csv"
FACTORIAL_MVIT = ROOT / "results" / "paper" / "quant_factorial_classification_mobilevit.csv"
# Deployment-resolution checks of Section IV-A, listed in Table I beside the settings of Table II:
# model -> (run, INT8 variant). Their data, input size and calibration are read from the run.
DEPLOY = {"MobileNetV2": ("stress_imagenette_dyn", "p00_int8"),
          "YOLOv8n": ("detection_640", "640_p00_int8")}
# The manuscript is ASCII; names that need an accent get it in the bibliography only.
BIB_ACCENTS = {"Schoenherr": r"Sch\"onherr", "Jegou, H.": r"J\'egou, H.",
               "Dollar, P.": r"Doll\'ar, P.", "Quintana-Orti": r"Quintana-Ort\'{\i}"}

ORDER = ["ResNet-50", "MobileNetV2", "YOLOv8n", "1-D conv AE", "DeiT-tiny", "MobileViT-XXS"]
CLASS = {"DeiT-tiny": "attention", "MobileViT-XXS": "hybrid"}          # everything else: conv.
METRIC = {"top1": "top-1", "mAP50-95": "mAP", "auc": "AUC"}   # Table I caption expands
ROMAN_NUM = {"I": 1, "II": 2, "III": 3, "IV": 4}

# (section, subsection letter or "", paragraph index within that subsection, kind, number).
# Each float is anchored after the paragraph that first discusses it; which page it lands on is
# llncs.cls's decision. Figures and tables are numbered by the order of their environments, so the
# build refuses a PLAN that would place Fig. 2 before Fig. 1.
PLAN = [
    ("III", "A", 1, "tab", 1),     # read in III-B ("Table I lists the six models ..."); anchored
                                   # one paragraph early so it takes the top of III-B's page and
                                   # leaves the next page's top to Table II
    ("III", "B", 1, "tab", 2),     # first cited in III-C ("the convolutional rows of Table II"),
                                   # read in IV-A: anchored with Table I so it prints near both
    ("IV", "D", 1, "fig", 1),      # the default activation type
    ("V", "A", 1, "fig", 2),       # "Fig. 4 plots the INT8 speedup ..."
    ("V", "B", 1, "tab", 3),       # "1.553 to 2.037 ms (Table III)"
    ("V", "C", 1, "fig", 3),       # the localization: where the step lives (Sec. V-C)
    ("V", "D", 1, "tab", 4),       # the scope run, "(462 sessions; Table IV)"
]


# ------------------------------------------------------------------------------ text conversion
def tex_escape(s: str) -> str:
    """Escape the characters LaTeX treats specially. Typography is applied separately."""
    s = s.replace("\\", "\\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    return s


def typeset_numbers(s: str, times: bool = True) -> str:
    """Times signs, en dashes and minus signs, on already-escaped text. References pass
    times=False: there an x belongs to a quoted title, e.g. "(9x9)" in a GitHub issue."""
    if times:
        s = re.sub(r"(\d+)x(\d+)", r"$\1\\times \2$", s)        # 64x64 -> 64 x 64
        s = re.sub(r"(\d)\s*x\b", r"\1$\\times$", s)            # 3.2x; the digit stays text
        s = re.sub(r"(?<=\s)x(?=\s+\d)", r"$\\times$", s)       # 11 releases x 2 models
    # An en dash belongs between two numbers, not inside a part number: 8-9 becomes 8--9, but
    # x86-64 and i7-13620H keep their hyphens, so both sides must be whole numbers. Release ranges
    # (1.16.3-1.23.2) are numbers too. \nobreakdash (amsmath, loaded by newtxmath) keeps a line
    # from ending in "26-" or "1.26.0-".
    s = re.sub(r"(?<![\w.])(\d+\.\d+\.\d+)-(\d+\.\d+\.\d+)(?![\w.])", r"\1\\nobreakdash--\2", s)
    # a sentence-ending period may follow the range ("pp. 740-755."), a decimal point may not
    s = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?!\w|\.\d)",
               r"\1\\nobreakdash--\2", s)
    s = re.sub(r"(?<=[\s(])-(?=\d)", r"$-$", s)                  # -0.28 pt: a minus, unbreakable
    return s


CITED: set[int] = set()
REFERRED: set[str] = set()
# Paragraphs typeset with \looseness=-1, named by their opening words in manuscript.md: TeX sets
# such a paragraph one line shorter when it can do so within its tolerance (slightly wider word
# spaces), and as usual otherwise. Only paragraphs whose last line held a few words, found with the
# per-paragraph line probe (round 3, 2026-09-19); the page budget (15 pages) has no line to spare.
LOOSEN = ("A scope experiment, its reading fixed in advance",
          "Shin [4] bounds the version effect to absolute latency",
          "On an x86 AVX-VNNI laptop CPU, the release moved",
          # widened after the final audit: the repairs cost about ten lines and the page budget
          # has none. TeX sets each of these one line shorter only where it can do so within its
          # tolerance, so an entry that does not help is simply inert.
          "Every ratio in this paper is taken within one protocol",
          "Table I lists the six models",
          "On the four convolutional models",
          "Table II prices the documented wrong choice",
          "On MobileNetV2 the runtime executes",
          "The speed half of the hybrid ranking",
          "The speedups above use U8S8 activations",
          "One set of files, FP32, static and dynamic",
          "The static MobileViT-XXS file ran at FP32 speed",
          "At 1.25.1 the static file did not change",
          "The step is reproduced by the static graph",
          "We did not examine the mechanism of YOLOv8n",
          "Post-training INT8 quantization (PTQ) is the standard way",
          "Integer-arithmetic inference [1]",
          "We measure what INT8 PTQ buys",
          "Section IV measures the baseline",
          "Section V gives the release effect",
          "The framework tax [14]",
          "All measurements ran on one laptop CPU",
          "The latency floor is 7%",
          "The gain holds at deployment resolution",
          "On DeiT-tiny, dynamic INT8 beat static",
          "On the hybrid MobileViT-XXS, full static INT8",
          "At four threads the dynamic speedup showed no step down",
          "Before each measurement below",
          "The mechanism of Section V-C is established in isolation")
LOOSENED: set[str] = set()


def _cite(m: re.Match) -> str:
    keys: list[int] = []
    for part in re.split(r"\s*,\s*", m.group(1)):
        r = re.fullmatch(r"(\d+)\s*(?:\\nobreakdash)?-{1,2}\s*(\d+)", part)
        keys += range(int(r[1]), int(r[2]) + 1) if r else [int(part)]
    CITED.update(keys)
    return "~\\cite{" + ",".join(f"ref{k}" for k in keys) + "}"


def _ref(label: str, text: str) -> str:
    REFERRED.add(label)
    return f"{text}~\\ref{{{label}}}"


def body_text(s: str) -> str:
    s = typeset_numbers(tex_escape(s))
    # [3], [1, 2], [16,17,18], [16-18] with any number of digits; a space before it becomes a tie
    s = re.sub(r"\s*\[(\d+(?:\s*(?:,|(?:\\nobreakdash)?-{1,2})\s*\d+)*)\]", _cite, s)
    # _cite ties a citation to the word before it; after an opening parenthesis the tie would
    # print as "( [7]", so it is dropped there
    s = s.replace("(~\\cite", "(\\cite")
    s = re.sub(r"\bFig\.\s*(\d)\b", lambda m: _ref(f"fig{m[1]}", "Fig."), s)
    s = re.sub(r"\bTable\s+(IV|III|II|I)\b", lambda m: _ref(f"tab{ROMAN_NUM[m[1]]}", "Table"), s)
    # llncs numbers sections in arabic, so the prose cannot carry its own roman numerals.
    s = re.sub(r"\bSection\s+([IVX]+)(?:-([A-G]))?\b",
               lambda m: _ref(f"sec{m[1]}{m[2] or ''}", "Section"), s)
    s = re.sub(r"(\d)\s*(pt|ms|px)\b", r"\1\\,\2", s)
    # program commands are set in typewriter font (LNCS instructions, Sect. 4.8)
    s = s.replace(r"quantize\_dynamic", r"\texttt{quantize\_dynamic}")
    # a C++ identifier from the ONNX Runtime sources; it may break after an underscore
    s = s.replace(r"has\_mlas\_transpose<uint16\_t>",
                  r"\texttt{has\_\allowbreak{}mlas\_\allowbreak{}transpose<uint16\_t>}")
    # statistics stay on one line: rho = 0.93, p = 0.16
    s = re.sub(r"\brho\s*=\s*(\d+(?:\.\d+)?)", r"$\\rho = \1$", s)
    # p = 4.1e-5 is set as 4.1 x 10^-5; typeset_numbers leaves "e-5" alone (no digit before '-')
    s = re.sub(r"(?<![\w.])p\s*=\s*(\d+\.\d+)(?:e-(\d+)\b)?",
               lambda m: f"$p = {m[1]}" + (f"\\times 10^{{-{m[2]}}}$" if m[2] else "$"), s)
    # QuantizeLinear/DequantizeLinear and INT8/FP32 may break after the slash; Q/DQ may not
    s = re.sub(r"(?<=[A-Za-z0-9])(?<!\bQ)/(?=[A-Za-z])", r"/\\allowbreak{}", s)
    return s


def ref_text(s: str) -> str:
    """A reference entry: URLs go into \\url so they break; everything else is escaped."""
    out = []
    for part in re.split(r"(https?://[^\s,]+)", s):
        if re.match(r"https?://", part):
            out.append("\\url{" + part.rstrip(".);") + "}" + part[len(part.rstrip(".);")):])
        else:
            part = typeset_numbers(tex_escape(part), times=False)
            for plain, accented in BIB_ACCENTS.items():
                part = part.replace(plain, accented)
            out.append(part)
    return "".join(out)


# ------------------------------------------------------------------------------------- tables
def _minus(s: str) -> str:
    return s.replace("-", "$-$")


def _speed(v: float) -> str:
    # three significant digits either way: 3.210 / 0.914 / 0.0676. A phantom fourth decimal keeps
    # the decimal points of a right-aligned column in line (3.210 above 0.0676).
    return f"{v:.3f}\\phantom{{0}}" if v >= 0.1 else f"{v:.4f}"


def _mid(s: str, rows: int = 2) -> str:
    """A cell that belongs to the next `rows` rows (e.g. one FP32 latency for a static and a
    dynamic row), set in the first row and lowered to their vertical middle, as \\multirow would,
    without the package."""
    if rows < 2 or not s:
        return s
    return f"\\raisebox{{-{(rows - 1) / 2:g}\\normalbaselineskip}}[0pt][0pt]{{{s}}}"


def _ms(v: float) -> str:
    return f"{v:#.3g}".rstrip(".")


def _fig_data() -> pd.DataFrame:
    if not FIG_DATA.exists():
        raise SystemExit(f"{FIG_DATA} missing: run scripts/paper/make_figures.py first")
    df = pd.read_csv(FIG_DATA)
    df["k"] = df["model"].map(ORDER.index)
    df["m"] = df["method"].map({"static": 0, "dynamic": 1})
    return df.sort_values(["k", "px", "m"])


def _summary(run: str) -> dict:
    return json.loads((ROOT / "results" / run / "summary.json").read_text(encoding="utf-8"))


def _dataset(ad: dict, cfg: dict) -> str:
    args = cfg.get("adapter_args", {})
    if ad.get("task") == "detection":
        return "COCO"                    # a val2017 subset; its size goes in the Table I caption
    if ad.get("task") == "anomaly":
        return {"ecg5000": "ECG5000"}.get(ad.get("source"), str(ad.get("source")))
    if ad.get("dataset") == "cifar10":
        return "CIFAR-10"
    if "imagenette" in str(args.get("data_root", "")).lower():
        return "Imagenette"
    return str(ad.get("dataset"))


def table_settings() -> str:
    """Table I: the six models, their data, input, metric, static calibration and dynamic ops."""
    df = _fig_data()
    rows, sizes, deploy_n = [], {}, {}
    for model in ORDER:
        sub = df[df["model"] == model]
        run = sub["run"].iloc[0]
        s = _summary(run)
        ad, cfg = s["adapter"], s["config"]
        sizes.setdefault(_dataset(ad, cfg), ad.get("n_val") or ad.get("n_test"))
        px = "/".join(str(p) for p in sorted(int(p) for p in sub["px"].unique()))
        calib = sub.loc[sub["method"] == "static", "calib"].dropna().unique()
        if len(calib) > 1:
            raise SystemExit(f"{model}: static calibration differs between settings: {calib}")
        calib = calib[0].replace("percentile", "pct.") if len(calib) else "--"
        dyn = [p["op_types_to_quantize"] for p in cfg["sweep"]["precisions"]
               if isinstance(p, dict) and p.get("method") == "dynamic"]
        dyn = ", ".join(dyn[0]) if dyn else "--"
        rows.append(f"{model} & {CLASS.get(model, 'conv.')} & {_dataset(ad, cfg)} & "
                    f"{px} & {METRIC[sub['metric'].iloc[0]]} & {calib} & {dyn} \\\\")
        if model in DEPLOY:
            # the deployment-resolution file of Section IV-A, read from its own run, on a second
            # row (one merged row would be too wide for the text block)
            drun, dvar = DEPLOY[model]
            dsweep = pd.read_csv(ROOT / "results" / drun / "sweep.csv")
            drow = dsweep[dsweep["variant"] == dvar].iloc[0]
            ds = _summary(drun)
            dpx = int(drow["img_size"]) if "img_size" in drow and pd.notna(drow["img_size"]) \
                else int(ds["adapter"]["img_size"])
            ddata = _dataset(ds["adapter"], ds["config"])
            dn = ds["adapter"].get("n_val") or ds["adapter"].get("n_test")
            sizes.setdefault(ddata, dn)
            if dn != sizes[ddata]:
                # this file was evaluated on a different number of samples than the model's
                # other settings: the caption names both counts
                deploy_n[ddata] = (dn, dpx)
            dcalib = str(drow["calib_method_chosen"]).replace("percentile", "pct.")
            rows.append(f" & & {ddata if ddata != _dataset(ad, cfg) else ''} & {dpx} & & "
                        f"{dcalib} & \\\\")
    evals = [f"{n:,} {'COCO val2017' if d == 'COCO' else d}"
             + (f" ({deploy_n[d][0]:,} for the {deploy_n[d][1]}\\,px row)" if d in deploy_n else "")
             for d, n in sizes.items()]
    evals = ", ".join(evals[:-1]) + " and " + evals[-1]
    return ("\\begin{table}[tb]\n\\centering\n"
            f"\\caption{{Models and settings. Quality is evaluated on {evals} samples. "
            "Input is in px, or samples for the autoencoder. Calib.: static INT8 calibration "
            "(pct.\\ = percentile), chosen per file "
            "by label-free SQNR (not logged for the autoencoder). Dynamic INT8 quantizes the "
            "listed operator types where one input is a weight, so attention-score MatMuls stay "
            "FP32; the autoencoder has no dynamic variant. A second row gives "
            "the deployment-resolution file of Section~\\ref{secIVA}.}\\label{tab1}\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}lllllll@{}}\n\\toprule\n"
            "Model & Class & Data & Input & Metric & Calib. & Dynamic ops \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table_results() -> str:
    """Table II: speed at 1 and 4 threads, quality and size, every setting and method."""
    df = _fig_data()
    # the release each row was timed on: the version protocol names it, a harness run logs it
    releases = {s.split()[-1] for s in df["source"] if s.startswith("version protocol")}
    releases |= {_summary(r)["env"]["onnxruntime"]
                 for r in df.loc[df["source"] == "harness", "run"].unique()}
    blocks = [("harness", "Harness protocol: static and dynamic from the same run"),
              ("version protocol", "Version protocol: median of three isolated sessions")]
    lines = []
    for key, title in blocks:
        sub = df[df["source"].str.startswith(key)]
        if lines:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        last_model, last_setting = None, None
        n_rows = sub.groupby(["model", "px"]).size()      # rows sharing a setting's cells
        m_rows = sub.groupby("model").size()               # rows sharing a model's name
        for _, r in sub.iterrows():
            setting = (r["model"], r["px"])
            first = setting != last_setting
            k = int(n_rows[setting])
            # the input size and FP32 latencies belong to all rows of the setting, the model name
            # to all rows of the model (YOLOv8n: three settings, six rows): set each at the
            # vertical middle of the rows it spans, not on the first row alone
            # a model's block is otherwise only marked by its centred name, which leaves a
            # multi-setting model (YOLOv8n) looking as if its first setting belonged above it
            if r["model"] != last_model and last_model is not None:
                lines.append("\\addlinespace[2pt]")
            # the name sits on the group's FIRST row, not centred over it: two readers read a
            # centred name as belonging to the row beside it (YOLOv8n has three settings)
            name = r["model"] if r["model"] != last_model else ""
            px = _mid(f"{int(r['px'])}", k) if first else ""
            fp = f"{_mid(_ms(r['fp1_ms']), k)} & {_mid(_ms(r['fp4_ms']), k)}" if first else "&"
            quality = _minus(f"{r['drop']:+.2f}")
            lines.append(f"{name} & {px} & {r['method']} & {fp} & {_speed(r['sp1'])} & "
                         f"{_speed(r['sp4'])} & {quality} & {r['size'] * 100:.0f} \\\\")
            last_model, last_setting = r["model"], setting
    if len(releases) != 1:
        raise SystemExit(f"Table II mixes ONNX Runtime releases {sorted(releases)}")
    rel = releases.pop()
    return ("\\begin{table}[tb]\n\\centering\n"
            f"\\caption{{What INT8 buys, per setting, on ONNX Runtime {rel} at batch 1. FP32 is the "
            "median latency of the FP32 sibling in ms; speedup is FP32 latency over INT8 latency "
            "at one (1T) and four (4T) threads; quality is the change in points of the metric in "
            "Table~\\ref{tab1}; size is the INT8 file as a percentage of the FP32 file. The two "
            "blocks come from the two protocols of Section~\\ref{secIIIC}.}\\label{tab2}\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4.5pt}\n"
            "\\begin{tabular}{@{}lrlrrrrrr@{}}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{FP32 (ms)} & \\multicolumn{2}{c}{Speedup} & Quality & "
            "Size \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Model & Input & Method & 1T & 4T & 1T & 4T & (pt) & (\\%) \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def table_versions() -> str:
    """Table III: MobileViT-XXS absolute latencies behind the Fig. 4 ratios, every release."""
    d = pd.read_csv(BISECT)
    d = d[d["family"] == "mobilevit"]
    med = d.groupby(["ort", "variant", "threads"])["p50_ms"].median()
    vers = sorted(d["ort"].unique(), key=lambda v: tuple(int(x) for x in v.split(".")))
    n_sess = d.groupby(["ort", "variant", "threads"]).size().unique()
    variants = ("p00_fp32", "p00_int8", "p00_int8_dyn_mm")
    lines = []
    for v in vers:
        if v == "1.24.4":
            lines.append("\\midrule")
        cells = [f"{med[(v, var, t)]:.3f}" for t in (1, 4) for var in variants]
        sp = [f"{med[(v, 'p00_fp32', t)] / med[(v, 'p00_int8', t)]:.3f}" for t in (1, 4)]
        lines.append(f"{v} & " + " & ".join(cells) + " & " + " & ".join(sp) + " \\\\")
    if len(n_sess) != 1:
        raise SystemExit(f"Table III: unequal session counts per cell {sorted(n_sess)}")
    sess = {2: "two", 3: "three", 4: "four", 5: "five"}.get(int(n_sess[0]), str(n_sess[0]))
    return ("\\begin{table}[tb]\n\\centering\n"
            "\\caption{Absolute latencies behind the MobileViT-XXS ratios of Fig.~\\ref{fig2}: "
            f"median of all {sess} sessions in ms under the version protocol, the same three "
            "files on every release, and the static speedup (FP32 over static) they give. The "
            "rule marks the 1.24.4 step.}\\label{tab3}\n"
            "\\footnotesize\\setlength{\\tabcolsep}{4.5pt}\n"
            "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{1 thread (ms)} & \\multicolumn{3}{c}{4 threads (ms)} & "
            "\\multicolumn{2}{c}{Static speedup} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-9}\n"
            "Release & FP32 & static & dynamic & FP32 & static & dynamic & 1T & 4T \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


SCOPE = ROOT / "results" / "paper" / "ort_version_scope.csv"
FLOOR_PCT = 7.0     # the latency floor of Sec. III-C: Table IV stars only steps beyond it
# Table IV columns: (family in the CSV, FP32 variant, INT8 variant, model, input px, method).
# The input sizes are the files' identities (the family names encode them), not measured values.
SCOPE_COLS = [
    ("mv64",    "p00_fp32",     "p00_int8",        "MobileViT-XXS", 64,  "st."),
    ("mv256",   "p00_fp32",     "p00_int8",        "MobileViT-XXS", 256, "st."),
    ("mv256",   "p00_fp32",     "p00_int8_dyn_mm", "MobileViT-XXS", 256, "dyn."),
    ("deit224", "p00_fp32",     "p00_int8",        "DeiT-tiny",     224, "st."),
    ("deit224", "p00_fp32",     "p00_int8_dyn_mm", "DeiT-tiny",     224, "dyn."),
    ("y256",    "256_p00_fp32", "256_p00_int8",    "YOLOv8n",       256, "st."),
    ("r50",     "p00_fp32",     "p00_int8",        "ResNet-50",     224, "st."),
    ("mnv2",    "p00_fp32",     "p00_int8",        "MobileNetV2",   32,  "st."),
]


def table_scope() -> str:
    """Table IV: the scope run of Sec. V-D - four-thread INT8 speedups on four releases for the
    hybrid at 64 and 256 px, DeiT-tiny at 224 px and the convolutional static files, and the INT8
    file's own four-thread latency change at 1.24.4. Ungated medians, as the text reports."""
    d = pd.read_csv(SCOPE)
    d = d[d["threads"] == 4]
    med = d.groupby(["ort", "family", "variant"])["p50_ms"].median()
    vers = sorted(d["ort"].unique(), key=lambda v: tuple(int(x) for x in v.split(".")))
    n_sess = d.groupby(["ort", "family", "variant"]).size().unique()
    if len(n_sess) != 1:
        raise SystemExit(f"Table IV: unequal session counts per cell {sorted(n_sess)}")
    if vers[:2] != ["1.23.2", "1.24.4"]:
        raise SystemExit(f"Table IV: the step row needs 1.23.2 and 1.24.4 first, got {vers}")
    lines = []
    for v in vers:
        cells = [f"{med[(v, f, fp)] / med[(v, f, q)]:.3f}" for f, fp, q, *_ in SCOPE_COLS]
        lines.append(f"{v} & " + " & ".join(cells) + " \\\\")
    # a step cell is starred when it is beyond the 7% floor of Sec. III-C AND the INT8 file's
    # 1.23.2 and 1.24.4 session ranges do not overlap (computed from the sessions, never typed in):
    # a disjoint step below the floor (DeiT-tiny dynamic, +3.8%) is not interpreted, so it is not
    # starred. \rlap keeps the numbers right-aligned.
    sess_ms = d.groupby(["ort", "family", "variant"])["p50_ms"]
    lo, hi = sess_ms.min(), sess_ms.max()

    def _disjoint(f: str, q: str) -> bool:
        a, b = ("1.23.2", f, q), ("1.24.4", f, q)
        return bool(hi[a] < lo[b] or hi[b] < lo[a])

    def _step(f: str, q: str) -> str:
        pct = (med[("1.24.4", f, q)] / med[("1.23.2", f, q)] - 1) * 100
        star = _disjoint(f, q) and abs(pct) > FLOOR_PCT
        return _minus(f"{pct:+.1f}") + ("\\rlap{*}" if star else "")

    step = [_step(f, q) for f, _, q, *_ in SCOPE_COLS]
    lines += ["\\midrule", "Step (\\%) & " + " & ".join(step) + " \\\\"]
    # header: models spanning their columns, then input and method
    groups, last = [], None
    for _, _, _, model, _, _ in SCOPE_COLS:
        if model == last:
            groups[-1][1] += 1
        else:
            groups.append([model, 1])
        last = model
    head1, rules, col = [], [], 2
    for model, n in groups:
        head1.append(f"\\multicolumn{{{n}}}{{c}}{{{model}}}" if n > 1 else model)
        if n > 1:
            rules.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    head2 = " & ".join(f"{px} {m}" for *_, px, m in SCOPE_COLS)
    sess = {2: "two", 3: "three", 4: "four"}.get(int(n_sess[0]), str(n_sess[0]))
    return ("\\begin{table}[tb]\n\\centering\n"
            "\\caption{Scope of the version effect: INT8 speedup over the FP32 sibling at four "
            f"threads (median of {sess} sessions), version protocol, a run separate "
            "from Fig.~\\ref{fig2}. Columns: input (px) and method (st.: static, dyn.: dynamic); "
            "DeiT-tiny 224 and MobileViT-XXS 256\\,px: latency only. Step: change of the INT8 "
            "file's own four-thread latency from 1.23.2 to 1.24.4; *: beyond 7\\%, sessions "
            "disjoint."
            "}\\label{tab4}\n"
            "\\footnotesize\\setlength{\\tabcolsep}{3.2pt}\n"
            f"\\begin{{tabular}}{{@{{}}l{'r' * len(SCOPE_COLS)}@{{}}}}\n\\toprule\n"
            " & " + " & ".join(head1) + " \\\\\n" + "".join(rules) + "\n"
            "Release & " + head2 + " \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


TABLES = {1: table_settings, 2: table_results, 3: table_versions, 4: table_scope}

# Named for help with the measurement runs, which is a statement about who produced data: it says
# help, not authorship, and it says nothing about which runs, because that is not recorded here.
ACKNOWLEDGEMENT = "The authors thank Wei-Yao Chang for assistance with the experiments."
DISCLOSURE = ("The authors have no competing interests to declare that are relevant to the "
              "content of this article.")
# TAAI has no supplementary upload, so the paper itself says where the evidence can be had. No
# public repository URL has been supplied: replace "from the author on request" with it if one is.
AVAILABILITY = ("The quantized ONNX files, measurement scripts, raw measurement CSVs (including "
                "the 396-row sweep) and dated decision log are available from the authors on "
                "request. All URLs in the references were last accessed 18-19 Sep 2026.")
# The paper's central claim is about byte-identical files, so it prints the hash prefixes of the
# sweep files of Sec. V-A and the 256 px static file of Sec. V-D, computed here from the files
# themselves (never typed in): a reader who obtains the files can check they are the ones measured. Groups: (lead text, [(label, path), ...]).
HASHED = [
    ("of the MobileViT-XXS files of Section~\\ref{secVA}:", [
        ("FP32", "results/classification_mobilevit/models/p00_fp32.onnx"),
        ("static", "results/classification_mobilevit/models/p00_int8.onnx"),
        ("dynamic", "results/classification_mobilevit/models/p00_int8_dyn_mm.onnx")]),
    ("DeiT-tiny", [
        ("FP32", "results/classification_deit/models/p00_fp32.onnx"),
        ("static", "results/classification_deit/models/p00_int8.onnx"),
        ("dynamic", "results/classification_deit/models/p00_int8_dyn_mm.onnx")]),
    ("MobileViT-XXS 256\\,px (Section~\\ref{secVD})", [
        ("static", "results/mobilevit_r256/models/p00_int8.onnx")]),
    # The scope run's YOLOv8n 256 px pair (results/detection_dyn/models/256_p00_fp32.onnx and
    # 256_p00_int8.onnx) would belong here too, but its one line pushes the paper to 16 pages
    # (tried 2026-09-19); add it if a line of the reference pages is freed.
]


def availability() -> str:
    import hashlib
    groups = []
    for lead, files in HASHED:
        cells = []
        for label, rel in files:
            digest = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            cells.append(f"{label} {digest[:12]}")
        groups.append(f"{lead} " + ", ".join(cells))
    return AVAILABILITY + " SHA-256 prefixes " + "; ".join(groups) + "."


# ------------------------------------------------------------------------------------ figures
def figure_captions() -> dict[int, tuple[str, str]]:
    df = _fig_data()
    fig = df[df["in_fig"]]
    yolo_px = int(fig.loc[fig["model"] == "YOLOv8n", "px"].iloc[0])
    worst = fig.loc[fig["drop"].idxmin()]
    ws = _summary(worst["run"])
    worst_data = _dataset(ws["adapter"], ws["config"])
    def fac_cell(path, fmt, act):
        f = pd.read_csv(path)
        f = f[(f["quant_format"] == fmt) & (f["activation_type"] == act)
              & (f["reduce_range"] == False)]  # noqa: E712 - the plotted cells
        return float(f["speedup_t4"].iloc[0])

    def sig3(v):
        return f"{v:#.3g}"

    # Fig. 3 is its own run: name its QDQ U8S8 cells beside the Table II cells they differ from
    t2 = df[df["method"] == "static"].set_index("model")["sp4"]
    run_u8 = (fac_cell(FACTORIAL, "qdq", "uint8"), fac_cell(FACTORIAL_MVIT, "qdq", "uint8"))
    tab_u8 = (t2["MobileNetV2"], t2["MobileViT-XXS"])
    return {
        1: ("fig3_default_config.pdf",
            "The quantizer's default is slower than FP32. Static INT8 speedup over FP32 at four "
            "threads for two formats (QDQ, QOperator) and two activation/weight types (U8S8, "
            "S8S8), harness protocol, ONNX Runtime 1.30.0, a separate run from "
            "Table~\\ref{tab2}, whose QDQ U8S8 cells "
            f"({sig3(tab_u8[0])}$\\times$ MobileNetV2, {sig3(tab_u8[1])}$\\times$ MobileViT-XXS) "
            f"differ from this run's ({sig3(run_u8[0])}$\\times$, {sig3(run_u8[1])}$\\times$) by "
            "run and, for MobileViT-XXS, by protocol. Models and settings are as in "
            "Table~\\ref{tab1}."),
        3: ("fig5_localization.pdf",
            "Where the four-thread step lives. Added latency of the same work at 1.24.4 against "
            "1.23.2: the full static file (black), then the isolation probe of "
            "Section~\\ref{secVC} (solid) and its controls (hatched). The probe holds only the "
            "static graph's 8-bit Transposes, wrapped in quantize/dequantize and taken net of a "
            "Transpose-free control; the probe's registered class-gated medians "
            "(Section~\\ref{secIIIC})."),
        2: ("fig4_version_sweep.pdf",
            "INT8 speedup (FP32 median over INT8 median of the same release, three sessions each) "
            "of one set of byte-identical files across eleven ONNX Runtime releases, version "
            "protocol. Untested releases omitted. Whiskers: range over the three INT8 "
            "sessions. Shaded: releases from 1.24.4 on. MobileViT-XXS static at four threads "
            "falls twice (Table~\\ref{tab3}); at one thread only the second fall appears."),
    }


def figure_block(n: int, figures: dict) -> str:
    pdf, cap = figures[n]
    return ("\\begin{figure}[tb]\n\\centering\n"
            f"\\includegraphics[width=0.93\\textwidth]{{{pdf}}}\n"
            f"\\caption{{{cap}}}\\label{{fig{n}}}\n"
            "\\end{figure}\n")


PREAMBLE = r"""% TAAI 2026, International Track. Springer LNCS A4, 10-15 pages including references.
% Generated by scripts/paper/build_tex.py from paper/manuscript.md - edit those, not this file.
% For Overleaf: open the Springer LNCS template, replace its main.tex with this file and upload
% the four figure PDFs next to it.
\documentclass[runningheads,a4paper]{llncs}
% article's a4paper sets \paperwidth; tell the PDF driver too (pdfTeX/XeTeX and LuaTeX)
\ifdefined\pdfpagewidth \pdfpagewidth=\paperwidth \pdfpageheight=\paperheight \fi
\ifdefined\pagewidth \pagewidth=\paperwidth \pageheight=\paperheight \fi

\usepackage[T1]{fontenc}
\usepackage{newtxtext}
\usepackage[varvw]{newtxmath}
\usepackage{graphicx}
\usepackage{array}
\hyphenation{QuantizeLinear DequantizeLinear ConvInteger MobileViT}
\brokenpenalty=10000
\usepackage{booktabs}
\usepackage{xurl}
% xurl lets a URL break after any character, so a long URL need not leave its line underfull.
% URL break points may stretch, so bibliography lines with a long URL are not left underfull;
% a tenth of a mu keeps that from visibly spacing out the URL's letters (half a mu set
% "https : //github . com" apart in ref. 13 at 200 dpi)
\Urlmuskip=0mu plus 0.1mu
% ...but not inside a run of digits: a break after a digit printed ref. 39's pull request 27691 as
% "pull/2769" with "1" on the next line. Removing \do0...\do9 from xurl's break list keeps the
% breaks after "/", "." and letters (url breaks after a listed character, so a number stays whole).
\makeatletter
\def\Url@nodigits#1\do0\do1\do2\do3\do4\do5\do6\do7\do8\do9#2\@nil{\def\UrlBreaks{#1#2}}
\expandafter\Url@nodigits\UrlBreaks\@nil
\makeatother

% a third line-breaking pass instead of lines that stick out past the margin
\setlength{\emergencystretch}{1em}

\begin{document}

\title{@@TITLE@@}
\titlerunning{@@RUNNING@@}

% Review is single-blind, so author details are required in the manuscript, not withheld.
\author{Yuan-Zhen Yang \and Yi-Hsiang Chao}
\authorrunning{Y.-Z. Yang and Y.-H. Chao}
\institute{Department of Industrial and Systems Engineering, Chung Yuan Christian University,\\
Taoyuan, Taiwan\\
\email{willyang2002@gmail.com}}

\maketitle

\begin{abstract}
@@ABSTRACT@@

\keywords{@@KEYWORDS@@}
\end{abstract}

"""


def main() -> int:
    if not SRC.exists():
        print(f"{SRC} not found", file=sys.stderr)
        return 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    figures = figure_captions()

    lines = [ln.rstrip() for ln in SRC.read_text(encoding="utf-8").split("\n")]
    raw_title = lines[0].strip()
    title = tex_escape(raw_title)
    running = tex_escape(raw_title.split(":")[0].strip())   # the full title overflows the head
    abstract = keywords = ""
    out, refs, labels = [], [], set()
    sec, sub, n_para = None, "", 0
    placed: list[tuple[str, int]] = []
    in_refs = False

    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("ABSTRACT:"):
            abstract = body_text(s[len("ABSTRACT:"):].strip())
            continue
        if s.startswith("INDEX TERMS:"):
            kw = s[len("INDEX TERMS:"):].strip().rstrip(".")
            keywords = " \\and ".join(tex_escape(k.strip()) for k in kw.split(","))
            continue
        if s.startswith("## "):
            head = s[3:].strip()
            m = re.match(r"([IVX]+)\.\s*(.+)", head)
            if (m.group(2) if m else head).lower().startswith("reference"):
                in_refs = True
                continue
            sec, sub, n_para = (m.group(1) if m else None), "", 0
            lab = f"sec{sec}" if sec else ""
            labels.add(lab)
            out.append(f"\n\\section{{{tex_escape(m.group(2) if m else head)}}}"
                       + (f"\\label{{{lab}}}" if lab else "") + "\n\n")
            continue
        if s.startswith("### "):
            # The manuscript letters its subsections IEEE-style; llncs numbers them itself.
            ms = re.match(r"([A-G])\.\s*(.+)", s[4:].strip())
            sub, text = (ms.group(1), ms.group(2)) if ms else ("", s[4:].strip())
            n_para = 0
            lab = f"sec{sec}{sub}" if sec and sub else ""
            labels.add(lab)
            out.append(f"\n\\subsection{{{tex_escape(text)}}}"
                       + (f"\\label{{{lab}}}" if lab else "") + "\n\n")
            continue
        if s.startswith("#### "):
            # run-in heading (llncs level 4, italic); the paragraph that follows continues its line
            out.append(f"\\paragraph{{{tex_escape(s[5:].strip())}}}\n")
            continue
        if in_refs:
            m = re.match(r"\[(\d+)\]\s*(.+)", s)
            if m:
                refs.append((int(m.group(1)), ref_text(m.group(2))))
            continue

        para = body_text(s)
        loose = [k for k in LOOSEN if s.startswith(k)]
        if loose:
            LOOSENED.update(loose)
            para = "\\looseness=-1 " + para
        out.append(para + "\n\n")
        n_para += 1
        for p_sec, p_sub, at, kind, num in PLAN:
            if (kind, num) in placed or (p_sec, p_sub, at) != (sec, sub, n_para):
                continue
            placed.append((kind, num))
            labels.add(f"{kind}{num}")
            out.append((figure_block(num, figures) if kind == "fig" else TABLES[num]()) + "\n")

    # ---- refuse to write a paper with a dangling reference or a missing float
    errors = []
    for kind in ("fig", "tab"):
        seq = [n for k, n in placed if k == kind]
        if seq != sorted(seq):
            errors.append(f"{kind} floats placed out of numbering order: {seq}")
    for _, _, _, k, n in PLAN:
        if (k, n) not in placed:
            errors.append(f"{k} {n} never placed (its PLAN anchor matches no paragraph)")
    ref_nums = {n for n, _ in refs}
    if CITED - ref_nums:
        errors.append(f"cited but not in the reference list: {sorted(CITED - ref_nums)}")
    if ref_nums - CITED:
        errors.append(f"in the reference list but never cited: {sorted(ref_nums - CITED)}")
    for lab in sorted(REFERRED - labels):
        errors.append(f"referenced but never defined: {lab}")
    for k in sorted(set(LOOSEN) - LOOSENED):
        errors.append(f"LOOSEN entry matches no paragraph: {k!r}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1

    # Springer's proceedings require a Disclosure of Interests before the references, as a 9-point
    # bold run-in heading: llncs v2.25 (paper/tex/llncs.cls) sets it in its credits environment.
    # The data and code availability statement is a second item of the same block.
    biblio = ["\n\\begin{credits}\n\\subsubsection{\\ackname}\n" + ACKNOWLEDGEMENT + "\n\n"
              "\\subsubsection{\\discintname}\n" + DISCLOSURE + "\n\n"
              "\\subsubsection{Data and Code Availability.}\n" + availability() + "\n\\end{credits}\n",
              # (a ragged-right bibliography stops URL letter-spacing but costs lines: tried
              # 2026-09-19, it pushed the paper to 16 pages; see \Urlmuskip above instead)
              "\\begin{thebibliography}{99}"]
    for num, text in sorted(refs):
        biblio.append(f"\\bibitem{{ref{num}}}\n{text}\n")
    biblio.append("\\end{thebibliography}\n\\end{document}\n")

    # Not %-formatting: the preamble is LaTeX, where '%' starts a comment, and Python would read
    # those comments as conversion specifiers. Sentinels survive however many comments accumulate.
    head = (PREAMBLE.replace("@@TITLE@@", title)
                    .replace("@@RUNNING@@", running)
                    .replace("@@ABSTRACT@@", abstract)
                    .replace("@@KEYWORDS@@", keywords))
    tex = head + "".join(out) + "\n".join(biblio)
    (OUTDIR / "main.tex").write_text(tex, encoding="utf-8")

    for pdf, _ in figures.values():
        src = FIGS / pdf
        if not src.exists():
            print(f"  ERROR: {src} missing - run make_figures.py", file=sys.stderr)
            return 1
        shutil.copy2(src, OUTDIR / pdf)
    stale = sorted(p.name for p in OUTDIR.glob("fig*.pdf")
                   if p.name not in {f for f, _ in figures.values()})

    print(f"wrote {OUTDIR / 'main.tex'}  ({len(tex.split())} words of TeX)")
    print(f"  figures copied: {sorted(f for f, _ in figures.values())}")
    if stale:
        print(f"  note: stale figure files in {OUTDIR.name}/ not used by main.tex: {stale}")
    print(f"  references: {len(refs)} (all cited)   floats placed: {len(placed)}/{len(PLAN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
