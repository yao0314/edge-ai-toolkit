"""Render the pitch deck to a single 16:9 PDF, with no browser and nothing to install.

The deck also lives as an Artifact, which exports PPTX and PDF from its own Share menu. This exists
so the file can simply be handed over: matplotlib is already a dependency of this project, and
python-pptx is not.

Slides are data here, not markup - a few block kinds (hero cards, stat cards, a table, callouts)
composed on a 1920x1080 canvas in the same palette and type scale as the Artifact, so the two read
as the same deck.

  python scripts/sec_deck_pdf.py --out submission/Valiant_Joy_pitch.pdf
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

W, H = 1920, 1080
MARGIN = 128
DARK, DARK2, LIGHT = "#10212E", "#19404F", "#F7F8F5"
BLUE, ORANGE = "#0072B2", "#D55E00"
INK, BODY, FAINT = "#15242E", "#44535E", "#8A959D"
ON_DARK, ON_DARK2, ON_DARK3 = "#F7F8F5", "#AFC9D8", "#7FB9D9"
CARD, CARD_EDGE = "#FFFFFF", "#DFE4E6"
FIGS = "submission/slides/bare"   # the report figures, titles stripped: the slide carries them


def new_slide(bg):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0, 0), W, H, color=bg, zorder=0))
    return fig, ax


def image(ax, path, x, y, w, h):
    """Place a figure inside x,y,w,h, keeping its aspect ratio and centring it in the box."""
    im = plt.imread(path)
    ih, iw = im.shape[0], im.shape[1]
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    ax.imshow(im, extent=(x + (w - dw) / 2, x + (w + dw) / 2,
                          y + (h + dh) / 2, y + (h - dh) / 2), zorder=3, aspect="auto")


def box(ax, x, y, w, h, fc, ec=None, r=20, z=2):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor=ec or fc, linewidth=1.4 if ec else 0, zorder=z))


def lines_of(s, wrap):
    """The lines `text` will draw. Split out so a card can be sized to its contents before it is
    painted — fixed card heights are why text used to run out of the bottom of a box."""
    out = []
    for para in s.split("\n"):
        if wrap and len(para) > wrap:
            cur = ""
            for word in para.split(" "):
                if len(cur) + len(word) + 1 > wrap and cur:
                    out.append(cur)
                    cur = word
                else:
                    cur = f"{cur} {word}".strip()
            out.append(cur)
        else:
            out.append(para)
    return out


def block_h(s, size, wrap=None, lh=1.35):
    return len(lines_of(s, wrap)) * size * lh


def text(ax, x, y, s, size, color, weight="normal", va="top", ha="left", wrap=None, lh=1.35, z=4):
    """Returns the y just past the block. `wrap` is a character budget per line."""
    lines = lines_of(s, wrap)
    step = size * lh
    for i, ln in enumerate(lines):
        ax.text(x, y + i * step + (size * 0.82 if va == "top" else 0), ln, fontsize=size * 0.75,
                color=color, fontweight=weight, ha=ha, va="baseline" if va == "top" else va,
                zorder=z, family="DejaVu Sans")
    return y + len(lines) * step


def eyebrow(ax, s, color):
    text(ax, MARGIN, MARGIN, s.upper(), 25, color, "bold")


def heading(ax, s, color, wrap=36, y=MARGIN + 54):
    return text(ax, MARGIN, y, s, 72, color, "bold", wrap=wrap, lh=1.1)


def footer(ax, n, color=FAINT):
    text(ax, MARGIN, H - MARGIN + 6, f"SEC 2026 / Software Optimization   ·   Valiant Joy   ·   {n}",
         24, color)


def stat_cards(ax, cards, y, h=None, gap=28, fc=CARD, ec=CARD_EDGE, num_c=BLUE, lab_c=INK,
               sub_c=BODY):
    n = len(cards)
    w = (W - 2 * MARGIN - gap * (n - 1)) / n
    # one height for the row, taken from whichever card needs the most, so the cards align AND
    # nothing overflows whatever the longest label happens to be
    need = max(40 + 82 * 1.0 + 10 + block_h(lab, 30, 24, 1.2)
               + (6 + block_h(sub, 27, 26, 1.25) if sub else 0) + 40
               for _, lab, sub in cards)
    h = max(h or 0, need)
    for i, (num, label, sub) in enumerate(cards):
        x = MARGIN + i * (w + gap)
        f = fc[i] if isinstance(fc, list) else fc
        e = (ec[i] if isinstance(ec, list) else ec)
        box(ax, x, y, w, h, f, e)
        nc = num_c[i] if isinstance(num_c, list) else num_c
        lc = lab_c[i] if isinstance(lab_c, list) else lab_c
        sc = sub_c[i] if isinstance(sub_c, list) else sub_c
        yy = text(ax, x + 44, y + 40, num, 82, nc, "bold", lh=1.0)
        yy = text(ax, x + 44, yy + 10, label, 30, lc, "bold", wrap=24, lh=1.2)
        if sub:
            text(ax, x + 44, yy + 6, sub, 27, sc, wrap=26, lh=1.25)


def table(ax, head, rows, y, widths, size=31, head_c=INK, body_c=BODY, rule="#DFE4E6",
          hl_row=None, hl_bg=None, cell_colors=None, total_w=None):
    total_w = total_w if total_w is not None else W - 2 * MARGIN
    xs, x = [], MARGIN
    for fr in widths:
        xs.append(x)
        x += total_w * fr
    rh = size * 2.1
    for j, cell in enumerate(head):
        text(ax, xs[j] if j == 0 else xs[j] + total_w * widths[j] - 12,
             y, cell, size, head_c, "bold", ha="left" if j == 0 else "right")
    y += rh
    ax.plot([MARGIN, MARGIN + total_w], [y - 10, y - 10], color=rule, linewidth=1.4, zorder=3)
    for i, row in enumerate(rows):
        if hl_row is not None and i == hl_row and hl_bg:
            box(ax, MARGIN - 20, y - 12, total_w + 40, rh, hl_bg, r=10, z=1)
        for j, cell in enumerate(row):
            c = body_c
            if cell_colors and (i, j) in cell_colors:
                c = cell_colors[(i, j)]
            text(ax, xs[j] if j == 0 else xs[j] + total_w * widths[j] - 12,
                 y, cell, size, c, "bold" if (cell_colors and (i, j) in cell_colors) else "normal",
                 ha="left" if j == 0 else "right")
        y += rh
    return y


# --------------------------------------------------------------------------------- the eight slides
def s1_cover():
    fig, ax = new_slide(DARK)
    ax.add_patch(mpatches.Rectangle((0, 0), W, H, color=DARK, zorder=0))
    eyebrow(ax, "SEC 2026  ·  Minimalist Edge AI Challenge", ON_DARK3)
    text(ax, MARGIN, 270, "Prune deeper.\nRecover smarter.", 126, ON_DARK, "bold", lh=1.02)
    text(ax, MARGIN, 540, "Structured pruning, teacher-guided recovery, static INT8 —\n"
                          "on a MobileNetV2 that was already distilled.", 36, ON_DARK2, lh=1.35)
    stat_cards(ax, [("8.94×", "faster on the edge CPU", ""),
                    ("5.58×", "fewer parameters", ""),
                    ("98.0%", "of accuracy kept", "")],
               y=690, h=230, fc=[BLUE, DARK2, DARK2], ec=[BLUE, DARK2, DARK2],
               num_c=["#FFFFFF", ON_DARK, ON_DARK], lab_c=["#D7ECF7", ON_DARK2, ON_DARK2])
    text(ax, MARGIN, H - MARGIN + 6, "VALIANT JOY   |   MobileNetV2 / EuroSAT   |   Software Optimization Track", 24, "#6E8C9E")
    return fig


def s2_challenge():
    fig, ax = new_slide(LIGHT)
    eyebrow(ax, "The challenge", BLUE)
    heading(ax, "The easy compression had\nalready been taken", INK)
    cards = [("What we were given", "A MobileNetV2 already distilled from a ResNet-50 and trained on "
                                    "EuroSAT. 2.24 M parameters, 99.09% top-1."),
             ("Why that is hard", "An already-compact model has no obvious redundancy left. Every "
                                  "channel removed costs accuracy almost immediately."),
             ("What is scored", "Accuracy, compression, resource use and energy — as ratios against "
                                "that baseline, over 10 land-use classes.")]
    w = (W - 2 * MARGIN - 2 * 28) / 3
    ch = max(40 + block_h(h, 32, 22) + 12 + block_h(b, 28, 24, 1.4) + 40 for h, b in cards)
    for i, (h, b) in enumerate(cards):
        x = MARGIN + i * (w + 28)
        box(ax, x, 400, w, ch, CARD, CARD_EDGE)
        yy = text(ax, x + 44, 440, h, 32, BLUE, "bold", wrap=22)
        text(ax, x + 44, yy + 12, b, 28, BODY, wrap=24, lh=1.4)
    text(ax, MARGIN, 760, "So the question is not how much can you cut. It is how far can you cut\n"
                          "before one class stops working — and that limit is not where anyone expects.",
         31, BODY, lh=1.45)
    footer(ax, "02")
    return fig


def s3_pipeline():
    fig, ax = new_slide(LIGHT)
    eyebrow(ax, "The pipeline", BLUE)
    heading(ax, "Four steps, one of them unusual", INK)
    steps = [("01", "Provided student", "MobileNetV2 FP32\n2.24 M params", False),
             ("02", "Structured pruning", "60% of channels\nremoved, not masked", False),
             ("03", "Teacher recovery", "BN recalibration +\n8 KD epochs from the\nprovided ResNet-50", True),
             ("04", "Static INT8", "QDQ, per-channel,\npercentile calibration", False)]
    gap, aw = 22, 46
    w = (W - 2 * MARGIN - 3 * (gap * 2 + aw)) / 4
    ch = max(32 + block_h(n, 28) + 10 + block_h(t, 29, 15) + 12 + block_h(b, 24, 19) + 36
             for n, t, b, _ in steps)
    x = MARGIN
    for num, title, body, hot in steps:
        box(ax, x, 400, w, ch, BLUE if hot else CARD, None if hot else CARD_EDGE)
        yy = text(ax, x + 34, 432, num, 28, "#8FC9E8" if hot else "#A9B5BD", "bold")
        yy = text(ax, x + 34, yy + 10, title, 29, "#FFFFFF" if hot else INK, "bold", wrap=15)
        text(ax, x + 34, yy + 12, body, 24, "#D7ECF7" if hot else BODY, wrap=19, lh=1.35)
        x += w
        if num != "04":
            ax.annotate("", xy=(x + gap + aw, 550), xytext=(x + gap, 550),
                        arrowprops=dict(arrowstyle="-|>", color="#A9B5BD", linewidth=2.4), zorder=4)
            x += gap * 2 + aw
    text(ax, MARGIN, 770, "Exported artifact:  ONNX opset 13  ·  static QDQ  ·  input pinned to "
                          "1×3×224×224  ·  0.60 MB", 30, BODY)
    footer(ax, "02")
    return fig


def s4_thesis():
    fig, ax = new_slide(DARK)
    eyebrow(ax, "The finding", ON_DARK3)
    heading(ax, "The collapse at 60% is a recovery\nlimit, not a capacity limit", ON_DARK)
    box(ax, MARGIN, 400, 1130, 430, "#FFFFFF", r=14)
    image(ax, f"{FIGS}/2_recovery.png", MARGIN + 16, 414, 1098, 402)
    text(ax, 1360, 420, "+7.8", 96, "#4FB3E8", "bold", lh=1.0)
    text(ax, 1360, 545, "points of the weakest class\nat 60% — for four lines\nof config. And nothing\n"
                        "at all at 50%.", 28, ON_DARK2, lh=1.35)
    text(ax, MARGIN, 836, "Weakest of the ten per-class recalls. Same cut, same architecture, same "
                          "quantization. The benefit peaks in a window: too shallow and hard labels "
                          "are already enough; too deep and there is no capacity left to teach.",
         27, ON_DARK2, wrap=98, lh=1.4)
    footer(ax, "03", "#6E8C9E")
    return fig


def s4b_curve():
    fig, ax = new_slide(LIGHT)
    eyebrow(ax, "How we chose the point", BLUE)
    heading(ax, "We swept it. The mean could not\npick a stopping point", INK)
    box(ax, MARGIN, 390, 1150, 480, "#FFFFFF", CARD_EDGE, r=14)
    image(ax, f"{FIGS}/5_curve.png", MARGIN + 18, 404, 1114, 452)
    bullets = [("Top panel", "every point from 0% to 20% is inside the ±0.36-point measurement "
                             "band. The bump at 10% is noise."),
               ("Bottom panel", "the weakest class is the statistic that moves, and the one that "
                                "located the cliff between 50% and 60%."),
               ("Both recoveries", "plotted together — the blue line is what let us ship the deeper "
                                   "point.")]
    y = 400
    for h, b in bullets:
        yy = text(ax, 1360, y, h, 29, BLUE, "bold")
        y = text(ax, 1360, yy + 6, b, 25, BODY, wrap=26, lh=1.35) + 26
    footer(ax, "04")
    return fig


def s5_accuracy():
    fig, ax = new_slide(LIGHT)
    eyebrow(ax, "Accuracy and compression", BLUE)
    heading(ax, "82% of the parameters gone,\n98% of the accuracy kept", INK)
    stat_cards(ax, [("98.04%", "of baseline accuracy", "99.09% → 97.15% top-1"),
                    ("5.58×", "fewer parameters", "2.24 M → 0.40 M"),
                    ("14.5×", "smaller on disk", "8.77 MB → 0.60 MB")], y=400)
    box(ax, MARGIN, 636, 1096, 286, "#FFFFFF", CARD_EDGE, r=14)
    image(ax, f"{FIGS}/4_perclass.png", MARGIN + 16, 648, 1064, 262)
    box(ax, 1264, 636, 528, 286, "#FFF4EC", "#F0D5C0", r=14)
    yy = text(ax, 1300, 672, "What it costs", 29, ORANGE, "bold")
    text(ax, 1300, yy + 10, "The weakest of the ten classes falls to 92.6%, below the baseline's own "
                            "94.7%. One class carries most of the loss, and we report it rather than "
                            "averaging it away.", 24, BODY, wrap=29, lh=1.35)
    footer(ax, "05")
    return fig


def s6_resources():
    fig, ax = new_slide(LIGHT)
    eyebrow(ax, "Resource and energy", BLUE)
    heading(ax, "Less compute, less memory, less energy", INK)
    y = table(ax, ["Metric", "Baseline", "Valiant Joy", "Change"],
              [["MACs", "0.326 G", "0.080 G", "4.10×"],
               ["CPU latency, 4 threads", "3.50 ms", "1.12 ms", "3.13×"],
               ["Energy per inference", "0.218 J", "0.071 J", "3.07×"],
               ["Peak RAM, raw process RSS", "78.9 MB", "62.1 MB", "1.27×"]],
              y=430, widths=[0.34, 0.22, 0.22, 0.22], size=32,
              cell_colors={(i, 3): BLUE for i in range(4)})
    # one paragraph, wrapped once: manual breaks AND a wrap budget fight each other, because each
    # hand-made line is then re-wrapped on its own and the block grows into the footer
    text(ax, MARGIN, y + 34, "Same machine, same script, same thread count for baseline and final. "
                             "Energy is CodeCarbon in TDP mode — so we checked it against physics: "
                             "implied power 62.2 W for the baseline and 63.4 W for ours, and a 3.07× "
                             "energy ratio against a 3.13× latency ratio. Peak RAM moves least, "
                             "because 56 MB of the process is the Python and ONNX Runtime floor, "
                             "not the model.", 26, BODY, wrap=96, lh=1.42)
    footer(ax, "06")
    return fig


def s7_edge():
    fig, ax = new_slide(DARK)
    eyebrow(ax, "Hardware-aware evaluation", ON_DARK3)
    heading(ax, "Our laptop said 3.1×.\nThe edge CPU said 8.9×.", ON_DARK)
    box(ax, MARGIN, 400, 1010, 400, "#FFFFFF", r=14)
    image(ax, f"{FIGS}/3_device.png", MARGIN + 16, 414, 978, 372)
    cards = [("Latency there", "27.97 ms → 3.13 ms"),
             ("Peak memory there", "32.33 MB → 19.45 MB   (1.66×)"),
             ("Profiled on", "Dragonwing RB3 Gen 2\nQCS6490, ONNX on CPU")]
    y = 408
    for lab, val in cards:
        box(ax, 1240, y, 552, 120, DARK2)
        yy = text(ax, 1278, y + 24, lab.upper(), 24, "#8FA9B8", "bold")
        text(ax, 1278, yy + 6, val, 27, ON_DARK, "bold", wrap=28, lh=1.3)
        y += 140
    text(ax, MARGIN, 826, "Same two ONNX files, profiled on both. A large out-of-order core with a deep "
                          "cache absorbs the cost of a fat model and hides most of what compression buys. "
                          "Choose a prune ratio on your laptop and you are reading a gradient three times "
                          "too shallow.", 28, ON_DARK2, wrap=96, lh=1.42)
    footer(ax, "07", "#6E8C9E")
    return fig


def s9_numbers():
    """Every measured quantity, absolute, with no framing. The slide a sceptic asks for."""
    fig, ax = new_slide(LIGHT)
    eyebrow(ax, "The measurements", BLUE)
    heading(ax, "Everything we measured, in full", INK)
    rows = [["Top-1 accuracy", "99.09 %", "97.15 %", "98.04 % retained"],
            ["Weakest of 10 class recalls", "94.66 %", "92.61 %", "−2.05 pp"],
            ["Parameters", "2,236,682", "400,954", "5.58×"],
            ["ONNX file size", "8.51 MB", "0.60 MB", "14.07×"],
            ["MACs", "326.2 M", "79.5 M", "4.10×"],
            ["Latency, 1 thread", "9.804 ms", "2.679 ms", "3.66×"],
            ["Latency, 4 threads", "3.501 ms", "1.119 ms", "3.13×"],
            ["CPU time per inference", "12.308 ms", "4.646 ms", "2.65×"],
            ["Peak RAM, raw process RSS", "78.85 MB", "62.14 MB", "1.27×"],
            ["Energy per inference", "0.2177 J", "0.0709 J", "3.07×"],
            ["Implied power (energy ÷ latency)", "62.2 W", "63.4 W", "1.02× — the check"],
            ["AI Hub latency, QCS6490", "27.969 ms", "3.127 ms", "8.94×"],
            ["AI Hub peak memory", "32.33 MB", "19.45 MB", "1.66×"]]
    y = table(ax, ["Measurement", "Baseline", "Valiant Joy", "Ratio"], rows, y=344,
              widths=[0.36, 0.20, 0.20, 0.24], size=20,
              cell_colors={(i, 3): BLUE for i in range(13) if i not in (1, 10)})
    # no caption: thirteen rows already reach the footer, and the footer carries the machine
    footer(ax, "09   ·   i7-13620H, ORT 1.30.0, one script for both")
    return fig


def s9_scorecard():
    """The closer: the rubric's own six rows, each with the figure that answers it.

    A jury scores against a list. Handing them that list with our number already beside each line
    is worth more at the end than another sentence about method.
    """
    fig, ax = new_slide(DARK)
    eyebrow(ax, "The scorecard", ON_DARK3)
    heading(ax, "Every number the rubric asks for", ON_DARK)
    rows = [["Model Accuracy & Reliability", "25", "retained vs. baseline,\nall 10 classes",
             "98.04% retained\nworst class 92.6%"],
            ["Model Compression", "20", "parameter count\nand file size",
             "5.58× parameters\n14.51× file"],
            ["Resource Efficiency", "20", "peak RAM and latency,\ncross-checked on AI Hub",
             "8.94× faster on QCS6490\n1.66× less peak memory"],
            ["Energy Parsimony", "20", "Joules per inference,\nratio to own baseline",
             "3.07× less\n0.2177 J → 0.0709 J"],
            ["Documentation & Reproducibility", "10", "rebuild from the provided\ncheckpoint",
             "four commands,\nevery figure one CSV"],
            ["Engineering Judgment", "5", "quality of the\ntradeoff analysis",
             "four documented\nnegative results"]]
    widths = [0.345, 0.075, 0.275, 0.305]
    xs, x = [], MARGIN
    for fr in widths:
        xs.append(x)
        x += (W - 2 * MARGIN) * fr
    y = 352
    for j, head in enumerate(["Criterion", "Pts", "What is measured", "Valiant Joy"]):
        text(ax, xs[j], y, head, 24, ON_DARK3, "bold")
    y += 42
    ax.plot([MARGIN, W - MARGIN], [y - 12, y - 12], color="#2C4A5B", linewidth=1.4, zorder=3)
    for i, row in enumerate(rows):
        h = max(block_h(c, 23, None, 1.3) for c in row) + 24
        if i < 4:
            box(ax, MARGIN - 18, y - 14, W - 2 * MARGIN + 36, h, DARK2, r=10, z=1)
        for j, cell in enumerate(row):
            c = ON_DARK if j == 3 else ("#4FB3E8" if j == 1 else "#C9D8E1")
            text(ax, xs[j], y, cell, 23, c, "bold" if j in (0, 3) else "normal", lh=1.3)
        y += h + 8
    footer(ax, "10   ·   85 of the 100 points are measured by the judges' own rig", "#6E8C9E")
    return fig


def s8_takeaways():
    fig, ax = new_slide(LIGHT)
    eyebrow(ax, "Engineering decisions", BLUE)
    heading(ax, "What we learned, including what failed", INK)
    cards = [("01", "Measure the weakest class", "The mean loses one point where the worst class loses "
                                                 "eight. The cliff is invisible in the average.", False),
             ("02", "Spend on recovery, not the cut", "Teacher-guided recovery moved the deepest usable "
                                                      "prune a full step down the ladder.", False),
             ("03", "Profile on the target", "The same two files read 3.13× on our laptop and 8.94× on "
                                             "the edge CPU.", False),
             ("04", "What did not work", "Per-layer allocation: better at matched size, but it stalls at "
                                         "46.6% — cheap parameters are a bounded pool.", True)]
    w = (W - 2 * MARGIN - 3 * 24) / 4
    ch = max(34 + block_h(n, 28) + 10 + block_h(t, 30, 17) + 12 + block_h(b, 25, 22) + 36
             for n, t, b, _ in cards)
    for i, (num, title, body, warn) in enumerate(cards):
        x = MARGIN + i * (w + 24)
        box(ax, x, 392, w, ch, "#FDF0E7" if warn else CARD, "#F0D5C0" if warn else CARD_EDGE)
        yy = text(ax, x + 36, 426, num, 28, ORANGE if warn else BLUE, "bold")
        yy = text(ax, x + 36, yy + 10, title, 30, INK, "bold", wrap=17)
        text(ax, x + 36, yy + 12, body, 25, BODY, wrap=22, lh=1.35)
    box(ax, MARGIN, 782, W - 2 * MARGIN, 140, DARK)
    text(ax, MARGIN + 44, 812, "98.04% accuracy retained · 5.58× fewer parameters · 14.5× smaller · "
                               "8.94× faster", 26, ON_DARK, "bold")
    text(ax, MARGIN + 44, 864, "Every figure traces to one CSV from one run. Both ladders, the failed "
                               "experiments and the conservative alternative ship with the code.",
         24, "#8FA9B8", wrap=104)
    footer(ax, "09")
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="submission/Valiant_Joy_pitch.pdf")
    a = ap.parse_args()
    # s2_challenge is dropped: to a jury of edge-computing experts it restates what they already
    # know, and the slot is worth more as the full measurement table.
    builders = [s1_cover, s3_pipeline, s4_thesis, s4b_curve, s5_accuracy,
                s6_resources, s7_edge, s8_takeaways, s9_numbers, s9_scorecard]
    with PdfPages(a.out) as pdf:
        for b in builders:
            fig = b()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    print(f"wrote {a.out}  ({len(builders)} slides, 16:9)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
