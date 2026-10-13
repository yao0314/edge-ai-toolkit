"""Copy the camera-ready additions into the public artifact repository, with its publication rules.

Private tool: it is NOT copied into the artifact. It applies the same rules as the first release
(2026-09-21): machine paths become placeholders, internal process notes are reworded, no number,
date, rule, command or verdict is changed, and every file whose published copy differs from the
original is listed with both SHA-256 values in docs/changed_files.csv.

    .venv/Scripts/python.exe scripts/paper/publish_artifact.py            # dry run: report only
    .venv/Scripts/python.exe scripts/paper/publish_artifact.py --write    # copy into the artifact
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

SRC = Path(r"C:/Users/User/Desktop/SEC/edge-ai-toolkit")
DST = Path(r"C:/Users/User/Desktop/SEC/taai2026-int8-cpu-artifact")

TREES = [("results/expA", "data/results/expA"), ("results/expA2", "data/results/expA2"),
         ("results/expB", "data/results/expB"), ("results/expW", "data/results/expW"),
         ("results/qc", "data/results/qc"), ("results/w_probe", "data/models/w_probe")]
PAPER = ["a_design_registered.md", "a2_design_registered.md", "b_design_registered.md",
         "w_design_registered.md", "A_RESULTS.md", "B_RESULTS.md", "W_RESULTS.md"]
SCRIPTS = ["a_check.py", "a_pool.py", "a2_exploratory_run.py", "a_analyse.py", "transpose_risk.py",
           "b_run.py", "b_analyse.py", "b_replicate_analyse.py", "w_probe.py", "qc_charts.py",
           "ort_version_bisect.py"]
TEXT = {".json", ".csv", ".md", ".txt", ".log", ".py", ".svg", ".out", ".jsonl", ".yaml", ".yml"}
ALLOWED_BINARY = {".png", ".onnx"}

SEP = r"[\\/]+"
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
PATHS = [
    (re.compile(r"[Cc]:" + SEP + "Users" + SEP + "User" + SEP + "AppData" + SEP + "Local" + SEP + "Temp" + SEP
                + "claude" + SEP + "C--Users-User-Desktop-SEC" + SEP + "(?:" + UUID + "|<session>)" + SEP
                + "scratchpad"), "<SCRATCH>"),
    (re.compile(r"[Cc]:" + SEP + "Users" + SEP + "User" + SEP + "Desktop" + SEP + "SEC" + SEP + "edge-ai-toolkit"),
     "<PROJECT_ROOT>"),
    (re.compile(r"/c/Users/User/Desktop/SEC/edge-ai-toolkit"), "<PROJECT_ROOT>"),
    (re.compile(r"[Dd]:" + SEP + "sec_models"), "<MODEL_STORE>"),
    (re.compile(r"[Dd]:" + SEP + "sec_envs"), "<VENVS>"),
    (re.compile(r"[Dd]:" + SEP + "sec_work"), "<SCRATCH>"),
]
RESIDUAL = re.compile(r"(?i)[a-z]:" + SEP + r"(users|sec_)|appdata|" + UUID + r"|C--Users-User|desktop" + SEP + "sec")

# Reworded internal process notes: (published path, old, new, expected count). "the author" is the
# paper's first author, who approved candidate lists and made the decisions these notes record.
AUTHOR_FILES = ["data/results/expA/candidates.json", "data/results/expA2/pool.json",
                "code/scripts/paper/a_pool.py", "data/results/paper/a_design_registered.md",
                "data/results/paper/a2_design_registered.md", "data/results/expB/v6_decision.md",
                "data/results/expB/replication_note.md"]
WORDING = [
    ("code/scripts/paper/b_run.py", "some of them may be the user's own.", "some of them may not be ours.", 1),
    ("data/results/expA/candidates.json", "from two independent collector agents.", "from two independent collectors.", 1),
    ("data/results/expA2/candidates_round2.json", "by the A2 collector agent under", "by the A2 collector under", 1),
    ("data/results/expA2/candidates_round2.json", '"A2 collector agent (round 2', '"A2 collector (round 2', 1),
    ("data/results/expA2/candidates_round3.json", "by the A2 top-up agent under", "by the A2 top-up collector under", 1),
    ("data/results/expA2/candidates_round3.json", '"A2 top-up agent (round 3', '"A2 top-up collector (round 3', 1),
    ("data/results/paper/a2_design_registered.md", "an agent that has not worked on it", "a reviewer that has not worked on it", 1),
    ("data/results/paper/b_design_registered.md", "by a separate critic agent;", "by a separate critic;", 1),
    ("data/results/paper/b_design_registered.md", "`CLAUDE.md` already records the same fact in the project's own words:",
     "The project's working rules already record the same fact in their own words:", 1),
    ("data/results/paper/b_design_registered.md", '(`CLAUDE.md`, "Measurement discipline")',
     '(the project\'s working rules, "Measurement discipline")', 1),
    ("data/results/paper/b_design_registered.md", "at the cost of one extra agent and about twenty minutes.",
     "at the cost of one extra classifier and about twenty minutes.", 1),
    ("data/results/paper/b_design_registered.md", "**Step 4 — an independent agent classifies.** A separate agent, which has not seen",
     "**Step 4 — an independent classifier.** A separate classifier, which has not seen", 1),
    ("data/results/paper/b_design_registered.md", "The blinded agent's verdict is final.", "The blinded classifier's verdict is final.", 1),
    ("data/results/paper/b_design_registered.md", "prompt. A disagreement is recorded", "instruction. A disagreement is recorded", 1),
    ("data/results/paper/b_design_registered.md", '(`CLAUDE.md`, "Working rules")', "(the project's working rules)", 1),
    ("data/results/paper/b_design_registered.md", '(`CLAUDE.md`, "Quantization defaults")',
     '(the project\'s working rules, "Quantization defaults")', 1),
    ("data/results/paper/w_design_registered.md", "Designed by an agent that is not the author of the claim under test",
     "Designed by an independent designer that is not the author of the claim under test", 1),
    ("data/results/paper/w_design_registered.md", '(project rule: *"Any check meant to refute a claim is designed by a separate agent,',
     "(project rule: any check meant to refute a claim is designed separately,", 1),
    ("data/results/paper/w_design_registered.md", "never by the claim's author.\"*, `CLAUDE.md`).", "never by the claim's author).", 1),
    ("data/results/paper/w_design_registered.md", "This is the check `MEMORY.md` demands",
     "This is the check the project's recorded lessons demand", 1),
    ("data/results/paper/w_design_registered.md", "agent. `LoadPercentage` and `other_load_pct` are barred by name (`MEMORY.md`: `LoadPercentage` read",
     "run. `LoadPercentage` and `other_load_pct` are barred by name (the project's recorded lessons: `LoadPercentage` read", 1),
    ("data/results/paper/w_design_registered.md", '(`CLAUDE.md` working rules; `MEMORY.md` "Workflow agents leave orphan',
     "(the project's working rules and recorded lessons", 1),
    ("data/results/paper/w_design_registered.md", '  processes").', "  on leftover processes).", 1),
    ("data/results/expB/v6_decision.md", "five resident `claude` processes", "five resident background processes", 1),
    ("data/results/expB/v6_decision.md", "only after the desktop sessions are closed", "only after those background processes are closed", 1),
    ("data/results/paper/a2_design_registered.md", "user approves the candidate list**. This registration approves rules, not URLs.",
     "author approves the candidate list**. This registration approves rules, not URLs.", 1),
    ("data/results/paper/a2_design_registered.md", "| user approves; downloads, admission and Arm 2 builds complete",
     "| author approves; downloads, admission and Arm 2 builds complete", 1),
    ("data/results/paper/a_design_registered.md", "| user approves the list; download and admission complete",
     "| author approves the list; download and admission complete", 1),
    ("data/results/expA2/pool.json", '"user_approval": {', '"author_approval": {', 1),
    ("code/scripts/paper/a_pool.py", '"user_approval": approval,', '"author_approval": approval,', 1),
    ("data/results/expA2/candidates_round2.json", "was inferred or completed from memory.", "was entered without a source of record.", 1),
]
LOG_BULLET_AFTER = '- Section 8: "Experiment E2 -- full-model counterfactual"'
LOG_BULLET = ('- Camera-ready additions (appended 2026-09-24, same rules as above; the added data files also use '
              '`<MODEL_STORE>` and `<VENVS>`, see the README): "Experiment A2 EXPLORATORY scan"; "Experiment B - V6 '
              'fired on batch 1, registration NOT amended"; "Experiment A2 — result of the third-party scan"; '
              '"Independent review of the camera-ready rewrite — corrections". The registrations and results of '
              'experiments A/A2, B and W are in `data/results/paper/`; B\'s V6 rule was later withdrawn as an '
              'operational rule (`data/results/expB/v6_decision.md`).')
LOG_WORDING = [("REJECTED by\nthe user:", "REJECTED by\nthe author:", 1)]
USER_RULES = [(re.compile(r"\bThe user's\b"), "The author's"), (re.compile(r"\bthe user's\b"), "the author's"),
              (re.compile(r"\bThe user\b"), "The author"), (re.compile(r"\bthe user\b"), "the author"),
              (re.compile(r" in chat\b"), "")]
FORBIDDEN = re.compile(r"(?i)claude|anthropic|\bagents?\b|\bin chat\b|conversation|memory\.md|\bthe user\b|"
                       r"user's|subagent|\bllm\b|\bprompt\b|session transcript|willyang|gmail")
LOG_START = "### Experiment A2 EXPLORATORY scan"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def plan() -> list[tuple[Path, str]]:
    items = []
    for s, d in TREES:
        for f in sorted((SRC / s).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                items.append((f, d + "/" + f.relative_to(SRC / s).as_posix()))
    items += [(SRC / "results/paper" / n, "data/results/paper/" + n) for n in PAPER]
    items += sorted((f, "data/results/paper/" + f.name) for f in (SRC / "results/paper").glob("a2_thirdparty_exploratory*.csv"))
    items += [(SRC / "scripts/paper" / n, "code/scripts/paper/" + n) for n in SCRIPTS]
    return items


def scrub_paths(text: str, counts: dict) -> str:
    for rx, rep in PATHS:
        text, n = rx.subn(rep, text)
        counts[rep] = counts.get(rep, 0) + n
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    problems, out, counts, used = [], {}, {}, set()
    for src, rel in plan():
        ext = src.suffix.lower()
        raw = src.read_bytes()
        if ext not in TEXT:
            if ext not in ALLOWED_BINARY:
                problems.append("unexpected binary type: " + rel)
            if ext == ".onnx" and not rel.startswith("data/models/w_probe/"):
                problems.append("ONNX outside w_probe: " + rel)
            if RESIDUAL.search(raw.decode("latin-1")):
                problems.append("machine path inside binary: " + rel)
            out[rel] = (raw, raw)
            continue
        text = raw.decode("utf-8")
        new = scrub_paths(text, counts)
        for i, (p, old, rep, k) in enumerate(WORDING):
            if p == rel:
                n = new.count(old)
                if n != k:
                    problems.append("%s: rule %r matched %d times, expected %d" % (rel, old[:50], n, k))
                new = new.replace(old, rep)
                used.add(i)
        if rel in AUTHOR_FILES:
            for rx, rep in USER_RULES:
                new = rx.sub(rep, new)
        if ext == ".json":
            try:
                json.loads(new)
            except ValueError as e:
                problems.append("JSON broken after scrub: %s (%s)" % (rel, e))
        out[rel] = (raw, new.encode("utf-8"))
    for i, (p, old, _, _) in enumerate(WORDING):
        if i not in used:
            problems.append("rule never applied (file not in plan): %s %r" % (p, old[:40]))

    # the research log: new FINDINGS entries, same rules. Line endings are kept as the published file
    # has them (CRLF), so that git shows only the added lines.
    findings = (SRC / "results/paper/FINDINGS.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    k = findings.find(LOG_START)
    if k < 0 or findings.count(LOG_START) != 1:
        problems.append("log start marker not found exactly once")
    tail = scrub_paths(findings[k:], counts)
    for old, rep, n in LOG_WORDING:
        if tail.count(old) != n:
            problems.append("log rule %r matched %d times" % (old, tail.count(old)))
        tail = tail.replace(old, rep)
    for rx, rep in USER_RULES:
        tail = rx.sub(rep, tail)
    log_path = DST / "docs/research_log.md"
    log_raw = log_path.read_bytes()
    eol = "\r\n" if b"\r\n" in log_raw else "\n"
    log_old = log_raw.decode("utf-8").replace("\r\n", "\n")
    if LOG_START in log_old or LOG_BULLET in log_old:
        problems.append("research_log already contains the new entries; restore it from HEAD first")
    lines = log_old.split("\n")
    at = [i for i, s in enumerate(lines) if s.startswith(LOG_BULLET_AFTER)]
    if len(at) != 1:
        problems.append("header anchor for the new bullet found %d times" % len(at))
    else:
        lines.insert(at[0] + 1, LOG_BULLET)
    log_new = ("\n".join(lines).rstrip("\n") + "\n\n" + tail.rstrip("\n") + "\n").replace("\n", eol)

    # the log's first 3486 lines were published on 2026-09-21 and are not re-examined here
    for rel, (_, b) in list(out.items()) + [("docs/research_log.md (appended part)", (None, tail.encode("utf-8")))]:
        if Path(rel).suffix.lower() in ALLOWED_BINARY:
            continue
        t = b.decode("utf-8")
        for m in RESIDUAL.finditer(t):
            problems.append("residual path in %s: %r" % (rel, t[max(0, m.start() - 30):m.end() + 30]))
        for m in FORBIDDEN.finditer(t):
            ctx = t[max(0, m.start() - 50):m.end() + 50].replace("\n", " ")
            problems.append("forbidden term in %s: %r" % (rel, ctx))

    changed = [(rel, sha(r), sha(b), r, b) for rel, (r, b) in out.items() if r != b]
    print("files planned: %d, changed by the rules: %d, placeholders: %s" % (len(out), len(changed), counts))
    big = [(rel, len(b)) for rel, (_, b) in out.items() if len(b) > 20 * 2**20]
    if big:
        problems.append("files over 20 MB: %s" % big)
    print("total bytes: %.1f MB" % (sum(len(b) for _, b in out.values()) / 2**20))
    if problems:
        print("\nPROBLEMS (%d):" % len(problems))
        for p in problems[:80]:
            print("  " + p)
        return 1
    if not a.write:
        print("dry run clean; nothing written")
        return 0

    for rel, (_, b) in out.items():
        p = DST / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    log_path.write_bytes(log_new.encode("utf-8"))
    rows = []
    for rel, s0, s1, r, b in changed:
        a_lines = r.decode("utf-8").splitlines()
        b_lines = b.decode("utf-8").splitlines()
        lines = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False).get_opcodes():
            if tag != "equal":
                lines += list(range(j1 + 1, j2 + 1))
        kind = "paths" if all(any(ph in b_lines[n - 1] for ph in ("<PROJECT_ROOT>", "<SCRATCH>", "<MODEL_STORE>", "<VENVS>"))
                              for n in lines) else "paths and wording"
        span = ";".join(str(n) for n in lines)
        rows.append({"published_path": rel, "original_sha256": s0, "published_sha256": s1,
                     "changed_lines": span, "change": kind})
    with open(DST / "docs/changed_files.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["published_path"]))
    print("written: %d files, research_log.md appended, docs/changed_files.csv (%d rows)" % (len(out), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
