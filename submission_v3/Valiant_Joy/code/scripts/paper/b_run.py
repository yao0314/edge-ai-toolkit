"""Experiment B runner: release drift of unchanged ONNX files across eleven ONNX Runtime releases.

Binding protocol: results/paper/b_design_registered.md (registered 2026-09-23). This script
implements the RUN side of it - the grids, the batch plan, stage 0, the anchors, the quiet-machine
checks, the exclusions, the seeds and the output files. It computes no verdict: MOVED / UNRESOLVED /
STILL, the denominators, the bootstraps and the sealed map are the analysis step's job and are not
in here.

    .venv/Scripts/python.exe scripts/paper/b_run.py --stage 0        # pre-flight, no timing
    .venv/Scripts/python.exe scripts/paper/b_run.py --stage map      # the fifteen batches
    .venv/Scripts/python.exe scripts/paper/b_run.py --stage post     # post-run hashes + merge
    .venv/Scripts/python.exe scripts/paper/b_run.py --stage all      # 0 -> map -> post

Engineering rules this file obeys, each because something went wrong without it:

  output to a FILE, never a pipe   Every harness invocation writes stdout+stderr to its own log
                                   file. A full pipe with nobody draining it hung an earlier run for
                                   11 minutes with no CPU use and no output.
  one batch per invocation         A batch that fails costs its own cells, is recorded, and is never
                                   retried silently (the a2_exploratory_run.py pattern).
  resumable                        A batch whose CSV already exists with the expected row count is
                                   not re-run. The harness writes its CSV only at the end of an
                                   invocation, so a batch is all-or-nothing and no cell can span two
                                   invocations (protocol V7).
  timestamped log                  Every stage appends to results/expB/b_run.log as it goes.
  no orphans                       Every subprocess is waited on; a timeout kills the whole process
                                   tree (taskkill /F /T), not just the harness.

Harness: scripts/paper/ort_version_bisect.py, unedited. It carries three opt-in flags that this run
needs - --drop-duplicate-envs, --status-rows, --exclude - and all three are used. Three further
harness changes the protocol lists in 11.2 were NOT added and are worked around here; see
DEVIATIONS below.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
HARNESS = "scripts/paper/ort_version_bisect.py"
DEFAULT_ENVS = Path("D:/sec_envs")
POOL = Path("D:/sec_models")
TW = timezone(timedelta(hours=8))

# --------------------------------------------------------------------------------------------
# DEVIATIONS from results/paper/b_design_registered.md 11.2, forced by the harness as it stands.
# Each is recorded in results/expB/manifest.json as well, so the artefact carries them.
DEVIATIONS = [
    ("11.2(a) t_utc / order_index columns",
     "The harness writes no per-session timestamp and no shuffle position. Per-BATCH start/end UTC "
     "are recorded in results/expB/batch_plan.json and added as columns to the merged CSVs, so the "
     "5.2 arm-centre figure and V4/V5 time regressions are computable at batch resolution, not "
     "session resolution."),
    ("11.2(c) --anchor injection every 40 sessions",
     "The harness cannot inject anchor cells inside a batch. Both anchors are measured in their own "
     "invocations at every batch BOUNDARY instead (16 boundaries x 2 anchors x --anchor-reps "
     "sessions), on the single 1.30.0 interpreter, into results/expB/b_drift_anchor.csv. V4's "
     "first-third / last-third medians are computable; the time resolution is one boundary "
     "(20-70 min) instead of ~3 min."),
    ("11.2(e) --not-applicable removal before the shuffle",
     "--exclude takes family:variant pairs and cannot express a per-RELEASE exclusion, so the "
     "NOT-APPLICABLE (file, release) cells of 3.0 cannot be removed from a batch without splitting "
     "the batch by release group - which would confound release with invocation and destroy the "
     "within-batch shuffle that 5.2 requires. They are therefore left in the grid, where they fail "
     "at session construction in ~1 s and are recorded as status=FAILED rows carrying the runtime's "
     "own error text. No timing is produced for them; the analysis drops them by name from every "
     "denominator using results/expB/load_probe.json. --exclude IS used for any (family, variant) "
     "that stage 0 shows fails on EVERY release, which is release-independent and so expressible."),
    ("11.1 artefact paths",
     "The protocol puts the CSVs under results/paper/. This run round is allowed to write only "
     "scripts/paper/b_run.py, results/expB/* and D:/sec_work/*, so every artefact keeps its "
     "registered FILE NAME under results/expB/ instead. Nothing under results/paper/ is written."),
    ("7 release-notes retrieval",
     "Stage 0's notes retrieval is implemented (--stage notes) but is OFF by default: it is the one "
     "step that needs the network, and downloading is not done unprompted."),
]

# --------------------------------------------------------------------------------------------
# 2.1 the ladder, and the binary fingerprints that identify it (V1). First 12 hex of the SHA-256 of
# each shipped binary in Lib/site-packages/onnxruntime/capi/.
RELEASES = ["1.16.3", "1.20.1", "1.21.1", "1.22.1", "1.23.2", "1.24.4",
            "1.25.1", "1.26.0", "1.27.0", "1.28.0", "1.30.0"]

EXPECTED_BINARIES = {
    "1.16.3": {"onnxruntime_pybind11_state.pyd": "11545c116906",
               "onnxruntime_providers_shared.dll": "fd04e715c9f4"},
    "1.20.1": {"onnxruntime_pybind11_state.pyd": "83f4061d6bc0",
               "onnxruntime.dll": "2f92a249542d", "onnxruntime_providers_shared.dll": "ff0a090202ab"},
    "1.21.1": {"onnxruntime_pybind11_state.pyd": "e6ca84c822e2",
               "onnxruntime.dll": "a93e75cd85a2", "onnxruntime_providers_shared.dll": "c80df011efa5"},
    "1.22.1": {"onnxruntime_pybind11_state.pyd": "cf57f10c1d27",
               "onnxruntime.dll": "b7d046825412", "onnxruntime_providers_shared.dll": "121329b5f87d"},
    "1.23.2": {"onnxruntime_pybind11_state.pyd": "967a5a759076",
               "onnxruntime.dll": "93457c4f1790", "onnxruntime_providers_shared.dll": "056da4c9eec7"},
    "1.24.4": {"onnxruntime_pybind11_state.pyd": "fbd869b4b1ff",
               "onnxruntime.dll": "ef720fc44a4e", "onnxruntime_providers_shared.dll": "f123ce892247"},
    "1.25.1": {"onnxruntime_pybind11_state.pyd": "fae1c8be8b8d",
               "onnxruntime.dll": "bb656592046f", "onnxruntime_providers_shared.dll": "c56c986e157e"},
    "1.26.0": {"onnxruntime_pybind11_state.pyd": "28a78fe15545",
               "onnxruntime.dll": "b7dfcb4dea88", "onnxruntime_providers_shared.dll": "669fa0fbb553"},
    "1.27.0": {"onnxruntime_pybind11_state.pyd": "071e3be75111",
               "onnxruntime.dll": "7b9ef55ecaa6", "onnxruntime_providers_shared.dll": "4cc699327866"},
    "1.28.0": {"onnxruntime_pybind11_state.pyd": "8fb176cc2c96",
               "onnxruntime.dll": "14e186627f10", "onnxruntime_providers_shared.dll": "d9a39bc6b34e"},
    "1.30.0": {"onnxruntime_pybind11_state.pyd": "06b712319c15",
               "onnxruntime.dll": "44741724166a", "onnxruntime_providers_shared.dll": "fabb99add45a"},
}

# 3.0 the expected NOT-APPLICABLE set. Stage 0 DECIDES it; this is only what the protocol expects.
EXPECTED_NOT_APPLICABLE = (
    [(f"{fam}/{var}", rel)
     for fam, var in [("r50", "p00_int8_dyn_conv"), ("mnv2", "p00_int8_dyn_conv"),
                      ("y256", "256_p00_int8_dyn_conv")]
     for rel in ["1.16.3", "1.20.1", "1.21.1", "1.22.1", "1.23.2"]]
    + [("c3_299x299/b05", "1.16.3")]
)

# --------------------------------------------------------------------------------------------
# 5.2 the batch plan: fifteen batches, fixed order, arms interleaved, heaviest first within an arm.
# Base seed 71; batch k in this order uses seed 71 + k. est_min is the protocol's own registered
# estimate; stage 0 recomputes it from the load probe and the pilot p50 table.
BATCHES = [
    dict(n=1,  arm="CONTRAST", name="r50",         d="results/stress_resnet50_dyn/models", shape="3,224,224",
         variants=["p00_fp32", "p00_int8", "p00_int8_dyn_conv"], seed=72, est_min=54.7),
    dict(n=2,  arm="BREADTH",  name="c3_224x224",  d="D:/sec_models/c3_224x224", shape="3,224,224",
         variants=["a01", "a02", "a03", "b02", "b03", "b15", "b16", "c01", "c05"], seed=73, est_min=69.4),
    dict(n=3,  arm="CONTRAST", name="y256",        d="results/detection_dyn/models", shape="3,256,256",
         variants=["256_p00_fp32", "256_p00_int8", "256_p00_int8_dyn_conv"], seed=74, est_min=23.6),
    dict(n=4,  arm="BREADTH",  name="c224_224x3",  d="D:/sec_models/c224_224x3", shape="224,224,3",
         variants=["a04", "c02"], seed=75, est_min=14.7),
    dict(n=5,  arm="CONTRAST", name="mv64",        d="results/mobilevit_r64/models", shape="3,64,64",
         variants=["p00_fp32", "p00_int8", "p00_int8_dyn_mm"], seed=76, est_min=22.1),
    dict(n=6,  arm="BREADTH",  name="c3_320x320",  d="D:/sec_models/c3_320x320", shape="3,320,320",
         variants=["b01", "b18"], seed=77, est_min=14.7),
    dict(n=7,  arm="CONTRAST", name="mv256",       d="results/mobilevit_r256/models", shape="3,256,256",
         variants=["p00_fp32", "p00_int8", "p00_int8_dyn_mm"], seed=78, est_min=22.1),
    dict(n=8,  arm="BREADTH",  name="c3_299x299",  d="D:/sec_models/c3_299x299", shape="3,299,299",
         variants=["b05"], seed=79, est_min=9.1),
    dict(n=9,  arm="CONTRAST", name="deit224",     d="results/deit_224/models", shape="3,224,224",
         variants=["p00_fp32", "p00_int8", "p00_int8_dyn_mm"], seed=80, est_min=22.1),
    dict(n=10, arm="BREADTH",  name="c1_64x64",    d="D:/sec_models/c1_64x64", shape="1,64,64",
         variants=["a05"], seed=81, est_min=7.4),
    dict(n=11, arm="CONTRAST", name="mnv2",        d="results/classification_dyn/models", shape="3,32,32",
         variants=["p00_fp32", "p00_int8", "p00_int8_dyn_conv"], seed=82, est_min=18.8),
    dict(n=12, arm="BREADTH",  name="c3_112x112",  d="D:/sec_models/c3_112x112", shape="3,112,112",
         variants=["c04"], seed=83, est_min=7.4),
    dict(n=13, arm="APPENDIX", name="ae",          d="results/anomaly/models", shape="1,140",
         variants=["p00_fp32", "p00_int8"], seed=84, est_min=14.7),
    dict(n=14, arm="BREADTH",  name="c3_240x320",  d="D:/sec_models/c3_240x320", shape="3,240,320",
         variants=["a06"], seed=85, est_min=7.4),
    dict(n=15, arm="BREADTH",  name="c56_56x3",    d="D:/sec_models/c56_56x3", shape="56,56,3",
         variants=["c03"], seed=86, est_min=7.4),
]

# 5.6 the two anchors, both on the single chosen 1.30.0 interpreter, 4 threads, own CSV.
ANCHORS = [
    dict(tag="A-small", name="mnv2", d="results/classification_dyn/models", shape="3,32,32",
         variant="p00_int8", threads=4, pilot_ms=0.4396),
    dict(tag="A-heavy", name="r50", d="results/stress_resnet50_dyn/models", shape="3,224,224",
         variant="p00_fp32", threads=4, pilot_ms=24.497),
]
ANCHOR_RELEASE = "1.30.0"

SEEDS = dict(base=71, batches={b["name"]: b["seed"] for b in BATCHES},
             replication=91, decoy_sampling=97, cluster_bootstrap=101)

REPEATS = 5                 # 5.1, never reduced
THREADS = "1,4"             # 3.1
SESSION_TIMEOUT = 900       # the harness's own per-session child timeout
PREFLIGHT_BUDGET_H = 6.0    # 8.3 cut trigger
HARD_STOP_H = 8.0           # 8.3 hard stop, evaluated at batch boundaries
QUIET_MAX_PCT = 15.0        # 5.3
QUIET_CHECKS = 10           # 5.3, one per minute
GATE = 0.95                 # reported, never gated (5.4, V6)
V6_BATCH_LIMIT = 0.45       # V6: below-gate share above this re-runs the batch once
FLOOR_REL = 0.07            # 4.2, quoted for the record only; this script computes no verdict


# ============================================================================================
# small helpers

def ts() -> str:
    return f"{datetime.now(TW):%Y-%m-%d %H:%M:%S}"


class Ctx:
    """Everything that depends on where the output goes, so smoke mode cannot touch the real run."""

    def __init__(self, out_root: Path, work: Path, envs: Path):
        self.out = out_root
        self.work = work
        self.envs = envs
        self.batches = out_root / "batches"
        self.logs = out_root / "logs"
        self.log_file = out_root / "b_run.log"
        for d in (self.out, self.work, self.batches, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"{ts()} {msg}"
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_json(p: Path):
    """utf-8-sig, so a plan or pair list hand-edited in a Windows editor (which adds a BOM) still
    loads. Everything this script writes is plain utf-8."""
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kill_tree(pid: int) -> None:
    """Kill the harness AND its ORT child. p.kill() alone leaves the child measuring into a dead
    pipe, which is exactly the orphan this project has been bitten by."""
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=60)
    except Exception:
        pass


def run_to_file(cmd, out_path: Path, timeout: int, cwd: Path = ROOT) -> int:
    """stdout+stderr to a FILE. Never a pipe: a pipe nobody drains is what hung an earlier run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {ts()} {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd))
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(p.pid)
            try:
                p.wait(60)
            except Exception:
                pass
            return -9


def csv_rows(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ============================================================================================
# 5.3 quiet-machine precondition, in the VA4 form. PerfOS_Processor _Total only: LoadPercentage and
# other_load_pct are on record as false alarms and must not appear.

PS_LOAD = (
    "$s=@();1..{n}|%{{$c=Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor|"
    "?{{$_.Name -eq '_Total'}};$s+=[double]$c.PercentProcessorTime;Start-Sleep -Milliseconds {ms}}};"
    "$s -join ','"
)


def cpu_samples(seconds: float = 5.0, n: int = 5) -> list[float]:
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           PS_LOAD.format(n=n, ms=int(seconds * 1000 / n))]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    vals = []
    for tok in (r.stdout or "").strip().splitlines()[-1].split(","):
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def on_ac_power() -> tuple[bool, str]:
    """Win32_Battery BatteryStatus 2 == on AC. No battery object at all == desktop/AC."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "(Get-CimInstance Win32_Battery | Select-Object -First 1)"
                            ".BatteryStatus"], capture_output=True, text=True, timeout=60)
        s = (r.stdout or "").strip()
        if not s:
            return True, "no battery object"
        return s in {"2", "6", "7", "8", "9"}, f"BatteryStatus={s}"
    except Exception as e:
        return True, f"unreadable ({e})"


def wsl_running() -> tuple[bool, str]:
    """`wsl -l --running` exits non-zero when nothing is running; its output is console-encoded."""
    try:
        r = subprocess.run(["wsl.exe", "-l", "--running"], capture_output=True, timeout=60)
    except Exception as e:
        return False, f"wsl not available ({e})"
    if r.returncode != 0:
        return False, f"rc={r.returncode} (no running distro)"
    txt = ""
    for enc in ("utf-16-le", "cp950", "utf-8"):
        try:
            txt = r.stdout.decode(enc)
            break
        except Exception:
            continue
    lines = [ln.strip() for ln in txt.replace("\x00", "").splitlines() if ln.strip()]
    return len(lines) > 1, f"rc=0, {len(lines)} line(s)"


def quiet_machine(ctx: Ctx, label: str, checks: int = QUIET_CHECKS) -> dict:
    """AC power, no WSL distro, machine-wide _Total <= 15% over 5 s, within `checks` one-minute
    tries. Appended to precondition.json whatever the outcome; a batch whose record is missing is
    invalid (5.4 iii)."""
    ac, ac_detail = on_ac_power()
    wsl, wsl_detail = wsl_running()
    rec = {"label": label, "utc": datetime.now(timezone.utc).isoformat(),
           "ac_power": ac, "ac_detail": ac_detail, "wsl_running": wsl, "wsl_detail": wsl_detail,
           "threshold_pct": QUIET_MAX_PCT, "attempts": []}
    ok = False
    for i in range(checks):
        s = cpu_samples()
        att = {"i": i, "utc": datetime.now(timezone.utc).isoformat(), "n": len(s),
               "mean": round(statistics.fmean(s), 2) if s else None,
               "max": round(max(s), 2) if s else None, "samples": s}
        rec["attempts"].append(att)
        ctx.log(f"    quiet check {i + 1}/{checks}: mean {att['mean']}% max {att['max']}% "
                f"(n={att['n']})")
        if s and statistics.fmean(s) <= QUIET_MAX_PCT:
            ok = True
            break
        if i < checks - 1:
            time.sleep(60)
    rec["passed"] = bool(ok and ac and not wsl)
    rec["reason"] = ("ok" if rec["passed"] else
                     ";".join(x for x in [("" if ac else "not on AC"),
                                          ("WSL distro running" if wsl else ""),
                                          ("" if ok else "CPU above threshold")] if x))
    path = ctx.out / "precondition.json"
    all_recs = load_json(path) if path.exists() else []
    all_recs.append(rec)
    path.write_text(json.dumps(all_recs, indent=1), encoding="utf-8")
    return rec


def stray_pythons() -> int:
    """8.3 thermal/process hygiene: count python processes, log them, kill nothing automatically -
    some of them may be the user's own."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "(Get-Process python -ErrorAction SilentlyContinue).Count"],
                           capture_output=True, text=True, timeout=60)
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return -1


# ============================================================================================
# the grid: the 38 files of stage 0, derived from the batch plan so the two cannot drift apart

def probe_files() -> list[dict]:
    out = []
    for b in BATCHES:
        for v in b["variants"]:
            d = Path(b["d"]) if re.match(r"^[A-Za-z]:", b["d"]) else ROOT / b["d"]
            out.append({"key": f"{b['name']}/{v}", "arm": b["arm"], "family": b["name"],
                        "variant": v, "path": str((d / f"{v}.onnx")).replace("\\", "/")})
    return out


def interpreter_map(envs: Path, drop_duplicates: bool = True) -> dict:
    """Reuse the harness's own interpreters(), so this script and the run agree by construction."""
    sys.path.insert(0, str(ROOT / "scripts" / "paper"))
    import importlib
    mod = importlib.import_module("ort_version_bisect")
    return mod.interpreters(envs, drop_duplicates=drop_duplicates)


def binary_triple(py: Path) -> dict:
    capi = py.parent.parent / "Lib" / "site-packages" / "onnxruntime" / "capi"
    out = {}
    if capi.is_dir():
        for f in sorted(capi.glob("*")):
            if f.is_file() and f.suffix.lower() in {".pyd", ".dll", ".so"}:
                out[f.name] = sha256(f)
    return out


PROBE_CHILD = '''
import json, sys, time
outp = sys.argv[2]
items = json.load(open(sys.argv[1], encoding="utf-8"))
def emit(rec):
    with open(outp, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\\n")
try:
    import numpy, onnxruntime as ort
    emit({"key": "__ENV__", "ort": ort.__version__, "numpy": numpy.__version__,
          "python": sys.version.split()[0]})
except Exception as e:
    emit({"key": "__ENV_FAIL__", "err": str(e)[:400]}); raise SystemExit(2)
for key, path in items:
    t0 = time.perf_counter()
    try:
        so = ort.SessionOptions(); so.log_severity_level = 3
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        n_in = s.get_inputs()[0].name
        emit({"key": key, "loads": True, "construct_s": time.perf_counter() - t0,
              "input": n_in, "err": ""})
        del s
    except Exception as e:
        emit({"key": key, "loads": False, "construct_s": time.perf_counter() - t0,
              "input": "", "err": str(e).replace("\\n", " ")[:400]})
emit({"key": "__DONE__"})
'''


def stage0(ctx: Ctx, args) -> int:
    """V1 hashes, V2 hashes, the V8 load probe, and the 8.3 pre-flight budget. No timing."""
    ctx.log("=" * 92)
    ctx.log("STAGE 0 - pre-flight. No timing measurement is taken in this stage.")
    t0 = time.time()
    q = quiet_machine(ctx, "stage 0", checks=1)   # recorded, non-blocking: no timing happens here
    ctx.log(f"stage-0 machine state: {q['reason']} "
            f"(mean {q['attempts'][0]['mean']}%, AC {q['ac_detail']}, WSL {q['wsl_detail']})")

    # ---- V1: one interpreter per release, resolved by bytes and by reported version
    pys_all = interpreter_map(ctx.envs, drop_duplicates=False)
    pys = interpreter_map(ctx.envs, drop_duplicates=True)
    dropped = sorted(set(pys_all) - set(pys))
    ctx.log(f"V1 interpreters: {len(pys_all)} found under {ctx.envs} (+project), "
            f"{len(pys)} kept, dropped as byte-identical: {dropped or 'none'}")

    interp = {}
    for env, py in pys.items():
        trip = binary_triple(Path(py))
        interp[env] = {"python": str(py).replace("\\", "/"),
                       "binaries": {k: v[:12] for k, v in trip.items()},
                       "binaries_full": trip}

    # ---- V2: pre-run SHA-256 of every model file
    files = probe_files()
    missing = [f for f in files if not Path(f["path"]).exists()]
    for f in files:
        p = Path(f["path"])
        f["sha256"] = sha256(p) if p.exists() else None
        f["bytes"] = p.stat().st_size if p.exists() else None
    ctx.log(f"V2 model files: {len(files)} in the grid, {len(files) - len(missing)} present, "
            f"{len(missing)} missing{': ' + ', '.join(m['key'] for m in missing) if missing else ''}")

    # ---- V8: the load probe. One process per release, 38 constructions each, incremental JSONL so
    #          a hard crash costs the remainder of one release, not the probe.
    items = [[f["key"], f["path"]] for f in files if Path(f["path"]).exists()]
    items_json = ctx.work / "probe_items.json"
    items_json.write_text(json.dumps(items), encoding="utf-8")
    child = ctx.work / "_probe_child.py"
    child.write_text(PROBE_CHILD, encoding="utf-8")

    probe = {}
    env_report = {}
    for env, py in pys.items():
        jl = ctx.work / f"probe_{env}.jsonl"
        if jl.exists():
            jl.unlink()
        pt = time.time()
        rc = run_to_file([py, child, items_json, jl], ctx.logs / f"probe_{env}.log",
                         timeout=3600)
        recs = []
        if jl.exists():
            for ln in jl.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    recs.append(json.loads(ln))
        env_rec = next((r for r in recs if r["key"] == "__ENV__"), None)
        done = any(r["key"] == "__DONE__" for r in recs)
        body = {r["key"]: r for r in recs if not r["key"].startswith("__")}

        # a crashed probe costs the remainder; re-probe the missing keys one process each
        if not done:
            missing_keys = [it for it in items if it[0] not in body]
            ctx.log(f"  probe {env}: incomplete (rc={rc}); re-probing {len(missing_keys)} file(s) "
                    f"one process each")
            for k, path in missing_keys:
                one = ctx.work / f"probe_{env}_{k.replace('/', '_')}.jsonl"
                (ctx.work / "one.json").write_text(json.dumps([[k, path]]), encoding="utf-8")
                run_to_file([py, child, ctx.work / "one.json", one],
                            ctx.logs / f"probe_{env}_one.log", timeout=600)
                if one.exists():
                    for ln in one.read_text(encoding="utf-8").splitlines():
                        if ln.strip():
                            r = json.loads(ln)
                            if not r["key"].startswith("__"):
                                body[r["key"]] = r
                else:
                    body[k] = {"key": k, "loads": False, "construct_s": None,
                               "err": "probe process died before writing a record"}
        ver = (env_rec or {}).get("ort", "?")
        env_report[env] = {"reported_ort": ver, "numpy": (env_rec or {}).get("numpy"),
                           "python": (env_rec or {}).get("python"), "probe_rc": rc,
                           "probe_wall_s": round(time.time() - pt, 1),
                           "n_loads": sum(1 for r in body.values() if r.get("loads")),
                           "n_fails": sum(1 for r in body.values() if not r.get("loads"))}
        probe[ver] = {"env_dir": env, "files": body}
        ctx.log(f"  probe {env}: ort {ver} numpy {(env_rec or {}).get('numpy')} -> "
                f"{env_report[env]['n_loads']} load, {env_report[env]['n_fails']} fail, "
                f"{env_report[env]['probe_wall_s']}s")

    # V1 verdict: reported version present, expected, unique, and hashes as registered
    v1 = {}
    for env, info in interp.items():
        rep = env_report.get(env, {}).get("reported_ort")
        exp = EXPECTED_BINARIES.get(rep, {})
        got = info["binaries"]
        v1[env] = {"reported": rep, "on_ladder": rep in RELEASES,
                   "hash_match": all(got.get(k) == v for k, v in exp.items()) if exp else None,
                   "expected": exp, "got": got,
                   "numpy": env_report.get(env, {}).get("numpy"),
                   "python_version": env_report.get(env, {}).get("python")}
    seen_ver = [r["reported"] for r in v1.values()]
    v1_ok = (sorted(x for x in seen_ver if x) == sorted(RELEASES)
             and all(r["hash_match"] is not False for r in v1.values()))
    ctx.log(f"V1 verdict: releases on the ladder {sorted(x for x in seen_ver if x)}")
    for env, r in sorted(v1.items()):
        if r["hash_match"] is False:
            ctx.log(f"  V1 MISMATCH {env} ({r['reported']}): expected {r['expected']} got {r['got']}")
    ctx.log(f"V1 {'PASS' if v1_ok else 'FAIL'} "
            f"({sum(1 for r in v1.values() if r['hash_match'])} / {len(v1)} hash-verified)")

    # ---- the NOT-APPLICABLE set, decided by the probe and by nothing else
    not_applicable = sorted((k, ver) for ver, blk in probe.items()
                            for k, r in blk["files"].items() if not r.get("loads"))
    globally_na = sorted({k for k, _ in not_applicable
                          if sum(1 for kk, _ in not_applicable if kk == k) == len(probe)})
    expected_set = {tuple(x) for x in EXPECTED_NOT_APPLICABLE}
    ctx.log(f"V8 NOT-APPLICABLE: {len(not_applicable)} (file, release) pairs; expected "
            f"{len(expected_set)}")
    for extra in sorted(set(not_applicable) - expected_set):
        ctx.log(f"  V8 UNEXPECTED not-applicable: {extra[0]} on {extra[1]}")
    for gone in sorted(expected_set - set(not_applicable)):
        ctx.log(f"  V8 expected not-applicable but it LOADS: {gone[0]} on {gone[1]}")
    if globally_na:
        ctx.log(f"  V8 globally not-applicable (excludable by --exclude): {globally_na}")

    probe_doc = {"experiment": "B", "stage": 0, "written_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol": "results/paper/b_design_registered.md",
                 "note": "Session-construction wall clock sizes the budget (8.3) and is NEVER a "
                         "result (12). No inference was timed in this stage.",
                 "interpreters": env_report, "n_files": len(files),
                 "files": {f["key"]: {"path": f["path"], "sha256": f["sha256"], "bytes": f["bytes"]}
                           for f in files},
                 "probe": probe,
                 "not_applicable": [[k, v] for k, v in not_applicable],
                 "globally_not_applicable": globally_na,
                 "expected_not_applicable": [list(x) for x in EXPECTED_NOT_APPLICABLE],
                 "matches_expected": sorted(not_applicable) == sorted(expected_set)}
    (ctx.out / "load_probe.json").write_text(json.dumps(probe_doc, indent=1), encoding="utf-8")
    ctx.log(f"wrote {ctx.out / 'load_probe.json'} "
            f"(sha256 {sha256(ctx.out / 'load_probe.json')[:12]})")

    # ---- 8.3 pre-flight budget and the cuts, decided NOW and not at 03:00
    plan = build_plan(ctx, probe_doc, args)

    manifest = {"experiment": "B", "written_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": "results/paper/b_design_registered.md",
                "protocol_sha256": sha256(ROOT / "results/paper/b_design_registered.md"),
                "harness": HARNESS, "harness_sha256": sha256(ROOT / HARNESS),
                "runner": "scripts/paper/b_run.py",
                "runner_sha256": sha256(Path(__file__).resolve()),
                "envs_dir": str(ctx.envs).replace("\\", "/"),
                "seeds": SEEDS, "repeats": REPEATS, "threads": THREADS, "spin": "runtime default",
                "V1": v1, "V1_pass": v1_ok, "dropped_duplicate_envs": dropped,
                "V2_pre_run": {f["key"]: f["sha256"] for f in files},
                "deviations": [{"item": a, "handling": b} for a, b in DEVIATIONS],
                "load_probe_sha256": sha256(ctx.out / "load_probe.json")}
    (ctx.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ctx.log(f"wrote {ctx.out / 'manifest.json'}")
    ctx.log(f"STAGE 0 done in {(time.time() - t0) / 60:.1f} min")
    return 0 if (v1_ok and not missing) else 0  # never blocks; the artefact carries the verdict


# ============================================================================================
# the cost model of 9, recomputed from the probe + the pilot p50 table

def pilot_p50() -> dict:
    """(family, variant, threads) -> median p50_ms from the three stored CSVs. Read-only; these rows
    are the pilot of 3.4 and never enter the map."""
    import pandas as pd
    out = {}
    for f in ["results/paper/ort_version_scope.csv",
              "results/paper/a2_thirdparty_exploratory.csv",
              "results/paper/ort_version_bisect.csv"]:
        p = ROOT / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d["p50_ms"].notna()]
        for (fam, var, t), g in d.groupby(["family", "variant", "threads"]):
            out.setdefault((str(fam), str(var), int(t)), float(g["p50_ms"].median()))
    return out


def session_seconds(p50_ms: float, construct_s: float, baseline_construct: float) -> float:
    """9: t = 0.52 + 1.5 (warm-up) + max(2.0, 40 x p50). The 0.52 was fitted on files up to
    24.56 MB; r50 p00_fp32 is 94 MB, so stage 0 MEASURES construction and the excess over the
    calibration baseline is added on top (R13)."""
    excess = max(0.0, (construct_s or 0.0) - baseline_construct)
    return 0.52 + excess + 1.5 + max(2.0, 40.0 * (p50_ms / 1000.0))


def build_plan(ctx: Ctx, probe_doc: dict, args) -> dict:
    pil = pilot_p50()
    probe = probe_doc["probe"]
    def vkey(v: str):
        try:
            return (0, [int(x) for x in v.split(".")])
        except ValueError:                      # a release whose probe never reported a version
            return (1, [0, 0, 0])
    releases = sorted(probe.keys(), key=vkey)
    n_env = len(releases)

    # calibration baseline: the construction cost of the files the 0.52 s constant was fitted on
    small = [r["construct_s"] for ver in releases for k, r in probe[ver]["files"].items()
             if r.get("loads") and ((probe_doc["files"].get(k) or {}).get("bytes") or 0) <= 25 << 20
             and r.get("construct_s")]
    baseline = statistics.median(small) if small else 0.0
    ctx.log(f"cost model: construction baseline {baseline * 1000:.0f} ms over {len(small)} "
            f"loads of files <= 25 MB")

    cuts_applied = []
    def compute(drop_threads_breadth=False, drop_releases_breadth=(), drop_ae=False):
        plan_batches, total_s = [], 0.0
        for b in BATCHES:
            if drop_ae and b["arm"] == "APPENDIX":
                continue
            thr = [4] if (drop_threads_breadth and b["arm"] == "BREADTH") else [1, 4]
            rels = [r for r in releases
                    if not (b["arm"] == "BREADTH" and r in drop_releases_breadth)]
            n_ok = n_fail = 0
            secs = 0.0
            for v in b["variants"]:
                key = f"{b['name']}/{v}"
                for rel in rels:
                    rec = probe.get(rel, {}).get("files", {}).get(key, {})
                    loads = bool(rec.get("loads"))
                    cs = rec.get("construct_s") or 0.0
                    for t in thr:
                        p50 = pil.get((b["name"], v, t))
                        if p50 is None:
                            p50 = pil.get((b["name"], v, 4)) or pil.get((b["name"], v, 1)) or 10.0
                            if t == 1:
                                p50 *= 2.5      # 1-thread cells are slower; conservative fill
                        for _ in range(REPEATS):
                            if loads:
                                secs += session_seconds(p50, cs, baseline)
                                n_ok += 1
                            else:
                                secs += 0.52 + cs        # fast fail, no warm-up, no window
                                n_fail += 1
            plan_batches.append({**{k: b[k] for k in
                                    ("n", "arm", "name", "d", "shape", "variants", "seed")},
                                 "threads": ",".join(str(t) for t in thr),
                                 "releases": rels, "n_releases": len(rels),
                                 "sessions_measured": n_ok, "sessions_not_applicable": n_fail,
                                 "rows_expected": len(b["variants"]) * len(rels) * len(thr) * REPEATS,
                                 "restrict_versions": (",".join(rels)
                                                       if len(rels) != n_env else None),
                                 "est_min": round(secs / 60, 1),
                                 "protocol_est_min": b["est_min"]})
            total_s += secs
        return plan_batches, total_s

    pb, total_s = compute()
    anchor_s = 2 * (len(BATCHES) + 1) * (args.anchor_reps * 4.02 + 12.0)
    ctx.log(f"pre-flight estimate: map {total_s / 3600:.2f} h + anchors {anchor_s / 60:.0f} min")
    if (total_s + anchor_s) / 3600 > PREFLIGHT_BUDGET_H:
        cuts_applied.append("1: BREADTH drops 1 thread, keeping 4")
        pb, total_s = compute(drop_threads_breadth=True)
        ctx.log(f"  cut 1 applied -> {total_s / 3600:.2f} h")
    if (total_s + anchor_s) / 3600 > PREFLIGHT_BUDGET_H:
        cuts_applied.append("2: BREADTH drops releases 1.21.1 and 1.22.1")
        pb, total_s = compute(drop_threads_breadth=True,
                              drop_releases_breadth=("1.21.1", "1.22.1"))
        ctx.log(f"  cut 2 applied -> {total_s / 3600:.2f} h")
    if (total_s + anchor_s) / 3600 > PREFLIGHT_BUDGET_H:
        cuts_applied.append("3: the ae appendix is dropped")
        pb, total_s = compute(drop_threads_breadth=True,
                              drop_releases_breadth=("1.21.1", "1.22.1"), drop_ae=True)
        ctx.log(f"  cut 3 applied -> {total_s / 3600:.2f} h")

    plan = {"written_utc": datetime.now(timezone.utc).isoformat(),
            "releases": releases, "n_interpreters": n_env, "repeats": REPEATS,
            "construction_baseline_s": baseline,
            "preflight_budget_h": PREFLIGHT_BUDGET_H, "hard_stop_h": HARD_STOP_H,
            "cuts_applied": cuts_applied,
            "anchor_reps": args.anchor_reps, "anchor_est_min": round(anchor_s / 60, 1),
            "map_est_h": round(total_s / 3600, 2),
            "total_est_h": round((total_s + anchor_s) / 3600, 2),
            "sessions_measured": sum(b["sessions_measured"] for b in pb),
            "sessions_not_applicable": sum(b["sessions_not_applicable"] for b in pb),
            "globally_not_applicable": probe_doc["globally_not_applicable"],
            "batches": pb}
    (ctx.out / "batch_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    ctx.log("batch plan:")
    ctx.log(f"  {'#':>2} {'arm':9s} {'batch':12s} {'rel':>3s} {'thr':>4s} {'meas':>5s} "
            f"{'NA':>4s} {'est':>6s} {'proto':>6s}")
    for b in pb:
        ctx.log(f"  {b['n']:2d} {b['arm']:9s} {b['name']:12s} {b['n_releases']:3d} "
                f"{b['threads']:>4s} {b['sessions_measured']:5d} "
                f"{b['sessions_not_applicable']:4d} {b['est_min']:6.1f} {b['protocol_est_min']:6.1f}")
    ctx.log(f"  TOTAL measured {plan['sessions_measured']} sessions, "
            f"{plan['sessions_not_applicable']} not-applicable, map {plan['map_est_h']:.2f} h, "
            f"with anchors {plan['total_est_h']:.2f} h; cuts: {cuts_applied or 'none'}")
    return plan


# ============================================================================================
# stage 1: the fifteen batches

def harness_cmd(ctx: Ctx, out_csv: Path, fam: str, d: str, shape: str, variants: list[str],
                threads: str, seed: int, repeats: int, exclude: str = "",
                versions: str | None = None) -> list:
    cmd = [PY, HARNESS,
           "--scratch", str(ctx.envs),
           "--threads", threads,
           "--repeats", str(repeats),
           "--seed", str(seed),
           "--families", f"{fam}={d}:{shape}",
           "--variants", ",".join(variants),
           "--drop-duplicate-envs",
           "--status-rows",
           "--exclude", exclude,
           "--out", str(out_csv).replace("\\", "/")]
    if versions:
        cmd += ["--versions", versions]
    return cmd


def batch_complete(csv_path: Path, rows_expected: int | None) -> tuple[bool, int]:
    if not csv_path.exists():
        return False, 0
    try:
        n = len(csv_rows(csv_path))
    except Exception:
        return False, 0
    if rows_expected is None:
        return n > 0, n
    return n == rows_expected, n


def run_anchor(ctx: Ctx, boundary: int, reps: int) -> None:
    """5.6 / V4. Both anchors, on the single 1.30.0 interpreter, into their own CSVs. Injected at
    batch boundaries - see DEVIATIONS 11.2(c)."""
    for a in ANCHORS:
        out = ctx.batches / f"anchor_{a['tag']}_b{boundary:02d}.csv"
        if out.exists() and len(csv_rows(out)) >= reps:
            continue
        cmd = harness_cmd(ctx, out, a["name"], a["d"], a["shape"], [a["variant"]],
                          str(a["threads"]), seed=SEEDS["base"], repeats=reps,
                          versions=ANCHOR_RELEASE)
        t0 = time.time()
        rc = run_to_file(cmd, ctx.logs / f"anchor_{a['tag']}_b{boundary:02d}.log",
                         timeout=60 * 20)
        n = len(csv_rows(out)) if out.exists() else 0
        ctx.log(f"  anchor {a['tag']} @boundary {boundary}: exit={rc} rows={n} "
                f"({time.time() - t0:.0f}s)")


def run_map(ctx: Ctx, args) -> int:
    plan_path = ctx.out / "batch_plan.json"
    if not plan_path.exists():
        ctx.log("no batch_plan.json - run --stage 0 first")
        return 2
    plan = load_json(plan_path)
    globally_na = set(plan.get("globally_not_applicable") or [])
    ctx.log("=" * 92)
    ctx.log(f"STAGE 1 - the map. {len(plan['batches'])} batches, {plan['sessions_measured']} "
            f"measured sessions, estimate {plan['total_est_h']:.2f} h, hard stop {HARD_STOP_H} h.")
    t_start = time.time()
    quiet_failures = 0
    done, skipped, failed = [], [], []

    run_anchor(ctx, 0, args.anchor_reps)
    for b in plan["batches"]:
        csv_path = ctx.batches / f"b{b['n']:02d}_{b['arm'].lower()}_{b['name']}.csv"
        ok, n = batch_complete(csv_path, b["rows_expected"])
        if ok:
            ctx.log(f"[{b['n']}/{len(plan['batches'])}] {b['name']}: complete already "
                    f"({n} rows) - not re-run")
            skipped.append(b["name"])
            continue
        if csv_path.exists():
            ctx.log(f"[{b['n']}/{len(plan['batches'])}] {b['name']}: incomplete CSV "
                    f"({n} rows, expected {b['rows_expected']}) - re-running the whole batch")
        if (time.time() - t_start) / 3600 > HARD_STOP_H:
            ctx.log(f"HARD STOP at batch boundary: {(time.time() - t_start) / 3600:.2f} h elapsed. "
                    f"Unstarted batches are listed in the log and in run_state.json.")
            break

        q = quiet_machine(ctx, f"batch {b['n']} {b['name']}")
        if not q["passed"]:
            quiet_failures += 1
            ctx.log(f"[{b['n']}] {b['name']}: QUIET-MACHINE FAILED ({q['reason']}); batch not "
                    f"started ({quiet_failures} of 3 allowed)")
            failed.append((b["name"], "quiet-machine"))
            if quiet_failures >= 3:
                ctx.log("three quiet-machine failures in one night: stopping (8.3).")
                break
            continue

        exclude = ";".join(sorted(k.replace("/", ":") for k in globally_na
                                  if k.split("/")[0] == b["name"]))
        cmd = harness_cmd(ctx, csv_path, b["name"], b["d"], b["shape"], b["variants"],
                          b["threads"], b["seed"], REPEATS, exclude=exclude,
                          versions=b.get("restrict_versions"))
        ctx.log(f"[{b['n']}/{len(plan['batches'])}] {b['arm']} {b['name']}: "
                f"{b['sessions_measured']} sessions, seed {b['seed']}, est {b['est_min']:.1f} min"
                f"{', exclude ' + exclude if exclude else ''}")
        b["start_utc"] = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        timeout = int(max(1800, b["est_min"] * 60 * 3 + SESSION_TIMEOUT))
        rc = run_to_file(cmd, ctx.logs / f"b{b['n']:02d}_{b['name']}.log", timeout=timeout)
        b["end_utc"] = datetime.now(timezone.utc).isoformat()
        b["wall_min"] = round((time.time() - t0) / 60, 1)
        ok, n = batch_complete(csv_path, b["rows_expected"])
        b["rows_written"] = n
        b["exit_code"] = rc
        b["complete"] = ok
        if ok:
            stats = batch_stats(csv_path)
            b.update(stats)
            ctx.log(f"  -> exit={rc} rows={n} in {b['wall_min']} min | OK {stats['n_ok']} "
                    f"FAILED {stats['n_failed']} NO_FILE {stats['n_no_file']} "
                    f"EXCLUDED {stats['n_excluded']} | below-gate "
                    f"{stats['below_gate_share']:.1%}")
            if stats["below_gate_share"] > V6_BATCH_LIMIT:
                ctx.log(f"  V6: below-gate share {stats['below_gate_share']:.1%} exceeds "
                        f"{V6_BATCH_LIMIT:.0%}; the protocol re-runs this batch once. Not done "
                        f"automatically - recorded for the analyst.")
            done.append(b["name"])
        else:
            ctx.log(f"  -> exit={rc} rows={n} (expected {b['rows_expected']}) in {b['wall_min']} "
                    f"min: BATCH FAILED, recorded, not retried")
            failed.append((b["name"], f"exit={rc} rows={n}"))
        sp = stray_pythons()
        ctx.log(f"  hygiene: {sp} python process(es) on the machine after the batch")
        run_anchor(ctx, b["n"], args.anchor_reps)
        plan["batches"] = [b if x["n"] == b["n"] else x for x in plan["batches"]]
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")

    state = {"finished_utc": datetime.now(timezone.utc).isoformat(),
             "elapsed_h": round((time.time() - t_start) / 3600, 2),
             "batches_complete": done, "batches_already_complete": skipped,
             "batches_failed": failed,
             "batches_unstarted": [x["name"] for x in plan["batches"]
                                   if not x.get("complete") and x["name"] not in
                                   [f[0] for f in failed] and x["name"] not in skipped]}
    (ctx.out / "run_state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    ctx.log(f"STAGE 1 finished: {len(done)} run, {len(skipped)} already complete, "
            f"{len(failed)} failed, {len(state['batches_unstarted'])} unstarted, "
            f"{state['elapsed_h']:.2f} h")
    return 0


def batch_stats(csv_path: Path) -> dict:
    rows = csv_rows(csv_path)
    st = [r.get("status", "OK") for r in rows]
    cd = []
    for r in rows:
        try:
            v = float(r.get("core_delivery") or "nan")
            if v == v:
                cd.append(v)
        except ValueError:
            pass
    return {"n_rows": len(rows), "n_ok": st.count("OK"), "n_failed": st.count("FAILED"),
            "n_no_file": st.count("NO_FILE"), "n_excluded": st.count("EXCLUDED"),
            "below_gate_share": (sum(1 for v in cd if v < GATE) / len(cd)) if cd else 0.0,
            "core_delivery_median": round(statistics.median(cd), 3) if cd else None}


# ============================================================================================
# stage post: V2 again, and the merged CSVs of 11.1

def merge(ctx: Ctx) -> int:
    plan_path = ctx.out / "batch_plan.json"
    plan = load_json(plan_path) if plan_path.exists() else {"batches": []}
    by_arm = {"CONTRAST": [], "BREADTH": []}
    for b in plan["batches"]:
        csv_path = ctx.batches / f"b{b['n']:02d}_{b['arm'].lower()}_{b['name']}.csv"
        if not csv_path.exists():
            continue
        arm = "CONTRAST" if b["arm"] in ("CONTRAST", "APPENDIX") else "BREADTH"
        for r in csv_rows(csv_path):
            r.update({"arm": b["arm"], "batch_name": b["name"], "batch_n": b["n"],
                      "batch_seed": b["seed"], "batch_start_utc": b.get("start_utc", ""),
                      "batch_end_utc": b.get("end_utc", "")})
            by_arm[arm].append(r)
    for arm, name in (("CONTRAST", "b_drift_contrast.csv"), ("BREADTH", "b_drift_breadth.csv")):
        rows = by_arm[arm]
        if rows:
            write_csv(ctx.out / name, rows)
            ctx.log(f"merged {len(rows)} rows -> {ctx.out / name}")

    arows = []
    for p in sorted(ctx.batches.glob("anchor_*.csv")):
        m = re.match(r"anchor_(.+)_b(\d+)\.csv", p.name)
        for r in csv_rows(p):
            r.update({"anchor": m.group(1), "boundary": int(m.group(2)),
                      "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime,
                                                          timezone.utc).isoformat()})
            arows.append(r)
    if arows:
        write_csv(ctx.out / "b_drift_anchor.csv", arows)
        ctx.log(f"merged {len(arows)} anchor rows -> {ctx.out / 'b_drift_anchor.csv'}")
        for a in ANCHORS:
            vals = [(r["boundary"], float(r["p50_ms"])) for r in arows
                    if r["anchor"] == a["tag"] and r.get("p50_ms") not in ("", None)
                    and r.get("status", "OK") == "OK"]
            vals.sort()
            if len(vals) >= 6:
                k = len(vals) // 3
                first = statistics.median(v for _, v in vals[:k])
                last = statistics.median(v for _, v in vals[-k:])
                drift = last / first - 1
                ctx.log(f"  V4 {a['tag']}: first-third {first:.4f} ms, last-third {last:.4f} ms, "
                        f"drift {drift:+.2%} -> {'FIRES' if abs(drift) >= 0.07 else 'quiet'}")

    # V2 post-run hashes
    files = probe_files()
    post = {f["key"]: (sha256(Path(f["path"])) if Path(f["path"]).exists() else None)
            for f in files}
    mp = ctx.out / "manifest.json"
    man = load_json(mp) if mp.exists() else {}
    pre = man.get("V2_pre_run", {})
    changed = [k for k in post if pre.get(k) and post[k] and pre[k] != post[k]]
    man["V2_post_run"] = post
    man["V2_changed"] = changed
    man["V2_pass"] = not changed
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    ctx.log(f"V2 post-run: {len(post)} files hashed, {len(changed)} changed "
            f"{'-> ' + ', '.join(changed) if changed else '(map premise holds)'}")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ============================================================================================
# 5.5 the replication stage: five FRESH sessions a side, its own stratum, never pooled with the map

def replicate(ctx: Ctx, args) -> int:
    src = Path(args.pairs) if args.pairs else ctx.out / "moves_sealed.json"
    if not src.exists():
        ctx.log(f"replication needs the analyst's pair-cell list at {src} "
                f"(MOVED + UNRESOLVED, each with abs_delta). Nothing to do.")
        return 2
    pairs = load_json(src)
    pairs = pairs.get("pair_cells", pairs) if isinstance(pairs, dict) else pairs
    want = [p for p in pairs if p.get("verdict") in (None, "MOVED", "UNRESOLVED")]
    want.sort(key=lambda p: (-abs(float(p.get("abs_delta", p.get("delta", 0)))),
                             p.get("arm", ""), p.get("model", ""), p.get("variant", ""),
                             int(p.get("threads", 0)), str(p.get("transition", ""))))
    ctx.log(f"replication: {len(want)} MOVED/UNRESOLVED pair-cells, descending |delta|, "
            f"cap {args.replicate_cap_min} min, seed {SEEDS['replication']}")

    pil = pilot_p50()
    by_model, order = {}, []
    for p in want:
        m = p["model"]
        if m not in by_model:
            by_model[m] = {"variants": set(), "threads": set(), "releases": set(), "cells": []}
            order.append(m)
        by_model[m]["variants"].add(p["variant"])
        by_model[m]["threads"].add(int(p["threads"]))
        by_model[m]["releases"].update([p["release_a"], p["release_b"]])
        by_model[m]["cells"].append(p)

    selected, not_tested, cum_s = [], [], 0.0
    for m in order:
        blk = by_model[m]
        b = next((x for x in BATCHES if x["name"] == m), None)
        if b is None:
            not_tested += blk["cells"]
            continue
        secs = 0.0
        for v in sorted(blk["variants"]):
            for t in sorted(blk["threads"]):
                p50 = pil.get((m, v, t)) or 10.0
                secs += len(blk["releases"]) * REPEATS * session_seconds(p50, 0.0, 0.0)
        if (cum_s + secs) / 60 > args.replicate_cap_min:
            not_tested += blk["cells"]
            ctx.log(f"  {m}: {secs / 60:.1f} min would exceed the cap -> NOT-TESTED "
                    f"({len(blk['cells'])} pair-cells)")
            continue
        cum_s += secs
        selected.append(m)
    ctx.log(f"  selected {len(selected)} model invocation(s), {cum_s / 60:.1f} min; "
            f"{len(not_tested)} pair-cells NOT-TESTED")

    for m in selected:
        blk = by_model[m]
        b = next(x for x in BATCHES if x["name"] == m)
        out = ctx.batches / f"rep_{m}.csv"
        if out.exists():
            ctx.log(f"  {m}: replication CSV exists, not re-run")
            continue
        q = quiet_machine(ctx, f"replication {m}")
        if not q["passed"]:
            ctx.log(f"  {m}: quiet-machine failed ({q['reason']}); skipped")
            continue
        cmd = harness_cmd(ctx, out, m, b["d"], b["shape"], sorted(blk["variants"]),
                          ",".join(str(t) for t in sorted(blk["threads"])),
                          SEEDS["replication"], REPEATS,
                          versions=",".join(sorted(blk["releases"])))
        rc = run_to_file(cmd, ctx.logs / f"rep_{m}.log", timeout=int(cum_s * 3 + 1800))
        ctx.log(f"  {m}: exit={rc} rows={len(csv_rows(out)) if out.exists() else 0}")

    rows = []
    for p in sorted(ctx.batches.glob("rep_*.csv")):
        for r in csv_rows(p):
            r["batch_name"] = p.stem[4:]
            r["stratum"] = "replication"
            rows.append(r)
    if rows:
        write_csv(ctx.out / "b_drift_replicate.csv", rows)
        ctx.log(f"wrote {ctx.out / 'b_drift_replicate.csv'} ({len(rows)} rows)")
    (ctx.out / "replication_selection.json").write_text(
        json.dumps({"seed": SEEDS["replication"], "cap_min": args.replicate_cap_min,
                    "selected_models": selected, "est_min": round(cum_s / 60, 1),
                    "not_tested": not_tested}, indent=1), encoding="utf-8")
    return 0


# ============================================================================================
# 7 step 1: retrieve the release notes verbatim. OFF by default - the one step that needs network.

def fetch_notes(ctx: Ctx) -> int:
    import urllib.request
    nd = ctx.out / "notes"
    nd.mkdir(parents=True, exist_ok=True)
    man = {}
    for v in RELEASES:
        url = f"https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v{v}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = json.loads(r.read().decode("utf-8")).get("body", "")
        except Exception as e:
            ctx.log(f"  notes {v}: FAILED ({e})")
            man[v] = {"url": url, "error": str(e)[:200]}
            continue
        p = nd / f"{v}.md"
        p.write_text(body, encoding="utf-8")
        man[v] = {"url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                  "sha256": sha256(p), "bytes": p.stat().st_size}
        ctx.log(f"  notes {v}: {p.stat().st_size} bytes, sha256 {man[v]['sha256'][:12]}")
    (ctx.out / "notes_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    ctx.log("notes retrieved. NOTHING is interpreted at retrieval time (7 step 1).")
    return 0


# ============================================================================================
# the smoke test: tiny, its own output root, proves the plumbing and touches no protocol output

def smoke(ctx: Ctx, args) -> int:
    ctx.log("=" * 92)
    ctx.log("SMOKE - at most 2 models x 2 releases x 1 thread x 1 session per invocation. "
            "Output stays under this root and is not a result.")
    pys_all = interpreter_map(ctx.envs, drop_duplicates=False)
    pys = interpreter_map(ctx.envs, drop_duplicates=True)
    ctx.log(f"  interpreters: {len(pys_all)} without --drop-duplicate-envs "
            f"({sorted(pys_all)}), {len(pys)} with it; dropped {sorted(set(pys_all) - set(pys))}")

    # A: OK + NO_FILE + EXCLUDED, two models, two releases, one thread, one session
    a_out = ctx.batches / "smoke_a.csv"
    cmd = [PY, HARNESS, "--scratch", str(ctx.envs), "--threads", "1", "--repeats", "1",
           "--seed", "72",
           "--families", "mnv2=results/classification_dyn/models:3,32,32;"
                         "c3_112x112=D:/sec_models/c3_112x112:3,112,112",
           "--variants", "p00_int8,c04,p00_int8_dyn_conv",
           "--versions", "1.23.2,1.24.4",
           "--drop-duplicate-envs", "--status-rows",
           "--exclude", "mnv2:p00_int8_dyn_conv",
           "--out", str(a_out).replace("\\", "/")]
    rc = run_to_file(cmd, ctx.logs / "smoke_a.log", timeout=1800)
    ctx.log(f"  smoke A exit={rc}")
    # B: the FAILED status, from a cell the protocol declares NOT-APPLICABLE below 1.24.4
    b_out = ctx.batches / "smoke_b.csv"
    cmd = [PY, HARNESS, "--scratch", str(ctx.envs), "--threads", "1", "--repeats", "1",
           "--seed", "82",
           "--families", "mnv2=results/classification_dyn/models:3,32,32",
           "--variants", "p00_int8_dyn_conv", "--versions", "1.23.2,1.24.4",
           "--drop-duplicate-envs", "--status-rows", "--exclude", "",
           "--out", str(b_out).replace("\\", "/")]
    rc = run_to_file(cmd, ctx.logs / "smoke_b.log", timeout=1800)
    ctx.log(f"  smoke B exit={rc}")
    # C: the anchor path (5.6) and the merge path, one session per anchor on 1.30.0
    run_anchor(ctx, 0, 1)
    merge(ctx)

    summary = {}
    for tag, p in (("A", a_out), ("B", b_out)):
        if not p.exists():
            ctx.log(f"  smoke {tag}: NO CSV")
            continue
        rows = csv_rows(p)
        cols = list(rows[0].keys()) if rows else []
        counts = {}
        for r in rows:
            counts[r.get("status", "(none)")] = counts.get(r.get("status", "(none)"), 0) + 1
        envs_used = sorted({r["env_dir"] for r in rows})
        orts = sorted({r["ort"] for r in rows if r["ort"]})
        ctx.log(f"  smoke {tag}: {len(rows)} rows, columns {cols}")
        ctx.log(f"  smoke {tag}: statuses {counts}, env_dir {envs_used}, ort {orts}")
        summary[tag] = {"rows": len(rows), "columns": cols, "statuses": counts,
                        "env_dirs": envs_used, "ort": orts,
                        "detail_samples": [r.get("detail", "")[:160] for r in rows
                                           if r.get("status") == "FAILED"][:3]}
    anchor_csv = ctx.out / "b_drift_anchor.csv"
    if anchor_csv.exists():
        ar = csv_rows(anchor_csv)
        summary["anchor"] = {"rows": len(ar),
                             "cells": sorted({(r["anchor"], r["family"], r["variant"], r["threads"],
                                               r["ort"], r["status"]) for r in ar}),
                             "p50_ms": [r["p50_ms"] for r in ar]}
        ctx.log(f"  smoke anchor: {len(ar)} rows -> {summary['anchor']['cells']}")
    summary["interpreters_without_dedup"] = sorted(pys_all)
    summary["interpreters_with_dedup"] = sorted(pys)
    summary["dropped"] = sorted(set(pys_all) - set(pys))
    (ctx.out / "smoke_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    ctx.log(f"wrote {ctx.out / 'smoke_summary.json'}")
    return 0


# ============================================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["0", "map", "post", "all", "smoke", "replicate", "notes", "plan"])
    ap.add_argument("--envs", default=str(DEFAULT_ENVS), help="directory holding the ortNNN venvs")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "expB"))
    ap.add_argument("--work", default="D:/sec_work/b_run", help="child scripts and probe scratch")
    ap.add_argument("--anchor-reps", type=int, default=3,
                    help="sessions per anchor per batch boundary (5.6)")
    ap.add_argument("--replicate-cap-min", type=float, default=75.0, help="5.5 cap")
    ap.add_argument("--pairs", default=None, help="5.5 pair-cell list (default moves_sealed.json)")
    a = ap.parse_args()

    ctx = Ctx(Path(a.out_root), Path(a.work), Path(a.envs))
    ctx.log("#" * 92)
    ctx.log(f"Experiment B runner | stage {a.stage} | out {ctx.out} | envs {ctx.envs}")
    ctx.log(f"protocol results/paper/b_design_registered.md "
            f"sha256 {sha256(ROOT / 'results/paper/b_design_registered.md')[:12]} | "
            f"harness {HARNESS} sha256 {sha256(ROOT / HARNESS)[:12]}")
    ctx.log(f"seeds: batches 72-86, replication {SEEDS['replication']}, "
            f"decoys {SEEDS['decoy_sampling']}, bootstrap {SEEDS['cluster_bootstrap']} | "
            f"repeats {REPEATS} | threads {THREADS} | spin: runtime default")

    if a.stage == "smoke":
        return smoke(ctx, a)
    if a.stage == "notes":
        return fetch_notes(ctx)
    if a.stage == "replicate":
        return replicate(ctx, a)
    if a.stage == "plan":
        probe = ctx.out / "load_probe.json"
        if not probe.exists():
            ctx.log("no load_probe.json - run --stage 0 first")
            return 2
        build_plan(ctx, load_json(probe), a)
        return 0
    if a.stage in ("0", "all"):
        rc = stage0(ctx, a)
        if rc:
            return rc
    if a.stage in ("map", "all"):
        rc = run_map(ctx, a)
        if rc:
            return rc
    if a.stage in ("post", "all"):
        return merge(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
