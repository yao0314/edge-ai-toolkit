"""Experiment A2, exploratory scan: predict, freeze, measure, analyse — in one run.

The registered pool minimum (18 admitted) was not reached: 13 were. Under the registration that bars a
VERDICT, not a measurement, so this run is labelled EXPLORATORY everywhere and no verdict is claimed from
it. The order that matters is kept: every prediction is written and hashed before the first timing row
exists.

Everything is written to files, never to a pipe, because a pipe with nobody draining it is what hung the
previous attempt.

    .venv/Scripts/python.exe scripts/paper/a2_exploratory_run.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # scripts/paper/<this file> -> the project root
PY = ROOT / ".venv" / "Scripts" / "python.exe"
ENVS = ROOT / ".envs"                      # holds ort1232 and ort1244
POOL = Path("D:/sec_models")
OUT = ROOT / "results" / "expA2"
LOG = OUT / "exploratory_run.log"
CSV = ROOT / "results" / "paper" / "a2_thirdparty_exploratory.csv"
WORK = Path("D:/sec_work/a2_explore")      # dumps and child scripts, off C:
TW = timezone(timedelta(hours=8))

CONTROLS = {                                # family -> (dir, c,h,w); variant is p00_int8
    "mobilevit": ("results/classification_mobilevit/models", (3, 64, 64)),
    "deit": ("results/classification_deit/models", (3, 32, 32)),
    "cls": ("results/classification/models", (3, 32, 32)),
}


def log(msg: str) -> None:
    line = f"{datetime.now(TW):%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], out_path: Path, timeout: int) -> int:
    """Run a command with stdout+stderr going to a FILE, so nothing can block on a full pipe."""
    with open(out_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(30)
            return -9


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest.get("measured_set") or manifest.get("measured") or []
    rows = rows if isinstance(rows, list) else list(rows.values())
    log(f"manifest: {len(rows)} admitted models")

    # ---- 1. predictions, from the files alone, before any timing
    models = []
    for r in rows:
        f = POOL / r["harness_family"] / f"{r['variant']}.onnx"
        if f.exists():
            models.append(f)
        else:
            log(f"MISSING {f}")
    # a_check indexes screen_ms by id / variant / stem / name / sha256 / path; the manifest keeps its
    # rows under "measured_set", which that lookup does not reach. Hand it a flat map instead, keyed by
    # both the absolute file path and the variant stem, built from the manifest and nothing else.
    smap = {}
    for r in rows:
        f = str((POOL / r["harness_family"] / f"{r['variant']}.onnx").resolve())
        for key in (f, f.replace("\\", "/"), r["variant"]):
            smap[key] = {"screen_ms": r.get("screen_ms"), "variant": r["variant"],
                         "sha256": r.get("sha256"), "path": f}
    (OUT / "screen_ms_map.json").write_text(json.dumps(smap, indent=1), encoding="utf-8")
    log(f"predicting on {len(models)} files (screen_ms map: {len(rows)} rows)")
    rc = run([PY, "scripts/paper/a_check.py", "predict", *models, "--work-dir", WORK / "check",
              "--manifest", OUT / "screen_ms_map.json", "--json", OUT / "predictions_exploratory.json"],
             OUT / "predict.log", timeout=3600)
    log(f"predict exit={rc}")
    if rc != 0 or not (OUT / "predictions_exploratory.json").exists():
        log("predict failed; see results/expA2/predict.log")
        return 1

    pred = json.loads((OUT / "predictions_exploratory.json").read_text(encoding="utf-8"))
    per = pred.get("models") or pred.get("predictions") or pred
    flagged = [k for k, v in (per.items() if isinstance(per, dict) else [])
               if isinstance(v, dict) and (v.get("prediction") or {}).get("predicted_class") == "PREDICT-STEP"]
    log(f"PREDICT-STEP: {len(flagged)} -> {flagged}")

    # ---- 2. the freeze: hash the prediction file before a single timing row exists
    seal = {"experiment": "A2-EXPLORATORY", "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "predictions_sha256": sha(OUT / "predictions_exploratory.json"),
            "manifest_sha256": sha(OUT / "manifest.json"),
            "a_check_sha256": sha(ROOT / "scripts" / "paper" / "a_check.py"),
            "n_models": len(models), "n_predict_step": len(flagged), "predict_step_ids": flagged,
            "note": "Pool minimum (18) not reached: 13 admitted. This run is EXPLORATORY: it produces "
                    "numbers, not a verdict, and may not be quoted as A2's result."}
    (OUT / "seal_exploratory.json").write_text(json.dumps(seal, indent=1), encoding="utf-8")
    with open(ROOT / "results" / "paper" / "FINDINGS.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Experiment A2 EXPLORATORY scan — predictions frozen "
                f"{datetime.now(TW):%Y-%m-%dT%H:%M:%S%z}\n\n"
                f"Pool minimum not reached (13 admitted of a required 18), so this scan carries NO verdict.\n"
                f"Predictions for {len(models)} third-party statically quantized models were written and hashed "
                f"before the first timing row: `results/expA2/predictions_exploratory.json` sha256 "
                f"{seal['predictions_sha256'][:16]}..., manifest {seal['manifest_sha256'][:16]}..., "
                f"checker {seal['a_check_sha256'][:16]}... PREDICT-STEP: {len(flagged)} "
                f"({', '.join(flagged) if flagged else 'none'}).\n")
    log(f"sealed: predictions {seal['predictions_sha256'][:12]}")

    # ---- 3. the measurement
    fams, variants = {}, []
    for r in rows:
        shape = r.get("input_shape") or r.get("chw")
        chw = tuple(shape[1:]) if len(shape) == 4 else tuple(shape)
        fams[r["harness_family"]] = (str(POOL / r["harness_family"]).replace("\\", "/"), chw)
        variants.append(r["variant"])
    for name, (d, chw) in CONTROLS.items():
        fams[name] = (d, chw)
    variants.append("p00_int8")
    # One invocation per shape family, not one for everything: a batch that stalls costs one family,
    # not the run, and each batch's output goes to its own file. Randomisation still happens inside a
    # batch (release x thread count x variant x session); the deviation from the registered single
    # invocation is recorded here and in the log.
    var_arg = ",".join(sorted(set(variants)))
    log(f"families: {len(fams)} | variants: {var_arg} | one batch per family")
    parts, failed = [], []
    for i, (name, (d, (c, h, w))) in enumerate(sorted(fams.items()), 1):
        part = CSV.with_name(f"{CSV.stem}_{name}.csv")
        t0 = time.time()
        rc = run([PY, "scripts/paper/ort_version_bisect.py", "--scratch", ENVS,
                  "--versions", "1.23.2,1.24.4", "--threads", "1,4", "--repeats", "5", "--seed", "63",
                  "--families", f"{name}={d}:{c},{h},{w}", "--variants", var_arg, "--out", part],
                 OUT / f"measure_{name}.log", timeout=5400)
        n = sum(1 for _ in open(part, encoding="utf-8")) - 1 if part.exists() else 0
        log(f"  [{i}/{len(fams)}] {name}: exit={rc} rows={n} in {(time.time()-t0)/60:.1f} min")
        (parts if rc == 0 and n > 0 else failed).append(name)
    if not parts:
        log("every batch failed; see results/expA2/measure_*.log")
        return 2
    if failed:
        log(f"batches with no rows: {failed} (recorded, not retried silently)")
    head, lines = None, []
    for name in parts:
        p = CSV.with_name(f"{CSV.stem}_{name}.csv")
        rows = p.read_text(encoding="utf-8").splitlines()
        head = head or rows[0]
        lines += rows[1:]
    CSV.write_text("\n".join([head, *lines]) + "\n", encoding="utf-8")
    log(f"merged {len(lines)} rows from {len(parts)} batches -> {CSV}")

    # ---- 4. analysis
    rc = run([PY, "scripts/paper/a_analyse.py", CSV], OUT / "analyse.log", timeout=1800)
    log(f"analyse exit={rc}")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
