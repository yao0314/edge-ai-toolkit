"""Experiment A, step 1: build the candidate pool (registration sections 1.1, 1.2, 1.3 and 4.1).

The registration is `results/paper/a_design_registered.md`, written and frozen before any model was
downloaded and before any timing. This script only executes its rules; it contains no judgement of
its own, it computes no prediction, and it runs no timing except the screening pass of section 4.1,
which exists solely to apply rule A8 and to give `rho` a denominator. It must not be used to produce
a latency, a baseline or a speed-up (section 8, "Barred in every branch").

    .venv\\Scripts\\python.exe scripts/paper/a_pool.py all
    .venv\\Scripts\\python.exe scripts/paper/a_pool.py licence      # publisher metadata only
    .venv\\Scripts\\python.exe scripts/paper/a_pool.py fetch        # HEAD, download, sha256
    .venv\\Scripts\\python.exe scripts/paper/a_pool.py admit        # A1..A9 (screens under 1.24.4)
    .venv\\Scripts\\python.exe scripts/paper/a_pool.py write        # pool.json + manifest.json

Every stage writes its intermediate result to `--work-dir` as JSON and every later stage reads it
back, so a stage can be re-run without repeating the network or the sessions. The two artefacts that
belong to the project are `results/expA/pool.json` (artefact 1) and `results/expA/manifest.json`
(artefact 2); everything else this script writes is scratch and may be deleted.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
* It does not compute `sig_b`, `sig_a`, `added_dispatches`, `A_ms` or `rho`. Those are artefact 3
  (`predictions.json`, `scripts/paper/a_check.py`) and by section 3 they are written AFTER the pool.
  Consequence, recorded rather than patched: `manifest.json`'s `stratum` field, which section 3
  lists as part of artefact 2, cannot be filled in at this point, because the strata of section 1.3
  are defined from the checker's flag. It is written as null with a note naming the artefact that
  must fill it. Section 1.3's SELECTION is unaffected as long as the admitted pool holds 25 models
  or fewer and |F| <= 12, which the script asserts and records.
* It never edits the registration, `transpose_probe.py`, `transpose_risk.py` or
  `ort_version_bisect.py`.

CONVENTIONS THAT THE REGISTRATION LEAVES TO THE IMPLEMENTATION, FIXED HERE AND RECORDED
---------------------------------------------------------------------------------------
* A7's "60 MB" and the 2.5 GB pool cap are decimal (60,000,000 and 2,500,000,000 bytes). This is the
  convention the approved candidate list already used: it rejected `yolov3-12-int8` at 62,912,863
  bytes under A7, which is below 60 MiB (62,914,560) and above 60 MB.
* "Quiet machine" (VA4) is read from `Win32_PerfFormattedData_PerfOS_Processor` `_Total`, never from
  `Win32_Processor.LoadPercentage`: 5 one-second samples, mean <= 15%, on AC power, re-checked
  before every screening session, up to 10 one-minute retries.
* A6's inference and the screening session use the same session options and the same P-core pinning
  as `.envs/_bisect_child.py`, the child `ort_version_bisect.py` runs, so nothing about the session
  changes between the screen and the primary run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.cpu_topology import pick_cpus  # noqa: E402

import onnx  # noqa: E402
from onnx import TensorProto  # noqa: E402

# ---- paths ------------------------------------------------------------------------------------
OUT_DIR = ROOT / "results" / "expA"
CANDIDATES = OUT_DIR / "candidates.json"
REGISTRATION = ROOT / "results" / "paper" / "a_design_registered.md"
A2_OUT_DIR = ROOT / "results" / "expA2"
A2_CANDIDATES2 = A2_OUT_DIR / "candidates_round2.json"
A2_CANDIDATES3 = A2_OUT_DIR / "candidates_round3.json"
A2_REGISTRATION = ROOT / "results" / "paper" / "a2_design_registered.md"
ENVS = ROOT / ".envs"
PY = {"1.23.2": ENVS / "ort1232" / "Scripts" / "python.exe",
      "1.24.4": ENVS / "ort1244" / "Scripts" / "python.exe"}

# `--experiment A` is Experiment A exactly as it ran: one candidate list, one registration, A 1.1's
# licence rule, A9 applied first-come. `--experiment A2` is the delta registered in
# `results/paper/a2_design_registered.md`: two candidate lists, L1-L8 in place of A 1.1 bullet 3,
# A3 decided from the file's own activation quantization, and A9a/A9b/A10/A12 in place of A9.
# Nothing under `results/expA` is read for A2 except A's candidate list and A's pool.json (for the
# carried-over sha256 values), and nothing under `results/expA` is ever written by A2.
EXPERIMENTS = {
    "A": {"out_dir": OUT_DIR,
          "candidate_files": [CANDIDATES],
          "registrations": [REGISTRATION]},
    "A2": {"out_dir": A2_OUT_DIR,
           "candidate_files": [CANDIDATES, A2_CANDIDATES2, A2_CANDIDATES3],
           "registrations": [REGISTRATION, A2_REGISTRATION]},
}

# ---- registered constants ---------------------------------------------------------------------
SIZE_CAP = 60_000_000          # A7
POOL_CAP = 2_500_000_000       # section 1.2 / abort criterion 7
SCREEN_CAP_MS = 60.0           # A8
MIN_QDQ = 8                    # A3
MAX_PER_ARCH = 2               # A9
MAX_PER_PUBLISHER = 6          # A9
POOL_MIN, POOL_TARGET, POOL_MAX = 18, 24, 40      # section 1.3
MEASURED_MAX = 25              # section 1.3
F_CAP = 12                     # section 1.3
SCREEN_THREADS, SCREEN_WARMUP, SCREEN_RUNS = 4, 3, 20   # section 4.1
QUIET_PCT, QUIET_SAMPLES, QUIET_RETRIES = 15.0, 5, 10    # VA4

# ---- A2-only registered constants (a2_design_registered.md) ------------------------------------
# A2 2.3, "Family definition, fixed now": architecture family is the architecture, not its width or
# depth variant. A's candidate list labelled MobileViT twice; A2 corrects the label and records A's
# original beside it. Nothing else in either candidate list is a width/depth variant of something
# else already in the list: inception_v1/inception_v3 and mobilenet_v1/mobilenet_v2 are different
# architectures, not -tiny/-small/-base pairs of one, and yolox / squeezenet / rtdetr are already
# labelled with one name each across their two members.
# Round 3 adds four aliases, all of them MERGES that make A9a's cap of 2 bind harder, never looser,
# and all of them decided before any A2 checker output existed:
#   resnet18 -> resnet and resnet50 -> resnet: ResNet-18 and ResNet-50 are one architecture at two
#     depths, which 2.3 puts in ONE family ("likewise any -tiny/-small/-base pair of one
#     architecture"). Keeping them apart would have let the pool hold three ResNets.
#   efficientnet_es -> efficientnet and efficientnet_lite4 -> efficientnet: EfficientNet-EdgeTPU-S
#     and EfficientNet-Lite4 are two edge variants of EfficientNet, not two architectures. Merging
#     them costs the pool one architecture family in the count that section 4's 8-family minimum
#     uses, which is the conservative direction.
# The canonical name is the bare architecture, so neither of the two members' own labels is
# privileged; pool.json records `architecture_family_as_labelled_in_A` beside every relabelled row.
A2_FAMILY_ALIASES = {"mobilevit_xxs": "mobilevit",
                     "resnet18": "resnet", "resnet50": "resnet",
                     "efficientnet_es": "efficientnet", "efficientnet_lite4": "efficientnet"}
# A2 2.3 / A12: the positive control's family, and the two negative controls' families.
A2_HOME_FAMILIES = {"mobilevit"}
A2_NEGATIVE_CONTROL_FAMILIES = {"deit", "mobilenet_v2"}
# A2 2.3 A9a caps, unchanged from A9; A9b decides WHICH candidates take the slots.
A2_MAX_PER_ARCH, A2_MAX_PER_PUBLISHER = MAX_PER_ARCH, MAX_PER_PUBLISHER
# A3, decided from the file (see `quant_mode` below): A's registered count of >= 8 8-bit Q/DQ or
# QLinear* nodes is unchanged; A2 adds the gate that the ACTIVATIONS must be quantized statically.
A2_MIN_STATIC_ACT_SITES = 1


def a2_family(name: str | None) -> str | None:
    """The fixed family definition of A2 2.3, applied to a candidate list's own label."""
    if name is None:
        return None
    return A2_FAMILY_ALIASES.get(name, name)

# The screening/A6 child. Same shape as `ort_version_bisect.py`'s CHILD and `.envs/_bisect_child.py`
# (one float32 NCHW tensor, ORT_SEQUENTIAL, ORT_ENABLE_ALL, pinned), but with section 4.1's own
# iteration counts - 3 warmup then exactly 20 runs, median - which the bisect child does not have
# (it warms up for 1.5 s and then runs for >= 2 s and >= 40 iterations).
CHILD = textwrap.dedent('''
    import json, statistics, sys, time
    import numpy as np, psutil, onnxruntime as ort

    mp, batch, t, aff, shape, mode, warmup, runs = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
        [int(x) for x in sys.argv[5].split(",")], sys.argv[6], int(sys.argv[7]), int(sys.argv[8]))
    proc = psutil.Process()
    if aff:
        proc.cpu_affinity([int(c) for c in aff.split(",")])
    so = ort.SessionOptions()
    so.intra_op_num_threads = t
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    s = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((batch, *shape)).astype(np.float32)
    feed = {name: x}
    out = {"ort": ort.__version__, "input_name": name, "threads": t}
    if mode == "a6":
        r = s.run(None, feed)
        out["n_outputs"] = len(r)
        out["output_shapes"] = [list(np.asarray(o).shape) for o in r]
    else:
        for _ in range(warmup):
            s.run(None, feed)
        lat = []
        c0, t0 = proc.cpu_times(), time.perf_counter()
        for _ in range(runs):
            a = time.perf_counter()
            s.run(None, feed)
            lat.append((time.perf_counter() - a) * 1e3)
        c1 = proc.cpu_times()
        cpu_ms = ((c1.user - c0.user) + (c1.system - c0.system)) * 1e3
        p50 = float(statistics.median(lat))
        out.update({"screen_ms": p50, "n": len(lat), "min_ms": min(lat), "max_ms": max(lat),
                    "cpu_ms_per_inf": cpu_ms / len(lat),
                    "core_delivery": (cpu_ms / len(lat)) / (t * p50) if p50 else None})
    print(json.dumps(out))
''')


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# ================================================================================================
# stage 1: licences, from the publisher's own metadata (section 1.1)
# ================================================================================================
def _curl(url: str, out: Path | None = None, head: bool = False, timeout: int = 120):
    cmd = ["curl", "-sSL", "--max-time", str(timeout)]
    if head:
        cmd += ["-I"]
    if out is not None:
        cmd += ["-o", str(out), "-w", "%{http_code}"]
    cmd += [url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
    return r


def hf_repo(url: str) -> str | None:
    """https://huggingface.co/<owner>/<name>/resolve/<rev>/<path> -> "<owner>/<name>"."""
    if "huggingface.co/" not in url:
        return None
    rest = url.split("huggingface.co/", 1)[1]
    parts = rest.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else None


def zoo_readme(url: str) -> str | None:
    """github.com/onnx/models/raw/main/validated/<dir>/model[s]/<file> -> the model's README URL."""
    if "github.com/onnx/models/raw/" not in url:
        return None
    rest = url.split("/raw/", 1)[1]           # main/validated/<dir>/model/<file>
    parts = rest.split("/")
    if len(parts) < 4:
        return None
    return ("https://raw.githubusercontent.com/onnx/models/"
            + "/".join(parts[:-2]) + "/README.md")


def stage_licence(cands, work: Path) -> dict:
    """Every candidate's licence as the PUBLISHER states it, with the URL where it is stated.

    A stated licence in the approved candidate list is not enough on its own: section 1.1 wants the
    publisher's own statement. Hugging Face states it in the repo metadata (`cardData.license` or a
    `license:*` tag); the ONNX model zoo states it as an SPDX identifier at the top of the model's
    own README. A candidate with no stated licence is recorded as rejected and never downloaded.
    """
    raw = work / "licence_raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in cands:
        cid, url = c["id"], c["url"]
        rec = {"id": cid, "declared_in_candidate_list": c.get("licence"),
               "declared_src": c.get("licence_src")}
        repo = hf_repo(url)
        if repo:
            api = f"https://huggingface.co/api/models/{repo}"
            f = raw / f"hf_{repo.replace('/', '_')}.json"
            r = _curl(api, f)
            rec["metadata_url"] = api
            rec["http"] = r.stdout.strip()[-3:] if r.stdout else str(r.returncode)
            lic, how = None, None
            if f.exists() and rec["http"] == "200":
                d = load_json(f)
                card = d.get("cardData") or {}
                if card.get("license"):
                    lic, how = card["license"], "cardData.license"
                else:
                    tags = [t for t in d.get("tags", []) if t.startswith("license:")]
                    if tags:
                        lic, how = tags[0].split(":", 1)[1], "tags[license:*]"
                rec["tags"] = d.get("tags", [])
            rec["licence"], rec["stated_in"] = lic, how
        else:
            rd = zoo_readme(url)
            f = raw / f"zoo_{cid}.md"
            rec["metadata_url"] = rd
            lic, how = None, None
            if rd:
                r = _curl(rd, f)
                rec["http"] = r.stdout.strip()[-3:] if r.stdout else str(r.returncode)
                if f.exists() and rec["http"] == "200":
                    head = f.read_text(encoding="utf-8", errors="replace").splitlines()[:3]
                    for line in head:
                        if "SPDX-License-Identifier:" in line:
                            lic = line.split("SPDX-License-Identifier:", 1)[1]
                            lic = lic.replace("-->", "").strip()
                            how = "README SPDX-License-Identifier"
                            break
            rec["licence"], rec["stated_in"] = lic, how
        rec["verified"] = bool(rec["licence"])
        rec["matches_candidate_list"] = (
            bool(rec["licence"]) and c.get("licence") not in (None, "verify")
            and rec["licence"].lower() == str(c.get("licence")).lower())
        out[cid] = rec
        print(f"  {cid} {c['name'][:44]:46s} {str(rec['licence']):14s} "
              f"({rec['stated_in'] or 'NOT STATED'})")
    save_json(work / "licences.json", out)
    return out


# ================================================================================================
# stage 1 (A2): licence STATE, gating and provenance, from the publisher's own metadata
#               (a2_design_registered.md L1-L5, VA15, and the fields of 2.2)
# ================================================================================================
LICENCE_FILE_NAMES = ("license", "licence", "license.md", "licence.md", "license.txt",
                      "licence.txt", "copying", "copying.txt", "notice", "notice.md")


def _is_licence_file(name: str) -> bool:
    return name.lower().lstrip("./") in LICENCE_FILE_NAMES


def stage_licence_a2(cands, work: Path, prior: dict | None = None) -> dict:
    """L1's three states, recorded BEFORE any download (VA15), from what the publisher states.

    DECLARED       an identifier is stated in the publisher's own metadata AND the URL where it is
                   stated resolves today.
    SILENT         the licence metadata field is empty AND no licence file sits in the repository
                   root.
    INDETERMINATE  anything else - a licence file with no identifier, an identifier the metadata
                   and the card disagree about, or a page that does not resolve. Not admitted (L2).

    L4 is checked here too: a gated repository, or one that needs a click-through, is not admitted.
    The hub's own `gated` flag is read, and the anonymous HEAD of stage 2 is the second check.

    `prior` is a licences_a2.json from an earlier round of the SAME fetch round (--resume). A
    candidate already in it keeps the state recorded then, with its original `recorded_utc`:
    re-deriving it now would move that timestamp after the file's `fetched_utc` and make the row
    invalid under VA15. Only candidates that have never been through this stage are looked up.
    """
    raw = work / "licence_raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in cands:
        cid, url = c["id"], c["url"]
        if prior and cid in prior and prior[cid].get("licence_state"):
            rec = dict(prior[cid])
            rec["reused_from_prior_round"] = True
            out[cid] = rec
            print(f"  {cid} reused: licence state {rec['licence_state']} recorded "
                  f"{rec.get('recorded_utc')}")
            continue
        rec = {"id": cid, "recorded_utc": utc(),
               "declared_in_candidate_list": c.get("licence"),
               "declared_state_in_candidate_list": c.get("licence_state"),
               "declared_src": c.get("licence_src") or c.get("licence_url")}
        repo = hf_repo(url)
        if repo:
            api = f"https://huggingface.co/api/models/{repo}"
            f = raw / f"hf_{repo.replace('/', '_')}.json"
            r = _curl(api, f)
            rec["metadata_url"] = api
            rec["repository"] = repo
            rec["http"] = r.stdout.strip()[-3:] if r.stdout else str(r.returncode)
            if not (f.exists() and rec["http"] == "200"):
                rec.update({"licence_state": "INDETERMINATE", "licence": None,
                            "licence_url": None, "stated_in": None, "gated": None,
                            "why": f"publisher metadata did not resolve (HTTP {rec['http']})"})
                out[cid] = rec
                print(f"  {cid} INDETERMINATE: metadata HTTP {rec['http']}")
                continue
            d = load_json(f)
            card = d.get("cardData") or {}
            tags = [t for t in d.get("tags", []) if t.startswith("license:")]
            ident = card.get("license") or (tags[0].split(":", 1)[1] if tags else None)
            where = ("cardData.license" if card.get("license")
                     else "tags[license:*]" if tags else None)
            siblings = [s.get("rfilename", "") for s in d.get("siblings", [])]
            root = [s for s in siblings if "/" not in s]
            lic_files = [s for s in root if _is_licence_file(s)]
            rec["repo_revision"] = d.get("sha")
            rec["repo_last_modified"] = d.get("lastModified")
            rec["gated"] = d.get("gated")
            rec["repo_root_files"] = root
            rec["licence_files_in_root"] = lic_files
            rec["tags"] = d.get("tags", [])
            rec["base_model_named_by_repo"] = (
                c.get("base_model_named_by_repo")
                or (card.get("base_model") if isinstance(card.get("base_model"), str) else None)
                or (card.get("base_model")[0] if isinstance(card.get("base_model"), list)
                    and card.get("base_model") else None))
            rec["base_model_url"] = c.get("base_model_url") or (
                f"https://huggingface.co/{rec['base_model_named_by_repo']}"
                if rec["base_model_named_by_repo"] else None)
            rec["base_model_licence_stated"] = c.get("base_model_licence_stated")
            rec["export_tooling"] = c.get("export_tooling") or c.get("quantizer")
            rec["export_tooling_version"] = c.get("export_tooling_version")
            rec["quantization_mode_as_published"] = c.get("quantization_mode_as_published")
            if ident:
                rec.update({"licence_state": "DECLARED", "licence": ident, "stated_in": where,
                            "licence_url": (f"https://huggingface.co/{repo}/blob/main/{lic_files[0]}"
                                            if lic_files else f"https://huggingface.co/{repo}"),
                            "why": f"identifier '{ident}' stated in {where}"})
            elif not lic_files:
                rec.update({"licence_state": "SILENT", "licence": None, "stated_in": None,
                            "licence_url": None,
                            "why": "licence metadata field empty and no licence file in the "
                                   "repository root (L1 SILENT)"})
            else:
                rec.update({"licence_state": "INDETERMINATE", "licence": None, "stated_in": None,
                            "licence_url": f"https://huggingface.co/{repo}/blob/main/{lic_files[0]}",
                            "why": f"no identifier in the metadata but {lic_files} in the root: "
                                   f"the terms cannot be resolved to an identifier (L1)"})
        else:
            rd = zoo_readme(url)
            f = raw / f"zoo_{cid}.md"
            rec["metadata_url"] = rd
            rec["repository"] = "onnx/models"
            rec["gated"] = False
            rec["repo_revision"] = None
            rec["base_model_named_by_repo"] = c.get("base_model_named_by_repo")
            rec["base_model_url"] = c.get("base_model_url")
            rec["base_model_licence_stated"] = c.get("base_model_licence_stated")
            rec["export_tooling"] = c.get("export_tooling") or c.get("quantizer")
            rec["export_tooling_version"] = c.get("export_tooling_version")
            rec["quantization_mode_as_published"] = c.get("quantization_mode_as_published")
            ident = None
            if rd:
                r = _curl(rd, f)
                rec["http"] = r.stdout.strip()[-3:] if r.stdout else str(r.returncode)
                if f.exists() and rec["http"] == "200":
                    for line in f.read_text(encoding="utf-8", errors="replace").splitlines()[:3]:
                        if "SPDX-License-Identifier:" in line:
                            ident = line.split("SPDX-License-Identifier:", 1)[1]
                            ident = ident.replace("-->", "").strip()
                            break
            if ident:
                rec.update({"licence_state": "DECLARED", "licence": ident,
                            "stated_in": "README SPDX-License-Identifier", "licence_url": rd,
                            "why": f"SPDX identifier '{ident}' at the top of the model's README"})
            else:
                rec.update({"licence_state": "INDETERMINATE", "licence": None, "stated_in": None,
                            "licence_url": rd,
                            "why": "the model's README did not resolve or states no SPDX "
                                   "identifier"})
        out[cid] = rec
        print(f"  {cid} {c['name'][:42]:44s} {rec['licence_state']:14s} "
              f"{str(rec['licence']):12s} gated={rec.get('gated')}")
    save_json(work / "licences_a2.json", out)
    return out


# ================================================================================================
# stage 2: HEAD, download, sha256 (section 1.1)
# ================================================================================================
def head_size(url: str) -> tuple[int | None, str]:
    r = _curl(url, head=True)
    if r.returncode != 0:
        return None, r.stderr.strip()[-200:]
    size, linked = None, None
    for line in r.stdout.splitlines():
        low = line.lower()
        if low.startswith("content-length:"):
            try:
                size = int(low.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif low.startswith("x-linked-size:"):
            try:
                linked = int(low.split(":", 1)[1].strip())
            except ValueError:
                pass
    return (linked if linked is not None else size), ""


def download(url: str, dest: Path) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["curl", "-sSL", "--fail", "--max-time", "900", "-o", str(dest), url],
                       capture_output=True, text=True, timeout=1000)
    return r.returncode == 0, r.stderr.strip()[-200:]


def dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def stage_fetch(cands, lic, work: Path, pool_dir: Path) -> dict:
    """HEAD size, then download, then sha256 and byte count; one re-fetch on mismatch.

    A candidate with no stated licence is not downloaded at all (section 1.1), and neither is one
    whose HEAD size already breaches A7 (abort criterion 6: never "included provisionally").
    """
    staging = pool_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in cands:
        cid = c["id"]
        rec = {"id": cid, "url": c["url"], "head_bytes_candidate_list": c.get("head_bytes")}
        if not lic[cid]["licence"]:
            rec.update({"downloaded": False, "skip_reason": "no stated licence (section 1.1)"})
            out[cid] = rec
            print(f"  {cid} not downloaded: no stated licence")
            continue
        hb, err = head_size(c["url"])
        rec["head_bytes"] = hb
        rec["head_error"] = err or None
        # section 1.1 wants a URL "that resolved on the day of the fetch": this is that moment.
        rec["resolved_utc"] = utc()
        if hb is not None and hb > SIZE_CAP:
            rec.update({"downloaded": False,
                        "skip_reason": f"HEAD size {hb} > {SIZE_CAP} (A7, abort criterion 6)"})
            out[cid] = rec
            print(f"  {cid} not downloaded: HEAD {hb} > cap")
            continue
        if dir_bytes(pool_dir) + (hb or 0) > POOL_CAP:
            rec.update({"downloaded": False, "skip_reason": "pool cap 2.5 GB reached"})
            out[cid] = rec
            continue
        dest = staging / f"{cid}.onnx"
        attempts = []
        ok_final = False
        for attempt in (1, 2):
            ok, err = download(c["url"], dest)
            a = {"attempt": attempt, "curl_ok": ok, "error": err or None}
            if ok and dest.exists():
                a["bytes"] = dest.stat().st_size
                a["sha256"] = sha256(dest)
                a["head_match"] = (hb is None or a["bytes"] == hb)
            attempts.append(a)
            if ok and dest.exists() and a.get("head_match"):
                ok_final = True
                break
        rec["attempts"] = attempts
        last = attempts[-1]
        rec["downloaded"] = bool(last.get("curl_ok") and dest.exists())
        rec["bytes"] = last.get("bytes")
        rec["sha256"] = last.get("sha256")
        rec["head_match"] = last.get("head_match")
        rec["path"] = str(dest) if rec["downloaded"] else None
        if rec["downloaded"] and not ok_final:
            rec["skip_reason"] = (f"HEAD size {hb} != downloaded {rec['bytes']} after one re-fetch "
                                  f"(section 1.1: rejected)")
        print(f"  {cid} {'ok  ' if ok_final else 'FAIL'} {rec['bytes']} bytes "
              f"sha {str(rec['sha256'])[:12]} head_match={rec['head_match']}")
        out[cid] = rec
    save_json(work / "fetch.json", out)
    return out


def a_admitted_sha256() -> dict:
    """A's admitted rows, id -> sha256, from `results/expA/pool.json`.

    A2 2.1: "A's admitted six may be re-used unchanged if their sha256 values still match
    `results/expA/pool.json`". This function only READS A's frozen artefact; A2 never writes it.
    """
    p = OUT_DIR / "pool.json"
    if not p.exists():
        return {}
    d = load_json(p)
    return {c["id"]: c["sha256"] for c in d.get("candidates", [])
            if c.get("verdict") == "ADMITTED" and c.get("sha256")}


def find_in_pool(pool_dir: Path, cid: str) -> Path | None:
    """The file A's own layout stage left at <pool_dir>/<shape family>/<id>.onnx, if it is there."""
    for p in sorted(pool_dir.glob(f"*/{cid}.onnx")):
        return p
    return None


def stage_fetch_a2(cands, lic, work: Path, pool_dir: Path, prior: dict | None = None) -> dict:
    """L2/L4/L6 and abort criterion 6'. SILENT is fetched; INDETERMINATE and gated are not.

    A file already on disk from Experiment A is re-used without a second HTTP GET when its sha256
    still matches `results/expA/pool.json` (A2 2.1). That is also what L6 asks for - one GET per
    admitted URL - and it is the only thing carried over: `screen_ms` is re-measured for every A2
    row (A2 section 5).

    `prior` is a fetch_a2.json from an earlier round of the same fetch round (--resume). A prior
    record is re-used when it is a DECISION - a completed download (sha256 present) or a recorded
    skip_reason - so that L6's one GET per URL is not spent twice and the recorded sha256, byte
    count and rejection reason of an already-decided candidate cannot move. A prior record that is
    NOT a decision (the download failed with no skip_reason, e.g. b06) is re-tried, which is the
    only way its verdict can become a fact about the file rather than about the network.
    """
    carried = a_admitted_sha256()
    staging = pool_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    out = {}
    for c in cands:
        cid = c["id"]
        p = (prior or {}).get(cid)
        if p and (p.get("sha256") or p.get("skip_reason")):
            rec = dict(p)
            rec["reused_from_prior_round"] = True
            out[cid] = rec
            print(f"  {cid} reused: {'sha ' + str(rec.get('sha256'))[:12] if rec.get('sha256') else rec.get('skip_reason', '')[:70]}")
            continue
        if p:
            print(f"  {cid} prior fetch record is not a decision "
                  f"(downloaded={p.get('downloaded')}, no skip_reason): re-fetching")
        l_ = lic[cid]
        rec = {"id": cid, "url": c["url"], "head_bytes_candidate_list": c.get("head_bytes"),
               "licence_state": l_["licence_state"],
               "licence_state_recorded_utc": l_["recorded_utc"]}
        if l_["licence_state"] == "INDETERMINATE":
            rec.update({"downloaded": False, "carried_over_from_A": False,
                        "skip_reason": f"licence state INDETERMINATE (L2, abort 6'): {l_['why']}"})
            out[cid] = rec
            print(f"  {cid} not fetched: INDETERMINATE licence state")
            continue
        if l_.get("gated"):
            rec.update({"downloaded": False, "carried_over_from_A": False,
                        "skip_reason": f"gated repository (L4): gated={l_['gated']}"})
            out[cid] = rec
            print(f"  {cid} not fetched: gated repository")
            continue

        have = find_in_pool(pool_dir, cid)
        if have is not None and cid in carried:
            s = sha256(have)
            if s == carried[cid]:
                rec.update({"downloaded": False, "carried_over_from_A": True,
                            "path": str(have), "bytes": have.stat().st_size, "sha256": s,
                            "head_bytes": None, "head_match": True,
                            "fetched_utc": None, "resolved_utc": None,
                            "carried_over_note": "sha256 matches results/expA/pool.json; re-used "
                                                 "without a second HTTP GET (A2 2.1, L6)"})
                out[cid] = rec
                print(f"  {cid} carried over from A, sha {s[:12]} ({rec['bytes']} bytes)")
                continue
            rec["carried_over_sha256_mismatch"] = {"on_disk": s, "expA_pool": carried[cid]}

        hb, err = head_size(c["url"])
        rec["head_bytes"] = hb
        rec["head_error"] = err or None
        rec["resolved_utc"] = utc()
        rec["carried_over_from_A"] = False
        if hb is None:
            rec.update({"downloaded": False,
                        "skip_reason": f"HEAD did not return a size: {err}"})
            out[cid] = rec
            print(f"  {cid} not fetched: HEAD failed")
            continue
        if hb > SIZE_CAP:
            rec.update({"downloaded": False,
                        "skip_reason": f"HEAD size {hb} > {SIZE_CAP} (A7, abort criterion 6')"})
            out[cid] = rec
            print(f"  {cid} not downloaded: HEAD {hb} > cap")
            continue
        if dir_bytes(pool_dir) + hb > POOL_CAP:
            rec.update({"downloaded": False,
                        "skip_reason": "pool cap 2.5 GB reached (abort criterion 7)"})
            out[cid] = rec
            print(f"  {cid} not downloaded: pool cap")
            continue
        dest = staging / f"{cid}.onnx"
        attempts, ok_final = [], False
        for attempt in (1, 2):
            ok, err = download(c["url"], dest)
            a = {"attempt": attempt, "curl_ok": ok, "error": err or None, "utc": utc()}
            if ok and dest.exists():
                a["bytes"] = dest.stat().st_size
                a["sha256"] = sha256(dest)
                a["head_match"] = a["bytes"] == hb
            attempts.append(a)
            if ok and dest.exists() and a.get("head_match"):
                ok_final = True
                break
        rec["attempts"] = attempts
        last = attempts[-1]
        rec["downloaded"] = bool(last.get("curl_ok") and dest.exists())
        rec["bytes"] = last.get("bytes")
        rec["sha256"] = last.get("sha256")
        rec["head_match"] = last.get("head_match")
        rec["fetched_utc"] = last.get("utc")
        rec["path"] = str(dest) if rec["downloaded"] else None
        if rec["downloaded"] and not ok_final:
            rec["skip_reason"] = (f"HEAD size {hb} != downloaded {rec['bytes']} after one re-fetch "
                                  f"(L6 / A 1.1 bullet 4: rejected)")
        print(f"  {cid} {'ok  ' if ok_final else 'FAIL'} {rec['bytes']} bytes "
              f"sha {str(rec['sha256'])[:12]} head_match={rec['head_match']}")
        out[cid] = rec
    save_json(work / "fetch_a2.json", out)
    return out


# ================================================================================================
# stage 3: admission rules A1..A9 (section 1.2)
# ================================================================================================
EIGHT_BIT = {TensorProto.INT8, TensorProto.UINT8}
SUB_BYTE = {getattr(TensorProto, n) for n in ("INT4", "UINT4") if hasattr(TensorProto, n)}
QLINEAR_PREFIXES = ("QLinear",)
QOPERATOR_EXTRA = {"QGemm", "MatMulIntegerToFloat", "ConvInteger", "MatMulInteger"}
BANNED_OPS = {"NonMaxSuppression", "Loop", "If", "Scan"}


def walk(graph):
    for n in graph.node:
        yield n
        for attr in n.attribute:
            if attr.HasField("g"):
                yield from walk(attr.g)
            for g in attr.graphs:
                yield from walk(g)


# The QOperator ops that consume an already-8-bit activation tensor together with a scale and a
# zero point that are graph constants. Their presence IS static activation quantization.
QLINEAR_STATIC_OPS = {
    "QLinearConv", "QLinearMatMul", "QLinearAdd", "QLinearMul", "QLinearConcat",
    "QLinearAveragePool", "QLinearGlobalAveragePool", "QLinearSigmoid", "QLinearLeakyRelu",
    "QLinearSoftmax", "QLinearReduceMean", "QGemm",
}
INTEGER_OPS = {"MatMulInteger", "ConvInteger", "MatMulIntegerToFloat"}


def const_maps(graph):
    """Every tensor in the file whose value is fixed, and its element type.

    A quantized graph can carry its scales, zero points and weights either as graph INITIALIZERS or
    as the outputs of `Constant` NODES, and exporters differ: the ONNX Model Zoo files use
    initializers, AMD's Ryzen AI exports put all 300-odd scales and zero points in `Constant` nodes.
    Resolving only initializers reads a perfectly ordinary static QDQ file as having no 8-bit
    quantization at all, so both are resolved here.

    Returns (fixed_names, dtype_by_name) where `fixed_names` is every initializer and every
    `Constant` output, and `dtype_by_name` also covers declared value_info/input/output types.
    """
    dtype = {i.name: i.data_type for i in graph.initializer}
    fixed = set(dtype)
    for n in walk(graph):
        if n.op_type != "Constant":
            continue
        for attr in n.attribute:
            if attr.name in ("value", "sparse_value") and attr.HasField("t"):
                for o in n.output:
                    fixed.add(o)
                    dtype[o] = attr.t.data_type
            elif attr.name in ("value_int", "value_ints", "value_float", "value_floats",
                               "value_string", "value_strings"):
                for o in n.output:
                    fixed.add(o)
    for v in list(graph.value_info) + list(graph.input) + list(graph.output):
        dtype.setdefault(v.name, v.type.tensor_type.elem_type)
    return fixed, dtype


def quant_mode(model) -> dict:
    """Static or dynamic 8-bit quantization, DECIDED FROM THE FILE (A2, rule A3).

    A3's reason for existing is that "dynamic quantization keeps activations in float, so its
    Transposes are float by construction" (`a_design_registered.md:82`). What separates the two is
    therefore not the file's name, not the repository's model card and not a `quantize_config.json`
    entry, but one graph fact: are the activation quantization PARAMETERS constants in the file, or
    are they computed at run time from the activation itself?

    * computed at run time -> `DynamicQuantizeLinear`, or a `QuantizeLinear` whose `scale` input is
      not an initializer/`Constant`, or an integer op (`MatMulInteger`, `ConvInteger`) fed by one of
      those. The activations are floats until inference, so nothing the graph transposes is 8-bit
      by construction.
    * constants in the file -> a `QuantizeLinear` with a constant 8-bit scale/zero point applied to
      a tensor that is NOT an initializer (i.e. to an activation, not to a weight), or a `QLinear*`
      / `QGemm` op, which by definition consumes an 8-bit activation with constant parameters.

    Counting only the ACTIVATION sites matters: a weight-only file dequantizes int8 initializers and
    runs every activation in float, which is the same trivial negative A3 excludes.

    Returns the counts and a label; the caller applies the >= 8 threshold.
    """
    g = model.graph
    const, dtype = const_maps(g)
    producer = {o: n for n in walk(g) for o in n.output}

    def eight_bit_zp(node, idx: int) -> bool:
        """QuantizeLinear/DequantizeLinear default to uint8 when the zero point is omitted."""
        if len(node.input) <= idx or not node.input[idx]:
            return True
        return dtype.get(node.input[idx]) in EIGHT_BIT

    c = {"dynamic_quantize_linear": 0, "quantize_linear_runtime_scale": 0,
         "quantize_linear_static_activation": 0, "quantize_linear_static_weight": 0,
         "dequantize_linear_static": 0, "qlinear_ops": 0,
         "integer_ops_fed_by_dynamic": 0, "integer_ops_fed_by_static": 0}
    dyn_outputs = set()
    for n in walk(g):
        if n.op_type == "DynamicQuantizeLinear":
            c["dynamic_quantize_linear"] += 1
            dyn_outputs.update(n.output)
    for n in walk(g):
        if n.op_type == "QuantizeLinear":
            if not eight_bit_zp(n, 2):
                continue
            if len(n.input) > 1 and n.input[1] in const:
                if n.input[0] in const:
                    c["quantize_linear_static_weight"] += 1
                else:
                    c["quantize_linear_static_activation"] += 1
            else:
                c["quantize_linear_runtime_scale"] += 1
        elif n.op_type == "DequantizeLinear":
            if eight_bit_zp(n, 2) and len(n.input) > 1 and n.input[1] in const:
                c["dequantize_linear_static"] += 1
        elif n.op_type in QLINEAR_STATIC_OPS:
            c["qlinear_ops"] += 1
        elif n.op_type in INTEGER_OPS:
            fed = any(i in dyn_outputs or producer.get(i, None) is not None
                      and producer[i].op_type == "DynamicQuantizeLinear" for i in n.input)
            c["integer_ops_fed_by_dynamic" if fed else "integer_ops_fed_by_static"] += 1

    static_sites = (c["quantize_linear_static_activation"] + c["qlinear_ops"]
                    + c["integer_ops_fed_by_static"])
    dynamic_sites = (c["dynamic_quantize_linear"] + c["quantize_linear_runtime_scale"]
                     + c["integer_ops_fed_by_dynamic"])
    if static_sites and dynamic_sites:
        label = "mixed"
    elif static_sites:
        label = "static"
    elif dynamic_sites:
        label = "dynamic"
    else:
        label = "none"
    return {"counts": c, "static_activation_sites": static_sites,
            "dynamic_activation_sites": dynamic_sites, "activation_quantization": label}


def pin_batch(model):
    """A copy with the single input's batch dim pinned to 1 (section 2.1 step 1 / A5).

    When the batch dim is symbolic, the SAME symbol usually also labels the batch dim of the graph
    outputs ('batch_size' in `mobilenetv2-12-qdq`, 'N' in `resnet50-v1-12-qdq`). Pinning the input
    pins that symbol, so every occurrence of it is substituted too; otherwise A5's "every graph
    output shape is static once the input is pinned" would fail on the pinned batch dimension
    itself, which is not a data-dependent shape, and would fail it only for the models whose tail
    ONNX shape inference happens not to re-infer (it overwrites a declared symbolic dim when it can
    infer a value and leaves it alone when it cannot). Nothing else is substituted: a dim that is
    still symbolic or missing after this is a genuinely non-static shape and fails A5.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    inits = {i.name for i in m.graph.initializer}
    real = [i for i in m.graph.input if i.name not in inits]
    d = real[0].type.tensor_type.shape.dim[0]
    sym = d.dim_param if d.HasField("dim_param") else None
    d.ClearField("dim_param")
    d.dim_value = 1
    if sym:
        for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
            if not vi.type.tensor_type.HasField("shape"):
                continue
            for dim in vi.type.tensor_type.shape.dim:
                if dim.HasField("dim_param") and dim.dim_param == sym:
                    dim.ClearField("dim_param")
                    dim.dim_value = 1
    return m


def static_shape(vi) -> list | None:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value") and d.dim_value > 0:
            dims.append(int(d.dim_value))
        else:
            return None
    return dims


def check_static(path: Path, exp: str = "A"):
    """A1..A5, plus the facts the manifest needs. Uses the project venv's onnx 1.22.0 only.

    `exp="A2"` changes exactly one rule: A3 is decided from the file's own activation quantization
    (`quant_mode`), so a dynamically quantized file fails A3 however many 8-bit Q/DQ and integer
    nodes it carries. Under `exp="A"` the A3 verdict is A's own count, unchanged; the static/dynamic
    analysis is still recorded in `info`, where it changes no verdict.
    """
    info = {}
    try:
        model = onnx.load(str(path))
    except Exception as e:                       # A1
        return {"A1": (False, f"onnx.load failed: {type(e).__name__}: {e}")}, info
    g = model.graph
    inits = {i.name for i in g.initializer}
    real_inputs = [i for i in g.input if i.name not in inits]
    res = {}
    info["opset"] = next((op.version for op in model.opset_import if op.domain in ("", "ai.onnx")),
                         None)
    info["ir_version"] = model.ir_version
    info["n_graph_inputs"] = len(real_inputs)
    # Recorded for EVERY file that parses, whatever rule it goes on to fail, and never used as a
    # verdict outside A3: the task of A2 is to say, per file, whether the publisher's export really
    # is static 8-bit. A file rejected at A1 or A2 still has to answer that question in the record.
    if exp == "A2":
        try:
            info["quant_mode"] = quant_mode(model)
        except Exception as e:                   # REPORTED-ONLY: never a verdict
            info["quant_mode"] = {"error": f"{type(e).__name__}: {e}"}
    res["A1"] = (len(real_inputs) == 1, f"{len(real_inputs)} graph inputs (need exactly 1)")
    if not res["A1"][0]:
        return res, info

    vi = real_inputs[0]
    tt = vi.type.tensor_type
    dims = [(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
            for d in tt.shape.dim]
    info["input_name"] = vi.name
    info["input_elem_type"] = int(tt.elem_type)
    info["input_dims"] = dims
    ok = (tt.elem_type == TensorProto.FLOAT and len(dims) == 4
          and (dims[0] == 1 or isinstance(dims[0], str))
          and all(isinstance(d, int) and d > 0 for d in dims[1:]))
    res["A2"] = (ok, f"elem_type={tt.elem_type} dims={dims}")
    if ok:
        info["chw"] = [int(dims[1]), int(dims[2]), int(dims[3])]
        info["batch"] = "symbolic" if isinstance(dims[0], str) else 1
    if not ok:
        return res, info

    nodes = list(walk(g))
    _fixed, _dtype = const_maps(g)
    qdq8, qlinear, qdq_total = 0, 0, 0
    for n in nodes:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            qdq_total += 1
            # An omitted zero point defaults to uint8; a zero point carried by a `Constant` NODE
            # rather than an initializer is resolved by `const_maps`, because whole exporters
            # (AMD's Ryzen AI files) put every scale and zero point in `Constant` nodes.
            zp = n.input[2] if len(n.input) > 2 and n.input[2] else None
            t = _dtype.get(zp) if zp else TensorProto.UINT8
            if t in EIGHT_BIT:
                qdq8 += 1
        elif n.op_type.startswith(QLINEAR_PREFIXES) or n.op_type in QOPERATOR_EXTRA:
            qlinear += 1
    info.update({"n_nodes": len(nodes), "n_qdq_8bit": qdq8, "n_qdq_total": qdq_total,
                 "n_qlinear": qlinear,
                 "quantization_format": ("QDQ+QOperator" if qdq8 and qlinear else
                                         "QDQ" if qdq8 else "QOperator" if qlinear else "none")})
    qm = info.get("quant_mode") or quant_mode(model)
    info["quant_mode"] = qm
    if exp == "A2":
        # Two gates, both from the file. The first is A3's registered count, unchanged from A: at
        # least 8 8-bit Q/DQ nodes or QLinear*/QOperator ops. The second is what A3's own
        # justification always meant and what the file, not the repository, decides: the
        # ACTIVATIONS must be quantized statically. A dynamically quantized file keeps its
        # activations in float until inference, so its Transposes are float by construction; so
        # does a weight-only file, which dequantizes int8 initializers and never quantizes an
        # activation at all. Either is the trivial negative A3 exists to keep out.
        count_ok = qdq8 + qlinear >= MIN_QDQ
        static_ok = qm["static_activation_sites"] > 0
        res["A3"] = (count_ok and static_ok,
                     f"{qdq8} 8-bit Q/DQ + {qlinear} QLinear*/QOperator = {qdq8 + qlinear} "
                     f"(need >= {MIN_QDQ}): {'ok' if count_ok else 'FAILS'}; activation "
                     f"quantization measured from the graph = {qm['activation_quantization']} "
                     f"({qm['static_activation_sites']} static 8-bit activation sites, "
                     f"{qm['dynamic_activation_sites']} run-time sites): "
                     f"{'ok' if static_ok else 'FAILS - activations are not quantized statically'}"
                     f"; format {info['quantization_format']}; counts {json.dumps(qm['counts'])}")
    else:
        res["A3"] = (qdq8 + qlinear >= MIN_QDQ,
                     f"{qdq8} 8-bit Q/DQ + {qlinear} QLinear*/QOperator = {qdq8 + qlinear} "
                     f"(need >= {MIN_QDQ}; {qdq_total} Q/DQ nodes in total)")
    if not res["A3"][0]:
        return res, info

    bad4 = [i.name for i in g.initializer if i.data_type in SUB_BYTE]
    f16 = [i.name for i in g.initializer if i.data_type == TensorProto.FLOAT16]
    bad4 += [v.name for v in list(g.value_info) + list(g.input) + list(g.output)
             if v.type.tensor_type.elem_type in SUB_BYTE]
    info["n_float16_initializers"] = len(f16)
    res["A4"] = (not bad4 and not f16,
                 f"{len(bad4)} sub-byte tensors, {len(f16)} float16 initializers")
    if not res["A4"][0]:
        return res, info

    banned = sorted({n.op_type for n in nodes if n.op_type in BANNED_OPS})
    topk_runtime = [n.name for n in nodes
                    if n.op_type == "TopK" and len(n.input) > 1 and n.input[1] not in inits]
    outs, why = [], []
    try:
        inferred = onnx.shape_inference.infer_shapes(pin_batch(model), data_prop=True,
                                                     strict_mode=False)
        for o in inferred.graph.output:
            s = static_shape(o)
            outs.append({"name": o.name, "shape": s})
            if s is None:
                why.append(f"output {o.name} not static")
    except Exception as e:
        why.append(f"shape inference failed: {type(e).__name__}: {e}")
    info["output_shapes_inferred"] = outs
    res["A5"] = (not banned and not topk_runtime and not why,
                 f"banned ops {banned}; runtime-K TopK {topk_runtime}; " + ("; ".join(why) or "ok"))
    return res, info


def run_child(py: Path, child: Path, model: Path, chw, mode: str, threads: int):
    aff = ",".join(map(str, pick_cpus(threads, "auto") or []))
    r = subprocess.run([str(py), str(child), str(model), "1", str(threads), aff,
                        ",".join(map(str, chw)), mode, str(SCREEN_WARMUP), str(SCREEN_RUNS)],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return None, r.stderr.strip()[-300:]
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]), None
    except Exception as e:
        return None, f"unparseable child output: {e}: {r.stdout[-200:]}"


def machine_quiet() -> dict:
    """VA4. PerfOS _Total over 5 s, and AC power. Never Win32_Processor.LoadPercentage."""
    ps = (f"$s=@(); for($i=0;$i -lt {QUIET_SAMPLES};$i++)"
          "{$s += (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor"
          " -Filter \"Name='_Total'\").PercentProcessorTime; Start-Sleep -Seconds 1};"
          "$b=(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus;"
          "$o=@{samples=$s; battery_status=$b}; $o | ConvertTo-Json -Compress")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                       capture_output=True, text=True, timeout=180)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    s = [float(x) for x in (d["samples"] if isinstance(d["samples"], list) else [d["samples"]])]
    bs = d.get("battery_status")
    # BatteryStatus 1 = discharging (on battery); 2 = on AC; no battery instance = desktop = AC.
    on_ac = (bs is None) or (int(bs) != 1)
    return {"utc": utc(), "source": "Win32_PerfFormattedData_PerfOS_Processor _Total",
            "samples_pct": s, "mean_pct": sum(s) / len(s), "max_pct": max(s),
            "battery_status": bs, "on_ac": on_ac,
            "quiet": (sum(s) / len(s) <= QUIET_PCT) and on_ac}


def wait_quiet(log: list) -> dict:
    """Up to 10 one-minute checks (VA4). Never measures on a machine that is not quiet."""
    for attempt in range(1, QUIET_RETRIES + 1):
        q = machine_quiet()
        q["attempt"] = attempt
        log.append(q)
        if q["quiet"]:
            return q
        print(f"    machine not quiet (mean {q['mean_pct']:.1f}%, AC={q['on_ac']}); "
              f"waiting 60 s [{attempt}/{QUIET_RETRIES}]")
        time.sleep(60)
    raise SystemExit("machine never went quiet in 10 checks; nothing was measured (VA4)")


def stage_admit(cands, lic, fetch, work: Path) -> dict:
    """A1..A9 in order, stopping at the first failure, for every candidate."""
    child = work / "_a_pool_child.py"
    child.write_text(CHILD, encoding="utf-8")
    quiet_log: list = []
    admitted: list = []
    results = {}
    by_id = {c["id"]: c for c in cands}
    for c in cands:
        cid = c["id"]
        f = fetch.get(cid, {})
        rec = {"id": cid, "rules": {}, "info": {}}

        def fail(rule, why):
            rec["rules"][rule] = {"pass": False, "detail": why}
            rec["verdict"] = "REJECTED"
            rec["first_failing_rule"] = rule
            print(f"  {cid} REJECTED at {rule}: {why[:110]}")

        if not lic[cid]["licence"]:
            fail("section 1.1 (licence)",
                 f"no licence stated by the publisher at {lic[cid]['metadata_url']}")
            results[cid] = rec
            continue
        if not f.get("downloaded") or f.get("skip_reason"):
            fail("A7" if "HEAD size" in str(f.get("skip_reason")) else "fetch",
                 f.get("skip_reason") or "download failed")
            results[cid] = rec
            continue

        path = Path(f["path"])
        static, info = check_static(path)
        rec["info"] = info
        stop = False
        for rule in ("A1", "A2", "A3", "A4", "A5"):
            if rule not in static:
                continue
            ok, why = static[rule]
            rec["rules"][rule] = {"pass": ok, "detail": why}
            if not ok:
                fail(rule, why)
                stop = True
                break
        if stop:
            results[cid] = rec
            continue

        chw = info["chw"]
        a6 = {}
        ok6 = True
        for ver in ("1.23.2", "1.24.4"):
            j, err = run_child(PY[ver], child, path, chw, "a6", SCREEN_THREADS)
            a6[ver] = {"ok": j is not None, "error": err,
                       "reported_ort": (j or {}).get("ort"),
                       "output_shapes": (j or {}).get("output_shapes")}
            ok6 &= j is not None
        rec["rules"]["A6"] = {"pass": ok6, "detail": json.dumps(
            {v: (a6[v]["reported_ort"] if a6[v]["ok"] else a6[v]["error"]) for v in a6})}
        rec["a6"] = a6
        if not ok6:
            fail("A6", rec["rules"]["A6"]["detail"])
            results[cid] = rec
            continue

        nbytes = f["bytes"]
        rec["rules"]["A7"] = {"pass": nbytes <= SIZE_CAP, "detail": f"{nbytes} bytes"}
        if nbytes > SIZE_CAP:
            fail("A7", f"{nbytes} bytes > {SIZE_CAP}")
            results[cid] = rec
            continue

        print(f"  {cid} screening under 1.24.4 at {SCREEN_THREADS} threads "
              f"({SCREEN_WARMUP} warmup + {SCREEN_RUNS} runs)")
        q = wait_quiet(quiet_log)
        q["for_candidate"] = cid
        j, err = run_child(PY["1.24.4"], child, path, chw, "screen", SCREEN_THREADS)
        if j is None:
            fail("A8", f"screening session failed: {err}")
            results[cid] = rec
            continue
        rec["screen"] = j
        rec["screen"]["quiet_check"] = q
        screen_ms = j["screen_ms"]
        rec["rules"]["A8"] = {"pass": screen_ms <= SCREEN_CAP_MS,
                              "detail": f"screen_ms {screen_ms:.3f} (cap {SCREEN_CAP_MS})"}
        print(f"    screen_ms {screen_ms:.3f}  cd {j['core_delivery']:.2f}  "
              f"(machine {q['mean_pct']:.1f}%)")
        if screen_ms > SCREEN_CAP_MS:
            fail("A8", rec["rules"]["A8"]["detail"])
            results[cid] = rec
            continue

        dup = [a for a in admitted if results[a]["sha256"] == f["sha256"]]
        arch = sum(1 for a in admitted if by_id[a]["family"] == c["family"])
        pub = sum(1 for a in admitted if by_id[a]["publisher"] == c["publisher"])
        ok9 = (not dup) and arch < MAX_PER_ARCH and pub < MAX_PER_PUBLISHER
        detail = (f"duplicate sha256 of {dup}" if dup else
                  f"architecture family '{c['family']}' already has {arch} admitted (cap "
                  f"{MAX_PER_ARCH})" if arch >= MAX_PER_ARCH else
                  f"publisher '{c['publisher']}' already has {pub} admitted (cap "
                  f"{MAX_PER_PUBLISHER})" if pub >= MAX_PER_PUBLISHER else
                  f"unique sha256; arch {arch + 1}/{MAX_PER_ARCH}, publisher "
                  f"{pub + 1}/{MAX_PER_PUBLISHER}")
        rec["rules"]["A9"] = {"pass": ok9, "detail": detail}
        rec["sha256"] = f["sha256"]
        if not ok9:
            fail("A9", detail)
            results[cid] = rec
            continue
        rec["verdict"] = "ADMITTED"
        rec["first_failing_rule"] = None
        admitted.append(cid)
        print(f"  {cid} ADMITTED  screen_ms {screen_ms:.3f}  shape {chw}")
        results[cid] = rec
    out = {"results": results, "admitted": admitted, "quiet_machine_checks": quiet_log}
    save_json(work / "admit.json", out)
    return out


# ================================================================================================
# stage 3 (A2): A1..A8 per candidate, then A9a/A9b over the whole passing set, then A10 and A12
# ================================================================================================
# Directory names that carry no model identity, stripped when a file's path inside its repository
# has to stand in for a base checkpoint (see `base_checkpoint_key`).
GENERIC_DIRS = {"model", "models", "onnx", "weights", "int8", "quantized", "quantization", "."}


def base_checkpoint_key(c, lic_rec) -> str:
    """A10's "same base checkpoint": the checkpoint the repository names, or - when it names none -
    the model's own place inside its repository.

    A10's second case is "the same repository with two quantization VARIANTS"
    (`a2_design_registered.md:180`), which is what an HF re-export repository looks like:
    `onnx/model_quantized.onnx` and `onnx/model_uint8.onnx` are two quantizations of one checkpoint.
    Taking the bare repository as the key would be right there and wrong for the ONNX Model Zoo,
    which is one GitHub repository holding hundreds of unrelated models: it would make
    shufflenet-v2, mobilenet-v2, resnet50, efficientnet-lite4, emotion-ferplus and ultraface
    "the same base checkpoint" and collapse five of them into PAIR-REPLICATEs of the sixth. So the
    fallback key is the repository plus the file's own directory with generic directory names
    (`model/`, `models/`, `onnx/`, `weights/`) stripped - which leaves the model's own directory in
    the zoo, and the repository root for a one-model HF repository.
    """
    b = lic_rec.get("base_model_named_by_repo")
    if b:
        return f"base:{b}"
    repo = lic_rec.get("repository") or c.get("repository") or c["publisher"]
    url = c.get("url", "")
    path = c.get("file_path")
    if not path:
        for sep in ("/resolve/main/", "/raw/main/"):
            if sep in url:
                path = url.split(sep, 1)[1]
                break
    parts = (path or "").split("/")[:-1]
    while parts and parts[-1].lower() in GENERIC_DIRS:
        parts.pop()
    return f"repo:{repo}/{'/'.join(parts)}"


def apply_a9(passing, by_id, lic, sha_of) -> tuple[list, dict]:
    """A9a's caps under A9b's deterministic, family-first ordering.

    A9b: "sort that publisher's passing candidates by (number of that architecture family already
    admitted ascending, number of that base checkpoint already admitted ascending, sha256
    ascending) and admit down the list until the cap is reached". The sort key is a function of
    what is ALREADY admitted, so it is re-evaluated after every admission: the rule is a greedy
    pass, not one static sort, and only a greedy pass produces the behaviour A2 section 4 predicts
    (one file from each family first, then the second MobileViT CHECKPOINT rather than a second
    export of a checkpoint already in).

    DECISION THE REGISTRATION DOES NOT COVER, recorded here and in `pool.json`: A9b is written per
    publisher, but the family cap couples publishers (yolox is published by both AMD and Thefalley;
    squeezenet by both AMD and ketiswp), so a per-publisher pass would need an order over
    publishers that the registration does not give. The greedy is therefore run over the whole
    passing set on A9b's own key, which is publisher-blind, family-first and deterministic. Where
    no cap binds the two readings admit the same set; they can differ only inside a binding cap.
    """
    admitted, trace = [], {}
    fam_n, ckpt_n, pub_n = {}, {}, {}
    shas = set()
    remaining = list(passing)
    while remaining:
        elig = []
        for cid in remaining:
            fam = a2_family(by_id[cid]["family"])
            pub = by_id[cid]["publisher"]
            ck = base_checkpoint_key(by_id[cid], lic[cid])
            if sha_of[cid] in shas:
                continue
            if fam_n.get(fam, 0) >= A2_MAX_PER_ARCH or pub_n.get(pub, 0) >= A2_MAX_PER_PUBLISHER:
                continue
            elig.append((fam_n.get(fam, 0), ckpt_n.get(ck, 0), sha_of[cid], cid))
        if not elig:
            break
        elig.sort()
        _, _, _, cid = elig[0]
        fam = a2_family(by_id[cid]["family"])
        pub = by_id[cid]["publisher"]
        ck = base_checkpoint_key(by_id[cid], lic[cid])
        trace[cid] = {"pass": True, "order": len(admitted) + 1,
                      "detail": f"A9b slot {len(admitted) + 1}: family '{fam}' "
                                f"{fam_n.get(fam, 0) + 1}/{A2_MAX_PER_ARCH}, publisher '{pub}' "
                                f"{pub_n.get(pub, 0) + 1}/{A2_MAX_PER_PUBLISHER}, checkpoint "
                                f"'{ck}' {ckpt_n.get(ck, 0) + 1}, sha256 {sha_of[cid][:12]}"}
        admitted.append(cid)
        shas.add(sha_of[cid])
        fam_n[fam] = fam_n.get(fam, 0) + 1
        ckpt_n[ck] = ckpt_n.get(ck, 0) + 1
        pub_n[pub] = pub_n.get(pub, 0) + 1
        remaining.remove(cid)
    for cid in remaining:
        fam = a2_family(by_id[cid]["family"])
        pub = by_id[cid]["publisher"]
        dup = [a for a in admitted if sha_of[a] == sha_of[cid]]
        why = (f"duplicate sha256 of {dup} (A9a)" if dup else
               f"architecture family '{fam}' is at its cap of {A2_MAX_PER_ARCH} admitted "
               f"(A9a); A9b gave the slots to {[a for a in admitted if a2_family(by_id[a]['family']) == fam]}"
               if fam_n.get(fam, 0) >= A2_MAX_PER_ARCH else
               f"publisher '{pub}' is at its cap of {A2_MAX_PER_PUBLISHER} admitted (A9a); A9b "
               f"gave the slots to {[a for a in admitted if by_id[a]['publisher'] == pub]}")
        trace[cid] = {"pass": False, "order": None, "detail": why}
    return admitted, trace


def apply_a10(admitted, by_id, lic, sha_of) -> dict:
    """A10: two admitted files that quantize the same base checkpoint contribute ONE matrix row -
    the one with the lexicographically smallest sha256. The other is measured and reported as a
    PAIR-REPLICATE and counts in no rate, no stratum and no family count."""
    groups = {}
    for cid in admitted:
        groups.setdefault(base_checkpoint_key(by_id[cid], lic[cid]), []).append(cid)
    out = {}
    for key, ids in groups.items():
        ids = sorted(ids, key=lambda i: sha_of[i])
        for k, cid in enumerate(ids):
            out[cid] = {"base_checkpoint_key": key, "matrix_row": k == 0,
                        "pair_replicate": k > 0,
                        "pair_replicate_of": ids[0] if k else None,
                        "checkpoint_group": ids}
    return out


def stage_admit_a2(cands, lic, fetch, work: Path, prior: dict | None = None,
                   pool_dir: Path | None = None) -> dict:
    """Every candidate gets a verdict and its FIRST failing rule, in the registered order:
    L2/L4 -> fetch -> A1..A5 -> A6 -> A7 -> A8 -> A9a/A9b. A9 is decided after every other rule,
    over the whole passing set, because A9b's ordering is not first-come.

    `prior` is an admit_a2.json from an earlier round of the same fetch round (--resume). A
    candidate whose fetch record was re-used and whose per-candidate verdict is already recorded
    keeps its rules A1..A8 verbatim - the file of a rejected candidate has been deleted (A 1.2), so
    re-running those rules now would replace a reason that came from the file with "download
    failed", which would be a worse record, and re-screening an admitted one would spend a session
    to overwrite a screen_ms measured on the same machine on the same day.

    What is NEVER re-used is A9: it is recomputed from scratch over the union of the old and new
    passing sets, because A9a's caps and A9b's ordering are defined over the whole set and adding
    candidates can change which rows they admit. A prior row that was REJECTED AT A9 cannot be
    re-used at all - its file is gone and the new ordering might admit it - so it is refused loudly
    instead of being silently kept out.
    """
    child = work / "_a_pool_child.py"
    child.write_text(CHILD, encoding="utf-8")
    quiet_log: list = list((prior or {}).get("quiet_machine_checks", []))
    results, passing = {}, []
    by_id = {c["id"]: c for c in cands}
    for c in cands:
        cid = c["id"]
        f = fetch.get(cid, {})
        l_ = lic[cid]
        p = (prior or {}).get("results", {}).get(cid)
        if p and f.get("reused_from_prior_round") and p.get("verdict"):
            if p["verdict"] == "REJECTED" and p.get("first_failing_rule") == "A9":
                raise SystemExit(
                    f"{cid} was rejected at A9 in the prior round, so its file was deleted; A9 must "
                    f"be decided over the whole passing set and cannot be resumed for it. Re-run "
                    f"the fetch stage without --resume for this candidate.")
            rec = dict(p)
            rec["reused_from_prior_round"] = True
            if p["verdict"] == "ADMITTED":
                have = find_in_pool(pool_dir, cid) if pool_dir else None
                if have is None or sha256(have) != p.get("sha256"):
                    raise SystemExit(
                        f"{cid} was admitted in the prior round but its file is missing from "
                        f"{pool_dir} or its sha256 has changed; refusing to re-use the record.")
                rec["rules"] = {k: v for k, v in p.get("rules", {}).items() if k != "A9"}
                rec["verdict"], rec["first_failing_rule"] = None, None
                rec["path_verified_utc"] = utc()
                passing.append(cid)
                print(f"  {cid} reused: A1..A8 from the prior round, file verified on disk, "
                      f"A9 re-decided below")
            else:
                print(f"  {cid} reused: REJECTED at {p.get('first_failing_rule')}")
            results[cid] = rec
            continue
        rec = {"id": cid, "rules": {}, "info": {}}

        def fail(rule, why):
            rec["rules"][rule] = {"pass": False, "detail": why}
            rec["verdict"] = "REJECTED"
            rec["first_failing_rule"] = rule
            print(f"  {cid} REJECTED at {rule}: {why[:110]}")

        if l_["licence_state"] == "INDETERMINATE":
            fail("L2 (licence state INDETERMINATE)", l_["why"])
            results[cid] = rec
            continue
        if l_.get("gated"):
            fail("L4 (gated repository)", f"hub gated flag = {l_['gated']}")
            results[cid] = rec
            continue
        if not (f.get("path") and Path(f["path"]).exists()) or f.get("skip_reason"):
            reason = f.get("skip_reason") or "download failed"
            fail("A7" if "HEAD size" in str(reason) else "fetch", reason)
            results[cid] = rec
            continue

        path = Path(f["path"])
        static, info = check_static(path, exp="A2")
        rec["info"] = info
        stop = False
        for rule in ("A1", "A2", "A3", "A4", "A5"):
            if rule not in static:
                continue
            ok, why = static[rule]
            rec["rules"][rule] = {"pass": ok, "detail": why}
            if not ok:
                fail(rule, why)
                stop = True
                break
        if stop:
            results[cid] = rec
            continue

        chw = info["chw"]
        a6, ok6 = {}, True
        for ver in ("1.23.2", "1.24.4"):
            j, err = run_child(PY[ver], child, path, chw, "a6", SCREEN_THREADS)
            a6[ver] = {"ok": j is not None, "error": err,
                       "reported_ort": (j or {}).get("ort"),
                       "output_shapes": (j or {}).get("output_shapes")}
            ok6 &= j is not None
        rec["rules"]["A6"] = {"pass": ok6, "detail": json.dumps(
            {v: (a6[v]["reported_ort"] if a6[v]["ok"] else a6[v]["error"]) for v in a6})}
        rec["a6"] = a6
        if not ok6:
            fail("A6", rec["rules"]["A6"]["detail"])
            results[cid] = rec
            continue

        nbytes = f["bytes"]
        rec["rules"]["A7"] = {"pass": nbytes <= SIZE_CAP, "detail": f"{nbytes} bytes"}
        if nbytes > SIZE_CAP:
            fail("A7", f"{nbytes} bytes > {SIZE_CAP}")
            results[cid] = rec
            continue

        print(f"  {cid} screening under 1.24.4 at {SCREEN_THREADS} threads "
              f"({SCREEN_WARMUP} warmup + {SCREEN_RUNS} runs)")
        q = wait_quiet(quiet_log)
        q["for_candidate"] = cid
        j, err = run_child(PY["1.24.4"], child, path, chw, "screen", SCREEN_THREADS)
        if j is None:
            fail("A8", f"screening session failed: {err}")
            results[cid] = rec
            continue
        rec["screen"] = j
        rec["screen"]["quiet_check"] = q
        screen_ms = j["screen_ms"]
        rec["rules"]["A8"] = {"pass": screen_ms <= SCREEN_CAP_MS,
                              "detail": f"screen_ms {screen_ms:.3f} (cap {SCREEN_CAP_MS})"}
        print(f"    screen_ms {screen_ms:.3f}  cd {j['core_delivery']:.2f}  "
              f"(machine {q['mean_pct']:.1f}%)")
        if screen_ms > SCREEN_CAP_MS:
            fail("A8", rec["rules"]["A8"]["detail"])
            results[cid] = rec
            continue

        rec["sha256"] = f["sha256"]
        rec["verdict"] = None          # A9 decides, after every candidate has been through A1..A8
        passing.append(cid)
        results[cid] = rec

    sha_of = {cid: results[cid]["sha256"] for cid in passing}
    admitted, trace = apply_a9(passing, by_id, lic, sha_of)
    for cid in passing:
        t = trace[cid]
        results[cid]["rules"]["A9"] = {"pass": t["pass"], "detail": t["detail"]}
        results[cid]["a9b_order"] = t["order"]
        if t["pass"]:
            results[cid]["verdict"] = "ADMITTED"
            results[cid]["first_failing_rule"] = None
            print(f"  {cid} ADMITTED  ({t['detail']})")
        else:
            results[cid]["verdict"] = "REJECTED"
            results[cid]["first_failing_rule"] = "A9"
            print(f"  {cid} REJECTED at A9: {t['detail'][:110]}")
    a10 = apply_a10(admitted, by_id, lic, sha_of)
    for cid, v in a10.items():
        results[cid]["a10"] = v
    out = {"results": results, "admitted": admitted, "passing_a1_a8": passing,
           "a10": a10, "quiet_machine_checks": quiet_log}
    save_json(work / "admit_a2.json", out)
    return out


# ================================================================================================
# stage 4: selection, harness families and variant stems (section 1.3 and 4.2)
# ================================================================================================
def family_name(chw) -> str:
    return f"c{chw[0]}_{chw[1]}x{chw[2]}"


def stage_layout(cands, admit, fetch, pool_dir: Path):
    """Section 1.3 selection, then the layout `ort_version_bisect.py` resolves.

    The harness builds its model path as ROOT / <family dir> / <variant>.onnx, and on Windows an
    absolute `D:/...` family dir replaces ROOT in that join, so the pool can live off the project
    drive. Family = one distinct input shape, because the harness feeds one (c,h,w) per family.
    """
    by_id = {c["id"]: c for c in cands}
    admitted = admit["admitted"]
    note = None
    if len(admitted) <= MEASURED_MAX:
        measured = list(admitted)
        note = (f"all {len(admitted)} admitted models are measured: section 1.3's selection only "
                f"has to drop models when the admitted set exceeds {MEASURED_MAX} or |F| > {F_CAP},"
                f" and neither can apply to a set this size (|F| <= {len(admitted)} <= {F_CAP}).")
    else:
        raise SystemExit("more than 25 admitted: selection needs the checker's flag (artefact 3); "
                         "run a_check.py first and pass its strata in.")
    layout = []
    for cid in measured:
        c, r, f = by_id[cid], admit["results"][cid], fetch[cid]
        chw = r["info"]["chw"]
        fam = family_name(chw)
        fam_dir = pool_dir / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        dest = fam_dir / f"{cid}.onnx"
        src = Path(f["path"])
        if src.exists():
            shutil.move(str(src), str(dest))
        sha_after = sha256(dest)
        layout.append({
            "id": cid, "variant": cid, "family": fam,
            "family_dir": str(fam_dir).replace("\\", "/"),
            "path": str(dest).replace("\\", "/"),
            "sha256": sha_after, "sha256_matches_download": sha_after == f["sha256"],
            "bytes": dest.stat().st_size, "chw": chw,
        })
    return measured, layout, note


def stage_layout_a2(cands, admit, fetch, pool_dir: Path):
    """The layout `ort_version_bisect.py` resolves, for A2's admitted set.

    The harness builds its model path as `ROOT / <family dir> / <variant>.onnx`, and an absolute
    `D:/...` family dir replaces ROOT in that join, so a family is ONE DISTINCT INPUT SHAPE and a
    variant stem must be globally unique. The candidate ids (`a01`..`a17`, `b01`..`b19`) are unique
    across both candidate lists, so they are the variant stems; A's six carried-over files are
    already at exactly this path and are not moved.
    """
    by_id = {c["id"]: c for c in cands}
    admitted = admit["admitted"]
    if len(admitted) <= MEASURED_MAX:
        measured = list(admitted)
        note = (f"all {len(admitted)} admitted models are measured: section 1.3's selection only "
                f"has to drop models when the admitted set exceeds {MEASURED_MAX} or |F| > "
                f"{F_CAP}, and neither can apply to a set this size.")
        pending = False
    else:
        measured = list(admitted)
        note = (f"{len(admitted)} admitted exceeds the measured maximum of {MEASURED_MAX}, so "
                f"section 1.3's SELECTION binds: 'take every model in F, up to 12 ... then take "
                f"models from U in ascending sha256 order, preferring U-near'. That rule is a "
                f"function of the checker's flag (artefact 3), which by section 3 is written after "
                f"this file, so it cannot be applied here. Every admitted row is listed and "
                f"`selection_pending_checker` is true: the measured set is the output of "
                f"a_check.py's strata applied to this list, and nothing here pre-empts it.")
        pending = True
    layout = []
    for cid in measured:
        r, f = admit["results"][cid], fetch[cid]
        chw = r["info"]["chw"]
        fam = family_name(chw)
        fam_dir = pool_dir / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        dest = fam_dir / f"{cid}.onnx"
        src = Path(f["path"])
        if src.exists() and src.resolve() != dest.resolve():
            shutil.move(str(src), str(dest))
        sha_after = sha256(dest)
        layout.append({
            "id": cid, "variant": cid, "family": fam,
            "family_dir": str(fam_dir).replace("\\", "/"),
            "path": str(dest).replace("\\", "/"),
            "sha256": sha_after, "sha256_matches_download": sha_after == f["sha256"],
            "bytes": dest.stat().st_size, "chw": chw,
            "carried_over_from_A": bool(f.get("carried_over_from_A")),
        })
        _ = by_id[cid]
    return measured, layout, note, pending


def cleanup_rejected(fetch, admit, pool_dir: Path) -> list:
    """Section 1.2: a rejected file is deleted after its sha256, size and reason are recorded."""
    deleted = []
    staging = pool_dir / "_staging"
    for cid, f in fetch.items():
        if admit["results"].get(cid, {}).get("verdict") == "ADMITTED":
            continue
        p = Path(f["path"]) if f.get("path") else None
        if p and p.exists():
            p.unlink()
            deleted.append(cid)
    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    return deleted


def cleanup_rejected_a2(fetch, admit, pool_dir: Path) -> list:
    """A 1.2 / A2: a rejected file is deleted once its sha256, size and reason are recorded.

    A2 also has to delete rejected files that are NOT in staging: a candidate carried over from A
    already sits in its shape-family directory, and a candidate dropped by the A9b ordering was
    screened before the cap was applied.
    """
    deleted = []
    for cid, f in fetch.items():
        if admit["results"].get(cid, {}).get("verdict") == "ADMITTED":
            continue
        for p in [Path(f["path"]) if f.get("path") else None, find_in_pool(pool_dir, cid)]:
            if p is not None and p.exists():
                p.unlink()
                deleted.append(str(p).replace("\\", "/"))
    staging = pool_dir / "_staging"
    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    for d in sorted(pool_dir.glob("*")):
        if d.is_dir() and d.name != "_staging" and not any(d.iterdir()):
            d.rmdir()
    return deleted


# ================================================================================================
# stage 5 (A2): results/expA2/pool.json and results/expA2/manifest.json
# ================================================================================================
PUBLISHER_MERGE_COUNTERFACTUAL = {"onnx (Hugging Face organisation)": "onnx/models"}


def publisher_counterfactual(admit, by_id, lic) -> dict:
    """A9a caps "per publisher", and the registration does not define publisher identity.

    The ONNX Model Zoo files (`publisher: onnx/models`, a GitHub monorepo) and
    `onnx/Inception-1-int8` (`publisher: onnx (Hugging Face organisation)`) are two repositories of
    one organisation. This function re-runs A9 with the two names merged and records exactly what
    changes, so a reader can apply the other reading without re-running anything.
    """
    passing = admit.get("passing_a1_a8", [])
    sha_of = {cid: admit["results"][cid]["sha256"] for cid in passing}
    merged = {}
    for cid in passing:
        c = dict(by_id[cid])
        c["publisher"] = PUBLISHER_MERGE_COUNTERFACTUAL.get(c["publisher"], c["publisher"])
        merged[cid] = c
    alt, _ = apply_a9(passing, merged, lic, sha_of)
    base = set(admit["admitted"])
    return {
        "question": "are the GitHub organisation 'onnx/models' and the Hugging Face organisation "
                    "'onnx' one publisher for A9a's cap of 6?",
        "reading_applied": "TWO publishers - the `publisher` string each candidate list records. "
                           "A9b requires 'no human choice', and the recorded publisher is the only "
                           "mechanical datum; asserting that two repositories are one organisation "
                           "is a judgement the publishers' own metadata does not state.",
        "alternative_reading": "ONE publisher, on the ground that the HF 'onnx' organisation is "
                               "the ONNX organisation and b16's file name "
                               "(`inception-v1-12-int8.onnx`) is the Model Zoo's own.",
        "merge_applied_in_counterfactual": PUBLISHER_MERGE_COUNTERFACTUAL,
        "admitted_under_reading_applied": sorted(base),
        "admitted_under_alternative": sorted(alt),
        "would_lose": sorted(base - set(alt)),
        "would_gain": sorted(set(alt) - base),
        "changes_the_outcome": sorted(base) != sorted(alt),
    }


def stage_write_a2(cands, lic, fetch, admit, measured, layout, note, pending, pool_dir, sources):
    by_id = {c["id"]: c for c in cands}
    lay = {x["id"]: x for x in layout}
    a10 = admit.get("a10", {})

    approval = {
        "round_1_candidate_list": "results/expA/candidates.json",
        "round_1_approval": "the user approved A's list as a whole in chat on 2026-09-23 "
                            "('資料都可下載', 'A開始吧'), "
                            "recorded in that file's own note; A2 re-examines the same 17 "
                            "candidates under the amended rules",
        "round_2_candidate_list": "results/expA2/candidates_round2.json",
        "round_2_approval": "the user approved the round-2 candidate list in chat on 2026-09-23 "
                            "with '可以' ('yes / go ahead'), which is the approval "
                            "a2_design_registered.md section 2.1 requires before any download",
        "round_2_approval_utc_date": "2026-09-23",
        "round_3_candidate_list": "results/expA2/candidates_round3.json",
        "round_3_approval": "the user instructed this round in chat on 2026-09-23 in Traditional "
                            "Chinese: '先再抓靜態量化資料補到18個 / 再跑' ('first fetch more "
                            "statically quantized data to top the pool up to 18, then run'). That "
                            "instruction approves a rule - only statically quantized ONNX models, "
                            "up to the registered minimum of 18 - and leaves the individual URLs "
                            "to the collector, unlike round 2, whose explicit list the user "
                            "approved URL by URL. Every candidate the collector then chose is "
                            "listed with its reason in results/expA2/candidates_round3.json, and "
                            "every source considered and dropped is listed beside them.",
        "round_3_approval_utc_date": "2026-09-23",
        "round_3_admissible_because": "A2 2.1 / D13 forbids ADDING a candidate once the first A2 "
                                      "checker output exists. When candidates_round3.json was "
                                      "written, results/expA2 held pool.json, manifest.json, "
                                      "candidates_round2.json and stage_records/ and no "
                                      "predictions.json, selftest.json or "
                                      "results/paper/a2_predictions_frozen.md, so no A2 flag "
                                      "existed for any model and the round is inside the one "
                                      "registered fetch round.",
        "one_fetch_round": "A2 2.1 / D13: the widened list is approved once; after the first A2 "
                           "checker output exists no candidate may be added, and there is no "
                           "second fetch round",
    }

    pool = {
        "experiment": "A2", "arm": 1, "arm_name": "PUBLISHED-INT8",
        "artefact": 1, "artefact_name": "results/expA2/pool.json",
        "written_utc": utc(),
        "registration": "results/paper/a2_design_registered.md",
        "registration_sha256": sha256(A2_REGISTRATION),
        "base_registration": "results/paper/a_design_registered.md",
        "base_registration_sha256": sha256(REGISTRATION),
        "built_by": "scripts/paper/a_pool.py",
        "built_by_sha256": sha256(Path(__file__)),
        "candidate_lists": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                             "sha256": sha256(p)} for p in sources],
        "user_approval": approval,
        "pool_dir": str(pool_dir).replace("\\", "/"),
        "pool_dir_bytes": dir_bytes(pool_dir),
        "pool_dir_cap_bytes": POOL_CAP,
        "redistributed": False,
        "redistribution_note": "L3: no model file is redistributed by this project. What this "
                               "artefact publishes is the URL, the repository revision, the "
                               "sha256, the byte count, the publisher and the fetch date.",
        "conventions": {
            "A7_size_cap_bytes": SIZE_CAP,
            "screen_cap_ms": SCREEN_CAP_MS,
            "screen_protocol": f"1.24.4, {SCREEN_THREADS} threads, {SCREEN_WARMUP} warmup + "
                               f"{SCREEN_RUNS} runs, median (A 4.1); re-measured for every A2 row "
                               f"including the rows carried over from A (A2 section 5)",
            "A3_rule": "static 8-bit decided FROM THE FILE: >= 8 static 8-bit ACTIVATION "
                       "quantization sites, where a site is a QuantizeLinear with a constant 8-bit "
                       "scale applied to a non-initializer tensor, or a QLinear*/QGemm op. A "
                       "DynamicQuantizeLinear, a QuantizeLinear with a run-time scale, or an "
                       "integer op fed by one of those is a run-time site and counts as dynamic. "
                       "The file's name, its model card and its quantize_config.json decide "
                       "nothing.",
            "A9_caps": {"per_architecture_family": A2_MAX_PER_ARCH,
                        "per_publisher": A2_MAX_PER_PUBLISHER},
            "A9b_ordering": "greedy on (families already admitted, base checkpoints already "
                            "admitted, sha256), re-evaluated after each admission, run over the "
                            "whole passing set (see apply_a9 for the one decision the "
                            "registration does not cover)",
            "family_definition": "A2 2.3: architecture, not width or depth variant. Applied "
                                 "aliases: " + json.dumps(A2_FAMILY_ALIASES),
            "rule_order": "L2/L4 -> fetch -> A1..A5 -> A6 -> A7 -> A8 -> A9a/A9b; evaluation stops "
                          "at the first failure, so later rules are recorded as not evaluated. A9 "
                          "is decided after every candidate has been through A1..A8, because "
                          "A9b's ordering is not first-come.",
        },
        "decisions_not_covered_by_the_registration": {
            "publisher_identity": publisher_counterfactual(admit, by_id, lic),
            "A9b_scope": "A9b is written 'sort that publisher's passing candidates', but the "
                         "family cap couples publishers (squeezenet is published by both AMD and "
                         "ketiswp), so a per-publisher pass would need an order over publishers "
                         "that the registration does not give. The greedy is run over the whole "
                         "passing set on A9b's own key, which is publisher-blind, family-first "
                         "and deterministic.",
            "A10_fallback_key": "A10's second case is 'the same repository with two quantization "
                                "variants'. Taken literally, the bare repository would make every "
                                "ONNX Model Zoo file a variant of every other, since the zoo is "
                                "one GitHub repository holding hundreds of unrelated models. The "
                                "fallback key is therefore the repository plus the file's own "
                                "directory with generic directory names stripped; see "
                                "base_checkpoint_key.",
            "A3_static_test": "A3 says 'static 8-bit quantization' but gives its test as a node "
                              "count, which a dynamically quantized QOperator file passes. The "
                              "count is kept unchanged and a second gate is added from the file: "
                              "the activations must be quantized statically. Verified against "
                              "known ground truth - this project's own p00_int8.onnx reads "
                              "static, its p00_int8_dyn_mm.onnx sibling reads dynamic.",
            "A10_grouping_corrected_in_round_3": (
                "The round-2 pool.json reported 8 matrix rows and 5 PAIR-REPLICATEs because its "
                "A10 grouping came from an admit_a2.json written before base_checkpoint_key's "
                "generic-directory rule was in place: every ONNX Model Zoo file keyed to "
                "'repo:onnx/models', which collapsed shufflenet-v2, mobilenetv2, resnet50, "
                "efficientnet-lite4 and emotion-ferplus into PAIR-REPLICATEs of ultraface - "
                "exactly the failure base_checkpoint_key's docstring was written to prevent. The "
                "write stage reads A10 from the admit record, so re-running `write` alone had left "
                "the stale grouping in place. Round 3 re-runs the admit stage, so every key is "
                "recomputed by the current code: the six zoo files now key to their own model "
                "directories and are six matrix rows. No file, sha256, rule verdict or screen_ms "
                "changed; only the A10 grouping did, and no A2 checker output existed at any "
                "point in between."),
            "family_definition_applied": "Collapsed: `mobilevit_xxs` -> `mobilevit` (round 2); "
                                         "`resnet18`/`resnet50` -> `resnet` and "
                                         "`efficientnet_es`/`efficientnet_lite4` -> "
                                         "`efficientnet` (round 3, before any A2 checker output, "
                                         "and both merges make the cap of 2 bind harder). "
                                         "inception_v1/inception_v3 and mobilenet_v1/mobilenet_v2 "
                                         "were kept apart: they are different architectures, not "
                                         "-tiny/-small/-base variants of one.",
        },
        "candidates": [],
        "counts": {},
        "quiet_machine_checks": admit["quiet_machine_checks"],
        "not_downloaded_rejections_from_candidate_lists": [],
    }
    for c in cands:
        cid = c["id"]
        r, f, l_ = admit["results"][cid], fetch.get(cid, {}), lic[cid]
        url = c["url"]
        repo = l_.get("repository") or hf_repo(url) or "onnx/models"
        fam_fixed = a2_family(c["family"])
        g = a10.get(cid, {})
        pool["candidates"].append({
            "id": cid, "name": c["name"], "publisher": c["publisher"], "repository": repo,
            "file_in_repository": c.get("file_path") or (
                url.split("/resolve/main/", 1)[-1] if "/resolve/main/" in url
                else url.split("/raw/main/", 1)[-1]),
            "url": url,
            "url_resolved_utc": f.get("resolved_utc"),
            "fetched_utc": f.get("fetched_utc"),
            "licence_state": l_["licence_state"],
            "licence": l_["licence"], "licence_url": l_.get("licence_url"),
            "licence_stated_in": l_.get("stated_in"),
            "licence_metadata_url": l_.get("metadata_url"),
            "licence_state_recorded_utc": l_["recorded_utc"],
            "licence_state_recorded_before_fetch": (
                f.get("fetched_utc") is None or l_["recorded_utc"] <= f["fetched_utc"]),
            "licence_evidence": l_.get("why"),
            "licence_files_in_repo_root": l_.get("licence_files_in_root"),
            "gated": l_.get("gated"),
            "base_model_named_by_repo": l_.get("base_model_named_by_repo"),
            "base_model_url": l_.get("base_model_url"),
            "base_model_licence_stated": l_.get("base_model_licence_stated"),
            "export_tooling": l_.get("export_tooling"),
            "export_tooling_version": l_.get("export_tooling_version"),
            "quantization_mode_as_published": l_.get("quantization_mode_as_published"),
            "quantization_mode_measured": (r["info"].get("quant_mode") or {}).get(
                "activation_quantization"),
            "quantization_mode_measured_detail": r["info"].get("quant_mode"),
            "repo_revision": l_.get("repo_revision"),
            "repo_last_modified": l_.get("repo_last_modified"),
            "head_bytes": f.get("head_bytes"),
            "head_bytes_candidate_list": c.get("head_bytes"),
            "bytes": f.get("bytes"), "sha256": f.get("sha256"),
            "head_size_matched": f.get("head_match"),
            "downloaded": bool(f.get("downloaded")),
            "download_attempts": len(f.get("attempts", [])),
            "carried_over_from_A": bool(f.get("carried_over_from_A")),
            "architecture_family": fam_fixed,
            "architecture_family_as_labelled_in_A": c["family"],
            "architecture_family_relabelled": fam_fixed != c["family"],
            "architecture": c.get("arch"),
            "home_architecture": fam_fixed in A2_HOME_FAMILIES,
            "control_architecture": ("negative" if fam_fixed in A2_NEGATIVE_CONTROL_FAMILIES
                                     else None),
            "opset": r["info"].get("opset"),
            "input_shape": ([1] + r["info"]["chw"]) if r["info"].get("chw") else None,
            "quantization_format": r["info"].get("quantization_format"),
            "n_qdq_8bit": r["info"].get("n_qdq_8bit"),
            "n_qlinear": r["info"].get("n_qlinear"),
            "screen_ms": (r.get("screen") or {}).get("screen_ms"),
            "screen_core_delivery": (r.get("screen") or {}).get("core_delivery"),
            "verdict": r["verdict"], "first_failing_rule": r["first_failing_rule"],
            "a9b_order": r.get("a9b_order"),
            "base_checkpoint_key": g.get("base_checkpoint_key"),
            "matrix_row": g.get("matrix_row"),
            "pair_replicate": g.get("pair_replicate"),
            "pair_replicate_of": g.get("pair_replicate_of"),
            "rules": r["rules"],
            "file_deleted_after_rejection": r["verdict"] == "REJECTED" and bool(f.get("sha256")),
            "redistributed": False,
            "harness_family": lay.get(cid, {}).get("family"),
            "harness_variant": lay.get(cid, {}).get("variant"),
        })
    for src in sources:
        d = load_json(src)
        for k in d.get("known_rejections_no_download", []):
            k = dict(k)
            k["from_candidate_list"] = str(src.relative_to(ROOT)).replace("\\", "/")
            pool["not_downloaded_rejections_from_candidate_lists"].append(k)

    adm = [c for c in pool["candidates"] if c["verdict"] == "ADMITTED"]
    rej = [c for c in pool["candidates"] if c["verdict"] == "REJECTED"]
    mat = [c for c in adm if c["matrix_row"]]
    pool["counts"] = {
        "candidates": len(pool["candidates"]), "admitted": len(adm), "rejected": len(rej),
        "downloaded": sum(1 for c in pool["candidates"] if c["downloaded"]),
        "carried_over_from_A": sum(1 for c in adm if c["carried_over_from_A"]),
        "by_first_failing_rule": {r_: sum(1 for c in rej if c["first_failing_rule"] == r_)
                                  for r_ in sorted({str(c["first_failing_rule"]) for c in rej})},
        "by_licence_state": {s: sum(1 for c in pool["candidates"] if c["licence_state"] == s)
                             for s in ("DECLARED", "SILENT", "INDETERMINATE")},
        "admitted_by_licence_state": {s: sum(1 for c in adm if c["licence_state"] == s)
                                      for s in ("DECLARED", "SILENT", "INDETERMINATE")},
        "admitted_by_publisher": {p: sum(1 for c in adm if c["publisher"] == p)
                                  for p in sorted({c["publisher"] for c in adm})},
        "admitted_by_architecture_family": {
            fa: sum(1 for c in adm if c["architecture_family"] == fa)
            for fa in sorted({c["architecture_family"] for c in adm})},
        "admitted_by_export_tooling": {
            str(t): sum(1 for c in adm if str(c["export_tooling"]) == str(t))
            for t in sorted({str(c["export_tooling"]) for c in adm})},
        "admitted_by_measured_quantization_mode": {
            str(m): sum(1 for c in adm if str(c["quantization_mode_measured"]) == str(m))
            for m in sorted({str(c["quantization_mode_measured"]) for c in adm})},
        "matrix_rows": len(mat),
        "pair_replicates": len(adm) - len(mat),
        "home_architecture_admitted": sum(1 for c in adm if c["home_architecture"]),
        "matrix_row_architecture_families": len({c["architecture_family"] for c in mat}),
        "anonymous_re_exports_admitted_reported_only": sum(
            1 for c in adm if not c["base_model_named_by_repo"]),
        "pool_minimum_18_met": len(adm) >= POOL_MIN,
        "pool_target_24_met": len(adm) >= POOL_TARGET,
    }
    A2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(A2_OUT_DIR / "pool.json", pool)

    manifest = {
        "experiment": "A2", "arm": 1, "arm_name": "PUBLISHED-INT8",
        "artefact": 2, "artefact_name": "results/expA2/manifest.json",
        "written_utc": utc(),
        "registration": "results/paper/a2_design_registered.md",
        "registration_sha256": pool["registration_sha256"],
        "base_registration": "results/paper/a_design_registered.md",
        "base_registration_sha256": pool["base_registration_sha256"],
        "built_by": "scripts/paper/a_pool.py",
        "built_by_sha256": pool["built_by_sha256"],
        "pool_json_sha256": sha256(A2_OUT_DIR / "pool.json"),
        "pool_dir": pool["pool_dir"],
        "redistributed": False,
        "selection_rule": "A 1.3, unchanged by A2 section 4",
        "selection_note": note,
        "selection_pending_checker": pending,
        "stratum_note": "null: F / U / U-near are defined from the checker's flag, which is "
                        "artefact 3 (results/expA2/predictions.json, scripts/paper/a_check.py) and "
                        "by section 3 is written after this file.",
        "screen_protocol": pool["conventions"]["screen_protocol"],
        "screen_ms_is_not_a_result": "A 4.1: screening latency may never be quoted as a latency, a "
                                     "baseline or a speed-up.",
        "measured_set": [],
        "harness": {
            "script": "scripts/paper/ort_version_bisect.py",
            "families_arg": None, "variants_arg": None,
            "seed_primary": 63, "seed_rerun": 64, "seed_persistence": 65,
            "controls": {
                "mobilevit": "results/classification_mobilevit/models:3,64,64",
                "deit": "results/classification_deit/models:3,32,32",
                "cls": "results/classification/models:3,32,32"},
        },
        "counts": {},
        "quiet_machine_checks": admit["quiet_machine_checks"],
    }
    fams = {}
    for cid in measured:
        c, r, l_ = by_id[cid], admit["results"][cid], lic[cid]
        x, g = lay[cid], a10.get(cid, {})
        fams[x["family"]] = (x["family_dir"], x["chw"])
        fam_fixed = a2_family(c["family"])
        manifest["measured_set"].append({
            "id": cid, "name": c["name"], "publisher": c["publisher"],
            "repository": l_.get("repository"), "url": c["url"],
            "repo_revision": l_.get("repo_revision"),
            "licence_state": l_["licence_state"], "licence": l_["licence"],
            "licence_url": l_.get("licence_url"),
            "licence_metadata_url": l_.get("metadata_url"),
            "base_model_named_by_repo": l_.get("base_model_named_by_repo"),
            "export_tooling": l_.get("export_tooling"),
            "quantization_mode_as_published": l_.get("quantization_mode_as_published"),
            "quantization_mode_measured": (r["info"].get("quant_mode") or {}).get(
                "activation_quantization"),
            "sha256": x["sha256"], "bytes": x["bytes"],
            "stratum": None,
            "screen_ms": r["screen"]["screen_ms"],
            "screen_core_delivery": r["screen"]["core_delivery"],
            "input_shape": [1] + x["chw"],
            "opset": r["info"]["opset"],
            "quantization_format": r["info"]["quantization_format"],
            "architecture_family": fam_fixed,
            "architecture_family_as_labelled_in_A": c["family"],
            "architecture": c.get("arch"),
            "home_architecture": fam_fixed in A2_HOME_FAMILIES,
            "control_architecture": ("negative" if fam_fixed in A2_NEGATIVE_CONTROL_FAMILIES
                                     else None),
            "carried_over_from_A": bool(fetch[cid].get("carried_over_from_A")),
            "base_checkpoint_key": g.get("base_checkpoint_key"),
            "matrix_row": g.get("matrix_row"),
            "pair_replicate": g.get("pair_replicate"),
            "pair_replicate_of": g.get("pair_replicate_of"),
            "redistributed": False,
            "harness_family": x["family"], "harness_family_dir": x["family_dir"],
            "variant": x["variant"], "path": x["path"],
        })
    manifest["harness"]["families_arg"] = ";".join(
        [f"{k}={v[0]}:{v[1][0]},{v[1][1]},{v[1][2]}" for k, v in sorted(fams.items())]
        + [f"{k}={v}" for k, v in manifest["harness"]["controls"].items()])
    manifest["harness"]["variants_arg"] = ",".join(sorted(measured) + ["p00_int8"])
    mrows = [m for m in manifest["measured_set"] if m["matrix_row"]]
    manifest["counts"] = {
        "measured": len(measured),
        "matrix_rows": len(mrows),
        "pair_replicates": len(measured) - len(mrows),
        "architecture_families_matrix_rows": len({m["architecture_family"] for m in mrows}),
        "publishers": len({m["publisher"] for m in manifest["measured_set"]}),
        "export_tooling_values_A11": len({str(m["export_tooling"])
                                          for m in manifest["measured_set"]}),
        "harness_families": len(fams),
        "by_licence_state": {s: sum(1 for m in manifest["measured_set"]
                                    if m["licence_state"] == s) for s in ("DECLARED", "SILENT")},
        "home_architecture_rows": sum(1 for m in manifest["measured_set"]
                                      if m["home_architecture"]),
        "carried_over_from_A": sum(1 for m in manifest["measured_set"]
                                   if m["carried_over_from_A"]),
        "minimums_section_4": {"|F|>=5": None, "|U|>=8": None, "|U-near|>=4": None,
                               "architecture_families>=8":
                                   len({m["architecture_family"] for m in mrows}) >= 8},
    }
    save_json(A2_OUT_DIR / "manifest.json", manifest)
    return pool, manifest


# ================================================================================================
# stage 5: the two artefacts
# ================================================================================================
def stage_write(cands, lic, fetch, admit, measured, layout, note, pool_dir: Path, raw_cands):
    by_id = {c["id"]: c for c in cands}
    lay = {x["id"]: x for x in layout}

    pool = {
        "experiment": "A", "artefact": 1, "artefact_name": "results/expA/pool.json",
        "written_utc": utc(),
        "registration": "results/paper/a_design_registered.md",
        "registration_sha256": sha256(REGISTRATION),
        "built_by": "scripts/paper/a_pool.py",
        "built_by_sha256": sha256(Path(__file__)),
        "candidate_list": "results/expA/candidates.json",
        "candidate_list_sha256": sha256(CANDIDATES),
        "pool_dir": str(pool_dir).replace("\\", "/"),
        "pool_dir_bytes": dir_bytes(pool_dir),
        "pool_dir_cap_bytes": POOL_CAP,
        "conventions": {
            "A7_size_cap_bytes": SIZE_CAP,
            "A7_note": "decimal MB, the convention the candidate list already used (yolov3-12-int8,"
                       " 62,912,863 bytes, is rejected under A7 and is below 60 MiB)",
            "screen_cap_ms": SCREEN_CAP_MS,
            "screen_protocol": f"1.24.4, {SCREEN_THREADS} threads, {SCREEN_WARMUP} warmup + "
                               f"{SCREEN_RUNS} runs, median (section 4.1)",
            "A9_caps": {"per_architecture_family": MAX_PER_ARCH,
                        "per_publisher": MAX_PER_PUBLISHER},
            "rule_order": "A1..A9 in order; evaluation stops at the first failure, so later rules "
                          "are recorded as not evaluated",
        },
        "candidates": [],
        "counts": {},
        "quiet_machine_checks": admit["quiet_machine_checks"],
        "not_downloaded_rejections_from_candidate_list": raw_cands.get(
            "known_rejections_no_download", []),
    }
    for c in cands:
        cid = c["id"]
        r, f, l_ = admit["results"][cid], fetch.get(cid, {}), lic[cid]
        url = c["url"]
        repo = hf_repo(url) or "onnx/models"
        pool["candidates"].append({
            "id": cid, "name": c["name"], "publisher": c["publisher"],
            "repository": repo,
            "file_in_repository": url.split("/resolve/main/", 1)[-1] if "/resolve/main/" in url
                                  else url.split("/raw/main/", 1)[-1],
            "url": url, "url_resolved_utc": f.get("resolved_utc") or pool["written_utc"],
            "declared_architecture_family": c["family"], "architecture": c.get("arch"),
            "quantizer": c.get("quantizer"),
            "licence": l_["licence"], "licence_stated_in": l_["stated_in"],
            "licence_metadata_url": l_["metadata_url"],
            "licence_in_candidate_list": l_["declared_in_candidate_list"],
            "licence_verified": l_["verified"],
            "head_bytes": f.get("head_bytes"),
            "head_bytes_candidate_list": c.get("head_bytes"),
            "bytes": f.get("bytes"), "sha256": f.get("sha256"),
            "head_size_matched": f.get("head_match"),
            "downloaded": bool(f.get("downloaded")),
            "download_attempts": len(f.get("attempts", [])),
            "opset": r["info"].get("opset"),
            "input_shape": ([1] + r["info"]["chw"]) if r["info"].get("chw") else None,
            "quantization_format": r["info"].get("quantization_format"),
            "n_qdq_8bit": r["info"].get("n_qdq_8bit"),
            "n_qlinear": r["info"].get("n_qlinear"),
            "screen_ms": (r.get("screen") or {}).get("screen_ms"),
            "verdict": r["verdict"], "first_failing_rule": r["first_failing_rule"],
            "rules": r["rules"],
            "file_deleted_after_rejection": r["verdict"] == "REJECTED" and bool(f.get("path")),
            "harness_family": lay.get(cid, {}).get("family"),
            "harness_variant": lay.get(cid, {}).get("variant"),
        })
    adm = [c for c in pool["candidates"] if c["verdict"] == "ADMITTED"]
    rej = [c for c in pool["candidates"] if c["verdict"] == "REJECTED"]
    pool["counts"] = {
        "candidates": len(pool["candidates"]), "admitted": len(adm), "rejected": len(rej),
        "downloaded": sum(1 for c in pool["candidates"] if c["downloaded"]),
        "by_first_failing_rule": {r_: sum(1 for c in rej if c["first_failing_rule"] == r_)
                                  for r_ in sorted({c["first_failing_rule"] for c in rej})},
        "pool_minimum_18_met": len(adm) >= POOL_MIN,
        "pool_target_24_met": len(adm) >= POOL_TARGET,
    }
    save_json(OUT_DIR / "pool.json", pool)

    manifest = {
        "experiment": "A", "artefact": 2, "artefact_name": "results/expA/manifest.json",
        "written_utc": utc(),
        "registration": "results/paper/a_design_registered.md",
        "registration_sha256": pool["registration_sha256"],
        "built_by": "scripts/paper/a_pool.py",
        "built_by_sha256": pool["built_by_sha256"],
        "pool_json_sha256": sha256(OUT_DIR / "pool.json"),
        "pool_dir": pool["pool_dir"],
        "selection_rule": "section 1.3",
        "selection_note": note,
        "stratum_note": "null: the strata F / U / U-near are defined from the checker's flag, which"
                        " is artefact 3 (results/expA/predictions.json, scripts/paper/a_check.py)"
                        " and by section 3 is written after this file. Section 3 lists 'stratum' as"
                        " a field of artefact 2; it can only be filled in by the checker step, and"
                        " it is left null here rather than guessed.",
        "screen_protocol": pool["conventions"]["screen_protocol"],
        "screen_ms_is_not_a_result": "section 4.1: screening latency may never be quoted as a "
                                     "latency, a baseline or a speed-up.",
        "measured_set": [],
        "harness": {
            "script": "scripts/paper/ort_version_bisect.py",
            "families_arg": None, "variants_arg": None,
            "controls": {
                "mobilevit": "results/classification_mobilevit/models:3,64,64",
                "deit": "results/classification_deit/models:3,32,32",
                "cls": "results/classification/models:3,32,32"},
        },
        "counts": {},
        "quiet_machine_checks": admit["quiet_machine_checks"],
    }
    fams = {}
    for cid in measured:
        c, r, l_ = by_id[cid], admit["results"][cid], lic[cid]
        x = lay[cid]
        fams[x["family"]] = (x["family_dir"], x["chw"])
        manifest["measured_set"].append({
            "id": cid, "name": c["name"], "publisher": c["publisher"],
            "url": c["url"], "licence": l_["licence"], "licence_metadata_url": l_["metadata_url"],
            "sha256": x["sha256"], "bytes": x["bytes"],
            "stratum": None,
            "screen_ms": r["screen"]["screen_ms"],
            "screen_core_delivery": r["screen"]["core_delivery"],
            "input_shape": [1] + x["chw"],
            "opset": r["info"]["opset"],
            "quantization_format": r["info"]["quantization_format"],
            "architecture_family": c["family"],
            "architecture": c.get("arch"),
            "harness_family": x["family"], "harness_family_dir": x["family_dir"],
            "variant": x["variant"], "path": x["path"],
        })
    manifest["harness"]["families_arg"] = ";".join(
        [f"{k}={v[0]}:{v[1][0]},{v[1][1]},{v[1][2]}" for k, v in sorted(fams.items())]
        + [f"{k}={v}" for k, v in manifest["harness"]["controls"].items()])
    manifest["harness"]["variants_arg"] = ",".join(sorted(measured) + ["p00_int8"])
    manifest["counts"] = {
        "measured": len(measured),
        "architecture_families": len({by_id[c]["family"] for c in measured}),
        "publishers": len({by_id[c]["publisher"] for c in measured}),
        "harness_families": len(fams),
        "minimums_section_1_3": {"|F|>=5": None, "|U|>=8": None, "|U-near|>=4": None,
                                 "architecture_families>=8":
                                     len({by_id[c]["family"] for c in measured}) >= 8},
    }
    save_json(OUT_DIR / "manifest.json", manifest)
    return pool, manifest


# ================================================================================================
def a2_candidates(sources) -> list:
    """A's 17 candidates and round 2's 19, in one list, ids unchanged and checked for collisions.

    A's list is re-examined in full under A2's rules (A2 2.1: "The fetch round is re-run from the
    beginning under these rules"), including the nine it rejected for a missing licence.
    """
    cands, seen = [], set()
    for src in sources:
        for c in load_json(src)["candidates"]:
            if c["id"] in seen:
                raise SystemExit(f"candidate id {c['id']} appears in two lists; ids must be "
                                 f"globally unique because they are the harness variant stems")
            seen.add(c["id"])
            c = dict(c)
            c["_source_list"] = str(src.relative_to(ROOT)).replace("\\", "/")
            cands.append(c)
    return cands


def main_a2(a, work: Path, pool_dir: Path) -> int:
    sources = EXPERIMENTS["A2"]["candidate_files"]
    cands = a2_candidates(sources)
    prior_dir = Path(a.resume) if getattr(a, "resume", None) else None
    if prior_dir is not None and not prior_dir.is_dir():
        raise SystemExit(f"--resume {prior_dir} is not a directory")

    def prior_of(name: str):
        if prior_dir is None:
            return None
        p = prior_dir / name
        return load_json(p) if p.exists() else None

    lic = fetch = admit = None
    if a.stage in ("licence", "all"):
        print(f"[licence] {len(cands)} candidates, publisher metadata only (L1-L5, VA15)")
        lic = stage_licence_a2(cands, work, prior=prior_of("licences_a2.json"))
    if a.stage in ("fetch", "admit", "write", "all"):
        lic = lic or load_json(work / "licences_a2.json")
    if a.stage in ("fetch", "all"):
        print("[fetch] HEAD, download, sha256 (L6); A's admitted files re-used on a sha256 match")
        fetch = stage_fetch_a2(cands, lic, work, pool_dir, prior=prior_of("fetch_a2.json"))
    if a.stage in ("admit", "write", "all"):
        fetch = fetch or load_json(work / "fetch_a2.json")
    if a.stage in ("admit", "all"):
        print("[admit] A1..A8 per candidate, then A9a/A9b over the passing set, then A10/A12")
        admit = stage_admit_a2(cands, lic, fetch, work, prior=prior_of("admit_a2.json"),
                               pool_dir=pool_dir)
    if a.stage in ("write", "all"):
        admit = admit or load_json(work / "admit_a2.json")
        measured, layout, note, pending = stage_layout_a2(cands, admit, fetch, pool_dir)
        deleted = cleanup_rejected_a2(fetch, admit, pool_dir)
        pool, manifest = stage_write_a2(cands, lic, fetch, admit, measured, layout, note,
                                        pending, pool_dir, sources)
        print(f"\nadmitted {pool['counts']['admitted']} / rejected {pool['counts']['rejected']} "
              f"of {pool['counts']['candidates']}; deleted {len(deleted)} rejected files")
        print(f"matrix rows {pool['counts']['matrix_rows']} in "
              f"{pool['counts']['matrix_row_architecture_families']} families; "
              f"pair-replicates {pool['counts']['pair_replicates']}")
        print(f"pool dir {pool['pool_dir']} = {pool['pool_dir_bytes']} bytes (cap {POOL_CAP})")
        if pool["counts"]["admitted"] < POOL_MIN:
            print(f"\nABORT CRITERION 2 (A section 7, unchanged): "
                  f"{pool['counts']['admitted']} admitted < {POOL_MIN}. No measurement may be "
                  f"run; the scan alone is the result.")
        print(f"wrote {A2_OUT_DIR / 'pool.json'} and {A2_OUT_DIR / 'manifest.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["licence", "fetch", "admit", "write", "all"])
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--pool-dir", default="D:/sec_models")
    ap.add_argument("--resume", default=None,
                    help="A2 only: a directory holding licences_a2.json / fetch_a2.json / "
                         "admit_a2.json from an EARLIER round of the same fetch round (normally "
                         "results/expA2/stage_records). Candidates already decided there keep "
                         "their recorded licence state, sha256 and rules A1..A8 instead of being "
                         "re-fetched and re-screened; A9/A9b/A10 are always recomputed over the "
                         "whole passing set. See stage_admit_a2.")
    ap.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="A",
                    help="A: Experiment A exactly as registered. A2: the delta registered in "
                         "results/paper/a2_design_registered.md (two candidate lists, L1-L8, A3 "
                         "from the file, A9a/A9b/A10/A12). A2 never writes under results/expA.")
    a = ap.parse_args()
    work, pool_dir = Path(a.work_dir), Path(a.pool_dir)
    work.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)
    if a.experiment == "A2":
        return main_a2(a, work, pool_dir)
    raw = load_json(CANDIDATES)
    cands = raw["candidates"]

    lic = fetch = admit = None
    if a.stage in ("licence", "all"):
        print(f"[licence] {len(cands)} candidates, publisher metadata only")
        lic = stage_licence(cands, work)
    if a.stage in ("fetch", "admit", "write", "all"):
        lic = lic or load_json(work / "licences.json")
    if a.stage in ("fetch", "all"):
        print("[fetch] HEAD, download, sha256")
        fetch = stage_fetch(cands, lic, work, pool_dir)
    if a.stage in ("admit", "write", "all"):
        fetch = fetch or load_json(work / "fetch.json")
    if a.stage in ("admit", "all"):
        print("[admit] A1..A9")
        admit = stage_admit(cands, lic, fetch, work)
    if a.stage in ("write", "all"):
        admit = admit or load_json(work / "admit.json")
        measured, layout, note = stage_layout(cands, admit, fetch, pool_dir)
        deleted = cleanup_rejected(fetch, admit, pool_dir)
        pool, manifest = stage_write(cands, lic, fetch, admit, measured, layout, note, pool_dir,
                                     raw)
        print(f"\nadmitted {pool['counts']['admitted']} / rejected {pool['counts']['rejected']} "
              f"of {pool['counts']['candidates']}; deleted {len(deleted)} rejected files")
        print(f"pool dir {pool['pool_dir']} = {pool['pool_dir_bytes']} bytes "
              f"(cap {POOL_CAP})")
        if pool["counts"]["admitted"] < POOL_MIN:
            print(f"\nABORT CRITERION 2 (section 7): {pool['counts']['admitted']} admitted < "
                  f"{POOL_MIN}. No measurement may be run; the scan alone is the result.")
        print(f"wrote {OUT_DIR / 'pool.json'} and {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
