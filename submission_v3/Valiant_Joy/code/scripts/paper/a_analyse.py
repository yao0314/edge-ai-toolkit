"""Experiment A: apply the registered outcome rule to results/paper/a_thirdparty.csv.

The rule is fixed in `results/paper/a_design_registered.md` (REGISTERED 2026-09-23T01:35+08:00),
sections 4.4, 5 and 6, BEFORE any model was downloaded and before any timing. This script only
executes that rule. It contains no judgement of its own, no timing, and no threshold that is not
written in the registration.

    "C:/Users/User/Desktop/SEC/edge-ai-toolkit/.venv/Scripts/python.exe" scripts/paper/a_analyse.py results/paper/a_thirdparty.csv

Writes `results/expA/verdict.json` and prints the verdict and every number behind it.

What it reads
-------------
* the timing CSV(s) written by `scripts/paper/ort_version_bisect.py` (columns `env_dir, ort, family,
  variant, threads, rep, spin, batch, pinned_cpus, p50_ms, iters, cpu_ms_per_inf, core_delivery`);
  a model is the harness pair (family, variant), which is what registration section 4.2 makes unique;
* the frozen artefacts of registration section 3: `results/expA/predictions.json` (3),
  `results/expA/manifest.json` (2), `results/expA/pool.json` (1), `results/expA/selftest.json` (4),
  `results/expA/precondition.json` (7) and the seal in `results/paper/FINDINGS.md` (6).

A missing cell, a missing session, a NaN, an unhashed artefact or a model that is in the CSV but not
in the frozen manifest never passes a check silently: it makes the run INCONCLUSIVE.

Reader contract for the frozen JSON
-----------------------------------
`predictions.json` and `manifest.json` do not exist when this script is written (they are artefacts 2
and 3 and are produced by `a_check.py` after it), so this script fixes the fields it needs rather
than a byte layout. Either file may be `{...}`, `[...]`, or `{"models": {...}|[...]}`; a list entry
identifies itself with `id` / `model_id` / `variant`. Per model it needs

  manifest    : sha256, harness family (`harness_family`, else `family`), variant stem (`variant`),
                architecture family (`arch_family` / `architecture_family` / `architecture` /
                `arch`, or `family` when the harness family came from `harness_family`),
                stratum (`F` / `U` / `U-near`) and `screen_ms` (reported only);
  predictions : `sig_b`, `sig_a`, `added_dispatches` (a number, or `{sc_b, sc_a}`), `A_ms`, `rho`
                and the PRIMARY class (`predicted_class` / `prediction` / `primary_class` /
                `class`, the strings `PREDICT-STEP` / `PREDICT-SAFE`).

`a_check.py`'s own shape is read directly: a list under `models`, each record
`{"model", "path", "sha256", "SC-B": {...}, "SC-A": {...}, "prediction": {...}}`, where `sig_b`,
`rho`, `A_ms` and `predicted_class` sit inside `prediction` and `added_dispatches` means SC-B's
inside `SC-B` and SC-A's inside `SC-A`. A record is matched to its manifest entry by the manifest
key, the variant stem, the sha256 or the file stem, in that order.

The class is re-derived from the registered rule of section 2.3 (`sig_b and rho >= 0.07`) and a
stored class that disagrees with it is a gating failure, not something this script repairs.

Ambiguities in the registration and how they are resolved here, all reported in `verdict.json`
under `resolutions` so the choice is visible rather than buried:

  R1  Band order. Section 5.3's table is read top to bottom: CONFIRMED, then PARTIAL, then REFUTED,
      then MIXED ("anything that falls in none of the first three bands"). When PARTIAL wins while
      a REFUTED condition also holds (possible: H >= 0.60 with M >= 0.30), the verdict is PARTIAL
      and `refuted_condition_also_met` is set.
  R2  Precedence of INCONCLUSIVE over INCONCLUSIVE-SPARSE. Section 6 says a gating failure gives
      INCONCLUSIVE "never a verdict", and VA3 says that if the chain cannot reproduce the effect
      "nothing measured beside it can be read", so validity is tested first and sparsity second.
      Both reason lists are always filled in, so either reading can be recovered from the file.
  R3  The section 1.3 minimum sizes are counted over the models that actually enter the confusion
      matrix (section 5.2 builds the matrix "after the exclusions of section 6", and the Fisher
      worked examples in 5.2 are matrix counts). The frozen strata counts are reported beside them.
  R4  VA7 completeness is evaluated on the valid sessions of section 4.4 only, never on the sessions
      the core-delivery gate removes in the gated arm; otherwise one below-gate session, which VA5
      explicitly tolerates, would make the gated arm incomplete and VA6 could never pass.
  R5  VA3's "both negative controls do not step" is read strictly as NOT-STEPPED (ratio < 1.07). A
      negative control that comes out INDETERMINATE fails VA3 and is named in the reason.
  R6  A model that produced no usable row at all is FAILED-TO-RUN (VA8) and is counted inside VA7's
      "more than 3 dropped" test, since it is also short of its 5 valid sessions.
  R7  A cell holding MORE than 5 sessions is a gating failure, not a pooled median: VA5 allows a
      re-run only as a replacement (`--replace-cell`), never pooled with seed 53.
  R8  Section 5.4's cap at PARTIAL needs the FN class (i)/(ii) diagnostic, which by its own words
      runs after this verdict exists. The verdict here is therefore uncapped; `--fn-class-ii N`
      records the cap as `verdict_after_fn_class_ii_cap` without touching `verdict`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ---- registered constants (a_design_registered.md; nothing here is a choice of this script) ----
OLD, NEW = "1.23.2", "1.24.4"          # section 4.2
T_PRIMARY, T_CONTROL = 4, 1            # section 4.2
FLOOR_REL = 0.07                       # section 5.1, the study's between-session floor
STEP_RATIO = 1.0 + FLOOR_REL           # 1.07
GATE = 0.95                            # VA5
MIN_SESSIONS = 5                       # section 4.2 / VA7
MIN_ITERS = 40                         # section 4.4
MAX_BELOW_GATE_PER_CELL = 1            # VA5: at most 1 of a cell's 5 sessions
MAX_EXCLUDED = 3                       # VA5
MAX_DROPPED = 3                        # VA7
H_CONFIRM, H_PARTIAL, H_REFUTE = 0.80, 0.60, 0.40   # section 5.3
M_CONFIRM, M_REFUTE = 0.10, 0.30                    # section 5.3
P_SIG = 0.05                                        # section 5.3
MIN_F, MIN_U, MIN_UNEAR, MIN_FAMILIES = 5, 8, 4, 8  # section 1.3
MAX_INDET_SHARE = 0.20                              # section 5.3
DISPATCH_US = 0.52                                  # section 2.3 (Exp 27b's 0.51-0.53 us)
MAG_LO, MAG_HI = 0.5, 2.0                           # section 2.3, REPORTED ONLY
FN_CLASS_II_CAP = 3                                 # section 5.4

# section 1.3: the three local controls, never in the confusion matrix
CONTROLS = {("mobilevit", "p00_int8"): "positive",
            ("deit", "p00_int8"): "negative",
            ("cls", "p00_int8"): "negative"}

# VA2: what the checker must reproduce on its own worked example (section 2.1 / 2.2)
SELFTEST_FLAGGED = 4
SELFTEST_ADDED = 832

REQUIRED_COLUMNS = ("ort", "family", "variant", "threads", "p50_ms", "iters",
                    "cpu_ms_per_inf", "core_delivery")

ARTEFACTS = {"pool": ("results", "expA", "pool.json"),
             "manifest": ("results", "expA", "manifest.json"),
             "predictions": ("results", "expA", "predictions.json"),
             "selftest": ("results", "expA", "selftest.json")}

HARNESS_FAM_KEYS = ("harness_family", "harness_family_dir", "family_dir_name", "fam")
VARIANT_KEYS = ("variant", "variant_stem", "stem", "harness_variant")
ARCH_KEYS = ("arch_family", "architecture_family", "architecture", "arch")

TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|UTC|[+-]\d{2}:?\d{2})?")


# ---- small helpers ----------------------------------------------------------------------------
def sha256_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick(d: dict, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _models_of(obj, what: str) -> dict:
    """predictions.json / manifest.json as {"models": ...}, a bare mapping, or a list."""
    if isinstance(obj, dict):
        for k in ("models", "measured", "entries", "records"):
            if isinstance(obj.get(k), (dict, list)):
                obj = obj[k]
                break
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                out[str(k)] = v
    elif isinstance(obj, list):
        for v in obj:
            if not isinstance(v, dict):
                continue
            mid = _pick(v, "id", "model_id", "model", *VARIANT_KEYS) or _pick(v, "name", "sha256")
            if mid is None:
                raise SystemExit(f"{what}: a list entry has no id / variant / name to key it by")
            out[str(mid)] = v
    else:
        raise SystemExit(f"{what}: expected an object or a list of model records")
    return out


def _deep_has_value(obj, value) -> bool:
    """VA2: the registered reproduction numbers must actually appear in selftest.json."""
    if isinstance(obj, dict):
        return any(_deep_has_value(v, value) for v in obj.values())
    if isinstance(obj, list):
        return any(_deep_has_value(v, value) for v in obj)
    if isinstance(obj, bool):
        return False
    if isinstance(obj, (int, float)):
        return float(obj) == float(value)
    if isinstance(obj, str):
        return obj.strip() == str(value)
    return False


def _parse_ts(s: str):
    s = s.strip()
    s = s.replace(" ", "T", 1) if "T" not in s and " " in s else s
    if s.endswith("UTC"):
        s = s[:-3].strip() + "+00:00"
    elif s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if re.search(r"[+-]\d{4}$", s):
        s = s[:-5] + s[-5:-2] + ":" + s[-2:]
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def fisher_one_sided(a: int, b: int, c: int, d: int):
    """Section 5.2: one-sided exact p on [[flagged&stepped, flagged&not], [cleared&stepped,
    cleared&not]], from math.comb and the hypergeometric tail. No SciPy dependency."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0 or r1 == 0:
        return None
    denom = math.comb(n, r1)
    tail = sum(math.comb(c1, k) * math.comb(n - c1, r1 - k) for k in range(a, min(r1, c1) + 1))
    return tail / denom


def _pred_fields(p: dict) -> dict:
    """One prediction record, flat or in `a_check.py`'s shape.

    `a_check.py` writes `{"model", "path", "sha256", "SC-B": {...}, "SC-A": {...},
    "prediction": {...}}`, so `sig_b`, `rho`, `A_ms` and `predicted_class` live one level down and
    `added_dispatches` means SC-B's inside `SC-B` and SC-A's inside `SC-A`. A flat record, or
    `added_dispatches` as `{sc_b, sc_a}`, is read just as well. Nothing is guessed: a field that is
    in none of these places comes back None and the caller decides what that voids.
    """
    if not isinstance(p, dict):
        p = {}
    pr = p.get("prediction") if isinstance(p.get("prediction"), dict) else {}
    scb = p.get("SC-B") if isinstance(p.get("SC-B"), dict) else (p.get("sc_b") if isinstance(p.get("sc_b"), dict) else {})
    sca = p.get("SC-A") if isinstance(p.get("SC-A"), dict) else (p.get("sc_a") if isinstance(p.get("sc_a"), dict) else {})
    sec = pr.get("secondary_SC_A") if isinstance(pr.get("secondary_SC_A"), dict) else {}

    def first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    add = _pick(p, "added_dispatches", "added_dispatches_b", "added")
    add_flat_b = _pick(add, "sc_b", "b", "SC-B") if isinstance(add, dict) else add
    add_flat_a = _pick(add, "sc_a", "a", "SC-A") if isinstance(add, dict) else None
    cls = None
    for cand in (_pick(p, "predicted_class", "primary_class", "primary", "predict", "class",
                       "predicted", "prediction"),
                 _pick(pr, "predicted_class", "primary_class", "class", "prediction")):
        if isinstance(cand, str):
            up = cand.upper()
            cls = "PREDICT-SAFE" if "SAFE" in up else ("PREDICT-STEP" if "STEP" in up else None)
            if cls:
                break
    return {
        "sig_b": first(_pick(p, "sig_b", "sigB", "sig"), _pick(pr, "sig_b"),
                       _pick(scb, "sig_b", "flagged")),
        "sig_a": first(_pick(p, "sig_a", "sigA"), _pick(sca, "sig_a", "flagged")),
        "added_b": _num(first(add_flat_b, _pick(pr, "added_dispatches"),
                              _pick(scb, "added_dispatches"))),
        "added_a": _num(first(add_flat_a, _pick(p, "added_dispatches_a", "added_dispatches_sc_a"),
                              _pick(sec, "added_dispatches_a"), _pick(sca, "added_dispatches"))),
        "rho": _num(first(_pick(p, "rho"), _pick(pr, "rho"))),
        "A_ms": _num(first(_pick(p, "A_ms", "a_ms", "predicted_added_ms"), _pick(pr, "A_ms"))),
        "class": cls,
        "variant": _pick(p, *VARIANT_KEYS),
    }


# ---- the frozen artefacts ------------------------------------------------------------------
def load_frozen(root: Path, paths: dict) -> dict:
    """Artefacts 1-5 and 7 of section 3, plus the per-model prediction each model is scored against."""
    fr = {"paths": {k: str(v) for k, v in paths.items()}, "sha256": {}, "errors": [], "notes": []}
    for name in ("pool", "manifest", "predictions", "selftest"):
        fr["sha256"][name] = sha256_file(paths[name])
        if fr["sha256"][name] is None:
            fr["errors"].append(f"VA1: {name}.json is missing ({paths[name]})")

    man_raw = json.loads(paths["manifest"].read_text(encoding="utf-8")) if paths["manifest"].exists() else {}
    pred_raw = json.loads(paths["predictions"].read_text(encoding="utf-8")) if paths["predictions"].exists() else {}
    man = _models_of(man_raw, "manifest.json") if man_raw else {}
    pred = _models_of(pred_raw, "predictions.json") if pred_raw else {}

    # predictions may be keyed by an id, by the variant stem, by the sha256 or by a_check.py's
    # "<parent>/<stem>" label; index every one of them so the manifest's key finds its record
    p_index = {}
    for k, v in pred.items():
        for key in (k, _pick(v, "id", "model_id", "model", "name"), _pick(v, *VARIANT_KEYS),
                    _pick(v, "sha256", "sha", "hash")):
            if isinstance(key, str) and key:
                p_index.setdefault(key.strip().lower(), v)
        path = _pick(v, "path", "file", "model_path")
        if isinstance(path, str) and path:
            p_index.setdefault(Path(path.replace("\\", "/")).stem.lower(), v)

    models, mismatches, derived = {}, [], []
    for mid, m in man.items():
        p = pred.get(mid, {})
        if not p:
            for key in (mid, _pick(m, *VARIANT_KEYS), _pick(m, "sha256", "sha", "hash"),
                        _pick(m, "path", "file", "model_path")):
                if isinstance(key, str) and key.strip().lower() in p_index:
                    p = p_index[key.strip().lower()]
                    break
        p = _pred_fields(p)
        harness_fam = _pick(m, *HARNESS_FAM_KEYS)
        arch_fam = _pick(m, *ARCH_KEYS)
        if harness_fam is None:
            harness_fam = _pick(m, "family")          # then "family" is the harness dir name
        elif arch_fam is None:
            arch_fam = _pick(m, "family")             # an explicit harness key frees "family"
        variant = _pick(m, *VARIANT_KEYS) or p["variant"]
        if harness_fam is None or variant is None:
            fr["errors"].append(f"VA1: manifest model {mid} has no harness family / variant stem")
            continue

        sig_b, sig_a = p["sig_b"], p["sig_a"]
        add_b, add_a, rho = p["added_b"], p["added_a"], p["rho"]
        a_ms = p["A_ms"]
        if a_ms is None and add_b is not None:
            a_ms = add_b * DISPATCH_US / 1000.0
        cls = p["class"]
        rule_cls = ("PREDICT-STEP" if (bool(sig_b) and rho is not None and rho >= FLOOR_REL)
                    else "PREDICT-SAFE")
        if sig_b is None or (rho is None and cls is None):
            fr["errors"].append(f"VA1: predictions.json has no sig_b/rho for {mid}")
            continue
        if cls is None:
            cls, = (rule_cls,)
            derived.append(str(mid))
        elif rho is not None and cls != rule_cls:
            mismatches.append({"model": str(mid), "stored": cls, "registered_rule": rule_cls,
                               "sig_b": bool(sig_b), "rho": rho})

        stratum = str(_pick(m, "stratum", "stratum_class") or "").strip().upper().replace("_", "-")
        u_near = bool(_pick(m, "u_near", "U_near", "is_u_near")) or stratum in ("U-NEAR", "UNEAR")
        models[str(mid)] = {
            "id": str(mid), "family": str(harness_fam), "variant": str(variant),
            "arch_family": None if arch_fam is None else str(arch_fam),
            "sha256": _pick(m, "sha256", "sha", "hash"),
            "bytes": _pick(m, "bytes", "size_bytes", "nbytes"),
            "screen_ms": _num(_pick(m, "screen_ms")),
            "opset": _pick(m, "opset"), "quantization": _pick(m, "quantization", "quant_format"),
            "publisher": _pick(m, "publisher"),
            "stratum_frozen": stratum or None, "u_near": u_near,
            "sig_b": bool(sig_b), "sig_a": None if sig_a is None else bool(sig_a),
            "added_dispatches_b": add_b, "added_dispatches_a": add_a,
            "A_ms": a_ms, "rho": rho, "prediction": cls,
            "sig_small": bool(sig_b) and rho is not None and rho < FLOOR_REL,
        }
    fr["models"] = models
    fr["prediction_class_derived_from_rule"] = derived
    fr["prediction_rule_mismatch"] = mismatches
    if mismatches:
        fr["errors"].append(
            "VA1: the frozen PRIMARY class disagrees with the registered rule of section 2.3 for "
            + ", ".join(x["model"] for x in mismatches))
    fr["predictions_meta"] = {
        "ort_versions": _pick(pred_raw, "ort_versions", "versions") if isinstance(pred_raw, dict) else None,
        "a_check_sha256": _pick(pred_raw, "a_check_sha256", "checker_sha256", "a_check.py")
        if isinstance(pred_raw, dict) else None}

    # --- VA1, the seal: the digests in FINDINGS.md, timestamped before the first row ---------
    seal = {"findings": str(paths["findings"]), "digest_in_findings": {}, "seal_utc": None,
            "compared_against": None, "run_start_utc": None}
    txt = paths["findings"].read_text(encoding="utf-8", errors="replace") if paths["findings"].exists() else ""
    at = None
    for name, dg in fr["sha256"].items():
        if not dg:
            seal["digest_in_findings"][name] = False
            continue
        i = txt.find(dg)
        how = "full"
        if i < 0:
            i = txt.find(dg[:16])          # FINDINGS.md quotes prefixes, e.g. "911de4ee..."
            how = "prefix16"
        seal["digest_in_findings"][name] = how if i >= 0 else False
        if name == "predictions" and i >= 0:
            at = i
    if at is not None:
        before = [m.group(0) for m in TS_RE.finditer(txt[max(0, at - 6000):at])]
        after = [m.group(0) for m in TS_RE.finditer(txt[at:at + 3000])]
        raw = before[-1] if before else (after[0] if after else None)
        seal["seal_utc"] = None if raw is None else (
            _parse_ts(raw).astimezone(timezone.utc).isoformat() if _parse_ts(raw) else None)
    fr["seal"] = seal

    # --- VA4, the quiet-machine record (artefact 7) ------------------------------------------
    pre_p = paths["precondition"]
    pre = {"recorded": pre_p.exists(), "ok": False}
    if pre_p.exists():
        j = json.loads(pre_p.read_text(encoding="utf-8"))
        pre.update(j if isinstance(j, dict) else {"raw": j})
        pre["ok"] = bool(_pick(j, "ok", "passed", "quiet") if isinstance(j, dict) else False)
        ts = _pick(j, "started_utc", "timestamp_utc", "generated_utc", "utc", "timestamp") \
            if isinstance(j, dict) else None
        if isinstance(ts, str) and _parse_ts(ts):
            seal["run_start_utc"] = _parse_ts(ts).astimezone(timezone.utc).isoformat()
            seal["compared_against"] = "precondition.json timestamp"
    fr["precondition"] = pre

    # --- VA2, the checker self-test (artefact 4) ----------------------------------------------
    st = {"recorded": paths["selftest"].exists(), "ok": False,
          "reproduces_4_flagged": False, "reproduces_832_added": False}
    if paths["selftest"].exists():
        j = json.loads(paths["selftest"].read_text(encoding="utf-8"))
        res = _pick(j, "result", "verdict", "status") if isinstance(j, dict) else None
        nfail = _num(_pick(j, "n_fail", "n_failed", "failures")) if isinstance(j, dict) else None
        nchk = _num(_pick(j, "n_checks", "n_check", "checks_run")) if isinstance(j, dict) else None
        st["declared_ok"] = bool(
            (isinstance(j, dict) and bool(_pick(j, "ok", "passed", "all_ok", "reproduced")))
            or (isinstance(res, str) and res.strip().upper() in ("PASS", "PASSED", "OK"))
            or (nfail == 0 and (nchk or 0) > 0))
        checks = j.get("checks") if isinstance(j, dict) else None
        if isinstance(checks, list) and checks:
            # the artefact keeps one record per check, with `computed` and `expected`
            st["n_checks"] = len(checks)
            st["n_failed_checks"] = sum(
                1 for c in checks
                if isinstance(c, dict) and str(c.get("result", "")).upper().startswith("FAIL"))
            def _reproduced(val):
                return any(isinstance(c, dict) and c.get("computed") == val
                           and c.get("expected") == val
                           and not str(c.get("result", "PASS")).upper().startswith("FAIL")
                           for c in checks)
            st["reproduces_4_flagged"] = _reproduced(SELFTEST_FLAGGED)
            st["reproduces_832_added"] = _reproduced(SELFTEST_ADDED)
        else:
            st["n_failed_checks"] = 0 if st["declared_ok"] else 1
            st["reproduces_4_flagged"] = _deep_has_value(j, SELFTEST_FLAGGED)
            st["reproduces_832_added"] = _deep_has_value(j, SELFTEST_ADDED)
        st["ok"] = bool(st["declared_ok"] and st["n_failed_checks"] == 0
                        and st["reproduces_4_flagged"] and st["reproduces_832_added"])
    fr["selftest"] = st
    return fr


# ---- the sessions and the cells --------------------------------------------------------------
def load_sessions(csv_paths, replace=()):
    frames, files = [], []
    for p in csv_paths:
        d = pd.read_csv(p)
        d["_src"] = str(p)
        files.append({"csv": str(p), "rows": int(len(d))})
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if miss:
        raise SystemExit(f"{csv_paths[0]} is not in ort_version_bisect.py format: missing {miss}")

    # a re-run cell REPLACES, never pools (VA3 / VA5, --seed 54 in its own CSV)
    replaced = []
    if replace:
        last = str(csv_paths[-1])
        for spec in replace:
            fam, var, rel, thr = [x.strip() for x in spec.split(",")]
            mask = ((df["family"] == fam) & (df["variant"] == var) & (df["ort"] == rel)
                    & (df["threads"].astype(int) == int(thr)) & (df["_src"] != last))
            replaced.append({"cell": f"{fam}|{var}|{rel}|{thr}", "dropped_rows": int(mask.sum()),
                             "kept_from": last})
            df = df[~mask]

    foreign = {}
    n0 = len(df)
    df = df[df["ort"].astype(str).isin([OLD, NEW])].copy()
    foreign["other_release"] = n0 - len(df)
    n0 = len(df)
    df["threads"] = df["threads"].astype(int)
    df = df[df["threads"].isin([T_CONTROL, T_PRIMARY])]
    foreign["other_thread_count"] = n0 - len(df)
    if "spin" in df.columns:
        n0 = len(df)
        df = df[df["spin"].astype(str).isin(["default", "", "nan"])]
        foreign["non_default_spin"] = n0 - len(df)
    if "batch" in df.columns:
        n0 = len(df)
        df = df[df["batch"].astype(float) == 1.0]
        foreign["other_batch"] = n0 - len(df)

    # section 4.4: what makes a session invalid. No outlier rule, by registration.
    bad = {}
    df["_p50"] = pd.to_numeric(df["p50_ms"], errors="coerce")
    df["_cd"] = pd.to_numeric(df["core_delivery"], errors="coerce")
    df["_it"] = pd.to_numeric(df["iters"], errors="coerce")
    bad["p50_nan"] = int(df["_p50"].isna().sum())
    bad["core_delivery_nan"] = int(df["_cd"].isna().sum())
    bad["iters_below_%d" % MIN_ITERS] = int((df["_it"].fillna(-1) < MIN_ITERS).sum())
    ok = df[(~df["_p50"].isna()) & (~df["_cd"].isna()) & (df["_it"] >= MIN_ITERS)].copy()
    return ok, {"files": files, "replaced_cells": replaced, "foreign_rows": foreign,
                "invalid_sessions": bad, "valid_rows": int(len(ok)), "raw_rows": int(len(df))}


def cell_stats(g, gate=GATE):
    v = sorted(float(x) for x in g["_p50"])
    cd = [float(x) for x in g["_cd"]]
    return {"n": len(v), "median": float(pd.Series(v).median()), "min": v[0], "max": v[-1],
            "p50s": v, "cd_min": min(cd), "cd_max": max(cd),
            "below_gate": int(sum(1 for x in cd if x < gate))}


def model_cells(rows, gate=GATE, gated=False):
    out = {}
    for (rel, thr), g in rows.groupby(["ort", "threads"]):
        if gated and int(thr) == T_PRIMARY:
            g = g[g["_cd"] >= gate]
        if len(g) == 0:
            continue
        out[(str(rel), int(thr))] = cell_stats(g, gate)
    return out


def status_of(cells):
    """Section 5.1, in exactly the registration's words."""
    o4, n4 = cells.get((OLD, T_PRIMARY)), cells.get((NEW, T_PRIMARY))
    o1, n1 = cells.get((OLD, T_CONTROL)), cells.get((NEW, T_CONTROL))
    r = {"ratio_4t": None, "disjoint": None, "status": None, "ratio_1t": None, "pure": None,
         "delta_ms_4t": None}
    if o4 and n4 and o4["median"] > 0:
        r["ratio_4t"] = n4["median"] / o4["median"]
        r["delta_ms_4t"] = n4["median"] - o4["median"]
        r["disjoint"] = bool(o4["max"] < n4["min"])
        if r["ratio_4t"] >= STEP_RATIO and r["disjoint"]:
            r["status"] = "STEPPED"
        elif r["ratio_4t"] < STEP_RATIO:
            r["status"] = "NOT-STEPPED"
        else:
            r["status"] = "INDETERMINATE"
    if o1 and n1 and o1["median"] > 0:
        r["ratio_1t"] = n1["median"] / o1["median"]
        r["pure"] = bool(abs(r["ratio_1t"] - 1.0) < FLOOR_REL)
    return r


def cell_of(prediction, status, pure):
    if status == "INDETERMINATE" or status is None:
        return None
    if prediction == "PREDICT-STEP":
        return "FP" if status == "NOT-STEPPED" else ("TP" if pure else "IMPURE")
    return "TN" if status == "NOT-STEPPED" else ("FN" if pure else "FN-IMPURE")


def band_of(counts, sizes_ok):
    """Section 5.3, read top to bottom (resolution R1). Validity is applied by the caller."""
    TP, IMP, FP = counts["TP"], counts["IMPURE"], counts["FP"]
    FN, FNI, TN = counts["FN"], counts["FN-IMPURE"], counts["TN"]
    flagged, cleared = TP + IMP + FP, FN + FNI + TN
    H = None if flagged == 0 else (TP) / flagged
    M = None if cleared == 0 else (FN + FNI) / cleared
    p = fisher_one_sided(TP + IMP, FP, FN + FNI, TN)
    refuted = (H is not None and H <= H_REFUTE) or (M is not None and M >= M_REFUTE)
    if not sizes_ok:
        band = "INCONCLUSIVE-SPARSE"
    elif (H is not None and H >= H_CONFIRM and M is not None and M <= M_CONFIRM
          and p is not None and p < P_SIG and IMP <= TP):
        band = "CONFIRMED"
    elif H is not None and H >= H_PARTIAL and p is not None and p < P_SIG:
        band = "PARTIAL"
    elif refuted:
        band = "REFUTED"
    else:
        band = "MIXED"
    return {"band": band, "H": H, "M": M, "fisher_p": p, "flagged_scored": flagged,
            "cleared_scored": cleared, "refuted_condition_also_met": bool(refuted),
            "counts": dict(counts)}


# ---- the analysis ------------------------------------------------------------------------------
def analyse(csv_paths, root=ROOT, gate=GATE, out_path=None, replace=(), fn_class_ii=None):
    paths = {k: root.joinpath(*v) for k, v in ARTEFACTS.items()}
    paths["precondition"] = root / "results" / "expA" / "precondition.json"
    paths["findings"] = root / "results" / "paper" / "FINDINGS.md"
    fr = load_frozen(root, paths)
    df, prov = load_sessions(csv_paths, replace=replace)

    rep = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "script": str(Path(__file__).resolve()),
           "script_sha256": sha256_file(Path(__file__).resolve()),
           "registration": {"path": str(root / "results" / "paper" / "a_design_registered.md"),
                            "sha256": sha256_file(root / "results" / "paper" / "a_design_registered.md")},
           "csv": [str(p) for p in csv_paths], "provenance": prov, "gate": gate,
           "registered": {"STEP_RATIO": STEP_RATIO, "FLOOR_REL": FLOOR_REL, "GATE": gate,
                          "MIN_SESSIONS": MIN_SESSIONS, "MIN_ITERS": MIN_ITERS,
                          "H_CONFIRM": H_CONFIRM, "H_PARTIAL": H_PARTIAL, "H_REFUTE": H_REFUTE,
                          "M_CONFIRM": M_CONFIRM, "M_REFUTE": M_REFUTE, "P_SIG": P_SIG,
                          "MIN_F": MIN_F, "MIN_U": MIN_U, "MIN_UNEAR": MIN_UNEAR,
                          "MIN_FAMILIES": MIN_FAMILIES, "MAX_INDET_SHARE": MAX_INDET_SHARE,
                          "MAX_EXCLUDED": MAX_EXCLUDED, "MAX_DROPPED": MAX_DROPPED,
                          "DISPATCH_US": DISPATCH_US, "MAG_BAND": [MAG_LO, MAG_HI],
                          "rule": "a_design_registered.md sections 4.4, 5 and 6"},
           "resolutions": {
               "R1": "section 5.3 bands read top to bottom; a REFUTED condition met under PARTIAL is reported, not applied",
               "R2": "a section 6 gating failure outranks INCONCLUSIVE-SPARSE",
               "R3": "section 1.3 sizes counted over the models entering the matrix; frozen strata reported beside",
               "R4": "VA7 completeness is judged on section 4.4 valid sessions, not on gate survivors",
               "R5": "VA3 negative controls must be NOT-STEPPED; INDETERMINATE fails VA3",
               "R6": "FAILED-TO-RUN models count inside VA7's dropped total",
               "R7": "a cell with more than 5 sessions is a gating failure (use --replace-cell)",
               "R8": "section 5.4's class-(ii) cap is post-hoc; --fn-class-ii records it separately"},
           "frozen": {"paths": fr["paths"], "sha256": fr["sha256"], "seal": fr["seal"],
                      "selftest": fr["selftest"], "precondition": fr["precondition"],
                      "predictions_meta": fr["predictions_meta"],
                      "prediction_class_derived_from_rule": fr["prediction_class_derived_from_rule"],
                      "prediction_rule_mismatch": fr["prediction_rule_mismatch"]}}

    models = fr["models"]
    by_key = {(m["family"], m["variant"]): mid for mid, m in models.items()}
    extra_hard = []

    # ---- rows that belong to no frozen model and to no control (VA8) -------------------------
    seen = {(str(f), str(v)) for f, v in zip(df["family"], df["variant"])}
    unknown = sorted(f"{f}|{v}" for (f, v) in seen if (f, v) not in by_key and (f, v) not in CONTROLS)

    # ---- per model ---------------------------------------------------------------------------
    per, oversized = {}, []
    for mid, m in models.items():
        rows = df[(df["family"] == m["family"]) & (df["variant"] == m["variant"])]
        cells_u = model_cells(rows, gate)
        cells_g = model_cells(rows, gate, gated=True)
        primary = [(OLD, T_PRIMARY), (NEW, T_PRIMARY), (OLD, T_CONTROL), (NEW, T_CONTROL)]
        n_valid = {f"{r}|{t}T": (cells_u.get((r, t)) or {}).get("n", 0) for r, t in primary}
        below = {f"{r}|{t}T": (cells_u.get((r, t)) or {}).get("below_gate", 0) for r, t in primary}
        over = [k for k, n in n_valid.items() if n > MIN_SESSIONS]
        if over:
            oversized.append(f"{mid} ({', '.join(over)})")
        breach = any((cells_u.get((r, T_PRIMARY)) or {}).get("below_gate", 0) > MAX_BELOW_GATE_PER_CELL
                     for r in (OLD, NEW))
        complete = all(n_valid[f"{r}|{t}T"] == MIN_SESSIONS for r, t in primary)
        if sum(n_valid.values()) == 0:
            disp = "FAILED-TO-RUN"
        elif over:
            disp = "OVERSIZED-CELL"      # R7: a pooled re-run, which VA5 forbids
        elif breach:
            disp = "EXCLUDED"
        elif not complete:
            disp = "DROPPED"
        else:
            disp = "SCORED"
        st_u = status_of(cells_u)
        st_g = status_of(cells_g)
        mag = None
        if m["A_ms"] and st_u["delta_ms_4t"] is not None and m["A_ms"] > 0:
            mag = st_u["delta_ms_4t"] / m["A_ms"]
        per[mid] = {**{k: m[k] for k in ("id", "family", "variant", "arch_family", "sha256",
                                         "screen_ms", "publisher", "stratum_frozen", "u_near",
                                         "sig_b", "sig_a", "added_dispatches_b",
                                         "added_dispatches_a", "A_ms", "rho", "prediction",
                                         "sig_small")},
                     "n_valid_sessions": n_valid, "below_gate": below, "disposition": disp,
                     "ungated": st_u, "gated": st_g,
                     "cells": {f"{r}|{t}T": {k: v for k, v in (cells_u.get((r, t)) or {}).items()}
                               for r, t in primary},
                     "magnitude_ratio": mag,
                     "magnitude_in_band": None if mag is None else bool(MAG_LO <= mag <= MAG_HI)}
        if disp == "SCORED":
            for arm, stt in (("ungated", st_u), ("gated", st_g)):
                per[mid][arm]["cell"] = cell_of(m["prediction"], stt["status"], stt["pure"])
    if oversized:
        extra_hard.append(
            "R7/VA5: a cell holds more than %d sessions, which pools a re-run instead of replacing "
            "it (use --replace-cell): %s" % (MIN_SESSIONS, "; ".join(oversized)))

    # ---- the three controls (VA3) -------------------------------------------------------------
    ctrl = {}
    for (fam, var), kind in CONTROLS.items():
        rows = df[(df["family"] == fam) & (df["variant"] == var)]
        cells_u = model_cells(rows, gate)
        st = status_of(cells_u)
        primary = [(OLD, T_PRIMARY), (NEW, T_PRIMARY), (OLD, T_CONTROL), (NEW, T_CONTROL)]
        ctrl[f"{fam}|{var}"] = {
            "kind": kind, "status": st["status"], "ratio_4t": st["ratio_4t"],
            "disjoint": st["disjoint"], "ratio_1t": st["ratio_1t"], "pure": st["pure"],
            "n_valid_sessions": {f"{r}|{t}T": (cells_u.get((r, t)) or {}).get("n", 0)
                                 for r, t in primary},
            "below_gate": {f"{r}|{t}T": (cells_u.get((r, t)) or {}).get("below_gate", 0)
                           for r, t in primary},
            "cells": {f"{r}|{t}T": {k: v for k, v in (cells_u.get((r, t)) or {}).items()}
                      for r, t in primary}}
    va3_why = []
    for name, c in ctrl.items():
        if c["kind"] == "positive" and c["status"] != "STEPPED":
            va3_why.append(f"the positive control {name} is {c['status']} (ratio "
                           f"{'-' if c['ratio_4t'] is None else round(c['ratio_4t'], 4)}, "
                           f"disjoint {c['disjoint']})")
        if c["kind"] == "negative" and c["status"] != "NOT-STEPPED":
            va3_why.append(f"the negative control {name} is {c['status']} (ratio "
                           f"{'-' if c['ratio_4t'] is None else round(c['ratio_4t'], 4)})")
    va3 = not va3_why

    # ---- dispositions, strata, and the matrix per arm -----------------------------------------
    excluded = sorted(mid for mid, r in per.items() if r["disposition"] == "EXCLUDED")
    dropped = sorted(mid for mid, r in per.items() if r["disposition"] == "DROPPED")
    failed = sorted(mid for mid, r in per.items() if r["disposition"] == "FAILED-TO-RUN")
    scored = sorted(mid for mid, r in per.items() if r["disposition"] == "SCORED")
    oversized_models = sorted(mid for mid, r in per.items() if r["disposition"] == "OVERSIZED-CELL")

    arms = {}
    for arm in ("ungated", "gated"):
        counts = {k: 0 for k in ("TP", "IMPURE", "FP", "FN", "FN-IMPURE", "TN")}
        indet, statused = [], []
        for mid in scored:
            st = per[mid][arm]
            if st["status"] is None:
                continue
            statused.append(mid)
            if st["status"] == "INDETERMINATE":
                indet.append(mid)
                continue
            counts[st["cell"]] += 1
        inmatrix = [mid for mid in statused if mid not in indet]
        F = [mid for mid in inmatrix if per[mid]["prediction"] == "PREDICT-STEP"]
        U = [mid for mid in inmatrix if per[mid]["prediction"] == "PREDICT-SAFE"]
        UN = [mid for mid in U if per[mid]["u_near"]]
        fams = sorted({per[mid]["arch_family"] for mid in inmatrix if per[mid]["arch_family"]})
        fams_unknown = [mid for mid in inmatrix if not per[mid]["arch_family"]]
        sizes = {"F": len(F), "U": len(U), "U_near": len(UN), "arch_families": len(fams),
                 "arch_families_list": fams, "arch_family_unknown": fams_unknown}
        sizes_ok = (len(F) >= MIN_F and len(U) >= MIN_U and len(UN) >= MIN_UNEAR
                    and len(fams) >= MIN_FAMILIES and not fams_unknown)
        sparse_why = []
        if len(F) < MIN_F:
            sparse_why.append(f"|F| = {len(F)} < {MIN_F}")
        if len(U) < MIN_U:
            sparse_why.append(f"|U| = {len(U)} < {MIN_U}")
        if len(UN) < MIN_UNEAR:
            sparse_why.append(f"|U-near| = {len(UN)} < {MIN_UNEAR}")
        if len(fams) < MIN_FAMILIES:
            sparse_why.append(f"{len(fams)} architecture families < {MIN_FAMILIES}")
        if fams_unknown:
            sparse_why.append("manifest records no architecture family for "
                              + ", ".join(fams_unknown))
        b = band_of(counts, sizes_ok)
        b.update({"sizes": sizes, "sizes_ok": sizes_ok, "sparse_because": sparse_why,
                  "indeterminate": indet, "indeterminate_share": (len(indet) / len(statused)
                                                                  if statused else None),
                  "n_statused": len(statused), "in_matrix": len(inmatrix),
                  "F_models": F, "U_models": U, "U_near_models": UN})
        arms[arm] = b

    u, g = arms["ungated"], arms["gated"]

    # ---- validity conditions of section 6 -----------------------------------------------------
    seal = fr["seal"]
    sha_ok, sha_bad = True, []
    for mid in per:
        m = models[mid]
        if not m["sha256"]:
            sha_ok = False
            sha_bad.append(f"{mid}: manifest has no sha256")
            continue
        f = root / m["family"] / f"{m['variant']}.onnx"
        got = sha256_file(f)
        if got is None:
            sha_ok = False
            sha_bad.append(f"{mid}: {f} not found for re-hashing")
        elif got.lower() != str(m["sha256"]).lower():
            sha_ok = False
            sha_bad.append(f"{mid}: sha256 {got[:16]} != manifest {str(m['sha256'])[:16]}")
    csv_mtime = datetime.fromtimestamp(Path(csv_paths[0]).stat().st_mtime, timezone.utc)
    if not seal["run_start_utc"]:
        seal["run_start_utc"] = csv_mtime.isoformat()
        seal["compared_against"] = "mtime of the timing CSV (no timestamp in precondition.json)"
    seal_before = None
    if seal["seal_utc"]:
        seal_before = _parse_ts(seal["seal_utc"]) < _parse_ts(seal["run_start_utc"])
    va1_ok = (all(fr["sha256"].values()) and all(seal["digest_in_findings"].values())
              and bool(seal_before) and sha_ok and not fr["prediction_rule_mismatch"])

    va5_excluded_ok = len(excluded) <= MAX_EXCLUDED
    va7_dropped = dropped + failed
    va7_ok = len(va7_dropped) <= MAX_DROPPED
    nan_reached = False  # section 4.4 removes them before any median is taken; asserted below
    for mid in scored:
        for k, c in per[mid]["cells"].items():
            if c and (math.isnan(c.get("median", float("nan"))) or
                      any(math.isnan(x) for x in c.get("p50s", []))):
                nan_reached = True
    va6_ok = u["band"] == g["band"]
    indet_ok = (u["indeterminate_share"] is None) or (u["indeterminate_share"] <= MAX_INDET_SHARE)

    validity = {
        "VA1_seal": {"ok": bool(va1_ok), "artefacts_present": {k: bool(v) for k, v in fr["sha256"].items()},
                     "digest_in_findings": seal["digest_in_findings"],
                     "seal_utc": seal["seal_utc"], "run_start_utc": seal["run_start_utc"],
                     "compared_against": seal["compared_against"],
                     "seal_before_first_row": seal_before,
                     "model_sha256_match": bool(sha_ok), "sha256_problems": sha_bad},
        "VA2_checker_selftest": {"ok": bool(fr["selftest"]["ok"]), **fr["selftest"]},
        "VA3_controls": {"ok": bool(va3), "why_not": va3_why, "controls": ctrl},
        "VA4_quiet_machine": {"ok": bool(fr["precondition"]["ok"]),
                              "recorded": bool(fr["precondition"]["recorded"])},
        "VA5_core_delivery": {"ok": bool(va5_excluded_ok), "excluded": excluded,
                              "n_excluded": len(excluded), "max_allowed": MAX_EXCLUDED,
                              "one_thread_cells_are_REPORTED_not_gated": True},
        "VA6_gated_equals_ungated": {"ok": bool(va6_ok), "ungated_band": u["band"],
                                     "gated_band": g["band"]},
        "VA7_completeness": {"ok": bool(va7_ok and not nan_reached), "dropped": dropped,
                             "failed_to_run": failed, "n_dropped_total": len(va7_dropped),
                             "max_allowed": MAX_DROPPED, "nan_reached_a_comparison": nan_reached},
        "VA8_no_reselection": {"ok": not unknown, "rows_not_in_manifest": unknown,
                               "note": "the freeze itself is VA1; this checks only that the CSV "
                                       "contains no model the frozen manifest does not name"},
        "VA9_one_measurement_no_orphans": {"ok": None,
                                           "note": "runner-side condition; not checkable from the CSV"},
        "VA10_reported_not_gating": {"ok": None, "note": "see per_model, controls and indeterminate"},
        "indeterminate_within_20pct": {"ok": bool(indet_ok),
                                       "share": u["indeterminate_share"],
                                       "models": u["indeterminate"]},
    }

    reasons = []
    if not va1_ok:
        bits = [k for k, v in seal["digest_in_findings"].items() if not v]
        reasons.append("VA1 the seal does not hold" + (f" (no digest in FINDINGS.md for {bits})" if bits else "")
                       + ("" if seal_before is not False else " (the seal is not earlier than the run)")
                       + ("" if sha_ok else f" (model sha256: {'; '.join(sha_bad)})")
                       + ("" if not fr["prediction_rule_mismatch"] else " (a frozen PRIMARY class "
                          "disagrees with the registered rule)"))
    if not fr["selftest"]["ok"]:
        reasons.append("VA2 the checker self-test does not reproduce the registered worked example "
                       f"({SELFTEST_FLAGGED} flagged, {SELFTEST_ADDED} added dispatches)")
    if not va3:
        reasons.append("VA3 " + "; ".join(va3_why))
    if not fr["precondition"]["ok"]:
        reasons.append("VA4 the quiet-machine record precondition.json is "
                       + ("absent" if not fr["precondition"]["recorded"] else "not recorded as passed"))
    if not va5_excluded_ok:
        reasons.append(f"VA5 {len(excluded)} models excluded for core delivery (> {MAX_EXCLUDED}): "
                       + ", ".join(excluded))
    if not va6_ok:
        reasons.append(f"VA6 gated ({g['band']}) and ungated ({u['band']}) verdicts differ")
    if not va7_ok:
        reasons.append(f"VA7 {len(va7_dropped)} models incomplete (> {MAX_DROPPED}): "
                       + ", ".join(va7_dropped))
    if nan_reached:
        reasons.append("VA7 a NaN reached a comparison")
    if unknown:
        reasons.append("VA8 the CSV names models the frozen manifest does not: " + ", ".join(unknown))
    for e in extra_hard + fr["errors"]:
        if e not in reasons:
            reasons.append(e)
    if not indet_ok:
        reasons.append(f"INDETERMINATE models are {u['indeterminate_share']:.1%} of the measured set "
                       f"(> {MAX_INDET_SHARE:.0%}): " + ", ".join(u["indeterminate"]))

    # ---- the verdict (R2: validity first, sparsity second) ------------------------------------
    if reasons:
        verdict = "INCONCLUSIVE"
    elif not u["sizes_ok"]:
        verdict = "INCONCLUSIVE-SPARSE"
    else:
        verdict = u["band"]

    fn_total = u["counts"]["FN"] + u["counts"]["FN-IMPURE"]
    fn_models = [mid for mid in scored
                 if per[mid]["ungated"].get("cell") in ("FN", "FN-IMPURE")]
    capped = None
    if fn_class_ii is not None and fn_class_ii >= FN_CLASS_II_CAP and verdict == "CONFIRMED":
        capped = "PARTIAL"

    wording = {"CONFIRMED": "A_CONFIRMED", "PARTIAL": "A_PARTIAL", "MIXED": "A_MIXED",
               "REFUTED": "A_REFUTED", "INCONCLUSIVE-SPARSE": "A_SPARSE",
               "INCONCLUSIVE": "A_INCONCLUSIVE"}[verdict]

    rep["per_model"] = per
    rep["controls"] = ctrl
    rep["dispositions"] = {"scored": scored, "excluded": excluded, "dropped": dropped,
                           "failed_to_run": failed, "oversized_cell": oversized_models,
                           "n_measured_models_in_manifest": len(models)}
    rep["strata_frozen"] = {
        "F": sorted(mid for mid, m in models.items() if m["prediction"] == "PREDICT-STEP"),
        "U": sorted(mid for mid, m in models.items() if m["prediction"] == "PREDICT-SAFE"),
        "U_near": sorted(mid for mid, m in models.items()
                         if m["prediction"] == "PREDICT-SAFE" and m["u_near"]),
        "SIG_SMALL": sorted(mid for mid, m in models.items() if m["sig_small"])}
    rep["arms"] = arms
    rep["validity"] = validity
    rep["magnitude_ratios_REPORTED"] = {
        mid: {"A_ms": per[mid]["A_ms"], "measured_delta_ms": per[mid]["ungated"]["delta_ms_4t"],
              "ratio": per[mid]["magnitude_ratio"], "in_band": per[mid]["magnitude_in_band"]}
        for mid in scored if per[mid]["prediction"] == "PREDICT-STEP"}
    rep["sc_a_vs_sc_b_disagreement_REPORTED"] = sorted(
        mid for mid, m in models.items()
        if m["sig_a"] is not None and bool(m["sig_a"]) != bool(m["sig_b"]))
    rep["final"] = {
        "verdict": verdict,
        "N_measured": len(scored) + len(excluded) + len(dropped) + len(failed)
        + len(oversized_models),
        "N_in_matrix": u["in_matrix"], "k_flagged_in_matrix": u["sizes"]["F"],
        "counts": u["counts"], "H": u["H"], "M": u["M"], "fisher_p": u["fisher_p"],
        "sizes": u["sizes"], "ungated_band": u["band"], "gated_band": g["band"],
        "inconclusive_because": reasons, "sparse_because": u["sparse_because"],
        "refuted_condition_also_met": u["refuted_condition_also_met"],
        "indeterminate": u["indeterminate"],
        "fn_total": fn_total, "fn_models": fn_models,
        "any_fn_bars_safety_claim": bool(fn_total),
        "fn_class_ii_cap": {"rule": "section 5.4: three or more FNs of class (ii) cap the verdict "
                                    "at PARTIAL; the class comes from the post-verdict diagnostic",
                            "fn_class_ii_given": fn_class_ii,
                            "verdict_after_fn_class_ii_cap": capped},
        "paper_wording_key": wording}

    out = Path(out_path) if out_path else (root / "results" / "expA" / "verdict.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    _print(rep)
    print(f"\nwrote {out}")
    return rep


# ---- printing ------------------------------------------------------------------------------
def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _print(rep):
    per, fin = rep["per_model"], rep["final"]
    print("\n" + "=" * 100)
    print("EXPERIMENT A - frozen prediction vs measured four-thread step (a_design_registered.md s.5)")
    print("=" * 100)
    print(f"{'model':16s} {'arch':14s} {'pred':13s} {'rho':>6s} {'m1232':>8s} {'m1244':>8s} "
          f"{'ratio':>6s} {'disj':>5s} {'1T':>6s} {'status':>14s} {'cell':>10s} {'disp':>13s}")
    for mid in sorted(per):
        r = per[mid]
        u = r["ungated"]
        c4o = r["cells"].get(f"{OLD}|{T_PRIMARY}T") or {}
        c4n = r["cells"].get(f"{NEW}|{T_PRIMARY}T") or {}
        print(f"{mid[:16]:16s} {str(r['arch_family'])[:14]:14s} {str(r['prediction']):13s} "
              f"{_f(r['rho']):>6s} {_f(c4o.get('median')):>8s} {_f(c4n.get('median')):>8s} "
              f"{_f(u['ratio_4t'], 3):>6s} {str(u['disjoint'])[:5]:>5s} {_f(u['ratio_1t'], 3):>6s} "
              f"{str(u['status']):>14s} {str(u.get('cell') or '-'):>10s} {r['disposition']:>13s}")
    print("\ncontrols (never in the matrix):")
    for name, c in rep["controls"].items():
        print(f"  {name:22s} {c['kind']:8s} ratio4T {_f(c['ratio_4t'])} disjoint {c['disjoint']} "
              f"ratio1T {_f(c['ratio_1t'])} -> {c['status']}")
    for arm in ("ungated", "gated"):
        a = rep["arms"][arm]
        c = a["counts"]
        print(f"\n===== {arm.upper()} =====")
        print(f"  TP {c['TP']}  IMPURE {c['IMPURE']}  FP {c['FP']} | FN {c['FN']}  "
              f"FN-IMPURE {c['FN-IMPURE']}  TN {c['TN']}   INDETERMINATE {len(a['indeterminate'])}")
        print(f"  H {_f(a['H'], 4)}   M {_f(a['M'], 4)}   Fisher one-sided p {_f(a['fisher_p'], 5)}")
        print(f"  |F| {a['sizes']['F']}  |U| {a['sizes']['U']}  |U-near| {a['sizes']['U_near']}  "
              f"arch families {a['sizes']['arch_families']}  -> band {a['band']}")
        for w in a["sparse_because"]:
            print(f"    sparse: {w}")
    v = rep["validity"]
    print("\nvalidity (section 6):")
    for k in ("VA1_seal", "VA2_checker_selftest", "VA3_controls", "VA4_quiet_machine",
              "VA5_core_delivery", "VA6_gated_equals_ungated", "VA7_completeness",
              "VA8_no_reselection", "indeterminate_within_20pct"):
        print(f"  {k:28s} {v[k]['ok']}")
    d = rep["dispositions"]
    print(f"\nEXCLUDED {d['excluded']}  DROPPED {d['dropped']}  FAILED-TO-RUN {d['failed_to_run']}"
          f"  INDETERMINATE {fin['indeterminate']}")
    print(f"\nFINAL: {fin['verdict']}   (N in matrix {fin['N_in_matrix']}, k flagged "
          f"{fin['k_flagged_in_matrix']}, H {_f(fin['H'], 3)}, M {_f(fin['M'], 3)}, "
          f"p {_f(fin['fisher_p'], 5)})  -> paper wording {fin['paper_wording_key']}")
    for r in fin["inconclusive_because"]:
        print(f"   ! {r}")
    if fin["any_fn_bars_safety_claim"]:
        print(f"   ~ section 5.4: {fin['fn_total']} FN ({', '.join(fin['fn_models'])}) - the check "
              f"may not be called necessary and no unflagged model may be called safe")
    if fin["refuted_condition_also_met"] and fin["verdict"] not in ("REFUTED", "INCONCLUSIVE"):
        print("   ~ a REFUTED condition of section 5.3 is also met; the table is read top to bottom (R1)")
    if fin["fn_class_ii_cap"]["verdict_after_fn_class_ii_cap"]:
        print(f"   ~ post-hoc: with {fin['fn_class_ii_cap']['fn_class_ii_given']} class-(ii) FNs the "
              f"reported verdict is capped at {fin['fn_class_ii_cap']['verdict_after_fn_class_ii_cap']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="results/paper/a_thirdparty.csv (plus a registered re-run CSV)")
    ap.add_argument("--root", default=str(ROOT),
                    help="project root holding results/expA and results/paper (default: this project)")
    ap.add_argument("--out", default=None, help="default <root>/results/expA/verdict.json")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--replace-cell", action="append", default=[],
                    metavar="FAMILY,VARIANT,ORT,THREADS",
                    help="for this cell keep only the rows of the LAST csv given (a VA3/VA5 "
                         "seed-54 re-run); repeatable")
    ap.add_argument("--fn-class-ii", type=int, default=None,
                    help="number of FNs the post-verdict diagnostic of section 5.4 classed (ii); "
                         "recorded separately, never changes the computed verdict")
    a = ap.parse_args()
    analyse([Path(c) for c in a.csv], root=Path(a.root), gate=a.gate, out_path=a.out,
            replace=tuple(a.replace_cell), fn_class_ii=a.fn_class_ii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
