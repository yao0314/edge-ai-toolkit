"""Fetch finished AI Hub profile jobs by id and write the figures as JSON.

Submission and retrieval are separate on purpose. The qai-hub client prints a progress spinner
containing U+23F3, which a cp950 console cannot encode, so the client raises while WAITING even
though the jobs were scheduled fine. Resubmitting would burn a free-tier queue slot for nothing;
this reattaches to the jobs that are already running.

  PYTHONIOENCODING=utf-8 python scripts/sec_aihub_fetch.py \
      --job p00_fp32=jp3o60znp --job p40_int8=jpv27klrg --out results/sec2026_real/aihub.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", action="append", required=True, metavar="NAME=JOBID")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import qai_hub as hub

    out = {"models": {}}
    if Path(a.out).exists():
        try:
            out = json.loads(Path(a.out).read_text(encoding="utf-8"))
            out.setdefault("models", {})
        except Exception:
            pass

    for spec in a.job:
        name, _, jid = spec.partition("=")
        try:
            pj = hub.get_job(jid)
            status = pj.get_status()
            if not status.finished:
                print(f"{name:10s} {jid}  still {status.code} - run again later")
                continue
            if not status.success:
                print(f"{name:10s} {jid}  FAILED on AI Hub: {status.message}")
                out["models"][name] = {"error": f"{status.code}: {status.message}"}
                continue
            s = pj.download_profile()["execution_summary"]
            peak = s.get("estimated_inference_peak_memory")
            out["models"][name] = {
                "aihub_device": pj.device.name,
                "aihub_latency_ms": s["estimated_inference_time"] / 1000.0,      # docs: microseconds
                "aihub_peak_memory_mb": peak / 2**20 if peak is not None else None,   # docs: bytes
                "aihub_profile_job": jid,
            }
            print(f"{name:10s} {jid}  {out['models'][name]['aihub_latency_ms']:.3f} ms, "
                  f"{out['models'][name]['aihub_peak_memory_mb']:.2f} MB peak")
        except Exception as e:
            print(f"{name:10s} {jid}  ERROR {e}")

    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
