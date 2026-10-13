"""Section 7 of the task sheet: compile and profile a model on Qualcomm AI Hub.

    python sec2026/aihub_profile.py --model sec2026/out/p00/student_model_int8.onnx --device "<PROFILE>"
    python sec2026/aihub_profile.py --list-devices

--device must be the profile the organisers announce (the sheet leaves it as "[PROFILE]"). The
duplicate graph.value_info entries the sheet warns about are stripped from a copy before upload.
The full profile is saved to <model dir>/aihub_<model stem>.json together with a short summary
(the figures metrics.json's ai_hub_profile_summary needs). Submit early: the free tier queues.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _hub():
    try:
        import truststore  # this laptop's antivirus re-signs TLS; use the Windows certificate store

        truststore.inject_into_ssl()
    except Exception:
        pass
    import qai_hub as hub

    return hub


def summarize(profile: dict) -> dict:
    """Pull the headline numbers out of an AI Hub profile; key names have moved between versions,
    so look them up defensively and keep whatever is present."""
    ex = profile.get("execution_summary", profile)
    pick = lambda *ks: next((ex[k] for k in ks if k in ex), None)
    t_us = pick("estimated_inference_time", "inference_time")
    peak = pick("estimated_inference_peak_memory", "inference_memory_peak_range")
    if isinstance(peak, (list, tuple)):
        peak = max(peak)
    return {"inference_time_ms": None if t_us is None else t_us / 1000.0,
            "peak_memory_mb": None if peak is None else peak / 2**20,
            "compute_units": sorted({l.get("compute_unit") for l in profile.get("execution_detail", [])
                                     if isinstance(l, dict) and l.get("compute_unit")}) or None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--device", help="the organisers' AI Hub device profile")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--options", default="--target_runtime onnx",
                    help="compile options; the default keeps the QDQ graph on ONNX Runtime (the TFLite "
                         "default rejects per-channel QDQ weights: 'tfl.transpose mismatched quantized axes')")
    ap.add_argument("--tag", default="", help="suffix for the output json")
    a = ap.parse_args(argv)
    hub = _hub()
    if a.list_devices:
        for d in hub.get_devices():
            print(f"{d.name} | {d.os} | {', '.join(sorted(x for x in d.attributes if x.startswith('chipset')))}")
        return
    if not (a.model and a.device):
        ap.error("--model and --device are required (or --list-devices)")

    import export_quant as eq

    src = Path(a.model)
    up = src.with_name(src.stem + "_aihub.onnx")
    shutil.copyfile(src, up)
    removed = eq.strip_output_value_info(up)
    device = hub.Device(a.device)
    compile_job = hub.submit_compile_job(model=str(up), device=device,
                                         input_specs={"input": (1, 3, 224, 224)}, options=a.options)
    print(f"compile job {compile_job.job_id} submitted ({compile_job.url})")
    compile_job.wait()
    profile_job = hub.submit_profile_job(model=compile_job.get_target_model(), device=device)
    print(f"profile job {profile_job.job_id} submitted ({profile_job.url})")
    profile_job.wait()
    prof = profile_job.download_profile()
    rec = {"model": str(src), "device": a.device, "compile_options": a.options, "value_info_removed": removed,
           "compile_job": compile_job.job_id, "profile_job": profile_job.job_id,
           "profile_url": profile_job.url, "summary": summarize(prof), "profile": prof}
    out = src.with_name(f"aihub_{src.stem}{a.tag}.json")
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    up.unlink(missing_ok=True)
    print(json.dumps(rec["summary"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
