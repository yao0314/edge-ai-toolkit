"""Profile sweep variants on Qualcomm AI Hub (optional; needs QAIHUB_API_TOKEN).

  python scripts/aihub_profile.py --check           # offline readiness check: no account needed
  python scripts/aihub_profile.py --list-devices     # first call that needs a real token
  python scripts/aihub_profile.py --config configs/classification.yaml            # all OK rows of sweep.csv
  python scripts/aihub_profile.py --config configs/classification.yaml --onnx m.onnx

Adds aihub_latency_ms / aihub_peak_memory_mb columns to results/<task>/sweep.csv.
"""
from _common import base_parser, resolve_config  # noqa: F401

import pandas as pd

from core import aihub
from core.config import build_adapter, results_dir, run_name, setup_env


def self_check(args) -> bool:
    """Offline readiness check. THIS PATH HAS NEVER BEEN RUN AGAINST A REAL ACCOUNT: everything below is
    verified locally, and the one remaining step (a real job) needs a Qualcomm AI Hub token."""
    import importlib.util
    import os
    from pathlib import Path

    ok = True

    def line(good, msg, fix=""):
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else 'MISSING'}] {msg}")
        if not good and fix:
            print(f"         -> {fix}")

    print("Qualcomm AI Hub readiness")
    line(importlib.util.find_spec("qai_hub") is not None, "qai-hub installed",
         "pip install -r requirements-optional.txt")
    has_token = bool(os.environ.get("QAIHUB_API_TOKEN"))
    has_ini = (Path.home() / ".qai_hub" / "client.ini").exists()
    line(has_token or has_ini, "API token available (env QAIHUB_API_TOKEN or ~/.qai_hub/client.ini)",
         "register at https://aihub.qualcomm.com, then: set QAIHUB_API_TOKEN=<token>  (never commit it)")
    if args.task or args.config:
        cfg = resolve_config(args)
        devices = cfg.get("aihub", {}).get("devices", [])
        line(bool(devices), f"device preference list in the config: {devices}",
             "add an aihub.devices list to the YAML")
        csv = Path(args.csv) if args.csv else results_dir(run_name(cfg), args.quick) / "sweep.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            models = [m for m in df.get("model_file", []) if isinstance(m, str) and m.endswith(".onnx")
                      and Path(m).exists()]
            line(bool(models), f"{len(models)} ONNX file(s) ready to profile from {csv.name}")
        else:
            line(False, f"no sweep results at {csv}", "run scripts/sweep.py first")
    print()
    print("Next (needs the token, ~2 min):")
    print("  python scripts/aihub_profile.py --list-devices")
    print("  python scripts/aihub_profile.py --config configs/detection.yaml")
    print()
    print("Known limits: AI Hub has no UNO Q / QRB2210 device (the config lists the closest proxies) and")
    print("its profiles report latency and peak memory but NO energy.")
    return ok


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="check everything that does not need the network (package, token, models, shapes) "
                         "and print what is still missing")
    ap.add_argument("--onnx", help="profile just this ONNX file")
    ap.add_argument("--csv", help="sweep csv to annotate (default results/<task>/sweep.csv)")
    args = ap.parse_args()
    setup_env()
    if args.check:
        raise SystemExit(0 if self_check(args) else 1)
    client = aihub.get_client()

    if args.list_devices:
        for d in aihub.list_devices(client):
            print(f"{d.name:45s} {d.os:8s} {', '.join(a for a in d.attributes if a.startswith(('chipset', 'framework')))}")
        return

    cfg = resolve_config(args)
    hub_cfg = cfg.get("aihub", {})
    device = aihub.pick_device(hub_cfg.get("devices", []), client)
    adapter = build_adapter(cfg, args.device)
    shape = tuple(adapter.example_input().shape)
    opts = dict(compile_options=hub_cfg.get("compile_options", "--target_runtime onnx --compute_unit cpu"),
                profile_options=hub_cfg.get("profile_options", ""), input_name=adapter.input_names()[0])

    if args.onnx:
        print(aihub.profile_onnx(args.onnx, shape, device, client, **opts))
        return

    csv = args.csv or results_dir(run_name(cfg), args.quick) / "sweep.csv"
    df = pd.read_csv(csv)
    for i, r in df.iterrows():
        path = r.get("model_file")
        if isinstance(r.get("error"), str) or not isinstance(path, str) or not path.endswith(".onnx"):
            continue
        try:
            res = aihub.profile_onnx(path, shape, device, client, name=r["variant"], **opts)
            for k, v in res.items():
                df.loc[i, k] = v
            print(r["variant"], res)
        except Exception as e:
            df.loc[i, "aihub_error"] = str(e)[:300]
            print(r["variant"], "FAILED", e)
        df.to_csv(csv, index=False)


if __name__ == "__main__":
    main()
