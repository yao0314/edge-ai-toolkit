"""Optional Qualcomm AI Hub wrapper (pip install -r requirements-optional.txt).

Written against qai-hub 0.55.0. Not exercised in CI (needs an account), but the whole path —
devices, compile, profile — was run against a real account on 2026-10-11: a 2.5 MB INT8 model on
the RB3 Gen 2 Vision Kit returned 1.27 ms and 19.83 MB peak, job jp1o7qr85, about 5 minutes wall
clock of which most is queueing and device provisioning.

Token: read from the environment variable QAIHUB_API_TOKEN (our convention; the qai-hub
client itself only reads ~/.qai_hub/client.ini) and passed in memory via
hub.Client(hub.ClientConfig(api_token=...)). Never put the token in code, configs or git.
Without the variable we fall back to the client.ini written by `qai-hub configure`.

AI Hub profiles report latency (us) and peak memory (bytes); they report NO energy. Energy can
only be corroborated indirectly: energy ~ power x latency, so the AI Hub latency ratio between two
models should track the locally measured energy ratio.

Device catalogue, checked 2026-10-11 (76 devices): the competition's target board is not in it.
`pick_device` walks the preference list from the YAML config (`aihub.devices`) and takes the first
available. Two traps found in that check:

  * `chipset:qualcomm-snapdragon-429`, the old first preference (chosen as the nearest 4x A53),
    does not exist at all — it silently fell through to the next entry.
  * "Arduino VENTUNO Q" IS listed and is the obvious pick by name. It is the WRONG board: the
    competition targets the Arduino UNO Q (Dragonwing QRB2210, 4x Cortex-A53 @ 2.0 GHz) while
    VENTUNO Q is a Dragonwing IQ8 / QCS8275 rated up to 40 dense TOPS. Picking it would
    understate latency and energy by a wide margin.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)
TOKEN_ENV = "QAIHUB_API_TOKEN"


def get_client():
    import qai_hub as hub

    token = os.environ.get(TOKEN_ENV)
    if token:
        return hub.Client(hub.ClientConfig(api_token=token))
    log.info("%s not set; using ~/.qai_hub/client.ini", TOKEN_ENV)
    return hub  # module-level functions use the ini-configured global client


def list_devices(client=None, name: str = "", attributes=None) -> list:
    client = client or get_client()
    return client.get_devices(name=name, attributes=attributes)


def pick_device(preferences: list[str], client=None):
    """First available device. Entries are device names or attributes like 'chipset:qualcomm-qcs6490'."""
    client = client or get_client()
    for pref in preferences:
        try:
            found = client.get_devices(attributes=pref) if ":" in pref else client.get_devices(name=pref)
        except Exception as e:  # unknown attribute etc.
            log.debug("device lookup %r failed: %s", pref, e)
            found = []
        if found:
            log.info("AI Hub device: %s (matched %r)", found[0].name, pref)
            return found[0]
    raise RuntimeError(f"none of the preferred AI Hub devices is available: {preferences}")


def profile_onnx(onnx_path: str | Path, input_shape: tuple, device, client=None, input_name: str = "input",
                 compile_options: str = "--target_runtime onnx --compute_unit cpu",
                 profile_options: str = "", name: str | None = None) -> dict:
    """Compile + profile one ONNX model on a hosted device. Blocks until the jobs finish."""
    client = client or get_client()
    cj = client.submit_compile_job(model=str(onnx_path), device=device, name=name,
                                   input_specs={input_name: tuple(int(s) for s in input_shape)},
                                   options=compile_options)
    target = cj.get_target_model()
    if target is None:
        raise RuntimeError(f"AI Hub compile failed: {cj.url if hasattr(cj, 'url') else cj}")
    pj = client.submit_profile_job(model=target, device=device, name=name, options=profile_options)
    prof = pj.download_profile()
    s = prof["execution_summary"]
    peak = s.get("estimated_inference_peak_memory")
    return {
        "aihub_device": device.name,
        "aihub_latency_ms": s["estimated_inference_time"] / 1000.0,   # docs: microseconds
        "aihub_peak_memory_mb": peak / 2**20 if peak is not None else float("nan"),  # docs: bytes
        "aihub_profile_job": getattr(pj, "job_id", ""),
    }
