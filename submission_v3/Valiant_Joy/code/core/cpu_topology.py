"""Pick which logical CPUs the measurement probe is pinned to.

Why: on hybrid CPUs (e.g. Intel P-cores + E-cores) the OS may schedule the same
benchmark on a fast or a slow core. We measured 0.9 ms vs 2.4 ms for the same model
on one laptop, which is larger than most compression gains. Pinning makes runs comparable.

`affinity` in the measure config:
  auto  -> highest-performance core class, one logical CPU per physical core (default)
  none  -> no pinning
  [ids] -> explicit logical CPU ids; the first `threads` of them are used
Topology comes from the OS (Windows GetSystemCpuSetInformation, Linux sysfs); when it
cannot be read, `auto` falls back to no pinning.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def cpu_layout() -> list[dict] | None:
    """[{cpu, core, perf}] per logical CPU; perf: larger = faster core class. None if unknown."""
    try:
        if sys.platform == "win32":
            return _windows_layout()
        if sys.platform.startswith("linux"):
            return _linux_layout()
    except Exception as e:  # never break measuring because of topology probing
        log.debug("cpu topology unavailable: %s", e)
    return None


def _windows_layout() -> list[dict]:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    need = ctypes.c_ulong(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = (ctypes.c_ubyte * need.value)()
    if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        raise OSError(ctypes.get_last_error())
    raw, off, out = bytes(buf), 0, []
    while off < need.value:
        size = int.from_bytes(raw[off:off + 4], "little")
        group = int.from_bytes(raw[off + 12:off + 14], "little")
        if group == 0:  # psutil affinity addresses processor group 0 only
            out.append({"cpu": raw[off + 14], "core": raw[off + 15], "perf": raw[off + 18]})
        off += size
    return sorted(out, key=lambda d: d["cpu"])


def _linux_layout() -> list[dict]:
    out = []
    for d in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cpu = int(d.name[3:])
        topo = d / "topology"
        core = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
        perf = 0
        for f in (d / "cpu_capacity", d / "cpufreq" / "cpuinfo_max_freq"):
            if f.exists():
                perf = int(f.read_text())
                break
        out.append({"cpu": cpu, "core": core, "perf": perf})
    return out


def pick_cpus(threads: int, affinity="auto") -> list[int] | None:
    """Logical CPU ids to pin a run with `threads` intra-op threads to (None = don't pin)."""
    if affinity in (None, "none", False):
        return None
    if isinstance(affinity, (list, tuple)):
        return [int(c) for c in affinity[:max(threads, 1)]]
    layout = cpu_layout()
    if not layout:
        return None
    # fastest class first; within it one logical CPU per physical core, then SMT siblings, then the rest
    order = sorted(layout, key=lambda d: (-d["perf"], d["cpu"]))
    first, siblings, seen = [], [], set()
    for d in order:
        (siblings if d["core"] in seen else first).append(d)
        seen.add(d["core"])
    best = max(d["perf"] for d in layout)
    ranked = [d for d in first if d["perf"] == best] + [d for d in siblings if d["perf"] == best] + \
             [d for d in first + siblings if d["perf"] != best]
    return [d["cpu"] for d in ranked[:max(threads, 1)]]


def describe() -> str:
    layout = cpu_layout()
    if not layout:
        return "unknown"
    classes = {}
    for d in layout:
        classes.setdefault(d["perf"], set()).add(d["core"])
    return ", ".join(f"class {p}: {len(c)} cores" for p, c in sorted(classes.items(), reverse=True))
