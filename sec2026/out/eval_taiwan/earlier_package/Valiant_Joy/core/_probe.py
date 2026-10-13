"""Isolated measurement probe. Run as a separate process by core/measure.py.

Self-contained on purpose (no torch import, no project imports) so its memory
footprint is just python + numpy + the runtime + the model.

Backends: ONNX Runtime (.onnx) and LiteRT / TFLite (.tflite, needs ai-edge-litert).

Prints one JSON line with latency percentiles, CPU usage and peak RSS.
measure.py also runs it on an "empty" baseline whose peak RSS it subtracts:
  onnx   -> a single Identity node with the same input shape (same code path)
  tflite -> `--model none`: runtime imported and input loaded, no interpreter
            (no TFLite writer without TensorFlow; documented in the README)
"""
import argparse
import json
import sys
import threading
import time

import numpy as np
import psutil


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        return int(psutil.Process().memory_info().peak_wset)
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def adapt_layout(x, expected):
    """NCHW <-> NHWC when needed (duplicated from core/runtime.py to keep the probe standalone)."""
    exp = [d if isinstance(d, int) and d > 0 else None for d in expected[1:]]

    def fits(a):
        return a.ndim == len(expected) and all(e is None or e == s for e, s in zip(exp, a.shape[1:]))

    if x.ndim != 4 or fits(x):
        return x
    for perm in ((0, 2, 3, 1), (0, 3, 1, 2)):
        if fits(x.transpose(perm)):
            return np.ascontiguousarray(x.transpose(perm))
    return x


class OrtRunner:
    def __init__(self, path, a):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[a.opt_level]
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        # Both default on. The arena holds on to what it allocates, so it raises the peak RSS this
        # probe reports; whether that costs latency has to be measured, not assumed.
        so.enable_cpu_mem_arena = not a.no_cpu_mem_arena
        so.enable_mem_pattern = not a.no_mem_pattern
        if a.no_spinning:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]

    def prepare(self, x):
        self.feed = {self.inp.name: adapt_layout(x, list(self.inp.shape)).astype(np.float32)}

    def run(self):
        self.sess.run(None, self.feed)


class TFLiteRunner:
    def __init__(self, path, a):
        from ai_edge_litert.interpreter import Interpreter

        self.interp = Interpreter(model_path=path, num_threads=a.threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]

    def prepare(self, x):
        x = adapt_layout(x, [int(d) for d in self.inp["shape"]])
        dt = self.inp["dtype"]
        if np.issubdtype(dt, np.integer):
            scale, zp = self.inp["quantization"]
            info = np.iinfo(dt)
            x = np.clip(np.round(x / scale + zp), info.min, info.max) if scale else x
        self.x = x.astype(dt)

    def run(self):
        self.interp.set_tensor(self.inp["index"], self.x)
        self.interp.invoke()


class NullRunner:
    """Empty baseline for backends without an 'identity model': runtime imported, nothing loaded."""

    def __init__(self, backend):
        if backend == "tflite":
            from ai_edge_litert.interpreter import Interpreter  # noqa: F401
        else:
            import onnxruntime  # noqa: F401

    def prepare(self, x):
        self.x = x

    def run(self):
        pass


class CpuSampler(threading.Thread):
    """This process' CPU% (100 = one fully busy core) and, separately, everyone ELSE's load.

    Measuring next to another job shifts the median a little and inflates the spread a lot (measured
    here: a 94 MB model went 1.0% -> 6.9% relative spread under a saturating load, with the median
    moving ~1%). Whether that happened should be recorded, not left to whoever reads the table -
    and on a shared machine (the competition sandbox) we do not control what else runs. `other_pct`
    is the system-wide busy fraction minus our own share, so ~0 means we had the machine.
    """

    def __init__(self, interval: float):
        super().__init__(daemon=True)
        self.interval, self.samples, self.other, self._halt = interval, [], [], threading.Event()
        self.proc = psutil.Process()
        self.n_cpu = psutil.cpu_count() or 1
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)          # prime both counters; the first call always returns 0.0

    def _sample(self):
        mine = self.proc.cpu_percent(None)                 # 100 = one core
        system = psutil.cpu_percent(None)                  # 100 = every core busy
        self.samples.append(mine)
        self.other.append(max(system - mine / self.n_cpu, 0.0))

    def run(self):
        while not self._halt.wait(self.interval):
            self._sample()

    def stop(self):
        self._halt.set()
        self.join()
        self._sample()


class RssSampler(threading.Thread):
    """High-frequency RSS sampler: peak memory from session creation to the end of inference.

    The OS lifetime peak (peak_wset / ru_maxrss) also includes import-time transients, which
    can hide the footprint of small models; this sampler isolates the session's own growth.
    """

    def __init__(self, interval: float = 0.002):
        super().__init__(daemon=True)
        self.proc, self.interval, self._halt = psutil.Process(), interval, threading.Event()
        self.start_rss = self.peak = self.proc.memory_info().rss

    def run(self):
        while not self._halt.wait(self.interval):
            self.peak = max(self.peak, self.proc.memory_info().rss)

    def stop(self) -> int:
        self._halt.set()
        self.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        return self.peak - self.start_rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".onnx | .tflite | 'none' (empty baseline)")
    ap.add_argument("--backend", default=None, help="onnx | tflite (default: from the file extension)")
    ap.add_argument("--input", required=True, help=".npy with a batch=1 input")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--no-spinning", action="store_true")
    ap.add_argument("--no-cpu-mem-arena", action="store_true",
                    help="ORT's arena allocates up front and does not give memory back: off lowers peak RSS")
    ap.add_argument("--no-mem-pattern", action="store_true",
                    help="skip ORT's static-shape memory pre-planning")
    ap.add_argument("--opt-level", default="all")
    ap.add_argument("--sample-interval", type=float, default=0.1)
    ap.add_argument("--min-time", type=float, default=2.0, help="minimum timed window (s)")
    ap.add_argument("--affinity", default="", help="comma-separated logical CPUs to pin to")
    a = ap.parse_args()
    backend = a.backend or ("tflite" if a.model.lower().endswith(".tflite") else "onnx")
    if a.affinity:  # before any runtime thread pool exists, so its threads inherit the mask
        psutil.Process().cpu_affinity([int(c) for c in a.affinity.split(",")])

    x = np.load(a.input).astype(np.float32)
    rss_sampler = RssSampler()
    rss_sampler.start()
    if a.model == "none":
        runner = NullRunner(backend)
    elif backend == "tflite":
        runner = TFLiteRunner(a.model, a)
    else:
        runner = OrtRunner(a.model, a)
    runner.prepare(x)
    rss_after_load = psutil.Process().memory_info().rss

    for _ in range(a.warmup):
        runner.run()

    # Latencies go into a preallocated buffer: a growing Python list would itself add RSS
    # proportional to the run count and bias the memory comparison between models.
    # Timed window: >= --runs inferences AND >= --min-time seconds, because OS CPU-time
    # counters are coarse (15.6 ms ticks on Windows) and need a long window to be accurate.
    proc = psutil.Process()
    sampler = CpuSampler(a.sample_interval)
    sampler.start()
    lat = np.empty(a.runs, dtype=np.float64)
    t_cpu0, t_wall0 = sum(proc.cpu_times()[:2]), time.perf_counter()
    for i in range(a.runs):
        t0 = time.perf_counter()
        runner.run()
        lat[i] = (time.perf_counter() - t0) * 1e3
    n = a.runs
    while time.perf_counter() - t_wall0 < a.min_time:
        runner.run()
        n += 1
    wall = time.perf_counter() - t_wall0
    cpu = sum(proc.cpu_times()[:2]) - t_cpu0
    sampler.stop()
    session_peak = rss_sampler.stop()

    samples = [s for s in sampler.samples if s is not None]
    other = [s for s in sampler.other if s is not None]
    print(json.dumps({
        "backend": backend,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p90_ms": float(np.percentile(lat, 90)),
        "latency_mean_ms": float(lat.mean()),
        "n_runs": int(n),
        # process CPU% over the timed window (100 = one core busy); = time-average of the samples
        "cpu_util_pct": 100.0 * cpu / wall if wall > 0 else float("nan"),
        "cpu_util_pct_sampled_max": float(np.max(samples)) if samples else float("nan"),
        "cpu_ms_per_inf": 1e3 * cpu / n,
        "rss_after_load_mb": rss_after_load / 2**20,
        "peak_rss_mb": peak_rss_bytes() / 2**20,          # OS lifetime max RSS of the process
        "session_peak_rss_mb": session_peak / 2**20,      # max RSS growth from session creation on
        "n_cpu_samples": len(samples),
        # competing load during the timed window: 0 = we had the machine to ourselves
        "other_load_pct": float(np.mean(other)) if other else float("nan"),
        "other_load_pct_max": float(np.max(other)) if other else float("nan"),
    }))


if __name__ == "__main__":
    main()
