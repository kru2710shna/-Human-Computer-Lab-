"""Hardware description and resource measurement.

The challenge asks us to report hardware and observed resource requirements, so
this module is written to produce numbers we can defend rather than numbers that
flatter. Specifically:

  * RSS is sampled on a background thread, because peak memory during a VLM call
    happens mid-forward-pass and a before/after reading misses it entirely.
  * On Apple silicon the GPU shares system RAM, so torch's allocator figures and
    process RSS measure overlapping things. We report both and say so, rather
    than adding them together into a meaningless total.
  * Timings come from the backend, which synchronises the device first. Wall time
    around an unsynchronised MPS/CUDA call measures queue submission, not work.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil


# ------------------------------------------------------------------ hardware
def _mac_chip() -> str | None:
    if platform.system() != "Darwin" or not shutil.which("sysctl"):
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _accelerator() -> dict:
    info: dict = {"type": "cpu", "name": platform.processor() or "unknown"}
    try:
        import torch
    except ImportError:
        return info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info = {"type": "cuda", "name": props.name,
                "vram_bytes": props.total_memory,
                "capability": f"{props.major}.{props.minor}",
                "count": torch.cuda.device_count()}
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info = {"type": "mps", "name": _mac_chip() or "Apple silicon",
                "memory_model": "unified with system RAM"}
    else:
        info["name"] = _mac_chip() or info["name"]
    return info


def hardware_report() -> dict:
    vm = psutil.virtual_memory()
    rep = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_bytes": vm.total,
        "ram_available_bytes": vm.available,
        "accelerator": _accelerator(),
    }
    try:
        import torch
        rep["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import transformers
        rep["transformers"] = transformers.__version__
    except ImportError:
        pass
    return rep


# ------------------------------------------------------------- measurement
@dataclass
class ResourceSample:
    label: str
    wall_s: float = 0.0
    rss_start_bytes: int = 0
    rss_peak_bytes: int = 0
    rss_delta_bytes: int = 0
    torch_peak_bytes: int | None = None
    samples: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "wall_s": round(self.wall_s, 3),
            "rss_start_mb": round(self.rss_start_bytes / 1e6, 1),
            "rss_peak_mb": round(self.rss_peak_bytes / 1e6, 1),
            "rss_delta_mb": round(self.rss_delta_bytes / 1e6, 1),
            "rss_samples": self.samples,
        }
        if self.torch_peak_bytes is not None:
            d["torch_peak_alloc_mb"] = round(self.torch_peak_bytes / 1e6, 1)
        if self.notes:
            d["notes"] = self.notes
        return d


class _RSSSampler(threading.Thread):
    """Polls process RSS so we catch the peak inside a call, not around it."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.count = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()

    def run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
            except psutil.Error:
                break
            self.peak = max(self.peak, rss)
            self.count += 1
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)


@contextmanager
def measure(label: str = "run", interval: float = 0.05):
    """Measure wall time and peak RSS for a block. Yields a ResourceSample that
    is filled in on exit."""
    sample = ResourceSample(label=label)
    proc = psutil.Process()
    torch = None
    try:
        import torch as _t
        torch = _t
    except ImportError:
        pass

    dev = None
    if torch is not None:
        if torch.cuda.is_available():
            dev = "cuda"
            torch.cuda.reset_peak_memory_stats()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"

    sample.rss_start_bytes = proc.memory_info().rss
    sampler = _RSSSampler(interval)
    sampler.peak = sample.rss_start_bytes
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield sample
    finally:
        sample.wall_s = time.perf_counter() - t0
        sampler.stop()
        rss_end = proc.memory_info().rss
        sample.rss_peak_bytes = max(sampler.peak, rss_end)
        sample.rss_delta_bytes = sample.rss_peak_bytes - sample.rss_start_bytes
        sample.samples = sampler.count
        if torch is not None and dev == "cuda":
            sample.torch_peak_bytes = int(torch.cuda.max_memory_allocated())
        elif torch is not None and dev == "mps":
            try:
                sample.torch_peak_bytes = int(torch.mps.driver_allocated_memory())
                sample.notes.append(
                    "MPS memory is unified with system RAM, so torch_peak_alloc and RSS "
                    "describe overlapping pools and must not be summed."
                )
            except (AttributeError, RuntimeError):
                pass


def summarise(samples: list[ResourceSample]) -> dict:
    if not samples:
        return {}
    walls = sorted(s.wall_s for s in samples)
    n = len(walls)
    return {
        "n": n,
        "wall_s_mean": round(sum(walls) / n, 3),
        "wall_s_median": round(walls[n // 2], 3),
        "wall_s_p90": round(walls[min(n - 1, int(0.9 * n))], 3),
        "wall_s_max": round(walls[-1], 3),
        "rss_peak_mb_max": round(max(s.rss_peak_bytes for s in samples) / 1e6, 1),
    }


def dump(path, payload: dict) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))