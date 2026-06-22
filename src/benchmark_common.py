"""
benchmark_common.py — Shared timing, memory sampling, and result-saving utilities.

All experiment scripts import from here to ensure consistent metric definitions:
  - TTFT:       time from inference start to first output token
  - TPOT:       (total_gen_time - ttft) / (total_tokens - 1), in milliseconds
  - Throughput: total_output_tokens / total_runtime_seconds
  - Peak RAM:   maximum resident set size during inference (GB)
  - Runtime:    wall-clock time from first forward pass call to final token
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ── Memory sampler ─────────────────────────────────────────────────────────────

class RamSampler:
    """Sample process RSS every `interval` seconds in a background thread."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if not _PSUTIL_AVAILABLE:
            print("WARNING: psutil not available — RAM sampling disabled.")

    def start(self):
        if not _PSUTIL_AVAILABLE:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        """Stop sampling and return peak RAM in GB."""
        if not _PSUTIL_AVAILABLE or self._thread is None:
            return 0.0
        self._stop.set()
        self._thread.join(timeout=5.0)
        return max(self._samples) if self._samples else 0.0

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / 1e9
                self._samples.append(rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)


# ── Disk I/O sampler ────────────────────────────────────────────────────────────

class DiskIoSampler:
    """Sample system-wide disk read bytes at `interval` intervals."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not _PSUTIL_AVAILABLE:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        if not _PSUTIL_AVAILABLE or self._thread is None:
            return []
        self._stop.set()
        self._thread.join(timeout=5.0)
        return self.samples

    def _run(self):
        prev = psutil.disk_io_counters()
        prev_time = time.perf_counter()
        while not self._stop.is_set():
            time.sleep(self.interval)
            curr = psutil.disk_io_counters()
            curr_time = time.perf_counter()
            if curr is None or prev is None:
                continue
            dt = curr_time - prev_time
            read_mb_s = (curr.read_bytes - prev.read_bytes) / 1e6 / dt
            self.samples.append({
                "t": round(curr_time, 3),
                "read_mb_s": round(read_mb_s, 3),
                "read_bytes_total": curr.read_bytes,
            })
            prev = curr
            prev_time = curr_time


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(
    ttft_seconds: float,
    total_output_tokens: int,
    total_runtime_seconds: float,
    peak_ram_gb: float,
    gpu_available: bool = False,
    peak_vram_gb: Optional[float] = None,
    power_estimate_w: Optional[float] = None,
) -> dict:
    """Compute derived metrics from raw measurements."""
    tpot_ms = None
    throughput = None

    if total_output_tokens > 1:
        decode_time = total_runtime_seconds - ttft_seconds
        if decode_time > 0:
            tpot_ms = round((decode_time / (total_output_tokens - 1)) * 1000, 3)

    if total_runtime_seconds > 0:
        throughput = round(total_output_tokens / total_runtime_seconds, 4)

    vram_note = None if gpu_available else "N/A — no CUDA/discrete GPU"

    metrics = {
        "ttft_seconds": round(ttft_seconds, 4),
        "tpot_ms": tpot_ms,
        "throughput_tokens_per_sec": throughput,
        "total_runtime_seconds": round(total_runtime_seconds, 4),
        "total_output_tokens": total_output_tokens,
        "peak_ram_gb": round(peak_ram_gb, 3),
        "peak_vram_gb": peak_vram_gb,
        "peak_vram_note": vram_note,
    }

    if power_estimate_w is not None:
        kwh = (power_estimate_w / 1000) * (total_runtime_seconds / 3600)
        metrics["power_estimate_w"] = round(power_estimate_w, 1)
        metrics["energy_kwh"] = round(kwh, 6)

    return metrics


# ── Result persistence ─────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: str, scenario: str, model_id: str,
                 extra: Optional[dict] = None):
    """Save metrics dict to a JSON file. Never overwrites — appends timestamp."""
    out = {
        "scenario": scenario,
        "model_id": model_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": metrics,
    }
    if extra:
        out["extra"] = extra

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Metrics saved: {out_path}")


def save_failure(error: Exception, path: str, scenario: str, context: str = ""):
    """Save failure information so negative results are documented."""
    import traceback
    record = {
        "scenario": scenario,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "context": context,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Failure log saved: {out_path}")


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_env_config() -> dict:
    """Load experiment parameters from environment variables (set via .env)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return {
        "model_id": os.getenv("MODEL_ID", "facebook/opt-6.7b"),
        "model_cache_dir": os.getenv("MODEL_CACHE_DIR", "./models"),
        "airllm_shard_dir": os.getenv("AIRLLM_SHARD_DIR", "./airllm_shards"),
        "airllm_cache_dir": os.getenv("AIRLLM_CACHE_DIR", "./airllm_cache"),
        "device": os.getenv("DEVICE", "cpu"),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "200")),
        "n_runs": int(os.getenv("N_RUNS", "3")),
        "hf_token": os.getenv("HF_TOKEN") or None,
        "cpu_tdp_watts": float(os.getenv("CPU_TDP_WATTS", "45")),
    }
