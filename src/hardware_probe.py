"""
hardware_probe.py — Collect and save hardware specifications.

Outputs results/raw/hardware_profile.json with:
  cpu_model, cpu_cores_physical, cpu_cores_logical,
  ram_total_gb, ram_available_gb, gpu_name, vram_gb,
  disk_type, disk_free_gb, os_name, os_version, python_version

Requires no model download. Run this first before any experiment.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


def _cpu_info() -> dict:
    info = {
        "cpu_model": platform.processor() or "unknown",
        "cpu_cores_physical": None,
        "cpu_cores_logical": None,
    }
    if _PSUTIL_AVAILABLE:
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
    return info


def _ram_info() -> dict:
    if _PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        return {
            "ram_total_gb": round(vm.total / 1e9, 2),
            "ram_available_gb": round(vm.available / 1e9, 2),
        }
    return {"ram_total_gb": None, "ram_available_gb": None}


def _gpu_info() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            parts = out.split(",")
            return {
                "gpu_name": parts[0].strip(),
                "vram_gb": round(float(parts[1].strip()) / 1024, 2),
                "gpu_available": True,
            }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return {
        "gpu_name": "N/A — no CUDA GPU detected",
        "vram_gb": None,
        "vram_note": "N/A — no CUDA/discrete GPU",
        "gpu_available": False,
    }


def _disk_info(path: str = ".") -> dict:
    if _PSUTIL_AVAILABLE:
        usage = psutil.disk_usage(path)
        return {
            "disk_free_gb": round(usage.free / 1e9, 2),
            "disk_total_gb": round(usage.total / 1e9, 2),
            "disk_path_checked": str(Path(path).resolve()),
        }
    return {"disk_free_gb": None, "disk_total_gb": None}


def _os_info() -> dict:
    return {
        "os_name": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "python_version": sys.version,
    }


def collect_profile(disk_path: str = ".") -> dict:
    profile = {}
    profile.update(_cpu_info())
    profile.update(_ram_info())
    profile.update(_gpu_info())
    profile.update(_disk_info(disk_path))
    profile.update(_os_info())
    profile["psutil_available"] = _PSUTIL_AVAILABLE

    if not _PSUTIL_AVAILABLE:
        profile["warning"] = (
            "psutil not installed — RAM and disk metrics are unavailable. "
            "Install with: pip install psutil"
        )
    return profile


def main():
    parser = argparse.ArgumentParser(description="Collect hardware profile.")
    parser.add_argument(
        "--output",
        default="results/raw/hardware_profile.json",
        help="Path to write JSON output (default: results/raw/hardware_profile.json)",
    )
    parser.add_argument(
        "--disk-path",
        default=".",
        help="Path to check for free disk space (default: current directory)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Collecting hardware profile...")
    profile = collect_profile(disk_path=args.disk_path)

    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Saved: {out_path}")
    print()
    for k, v in profile.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
