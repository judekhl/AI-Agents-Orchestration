"""Capture installed package versions and import health; write environment_setup.json."""
import json
import subprocess
import sys
import datetime
import importlib
from pathlib import Path


def _version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "NOT_INSTALLED"
    except Exception as e:
        return f"ERROR: {e}"


def _try_import(module_name: str) -> dict:
    try:
        importlib.import_module(module_name)
        ver = _version(module_name)
        return {"status": "ok", "version": ver}
    except ImportError as e:
        return {"status": "not_installed", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _pip_list() -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        packages = json.loads(result.stdout)
        return {p["name"]: p["version"] for p in packages}
    except Exception as e:
        return {"error": str(e)}


def main(output_json: str = "results/raw/environment_setup.json",
         output_md: str = "results/processed/environment_summary.md"):

    pip_packages = _pip_list()

    core_imports = {
        "psutil":       _try_import("psutil"),
        "pandas":       _try_import("pandas"),
        "matplotlib":   _try_import("matplotlib"),
        "numpy":        _try_import("numpy"),
        "tqdm":         _try_import("tqdm"),
        "dotenv":       _try_import("dotenv"),
    }

    ml_imports = {
        "torch":        _try_import("torch"),
        "transformers": _try_import("transformers"),
        "accelerate":   _try_import("accelerate"),
        "safetensors":  _try_import("safetensors"),
        "sentencepiece": _try_import("sentencepiece"),
    }

    optional_imports = {
        "llama_cpp":    _try_import("llama_cpp"),
        "airllm":       _try_import("airllm"),
    }

    # Check torch device
    torch_device_info = "unknown"
    try:
        import torch
        torch_device_info = f"cpu available: True, cuda available: {torch.cuda.is_available()}"
    except Exception:
        pass

    data = {
        "probe_timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "venv_outside_onedrive": not sys.executable.startswith("C:\\Users\\HP\\OneDrive"),
        "torch_device_info": torch_device_info,
        "core_imports": core_imports,
        "ml_imports": ml_imports,
        "optional_imports": optional_imports,
        "installed_packages": pip_packages,
        "models_downloaded": False,
        "inference_run": False,
        "note": (
            "Environment probe only. No model weights downloaded. "
            "No inference run. Optional imports (llama_cpp, airllm) "
            "may be NOT_INSTALLED if wheels were unavailable."
        ),
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_json}")

    # Write markdown summary
    def status_icon(r):
        return "OK" if r.get("status") == "ok" else "FAIL"

    lines = [
        "# Environment Setup Summary",
        f"Generated: {data['probe_timestamp_utc']}",
        "",
        f"**Python:** `{sys.executable}`",
        f"**Venv outside OneDrive:** {data['venv_outside_onedrive']}",
        f"**Torch device info:** {torch_device_info}",
        "",
        "## Core Imports",
        "| Package | Status | Version |",
        "|---|---|---|",
    ]
    for pkg, res in core_imports.items():
        lines.append(f"| `{pkg}` | {status_icon(res)} | {res.get('version', res.get('error', ''))} |")

    lines += [
        "",
        "## ML Imports",
        "| Package | Status | Version |",
        "|---|---|---|",
    ]
    for pkg, res in ml_imports.items():
        lines.append(f"| `{pkg}` | {status_icon(res)} | {res.get('version', res.get('error', ''))} |")

    lines += [
        "",
        "## Optional Imports",
        "| Package | Status | Note |",
        "|---|---|---|",
    ]
    for pkg, res in optional_imports.items():
        note = res.get("version", res.get("error", ""))
        lines.append(f"| `{pkg}` | {status_icon(res)} | {note} |")

    lines += [
        "",
        "## Safety Confirmation",
        f"- Models downloaded: {data['models_downloaded']}",
        f"- Inference run: {data['inference_run']}",
        "",
        "_Raw data: `results/raw/environment_setup.json`_",
    ]

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {output_md}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", default="results/raw/environment_setup.json")
    p.add_argument("--output-md",   default="results/processed/environment_summary.md")
    args = p.parse_args()
    main(args.output_json, args.output_md)
