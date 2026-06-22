"""
model_selection.py — Document and validate the model choice for experiments.

Run after hardware_probe.py to check whether the chosen model fits within
available RAM and disk space. Prints a go/no-go recommendation.

Does NOT download the model. Requires no GPU or heavy dependencies.
"""

import argparse
import json
from pathlib import Path

# ── Model catalogue ───────────────────────────────────────────────────────────
# Add entries here as needed. Sizes are approximate FP16 disk footprints.
MODEL_CATALOGUE = {
    "facebook/opt-6.7b": {
        "params_b": 6.7,
        "fp32_disk_gb": 26.8,
        "fp16_disk_gb": 13.4,
        "fp16_ram_gb": 14.0,
        "requires_token": False,
        "notes": "Public, no auth required. Good baseline stress test for 16 GB RAM.",
    },
    "facebook/opt-13b": {
        "params_b": 13.0,
        "fp32_disk_gb": 52.0,
        "fp16_disk_gb": 26.0,
        "fp16_ram_gb": 27.0,
        "requires_token": False,
        "notes": "Public, no auth. Will OOM on 16 GB RAM — documents baseline failure.",
    },
    "EleutherAI/gpt-j-6b": {
        "params_b": 6.1,
        "fp32_disk_gb": 24.2,
        "fp16_disk_gb": 12.1,
        "fp16_ram_gb": 12.5,
        "requires_token": False,
        "notes": "Public, no auth. Similar scale to OPT-6.7B.",
    },
    "EleutherAI/gpt-neox-20b": {
        "params_b": 20.0,
        "fp32_disk_gb": 80.0,
        "fp16_disk_gb": 40.0,
        "fp16_ram_gb": 42.0,
        "requires_token": False,
        "notes": "Public, no auth. 20B — stresses even 64 GB RAM machines.",
    },
    "meta-llama/Llama-2-7b-hf": {
        "params_b": 7.0,
        "fp32_disk_gb": 28.0,
        "fp16_disk_gb": 14.0,
        "fp16_ram_gb": 14.5,
        "requires_token": True,
        "notes": "Gated — requires HF token and Meta license acceptance.",
    },
}

# Default fallback: public, no token, stresses a 16 GB system
DEFAULT_PRIMARY_MODEL = "facebook/opt-6.7b"
DEFAULT_FALLBACK_MODEL = "EleutherAI/gpt-j-6b"


def check_feasibility(model_id: str, hardware_profile: dict) -> dict:
    """Return a go/no-go assessment given hardware_profile.json contents."""
    meta = MODEL_CATALOGUE.get(model_id)
    if meta is None:
        return {
            "model_id": model_id,
            "status": "UNKNOWN",
            "message": f"Model not in catalogue. Add an entry to MODEL_CATALOGUE.",
        }

    ram_gb = hardware_profile.get("ram_total_gb")
    disk_free_gb = hardware_profile.get("disk_free_gb")
    required_ram = meta["fp16_ram_gb"]
    required_disk = meta["fp16_disk_gb"]

    issues = []
    if ram_gb is not None and ram_gb < required_ram:
        issues.append(
            f"RAM insufficient: {ram_gb} GB available, {required_ram} GB required for FP16. "
            f"Expect OOM or slow OS swap. This is a valid negative result — document it."
        )
    if disk_free_gb is not None and disk_free_gb < required_disk * 1.2:
        issues.append(
            f"Disk space tight: {disk_free_gb} GB free, {required_disk} GB required "
            f"(with 20% buffer = {required_disk * 1.2:.1f} GB)."
        )

    return {
        "model_id": model_id,
        "params_b": meta["params_b"],
        "fp16_disk_gb": required_disk,
        "fp16_ram_gb": required_ram,
        "requires_token": meta["requires_token"],
        "notes": meta["notes"],
        "ram_available_gb": ram_gb,
        "disk_free_gb": disk_free_gb,
        "status": "WARNING" if issues else "GO",
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate model choice against hardware.")
    parser.add_argument(
        "--hardware-profile",
        default="results/raw/hardware_profile.json",
        help="Path to hardware_profile.json from hardware_probe.py",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_PRIMARY_MODEL,
        help=f"HuggingFace model ID to check (default: {DEFAULT_PRIMARY_MODEL})",
    )
    parser.add_argument(
        "--output",
        default="results/raw/model_selection.json",
        help="Path to write selection report JSON",
    )
    args = parser.parse_args()

    hardware_path = Path(args.hardware_profile)
    if not hardware_path.exists():
        print(f"ERROR: {hardware_path} not found. Run hardware_probe.py first.")
        raise SystemExit(1)

    with open(hardware_path) as f:
        hw = json.load(f)

    report = {
        "primary_model": check_feasibility(args.model_id, hw),
        "fallback_model": check_feasibility(DEFAULT_FALLBACK_MODEL, hw),
        "catalogue": {mid: meta for mid, meta in MODEL_CATALOGUE.items()},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Primary model: {args.model_id}")
    print(f"  Status: {report['primary_model']['status']}")
    for issue in report["primary_model"].get("issues", []):
        print(f"  WARNING: {issue}")
    print(f"\nFallback model: {DEFAULT_FALLBACK_MODEL}")
    print(f"  Status: {report['fallback_model']['status']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
