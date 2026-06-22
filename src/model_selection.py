"""
model_selection.py — Evaluate candidate models against hardware constraints.

Supports three experiment roles:
  warm_up  — small model that verifies the pipeline works without stressing RAM
  stress   — large model that exceeds RAM in FP16, producing a documentable OOM
  fallback — quantized (GGUF) model that fits in RAM for successful throughput measurement

Does NOT download any model. Requires no GPU or heavy ML packages.
Run after hardware_probe.py.

Usage (evaluate all three roles and save reports):
    python src/model_selection.py \
        --hardware-profile results/raw/hardware_profile.json \
        --output-json results/raw/model_selection.json \
        --output-md results/processed/model_selection.md
"""

import argparse
import json
import sys
from pathlib import Path

# ── Model catalogue ────────────────────────────────────────────────────────────
# format:   "hf_repo_id": { params_b, format, requires_token, ... }
# format="hf"   → standard HuggingFace safetensors/bin weights
# format="gguf" → single-file GGUF; sizes vary by quantization level
MODEL_CATALOGUE = {
    # ── Warm-up candidates ──
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "params_b": 0.5,
        "format": "hf",
        "fp16_disk_gb": 1.0,
        "fp16_ram_gb": 1.5,
        "fp32_disk_gb": 2.0,
        "requires_token": False,
        "role_hint": "warm_up",
        "notes": (
            "0.5B params. Fits easily in 8 GB RAM. Public, no HF token. "
            "Used to verify the inference pipeline end-to-end before attempting the stress model."
        ),
    },
    "facebook/opt-1.3b": {
        "params_b": 1.3,
        "format": "hf",
        "fp16_disk_gb": 2.6,
        "fp16_ram_gb": 3.0,
        "fp32_disk_gb": 5.2,
        "requires_token": False,
        "role_hint": "warm_up",
        "notes": "1.3B params. Public. Alternate warm-up if Qwen 0.5B is unavailable.",
    },

    # ── Stress candidates ──
    "facebook/opt-6.7b": {
        "params_b": 6.7,
        "format": "hf",
        "fp16_disk_gb": 13.4,
        "fp16_ram_gb": 14.0,
        "fp32_disk_gb": 26.8,
        "requires_token": False,
        "role_hint": "stress",
        "notes": (
            "6.7B params. FP16 needs ~14 GB RAM — exceeds 8.22 GB total on this machine. "
            "Naive baseline load will OOM. Public, no token. "
            "AirLLM can shard this to ~0.4 GB peak RAM (one layer at a time)."
        ),
    },
    "EleutherAI/gpt-j-6b": {
        "params_b": 6.1,
        "format": "hf",
        "fp16_disk_gb": 12.1,
        "fp16_ram_gb": 12.5,
        "fp32_disk_gb": 24.2,
        "requires_token": False,
        "role_hint": "stress",
        "notes": "6.1B params. Public. Alternate stress model; same RAM situation as OPT-6.7B.",
    },

    # ── Fallback GGUF candidates ──
    "bartowski/Qwen2.5-7B-Instruct-GGUF": {
        "params_b": 7.0,
        "format": "gguf",
        "requires_token": False,
        "role_hint": "fallback",
        "gguf_variants": {
            "Q4_K_M": {"disk_gb": 4.4, "ram_gb": 4.8},
            "Q6_K":   {"disk_gb": 5.7, "ram_gb": 6.1},
            "Q8_0":   {"disk_gb": 7.7, "ram_gb": 8.2},
        },
        "recommended_quant": "Q4_K_M",
        "notes": (
            "7B Qwen2.5-Instruct in GGUF format. bartowski is a reliable GGUF publisher. "
            "Q4_K_M: ~4.4 GB disk, ~4.8 GB RAM — fits in 8.22 GB. "
            "Q8_0: ~7.7 GB disk, ~8.2 GB RAM — extremely tight, may trigger swap. "
            "Enables successful throughput measurement where baseline OOMs. "
            "Inference via llama-cpp-python; no transformers/torch required."
        ),
    },
    "Qwen/Qwen2.5-7B-Instruct-GGUF": {
        "params_b": 7.0,
        "format": "gguf",
        "requires_token": False,
        "role_hint": "fallback",
        "gguf_variants": {
            "Q4_K_M": {"disk_gb": 4.4, "ram_gb": 4.8},
            "Q8_0":   {"disk_gb": 7.7, "ram_gb": 8.2},
        },
        "recommended_quant": "Q4_K_M",
        "notes": (
            "Official Qwen2.5-7B GGUF repo. Same weights as bartowski variant. "
            "Alternate source if bartowski repo is unavailable."
        ),
    },
}

# ── Selected roles ─────────────────────────────────────────────────────────────
SELECTED = {
    "warm_up": "Qwen/Qwen2.5-0.5B-Instruct",
    "stress":  "facebook/opt-6.7b",
    "fallback": "bartowski/Qwen2.5-7B-Instruct-GGUF",
}
SELECTED_FALLBACK_QUANT = "Q4_K_M"


def _check_hf_model(model_id: str, meta: dict, hw: dict) -> dict:
    ram_total = hw.get("ram_total_gb", 0)
    disk_free = hw.get("disk_free_gb", 0)
    required_ram = meta["fp16_ram_gb"]
    required_disk = meta["fp16_disk_gb"]

    ram_ok = ram_total >= required_ram
    disk_ok = disk_free >= required_disk * 1.1  # 10% buffer

    issues = []
    if not ram_ok:
        issues.append(
            f"FP16 needs {required_ram} GB RAM; only {ram_total} GB available. "
            "OOM expected on naive load — valid negative result."
        )
    if not disk_ok:
        issues.append(
            f"FP16 needs {required_disk:.1f} GB disk (+ 10% buffer); "
            f"only {disk_free} GB free."
        )

    return {
        "model_id": model_id,
        "format": "hf",
        "params_b": meta["params_b"],
        "fp16_disk_gb": required_disk,
        "fp16_ram_gb": required_ram,
        "requires_token": meta["requires_token"],
        "role": meta.get("role_hint"),
        "feasible_fp16": ram_ok and disk_ok,
        "ram_ok": ram_ok,
        "disk_ok": disk_ok,
        "issues": issues,
        "notes": meta["notes"],
    }


def _check_gguf_model(model_id: str, meta: dict, hw: dict, quant: str) -> dict:
    ram_total = hw.get("ram_total_gb", 0)
    disk_free = hw.get("disk_free_gb", 0)
    variant = meta["gguf_variants"].get(quant, {})
    required_ram = variant.get("ram_gb", 0)
    required_disk = variant.get("disk_gb", 0)

    ram_ok = ram_total >= required_ram
    disk_ok = disk_free >= required_disk * 1.1

    issues = []
    if not ram_ok:
        issues.append(
            f"{quant} needs {required_ram} GB RAM; only {ram_total} GB available."
        )
    if not disk_ok:
        issues.append(
            f"{quant} needs {required_disk:.1f} GB disk (+ 10% buffer); "
            f"only {disk_free} GB free."
        )

    return {
        "model_id": model_id,
        "format": "gguf",
        "params_b": meta["params_b"],
        "selected_quant": quant,
        "gguf_disk_gb": required_disk,
        "gguf_ram_gb": required_ram,
        "all_variants": meta["gguf_variants"],
        "requires_token": meta["requires_token"],
        "role": meta.get("role_hint"),
        "feasible": ram_ok and disk_ok,
        "ram_ok": ram_ok,
        "disk_ok": disk_ok,
        "issues": issues,
        "notes": meta["notes"],
    }


def evaluate_roles(hw: dict) -> dict:
    results = {}
    for role, model_id in SELECTED.items():
        meta = MODEL_CATALOGUE[model_id]
        if meta["format"] == "gguf":
            results[role] = _check_gguf_model(
                model_id, meta, hw, SELECTED_FALLBACK_QUANT
            )
        else:
            results[role] = _check_hf_model(model_id, meta, hw)
    return results


def disk_budget(hw: dict, roles: dict) -> dict:
    free = hw.get("disk_free_gb", 0)
    warm_disk = roles["warm_up"].get("fp16_disk_gb", 0)
    stress_disk = roles["stress"].get("fp16_disk_gb", 0)
    airllm_shards = stress_disk          # AirLLM stores a copy of shards
    fallback_disk = roles["fallback"].get("gguf_disk_gb", 0)
    total_needed = warm_disk + stress_disk + airllm_shards + fallback_disk

    return {
        "disk_free_gb": free,
        "warm_up_fp16_gb": warm_disk,
        "stress_fp16_gb": stress_disk,
        "airllm_shards_gb": airllm_shards,
        "fallback_gguf_gb": fallback_disk,
        "total_needed_gb": round(total_needed, 2),
        "remaining_after_downloads_gb": round(free - total_needed, 2),
        "feasible": free >= total_needed * 1.05,
        "note": (
            "AirLLM shards the stress model into per-layer files in a separate directory. "
            "Budget assumes both original weights and shards coexist on disk during sharding. "
            "After sharding completes, the original checkpoint can be deleted to reclaim space."
        ),
    }


def build_report(hw: dict) -> dict:
    roles = evaluate_roles(hw)
    budget = disk_budget(hw, roles)
    return {
        "hardware_summary": {
            "ram_total_gb": hw.get("ram_total_gb"),
            "ram_available_gb_at_probe": hw.get("ram_available_gb_at_probe"),
            "gpu_available": hw.get("gpu_available"),
            "disk_free_gb": hw.get("disk_free_gb"),
            "disk_type": hw.get("disk_type"),
            "cpu_model_name": hw.get("cpu_model_name"),
        },
        "selected_models": SELECTED,
        "selected_fallback_quant": SELECTED_FALLBACK_QUANT,
        "role_assessments": roles,
        "disk_budget": budget,
        "assignment_rationale": {
            "warm_up": (
                "Qwen2.5-0.5B-Instruct (0.5B params, ~1.5 GB RAM in FP16) proves the "
                "inference pipeline works on this hardware before attempting the stress model. "
                "Provides a meaningful but non-stressful timing baseline."
            ),
            "stress": (
                "OPT-6.7B (6.7B params, ~14 GB RAM in FP16) far exceeds the 8.22 GB RAM "
                "ceiling. Naive baseline load will trigger OOM — this is the expected, "
                "documentable negative result that motivates AirLLM and quantization. "
                "AirLLM can run this model by loading one layer (~0.4 GB) at a time."
            ),
            "fallback": (
                "Qwen2.5-7B-Instruct Q4_K_M GGUF (~4.4 GB disk, ~4.8 GB RAM) fits within "
                "8.22 GB RAM and enables a successful throughput measurement. "
                "Q4_K_M is the recommended balance of compression and quality. "
                "Inference uses llama-cpp-python; no torch or transformers needed."
            ),
        },
    }


def write_markdown(report: dict, out_path: Path):
    hw = report["hardware_summary"]
    roles = report["role_assessments"]
    budget = report["disk_budget"]
    rationale = report["assignment_rationale"]

    def feasible_str(r: dict) -> str:
        if r.get("feasible_fp16") is True or r.get("feasible") is True:
            return "✓ Fits"
        return "✗ OOM expected"

    lines = [
        "# Model Selection Report",
        f"Generated: 2026-06-23",
        f"Hardware: {hw['cpu_model_name']}, {hw['ram_total_gb']} GB RAM, "
        f"no GPU, {hw['disk_free_gb']} GB free ({hw['disk_type']})",
        "",
        "## Selected Models",
        "",
        "| Role | Model | Format | RAM needed | Fits in 8.22 GB RAM? |",
        "|---|---|---|---|---|",
        f"| Warm-up | `{SELECTED['warm_up']}` | HF FP16 | "
        f"~{roles['warm_up']['fp16_ram_gb']} GB | {feasible_str(roles['warm_up'])} |",
        f"| Stress | `{SELECTED['stress']}` | HF FP16 | "
        f"~{roles['stress']['fp16_ram_gb']} GB | {feasible_str(roles['stress'])} |",
        f"| Fallback | `{SELECTED['fallback']}` | GGUF {SELECTED_FALLBACK_QUANT} | "
        f"~{roles['fallback']['gguf_ram_gb']} GB | {feasible_str(roles['fallback'])} |",
        "",
        "## Disk Budget",
        "",
        "| Item | GB |",
        "|---|---|",
        f"| Disk free at probe | {budget['disk_free_gb']} |",
        f"| Warm-up model (FP16) | {budget['warm_up_fp16_gb']} |",
        f"| Stress model (FP16) | {budget['stress_fp16_gb']} |",
        f"| AirLLM shards (stress model copy) | {budget['airllm_shards_gb']} |",
        f"| Fallback GGUF ({SELECTED_FALLBACK_QUANT}) | {budget['fallback_gguf_gb']} |",
        f"| **Total needed** | **{budget['total_needed_gb']}** |",
        f"| **Remaining after all downloads** | **{budget['remaining_after_downloads_gb']}** |",
        f"| Budget feasible? | {'✓ Yes' if budget['feasible'] else '✗ No — reduce scope'} |",
        "",
        f"> {budget['note']}",
        "",
        "## Role Rationale",
        "",
        f"**Warm-up:** {rationale['warm_up']}",
        "",
        f"**Stress:** {rationale['stress']}",
        "",
        f"**Fallback:** {rationale['fallback']}",
        "",
        "## Feasibility Details",
        "",
    ]

    for role, assessment in roles.items():
        lines.append(f"### {role.replace('_', '-').title()} — `{assessment['model_id']}`")
        issues = assessment.get("issues", [])
        if issues:
            for issue in issues:
                lines.append(f"- ⚠ {issue}")
        else:
            lines.append("- No feasibility issues.")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate candidate models against hardware constraints."
    )
    parser.add_argument(
        "--hardware-profile",
        default="results/raw/hardware_profile.json",
    )
    parser.add_argument(
        "--output-json",
        default="results/raw/model_selection.json",
    )
    parser.add_argument(
        "--output-md",
        default="results/processed/model_selection.md",
    )
    args = parser.parse_args()

    hw_path = Path(args.hardware_profile)
    if not hw_path.exists():
        print(f"ERROR: {hw_path} not found. Run hardware_probe.py first.")
        sys.exit(1)

    with open(hw_path) as f:
        hw = json.load(f)

    report = build_report(hw)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    write_markdown(report, Path(args.output_md))

    print("=== Model Selection ===")
    for role, model_id in SELECTED.items():
        a = report["role_assessments"][role]
        feasible = a.get("feasible_fp16", a.get("feasible"))
        status = "GO" if feasible else "OOM expected (intended)"
        fmt = a.get("format", "?")
        if fmt == "gguf":
            size = f"{a['gguf_ram_gb']} GB RAM ({SELECTED_FALLBACK_QUANT})"
        else:
            size = f"{a['fp16_ram_gb']} GB RAM (FP16)"
        print(f"  {role:10s}: {model_id}")
        print(f"             {size} — {status}")
        for issue in a.get("issues", []):
            print(f"             WARNING: {issue}")

    budget = report["disk_budget"]
    print(f"\nDisk budget: {budget['total_needed_gb']} GB needed / "
          f"{budget['disk_free_gb']} GB free -> "
          f"{budget['remaining_after_downloads_gb']} GB remaining")
    print(f"Budget OK: {'yes' if budget['feasible'] else 'NO — reduce scope'}")
    print(f"\nSaved: {out_json}")
    print(f"Saved: {args.output_md}")


if __name__ == "__main__":
    main()
