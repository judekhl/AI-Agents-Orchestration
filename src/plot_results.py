"""
plot_results.py — Generate all benchmark graphs and summary table from raw data.

Reads:  results/raw/*_metrics.json
Writes: figures/*.png
        results/processed/summary_table.csv

Graphs produced:
  - ttft_comparison.png
  - tpot_comparison.png
  - throughput_comparison.png
  - memory_comparison.png
  - runtime_comparison.png
  - quant_tradeoff.png
  - economic_breakeven.png

Usage:
    python src/plot_results.py --input-dir results/raw/ \
        --processed-dir results/processed/ \
        --output-dir figures/
"""

import argparse
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


SCENARIO_ORDER = ["baseline_warmup", "baseline", "airllm", "quant_fp32", "quant_fp16",
                  "quant_bf16", "quant_q8", "quant_q8_0", "quant_q4_k_m"]
SCENARIO_LABELS = {
    "baseline_warmup": "Baseline\n(0.5B FP16)",
    "baseline":        "Baseline\n(FP32, full load)",
    "airllm":          "AirLLM\n(layer paging)",
    "quant_fp32":      "Quant FP32",
    "quant_fp16":      "Quant FP16",
    "quant_bf16":      "Quant BF16",
    "quant_q8":        "Quant Q8",
    "quant_q8_0":      "Quant Q8_0\n(GGUF)",
    "quant_q4_k_m":    "Quant Q4_K_M\n(GGUF)",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
          "#937860", "#DA8BC3", "#8C8C8C"]


def load_all_metrics(input_dir: Path) -> list[dict]:
    records = []
    for path in sorted(input_dir.glob("*_metrics.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            row = {"scenario": data.get("scenario", path.stem)}
            m = data.get("metrics", {})
            row.update({
                "ttft_seconds":               m.get("ttft_seconds"),
                "tpot_ms":                    m.get("tpot_ms"),
                "throughput_tokens_per_sec":  m.get("throughput_tokens_per_sec"),
                "peak_ram_gb":                m.get("peak_ram_gb"),
                "total_runtime_seconds":      m.get("total_runtime_seconds"),
                "peak_vram_note":             m.get("peak_vram_note", "N/A"),
            })
            records.append(row)
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")
    return records


def _bar_chart(ax, scenarios, values, ylabel, title, colors):
    clean = [v if v is not None else 0 for v in values]
    bars = ax.bar(range(len(scenarios)), clean, color=colors[:len(scenarios)],
                  width=0.6, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, values):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)


def make_bar_figure(scenarios, values, ylabel, title, out_path):
    if all(v is None for v in values):
        print(f"  Skipping {Path(out_path).name} — all values are None")
        return
    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 1.4), 4))
    _bar_chart(ax, scenarios, values, ylabel, title, COLORS)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_quant_tradeoff(records, out_path):
    """Multi-panel: memory vs throughput vs TPOT for quantization configs."""
    quant = [r for r in records if r["scenario"].startswith("quant_")]
    if not quant:
        print("  No quantization records found — skipping quant_tradeoff.png")
        return

    scenarios = [r["scenario"] for r in quant]
    ram = [r["peak_ram_gb"] or 0 for r in quant]
    tpt = [r["throughput_tokens_per_sec"] or 0 for r in quant]
    tpot = [r["tpot_ms"] or 0 for r in quant]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, label in zip(axes,
                                [ram, tpt, tpot],
                                ["Peak RAM (GB)", "Throughput (tok/s)", "TPOT (ms/tok)"]):
        _bar_chart(ax, scenarios, vals, label, label, COLORS)
    fig.suptitle("Quantization Tradeoff: Memory vs Speed", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_economic_breakeven(econ_path: Path, out_path: Path):
    if not econ_path.exists():
        print(f"  No economic analysis file found at {econ_path} — skipping breakeven graph")
        return

    with open(econ_path) as f:
        econ = json.load(f)

    curve = econ.get("monthly_cost_curve", [])
    if not curve:
        print("  No cost curve data — skipping breakeven graph")
        return

    xs = [r["requests_per_month"] for r in curve]
    api = [r["api_cost_usd"] for r in curve]
    onprem = [r["on_prem_cost_usd"] for r in curve]
    breakeven = econ.get("breakeven_requests_per_month")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, api, "o-", label="API cost", color="#4C72B0", linewidth=2)
    ax.plot(xs, onprem, "s--", label="On-prem cost", color="#DD8452", linewidth=2)
    if breakeven and breakeven != float("inf"):
        ax.axvline(breakeven, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Break-even: {breakeven:.0f} req/month")
    ax.set_xlabel("Requests / month", fontsize=10)
    ax.set_ylabel("Monthly cost (USD)", fontsize=10)
    ax.set_title("API vs On-Premises: Monthly Cost Break-Even", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pricing_note = econ.get("pricing_note", "")
    if pricing_note:
        ax.text(0.01, -0.18, pricing_note, transform=ax.transAxes,
                fontsize=6, color="gray", wrap=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary_csv(records, out_path: Path):
    if not _PANDAS_AVAILABLE:
        print("  pandas not installed — skipping CSV summary")
        return
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"  Summary table saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate all benchmark plots.")
    parser.add_argument("--input-dir", default="results/raw/")
    parser.add_argument("--processed-dir", default="results/processed/")
    parser.add_argument("--output-dir", default="figures/")
    args = parser.parse_args()

    if not _MPL_AVAILABLE:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return

    in_dir = Path(args.input_dir)
    proc_dir = Path(args.processed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")
    records = load_all_metrics(in_dir)
    if not records:
        print(f"No *_metrics.json files found in {in_dir}")
        print("Run experiments first.")
        return

    # Sort by canonical order
    order = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    records.sort(key=lambda r: order.get(r["scenario"], 99))

    scenarios = [r["scenario"] for r in records]

    print(f"\nGenerating plots for {len(records)} scenarios...")

    make_bar_figure(scenarios, [r["ttft_seconds"] for r in records],
                    "TTFT (seconds)", "Time To First Token by Scenario",
                    out_dir / "ttft_comparison.png")

    make_bar_figure(scenarios, [r["tpot_ms"] for r in records],
                    "TPOT (ms / token)", "Time Per Output Token by Scenario",
                    out_dir / "tpot_comparison.png")

    make_bar_figure(scenarios, [r["throughput_tokens_per_sec"] for r in records],
                    "Throughput (tokens/sec)", "Throughput by Scenario",
                    out_dir / "throughput_comparison.png")

    make_bar_figure(scenarios, [r["peak_ram_gb"] for r in records],
                    "Peak RAM (GB)", "Peak RAM Usage by Scenario",
                    out_dir / "memory_comparison.png")

    make_bar_figure(scenarios, [r["total_runtime_seconds"] for r in records],
                    "Runtime (seconds)", "Total Inference Runtime by Scenario",
                    out_dir / "runtime_comparison.png")

    make_quant_tradeoff(records, out_dir / "quant_tradeoff.png")

    make_economic_breakeven(
        proc_dir / "economic_analysis.json",
        out_dir / "economic_breakeven.png",
    )

    save_summary_csv(records, proc_dir / "summary_table.csv")
    print("\nAll plots complete.")


if __name__ == "__main__":
    main()
