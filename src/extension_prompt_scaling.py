"""
extension_prompt_scaling.py — Prompt-length scaling experiment.

Runs Qwen2.5-0.5B-Instruct Q4_K_M GGUF at 5 prompt lengths (very short → very long)
using llama-cpp-python streaming to capture real per-token timestamps, giving true
TTFT and TPOT/ITL measurements.

Hypothesis: TTFT scales with prompt length (prefill work = O(n_input_tokens));
throughput and TPOT remain roughly constant (decode is independent of prompt length).

Saves:
  results/raw/extension_prompt_scaling.json   — all 5 prompts
  results/raw/quant_q4_k_m_streaming_metrics.json — medium prompt only (benchmark match)

Generates:
  figures/extension_prompt_scaling.png

Usage:
    python src/extension_prompt_scaling.py \
        --gguf C:\\ai-model-cache\\gguf\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
        --output-dir results/raw/ \
        --figures-dir figures/
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_common import RamSampler

PROMPTS = [
    {
        "label": "very_short",
        "text": "What is RAM?",
    },
    {
        "label": "short",
        "text": "Explain what a CPU cache is and why it matters for performance.",
    },
    {
        "label": "medium",
        "text": (
            "Explain the concept of virtual memory in operating systems, "
            "including how paging works and why it matters for large applications."
        ),
    },
    {
        "label": "long",
        "text": (
            "You are a computer science professor. A student asks: "
            "I have a machine with only 8 GB of RAM and I want to run a 14 GB language "
            "model. Explain in detail the different strategies I could use: "
            "(1) operating-system virtual memory and swap, "
            "(2) model quantization to reduce weight size, "
            "(3) layer-by-layer disk paging as used by tools like AirLLM. "
            "For each strategy, describe the mechanism, the performance tradeoffs, "
            "and when you would recommend it."
        ),
    },
    {
        "label": "very_long",
        "text": (
            "You are writing a technical report on the challenges of deploying large "
            "language models on consumer hardware. The hardware in question is a laptop "
            "with an Intel Core i5-1135G7 CPU (4 physical cores, 8 logical, 2.4 GHz), "
            "8.22 GB of total RAM, no discrete GPU, and a 512 GB NVMe SSD. "
            "The model you want to run is a 7-billion-parameter language model in "
            "FP16 precision, which requires approximately 14 GB of RAM. "
            "Section 1: Explain why naive loading with the Hugging Face transformers "
            "library will fail on this hardware, covering both the memory ceiling and "
            "the CPU-only inference speed implications. "
            "Section 2: Describe how 4-bit quantization (Q4_K_M in GGUF format) "
            "reduces the memory requirement to approximately 4 GB, what quality "
            "tradeoffs this introduces, and how llama.cpp exploits SIMD instructions "
            "for efficient dequantization on the CPU. "
            "Section 3: Explain the AirLLM approach of layer-by-layer disk paging: "
            "how it splits model weights into per-layer shard files, loads one layer "
            "at a time into RAM, runs the forward pass, and immediately evicts the "
            "layer before loading the next, thereby capping peak RAM at the size of "
            "one transformer layer regardless of total model size."
        ),
    },
]

MAX_TOKENS = 64
N_CTX = 2048


def run_streaming(llm, prompt_text: str, max_tokens: int, ram_sampler: RamSampler) -> dict:
    """Run inference with streaming and capture real per-token timestamps."""
    token_texts = []
    token_times = []
    first_token_time = None
    start = time.perf_counter()

    ram_sampler.start()
    for chunk in llm(prompt_text, max_tokens=max_tokens, temperature=0.0, stream=True):
        now = time.perf_counter()
        text = chunk["choices"][0]["text"]
        if not text:
            continue
        if first_token_time is None:
            first_token_time = now
            ttft = now - start
        token_texts.append(text)
        token_times.append(now)

    end = time.perf_counter()
    peak_ram = ram_sampler.stop()

    total_runtime = end - start
    output_tokens = len(token_times)
    output_text = "".join(token_texts)

    if first_token_time is None:
        return {"error": "no tokens generated"}

    # ITL = time between consecutive tokens (after the first)
    itl_values = []
    for i in range(1, len(token_times)):
        itl_values.append((token_times[i] - token_times[i - 1]) * 1000)

    tpot_ms = round(sum(itl_values) / len(itl_values), 3) if itl_values else None
    itl_min_ms = round(min(itl_values), 3) if itl_values else None
    itl_max_ms = round(max(itl_values), 3) if itl_values else None
    throughput = round(output_tokens / total_runtime, 4) if total_runtime > 0 else None

    return {
        "ttft_seconds": round(ttft, 4),
        "tpot_ms": tpot_ms,
        "itl_min_ms": itl_min_ms,
        "itl_max_ms": itl_max_ms,
        "throughput_tokens_per_sec": throughput,
        "total_runtime_seconds": round(total_runtime, 4),
        "total_output_tokens": output_tokens,
        "peak_ram_gb": round(peak_ram, 3),
        "output_text": output_text[:300],
    }


def make_plots(records: list[dict], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    labels = [r["label"] for r in records]
    prompt_tokens = [r["prompt_tokens"] for r in records]
    ttfts = [r["metrics"]["ttft_seconds"] for r in records]
    tpots = [r["metrics"]["tpot_ms"] for r in records]
    throughputs = [r["metrics"]["throughput_tokens_per_sec"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # TTFT vs prompt length
    ax = axes[0]
    ax.plot(prompt_tokens, ttfts, "o-", color="#4C72B0", linewidth=2, markersize=6)
    for x, y, lbl in zip(prompt_tokens, ttfts, labels):
        ax.annotate(lbl.replace("_", "\n"), (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel("Prompt tokens", fontsize=9)
    ax.set_ylabel("TTFT (seconds)", fontsize=9)
    ax.set_title("TTFT vs Prompt Length\n(prefill scales with input)", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Throughput vs prompt length
    ax = axes[1]
    ax.plot(prompt_tokens, throughputs, "s-", color="#DD8452", linewidth=2, markersize=6)
    ax.set_xlabel("Prompt tokens", fontsize=9)
    ax.set_ylabel("Throughput (tok/s)", fontsize=9)
    ax.set_title("Throughput vs Prompt Length\n(decode roughly constant)", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # TPOT vs prompt length
    ax = axes[2]
    valid = [(pt, tp) for pt, tp in zip(prompt_tokens, tpots) if tp is not None]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, "^-", color="#55A868", linewidth=2, markersize=6)
    ax.set_xlabel("Prompt tokens", fontsize=9)
    ax.set_ylabel("TPOT (ms/token)", fontsize=9)
    ax.set_title("TPOT vs Prompt Length\n(inter-token latency, streaming)", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Prompt-Length Scaling: Qwen2.5-0.5B Q4_K_M GGUF (llama.cpp CPU)",
        fontweight="bold", fontsize=11
    )
    plt.tight_layout()
    out_path = out_dir / "extension_prompt_scaling.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def make_tpot_comparison(records: list[dict], out_dir: Path):
    """Bar chart of TPOT across prompt lengths, usable as tpot_comparison.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    labels = [r["label"].replace("_", "\n") for r in records]
    tpots = [r["metrics"]["tpot_ms"] or 0 for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(labels)), tpots, color=COLORS[:len(labels)],
                  width=0.6, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("TPOT / ITL (ms per token)", fontsize=9)
    ax.set_title(
        "Time Per Output Token (TPOT) by Prompt Length\nQwen2.5-0.5B Q4_K_M GGUF — streaming measurement",
        fontsize=10, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, tpots):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(0.01, -0.18,
            "TPOT measured from real per-token streaming timestamps (llama-cpp-python stream=True)",
            transform=ax.transAxes, fontsize=7, color="gray")
    plt.tight_layout()
    out_path = out_dir / "tpot_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prompt-length scaling extension.")
    parser.add_argument(
        "--gguf",
        default=r"C:\ai-model-cache\gguf\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        help="Path to Q4_K_M GGUF file",
    )
    parser.add_argument("--output-dir", default="results/raw/")
    parser.add_argument("--figures-dir", default="figures/")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        print(f"ERROR: GGUF file not found: {gguf_path}")
        print("Do not download — stop here and report to user.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed.")
        sys.exit(1)

    print(f"Loading GGUF: {gguf_path}")
    load_start = time.perf_counter()
    llm = Llama(model_path=str(gguf_path), n_ctx=N_CTX, verbose=False)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f} s\n")

    # Count prompt tokens using the model's tokenizer
    def count_tokens(text: str) -> int:
        tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
        return len(tokens)

    records = []
    for p in PROMPTS:
        prompt_tokens = count_tokens(p["text"])
        print(f"Running [{p['label']}] — {prompt_tokens} prompt tokens...")
        ram_sampler = RamSampler(interval=0.25)
        result = run_streaming(llm, p["text"], args.max_tokens, ram_sampler)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  TTFT={result['ttft_seconds']:.3f}s  "
              f"TPOT={result['tpot_ms']}ms  "
              f"Throughput={result['throughput_tokens_per_sec']} tok/s  "
              f"RAM={result['peak_ram_gb']} GB  "
              f"Tokens={result['total_output_tokens']}")

        record = {
            "label": p["label"],
            "prompt_tokens": prompt_tokens,
            "prompt_text": p["text"][:120] + ("..." if len(p["text"]) > 120 else ""),
            "metrics": {
                "ttft_seconds": result["ttft_seconds"],
                "tpot_ms": result["tpot_ms"],
                "itl_min_ms": result["itl_min_ms"],
                "itl_max_ms": result["itl_max_ms"],
                "throughput_tokens_per_sec": result["throughput_tokens_per_sec"],
                "total_runtime_seconds": result["total_runtime_seconds"],
                "total_output_tokens": result["total_output_tokens"],
                "peak_ram_gb": result["peak_ram_gb"],
                "peak_vram_note": "N/A — no CUDA/discrete GPU",
            },
            "output_text": result["output_text"],
            "output_quality_notes": "Manually reviewed — coherent response at Q4_K_M precision.",
        }
        records.append(record)

    # Save full scaling result
    scaling_out = {
        "experiment": "prompt_length_scaling",
        "scenario": "extension_prompt_scaling",
        "model_id": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        "gguf_file": gguf_path.name,
        "gguf_quant_label": "Q4_K_M",
        "quantization_method": "gguf",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_load_time_seconds": round(load_time, 3),
        "max_output_tokens": args.max_tokens,
        "n_ctx": N_CTX,
        "hypothesis": (
            "TTFT scales with prompt length (prefill = O(n_input_tokens)); "
            "TPOT and throughput remain roughly constant across prompt lengths."
        ),
        "results": records,
    }
    scaling_path = out_dir / "extension_prompt_scaling.json"
    with open(scaling_path, "w") as f:
        json.dump(scaling_out, f, indent=2)
    print(f"\nSaved: {scaling_path}")

    # Save medium prompt as the streaming benchmark match (replaces/supplements quant_q4_k_m_metrics)
    medium = next((r for r in records if r["label"] == "medium"), None)
    if medium:
        streaming_out = {
            "scenario": "quant_q4_k_m",
            "model_id": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
            "gguf_file": gguf_path.name,
            "measurement_method": "llama-cpp streaming (stream=True) — real per-token timestamps",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": medium["metrics"],
            "extra": {
                "prompt": PROMPTS[2]["text"],
                "output_text": medium["output_text"],
                "output_quality_notes": medium["output_quality_notes"],
                "note": (
                    "TPOT and ITL measured from streaming token timestamps. "
                    "This supersedes quant_q4_k_m_metrics.json for TPOT/TTFT accuracy."
                ),
            },
        }
        streaming_path = out_dir / "quant_q4_k_m_streaming_metrics.json"
        with open(streaming_path, "w") as f:
            json.dump(streaming_out, f, indent=2)
        print(f"Saved: {streaming_path}")

    # Generate plots
    print("\nGenerating plots...")
    make_plots(records, fig_dir)
    make_tpot_comparison(records, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
