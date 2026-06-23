"""
run_baseline.py — Baseline inference: standard HuggingFace transformers pipeline,
no quantization, no AirLLM, no optimization.

This is the naive approach. On hardware-constrained machines it may:
  - Succeed but be very slow (valid result — document TTFT and TPOT)
  - Run out of RAM and crash with MemoryError (valid negative result)
  - Trigger OS swap and become unusably slow (valid negative result)

All outcomes are saved honestly. Run this first to establish the baseline.

Usage:
    python src/run_baseline.py --config experiments/configs/default_config.json \
        --output results/raw/baseline_metrics.json
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path so benchmark_common is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_common import (
    RamSampler,
    compute_metrics,
    load_config,
    load_env_config,
    save_failure,
    save_metrics,
)


def run_baseline(config: dict, output_path: str,
                 timeout_seconds: int = 0, max_ram_gb: float = 0.0):
    model_id = config.get("model_id", "facebook/opt-6.7b")
    max_new_tokens = config.get("max_new_tokens", 200)
    device = config.get("device", "cpu")
    prompt = config.get("benchmark_prompt", "Explain the concept of virtual memory in operating systems.")
    hf_token = config.get("hf_token")
    model_cache_dir = config.get("model_cache_dir", "./models")

    # Derive failure path: _metrics.json → _failure.json
    failure_path = output_path.replace("_metrics.json", "_failure.json")
    if failure_path == output_path:
        failure_path = output_path.rsplit(".", 1)[0] + "_failure.json"

    print(f"=== Baseline Experiment ===")
    print(f"Model:       {model_id}")
    print(f"Device:      {device}")
    print(f"Timeout:     {timeout_seconds}s" if timeout_seconds else "Timeout:     none")
    print(f"RAM abort:   {max_ram_gb:.1f} GB" if max_ram_gb else "RAM abort:   none")
    print(f"Failure out: {failure_path}")
    print(f"Prompt: {prompt[:80]}...")
    print()

    # ── Timeout guard (daemon thread calls os._exit on expiry) ──────────────
    if timeout_seconds > 0:
        def _on_timeout():
            print(f"\nTIMEOUT: {timeout_seconds}s elapsed — saving failure and exiting.")
            save_failure(
                TimeoutError(f"Run exceeded {timeout_seconds}s timeout"),
                failure_path, scenario="baseline",
                context=f"Timeout guard fired after {timeout_seconds}s. model_id={model_id}",
            )
            os._exit(5)
        _timer = threading.Timer(timeout_seconds, _on_timeout)
        _timer.daemon = True
        _timer.start()

    # ── RAM abort guard (background thread) ─────────────────────────────────
    if max_ram_gb > 0:
        def _ram_guard():
            try:
                import psutil
                proc = psutil.Process(os.getpid())
                while True:
                    time.sleep(2.0)
                    rss = proc.memory_info().rss / 1e9
                    if rss > max_ram_gb:
                        print(f"\nRAM ABORT: {rss:.2f} GB > {max_ram_gb:.1f} GB threshold — saving failure.")
                        save_failure(
                            MemoryError(f"RAM {rss:.2f} GB exceeded abort threshold {max_ram_gb:.1f} GB"),
                            failure_path, scenario="baseline",
                            context=f"RAM abort guard: {rss:.3f} GB > {max_ram_gb} GB. model_id={model_id}",
                        )
                        os._exit(4)
            except Exception:
                pass
        _ram_thread = threading.Thread(target=_ram_guard, daemon=True)
        _ram_thread.start()

    # ── Import heavy dependencies inside the function so the module is importable
    # even when torch/transformers are not installed.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        save_failure(e, failure_path, scenario="baseline",
                     context="torch or transformers not installed")
        print(f"IMPORT ERROR: {e}")
        print("Install with: pip install -e '.[transformers]'")
        sys.exit(1)

    ram_sampler = RamSampler(interval=0.5)

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=model_cache_dir,
        )

        print(f"Loading model (device={device}, no quantization)...")
        ram_sampler.start()
        model_load_start = time.perf_counter()

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=model_cache_dir,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            model = model.to("cpu")
        model.eval()

        model_load_time = time.perf_counter() - model_load_start
        print(f"Model loaded in {model_load_time:.1f}s")

        # ── Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.shape[-1]

        # ── Inference with TTFT measurement
        print(f"Running inference (max_new_tokens={max_new_tokens})...")
        first_token_time = None
        generated_tokens = 0

        def _first_token_hook(*_args, **_kwargs):
            nonlocal first_token_time
            if first_token_time is None:
                first_token_time = time.perf_counter()

        inference_start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        inference_end = time.perf_counter()

        # TTFT approximation: use total time if streaming hooks not available.
        # For a more accurate TTFT, use streaming generation — see TODO below.
        # TODO: Replace with streamer-based generation to get true TTFT.
        total_runtime = inference_end - inference_start
        ttft = total_runtime  # Conservative: full prefill+first-decode estimate
        generated_tokens = output.shape[-1] - n_input_tokens

        peak_ram = ram_sampler.stop()

        metrics = compute_metrics(
            ttft_seconds=ttft,
            total_output_tokens=generated_tokens,
            total_runtime_seconds=total_runtime,
            peak_ram_gb=peak_ram,
            gpu_available=False,
        )
        metrics["model_load_time_seconds"] = round(model_load_time, 3)
        metrics["n_input_tokens"] = n_input_tokens
        metrics["ttft_note"] = (
            "Approximated as total runtime (no streaming hook). "
            "True TTFT ≤ this value."
        )

        output_text = tokenizer.decode(output[0][n_input_tokens:], skip_special_tokens=True)
        extra = {
            "prompt": prompt,
            "output_text": output_text[:500],
            "output_quality_notes": "TODO: manual assessment after experiment",
        }

        save_metrics(metrics, output_path, scenario="baseline", model_id=model_id, extra=extra)

        print()
        print(f"TTFT (approx):  {metrics['ttft_seconds']:.2f}s")
        print(f"TPOT:           {metrics['tpot_ms']} ms/token")
        print(f"Throughput:     {metrics['throughput_tokens_per_sec']} tok/s")
        print(f"Peak RAM:       {metrics['peak_ram_gb']:.2f} GB")
        print(f"Output tokens:  {generated_tokens}")
        print()
        print("Output snippet:")
        print(output_text[:300])

    except MemoryError as e:
        ram_sampler.stop()
        save_failure(e, failure_path, scenario="baseline",
                     context=f"MemoryError loading {model_id} on {device}")
        print(f"\nOUT OF MEMORY: {e}")
        print("This is a valid negative result — documented in failure file.")
        sys.exit(2)

    except Exception as e:
        ram_sampler.stop()
        save_failure(e, failure_path, scenario="baseline", context=str(e))
        print(f"\nFAILED: {e}")
        sys.exit(3)


def main():
    parser = argparse.ArgumentParser(description="Run baseline inference experiment.")
    parser.add_argument("--config", default="experiments/configs/default_config.json")
    parser.add_argument("--output", default="results/raw/baseline_metrics.json")
    parser.add_argument("--timeout-seconds", type=int, default=0,
                        help="Abort after N seconds (0 = no timeout)")
    parser.add_argument("--max-ram-gb", type=float, default=0.0,
                        help="Abort if process RAM exceeds N GB (0 = no limit)")
    args = parser.parse_args()

    cfg = load_env_config()
    if Path(args.config).exists():
        cfg.update(load_config(args.config))

    run_baseline(cfg, args.output,
                 timeout_seconds=args.timeout_seconds,
                 max_ram_gb=args.max_ram_gb)


if __name__ == "__main__":
    main()
