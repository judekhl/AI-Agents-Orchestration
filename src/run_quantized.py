"""
run_quantized.py — Quantization experiment: run the same model at multiple
precision levels and save metrics for each.

Strategy A (transformers, no GPU required):
  - FP32: torch_dtype=torch.float32 (full precision)
  - FP16: torch_dtype=torch.float16 (half precision)
  - BF16: torch_dtype=torch.bfloat16 (brain float, same size as FP16)

Strategy B (GGUF via llama-cpp-python, CPU-optimized):
  - Q8_0:  8-bit integer quantization
  - Q4_K_M: 4-bit with mixed precision (K-quant variant)

This script tries Strategy A first. If the model is too large for FP32/FP16 on
available RAM, it falls back to Strategy B with a pre-quantized GGUF file.

Usage:
    python src/run_quantized.py --config experiments/configs/default_config.json \
        --output-dir results/raw/
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_common import (
    RamSampler,
    compute_metrics,
    load_config,
    load_env_config,
    save_failure,
    save_metrics,
)

QUANT_CONFIGS = [
    {"id": "fp32", "torch_dtype": "float32",  "description": "Full precision FP32"},
    {"id": "fp16", "torch_dtype": "float16",  "description": "Half precision FP16"},
    {"id": "bf16", "torch_dtype": "bfloat16", "description": "Brain float BF16"},
]

GGUF_QUANT_CONFIGS = [
    {"id": "q8_0",   "quant_label": "Q8_0",   "description": "8-bit integer (GGUF)"},
    {"id": "q4_k_m", "quant_label": "Q4_K_M", "description": "4-bit K-quant mixed (GGUF)"},
]


def run_single_transformers(model_id, dtype_str, prompt, max_new_tokens, hf_token,
                            cache_dir, output_path, quant_id):
    print(f"\n--- Quantization: {dtype_str} ---")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, cache_dir=cache_dir
        )
        ram_sampler = RamSampler(interval=0.5)
        ram_sampler.start()

        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        load_time = time.perf_counter() - load_start

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.shape[-1]

        inf_start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        inf_end = time.perf_counter()
        peak_ram = ram_sampler.stop()

        total_runtime = inf_end - inf_start
        generated_tokens = output.shape[-1] - n_input_tokens
        output_text = tokenizer.decode(output[0][n_input_tokens:], skip_special_tokens=True)

        metrics = compute_metrics(
            ttft_seconds=total_runtime,
            total_output_tokens=generated_tokens,
            total_runtime_seconds=total_runtime,
            peak_ram_gb=peak_ram,
        )
        metrics["quantization_method"] = "transformers_dtype"
        metrics["torch_dtype"] = dtype_str
        metrics["model_load_time_seconds"] = round(load_time, 3)

        save_metrics(metrics, output_path, scenario=f"quant_{quant_id}",
                     model_id=model_id,
                     extra={"prompt": prompt, "output_text": output_text[:500],
                            "output_quality_notes": "TODO: manual assessment"})

        print(f"  Throughput: {metrics['throughput_tokens_per_sec']} tok/s")
        print(f"  Peak RAM:   {metrics['peak_ram_gb']:.2f} GB")
        del model

    except MemoryError as e:
        save_failure(e, output_path.replace("_metrics.json", "_failure.txt"),
                     scenario=f"quant_{quant_id}",
                     context=f"OOM loading {model_id} at {dtype_str}")
        print(f"  OOM at {dtype_str} — failure logged.")

    except Exception as e:
        save_failure(e, output_path.replace("_metrics.json", "_failure.txt"),
                     scenario=f"quant_{quant_id}", context=str(e))
        print(f"  FAILED at {dtype_str}: {e}")


def run_single_gguf(gguf_path, quant_label, prompt, max_new_tokens, output_path, quant_id, model_id):
    print(f"\n--- GGUF Quantization: {quant_label} ---")
    try:
        from llama_cpp import Llama

        ram_sampler = RamSampler(interval=0.5)
        ram_sampler.start()

        load_start = time.perf_counter()
        llm = Llama(model_path=str(gguf_path), n_ctx=2048, verbose=False)
        load_time = time.perf_counter() - load_start

        inf_start = time.perf_counter()
        result = llm(prompt, max_tokens=max_new_tokens, temperature=0.0)
        inf_end = time.perf_counter()
        peak_ram = ram_sampler.stop()

        total_runtime = inf_end - inf_start
        output_text = result["choices"][0]["text"]
        generated_tokens = result.get("usage", {}).get("completion_tokens", max_new_tokens)

        metrics = compute_metrics(
            ttft_seconds=total_runtime,
            total_output_tokens=generated_tokens,
            total_runtime_seconds=total_runtime,
            peak_ram_gb=peak_ram,
        )
        metrics["quantization_method"] = "gguf"
        metrics["gguf_quant_label"] = quant_label
        metrics["model_load_time_seconds"] = round(load_time, 3)

        save_metrics(metrics, output_path, scenario=f"quant_{quant_id}",
                     model_id=model_id,
                     extra={"prompt": prompt, "output_text": output_text[:500],
                            "output_quality_notes": "TODO: manual assessment"})

        print(f"  Throughput: {metrics['throughput_tokens_per_sec']} tok/s")
        print(f"  Peak RAM:   {metrics['peak_ram_gb']:.2f} GB")

    except ImportError as e:
        save_failure(e, output_path.replace("_metrics.json", "_failure.txt"),
                     scenario=f"quant_{quant_id}",
                     context="llama-cpp-python not installed")
        print(f"  SKIPPED: llama-cpp-python not installed.")

    except Exception as e:
        save_failure(e, output_path.replace("_metrics.json", "_failure.txt"),
                     scenario=f"quant_{quant_id}", context=str(e))
        print(f"  FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run quantization experiments.")
    parser.add_argument("--config", default="experiments/configs/default_config.json")
    parser.add_argument("--output-dir", default="results/raw/")
    parser.add_argument(
        "--strategy",
        choices=["transformers", "gguf", "both"],
        default="transformers",
        help="Which quantization strategy to run",
    )
    parser.add_argument(
        "--gguf-dir",
        default="./models/gguf/",
        help="Directory containing pre-downloaded GGUF files",
    )
    args = parser.parse_args()

    cfg = load_env_config()
    if Path(args.config).exists():
        cfg.update(load_config(args.config))

    model_id = cfg["model_id"]
    prompt = cfg.get("benchmark_prompt",
                     "Explain the concept of virtual memory in operating systems.")
    max_new_tokens = cfg["max_new_tokens"]
    hf_token = cfg.get("hf_token")
    cache_dir = cfg.get("model_cache_dir", "./models")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Quantization Experiment ===")
    print(f"Model: {model_id}")
    print(f"Strategy: {args.strategy}")

    if args.strategy in ("transformers", "both"):
        for qcfg in QUANT_CONFIGS:
            out_path = str(out_dir / f"quant_{qcfg['id']}_metrics.json")
            run_single_transformers(
                model_id=model_id,
                dtype_str=qcfg["torch_dtype"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                hf_token=hf_token,
                cache_dir=cache_dir,
                output_path=out_path,
                quant_id=qcfg["id"],
            )

    if args.strategy in ("gguf", "both"):
        gguf_dir = Path(args.gguf_dir)
        for qcfg in GGUF_QUANT_CONFIGS:
            # Look for a matching GGUF file by quant label
            pattern = f"*{qcfg['quant_label']}*"
            matches = list(gguf_dir.glob(pattern)) if gguf_dir.exists() else []
            out_path = str(out_dir / f"quant_{qcfg['id']}_metrics.json")
            if not matches:
                print(f"\n--- GGUF {qcfg['quant_label']}: no file found in {gguf_dir} ---")
                from src.benchmark_common import save_failure
                save_failure(
                    FileNotFoundError(f"No GGUF file matching {pattern} in {gguf_dir}"),
                    out_path.replace("_metrics.json", "_failure.txt"),
                    scenario=f"quant_{qcfg['id']}",
                    context=f"Download a {qcfg['quant_label']} GGUF file to {gguf_dir}",
                )
                continue
            run_single_gguf(
                gguf_path=matches[0],
                quant_label=qcfg["quant_label"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                output_path=out_path,
                quant_id=qcfg["id"],
                model_id=model_id,
            )

    print("\nQuantization experiments complete.")


if __name__ == "__main__":
    main()
