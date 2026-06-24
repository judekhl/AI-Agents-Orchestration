"""
run_airllm.py — AirLLM layer-sharding inference experiment.

AirLLM loads one transformer layer at a time from disk, runs its forward pass,
then discards it before loading the next. This means peak RAM is proportional to
a single layer's size rather than the full model — enabling inference on machines
with far less RAM than the model requires.

Expected behavior on memory-constrained hardware:
  - First run: AirLLM shards the model into per-layer files (slow, one-time)
  - Subsequent runs: loads directly from cached shards
  - TTFT will be significantly higher than baseline (disk I/O bottleneck)
  - Peak RAM will be significantly lower than baseline

Usage:
    python src/run_airllm.py --config experiments/configs/default_config.json \
        --output results/raw/airllm_metrics.json
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_common import (
    DiskIoSampler,
    RamSampler,
    compute_metrics,
    load_config,
    load_env_config,
    save_failure,
    save_metrics,
)


def run_airllm(config: dict, output_path: str):
    model_id = config.get("model_id", "facebook/opt-6.7b")
    max_new_tokens = config.get("max_new_tokens", 200)
    shard_dir = config.get("airllm_shard_dir", "./airllm_shards")
    cache_dir = config.get("airllm_cache_dir", "./airllm_cache")
    prompt = config.get("benchmark_prompt", "Explain the concept of virtual memory in operating systems.")
    hf_token = config.get("hf_token")

    print(f"=== AirLLM Experiment ===")
    print(f"Model:      {model_id}")
    print(f"Shard dir:  {shard_dir}")
    print(f"Cache dir:  {cache_dir}")
    print(f"Prompt:     {prompt[:80]}...")
    print()

    try:
        from airllm import AutoModel as AirLLMAutoModel
    except ImportError as e:
        save_failure(e, output_path.replace("_metrics.json", "_failure.json"),
                     scenario="airllm",
                     context="airllm package not installed — run: pip install airllm")
        print(f"IMPORT ERROR: {e}")
        print("Install with: pip install -e '.[airllm]'")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        save_failure(e, output_path.replace("_metrics.json", "_failure.json"),
                     scenario="airllm", context="transformers not installed")
        sys.exit(1)

    try:
        import torch
    except ImportError as e:
        save_failure(e, output_path.replace("_metrics.json", "_failure.json"),
                     scenario="airllm", context="torch not installed")
        sys.exit(1)

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_version = torch.version.cuda if cuda_available else None
    print(f"Device:     {'GPU ' + gpu_name + ' (CUDA ' + str(cuda_version) + ')' if cuda_available else 'CPU only'}")
    print()

    ram_sampler = RamSampler(interval=0.5)
    disk_sampler = DiskIoSampler(interval=0.1)

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
        )

        print("Initializing AirLLM model (may shard on first run — this is slow)...")
        ram_sampler.start()
        disk_sampler.start()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()

        # AirLLM API: per-layer shard files are written to layer_shards_saving_path.
        # NOTE: AirLLM does NOT accept `cache_dir` here (that is a transformers arg) —
        # passing it raises TypeError before the model loads. Use layer_shards_saving_path.
        # The constructor param is `hf_token` (not `token`); only pass it when set so the
        # public-model path doesn't trip AutoConfig's token branch.
        model = AirLLMAutoModel.from_pretrained(
            model_id,
            layer_shards_saving_path=shard_dir,
            **({"hf_token": hf_token} if hf_token else {}),
        )

        load_time = time.perf_counter() - load_start
        print(f"AirLLM model ready in {load_time:.1f}s")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.shape[-1]

        print(f"Running AirLLM inference (max_new_tokens={max_new_tokens})...")
        inference_start = time.perf_counter()

        # AirLLM uses standard generate() interface
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        inference_end = time.perf_counter()
        disk_io_samples = disk_sampler.stop()
        peak_ram = ram_sampler.stop()
        peak_vram = (torch.cuda.max_memory_allocated() / 1e9) if cuda_available else None

        total_runtime = inference_end - inference_start
        generated_tokens = output.shape[-1] - n_input_tokens

        # AirLLM TTFT ≈ total runtime because every decode step reloads layers.
        # True streaming TTFT not available without AirLLM internals hook.
        metrics = compute_metrics(
            ttft_seconds=total_runtime,
            total_output_tokens=generated_tokens,
            total_runtime_seconds=total_runtime,
            peak_ram_gb=peak_ram,
            gpu_available=cuda_available,
            peak_vram_gb=round(peak_vram, 3) if peak_vram is not None else None,
        )
        metrics["model_load_time_seconds"] = round(load_time, 3)
        metrics["n_input_tokens"] = n_input_tokens
        metrics["ttft_note"] = (
            "AirLLM reloads layers every decode step; TTFT ≈ full runtime. "
            "True streaming TTFT not measured."
        )
        metrics["disk_io_sample_count"] = len(disk_io_samples)
        if disk_io_samples:
            max_read = max(s["read_mb_s"] for s in disk_io_samples)
            avg_read = sum(s["read_mb_s"] for s in disk_io_samples) / len(disk_io_samples)
            metrics["disk_io_peak_read_mb_s"] = round(max_read, 2)
            metrics["disk_io_avg_read_mb_s"] = round(avg_read, 2)

        output_text = tokenizer.decode(output[0][n_input_tokens:], skip_special_tokens=True)
        extra = {
            "prompt": prompt,
            "output_text": output_text[:500],
            "output_quality_notes": "TODO: manual assessment after experiment",
            "disk_io_samples": disk_io_samples,
            "shard_dir": str(shard_dir),
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "device": "cuda:0" if cuda_available else "cpu",
        }

        save_metrics(metrics, output_path, scenario="airllm", model_id=model_id, extra=extra)

        print()
        print(f"Total runtime:      {total_runtime:.2f}s")
        print(f"TPOT:               {metrics['tpot_ms']} ms/token")
        print(f"Throughput:         {metrics['throughput_tokens_per_sec']} tok/s")
        print(f"Peak RAM:           {metrics['peak_ram_gb']:.2f} GB")
        print(f"Peak VRAM:          {f'{peak_vram:.2f} GB' if peak_vram is not None else 'N/A'}")
        print(f"Disk peak read:     {metrics.get('disk_io_peak_read_mb_s', 'N/A')} MB/s")
        print(f"Output tokens:      {generated_tokens}")

    except MemoryError as e:
        disk_sampler.stop()
        ram_sampler.stop()
        save_failure(e, output_path.replace("_metrics.json", "_failure.json"),
                     scenario="airllm", context="MemoryError during AirLLM inference")
        print(f"\nOUT OF MEMORY: {e}")
        sys.exit(2)

    except Exception as e:
        disk_sampler.stop()
        ram_sampler.stop()
        save_failure(e, output_path.replace("_metrics.json", "_failure.json"),
                     scenario="airllm", context=str(e))
        print(f"\nFAILED: {e}")
        sys.exit(3)


def main():
    parser = argparse.ArgumentParser(description="Run AirLLM inference experiment.")
    parser.add_argument("--config", default="experiments/configs/default_config.json")
    parser.add_argument("--output", default="results/raw/airllm_metrics.json")
    args = parser.parse_args()

    cfg = load_env_config()
    if Path(args.config).exists():
        cfg.update(load_config(args.config))

    run_airllm(cfg, args.output)


if __name__ == "__main__":
    main()
