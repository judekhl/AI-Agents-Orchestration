# Baseline Warm-up Summary — Qwen/Qwen2.5-0.5B-Instruct

**Date:** 2026-06-23  
**Scenario:** Standard HuggingFace `transformers`, no quantization, no AirLLM  
**Hardware:** Intel i5-1135G7, 8.22 GB RAM, CPU-only (no GPU)

---

## Key Metrics

| Metric | Value |
|---|---|
| Model load time | 208.5 s (first run; includes ~1 GB download) |
| TTFT (approx) | 10.33 s |
| TPOT | N/A (TTFT approximated as full runtime — no streaming hook) |
| Throughput | 6.20 tok/s |
| Peak RAM | 2.73 GB |
| Output tokens | 64 |
| Input tokens | 23 |
| OOM | No |

## Notes

- TTFT is a conservative upper bound (total runtime). True TTFT ≤ 10.33 s.
- TPOT is null because decode_time = total_runtime − ttft_approx ≈ 0. Accurate TPOT requires a streaming hook (planned).
- Peak RAM (2.73 GB) well within 8.22 GB limit. Model fits comfortably.
- Model cache: `C:\ai-model-cache\hf`

## Output Snippet

> Virtual memory is a technique used by operating systems to allow programs to access larger amounts of memory than are physically available on the system's physical memory. It does this by creating a separate "virtual" memory space that can be accessed by the program without having to worry about the actual physical memory being allocated. Paging is one...

**Quality assessment:** Coherent, on-topic. Partial sentence at cut-off (max_new_tokens=64). Acceptable for benchmarking purposes.

## Raw Evidence

`results/raw/baseline_warmup_metrics.json`
