# Model Selection Report
Generated: 2026-06-23
Hardware: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz, 8.22 GB RAM, no GPU, 38.44 GB free (NVMe SSD)

## Selected Models

| Role | Model | Format | RAM needed | Fits in 8.22 GB RAM? |
|---|---|---|---|---|
| Warm-up | `Qwen/Qwen2.5-0.5B-Instruct` | HF FP16 | ~1.5 GB | ✓ Fits |
| Stress | `facebook/opt-6.7b` | HF FP16 | ~14.0 GB | ✗ OOM expected |
| Fallback | `bartowski/Qwen2.5-7B-Instruct-GGUF` | GGUF Q4_K_M | ~4.8 GB | ✓ Fits |

## Disk Budget

| Item | GB |
|---|---|
| Disk free at probe | 38.44 |
| Warm-up model (FP16) | 1.0 |
| Stress model (FP16) | 13.4 |
| AirLLM shards (stress model copy) | 13.4 |
| Fallback GGUF (Q4_K_M) | 4.4 |
| **Total needed** | **32.2** |
| **Remaining after all downloads** | **6.24** |
| Budget feasible? | ✓ Yes |

> AirLLM shards the stress model into per-layer files in a separate directory. Budget assumes both original weights and shards coexist on disk during sharding. After sharding completes, the original checkpoint can be deleted to reclaim space.

## Role Rationale

**Warm-up:** Qwen2.5-0.5B-Instruct (0.5B params, ~1.5 GB RAM in FP16) proves the inference pipeline works on this hardware before attempting the stress model. Provides a meaningful but non-stressful timing baseline.

**Stress:** OPT-6.7B (6.7B params, ~14 GB RAM in FP16) far exceeds the 8.22 GB RAM ceiling. Naive baseline load will trigger OOM — this is the expected, documentable negative result that motivates AirLLM and quantization. AirLLM can run this model by loading one layer (~0.4 GB) at a time.

**Fallback:** Qwen2.5-7B-Instruct Q4_K_M GGUF (~4.4 GB disk, ~4.8 GB RAM) fits within 8.22 GB RAM and enables a successful throughput measurement. Q4_K_M is the recommended balance of compression and quality. Inference uses llama-cpp-python; no torch or transformers needed.

## Feasibility Details

### Warm-Up — `Qwen/Qwen2.5-0.5B-Instruct`
- No feasibility issues.

### Stress — `facebook/opt-6.7b`
- ⚠ FP16 needs 14.0 GB RAM; only 8.22 GB available. OOM expected on naive load — valid negative result.

### Fallback — `bartowski/Qwen2.5-7B-Instruct-GGUF`
- No feasibility issues.
