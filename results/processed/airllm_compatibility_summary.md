# AirLLM Compatibility Summary

**Date:** 2026-06-23  
**Result:** BLOCKED — AirLLM cannot run on this hardware within constraints

---

> ## ✅ RESOLVED 2026-06-24 (see `results/raw/airllm_gpu_metrics.json`)
>
> This document records the **original** investigation on the primary i5-1135G7 laptop (no CUDA GPU).
> AirLLM was later **run successfully** on a second machine with an NVIDIA RTX 4060 Laptop GPU:
> `Qwen/Qwen2.5-7B-Instruct` layer-streamed at **1.16 GB peak VRAM**, 0.039 tok/s, 32 tokens.
>
> Two findings below were refined by the working run:
> - **Model format:** AirLLM 2.11.0 accepts *either* `model.safetensors.index.json` *or*
>   `pytorch_model.bin.index.json` (a multi-shard checkpoint). A single-file model (0.5B) has no
>   index and cannot be sharded — which is the real reason 0.5B failed. The working target is the
>   multi-shard `Qwen/Qwen2.5-7B-Instruct`. (`facebook/opt-6.7b` turned out to be unsupported for a
>   different reason: AirLLM has no OPT class, so its `model.decoder.layers.*` naming is not found.)
> - **GPU:** the only real hard blocker on the primary laptop. A CUDA GPU resolved it.
> - Runtime also required `transformers==4.45.2` (newer drops the per-layer RoPE fallback AirLLM uses)
>   and three small `run_airllm.py` fixes (`layer_shards_saving_path`, `hf_token`, `input_ids.to(device)`).

---

## What Was Tested

- `airllm` package import and `AutoModel` API
- AirLLM initialization with `Qwen/Qwen2.5-0.5B-Instruct` (only fully-cached model)
- Correct API parameters (`device`, `dtype`, `layer_shards_saving_path`)

## Findings

### Imports — PASS
- `airllm.AutoModel` imports correctly
- `AirLLMQWen2` is correctly auto-selected for Qwen2 architecture
- All dependencies (`transformers`, `torch`) import fine

### Bug in `src/run_airllm.py`
The script passes `cache_dir=` to `AirLLMAutoModel.from_pretrained()`, but AirLLM does not accept `cache_dir`. The correct parameter is `layer_shards_saving_path`. This would cause an immediate `TypeError` on the original script.

### Blocker 1 — Model Format
AirLLM requires `model.safetensors.index.json` (the HuggingFace sharded format for large models). This file only exists when a model is stored as multiple shard files.

- **Qwen2.5-0.5B**: single `model.safetensors` → no index file → `AssertionError` on init
- **OPT-6.7B**: would have the index file, but requires full 13.5 GB download (only 4 GB cached; previous download stalled at timeout)

### Blocker 2 — No GPU
AirLLM defaults to `device='cuda:0'`. This machine has no CUDA GPU. CPU-mode operation is undocumented and `dtype=float16` is unsupported on CPU without explicit workarounds.

---

## Conclusion

AirLLM is **BLOCKED** on this machine due to two independent hard constraints:

1. No usable model in the right format (tiny model = no shard index; large model = incomplete download)
2. No GPU (AirLLM is GPU-native)

**AirLLM is documented as a compatibility constraint in the report rather than a runnable experiment.**

---

## Next Step

Proceed to quantization experiment: `llama.cpp` + `Qwen2.5-7B-Instruct-GGUF` (Q4\_K\_M, ~4.4 GB).
