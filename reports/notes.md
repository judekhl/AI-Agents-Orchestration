# Working Notes — Assignment 05

Use this file to track decisions, observations, and questions during experiments.
These notes do NOT need to be polished — they are working notes only.
The README.md is the final deliverable.

---

## Decisions

- [x] Model selection: confirmed 2026-06-23
      Warm-up:  Qwen/Qwen2.5-0.5B-Instruct
      Stress:   facebook/opt-6.7b  (OOM expected on naive load)
      Fallback: bartowski/Qwen2.5-7B-Instruct-GGUF Q4_K_M
- [x] Cache paths: model files stored OUTSIDE the OneDrive repo (see below)
- [ ] Quantization strategy: GGUF Q4_K_M / Q8_0 via llama-cpp-python (primary);
      transformers FP16/BF16 for warm-up model
- [ ] Extension choice: disk I/O observation (tentative) — revisit after AirLLM run

## Architecture Decision — Cache Outside OneDrive (2026-06-23)

**Decision:** Model files are stored outside the OneDrive repo to avoid Git bloat
and OneDrive sync problems.

**Why:**
1. This repo lives at C:\Users\HP\OneDrive\Documents\Jam3a\AI-Agents-Orchestration.
   OneDrive monitors and syncs everything under its folder tree.
2. Model weights are multi-GB binary files (OPT-6.7B = ~13.4 GB, AirLLM shards = ~13.4 GB).
   Syncing them would waste bandwidth, consume cloud storage quota, and slow down experiments.
3. AirLLM writes layer shard files during inference. OneDrive's background sync can hold
   file locks that conflict with AirLLM's writes, causing runtime errors mid-experiment.
4. Git already ignores model files via .gitignore patterns (*.bin, *.safetensors, *.gguf, etc.)
   but .gitignore does NOT prevent OneDrive from syncing — only moving files outside
   the OneDrive tree provides that guarantee.

**Cache layout (create once before any download):**
  C:\ai-model-cache\hf             — HuggingFace model downloads (HF_HOME)
  C:\ai-model-cache\airllm-cache   — AirLLM runtime cache
  C:\ai-model-cache\airllm-shards  — AirLLM per-layer shard files (~13.4 GB for OPT-6.7B)
  C:\ai-model-cache\gguf           — GGUF single-file downloads (~4.4 GB for Q4_K_M)

**Where this is documented:**
  .env.example              — recommended paths + explanation comment block
  experiments/configs/default_config.json — absolute paths as defaults
  README.md Section 3       — "Step 0: Create external cache folders" instruction

---

## Experiment Log

_Add entries as experiments run. Include date, outcome, and any errors._

### hardware_probe.py
- Status: DONE (2026-06-23)
- Result: i5-1135G7, 4P/8L cores, 8.22 GB RAM, no GPU, NVMe SSD, 38.44 GB free
- Evidence: results/raw/hardware_profile.json

### model_selection.py
- Status: DONE (2026-06-23)
- Result: three-role selection confirmed; disk budget 32.2 GB / 38.44 GB free
- Evidence: results/raw/model_selection.json

### baseline — warm-up (Qwen2.5-0.5B-Instruct)
- Status: NOT RUN
- Notes:

### baseline — stress (OPT-6.7B, OOM expected on naive load)
- Status: NOT RUN
- Notes:

### run_airllm.py (stress: OPT-6.7B via AirLLM)
- Status: NOT RUN
- Notes:

### run_quantized.py (fallback: Qwen2.5-7B Q4_K_M / Q8_0 GGUF)
- Status: NOT RUN
- Notes:

### extension
- Status: NOT RUN
- Planned: disk I/O observation during AirLLM run
- Notes:

---

## Blockers / Issues

_Document anything that blocks progress._

---

## Reminders

- Never type a number into README.md without a corresponding results/raw/*.json file
- Commit after each experiment with a descriptive message
- Save BOTH success metrics AND failure logs — negative results count
- Screenshot terminal output during each run (save to figures/screenshots/)
- Check disk space at C:\ai-model-cache\ before each model download
- Do NOT move cache paths back inside the repo or OneDrive
