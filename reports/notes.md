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
  README.md Section 3       — "Step 0: Create external folders" instruction

---

## Architecture Decision — Venv Outside OneDrive (2026-06-23)

**Decision:** The Python virtual environment is stored outside the OneDrive repo at
`C:\ai-envs\ai-agents-ex05` to avoid OneDrive sync overhead.

**Why:**
1. A Python venv created with `python -m venv .venv` inside the repo would sit under
   the OneDrive folder tree. OneDrive would attempt to sync every `.py`, `.pyd`, `.dll`,
   and `.pth` file inside it — typically 5 000–15 000 small files.
2. Continuous sync of venv files wastes bandwidth, slows `pip install` (OneDrive holds
   file locks during upload), and can cause "file in use" errors during package installation.
3. Moving the venv to `C:\ai-envs\` (outside OneDrive) eliminates all three problems.
   The venv is not version-controlled (it's gitignored) so there is no loss of portability.

**Venv path:** `C:\ai-envs\ai-agents-ex05`
**Activate:**  `C:\ai-envs\ai-agents-ex05\Scripts\activate`

**Where this is documented:**
  README.md Section 3 Step 0 — mkdir C:\ai-envs, Step 1 — venv creation command

---

## Compatibility Fix — airllm vs transformers (2026-06-23)

**Problem:** airllm 2.11.0 imports `optimum.bettertransformer.BetterTransformer`, which
requires `transformers < 4.49`. The default pip install gave transformers 5.12.1.
The import failed at module level:
  `ModuleNotFoundError: No module named 'optimum.bettertransformer'` (with optimum 2.2)
  `RuntimeError: BetterTransformer requires transformers<4.49 but found 5.12.1` (with optimum 1.27)

**Fix applied:**
  pip install "transformers>=4.44,<4.49"   # downgraded to 4.48.3
  pip install "optimum>=1.16,<2.0"          # pinned to 1.27.0

**Final resolved versions:**
  transformers: 4.48.3
  optimum: 1.27.0
  airllm: 2.11.0 (AutoModel importable)

**Impact on experiments:**
  Qwen2.5 and OPT models are supported in transformers 4.48.3 — no impact on planned experiments.
  Shard format and AutoModel API are unaffected.

**Evidence:** `results/raw/environment_setup.json` → `optional_imports.airllm.status: ok`

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

### environment setup (save_environment_info.py)
- Status: DONE (2026-06-23)
- Result: all packages installed and importable
  Venv: C:\ai-envs\ai-agents-ex05 (Python 3.10.0, outside OneDrive)
  Core: psutil 7.2.2, pandas 2.3.3, matplotlib 3.10.9, numpy 2.2.6
  ML: torch 2.12.1+cpu, transformers 4.48.3, accelerate 1.14.0
  Optional: llama_cpp 0.3.31 (ok), airllm 2.11.0 (ok after downgrade)
  Note: transformers pinned to 4.48.3 due to airllm/optimum compatibility
- Evidence: results/raw/environment_setup.json, results/processed/environment_summary.md

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
