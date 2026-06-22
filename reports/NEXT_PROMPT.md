# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Run only the stress baseline with facebook/opt-6.7b. Use C:\ai-model-cache\hf for model cache. Use max_new_tokens=64. Save raw output to results/raw/baseline_stress_failure.json (or results/raw/baseline_stress_metrics.json if it somehow succeeds). Do not run AirLLM yet. Do not run quantization yet. Update PROJECT_STATE.md with result. The expected outcome is MemoryError or OS swap making inference unusably slow — both are valid negative results to document.

---

## Context before running

- Read CLAUDE.md first (compact session context).
- Read reports/PROJECT_STATE.md for current state.
- Venv: `C:\ai-envs\ai-agents-ex05\Scripts\activate`
- Model will attempt to download to `C:\ai-model-cache\hf` (~13 GB FP16 — may fail mid-download due to RAM).
- Script: `src/run_baseline.py`
- Config: `experiments/configs/default_config.json`
- Set MODEL_ID=facebook/opt-6.7b (env var) — config key is model_id_stress.
- Warning: this run may take a very long time or crash. Set timeout to 20+ minutes.

## After the run

Update this file with the next task (AirLLM experiment) once stress baseline is complete.
