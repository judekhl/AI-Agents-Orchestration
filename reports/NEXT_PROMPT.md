# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Run only the warm-up baseline with Qwen/Qwen2.5-0.5B-Instruct. Use C:\ai-model-cache\hf for model cache. Use small max_new_tokens=32 or 64. Save raw metrics to results/raw/baseline_warmup_metrics.json and readable summary to results/processed/baseline_warmup_summary.md. Do not run OPT-6.7B yet. Do not run AirLLM yet. Do not run quantization yet. Update README and REQUIREMENTS_MATRIX only for evidence that exists.

---

## Context before running

- Read CLAUDE.md first (compact session context).
- Read reports/PROJECT_STATE.md for current state.
- Venv: `C:\ai-envs\ai-agents-ex05\Scripts\activate`
- Model will download to `C:\ai-model-cache\hf` (~1 GB — confirm disk space first).
- Script: `src/run_baseline.py`
- Config: `experiments/configs/default_config.json`
- Relevant src: `src/benchmark_common.py` (RamSampler, compute_metrics)

## After the run

Update this file with the next task once warm-up baseline is complete.
