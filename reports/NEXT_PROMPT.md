# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Run the AirLLM experiment with facebook/opt-6.7b using layer-sharding. The model is partially downloaded (~4 GB / 13.5 GB in C:\ai-model-cache\hf). AirLLM needs the full model — it will either resume the download or re-download. Use C:\ai-model-cache\airllm-cache for AirLLM cache and C:\ai-model-cache\airllm-shards for shard storage. Use max_new_tokens=32 or 64. Save metrics to results/raw/airllm_metrics.json (or results/raw/airllm_failure.json if it fails). Do not run quantization yet. Update README Section 5, PROJECT_STATE.md, and REQUIREMENTS_MATRIX.md with the result.

---

## Context before running

- Read CLAUDE.md, reports/PROJECT_STATE.md, reports/NEXT_PROMPT.md, src/run_airllm.py
- Venv: `C:\ai-envs\ai-agents-ex05\Scripts\activate`
- Model partially cached in `C:\ai-model-cache\hf` (~4 GB of 13.5 GB). AirLLM needs full model.
- AirLLM shards: `C:\ai-model-cache\airllm-shards` (shard step writes ~13.5 GB here)
- AirLLM cache: `C:\ai-model-cache\airllm-cache`
- Disk free: ~38 GB (check before running — need ~13.5 GB for remaining download + 13.5 GB shards)
- Script: `src/run_airllm.py`
- Config: `experiments/configs/default_config.json`
- Warning: first AirLLM run requires download + sharding. Sharding alone may take 30+ min.
- Set timeout to 3600s (1 hour) to allow sharding to complete.

## Stress baseline summary (completed)

- Outcome: TimeoutError after 1200s
- Download stalled at 4.049 GB / 13.5 GB (~30%)
- Model load not reached; OOM was expected (~14 GB FP16 vs 8.22 GB RAM)
- Evidence: results/raw/baseline_stress_failure.json ✓

## After AirLLM run

Update this file with the next task (quantization experiment) once AirLLM is complete.
