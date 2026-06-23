# Next Prompt

Paste the text below as the next Claude Code prompt.

---

AirLLM compatibility check only. Do not continue downloading OPT-6.7B. Do not shard OPT-6.7B. Do not run a long AirLLM experiment. Read src/run_airllm.py only if needed. Verify airllm imports and inspect whether the script can run with a tiny/warm-up model without large downloads. If a model run is needed, use only a tiny model and a strict 5-minute timeout. If AirLLM requires the full OPT-6.7B download/sharding, stop and document AirLLM as blocked by download/time constraints. Save evidence to results/raw/airllm_compatibility.json and results/processed/airllm_compatibility_summary.md. Do not run quantization yet.

---

## Context before running

- Read CLAUDE.md, reports/PROJECT_STATE.md, reports/NEXT_PROMPT.md, src/run_airllm.py
- Venv: `C:\ai-envs\ai-agents-ex05\Scripts\activate`
- OPT-6.7B is partially cached (~4 GB / 13.5 GB) — do NOT resume download
- Runtime budget: 5 minutes max for any model run
- If AirLLM cannot run without the full OPT-6.7B, document as BLOCKED and move on to quantization

## After AirLLM check

Update this file with the next task (quantization experiment) once AirLLM compatibility check is complete.
