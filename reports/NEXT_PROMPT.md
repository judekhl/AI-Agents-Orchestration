# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Quantization experiment with small GGUF model. Read CLAUDE.md, reports/PROJECT_STATE.md, reports/NEXT_PROMPT.md, and src/run_quantized.py only. Do not read other files speculatively.

**Background:** The Q4_K_M 7B GGUF download was attempted for ~9 hours and produced nothing. Do not attempt the 7B Q4 download again inside Claude.

Goal: find a smaller GGUF quantized model already suitable for CPU-only 8 GB RAM (preferably 0.5B–1.5B parameters), get user approval before downloading, run llama.cpp benchmark, and document it as quantized fallback metrics.

Steps:
1. Identify a small GGUF Q4_K_M model: prefer `Qwen2.5-0.5B-Instruct-GGUF` or `Qwen2.5-1.5B-Instruct-GGUF` from a reliable HuggingFace repo (e.g. bartowski or lmstudio-community). Note the exact filename and size.
2. **Ask the user for approval** before downloading. State the filename, size, and source repo.
3. After approval, download the GGUF file to `C:\ai-model-cache\gguf\`.
4. Check if llama-cpp-python is installed in the venv (`C:\ai-envs\ai-agents-ex05\Scripts\activate`). If not, install CPU-only build.
5. Run `src/run_quantized.py --strategy gguf --gguf-dir C:\ai-model-cache\gguf\` and save output to `results/raw/quant_q4km_metrics.json`.
6. Update `reports/PROJECT_STATE.md` with results and update `reports/NEXT_PROMPT.md` to the next task (benchmark summary table + graphs).

Runtime budget: ask before any step expected to take > 5 min. No polling.

## Context
- Venv: C:\ai-envs\ai-agents-ex05\Scripts\activate
- GGUF target dir: C:\ai-model-cache\gguf\
- Q4_K_M 7B (~4.4 GB) download FAILED after 9 hours — do not retry
- AirLLM is BLOCKED (documented in results/raw/airllm_compatibility.json)
- Do not touch AirLLM

---

## After quantization experiment

Update this file with the next task (benchmark summary table + graphs).
