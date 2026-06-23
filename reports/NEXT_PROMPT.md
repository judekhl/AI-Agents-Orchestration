# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Quantization experiment. Read CLAUDE.md, reports/PROJECT_STATE.md, reports/NEXT_PROMPT.md, and src/run_quantized.py only. Do not read other files speculatively.

Goal: run llama.cpp inference with Qwen2.5-7B-Instruct-GGUF Q4_K_M and produce results/raw/quant_q4km_metrics.json. If time allows, repeat with Q8_0.

Steps:
1. Check if llama-cpp-python is installed in the venv. If not, install it (CPU-only build).
2. Check if Q4_K_M GGUF file already exists at C:\ai-model-cache\gguf\. If not, download it from bartowski/Qwen2.5-7B-Instruct-GGUF on HuggingFace (file: Qwen2.5-7B-Instruct-Q4_K_M.gguf, ~4.4 GB). Ask before downloading — confirm the user wants to proceed (download will take several minutes on this connection).
3. Run src/run_quantized.py with the correct config pointing to the GGUF file and output results/raw/quant_q4km_metrics.json.
4. If Q8_0 (~8 GB) is feasible within RAM (8.22 GB total), run it too. Otherwise skip and note it as infeasible.
5. Update reports/PROJECT_STATE.md with results.

Runtime budget: ask before any step expected to take > 5 min. No polling.

## Context
- Venv: C:\ai-envs\ai-agents-ex05\Scripts\activate
- GGUF target dir: C:\ai-model-cache\gguf\
- HF cache: C:\ai-model-cache\hf\
- Q4_K_M size: ~4.4 GB (fits in 8.22 GB RAM)
- Q8_0 size: ~8 GB (borderline — may OOM; test only if time allows)
- AirLLM is BLOCKED (documented in results/raw/airllm_compatibility.json)
- Do not touch AirLLM

---

## After quantization experiment

Update this file with the next task (benchmark summary table + graphs).
