# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Generate benchmark summary table and graphs from existing raw results. Read CLAUDE.md, reports/PROJECT_STATE.md, reports/NEXT_PROMPT.md, and src/plot_results.py only. Do not read other files speculatively.

**Background:**
- Warm-up baseline: `results/raw/baseline_warmup_metrics.json` — 6.20 tok/s, 2.73 GB RAM ✓
- Stress baseline: `results/raw/baseline_stress_failure.json` — timeout/OOM ✓
- AirLLM: BLOCKED — `results/raw/airllm_compatibility.json` ✓
- Q4_K_M quantization: `results/raw/quant_q4_k_m_metrics.json` — 26.24 tok/s, 0.55 GB RAM ✓

Goal: produce `results/processed/summary_table.csv` and all graphs under `figures/`.

Steps:
1. Read `src/plot_results.py`. If it is complete and correct, run it. If it needs fixes (missing import, wrong path, missing scenario handling for failure files), fix it minimally and run.
2. Expected outputs:
   - `results/processed/summary_table.csv`
   - `figures/ttft_comparison.png`
   - `figures/throughput_comparison.png`
   - `figures/memory_comparison.png`
   - `figures/runtime_comparison.png`
   - `figures/quant_tradeoff.png`
   (skip tpot_comparison and economic_breakeven — not enough data yet)
3. Update README.md Section 7 to embed graph images and replace the summary table with real data from the CSV.
4. Update reports/PROJECT_STATE.md and reports/REQUIREMENTS_MATRIX.md.
5. Set reports/NEXT_PROMPT.md to: "Economic analysis + self-assessment + final README polish."

Runtime budget: ask before any step expected to take > 5 min. No polling.

## Context
- Venv: C:\ai-envs\ai-agents-ex05\Scripts\activate
- Raw results dir: results/raw/
- Output dirs: results/processed/, figures/
- Do not download models. Do not run inference. Graphs only.

---

## After graphs are generated

Update this file with the next task (economic analysis + self-assessment + final README polish).
