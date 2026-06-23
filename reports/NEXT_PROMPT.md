# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Economic analysis + self-assessment + final README polish. Read CLAUDE.md, reports/PROJECT_STATE.md, reports/NEXT_PROMPT.md, README.md, and reports/REQUIREMENTS_MATRIX.md only.

**Background:**
All experiment runs are complete. Available real data:
- Warm-up baseline: 6.20 tok/s, 2.73 GB RAM, 10.33 s TTFT — `results/raw/baseline_warmup_metrics.json` ✓
- Stress baseline: timeout/OOM — `results/raw/baseline_stress_failure.json` ✓
- AirLLM: BLOCKED — `results/raw/airllm_compatibility.json` ✓
- Q4_K_M quantization: 26.24 tok/s, 0.55 GB RAM, 2.44 s TTFT — `results/raw/quant_q4_k_m_metrics.json` ✓
- Summary table: `results/processed/summary_table.csv` ✓
- Graphs: `figures/ttft_comparison.png`, `throughput_comparison.png`, `memory_comparison.png`, `runtime_comparison.png`, `quant_tradeoff.png` ✓

**Goal:** Complete README Sections 8 (Economic Analysis) and 12 (Self-Assessment), generate `figures/economic_breakeven.png`, and polish the README.

Steps:
1. Write `results/processed/economic_analysis.json` with:
   - API pricing: use Claude Haiku (cheapest current option, real published price as of 2026-06-23)
   - Hardware amortization: $800 laptop, 3-year life = ~$22/month
   - Electricity: i5-1135G7 TDP 28W × estimated inference hours × $0.12/kWh
   - Break-even calculation
   - monthly_cost_curve array for the breakeven graph
2. Run `src/plot_results.py` again to generate `figures/economic_breakeven.png` (the economic_analysis.json now exists).
3. Fill README Section 8 with real numbers from the JSON (all numbers must trace to the JSON file).
4. Fill README Section 12 (Self-Assessment) honestly — cite evidence files for each completed item.
5. Replace all remaining `_TBD_` in README with real values or explicit "not measured".
6. Update reports/PROJECT_STATE.md, reports/REQUIREMENTS_MATRIX.md, reports/NEXT_PROMPT.md.
7. Set NEXT_PROMPT.md to: "Final pre-submission review — check all requirements, fix any gaps."
8. Validate: python -m compileall src; git diff --check; git status
9. Commit and push.

Runtime budget: no model downloads, no inference. All steps are file writes + one short script run. No step should take > 1 min.

## Context
- Venv: C:\ai-envs\ai-agents-ex05\Scripts\activate
- Do not download models. Do not run inference. Data writing + plots only.
- Do not touch AirLLM or quantization.
