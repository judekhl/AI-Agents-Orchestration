# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Second quantization level Q8_0 — ask user approval before download.

**Background:**
All major work is done (48/74 requirements DONE, ~80/100 estimated). Extension complete, real TPOT measured.
Remaining Critical gap: E1 — only one explicit quantization level (Q4_K_M). Need a second level.

**Goal:** Add Q8_0 as second quantization level to satisfy E1.

**Before doing anything:**
1. Check if `bartowski/Qwen2.5-0.5B-Instruct-Q8_0.gguf` (or equivalent) exists at `C:\ai-model-cache\gguf\`
2. If not found: estimate download size (~520 MB for 0.5B Q8_0) and report to user
3. **Ask user for approval before downloading anything**
4. Only proceed with download + run after explicit user approval

**If user approves:**
- Download Q8_0 GGUF to `C:\ai-model-cache\gguf\` (not OneDrive)
- Run `src/run_quantized.py --strategy gguf --quant-filter q8_0 --gguf-dir C:\ai-model-cache\gguf\`
- Save `results/raw/quant_q8_0_metrics.json`
- Regenerate `figures/quant_tradeoff.png` with 3 data points: FP16 baseline, Q8_0, Q4_K_M
- Update README Section 6 and Section 7 with Q8_0 results
- Mark E1 DONE in REQUIREMENTS_MATRIX

**Do NOT:**
- Download anything without user approval
- Run 7B models
- Touch AirLLM
