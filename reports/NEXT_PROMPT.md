# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Final submission review — check all requirements, fix gaps. Read CLAUDE.md, reports/PROJECT_STATE.md, reports/NEXT_PROMPT.md, README.md, and reports/REQUIREMENTS_MATRIX.md only.

**Background:**
All major work is done:
- Hardware, baseline (warm-up + stress), AirLLM (BLOCKED/documented), Q4_K_M quantization, graphs, economic analysis, self-assessment all complete.
- Estimated score: ~65/100. Missing: extension (J1-J3), TPOT, second explicit quant level.

**Goal:** Final review pass — find and fix any remaining gaps before submission.

Steps:
1. Grep README.md for any remaining `_TBD_` markers. Fix each with "not measured" or the real value.
2. Check REQUIREMENTS_MATRIX.md — for any requirement whose evidence file now exists, mark DONE.
3. Verify git status shows no untracked important files are missing from staging.
4. Validate: python -m json.tool results/processed/economic_analysis.json; python -m compileall src; git diff --check
5. Final commit and push if any changes were made.

Do NOT:
- Download models
- Run inference
- Generate new experiments
- Touch AirLLM

This is a documentation-only final pass.
