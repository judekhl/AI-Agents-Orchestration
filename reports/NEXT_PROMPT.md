# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Final submission review — validate and commit all closure-sprint files.

Read only: CLAUDE.md, reports/PROJECT_STATE.md, reports/REQUIREMENTS_MATRIX.md

Tasks (in order):
1. Run: `python -m json.tool results/processed/quality_scores.json` — must succeed
2. Run: `python -m json.tool results/processed/economic_analysis.json` — must succeed
3. Run: `python -m compileall src/ -q` — must succeed
4. Run: `git diff --check` — must return clean (no trailing whitespace)
5. Grep README.md for remaining `_TBD_` or `TODO` markers
6. Confirm all figures referenced in README exist in figures/ and figures/screenshots/
7. If all pass: `git add` new/changed files, commit, push

Files to stage (not yet committed):
- `results/processed/quality_scores.json`
- `figures/quality_comparison.png`
- `figures/screenshots/` (4 PNGs)
- `README.md`
- `reports/REQUIREMENTS_MATRIX.md`
- `reports/PROJECT_STATE.md`
- `reports/NEXT_PROMPT.md`

Commit message: "docs: 90+ closure sprint — quality scoring, evidence snapshots, matrix closure"

Do NOT run models, download anything, or touch AirLLM.
Keep output under 25 lines.
