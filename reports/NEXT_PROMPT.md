# Next Prompt

Paste the text below as the next Claude Code prompt.

---

Final 90+ readiness review.

Read only: CLAUDE.md, reports/PROJECT_STATE.md, reports/REQUIREMENTS_MATRIX.md, README.md

Task: Strict final review — check for any remaining gaps, stale text, broken links, or fabricated numbers before final submission.

Steps:
1. Grep README.md for any remaining _TBD_, TODO, NOT YET, BLOCKED markers that should now be resolved.
2. Verify REQUIREMENTS_MATRIX counts are consistent with DONE items listed.
3. Confirm all figures referenced in README exist in figures/.
4. Run: python -m json.tool on all results/raw/*.json files.
5. Run: python -m compileall src/ && git diff --check
6. Final commit and push if any fixes are made.

Do NOT run models, download anything, or touch AirLLM.
