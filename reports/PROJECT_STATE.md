# PROJECT STATE — Assignment 05
Last updated: 2026-06-23 (final readiness review — 69/74 DONE, 85–90/100 estimated)

---

## Requirement Counts

| Status | Count | Out of |
|---|---|---|
| DONE | 69 | 74 |
| IN_PROGRESS | 2 | 74 |
| NOT_STARTED | 2 | 74 |
| BLOCKED | 1 | 74 |

DONE items: A1–A7, B1–B5, C1–C4, D1, D3, D4, D5, E1–E3, F1–F8, G1–G8, H1–H7, H9, I1, I3–I13, J1–J3, K1–K5, K7, K8

IN_PROGRESS: I2 (TPOT: Q4 streaming done; FP16 baseline null), K6 (try/except present in all scripts; OOM handler partial)
NOT_STARTED: G9 (roofline — optional), H8 (cloud GPU comparison — optional)
BLOCKED: D2 (AirLLM no GPU)

Full detail: `reports/REQUIREMENTS_MATRIX.md`

---

## Completed Milestones

1. Requirements matrix (`reports/REQUIREMENTS_MATRIX.md`)
2. Repository scaffold (all `src/` scripts, `experiments/`, `results/`, `figures/`)
3. Hardware profile (`results/raw/hardware_profile.json`) — real measured data
4. Model selection (`results/raw/model_selection.json`) — three-role plan
5. Cache path safety — all paths redirected outside OneDrive to `C:\ai-model-cache\`
6. Environment setup (`results/raw/environment_setup.json`) — all packages installed
7. Warm-up baseline — `results/raw/baseline_warmup_metrics.json` — 6.2 tok/s, 2.73 GB RAM, no OOM
8. Stress baseline — `results/raw/baseline_stress_failure.json` — TimeoutError after 1200s; download stalled at 4 GB / 13.5 GB; confirms OOM risk on load
9. AirLLM compatibility check — `results/raw/airllm_compatibility.json` — BLOCKED: requires sharded model format (large models only) and CUDA GPU; neither available.
10. Q4_K_M GGUF benchmark — `results/raw/quant_q4_k_m_metrics.json` — **26.24 tok/s, 0.55 GB RAM** (Qwen2.5-0.5B-Instruct Q4_K_M, llama-cpp-python, CPU-only).
11. Graphs + summary table — `results/processed/summary_table.csv` + 9 figures in `figures/`
12. Economic analysis — `results/processed/economic_analysis.json` + `figures/economic_breakeven.png`. Break-even: ~261K requests/month.
13. Self-assessment + README polish — all `_TBD_` markers replaced; Section 12 written.
14. Streaming TPOT extension — `results/raw/extension_prompt_scaling.json` — 5 prompts; real per-token timestamps; `figures/extension_prompt_scaling.png` + `figures/tpot_comparison.png`
15. Q8_0 second quantization level — `results/raw/quant_q8_0_metrics.json` — 17.56 tok/s, 0.58 GB RAM
16. Quality scoring — `results/processed/quality_scores.json` — FP16: 23/25, Q4_K_M: 22/25, Q8_0: 17/25; `figures/quality_comparison.png`
17. Evidence snapshots — `figures/screenshots/` — 4 PNGs generated programmatically from raw JSON
18. Matrix closure sprint — A2–A6, B3, C1–C4, F8, K1–K5, K7, K8 all closed to DONE

---

## Remaining Gaps (honest accounting)

- **F2 / I2** — FP16 baseline TPOT is null (non-streaming inference; TTFT ≈ total runtime)
- **K6** — Error handling is partial (some try/except, no OOM handler for stress; stress failed before load)
- **G9** — Roofline model graph not generated (optional requirement)
- **H8** — Cloud GPU cost comparison not done (optional)
- **D2** — AirLLM permanently BLOCKED (no CUDA GPU)

---

## Latest Safe Next Step

Final review and commit:
- `python -m json.tool results/processed/quality_scores.json` — validate JSON
- `python -m compileall src/ -q` — validate scripts
- `git diff --check` — no trailing whitespace
- `git add` new files + commit + push

Exact prompt: see `reports/NEXT_PROMPT.md`

---

## Important Files

| File | Purpose |
|---|---|
| `reports/REQUIREMENTS_MATRIX.md` | Grading control — 74 requirements with status |
| `results/raw/hardware_profile.json` | Measured hardware (CPU, RAM, disk, OS) |
| `results/raw/model_selection.json` | Three-role model feasibility analysis |
| `results/raw/environment_setup.json` | Installed package versions, import health |
| `results/processed/quality_scores.json` | Manual quality rubric — 3 models × 5 dimensions |
| `results/processed/economic_analysis.json` | Break-even analysis (~261K req/month) |
| `figures/screenshots/` | 4 evidence snapshots (programmatically generated) |
| `experiments/configs/default_config.json` | All cache paths, model IDs, experiment params |
| `src/run_baseline.py` | Baseline inference script |
| `src/benchmark_common.py` | RamSampler, DiskIoSampler, compute_metrics |

---

## Safety Rules

- Before any model download: confirm path is under `C:\ai-model-cache\` (not OneDrive).
- Before updating README numbers: confirm the raw JSON file exists with real data.
- Before marking any requirement DONE: evidence file must exist and contain real data.
- Do not upgrade `transformers` — pinned to 4.48.3 for airllm compatibility.

## Runtime and Token Budget Rules

- **No polling.** Run long commands once with a timeout; wait silently for completion. No repeated status checks.
- **Ask before long tasks.** If a command or download will take > 5 min, ask first.
- **No broad Explore.** Only read files named in the prompt.
- **Output under 25 lines.** No large tables or long diffs unless asked.
- **Prefer small validation.** Use `json.tool`, `compileall`, `git diff --check` — not full repo scans.
