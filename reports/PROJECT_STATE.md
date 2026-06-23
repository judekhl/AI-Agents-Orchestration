# PROJECT STATE — Assignment 05
Last updated: 2026-06-23 (graphs + summary table generated)

---

## Requirement Counts

| Status | Count | Out of |
|---|---|---|
| DONE | 7 | 74 |
| IN_PROGRESS | 17 | 74 |
| NOT_STARTED | 50 | 74 |

DONE items: A1, A7, B1, B2, B4, B5, D1, D3, D4, D5 (C1-C4 IN_PROGRESS; E1 IN_PROGRESS — Q4_K_M done)

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
9. AirLLM compatibility check — `results/raw/airllm_compatibility.json` — BLOCKED: requires sharded model format (large models only) and CUDA GPU; neither available. Also found bug in run_airllm.py (wrong kwarg `cache_dir` → should be `layer_shards_saving_path`).
10. Q4_K_M GGUF benchmark — `results/raw/quant_q4_k_m_metrics.json` — **26.24 tok/s, 0.55 GB RAM** (Qwen2.5-0.5B-Instruct Q4_K_M, llama-cpp-python, CPU-only). vs warm-up baseline: +323% throughput, −80% RAM.
11. Graphs + summary table — `results/processed/summary_table.csv` ✓ + 5 figures in `figures/` ✓ (ttft, throughput, memory, runtime, quant_tradeoff).

---

## Not Yet Done (in order)

- [x] Warm-up baseline inference — Qwen2.5-0.5B → `results/raw/baseline_warmup_metrics.json` ✓
- [x] Stress baseline — OPT-6.7B → `results/raw/baseline_stress_failure.json` ✓ (TimeoutError: download stalled at 4/13.5 GB; OOM expected on load)
- [BLOCKED] AirLLM experiment — BLOCKED: requires GPU (cuda:0) + sharded model format; `results/raw/airllm_compatibility.json` documents both blockers
- [BLOCKED] Q4_K_M 7B GGUF download — stalled after ~9 hours; no file written; `results/raw/quant_q4_download_failure.json` documents failure
- [x] Quantization experiment (Q4_K_M) — Qwen2.5-0.5B GGUF Q4_K_M ✓ — 26.24 tok/s, 0.55 GB RAM; `results/raw/quant_q4_k_m_metrics.json` ✓
- [x] Benchmark summary table — `results/processed/summary_table.csv` ✓
- [x] Graph generation — `figures/*.png` — 5 graphs generated ✓ (ttft, throughput, memory, runtime, quant_tradeoff)
- [ ] Quality comparison — `results/processed/quality_scores.json`
- [ ] Economic analysis — `results/processed/economic_analysis.json`
- [ ] Original extension — `results/raw/extension_disk_io.json`
- [ ] Final README polish — fill all `_TBD_` placeholders with real numbers
- [ ] Final self-assessment — `README.md` Section 13

---

## Latest Safe Next Step

Economic analysis + self-assessment + final README polish:
- Write `results/processed/economic_analysis.json` with API cost, on-prem cost, break-even
- Generate `figures/economic_breakeven.png`
- Fill README Sections 8, 12 with real data
- Final self-assessment

Exact prompt: see `reports/NEXT_PROMPT.md`

---

## Important Files

| File | Purpose |
|---|---|
| `reports/REQUIREMENTS_MATRIX.md` | Grading control — 74 requirements with status |
| `results/raw/hardware_profile.json` | Measured hardware (CPU, RAM, disk, OS) |
| `results/raw/model_selection.json` | Three-role model feasibility analysis |
| `results/raw/environment_setup.json` | Installed package versions, import health |
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
