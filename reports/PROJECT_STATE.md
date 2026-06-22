# PROJECT STATE — Assignment 05
Last updated: 2026-06-23

---

## Requirement Counts

| Status | Count | Out of |
|---|---|---|
| DONE | 6 | 74 |
| IN_PROGRESS | 14 | 74 |
| NOT_STARTED | 54 | 74 |

DONE items: A1, A7, B1, B2, B4, B5

Full detail: `reports/REQUIREMENTS_MATRIX.md`

---

## Completed Milestones

1. Requirements matrix (`reports/REQUIREMENTS_MATRIX.md`)
2. Repository scaffold (all `src/` scripts, `experiments/`, `results/`, `figures/`)
3. Hardware profile (`results/raw/hardware_profile.json`) — real measured data
4. Model selection (`results/raw/model_selection.json`) — three-role plan
5. Cache path safety — all paths redirected outside OneDrive to `C:\ai-model-cache\`
6. Environment setup (`results/raw/environment_setup.json`) — all packages installed

---

## Not Yet Done (in order)

- [ ] Warm-up baseline inference — Qwen2.5-0.5B → `results/raw/baseline_warmup_metrics.json`
- [ ] Stress baseline — OPT-6.7B OOM → `results/raw/baseline_stress_failure.json`
- [ ] AirLLM experiment — OPT-6.7B via layer-sharding → `results/raw/airllm_metrics.json`
- [ ] Quantization experiment — Q4_K_M + Q8_0 → `results/raw/quant_*_metrics.json`
- [ ] Benchmark summary table — `results/processed/summary_table.csv`
- [ ] Graph generation — `figures/*.png` (7 graphs)
- [ ] Quality comparison — `results/processed/quality_scores.json`
- [ ] Economic analysis — `results/processed/economic_analysis.json`
- [ ] Original extension — `results/raw/extension_disk_io.json`
- [ ] Final README polish — fill all `_TBD_` placeholders with real numbers
- [ ] Final self-assessment — `README.md` Section 13

---

## Latest Safe Next Step

Run **warm-up baseline only** using `Qwen/Qwen2.5-0.5B-Instruct`.
- Model cache: `C:\ai-model-cache\hf`
- `max_new_tokens`: 32 or 64
- Output: `results/raw/baseline_warmup_metrics.json`
- Summary: `results/processed/baseline_warmup_summary.md`
- Do NOT run OPT-6.7B yet.
- Do NOT run AirLLM yet.

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
