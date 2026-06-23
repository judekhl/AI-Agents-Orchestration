# PROJECT STATE — Assignment 05
Last updated: 2026-06-23

---

## Requirement Counts

| Status | Count | Out of |
|---|---|---|
| DONE | 7 | 74 |
| IN_PROGRESS | 17 | 74 |
| NOT_STARTED | 50 | 74 |

DONE items: A1, A7, B1, B2, B4, B5 (C1-C4 IN_PROGRESS — warm-up + stress both done)

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

---

## Not Yet Done (in order)

- [x] Warm-up baseline inference — Qwen2.5-0.5B → `results/raw/baseline_warmup_metrics.json` ✓
- [x] Stress baseline — OPT-6.7B → `results/raw/baseline_stress_failure.json` ✓ (TimeoutError: download stalled at 4/13.5 GB; OOM expected on load)
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

Run **AirLLM experiment** using `facebook/opt-6.7b` via layer-sharding.
- AirLLM cache: `C:\ai-model-cache\airllm-cache`
- AirLLM shards: `C:\ai-model-cache\airllm-shards`
- OPT-6.7B weights are partially cached in `C:\ai-model-cache\hf` (4 GB / 13.5 GB); AirLLM will need the full model
- Output: `results/raw/airllm_metrics.json`
- Script: `src/run_airllm.py`

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
