# PROJECT STATE — Assignment 05
Last updated: 2026-06-24 (AirLLM D2 measured on GPU — 74/74 DONE, 0 BLOCKED, 93–97/100 estimated)

---

## Requirement Counts

| Status | Count | Out of |
|---|---|---|
| DONE | 74 | 74 |
| IN_PROGRESS | 0 | 74 |
| NOT_STARTED | 0 | 74 |
| BLOCKED | 0 | 74 |

DONE items: A1–A7, B1–B5, C1–C4, D1–D5, E1–E3, F1–F8, G1–G9, H1–H9, I1–I13, J1–J3, K1–K8

D2 RESOLVED: AirLLM ran on a CUDA GPU (RTX 4060, second machine) — Qwen2.5-7B layer-streamed, 1.16 GB peak VRAM. Evidence: `results/raw/airllm_gpu_metrics.json`.

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

- **None blocking** — all 74 requirements DONE with real measured evidence.
- **D2 RESOLVED** — AirLLM was BLOCKED on the primary no-GPU laptop; completed on a second machine with an RTX 4060 (CUDA 12.4). Real metrics in `results/raw/airllm_gpu_metrics.json` (1.16 GB peak VRAM, 0.039 tok/s, 7589 disk-I/O samples). The CPU-only-laptop block is preserved in `airllm_compatibility.json` for honesty.

## Final Milestone

19. FP16 baseline streaming TPOT — `results/raw/baseline_warmup_streaming_metrics.json` ✓
    - TTFT: 14.29s (prefill, manual decode loop), TPOT: 387 ms/token, ITL range: 101–7701 ms
    - 12.5× slower decode than Q4_K_M (31 ms/token) — memory bandwidth bottleneck confirmed
    - I2 closed: decode stage concept connected to both FP16 and Q4_K_M real measurements

20. AirLLM GPU run (D2 RESOLVED) — `results/raw/airllm_gpu_metrics.json` ✓
    - Qwen/Qwen2.5-7B-Instruct layer-streamed on RTX 4060 Laptop GPU (CUDA 12.4), 2nd machine
    - Peak VRAM 1.16 GB (7B on an 8 GB GPU), peak RAM 5.87 GB, 819 s / 32 tok = 0.039 tok/s
    - Disk peak 2.90 GB/s, avg 636 MB/s across 7589 samples — direct evidence of layer paging
    - Fixes: `layer_shards_saving_path` + `hf_token` kwargs in run_airllm.py; `transformers==4.45.2`
      (newer drops the per-layer RoPE fallback AirLLM relies on); `input_ids.to(model.device)`
    - Env (2nd machine): `C:\ai-envs\ai-agents-ex05`, Python 3.12, torch 2.5.1+cu124, airllm 2.11.0

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
