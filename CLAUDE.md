# CLAUDE.md — Context for Claude Code Sessions

**Project:** Assignment 05 — Running a Massive LLM Locally: AirLLM, Quantization, Performance Benchmarking
**Repo:** https://github.com/judekhl/AI-Agents-Orchestration
**Target grade:** 90+ — but never claim 90+ until `reports/REQUIREMENTS_MATRIX.md` proves it with real evidence.

---

## Session Start Protocol

At the start of every session, read **only** these three files — nothing else unless the task names specific files:

1. `CLAUDE.md` (this file)
2. `reports/PROJECT_STATE.md`
3. `reports/NEXT_PROMPT.md`

Read `README.md` or `reports/REQUIREMENTS_MATRIX.md` **only when the task requires editing them.**
Do not read or explore other files speculatively.

---

## Core Rules

- **Do not fabricate** benchmark numbers, hardware specs, model results, costs, or screenshots.
- **Do not download models** unless the prompt explicitly says to download.
- **Do not store models inside the repo or inside OneDrive.**
- **Do not use a HuggingFace token** unless explicitly requested.
- **Do not use the `Explore` agent** unless the prompt explicitly asks for repo exploration.
- **Do not read the whole repo** — read only files needed for the current task.
- **Do not paste full files or long diffs** unless asked.
- **Do not over-explain** decisions already recorded in `reports/PROJECT_STATE.md`.
- After each meaningful milestone, update `reports/PROJECT_STATE.md` and commit if requested.

## Output Format

Keep final responses under 25 lines unless more is requested. Format:
1. Changed files (bullet list)
2. Validation result (one line each)
3. `git status`
4. Committed/pushed yes/no
5. Next step (one sentence)

---

## Runtime and Token Budget Rules

- **No polling.** Do not repeatedly check process status with shell commands while waiting. Run once, wait for completion, then summarize.
- **No broad Explore.** Do not use the Explore agent unless the prompt explicitly says to.
- **Ask before long tasks.** If a command, model download, or inference run is expected to take more than 5 minutes, ask the user before starting it.
- **Silent waiting.** For long background commands, run once with a clear timeout and wait silently for the completion notification. Do not issue intermediate status checks.
- **Read only named files.** For each task, read only the files explicitly listed in the prompt.
- **No large tables unprompted.** Do not print wide tables or long diffs unless asked.
- **Prefer small validation commands.** Use targeted checks (`json.tool`, `compileall`, `git diff --check`) not full repo scans.

---

## Hardware (measured — do not change)

| Field | Value |
|---|---|
| CPU | Intel i5-1135G7 @ 2.40 GHz |
| Cores | 4 physical / 8 logical |
| RAM | 8.22 GB total |
| GPU | None — no CUDA, VRAM N/A |
| Disk free at probe | 38.44 GB (NVMe SSD) |
| OS | Windows 11 24H2 |

Evidence: `results/raw/hardware_profile.json`

---

## External Paths (outside OneDrive — mandatory)

| Purpose | Path |
|---|---|
| Python venv | `C:\ai-envs\ai-agents-ex05` |
| HuggingFace cache | `C:\ai-model-cache\hf` |
| AirLLM cache | `C:\ai-model-cache\airllm-cache` |
| AirLLM shards | `C:\ai-model-cache\airllm-shards` |
| GGUF files | `C:\ai-model-cache\gguf` |

**Why external:** repo lives inside OneDrive; model weights and venv files must not be synced.

Activate venv: `C:\ai-envs\ai-agents-ex05\Scripts\activate`

---

## Selected Models

| Role | Model | Reason |
|---|---|---|
| Warm-up | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB, fits in RAM, proves pipeline |
| Stress | `facebook/opt-6.7b` | ~14 GB FP16 > 8.22 GB RAM — OOM expected |
| Quantized fallback | `bartowski/Qwen2.5-7B-Instruct-GGUF` Q4\_K\_M | ~4.4 GB, fits in RAM |

Evidence: `results/raw/model_selection.json`

---

## Package Compatibility Note

- airllm 2.11.0 requires `optimum < 2.0` and `transformers < 4.49`
- **Working pinned versions:** `transformers==4.48.3`, `optimum==1.27.0`
- Do NOT `pip install --upgrade transformers` — it will break airllm import.

Evidence: `results/raw/environment_setup.json`

---

## Current Repo State

| Item | Status |
|---|---|
| GitHub repo pushed | ✓ |
| `reports/REQUIREMENTS_MATRIX.md` | ✓ |
| Repository scaffold | ✓ |
| `results/raw/hardware_profile.json` | ✓ |
| `results/raw/model_selection.json` | ✓ |
| Cache paths outside OneDrive | ✓ |
| `results/raw/environment_setup.json` | ✓ |
| Model weights downloaded | Partial (Qwen 0.5B ✓, OPT-6.7B 4/13.5 GB) |
| Warm-up baseline run | ✓ |
| Stress baseline / OOM evidence | ✓ (TimeoutError — download stalled) |
| AirLLM experiment | ✗ |
| Quantization experiment | ✗ |
| Graphs / summary table | ✗ |
| Economic analysis | ✗ |
| Original extension | ✗ |
| Final README polish | ✗ |

Requirement counts: **DONE 6 / IN_PROGRESS 17 / NOT_STARTED 51** (total 74)

---

## Key Files

```
reports/REQUIREMENTS_MATRIX.md      ← grading control; check before claiming DONE
reports/PROJECT_STATE.md            ← compact current state; update after milestones
reports/NEXT_PROMPT.md              ← next task prompt ready to paste
reports/notes.md                    ← working notes; decisions; compatibility fixes
experiments/configs/default_config.json
src/run_baseline.py                 ← warm-up and stress baseline
src/run_airllm.py
src/run_quantized.py
src/benchmark_common.py             ← RamSampler, DiskIoSampler, compute_metrics
results/raw/                        ← all raw experiment JSON files go here
results/processed/                  ← summaries, CSVs derived from raw
figures/                            ← all graphs go here
```

---

## Next Task

See `reports/NEXT_PROMPT.md` for the exact next prompt to run.
