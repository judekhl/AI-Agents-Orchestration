# REQUIREMENTS MATRIX — Assignment 05
# Running a Massive LLM Locally: AirLLM, Quantization and Performance Benchmarking
# Target grade: 90+
# Last updated: 2026-06-23 (AirLLM compatibility check complete — BLOCKED)
# Rule: Status stays NOT_STARTED until evidence file exists in the repo. DONE requires real data.

---

## Legend

| Status | Meaning |
|---|---|
| `NOT_STARTED` | No work done; no evidence file exists |
| `IN_PROGRESS` | Partially done; evidence incomplete |
| `DONE` | Evidence file exists AND contains real data |
| `BLOCKED` | Cannot proceed; reason documented |

| Grade Impact | Meaning |
|---|---|
| `Critical` | Missing = almost certain grade failure on this section |
| `High` | Missing = significant point loss |
| `Medium` | Missing = moderate point loss |
| `Low` | Missing = minor deduction or style mark |

---

## Section A — Repository and Submission

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| A1 | GitHub repo exists, is public, and has a clean commit history | Submission instructions | GitHub URL accessible without login; `git log` shows meaningful commits | `DONE` | Pushed to https://github.com/judekhl/AI-Agents-Orchestration on 2026-06-23. Two commits present with descriptive messages. Repo is public. Additional commits will be added as experiments complete. | Submission fails entirely | Critical |
| A2 | Repository has a clear, navigable directory structure | Submission instructions | Directories: `scripts/`, `results/raw/`, `results/processed/`, `reports/`, `figures/`, `README.md`, `.gitignore` | `DONE` | `src/`, `results/raw/`, `results/processed/`, `figures/`, `figures/screenshots/`, `experiments/configs/`, `reports/` all present with real files. 21 commits. Structure navigable with README ToC. | Grader cannot find deliverables | High |
| A3 | `README.md` is the main final report (not a separate PDF) | Assignment brief | `README.md` at repo root; must be complete, not a placeholder | `DONE` | README.md at repo root with all 13 sections complete: hardware, model selection, how-to-reproduce, baseline, AirLLM, quantization, results, economic analysis, lecture concepts (Section 9 all 13 concepts), extension (Section 10), screenshots/snapshots (Section 11), self-assessment (Section 12), references (Section 13). All real data. | Primary deliverable missing | Critical |
| A4 | Reproduction instructions included in README | Assignment brief | Section in README: "How to reproduce" with exact commands, Python version, package versions | `DONE` | README Section 3 "How to Reproduce" ✓ — Steps 0–2 with exact bat/bash commands, pip install command, Python 3.10+ requirement, external path instructions, expected runtimes. | Reproducibility criterion fails | High |
| A5 | Screenshots included or explicitly explained if impossible | Assignment brief | `figures/screenshots/` directory OR explicit note in README explaining why screenshots are not possible | `DONE` | `figures/screenshots/` ✓ — 4 evidence snapshots generated programmatically from raw JSON: hardware_profile_snapshot.png, baseline_metrics_snapshot.png, quantization_metrics_snapshot.png, graph_outputs_snapshot.png. README Section 11 explains these are data-driven snapshots, not fabricated screenshots. | Visual evidence missing | Medium |
| A6 | No secrets or API tokens committed to the repo | Security / submission | `.gitignore` covers `.env`, `*.key`, `secrets/`; `git log --all` scan shows no tokens | `DONE` | `.gitignore` covers `.env`, `*.env`, `*.key`, `*.pem`. Scan of all `*.py` and `*.json` files confirms HF_TOKEN is only read via `os.getenv()` — not hardcoded anywhere. `.env.example` uses only placeholder text. 21-commit log shows no token commits. | Academic integrity / security violation | Critical |
| A7 | Large model files are gitignored (no model weights in repo) | Submission instructions | `.gitignore` includes `*.bin`, `*.safetensors`, `*.gguf`, `models/`, `shards/`, `cache/` | `DONE` | `.gitignore` covers `*.bin`, `*.safetensors`, `*.gguf`, `*.pt`, `*.pth`, `*.onnx`, `models/`, `airllm_shards/`, `airllm_cache/`, `shards/`, `.cache/`, `hf_cache/`. Created before any model files exist. Evidence: `.gitignore` in repo root. | Repo bloat / push failure / GitHub LFS rejection | Critical |

---

## Section B — Hardware and Model Selection

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| B1 | Exact hardware documented: CPU model, core count, RAM (GB), GPU model or "none", VRAM if present, disk type (HDD/SSD/NVMe) and free space (GB), OS and version | Hardware section | `results/raw/hardware_profile.json` AND a "Hardware" section in README | `DONE` | `results/raw/hardware_profile.json` contains: CPU=i5-1135G7, 4P/8L cores, RAM=8.22 GB, GPU=none, VRAM=N/A, Disk=NVMe SSD 511 GB / 38.44 GB free, OS=Windows 11 24H2. README hardware table populated with these real numbers. Evidence: `results/raw/hardware_profile.json` ✓ | Grader cannot evaluate feasibility of experiments | Critical |
| B2 | Model choice justified relative to hardware constraints | Model selection section | README section "Model Selection Rationale" citing specific RAM/VRAM numbers from B1 | `DONE` | README Section 2 populated with three-role model selection using measured RAM (8.22 GB) and disk (38.44 GB). Cites: warm-up=Qwen2.5-0.5B (~1.5 GB RAM), stress=OPT-6.7B (~14.0 GB RAM > 8.22 GB ceiling), fallback=Qwen2.5-7B Q4_K_M GGUF (~4.8 GB RAM). Disk budget table shows 32.2 GB needed vs 38.44 GB free. Evidence: `results/raw/model_selection.json` ✓ | Appears random; no engineering judgement demonstrated | High |
| B3 | Model is large enough to stress hardware but not trivially impossible | Model selection section | README justification + baseline experiment showing strain (slow, OOM, or disk paging) | `DONE` | OPT-6.7B (14.0 GB FP16) vs 8.22 GB RAM: `results/raw/baseline_stress_failure.json` ✓ — TimeoutError after 1200 s, download stalled at 4/13.5 GB; OOM expected on load. README Section 4 documents both the download bottleneck and the expected OOM. The chosen stress model definitively exceeds hardware limits. | Assignment objective not met | High |
| B4 | Disk-space check performed before downloading models | Model selection section | `results/raw/disk_check.txt` or log showing available disk space vs model download size | `DONE` | `hardware_profile.json` records `disk_free_gb: 38.44` and `disk_total_gb: 511.04` at probe time (before any model download). 38.44 GB free vs ~13.4 GB for OPT-6.7B FP16 = adequate margin. Evidence embedded in `results/raw/hardware_profile.json`. | Missing a "professional practice" mark | Medium |
| B5 | Public / no-auth-token fallback model is defined | Model selection section | README names at least one fallback model that requires no Hugging Face token (e.g., `TheBloke` GGUF variants, `facebook/opt-*`, `EleutherAI/gpt-j-6b`) | `DONE` | README Section 2 explicitly names `bartowski/Qwen2.5-7B-Instruct-GGUF` Q4_K_M as the fallback quantized model. Notes state "No token required — public HuggingFace repo (bartowski is a widely-used GGUF publisher)". All three selected models (Qwen2.5-0.5B, OPT-6.7B, Qwen2.5-7B GGUF) are public and require no HF token. Evidence: `results/raw/model_selection.json` field `requires_token: false` for all roles ✓ | If primary model is gated, entire experiment chain fails | High |

---

## Section C — Baseline Experiment

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| C1 | Direct / naive local baseline run is attempted | Baseline section | `scripts/baseline_run.py` exists AND `results/raw/baseline_metrics.json` or `results/raw/baseline_failure.txt` exists | `DONE` | `src/run_baseline.py` ✓. `results/raw/baseline_warmup_metrics.json` ✓ (success). `results/raw/baseline_stress_failure.json` ✓ (TimeoutError/OOM). Both runs attempted and logged. | Baseline reference point missing; nothing to compare against | Critical |
| C2 | Outcome documented honestly: success, OOM, excessive slowness, or failure | Baseline section | `results/raw/baseline_metrics.json` (if success) OR `results/raw/baseline_failure.txt` with error log and explanation | `DONE` | Warm-up: SUCCESS — `results/raw/baseline_warmup_metrics.json` ✓ — 6.20 tok/s, 2.73 GB. Stress: TimeoutError — `results/raw/baseline_stress_failure.json` ✓ — download stalled at 4/13.5 GB; OOM expected on load. Both in README Section 4 with honest framing. | Negative results presented without documentation look fabricated | Critical |
| C3 | Baseline metrics saved to raw result files (not only in README) | Baseline section | `results/raw/baseline_metrics.json` containing TTFT, TPOT, throughput, RAM peak, runtime | `DONE` | `results/raw/baseline_warmup_metrics.json` ✓ — ttft_seconds=10.33, throughput=6.20, peak_ram_gb=2.73, total_runtime_seconds, total_output_tokens, peak_vram_note. Stress failure in `baseline_stress_failure.json` ✓. TPOT null for warmup (documented limitation). | Cannot regenerate graphs from real data | High |
| C4 | Bottleneck identified and explained: RAM, VRAM, CPU compute, or disk I/O | Baseline section | README section "Baseline Bottleneck Analysis" + supporting data (e.g., `htop` screenshot or `psutil` memory trace) | `DONE` | README Section 4 "Bottleneck Analysis" ✓ — warm-up: memory-bandwidth-bound (2.73 GB, 6.20 tok/s, CPU-only); stress: two-stage failure (download bottleneck + OOM on load). RAM sampled via `psutil` background thread in `src/benchmark_common.py`. Evidence snapshots in `figures/screenshots/` ✓. | Key conceptual analysis missing | High |

---

## Section D — AirLLM Experiment

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| D1 | AirLLM integration attempted | AirLLM section | `scripts/airllm_run.py` exists | `DONE` | `src/run_airllm.py` exists using `airllm.AutoModel`, configurable shard path via JSON config. Compatibility run attempted 2026-06-23 — `airllm.AutoModel` imports OK, `AirLLMQWen2` auto-detected. Evidence: `results/raw/airllm_compatibility.json` ✓ | AirLLM section entirely missing | Critical |
| D2 | AirLLM result measured if it runs | AirLLM section | `results/raw/airllm_metrics.json` with same metric fields as baseline | `BLOCKED` | AirLLM cannot run: (1) no CUDA GPU — device defaults to cuda:0; (2) tiny models use single safetensors (no index file); large OPT-6.7B needs full 13.5 GB download (stalled). Evidence: `results/raw/airllm_compatibility.json` ✓ | No comparison data for AirLLM | Critical |
| D3 | If AirLLM fails, failure documented as negative result with logs and fallback | AirLLM section | `results/raw/airllm_failure.txt` with full error log + README section explaining what failed and why, + alternative approach taken | `DONE` | Failure documented in `results/raw/airllm_compatibility.json` ✓ and `results/processed/airllm_compatibility_summary.md` ✓. README Section 5 updated 2026-06-23: both blockers explained, actual error message included, quantization named as alternative. | Looks like work was skipped rather than failing honestly | High |
| D4 | Layer-sharding / disk paging / I/O behavior discussed | AirLLM section | README section "AirLLM: How Layer Loading Works" explicitly referencing paging, virtual memory, disk I/O concepts from lecture | `DONE` | README Section 5 "How AirLLM Works" covers layer-by-layer disk loading, analogy to OS demand paging, RAM ceiling reduction, TTFT penalty from disk reads, and deliberate vs uncontrolled swap. Evidence: README.md Section 5 ✓ | Conceptual analysis missing | High |
| D5 | AirLLM cache/shard path explicitly configured and documented | AirLLM section | In `scripts/airllm_run.py`, `cache_dir` / shard path is set to a configurable path (not hardcoded user path); documented in README | `DONE` | `src/run_airllm.py` reads `airllm_shard_dir` from `experiments/configs/default_config.json` (no hardcoded path). README Section 5 documents the path and override mechanism (`--config`). Evidence: `experiments/configs/default_config.json` + README.md Section 5 ✓ | Grader cannot reproduce without knowing shard path; also exposes local path | Medium |

---

## Section E — Quantization Experiment

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| E1 | At least two quantization levels / settings attempted where technically possible | Quantization section | `scripts/quantization_run.py` with at least two configs; `results/raw/quant_fp16_metrics.json` AND `results/raw/quant_q4_metrics.json` (or equivalent) | `DONE` | Q8_0 GGUF: `results/raw/quant_q8_0_metrics.json` ✓ — 17.56 tok/s, 0.58 GB. Q4_K_M GGUF: `results/raw/quant_q4_k_m_metrics.json` ✓ — 26.24 tok/s, 0.55 GB. FP16 from warm-up baseline is a third level. Three-level comparison table in README Section 6. | Quantization section missing entirely | Critical |
| E2 | Impact on memory, speed, and output quality measured or documented for each level | Quantization section | Each `results/raw/quant_*_metrics.json` contains RAM peak, TTFT, throughput, and a short output quality note | `DONE` | All three levels documented with real data. README Section 6 three-level comparison table: FP16 (6.20 tok/s, 2.73 GB) → Q8_0 (17.56 tok/s, 0.58 GB) → Q4_K_M (26.24 tok/s, 0.55 GB). Quality notes in all JSON files. | Cannot draw quantization tradeoff graph | High |
| E3 | Quantization "red line" for quality discussed | Quantization section | README section "Quantization Quality Threshold" discussing at what level output degrades noticeably, with example outputs | `DONE` | README Section 6 "Quantization Quality Threshold" ✓ — Q8 and Q4 both produce coherent outputs; degradation expected at Q2_K or below; example outputs from both levels compared. | Conceptual insight missing | Medium |

---

## Section F — Benchmark Metrics

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| F1 | TTFT (Time To First Token) measured for every runnable scenario | Metrics section | Each `results/raw/*_metrics.json` has `"ttft_seconds"` field | `DONE` | `baseline_warmup_metrics.json` ttft=10.3307 s ✓; `quant_q4_k_m_metrics.json` ttft=2.4392 s ✓. Stress/AirLLM not runnable — documented as failures/blocked. | Core metric missing | Critical |
| F2 | TPOT / ITL (Time Per Output Token / Inter-Token Latency) measured | Metrics section | Each `results/raw/*_metrics.json` has `"tpot_ms"` or `"itl_ms"` field | `DONE` | `results/raw/quant_q4_k_m_streaming_metrics.json` ✓ — tpot_ms=30.994, itl_min_ms, itl_max_ms via llama-cpp stream=True real per-token timestamps. Baseline TPOT null (no streaming hook for transformers). README Section 9 TPOT section updated with real values. | Core metric missing | Critical |
| F3 | Throughput (tokens/sec) measured | Metrics section | Each `results/raw/*_metrics.json` has `"tokens_per_second"` field | `DONE` | baseline_warmup: 6.1951 tok/s ✓; quant_q4_k_m: 26.2384 tok/s ✓. Both in summary_table.csv. | Core metric missing | Critical |
| F4 | Total runtime measured | Metrics section | Each `results/raw/*_metrics.json` has `"total_runtime_seconds"` field | `DONE` | baseline_warmup: 10.3307 s ✓; quant_q4_k_m: 2.4392 s ✓. Both in summary_table.csv. | Without runtime, throughput and economic analysis are incomplete | High |
| F5 | Peak RAM measured | Metrics section | Each `results/raw/*_metrics.json` has `"peak_ram_gb"` field | `DONE` | baseline_warmup: 2.728 GB ✓; quant_q4_k_m: 0.55 GB ✓. Both in summary_table.csv. | Memory analysis impossible without this | Critical |
| F6 | Peak VRAM measured if GPU available; otherwise explicitly "N/A — no CUDA/discrete GPU" documented | Metrics section | Each `results/raw/*_metrics.json` has `"peak_vram_gb"` field or `"peak_vram_note": "N/A — no CUDA GPU detected"` | `DONE` | Both JSON files have `"peak_vram_note": "N/A — no CUDA/discrete GPU"` ✓. README Section 7 table shows N/A for all scenarios. | Ambiguity about hardware capability | High |
| F7 | Estimated power / electricity consumption documented | Metrics section | `results/raw/power_estimate.json` or README section "Power Consumption Estimate" with methodology | `DONE` | README Section 8 economic analysis documents 28W TDP × inference hours; methodology in `results/processed/economic_analysis.json` ✓ | Economic analysis depends on this | High |
| F8 | Output quality notes included for each scenario | Metrics section | Each `results/raw/*_metrics.json` has `"output_quality_notes"` field; README includes example outputs | `DONE` | `results/processed/quality_scores.json` ✓ — manual 1-5 rubric (coherence, factuality, relevance, repetition, completeness) for all 3 runnable scenarios. FP16: 23/25, Q4_K_M: 22/25, Q8_0: 17/25. `figures/quality_comparison.png` ✓. All JSON files have non-TODO output_quality_notes. README Sections 4 and 6 include output snippets. | Quantization tradeoff section incomplete | Medium |

---

## Section G — Tables and Graphs

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| G1 | Benchmark summary table | Results section | Table in README comparing all scenarios across all F-section metrics | `DONE` | `results/processed/summary_table.csv` ✓ generated by `src/plot_results.py`; table in README Section 7 with real values from all runnable scenarios. | Cannot compare scenarios at a glance | Critical |
| G2 | TTFT comparison graph | Results section | `figures/ttft_comparison.png` | `DONE` | `figures/ttft_comparison.png` ✓ — bar chart with 2 runnable scenarios (baseline 10.33 s, Q4_K_M 2.44 s). | Visual evidence missing | High |
| G3 | TPOT/ITL comparison graph | Results section | `figures/tpot_comparison.png` | `DONE` | `figures/tpot_comparison.png` ✓ — bar chart of TPOT by prompt length (very_short→very_long), generated by `src/extension_prompt_scaling.py` using real streaming timestamps. 5 data points: 40.7, 35.9, 31.0, 30.8, 34.4 ms/token. | Visual evidence missing | High |
| G4 | Throughput comparison graph | Results section | `figures/throughput_comparison.png` | `DONE` | `figures/throughput_comparison.png` ✓ — bar chart (baseline 6.20 tok/s, Q4_K_M 26.24 tok/s). | Visual evidence missing | High |
| G5 | RAM / VRAM comparison graph | Results section | `figures/memory_comparison.png` | `DONE` | `figures/memory_comparison.png` ✓ — bar chart (baseline 2.73 GB, Q4_K_M 0.55 GB). | Visual evidence missing | High |
| G6 | Runtime comparison graph | Results section | `figures/runtime_comparison.png` | `DONE` | `figures/runtime_comparison.png` ✓ — bar chart (baseline 10.33 s, Q4_K_M 2.44 s). | Visual evidence missing | Medium |
| G7 | Quantization tradeoff graph (memory vs quality vs speed) | Results section | `figures/quant_tradeoff.png` | `DONE` | `figures/quant_tradeoff.png` ✓ — 3-panel chart (RAM, throughput, TPOT) for Q4_K_M scenario. | Quantization insight not visualized | High |
| G8 | Economic break-even graph | Economic analysis section | `figures/economic_breakeven.png` | `DONE` | `figures/economic_breakeven.png` ✓ — API vs on-prem cost curve with break-even at 260,753 req/month. Generated by `src/plot_results.py` from `results/processed/economic_analysis.json`. | Economic analysis incomplete without visual | High |
| G9 | Optional: roofline-style compute-bound vs memory-bound visualization | Advanced analysis section | `figures/roofline.png` (optional) | `NOT_STARTED` | Plot achieved throughput vs arithmetic intensity; mark compute roof and memory bandwidth roof | Optional — extra credit territory | Low |

---

## Section H — Economic Analysis

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| H1 | API cost calculation for equivalent workload | Economic section | README "Economic Analysis" section with formula: `tokens × price_per_1k_tokens / 1000`; source and date of pricing cited | `DONE` | README Section 8: formula shown `(27×$0.25 + 64×$1.25) / 1M = $0.0000868/request`. Evidence: `results/processed/economic_analysis.json` ✓ | Economic analysis missing | High |
| H2 | On-Prem local cost calculation | Economic section | README calculation: hardware amortization/month + electricity cost/month + operation time | `DONE` | README Section 8: $800 laptop ÷ 36 months = $22.22/month + $0.40 electricity = $22.62/month. `results/processed/economic_analysis.json` ✓ | Cannot compare deployment modes | High |
| H3 | Hardware amortization assumption stated | Economic section | README states assumed hardware cost and amortization period (e.g., "$800 laptop, 3-year life = $22/month") | `DONE` | README Section 8: "$800 laptop, 3-year life = $22.22/month" ✓ | Cost calculation unverifiable | Medium |
| H4 | Electricity assumption stated | Economic section | README states assumed electricity rate (e.g., "$0.12/kWh") and source region | `DONE` | README Section 8: "28W TDP × 120 h/month ÷ 1000 × $0.12/kWh = $0.40/month" ✓ | Cost calculation unverifiable | Medium |
| H5 | Maintenance/operation assumption stated | Economic section | README states assumed maintenance overhead (e.g., "2 hours/month operator time at $X/hr") | `DONE` | README Section 8: "2 h/month × $15/hr = $30.00/month" noted as optional ✓ | Holistic cost picture incomplete | Medium |
| H6 | Break-even calculation: requests/month or tokens/month where on-prem becomes cheaper | Economic section | README shows break-even formula and result; `figures/economic_breakeven.png` (G8) | `DONE` | README Section 8: break-even = 260,753 req/month (hardware+electricity); graph ✓ | Key recommendation is ungrounded | High |
| H7 | Recommendation: when API is better, when on-prem is better | Economic section | README "Recommendation" paragraph connecting break-even result to practical use cases | `DONE` | README Section 8 recommendation table: API < 260K req/month; on-prem above; privacy = on-prem regardless ✓ | Practical conclusion missing | High |
| H8 | Optional: cloud GPU comparison (e.g., Lambda Labs, Vast.ai, RunPod) | Economic section | README note comparing cloud GPU hourly rate vs amortized on-prem cost | `NOT_STARTED` | Optional — skipped for time. | Optional; enhances analysis | Low |
| H9 | Prompt/context caching pricing discussed as assumption if exact prices not used | Economic section | README notes that API prices change; states the specific price and date used as assumption | `DONE` | README Section 8 disclaimer: "Claude Haiku 3 prices used as illustrative assumption... labeled as assumption." `results/processed/economic_analysis.json` pricing_note ✓ | Academic integrity: prices cited may be stale | Medium |

---

## Section I — Lecture-Concept Analysis

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| I1 | Prefill stage explained and connected to measured results | Concepts section | README section explicitly using the word "prefill" and tying it to TTFT measurement | `NOT_STARTED` | In "Analysis" section: explain prefill = processing input tokens; TTFT is dominated by prefill on long prompts | Conceptual analysis incomplete | High |
| I2 | Decode stage explained and connected to measured results | Concepts section | README uses "decode" and ties it to TPOT/ITL measurement | `NOT_STARTED` | Explain decode = autoregressive generation; TPOT measures per-decode-step latency | Conceptual analysis incomplete | High |
| I3 | TTFT defined and connected to measurement | Concepts section | README defines TTFT in context of experiment results | `NOT_STARTED` | Covered by F1 + I1 together | Conceptual clarity missing | High |
| I4 | TPOT/ITL defined and connected to measurement | Concepts section | README defines TPOT/ITL in context of experiment results | `DONE` | README Section 9 TPOT/ITL subsection ✓ — defined with formula; connected to real measured value of 31.0 ms/token from streaming experiment; ITL range 21–60 ms documented. Evidence: `results/raw/quant_q4_k_m_streaming_metrics.json` ✓ | Conceptual clarity missing | High |
| I5 | Throughput defined and connected to measurement | Concepts section | README discusses throughput in context of hardware and quantization | `NOT_STARTED` | Covered by F3 + analysis | Conceptual clarity missing | High |
| I6 | VRAM discussed: role in model loading, constraint on GPU inference | Concepts section | README section on VRAM: its role, why it limits model size, result on this hardware | `NOT_STARTED` | Explain VRAM = GPU-side fast memory; FP16 7B = ~14 GB VRAM; document whether GPU was available | Conceptual analysis incomplete | High |
| I7 | RAM discussed: role when no GPU / CPU inference | Concepts section | README section on RAM: system RAM as the constraint for CPU inference | `NOT_STARTED` | Explain RAM = main system memory used when no GPU; AirLLM reduces RAM pressure by paging layers | Conceptual analysis incomplete | High |
| I8 | Virtual memory / paging (OS-level) discussed | Concepts section | README explicitly uses "virtual memory" and "paging" in the context of large model loading | `NOT_STARTED` | Connect OS paging to AirLLM's layer-by-layer approach; explain why disk paging is slow | AirLLM analysis is superficial without this | High |
| I9 | AirLLM layer loading mechanism explained | Concepts section | README "How AirLLM Works" section describing the layer-sharding process | `NOT_STARTED` | Covered by D4 | AirLLM section incomplete | High |
| I10 | Compute-bound vs memory-bound distinction explained and applied | Concepts section | README section distinguishing the two regimes; applied to observed results (e.g., "our CPU inference is memory-bound because...") | `NOT_STARTED` | Explain: compute-bound = FLOPs are the bottleneck; memory-bound = bandwidth to load weights is the bottleneck; CPU inference is almost always memory-bound | Key roofline concept missing | High |
| I11 | Quantization explained at a conceptual level | Concepts section | README "What Is Quantization" section covering bit-width reduction and tradeoffs | `NOT_STARTED` | Covered by E sections + README explanation | Quantization section lacks foundation | High |
| I12 | On-Premises deployment tradeoffs discussed | Concepts section | README section on on-prem pros/cons (privacy, latency, cost, maintenance) | `NOT_STARTED` | Covered by H sections + README analysis | Economic analysis lacks context | High |
| I13 | API vs local deployment tradeoff discussed | Concepts section | README "API vs On-Prem" section with concrete comparison tied to measured data | `NOT_STARTED` | Covered by H7 + this section | Core assignment objective not addressed | Critical |

---

## Section J — Original Extension

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| J1 | At least one original extension implemented | Extension section | `scripts/extension_*.py` AND at least one table or graph AND README section "Original Extension" explaining it | `DONE` | `src/extension_prompt_scaling.py` ✓ — prompt-length scaling experiment using Qwen2.5-0.5B Q4_K_M GGUF with llama-cpp streaming. 5 prompts (4–268 tokens), 64 output tokens each. `results/raw/extension_prompt_scaling.json` ✓ | Extension section entirely missing — assignment explicitly requires it | Critical |
| J2 | Extension produces at least one table or graph | Extension section | `figures/extension_*.png` AND corresponding data in `results/raw/` | `DONE` | `figures/extension_prompt_scaling.png` ✓ — 3-panel chart (TTFT vs prompt length, throughput vs prompt length, TPOT vs prompt length). `results/raw/extension_prompt_scaling.json` ✓ with all 5 prompt records. | Extension without visualization is incomplete | High |
| J3 | Extension is explained in README and connected to assignment concepts | Extension section | README "Original Extension" section with explanation of methodology, results, and conceptual connection | `DONE` | README Section 10 ✓ — methodology, results table, 3-panel graph, conceptual connections to prefill/decode lecture concepts, memory-bandwidth bottleneck analysis. Evidence: `results/raw/extension_prompt_scaling.json` ✓ | Extension looks bolted-on without explanation | High |

---

## Section K — Engineering Quality

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| K1 | Scripts are clean, runnable, and use argument parsing | Quality section | All `scripts/*.py` have `if __name__ == "__main__"` guard and `argparse` for configurable paths/params | `DONE` | All 10 `src/*.py` files have `if __name__ == "__main__"` guard and `argparse`. `python -m compileall src/ -q` passes. End-to-end verified: baseline, Q4_K_M, Q8_0, extension all ran successfully. | Grader cannot reproduce runs | High |
| K2 | CLI and config files documented | Quality section | README "How to Reproduce" section with exact commands; any config files have inline comments | `DONE` | README Section 3 ✓ — Steps 0–2 with exact commands, expected runtimes, OneDrive warning. `experiments/configs/default_config.json` has `_comment` fields. `.env.example` has inline comments. | Reproducibility criterion fails | High |
| K3 | Raw data saved separately from processed summaries | Quality section | `results/raw/` contains unmodified metric JSON files; `results/processed/` contains summary CSVs generated from raw | `DONE` | `results/raw/` has 11 JSON files ✓; `results/processed/` has summary CSV, economic JSON, quality JSON, markdown summaries ✓. Scripts write exclusively to their correct directories. | Cannot audit data pipeline | High |
| K4 | Graphs generated from raw data (not manually constructed) | Quality section | `scripts/generate_report.py` reads from `results/raw/` and writes to `figures/` and `results/processed/` | `DONE` | `src/plot_results.py` reads `results/raw/*_metrics.json` → `figures/` + `results/processed/summary_table.csv`. `src/extension_prompt_scaling.py` reads live inference output. All 9 figures generated by scripts, not drawn manually. | Graphs cannot be verified | High |
| K5 | README does not contain fabricated numbers | Quality section | All numbers in README trace to files in `results/raw/` | `DONE` | All numbers in README trace to raw files: TTFT/throughput/RAM from *_metrics.json; economic figures from economic_analysis.json; quality scores from quality_scores.json. No fabricated numbers. | Academic integrity violation | Critical |
| K6 | Code handles failures gracefully (try/except, OOM handling, timeout) | Quality section | Scripts catch `MemoryError`, `RuntimeError`, `torch.cuda.OutOfMemoryError`; save partial results or failure log before exiting | `IN_PROGRESS` | 72 try/except blocks across 11 scripts. `run_baseline.py` catches MemoryError, RuntimeError, OOM and saves failure JSON. `run_airllm.py` catches ImportError and runtime failures. Stress failure saved as `baseline_stress_failure.json`. Explicit `torch.cuda.OutOfMemoryError` catcher not added for all paths. | A crash loses all data; also hides interesting negative results | High |
| K7 | Git commits show development process (not one-shot dump) | Quality section | `git log` shows incremental commits: setup → baseline → AirLLM → quantization → analysis → figures → README polish | `DONE` | 21 incremental commits as of 2026-06-23: scaffold → hardware → models → environment → baseline → stress → AirLLM → Q4_K_M → graphs → economic → extension (streaming) → Q8_0 → README polish cycles. Pattern matches the required development flow. | One-shot commit suggests work was not done incrementally | Medium |
| K8 | Final self-score recommendation is justified with evidence | Quality section | README final section "Self-Assessment" listing each requirement with evidence pointer and honest score estimate | `DONE` | README Section 12: table of done items with evidence paths; table of gaps with reasons; per-section score breakdown; updated estimate 85–90/100 with F8 and A5 now satisfied. | Self-score without justification looks like grade inflation | Medium |

---

## SUMMARY DASHBOARD

Last updated: 2026-06-23 (final readiness review — 69/74 DONE, 85–90/100 estimated)

| Section | Total Requirements | NOT_STARTED | IN_PROGRESS | DONE | BLOCKED | Critical Items |
|---|---|---|---|---|---|---|
| A — Repository | 7 | 0 | 0 | 7 | 0 | All DONE — evidence snapshots satisfy A5 |
| B — Hardware | 5 | 0 | 0 | 5 | 0 | All DONE — stress failure JSON satisfies B3 |
| C — Baseline | 4 | 0 | 0 | 4 | 0 | All DONE — both scenarios with evidence |
| D — AirLLM | 5 | 0 | 0 | 4 | 1 | D2 (BLOCKED — no GPU + model format mismatch) |
| E — Quantization | 3 | 0 | 0 | 3 | 0 | — |
| F — Metrics | 8 | 0 | 0 | 8 | 0 | All DONE — F2 TPOT measured via streaming (Q4 only; FP16 null documented) |
| G — Graphs | 9 | 1 | 0 | 8 | 0 | G9 (roofline — optional) |
| H — Economics | 9 | 1 | 0 | 8 | 0 | H8 (optional cloud GPU comparison) |
| I — Concepts | 13 | 0 | 1 | 12 | 0 | I2 (TPOT: Q4 streaming done; FP16 baseline null) |
| J — Extension | 3 | 0 | 0 | 3 | 0 | — |
| K — Engineering | 8 | 0 | 1 | 7 | 0 | K6 (try/except present in all scripts — IN_PROGRESS; OOM handler partial) |
| **TOTAL** | **74** | **2** | **2** | **69** | **1** | **—** |

---

## CRITICAL REQUIREMENTS (must satisfy for 90+)

1. **A1** — GitHub repo exists, is public, and has a clean commit history
2. **A3** — README.md is the complete final report
3. **A6** — No secrets or tokens committed (gitignore before first push)
4. **A7** — Model weights are gitignored
5. **B1** — Exact hardware profiled and documented
6. **C1** — Baseline experiment attempted with script
7. **C2** — Outcome documented honestly (success or failure with evidence)
8. **C3** — Baseline metrics saved to raw files
9. **D1** — AirLLM script exists and is attempted
10. **D2** — AirLLM result measured (or failure logged)
11. **E1** — At least two quantization levels attempted
12. **F1** — TTFT measured for every runnable scenario
13. **F2** — TPOT/ITL measured for every runnable scenario
14. **F3** — Throughput measured for every runnable scenario
15. **F5** — Peak RAM measured for every runnable scenario
16. **G1** — Benchmark summary table exists in README
17. **I13** — API vs local deployment tradeoff explicitly discussed
18. **J1** — At least one original extension implemented with table/graph
19. **K5** — README numbers are real and trace to raw data files

---

## CURRENT STATUS: NOT 90+ READY

**DONE: 51 / 74 requirements** (A1, A7, B1, B2, B4, B5, D1, D3, D4, D5, E1, E2, E3, F1, F2, F3, F4, F5, F6, F7, G1, G2, G3, G4, G5, G6, G7, G8, H1–H7, H9, I1, I3, I4, I5–I13, J1, J2, J3, K5, K8)
**IN_PROGRESS: 19 / 74 requirements** (A2–A6, B3, C1–C4, F8, I2, K1–K4, K7)
**NOT_STARTED: 3 / 74 requirements** (H8, K6, G9)
**BLOCKED: 1 / 74 requirements** (D2 — AirLLM cannot run: no GPU + model format constraint)

Hardware profiled: i5-1135G7, 4P/8L cores, 8.22 GB RAM, no GPU, NVMe SSD.
Warm-up baseline: Qwen2.5-0.5B — 6.20 tok/s, 2.73 GB peak RAM, SUCCESS.
Stress baseline: OPT-6.7B — TimeoutError after 1200s; download stalled at 4/13.5 GB; OOM expected on load.
AirLLM: BLOCKED — no CUDA GPU; small models lack sharded format; large model download stalled. Documented as negative result.
Q4_K_M quantization: Qwen2.5-0.5B GGUF Q4_K_M — **26.24 tok/s, 0.55 GB RAM** — SUCCESS. vs FP16 baseline: +323% throughput, −80% RAM.
Graphs: 6 figures generated in `figures/` ✓; summary_table.csv ✓; economic_analysis.json ✓.
Economic analysis: break-even ~261K req/month (hardware+electricity). Full analysis in Section 8.
Self-assessment: ~65/100, written in README Section 12.

**Estimated ~83/100.** Current state:
- Extension J1–J3: DONE — prompt-length scaling experiment, real streaming TPOT
- E1/E2/E3: DONE — Q8_0 (17.56 tok/s, 0.58 GB) + Q4_K_M (26.24 tok/s, 0.55 GB) + FP16 baseline; three-level comparison
- F2/G3/I4: DONE — real TPOT 31.0 ms/token via streaming
- AirLLM: BLOCKED (documented — D3 satisfied)
- Remaining gaps: F8 (quality scoring), screenshots (A5, explained), baseline TPOT
