# REQUIREMENTS MATRIX — Assignment 05
# Running a Massive LLM Locally: AirLLM, Quantization and Performance Benchmarking
# Target grade: 90+
# Last updated: 2026-06-23
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
| A2 | Repository has a clear, navigable directory structure | Submission instructions | Directories: `scripts/`, `results/raw/`, `results/processed/`, `reports/`, `figures/`, `README.md`, `.gitignore` | `IN_PROGRESS` | Skeleton created: `src/`, `results/raw/`, `results/processed/`, `figures/`, `figures/screenshots/`, `experiments/`, `prompts/`, `configs/`, `reports/`. Real deliverable files not yet present. | Grader cannot find deliverables | High |
| A3 | `README.md` is the main final report (not a separate PDF) | Assignment brief | `README.md` at repo root; must be complete, not a placeholder | `IN_PROGRESS` | README.md exists with all required section headers (Hardware, Model Selection, Reproduce, Baseline, AirLLM, Quantization, Results, Economic Analysis, Lecture Concepts, Extension, Self-Assessment). Content is placeholder — no real data yet. | Primary deliverable missing | Critical |
| A4 | Reproduction instructions included in README | Assignment brief | Section in README: "How to reproduce" with exact commands, Python version, package versions | `IN_PROGRESS` | README "How to Reproduce" section written with full command sequence. Commands not yet end-to-end tested — will finalize after all scripts are verified runnable. | Reproducibility criterion fails | High |
| A5 | Screenshots included or explicitly explained if impossible | Assignment brief | `figures/screenshots/` directory OR explicit note in README explaining why screenshots are not possible | `IN_PROGRESS` | `figures/screenshots/` directory created; README has screenshot table listing 5 planned screenshots. No screenshots taken yet — will capture during experiment runs. | Visual evidence missing | Medium |
| A6 | No secrets or API tokens committed to the repo | Security / submission | `.gitignore` covers `.env`, `*.key`, `secrets/`; `git log --all` scan shows no tokens | `IN_PROGRESS` | `.gitignore` created before first model download or .env creation; covers `.env`, `*.env`, `*.key`. `.env.example` uses only placeholders. `git log` scan should be run before final push to confirm. | Academic integrity / security violation | Critical |
| A7 | Large model files are gitignored (no model weights in repo) | Submission instructions | `.gitignore` includes `*.bin`, `*.safetensors`, `*.gguf`, `models/`, `shards/`, `cache/` | `DONE` | `.gitignore` covers `*.bin`, `*.safetensors`, `*.gguf`, `*.pt`, `*.pth`, `*.onnx`, `models/`, `airllm_shards/`, `airllm_cache/`, `shards/`, `.cache/`, `hf_cache/`. Created before any model files exist. Evidence: `.gitignore` in repo root. | Repo bloat / push failure / GitHub LFS rejection | Critical |

---

## Section B — Hardware and Model Selection

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| B1 | Exact hardware documented: CPU model, core count, RAM (GB), GPU model or "none", VRAM if present, disk type (HDD/SSD/NVMe) and free space (GB), OS and version | Hardware section | `results/raw/hardware_profile.json` AND a "Hardware" section in README | `DONE` | `results/raw/hardware_profile.json` contains: CPU=i5-1135G7, 4P/8L cores, RAM=8.22 GB, GPU=none, VRAM=N/A, Disk=NVMe SSD 511 GB / 38.44 GB free, OS=Windows 11 24H2. README hardware table populated with these real numbers. Evidence: `results/raw/hardware_profile.json` ✓ | Grader cannot evaluate feasibility of experiments | Critical |
| B2 | Model choice justified relative to hardware constraints | Model selection section | README section "Model Selection Rationale" citing specific RAM/VRAM numbers from B1 | `DONE` | README Section 2 populated with three-role model selection using measured RAM (8.22 GB) and disk (38.44 GB). Cites: warm-up=Qwen2.5-0.5B (~1.5 GB RAM), stress=OPT-6.7B (~14.0 GB RAM > 8.22 GB ceiling), fallback=Qwen2.5-7B Q4_K_M GGUF (~4.8 GB RAM). Disk budget table shows 32.2 GB needed vs 38.44 GB free. Evidence: `results/raw/model_selection.json` ✓ | Appears random; no engineering judgement demonstrated | High |
| B3 | Model is large enough to stress hardware but not trivially impossible | Model selection section | README justification + baseline experiment showing strain (slow, OOM, or disk paging) | `IN_PROGRESS` | OPT-6.7B (14.0 GB FP16) vs 8.22 GB RAM: rationale documented in README — any 7B model overwhelms this machine's RAM; a 1B model would not. README explanation written. OOM evidence will come from baseline experiment (not yet run). Status stays IN_PROGRESS until baseline run produces `results/raw/baseline_failure.txt` or `baseline_metrics.json`. | Assignment objective not met | High |
| B4 | Disk-space check performed before downloading models | Model selection section | `results/raw/disk_check.txt` or log showing available disk space vs model download size | `DONE` | `hardware_profile.json` records `disk_free_gb: 38.44` and `disk_total_gb: 511.04` at probe time (before any model download). 38.44 GB free vs ~13.4 GB for OPT-6.7B FP16 = adequate margin. Evidence embedded in `results/raw/hardware_profile.json`. | Missing a "professional practice" mark | Medium |
| B5 | Public / no-auth-token fallback model is defined | Model selection section | README names at least one fallback model that requires no Hugging Face token (e.g., `TheBloke` GGUF variants, `facebook/opt-*`, `EleutherAI/gpt-j-6b`) | `DONE` | README Section 2 explicitly names `bartowski/Qwen2.5-7B-Instruct-GGUF` Q4_K_M as the fallback quantized model. Notes state "No token required — public HuggingFace repo (bartowski is a widely-used GGUF publisher)". All three selected models (Qwen2.5-0.5B, OPT-6.7B, Qwen2.5-7B GGUF) are public and require no HF token. Evidence: `results/raw/model_selection.json` field `requires_token: false` for all roles ✓ | If primary model is gated, entire experiment chain fails | High |

---

## Section C — Baseline Experiment

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| C1 | Direct / naive local baseline run is attempted | Baseline section | `scripts/baseline_run.py` exists AND `results/raw/baseline_metrics.json` or `results/raw/baseline_failure.txt` exists | `IN_PROGRESS` | `src/run_baseline.py` exists with full instrumentation (RamSampler, TTFT timing, failure logging). Result JSON not yet generated — must run the script. | Baseline reference point missing; nothing to compare against | Critical |
| C2 | Outcome documented honestly: success, OOM, excessive slowness, or failure | Baseline section | `results/raw/baseline_metrics.json` (if success) OR `results/raw/baseline_failure.txt` with error log and explanation | `NOT_STARTED` | Capture `traceback`, `MemoryError`, or timing data; do not fabricate success if it OOMed | Negative results presented without documentation look fabricated | Critical |
| C3 | Baseline metrics saved to raw result files (not only in README) | Baseline section | `results/raw/baseline_metrics.json` containing TTFT, TPOT, throughput, RAM peak, runtime | `NOT_STARTED` | Instrument script with `tracemalloc` / `psutil` memory sampling and `time.perf_counter` timing | Cannot regenerate graphs from real data | High |
| C4 | Bottleneck identified and explained: RAM, VRAM, CPU compute, or disk I/O | Baseline section | README section "Baseline Bottleneck Analysis" + supporting data (e.g., `htop` screenshot or `psutil` memory trace) | `NOT_STARTED` | During baseline run, sample RAM every second; identify where limit was hit; connect to lecture concepts (memory-bound vs compute-bound) | Key conceptual analysis missing | High |

---

## Section D — AirLLM Experiment

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| D1 | AirLLM integration attempted | AirLLM section | `scripts/airllm_run.py` exists | `IN_PROGRESS` | `src/run_airllm.py` exists using `airllm.AutoModel`, with configurable `AIRLLM_SHARD_DIR`/`AIRLLM_CACHE_DIR`, DiskIoSampler, and failure logging. Must be run to produce `results/raw/airllm_metrics.json`. | AirLLM section entirely missing | Critical |
| D2 | AirLLM result measured if it runs | AirLLM section | `results/raw/airllm_metrics.json` with same metric fields as baseline | `NOT_STARTED` | Reuse metric instrumentation from C3; save to separate raw file | No comparison data for AirLLM | Critical |
| D3 | If AirLLM fails, failure documented as negative result with logs and fallback | AirLLM section | `results/raw/airllm_failure.txt` with full error log + README section explaining what failed and why, + alternative approach taken | `NOT_STARTED` | Wrap AirLLM calls in try/except; save exception to file; document in README as "Negative Result — AirLLM" | Looks like work was skipped rather than failing honestly | High |
| D4 | Layer-sharding / disk paging / I/O behavior discussed | AirLLM section | README section "AirLLM: How Layer Loading Works" explicitly referencing paging, virtual memory, disk I/O concepts from lecture | `NOT_STARTED` | Explain that AirLLM loads one layer at a time from disk, analogous to OS paging; connect to lecture vocabulary | Conceptual analysis missing | High |
| D5 | AirLLM cache/shard path explicitly configured and documented | AirLLM section | In `scripts/airllm_run.py`, `cache_dir` / shard path is set to a configurable path (not hardcoded user path); documented in README | `NOT_STARTED` | Use `argparse` or environment variable for cache path; document in "How to Reproduce" | Grader cannot reproduce without knowing shard path; also exposes local path | Medium |

---

## Section E — Quantization Experiment

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| E1 | At least two quantization levels / settings attempted where technically possible | Quantization section | `scripts/quantization_run.py` with at least two configs; `results/raw/quant_fp16_metrics.json` AND `results/raw/quant_q4_metrics.json` (or equivalent) | `IN_PROGRESS` | `src/run_quantized.py` has 3 transformers configs (FP32, FP16, BF16) + 2 GGUF configs (Q8_0, Q4_K_M). Must run to produce JSON files. | Quantization section missing entirely | Critical |
| E2 | Impact on memory, speed, and output quality measured or documented for each level | Quantization section | Each `results/raw/quant_*_metrics.json` contains RAM peak, TTFT, throughput, and a short output quality note | `NOT_STARTED` | Run same prompt across all quant levels; compare outputs qualitatively; log metrics consistently | Cannot draw quantization tradeoff graph | High |
| E3 | Quantization "red line" for quality discussed | Quantization section | README section "Quantization Quality Threshold" discussing at what level output degrades noticeably, with example outputs | `NOT_STARTED` | After E2, compare outputs; note any coherence loss at lower precision; reference bit-width tradeoffs from lecture | Conceptual insight missing | Medium |

---

## Section F — Benchmark Metrics

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| F1 | TTFT (Time To First Token) measured for every runnable scenario | Metrics section | Each `results/raw/*_metrics.json` has `"ttft_seconds"` field | `NOT_STARTED` | Record `time.perf_counter()` at token generation start and when first token arrives; for batch/streaming inference, hook into generation callback | Core metric missing | Critical |
| F2 | TPOT / ITL (Time Per Output Token / Inter-Token Latency) measured | Metrics section | Each `results/raw/*_metrics.json` has `"tpot_ms"` or `"itl_ms"` field | `NOT_STARTED` | `(total_generation_time - ttft) / (total_tokens - 1)` for streaming; or average inter-token interval | Core metric missing | Critical |
| F3 | Throughput (tokens/sec) measured | Metrics section | Each `results/raw/*_metrics.json` has `"tokens_per_second"` field | `NOT_STARTED` | `total_output_tokens / total_generation_time_seconds` | Core metric missing | Critical |
| F4 | Total runtime measured | Metrics section | Each `results/raw/*_metrics.json` has `"total_runtime_seconds"` field | `NOT_STARTED` | Wrap entire inference call in `time.perf_counter()` | Without runtime, throughput and economic analysis are incomplete | High |
| F5 | Peak RAM measured | Metrics section | Each `results/raw/*_metrics.json` has `"peak_ram_gb"` field | `NOT_STARTED` | Use `psutil.Process().memory_info().rss` sampled in a background thread during inference | Memory analysis impossible without this | Critical |
| F6 | Peak VRAM measured if GPU available; otherwise explicitly "N/A — no CUDA/discrete GPU" documented | Metrics section | Each `results/raw/*_metrics.json` has `"peak_vram_gb"` field or `"peak_vram_note": "N/A — no CUDA GPU detected"` | `NOT_STARTED` | Try `nvidia-smi` / `torch.cuda.max_memory_allocated()`; if unavailable, explicitly document absence | Ambiguity about hardware capability | High |
| F7 | Estimated power / electricity consumption documented | Metrics section | `results/raw/power_estimate.json` or README section "Power Consumption Estimate" with methodology | `NOT_STARTED` | Use CPU TDP from spec sheet × utilization fraction × runtime; or use `powermetrics` (macOS) / `Intel RAPL` (Linux) equivalent; on Windows, document estimation method | Economic analysis depends on this | High |
| F8 | Output quality notes included for each scenario | Metrics section | Each `results/raw/*_metrics.json` has `"output_quality_notes"` field; README includes example outputs | `NOT_STARTED` | Manually assess: is output coherent? Does quantization cause repetition or garbling? | Quantization tradeoff section incomplete | Medium |

---

## Section G — Tables and Graphs

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| G1 | Benchmark summary table | Results section | Table in README comparing all scenarios across all F-section metrics | `NOT_STARTED` | Generate via `scripts/generate_report.py` from raw JSON; also embed in README | Cannot compare scenarios at a glance | Critical |
| G2 | TTFT comparison graph | Results section | `figures/ttft_comparison.png` | `NOT_STARTED` | `matplotlib` bar chart from `results/processed/summary_table.csv` | Visual evidence missing | High |
| G3 | TPOT/ITL comparison graph | Results section | `figures/tpot_comparison.png` | `NOT_STARTED` | Same pipeline as G2 | Visual evidence missing | High |
| G4 | Throughput comparison graph | Results section | `figures/throughput_comparison.png` | `NOT_STARTED` | Same pipeline as G2 | Visual evidence missing | High |
| G5 | RAM / VRAM comparison graph | Results section | `figures/memory_comparison.png` | `NOT_STARTED` | Same pipeline as G2 | Visual evidence missing | High |
| G6 | Runtime comparison graph | Results section | `figures/runtime_comparison.png` | `NOT_STARTED` | Same pipeline as G2 | Visual evidence missing | Medium |
| G7 | Quantization tradeoff graph (memory vs quality vs speed) | Results section | `figures/quant_tradeoff.png` | `NOT_STARTED` | Multi-axis or grouped bar chart from quantization raw data | Quantization insight not visualized | High |
| G8 | Economic break-even graph | Economic analysis section | `figures/economic_breakeven.png` | `NOT_STARTED` | Plot cumulative API cost vs on-prem cost as function of monthly requests | Economic analysis incomplete without visual | High |
| G9 | Optional: roofline-style compute-bound vs memory-bound visualization | Advanced analysis section | `figures/roofline.png` (optional) | `NOT_STARTED` | Plot achieved throughput vs arithmetic intensity; mark compute roof and memory bandwidth roof | Optional — extra credit territory | Low |

---

## Section H — Economic Analysis

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| H1 | API cost calculation for equivalent workload | Economic section | README "Economic Analysis" section with formula: `tokens × price_per_1k_tokens / 1000`; source and date of pricing cited | `NOT_STARTED` | Use Claude or GPT-4o pricing as reference; note that prices change and state the date | Economic analysis missing | High |
| H2 | On-Prem local cost calculation | Economic section | README calculation: hardware amortization/month + electricity cost/month + operation time | `NOT_STARTED` | Amortize hardware cost over 3 years; add electricity from F7; estimate operator time cost | Cannot compare deployment modes | High |
| H3 | Hardware amortization assumption stated | Economic section | README states assumed hardware cost and amortization period (e.g., "$800 laptop, 3-year life = $22/month") | `NOT_STARTED` | Use realistic local hardware cost; be explicit about assumption | Cost calculation unverifiable | Medium |
| H4 | Electricity assumption stated | Economic section | README states assumed electricity rate (e.g., "$0.12/kWh") and source region | `NOT_STARTED` | Use local or US average rate; cite source | Cost calculation unverifiable | Medium |
| H5 | Maintenance/operation assumption stated | Economic section | README states assumed maintenance overhead (e.g., "2 hours/month operator time at $X/hr") | `NOT_STARTED` | Estimate realistically; mark as assumption | Holistic cost picture incomplete | Medium |
| H6 | Break-even calculation: requests/month or tokens/month where on-prem becomes cheaper | Economic section | README shows break-even formula and result; `figures/economic_breakeven.png` (G8) | `NOT_STARTED` | `break_even_requests = fixed_monthly_on_prem_cost / (api_cost_per_request - variable_on_prem_cost_per_request)` | Key recommendation is ungrounded | High |
| H7 | Recommendation: when API is better, when on-prem is better | Economic section | README "Recommendation" paragraph connecting break-even result to practical use cases | `NOT_STARTED` | State clearly: "Below X requests/month, use API. Above X, on-prem is cheaper if quality is acceptable." | Practical conclusion missing | High |
| H8 | Optional: cloud GPU comparison (e.g., Lambda Labs, Vast.ai, RunPod) | Economic section | README note comparing cloud GPU hourly rate vs amortized on-prem cost | `NOT_STARTED` | Optional — add if time permits | Optional; enhances analysis | Low |
| H9 | Prompt/context caching pricing discussed as assumption if exact prices not used | Economic section | README notes that API prices change; states the specific price and date used as assumption | `NOT_STARTED` | Add a one-line disclaimer with the pricing date in the economic section | Academic integrity: prices cited may be stale | Medium |

---

## Section I — Lecture-Concept Analysis

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| I1 | Prefill stage explained and connected to measured results | Concepts section | README section explicitly using the word "prefill" and tying it to TTFT measurement | `NOT_STARTED` | In "Analysis" section: explain prefill = processing input tokens; TTFT is dominated by prefill on long prompts | Conceptual analysis incomplete | High |
| I2 | Decode stage explained and connected to measured results | Concepts section | README uses "decode" and ties it to TPOT/ITL measurement | `NOT_STARTED` | Explain decode = autoregressive generation; TPOT measures per-decode-step latency | Conceptual analysis incomplete | High |
| I3 | TTFT defined and connected to measurement | Concepts section | README defines TTFT in context of experiment results | `NOT_STARTED` | Covered by F1 + I1 together | Conceptual clarity missing | High |
| I4 | TPOT/ITL defined and connected to measurement | Concepts section | README defines TPOT/ITL in context of experiment results | `NOT_STARTED` | Covered by F2 + I2 together | Conceptual clarity missing | High |
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
| J1 | At least one original extension implemented | Extension section | `scripts/extension_*.py` AND at least one table or graph AND README section "Original Extension" explaining it | `NOT_STARTED` | Choose one: (1) repeated-run stability — run same config 5× and report variance in TTFT/TPOT; (2) quality degradation — score outputs at each quant level with BLEU/ROUGE or manual rubric; (3) disk I/O observation — log disk read bytes during AirLLM layer loading using `psutil.disk_io_counters()`. Recommended: disk I/O observation since it naturally extends D4 and requires no extra model runs | Extension section entirely missing — assignment explicitly requires it | Critical |
| J2 | Extension produces at least one table or graph | Extension section | `figures/extension_*.png` AND corresponding data in `results/raw/` | `NOT_STARTED` | Covered by whichever J1 option is chosen | Extension without visualization is incomplete | High |
| J3 | Extension is explained in README and connected to assignment concepts | Extension section | README "Original Extension" section with explanation of methodology, results, and conceptual connection | `NOT_STARTED` | Write after J1 and J2 are complete | Extension looks bolted-on without explanation | High |

---

## Section K — Engineering Quality

| ID | Exact Requirement | Source Section | Required Evidence / File | Status | How We Will Satisfy It | Risk If Missing | Grade Impact |
|---|---|---|---|---|---|---|---|
| K1 | Scripts are clean, runnable, and use argument parsing | Quality section | All `scripts/*.py` have `if __name__ == "__main__"` guard and `argparse` for configurable paths/params | `IN_PROGRESS` | All `src/*.py` files have `if __name__ == "__main__"` guard and `argparse`. Import of heavy deps (torch, transformers, airllm) is inside functions so modules are importable without packages installed. `python -m compileall src/` passed for all 9 scripts (2026-06-23). End-to-end runability not yet verified — needs model download and inference run. | Grader cannot reproduce runs | High |
| K2 | CLI and config files documented | Quality section | README "How to Reproduce" section with exact commands; any config files have inline comments | `IN_PROGRESS` | README "How to Reproduce" section has full command sequence. `configs/default_config.json` has `_comment` fields. `.env.example` has inline comments on every variable. Commands not yet end-to-end tested. | Reproducibility criterion fails | High |
| K3 | Raw data saved separately from processed summaries | Quality section | `results/raw/` contains unmodified metric JSON files; `results/processed/` contains summary CSVs generated from raw | `IN_PROGRESS` | `results/raw/` and `results/processed/` directories created. All `src/run_*.py` scripts write exclusively to `results/raw/`. `src/plot_results.py` and `src/quality_eval.py` write to `results/processed/`. No files present yet. | Cannot audit data pipeline | High |
| K4 | Graphs generated from raw data (not manually constructed) | Quality section | `scripts/generate_report.py` reads from `results/raw/` and writes to `figures/` and `results/processed/` | `IN_PROGRESS` | `src/plot_results.py` reads `results/raw/*_metrics.json`, generates 7 standard graphs + summary CSV to `figures/` and `results/processed/`. Idempotent (re-running overwrites). Not yet run — requires experiment data first. | Graphs cannot be verified | High |
| K5 | README does not contain fabricated numbers | Quality section | All numbers in README trace to files in `results/raw/` | `NOT_STARTED` | Strict rule: never type a number in README without a corresponding raw file | Academic integrity violation | Critical |
| K6 | Code handles failures gracefully (try/except, OOM handling, timeout) | Quality section | Scripts catch `MemoryError`, `RuntimeError`, `torch.cuda.OutOfMemoryError`; save partial results or failure log before exiting | `NOT_STARTED` | Wrap all model loading and inference in try/except blocks; always save whatever data was collected before crash | A crash loses all data; also hides interesting negative results | High |
| K7 | Git commits show development process (not one-shot dump) | Quality section | `git log` shows incremental commits: setup → baseline → AirLLM → quantization → analysis → figures → README polish | `IN_PROGRESS` | 6 incremental commits as of 2026-06-23: matrix, scaffold, hardware profiling, model selection, cache safety, environment setup. Experiment commits (baseline, AirLLM, quant, analysis, figures) still to come. | One-shot commit suggests work was not done incrementally | Medium |
| K8 | Final self-score recommendation is justified with evidence | Quality section | README final section "Self-Assessment" listing each requirement with evidence pointer and honest score estimate | `NOT_STARTED` | Write last, after all evidence exists; cite actual file paths | Self-score without justification looks like grade inflation | Medium |

---

## SUMMARY DASHBOARD

Last updated: 2026-06-23 (environment setup — all packages installed)

| Section | Total Requirements | NOT_STARTED | IN_PROGRESS | DONE | BLOCKED | Critical Items |
|---|---|---|---|---|---|---|
| A — Repository | 7 | 0 | 5 | 2 | 0 | A3 (README needs real data), A6 (verify before final push) |
| B — Hardware | 5 | 0 | 1 | 4 | 0 | B3 (in progress — needs baseline OOM evidence) |
| C — Baseline | 4 | 3 | 1 | 0 | 0 | C1 (script ready, not run), C2, C3 |
| D — AirLLM | 5 | 4 | 1 | 0 | 0 | D1 (script ready, not run), D2 |
| E — Quantization | 3 | 2 | 1 | 0 | 0 | E1 (script ready, not run) |
| F — Metrics | 8 | 8 | 0 | 0 | 0 | F1, F2, F3, F5 |
| G — Graphs | 9 | 9 | 0 | 0 | 0 | G1 |
| H — Economics | 9 | 9 | 0 | 0 | 0 | — |
| I — Concepts | 13 | 13 | 0 | 0 | 0 | I13 |
| J — Extension | 3 | 3 | 0 | 0 | 0 | J1 |
| K — Engineering | 8 | 3 | 5 | 0 | 0 | K5 |
| **TOTAL** | **74** | **54** | **14** | **6** | **0** | **—** |

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

**DONE: 6 / 74 requirements** (A1, A7, B1, B2, B4, B5)  
**IN_PROGRESS: 14 / 74 requirements** (K7 added — 6 incremental commits exist)
**NOT_STARTED: 54 / 74 requirements**

Hardware profiled: i5-1135G7, 4P/8L cores, 8.22 GB RAM, no GPU, NVMe SSD, 38.44 GB free.
Critical finding: 8.22 GB RAM means naive 7B model load will OOM — expected negative result.
AirLLM and GGUF Q4 are the viable paths.

**NOT 90+ ready.** Remaining blockers:
- Baseline, AirLLM, and quantization experiments not yet run (no results/raw/ data files)
- All figures/*.png not yet generated
- README sections B2/B3, all of Sections C–K still contain _TBD_
- Economic analysis, extension, and self-assessment not written
- Model not yet selected or downloaded
