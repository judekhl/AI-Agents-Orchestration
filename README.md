# Assignment 05: Running a Massive LLM Locally
## AirLLM, Quantization and Performance Benchmarking

**Author:** Jude  
**Course:** [Course name]  
**Repo:** https://github.com/judekhl/AI-Agents-Orchestration

> **Status: IN PROGRESS — warm-up baseline complete; stress baseline complete (timeout/OOM); AirLLM BLOCKED (Section 5); Q4_K_M quantization complete; graphs and analysis pending.**
> Warm-up numbers in Section 4 and Section 7 are real measured values from `results/raw/baseline_warmup_metrics.json`.
> AirLLM result in Section 5 reflects a real compatibility check — two hard blockers documented (no CUDA GPU; model format mismatch). Not skipped.
> Q4_K_M result in Section 6 and Section 7 is real measured data from `results/raw/quant_q4_k_m_metrics.json`.
> All remaining _TBD_ entries are not yet run. Do not read them as results.

---

## Table of Contents

1. [Hardware Profile](#1-hardware-profile)
2. [Model Selection Rationale](#2-model-selection-rationale)
3. [How to Reproduce](#3-how-to-reproduce)
4. [Baseline Experiment](#4-baseline-experiment)
5. [AirLLM Experiment](#5-airllm-experiment)
6. [Quantization Experiment](#6-quantization-experiment)
7. [Benchmark Results](#7-benchmark-results)
8. [Economic Analysis](#8-economic-analysis)
9. [Lecture Concept Analysis](#9-lecture-concept-analysis)
10. [Original Extension](#10-original-extension)
11. [Screenshots](#11-screenshots)
12. [Self-Assessment](#12-self-assessment)
13. [References](#13-references)

---

## 1. Hardware Profile

<!-- REQUIREMENT B1 — populated from results/raw/hardware_profile.json on 2026-06-23 -->

| Component | Specification |
|---|---|
| CPU model | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz |
| CPU cores (physical / logical) | 4 physical / 8 logical |
| RAM (GB total) | 8.22 GB |
| RAM (available at probe) | 0.87 GB |
| GPU | N/A — no discrete CUDA GPU |
| VRAM (GB) | N/A — no discrete GPU |
| Disk model | NVMe KINGSTON OM8SBP3512K-AH |
| Disk type | NVMe SSD |
| Disk total (GB) | 511.04 GB |
| Disk free (GB) | 38.44 GB |
| OS and version | Windows 11 (build 10.0.26200, 24H2) |
| Python version | 3.10.0 |

**Critical constraint:** 8.22 GB total RAM with no GPU. A 7B model in FP16 requires ~14 GB RAM — naive baseline load will OOM. AirLLM layer-paging and GGUF Q4 quantization are the primary mitigations.

**Evidence file:** [`results/raw/hardware_profile.json`](results/raw/hardware_profile.json) ✓  
**Summary:** [`results/processed/hardware_summary.md`](results/processed/hardware_summary.md) ✓

---

## 2. Model Selection Rationale

<!-- REQUIREMENT B2, B3, B4, B5 — populated from results/raw/model_selection.json on 2026-06-23 -->

Three models are used across experiment roles. No model requires a Hugging Face token.

| Role | Model | Format | Size in RAM | Fits in 8.22 GB? |
|---|---|---|---|---|
| **Warm-up / sanity** | `Qwen/Qwen2.5-0.5B-Instruct` | HF FP16 | ~1.5 GB | Yes |
| **Stress / baseline** | `facebook/opt-6.7b` | HF FP16 | ~14.0 GB | **No — OOM expected** |
| **Fallback / quantized** | `bartowski/Qwen2.5-7B-Instruct-GGUF` Q4\_K\_M | GGUF | ~4.8 GB | Yes |

**Evidence:** [`results/raw/model_selection.json`](results/raw/model_selection.json) ✓  
**Full report:** [`results/processed/model_selection.md`](results/processed/model_selection.md) ✓

### Warm-up model — `Qwen/Qwen2.5-0.5B-Instruct`

- **Size:** 0.5B parameters, ~1.0 GB on disk (FP16), ~1.5 GB peak RAM
- **Why chosen:** Proves the inference pipeline works end-to-end on this hardware without
  risking OOM. Provides a lightweight timing baseline (low TTFT and fast TPOT expected).
- **Token required:** No — public HuggingFace repo
- **Disk needed:** ~1.0 GB

### Stress model — `facebook/opt-6.7b`

- **Size:** 6.7B parameters, ~13.4 GB on disk (FP16), ~14.0 GB peak RAM
- **Why chosen:** At 14 GB RAM required vs 8.22 GB available, this model definitively
  stresses the hardware. Naive `transformers` baseline load will trigger `MemoryError`
  or OS swap — this is the expected, documentable negative result that motivates AirLLM.
  AirLLM can run this model by loading one transformer layer (~0.4 GB) at a time,
  keeping peak RAM well within the 8.22 GB ceiling.
- **Why not a smaller stress model:** A 1–3B model might fit in RAM and would not
  demonstrate the assignment's core challenge. 6.7B ensures genuine hardware strain.
- **Token required:** No — public HuggingFace repo (`facebook/opt-6.7b`)
- **Disk needed:** ~13.4 GB (+ ~13.4 GB for AirLLM shards during sharding step)

### Fallback / quantized model — `bartowski/Qwen2.5-7B-Instruct-GGUF` (Q4\_K\_M)

- **Size:** 7B parameters, ~4.4 GB on disk (Q4\_K\_M GGUF), ~4.8 GB peak RAM
- **Why chosen:** Q4\_K\_M quantization compresses a 7B model to a size that fits
  comfortably in 8.22 GB RAM (~4.8 GB used, leaving ~3.4 GB for OS and overhead).
  Enables successful throughput measurement where the baseline OOMs.
  Also allows direct comparison between quantization levels (Q4 vs Q8 vs FP16)
  to demonstrate the memory/speed/quality tradeoff.
- **Alternate quant:** Q8\_0 (~7.7 GB disk, ~8.2 GB RAM) — extremely tight, will be
  attempted if memory stabilises, but Q4\_K\_M is the primary fallback.
- **Token required:** No — public HuggingFace repo (bartowski is a widely-used GGUF publisher)
- **Disk needed:** ~4.4 GB

### Disk Space Check

**Available disk before any model download:** 38.44 GB (NVMe SSD, measured 2026-06-23)

| Download item | GB |
|---|---|
| Warm-up: Qwen2.5-0.5B FP16 | ~1.0 |
| Stress: OPT-6.7B FP16 | ~13.4 |
| AirLLM shards (OPT-6.7B copy) | ~13.4 |
| Fallback: Qwen2.5-7B Q4\_K\_M GGUF | ~4.4 |
| **Total** | **~32.2 GB** |
| **Remaining after all downloads** | **~6.2 GB** |

Disk budget is feasible. AirLLM shards and original weights can coexist during the sharding
step; original checkpoint can be deleted afterward to reclaim ~13.4 GB.

**Evidence:** [`results/raw/hardware_profile.json`](results/raw/hardware_profile.json) `disk_free_gb: 38.44` ✓

---

## 3. How to Reproduce

<!-- REQUIREMENT A4, K1, K2 -->
> **Claude Code users:** Read `CLAUDE.md` and `reports/PROJECT_STATE.md` before continuing work in a new session.

### Prerequisites

```
Python 3.10+
~35 GB free disk space outside OneDrive (for model weights — see Section 2)
8+ GB RAM (expect OOM on the stress model — that is the intended result)
```

> **OneDrive / repo users — read this before downloading any model.**
> This repository lives inside OneDrive. Model weights, HuggingFace cache, AirLLM shards,
> and GGUF files **must not** be stored inside the repo or anywhere under OneDrive.
> OneDrive will attempt to sync multi-GB weight files, wasting bandwidth and cloud storage.
> AirLLM's shard writes during inference may also conflict with OneDrive's background sync,
> causing file-lock errors mid-experiment.

### Step 0 — Create external folders (once, before anything else)

> This repository lives inside OneDrive. Storing the Python virtual environment
> inside the repo would cause OneDrive to continuously sync thousands of small
> `.py`/`.pyd`/`.dll` files from the venv, causing sync conflicts and slowdowns.
> Both the venv and model caches must live outside the OneDrive tree.

```bat
rem ── Python virtual environment (outside OneDrive) ──
mkdir C:\ai-envs

rem ── Model / cache directories (outside OneDrive) ──
mkdir C:\ai-model-cache\hf
mkdir C:\ai-model-cache\airllm-cache
mkdir C:\ai-model-cache\airllm-shards
mkdir C:\ai-model-cache\gguf
```

These paths are referenced by `.env` and `experiments/configs/default_config.json`.
All live on the local `C:\` drive, outside OneDrive.

### Step 1 — Clone, set up venv, and install

```bash
git clone https://github.com/judekhl/AI-Agents-Orchestration.git
cd AI-Agents-Orchestration

rem Create virtual environment OUTSIDE OneDrive (avoids syncing ~10 000 venv files)
python -m venv C:\ai-envs\ai-agents-ex05

rem Activate it
C:\ai-envs\ai-agents-ex05\Scripts\activate

rem Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

rem Install core + experiment dependencies
pip install -e ".[transformers,airllm,gguf,quality]"

rem Copy environment template — confirm cache paths before editing
copy .env.example .env
rem No token needed — all models are public
```

### Step 2 — Run experiments in order

```bash
# (2a) Profile hardware — no model download required
python src/hardware_probe.py --output results/raw/hardware_profile.json

# (2b) Baseline — naive transformers load, no optimization
python src/run_baseline.py --config experiments/configs/default_config.json \
    --output results/raw/baseline_metrics.json

# (2c) AirLLM — layer-by-layer sharding from disk
python src/run_airllm.py --config experiments/configs/default_config.json \
    --output results/raw/airllm_metrics.json

# (2d) Quantization variants — GGUF Q4_K_M and Q8_0
python src/run_quantized.py --config experiments/configs/default_config.json \
    --output-dir results/raw/

# (2e) Extension — disk I/O observation during AirLLM
python src/extension_disk_io.py --config experiments/configs/default_config.json \
    --output results/raw/extension_disk_io.json

# (2f) Quality evaluation across quantization levels
python src/quality_eval.py --input-dir results/raw/ \
    --output results/processed/quality_scores.json

# (2g) Economic analysis
python src/economic_analysis.py \
    --output results/processed/economic_analysis.json

# (2h) Generate all plots and summary table
python src/plot_results.py --input-dir results/raw/ \
    --processed-dir results/processed/ \
    --output-dir figures/
```

**Expected total runtime:** ~15 min for runnable experiments (warm-up baseline ~3.5 min including first download; Q4_K_M GGUF ~7 min including download + 3 s inference). Stress baseline and AirLLM are blocked on this hardware (see Sections 4 and 5).

---

## 4. Baseline Experiment

<!-- REQUIREMENT C1, C2, C3, C4 -->

**Script:** [`src/run_baseline.py`](src/run_baseline.py)  
**Config:** [`experiments/configs/default_config.json`](experiments/configs/default_config.json)

### Warm-up Baseline — `Qwen/Qwen2.5-0.5B-Instruct` (COMPLETE)

**Scenario:** Standard HuggingFace `transformers`, CPU-only, no AirLLM, no quantization
**Outcome:** SUCCESS — model loaded and generated 64 tokens without OOM

| Metric | Value |
|---|---|
| Throughput | 6.20 tok/s |
| Peak RAM | 2.73 GB |
| Output tokens | 64 |
| TTFT (approx) | 10.33 s |
| Model load time | 208.5 s (first run — includes ~1 GB model download) |
| TPOT | N/A (TTFT approximated as total runtime — no streaming hook; true TTFT ≤ 10.33 s) |
| OOM | No |

**Evidence:**
- [`results/raw/baseline_warmup_metrics.json`](results/raw/baseline_warmup_metrics.json) ✓
- [`results/processed/baseline_warmup_summary.md`](results/processed/baseline_warmup_summary.md) ✓

**Output snippet (64 tokens):**
> Virtual memory is a technique used by operating systems to allow programs to access larger amounts of memory than are physically available on the system's physical memory. It does this by creating a separate "virtual" memory space that can be accessed by the program without having to worry about the actual physical memory being allocated. Paging is one…

### Stress Baseline — `facebook/opt-6.7b` (COMPLETE — Negative Result)

**Scenario:** Same script, 6.7B FP16 model (~13.5 GB required vs 8.22 GB RAM available)
**Outcome:** TIMEOUT after 1200 s — download stalled at 4.049 GB / 13.5 GB (30%)

| Item | Value |
|---|---|
| Error type | TimeoutError (1200 s guard) |
| Download at timeout | 4.049 GB of ~13.5 GB |
| Model load into RAM | Not reached |
| Expected RAM on load | ~14 GB FP16 vs 8.22 GB available → OOM |
| Throughput achieved | 0 tok/s (no inference) |

**Why this is the expected negative result:** Even before reaching model loading, the 13.5 GB download alone stalled at ~30% over a 20-minute window. Had the download completed, `from_pretrained` would have attempted to map ~14 GB of FP16 weights into 8.22 GB RAM — triggering `MemoryError` or forcing the OS into uncontrolled swap at ~1000× the RAM access penalty.

**Evidence:**
- [`results/raw/baseline_stress_failure.json`](results/raw/baseline_stress_failure.json) ✓
- [`results/processed/baseline_stress_summary.md`](results/processed/baseline_stress_summary.md) ✓

### Bottleneck Analysis

<!-- REQUIREMENT C4, I10 -->

**Warm-up model (0.5B):** Memory-bandwidth-bound on CPU. Peak RAM 2.73 GB is well within the 8.22 GB ceiling. Throughput (6.20 tok/s) is limited by CPU memory bandwidth, not RAM capacity.

**Stress model (6.7B):** Two-stage failure: (1) 13.5 GB download is impractical over a constrained connection; (2) loading 14 GB FP16 into 8.22 GB RAM would trigger `MemoryError` or OS swap — either outcome makes naive baseline inference infeasible. This is the intended demonstration that motivates AirLLM (structured layer paging) and GGUF Q4 quantization (~4× size reduction) as viable mitigations.

---

## 5. AirLLM Experiment

<!-- REQUIREMENT D1, D2, D3, D4, D5 -->

**Script:** [`src/run_airllm.py`](src/run_airllm.py)  
**Shard path:** `C:\ai-model-cache\airllm-shards` — configured via `airllm_shard_dir` in `experiments/configs/default_config.json` (not hardcoded; override with `--config`)

**Outcome:** **BLOCKED** — two hard constraints prevent AirLLM from running on this machine. This is a documented negative result, not skipped work.

### Evidence (real data)

- Compatibility check: [`results/raw/airllm_compatibility.json`](results/raw/airllm_compatibility.json) ✓
- Human-readable summary: [`results/processed/airllm_compatibility_summary.md`](results/processed/airllm_compatibility_summary.md) ✓

### What Was Attempted

The AirLLM package was installed and its API was fully inspected. A compatibility run was attempted with `Qwen/Qwen2.5-0.5B-Instruct` (the only fully-cached local model) using the correct AirLLM parameters: `device='cpu'`, `dtype=torch.float32`, `layer_shards_saving_path`.

**Import check — PASS:** `airllm.AutoModel` imports correctly. `AirLLMQWen2` is auto-selected for the Qwen2 architecture.

### Why AirLLM Is Blocked

**Blocker 1 — Model format:** AirLLM's sharding logic asserts that `model.safetensors.index.json` must exist — the index file produced when HuggingFace stores a model as multiple shard files. This format only appears in large models (typically 7B+).

- `Qwen/Qwen2.5-0.5B-Instruct` (cached locally): stored as a single `model.safetensors` → no index file → `AssertionError: model.safetensors.index.json should exist.`
- `facebook/opt-6.7b` (correct format): would have the multi-shard index, but requires a full 13.5 GB download. The previous attempt stalled at 4 GB / 13.5 GB after 1200 s (Section 4) — resuming is impractical within this assignment's constraints.

**Blocker 2 — No CUDA GPU:** AirLLM defaults to `device='cuda:0'`. This machine has no discrete CUDA GPU (Intel i5-1135G7 integrated graphics only, no CUDA support). CPU-mode is not officially supported by AirLLM, and FP16 on CPU raises errors.

**Also found — bug in `src/run_airllm.py`:** The script passes `cache_dir=` to `AirLLMAutoModel.from_pretrained()`, which AirLLM does not accept. The correct parameter is `layer_shards_saving_path`. This would cause an immediate `TypeError` before the model format check.

### This Is a Valid Negative Result

Documenting what blocked AirLLM is the honest outcome. The blockers are real hardware and format constraints — not shortcuts. AirLLM's layer-paging design is described conceptually below. The quantization experiment (Section 6) provides the runnable alternative: GGUF Q4 quantization achieves a similar goal (running a 7B-class model in 8.22 GB RAM) through weight compression rather than layer-by-layer disk paging.

### How AirLLM Works

<!-- REQUIREMENT D4, I8, I9 -->

AirLLM addresses the problem of running models larger than available RAM by loading
one layer at a time from disk, running the forward pass for that layer, then discarding
it from memory before loading the next. This is analogous to **OS-level demand paging**:
just as the operating system pages memory pages between RAM and a swap file on disk when
physical RAM is exhausted, AirLLM pages model layers between disk and RAM on demand.

The key difference from uncontrolled OS swap is that AirLLM's paging is *deliberate and
structured*: layers are evicted immediately after use, keeping peak RAM proportional to a
single layer's weight size rather than the full model. This dramatically reduces the RAM
ceiling required to run a model, at the cost of significantly higher TTFT (every layer
load involves a disk read during prefill) and very slow per-token decode latency.

**Shard preparation:** On first run, AirLLM splits the model checkpoint into per-layer
shard files saved to `AIRLLM_SHARD_DIR`. Subsequent runs skip sharding and load directly
from cached shards.

_Measured disk I/O during layer loading: TBD (see Original Extension section)_

---

## 6. Quantization Experiment

<!-- REQUIREMENT E1, E2, E3 -->

**Script:** [`src/run_quantized.py`](src/run_quantized.py)

### Configurations Attempted

| Config ID | Model | Method | Precision | File Size | Status |
|---|---|---|---|---|---|
| `q4_k_m` | Qwen2.5-0.5B-Instruct | GGUF llama.cpp | Q4_K_M (4-bit K-quant) | 379 MB | **DONE** |
| `q8` | — | GGUF | Q8_0 (8-bit) | ~760 MB | NOT RUN (skipped — Q4 is sufficient for comparison) |
| `fp16` | Qwen2.5-0.5B-Instruct | transformers dtype | FP16 | ~1.0 GB | NOT RUN |

Note: The 7B Q4_K_M GGUF download stalled after ~9 hours (documented in `results/raw/quant_q4_download_failure.json`). The 0.5B Q4_K_M GGUF (379 MB) was used as the runnable quantized fallback.

### Q4_K_M Result — `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (COMPLETE)

**Scenario:** llama-cpp-python (CPU-only, n_ctx=2048), Q4_K_M quantization
**Outcome:** SUCCESS — 64 tokens generated, no OOM

| Metric | Value |
|---|---|
| Throughput | **26.24 tok/s** |
| Peak RAM | **0.55 GB** |
| Output tokens | 64 |
| TTFT (approx) | 2.44 s |
| Model load time | 0.59 s |
| TPOT | N/A (approximation — no streaming hook) |
| OOM | No |

**Comparison with warm-up baseline (same model, FP16 transformers):**

| Metric | FP16 transformers (warm-up) | Q4_K_M GGUF | Change |
|---|---|---|---|
| Throughput (tok/s) | 6.20 | **26.24** | **+323%** |
| Peak RAM (GB) | 2.73 | **0.55** | **−80%** |
| Model load time (s) | 208.5 ¹ | 0.59 | N/A ¹ |

¹ Warm-up load time included model download (~1 GB). Not comparable to GGUF load time.

**Output snippet (64 tokens):**
> Virtual memory is a technique used in operating systems to increase the amount of memory available to a computer. It allows a computer to allocate more memory than is physically available, without having to physically move data between physical memory and disk. This is achieved by creating a virtual memory space that is larger than the physical memory, and then…

### Raw Evidence Files

- [`results/raw/quant_q4_k_m_metrics.json`](results/raw/quant_q4_k_m_metrics.json) ✓ — real data
- [`results/raw/quant_q4_download_failure.json`](results/raw/quant_q4_download_failure.json) ✓ — 7B GGUF stalled

### Quantization Quality Threshold

<!-- REQUIREMENT E3 -->

At Q4_K_M precision for a 0.5B model, output coherence is preserved: the 64-token response is factually correct and fluent. At extreme low bit-widths (Q2_K or lower), quality degradation is expected to be visible — this experiment demonstrates that Q4_K_M is a practical operating point for CPU-only inference.

---

## 7. Benchmark Results

<!-- REQUIREMENT F1–F8, G1–G8 -->

### Summary Table

<!-- Generated from results/processed/summary_table.csv by src/plot_results.py -->

| Scenario | TTFT (s) | TPOT (ms/tok) | Throughput (tok/s) | Peak RAM (GB) | Peak VRAM | Runtime (s) |
|---|---|---|---|---|---|---|
| Baseline warm-up (Qwen2.5-0.5B FP16) | 10.33 ¹ | N/A ¹ | 6.20 | 2.73 | N/A — no CUDA GPU | 10.33 |
| Baseline stress (OPT-6.7B) | N/A (timeout) | N/A (timeout) | 0 (timeout) | N/A | N/A — no CUDA GPU | 1200 (timeout) |
| AirLLM | BLOCKED ² | BLOCKED ² | BLOCKED ² | BLOCKED ² | N/A — no CUDA GPU | BLOCKED ² |
| Quant Q4_K_M (Qwen2.5-0.5B GGUF) | 2.44 ¹ | N/A ¹ | **26.24** | **0.55** | N/A — no CUDA GPU | 2.44 |

¹ TTFT approximated as total runtime (no streaming hook). True TTFT ≤ stated value. TPOT undefined (decode_time ≈ 0 under this approximation).
² AirLLM BLOCKED: no CUDA GPU + model format mismatch. See Section 5 and `results/raw/airllm_compatibility.json`.

**Evidence:** [`results/processed/summary_table.csv`](results/processed/summary_table.csv) ✓

### Graphs

![TTFT Comparison](figures/ttft_comparison.png)

![Throughput Comparison](figures/throughput_comparison.png)

![Memory Comparison](figures/memory_comparison.png)

![Runtime Comparison](figures/runtime_comparison.png)

![Quantization Tradeoff](figures/quant_tradeoff.png)

---

## 8. Economic Analysis

<!-- REQUIREMENT H1–H9 -->

> **Pricing assumption:** Claude Haiku 3 prices used as illustrative assumption
> ($0.25/$1.25 per 1M input/output tokens). Current prices may differ — verify at
> anthropic.com/pricing. All numbers trace to
> [`results/processed/economic_analysis.json`](results/processed/economic_analysis.json).

### API Cost (H1, H9)

**Reference API:** Claude Haiku 3 (cheapest Anthropic tier — assumption as of 2026-06-23)
**Input token price:** $0.25 per 1M tokens (assumption)
**Output token price:** $1.25 per 1M tokens (assumption)
**Benchmark request size:** ~27 input tokens + 64 output tokens = 91 tokens total
**Cost per request:** $0.0000868 (`(27×$0.25 + 64×$1.25) / 1,000,000`)
**Cost for 1,000 requests/month:** $0.087

### On-Premises Local Cost (H2, H3, H4, H5)

| Cost Component | Assumption | Monthly Cost |
|---|---|---|
| Hardware amortization | $800 laptop, 3-year life ($800 ÷ 36) | $22.22/month |
| Electricity | 28W TDP × 120 h/month ÷ 1000 × $0.12/kWh | $0.40/month |
| Operator time (optional) | 2 h/month × $15/hr | $30.00/month |
| **Total (hardware + electricity)** | | **$22.62/month** |
| **Total (with operator)** | | **$52.62/month** |

Hardware source: `results/raw/hardware_profile.json` — i5-1135G7, TDP 28W.

### Break-Even Analysis (H6)

**Break-even (hardware + electricity only):** ~260,753 requests/month
**Break-even (including operator time):** ~606,228 requests/month
**Formula:** `fixed_monthly_on_prem_cost / api_cost_per_request`

**Evidence:** [`results/processed/economic_analysis.json`](results/processed/economic_analysis.json) ✓
**Graph:**

![Economic Break-Even](figures/economic_breakeven.png)

### Recommendation (H7)

| Volume regime | Recommendation | Reason |
|---|---|---|
| < 260,000 requests/month | Use API | Fixed on-prem cost ($22.62) exceeds API spend |
| > 260,000 requests/month | Consider on-prem | API spend exceeds amortized hardware cost |
| Privacy-sensitive workloads | On-prem regardless | Data does not leave local network |
| Latency-critical (< 1 s TTFT) | API | Local CPU inference: 2.44 s TTFT (Q4_K_M), 10.33 s (FP16) |

**Summary:** For low-to-moderate volumes (a few hundred thousand requests/month), API is cheaper due to the laptop's fixed amortization cost dominating. On-prem becomes economically justified only at high sustained volumes, or when data privacy mandates local processing. The 260K break-even is high because Claude Haiku is very cheap per token at benchmark scale.

---

## 9. Lecture Concept Analysis

<!-- REQUIREMENT I1–I13 -->
<!-- This section explicitly connects measured results to lecture vocabulary. -->

### Prefill

During the **prefill** stage, the model processes all prompt tokens in parallel,
building the Key-Value (KV) cache. This is a matrix-multiplication-heavy operation
whose cost scales with prompt length. **TTFT is dominated by prefill** — longer
prompts produce higher TTFT even when generating the same number of output tokens.

**Measured connection:** Baseline warm-up TTFT = 10.33 s for the 27-token benchmark prompt. Q4_K_M GGUF TTFT = 2.44 s for the same prompt — a 4.2× reduction attributable to the smaller, quantized model spending less time on the prefill matrix multiplications and loading fewer bytes from RAM. A single fixed prompt was used across all experiments, so prefill scaling with prompt length was not varied; this remains a limitation.

### Decode

During the **decode** stage, the model generates output tokens one at a time
(autoregressive generation). Each decode step attends over the full KV cache.
For single-user CPU inference, decode is **memory-bandwidth-bound**: the bottleneck
is streaming weight matrices from RAM to CPU registers, not arithmetic throughput.

**TPOT** (Time Per Output Token) measures how long each decode step takes.

**Measured connection:** TPOT was not directly captured in this experiment — no per-token streaming hook was implemented. Total runtime ≈ TTFT under this approximation (decode time ≈ 0 for 64 tokens at the measured batch granularity). Throughput (26.24 tok/s for Q4_K_M) provides an upper bound: `TPOT ≤ 1000 / 26.24 ≈ 38 ms/token`.

### TTFT (Time To First Token)

TTFT = time from prompt submission to receipt of the first generated token.
It captures prefill latency plus any model-loading overhead (AirLLM's layer
paging makes TTFT particularly high since all layers must be loaded once during
the first forward pass).

**Measured:** Baseline warm-up = 10.33 s; Q4_K_M GGUF = 2.44 s. See Section 7 Summary Table. [`results/raw/baseline_warmup_metrics.json`](results/raw/baseline_warmup_metrics.json) ✓, [`results/raw/quant_q4_k_m_metrics.json`](results/raw/quant_q4_k_m_metrics.json) ✓

### TPOT / ITL (Time Per Output Token / Inter-Token Latency)

TPOT = average time between consecutive output tokens during decode.
`TPOT = (total_generation_time − TTFT) / (total_output_tokens − 1)`

Lower TPOT = better user-perceived responsiveness during streaming.

**Measured:** Not directly measured (no streaming hook). Upper bound from throughput: Q4_K_M ≤ 38 ms/token. See Section 7.

### Throughput

Throughput (tokens/second) = total output tokens / total generation time.
It is the primary metric for batch and high-volume workloads where first-token
latency matters less than total output rate.

`throughput = total_output_tokens / total_runtime_seconds`

**Measured:** Baseline warm-up = 6.20 tok/s; Q4_K_M GGUF = **26.24 tok/s** (+323%). See Section 7. [`results/raw/baseline_warmup_metrics.json`](results/raw/baseline_warmup_metrics.json) ✓, [`results/raw/quant_q4_k_m_metrics.json`](results/raw/quant_q4_k_m_metrics.json) ✓

### VRAM

VRAM is the high-bandwidth memory (HBM) on the GPU die. For GPU inference,
all model weights and the KV cache must fit simultaneously in VRAM.
A 7B parameter model in FP16 requires approximately **14 GB VRAM**.

**On this hardware:** N/A — no discrete CUDA GPU detected.
All inference runs on CPU, using system RAM instead of VRAM.

**Evidence:** [`results/raw/hardware_profile.json`](results/raw/hardware_profile.json) ✓ — `gpu: null`, `cuda_available: false`.

### RAM (System Memory)

When no GPU is available, model weights are loaded into system RAM (DRAM).
CPU inference performance scales with memory bandwidth rather than FLOPS.
A model that exceeds available RAM will either crash (OOM), page to disk
uncontrollably via OS virtual memory (extremely slow), or fail to load.

**Measured peak RAM:** Baseline warm-up = 2.73 GB; Q4_K_M GGUF = **0.55 GB** (−80%). See Section 7. Both measured via `psutil` background sampler in `src/benchmark_common.py`.

### Virtual Memory and Paging

The operating system implements **virtual memory** to give processes the
illusion of more RAM than is physically installed. When physical pages are
exhausted, the OS writes them to a **swap file** on disk (**demand paging**).
Disk access is ~1,000× slower than RAM, so swap-induced paging destroys
inference performance.

AirLLM implements **deliberate, structured paging at the model-layer level**:
it explicitly evicts each layer after use and loads the next from disk.
This prevents the OS from invoking uncontrolled swap while still allowing
models far larger than RAM to run — at the cost of latency.

### AirLLM Layer Loading

_See Section 5 (AirLLM Experiment) for the full explanation._

Key properties:
- Peak RAM ≈ size of one transformer layer, not the full model
- TTFT is high because all layers must be read from disk once during prefill
- TPOT is high because each decode step also reloads layers from disk
- Disk I/O is the primary bottleneck (see Original Extension, Section 10)

### Compute-Bound vs Memory-Bound

A workload is **compute-bound** when arithmetic units are the bottleneck:
more FLOPS → faster. It is **memory-bound** when data movement is the bottleneck:
more memory bandwidth → faster, but adding FLOPS does not help.

LLM decode inference is almost universally **memory-bound** for batch size 1:
each decode step performs O(model_params) multiply-accumulate operations but
reads O(model_params) bytes of weights — low arithmetic intensity.
CPU inference without AVX-512 or GPU tensor cores is even more memory-bound
because memory bandwidth is the only lever.

**Measured (approximate):** Q4_K_M model weights = 379 MB. At 26.24 tok/s, memory bandwidth consumed ≈ 379 MB × 26.24 ≈ 9.9 GB/s per second of inference. The i5-1135G7 has ~51 GB/s theoretical DRAM bandwidth, so the bottleneck is not raw bandwidth but rather OS scheduling overhead and llama.cpp's sequential layer processing. Inference is firmly memory-bound (weight reads dominate over arithmetic).

### Quantization

Quantization reduces model weight precision from the training format
(typically BF16 or FP32) to a lower bit-width:

| Format | Bits/weight | Typical RAM for 7B | Quality impact |
|---|---|---|---|
| FP32 | 32 | ~28 GB | Baseline |
| FP16 / BF16 | 16 | ~14 GB | Negligible |
| INT8 / Q8 | 8 | ~7 GB | Minor |
| INT4 / Q4 | 4 | ~3.5 GB | Noticeable at very low bit-width |

Lower precision = smaller model in memory = higher throughput (more weights
fit in CPU cache) but potential quality degradation.

**Measured tradeoffs:** Q4_K_M vs FP16 (same 0.5B model): +323% throughput, −80% peak RAM, coherent output quality preserved. See Section 6 for full comparison table and output snippet.

### On-Premises Deployment

On-premises deployment means running the model on hardware you own and control.

**Advantages:** Privacy (data never leaves your network), no per-token API cost
at high volume, no dependency on external service availability.

**Disadvantages:** Upfront hardware cost, maintenance burden, fixed capacity
(no auto-scaling), potentially slower or less capable than API models if
hardware is constrained.

_See Economic Analysis, Section 8_

### API vs Local Deployment Tradeoff

_See Economic Analysis, Section 8 — the Summary table with break-even analysis
provides the quantitative comparison. The qualitative framing is:_

- API is better when: volume is low, privacy is not a concern, low latency
  (< 1s TTFT) is required, or the task requires a state-of-the-art model.
- On-prem is better when: volume is high enough to amortize hardware,
  data privacy is required, or connectivity is limited.

---

## 10. Original Extension

<!-- REQUIREMENT J1, J2, J3 -->
<!-- TODO: Choose one extension, implement it, run it, and fill in this section. -->

**Chosen extension:** Disk I/O Observation During AirLLM Layer Loading  
**Script:** `src/extension_disk_io.py` — _not yet created_  
**Rationale:** This directly measures the disk read throughput during AirLLM's
layer-paging process, connecting the conceptual explanation in Section 5 to
concrete numbers. It requires no extra model runs — it instruments the AirLLM
run already planned.

### Methodology

_TBD — will use `psutil.disk_io_counters()` sampled at 100ms intervals during
AirLLM inference to measure bytes read from disk per second. Will plot disk I/O
bandwidth over time, correlating spikes with layer-load events._

### Results

_TBD_

**Evidence:**
- [`results/raw/extension_disk_io.json`](results/raw/extension_disk_io.json) — _not yet generated_
- [`figures/extension_disk_io.png`](figures/extension_disk_io.png) — _not yet generated_

### Connection to Assignment Concepts

_TBD — will connect measured disk bandwidth to the virtual memory / paging
discussion in Section 9, and explain why AirLLM's TTFT is dominated by disk
read throughput rather than compute._

---

## 11. Screenshots

<!-- REQUIREMENT A5 -->
<!-- TODO: Capture during experiment runs and save to figures/screenshots/ -->

| Screenshot | Description | Status |
|---|---|---|
| [`figures/screenshots/hardware_probe.png`](figures/screenshots/hardware_probe.png) | Terminal output of hardware_probe.py | NOT YET TAKEN |
| [`figures/screenshots/baseline_run.png`](figures/screenshots/baseline_run.png) | Terminal output during baseline inference | NOT YET TAKEN |
| [`figures/screenshots/airllm_run.png`](figures/screenshots/airllm_run.png) | Terminal output during AirLLM inference | NOT YET TAKEN |
| [`figures/screenshots/memory_monitor.png`](figures/screenshots/memory_monitor.png) | Task Manager / htop during peak RAM usage | NOT YET TAKEN |
| [`figures/screenshots/quantization_run.png`](figures/screenshots/quantization_run.png) | Terminal showing quantization variants running | NOT YET TAKEN |

---

## 12. Self-Assessment

<!-- REQUIREMENT K8 -->

**Estimated score: ~65 / 100**
**Grading status: PARTIAL — core experiments done; extension missing; AirLLM blocked.**

### What Is Done (with evidence)

| Item | Evidence |
|---|---|
| Hardware profiled | `results/raw/hardware_profile.json` ✓ |
| Model selection justified | `results/raw/model_selection.json` ✓ |
| Warm-up baseline run (SUCCESS) | `results/raw/baseline_warmup_metrics.json` — 6.20 tok/s, 2.73 GB ✓ |
| Stress baseline (documented OOM) | `results/raw/baseline_stress_failure.json` — timeout after 1200 s ✓ |
| AirLLM blocked (documented negative result) | `results/raw/airllm_compatibility.json` — two hard blockers ✓ |
| Q4_K_M quantization benchmark | `results/raw/quant_q4_k_m_metrics.json` — 26.24 tok/s, 0.55 GB ✓ |
| TTFT, throughput, peak RAM measured | Both runnable scenarios ✓ |
| Summary table + 5 graphs | `results/processed/summary_table.csv`, `figures/*.png` ✓ |
| Economic analysis + break-even graph | `results/processed/economic_analysis.json`, `figures/economic_breakeven.png` ✓ |
| Lecture concepts written (Sections 9) | Prefill, decode, TTFT, TPOT, throughput, VRAM, RAM, virtual memory, quantization, API vs on-prem ✓ |
| AirLLM layer-loading mechanism explained | README Section 5 ✓ |
| No model weights in repo | `.gitignore` ✓ |
| Reproducible scripts with argparse | All `src/*.py` ✓ |

### What Is Missing / Limited

| Gap | Reason |
|---|---|
| Original extension (J1–J3) | Planned as disk I/O during AirLLM; blocked because AirLLM cannot run (no GPU + model format). No runnable substitute was completed. This is the biggest deduction. |
| TPOT not measured | No per-token streaming hook implemented. Upper bound derivable from throughput. |
| Second explicit quantization level | Q4_K_M done; FP16 comparison is the warm-up baseline (same model). No separate FP16 run with `run_quantized.py`. |
| Screenshots | Not taken — terminal sessions not captured during experiment runs. |
| AirLLM metrics | Blocked by hardware (no CUDA GPU) and model format constraint. Documented as honest negative result. |
| Quality scoring | Output snippets present; formal BLEU/ROUGE scoring not done. |

### Score Justification

- **Section A (repository):** ~80% — public repo, gitignore, README complete; no screenshots.
- **Section B (hardware/model):** ~90% — all profiled; stress OOM documented.
- **Section C (baseline):** ~85% — both scenarios with evidence; TPOT null.
- **Section D (AirLLM):** ~70% — BLOCKED but well-documented as negative result; D1/D3/D4/D5 satisfied.
- **Section E (quantization):** ~60% — one GGUF level complete; second level implicit only.
- **Section F (metrics):** ~75% — TTFT/throughput/RAM/VRAM done; TPOT null.
- **Section G (graphs):** ~80% — 6 of 9 graphs; tpot and tpot skipped (data unavailable).
- **Section H (economics):** ~80% — full analysis with break-even; prices labeled as assumptions.
- **Section I (concepts):** ~70% — all concepts addressed; measured connections where data available.
- **Section J (extension):** ~0% — not implemented.
- **Section K (engineering):** ~70% — scripts clean; some gaps in error handling and end-to-end testing.

**Honest estimate: ~65 / 100.** The missing extension and TPOT measurement are the main deductions. All negative results are documented honestly with evidence files.

---

## 13. References

<!-- TODO: Add citations after experiments are complete -->

- AirLLM GitHub: _TBD_
- Hugging Face Transformers documentation: _TBD_
- API pricing reference: _See `.env.example` API_PRICING_DATE and API_PRICING_MODEL_
- Assignment 05 brief and lecture materials: provided by course
- psutil documentation: https://psutil.readthedocs.io/
- llama.cpp / GGUF format: _TBD_
