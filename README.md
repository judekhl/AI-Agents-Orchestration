# Assignment 05: Running a Massive LLM Locally
## AirLLM, Quantization and Performance Benchmarking

**Author:** Jude  
**Course:** [Course name]  
**Repo:** https://github.com/judekhl/AI-Agents-Orchestration

> **Status: SCAFFOLD — experiments not yet run.**
> All section headers are in place. Numbers, graphs, and analysis will be added
> after real measurements are complete. Do not read _TBD_ entries as results.

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

**Expected total runtime:** _TBD_ (depends heavily on hardware)

---

## 4. Baseline Experiment

<!-- REQUIREMENT C1, C2, C3, C4 -->
<!-- TODO: After running src/run_baseline.py, fill in this section.
     Report honestly — OOM and failure are valid results if documented. -->

**Script:** [`src/run_baseline.py`](src/run_baseline.py)  
**Config:** [`experiments/configs/default_config.json`](experiments/configs/default_config.json)  
**Outcome:** _TBD — Success / OOM / Excessive slowness / Failure_

### Raw Evidence

- Success case: [`results/raw/baseline_metrics.json`](results/raw/baseline_metrics.json) — _not yet generated_
- Failure case: [`results/raw/baseline_failure.txt`](results/raw/baseline_failure.txt) — _not yet generated_

### Bottleneck Analysis

<!-- REQUIREMENT C4, I10 -->
<!-- TODO: Identify the limiting resource. Connect to lecture concepts. -->

**Bottleneck identified:** _TBD (RAM / CPU compute / Disk I/O)_

_Analysis: TBD — will explain whether this is compute-bound or memory-bound,
referencing peak RAM from the metrics file and CPU utilization observations._

---

## 5. AirLLM Experiment

<!-- REQUIREMENT D1, D2, D3, D4, D5 -->
<!-- TODO: After running src/run_airllm.py, fill in this section. -->

**Script:** [`src/run_airllm.py`](src/run_airllm.py)  
**Shard directory:** Configured via `AIRLLM_SHARD_DIR` in `.env` (default: `./airllm_shards/`)  
**Cache directory:** Configured via `AIRLLM_CACHE_DIR` in `.env` (default: `./airllm_cache/`)  
**Outcome:** _TBD — Success / Failure (if failure, see airllm_failure.txt)_

### Raw Evidence

- Success case: [`results/raw/airllm_metrics.json`](results/raw/airllm_metrics.json) — _not yet generated_
- Failure case: [`results/raw/airllm_failure.txt`](results/raw/airllm_failure.txt) — _not yet generated_

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
<!-- TODO: After running src/run_quantized.py, fill in this section. -->

**Script:** [`src/run_quantized.py`](src/run_quantized.py)

### Configurations Attempted

| Config ID | Method | Precision | Expected RAM Reduction |
|---|---|---|---|
| `fp32` | `transformers` dtype | FP32 (full) | baseline |
| `fp16` | `transformers` dtype | FP16 / BF16 | ~50% vs FP32 |
| `q8` | `bitsandbytes` or GGUF | INT8 | ~50% vs FP16 |
| `q4` | `bitsandbytes` or GGUF | INT4 | ~50% vs Q8 |

### Raw Evidence Files

- [`results/raw/quant_fp32_metrics.json`](results/raw/quant_fp32_metrics.json) — _not yet generated_
- [`results/raw/quant_fp16_metrics.json`](results/raw/quant_fp16_metrics.json) — _not yet generated_
- [`results/raw/quant_q8_metrics.json`](results/raw/quant_q8_metrics.json) — _not yet generated_
- [`results/raw/quant_q4_metrics.json`](results/raw/quant_q4_metrics.json) — _not yet generated_

### Quantization Quality Threshold

<!-- REQUIREMENT E3 -->
<!-- TODO: After experiments, compare outputs at each level. -->

_TBD — will discuss at which quantization level coherence or factual accuracy
noticeably degrades, with example output comparisons._

---

## 7. Benchmark Results

<!-- REQUIREMENT F1–F8, G1–G8 -->
<!-- TODO: This section is generated from results/processed/summary_table.csv
     by src/plot_results.py. Do not fill in numbers manually. -->

### Summary Table

<!-- Generated from results/processed/summary_table.csv -->

| Scenario | TTFT (s) | TPOT (ms/tok) | Throughput (tok/s) | Peak RAM (GB) | Peak VRAM | Runtime (s) | Power est. (W) |
|---|---|---|---|---|---|---|---|
| Baseline | _TBD_ | _TBD_ | _TBD_ | _TBD_ | N/A — no CUDA GPU | _TBD_ | _TBD_ |
| AirLLM | _TBD_ | _TBD_ | _TBD_ | _TBD_ | N/A — no CUDA GPU | _TBD_ | _TBD_ |
| Quant FP32 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | N/A — no CUDA GPU | _TBD_ | _TBD_ |
| Quant FP16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | N/A — no CUDA GPU | _TBD_ | _TBD_ |
| Quant Q8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | N/A — no CUDA GPU | _TBD_ | _TBD_ |
| Quant Q4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | N/A — no CUDA GPU | _TBD_ | _TBD_ |

**Evidence:** [`results/processed/summary_table.csv`](results/processed/summary_table.csv) — _not yet generated_

### Graphs

<!-- Graphs are generated by src/plot_results.py from raw JSON files. -->
<!-- Replace the status notes below with actual image embeds after generation. -->

| Graph | File | Status |
|---|---|---|
| TTFT comparison | `figures/ttft_comparison.png` | NOT YET GENERATED |
| TPOT / ITL comparison | `figures/tpot_comparison.png` | NOT YET GENERATED |
| Throughput comparison | `figures/throughput_comparison.png` | NOT YET GENERATED |
| RAM / VRAM comparison | `figures/memory_comparison.png` | NOT YET GENERATED |
| Runtime comparison | `figures/runtime_comparison.png` | NOT YET GENERATED |
| Quantization tradeoff | `figures/quant_tradeoff.png` | NOT YET GENERATED |
| Economic break-even | `figures/economic_breakeven.png` | NOT YET GENERATED |

<!-- After graphs are generated, embed with:
![TTFT Comparison](figures/ttft_comparison.png)
-->

---

## 8. Economic Analysis

<!-- REQUIREMENT H1–H9 -->
<!-- TODO: After src/economic_analysis.py runs, fill in this section.
     All numbers must trace to results/processed/economic_analysis.json. -->

> **Pricing disclaimer:** API prices change frequently. The figures below use prices
> from `.env.example` as of **2026-06-23** and are illustrative assumptions.
> See `API_PRICING_DATE` and `API_PRICING_MODEL` in `.env.example` for the reference.

### API Cost

**Reference API:** _See `.env.example` for model and date_  
**Input token price:** $_TBD_ per 1M tokens  
**Output token price:** $_TBD_ per 1M tokens  
**Cost per benchmark request (~_TBD_ tokens):** $_TBD_  
**Cost for 1,000 requests/month:** $_TBD_

### On-Premises Local Cost

| Cost Component | Assumption | Monthly Cost |
|---|---|---|
| Hardware amortization | $_TBD_ hardware, 3-year life | $_TBD_/month |
| Electricity | _TBD_ W × _TBD_ hrs/month ÷ 1000 × $_TBD_/kWh | $_TBD_/month |
| Operator time | _TBD_ hrs/month × $_TBD_/hr | $_TBD_/month |
| **Total on-prem (fixed)** | | **$_TBD_/month** |

### Break-Even Analysis

**Break-even:** _TBD_ requests/month  
**Formula:** `break_even = fixed_monthly_on_prem_cost / api_cost_per_request`

**Evidence:** [`results/processed/economic_analysis.json`](results/processed/economic_analysis.json) — _not yet generated_  
**Graph:** [`figures/economic_breakeven.png`](figures/economic_breakeven.png) — _not yet generated_

### Recommendation

<!-- REQUIREMENT H7 -->
<!-- TODO: After break-even is calculated, write the recommendation. -->

| Volume regime | Recommendation | Reason |
|---|---|---|
| < _TBD_ requests/month | Use API | Fixed on-prem cost exceeds API spend |
| > _TBD_ requests/month | Consider on-prem | API spend exceeds amortized hardware |
| Privacy-sensitive workloads | On-prem regardless | Data does not leave local network |
| Latency-critical (< 1s TTFT) | API | Local hardware produces _TBD_ s TTFT |

_Full recommendation: TBD after economic analysis is complete._

---

## 9. Lecture Concept Analysis

<!-- REQUIREMENT I1–I13 -->
<!-- This section explicitly connects measured results to lecture vocabulary. -->

### Prefill

During the **prefill** stage, the model processes all prompt tokens in parallel,
building the Key-Value (KV) cache. This is a matrix-multiplication-heavy operation
whose cost scales with prompt length. **TTFT is dominated by prefill** — longer
prompts produce higher TTFT even when generating the same number of output tokens.

_Measured connection: TBD — baseline TTFT vs prompt length observation_

### Decode

During the **decode** stage, the model generates output tokens one at a time
(autoregressive generation). Each decode step attends over the full KV cache.
For single-user CPU inference, decode is **memory-bandwidth-bound**: the bottleneck
is streaming weight matrices from RAM to CPU registers, not arithmetic throughput.

**TPOT** (Time Per Output Token) measures how long each decode step takes.

_Measured connection: TBD — TPOT values from benchmark table_

### TTFT (Time To First Token)

TTFT = time from prompt submission to receipt of the first generated token.
It captures prefill latency plus any model-loading overhead (AirLLM's layer
paging makes TTFT particularly high since all layers must be loaded once during
the first forward pass).

_Measured: TBD — see Summary Table, Section 7_

### TPOT / ITL (Time Per Output Token / Inter-Token Latency)

TPOT = average time between consecutive output tokens during decode.
`TPOT = (total_generation_time − TTFT) / (total_output_tokens − 1)`

Lower TPOT = better user-perceived responsiveness during streaming.

_Measured: TBD — see Summary Table, Section 7_

### Throughput

Throughput (tokens/second) = total output tokens / total generation time.
It is the primary metric for batch and high-volume workloads where first-token
latency matters less than total output rate.

`throughput = total_output_tokens / total_runtime_seconds`

_Measured: TBD — see Summary Table, Section 7_

### VRAM

VRAM is the high-bandwidth memory (HBM) on the GPU die. For GPU inference,
all model weights and the KV cache must fit simultaneously in VRAM.
A 7B parameter model in FP16 requires approximately **14 GB VRAM**.

**On this hardware:** N/A — no discrete CUDA GPU detected.
All inference runs on CPU, using system RAM instead of VRAM.

_Evidence: `results/raw/hardware_profile.json` — not yet generated_

### RAM (System Memory)

When no GPU is available, model weights are loaded into system RAM (DRAM).
CPU inference performance scales with memory bandwidth rather than FLOPS.
A model that exceeds available RAM will either crash (OOM), page to disk
uncontrollably via OS virtual memory (extremely slow), or fail to load.

_Measured peak RAM: TBD — see Summary Table, Section 7_

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

_Measured: TBD — will calculate arithmetic intensity from throughput data_

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

_Measured tradeoffs: TBD — see Quantization Experiment, Section 6_

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
<!-- TODO: Fill in last, after ALL experiments, measurements, graphs, and
     README sections are complete. Cite evidence for every claim.
     Negative results with documentation are acceptable and honest. -->

**Estimated score:** _TBD / 100_  
**Grading status:** NOT READY — experiments not run, evidence files do not exist.

_This section will be completed after all raw data files exist in results/raw/,
all graphs exist in figures/, and all README sections above are filled with
real measured values. See reports/REQUIREMENTS_MATRIX.md for the itemized checklist._

---

## 13. References

<!-- TODO: Add citations after experiments are complete -->

- AirLLM GitHub: _TBD_
- Hugging Face Transformers documentation: _TBD_
- API pricing reference: _See `.env.example` API_PRICING_DATE and API_PRICING_MODEL_
- Assignment 05 brief and lecture materials: provided by course
- psutil documentation: https://psutil.readthedocs.io/
- llama.cpp / GGUF format: _TBD_
