# Hardware Summary
Generated from: `results/raw/hardware_profile.json`
Probe date: 2026-06-23

## Specifications

| Component | Detail |
|---|---|
| **CPU** | 11th Gen Intel Core i5-1135G7 @ 2.40 GHz |
| **CPU Cores** | 4 physical / 8 logical (Hyper-Threading) |
| **RAM (total)** | 8.22 GB |
| **RAM (available at probe)** | 0.87 GB |
| **GPU** | N/A — no discrete CUDA GPU |
| **VRAM** | N/A — no discrete GPU |
| **Disk model** | NVMe KINGSTON OM8SBP3512K-AH |
| **Disk type** | NVMe SSD |
| **Disk total** | 511.04 GB |
| **Disk free** | 38.44 GB |
| **OS** | Windows 11 (build 10.0.26200, 24H2) |
| **Python** | 3.10.0 |

## Critical Constraints for This Assignment

| Constraint | Value | Impact |
|---|---|---|
| RAM total | 8.22 GB | A 7B FP16 model needs ~14 GB — baseline will OOM |
| RAM available | 0.87 GB | Almost no headroom; even 1B models may struggle in FP32 |
| GPU | None | All inference is CPU-only; no CUDA acceleration |
| VRAM | None | VRAM metric will be "N/A" for all experiments |
| Disk free | 38.44 GB | Enough for OPT-6.7B FP16 (~13.4 GB) with margin |
| Disk type | NVMe SSD | Favorable for AirLLM — faster layer-paging than HDD |

## Model Feasibility at a Glance

| Model | Size (FP16) | Fits in RAM? | Verdict |
|---|---|---|---|
| facebook/opt-1.3b | ~2.6 GB | Possibly (at ~8 GB total) | Borderline — depends on available RAM |
| facebook/opt-6.7b | ~13.4 GB | No | OOM on naive load; AirLLM or GGUF Q4 required |
| EleutherAI/gpt-j-6b | ~12.1 GB | No | OOM on naive load |
| Any GGUF Q4_K_M 7B | ~3.5–4 GB | Yes | Viable for quantization experiments |

## Implication for Assignment Strategy

The 8.22 GB RAM ceiling is the defining hardware constraint.
- **Baseline experiment**: loading a 7B model in FP16 will trigger OOM — this is an expected,
  documentable negative result, not a failure to attempt.
- **AirLLM**: by paging one layer at a time, peak RAM stays proportional to one layer's weight
  (~100–200 MB for a 7B model), making inference feasible at the cost of high TTFT.
- **Quantization**: GGUF Q4_K_M reduces a 7B model to ~3.5 GB — fits within available RAM
  and enables a successful throughput measurement.
- **NVMe SSD**: reduces AirLLM's disk I/O bottleneck compared to HDD; sequential read
  speeds of ~2000+ MB/s mean layer loads are measured in milliseconds rather than seconds.
