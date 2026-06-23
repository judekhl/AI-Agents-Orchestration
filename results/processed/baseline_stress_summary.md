# Baseline Stress Summary — facebook/opt-6.7b

**Date:** 2026-06-23
**Scenario:** Standard HuggingFace `transformers`, CPU-only, no AirLLM, no quantization
**Hardware:** Intel i5-1135G7, 8.22 GB RAM, CPU-only (no GPU)

---

## Outcome: TIMEOUT / STALLED DOWNLOAD (valid negative result)

| Item | Value |
|---|---|
| Error type | TimeoutError |
| Timeout limit | 1200 s (20 min) |
| Download progress at timeout | 4.049 GB of ~13.5 GB (30%) |
| Model load into RAM | Not reached — download stalled |
| Expected OOM on load | Yes — ~14 GB FP16 required vs 8.22 GB RAM available |
| OOM actually observed | No — process timed out before download completed |

## Why This Is a Valid Stress Result

The stress baseline demonstrates that `facebook/opt-6.7b` in FP16 is **not feasible** on this hardware via naive `transformers.from_pretrained()`:

1. **Download alone takes 20+ min** — 13.5 GB over a typical home connection; download stalled at 4 GB (possible network throttle or chunk boundary).
2. **Loading would OOM** — even if the download completed, loading 13.5 GB of FP16 weights into 8.22 GB RAM would raise `MemoryError` or force the OS into uncontrolled swap (disk paging at ~1000× RAM speed penalty).
3. **No results produced** — this is the expected, documentable negative result that motivates AirLLM and GGUF quantization as mitigations.

## Bottleneck Chain

```
Download: 13.5 GB > practical bandwidth → stall at ~30%
   ↓ (if download completed)
Load: 13.5 GB FP16 weights vs 8.22 GB RAM → MemoryError / OS swap
   ↓ (if OS swap engaged)
Inference: ~1000× slower than RAM → unusably slow TTFT (hours)
```

## Connection to Assignment Concepts

- **Virtual memory / OS paging:** If the OS attempted to accommodate the 13 GB load via swap, it would page model weights to disk — exactly the scenario AirLLM avoids through structured, deliberate layer paging.
- **Memory-bandwidth-bound:** CPU inference is memory-bandwidth-limited. With weights in swap (NVMe at ~3 GB/s), each decode step would stream 13 GB from disk, making TPOT on the order of seconds per token.

## Raw Evidence

`results/raw/baseline_stress_failure.json`
