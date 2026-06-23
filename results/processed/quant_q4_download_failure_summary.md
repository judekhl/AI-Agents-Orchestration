# Quantization Q4_K_M Download Failure Summary

**Date:** 2026-06-23

## What Was Attempted

Download `Qwen2.5-7B-Instruct-Q4_K_M.gguf` from `bartowski/Qwen2.5-7B-Instruct-GGUF` on HuggingFace.

| Field | Value |
|---|---|
| Intended file | Qwen2.5-7B-Instruct-Q4_K_M.gguf |
| Intended size | ~4.4 GB |
| Target directory | C:\ai-model-cache\gguf |
| Duration left running | ~9 hours |

## Actual Result

No GGUF file was written. Directory contained only `.cache` with 0 GB of usable data. The download stalled or hung without producing the target file.

## Conclusion

The Q4_K_M 7B GGUF download is not reliable inside Claude on this machine/connection. There is no file to benchmark. This experiment is blocked until a GGUF file is obtained.

## Evidence File

`results/raw/quant_q4_download_failure.json`

## Next Action

Use a smaller GGUF quantized model (0.5B–1.5B parameters) that can be downloaded reliably, or manually download the Q4_K_M file outside Claude and place it at `C:\ai-model-cache\gguf\Qwen2.5-7B-Instruct-Q4_K_M.gguf`. Do not attempt 7B Q4 download again inside Claude.
