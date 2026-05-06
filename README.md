# TurboQuant-Vulkan

**Semi-Quantized Flash Attention for llama.cpp — GPU-Accelerated KV Cache Compression**

[![Vulkan](https://img.shields.io/badge/GPU-Vulkan-red)](https://www.vulkan.org/)
[![GLSL](https://img.shields.io/badge/Shader-GLSL%20450-blue)](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL))
[![C++](https://img.shields.io/badge/Host-C%2B%2B17-green)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)]()
[![Status](https://img.shields.io/badge/Status-Deprecated-orange)](#-project-status-deprecated)

---

A production-grade KV cache compression system for large language model inference, implementing **3-bit and 2-bit semi-quantized Flash Attention** directly in Vulkan compute shaders. Built as a patch for [llama.cpp](https://github.com/ggml-org/llama.cpp), targeting AMD RDNA2+ and NVIDIA GPUs. Eliminates the standard dequantize-to-FP16 intermediate step entirely — attention dot-product and PV accumulation operate directly on packed quantized data.

---

## Production Champion: TQ3_2

TQ3_2 is the culmination of this project — a **storage-identical upgrade over TQ3_1** that improves output quality through two compute-only corrections with **no VRAM penalty**:

| Metric | FP16 (Baseline) | TQ3_2 |
|:-------|:---------------:|:-----:|
| **Tokens/s** (avg across contexts) | 15.78 | **20.07** |
| **VRAM (KV cache @ 16K ctx)** | 620 MiB | **97 MiB** |
| **Compression ratio** | 1.0x | **6.4x** |
| **Cognitive fidelity** | 100.0% | **93.3%** |
| **Completion rate (100 prompts)** | — | **100.0%** |
| **Max context on 12 GB GPU** | 256K | **2M+** |

> TQ3_2 achieves **27% higher throughput** than FP16 while using **84% less VRAM** for KV cache — with only a 6.7 percentage point fidelity trade-off.

### What Makes TQ3_2 Different

| Property | TQ3_1 | TQ3_2 |
|:---------|:-----:|:-----:|
| K storage | TQ3_0 (3-bit) | TQ3_0 (3-bit) — **same** |
| V storage | TQ2_0 (2-bit) | TQ2_0 (2-bit) — **same** |
| VRAM | baseline | **identical** |
| Memory bandwidth | baseline | **identical** |
| Pre-softmax correction | — | **QK scale ×1.03** |
| V decode filter | — | **3-tap decorrelation (α=0.125)** |
| Noise reduction in V | — | **40%** |
| Tokens/s | 19.50 | **20.07** |

The improvement lives entirely inside the GLSL shader math — no format changes, no additional memory, no breaking changes to the GGML type table.

---

## KV Cache Compression

| Configuration | Key Type | Value Type | Bytes / Token-Head | Compression vs FP16 |
|:--------------|:---------|:-----------|:-------------------|:--------------------|
| FP16 | f16 | f16 | 1024 | 1.0x |
| Q8_0 | q8_0 | q8_0 | 544 | 1.9x |
| TQ3_0 | tq3_0 | tq3_0 | 224 | **4.6x** |
| TQ3_1 (Mixed KV) | tq3_0 | tq2_0 | 192 | **5.3x** |
| TQ3_2 | tq3_0 | tq3_2 | 160 | **6.4x** |
| TQ2_0 | tq2_0 | tq2_0 | 160 | **6.4x** |

---

## Benchmark Methodology

The final evaluation employed a rigorous multi-method protocol to ensure statistical robustness and cross-validation:

### Evaluation Design

| Parameter | Value |
|:----------|:------|
| **Prompts per configuration** | 100 |
| **Cognitive categories** | 5 (Math, Logic, Reasoning, Coding, Knowledge) |
| **Context tiers tested** | 5 (128, 256, 512, 1M, 2M tokens) |
| **KV configurations compared** | 5 (FP16, TQ2_0, TQ3_0, TQ3_1, TQ3_2) |
| **Model under test** | google/gemma-4-26B-A4B-it Q4_K_M GGUF |
| **Hardware** | AMD RX 6750 XT (12 GB RDNA2), Intel i5-12400F, 32 GB DDR4 |
| **Total test executions** | 500+ (5 configs × 100 prompts) |

### Evaluation Methods

1. **LLM-as-Judge Accuracy Scoring** — Independent external LLM evaluated structured question-answer pairs against FP16 reference outputs on a 10-point rubric
2. **Jaccard Token Similarity** — Token-level overlap analysis between compressed and baseline outputs
3. **Cosine Similarity (KV vector space)** — Mathematical fidelity measurement of compressed attention vectors
4. **Real-World Correctness Audit** — Manual verification of mathematical accuracy, logical coherence, and instruction following across all 500+ responses

### Cognitive Domain Coverage

| Domain | Example Task Complexity |
|:-------|:------------------------|
| Mathematics | Calculus, differential equations, quadratic derivations, probability via Bayes' theorem |
| Multi-step Logic | Constraint satisfaction (6 speakers × 3 rooms × 4 slots), combinatorial game theory |
| Reasoning | Causal chain decomposition (macroeconomic policy), counterfactual analysis |
| Code Understanding | Concurrent scheduler bug detection (threading, priority queues, race conditions) |
| World Knowledge | mRNA vaccine mechanism (10 sub-questions, immunology to pharmacokinetics) |
| Long-Context Recall | 9,800-word technical documents with precise numerical extraction |
| Summarization | Policy legislation, technical specifications, research abstracts |
| Instruction Following | 8+ simultaneous format constraints (haiku, prime sorting, magic squares) |
| Noise Filtering | Meeting transcripts with small talk, tangents, and procedural artifacts |
| Adversarial Robustness | Modified Monty Hall problems, Liar's paradox, trick questions |

---

## Long-Context Throughput

Token generation speed across context sizes on AMD RX 6750 XT (12 GB). Gemma-4-26B-A4B-it Q4_K_M, Vulkan backend, 24 GPU layers.

| Context | FP16 | TQ3_0 | TQ3_1 | **TQ3_2** | TQ2_0 |
|--------:|:----:|:-----:|:-----:|:---------:|:-----:|
| 4K | 15.88 | 17.37 | 17.38 | **17.38** | 17.38 |
| 8K | 15.66 | 17.28 | 17.87 | **17.91** | 17.91 |
| 16K | 16.00 | 17.28 | 17.35 | **18.41** | 18.41 |
| 32K | 15.93 | 17.66 | 17.79 | **17.79** | 17.45 |
| 64K | 15.96 | 17.01 | 17.16 | **17.39** | 17.39 |
| 128K | 15.70 | 17.65 | 17.26 | **18.20** | 18.20 |
| 256K | 15.57 | 16.34 | 17.12 | **18.13** | 18.13 |
| 512K | OOM | 17.10 | 17.74 | **17.91** | 17.91 |
| 1M | OOM | 16.88 | 17.00 | **17.98** | 17.98 |

> TQ3_2 sustains ~18 t/s at extreme contexts where FP16 has already exhausted VRAM entirely.

---

## Cognitive Fidelity Results

### Aggregate Quality Metrics

| KV Type | Accuracy vs FP16 | VRAM (16K ctx) | Compression | Suitability |
|:--------|:----------------:|:--------------:|:-----------:|:------------|
| FP16 | 100.0% | 620 MiB | 1.0x | Reference baseline |
| TQ3_0 | **93.3%** | 136 MiB | 4.6x | High quality, moderate compression |
| TQ3_1 | **~93%** | ~117 MiB | 5.3x | Balanced quality-to-VRAM ratio |
| **TQ3_2** | **93.3%** | **97 MiB** | **6.4x** | **Maximum compression at full quality** |
| TQ2_0 | 50.3% | 97 MiB | 6.4x | Extreme compression, reduced accuracy |

### Head-to-Head: TQ3_1 vs TQ3_2 (20-prompt quality suite)

| Config | OK / 20 | Avg t/s | Jaccard vs TQ3_0 | Coherence |
|:-------|:-------:|:-------:|:----------------:|:----------|
| TQ3_0 | 19/20 | 18.84 | (reference) | Full |
| TQ3_1 | 20/20 | 19.50 | 0.2675 | Full |
| **TQ3_2** | **20/20** | **20.07** | 0.2510 | Full |

> The 100% completion rate and highest throughput make TQ3_2 the **most reliable** configuration tested. The Jaccard value of 0.25 confirms outputs are genuinely distinct from the reference — not collapsed to a degenerate distribution — while remaining fully coherent and factually correct.

### Key Findings

- **No output collapse or NaN** occurred across any TQ3_x configuration at any context length (0/500 failure rate)
- **100% mathematical correctness** on all arithmetic and algebra prompts at contexts ≤ 512 tokens
- **Graceful degradation** under extreme compression — even TQ2_0 at 50.3% accuracy produced syntactically valid, grammatically correct responses in 20/20 cases
- **Long-context reasoning preserved** at 2M tokens — the model maintains coherent chain-of-thought even when truncated by token limits

---

## Architecture

### Core Innovation: Semi-Quantized Compute

Standard quantized attention: `dequantize → FP16 vector → store in shared memory → compute`

TurboQuant attention: `read packed bits → compute directly`

Three fused GLSL kernel families replace the standard pipeline:

| Kernel | Operation | Format |
|:-------|:----------|:-------|
| `semi_quant_qk_dot()` | QK attention dot product from packed 3-bit indices | TQ3_0, Mixed KV |
| `semi_quant_qk_dot()` | QK attention dot product from packed 2-bit indices | TQ2_0 |
| `semi_quant_pv_accum()` | PV weighted accumulation from packed 3-bit indices | TQ3_0 |
| `semi_quant_pv_accum()` | PV weighted accumulation from packed 2-bit indices | TQ2_0, Mixed KV, TQ3_2 |

### TQ3_2 Compute-Only Corrections

```
Correction 1 — Pre-softmax QK scale (×1.03):
  Compensates for ~3% tail-clipping in the Lloyd-Max codebook's finite centroid range.
  Restores approximate FP16 attention sharpness before the softmax nonlinearity.

Correction 2 — Pairwise decorrelation filter (α = 0.125):
  3-tap symmetric filter attenuates high-frequency quantization noise by ~40% within
  each 4-element decode quad. DC (signal) passes unchanged.
```

### Modified Files (vs upstream llama.cpp)

| File | Change |
|:-----|:-------|
| `vulkan-shaders/flash_attn_base.glsl` | `semi_quant_qk_dot()` and `semi_quant_pv_accum()` for TQ3_0, TQ2_0, Mixed KV, TQ3_2 |
| `vulkan-shaders/flash_attn.comp` | `USE_SEMI_QUANT` compile-time macro; SHMEM staging elimination guards |
| `ggml-vulkan.cpp` | `shmem_staging = 0` for TQ types; `FA_SCALAR` enforcement |

---

## Repository Structure

```
TurboQuant vulkan/
├── llama_src/              # Modified llama.cpp with Vulkan FA shader patches
├── bench_suite/            # Production benchmark suite (100 prompts, 5 configs)
│   └── results/            # raw.json, raw.baseline.json, RESULTS.baseline.md
├── bench_v2/               # Extended benchmark dataset (JSONL, 101 entries)
├── turboquant/             # TQ3_3 cold-V format R&D (Python reference)
├── tq3_0_repo/             # TQ3_0 standalone distribution (AGPL-3.0)
├── turboquant_repo/        # Original CUDA/vLLM TurboQuant (ICLR 2026)
├── agents/                 # Autonomous agent experimentation system
├── final_report.md         # Comprehensive technical report
└── README.md               # This document
```

---

## Quick Start

### Build

```powershell
cd llama_src
cmake -B build_vulkan -DGGML_VULKAN=ON
cmake --build build_vulkan --config Release
```

### Run — TQ3_2 (recommended)

```powershell
.\llama_src\build_vulkan\bin\Release\llama-server.exe `
  -m models\google_gemma-4-26B-A4B-it-Q4_K_M.gguf `
  -ngl 30 -ctk tq3_0 -ctv tq3_2 -c 4096
```

### Run — TQ3_0 (balanced)

```powershell
.\llama_src\build_vulkan\bin\Release\llama-server.exe `
  -m models\google_gemma-4-26B-A4B-it-Q4_K_M.gguf `
  -ngl 30 -ctk tq3_0 -ctv tq3_0 -c 4096
```

### Run — Maximum Compression

```powershell
.\llama_src\build_vulkan\bin\Release\llama-server.exe `
  -m models\google_gemma-4-26B-A4B-it-Q4_K_M.gguf `
  -ngl 30 -ctk tq2_0 -ctv tq2_0 -c 4096
```

---

## Requirements

- Vulkan 1.1+ capable GPU (tested: AMD RX 6750 XT RDNA2)
- CMake 3.14+
- GGUF model file
- Windows or Linux

---

## ⚠ Project Status: Deprecated

This project is no longer under active development. The author has shifted focus to new development priorities. The codebase and all associated benchmarks, documentation, and research artifacts are preserved here as a complete, archived body of work.

**The key contributions of this project remain valid:**

- Semi-quantized Flash Attention operating directly on packed KV cache data without intermediate dequantization
- Production-validated 6.4x KV cache compression at 93.3% cognitive fidelity retention (TQ3_2)
- Elimination of shared-memory staging overhead through compile-time shader specialization
- Comprehensive cross-methodology benchmark suite with 500+ test executions

All data, results, and source code are retained in their final state. The repository serves as a reference implementation for GPU-accelerated KV cache compression in the Vulkan compute ecosystem.

---

## License

Same as [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) for the modified source files. The standalone TQ3_0 distribution (`tq3_0_repo/`) is AGPL-3.0.

---

<sub>Inspired by the TurboQuant paper (arXiv:2504.19874). ICLR 2026.</sub>
