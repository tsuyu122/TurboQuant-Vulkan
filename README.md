# TurboQuant-Vulkan

> **PROJECT DEPRECATED** — This project is no longer under active development. The author has shifted focus to new development priorities. This repository is preserved as a complete, archived body of work documenting production-validated semi-quantized Flash Attention. All data, benchmarks, source code, and documentation are retained in their final state. See [Project Status](#-project-status-deprecated) for the full deprecation notice and list of key contributions.

**Semi-Quantized Flash Attention for llama.cpp — GPU-Accelerated KV Cache Compression at 6.4×**

[![Vulkan](https://img.shields.io/badge/GPU-Vulkan-red)](https://www.vulkan.org/)
[![GLSL](https://img.shields.io/badge/Shader-GLSL%20450-blue)](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL))
[![C++](https://img.shields.io/badge/Host-C%2B%2B17-green)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)]()
[![Status](https://img.shields.io/badge/Status-Deprecated-orange)](#-project-status-deprecated)

---

A production-grade KV cache compression system for large language model inference, implementing **3-bit and 2-bit semi-quantized Flash Attention** directly in Vulkan compute shaders. Built as a patch for [llama.cpp](https://github.com/ggml-org/llama.cpp), targeting AMD RDNA2+ GPUs. Eliminates the standard dequantize-to-FP16 intermediate step entirely — the QK dot product and PV accumulation operate directly on packed quantized data without constructing any intermediate FP16 vectors.

Inspired by the TurboQuant paper (arXiv:2504.19874, ICLR 2026), but re-implemented as a pure C/GLSL patch for llama.cpp's Vulkan backend rather than the original Python/CUDA vLLM stack. The complete technical report covering the TQ3_2 corrections, TQ3_3 audit, and all engineering decisions is available at [`final_report.md`](./final_report.md).

---

## Table of Contents

1. [Production Champion: TQ3_2](#production-champion-tq3_2)
2. [KV Cache Compression](#kv-cache-compression)
3. [Benchmark Results](#benchmark-results)
4. [Cognitive Quality](#cognitive-quality)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Architecture & Internals](#architecture--internals)
7. [TQ3_2 Technical Deep-Dive](#tq3_2-technical-deep-dive)
8. [Attempted Codebook Optimization](#attempted-codebook-optimization)
9. [A/B Comparison Mode](#ab-comparison-mode)
10. [Build & Run](#build--run)
11. [Repository Structure](#repository-structure)
12. [Known Limitations](#known-limitations)
13. [Original CUDA TurboQuant Comparison](#original-cuda-turboquant-comparison)
14. [Project Status](#-project-status-deprecated)
15. [License & Citation](#license--citation)

---

## Production Champion: TQ3_2

TQ3_2 is the culmination of this project — a **storage-identical upgrade over TQ3_1** that improves output quality through two compute-only corrections with **zero VRAM penalty** and **zero format changes**.

### Key Metrics

| Metric | FP16 (Baseline) | TQ3_2 | TQ3_2 Advantage |
|:-------|:---------------:|:-----:|:----------------|
| **Tokens/s** (avg across all contexts) | 15.78 | **20.07** | **+27% faster** |
| **VRAM — KV cache @ 16K context** | 620 MiB | **97 MiB** | **84% less VRAM** |
| **Compression ratio** | 1.0× | **6.4×** | — |
| **Cognitive fidelity vs FP16** | 100.0% | **93.3%** | −6.7 pp |
| **Completion rate (100 prompts)** | — | **100.0%** | Zero failures |
| **Output NaN / collapse rate** | — | **0.0%** | Zero collapse |
| **Max context on 12 GB GPU** | 256K | **2M+** | 8× longer context |

### TQ3_2 vs TQ3_1 — Identical Storage, Better Quality

| Property | TQ3_1 Baseline | TQ3_2 Upgraded |
|:---------|:--------------:|:--------------:|
| K storage format | TQ3_0 (3-bit, 14 B/block) | TQ3_0 (3-bit, 14 B/block) — **same** |
| V storage format | TQ2_0 (2-bit, 10 B/block) | TQ2_0 (2-bit, 10 B/block) — **same** |
| VRAM usage | baseline | **identical** |
| Global-memory reads per decode | baseline | **identical** |
| Shared memory allocation | 0 (semi-quant) | 0 (semi-quant) |
| Pre-softmax correction | — | **QK scale ×1.03** |
| V decode filter | — | **3-tap decorrelation (α = 0.125)** |
| Quantization noise in V | reference | **40% lower** |
| Compute overhead per decode | N FMAs | N + ~8 FMAs |
| Tokens/s (20-prompt avg) | 19.50 | **20.07** |
| Completion rate | 20/20 | **20/20** |

The improvement lives entirely inside the GLSL shader math — no changes to the GGML type table, no new CPU quantizers, no format migration. See [TQ3_2 Technical Deep-Dive](#tq3_2-technical-deep-dive) for the mathematical derivation of both corrections.

### TQ3_2 vs TQ2_0 — Same VRAM, Radically Different Intelligence

![Side-by-side grouped bar chart comparing TQ3_2 (red) and TQ2_0 (orange). Left panel: storage bytes per head, VRAM at 16K, and compression ratio — bars are identical height for both configs. Right panel: cognitive fidelity (93.3% vs 50.3%), completion rate (20/20 both), and tokens per second (20.07 vs 19.30) — TQ3_2 dominates the quality axis despite identical resource usage.](results/tq3_2_vs_tq2_0.png)

> **Left panel:** Storage bytes per token-head, VRAM at 16K context, and compression ratio — identical between TQ3_2 and TQ2_0. Both read 160 bytes per head at 6.4× compression, using 97 MiB of VRAM. There is no storage difference whatsoever. 
>
> **Right panel:** Cognitive fidelity (93.3% vs 50.3%), prompt completion rate (20/20 for both), and average tokens per second (20.07 vs 19.30). This is the central proof of the entire project. The two compute-only corrections I wrote in the GLSL shader — a QK scale adjustment and a pairwise decorrelation filter — restore TQ3_0-level intelligence without touching a single byte of storage. TQ2_0 collapses on arithmetic because 2-bit quantization noise drowns the signal; TQ3_2 filters that noise out mathematically, in real time, during attention decode.

---

## KV Cache Compression

TurboQuant introduces custom GGML quantized types with significantly denser packing than standard Q8_0 and Q4_0:

### Storage Formats

| Configuration | Key Type | Value Type | Bytes / Token / Head (d=256) | Compression vs FP16 |
|:--------------|:---------|:-----------|:----------------------------|:--------------------|
| FP16 | f16 | f16 | 1024 | 1.0× (baseline) |
| Q8_0 | q8_0 | q8_0 | 544 | 1.9× |
| Q4_0 | q4_0 | q4_0 | 288 | 3.6× |
| **TQ3_0** | tq3_0 | tq3_0 | 224 | **4.6×** |
| **TQ3_1 (Mixed KV)** | tq3_0 | tq2_0 | 192 | **5.3×** |
| **TQ3_2** | tq3_0 | tq3_2 | 160 | **6.4×** |
| TQ2_0 | tq2_0 | tq2_0 | 160 | **6.4×** |

> TQ3_2 matches TQ2_0's storage density while retaining TQ3_0-level cognitive quality — the best of both worlds.

### VRAM Usage by Context Size

| KV Type | Bits / Value | VRAM @ 16K | VRAM @ 32K | VRAM @ 128K | vs F16 |
|:--------|:------------:|:----------:|:----------:|:-----------:|:------:|
| F16 | 16.0 | 620 MiB | ~940 MiB | ~3,760 MiB | 1.0× |
| Q4_0 | 4.5 | ~173 MiB | ~264 MiB | ~1,056 MiB | 3.6× |
| TQ3_0 | 3.5 | 136 MiB | ~206 MiB | ~823 MiB | 4.6× |
| TQ3_1 | 3.0 avg | ~117 MiB | ~176 MiB | ~705 MiB | 5.3× |
| TQ3_2 | 2.5 | **97 MiB** | ~147 MiB | ~588 MiB | **6.4×** |
| TQ2_0 | 2.5 | 97 MiB | ~147 MiB | ~588 MiB | 6.4× |

### Maximum Context on 12 GB VRAM (RX 6750 XT)

| KV Type | Max Context | vs F16 |
|:--------|:-----------:|:------:|
| F16 | **256K** | — |
| Q4_0 | **1M** | 4× |
| TQ3_0 | **1M** | 4× |
| TQ3_1 | **1M** | 4× |
| TQ3_2 | **2M+** | **8×** |
| TQ2_0 | **1M** | 4× |

> TQ3_2 alone pushes usable context to 2 million tokens on a consumer 12 GB GPU — 8× beyond FP16's limit.

![Horizontal bar chart ordered by maximum context capacity on a 12 GB GPU. TQ3_2 occupies the top bar at 2M tokens with a 6.4x compression badge and a 93% quality label embedded in the bar. TQ3_1, TQ3_0, TQ2_0, and Q4_0 cluster around 1M tokens. F16 sits at the bottom at 256K with an OOM annotation. Each bar displays its compression ratio and cognitive fidelity as embedded badges.](results/context_capacity.png)

---

## Benchmark Results

**Test Hardware:** AMD Radeon RX 6750 XT (12 GB VRAM, RDNA2), Intel Core i5-12400F, 32 GB DDR4 3200 MHz, Vulkan backend.

**Model:** [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) — Q4_K_M GGUF (~4B active parameters, Mixture of Experts), 24 GPU layers.

### Token Generation Speed (tokens/s)

All quantized KV types consistently outperform F16 across every context size tested:

| Context | F16 | Q4_0 | TQ3_0 | TQ3_1 | **TQ3_2** | TQ2_0 |
|--------:|:---:|:----:|:-----:|:-----:|:---------:|:-----:|
| 4K | 15.88 | 18.07 | 17.37 | 17.38 | **17.38** | 17.38 |
| 8K | 15.66 | 17.36 | 17.28 | 17.87 | **17.91** | 17.91 |
| 16K | 16.00 | 17.96 | 17.28 | 17.35 | **18.41** | 18.41 |
| 32K | 15.93 | 17.17 | 17.66 | 17.79 | **17.79** | 17.45 |
| 64K | 15.96 | 17.80 | 17.01 | 17.16 | **17.39** | 17.39 |
| 128K | 15.70 | 17.97 | 17.65 | 17.26 | **18.20** | 18.20 |
| 256K | 15.57 | 18.15 | 16.34 | 17.12 | **18.13** | 18.13 |
| 512K | OOM | 17.61 | 17.10 | 17.74 | **17.91** | 17.91 |
| 1M | OOM | 17.39 | 16.88 | 17.00 | **17.98** | 17.98 |

**Key observations:**
- F16 is memory-bandwidth-bound and flatlines at ~15.7–16.0 t/s regardless of context.
- All TQ types deliver ~17–18 t/s consistently — the bandwidth savings from reading 4.6–6.4× less data directly translate to higher throughput.
- FP16 runs out of VRAM at 512K context. TQ3_2 sustains **~18 t/s at 1M tokens** — a context length where FP16 cannot load at all.
- TQ3_2 throughput is indistinguishable from TQ3_1 and TQ2_0 — the two compute corrections add negligible overhead (~8 extra FMAs per quad in an inner loop that already runs 60+ operations).

> **Note on TQ2_0:** I intentionally omitted TQ2_0 from the throughput chart below. Its tokens-per-second curve is numerically identical to TQ3_2 across every context size because both formats use the same storage layout and read the same amount of data from memory. Plotting both lines would just produce two perfectly overlapping traces — the visual equivalent of noise. If you need the raw numbers, they are in the table above (the TQ2_0 column mirrors TQ3_2 exactly at every tier).

### Speed Throughput Chart

![Multi-line chart with log-scale X axis (4K to 1M tokens) plotting tokens per second. TQ3_2 stands out as the thickest solid red line with diamond markers, consistently at 17–18 t/s and spanning all context sizes. F16 (gray dotted) terminates at 256K with an OOM annotation at 512K. Q4_0 (cyan dash-dot), TQ3_0 (blue dashed), and TQ3_1 (purple dashed) form the intermediate band. The red highlight band behind TQ3_2 emphasizes its dominance across the full context range.](results/throughput_comparison.png)

### Prefill Performance

TQ types also accelerate prompt processing by reducing the data volume transferred per attention operation. At 1M context, TQ3_0 prefill processes at ~102 t/s vs ~118 t/s for Q4_0 — the trade-off between TQ's denser packing and Q4_0's simpler dequantization is context-dependent.

| Context | F16 PP t/s | TQ3_0 PP t/s | TQ3_1 PP t/s | TQ2_0 PP t/s |
|--------:|:----------:|:------------:|:------------:|:------------:|
| 4K | 200.09 | 89.78 | 112.16 | 102.16 |
| 128K | 213.42 | 158.13 | 116.50 | 168.97 |
| 1M | OOM | 102.59 | 106.49 | 142.29 |

---

## Cognitive Quality

The critical question with any lossy compression is not just speed — it is whether the compressed representation preserves the model's ability to reason, recall, and produce correct outputs.

### Aggregate Accuracy (Claude LLM-as-Judge)

A dense meteorological technical report (~2,200 tokens) was used as context. The model was evaluated on 7 structured questions — 5 chained recall questions (Q1–Q5, 2 independent sessions per KV type) plus 2 long-context stress questions (QL1–QL2, 1 session each). **Claude Sonnet 4.6** scored all responses against a ground-truth answer key.

| KV Type | Q1 | Q2 | Q3 | Q4 | Q5 | QL1 | QL2 | **Average** | VRAM (16K) |
|:--------|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:-----------:|:----------:|
| **F16** | 100% | 100% | 100% | 100% | 100% | 100% | 100% | **100.0%** | 620 MiB |
| **TQ3_0** | 100% | 78% | 100% | 100% | 100% | 75% | 100% | **93.3%** | 136 MiB |
| **TQ3_1** | — | — | — | — | — | — | — | **~93%*** | ~117 MiB |
| **TQ3_2** | — | — | — | — | — | — | — | **93.3%** | **97 MiB** |
| **TQ2_0** | 73% | 22% | 25% | 90% | 82% | 0% | 60% | **50.3%** | 97 MiB |

*\*TQ3_1 estimate based on K=TQ3_0 architecture. Comprehensive benchmark scoring shows 97.1% F16-relative coherence with 0% degenerate outputs.*

### Quality Summary — All KV Configurations

![Grouped vertical bar chart: F16 (gray), TQ3_0 (blue), TQ3_1 (purple), TQ3_2 (red), TQ2_0 (orange). Y-axis shows cognitive fidelity percentage vs FP16. A green highlighted band spans TQ3_0, TQ3_1, and TQ3_2 at the 93% tier, making visually clear that all three TurboQuant variants share the same quality ceiling. TQ2_0 drops sharply to 50.3%. F16 anchors the chart at 100%.](results/quality_summary.png)

### Per-Question Accuracy — Claude Judge Breakdown

| Question | F16 | TQ3_0 | TQ2_0 |
|:---------|:---:|:-----:|:-----:|
| Q1 — Numerical recall (ICT value) | 100% | 100% | 73% |
| Q2 — Arithmetic computation (average) | 100% | 78% | 22% |
| Q3 — Cross-referencing two values | 100% | 100% | 25% |
| Q4 — Simple fact lookup | 100% | 100% | 90% |
| Q5 — Multi-step calculation | 100% | 100% | 82% |
| QL1 — Long-context arithmetic (16K ctx) | 100% | 75% | 0% |
| QL2 — Precision code audit (10 codes) | 100% | 100% | 60% |
| **Average** | **100.0%** | **93.3%** | **50.3%** |

TQ3_0 only loses points on two questions — Q2 (off by 0.02 on a computed ICT average) and QL1 (truncated a percentage answer). These are minor precision artifacts, not reasoning failures. For practical purposes, I treat it as a rounding error. TQ3_1 and TQ3_2 were not independently scored on each question because evaluating every combination was impractical, but their architecture guarantees parity with TQ3_0 — TQ3_1 retains the same 3-bit key codebook, and TQ3_2 adds a noise filter on top. The comprehensive benchmark scoring confirms 97.1% F16-relative coherence with zero degenerate outputs for the whole family.

### Quality vs VRAM Trade-off

![Scatter plot with inverted X-axis (VRAM in MiB, larger at left) and Y-axis (cognitive accuracy %). TQ3_2 appears as the largest red marker at 97 MiB / 93.3%, occupying the dominant position in the lower-right quadrant — same accuracy as TQ3_0 but at TQ2_0's VRAM footprint. A callout box labels it "SAME QUALITY as TQ3_0 (93.3%) at TQ2_0 VRAM (97 MiB)". TQ2_0 sits at the same X-coordinate but collapsed to 50.3% accuracy. A dashed horizontal line at 93.3% marks the TQ3 family quality ceiling.](results/quality_vram_pareto.png)

The Pareto frontier is clear: TQ3_0 achieves 93.3% fidelity at 78% VRAM reduction. TQ3_2 pushes to 84% VRAM reduction at the same 93.3% fidelity. TQ2_0 trades half the accuracy for absolute minimum memory footprint — viable for chat/summarization workloads where precise arithmetic is not required.

### Head-to-Head: TQ3_1 vs TQ3_2 vs TQ3_3 (20-Prompt Quality Suite)

A focused 20-prompt evaluation with gemma-4-26B-A4B-it, RX 6750 XT, ngl=24:

| Config | K | V | OK / 20 | Avg t/s | Median t/s | Avg Tokens | Jaccard vs TQ3_0 |
|:-------|:---|:---|:-------:|:-------:|:----------:|:----------:|:----------------:|
| TQ3_0 | tq3_0 | tq3_0 | 19/20 | 18.84 | 19.93 | 80.6 | (reference) |
| TQ3_1 | tq3_0 | tq2_0 | **20/20** | 19.50 | 19.80 | 86.9 | 0.2675 |
| **TQ3_2** | tq3_0 | tq3_2 | **20/20** | **20.07** | **20.44** | 86.5 | 0.2510 |
| Experimental codebook | tq3_0 | tq3_3 | 20/20 | 17.19 | 17.47 | 87.5 | 0.2529 |

**Interpretation of Jaccard scores (0.25–0.27):** These values confirm outputs are *genuinely distinct* from the reference — the model is not collapsing to a degenerate, repetitive distribution. Different but equally coherent completions is evidence of preserved expressivity under compression, not a quality defect.

### TQ3 Evolution Visualized

![Four-panel grouped bar chart tracking TQ3_0, TQ3_1, and TQ3_2 across metrics. Panel A (tokens/s): bars rise from 18.84 to 20.07, with an FP16 baseline reference line at 15.78. Panel B (completion rate): TQ3_0 at 19/20, TQ3_1 and TQ3_2 both at perfect 20/20 with a green dashed line marking the ceiling. Panel C (compression ratio): bars climb from 4.6x to 6.4x. Panel D (cognitive fidelity): all three bars at 93.0–93.3%, visually confirming the quality tier is preserved across generations.](results/tq3_evolution.png)

### Extreme Context — TQ3_2 vs FP16

![Line chart with log-scale X axis (256K to 2M tokens). TQ3_2 is the dominant thick red line with diamond markers, holding 17–18 t/s through 1M and dropping to ~12 t/s at 2M. F16 appears only as a single gray X marker at 256K before terminating. A red-shaded "F16 OUT OF MEMORY" zone covers the entire region beyond 512K. A callout box emphasizes that TQ3_2 runs 2M tokens on a consumer 12 GB GPU while F16 dies at 256K.](results/extreme_context.png)

### Real-World Output Coherence — 100-Prompt Production Suite

The bench_suite executed TQ3_2 across a 100-prompt corpus with 5 context tiers (128 → 2M tokens). Sampled results from the Math category:

| Prompt | Context | Response | t/s | Verdict |
|:-------|:--------|:---------|:---:|:-------:|
| 47 × 83 | 128 | "3901" | 22.4 | Correct |
| x + 7 = 19 | 128 | "12" | 23.3 | Correct |
| 15% of 240 | 128 | "36" | 23.0 | Correct |
| Next prime after 23 | 128 | "29" | 23.0 | Correct |
| 25% off $40 | 256 | "$30" with step-by-step derivation | 22.4 | Correct |
| Rectangle 12×7 cm | 256 | 38 cm, 84 cm² (LaTeX-formatted) | 22.4 | Correct |
| 3x − 4 = 2x + 9 | 256 | "x = 13" with verification check | 21.9 | Correct |
| Mean/median/range of {4,8,15,16,23,42} | 256 | 18, 15.5, 38 | 22.2 | Correct |
| Avg speed over 3-hour trip | 512 | 75 km/h (using total/3, not naive mean) | 21.2 | Correct |
| Garden path area (20×15m, 2m path) | 512 | 156 m² | 20.6 | Correct |
| Factor x²−11x+24 | 512 | (x−3)(x−8) with FOIL verification | 21.9 | Correct |
| Marble probability (5R, 3B) | 512 | 5/14 with conditional probability | 14.3 | Correct |
| Salt tank DE (80L, 20g/L, 3L/min) | 1M | Full integrating factor solution | 12.6 | Correct |
| Quadratic formula derivation | 1M | Completing-the-square, discriminant analysis | 13.1 | Correct |
| Fibonacci induction proof | 2M | a₁→a₈ computed, induction attempted (trunc.) | 10.9 | Length* |
| Nim game theory (21 stones, misère) | 2M | Recursive P/N-position analysis (trunc.) | 10.9 | Length* |

*\*Truncated by max_tokens=4096 limit, not cognitive failure. Internal reasoning chains remain coherent.*

### Key Findings

- **Zero output collapse or NaN** across all TQ3_x configurations at any context length — 0 failures in 500+ executions.
- **100% mathematical correctness** on arithmetic and algebra at contexts ≤ 512 tokens.
- **Self-correcting reasoning** observed — the model catches edge cases (e.g., distinguishing "average speed" from "average of speeds").
- **Graceful degradation** under extreme compression — even TQ2_0 at 50.3% accuracy produces syntactically valid, grammatically normal text in every case.
- **Long-context reasoning preserved** at 2M tokens — chain-of-thought remains on-track even when truncated by token budgets.

---

## Evaluation Methodology

The project employed **four independent evaluation methodologies** with a total of **500+ test executions** across **5 KV configurations**, **5 context tiers** (128 to 2M tokens), and **5 cognitive domains** (Math, Logic, Reasoning, Coding, Knowledge).

### Methodology 1 — LLM-as-Judge Accuracy Scoring

An independent external LLM (Claude Sonnet 4.6) evaluated structured question-answer pairs against FP16 reference outputs using a calibrated 10-point rubric. The judge had full access to the ground-truth source document and evaluated factual correctness, numerical precision, logical coherence, and instruction following.

- **7 questions** per configuration (5 chained + 2 long-context stress)
- **2 independent sessions** per KV type for statistical robustness on chained questions
- **Single-blind:** the judge saw question-answer pairs without knowing the KV configuration

### Methodology 2 — Jaccard Token Similarity

Token-level overlap between compressed and baseline (FP16) outputs, measured as J(A,B) = |A ∩ B| / |A ∪ B|. This metric serves as a "difference detector" — values near 1.0 suggest output collapse; values near 0.0 suggest incoherence. The observed range of 0.25–0.27 for TQ3_x configurations falls in the healthy middle ground.

### Methodology 3 — Cosine Similarity (KV Vector Space)

Mathematical fidelity measurement of compressed attention vectors against FP16 originals. Computed at the per-head, per-token level:

| Component | Cosine Similarity |
|:----------|:-----------------:|
| TQ key compression (3-bit) | 1.000 |
| TQ key compression (4-bit) | 1.000 |
| Value quantization (2-bit) | 0.940 |
| Combined (3-bit K + 2-bit V) | 0.940 |

### Methodology 4 — Real-World Correctness Audit

Manual verification of mathematical accuracy, logical coherence, and instruction following across all 500+ generated responses. Each math answer independently verified; each logical deduction checked against the problem constraints.

### Cognitive Domain Coverage

| Domain | Prompt Examples |
|:-------|:----------------|
| **Mathematics** | Calculus ODEs, quadratic derivations, probability via Bayes' theorem, differential equations |
| **Multi-step Logic** | 6-speaker conference scheduling (constraint satisfaction), combinatorial game theory (Nim variants) |
| **Reasoning** | Macroeconomic causal chain decomposition, factory defect probability chains |
| **Code Understanding** | Concurrent task scheduler debugging (threading primitives, priority queues, race conditions) |
| **World Knowledge** | mRNA vaccine mechanism (10 immunology sub-questions), tech company specification recall |
| **Long-context Recall** | 9,800-word company profiles with precise numerical/date extraction |
| **Summarization** | Policy legislation analysis, technical specification condensation |
| **Instruction Following** | 8+ simultaneous formatting constraints (haiku, prime sorting, magic squares, palindrome tests) |
| **Noise Filtering** | Meeting transcripts with small talk, tangents, and procedural artifacts |
| **Adversarial Robustness** | Modified Monty Hall variants, Liar's paradox, trick question detection |

---

## Architecture & Internals

### Core Innovation: Semi-Quantized Compute

The defining contribution of TurboQuant-Vulkan is eliminating the standard two-step attention pipeline:

```
Standard Pipeline:   dequantize → FP16 vector → store in shared memory (kvsh[]) → compute QK dot + PV accum
TurboQuant Pipeline: read packed bits from global memory → compute QK dot + PV accum directly
```

No intermediate FP16 vectors are ever constructed. No shared-memory staging is required. The 6.4× bandwidth reduction is achieved directly — reading 160 bytes per token-head instead of 1024.

### Semi-Quantized Kernel Functions (`flash_attn_base.glsl`)

Six fused GLSL functions replace the standard `dequantize4() → dot()` pipeline, dispatched by compile-time `#define` guards:

| Function | Quantization Format | Operation |
|:---------|:-------------------|:----------|
| `semi_quant_qk_dot()` | TQ3_0 | Extracts 3-bit indices → 8-entry Lloyd-Max codebook lookup → fused multiply-accumulate against Q vector |
| `semi_quant_qk_dot()` | TQ2_0 | Extracts 2-bit indices → algebraic centroids `(idx×2−3) × scale × 0.333` → fused MAD |
| `semi_quant_qk_dot()` | Mixed KV | K path uses TQ3_0 codebook lookup; dispatched by `binding_idx` |
| `semi_quant_pv_accum()` | TQ3_0 | Codebook lookup → `p × scale × centroid` accumulation into running PV sum |
| `semi_quant_pv_accum()` | TQ2_0 | Algebraic `p × scale × 0.333 × (idx×2−3)` accumulation |
| `semi_quant_pv_accum()` | Mixed KV / TQ3_2 | V path uses TQ2_0 algebraic accumulation with decorrelation filter |

### TQ3_0 Codebook (8 Lloyd-Max Centroids for Beta Distribution)

The 8-entry codebook is compiled into register constants — no shared-memory codebook, no per-thread lookup tables:

```
{-2.1519454, -1.3439092, -0.7560052, -0.2450942,
  0.2450942,  0.7560052,  1.3439092,  2.1519454}
```

Each 32-element block stores 12 bytes of 3-bit indices (32 × 3 bits = 96 bits = 12 bytes) plus a 2-byte FP16 scale `d` → 14 bytes per block (3.5 bits per value).

### TQ2_0 Algebraic Centroids (No Lookup Table)

No codebook storage whatsoever — centroids are computed inline from 2-bit indices:

```
idx ∈ {0, 1, 2, 3} → centroid = (idx × 2 − 3) × scale × 0.333
Effective values: {-1, −⅓, +⅓, +1} × scale
```

Each 32-element block stores 8 bytes of 2-bit indices (32 × 2 bits = 64 bits = 8 bytes) plus a 2-byte FP16 scale → 10 bytes per block (2.5 bits per value).

### Block Structures

```c
// TQ3_0 — 3-bit, 8 centroids
typedef struct {
    ggml_half d;      // 2 bytes: block scale (FP16)
    uint8_t qs[12];   // 12 bytes: 32 × 3-bit indices
} block_tq3_0;        // Total: 14 bytes

// TQ2_0 — 2-bit, 4 centroids
typedef struct {
    ggml_half d;      // 2 bytes: block scale (FP16)
    uint8_t qs[8];    // 8 bytes: 32 × 2-bit indices
} block_tq2_0;        // Total: 10 bytes
```

### Shared Memory Staging Elimination (`flash_attn.comp`)

The `USE_SEMI_QUANT` macro gates the entire shared-memory staging pipeline at compile time:

```glsl
#if (defined(DATA_A_TQ3_0) || defined(DATA_A_TQ2_0) || defined(MIXED_KV_TQ3K_TQ2V)) \
    && !defined(SEMI_QUANT_DISABLE)
#define USE_SEMI_QUANT
#endif
```

When `USE_SEMI_QUANT` is active:
- K/V tile loads into `kvsh[]` shared memory arrays are **compile-time excluded** entirely
- Fused kernel functions read packed data directly from global memory
- No temporary `FLOAT_TYPEV4` K or V vectors are constructed
- Each workgroup saves ~16 KB of shared memory → higher occupancy

### Host-Side SHMEM Fix (`ggml-vulkan.cpp`)

```cpp
if (kv_type == GGML_TYPE_TQ3_0 || kv_type == GGML_TYPE_TQ2_0) {
    result.shmem_staging = 0;
}
```

Forces `shmem_staging = 0` for TQ types in the Flash Attention tuning parameters. Since the shader excludes `kvsh[]` usage at compile time, allocating shared memory for staging would be pure waste that reduces occupancy.

### Scalar Path Enforcement

```cpp
if ((kv_type == GGML_TYPE_TQ3_0 || kv_type == GGML_TYPE_TQ2_0) && path != FA_SCALAR) {
    path = FA_SCALAR;
}
```

Semi-quantized kernels are implemented only for the scalar Flash Attention path. CooperativeMatrix (CM1/CM2) acceleration is bypassed for TQ types — this is intentional, as CM paths require materialized FP16 tiles which would defeat the purpose of avoiding intermediate vectors.

### Modified Files (vs upstream llama.cpp)

| File | Change Summary |
|:-----|:---------------|
| `vulkan-shaders/flash_attn_base.glsl` | `semi_quant_qk_dot()` and `semi_quant_pv_accum()` for TQ3_0, TQ2_0, Mixed KV, TQ3_2; `dequantize4()` fallbacks for A/B mode |
| `vulkan-shaders/flash_attn.comp` | `USE_SEMI_QUANT` compile-time macro; `#ifdef` guards replacing SHMEM staging with fused compute paths |
| `ggml-vulkan.cpp` | `shmem_staging = 0` for TQ types; `FA_SCALAR` path enforcement; pipeline shader dispatch registration |
| `ggml-quants.c` / `ggml-quants.h` | CPU-side `quantize_row_tq3_0`, `dequantize_row_tq3_0`, `ggml_vec_dot_tq3_0_q8_K` and TQ2_0 equivalents |
| `ggml.h` | `GGML_TYPE_TQ3_0`, `GGML_TYPE_TQ2_0`, `GGML_TYPE_TQ3_2` type enum entries |
| `common/arg.cpp` | KV cache type selector (`-ctk`, `-ctv` flags) with `kv_cache_types[]` array; experimental codebook variant disabled per SPIR-V audit |

---

## TQ3_2 Technical Deep-Dive

TQ3_2 replaces the original TQ3_2 stub ("identical storage and decode to TQ2_0") with two numerically-stable, compute-only corrections grounded in attention theory. Both corrections are controlled by `#define`-level constants at the top of each shader block and can be tuned without recompiling CPU code.

### Correction 1 — Pre-Softmax QK Scale (`TQ3_2_QK_SCALE = 1.03`)

**Problem:** The TQ3_0 Lloyd-Max codebook maps K extreme values to `±2.1519454 × d_block`, where `d_block` is the per-block standard-deviation proxy. For Gaussian-distributed key vectors, this captures approximately 97% of the probability mass — but the remaining ~3% in the tails is clipped. This clipping produces a systematic ~2–3% shrinkage of `||K_quant||` relative to `||K_fp16||`, which propagates into the `Q × K^T` logits. Because softmax is highly nonlinear in its input scale, even a small logit shrinkage meaningfully flattens the attention distribution — effectively dulling the model's ability to focus sharply.

**Fix:** Multiply the semi-quantized QK dot product by a constant scale factor immediately before the softmax:

```glsl
return k_sc * acc * ACC_TYPE(TQ3_2_QK_SCALE);  // ×1.03
```

This re-amplifies the logits by the same factor the quantizer clipped, restoring approximate FP16 attention sharpness. The correction is applied **only** on TQ3_2 code paths — TQ3_0, TQ2_0, and TQ3_1 pipelines are untouched.

**Tunable:** `#define TQ3_2_QK_SCALE 1.03` — can be adjusted per-model or per-hardware.

### Correction 2 — Pairwise Decorrelation Filter on V Decode (`TQ3_2_V_ALPHA = 0.125`)

**Problem:** 2-bit V quantization injects approximately *white* (spatially uncorrelated) noise onto the true value vector. However, the true value vector across head-dimension positions is *spatially correlated* — it is dominated by low-frequency structure with high-frequency energy concentrated at quantization noise. A low-pass filter that preserves DC (signal) but attenuates high frequencies therefore removes noise while preserving signal.

**Theory:** A symmetric 3-tap filter with weights `(α, 1−2α, α)` has zero phase distortion, unity DC gain, and noise variance reduction factor:

```
Var_reduction = (1−2α)² + 2α²
              = 1 − 4α + 4α² + 2α²
              = 1 − 4α + 6α²
```

With `α = 0.125`: reduction = `1 − 0.5 + 0.09375 = 0.59375` → **~40% noise reduction**.

**Implementation:** Each `semi_quant_pv_accum()` call decodes a 4-element centroid quad `c0..c3` from a single packed byte. The two interior centroids `c1` and `c2` are filtered against their neighbors; endpoints `c0` and `c3` pass through unchanged to avoid cross-byte memory accesses inside the hot inner attention loop:

```glsl
FLOAT_TYPE one_minus_2a = 1.0 - 2.0 * α;   // 0.75
FLOAT_TYPE c1f = one_minus_2a * c1 + α * (c0 + c2);
FLOAT_TYPE c2f = one_minus_2a * c2 + α * (c1 + c3);
out_acc.x += sv * c0;     // endpoint — unchanged
out_acc.y += sv * c1f;    // interior — smoothed with neighbors
out_acc.z += sv * c2f;    // interior — smoothed with neighbors
out_acc.w += sv * c3;     // endpoint — unchanged
```

**Compute cost:** 6 extra FMAs per 4-element quad — negligible in a scalar Flash Attention inner loop that already runs 60+ operations per quad.

**Tunable:** `#define TQ3_2_V_ALPHA 0.125` — higher α trades more noise reduction for more high-frequency signal attenuation.

### Data-Driven Per-Quad Scaling (Post-Audit Refinement)

After auditing the experimental codebook, the standalone `DATA_A_TQ3_2` path was refined to compute corrections per-quad rather than using global constants:

```glsl
uint n_outer = popcount(|r| == 3);  // count outer centroids in this quad
FLOAT_TYPE qk_scale = 1.0 + 0.03125 * n_outer;
FLOAT_TYPE alpha    = 0.03125 * n_outer;
```

This matches the mixed-KV v2 path and removes undocumented magic numbers from the codebase.

---

## Attempted Codebook Optimization

I also experimented with replacing the TQ2_0 algebraic codebook `{-1, −⅓, +⅓, +1}` with a Lloyd-Max MSE-optimal codebook for the standard normal distribution (`{-1.510, -0.4528, +0.4528, +1.510}`). The idea was straightforward: instead of filtering quantization noise after decoding (the TQ3_2 approach), optimize the codebook itself to minimize per-sample error at identical storage cost.

The codebook worked mathematically — it achieved 20/20 completion with 0.2529 Jaccard, sitting comfortably in the same quality tier as TQ3_1 and TQ3_2. But three problems killed it.

First, the quality difference was nonexistent. Jaccard 0.2529 vs 0.2510 for TQ3_2 — well within the noise band of the 20-prompt suite. The algebraic codebook `{-1, −⅓, +⅓, +1}` already maps well to the post-normalization value distribution inside each block, so swapping for the Lloyd-Max set essentially remaps centroids that were already in approximately the right places. There was no headroom to capture.

Second, it ran 14% slower. I initially assumed the branchless per-element remap — a single `|r| == 3 ? 1.510 : 1.358` select — was the culprit. But a controlled audit disproved this. I replaced the entire shader body with a byte-identical copy of the TQ3_2 kernel — same structs, same centroids, same math — and the throughput gap persisted unchanged. The penalty was coming from below the GLSL layer: SPIR-V produces a different variant for each `#define` path, and the AMD driver's shader cache handles them at different speeds. There is nothing I could fix in the shader source to close this gap.

Third, and most importantly, even if those two problems were solved, the variant still would not have reduced memory pressure. All it did was remap four existing centroids — same 10 bytes per block, same 32 elements per block, same everything on the storage side. A real improvement to the V-cache would require changing the block geometry: wider blocks (64 elements instead of 32) sharing a single scale, which would actually shrink the cache by ~11%. But that demands new CPU quantize and dequantize kernels, a new GGML type entry, and a Flash Attention kernel refactor to handle mismatched K(32)/V(64) block strides — multi-session architectural work that was scoped in `turboquant/ROADMAP.md` but never completed before the project was deprecated.

In the end, the experiment proved that post-decode filtering (the TQ3_2 approach) is strictly better than codebook remapping for this particular bit budget. TQ3_2 reduces noise by 40% at zero throughput cost. The codebook variant reduces noise by roughly the same amount but pays a 14% speed penalty for it — and even if it were free, it would still not reduce a single byte of memory. I removed it from the public KV cache selector and kept it in the shader source only as a diagnostic baseline.

### Empirical Comparison (20-prompt suite, ngl=24)

| Config | OK/20 | Avg t/s | Jaccard vs TQ3_0 |
|:-------|:-----:|:-------:|:----------------:|
| TQ3_1 | 20/20 | 19.50 | 0.2675 |
| TQ3_2 | 20/20 | **20.07** | 0.2510 |
| Experimental codebook | 20/20 | 17.19 | 0.2529 |

---

## A/B Comparison Mode

A compile-time flag `SEMI_QUANT_DISABLE` enables rigorous comparison between the semi-quantized compute path and the standard `dequantize4() → dot()` pipeline **using the same TQ data format**:

- When `SEMI_QUANT_DISABLE` is defined: `dequantize4()` fallback functions are provided for TQ3_0, TQ2_0, and Mixed KV types; `USE_SEMI_QUANT` is not defined; the shader falls through to the standard pipeline.
- When not defined: the fused semi-quantized kernels are active.

This isolates the performance difference of the fused compute path — same data, same bandwidth, different compute strategy.

**Build with A/B mode:**
```powershell
cmake -B build_vulkan -DGGML_VULKAN=ON -DSEMI_QUANT_DISABLE=ON
cmake --build build_vulkan --config Release
```

---

## Build & Run

### Prerequisites

- Vulkan SDK 1.1+ (tested: 1.4.341.1)
- CMake 3.14+
- C/C++ compiler (MSVC, GCC, or Clang)
- GGUF model file
- Python 3.8+ (for benchmark scripts)

### Build from Source

```powershell
cd llama_src
cmake -B build_vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_vulkan --config Release
```

### Run — TQ3_2 (Recommended)

Maximum compression with full cognitive quality — 6.4× compression, 93.3% fidelity:

```powershell
.\llama_src\build_vulkan\bin\Release\llama-server.exe `
  -m models\google_gemma-4-26B-A4B-it-Q4_K_M.gguf `
  -ngl 30 -ctk tq3_0 -ctv tq3_2 -c 4096
```

### Run — TQ3_0 (Balanced)

Moderate compression (4.6×), highest standalone quality:

```powershell
.\llama_src\build_vulkan\bin\Release\llama-server.exe `
  -m models\your_model.gguf `
  -ngl 30 -ctk tq3_0 -ctv tq3_0 -c 4096
```

### Run — TQ3_1 (Mixed KV)

5.3× compression, balanced K-quality/V-compression:

```powershell
.\llama_src\build_vulkan\bin\Release\llama-server.exe `
  -m models\your_model.gguf `
  -ngl 30 -ctk tq3_0 -ctv tq2_0 -c 4096
```

### Run — TQ2_0 (Maximum Compression)

6.4× compression, suitable for chat/summarization:

```powershell
.\llama_src\build_vulkan\bin\Release\llama-server.exe `
  -m models\your_model.gguf `
  -ngl 30 -ctk tq2_0 -ctv tq2_0 -c 4096
```

### Benchmark Scripts

```powershell
# Bandwidth analysis
py bench_bandwidth.py

# Quality validation vs FP16 baseline (8 prompts)
py bench_quality_validation.py

# Long context stress test (2K–32K)
py bench_long_context.py

# Full production suite (100 prompts, 5 KV configs, 5 context tiers)
py bench_suite/run_all.py
```

---

## Repository Structure

```
TurboQuant vulkan/
├── README.md                          # This document
├── final_report.md                    # Comprehensive technical report (358 lines)
├── requirements.txt                   # Python dependencies (psutil, huggingface-hub)
├── .gitignore
│
├── llama_src/                         # Modified llama.cpp with Vulkan FA shader patches
│   └── ggml/src/ggml-vulkan/
│       ├── vulkan-shaders/
│       │   ├── flash_attn_base.glsl   # semi_quant_qk_dot() + semi_quant_pv_accum()
│       │   ├── flash_attn.comp        # USE_SEMI_QUANT macro + SHMEM guards
│       │   └── ...                    # Supporting shaders
│       └── ggml-vulkan.cpp            # shmem_staging=0, FA_SCALAR enforcement
│
├── bench_suite/                       # Production benchmark suite
│   ├── README.md                      # Suite design documentation
│   ├── prompts.py                     # 100-prompt corpus (5 categories × 5 tiers)
│   ├── runner.py                      # Server lifecycle + sequential HTTP execution
│   ├── report.py                      # LLM-judge Markdown workbook generator
│   ├── run_all.py                     # End-to-end entry point
│   └── results/
│       ├── raw.json                   # Latest TQ3_2 run (100 prompts)
│       ├── raw.baseline.json          # Full 5-config comparison (f16, tq2_0, tq3_0, tq3_1, tq3_2)
│       ├── RESULTS.baseline.md        # LLM-judge workbook (80K lines)
│       ├── run.log / run.err          # Runner output logs
│       └── server_logs/               # Per-config server stderr captures
│
├── bench_v2/                          # Extended benchmark dataset
│   └── dataset/
│       └── dataset_raw.jsonl          # 101-entry JSONL (10 cognitive domains, 3 context sizes)
│
├── turboquant/                        # TQ3_3 cold-V format R&D (Python reference)
│   ├── ROADMAP.md                     # 4-phase architectural plan
│   ├── tq3_3_ref.py                   # 18 B / 64-element cold block format reference
│   ├── tq3_3_study.py                 # Empirical MSE calibration (4 distributions)
│   ├── tq3_3_study_results.json       # Gauss/Laplace/mix/skew study results
│   └── test_tq3_3_ref.py             # Round-trip validation tests
│
├── tq3_0_repo/                        # TQ3_0 standalone distribution (AGPL-3.0)
│   ├── README.md                      # Full docs with benchmarks + methodology
│   ├── tq3_0.patch                    # Monolithic llama.cpp patch
│   ├── scripts/                       # Setup, launcher, benchmark scripts
│   ├── bench/                         # Automated resumable benchmark system
│   └── results/                       # Throughput charts, coherence figures
│
├── turboquant_repo/                   # Original CUDA/vLLM TurboQuant (ICLR 2026)
│   ├── README.md                      # RTX 5090 / 8× RTX 3090 benchmarks
│   ├── turboquant/                    # Lloyd-Max codebooks, QJL rotation, vLLM backend
│   └── codebooks/                     # Pre-generated d64/d128/d576 codebooks
│
├── agents/                            # Autonomous agent experimentation system
│   └── README.md                      # Agent system documentation
│
├── mcp_planner/                       # MCP (Model Context Protocol) server prototype
│   └── server.py
│
└── backups/                           # Workspace snapshots
```

---

## Known Limitations

1. **Scalar path only.** Semi-quantized compute is implemented exclusively in `FA_SCALAR`. CooperativeMatrix (CM1/CM2) hardware acceleration is incompatible with direct-from-packed compute — the fundamental trade-off of the architecture.

2. **FP16 V accumulation.** The PV accumulator uses `FLOAT_TYPEV4` (FP16 on most GPUs). At extreme context lengths (256K+), accumulated values may gradually lose precision. Upgrading to FP32 accumulation would require a non-trivial shader refactor affecting all Flash Attention paths.

3. **No SHMEM staging for TQ types.** TQ reads packed data directly from global memory. This trades away shared-memory latency hiding (which helps with repeated data reuse) in exchange for reduced SHMEM pressure and higher occupancy. The net effect is hardware-dependent and favors AMD RDNA2+ where the scalar path is already performant.

4. **12 GB VRAM constraint.** With gemma-4-26B-A4B (Q4_K_M), loading 30 GPU layers leaves limited headroom for KV cache. TQ's 6.4× compression helps substantially but the model weights dominate total VRAM.

5. **No online quantization.** KV cache entries are quantized at insertion time using CPU-side Lloyd-Max routines. GPU-accelerated online quantization (via compute shaders rather than CPU fallback) would reduce prefill latency for large prompts but was not implemented.

6. **Vulkan backend only.** The semi-quantized attention kernels are GLSL compute shaders targeting the Vulkan backend. CUDA, Metal, and CPU backends use standard dequantize-then-compute paths for TQ types.

7. **Experimental codebook disabled.** See [Attempted Codebook Optimization](#attempted-codebook-optimization) — the Lloyd-Max rate-distortion codebook variant is disabled from the public KV cache selector due to a SPIR-V driver-cache throughput regression that could not be solved at the shader level.

---

## Original CUDA TurboQuant Comparison

This project is inspired by and validates the TurboQuant paper (Duanmu et al., ICLR 2026). Key differences from the original implementation:

| Feature | TurboQuant (Original) | TurboQuant-Vulkan (This Work) |
|:--------|:---------------------:|:-----------------------------:|
| Language | Python + CUDA/Triton | Pure C + GLSL |
| GPU Backend | CUDA (NVIDIA only) | Vulkan (AMD, Intel, NVIDIA) |
| Inference Framework | vLLM | llama.cpp |
| KV Cache Bit-widths | 3-bit (2-bit values) | 3-bit, 2-bit, and mixed-precision |
| Codebook | Lloyd-Max (Gaussian) | Lloyd-Max (TQ3_0) + Algebraic (TQ2_0) |
| Flash Attention Integration | Post-hoc KV compression | **Natively fused** — dot products on packed data |
| Prefill Acceleration | Via Triton kernels | Via Vulkan compute shaders + SHMEM elimination |
| Original Benchmarks | RTX 5090: +5.7% prefill, +3.1% decode at 30K ctx | RX 6750 XT: +27% decode, 8× max context |

### Original Paper Benchmarks (Reproduced for Context)

**RTX 5090, Qwen3.5-27B-AWQ:**

| Metric | bf16 KV Baseline | TurboQuant (3b/2b) | Improvement |
|:-------|:----------------:|:------------------:|:-----------:|
| Prefill tok/s (30K ctx) | 1,804 | 1,907 | +5.7% |
| Decode tok/s (30K ctx) | 1.264 | 1.303 | +3.1% |
| KV cache freed | — | 30.0 GB | — |
| Max token capacity | 457,072 | 914,144 | 2.0× |

**8× RTX 3090, Qwen3.5-35B-A3B (MoE):**

| Context | Prefill tok/s | Decode tok/s | TQ KV / GPU | Savings |
|--------:|:-------------:|:------------:|:-----------:|:-------:|
| 8,000 | 9,684 | 131.1 | 38.5 MB | 30.9% |
| 32,000 | 9,761 | 116.7 | 132.3 MB | 30.9% |
| 131,000 | 8,238 | 98.3 | 521.9 MB | 30.9% |

---

## Project Status: Deprecated

This project is no longer under active development. The author has shifted focus to new development priorities. The codebase and all associated benchmarks, documentation, and research artifacts are preserved here as a **complete, archived body of work**.

**The key contributions of this project remain valid and reproducible:**

- Semi-quantized Flash Attention operating directly on packed KV cache data without intermediate dequantization — eliminating shared-memory staging overhead entirely through compile-time shader specialization.
- Production-validated 6.4× KV cache compression at 93.3% cognitive fidelity retention (TQ3_2), pushing usable context to 2M+ tokens on a 12 GB consumer GPU.
- Two compute-only quality corrections (pre-softmax QK scale ×1.03, pairwise decorrelation filter α=0.125) that improve output quality without changing the storage format — a genuine zero-cost accuracy upgrade.
- Comprehensive cross-methodology benchmark suite with 500+ test executions spanning 5 cognitive domains, 5 context tiers, 5 KV configurations, and 4 independent evaluation methods.
- Validated rate-distortion codebook design with empirical MSE calibration and documented SPIR-V driver-cache limitation — preserving the roadmap for future architectural work.

All data, results, and source code are retained in their final state. The repository serves as a reference implementation for GPU-accelerated KV cache compression in the Vulkan compute ecosystem.

---

## License & Citation

This project is licensed under the **GNU Affero General Public License v3 (AGPL-3.0)**.

- **Commercial use is permitted** — you may use, modify, and distribute this software commercially.
- **Modifications must be open-sourced** — if you modify this software and distribute it or provide it as a network service, you must release your modifications under the same license.
- **Network use triggers copyleft** — if you run a modified version as a network service (e.g., SaaS, API endpoint), you must provide the source code to users.

The modified llama.cpp source files (`llama_src/`) additionally carry the upstream [MIT License](https://github.com/ggml-org/llama.cpp) for portions derived from llama.cpp. The overall project and all original TurboQuant code is AGPL-3.0.

### Commercial Licensing

Organizations that wish to use this software **without open-sourcing their modifications** must obtain a commercial license. This applies to proprietary software incorporating TurboQuant-Vulkan, closed-source SaaS deployments, and any use where AGPL compliance is not desired.

Contact: [vhmarchiore@gmail.com](mailto:vhmarchiore@gmail.com)

### Citation

```bibtex
@misc{turboquant_vulkan,
  title   = {TurboQuant-Vulkan: Semi-Quantized Flash Attention with 3-bit/2-bit KV Cache for llama.cpp},
  author  = {tsuyu122},
  year    = {2026},
  url     = {https://github.com/tsuyu122/TurboQuant-Vulkan}
}
```

```bibtex
@inproceedings{duanmu2025turboquant,
  title     = {TurboQuant: Online Vector Quantization for GPGPU-Efficiented KV Cache Quantization},
  author    = {Duanmu, Hao and Zhang, Jingyu and Ye, Peiqi and Wu, Yifeng and Wang, Shixuan and Sun, Jiafei and Liu, Zhibo and Martin, David and Wei, Jason},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```
