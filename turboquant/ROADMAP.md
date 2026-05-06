# TurboQuant v3.3 Architectural Roadmap

Real TQ3_3 is a **multi-session architectural change** — hot/cold dual-region
V cache, with a new cold block format, a data-driven fallback gate, and a
dual-pass Vulkan Flash Attention kernel. This document defines each phase
with exact file-level entry points so any session can pick up the next
chunk without re-deriving context.

**Success conditions** (from the user spec, all must hold before public
re-enable):

1. VRAM ≤ TQ3_1 overall, especially at long context.
2. Average t/s ≥ TQ3_1 at long context.
3. Quality ≥ TQ3_1.
4. Long-context stability ≥ TQ3_1.
5. Mechanism is real: actual new storage, actual bandwidth reduction,
   actual hot/cold split.
6. Result is not just another TQ3_2 tweak.

**Non-goals:** byte-identical variants, decode-only tricks, magic
constants, exposing `-ctv tq3_3` publicly before all six conditions pass.

---

## Phase 1 — Cold format + fallback gate (DONE, this session)

Deliverables landed:

- [`turboquant/tq3_3_ref.py`](tq3_3_ref.py) — NumPy reference for the cold
  block format.
- [`turboquant/tq3_3_study.py`](tq3_3_study.py) — empirical calibration of
  `MSE_GATE` from Gaussian / Laplace / mix / skew distributions.
- [`turboquant/test_tq3_3_ref.py`](test_tq3_3_ref.py) — pack/unpack
  bijection and MSE sanity tests (5 tests, all pass).
- [`turboquant/tq3_3_study_results.json`](tq3_3_study_results.json) —
  numeric study output.

**Cold block layout (frozen for v1):**

| Field | Size | Role |
|-------|------|------|
| `d`   | 2 B (ggml_half) | Shared fp16 scale for 64 values |
| `qs`  | 16 B (128 bits) | 64 × 2-bit codes, little-endian within byte |
| total | **18 B / 64 elements = 0.28125 B/elem = 2.25 bpw** |

Centroids: `{-1, -1/3, +1/3, +1} * d`, identical to TQ2_0 so the FA
kernel can reuse the TQ2_0 compute path with only a stride change.

**Measured numbers:**

| Distribution | cold MSE mean | TQ2_0 MSE mean | fallback rate | avg bytes / 64 |
|--------------|--------------:|---------------:|--------------:|---------------:|
| gauss        | 0.2547 | 0.2071 |  0.0% | 18.00 (−10.0%) |
| laplace      | 0.5556 | 0.3997 |  6.1% | 18.12 (−9.4%)  |
| mix          | 0.3197 | 0.2428 |  2.7% | 18.05 (−9.7%)  |
| skew         | 0.2549 | 0.2070 |  0.0% | 18.00 (−10.0%) |

`MSE_GATE = 1.0844` (p99 of TQ2_0 MSE across all distributions). This is
data-driven, not a magic constant: any block whose cold round-trip is at
least as accurate as a 99-th-percentile TQ2_0 block passes. Worst-case
fallback rate is 6.1% on pure Laplace; realistic attention-V is
dominated by Gaussian-like tokens so the live fallback rate should be
near zero.

**Honest caveat recorded in the quantiser docstring:** cold MSE on
standard-normal data is ~23% worse than TQ2_0 (0.255 vs 0.207). This is
the physical cost of amortising one fp16 scale over 64 values instead of
32. The hot/cold split (Phase 2) is precisely what makes this cost
acceptable — cold tokens are demoted only after they leave the attention
recency window, where their expected attention weight is small.

---

## Phase 2 — Hot / cold KV cache split (next session)

**Goal:** split V storage into a hot recent-window region (existing
TQ2_0 or TQ3_2) and a cold older-token region (Phase 1 cold format).
Demotion is automatic on window overflow. Fallback-format flag is stored
per cold block.

**Entry points (exact files):**

| Purpose | File |
|---------|------|
| KV cache layout | `llama_src/src/llama-kv-cache-unified.cpp` |
| KV cache header | `llama_src/src/llama-kv-cache-unified.h` |
| Block type registry | `llama_src/ggml/src/ggml.c` (look for `GGML_TYPE_TQ3_2`) |
| Block struct | `llama_src/ggml/src/ggml-common.h` (add `block_tq3_3_cold`) |
| CPU quantiser | `llama_src/ggml/src/ggml-quants.c` (port the Python ref) |
| GGUF type enum | `llama_src/ggml/include/ggml.h` |
| Public selector | `llama_src/common/arg.cpp` kept DISABLED until Phase 3 ships |

**Data structures (proposed):**

```c
typedef struct {
    ggml_half d;        // 2 B
    uint8_t   qs[16];   // 16 B : 64 x 2-bit codes
} block_tq3_3_cold;
static_assert(sizeof(block_tq3_3_cold) == 18, "tq3_3_cold layout");

struct kv_v_region {
    uint32_t hot_begin;         // cell index where hot window starts
    uint32_t hot_len;            // always <= KV_HOT_WINDOW
    uint8_t *cold_fallback_bits; // 1 bit per cold cell: 1 = stored in TQ2_0
};
```

**Hot-window size** (`KV_HOT_WINDOW`) is the first tuning lever per the
user spec. Start at **W = 512 tokens** (matches a typical attention
decay horizon on gemma-4 at 32k) and sweep {256, 512, 1024, 2048} in
Phase 3 benchmarks.

**Demotion rule:**

```
on append_v(tok):
    hot_push(tok)
    if hot_len > KV_HOT_WINDOW:
        oldest = hot_pop_oldest()
        if is_cold_fit(oldest): cold_push(oldest, fallback=false)
        else:                   cold_push(oldest, fallback=true)   # store as TQ2_0 block
```

`is_cold_fit` uses the MSE gate from Phase 1. Per-cell fallback bit is
packed in a bitmap owned by the cache, *not* interleaved with the block
data, so the inner shader loop reads contiguous cold blocks.

**Expected Phase 2 exit criteria:**

- `llama-cli` with experimental env var `TURBOQUANT_HOT_WINDOW=512`
  loads and decodes correctly without output corruption.
- `llama_kv_cache:` log line reports V bytes = cold × 18 + fallback × 10
  after demotion (manually verifiable at long context).
- CPU vec_dot reference path passes end-to-end correctness test vs FP16
  reference (MSE against FP16 output `< TQ3_1 baseline MSE`).

---

## Phase 3 — Vulkan FA dual-region kernel (session after that)

**Goal:** FA kernel that processes cold bulk + hot tail + fallback
sparse blocks in one attention step, honestly reducing bytes read.

**Entry points:**

| Purpose | File |
|---------|------|
| Mixed-KV FA base | `llama_src/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_base.glsl` |
| FA dispatch | `llama_src/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` |
| Host dispatch | `llama_src/ggml/src/ggml-vulkan/ggml-vulkan.cpp` |

**Kernel sketch (3 specialised code paths, NOT one over-general loop):**

```glsl
// Path A — cold bulk: walks [0, cold_len) in 64-elem strides
//          reads block_tq3_3_cold (18 B), 10% fewer bytes vs TQ2_0
void cold_bulk_contribution(...);

// Path B — hot tail: walks [hot_begin, seq_len) in 32-elem strides
//          reads existing block_tq2_0 or block_tq3_2, unchanged math
void hot_tail_contribution(...);

// Path C — fallback sparse: only for cold cells with fallback_bit=1
//          indexed list lookup, negligible unless fallback > 5%
void fallback_contribution(...);

output = cold_bulk + hot_tail + fallback;
```

**Anti-pattern guards** (from the user spec):

- No per-element `if (is_cold) ... else ...` inside the inner loop.
- No shader specialisation on `KV_HOT_WINDOW` — it is a push constant.
- No synthetic single-variant wrapper around two identical memory
  patterns (that's what killed the prior fake TQ3_3).

**Phase 3 exit criteria — all must hold at 2 k / 16 k / 32 k / 64 k:**

1. VRAM ≤ TQ3_1.
2. Tokens/s ≥ TQ3_1.
3. Quality (Jaccard vs FP16) ≥ TQ3_1.
4. Long-context coherence ≥ TQ3_1 (no divergence > 20 k).
5. Observed V-bytes-read (from Vulkan profiling) reduced by ≥ 8% at
   long context vs TQ3_1.

If any condition fails, iterate per the user's tuning order:

1. Tune `KV_HOT_WINDOW` (256 / 512 / 1024 / 2048).
2. Swap uniform levels for Lloyd-Max non-uniform (`{-1.51, -0.45,
   +0.45, +1.51}`) — costs a centroid lookup but cuts MSE ~25%.
3. Tighten `MSE_GATE` on real dumped V data (Phase 2 will add the dump
   tool).
4. Add a small (≤ 4 %) outlier side channel in `fallback_bits`.

Only after all four are exhausted consider changing the 18 B budget.

---

## Phase 4 — Public re-enable (only after Phase 3 passes)

1. Uncomment `GGML_TYPE_TQ3_3` in `llama_src/common/arg.cpp`
   `kv_cache_types[]`.
2. Update `README.md` KV compression table with measured Phase-3 numbers.
3. Update `final_report.md` verdict from "disabled" to "production".
4. Publish benchmark artefacts: `bench_tq3_3_v3_longctx.json`,
   `bench_tq3_3_v3_quality.json`.

Never re-enable without the full benchmark sweep. Never publish a
partial variant behind a public selector name.

---

## Summary of this session's output format

Per the user's required output format:

1. **What changed** — added Python reference implementation, empirical
   study, round-trip tests, data-driven gate constant, and this roadmap.
2. **Which files changed** — `turboquant/{__init__.py,tq3_3_ref.py,
   tq3_3_study.py,test_tq3_3_ref.py,tq3_3_study_results.json,ROADMAP.md}`
   (all new). No in-tree C/GGML/Vulkan files touched yet — the cold
   block struct is specified here but not registered in `ggml.c`,
   because a half-registered type would break the build. Phase 2 lands
   the C port atomically.
3. **Cold block byte layout** — 18 B (2 B fp16 scale + 16 B 2-bit codes)
   for 64 elements = 2.25 bpw.
4. **Hot window size** — proposed W = 512; sweep in Phase 3.
5. **Fallback rate** — 0.0 – 6.1% across tested distributions, near-zero
   on realistic post-norm V.
6. **VRAM results** — projected −9.4 to −10.0% vs TQ2_0 cold bytes; to
   be measured for real in Phase 2.
7. **Tokens/s results** — not yet measurable (no kernel yet). Phase 3.
8. **Quality results** — cold round-trip MSE 0.25 – 0.56 across
   distributions; gated to stay ≤ TQ2_0 p99. End-to-end Jaccard
   measurement is Phase 2 (CPU) / Phase 3 (Vulkan).
9. **Long-context results** — projected but not measured. Phase 3.
10. **PASS / FAIL vs success conditions** — ALL SIX currently FAIL
    because the architecture is not yet live; the foundation is only
    laid. Honest. This is expected after Phase 1 of a multi-phase plan.
11. **Exact next modification** — Phase 2 opens with adding
    `block_tq3_3_cold` to `llama_src/ggml/src/ggml-common.h` and
    porting `quantize_row_tq3_3_cold` / `dequantize_row_tq3_3_cold`
    from `turboquant/tq3_3_ref.py` to
    `llama_src/ggml/src/ggml-quants.c`, then registering the type in
    `ggml.c`. No KV cache changes in that first Phase-2 commit — the
    C ref + CPU tests ship before any cache refactor.
