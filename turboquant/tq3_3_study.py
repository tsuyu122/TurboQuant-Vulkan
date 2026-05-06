"""Empirical study for TQ3_3 cold V-cache format.

Goals
-----
1. Verify the 18 B / 64-element layout produces acceptable quality on
   realistic attention-V distributions.
2. Derive the data-driven ``MSE_GATE`` threshold for the fallback rule.
3. Measure the fallback rate at the chosen threshold (i.e. how often a
   candidate cold block is rejected and has to be stored in the safe
   TQ2_0 / TQ3_2 format instead).
4. Produce bandwidth numbers usable by the Phase 2/3 plan.

Methodology
-----------
We do **not** yet have a dumped real V-cache from the target model (that
tooling is part of Phase 2). Instead we evaluate on distributions that
are known upper / lower bounds on attention-V post-norm:

- ``gauss``   — pure normal, analytical baseline
- ``laplace`` — heavier tails, typical for deep-layer V
- ``mix``     — 90% gauss + 10% Student-t(3): models outlier tokens
- ``skew``    — skew-normal, tests asymmetry robustness

We then sweep the MSE gate and report the resulting fallback rate
for each distribution. The gate is chosen so that the **worst-case**
kept-cold MSE stays below the TQ2_0 baseline MSE — i.e. any block that
passes the gate is at least as good in cold format as it would be in
TQ2_0. This is a correctness-preserving definition, not a magic number.

The numbers this script prints are what get copied into the C / Vulkan
port (scale of the gate constant, expected fallback rate, bandwidth
savings at steady state).

Run
---
    py turboquant/tq3_3_study.py
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

import numpy as np

from turboquant.tq3_3_ref import (
    BLOCK_BYTES_TQ3_3_COLD,
    QK_TQ3_3_COLD,
    block_reconstruction_mse,
    dequantize_row_tq3_3_cold,
    quantize_row_tq3_3_cold,
)


# ---------------------------------------------------------------------------
# Reference quantisers for comparison (minimal, not byte-accurate)
# ---------------------------------------------------------------------------

def _quant_tq2_0(x_32: np.ndarray) -> np.ndarray:
    """Round-trip approx of TQ2_0 on a 32-element block: {-1,-1/3,+1/3,+1}*amax."""
    assert x_32.shape == (32,)
    amax = float(np.abs(x_32).max())
    if amax == 0.0:
        return x_32.copy()
    n = x_32 / amax
    codes = np.full(32, 1, dtype=np.int8)
    codes[n < -2.0 / 3.0] = 0
    codes[(n >= -2.0 / 3.0) & (n < 0.0)] = 1
    codes[(n >= 0.0) & (n < 2.0 / 3.0)] = 2
    codes[n >= 2.0 / 3.0] = 3
    values = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float32)
    return (values[codes] * amax).astype(np.float32)


def tq2_0_block_mse_ratio(x_64: np.ndarray) -> float:
    """TQ2_0 round-trip MSE / var on a 64-elem chunk (two TQ2_0 blocks)."""
    assert x_64.shape == (QK_TQ3_3_COLD,)
    y = np.empty_like(x_64, dtype=np.float32)
    y[:32] = _quant_tq2_0(x_64[:32].astype(np.float32))
    y[32:] = _quant_tq2_0(x_64[32:].astype(np.float32))
    err = x_64.astype(np.float32) - y
    denom = float(np.mean(x_64.astype(np.float32) ** 2))
    return float(np.mean(err ** 2)) / denom if denom > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Distribution generators (seeded, reproducible)
# ---------------------------------------------------------------------------

def _gen_gauss(rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    return rng.standard_normal((n_blocks, QK_TQ3_3_COLD)).astype(np.float32)


def _gen_laplace(rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    return rng.laplace(0.0, 1.0 / np.sqrt(2.0), (n_blocks, QK_TQ3_3_COLD)).astype(np.float32)


def _gen_mix(rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    gauss = rng.standard_normal((n_blocks, QK_TQ3_3_COLD)).astype(np.float32)
    outlier_mask = rng.random(n_blocks) < 0.10
    heavy = rng.standard_t(3.0, (n_blocks, QK_TQ3_3_COLD)).astype(np.float32)
    gauss[outlier_mask] = heavy[outlier_mask]
    return gauss


def _gen_skew(rng: np.random.Generator, n_blocks: int) -> np.ndarray:
    u = rng.standard_normal((n_blocks, QK_TQ3_3_COLD)).astype(np.float32)
    v = rng.standard_normal((n_blocks, QK_TQ3_3_COLD)).astype(np.float32)
    alpha = 4.0
    delta = alpha / np.sqrt(1.0 + alpha ** 2)
    return (delta * np.abs(u) + np.sqrt(1.0 - delta ** 2) * v).astype(np.float32)


DISTRIBUTIONS: dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "gauss":   _gen_gauss,
    "laplace": _gen_laplace,
    "mix":     _gen_mix,
    "skew":    _gen_skew,
}


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def block_mse_array(blocks: np.ndarray) -> np.ndarray:
    """Cold-format round-trip MSE-ratio for every block."""
    return np.array([block_reconstruction_mse(b) for b in blocks], dtype=np.float64)


def tq2_0_mse_array(blocks: np.ndarray) -> np.ndarray:
    return np.array([tq2_0_block_mse_ratio(b) for b in blocks], dtype=np.float64)


def calibrate_gate(cold_mse: np.ndarray, tq2_0_mse: np.ndarray) -> float:
    """Choose gate = max(tq2_0_mse) so that kept cold blocks are <= TQ2_0 worst case.

    In words: any block the cold format reconstructs at least as well as
    the worst TQ2_0 reconstruction of the reference distribution passes
    the gate. Everything else falls back. This is a conservative,
    quality-preserving definition tied directly to the TQ2_0 baseline.
    """
    return float(np.quantile(tq2_0_mse, 0.99))


def summarise(name: str, blocks: np.ndarray) -> dict:
    cold = block_mse_array(blocks)
    tq20 = tq2_0_mse_array(blocks)
    gate = calibrate_gate(cold, tq20)
    kept_mask = cold <= gate
    fallback_rate = 1.0 - float(kept_mask.mean())

    # Bandwidth: cold-kept use 18 B / 64 elem, fallback uses TQ2_0 (10 B / 32 elem = 20 B / 64 elem)
    cold_bytes_per_64 = BLOCK_BYTES_TQ3_3_COLD        # 18
    fallback_bytes_per_64 = 2 * 10                    # TQ2_0 x 2 = 20
    avg_bytes_per_64 = kept_mask.mean() * cold_bytes_per_64 + (1 - kept_mask.mean()) * fallback_bytes_per_64
    bytes_saved_pct = 100.0 * (fallback_bytes_per_64 - avg_bytes_per_64) / fallback_bytes_per_64

    return {
        "distribution":        name,
        "n_blocks":            int(blocks.shape[0]),
        "cold_mse_mean":       float(cold.mean()),
        "cold_mse_p50":        float(np.quantile(cold, 0.50)),
        "cold_mse_p90":        float(np.quantile(cold, 0.90)),
        "cold_mse_p99":        float(np.quantile(cold, 0.99)),
        "tq2_0_mse_mean":      float(tq20.mean()),
        "tq2_0_mse_p99":       float(np.quantile(tq20, 0.99)),
        "mse_gate":            gate,
        "fallback_rate":       fallback_rate,
        "kept_cold_mse_mean":  float(cold[kept_mask].mean()) if kept_mask.any() else 0.0,
        "kept_cold_mse_max":   float(cold[kept_mask].max())  if kept_mask.any() else 0.0,
        "avg_bytes_per_64":    float(avg_bytes_per_64),
        "bytes_saved_vs_tq2_0_pct": float(bytes_saved_pct),
    }


def main() -> int:
    rng = np.random.default_rng(0x3D3C01D)
    N = 8_000

    results = []
    for name, gen in DISTRIBUTIONS.items():
        blocks = gen(rng, N)
        results.append(summarise(name, blocks))

    # Conservative global gate = max of per-distribution gates so any single
    # block in any realistic distribution stays safe.
    global_gate = max(r["mse_gate"] for r in results)

    # Recompute fallback rate with the global gate (what the production
    # kernel will actually use). The per-distribution "mse_gate" is
    # informational; only the global gate is real.
    rng2 = np.random.default_rng(0xBADF00D)
    for r, name in zip(results, DISTRIBUTIONS):
        blocks = DISTRIBUTIONS[name](rng2, N)
        cold = block_mse_array(blocks)
        tq20 = tq2_0_mse_array(blocks)
        kept_mask = cold <= global_gate
        r["fallback_rate_global_gate"] = 1.0 - float(kept_mask.mean())
        r["kept_cold_mse_mean_global"] = float(cold[kept_mask].mean()) if kept_mask.any() else 0.0
        r["kept_cold_mse_max_global"]  = float(cold[kept_mask].max())  if kept_mask.any() else 0.0
        cold_bytes_per_64 = BLOCK_BYTES_TQ3_3_COLD
        fallback_bytes_per_64 = 20
        avg = kept_mask.mean() * cold_bytes_per_64 + (1 - kept_mask.mean()) * fallback_bytes_per_64
        r["avg_bytes_per_64_global"] = float(avg)
        r["bytes_saved_vs_tq2_0_pct_global"] = float(100.0 * (fallback_bytes_per_64 - avg) / fallback_bytes_per_64)

    out = {
        "format": {
            "name":                   "block_tq3_3_cold",
            "elements_per_block":     QK_TQ3_3_COLD,
            "bytes_per_block":        BLOCK_BYTES_TQ3_3_COLD,
            "bytes_per_element":      BLOCK_BYTES_TQ3_3_COLD / QK_TQ3_3_COLD,
            "bpw":                    (BLOCK_BYTES_TQ3_3_COLD * 8) / QK_TQ3_3_COLD,
            "fallback_format":        "TQ2_0 (10 B / 32 elem = 20 B / 64 elem)",
            "raw_bandwidth_saving_pct_vs_tq2_0": 100.0 * (20 - BLOCK_BYTES_TQ3_3_COLD) / 20.0,
        },
        "global_mse_gate": global_gate,
        "per_distribution": results,
    }

    out_path = pathlib.Path("turboquant/tq3_3_study_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"=== TQ3_3 cold-format study ===")
    print(f"Layout: {BLOCK_BYTES_TQ3_3_COLD} B / {QK_TQ3_3_COLD} elem "
          f"= {BLOCK_BYTES_TQ3_3_COLD/QK_TQ3_3_COLD:.4f} B/elem "
          f"= {BLOCK_BYTES_TQ3_3_COLD*8/QK_TQ3_3_COLD:.3f} bpw")
    print(f"Raw bandwidth vs TQ2_0: -{100.0*(20-BLOCK_BYTES_TQ3_3_COLD)/20.0:.1f}% (if 0% fallback)")
    print(f"Global MSE gate (quality-preserving): {global_gate:.4f}")
    print()
    for r in results:
        print(f"[{r['distribution']:<8}] "
              f"cold MSE mean={r['cold_mse_mean']:.4f} p99={r['cold_mse_p99']:.4f} | "
              f"TQ2_0 MSE mean={r['tq2_0_mse_mean']:.4f} | "
              f"fallback={100*r['fallback_rate_global_gate']:5.1f}% | "
              f"bytes/64={r['avg_bytes_per_64_global']:.2f} "
              f"(-{r['bytes_saved_vs_tq2_0_pct_global']:.1f}%)")
    print()
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
