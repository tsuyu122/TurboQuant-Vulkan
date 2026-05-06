"""Reference implementation of the TQ3_3 cold V-cache format.

Cold-V block layout (target for the C / Vulkan port):

  block_tq3_3_cold {
      ggml_half d;        // 2 bytes: shared fp16 scale for all 64 values
      uint8_t   qs[16];   // 16 bytes: 64 x 2-bit codes, little-endian within byte
  }                       // 18 bytes total / 64 elements = 0.28125 B/elem = 2.25 bpw

Centroid table (algebraic, identical to TQ2_0, proven on LayerNorm'd V):

  code ∈ {0, 1, 2, 3} → value = (code * 2 - 3) * d / 3
                              = {-1, -1/3, +1/3, +1} * d

Compared to TQ2_0 (10 B / 32 values = 0.3125 B/elem), this format saves
**10% of V-bytes-read** by amortising the fp16 scale over 64 values
instead of 32. No compute change vs TQ2_0: the centroid formula is the
same, so the FA kernel can reuse the TQ2_0 dot-product code with a
different stride.

Fallback gate — NEVER store a block that the cold format can't represent:

A block is *unfit* for cold storage iff its normalised reconstruction MSE
exceeds ``MSE_GATE`` (data-driven, see ``tq3_3_study.py``). Candidates
that fail the gate stay in the existing TQ2_0 / TQ3_2 hot / fallback format.
The gate is computed per block from the DATA, not from an arbitrary
outlier heuristic, which means it stays valid across arbitrary V
distributions without re-tuning.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QK_TQ3_3_COLD",
    "BLOCK_BYTES_TQ3_3_COLD",
    "TQ3_3_COLD_CENTROIDS",
    "quantize_row_tq3_3_cold",
    "dequantize_row_tq3_3_cold",
    "block_reconstruction_mse",
    "is_cold_fit",
]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

QK_TQ3_3_COLD = 64                    # values per block
BLOCK_BYTES_TQ3_3_COLD = 2 + 16       # fp16 scale + 16 packed code bytes

# Algebraic centroids, identical to TQ2_0. With amax-scaling ``d = amax``
# centroid codes in {0,1,2,3} map to {-1, -1/3, +1/3, +1}*d. The factor
# 1/3 keeps the mid-values distinct from the extremes.
_CENTROIDS_UNIT = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float32)
TQ3_3_COLD_CENTROIDS = _CENTROIDS_UNIT.copy()


# ---------------------------------------------------------------------------
# Quantisation / dequantisation (reference, not perf-tuned)
# ---------------------------------------------------------------------------


def _pack_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack 64 codes in {0..3} into 16 little-endian bytes (4 codes/byte)."""
    assert codes.shape == (QK_TQ3_3_COLD,)
    assert codes.dtype == np.uint8
    reshaped = codes.reshape(QK_TQ3_3_COLD // 4, 4).astype(np.uint32)
    packed = (
        reshaped[:, 0]
        | (reshaped[:, 1] << 2)
        | (reshaped[:, 2] << 4)
        | (reshaped[:, 3] << 6)
    ).astype(np.uint8)
    return packed


def _unpack_2bit(packed: np.ndarray) -> np.ndarray:
    assert packed.shape == (QK_TQ3_3_COLD // 4,)
    assert packed.dtype == np.uint8
    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03
    return np.stack([c0, c1, c2, c3], axis=1).reshape(QK_TQ3_3_COLD).astype(np.uint8)


def quantize_row_tq3_3_cold(x: np.ndarray) -> tuple[np.float16, np.ndarray]:
    """Quantize a single 64-element block.

    Returns ``(d, qs)`` where ``d`` is the fp16 scale and ``qs`` is a
    ``uint8[16]`` packed array. The amax of ``x`` is mapped to ±d.
    Values are assigned to the nearest centroid via boundary thresholds
    derived from the unit centroid table.
    """
    if x.shape != (QK_TQ3_3_COLD,):
        raise ValueError(f"expected shape ({QK_TQ3_3_COLD},), got {x.shape}")
    amax = float(np.abs(x).max())
    d = np.float16(amax) if amax > 0.0 else np.float16(0.0)
    if amax == 0.0:
        return d, np.zeros(QK_TQ3_3_COLD // 4, dtype=np.uint8)
    normed = (x / amax).astype(np.float32)
    # Boundaries between {-1, -1/3, +1/3, +1}: -2/3, 0, +2/3
    codes = np.full(QK_TQ3_3_COLD, 1, dtype=np.uint8)           # default: -1/3
    codes[normed <  -2.0 / 3.0] = 0                              # -1
    codes[(normed >= -2.0 / 3.0) & (normed < 0.0)] = 1           # -1/3
    codes[(normed >= 0.0) & (normed < 2.0 / 3.0)] = 2            # +1/3
    codes[normed >= 2.0 / 3.0] = 3                               # +1
    return d, _pack_2bit(codes)


def dequantize_row_tq3_3_cold(d: np.float16, qs: np.ndarray) -> np.ndarray:
    if qs.shape != (QK_TQ3_3_COLD // 4,):
        raise ValueError(f"expected qs shape ({QK_TQ3_3_COLD // 4},), got {qs.shape}")
    codes = _unpack_2bit(qs)
    return (TQ3_3_COLD_CENTROIDS[codes] * float(d)).astype(np.float32)


def block_reconstruction_mse(x: np.ndarray) -> float:
    """MSE of cold-format round-trip, normalised by block variance.

    ``0.0`` means lossless, ``1.0`` means no better than mean-zero.
    """
    d, qs = quantize_row_tq3_3_cold(x)
    y = dequantize_row_tq3_3_cold(d, qs)
    err = x.astype(np.float32) - y
    denom = float(np.mean(x.astype(np.float32) ** 2))
    return float(np.mean(err ** 2)) / denom if denom > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Fallback gate
# ---------------------------------------------------------------------------
#
# ``MSE_GATE`` is set by ``tq3_3_study.py`` from measured cold-format error
# on a representative attention-V distribution. It is a *ratio*: blocks
# whose cold round-trip MSE / var exceeds the gate are demoted to the
# fallback (hot-safe) format. Because it's a ratio, the gate is invariant
# to scale and is therefore safe to reuse across layers / heads / models
# without re-tuning.
#
# Default value is deliberately coarse until the empirical study runs;
# the study overwrites this via ``calibrate_gate()`` below.

# Measured by turboquant/tq3_3_study.py on a mixture of Gaussian,
# Laplace, Gaussian+Student-t(3) outlier mix, and skew-normal
# distributions (8 000 blocks each). Value is the p99 of TQ2_0 round-trip
# MSE ratio across all distributions, so any block the cold format
# reconstructs at least as well as the 99-th percentile TQ2_0 block
# passes the gate.
MSE_GATE: float = 1.0844


def is_cold_fit(x: np.ndarray, gate: float | None = None) -> bool:
    """Return True iff the cold format can represent this block within gate."""
    g = MSE_GATE if gate is None else gate
    return block_reconstruction_mse(x) <= g
