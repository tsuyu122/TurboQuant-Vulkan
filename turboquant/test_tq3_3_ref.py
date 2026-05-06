"""Round-trip + boundary tests for the TQ3_3 cold reference implementation."""

from __future__ import annotations

import numpy as np

from turboquant.tq3_3_ref import (
    BLOCK_BYTES_TQ3_3_COLD,
    QK_TQ3_3_COLD,
    TQ3_3_COLD_CENTROIDS,
    _pack_2bit,
    _unpack_2bit,
    block_reconstruction_mse,
    dequantize_row_tq3_3_cold,
    quantize_row_tq3_3_cold,
)


def test_layout() -> None:
    assert QK_TQ3_3_COLD == 64
    assert BLOCK_BYTES_TQ3_3_COLD == 18
    assert TQ3_3_COLD_CENTROIDS.shape == (4,)


def test_pack_unpack_bijection() -> None:
    rng = np.random.default_rng(0)
    for _ in range(256):
        codes = rng.integers(0, 4, QK_TQ3_3_COLD, dtype=np.uint8)
        round_trip = _unpack_2bit(_pack_2bit(codes))
        assert np.array_equal(round_trip, codes)


def test_quantize_zero_block() -> None:
    d, qs = quantize_row_tq3_3_cold(np.zeros(QK_TQ3_3_COLD, dtype=np.float32))
    y = dequantize_row_tq3_3_cold(d, qs)
    assert np.allclose(y, 0.0)


def test_quantize_centroid_values_are_exact() -> None:
    # A block composed purely of {-1, -1/3, +1/3, +1} * s must be lossless.
    s = 0.7
    template = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0] * (QK_TQ3_3_COLD // 4),
                        dtype=np.float32) * s
    d, qs = quantize_row_tq3_3_cold(template)
    y = dequantize_row_tq3_3_cold(d, qs)
    # fp16 scale introduces a tiny rounding; allow a loose tolerance.
    assert np.allclose(y, template, atol=1e-3), f"max err = {np.max(np.abs(y-template))}"


def test_reconstruction_mse_bounded_for_gauss() -> None:
    rng = np.random.default_rng(42)
    errs = [block_reconstruction_mse(rng.standard_normal(QK_TQ3_3_COLD).astype(np.float32))
            for _ in range(512)]
    mean_mse = float(np.mean(errs))
    # Physics bound: amax-scaled 4-level uniform quantiser on N(0,1) has
    # MSE ratio ~0.257 (measured). The per-block scale burns one level
    # on the outlier, leaving 3 levels for the inlier density.
    # Anything <0.30 means the quantiser is behaving as expected; higher
    # would indicate a bug. Lloyd-Max non-uniform levels can push this
    # to ~0.20 as a follow-up, see ROADMAP §Iteration 2.
    assert mean_mse < 0.30, f"mean cold MSE ratio on Gaussian = {mean_mse:.4f}"


def run() -> int:
    test_layout()
    test_pack_unpack_bijection()
    test_quantize_zero_block()
    test_quantize_centroid_values_are_exact()
    test_reconstruction_mse_bounded_for_gauss()
    print("tq3_3_ref tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
