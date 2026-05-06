# Copyright (c) 2026 tsuyu122
# Licensed under the GNU Affero General Public License v3 (AGPL-3.0)
# See LICENSE file for details.
"""
Generate Lloyd-Max codebook for TQ3_0 quantization.
Uses Beta(d/2-1, d/2-1) distribution for post-Hadamard-rotated coordinates.
For d=128 (typical head_dim), this is Beta(63, 63).

After rotation, each coordinate follows Beta(d/2-1, d/2-1) on [0,1],
which when centered becomes symmetric on [-0.5, 0.5].
We quantize with 3 bits = 8 levels using Lloyd-Max (minimum MSE).
"""

import numpy as np
from scipy import stats, integrate

def compute_lloyd_max_codebook(dim, bits, n_iter=200):
    """
    Compute Lloyd-Max codebook for Beta(dim/2-1, dim/2-1) distribution.
    The distribution is centered and scaled to have zero mean.
    
    Returns (centroids, boundaries) for the optimal quantizer.
    """
    alpha = dim / 2 - 1  # Beta parameter
    n_levels = 2 ** bits
    
    # Beta(alpha, alpha) on [0, 1] — symmetric around 0.5
    dist = stats.beta(alpha, alpha)
    
    # Initialize centroids uniformly in the quantile space
    quantiles = np.linspace(0, 1, n_levels + 1)
    # Use midpoints of quantile intervals as initial centroids
    centroids = np.array([
        dist.ppf((quantiles[i] + quantiles[i+1]) / 2)
        for i in range(n_levels)
    ])
    
    for iteration in range(n_iter):
        # Step 1: Update boundaries (midpoints between centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        
        # Step 2: Update centroids (conditional expectations)
        full_bounds = np.concatenate([[0.0], boundaries, [1.0]])
        new_centroids = np.zeros(n_levels)
        
        for i in range(n_levels):
            lo, hi = full_bounds[i], full_bounds[i+1]
            # E[X | lo <= X <= hi] = integral(x * f(x), lo, hi) / integral(f(x), lo, hi)
            prob = dist.cdf(hi) - dist.cdf(lo)
            if prob < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0
            else:
                numerator, _ = integrate.quad(
                    lambda x: x * dist.pdf(x), lo, hi
                )
                new_centroids[i] = numerator / prob
        
        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < 1e-12:
            print(f"  Converged at iteration {iteration}")
            centroids = new_centroids
            break
        centroids = new_centroids
    
    # Final boundaries
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    
    # Center: shift from [0,1] Beta to zero-mean [-0.5, 0.5]
    centroids_centered = centroids - 0.5
    boundaries_centered = boundaries - 0.5
    
    return centroids_centered, boundaries_centered


def format_c_array(name, values, type_str="float"):
    """Format as C static const array."""
    vals = ", ".join(f"{v: .10f}f" for v in values)
    return f"static const {type_str} {name}[{len(values)}] = {{{vals}}};"


if __name__ == "__main__":
    print("=" * 70)
    print("Lloyd-Max Codebook Generation for TQ3_0")
    print("=" * 70)
    
    for dim in [128, 256]:
        print(f"\n--- dim = {dim}, Beta({dim//2-1}, {dim//2-1}), 3-bit (8 levels) ---")
        centroids, boundaries = compute_lloyd_max_codebook(dim, bits=3)
        
        print(f"\nCentroids (8 reconstruction levels):")
        for i, c in enumerate(centroids):
            print(f"  [{i}] = {c: .10f}")
        
        print(f"\nBoundaries (7 decision thresholds):")
        for i, b in enumerate(boundaries):
            print(f"  [{i}] = {b: .10f}")
        
        # Verify symmetry (should be symmetric around 0 for Beta(a,a))
        print(f"\nSymmetry check:")
        for i in range(len(centroids) // 2):
            j = len(centroids) - 1 - i
            print(f"  c[{i}] + c[{j}] = {centroids[i] + centroids[j]:.2e} (should be ~0)")
        
        print(f"\n// C code for dim={dim}:")
        print(format_c_array(f"tq3_centroids_{dim}", centroids))
        print(format_c_array(f"tq3_boundaries_{dim}", boundaries))
    
    # Also generate for "generic" case — Gaussian approximation
    # For large d, Beta(d/2-1, d/2-1) ~ N(0.5, 1/(2*sqrt(d-2)))
    # This is the fallback for unusual head dimensions
    print(f"\n--- Gaussian approximation (fallback for any dim) ---")
    dist = stats.norm(0, 1)
    n_levels = 8
    quantiles = np.linspace(0, 1, n_levels + 1)
    centroids_init = np.array([
        dist.ppf((quantiles[i] + quantiles[i+1]) / 2) for i in range(n_levels)
    ])
    
    for _ in range(200):
        boundaries = (centroids_init[:-1] + centroids_init[1:]) / 2.0
        full_bounds = np.concatenate([[-6.0], boundaries, [6.0]])
        new_c = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = full_bounds[i], full_bounds[i+1]
            prob = dist.cdf(hi) - dist.cdf(lo)
            if prob < 1e-15:
                new_c[i] = (lo + hi) / 2
            else:
                num, _ = integrate.quad(lambda x: x * dist.pdf(x), lo, hi)
                new_c[i] = num / prob
        if np.max(np.abs(new_c - centroids_init)) < 1e-12:
            break
        centroids_init = new_c
    
    boundaries = (centroids_init[:-1] + centroids_init[1:]) / 2.0
    print(f"\nGaussian Lloyd-Max centroids:")
    for i, c in enumerate(centroids_init):
        print(f"  [{i}] = {c: .10f}")
    print(f"\nGaussian Lloyd-Max boundaries:")
    for i, b in enumerate(boundaries):
        print(f"  [{i}] = {b: .10f}")
    
    print(f"\n// C code (Gaussian fallback):")
    print(format_c_array("tq3_centroids_gauss", centroids_init))
    print(format_c_array("tq3_boundaries_gauss", boundaries))
    
    # Compute theoretical distortion for comparison
    print(f"\n\n--- Distortion comparison (dim=128) ---")
    dim = 128
    alpha = dim / 2 - 1
    dist_beta = stats.beta(alpha, alpha)
    sigma = dist_beta.std()
    
    # Lloyd-Max distortion
    centroids_128, boundaries_128 = compute_lloyd_max_codebook(dim, bits=3)
    centroids_128_raw = centroids_128 + 0.5  # shift back to [0,1]
    boundaries_128_raw = boundaries_128 + 0.5
    full_b = np.concatenate([[0.0], boundaries_128_raw, [1.0]])
    
    distortion_lm = 0.0
    for i in range(8):
        lo, hi = full_b[i], full_b[i+1]
        d, _ = integrate.quad(
            lambda x: (x - centroids_128_raw[i])**2 * dist_beta.pdf(x), lo, hi
        )
        distortion_lm += d
    
    # Uniform quantizer distortion (for comparison)
    uniform_centroids = np.linspace(1/16, 15/16, 8)
    uniform_boundaries = (uniform_centroids[:-1] + uniform_centroids[1:]) / 2.0
    full_b_u = np.concatenate([[0.0], uniform_boundaries, [1.0]])
    
    distortion_uniform = 0.0
    for i in range(8):
        lo, hi = full_b_u[i], full_b_u[i+1]
        d, _ = integrate.quad(
            lambda x: (x - uniform_centroids[i])**2 * dist_beta.pdf(x), lo, hi
        )
        distortion_uniform += d
    
    print(f"  Variance of Beta({alpha},{alpha}): {sigma**2:.6e}")
    print(f"  Lloyd-Max MSE: {distortion_lm:.6e}")
    print(f"  Uniform MSE:   {distortion_uniform:.6e}")
    print(f"  Improvement:   {(1 - distortion_lm/distortion_uniform)*100:.2f}%")
    print(f"  SNR (Lloyd-Max): {10*np.log10(sigma**2/distortion_lm):.2f} dB")
    print(f"  SNR (Uniform):   {10*np.log10(sigma**2/distortion_uniform):.2f} dB")
