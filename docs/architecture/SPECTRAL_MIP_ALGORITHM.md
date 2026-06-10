# Spectral MIP Algorithm — O(n³) Minimum Information Partition via Fiedler Ordering

## Overview

The Spectral MIP Finder computes the **Minimum Information Partition (MIP)** of a
system's covariance structure in O(n³) time, replacing the NP-hard O(2^n) exhaustive
search. This enables real-time Integrated Information (Φ) computation within the
cognitive loop's 50Hz (20ms) budget.

**Location**: `symthaea-core/src/consciousness_metrics/spectral_mip.rs`
**Tests**: `symthaea-core/src/consciousness_metrics/tests/spectral_mip_tests.rs`
**Benchmarks**: `symthaea-core/benches/spectral_mip.rs`

## Motivation

Integrated Information Theory (IIT) defines Φ as the difference between a system's
total mutual information and the mutual information of its weakest bipartition (the MIP):

```
Φ = MI_total(X) - MI(X_A; X_B) where (A, B) = argmin MI(X_A; X_B)
```

Finding the MIP requires evaluating all 2^(n-1) - 1 bipartitions — exponential in the
number of subsystems. Previous approaches either:
- **Exhaustive search** (impractical beyond n ≈ 20)
- **Greedy/heuristic** (no quality guarantees)
- **Subsampling** (old approach: 64 of 16,384 dimensions = 0.4% coverage)

The spectral approach achieves O(n³) with principled partition quality by exploiting
the graph-theoretic structure of mutual information.

## Algorithm

### Step 1: Covariance Estimation

Maintain a sliding window of T HDC state snapshots (default T=50). Compute the n×n
sample covariance matrix with Tikhonov regularization (ε = 10⁻⁶):

```
Σ̂ = (1/T) Σ_t (x_t - μ)(x_t - μ)ᵀ + εI
```

**Complexity**: O(n² · T) — amortized O(n²) per push, O(n² · T) at compute time.

### Step 2: MI Laplacian

Compute the pairwise Gaussian mutual information matrix and its graph Laplacian.

For Gaussian variables, MI between dimensions i and j is:

```
MI(i, j) = -½ ln(1 - ρ²ᵢⱼ)
```

where ρᵢⱼ = Σᵢⱼ / √(Σᵢᵢ · Σⱼⱼ) is the Pearson correlation.

The MI Laplacian L = D - W, where W[i,j] = MI(i,j) and D = diag(Σⱼ W[i,j]).

**Complexity**: O(n²)

### Step 3: Fiedler Ordering (Key Insight)

The **Fiedler vector** (second-smallest eigenvector of L) provides a 1D embedding that
clusters tightly-coupled dimensions together. This transforms the NP-hard graph bisection
problem into finding the weakest link in a 1D chain.

**Theorem** (Fiedler, 1973): The Fiedler vector of a connected graph's Laplacian minimizes
the ratio cut objective. Sorting vertices by Fiedler value produces an ordering where
highly-connected groups are contiguous.

**Implementation**: Shifted inverse iteration with deflation.

1. Shift: M = L + σI (σ = 10⁻⁸) to make the matrix positive definite
2. Cholesky factorization: M = LLᵀ — one-time O(n³/6) cost
3. Inverse iteration with deflation:
   - Start with v orthogonal to the constant eigenvector (which has eigenvalue 0)
   - Repeat 30 times: solve Mw = v via Cholesky (O(n²)), deflate, normalize
   - Converges to the Fiedler vector (eigenvector of smallest nonzero eigenvalue)
4. Sort dimensions by Fiedler value → spectral order

**Complexity**: O(n³/6 + 30·n²) ≈ O(n³/6)

**Fallback**: If Cholesky fails (non-PD matrix), falls back to full eigendecomposition
via nalgebra's SymmetricEigen — still O(n³) but with larger constant.

### Step 4: Bordered Cholesky Sweep (Novel Contribution)

With dimensions in Fiedler order, we only need to evaluate n-1 **contiguous** bipartitions
(cuts at each bond in the 1D chain). This is justified because the Fiedler ordering places
the weakest information link at a contiguous boundary.

For each cut k (left = {0,...,k}, right = {k+1,...,n-1}):

```
MI_cut(k) = MI_left(k) + MI_right(k)
```

where MI of a block = ½(Σ ln(σ²ᵢ) - ln(det(Σ_block))).

**Key optimization**: Each successive block grows by one dimension. The determinant
can be updated via **bordered Cholesky**:

Given the Cholesky factor L_k of the k×k sub-covariance, extending to (k+1) dimensions:

```
L_{k+1} = [[L_k, 0], [xᵀ, l_new]]

where: L_k · x = c   (forward substitution, O(k²))
       l_new = √(d - xᵀx)
       c = Σ[0:k, k]  (cross-covariance column)
       d = Σ[k, k]     (new diagonal element)
```

This gives ln(det) incrementally: `ln_det(k+1) = ln_det(k) + 2·ln(l_new)`.

**Left sweep** grows from index 0 → n-1. **Right sweep** grows from index n-1 → 0.
Both sweeps are independent and run in parallel via `rayon::join`.

**Complexity**: O(Σ_{k=1}^{n} k²) = O(n³/3), parallelized to O(n³/6) wall time.

### Step 5: MIP Selection

```
MIP = argmin_k MI_cut(k)
Φ = MI_total - MI_cut(MIP)
```

**Complexity**: O(n) scan.

### Total Complexity

| Step | Operation | Complexity |
|------|-----------|------------|
| 1 | Covariance matrix | O(n²·T) |
| 2 | MI Laplacian | O(n²) |
| 3 | Fiedler ordering | O(n³/6) |
| 4 | Bordered Cholesky sweep | O(n³/6) parallel |
| 5 | MIP selection | O(n) |
| **Total** | | **O(n³)** |

## Benchmarks

Measured on AMD Ryzen (single-threaded equivalent):

| Operation | n=64 | n=128 | n=256 |
|-----------|------|-------|-------|
| push (per snapshot) | 239 ns | 442 ns | 772 ns |
| full pipeline | — | 5.50 ms | — |

The n=128 full pipeline at 5.50ms is well within the 20ms budget, allowing computation
every 50 cycles (1 second at 50Hz) with negligible impact on throughput.

## Integration in Cognitive Loop

```
cycle.rs (every cycle):
  spectral_mip_finder.push(&encoding_result.hdv)     // O(n) subsample + push

cycle.rs (every 50 cycles):
  let result = spectral_mip_finder.compute()          // O(n³), ~5.5ms
  carryover.last_spectral_mip_phi = result.phi        // cached for inter-cycle use
  carryover.last_sigma = result.phi                   // backward compat for memory coordinator
```

The SpectralMIPFinder maintains a sliding window (default 50 snapshots) of subsampled
HDC state vectors (128 of 16,384 dimensions, evenly spaced). Computation is amortized
over 50 cycles, giving an effective per-cycle cost of ~110μs.

## Configuration

```rust
SpectralMIPConfig {
    num_components: 128,   // Dimensions tracked (subsampled from 16,384)
    window_size: 50,       // Sliding window of state snapshots
    min_samples: 10,       // Minimum window fill before computing
    regularization: 1e-6,  // Tikhonov regularization on covariance diagonal
}
```

## Result Structure

```rust
SpectralMIPResult {
    phi: f64,                    // Φ = MI_total - MI_mip (non-negative)
    total_mi: f64,               // Total Gaussian MI of the system
    mip_mi: f64,                 // MI at the minimum information partition
    mip: TruePartition,         // { part_a: Vec<usize>, part_b: Vec<usize> }
    spectral_order: Vec<usize>,  // Fiedler-sorted dimension indices
    mip_bond: usize,             // Cut position in spectral order
    fiedler_zero_crossing: usize,// Where Fiedler vector changes sign
    cut_mis: Vec<f64>,           // MI at each of n-1 contiguous cuts
    window_used: usize,          // Number of snapshots used
    num_components: usize,       // Number of dimensions tracked
}
```

## Property-Based Testing

Seven proptest properties verify algorithm invariants:

1. **phi_nonnegative**: Φ ≥ 0 and finite for all valid covariance matrices
2. **total_mi_finite**: Total MI is finite and non-negative
3. **partition_valid**: MIP partition covers all elements, non-overlapping, both non-empty
4. **cut_mis_valid**: Exactly n-1 cut MIs, all finite
5. **mip_mi_leq_total_mi**: MIP MI ≤ total MI (Φ ≥ 0 by construction)
6. **bordered_cholesky_finite**: ln(det) is finite for all sub-matrices of PD input
7. **spectral_order_is_permutation**: Spectral order is a valid permutation of [0, n)

## Theoretical Justification

### Why Fiedler Ordering Works for MIP

The MIP seeks the bipartition minimizing cross-partition MI. The Fiedler vector of the
MI Laplacian provides the optimal 2-way normalized cut relaxation (Shi & Malik, 2000).
Dimensions with similar Fiedler values share strong mutual information; the MIP naturally
falls at a boundary where Fiedler values change most rapidly.

For Gaussian systems (our case), the MI structure is fully determined by the correlation
matrix. The spectral ordering preserves this structure: highly correlated dimensions
cluster together, and the weakest information link becomes a contiguous cut boundary.

### Approximation Quality

The spectral MIP is an **approximation** — it only searches contiguous bipartitions in
Fiedler order (n-1 candidates) rather than all 2^(n-1)-1 possibilities. The approximation
is tight when:

1. The MI graph has clear community structure (common in neural systems)
2. The Fiedler gap (λ₂/λ₃) is large (indicating well-separated clusters)
3. The system has hierarchical organization (natural in HDC encodings)

For adversarial inputs (uniform random correlation), the approximation may miss the true
MIP. However, real HDC state trajectories exhibit strong temporal and spatial correlation
patterns that make the spectral approach highly effective.

### Comparison with Kitazono et al. (2018)

Kitazono et al. proposed hierarchical bipartitioning for MIP search. Our approach differs:

1. **Fiedler ordering** provides a principled dimension ordering (vs. arbitrary hierarchy)
2. **Bordered Cholesky** gives O(k²) per cut update (vs. O(k³) for fresh determinants)
3. **Parallel left/right sweeps** exploit rayon for 2× wall-time improvement
4. **Sliding window** enables online computation (vs. batch processing)

## References

- Fiedler, M. (1973). "Algebraic connectivity of graphs." *Czechoslovak Mathematical Journal*, 23(2), 298-305.
- Kitazono, J., et al. (2018). "Efficient algorithms for searching the minimum information partition in Integrated Information Theory." *Entropy*, 20(3), 173.
- Oizumi, M., et al. (2016). "Measuring integrated information from the decoding perspective." *PLOS Computational Biology*, 12(1).
- Shi, J. & Malik, J. (2000). "Normalized cuts and image segmentation." *IEEE TPAMI*, 22(8), 888-905.
- Tononi, G. (2008). "Consciousness as integrated information: A provisional manifesto." *Biological Bulletin*, 215(3), 216-242.
