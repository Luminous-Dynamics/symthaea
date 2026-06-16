# Nalgebra Migration Analysis: Eigenvalue Variance Investigation

**Date**: January 3, 2026
**Status**: ROOT CAUSE IDENTIFIED

## Summary

Replacing the custom eigenvalue implementation with `nalgebra::SymmetricEigen` produces **mathematically correct** eigenvalues, but reveals that **normalized Laplacian algebraic connectivity does not match IIT predictions**.

## Key Findings

### 1. Original vs New Implementation Results

| Topology | Original Φ | New Φ | Original λ₂ (estimated) | New λ₂ |
|----------|------------|-------|-------------------------|--------|
| Star     | 0.4543     | 0.5647| ~0.91                   | 1.1294 |
| Random   | 0.4318     | 0.5689| ~0.86                   | 1.1379 |
| **Δ**    | **+5.20%** | **-0.77%** | - | - |

### 2. Root Cause: Original Implementation Had Numerical Issues

The original `find_algebraic_connectivity()` method used **inverse power iteration with shift**:

```rust
// Add small shift to make matrix invertible (since λ₁ = 0)
let shift = 1e-6;
```

**Problems identified**:
1. **Shift too small**: 1e-6 is insufficient for stable inverse iteration
2. **Numerical instability**: The LU decomposition solve accumulated errors
3. **Projection errors**: Orthogonalization to null space was imprecise

These errors **coincidentally** produced results favoring Star topology.

### 3. nalgebra Produces Correct Eigenvalues

Verified with K₃ (complete graph):
- Expected: λ = [0, 1.5, 1.5]
- nalgebra: λ = [0, 1.5, 1.5] ✓

### 4. Why Normalized Laplacian Favors Random

The normalized Laplacian L_norm = I - D^(-1/2) A D^(-1/2) has eigenvalues in [0, 2]:

- **Star topology**: Hub has degree (n-1), spokes have degree 1
  - Degree normalization "penalizes" the hub's high connectivity
  - Results in λ₂ = 1.1294

- **Random topology**: More uniform degree distribution
  - Less penalty from degree normalization
  - Results in λ₂ = 1.1379 (higher!)

**Mathematical insight**: Normalized Laplacian measures **mixing time** (how quickly a random walk reaches equilibrium), NOT IIT-style "integration".

### 5. IIT vs Spectral Graph Theory

| Property | IIT Φ | Normalized Laplacian λ₂ |
|----------|-------|-------------------------|
| Measures | Integrated information | Mixing time / spectral gap |
| Star prediction | HIGH (bottleneck = integration) | LOW (non-uniform degrees) |
| Random prediction | LOW (reducible) | MODERATE (uniform degrees) |
| Uniform k-regular | Variable | Favored |

## Decision: Accept nalgebra as Correct

The nalgebra implementation is **mathematically correct**. The variance reflects:
1. The original implementation was numerically buggy
2. Normalized algebraic connectivity ≠ IIT-style Φ

## Implications for Symthaea

### Option A: Accept New Results (Recommended)
- Document that RealHV Φ uses algebraic connectivity (mixing time metric)
- Note it does NOT directly correspond to IIT's Φ
- The probabilistic binary Φ method still shows Star > Random (+5.52%)

### Option B: Alternative Metrics (Future Work)
- Implement unnormalized Laplacian: L = D - A
- Implement actual IIT MIP calculation (expensive)
- Implement weighted connectivity preserving absolute similarity

## Updated Documentation

The following should be updated in project docs:

1. **phi_real.rs**: Update docstring to clarify metric is algebraic connectivity
2. **CLAUDE.md**: Note that RealHV Φ measures spectral gap, not IIT Φ
3. **Validation results**: Update to reflect correct measurements

## Code Quality Improvement

Despite the result variance, the nalgebra integration provides:
- ✅ 120+ lines of complex eigenvalue code removed
- ✅ Validated, numerically stable library
- ✅ Correct eigenvalue computation
- ✅ Better maintainability
- ✅ Reduced technical debt

## Conclusion

The original implementation's "correct" results (Star > Random) were a happy accident caused by numerical bugs. The mathematically correct results from nalgebra reveal that:

1. **Normalized algebraic connectivity** does not match IIT predictions
2. **Probabilistic binarization** (in binary Φ calculator) still validates the hypothesis
3. The metric measures **mixing time**, not **integration**

This is valuable scientific insight: different mathematical formulations of "connectivity" produce different results, and care must be taken when choosing metrics for consciousness measurement.
