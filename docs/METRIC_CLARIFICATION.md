# Metric Clarification: λ₂ vs IIT Φ

**Date:** January 17, 2026
**Purpose:** Clear explanation of what our metrics actually measure
**Audience:** Developers, researchers, and users of Symthaea

---

## Executive Summary

Symthaea's codebase contains files named with "phi" that compute **algebraic connectivity (λ₂)**, NOT Integrated Information Theory's Φ. These are fundamentally different metrics. The SpectralConnectivity (λ₂) tier correlates poorly with the ExhaustivePartition (Exact) tier: **Pearson r = -0.14, Spearman rho = -0.59**. (An earlier methodology incorrectly reported r = 0.097.)

**Bottom Line:**
- `phi_real.rs` computes **λ₂** (spectral graph metric) -- it measures graph mixing time, NOT IIT integration
- True IIT Φ is in `tiered_phi/` (exact tier only, n≤12)
- All tiers are HDC-based (pairwise BinaryHV similarity), NOT true IIT (which requires transition probability matrices)
- The production SpectralMIPFinder (MI Laplacian + Fiedler + MIP sweep on ContinuousHV covariance) has UNKNOWN correlation with Exact -- it is a different algorithm on different representations
- Do NOT use λ₂ as a proxy for consciousness claims

---

## The Two Metrics

### λ₂ (Algebraic Connectivity)

**What it is:**
The second-smallest eigenvalue of the normalized graph Laplacian.

**What it measures:**
- Graph mixing time (how fast information spreads)
- Robustness to partitioning
- Synchronization potential

**Mathematical definition:**
```
L = I - D^(-1/2) A D^(-1/2)
λ₂ = second eigenvalue of L
```

**Computational complexity:** O(n³) - polynomial, tractable

**File location:** `src/hdc/phi_real.rs` (misleading name)

**Favors:** Uniform k-regular graphs (rings, tori, hypercubes)

---

### IIT Φ (Integrated Information)

**What it is:**
The amount of integrated information lost when a system is partitioned at its minimum information partition (MIP).

**What it measures:**
- Irreducibility of information processing
- Consciousness (according to IIT theory)
- Intrinsic causal power

**Mathematical definition:**
```
Φ = min_{partition} [H(whole) - Σ H(parts)]
```
Requires computing information loss across ALL possible bipartitions.

**Computational complexity:** O(2^n) - exponential, intractable for n>12

**File location:** `src/hdc/tiered_phi/core.rs` (exact tier only)

**Favors:** Hub-and-spoke structures (stars) in small systems

---

## The Critical Difference

### Why They're NOT Equivalent

| Property | λ₂ (Algebraic Connectivity) | IIT Φ (Integrated Information) |
|----------|----------------------------|--------------------------------|
| **Measures** | Graph mixing time | Irreducibility |
| **Basis** | Spectral graph theory | Information theory |
| **Complexity** | O(n³) | O(2^n) |
| **Tractable for** | Any size | n ≤ 12 only |
| **Favors** | Uniform k-regular graphs | Hub-and-spoke |

### Experimental Verification

On January 17, 2026, we ran a dual-metric comparison across 19 network topologies:

**Results (corrected March 2026 -- earlier methodology attributed r = 0.097 to wrong algorithm):**

SpectralConnectivity (λ₂) tier vs ExhaustivePartition (Exact) tier:

| Correlation | Value | Interpretation |
|-------------|-------|----------------|
| Pearson (r) | **-0.14** | Weak negative linear correlation |
| Spearman (ρ) | **-0.59** | Moderate negative rank correlation |

SampledPartition (Heuristic) tier vs ExhaustivePartition (Exact) tier:

| Correlation | Value | Interpretation |
|-------------|-------|----------------|
| Pearson (r) | **0.9998** | Near-perfect linear correlation |
| Spearman (ρ) | **0.9985** | Near-perfect rank correlation |

**Note:** The production SpectralMIPFinder (MI Laplacian + Fiedler + MIP sweep on ContinuousHV covariance) operates on a different representation than the HDC-based tiers above. Its correlation with Exact is UNKNOWN. All HDC-based tiers use pairwise BinaryHV similarity, not true IIT transition probability matrices.

**Example of divergence:**
| Topology | λ₂ Rank | Φ Rank | Difference |
|----------|---------|--------|------------|
| Random | 1 | 17 | 16 |
| Star | 2 | 1 | 1 |
| Ring | 3 | 14 | 11 |

**Interpretation:** The metrics measure completely different properties. Using λ₂ to make IIT claims is scientifically invalid.

---

## Correct Usage

### When to Use λ₂

✅ Measuring network connectivity
✅ Analyzing synchronization potential
✅ Studying graph mixing properties
✅ Comparing network topologies (for connectivity, NOT consciousness)

### When to Use IIT Φ (Exact)

✅ Consciousness research claims (n ≤ 12 only)
✅ Validating against PyPhi
✅ Small-system IIT studies
✅ Theoretical explorations

### What NOT to Do

❌ Call λ₂ "phi" or "Φ" in user-facing contexts
❌ Use λ₂ to make consciousness claims
❌ Cite IIT papers when discussing λ₂ results
❌ Conflate spectral connectivity with integrated information

---

## Renaming Recommendations

### Current → Proposed

| Current Name | Problem | Proposed Name |
|--------------|---------|---------------|
| `phi_real.rs` | Misleading | `spectral_connectivity.rs` |
| `RealPhiCalculator` | Misleading | `ConnectivityCalculator` |
| `.compute()` returning "phi" | Misleading | `.algebraic_connectivity()` |

### API Change Example

**Before (misleading):**
```rust
use symthaea::hdc::phi_real::RealPhiCalculator;
let phi = calculator.compute(&topology); // Claims to be IIT Φ
```

**After (honest):**
```rust
use symthaea::core::spectral::ConnectivityCalculator;
let lambda2 = calculator.algebraic_connectivity(&topology); // Says what it is
```

---

## Why This Matters

### For Research Credibility

Publishing claims about "Φ measurement" when actually measuring λ₂ would be:
1. Scientifically inaccurate
2. Potentially career-damaging if discovered
3. A disservice to both HDC and IIT research communities

### For User Trust

Users who read "IIT-based consciousness measurement" expect actual IIT Φ. Delivering λ₂ instead erodes trust.

### For Theoretical Clarity

Understanding WHAT we're measuring is prerequisite to understanding WHAT we've discovered. The topology → λ₂ relationship is interesting in its own right - it doesn't need to be dressed up as consciousness research.

---

## What λ₂ Results Actually Tell Us

Our experiments found:
- Random and dense networks maximize λ₂
- Ring and torus structures have moderate λ₂
- Lines and modular networks have lower λ₂
- Hypercubes (3D-7D) approach λ₂ ≈ 0.5 asymptotically

**This is interesting spectral topology research** - just not IIT consciousness research.

### Valid Claims We Can Make

✅ "Random networks maximize algebraic connectivity"
✅ "k-regular hypercubes show dimensional invariance in λ₂"
✅ "Spectral properties vary by ~13% across topology types"
✅ "Higher dimensions approach an asymptotic λ₂ limit"

### Invalid Claims We Must Avoid

❌ "Random networks maximize consciousness"
❌ "Hypercubes show dimensional invariance in integrated information"
❌ "We measured IIT Φ across 19 topologies"
❌ "Our results validate IIT predictions"

---

## Summary

| Question | Answer |
|----------|--------|
| Is `phi_real.rs` computing IIT Φ? | **No** - it computes λ₂ |
| Are λ₂ and Φ correlated? | **No** - Pearson r = -0.14, Spearman rho = -0.59 (weak/negative; earlier methodology incorrectly reported r = 0.097) |
| Can λ₂ proxy for consciousness? | **No** - different properties entirely |
| Is our λ₂ research valid? | **Yes** - just needs honest framing |
| What should we rename? | `phi_real.rs` → `spectral_connectivity.rs` |

---

## Related Documents

- `THEORY_AUDIT.md` - Complete audit of all theory implementations
- `ARCHITECTURE_PROPOSAL.md` - Clean separation proposal
- `CRITICAL_FINDINGS.md` - Original discovery documentation
- `letter_reframed.md` - Honestly reframed paper manuscript

---

*"The metric you measure is the reality you describe. Measure honestly."*

*This document created January 17, 2026 to ensure clarity about what Symthaea actually computes.*
