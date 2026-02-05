# Φ Implementation Audit: Consistency Analysis

**Date**: January 3, 2026
**Status**: INCONSISTENCY IDENTIFIED - ACTION REQUIRED

## Summary

The codebase has **two fundamentally different metrics** both called "Φ":

1. **Spectral Φ** (Algebraic Connectivity) - Measures mixing time
2. **IIT Φ** (Integrated Information) - Measures integration

These produce **opposite predictions** for Star vs Random topologies!

---

## Implementation Inventory

### 1. phi_real.rs - "RealPhiCalculator"
| Property | Value |
|----------|-------|
| **Algorithm** | Normalized Laplacian algebraic connectivity (λ₂) |
| **Complexity** | O(n³) eigenvalue decomposition |
| **What it measures** | Spectral gap / mixing time |
| **Star vs Random** | Random > Star (uniform degrees favored) |
| **IIT-aligned?** | ❌ NO - opposite of IIT predictions |

### 2. tiered_phi.rs - "TieredPhi"
| Tier | Algorithm | Complexity | IIT-aligned? |
|------|-----------|------------|--------------|
| Mock | Deterministic | O(1) | N/A (testing) |
| Heuristic | 1 - avg_similarity | O(n) | ❓ Unclear |
| **Spectral** | **Algebraic connectivity** | O(n²) | ❌ NO |
| Exact | MIP search | O(2^n) | ✅ YES |

### 3. integrated_information.rs - "IntegratedInformation"
| Property | Value |
|----------|-------|
| **Algorithm** | Φ = EI(System) - ΣEI(Parts) |
| **Complexity** | O(n²) to O(2^n) depending on partition search |
| **What it measures** | True integrated information |
| **Star vs Random** | Star > Random (bottleneck = integration) |
| **IIT-aligned?** | ✅ YES |

### 4. phi_topology_validation.rs - "PhiTopologyValidation"
| Property | Value |
|----------|-------|
| **Uses** | TieredPhi + RealPhiCalculator |
| **Binarization methods** | Mean, Median, Probabilistic, Quantile |
| **Key finding** | Probabilistic binarization: Star > Random (+5.52%) |
| **IIT-aligned?** | ✅ YES (with probabilistic binarization) |

### 5. phi_exact.rs - "PyPhiValidator"
| Property | Value |
|----------|-------|
| **Algorithm** | PyPhi (true IIT 3.0 MIP) |
| **Complexity** | O(2^n) - intractable for n > 8 |
| **IIT-aligned?** | ✅ YES (ground truth) |

### 6. phi_resonant.rs - "ResonatorPhi"
| Property | Value |
|----------|-------|
| **Algorithm** | Coupled oscillator energy convergence |
| **Complexity** | O(n log N) |
| **IIT-aligned?** | ❓ Needs validation |

### 7. phi_engine/calculator.rs - "PhiCalculator" trait
| Property | Value |
|----------|-------|
| **Purpose** | Unified interface |
| **Issue** | Doesn't distinguish spectral vs IIT |

---

## The Core Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TWO DIFFERENT "Φ" METRICS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   SPECTRAL Φ (Algebraic Connectivity)    │    IIT Φ (Integration)       │
│   ────────────────────────────────────   │   ──────────────────────     │
│   • Measures: Mixing time                │   • Measures: Integration    │
│   • Favors: Uniform k-regular            │   • Favors: Bottlenecks      │
│   • Star < Random                        │   • Star > Random            │
│   • Fast: O(n³)                          │   • Slow: O(2^n) exact       │
│                                                                         │
│   Used by:                               │   Used by:                   │
│   • phi_real.rs                          │   • integrated_information.rs│
│   • tiered_phi.rs (Spectral tier)        │   • phi_exact.rs (PyPhi)     │
│                                          │   • tiered_phi.rs (Exact)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Recommended Actions

### Priority 1: Naming Clarity (Immediate)

1. **Rename `phi_real.rs`** → Document as "Spectral Φ" or "Algebraic Connectivity"
   - ✅ DONE (updated docstrings)

2. **Update `tiered_phi.rs` Tier 2** documentation
   - Clarify "Spectral" tier uses algebraic connectivity, NOT IIT Φ
   - Add warning about opposite predictions

3. **Create naming convention**:
   - `spectral_phi` or `lambda2` for algebraic connectivity
   - `iit_phi` or `integrated_info` for true IIT

### Priority 2: Unified API (This Session)

Create a clear dual-metric API:

```rust
pub enum PhiMetric {
    /// Algebraic connectivity (λ₂ of normalized Laplacian)
    /// Fast O(n³), but NOT IIT-aligned
    Spectral,

    /// True IIT integrated information
    /// Slow O(2^n) exact, or O(n²) approximation
    IIT,
}

pub trait UnifiedPhiCalculator {
    fn compute(&self, hvs: &[RealHV], metric: PhiMetric) -> f64;
    fn metric_type(&self) -> PhiMetric;
}
```

### Priority 3: IIT-Aligned Approximation (Future)

Implement a **tractable IIT approximation** that:
- Is O(n²) or O(n³) (not O(2^n))
- Still correlates with IIT predictions (Star > Random)
- Options:
  - Unnormalized Laplacian connectivity
  - Greedy MIP approximation
  - Probabilistic partition sampling

---

## Validation Status

### Confirmed Working (IIT-aligned)
- ✅ `phi_topology_validation.rs` with **probabilistic binarization** → Star > Random (+5.52%)
- ✅ `phi_exact.rs` (PyPhi) → Ground truth IIT 3.0
- ✅ `integrated_information.rs` → HDC-adapted IIT

### Not IIT-Aligned (Spectral only)
- ❌ `phi_real.rs` → Random > Star (spectral gap metric)
- ❌ `tiered_phi.rs` Tier 2 (Spectral) → Same issue

### Unknown / Needs Validation
- ❓ `phi_resonant.rs` → Needs empirical validation against IIT
- ❓ `tiered_phi.rs` Tier 1 (Heuristic) → Unclear correlation

---

## Conclusion

The codebase has grown organically with multiple Φ implementations that measure **different things**. This needs consolidation:

1. **Clear naming** distinguishing spectral vs IIT metrics
2. **Unified interface** that makes the metric choice explicit
3. **Documentation** explaining when to use which
4. **Deprecation** of misleading implementations or clear warnings

The nalgebra integration is **correct** - it exposed a pre-existing conceptual inconsistency in the codebase.
