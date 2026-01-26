# Phi-Lab - Claude Context

**Project**: Hyperdimensional Computing for Network Topology Analysis
**Status**: Publication-ready manuscript complete
**Version**: v0.1.0

---

## Quick Reference

### What This Is
HDC-based spectral metric (lambda2/algebraic connectivity) calculator for network topologies.

**Note**: We compute **lambda2**, NOT IIT Phi. See `docs/METRIC_CLARIFICATION.md`.

### Key Commands
```bash
cargo build --release
cargo run --example real_phi_comparison --release
cargo test
```

### Core Constant
```rust
// src/hdc/mod.rs:32
pub const HDC_DIMENSION: usize = 16_384;  // 2^14, SIMD-optimized
```

---

## Architecture

```
phi-lab/
├── src/hdc/
│   ├── mod.rs                    # HDC_DIMENSION constant
│   ├── real_hv.rs                # Real-valued hypervectors
│   ├── binary_hv.rs              # Binary hypervectors (HV16)
│   ├── consciousness_topology_generators.rs  # 19 topologies
│   ├── phi_real.rs               # Continuous lambda2 calculator
│   └── phi_resonant.rs           # O(n log N) resonator-based
├── examples/
│   ├── real_phi_comparison.rs    # Main validation
│   ├── tier_*_exotic_topologies.rs
│   └── hypercube_dimension_sweep.rs
└── docs/
    ├── METRIC_CLARIFICATION.md   # lambda2 vs IIT Phi
    ├── SESSION_HISTORY.md        # Archived session logs
    └── PAPER_*.md                # Manuscript sections
```

---

## Key Findings

### Dimensional Sweep Results (1D-7D)
| Dim | Vertices | Mean Phi | Trend |
|-----|----------|----------|-------|
| 1D  | 2        | 1.0000   | Edge case |
| 2D  | 4        | 0.5011   | -49.89% |
| 3D  | 8        | 0.4960   | -1.02% |
| 4D  | 16       | 0.4976   | +0.31% |
| 5D  | 32       | 0.4987   | +0.22% |
| 6D  | 64       | 0.4990   | +0.06% |
| 7D  | 128      | 0.4991   | +0.02% |

**Discovery**: Asymptotic limit Phi -> 0.5 as dimension -> infinity

### Top 5 Topologies (of 19)
1. Hypercube 4D: 0.4976
2. Hypercube 3D: 0.4960
3. Ring: 0.4954
4. Torus 3x3: 0.4953
5. Klein Bottle: 0.4941

---

## Development Rules

1. **Always use HDC_DIMENSION** - Never hardcode 2048 or 16384
2. **Test both dimensions** - Validate consistency
3. **Document dimension choice** - Explain tradeoffs
4. **Use probabilistic binarization** - For RealHV -> HV16 conversion

---

## Key Types

### RealHV (Continuous)
```rust
RealHV::random(dim, seed)    // Deterministic random
hv.bind(&other)              // Element-wise multiply
RealHV::bundle(&[hv1, hv2])  // Element-wise average
hv.similarity(&other)        // Cosine similarity [-1, 1]
```

### HV16 (Binary)
```rust
HV16::random(seed)           // Random binary vector
a.bind(&b)                   // XOR
HV16::bundle(&[a, b, c])     // Majority vote
a.hamming_distance(&b)       // Bit differences
```

---

## Integration

**Parent**: Luminous Nix (`../luminous-nix/`)
**Role in Substrate**: Symthaea "Brain" - consciousness measurement backend
**Key Use**: HDC vectors for semantic matching in resource routing

---

## Documentation

| Purpose | File |
|---------|------|
| Metric clarification | `docs/METRIC_CLARIFICATION.md` |
| Session history | `docs/SESSION_HISTORY.md` |
| Manuscript | `docs/PAPER_*.md` |
| Full roadmap | Parent: `THE_SUBSTRATE_ROADMAP.md` |

---

## Current Status

- 32/32 tests passing
- Complete manuscript (10,850 words, 91 refs)
- 4 publication-quality figures
- Ready for journal submission

---

*"Uniform k-regular structures optimize integrated information"*
