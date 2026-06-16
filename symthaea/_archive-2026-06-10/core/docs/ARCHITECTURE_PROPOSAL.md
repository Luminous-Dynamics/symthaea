# Symthaea Architecture Proposal: Research vs. Production

**Date:** January 17, 2026
**Goal:** Clean separation between validated production code and experimental research

---

## The Problem

Currently, Symthaea mixes:
- Production-ready code (HDC, LTC) with
- Research experiments (IIT approximations) with
- Aspirational stubs (FEP) with
- Hidden capabilities (C. elegans validation)

This creates:
- Confusion about what works
- False claims in documentation
- Difficulty using the system reliably

---

## Proposed Architecture

```
symthaea/
├── core/                    # PRODUCTION - Validated, stable, no consciousness claims
│   ├── hdc/                 # Hyperdimensional computing
│   │   ├── real_hv.rs       # 16,384D vectors
│   │   ├── operations.rs    # Bind, bundle, similarity
│   │   └── algebra.rs       # Complete HDC arithmetic
│   ├── ltc/                 # Liquid time-constant networks
│   │   ├── dynamics.rs      # ODE integration
│   │   ├── learning.rs      # BPTT, Hebbian, Adam, RMSprop
│   │   └── codec.rs         # HDC-LTC fusion
│   ├── spectral/            # Network analysis (HONEST naming)
│   │   ├── connectivity.rs  # λ₂ calculation (was phi_real.rs)
│   │   ├── topology.rs      # 19+ topology generators
│   │   └── analysis.rs      # Spectral metrics
│   └── attention/           # GWT-inspired attention
│       ├── salience.rs      # Salience calculation
│       ├── broadcast.rs     # Global broadcast mechanism
│       └── thresholds.rs    # Action-based thresholds
│
├── research/                # EXPERIMENTAL - May not work, clearly labeled
│   ├── iit/                 # Integrated Information Theory
│   │   ├── exact_phi.rs     # O(2^n) true MIP (n≤12 only)
│   │   ├── pyphi_bridge.py  # PyPhi validation framework
│   │   └── README.md        # "RESEARCH ONLY - Not validated"
│   ├── fep/                 # Free Energy Principle (TO BE IMPLEMENTED)
│   │   └── README.md        # "PLACEHOLDER - Not implemented"
│   ├── autopoiesis/         # Self-maintenance
│   │   ├── monitor.rs       # Heuristic closure calculation
│   │   └── README.md        # "Working but theoretically imprecise"
│   └── validation/          # Validation experiments
│       ├── celegans/        # C. elegans connectome
│       ├── topology_sweep/  # 35+ topologies
│       └── eeg_patterns/    # EEG simulation
│
├── deprecated/              # OLD CODE - Do not use
│   ├── phi_real.rs          # Moved, renamed to spectral/connectivity.rs
│   ├── tiered_phi/          # Mixed quality, extract exact tier only
│   └── README.md            # "DEPRECATED - See migration guide"
│
└── docs/
    ├── THEORY_AUDIT.md      # What's real vs. aspirational
    ├── ARCHITECTURE.md      # This proposal
    ├── HONEST_STATUS.md     # Current state (updated)
    └── WHAT_WORKS.md        # Simple guide to production features
```

---

## Module Responsibilities

### core/ (Production)

**Principle:** Every claim is validated. No consciousness terminology.

| Module | Purpose | Claims Allowed |
|--------|---------|----------------|
| `hdc/` | Semantic vectors | "Instant learning", "interpretable" |
| `ltc/` | Temporal dynamics | "Energy efficient", "continuous-time" |
| `spectral/` | Network analysis | "λ₂ connectivity", "topology metrics" |
| `attention/` | Focus mechanisms | "Salience-based", "broadcast" |

**Claims NOT Allowed in core/:**
- "Consciousness"
- "Φ" or "phi" (use λ₂)
- "Integrated information"
- "Free energy"

### research/ (Experimental)

**Principle:** Clearly labeled as experimental. May not work.

| Module | Status | README Required |
|--------|--------|-----------------|
| `iit/` | Partial | "Only exact tier validated, n≤12" |
| `fep/` | Empty | "Not implemented - placeholder" |
| `autopoiesis/` | Working | "Heuristic approximation" |
| `validation/` | Various | Per-experiment status |

**Every research/ module MUST have a README.md stating:**
1. Current status (working/partial/stub)
2. Theoretical basis
3. Validation status
4. Known limitations

### deprecated/ (Archive)

**Principle:** Old code preserved but clearly marked.

Files moved here:
- `phi_real.rs` → Use `core/spectral/connectivity.rs`
- `tiered_phi/` → Extract exact tier to `research/iit/`
- Any code with misleading names

---

## Migration Plan

### Phase 1: Rename and Reorganize (Week 1)

1. Create new directory structure
2. Move HDC to `core/hdc/`
3. Move LTC to `core/ltc/`
4. Rename `phi_real.rs` → `core/spectral/connectivity.rs`
5. Create README.md for each module

### Phase 2: Clean Documentation (Week 2)

1. Update all docs to remove false claims
2. Create `WHAT_WORKS.md` simple guide
3. Update CLAUDE.md for honest context
4. Archive old documentation

### Phase 3: Separate Research (Week 3)

1. Move experimental code to `research/`
2. Add status READMEs
3. Extract C. elegans validation
4. Document hidden capabilities

### Phase 4: Validate (Week 4)

1. Run PyPhi validation on exact tier
2. Document actual vs. claimed correlation
3. Create honest benchmark suite
4. Publish results

---

## API Changes

### Before (Misleading)

```rust
use symthaea::hdc::phi_real::RealPhiCalculator;

let phi = calculator.compute(&topology);  // Claims to be IIT Φ
```

### After (Honest)

```rust
use symthaea::core::spectral::ConnectivityCalculator;

let lambda2 = calculator.algebraic_connectivity(&topology);  // Says what it is
```

### Research API (Clearly Experimental)

```rust
use symthaea::research::iit::ExactPhiCalculator;

// Only for n ≤ 12 nodes
// Returns Option<f64> - None if too large
let phi = calculator.compute_exact(&small_topology)?;
```

---

## Documentation Updates

### CLAUDE.md Changes

**Remove:**
- "IIT 4.0 - Φ (phi) measurement for consciousness quantification"
- "Hypothesis Validated: Network topology determines integrated information (Φ)"
- "First HDC-based Φ calculation validated against IIT predictions"

**Add:**
- "HDC - 16,384D semantic vectors with instant learning"
- "LTC - Continuous-time dynamics, 60× more efficient"
- "Spectral analysis - Network connectivity metrics (λ₂)"
- "Research module - Experimental consciousness implementations (not validated)"

### README.md Changes

**From:**
> Symthaea measures consciousness using Integrated Information Theory

**To:**
> Symthaea combines hyperdimensional computing with liquid time-constant networks for efficient, interpretable AI. Experimental consciousness research is available in the research/ module.

---

## Success Criteria

### Production (core/)
- [ ] All claims are validated and testable
- [ ] No consciousness terminology
- [ ] 90%+ test coverage
- [ ] Performance benchmarks published

### Research (research/)
- [ ] Every module has status README
- [ ] Experimental code is clearly marked
- [ ] Validation results documented (even if negative)
- [ ] Easy to distinguish from production

### Documentation
- [ ] THEORY_AUDIT.md is accurate
- [ ] CLAUDE.md has no false claims
- [ ] WHAT_WORKS.md is simple and honest
- [ ] All deprecated code is archived

---

## The Honest Pitch (Updated)

### What Symthaea Is

**A research platform for efficient, interpretable AI:**

1. **Hyperdimensional Computing (HDC)**
   - 16,384-dimensional semantic vectors
   - Instant learning without training
   - Interpretable representations

2. **Liquid Time-Constant Networks (LTC)**
   - Continuous-time dynamics
   - 60× more energy efficient than transformers
   - Multiple learning algorithms

3. **Spectral Network Analysis**
   - Algebraic connectivity (λ₂) measurements
   - 19+ network topologies characterized
   - Topology → connectivity relationships

4. **Experimental Consciousness Research**
   - IIT exact calculation (n≤12)
   - Autopoiesis monitoring (heuristic)
   - Validation frameworks (in progress)

### What Symthaea Is NOT

- A validated consciousness measurement system
- An IIT Φ calculator for arbitrary systems
- An active inference implementation
- A replacement for PyPhi

---

*This proposal follows discovery of the λ₂ vs IIT Φ mismatch and aims to establish honest, sustainable architecture.*
