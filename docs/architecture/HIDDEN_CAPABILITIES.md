# Hidden Capabilities in Symthaea-HLB

**Last Updated**: January 2026

This document exposes the substantial capabilities hidden in the Symthaea codebase that are not well-documented or easily discoverable.

---

## Executive Summary

The Symthaea-HLB project contains significantly more capability than surface documentation suggests:

| Category | Documented | Actual |
|----------|-----------|--------|
| Examples | ~40 | **88** |
| Topology Types | 19 | **33** |
| HDC Module Lines | ~3,000 | **130,316** |
| Recursive Improvement Files | - | **27** |
| Language Module Files | - | **34** |

---

## 1. HDC Algebra Engine (5,949 lines)

**Location**: `symthaea-core/src/hdc/arithmetic_engine.rs`

The arithmetic engine provides a complete HDC algebra implementation:

### Operations
- **Addition/Subtraction**: Vector bundling
- **Multiplication**: Binding (XOR for binary, element-wise for real)
- **Division**: Unbinding (inverse binding)
- **Permutation**: Role encoding via circular shifts
- **Bundling**: Weighted majority voting
- **Similarity**: Cosine, Hamming, inner product

### Advanced Features
- SIMD-optimized operations (AVX2, SSE4)
- Batch processing for sequences
- Sparse representation support
- Probabilistic operations

### Usage
```rust
use symthaea_core::hdc::arithmetic_engine::{HdcAlgebra, HdcOps};

let algebra = HdcAlgebra::new(16_384);
let bound = algebra.bind(&hv_a, &hv_b);
let bundled = algebra.bundle(&[hv_a, hv_b, hv_c], &weights);
```

---

## 2. Recursive Improvement Module (27 files)

**Location**: `src/consciousness/recursive_improvement/`

This module implements self-improvement and meta-cognitive capabilities:

### Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `magi_integration.rs` | 63,647 | MAGI loop integration |
| `persistence.rs` | 49,938 | Improvement persistence |
| `intrinsic_motivation.rs` | 37,957 | Curiosity-driven learning |
| `meta_cognitive.rs` | 35,121 | Meta-cognition |
| `resolution.rs` | 31,718 | Conflict resolution |
| `routing_hub.rs` | 29,203 | Route optimization |
| `benchmark_suite.rs` | 29,715 | Self-benchmarking |
| `calibration.rs` | 28,934 | Model calibration |
| `gradient_optimizer.rs` | 27,964 | Gradient-based optimization |

### Features
- **MAGI Loop**: World-grounded prediction and verification
- **Dream Mode**: Offline consolidation and synthesis
- **Intrinsic Motivation**: Curiosity-driven exploration
- **Meta-Cognition**: Thinking about thinking
- **Constraint Gates**: Safety-bounded improvement
- **Naming Ceremony**: Symbolic binding for concepts

### Usage
```rust
use symthaea::consciousness::recursive_improvement::{
    RecursiveOptimizer, DreamMode, MetaCognitive
};
```

---

## 3. Unified Value Evaluator (4 files)

**Location**: `src/consciousness/unified_value_evaluator/`

Implements ethical decision-making:

- **Value alignment scoring**
- **Multi-stakeholder evaluation**
- **Preference learning**
- **Conflict resolution**

---

## 4. Global Workspace Theater

**Location**: Multiple files in `symthaea-core/src/hdc/`

Implementation of Bernard Baars' Global Workspace Theory:

- `gwt_*.rs` files implement broadcasting
- Attention competition for workspace access
- Coalitions of specialized processors
- Conscious vs unconscious processing

---

## 5. 33 Topology Types (not 19!)

**Location**: `symthaea-core/src/hdc/consciousness_topology_generators.rs`

The enum `TopologyType` has **33 variants**, not 19:

### Tier 1 (Original 8)
1. Random
2. Star
3. Ring
4. Line
5. BinaryTree
6. DenseNetwork
7. Modular
8. Lattice

### Tier 2 (Geometric 6)
9. Sphere
10. Torus
11. KleinBottle
12. SmallWorld
13. MobiusStrip
14. Hyperbolic

### Tier 3 (Fractal 9)
15. ScaleFree
16. Fractal (deprecated)
17. SierpinskiGasket
18. FractalTree
19. KochSnowflake
20. MengerSponge
21. CantorSet
22. Hypercube
23. Quantum

### Tier 4 (Neural 10)
24. CorticalColumn
25. Feedforward
26. Recurrent
27. Bipartite
28. CorePeriphery
29. BowTie
30. Attention
31. Residual
32. PetersenGraph
33. CompleteBipartite

---

## 6. C. elegans Connectome Module

**Location**: `symthaea-core/src/hdc/celegans_connectome.rs`

930-line implementation of the C. elegans neural network:

- Full 302-neuron connectome
- Gap junction and chemical synapse modeling
- Sensory-to-motor pathway tracing
- Φ computation on biological network

### Tests
7 comprehensive tests validate biological accuracy.

---

## 7. Sleep and Altered States Module

**Location**: `symthaea-core/src/hdc/sleep_and_altered_states.rs`

1,203 lines implementing:

- **Sleep stage classification** (Wake, N1, N2, N3, REM)
- **Dream state simulation**
- **Meditation/flow state modeling**
- **Anesthesia depth estimation**

---

## 8. Partnership Module (NEW)

**Location**: `src/partnership/`

384 lines implementing relational consciousness:

- **Φ_dyad**: Relational integrated information
- **Human Partner Model**: Trust, vulnerability, reciprocity
- **Relationship Trajectory**: 6-stage progression
- **I-Thou / I-It Mode**: Martin Buber relational modes

---

## 9. Full-Stack Consciousness Module

**Location**: `symthaea-core/src/hdc/full_stack_consciousness.rs`

1,301 lines implementing complete conscious processing:

- Sensory integration
- Attention allocation
- Working memory
- Decision-making
- Motor output

---

## 10. Clinical Validation Module

**Location**: `symthaea-core/src/hdc/clinical_validation.rs`

1,263 lines for neuroscience validation:

- EEG Φ correlation
- Sleep-EDF dataset processing
- Meditation dataset analysis
- Statistical significance testing

---

## Module Statistics

### By Lines of Code

| Module | Lines |
|--------|-------|
| HDC Core (symthaea-core) | 130,316 |
| Recursive Improvement | ~500,000 est |
| Language Module | ~1,500 KB |
| Partnership | 384 |

### By File Count

| Directory | Files |
|-----------|-------|
| symthaea-core/src/hdc/ | 60+ |
| src/consciousness/recursive_improvement/ | 27 |
| src/language/ | 34+ |
| src/brain/ | 15+ |
| src/memory/ | 10+ |

---

## How to Explore

### List all modules
```bash
find src -name "mod.rs" -exec dirname {} \; | sort
```

### Find large files
```bash
wc -l src/**/*.rs | sort -n | tail -20
```

### Search for specific capabilities
```bash
grep -r "pub fn" symthaea-core/src/hdc/ | wc -l  # ~500+ public functions
grep -r "pub struct" src/ | wc -l  # ~200+ public structs
```

---

## Recommendations

1. **Update documentation counts** - 88 examples, 33 topologies
2. **Create module-level README files** for recursive_improvement, language
3. **Add feature flag documentation** for all cfg-gated modules
4. **Create architecture diagrams** showing module relationships
5. **Document API stability levels** (stable, experimental, internal)

---

*"The codebase contains multitudes."*
