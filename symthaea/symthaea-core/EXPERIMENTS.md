# Symthaea Research Tracks

This document organizes the ~320K line codebase into distinct research tracks with clear purposes and honest naming.

## Overview

| Track | Purpose | Status |
|-------|---------|--------|
| **Spark Engineering** | Fusion reactor design assessment | Active |
| **Phenomenal Research** | Scientific consciousness hypotheses (H1, H2) | Active |
| **Cognitive Architecture** | GWT/IIT/HOT integration engine | Mature |
| **HDC Primitives** | Core hyperdimensional computing | Mature |
| **Φ Measurement** | Integrated information quantification | Mature |
| **Empirical Validation** | Bio/neural grounding | Research |
| **Phenomenology** | Experiential states (sleep, creativity) | Exploratory |

---

## Track 1: Spark Engineering (Applied)

**Purpose**: Fusion reactor design analysis using HDC for dimensionality reduction and coupling measurement.

**Note**: Previously misnamed "physics_consciousness_integration". These are engineering metrics, not consciousness.

### Modules

| Module | Description |
|--------|-------------|
| `design_integration.rs` | HDC-based design coupling metrics |
| `trajectory_analysis.rs` | Temporal evolution analysis |
| `coupled_physics.rs` | Multi-physics simulation |
| `uncertainty.rs` | Monte Carlo uncertainty quantification |
| `manufacturing.rs` | DFM assessment |
| `economics.rs` | LCOE, NPV analysis |
| `design_space.rs` | Pareto multi-objective optimization |

### Key Metrics

- **Integration Score**: Geometric mean of thermal/damage/geometry coherence
- **Trajectory Quality**: Temporal coherence of operational states
- **Coupling Index**: HDC similarity patterns (legacy name: phenomenal_index)

### Examples

```bash
cargo run --example design_integration_demo --release
cargo run --example trajectory_analysis_demo --release
cargo run --example engineering_assessment_demo --release
```

---

## Track 2: Phenomenal Research (Scientific)

**Purpose**: Test scientific hypotheses about consciousness using HDC topology analysis.

### Hypothesis H1: Phenomenal Topology

**Claim**: LLM internal representations for phenomenal concepts (e.g., "what it is like to see red") exhibit different topological properties than functional concepts (e.g., "recursive function").

**Status**: Infrastructure ready, awaiting neural-bridge LLM embedding integration.

**Files**:
- `consciousness_topology.rs` - Betti numbers, unity scores
- Concept corpora: 52 phenomenal, 52 functional concepts
- K-fold cross-validation framework

**Test**:
```bash
cargo test --release test_phenomenal_vs_functional_topology -- --nocapture
```

### Hypothesis H2: Binding Unity

**Claim**: HDC binding (⊗) produces representations with higher topological integration than bundling (⊕) for phenomenally unified concept pairs.

**Status**: Preliminary support (interaction effect +0.1).

**Files**:
- `phenomenal_binding_study.rs` - 2x2 ANOVA framework
- Concept pairs: 50 unified, 50 separate

**Test**:
```bash
cargo test --release test_phenomenal_binding_vs_bundling -- --nocapture
```

---

## Track 3: Cognitive Architecture (Theoretical)

**Purpose**: Unified consciousness engine combining multiple theories.

### Theories Integrated

| Theory | Module | Key Metric |
|--------|--------|------------|
| IIT (Tononi) | `integrated_information.rs` | Φ |
| GWT (Baars) | `global_workspace.rs` | Workspace access |
| HOT (Rosenthal) | `higher_order_thought.rs` | Meta-awareness |
| Predictive Processing | `predictive_coding.rs` | Prediction error |

### Key Finding: Bridge Hypothesis

Correlation r = -0.72 between bridge ratio and Φ. Optimal: 40-45% bridges.

**Cognitive Modes**:
| Mode | Bridge Ratio | Use Case |
|------|--------------|----------|
| DeepSpecialization | 22-25% | Expert flow |
| Balanced | 40-45% | Normal waking |
| GlobalAwareness | 60-65% | Meditation |

### Files

- `unified_consciousness_engine.rs` - Crown jewel integration
- `fractal_consciousness.rs` - Multi-scale self-similarity
- `adaptive_topology.rs` - Dynamic bridge adjustment

---

## Track 4: HDC Primitives (Infrastructure)

**Purpose**: Core hyperdimensional computing operations.

### Vector Types

| Type | Bits | Memory | Use Case |
|------|------|--------|----------|
| `HV16` | 16,384 binary | 2KB | Fast operations |
| `ContinuousHV` | 16,384 × f32 | 64KB | Gradients, topology |
| `RealHV` | Variable × f32 | Variable | Temporal binding |

### Operations

- **Bind (⊗)**: Element-wise XOR/multiply - creates associations
- **Bundle (⊕)**: Majority vote/average - creates superpositions
- **Similarity**: Hamming/cosine distance

### Key Modules

- `unified_hv.rs` - Unified interface
- `simd_hv16.rs` - SIMD-optimized (8x faster)
- `temporal_encoder.rs` - Circular time encoding
- `sequence_encoder.rs` - Order-preserving sequences

---

## Track 5: Φ Measurement (Tooling)

**Purpose**: Quantify integrated information at various fidelity levels.

### Tiered System

| Tier | Complexity | Use Case |
|------|------------|----------|
| 0: Mock | O(1) | Testing |
| 1: Heuristic | O(n) | Real-time |
| 2: Spectral | O(n²) | Analysis |
| 3: Exact | O(2^n) | Small systems |

### Files

- `tiered_phi/core.rs` - Multi-tier implementation
- `phi_gradient_learning.rs` - Gradient-based optimization
- `differentiable_phi.rs` - Soft-partitioned Φ

---

## Track 6: Empirical Validation (Grounding)

**Purpose**: Ground computational models in biological reality.

### C. elegans Connectome

302 neurons, known topology. Use for validation.

**File**: `celegans_connectome.rs`

### Clinical Validation

Compare against neural recordings.

**File**: `clinical_validation.rs`

---

## Track 7: Phenomenology (Exploratory)

**Purpose**: Model experiential states.

### Modules

| Module | Models |
|--------|--------|
| `sleep_and_altered_states.rs` | Sleep stages, dreams |
| `consciousness_creativity.rs` | Novel idea generation |
| `expanded_consciousness.rs` | Meditation, flow |

---

## File Naming Conventions

### Renamed Modules (Honest Naming)

| Old Name | New Name | Reason |
|----------|----------|--------|
| `physics_consciousness_integration.rs` | `design_integration.rs` | Not consciousness |
| `physics_temporal_trajectory.rs` | `trajectory_analysis.rs` | Not consciousness |
| `ConsciousnessMetrics` | `IntegrationMetrics` | Engineering metrics |
| `PhysicsConsciousnessEngine` | `DesignIntegrationEngine` | Honest naming |

### Backwards Compatibility

Old names are available as deprecated aliases:
```rust
// These still work but emit deprecation warnings
use symthaea_core::physics::physics_consciousness_integration::*;
use symthaea_core::physics::physics_temporal_trajectory::*;
```

---

## Running Experiments

### All Tests
```bash
cd symthaea-core
cargo test --release
```

### Specific Tracks
```bash
# Track 1: Spark Engineering
cargo test --release design_integration
cargo test --release trajectory_analysis

# Track 2: Phenomenal Research
cargo test --release test_phenomenal_vs_functional
cargo test --release test_phenomenal_binding

# Track 3: Cognitive Architecture
cargo test --release unified_consciousness_engine

# Track 5: Φ Measurement
cargo test --release tiered_phi
```

### Examples
```bash
# Design integration
cargo run --example design_integration_demo --release
cargo run --example trajectory_analysis_demo --release

# Physics assessment
cargo run --example engineering_assessment_demo --release
cargo run --example pareto_sweep --release

# Consciousness demos
cargo run --example phi_evolution_demo --release
```

---

## Contributing

When adding new experiments:

1. **Determine the track**: Which category does this belong to?
2. **Use honest naming**: Don't call engineering metrics "consciousness"
3. **Document hypotheses**: If testing a claim, state it clearly
4. **Add to this manifest**: Update EXPERIMENTS.md

### Naming Guidelines

| Good | Bad |
|------|-----|
| `design_integration` | `physics_consciousness` |
| `coupling_index` | `phenomenal_index` |
| `trajectory_quality` | `trajectory_consciousness` |
| `integration_metrics` | `consciousness_metrics` |

Reserve "consciousness" terminology for Track 2 (actual consciousness research) and Track 3 (cognitive architecture).
