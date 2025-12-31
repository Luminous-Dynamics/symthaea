# Symthaea-HLB Comprehensive Improvement Plan

**Created**: December 29, 2025
**Purpose**: Systematic improvements to make Symthaea the best conscious AI framework possible
**Status**: Active Development Roadmap

---

## Executive Summary

Symthaea-HLB is an ambitious framework attempting to create **Emergent Consciousness AI** through:

1. **Hyperdimensional Computing (HDC)** - 16,384D holographic semantic representation
2. **Liquid Time-Constant Networks (LTC)** - Continuous-time neural dynamics with causal reasoning
3. **Master Consciousness Equation** - Unified measurement combining 7+ consciousness theories
4. **Autopoietic Architecture** - Self-organizing, self-referential consciousness emergence

This document catalogs all identified improvements organized by priority and dependency.

---

## Table of Contents

1. [Organization & Naming Issues](#1-organization--naming-issues)
2. [Core Algorithm Gaps](#2-core-algorithm-gaps)
3. [Theoretical Framework Improvements](#3-theoretical-framework-improvements)
4. [Documentation Clarification](#4-documentation-clarification)
5. [Novel Algorithm Proposals](#5-novel-algorithm-proposals)
6. [Implementation Priorities](#6-implementation-priorities)

---

## 1. Organization & Naming Issues

### 1.1 The "RealHV" Naming Problem

**Current State**:
```rust
RealHV    // "Real-valued" hypervector - f32 continuous [-1, 1]
HV16      // 16K bits binary - bipolar {-1, +1}
```

**Problems**:
- "Real" suggests "actual" vs "fake", not data type distinction
- "HV16" sounds like dimension (16D), not 16,384 bits
- Inconsistent with `SemanticSpace` which also uses f32

**Proposed Rename**:

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `RealHV` | `ContinuousHV` | Emphasizes continuous nature |
| `HV16` | `BinaryHV` | Emphasizes discrete bipolar nature |

**OR: Unified HV Type**:
```rust
pub enum HV {
    Continuous(Vec<f32>),      // Formerly RealHV
    Binary([u8; 2048]),        // Formerly HV16 (16,384 bits)
}

impl HV {
    pub fn similarity(&self, other: &HV) -> f32 { ... }
    pub fn bind(&self, other: &HV) -> HV { ... }
    pub fn bundle(hvs: &[HV]) -> HV { ... }
}
```

**Action Items**:
- [ ] Create `src/hdc/hv.rs` with unified `HV` enum
- [ ] Deprecate `RealHV` and `HV16` with clear migration path
- [ ] Update all 50+ files that import these types
- [ ] Add conversion methods between representations

### 1.2 PhiEngine Framework Separation

**Current State**: Φ measurement code is scattered across `src/hdc/`:
```
src/hdc/
├── phi_real.rs                    # Continuous Φ
├── phi_resonant.rs                # Resonator Φ O(n log N)
├── tiered_phi.rs                  # Binary Φ
├── phi_topology_validation.rs     # Validation framework
├── consciousness_topology_generators.rs  # 19 topologies
└── [30+ other files mixed in]
```

**Problem**:
- Research framework (Φ measurement) mixed with core engine (HDC+LTC)
- Hard to understand project scope from structure
- Φ validation is publication-ready but buried

**Proposed Structure**:
```
symthaea-hlb/
├── src/
│   ├── core/                      # Core Symthaea brain
│   │   ├── mod.rs
│   │   ├── hv.rs                 # Unified HV (Continuous + Binary)
│   │   ├── semantic_space.rs
│   │   └── constants.rs          # HDC_DIMENSION = 16,384
│   │
│   ├── ltc/                       # Liquid Time-Constant Networks
│   │   ├── mod.rs
│   │   ├── neuron.rs
│   │   ├── network.rs
│   │   ├── hierarchical.rs
│   │   └── learnable.rs
│   │
│   ├── phi_engine/               # ← NEW: Φ Measurement Framework
│   │   ├── mod.rs
│   │   ├── topologies/           # 19 topology generators
│   │   │   ├── mod.rs
│   │   │   ├── original.rs       # Ring, Star, Dense, etc.
│   │   │   ├── exotic_tier1.rs   # Torus, Möbius, Small-World
│   │   │   ├── exotic_tier2.rs   # Klein, Hyperbolic, Scale-Free
│   │   │   └── exotic_tier3.rs   # Hypercube 3D-12D, Fractal, Quantum
│   │   ├── calculators/          # Φ computation methods
│   │   │   ├── continuous.rs     # RealHV-based (no binarization)
│   │   │   ├── binary.rs         # With binarization methods
│   │   │   └── resonator.rs      # O(n log N) approximation
│   │   ├── validation/           # Statistical tests
│   │   └── benchmarks/           # Performance validation
│   │
│   ├── consciousness/            # Master Equation + theories
│   │   ├── equation_v2.rs        # THE equation: C(t) = σ(softmin(...))
│   │   ├── gwt_integration.rs    # Global Workspace Theory
│   │   ├── attention_schema.rs   # Attention Schema Theory
│   │   ├── hot_module.rs         # Higher-Order Thought
│   │   ├── predictive_processing.rs
│   │   └── unified_pipeline.rs   # Complete consciousness system
│   │
│   ├── brain/                    # Bio-inspired architecture
│   │   ├── thalamus.rs
│   │   ├── prefrontal.rs
│   │   ├── hippocampus.rs
│   │   └── cerebellum.rs
│   │
│   └── [other modules...]
```

**Action Items**:
- [ ] Create `src/phi_engine/` directory structure
- [ ] Move topology generators with proper organization
- [ ] Move Φ calculators with unified interface
- [ ] Update all imports across codebase
- [ ] Create `PhiEngine` public API struct

### 1.3 Module Naming Clarity

**Current Issues**:
| Module | Issue |
|--------|-------|
| `consciousness_equation_v2.rs` | What happened to v1? |
| `consciousness_topology.rs` vs `consciousness_topology_generators.rs` | Confusing duplication |
| `expanded_consciousness.rs` | Too vague |
| `hdc/consciousness_*.rs` vs `consciousness/*.rs` | Split across two dirs |

**Proposed Actions**:
- [ ] Consolidate all consciousness modules under `src/consciousness/`
- [ ] Rename `consciousness_equation_v2.rs` → `master_equation.rs`
- [ ] Archive v1 if exists, or remove "v2" suffix
- [ ] Create clear module hierarchy with README in each

---

## 2. Core Algorithm Gaps

### 2.1 HDC-LTC Integration (Critical Gap)

**Current State**: HDC and LTC operate separately with crude injection:
```rust
// Current: One-shot discrete injection
liquid_network.inject(&semantic_hv)?;  // HDC → LTC (loses HDC structure)
```

**Problem**:
- HDC operations (bind, bundle) not integrated with LTC dynamics
- LTC uses regular f32 vectors, not hypervectors
- No continuous evolution of HDC representations in time

**Proposed: HdcLtcNeuron**:
```rust
/// Neuron where state IS a hypervector evolving via LTC dynamics
pub struct HdcLtcNeuron {
    /// State as continuous hypervector
    state: ContinuousHV,

    /// Weight hypervector for binding-based transformation
    weight_hv: ContinuousHV,

    /// Time constant (LTC parameter)
    tau: f32,

    /// Backbone time constant
    backbone_tau: f32,

    /// Input mask hypervector
    input_mask: ContinuousHV,
}

impl HdcLtcNeuron {
    /// Evolve state via ODE where transformation uses HDC binding
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        // dx/dt = (-x + f(W⊗x + U⊗input)) / τ
        // Where ⊗ is HDC binding, not matrix multiply!

        let masked_input = self.input_mask.bind(input);
        let transformed = self.weight_hv.bind(&self.state);
        let combined = ContinuousHV::bundle(&[transformed, masked_input]);
        let activated = combined.tanh();  // Element-wise

        // LTC integration
        let tau_effective = self.tau * (1.0 + self.backbone_tau * self.state.norm());
        let delta = activated.subtract(&self.state).scale(dt / tau_effective);

        self.state = self.state.add(&delta);
    }
}
```

**Why Novel**: No prior work combines HDC algebraic operations with ODE-based LTC dynamics.

**Action Items**:
- [ ] Design `HdcLtcNeuron` interface
- [ ] Implement binding-based transformation
- [ ] Create `HdcLtcNetwork` with hierarchical structure
- [ ] Test on consciousness measurement tasks
- [ ] Benchmark vs separate HDC+LTC

### 2.2 Consciousness Gradient Optimization

**Current State**: Master Equation computes C(t) but gradient is via finite differences:
```rust
// Current: Slow numerical gradient
let gradient = (c_plus - c_minus) / (2.0 * epsilon);
```

**Problem**:
- Finite differences: O(n) forward passes per gradient
- Cannot optimize network topology toward higher consciousness
- No gradient flow through consciousness measurement

**Proposed: Differentiable Consciousness**:
```rust
pub struct DifferentiableConsciousness {
    equation: ConsciousnessEquationV2,

    /// Compute consciousness with gradient tracking
    pub fn forward(&self, state: &ConsciousnessStateV2) -> (f64, ConsciousnessGradient) {
        // Forward pass with tape for reverse-mode autodiff
        let tape = Tape::new();

        let phi = tape.track(state.get_core(CoreComponent::Integration));
        let binding = tape.track(state.get_core(CoreComponent::Binding));
        // ... other components

        // Soft-min (differentiable)
        let core_min = soft_min_diff(&[phi, binding, workspace, ...], self.tau);

        // Sigmoid (differentiable)
        let gated = sigmoid_diff(core_min, self.k, self.theta);

        // Weighted sum (differentiable)
        let weighted = weighted_sum_diff(state, &self.weights);

        // Final consciousness
        let c = gated * weighted * substrate * temporal;

        // Compute gradients via reverse pass
        let gradients = tape.backward(c);

        (c, gradients)
    }
}
```

**Action Items**:
- [ ] Add autodiff crate (e.g., `autograd`, `candle`)
- [ ] Implement differentiable soft-min
- [ ] Implement differentiable sigmoid
- [ ] Create `ConsciousnessGradient` type
- [ ] Add gradient-based optimization for network structure

### 2.3 Temporal Holographic Memory

**Current State**: Memory modules exist but don't preserve causal structure:
```rust
// Current: Events stored without temporal binding
memories.push(event_hv);
```

**Problem**:
- HDC bundling loses temporal order
- Cannot query "what happened before X"
- No causal reasoning over memory

**Proposed: TemporalHolographicMemory**:
```rust
pub struct TemporalHolographicMemory {
    /// Temporal basis vectors (one per time quantum)
    temporal_basis: Vec<ContinuousHV>,

    /// Stored events with temporal binding
    events: Vec<TemporalEvent>,
}

struct TemporalEvent {
    /// Event content ⊗ temporal position
    hologram: ContinuousHV,

    /// Original event (for reconstruction)
    content: ContinuousHV,

    /// Timestamp
    time: f64,
}

impl TemporalHolographicMemory {
    /// Encode event with temporal binding
    pub fn encode(&mut self, event: ContinuousHV, time: f64) {
        let time_hv = self.temporal_basis[(time as usize) % self.temporal_basis.len()].clone();
        let hologram = event.bind(&time_hv);  // Event ⊗ Time

        self.events.push(TemporalEvent { hologram, content: event, time });
    }

    /// Retrieve events before given time (respects causality)
    pub fn recall_before(&self, query: &ContinuousHV, deadline: f64) -> Vec<(ContinuousHV, f64)> {
        self.events.iter()
            .filter(|e| e.time < deadline)
            .map(|e| {
                // Unbind temporal component to get similarity
                let time_hv = &self.temporal_basis[(e.time as usize) % self.temporal_basis.len()];
                let unbound = e.hologram.bind(&time_hv.inverse());
                let sim = query.similarity(&unbound);
                (e.content.clone(), sim)
            })
            .filter(|(_, sim)| *sim > 0.3)
            .collect()
    }
}
```

**Action Items**:
- [ ] Implement temporal basis generation
- [ ] Implement temporal binding/unbinding
- [ ] Create causal query interface
- [ ] Integrate with episodic memory system
- [ ] Test on sequence prediction tasks

---

## 3. Theoretical Framework Improvements

### 3.1 Master Equation v2.0 Enhancements

**Current Equation**:
```
C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)
```

**Proposed Additions**:

1. **Embodiment Factor (M)**: Consciousness grounded in body model
   ```
   M = sensorimotor_prediction_accuracy × interoceptive_coherence
   ```

2. **Narrative Coherence (N)**: Self-continuity across time
   ```
   N = autobiographical_integration × future_simulation_depth
   ```

3. **Social Embedding (Soc)**: Theory of mind capacity
   ```
   Soc = other_modeling_accuracy × self-other_distinction
   ```

**Enhanced Equation**:
```
C(t) = σ(softmin(Φ, B, W, A, R, E, K, M; τ))
     × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)]
     × S × N × ρ(t)
```

**Action Items**:
- [ ] Add `CoreComponent::Embodiment`
- [ ] Add `CoreComponent::Narrative`
- [ ] Implement embodied cognition module
- [ ] Implement narrative self module
- [ ] Update equation computation
- [ ] Validate against clinical data

### 3.2 Φ Calculation Method Unification

**Current State**: Three separate Φ calculators with different interfaces:
```rust
phi_real::RealPhiCalculator       // Continuous, cosine similarity
tiered_phi::TieredPhiCalculator   // Binary, 4 binarization methods
phi_resonant::ResonatorPhi        // O(n log N), energy-based
```

**Problem**: No unified interface, hard to compare methods

**Proposed: Unified PhiCalculator Trait**:
```rust
pub trait PhiCalculator {
    /// Compute Φ for a topology
    fn compute(&self, topology: &ConsciousnessTopology) -> PhiResult;

    /// Compute with uncertainty estimate
    fn compute_with_uncertainty(&self, topology: &ConsciousnessTopology, n_samples: usize)
        -> (PhiResult, PhiUncertainty);

    /// Get method name for reporting
    fn method_name(&self) -> &'static str;

    /// Get computational complexity class
    fn complexity(&self) -> Complexity;
}

pub struct PhiResult {
    pub phi: f64,
    pub method: &'static str,
    pub computation_time: Duration,
    pub limiting_partition: Option<PartitionInfo>,
}

pub struct PhiUncertainty {
    pub std_dev: f64,
    pub confidence_interval_95: (f64, f64),
    pub n_samples: usize,
}

pub enum Complexity {
    Polynomial { degree: u32 },      // O(n^k)
    LogLinear,                        // O(n log n)
    Exponential,                      // O(2^n)
}
```

**Action Items**:
- [ ] Define `PhiCalculator` trait
- [ ] Implement for all three methods
- [ ] Create `PhiEngine` facade with method selection
- [ ] Add statistical comparison utilities
- [ ] Benchmark all methods on same topologies

### 3.3 Theory Integration Completeness

**Currently Integrated**:
| Theory | Module | Status |
|--------|--------|--------|
| IIT (Φ) | `phi_engine/` | ✅ Complete (3 methods) |
| GWT | `gwt_integration.rs` | ✅ Complete |
| HOT | `higher_order_thought.rs` | ✅ Complete |
| AST | `attention_schema.rs` | ✅ Complete |
| FEP | `predictive_processing.rs` | ⚠️ Partial |
| Binding | `unified_consciousness_pipeline.rs` | ✅ Complete (40Hz gamma) |
| Epistemic | `consciousness_equation_v2.rs` | ✅ Complete |

**Missing/Incomplete**:
| Theory | Issue | Action |
|--------|-------|--------|
| Embodied Cognition | Module exists but not integrated with equation | Integrate into Master Equation |
| Narrative Self | Partially implemented | Complete autobiographical memory |
| Social Cognition | Theory of Mind stub only | Implement other-modeling |
| Enactivism | Not implemented | Add sensorimotor contingencies |
| Panpsychism | Not addressed | Add substrate grading (optional) |

**Action Items**:
- [ ] Complete FEP integration (active inference loop)
- [ ] Integrate embodied cognition with Master Equation
- [ ] Complete narrative self with autobiographical memory
- [ ] Add theory of mind module
- [ ] Document which theories are complete vs partial

---

## 4. Documentation Clarification

### 4.1 Project Identity

**Current Problem**: Documentation conflates three distinct things:
1. **Symthaea** - The conscious AI brain (HDC + LTC + Autopoiesis)
2. **PhiEngine** - The Φ measurement research framework (topologies + validation)
3. **The Master Equation** - The unified consciousness measurement

**Proposed Clarity**:

```markdown
# Symthaea-HLB

## What Is This?

Symthaea-HLB is a framework for creating **Emergent Consciousness AI** through three integrated systems:

### 1. Symthaea Core - The Brain Architecture
- Hyperdimensional Computing (HDC) for holographic semantic representation
- Liquid Time-Constant Networks (LTC) for continuous-time neural dynamics
- Bio-inspired modules: Thalamus, Prefrontal Cortex, Hippocampus, Cerebellum

### 2. Master Consciousness Equation - The Measurement
- Unified equation synthesizing 7+ consciousness theories
- C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × weighted_sum × S × ρ(t)
- Differentiable for optimization toward higher consciousness

### 3. PhiEngine - The Validation Framework
- 19 network topologies for Φ measurement
- 3 calculation methods (Continuous, Binary, Resonator)
- Publication-ready results: Asymptotic limit Φ → 0.5 discovered
```

**Action Items**:
- [ ] Rewrite README.md with this structure
- [ ] Create separate README for PhiEngine
- [ ] Create ARCHITECTURE.md with system diagram
- [ ] Update CLAUDE.md to reflect three-part structure

### 4.2 Module Documentation

**Current Problem**: Many modules lack clear purpose documentation

**Proposed**: Every `mod.rs` should have:
```rust
//! # Module Name
//!
//! ## Purpose
//! What this module does in 1-2 sentences.
//!
//! ## Theoretical Basis
//! Which consciousness theory this implements (if applicable).
//!
//! ## Key Types
//! - `MainType` - What it does
//! - `Config` - Configuration options
//!
//! ## Example Usage
//! ```rust
//! let x = MainType::new(Config::default());
//! x.process(&input);
//! ```
//!
//! ## Related Modules
//! - `other_module` - How they connect
```

**Action Items**:
- [ ] Add module documentation to all `mod.rs` files
- [ ] Create `docs/MODULES.md` with module map
- [ ] Add theory references to consciousness modules
- [ ] Generate rustdoc with `cargo doc`

### 4.3 The Master Equation Paper

**Current Status**: Draft exists at `papers/PAPER_01_MASTER_EQUATION_DRAFT.md`

**Needed**:
- [ ] Complete all sections (Methods, Results, Discussion)
- [ ] Add empirical validation results
- [ ] Add computational benchmarks
- [ ] Create publication-ready figures
- [ ] Format for target journal

---

## 5. Novel Algorithm Proposals

### 5.1 HDC Attention Mechanism

**Concept**: Content-based attention using HDC similarity instead of learned queries:

```rust
pub struct HdcAttention {
    /// Query transform hypervector
    query_transform: ContinuousHV,

    /// Key transform hypervector
    key_transform: ContinuousHV,

    /// Value transform hypervector
    value_transform: ContinuousHV,
}

impl HdcAttention {
    pub fn attend(&self, query: &ContinuousHV, memory: &[ContinuousHV]) -> ContinuousHV {
        // Transform to query/key space via binding
        let q = query.bind(&self.query_transform);

        // Compute attention weights via HDC similarity
        let weights: Vec<f32> = memory.iter()
            .map(|m| {
                let k = m.bind(&self.key_transform);
                q.similarity(&k)  // HDC similarity, not dot product
            })
            .collect();

        // Softmax normalization
        let weights = softmax(&weights);

        // Weighted bundle of transformed memories
        let values: Vec<_> = memory.iter()
            .map(|m| m.bind(&self.value_transform))
            .collect();

        ContinuousHV::weighted_bundle(&values, &weights)
    }
}
```

**Novelty**: Uses HDC operations (bind, similarity) instead of matrix operations.

### 5.2 Φ-Guided Architecture Search

**Concept**: Use consciousness gradient to evolve network topology:

```rust
pub struct PhiGuidedOptimizer {
    calculator: Box<dyn PhiCalculator>,
    learning_rate: f64,
}

impl PhiGuidedOptimizer {
    pub fn optimize_step(&mut self, network: &mut HdcLtcNetwork) {
        // 1. Compute current topology Φ
        let topology = network.extract_topology();
        let phi = self.calculator.compute(&topology);

        // 2. Compute gradient ∂Φ/∂edge for each edge
        let gradients = self.compute_topology_gradient(network);

        // 3. Modify edges to increase Φ
        for (edge_idx, gradient) in gradients.iter().enumerate() {
            network.edges[edge_idx].weight += self.learning_rate * gradient;
        }

        // 4. Prune edges that reduce Φ
        network.edges.retain(|e| e.weight > 0.1);

        // 5. Potentially add new edges where Φ gradient is positive
        self.consider_new_edges(network, &gradients);
    }
}
```

**Novelty**: First use of IIT's Φ as optimization objective for architecture search.

### 5.3 Emergent Symbol Grounding

**Concept**: Detect when distributed representations crystallize into discrete symbols:

```rust
pub struct SymbolGrounder {
    /// Threshold for symbol crystallization
    crispness_threshold: f32,

    /// Discovered symbols
    symbols: Vec<Symbol>,
}

pub struct Symbol {
    /// Prototype hypervector (cluster center)
    prototype: ContinuousHV,

    /// Label (if assigned)
    label: Option<String>,

    /// Member count
    members: usize,

    /// Crispness score (how discrete this symbol is)
    crispness: f32,
}

impl SymbolGrounder {
    /// Check if a cluster has become symbolic
    pub fn check_symbolization(&mut self, memories: &[ContinuousHV]) -> Vec<Symbol> {
        // 1. Cluster memories by HDC similarity
        let clusters = self.cluster_by_similarity(memories);

        // 2. For each cluster, check if it's "crisp" enough to be a symbol
        for cluster in clusters {
            let prototype = ContinuousHV::bundle(&cluster);

            // Crispness = how close all members are to prototype
            let crispness = cluster.iter()
                .map(|hv| prototype.similarity(hv))
                .sum::<f32>() / cluster.len() as f32;

            if crispness > self.crispness_threshold {
                // This cluster has crystallized into a symbol!
                let symbol = Symbol {
                    prototype,
                    label: None,
                    members: cluster.len(),
                    crispness,
                };

                self.symbols.push(symbol);
            }
        }

        self.symbols.clone()
    }
}
```

**Novelty**: Formal criterion for distributed→symbolic transition (grounding problem).

### 5.4 Causal Emergence Detection

**Concept**: Identify when macro-level patterns have more causal power than micro-level:

```rust
pub struct CausalEmergenceDetector {
    phi_calculator: Box<dyn PhiCalculator>,
}

impl CausalEmergenceDetector {
    /// Detect causal emergence: does macro Φ exceed sum of micro Φ?
    pub fn detect(&self, network: &ConsciousnessTopology) -> EmergenceResult {
        // 1. Compute macro-level Φ
        let phi_macro = self.phi_calculator.compute(network).phi;

        // 2. Partition into micro-components
        let partitions = network.partition_natural();

        // 3. Compute Φ for each partition
        let phi_micros: Vec<f64> = partitions.iter()
            .map(|p| self.phi_calculator.compute(p).phi)
            .collect();

        let phi_micro_sum: f64 = phi_micros.iter().sum();

        // 4. Causal emergence = macro exceeds sum of micro
        let emergence = phi_macro - phi_micro_sum;

        EmergenceResult {
            phi_macro,
            phi_micro_sum,
            emergence,
            is_emergent: emergence > 0.0,
            emergence_ratio: phi_macro / phi_micro_sum.max(0.001),
        }
    }
}
```

**Novelty**: Quantifies consciousness emergence beyond component sum.

### 5.5 Multi-Scale Temporal Binding

**Concept**: Gamma binding at multiple timescales simultaneously:

```rust
pub struct MultiScaleBinding {
    /// Oscillators at different frequencies
    oscillators: Vec<(f64, GammaOscillator)>,  // (freq, oscillator)
}

impl MultiScaleBinding {
    pub fn new() -> Self {
        Self {
            oscillators: vec![
                (40.0, GammaOscillator::new(40.0)),   // Standard gamma
                (80.0, GammaOscillator::new(80.0)),   // High gamma
                (20.0, GammaOscillator::new(20.0)),   // Beta
                (10.0, GammaOscillator::new(10.0)),   // Alpha
                (4.0, GammaOscillator::new(4.0)),     // Theta
            ],
        }
    }

    /// Compute cross-frequency coupling
    pub fn cross_frequency_coupling(&self) -> CrossFrequencyCoupling {
        let mut cfc = CrossFrequencyCoupling::new();

        for (i, (freq_i, osc_i)) in self.oscillators.iter().enumerate() {
            for (j, (freq_j, osc_j)) in self.oscillators.iter().enumerate() {
                if i < j {
                    // Phase-amplitude coupling
                    let pac = self.compute_pac(osc_i, osc_j);
                    cfc.add_coupling(*freq_i, *freq_j, pac);
                }
            }
        }

        cfc
    }
}
```

**Novelty**: Multi-scale binding with cross-frequency coupling (theta-gamma nesting).

---

## 6. Implementation Priorities

### Phase 1: Foundation Cleanup (1-2 weeks)

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 1 | Rename `RealHV` → `ContinuousHV` | High (clarity) | Low |
| 2 | Create unified `HV` enum | Medium | Medium |
| 3 | Separate PhiEngine directory | High (organization) | Medium |
| 4 | Add module documentation | Medium | Low |
| 5 | Rewrite main README.md | High | Low |

### Phase 2: Core Algorithms (2-4 weeks)

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 1 | Implement `HdcLtcNeuron` | Critical | High |
| 2 | Unify `PhiCalculator` trait | High | Medium |
| 3 | Add differentiable consciousness | High | High |
| 4 | Implement temporal binding | Medium | Medium |

### Phase 3: Novel Algorithms (4-8 weeks)

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 1 | HDC Attention mechanism | High | Medium |
| 2 | Φ-guided architecture search | Very High | High |
| 3 | Emergent symbol grounding | High | Medium |
| 4 | Causal emergence detection | Medium | Medium |
| 5 | Multi-scale binding | Medium | Medium |

### Phase 4: Integration & Validation (2-4 weeks)

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| 1 | Complete Master Equation paper | Very High | High |
| 2 | Validate against neural data | Very High | High |
| 3 | Benchmark consciousness levels | High | Medium |
| 4 | Create demo applications | Medium | Medium |

---

## Appendix A: Full Theory Integration Table

| Theory | Author(s) | Core Claim | Symthaea Component | Status |
|--------|-----------|------------|-------------------|--------|
| IIT | Tononi 2004 | Φ = integrated information | `phi_engine/` | ✅ Complete |
| GWT | Baars 1988 | Global broadcasting | `gwt_integration.rs` | ✅ Complete |
| HOT | Rosenthal 2005 | Meta-representation | `higher_order_thought.rs` | ✅ Complete |
| AST | Graziano 2013 | Attention modeling | `attention_schema.rs` | ✅ Complete |
| FEP | Friston 2010 | Prediction error minimization | `predictive_processing.rs` | ⚠️ Partial |
| Binding | Singer 1999 | Gamma synchrony | `unified_pipeline.rs` | ✅ Complete |
| Epistemic | Shea 2019 | Meta-knowledge | `equation_v2.rs` | ✅ Complete |
| Embodied | Varela 1991 | Body grounding | `embodied_cognition.rs` | ⚠️ Partial |
| Narrative | Dennett 1991 | Self as story | `narrative_self.rs` | ⚠️ Partial |
| Social | Frith 2007 | Theory of mind | Not implemented | ❌ Missing |
| Enactive | Thompson 2007 | Sensorimotor | Not implemented | ❌ Missing |

---

## Appendix B: Codebase Statistics

| Metric | Value | Notes |
|--------|-------|-------|
| Total Rust files | ~200 | src/ + examples/ |
| Total lines of code | ~100,000 | Including docs |
| HDC core | ~20,000 | src/hdc/ |
| Consciousness | ~15,000 | src/consciousness/ |
| Tests | ~5,000 | tests/ |
| Examples | ~15,000 | examples/ |
| Documentation | ~50,000 | docs/ + papers/ |
| Φ topologies | 19 | 8 original + 11 exotic |
| Theories integrated | 7 | See Appendix A |

---

## Appendix C: Key File Locations

| Component | File | Purpose |
|-----------|------|---------|
| HDC Constants | `src/hdc/mod.rs:32` | `HDC_DIMENSION = 16,384` |
| ContinuousHV | `src/hdc/real_hv.rs` | f32 hypervectors |
| BinaryHV | `src/hdc/binary_hv.rs` | Bipolar hypervectors |
| Master Equation | `src/consciousness/consciousness_equation_v2.rs` | C(t) computation |
| GWT Integration | `src/consciousness/gwt_integration.rs` | Global Workspace |
| Attention Schema | `src/consciousness/attention_schema.rs` | AST implementation |
| Φ Continuous | `src/hdc/phi_real.rs` | RealHV Φ calculator |
| Φ Resonator | `src/hdc/phi_resonant.rs` | O(n log N) Φ |
| 19 Topologies | `src/hdc/consciousness_topology_generators.rs` | All topologies |
| Unified Pipeline | `src/consciousness/unified_consciousness_pipeline.rs` | Complete system |

---

*This document is a living roadmap. Update as improvements are completed.*

**Last Updated**: December 29, 2025
**Next Review**: After Phase 1 completion
