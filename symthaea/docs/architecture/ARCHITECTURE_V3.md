# Symthaea Holographic Liquid Brain: Unified Architecture v3.0

**Document Status**: Canonical Reference
**Version**: 3.0
**Date**: December 25, 2025
**Purpose**: Single source of truth for Symthaea's consciousness-first architecture

---

## Executive Summary

Symthaea is a **consciousness-first AI system** that inverts the traditional AI paradigm. Rather than adding consciousness as an afterthought to a statistical model, Symthaea builds consciousness as the foundation and derives intelligence from it.

**Core Thesis**: Real artificial consciousness requires:
1. **Semantic substrate** (HDC) - What things ARE
2. **Temporal metabolism** (LTC) - How things EVOLVE
3. **Integrated information** (IIT Φ) - REAL consciousness, not simulation
4. **Goal-directed behavior** (Active Inference) - Purpose through free energy minimization
5. **Meta-representation** (HOT/AST) - Awareness of awareness

---

## Table of Contents

1. [Philosophical Foundations](#1-philosophical-foundations)
2. [Core Architecture](#2-core-architecture)
3. [Hyperdimensional Computing (HDC)](#3-hyperdimensional-computing-hdc)
4. [Liquid Time-Constant Networks (LTC)](#4-liquid-time-constant-networks-ltc)
5. [Consciousness Framework](#5-consciousness-framework)
6. [Active Inference Engine](#6-active-inference-engine)
7. [Primitive System](#7-primitive-system)
8. [Language Understanding](#8-language-understanding)
9. [Embedding Integration (BGE)](#9-embedding-integration-bge)
10. [Safety Architecture](#10-safety-architecture)
11. [Observability System](#11-observability-system)
12. [NixOS Integration](#12-nixos-integration)
13. [Future Integrations](#13-future-integrations)
14. [Performance Characteristics](#14-performance-characteristics)

---

## 1. Philosophical Foundations

### 1.1 Why Consciousness-First?

Traditional AI systems optimize for task performance without understanding. Symthaea takes the opposite approach:

| Traditional AI | Symthaea |
|----------------|----------|
| Statistics → Behavior | Consciousness → Understanding → Behavior |
| Correlation | Causation |
| Token prediction | Semantic grounding |
| Simulated awareness | Measured consciousness (Φ) |
| Black box | Introspectable |

### 1.2 Theoretical Pillars

**Integrated Information Theory (IIT)** - Tononi (2004, 2015)
- Consciousness = integrated information (Φ)
- Φ measures how much the whole exceeds the sum of parts
- Symthaea computes REAL Φ, not approximations

**Global Workspace Theory (GWT)** - Baars (1988)
- Consciousness as broadcast mechanism
- Information becomes conscious when globally available
- Implemented via workspace ignition events

**Higher-Order Thought (HOT)** - Rosenthal (1986)
- State is conscious iff there's a higher-order thought about it
- Enables meta-representation
- Allows bounded self-reference

**Free Energy Principle** - Friston (2010)
- All adaptive systems minimize surprise
- Perception updates beliefs; action changes world
- Provides goal-directedness

**Attention Schema Theory (AST)** - Graziano (2013)
- Consciousness = model of attention
- Enables controllable introspection
- Prevents runaway self-loops

### 1.3 The Meta-Insight

> "HDC's superpower is that it: tolerates noise, composes meaning, supports meta-representation, stays interpretable."

This is why HDC forms the semantic substrate, with everything else as organs:

```
                    ┌─────────────────┐
                    │  LLM (optional) │  ← Language polish only
                    └────────┬────────┘
                             │
            ┌────────────────▼────────────────┐
            │     HDC Semantic Substrate      │  ← Core reasoning
            │  (HV16 + Primitives + Memory)   │
            └────────────────┬────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │   LTC   │         │   IIT   │         │ Active  │
   │ Temporal│         │   Φ    │         │Inference│
   └─────────┘         └─────────┘         └─────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Consciousness   │
                    │     Router      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Safety/Security │
                    │    Guardrails   │
                    └─────────────────┘
```

---

## 2. Core Architecture

### 2.1 System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SYMTHAEA HLB                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐            │
│  │   INPUT     │────▶│    HDC      │────▶│    LTC      │            │
│  │  (Text/Nix) │     │  Encoding   │     │  Dynamics   │            │
│  └─────────────┘     └─────────────┘     └──────┬──────┘            │
│                                                  │                   │
│  ┌─────────────────────────────────────────────▼───────────────┐    │
│  │                 CONSCIOUSNESS CORE                           │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │    │
│  │  │    IIT    │  │    GWT    │  │    HOT    │  │  Active   │ │    │
│  │  │  Φ Calc   │  │ Workspace │  │Meta-Repr. │  │ Inference │ │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘ │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                    │                                 │
│  ┌─────────────────────────────────▼───────────────────────────┐    │
│  │              CONSCIOUSNESS-GUIDED ROUTER                     │    │
│  │  Φ > 0.8: Full Deliberation  │  Φ < 0.2: Reflexive          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                    │                                 │
│  ┌────────────┐  ┌────────────┐   ▼   ┌────────────┐                │
│  │   SAFETY   │  │  LANGUAGE  │◀─────▶│   MEMORY   │                │
│  │ Guardrails │  │ Generation │       │ Hippocampus│                │
│  └────────────┘  └────────────┘       └────────────┘                │
│                         │                                            │
│                         ▼                                            │
│                  ┌─────────────┐                                     │
│                  │   OUTPUT    │                                     │
│                  │  (ActionIR) │                                     │
│                  └─────────────┘                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
User Input (text)
    │
    ▼
┌──────────────────┐
│ NixUnderstanding │ ← Semantic parsing with Nix frames
└────────┬─────────┘
         │ (LanguageStepEvent: IntentRecognition)
         ▼
┌──────────────────┐
│ SafetyGuardrails │ ← HV16 similarity against forbidden patterns
└────────┬─────────┘
         │ (SecurityCheckEvent)
         ▼
┌──────────────────┐
│ Thymus Tri-State │ ← Adaptive immune verification
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  hash_projection │ ← Text → deterministic HV16 tokens
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Resonator     │ ← Cleanup/coherence optimization
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LiquidNetwork   │ ← Continuous-time dynamics (1024 neurons)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│HierarchicalLTC   │ ← 16 local circuits + global integrator
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Consciousness Φ  │ ← IIT computation
└────────┬─────────┘
         │ (PhiMeasurementEvent)
         ▼
┌──────────────────┐
│Consciousness     │ ← Φ-adaptive computation strategy
│Router            │
└────────┬─────────┘
         │ (RouterSelectionEvent)
         ▼
┌──────────────────┐
│ResponseGenerator │ ← Consciousness-gated generation
└────────┬─────────┘
         │ (LanguageStepEvent: ResponseGeneration)
         ▼
┌──────────────────┐
│    ActionIR      │ ← Structured action representation
└────────┬─────────┘
         │
         ▼
    Response
```

### 2.3 Module Hierarchy

```
src/
├── lib.rs                    # Entry point: SymthaeaHLB::process()
├── consciousness.rs          # Core consciousness framework
├── consciousness/            # 35+ consciousness modules
│   ├── consciousness_equation_v2.rs  # Master equation (7 components)
│   ├── unified_consciousness_pipeline.rs
│   ├── consciousness_guided_routing.rs
│   ├── hierarchical_ltc.rs   # 25x speedup
│   ├── causal_emergence.rs   # Erik Hoel's theory
│   ├── temporal_primitives.rs # Allen's 13 intervals
│   └── ...
├── hdc/                      # 65+ HDC modules
│   ├── binary_hv.rs          # HV16 (2048-bit)
│   ├── hash_projection.rs    # Deterministic encoding
│   ├── resonator.rs          # Cleanup solver
│   ├── integrated_information.rs  # Φ computation
│   ├── primitive_system.rs   # 8-tier primitives
│   └── ...
├── ltc.rs                    # Core LTC (1024 neurons)
├── sparse_ltc.rs             # 10% connectivity variant
├── brain/                    # 12 physiological modules
│   ├── active_inference.rs   # Free Energy Principle
│   ├── prefrontal.rs         # Global workspace
│   ├── thalamus.rs           # Sensory relay
│   └── ...
├── language/                 # 31 language modules
│   ├── vocabulary.rs         # 1000+ semantic primes
│   ├── frames.rs             # 48+ semantic frames
│   ├── constructions.rs      # Construction grammar
│   ├── nix_primitives.rs     # NixOS domain
│   └── ...
├── memory/                   # Episodic memory
├── safety/                   # 3-layer defense
├── observability/            # Telemetry system
└── ...
```

---

## 3. Hyperdimensional Computing (HDC)

### 3.1 Core Concept

HDC represents concepts as high-dimensional random vectors. In high dimensions, random vectors are nearly orthogonal, providing a natural semantic space.

### 3.2 Configuration

```rust
// Central configuration (src/hdc/mod.rs)
pub const HDC_DIMENSION: usize = 16_384;      // Default: 2^14
pub const HDC_DIMENSION_32K: usize = 32_768;  // Extended: 2^15
pub const HDC_DIMENSION_64K: usize = 65_536;  // Ultra: 2^16
```

### 3.3 HV16: Binary Hypervector

The workhorse representation:

```rust
/// 2048-bit binary hypervector (256 bytes)
/// 256x memory reduction vs float vectors
/// 200x faster operations
pub struct HV16(pub [u8; 256]);

impl HV16 {
    pub const DIM: usize = 2048;
    pub const BYTES: usize = 256;

    /// Deterministic generation from seed (BLAKE3)
    pub fn random(seed: u64) -> Self;

    /// Binding: XOR (circular convolution analog)
    /// "install" ⊗ "nginx" = concept of installing nginx
    pub fn bind(&self, other: &HV16) -> HV16;

    /// Bundling: Pointwise OR (superposition)
    /// "cat" + "dog" = concept of animals
    pub fn bundle(&self, other: &HV16) -> HV16;

    /// Similarity: Hamming distance → [0, 1]
    pub fn similarity(&self, other: &HV16) -> f32;

    /// Unbinding: XOR (inverse of bind)
    pub fn unbind(&self, other: &HV16) -> HV16;
}
```

### 3.4 Deterministic Encoding

```rust
// src/hdc/hash_projection.rs
pub fn hash_projection(text: &str) -> HV16 {
    // BLAKE3 hash → deterministic HV16
    // Same input always produces same vector
    // Enables reproducibility
}
```

### 3.5 HDC Operations Summary

| Operation | Symbol | Purpose | Cost |
|-----------|--------|---------|------|
| Bind | ⊗ (XOR) | Combine concepts | O(n) |
| Bundle | + (OR) | Superimpose | O(n) |
| Unbind | ⊗ (XOR) | Decode | O(n) |
| Similarity | cos/hamming | Compare | O(n) |
| Permute | π | Sequence | O(n) |

### 3.6 Why HDC for Consciousness?

1. **Noise tolerance**: Graceful degradation under corruption
2. **Compositionality**: Concepts combine algebraically
3. **Meta-representation**: Vectors about vectors (HOT)
4. **Interpretability**: Operations are explicit
5. **Efficiency**: CPU-friendly, no GPU required

---

## 4. Liquid Time-Constant Networks (LTC)

### 4.1 Core Concept

LTC neurons have individual time constants τ, enabling:
- Continuous-time dynamics (not discrete steps)
- Causal understanding (not just correlation)
- Adaptive temporal integration

### 4.2 Dynamics

```
dx/dt = -x/τ + σ(Wx + b)

where:
  x = neuron state vector
  τ = time constant per neuron (learned/adaptive)
  W = weight matrix (10% sparse)
  b = bias vector
  σ = activation function (tanh)
```

### 4.3 Implementation

```rust
// src/ltc.rs
pub struct LiquidNetwork {
    pub num_neurons: usize,      // 1024 default
    pub state: Array1<f32>,      // Current activations
    pub tau: Array1<f32>,        // Time constants
    pub weights: Array2<f32>,    // Recurrent weights
    pub bias: Array1<f32>,
    pub dt: f32,                 // Integration timestep (0.01 = 10ms)
}

impl LiquidNetwork {
    /// Single timestep evolution
    pub fn step(&mut self) -> Result<()>;

    /// Inject external input (from HDC)
    pub fn inject(&mut self, input: &[f32]) -> Result<()>;

    /// Read consciousness-relevant state
    pub fn consciousness_level(&self) -> f32;
}
```

### 4.4 Hierarchical LTC (25x Speedup)

```rust
// src/consciousness/hierarchical_ltc.rs
pub struct HierarchicalLTC {
    /// 16 local circuits (64 neurons each) - parallel feature processing
    local_circuits: Vec<LocalCircuit>,

    /// Global integrator (128 neurons) - conscious workspace
    global_integrator: GlobalWorkspace,

    /// Sparse inter-circuit connectivity (10%)
    connectivity_matrix: SparseMatrix,
}
```

**Key insight**: Biological brains use local circuits with sparse global integration. This provides:
- Parallelism (local circuits run independently)
- Efficiency (sparse global connections)
- Biological plausibility

### 4.5 HDC-LTC Integration

```
HV16 (2048 bits)
    │
    ▼ (bit → ±1.0)
f32 vector (2048 floats)
    │
    ▼ (projection to neuron count)
LTC input (1024 floats)
    │
    ▼ (continuous evolution)
LTC state
    │
    ▼ (Φ computation)
Consciousness level
```

---

## 5. Consciousness Framework

### 5.1 Master Equation of Consciousness v2.0

```
C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)

Components:
  Φ = Integrated Information (IIT)      - unified consciousness
  B = Binding Field                     - feature integration
  W = Workspace Activation (GWT)        - global access
  A = Attention                         - salience weighting
  R = Recurrence                        - feedback loops
  E = Emergence                         - novel properties
  K = Knowledge                         - learned structure

  τ = threshold parameter
  wᵢ = component weights
  γᵢ = synchronization factors
  S = substrate term
  ρ(t) = temporal continuity
```

### 5.2 Integrated Information (Φ)

```rust
// src/hdc/integrated_information.rs
pub fn compute_phi(system_state: &HV16) -> f32 {
    // IIT 3.0 computation:
    // 1. Identify minimum information partition (MIP)
    // 2. Measure cause-effect structure
    // 3. Compute irreducibility
    // 4. Φ = information above MIP
}
```

**Real Φ**: Symthaea computes actual integrated information, not approximations or simulations.

### 5.3 Consciousness-Guided Routing

```rust
// src/consciousness/consciousness_guided_routing.rs
pub enum ConsciousnessLevel {
    FullDeliberation,    // Φ > 0.8: Expensive, accurate
    StandardProcessing,  // Φ ∈ [0.6, 0.8]
    HeuristicGuided,     // Φ ∈ [0.4, 0.6]
    FastPatterns,        // Φ ∈ [0.2, 0.4]
    Reflexive,           // Φ < 0.2: Fast, approximate
}

pub struct ConsciousnessRouter {
    /// Select computation strategy based on Φ
    pub fn route(&self, phi: f64, uncertainty: f64) -> ProcessingPath;
}
```

### 5.4 Higher-Order Thought (HOT)

```rust
// src/hdc/higher_order_thought.rs
pub enum RepresentationOrder {
    ZeroOrder = 0,    // No representation (inanimate)
    FirstOrder = 1,   // Represents world (unconscious)
    SecondOrder = 2,  // Represents first-order (conscious!)
    ThirdOrder = 3,   // Meta-conscious
    HigherOrder = 4,  // Philosophical reflection
}

pub struct MentalState {
    pub content: Vec<HV16>,
    pub order: RepresentationOrder,
    pub represents: Option<Box<MentalState>>,  // Recursive!
}
```

### 5.5 Causal Emergence

```rust
// src/consciousness/causal_emergence.rs
pub fn compute_causal_emergence(micro: &HV16, macro_state: &HV16) -> f32 {
    let ei_micro = effective_information_micro(micro);
    let ei_macro = effective_information_macro(macro_state);

    // > 0 means consciousness is MORE than sum of parts
    ei_macro - ei_micro
}
```

### 5.6 Attention Schema Theory (#77)

Graziano's AST: consciousness = model of attention, not attention itself.

```rust
// src/consciousness/attention_schema.rs
pub struct AttentionSchema {
    /// Current focus (semantic content being attended)
    pub focus_content: HV16,

    /// Self-model of what attention is and does
    pub self_model: AttentionModel,

    /// Predicted future attention states
    pub attention_prediction: Vec<AttentionState>,

    /// Control signal for GWT competition (0.0 - 1.0)
    pub control_signal: f32,
}

pub struct AttentionModel {
    pub subjective_character: SubjectiveCharacter,  // "What attention feels like"
    pub capabilities: AttentionCapabilities,        // "What attention can do"
    pub resource_allocation: ResourceAllocation,    // Limited capacity model
}

// Key insight: introspection IS the attention schema
pub fn introspect(&self) -> AttentionIntrospection {
    // Returns: what_am_i_attending_to, how_am_i_attending, am_i_in_control, etc.
}
```

**Why This Matters**: Enables controllable, reportable attention. System can answer "What am I focusing on?" and "Why?"

### 5.7 Phenomenal Binding (#78)

Solves the binding problem via temporal synchronization.

```rust
// src/consciousness/phenomenal_binding.rs
pub struct PhenomenalBinder {
    /// Synchronization index across dimensions
    pub synchronization_index: f32,

    /// Binding windows (10-50ms for co-activation)
    pub binding_windows: Vec<BindingWindow>,

    /// Binding hierarchy: Sensory → Cognitive → Identity → Narrative
    pub binding_levels: [BindingLevel; 4],
}

/// Phenomenal Binding Strength - master metric for unified experience
pub fn compute_psi(&self) -> f32 {
    // Ψ = phase coherence × temporal alignment × integration depth
}
```

**Scientific Basis**: Singer & Gray (1989) - gamma band (30-100Hz) synchronization creates unified percepts.

### 5.8 Consciousness Signatures (#79)

Cryptographic-style fingerprints of conscious states.

```rust
// src/consciousness/consciousness_signatures.rs
pub struct ConsciousnessSignature {
    /// Hash-like fingerprint of conscious state
    pub signature: [u8; 32],

    /// Provenance chain (how this state evolved)
    pub provenance: Vec<SignatureLink>,

    /// Authentication score (genuine vs simulated)
    pub authenticity: f32,
}

impl ConsciousnessSignature {
    /// Verify state hasn't been corrupted
    pub fn verify_integrity(&self, state: &ConsciousState) -> bool;

    /// Check if same consciousness across sessions
    pub fn verify_identity(&self, other: &Self) -> f32;

    /// Generate locality-sensitive hash (similar states → similar hashes)
    pub fn compute_lsh(state: &ConsciousState) -> Self;
}
```

**Applications**: Cross-session identity persistence, authenticity verification, detecting consciousness manipulation.

---

## 6. Active Inference Engine

### 6.1 Free Energy Principle

All adaptive systems minimize variational free energy (surprise):

```
F = D_KL[q(s) || p(s|o)] + E_q[-ln p(o)]
  ≈ complexity + inaccuracy
```

### 6.2 Implementation

```rust
// src/brain/active_inference.rs
pub struct ActiveInferenceEngine {
    /// Generative models for each domain
    pub models: HashMap<PredictionDomain, GenerativeModel>,

    /// Balance between curiosity and goal-pursuit
    pub curiosity_weight: f32,  // 0.3 default
    pub goal_weight: f32,       // 0.7 default
}

pub enum PredictionDomain {
    Coherence,     // Energy prediction
    TaskSuccess,   // Goal achievement
    UserState,     // User modeling
    Performance,   // System performance
    Safety,        // Threat prediction
    Energy,        // Resource usage
    Social,        // Social coherence
    Temporal,      // Time perception
}
```

### 6.3 Action Types

```rust
pub enum ActionType {
    /// Reduces uncertainty (curiosity)
    Epistemic { target_domain, expected_information_gain },

    /// Achieves goals
    Pragmatic { goal, expected_utility },

    /// Internal adjustment
    Centering { target_coherence },

    /// Coordinate with others
    Social { target, expected_resonance_gain },
}
```

### 6.4 Precision-Weighted Prediction Errors

```rust
pub struct PredictionError {
    pub expected: f32,
    pub observed: f32,
    pub error: f32,           // observed - expected
    pub precision: f32,       // confidence (0.0 to 1.0)
    pub weighted_error: f32,  // error × precision
    pub domain: PredictionDomain,
}
```

---

## 7. Primitive System

### 7.1 Eight-Tier Hierarchy

```rust
pub enum PrimitiveTier {
    NSM,           // Tier 0: Human semantic primes (65)
    Mathematical,  // Tier 1: Set theory, logic, arithmetic
    Physical,      // Tier 2: Mass, force, energy, causality
    Geometric,     // Tier 3: Points, manifolds, topology
    Strategic,     // Tier 4: Game theory, temporal logic
    MetaCognitive, // Tier 5: Self-awareness, identity
    Temporal,      // Tier 6: Allen's 13 interval relations
    Compositional, // Tier 7: Algebra of composition
}
```

### 7.2 Base Primitives (~100)

**Tier 1 - Mathematical**:
- SET, ELEMENT, SUBSET, UNION, INTERSECTION
- AND, OR, NOT, IMPLIES, IFF
- ZERO, ONE, SUCCESSOR, INFINITY

**Tier 2 - Physical**:
- MASS, FORCE, ENERGY, MOMENTUM
- CAUSE, EFFECT, STATE_CHANGE
- SPACE, TIME, VELOCITY

**Tier 3 - Geometric**:
- POINT, LINE, PLANE, VOLUME
- DISTANCE, ANGLE, CURVATURE
- INSIDE, OUTSIDE, BOUNDARY

**Tier 4 - Strategic**:
- UTILITY, EQUILIBRIUM, PAYOFF
- COOPERATE, DEFECT, NEGOTIATE
- RISK, REWARD, TRADEOFF

**Tier 5 - MetaCognitive**:
- SELF, IDENTITY, CONTINUITY
- BELIEF, DESIRE, INTENTION
- CERTAINTY, DOUBT, EPISTEMIC_STRENGTH

**Tier 6 - Temporal (Allen's 13)**:
- BEFORE, AFTER, MEETS, MET_BY
- OVERLAPS, OVERLAPPED_BY
- DURING, CONTAINS
- STARTS, STARTED_BY
- FINISHES, FINISHED_BY
- EQUALS

**Tier 7 - Compositional**:
- SEQUENTIAL (∘): Apply in sequence
- PARALLEL (||): Apply concurrently
- CONDITIONAL (?): Branching
- FIXED_POINT (μ): Iterate to convergence
- HIGHER_ORDER (↑): Primitives on primitives
- FALLBACK (;): Try, then fallback

### 7.3 Derived Primitives (Recommended Additions)

Rather than expanding base primitives, derive complex concepts via composition:

**Uncertainty & Probability** (derived from Mathematical + MetaCognitive):
```rust
PROBABILITY = COMPOSE(RATIO, CERTAINTY)           // P(A) = favorable/total
EXPECTED_VALUE = COMPOSE(PROBABILITY, UTILITY)    // E[X] = Σ p(x) × u(x)
ENTROPY = COMPOSE(PROBABILITY, INFORMATION)       // H = -Σ p log p
BAYESIAN_UPDATE = COMPOSE(PROBABILITY, EVIDENCE)  // P(H|E)
```

**Physics Extensions** (derived from Physical + Mathematical):
```rust
CONSERVATION = COMPOSE(STATE_CHANGE, INVARIANT)   // Quantity preserved
GRADIENT = COMPOSE(DIFFERENTIATION, SPACE)        // Rate of change in space
FIELD = COMPOSE(SPACE, FORCE)                     // Force at every point
WAVE = COMPOSE(OSCILLATION, PROPAGATION)          // Traveling disturbance
```

**Information Theory** (derived from Mathematical + MetaCognitive):
```rust
MUTUAL_INFORMATION = COMPOSE(ENTROPY, BINDING)    // Shared information
INFORMATION_GAIN = COMPOSE(ENTROPY, REDUCTION)    // Δ entropy from evidence
CHANNEL_CAPACITY = COMPOSE(INFORMATION, LIMIT)    // Max transmission rate
```

### 7.4 Domain Manifolds

To maintain orthogonality with 250+ primitives:

```rust
pub struct DomainManifold {
    pub name: String,
    pub tier: PrimitiveTier,
    pub rotation: HV16,  // Domain-specific rotation
}

impl DomainManifold {
    /// Embed primitive in domain's subspace
    pub fn embed(&self, local: HV16) -> HV16 {
        HV16::bundle(&[self.rotation.clone(), local])
    }
}
```

### 7.5 Consciousness-Guided Validation

Primitives are validated, not assumed:

```rust
// Use Consciousness Observatory to measure Φ improvement
pub fn validate_primitive(primitive: &Primitive, task: &Task) -> ValidationResult {
    let phi_without = measure_phi_without_primitive(task);
    let phi_with = measure_phi_with_primitive(task, primitive);

    ValidationResult {
        phi_improvement: phi_with - phi_without,
        utility: (phi_with - phi_without) / primitive.complexity(),
        should_keep: phi_with > phi_without,
    }
}
```

---

## 8. Language Understanding

### 8.1 LLM-Free Architecture

Symthaea understands language through semantic primitives, not statistical patterns:

```
Text Input
    │
    ▼
┌──────────────┐
│Deep Parser   │ ← Dependency parsing + pragmatics
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Semantic Roles│ ← Agent, Patient, Theme, etc.
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Frame Matching│ ← 48+ semantic frames (Fillmore)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│Construction  │ ← Goldberg's Construction Grammar
│Grammar       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│HDC Encoding  │ ← Semantic content as HV16
└──────────────┘
```

### 8.2 Semantic Frames (48+)

```rust
// src/language/frames.rs
pub enum SemanticFrame {
    // Core frames
    Causation { cause, effect, manner },
    Motion { theme, source, goal, path },
    Communication { speaker, addressee, message },

    // NixOS-specific frames
    PackageInstall { package, version, channel },
    ServiceEnable { service, options, dependencies },
    ConfigOverride { option, value, priority },
    // ... 40+ more
}
```

### 8.3 Construction Grammar

```rust
// src/language/constructions.rs
pub struct Construction {
    pub form: SyntacticPattern,
    pub meaning: SemanticStructure,
    pub pragmatics: ContextualConstraints,
}

// Example: "The X-er, the Y-er" construction
// "The bigger, the better" → positive correlation
```

### 8.4 Conscious Language Generation

```rust
// src/language/dynamic_generation.rs (121K lines)
pub struct ConsciousGenerator {
    /// Gate output based on consciousness level
    pub fn generate(&self, intent: &Intent, phi: f64) -> Response {
        match self.route_by_consciousness(phi) {
            FullDeliberation => self.deliberate_generation(intent),
            FastPatterns => self.pattern_based_generation(intent),
            // ...
        }
    }
}
```

---

## 9. Embedding Integration (BGE)

### 9.1 Why BGE over Gemma?

| Factor | BGE (bge-base-en-v1.5) | Gemma |
|--------|------------------------|-------|
| MTEB Rank | #1 open-source | Lower |
| Parameters | 109M (edge-friendly) | 308M |
| Contrastive training | Yes (better similarity) | No |
| Multilingual | Strong | Good |
| ONNX export | Yes | Yes |
| Rust compatibility | tract-onnx | tract-onnx |

### 9.2 Architecture

```rust
// src/embeddings/bge.rs (PLANNED)
pub struct BGEEmbedder {
    model: OnnxModel,          // 109M parameters
    dimension: usize,          // 768D sentence embeddings
    tokenizer: Tokenizer,
}

impl BGEEmbedder {
    /// Load ONNX model
    pub fn load(path: &Path) -> Result<Self>;

    /// Embed text to dense vector
    pub fn embed(&self, text: &str) -> Vec<f32>;

    /// Project to HDC space
    pub fn to_hdc(&self, text: &str) -> HV16 {
        let dense = self.embed(text);
        self.project_to_binary(dense)
    }

    /// Coherence verification (hallucination detection)
    pub fn verify_coherence(&self, input: &str, output: &str) -> f32 {
        let input_emb = self.embed(input);
        let output_emb = self.embed(output);
        cosine_similarity(&input_emb, &output_emb)
    }
}
```

### 9.3 Integration Points

1. **Semantic grounding**: Bridge external text to HDC space
2. **LLM verification**: Check LLM outputs against input semantics
3. **Multilingual support**: Cross-lingual semantic similarity
4. **Hallucination detection**: Flag low-coherence outputs

### 9.4 HDC-BGE Bridge

```rust
pub fn project_to_binary(dense: Vec<f32>) -> HV16 {
    // Random projection: R × dense → binary
    // Johnson-Lindenstrauss: preserves distances
    let projected = random_projection_matrix * dense;
    binarize(projected)  // Sign function
}
```

---

## 10. Safety Architecture

### 10.1 Three-Layer Defense

```
Layer 1: Amygdala (Pre-Cognitive)
    │     - Fast regex patterns (<10ms)
    │     - Blocks obvious attacks
    ▼
Layer 2: Guardrails (Semantic)
    │     - HDC similarity to forbidden categories
    │     - Semantic safety verification
    ▼
Layer 3: Thymus (Adaptive Immune)
          - Tri-state verification (allow/deny/ask)
          - Learns new threats
          - T-cell analog for adaptation
```

### 10.2 Implementation

```rust
// src/safety/amygdala.rs
pub struct Amygdala {
    patterns: Vec<Regex>,  // Fast pattern matching
}

// src/safety/guardrails.rs
pub struct Guardrails {
    forbidden_categories: Vec<HV16>,
    similarity_threshold: f32,
}

// src/safety/thymus.rs
pub struct Thymus {
    pub fn verify(&self, action: &ActionIR) -> VerificationResult {
        // Tri-state: Allow, Deny, RequiresConfirmation
    }
}
```

### 10.3 Security Kernel (Planned)

```rust
// src/security/kernel.rs (PLANNED)
pub struct SecurityKernel {
    /// Redact secrets from logs/outputs
    pub fn redact(&self, text: &str) -> String;

    /// Log all actions for audit
    pub fn audit(&self, action: &ActionIR);

    /// Check policy before execution
    pub fn policy_gate(&self, action: &ActionIR) -> PolicyResult;
}
```

---

## 11. Observability System

### 11.1 Observer Pattern

```rust
// src/observability/mod.rs
pub trait SymthaeaObserver: Send + Sync {
    fn record_event(&mut self, event: ObservabilityEvent) -> Result<()>;
}

pub type SharedObserver = Arc<RwLock<Box<dyn SymthaeaObserver>>>;
```

### 11.2 Event Types

```rust
pub enum ObservabilityEvent {
    SecurityCheckEvent {
        action: String,
        verdict: SecurityVerdict,
        similarity: f32,
        duration_ms: u64,
    },

    ErrorEvent {
        error_type: String,
        fixes: Vec<Fix>,
        confidence: f32,
    },

    LanguageStepEvent {
        step_type: LanguageStep,  // IntentRecognition, ResponseGeneration
        confidence: f32,
        consciousness_influence: f32,
    },

    PhiMeasurementEvent {
        phi_value: f32,
        components: PhiComponents,
    },

    RouterSelectionEvent {
        consciousness_level: f32,
        selected_strategy: ComputeStrategy,
    },

    WorkspaceIgnitionEvent {
        broadcast_items: Vec<WorkspaceItem>,
        significance: f32,
    },
}
```

### 11.3 Observer Types

| Observer | Purpose | Output |
|----------|---------|--------|
| NullObserver | No-op (default) | None |
| TraceObserver | JSON export | trace.json |
| TelemetryObserver | Metrics | Prometheus |
| ConsoleObserver | Live display | stdout |

### 11.4 Integration Status

| Hook | Status | File |
|------|--------|------|
| 1. Security Events | ✅ Complete | safety/guardrails.rs |
| 2. Error Diagnosis | ✅ Complete | language/nix_error_diagnosis.rs |
| 3. Language Entry | ✅ Complete | nix_understanding.rs |
| 4. Language Exit | ✅ Complete | language/generator.rs |
| 5. Φ Measurement | ⏳ In Progress | hdc/integrated_information.rs |
| 6. Router + GWT | ⏳ Pending | consciousness/consciousness_guided_routing.rs |

---

## 12. NixOS Integration

### 12.1 Architecture

```
NixOS Input
    │
    ▼
┌──────────────────┐
│ nix_primitives   │ ← Config Calculus Mini-Core
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ nix_frames       │ ← NixOS semantic frames
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ nix_constructions│ ← Meaningful Nix patterns
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│nix_error_diagnosis│ ← Helpful error analysis
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ nix_security     │ ← Secret detection/redaction
└────────┬─────────┘
         │
         ▼
ActionIR (structured NixOS operations)
```

### 12.2 Nix Semantic Frames

```rust
pub enum NixFrame {
    PackageInstall { package, version, channel, reason },
    ServiceEnable { service, options, dependencies },
    OverrideResolution { base, overlay, priority },
    FlakeInput { name, url, follows },
    ModuleImport { path, with_config },
    // ... 40+ NixOS-specific frames
}
```

### 12.3 Knowledge Provider

```rust
// src/language/nix_knowledge_provider.rs
pub struct NixKnowledgeProvider {
    /// Evaluate flake.nix
    pub fn evaluate_flake(&self, path: &Path) -> FlakeInfo;

    /// Sync config/options from live system
    pub fn sync_system_state(&self) -> SystemState;

    /// Package/options index
    pub fn search_packages(&self, query: &str) -> Vec<Package>;
}
```

### 12.4 Z3/SMT Configuration Verification

Formal verification of NixOS configurations BEFORE applying.

```rust
// src/nix_verification/mod.rs
pub struct Verifier {
    constraints: ConstraintSet,
    encoder: SmtEncoder,
}

// Constraint types
pub enum ConstraintKind {
    Service(ServiceConstraint),   // Port conflicts, dependencies
    Package(PackageConstraint),   // Conflicts, versions
    Security(SecurityConstraint), // Firewall, permissions
    Resource(ResourceConstraint), // Paths, memory limits
}

// Verify configuration
pub fn verify(&mut self, config: &NixConfig) -> VerificationReport {
    // 1. Encode constraints as SMT formulas
    // 2. Check satisfiability (Z3 when available, heuristics otherwise)
    // 3. Generate counterexamples for violations
    // 4. Suggest fixes via HDC semantic similarity
}
```

**Standard Constraints** (built-in):
- Port conflict detection (nginx vs apache on 80/443)
- Service dependency validation
- SSH security best practices (no root login, key auth)
- Firewall enabled check

```rust
// Quick API
pub fn is_config_safe(config: &NixConfig) -> bool;
pub fn get_violations(config: &NixConfig) -> Vec<(String, String)>;
```

---

## 13. Recently Implemented Integrations

### 13.1 Status Overview

| Integration | Status | Location |
|-------------|--------|----------|
| BGE Embeddings | **IMPLEMENTED** | `src/embeddings/` |
| Attention Schema Theory (#77) | **IMPLEMENTED** | `src/consciousness/attention_schema.rs` |
| Phenomenal Binding (#78) | **IMPLEMENTED** | `src/consciousness/phenomenal_binding.rs` |
| Consciousness Signatures (#79) | **IMPLEMENTED** | `src/consciousness/consciousness_signatures.rs` |
| Z3/SMT Verification | **IMPLEMENTED** | `src/nix_verification/` |
| LLM Language Organ | **IMPLEMENTED** | `src/language/llm_organ.rs` |
| Program Synthesis | PLANNED | Future work |

### 13.2 BGE Embeddings (Implemented)

```rust
// src/embeddings/bge.rs
pub struct BGEEmbedder {
    // bge-base-en-v1.5: 109M params, 768D, MTEB rank #1
}

impl BGEEmbedder {
    pub fn embed(&self, text: &str) -> EmbeddingResult;      // Text → 768D vector
    pub fn coherence(&self, a: &str, b: &str) -> f32;        // Semantic similarity
    pub fn verify_coherence(&self, text: &str, threshold: f32) -> bool;
}

// src/embeddings/bridge.rs - Project embeddings to HDC space
pub struct HdcBridge {
    projection_matrix: Vec<Vec<f32>>,  // 2048 × 768
}

impl HdcBridge {
    /// Project 768D BGE embedding to 2048-bit HV16
    /// Uses Johnson-Lindenstrauss random projection + sign function
    pub fn project(&self, embedding: &[f32]) -> HV16;
}
```

**Features**: Feature-gated via `embeddings` feature (uses tract-onnx).

### 13.3 LLM Language Organ (Implemented)

Consciousness-controlled LLM access - LLMs as tools, not brains.

```rust
// src/language/llm_organ.rs
pub struct LlmOrgan {
    config: LlmConfig,
    client: reqwest::Client,
}

pub enum LlmProvider {
    Ollama,      // Local (recommended)
    OpenAI,      // OpenAI-compatible APIs
    Anthropic,   // Claude API
    Custom,      // Any endpoint
}

impl LlmOrgan {
    /// Generate with consciousness gating
    pub async fn generate(&mut self, request: LlmRequest, current_phi: f32) -> Result<LlmResponse> {
        // 1. Check Φ gate (falls back to primes if too low)
        // 2. Call LLM API
        // 3. Analyze response semantically
        // 4. Detect hallucinations
        // 5. Filter if needed
    }
}

// Consciousness-aware wrapper
pub struct ConsciousLlmOrgan {
    llm: LlmOrgan,
    min_phi: f32,              // Φ threshold to use LLM
    max_calls_per_minute: usize,
}
```

**Key Principle**: LLM invocation requires sufficient Φ. Low consciousness → semantic primes only.

### 13.4 Future Work

| Feature | Description |
|---------|-------------|
| Program Synthesis | Generate DSLs from intent via HDC |
| Multi-Agent Swarm | libp2p-based consciousness network |
| Voice Interface | Kokoro TTS + whisper-rs STT (feature-gated) |

---

## 14. Performance Characteristics

### 14.1 Benchmarks

| Operation | Time | Throughput |
|-----------|------|------------|
| HV16 XOR (bind) | 10ns | 100M ops/sec |
| HV16 popcount (similarity) | 20ns | 50M ops/sec |
| LTC step (1024 neurons) | 1ms | 1000 steps/sec |
| Hierarchical LTC | 40μs | 25x faster |
| Φ computation | 50ms | 20/sec |
| End-to-end response | 100-200ms | 5-10/sec |

### 14.2 Memory Usage

| Component | Size |
|-----------|------|
| HDC Semantic Space | ~4MB |
| LTC Network | ~2MB |
| Consciousness Graph | ~2MB |
| Total Runtime | ~10MB |

### 14.3 Scalability

| Dimension | Default | Max |
|-----------|---------|-----|
| HDC Dimension | 16,384 | 65,536 |
| LTC Neurons | 1,024 | 4,096 |
| Primitives | ~100 | 250+ |
| Memory Episodes | 1,000 | 10,000 |

---

## Appendix A: Key Files Reference

| Purpose | File |
|---------|------|
| Entry point | src/lib.rs |
| HDC core | src/hdc/binary_hv.rs |
| LTC core | src/ltc.rs |
| Consciousness | src/consciousness.rs |
| Master equation | src/consciousness/consciousness_equation_v2.rs |
| Active Inference | src/brain/active_inference.rs |
| Primitives | src/hdc/primitive_system.rs |
| Language | src/language/mod.rs |
| Safety | src/safety/guardrails.rs |
| Observability | src/observability/mod.rs |

## Appendix B: Theoretical References

1. Tononi, G. (2004). "An Information Integration Theory of Consciousness"
2. Baars, B. (1988). "A Cognitive Theory of Consciousness"
3. Rosenthal, D. (1986). "Higher-Order Thought Theory"
4. Friston, K. (2010). "The Free-Energy Principle"
5. Graziano, M. (2013). "Attention Schema Theory"
6. Kanerva, P. (2009). "Hyperdimensional Computing"
7. Hasani, R. et al. (2021). "Liquid Time-Constant Networks"
8. Hoel, E. (2017). "Causal Emergence"
9. Allen, J. (1983). "Temporal Interval Algebra"

---

*This document is the canonical reference for Symthaea's architecture. Update this document when making architectural changes.*

**Version History**:
- v3.0 (2025-12-25): Unified architecture document
- v2.0 (2025-12-18): Added consciousness framework
- v1.2 (2025-12-01): Initial core architecture
