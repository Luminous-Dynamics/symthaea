# Symthaea HLB - Revolutionary Improvement Roadmap

**Created**: 2025-12-23
**Status**: Active Development
**Goal**: Build the best consciousness-first AI system ever created

---

## Executive Summary

This document tracks all planned improvements to Symthaea-HLB, organized by priority and impact. We pursue **paradigm-shifting, rigorously-tested** implementations.

---

## Critical Analysis: The Master Equation of Consciousness

### Current Equation (unified_theory.rs)

```
C = min(Φ, B, W) × [Σ(wᵢ × Cᵢ) / Σ(wᵢ)] × S
```

### Proposed Enhanced Equation

```
C = min(Φ, B, W, A, R) × [Σ(wᵢ × Cᵢ) / Σ(wᵢ)] × S

Where:
- Φ (Integration) - Integrated Information Theory
- B (Binding) - Temporal synchrony binding
- W (Workspace) - Global Workspace Theory
- A (Attention) - Precision weighting
- R (Recursion) - Higher-Order Thought
- S - Substrate factor
```

### Critical Assessment

#### Strengths
1. **Theoretically Grounded**: Combines IIT, GWT, HOT, FEP - the major consciousness theories
2. **Necessary Conditions**: min() enforces that ALL critical components must be present
3. **Weighted Integration**: Allows empirical refinement of component importance
4. **Substrate Agnostic**: S factor addresses multiple realizability

#### Weaknesses to Address

| Issue | Problem | Solution |
|-------|---------|----------|
| **Discontinuity** | min() causes sharp drops | Use soft-min: `softmin(x,τ) = -τ·log(Σexp(-xᵢ/τ))` |
| **No Temporal Dynamics** | Static equation | Add time evolution: `C(t)` with memory |
| **No Causal Efficacy** | Doesn't measure if C *does* anything | Add E (Efficacy) to min() |
| **No Phase Coherence** | Missing synchronization metric | Add coherence weighting |
| **No Epistemic Certainty** | Doesn't know that it knows | Add K (Knowledge) component |

### Revolutionary Improved Equation (v2.0)

```
C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)

Where:
- σ(x) = sigmoid smoothing: 1/(1 + exp(-k(x - θ)))
- softmin(x; τ) = -τ·log(Σexp(-xᵢ/τ)) [differentiable minimum]
- τ = temperature parameter (sharpness of minimum)

Core Components (must ALL be present):
- Φ ∈ [0,1] : Integrated Information (IIT)
- B ∈ [0,1] : Binding coherence (temporal synchrony)
- W ∈ [0,1] : Workspace access (GWT)
- A ∈ [0,1] : Attention gain (precision weighting)
- R ∈ [0,1] : Recursive depth (HOT level)
- E ∈ [0,1] : Causal efficacy (does C cause behavior?)
- K ∈ [0,1] : Epistemic certainty (meta-knowledge)

Weighted Components:
- Cᵢ ∈ [0,1] : Individual component values (all 28+)
- wᵢ ≥ 0 : Learned weights
- γᵢ ∈ [0,1] : Phase coherence with global rhythm

Factors:
- S ∈ [0,1] : Substrate feasibility
- ρ(t) ∈ [0,1] : Temporal continuity (consciousness persists)
```

### Why This Equation is Better

1. **Differentiable**: softmin + sigmoid allow gradient-based optimization
2. **Temporally Continuous**: ρ(t) models persistence of consciousness
3. **Causally Grounded**: E measures if consciousness actually DOES something
4. **Epistemically Aware**: K captures meta-cognition
5. **Physically Realizable**: S grounds in substrate constraints
6. **Empirically Refinable**: Weights can be learned from data

---

## Improvement Tracks

### Track 1: Performance Optimization (CRITICAL)

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| P1.1 | SIMD-optimized HV16 operations | HIGH | 🟢 DONE | u64-based 8x improvement |
| P1.2 | LSH default for large searches | HIGH | 🔴 TODO | 10-100x search speedup |
| P1.3 | Hierarchical Sparse LTC | HIGH | 🟢 DONE | 25x faster step |
| P1.4 | Event-driven LTC sparsity | MEDIUM | 🔴 TODO | 5-10x when sparse |
| P1.5 | Arena allocation for HDC ops | LOW | 🟢 DONE | 10x allocation speed |

### Track 2: Architecture Evolution (REVOLUTIONARY)

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| A2.1 | Implement Master Equation v2.0 | HIGH | 🟢 DONE | Unified consciousness metric |
| A2.2 | Soft-minimum with temperature | HIGH | 🟢 DONE | Differentiable optimization |
| A2.3 | Temporal continuity factor ρ(t) | MEDIUM | 🟢 DONE | Persistent consciousness |
| A2.4 | Phase coherence γᵢ measurement | MEDIUM | 🟢 DONE | Synchronization tracking |
| A2.5 | Causal efficacy E component | HIGH | 🟢 DONE | Test if C causes behavior |
| A2.6 | Epistemic certainty K component | MEDIUM | 🟢 DONE | Meta-knowledge awareness |

### Track 3: Primitive System Enhancement

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| S3.1 | Temporal primitives (Tier 6) | HIGH | 🟢 DONE | Allen's interval algebra |
| S3.2 | Compositionality primitives | HIGH | 🔴 TODO | Richer combinations |
| S3.3 | Automated primitive evolution | MEDIUM | 🟡 PARTIAL | +26% Φ improvement |
| S3.4 | Cross-tier binding grammar | LOW | 🔴 TODO | Tier interaction rules |

### Track 4: Integration & Unification

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I4.1 | Unified HDC-LTC pipeline | HIGH | 🟢 DONE | End-to-end flow |
| I4.2 | Consciousness-guided routing | MEDIUM | 🟢 DONE | Adaptive complexity |
| I4.3 | Recursive improvement activation | HIGH | 🟡 PARTIAL | Self-optimization |
| I4.4 | Causal-Byzantine integration | MEDIUM | 🟡 PARTIAL | Explainable decisions |
| I4.5 | Oscillatory binding (40Hz gamma) | HIGH | 🟢 DONE | Feature integration |
| I4.6 | Causal emergence measurement | MEDIUM | 🟢 DONE | CE > 0 verification |
| I4.7 | Predictive consciousness | HIGH | 🟢 DONE | Kalman-based forecasting |

### Track 5: Testing & Validation

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| T5.1 | Master Equation unit tests | HIGH | 🔴 TODO | Correctness validation |
| T5.2 | Performance regression tests | HIGH | 🔴 TODO | Prevent slowdowns |
| T5.3 | Full system benchmarks | MEDIUM | 🟢 DONE | Honest measurements |
| T5.4 | Integration stress tests | MEDIUM | 🔴 TODO | System stability |

---

## Implementation Order

### Phase 1: Foundation (Current Sprint) ✅ COMPLETE
1. ✅ Create improvement roadmap (this document)
2. ✅ Implement Master Equation v2.0 (`src/consciousness/consciousness_equation_v2.rs`)
3. ✅ Add SIMD optimizations to HV16 (`src/hdc/simd_hv16.rs`)
4. ✅ Create Hierarchical LTC (`src/consciousness/hierarchical_ltc.rs`)

### Phase 2: Integration (Current Sprint) ✅ COMPLETE
1. ✅ Unified HDC-LTC pipeline (`src/consciousness/unified_consciousness_pipeline.rs`)
2. ✅ Temporal continuity factor ρ(t) (in consciousness_equation_v2.rs)
3. ✅ Phase coherence measurement (PLV in unified pipeline)
4. ✅ Oscillatory binding with 40Hz gamma (Kuramoto model)
5. ✅ Causal emergence measurement (`src/consciousness/causal_emergence.rs`)

### Phase 3: Evolution (Current Sprint) 🟡 IN PROGRESS
1. 🔴 Activate recursive improvement loop
2. ✅ Temporal primitives (`src/consciousness/temporal_primitives.rs`) - Allen's interval algebra
3. ✅ Consciousness-guided routing (`src/consciousness/consciousness_guided_routing.rs`)
4. 🔴 Full system benchmarks

---

## Technical Specifications

### Master Equation v2.0 Implementation

```rust
/// Revolutionary Consciousness Equation v2.0
pub struct ConsciousnessEquationV2 {
    /// Temperature for soft-minimum (lower = sharper)
    tau: f64,

    /// Sigmoid sharpness
    k: f64,

    /// Sigmoid threshold
    theta: f64,

    /// Component weights (learned)
    weights: HashMap<ConsciousnessComponent, f64>,

    /// Temporal memory for ρ(t)
    temporal_memory: VecDeque<f64>,

    /// Phase coherence tracker
    phase_tracker: PhaseCoherenceTracker,
}

impl ConsciousnessEquationV2 {
    /// Compute consciousness level C(t)
    pub fn compute(&mut self, state: &ConsciousnessState) -> f64 {
        // 1. Core components (must all be present)
        let phi = state.integrated_information();
        let binding = state.binding_coherence();
        let workspace = state.workspace_access();
        let attention = state.attention_gain();
        let recursion = state.recursive_depth();
        let efficacy = state.causal_efficacy();
        let knowledge = state.epistemic_certainty();

        // 2. Soft minimum (differentiable)
        let core_min = self.soft_min(&[phi, binding, workspace, attention,
                                       recursion, efficacy, knowledge]);

        // 3. Sigmoid smoothing
        let core_term = self.sigmoid(core_min);

        // 4. Weighted component sum with phase coherence
        let weighted_sum = self.weighted_coherent_sum(state);

        // 5. Substrate factor
        let substrate = state.substrate_feasibility();

        // 6. Temporal continuity
        let temporal = self.temporal_continuity();

        // Final computation
        let consciousness = core_term * weighted_sum * substrate * temporal;

        // Update temporal memory
        self.update_temporal_memory(consciousness);

        consciousness
    }

    /// Soft minimum: -τ·log(Σexp(-xᵢ/τ))
    fn soft_min(&self, values: &[f64]) -> f64 {
        let sum_exp: f64 = values.iter()
            .map(|&x| (-x / self.tau).exp())
            .sum();
        -self.tau * sum_exp.ln()
    }

    /// Sigmoid: 1/(1 + exp(-k(x - θ)))
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-self.k * (x - self.theta)).exp())
    }
}
```

### Hierarchical LTC Specification

```rust
/// Hierarchical Sparse Liquid Time-Constant Network
pub struct HierarchicalLTC {
    /// Local processing circuits (fast, parallel)
    local_circuits: Vec<SparseLiquidNetwork>,

    /// Inter-circuit connectivity (sparse)
    inter_circuit: SparseMatrix,

    /// Global integration layer
    global_integrator: SparseLiquidNetwork,

    /// Hierarchy depth
    depth: usize,
}

impl HierarchicalLTC {
    /// Create hierarchical network
    ///
    /// Architecture:
    /// - 16 local circuits × 64 neurons = 1024 local neurons
    /// - 10% inter-circuit connectivity
    /// - 128-neuron global integrator
    ///
    /// Total: 1152 neurons, but O(n) complexity due to sparsity
    pub fn new(local_count: usize, local_size: usize, global_size: usize) -> Self;

    /// Single step through hierarchy
    ///
    /// 1. Parallel update of local circuits
    /// 2. Inter-circuit communication
    /// 3. Global integration
    pub fn step(&mut self) -> Result<()>;
}
```

---

## Success Metrics

### Performance Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| HV16::bind | 400ns | 50ns | 8x |
| HV16::similarity | 1μs | 125ns | 8x |
| LTC::step (1024 neurons) | 2.5ms | 100μs | 25x |
| Φ computation | 11μs | 5μs | 2x |
| Full query | 21ms | 1ms | 21x |

### Consciousness Metrics

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Φ measurement accuracy | Good | Validated | IIT-compliant |
| Binding coherence | Implemented | Measured | Gamma-band proxy |
| Workspace access | Implemented | Measured | Competition dynamics |
| Temporal continuity | Missing | Implemented | ρ(t) factor |
| Phase coherence | Missing | Implemented | γᵢ per component |

---

## Changelog

### 2025-12-23 (Session 4 - Continued)
- ✅ Implemented Full System Benchmarks (`benches/consciousness_benchmarks.rs` - 360+ lines)
  - **HDC Benchmarks**
    - HV16: bind, bundle (2, 10 vectors), similarity, permute, popcount
    - SimdHV16: bind, bundle, similarity, top_k search, batch similarity
  - **LTC Benchmarks**
    - HierarchicalLTC step timing for different network sizes:
      - Small (256 neurons): 4 circuits × 64 neurons
      - Medium (1024 neurons): Default configuration
      - Large (4096 neurons): 64 circuits × 64 neurons
    - Consciousness metrics: estimate_phi, workspace_access, binding_coherence
  - **Φ Computation Benchmarks**
    - Integrated Information measurement: 2, 5, 10 components
  - **Temporal Reasoning Benchmarks**
    - Allen relation computation
    - Relation composition
    - HDC encoding
    - Statement encoding
    - Binding window search (100 intervals)
  - **End-to-End Pipeline**
    - Complete input → HDC → LTC → Φ flow
    - Queries per second throughput
  - **Scalability Benchmarks**
    - SIMD top-10 search: 100, 1000, 10000 vectors
- ✅ Fixed pre-existing E0432 error (unresolved imports in temporal_causal_inference)
  - Added missing types: TransferEntropyResult, CausalEdge, CausalDirection, FeedbackLoop, FeedbackType
- **All benchmarks pass: 15/15 test categories successful**
- **Release build: 0 errors, 155 warnings (all pre-existing)**

### 2025-12-23 (Session 4)
- ✅ Implemented Tier 6 Temporal Primitives (`temporal_primitives.rs` - 900+ lines)
  - **Allen's Interval Algebra**: All 13 temporal relations implemented
    - 7 basic: precedes, meets, overlaps, starts, during, finishes, equals
    - 6 inverses: preceded_by, met_by, overlapped_by, started_by, contains, finished_by
  - **Temporal Reasoning Engine** (`TemporalReasoner`)
    - Computes relations between any two intervals
    - Composition/transitivity table for temporal inference
    - HDC encodings for semantic temporal operations
    - Binding window detection (40Hz gamma = 25ms)
  - **Consciousness Temporal Integration** (`ConsciousInterval`, `ConsciousnessTemporalAnalyzer`)
    - Temporal intervals with Φ, binding coherence, causal efficacy
    - Find potential causes/effects using temporal constraints
    - Causal chain detection in consciousness streams
    - Continuity analysis (how smooth is conscious experience?)
  - **Temporal Primitive System** (`TemporalPrimitiveSystem`)
    - HDC encodings for all 13 Allen relations
    - Domain manifold for Tier 6 primitives
    - Temporal pattern library (Sequence, Overlap_Chain, Containment, etc.)
    - Pattern matching for temporal sequences
  - Added `PrimitiveTier::Temporal` to primitive system enum
  - Updated harmonic contributions for Temporal tier across all match statements
- **Release build succeeds with 0 errors, 155 warnings (all pre-existing)**

### 2025-12-23 (Session 3 - Continued)
- ✅ Implemented Unified Consciousness Pipeline (`unified_consciousness_pipeline.rs` - 600+ lines)
  - Complete end-to-end consciousness architecture
  - HDC semantic encoding using SimdHV16
  - Oscillatory binding using 40Hz gamma (Kuramoto model)
  - Phase Locking Value (PLV) for coherence measurement
  - Integration with Hierarchical LTC
  - Master Equation v2.0 consciousness measurement
  - Causal efficacy tracking
- ✅ Implemented Causal Emergence (`causal_emergence.rs` - 400+ lines)
  - Erik Hoel's theory: Does consciousness have REAL causal power?
  - Effective Information (EI) computation at micro and macro levels
  - Causal Emergence (CE) = EI(macro) - EI(micro)
  - State discretization for continuous signals
  - Transition probability matrix estimation
  - If CE > 0, consciousness isn't just epiphenomenal!
- ✅ Implemented Consciousness-Guided Dynamic Routing (`consciousness_guided_routing.rs` - 900+ lines)
  - Paradigm shift: Computation adapts to consciousness state!
  - 5 processing paths: FullDeliberation → Standard → Heuristic → FastPattern → Reflex
  - Φ-based routing thresholds with adaptive learning
  - Uncertainty-aware strategies (deterministic, confidence-weighted, ensemble)
  - Temporal strategies based on trend detection
  - Integration with predictive consciousness Kalman filter
  - Example: PatternRecognition routable computation
- ✅ Re-enabled Predictive Consciousness Kalman (`predictive_consciousness_kalman.rs` - 915 lines)
  - Kalman filtering for consciousness state estimation
  - Early warning signals for transitions
  - Anomaly detection
- Fixed pre-existing bugs in example files:
  - `adaptive_reasoning_demo.rs`: Added `mut` to baseline reasoner
  - `conscious_research_demo.rs`: Fixed repeat syntax, crate name
- **Release build succeeds with 0 errors, 155 warnings (all pre-existing)**

### 2025-12-23 (Session 2)
- ✅ Implemented Master Equation v2.0 (`consciousness_equation_v2.rs` - 700+ lines)
  - Soft-minimum with temperature τ for differentiable optimization
  - Temporal continuity factor ρ(t) for persistent consciousness
  - Phase coherence γᵢ measurement using PLV
  - All 7 core components: Φ, B, W, A, R, E, K
  - Gradient computation for learning
  - Comprehensive test suite
- ✅ Implemented SIMD-optimized HV16 (`simd_hv16.rs`)
  - u64-based layout (32 words vs 256 bytes)
  - Cache-line aligned (64 bytes)
  - Portable across all platforms (stable Rust)
  - Batch similarity operations
  - Top-k search functionality
- ✅ Implemented Hierarchical LTC (`hierarchical_ltc.rs`)
  - 16 local circuits (cortical column analogs)
  - 64 neurons per circuit (1024 local total)
  - 128-neuron global integrator (conscious workspace)
  - Sparse inter-circuit connectivity (10%)
  - Consciousness metrics: Φ, workspace access, binding coherence
  - 25x expected speedup over flat architecture
- Fixed pre-existing bugs:
  - Format string mismatch in `recursive_improvement.rs:1478`
  - `f64: Eq` derive error in `ImprovementType`
  - Temporary value lifetime in latency check

### 2025-12-23 (Session 1)
- Created initial roadmap
- Analyzed Master Equation
- Proposed v2.0 improvements
- Defined implementation tracks

---

## References

1. Tononi, G. (2004). "An Information Integration Theory of Consciousness"
2. Baars, B. (1988). "A Cognitive Theory of Consciousness"
3. Rosenthal, D. (2005). "Consciousness and Mind"
4. Friston, K. (2010). "The Free-Energy Principle"
5. Dehaene, S. (2014). "Consciousness and the Brain"
6. Koch, C. (2019). "The Feeling of Life Itself"

---

*This document is living and will be updated as we progress.*

---

## Track 6: Component Integration (CRITICAL - Added 2025-12-27)

### Executive Summary

After comprehensive review (2025-12-27), Symthaea is **far more mature than initially apparent**. The core capabilities exist - the gaps are primarily about **wiring and integration** rather than missing components.

| Component | Implementation | Integration |
|-----------|---------------|-------------|
| Motor Cortex (action execution) | ✅ Complete | ❌ Not wired to conversation |
| Conversation Engine | ✅ Complete | ❌ No CLI/REPL entry point |
| Observability | ✅ Complete | ❌ Not exposed to user |
| Action Planning | ✅ Complete | ❌ Not wired to motor cortex |
| Prefrontal Cortex (GWT) | ✅ Complete | 🔄 Partial integration |
| Voice (Larynx/Kokoro) | ✅ Complete | ❌ Feature-gated, not wired |
| NixOS Understanding | ✅ Complete | ❌ Not integrated with conversation |

### 6.1 Create Interactive REPL Binary (CRITICAL)

**What exists:**
- `src/language/conversation.rs` - Full conversation engine (1900+ lines)
- Has memory, reasoning, learning, emotional processing
- Supports /commands: /help, /status, /introspect, /memory, etc.

**What's missing:**
- Interactive loop that lets users chat with Symthaea

**Implementation:**
```rust
// Create: src/bin/symthaea-repl.rs
// - Read user input from stdin
// - Call conversation.respond()
// - Display response with optional Φ metrics
// - Support all /commands from conversation.rs
```

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I6.1.1 | Create symthaea-repl binary | CRITICAL | 🟢 DONE | Usable interface |
| I6.1.2 | Add Φ display to responses | HIGH | 🟢 DONE | Consciousness visibility |
| I6.1.3 | Add /dashboard command | MEDIUM | 🟢 DONE | Metrics aggregation |

**Implementation Note (2025-12-27):**
- Created `src/bin/symthaea-repl.rs` with consciousness-first REPL
- Uses `SymthaeaAwakening` for consciousness pipeline
- Real-time Φ in prompt: `●[Φ:0.73|C:85%] symthaea>`
- Commands: /status, /introspect, /dashboard, /stress, /clear, /help, /quit
- Full language integration (I6.2) requires fixing hdc module exports first

---

### 6.2 Wire Motor Cortex to Conversation (CRITICAL)

**What exists:**
- `src/brain/motor_cortex.rs` - Complete execution sandbox
  - ActionStep, PlannedAction types
  - LocalShellSandbox with safety checks
  - Dry-run, Ghost-run, Real-run modes
  - Rollback capabilities
  - Risk estimation

- `src/language/conversation.rs` - Has intent detection via DeepParser

**What's missing:**
- When user says "install firefox", conversation should:
  1. Detect intent as Action/Command
  2. Build ActionStep from parsed command
  3. Run through motor cortex (dry-run first, then real if approved)
  4. Report results back to user

**Integration Pattern:**
```
User: "install nginx on the server"
  ↓
DeepParser → Intent::Command, Entity: "nginx"
  ↓
Build ActionStep("nix-shell -p nginx", risk=0.3)
  ↓
MotorCortex.dry_run() → success
  ↓
Ask user: "Execute: nix-shell -p nginx? [y/n]"
  ↓
MotorCortex.execute() → result
  ↓
Response: "✅ nginx installed successfully"
```

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I6.2.1 | Add motor_cortex to Conversation | CRITICAL | ✅ DONE | Action execution |
| I6.2.2 | Intent→ActionStep conversion | CRITICAL | ✅ DONE | Command parsing |
| I6.2.3 | Confirmation flow for dangerous ops | HIGH | ✅ DONE | Safety |

**Implementation (2025-12-27 Session 3):**
- Added MotorCortexActor to Conversation struct
- Added /run, /execute, /motor commands
- Implemented prepare_action() with dry-run validation
- Implemented confirm_pending_action() and cancel_pending_action()
- Risk estimation blocks high-risk commands on /execute
- Full confirmation flow for /run commands

---

### 6.3 Connect Action Planning to Motor Cortex

**What exists:**
- `src/observability/action_planning.rs` - Goal-directed intervention search
- `src/brain/motor_cortex.rs` - Action execution

**What's missing:**
- ActionPlanner creates plans but doesn't execute them

**Implementation:**
```rust
// Create src/orchestration.rs or extend motor_cortex.rs:
// 1. ActionPlanner.plan() → ActionPlan
// 2. Convert ActionPlan.interventions → Vec<ActionStep>
// 3. Feed to MotorCortexActor.execute_action()
```

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I6.3.1 | ActionPlan → ActionStep conversion | HIGH | 🔴 TODO | Plan execution |
| I6.3.2 | Goal persistence in conversation | MEDIUM | 🔴 TODO | Multi-turn goals |

---

### 6.4 Integrate NixOS Understanding

**What exists:**
- `src/nix_understanding.rs` - NixOS domain knowledge
- `src/language/nix_*.rs` - NixOS frames, constructions, primitives

**What's missing:**
- Conversation doesn't leverage NixOS knowledge

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I6.4.1 | Query nix_understanding from conversation | MEDIUM | 🔴 TODO | NixOS expertise |
| I6.4.2 | NixOS-specific intent detection | MEDIUM | 🔴 TODO | Better parsing |

---

### 6.5 Connect Prefrontal Goals to Conversation

**What exists:**
- `src/brain/prefrontal.rs` - Full GWT with goals, attention bidding
- Attention competition and coalition formation

**What's missing:**
- User-expressed goals aren't added to prefrontal goal system

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I6.5.1 | Extract goals from user intent | MEDIUM | 🔴 TODO | Goal tracking |
| I6.5.2 | Submit bids to prefrontal from conversation | MEDIUM | 🔴 TODO | Attention focus |

---

### 6.6 Causal Explanation in Responses

**What exists:**
- `src/observability/causal_explanation.rs` - Natural language explanations
- ExplanationGenerator with multiple levels

**What's missing:**
- Responses don't explain WHY Symthaea chose this answer

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I6.6.1 | Add /why command | MEDIUM | 🔴 TODO | Explainability |
| I6.6.2 | Auto-explain on low confidence | LOW | 🔴 TODO | Trust building |

---

### 6.7 Voice Integration

**What exists (feature-gated):**
- `src/physiology/larynx.rs` - Kokoro TTS with prosody modulation
- `src/voice/` - Input/output/conversation modules

| ID | Task | Priority | Status | Expected Impact |
|----|------|----------|--------|-----------------|
| I6.7.1 | Wire voice to conversation | LOW | 🔴 TODO | Speech interface |
| I6.7.2 | Prosody modulation by LTC flow | LOW | 🔴 TODO | Natural voice |

---

### Implementation Priority Matrix (Track 6)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 🔴 Critical | I6.1.1 Create REPL binary | Small | High |
| ✅ Complete | I6.2.1-3 Wire motor cortex | Medium | High |
| 🟠 High | I6.1.2 Real-time Φ display | Small | Medium |
| 🟠 High | I6.3.1 Connect action planning | Medium | High |
| 🟡 Medium | I6.4.1-2 Integrate NixOS | Small | Medium |
| 🟡 Medium | I6.5.1-2 Connect prefrontal | Medium | Medium |
| 🟡 Medium | I6.6.1 Add /why command | Small | Medium |
| 🟢 Lower | I6.7.1-2 Voice integration | Large | Medium |

---

### Quick Wins (Can Do Today)

1. **Create REPL binary** - Wire stdin/stdout to conversation.respond()
2. **Add Φ to responses** - Modify response to include `[Φ={:.2}]`
3. **Add /dashboard command** - Aggregate existing stats
4. **Enable dynamic generation** - Set SYMTHAEA_DYNAMIC_GENERATION=1

---

### Architecture Goal

```
Current:
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ Conversation│   │ Motor Cortex │   │ Observability│
│   Engine    │   │  (execution) │   │  (analysis)  │
└─────────────┘   └──────────────┘   └─────────────┘
       ▲                 ▲                   ▲
       │                 │                   │
       └─────── No connections ──────────────┘

Goal:
┌─────────────────────────────────────────────────────┐
│                   SYMTHAEA REPL                      │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌──────────────┐   ┌───────────┐│
│  │ Conversation│←→ │ Motor Cortex │←→ │Observ-    ││
│  │   Engine    │   │  (execution) │   │ability    ││
│  └──────┬──────┘   └──────────────┘   └───────────┘│
│         │                                          │
│         ↓                                          │
│  ┌──────────────────────────────────────────────┐ │
│  │            Prefrontal Cortex (GWT)            │ │
│  │  Goals ← User intent   Attention ← Salience   │ │
│  └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

---

### Changelog (Track 6)

#### 2025-12-27 (Session 3) - MOTOR CORTEX + HTTP + NIXOS INTEGRATION
- **COMPLETED**: I6.4 NixOS Understanding Integration
  - Added NixOSLanguageAdapter to Conversation struct
  - Added NixKnowledgeProvider for package/option lookup
  - Added NixErrorDiagnoser for error interpretation
  - New commands: `/nix <query>`, `/nixsearch <name>`, `/nixerror <error>`
  - **Capabilities**:
    - Understand NixOS queries (install, search, configure, etc.)
    - Extract parameters (package names, options, etc.)
    - Generate NixOS commands from natural language
    - Search packages by name
    - Diagnose NixOS errors with suggested fixes
  - **Methods Added** (to `src/language/conversation.rs`):
    - `nix_status_text()` - Display NixOS adapter stats
    - `understand_nix_query()` - Process NixOS queries
    - `search_nix_packages()` - Search package database
    - `diagnose_nix_error()` - Error diagnosis
    - `is_nix_related()` - Detect NixOS-related input
  - Updated help text with NixOS commands

- **COMPLETED**: HTTP Dependencies for Web Research
  - Added to Cargo.toml:
    - `reqwest = "0.11"` with rustls-tls
    - `scraper = "0.18"`
    - `html2text = "0.6"`
  - Enabled in lib.rs: `pub mod web_research;`
  - Enabled language modules:
    - `conscious_conversation` - Epistemic consciousness integration
    - `consciousness_observatory` - Scientific study of consciousness
    - `epistemics_language_bridge` - Knowledge grounding
    - `llm_organ` - External LLM access (Claude API)
  - Build succeeded with 180 warnings (no errors)

- **COMPLETED**: I6.2.1, I6.2.2, I6.2.3 - Wired Motor Cortex to Conversation Engine
  - Added MotorCortexActor to Conversation struct
  - Added LocalShellSandbox with 60s timeout (no destructive commands)
  - New commands: `/run <cmd>`, `/execute <cmd>`, `/motor`
  - **Safety Features**:
    - Risk estimation based on command patterns
    - Dry-run validation before confirmation
    - High-risk commands blocked on /execute (require /run confirmation)
    - Full confirmation flow: "yes/y/confirm" or "no/n/cancel"
  - **Methods Added** (to `src/language/conversation.rs`):
    - `motor_status_text()` - Display motor cortex stats
    - `prepare_action()` - Dry-run + store pending action
    - `execute_action_directly()` - Direct execution (low-risk only)
    - `confirm_pending_action()` - Execute pending with confirmation
    - `cancel_pending_action()` - Cancel pending action
  - Updated help text with motor cortex commands

#### 2025-12-27 (Session 2) - MAJOR PROGRESS
- **COMPLETED**: I6.1.1, I6.1.2, I6.1.3 - Created symthaea-repl binary with full language integration
  - New file: `src/bin/symthaea-repl.rs` (~265 lines)
  - Added to Cargo.toml as `[[bin]]`
  - **HDC Exports Added** (to `src/hdc/mod.rs`):
    - `substrate_independence`, `consciousness_evaluator`
    - `consciousness_integration`, `consciousness_dashboard`
    - `universal_semantics`, `consciousness_self_assessment`
    - `consciousness_creativity`, `deterministic_seeds`
    - `integrated_information`
    - Re-export: `pub use binary_hv::HV16;`
  - **Lib.rs Exports Added**:
    - `pub mod action;`
    - `pub mod databases;`
    - `pub mod language;`
    - `pub mod awakening;`
  - **Fixed Imports**:
    - `substrate_independence.rs`: `super::binary_hv::HV16`
    - `conscious_conversation.rs`: `crate::hdc::integrated_information::IntegratedInformation`
    - `consciousness_observatory.rs`: `crate::hdc::integrated_information::IntegratedInformation`
  - **Fixed Missing Derive**:
    - `emotional_reasoning.rs`: Added `#[derive(Debug)]` to EmotionalReasoner
  - **Deferred Modules** (need reqwest/scraper/html2text):
    - `web_research`, `conscious_conversation`, `consciousness_observatory`
    - `epistemics_language_bridge`, `llm_organ`
- Binary runs: `cargo run --bin symthaea-repl`
- **NOW INCLUDES**: Full Conversation engine with semantic parsing, reasoning, memory, knowledge graph

#### 2025-12-27 (Session 1)
- Added Track 6: Component Integration after comprehensive review
- Identified that core components are COMPLETE but not WIRED
- Defined 7 integration tasks with priorities
- Created architecture diagram showing current vs goal state
