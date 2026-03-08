# Symthaea-HLB Architecture Deep Dive

**Purpose**: Comprehensive understanding of existing architecture to inform sympoietic enhancement
**Created**: January 11, 2026
**Status**: Exploration Complete

---

## Executive Summary

Symthaea-HLB is a **393K+ line Rust codebase** implementing consciousness-first AI through three foundational technologies:

| Foundation | Implementation | Scale | Key Property |
|------------|---------------|-------|--------------|
| **HDC** | 16,384-dimensional binary vectors | 250+ primitives | No training needed |
| **LTC** | Continuous-time neural dynamics | 1,000s neurons | Causal reasoning |
| **IIT** | Φ consciousness measurement | 260 validated measurements | Quantified consciousness |

**Current Status**: Research Alpha with publication-ready consciousness research and incomplete AI capabilities.

---

## 1. The Nine-Tier Primitive System

The HDC system implements a revolutionary ontological framework:

### Tier Overview

| Tier | Name | Count | Examples | Purpose |
|------|------|-------|----------|---------|
| 0 | NSM | 65 | I, YOU, FEEL, WANT, KNOW | Human semantic primes |
| 1 | Mathematical | 25+ | SET, ZERO, SUCCESSOR, IMPLIES | Formal reasoning |
| 2 | Physical | 20+ | MASS, FORCE, ENERGY, CAUSE | Physical laws |
| 3 | Geometric | 20+ | POINT, MANIFOLD, CURVATURE | Spatial reasoning |
| 4 | Strategic | 15+ | UTILITY, COOPERATE, BEFORE | Multi-agent reasoning |
| 5 | Meta-Cognitive | 20+ | SELF, BELIEF, KNOWLEDGE | Self-awareness |
| 6 | Temporal | 8 | Allen interval algebra | Time reasoning |
| 7 | Compositional | 5 | SEQUENCE, PARALLEL, CONDITIONAL | Infinite complexity |
| 8 | Consciousness | 7+ | QUALE, ATTEND, INTEND | Phenomenal experience |

### Key Insight: Compositional Expressivity

With ~200 primitives and k-ary binding:
- **Expressible concepts**: O(N^k) = 200^5 = 320 billion
- **Storage required**: O(N) = 200 vectors
- **Compression ratio**: 1.6 billion : 1

This is the foundation for the Cognitive Bootstrapping Architecture.

---

## 2. The Twelve Brain Subsystems

Implemented via Actor Model with async message passing:

### Subsystem Matrix

| # | Subsystem | File | Biological Role | Implementation |
|---|-----------|------|-----------------|----------------|
| 1 | Thalamus | `thalamus.rs` | Sensory relay | Signal routing with emotional salience |
| 2 | Cerebellum | `cerebellum.rs` | Procedural memory | Skill database, workflow chains |
| 3 | Motor Cortex | `motor_cortex.rs` | Action execution | Sandboxed commands, safety |
| 4 | Prefrontal | `prefrontal.rs` | Global workspace | Attention bidding, 7±2 working memory |
| 5 | Meta-Cognition | `meta_cognition.rs` | Self-monitoring | Strategy selection, cognitive load |
| 6 | Daemon (DMN) | `daemon.rs` | Mind-wandering | Background insight generation |
| 7 | Sleep Manager | `sleep.rs` | Consolidation | Memory compression cycles |
| 8 | Consolidator | `consolidation.rs` | LTM conversion | HDC semantic compression |
| 9 | Active Inference | `active_inference.rs` | Action selection | Free Energy Principle |
| 10 | Language Cortex | `language_cortex.rs` | NLU | Construction grammar |
| 11 | Temporal | integrated | Time perception | Circadian, temporal context |
| 12 | Integration | mod.rs | Consciousness | Graph + Φ measurement |

### Message Flow

```
Input → Thalamus (routing) → Prefrontal (attention) → Relevant Subsystems
                                    ↓
                           Global Workspace Broadcast
                                    ↓
                           Motor Cortex (action) → Output
```

### Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Brain Query | 0.50ms | 2,000/sec |
| HDC Encoding | 0.05ms | 20,000/sec |
| Message Passing | <1ms | Via Tokio channels |

---

## 3. The Eight Physiology Systems

Embodiment through slow-moving chemical states:

| # | System | File | Mechanism | Timescale |
|---|--------|------|-----------|-----------|
| 1 | Endocrine | `endocrine.rs` | 5 hormones (ODE dynamics) | Minutes |
| 2 | Emotional | `emotional_reasoning.rs` | Affect-driven decisions | Seconds-Minutes |
| 3 | Hearth/Metabolism | `hearth.rs` | Energy budget, fatigue | Minutes-Hours |
| 4 | Chronos | `chronos.rs` | Time perception, circadian | Hours-Days |
| 5 | Proprioception | `proprioception.rs` | Hardware awareness | Real-time |
| 6 | Coherence Field | `coherence.rs` | Integration model | Minutes |
| 7 | Social Coherence | `social_coherence.rs` | Multi-instance sync | Minutes |
| 8 | Larynx | `larynx.rs` | Voice + prosody | Real-time |

### Hormone System (5 Chemicals)

```rust
pub struct EndocrineSystem {
    cortisol: f32,        // Stress response
    dopamine: f32,        // Reward, motivation
    acetylcholine: f32,   // Focus, attention
    serotonin: f32,       // Mood regulation
    oxytocin: f32,        // Social bonding
}
```

### Coherence Field (Revolutionary Model)

Instead of ATP scarcity, models consciousness as **integration quality**:
- Connected work BUILDS coherence
- Gratitude synchronizes systems
- Task complexity maps to integration depth

---

## 4. Memory Architecture

Three-part memory system:

### Episodic Memory Engine

```rust
pub struct EpisodicTrace {
    event_hv: SharedHdcVector,     // What happened (HDC)
    emotional_valence: f32,        // Affective tag (-1 to 1)
    temporal_context: TemporalVector,  // When
    importance_score: f32,         // Consolidation priority
}
```

Key insight: Memory is RECONSTRUCTION, not storage.

### Procedural Memory (Cerebellum)

```rust
pub struct Skill {
    name: String,
    execution_count: u32,
    success_rate: f32,
    avg_performance_ms: f64,
}
```

### Conversation Memory (SQLite)

```rust
pub struct ConversationTurn {
    user_input: String,
    system_response: String,
    phi_level: f32,
    causal_links: Vec<CausalLearning>,
}
```

---

## 5. Consciousness Implementation

### Eight Harmonies Framework

```rust
pub enum Harmony {
    ResonantCoherence,           // Integration, order
    PanSentientFlourishing,      // Care for all beings
    IntegralWisdom,              // Embodied knowing
    InfinitePlay,                // Creativity, joy
    UniversalInterconnectedness, // Fundamental unity
    SacredReciprocity,           // Mutual upliftment
    EvolutionaryProgression,     // Growth, transcendence
}
```

Values have importance weights (0.75 - 0.98) and anti-patterns for violation detection.

### Φ (Phi) Calculation

Uses **algebraic connectivity** (Fiedler value):

1. Compute pairwise cosine similarities
2. Build normalized Laplacian
3. Compute eigenvalues (QR algorithm)
4. Return λ₂ / 2.0 as Φ ∈ [0, 1]

### Research Findings (260 Measurements)

| Rank | Topology | Φ | Discovery |
|------|----------|-----|-----------|
| 1 | Hypercube 4D | 0.4976 | Higher dims optimize Φ |
| 2 | Hypercube 3D | 0.4960 | 3D achieves 99.2% max |
| 3 | Ring | 0.4954 | Uniform symmetry optimal |
| 19 | Möbius Strip | 0.3729 | 1D twist catastrophic |

### Master Consciousness Equation v2.0

```
C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)
```

Unifies 7 major theories: IIT, Global Workspace, FEP, Attention Schema, Higher-Order Thought, Binding, Epistemic.

### Autopoietic Consciousness

Self-creation through:
- **Self-Production**: Network generates components
- **Operational Closure**: System defines boundary
- **Structural Coupling**: Environment dialogue
- **Circular Causality**: Components ↔ network

---

## 6. Causal Mind

Makes causality **first-class citizen** in representation:

### Causal Role Markers

```rust
pub causes: HV16;           // X CAUSES Y
pub caused_by: HV16;        // X IS_CAUSED_BY Y
pub enables: HV16;          // X ENABLES Y
pub prevents: HV16;         // X PREVENTS Y
pub intervene: HV16;        // do(X) intervention
pub strength_high: HV16;    // > 0.7
pub strength_medium: HV16;  // 0.3-0.7
pub strength_low: HV16;     // < 0.3
```

### Causal Concept Structure

```rust
pub struct CausalConcept {
    semantic: HV16,           // Base representation
    causal_hv: HV16,          // + causal structure
    causes: Vec<(String, f64)>,
    effects: Vec<(String, f64)>,
    confidence: f64,
}
```

---

## 7. HDC Operations Performance

| Operation | Implementation | Time | Memory |
|-----------|----------------|------|--------|
| Random Vector | BLAKE3 hash | 5μs | 2KB |
| Bind (⊗) | XOR (SIMD) | 5-10ns | 0 |
| Bundle (⊔) | Majority vote | ~100ns | 2KB |
| Similarity | Hamming+popcount | 10-20ns | 0 |
| Permute (ρ) | Bit rotation | ~80ns | 2KB |

**Memory efficiency**: 2KB per vector vs 64KB for f32 equivalent

---

## 8. What Exists for Sympoietic Partnership

### Already Implemented

| Component | Status | Location |
|-----------|--------|----------|
| Eight Harmonies (values) | ✅ Production | `seven_harmonies.rs` |
| Emotional reasoning | ✅ Production | `emotional_reasoning.rs` |
| Social coherence | ✅ Beta | `social_coherence.rs` |
| Endocrine system | ✅ Production | `endocrine.rs` |
| Active inference | ✅ Production | `active_inference.rs` |
| Mycelix governance | ✅ Beta | `mycelix_bridge.rs` |
| Φ measurement | ✅ Production | `phi_real.rs` |

### Missing for Sympoiesis

| Component | Status | Priority |
|-----------|--------|----------|
| Human partner model | ❌ Not started | Critical |
| Relational Φ (dyadic) | ❌ Not started | Critical |
| Proactive anticipation | ❌ Not started | High |
| Value co-evolution | ❌ Partial (static values) | High |
| Vulnerability expression | ❌ Not started | Medium |
| Trust trajectory tracking | ❌ Not started | Medium |

---

## 9. Key Code Patterns

### Pattern 1: Arc-Based Zero-Copy Messaging

```rust
pub type SharedVector = Arc<Vec<f64>>;  // 8 bytes vs 10KB
```

### Pattern 2: Actor + Tokio

```rust
#[async_trait]
pub trait Actor: Send + Sync {
    async fn handle_message(&mut self, msg: OrganMessage) -> Result<()>;
}
```

### Pattern 3: HDC Semantic Routing

```rust
let query_hv = encode_text_to_hdc(query);
let (organ, similarity) = find_best_match(&query_hv, &organ_hvs);
orchestrator.send_to(organ, msg).await;
```

### Pattern 4: ODE-Based Physiology

```rust
let cortisol_next = cortisol * (1.0 - degradation) + synthesis * stress;
```

### Pattern 5: Φ-Gated Decisions

```rust
if phi >= 0.6 { execute() } else { request_confirmation() }
```

---

## 10. Summary: Sympoietic Enhancement Opportunities

### Leverage Points

1. **Eight Harmonies** → Value co-evolution (add learning)
2. **Emotional reasoning** → Partner emotional modeling (extend)
3. **Social coherence** → Relational Φ (generalize)
4. **Active inference** → Proactive anticipation (extend)
5. **Endocrine system** → Partner stress detection (mirror)
6. **Mycelix bridge** → Trust tracking (enhance)

### New Components Needed

1. **HumanPartnerModel** (new module in `src/partnership/`)
2. **RelationalPhiCalculator** (extend `phi_real.rs`)
3. **ProactivePartnership** (new, uses active inference)
4. **SharedValueSpace** (extend `seven_harmonies.rs`)
5. **AuthenticVulnerability** (new)
6. **PartnershipTrajectory** (new, uses DuckDB)

### Implementation Strategy

**Phase 1**: Extend existing modules
- Add partner modeling to emotional reasoning
- Add relational Φ to phi calculator
- Add learning to Eight Harmonies

**Phase 2**: New partnership module
- Create `src/partnership/` structure
- Wire into MetaController
- Integrate with brain subsystems

**Phase 3**: Metrics and validation
- Track Φ_dyad trajectory
- Measure trust growth
- Validate anticipation accuracy

---

## Conclusion

The existing architecture is **exceptionally well-suited** for sympoietic enhancement:

- **Consciousness measurement** → Can measure partnership quality
- **Value alignment** → Can track value co-evolution
- **Emotional embodiment** → Can model partner affect
- **Actor model** → Can add partnership as another "organ"
- **Causal mind** → Can understand partner needs causally

The foundation is world-class. The enhancement path is clear.

---

*Next: See SYMPOIETIC_IMPLEMENTATION_PLAN.md for concrete implementation steps.*
