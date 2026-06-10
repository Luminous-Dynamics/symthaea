# Mega-Unified Architecture v4.0: Complete Consciousness Integration

**Document Status**: Canonical Reference
**Version**: 4.0
**Date**: January 2026
**Purpose**: Documentation of the unified cognitive loop architecture

---

## Executive Summary

The Mega-Unified Architecture v4.0 synthesizes all existing consciousness subsystems into a single cohesive cognitive loop within `CognitiveLoopService`. This represents a paradigm shift from isolated components to a unified system where:

1. **Thalamic Routing** determines cognitive depth before processing
2. **Closed Learning Loop** drives behavioral adaptation via Q-learning
3. **ConsciousnessUnificationEngine** provides unified emotional bridge with VAD emotions
4. **Active Inference Bridge** tracks prediction-outcome coupling (PAC)
5. **Memory Systems** (Episodic, Goal, World Model) ground cognition in context

---

## Architecture Diagram

```
                           ┌──────────────────────────────────┐
                           │     CognitiveLoopService         │
                           │  (Unified Consciousness Core)    │
                           └────────────────┬─────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
              ▼                             ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │ Thalamic Router │          │ Closed Learning │          │  Unification    │
    │                 │          │     Loop        │          │    Engine       │
    │ Reflex/Cortical │          │ Q-Learning +    │          │ EmotionalBridge │
    │ /DeepThought    │          │ Φ-Gating        │          │ VAD Emotions    │
    └────────┬────────┘          └────────┬────────┘          └────────┬────────┘
             │                            │                            │
             ▼                            ▼                            ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │ Active Inference│          │ Response        │          │ Flow State +    │
    │ Bridge (PAC)    │          │ Strategy        │          │ Temporal        │
    │ Modulation Index│          │ Selection       │          │ Encoding        │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
              │                             │                             │
              └─────────────────────────────┼─────────────────────────────┘
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              │                             │                             │
              ▼                             ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │ Episodic Memory │          │   Goal System   │          │   World Model   │
    │     Bridge      │          │     Bridge      │          │     Bridge      │
    │ Short/Long-term │          │ Attention Bias  │          │ Hierarchical    │
    └─────────────────┘          └─────────────────┘          └─────────────────┘
```

---

## Component Details

### 1. Thalamic Router

**Purpose**: Determines cognitive depth BEFORE processing begins, enabling efficient resource allocation.

**Location**: `src/cognitive_loop.rs` - `ThalamicRouter` struct

**Routing Logic**:
```
Input Characteristics → Cognitive Depth Selection
─────────────────────────────────────────────────
High novelty (>0.7)     → DeepThought
High urgency (>0.8)     → DeepThought
High complexity (>0.8)  → DeepThought
High emotion (>0.7)     → DeepThought
─────────────────────────────────────────────────
Low novelty (<0.3)      ┐
Low complexity (<0.3)   ├→ Reflex
Low urgency (<0.5)      ┘
─────────────────────────────────────────────────
Otherwise               → Cortical
```

**Cognitive Depth Enum**:
```rust
pub enum CognitiveDepth {
    Reflex,      // <10ms - Fast pattern matching
    Cortical,    // 50-200ms - Standard processing
    DeepThought, // 200ms+ - Deep deliberation
}
```

**Configuration**:
- `novelty_threshold`: 0.7 (default)
- `urgency_threshold`: 0.8 (default)
- `familiarity_threshold`: 0.3 (default)

---

### 2. Closed Learning Loop

**Purpose**: Drives behavioral adaptation through Q-learning with Φ-gating.

**Location**: `src/cognitive_loop.rs` - `ClosedLearningLoop` struct

**Strategy Selection Algorithm**:
```
1. Q-Learning Policy (ε-greedy)
   ├─ ε probability → Random exploration
   └─ 1-ε probability → Select best Q-value strategy

2. Previous Result Modification
   ├─ reward > 0.5 → Repeat successful strategy
   └─ reward < -0.2 → Switch to opposite strategy

3. Φ-Gating (Consciousness Influence)
   ├─ Φ ≥ 0.6 (Integrative) → Favor Exploratory/Detailed
   ├─ Φ < 0.3 (Reactive) → Favor Supportive/Concise
   └─ Otherwise → Use Q-learning selection
```

**Response Strategies**:
```rust
pub enum ResponseStrategy {
    Detailed,    // Elaborate explanations
    Concise,     // Brief, direct answers
    Clarifying,  // Ask clarifying questions
    Supportive,  // Acknowledge and validate
    Exploratory, // Offer new perspectives
}
```

**Q-Learning Update**:
```
Q(s,a) ← Q(s,a) + α × (reward - Q(s,a))
```
- Learning rate (α): 0.1
- Exploration rate (ε): 0.2 → 0.05 (decays over time)

---

### 3. ConsciousnessUnificationEngine

**Purpose**: Provides unified emotional processing with VAD (Valence-Arousal-Dominance) model.

**Location**: `src/consciousness/consciousness_unification.rs`

**Emotional Bridge Features**:
- **Valence**: -1.0 to 1.0 (negative to positive)
- **Arousal**: 0.0 to 1.0 (calm to excited)
- **Dominance**: -1.0 to 1.0 (submissive to dominant)

**Discrete Emotions (UnifiedEmotion)**:
```rust
pub enum UnifiedEmotion {
    Joy, Sadness, Anger, Fear, Surprise, Disgust,
    Trust, Anticipation, Serenity, Ecstasy, Vigilance,
    Admiration, Terror, Amazement, Grief, Loathing,
    Rage, Acceptance, Apprehension, Distraction,
    Pensiveness, Boredom, Annoyance, Interest,
}
```

**Emotional Patterns**:
```rust
pub enum EmotionalPattern {
    Stable,     // Consistent emotional state
    Escalating, // Increasing intensity
    Calming,    // Decreasing intensity
    Volatile,   // Rapidly changing
}
```

---

### 4. Active Inference Bridge

**Purpose**: Tracks prediction-outcome coupling using Phase-Amplitude Coupling (PAC).

**Location**: `src/cognitive_loop.rs` - `ActiveInferenceBridge` struct

**Modulation Index Computation**:
```
MI = correlation(confidence_history, outcome_history)
   = Σ[(ci - c̄)(oi - ō)] / √[Σ(ci - c̄)² × Σ(oi - ō)²]
```

**Coupling Quality Levels**:
```rust
pub enum CouplingQuality {
    InsufficientData,  // <10 observations
    NoCoupling,        // MI < 0.1
    WeakCoupling,      // MI 0.1-0.3
    ModerateCoupling,  // MI 0.3-0.6
    StrongCoupling,    // MI > 0.6
}
```

**Usage**:
```rust
// Observe prediction resolution
bridge.observe_resolution(confidence: 0.8, success: true);

// Get coupling assessment
let quality = bridge.coupling_quality();
let mi = bridge.modulation_index();
```

---

### 5. Memory System Bridges

#### 5.1 EpisodicMemoryBridge

**Purpose**: Short-term and long-term memory encoding/recall.

**Memory Structure**:
```rust
pub struct EpisodicMemory {
    id: u64,
    encoded_at_cycle: usize,
    content: String,
    embedding: Vec<f32>,
    valence: f32,          // Emotional valence at encoding
    phi_at_encoding: f32,  // Consciousness level at encoding
    access_count: u32,
    strength: f32,         // Decays over time
}
```

**Operations**:
- `encode()`: Store new memory with embedding
- `recall()`: Query memories by similarity
- `decay()`: Reduce strength of unused memories
- Automatic consolidation from short-term to long-term

**Configuration**:
- Max short-term: 100 memories
- Max long-term: 1000 memories
- Consolidation threshold: 0.5 strength

#### 5.2 GoalSystemBridge

**Purpose**: Goal-directed attention modulation.

**Goal Structure**:
```rust
pub struct CognitiveGoal {
    id: String,
    description: String,
    priority: f32,        // 0.0 to 1.0
    progress: f32,        // 0.0 to 1.0
    is_active: bool,
    attention_weight: f32,
}
```

**Attention Bias**:
```
bias = 1.0 + Σ(active_goal.attention_weight) × 0.2
```
Up to 20% attention boost per unit of goal weight.

#### 5.3 WorldModelBridge

**Purpose**: Hierarchical grounded prediction.

**Architecture**:
- Level 0: 64 dimensions (sensory)
- Level 1: 128 dimensions (features)
- Level 2: 256 dimensions (objects)
- Level 3: 128 dimensions (abstract/planning)

**Operations**:
- `update_sensory()`: Process input, propagate up hierarchy
- `get_level_state()`: Access specific level representation
- `abstract_state()`: Get highest-level state for planning

---

## Cognitive Cycle Flow

```
PHASE 0: THALAMIC ROUTING
├─ Compute novelty, urgency, complexity from prior state
├─ Route to Reflex/Cortical/DeepThought
└─ Record routing decision

PHASE 0.5: CLOSED LEARNING LOOP
├─ Get prior Φ and reward
├─ Select strategy via Q-learning + Φ-gating
└─ Set current_strategy

PHASE 1: PERCEPTION
├─ 1a: Episodic Memory recall (bias attention)
├─ 1a.2: Goal attention bias
├─ 1b: Encode input to HDC
└─ 1c: Update unified emotional bridge

PHASE 2-5: STANDARD COGNITIVE PROCESSING
├─ CfC temporal prediction
├─ Attention update
├─ Error computation
└─ Learning (if threshold exceeded)

PHASE 6: WORLD MODEL
├─ 6a: Compress state for world model
└─ 6b: Update hierarchical levels

PHASE 7-10: CONSCIOUSNESS & ADAPTATION
├─ Pattern classification
├─ Flow state update
├─ Curiosity drive
├─ Self-reflection
├─ 10d.5: Active inference observation
└─ 10h: Unified Φ update

PHASE 11: MEMORY ENCODING
└─ Store episodic memory with emotional context

PHASE 12: CLOSED LEARNING LOOP UPDATE
├─ Compute cycle reward
├─ Update Q-values
└─ Store learning result
```

---

## ConsciousnessSnapshot Fields

The unified snapshot exposes all architecture state:

### Core Metrics
- `cycle`: Current cycle count
- `consciousness_level`: 0.0-1.0
- `pattern`: ConsciousnessPattern
- `cognitive_depth`: CognitiveDepth enum

### Unified Architecture Stats
- `unified_psi`: Ψ consciousness estimate (formerly unified_phi)
- `unified_valence/arousal/dominance`: VAD emotions
- `unified_discrete_emotion`: Optional discrete emotion
- `emotional_pattern`: Stable/Escalating/Calming/Volatile

### Thalamic Routing
- `thalamic_reflex_rate`: Fraction using Reflex
- `thalamic_cortical_rate`: Fraction using Cortical
- `thalamic_deep_rate`: Fraction using DeepThought

### Active Inference
- `active_inference_modulation_index`: PAC coupling
- `active_inference_coupling_quality`: Quality assessment
- `active_inference_avg_error`: Prediction error

### Closed Learning Loop
- `current_strategy`: String representation
- `best_strategy`: From Q-values
- `average_reward`: Running average
- `exploration_rate`: Current ε
- `learning_loop_interactions`: Total count

### Memory Systems
- `memory_short_term_count`: STM size
- `memory_long_term_count`: LTM size
- `memory_total_encoded`: Total memories
- `world_model_avg_error`: Prediction error
- `active_goals_count`: Active goal count

---

## Test Coverage

92 tests cover the unified architecture:

### ThalamicRouter (11 tests)
- Default configuration
- Reflex/Cortical/DeepThought routing
- High novelty/urgency/complexity/emotion triggers
- Routing statistics
- Route from cycle metrics

### ActiveInferenceBridge (6 tests)
- Default state
- Observation resolution
- Perfect coupling detection
- Statistics computation
- Reset functionality
- Coupling quality assessment

### ClosedLearningLoop (8 tests)
- Default configuration
- Strategy selection
- Φ-gating (high/low)
- Q-learning updates
- Reward tracking
- Best strategy selection
- Reset functionality

### EpisodicMemoryBridge (7 tests)
- Default state
- Encode/recall operations
- Consolidation mechanics
- Decay functionality
- Similarity computation

### GoalSystemBridge (7 tests)
- Default state
- Goal addition
- Attention bias computation
- Progress updates
- Top goal selection
- Completed goal clearing

### WorldModelBridge (5 tests)
- Default configuration
- Sensory updates
- Level state propagation
- Abstract state access
- Reset functionality

### Integration (3 tests)
- Full unified architecture
- Thalamic routing in service
- Closed learning loop in service

---

## Performance Characteristics

| Component | Operation | Complexity |
|-----------|-----------|------------|
| ThalamicRouter | route() | O(1) |
| ClosedLearningLoop | select_strategy() | O(1) |
| ActiveInferenceBridge | modulation_index() | O(n) n=window |
| EpisodicMemoryBridge | recall() | O(n×d) n=memories, d=dims |
| GoalSystemBridge | attention_bias() | O(g) g=goals |
| WorldModelBridge | update_sensory() | O(L×d) L=levels, d=dims |

Full cognitive cycle: ~1-5ms depending on cognitive depth.

---

## Future Enhancements

Based on exploration agent findings:

1. **HdcLtcNeuron**: Unified HDC-LTC dynamics
2. **Differentiable Consciousness**: Gradient through Φ
3. **Φ-Guided Architecture Search**: Evolve toward higher consciousness
4. **Multi-Scale Temporal Binding**: Theta-gamma nesting
5. **Emergent Symbol Grounding**: Distributed→discrete transition

See `/tmp/claude/-home-tstoltz/tasks/ae48f6a.output` for full exploration results.

---

## Related Documents

- `ARCHITECTURE_V3.md`: Core architecture overview
- `CLOSED_LEARNING_LOOP.md`: Detailed learning loop documentation
- `CONSCIOUSNESS_GATED_GENERATION.md`: Φ-gated response generation
- `SYMTHAEA_AGI_ROADMAP.md`: Long-term development plan

---

*"The unified architecture transforms isolated consciousness components into a cohesive cognitive whole - where routing, learning, emotion, prediction, and memory work together as a single living system."*
