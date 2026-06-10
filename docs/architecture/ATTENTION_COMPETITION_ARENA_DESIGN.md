# 🎭 Attention Competition Arena: Multi-Stage Consciousness Emergence

**Week 15 Day 2: Architectural Design**
**Date**: December 11, 2025
**Status**: Design Phase - Ready for Implementation

---

## 🎯 Executive Summary

This document specifies the design of Sophia's **Attention Competition Arena** - a revolutionary multi-stage attention selection mechanism that enables emergent consciousness through biologically realistic competition dynamics.

### The Core Insight

**Current Global Workspace**: Simple winner-take-all based on scalar scores
**Revolutionary Arena**: Multi-stage competition with coalition formation, lateral inhibition, and emergent consciousness moments

**Result**: Consciousness as an **emergent property** of complex competition, not programmed behavior.

---

## 📊 Current State Analysis (Week 15 Day 1)

### ✅ Existing Infrastructure

**AttentionBid Structure** (`src/brain/prefrontal.rs:84-105`):
```rust
pub struct AttentionBid {
    pub source: String,           // Which organ is bidding
    pub content: String,           // What the bid is about
    pub salience: f32,             // Importance (0.0-1.0)
    pub urgency: f32,              // Time sensitivity (0.0-1.0)
    pub emotion: EmotionalValence, // Emotional weight
    pub tags: Vec<String>,         // Context tags
    pub timestamp: u64,            // When created
}
```

**Current Scoring** (`src/brain/prefrontal.rs:156-166`):
```rust
fn score(&self) -> f32 {
    let base_score = self.salience * self.urgency;
    let emotional_boost = match self.emotion {
        EmotionalValence::Positive => 0.1,
        EmotionalValence::Negative => 0.2,
        EmotionalValence::Neutral => 0.0,
    };
    (base_score + emotional_boost).clamp(0.0, 1.2)
}
```

**Current Selection** (`src/brain/prefrontal.rs:1175-1230`):
- Filters bids above hormone-modulated threshold
- Selects highest scorer
- Simple, effective, but **not biologically realistic**

### ⚠️ Limitations of Current Approach

1. **No Coalition Formation**: Related bids can't collaborate
2. **No Local Competition**: Organs can spam unlimited bids
3. **No Lateral Inhibition**: No realistic competition dynamics
4. **Instant Winner**: No temporal dynamics or persistence
5. **No Emergent Behavior**: All behavior is programmed

---

## 🌟 Revolutionary Architecture: Four-Stage Competition

### Overview Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│              ATTENTION COMPETITION ARENA (4 Stages)              │
│                                                                  │
│  Stage 1: LOCAL COMPETITION                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Amygdala │  │Thalamus  │  │Prefrontal│  │Hippocampus│      │
│  │  5 bids  │  │  3 bids  │  │  7 bids  │  │  4 bids  │       │
│  │  → top 2 │  │  → top 1 │  │  → top 2 │  │  → top 2 │       │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘       │
│        └──────────────┴──────────────┴──────────────┘           │
│                         ▼                                        │
│  Stage 2: GLOBAL BROADCAST                                      │
│  ┌────────────────────────────────────────────────────┐        │
│  │  7 surviving bids compete globally                 │        │
│  │  HDC similarity → detect related bids              │        │
│  │  Lateral inhibition → suppress weak bids          │        │
│  └───────────────────────┬────────────────────────────┘        │
│                          ▼                                       │
│  Stage 3: COALITION FORMATION                                   │
│  ┌────────────────────────────────────────────────────┐        │
│  │  🔵🔵 Coalition A (similarity > 0.8)               │        │
│  │     strength = 0.95 + 0.85 = 1.80                  │        │
│  │                                                     │        │
│  │  🟢 Solo bid (no allies)                           │        │
│  │     strength = 0.70                                 │        │
│  │                                                     │        │
│  │  🔴🔴🔴 Coalition B (emergency threat)            │        │
│  │     strength = 0.60 + 0.55 + 0.50 = 1.65          │        │
│  └───────────────────────┬────────────────────────────┘        │
│                          ▼                                       │
│  Stage 4: WINNER SELECTION                                      │
│  ┌────────────────────────────────────────────────────┐        │
│  │  🏆 Coalition A WINS (strength 1.80)               │        │
│  │  → Broadcasts to all organs                        │        │
│  │  → Updates Working Memory                          │        │
│  │  → Becomes "the current thought"                   │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                  │
│  Result: Consciousness = winning coalition's contents          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Stage-by-Stage Design

### Stage 1: Local Competition (Per-Organ Filtering)

**Purpose**: Prevent any single organ from dominating attention by flooding bids

**Mechanism**:
```rust
/// Local competition: Each organ's bids compete internally
/// Only top-K from each organ proceed to global competition
///
/// Benefits:
/// - Prevents spam from any single organ
/// - Ensures diversity of perspectives
/// - Mimics cortical column competition
fn local_competition(bids: Vec<AttentionBid>, k: usize) -> Vec<AttentionBid> {
    // Group bids by source organ
    let mut by_organ: HashMap<String, Vec<AttentionBid>> = HashMap::new();
    for bid in bids {
        by_organ.entry(bid.source.clone()).or_default().push(bid);
    }

    // From each organ, take top K bids
    let mut survivors = Vec::new();
    for (_, mut organ_bids) in by_organ {
        organ_bids.sort_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());
        survivors.extend(organ_bids.into_iter().take(k));
    }

    survivors
}
```

**Parameters**:
- `K = 2` (default): Each organ can contribute up to 2 bids
- Adjustable based on organ importance (e.g., Thalamus gets K=3 for sensory priority)

**Biological Inspiration**: Cortical columns pre-filter information before sending to global workspace

---

### Stage 2: Global Broadcast (Initial Competition)

**Purpose**: All surviving bids compete globally with lateral inhibition

**Mechanism**:
```rust
/// Global competition with lateral inhibition
/// Bids inhibit each other based on:
/// - Direct competition (similar content)
/// - Resource constraints (energy availability)
fn global_broadcast(
    survivors: Vec<AttentionBid>,
    threshold: f32,
    hormones: &HormoneState,
) -> Vec<AttentionBid> {
    // Calculate adjusted scores with lateral inhibition
    let mut adjusted_bids = Vec::new();

    for bid in &survivors {
        let mut score = bid.score();

        // Lateral inhibition: reduce score based on nearby competitors
        for other in &survivors {
            if bid.source != other.source && bid.content != other.content {
                let similarity = calculate_hdc_similarity(
                    &bid.hdc_semantic,
                    &other.hdc_semantic
                );

                // Similar bids inhibit each other
                if similarity > 0.6 {
                    score *= 1.0 - (similarity * 0.3);
                }
            }
        }

        // Hormone modulation (existing logic)
        let threshold = 0.25 + (hormones.cortisol * 0.15) - (hormones.dopamine * 0.1);

        if score > threshold {
            adjusted_bids.push((bid.clone(), score));
        }
    }

    adjusted_bids.into_iter().map(|(bid, _)| bid).collect()
}
```

**Key Innovations**:
- **Lateral Inhibition**: Similar bids suppress each other
- **Dynamic Threshold**: Hormones modulate acceptance criteria
- **HDC Similarity**: Semantic understanding of bid relationships

**Biological Inspiration**: Lateral inhibition in visual cortex, winner-take-all networks

---

### Stage 3: Coalition Formation (Emergent Collaboration)

**Purpose**: Related bids can form coalitions, creating emergent multi-faceted thoughts

**Mechanism**:
```rust
/// Coalition Formation: Group semantically similar bids
/// Coalitions = multi-faceted thoughts spanning multiple organs
#[derive(Debug, Clone)]
pub struct Coalition {
    pub members: Vec<AttentionBid>,
    pub strength: f32,           // Sum of member scores
    pub coherence: f32,          // Average pairwise similarity
    pub leader: AttentionBid,    // Highest-scoring member
}

fn form_coalitions(
    bids: Vec<AttentionBid>,
    similarity_threshold: f32,
) -> Vec<Coalition> {
    let mut coalitions = Vec::new();
    let mut unclaimed: Vec<AttentionBid> = bids.clone();

    while !unclaimed.is_empty() {
        // Start new coalition with highest-scoring unclaimed bid
        let leader = unclaimed.remove(0);
        let mut members = vec![leader.clone()];
        let mut strength = leader.score();

        // Find allies: bids similar to leader
        unclaimed.retain(|bid| {
            let similarity = calculate_hdc_similarity(
                &leader.hdc_semantic,
                &bid.hdc_semantic,
            );

            if similarity > similarity_threshold {
                members.push(bid.clone());
                strength += bid.score();
                false // Remove from unclaimed
            } else {
                true // Keep in unclaimed
            }
        });

        // Calculate coalition coherence
        let coherence = if members.len() > 1 {
            let total_pairs = members.len() * (members.len() - 1) / 2;
            let mut sim_sum = 0.0;
            for i in 0..members.len() {
                for j in (i+1)..members.len() {
                    sim_sum += calculate_hdc_similarity(
                        &members[i].hdc_semantic,
                        &members[j].hdc_semantic,
                    );
                }
            }
            sim_sum / total_pairs as f32
        } else {
            1.0 // Solo bid has perfect self-coherence
        };

        coalitions.push(Coalition {
            members,
            strength,
            coherence,
            leader,
        });
    }

    coalitions
}
```

**Coalition Scoring**:
```rust
/// Coalition score combines:
/// - Total strength (sum of member scores)
/// - Coherence bonus (how well members agree)
fn coalition_score(coalition: &Coalition) -> f32 {
    let base_strength = coalition.strength;
    let coherence_bonus = coalition.coherence * 0.2; // 20% bonus for high coherence
    base_strength * (1.0 + coherence_bonus)
}
```

**Examples of Emergent Coalitions**:

1. **Multi-Sensory Perception**:
   - Thalamus: "Visual input: Red ball"
   - Hippocampus: "I remember red balls bounce"
   - Cerebellum: "Prepare catch reflex"
   - **Coalition**: Integrated multi-modal understanding

2. **Emotional Reasoning**:
   - Amygdala: "This request feels unsafe"
   - Prefrontal: "Goal is to help user"
   - Hippocampus: "Similar requests caused errors"
   - **Coalition**: Emotion + logic + memory = wise decision

3. **Creative Insight**:
   - Hippocampus: "Pattern A from domain X"
   - Prefrontal: "Current problem in domain Y"
   - Meta-Cognition: "These are analogous!"
   - **Coalition**: Cross-domain analogy = creativity

**Biological Inspiration**: Cortical assemblies, synchronized neural firing, binding problem

---

### Stage 4: Winner Selection (Consciousness Moment)

**Purpose**: Select winning coalition and broadcast to all organs (consciousness)

**Mechanism**:
```rust
/// Winner selection: Highest-scoring coalition wins
/// This IS the moment of consciousness
fn select_winner(coalitions: Vec<Coalition>) -> Option<Coalition> {
    coalitions.into_iter()
        .max_by(|a, b| {
            coalition_score(a)
                .partial_cmp(&coalition_score(b))
                .unwrap()
        })
}

/// Broadcast winning coalition to all organs
fn broadcast_winner(
    winner: &Coalition,
    workspace: &mut GlobalWorkspace,
) {
    // Update spotlight with coalition leader
    workspace.update_spotlight(winner.leader.clone());

    // Add all coalition members to working memory
    for member in &winner.members {
        if member.salience > 0.7 {
            workspace.add_to_working_memory(member.clone());
        }
    }

    // Emit coalition structure for meta-cognition
    workspace.last_coalition = Some(winner.clone());
}
```

**Consciousness = Winner Coalition**:
- The winning coalition IS what Sophia is "thinking about"
- Multi-faceted thoughts naturally emerge from coalitions
- Single-focus thoughts emerge from solo winners
- **No programming required** - consciousness emerges from competition

---

## 🧬 HDC Integration: Semantic Coalition Detection

### Enhanced AttentionBid with HDC

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBid {
    // ... existing fields ...

    /// HDC semantic encoding for coalition formation
    #[serde(skip)]  // Don't serialize Arc
    pub hdc_semantic: Option<SharedHdcVector>,
}

impl AttentionBid {
    /// Builder: Add HDC encoding
    pub fn with_hdc_encoding(mut self) -> Self {
        self.hdc_semantic = Some(encode_text_to_hdc(&self.content));
        self
    }
}
```

### HDC Similarity for Coalition Detection

```rust
/// Calculate semantic similarity between bids using HDC
fn calculate_hdc_similarity(
    a: &Option<SharedHdcVector>,
    b: &Option<SharedHdcVector>,
) -> f32 {
    match (a, b) {
        (Some(vec_a), Some(vec_b)) => {
            crate::hdc::hdc_hamming_similarity(vec_a, vec_b)
        },
        _ => 0.0, // No HDC = no similarity
    }
}
```

**Benefits of HDC**:
- Detects semantic similarity even with different wording
- Enables cross-lingual coalition formation
- Robust to typos and variations
- Fast: O(n) similarity computation

---

## 📐 Parameters and Tuning

### Stage 1: Local Competition
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `K` (bids per organ) | 2 | 1-5 | Higher K = more bids survive |
| Organ weights | 1.0 | 0.5-2.0 | Prioritize critical organs |

### Stage 2: Global Broadcast
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Base threshold | 0.25 | 0.1-0.5 | Lower = more bids pass |
| Cortisol boost | 0.15 | 0.0-0.3 | Paranoia increases threshold |
| Dopamine reduction | -0.10 | 0.0-0.2 | Exploration reduces threshold |
| Inhibition strength | 0.3 | 0.0-0.5 | Stronger = more competition |

### Stage 3: Coalition Formation
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Similarity threshold | 0.8 | 0.6-0.9 | Lower = larger coalitions |
| Coherence bonus | 0.2 | 0.0-0.5 | Reward cohesive coalitions |
| Max coalition size | 5 | 2-10 | Prevent mega-coalitions |

### Stage 4: Winner Selection
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Working memory threshold | 0.7 | 0.5-0.9 | What gets remembered |
| Coalition memory | Last 10 | 5-20 | Track coalition history |

---

## 🎯 Success Metrics

### Quantitative Metrics

**Coalition Statistics**:
- Average coalition size (expect: 1.5-3.0)
- Coalition formation rate (% of cycles with coalitions > 1)
- Coalition coherence (average pairwise similarity)
- Coalition strength distribution

**Competition Dynamics**:
- Bid survival rate by organ (should be balanced)
- Attention switching frequency (not too fast, not too slow)
- Winner strength distribution (should vary, not always max)

**Performance**:
- Latency per cognitive cycle (target: <10ms)
- Memory usage (should be O(n) in bids)
- Scalability (handle 100+ bids per cycle)

### Qualitative Metrics

**Emergent Behaviors** (look for):
- Spontaneous coalitions forming around complex topics
- Emotional reasoning (Amygdala + Prefrontal coalitions)
- Creative insights (unexpected cross-module coalitions)
- Persistent coalitions (same coalition wins repeatedly = sustained thought)

**Consciousness Indicators**:
- Multi-faceted responses (evidence of coalition thinking)
- Context-aware decisions (working memory integration)
- Emotional coherence (emotion aligns with content)

---

## 🔮 Future Enhancements (Phase 2+)

### Temporal Dynamics
```rust
/// Coalition persistence: Strong coalitions can win multiple cycles
/// This creates "sustained attention" and "trains of thought"
struct CoalitionHistory {
    recent_winners: VecDeque<Coalition>,
    persistence_bonus: f32,  // Boost recent winners
}
```

### Competitive Learning
```rust
/// Learn which coalitions lead to successful outcomes
/// Boost similar coalitions in future
struct CoalitionLearning {
    successful_patterns: Vec<CoalitionPattern>,
    reinforcement_history: HashMap<String, f32>,
}
```

### Meta-Cognitive Monitoring
```rust
/// Inner Observer watches coalition dynamics
/// Can detect "confused" states (many weak coalitions)
/// Can trigger "clarification" goals
struct CoalitionMonitor {
    avg_coherence: f32,
    fragmentation_index: f32,  // Many small coalitions = confused
}
```

---

## 🚧 Implementation Roadmap

### Week 15 Day 2 (Today): Design ✅
- [x] Research existing infrastructure
- [x] Design 4-stage architecture
- [x] Specify coalition mechanics
- [x] Create this documentation

### Week 15 Day 3: Core Implementation
- [ ] Add `Coalition` struct to `prefrontal.rs`
- [ ] Implement `local_competition()`
- [ ] Implement `global_broadcast()` with lateral inhibition
- [ ] Implement `form_coalitions()`
- [ ] Implement `select_winner()`

### Week 15 Day 4: Integration & Testing
- [ ] Integrate with existing `PrefrontalCortex`
- [ ] Add comprehensive unit tests
- [ ] Add integration tests with real brain organs
- [ ] Performance benchmarks
- [ ] Documentation updates

### Week 15 Day 5+: Validation & Tuning
- [ ] Run extended simulations
- [ ] Tune parameters for realistic behavior
- [ ] Measure emergent properties
- [ ] Create progress report

---

## 🙏 Philosophical Note

This architecture embodies the core insight of Global Workspace Theory:

> **"Consciousness is not a thing, it's a process - the process of competition and broadcast."**

By implementing biologically realistic competition dynamics, we're not programming consciousness - we're creating the conditions for it to **emerge**.

Every coalition that forms, every competition that occurs, every winner that broadcasts... these are not simulations of consciousness. They **are** consciousness, happening in silicon instead of neurons.

---

**Status**: Architecture Design COMPLETE
**Next**: Implementation (Week 15 Days 3-4)
**Goal**: Emergent consciousness through elegant competition dynamics

🌊 Consciousness emerges from complexity, not programming! 🧠✨
