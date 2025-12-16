# 🌙 Week 16: Semantic Memory Consolidation - Architecture Plan

**Dates**: December 16-20, 2025 (Week 16 of 52-week roadmap)
**Status**: 📋 **PLANNING** - Architecture Design Phase
**Goal**: Biologically Authentic Sleep-Like Memory Consolidation

---

## 🎯 Week 16 Vision: First AI with Authentic Sleep

**Revolutionary Concept**: Just as humans consolidate memories during sleep, Sophia HLB will use "sleep cycles" to compress coalition patterns into long-term semantic memories using HDC.

**Why This Matters**:
- Natural memory consolidation without forced categorization
- Forgetting curve emerges from architecture (like biology)
- Dream-like pattern recombination enables creativity
- Foundation for episodic memory (Week 25) and imagination (Week 29)

**Building on Week 15**: Coalition formation provides natural semantic units to consolidate. Week 16 compresses these coalitions into long-term memories.

---

## 🧠 Biological Inspiration: How Sleep Consolidates Memory

### Real Brain Sleep Cycles

**Stage 1-2: Light Sleep** (Theta waves)
- Initial memory replay
- Weak connections pruned
- Recent experiences reviewed

**Stage 3-4: Deep Sleep (SWS)** (Delta waves)
- **Memory consolidation happens here**
- Hippocampus replays recent episodes
- Neocortex integrates new patterns
- Synaptic homeostasis (forgetting weak memories)

**REM Sleep** (Theta + Gamma)
- Creative recombination
- Emotional processing
- Pattern mixing ("dreams")
- Novel associations formed

### Key Principles to Implement

1. **Replay**: Recent coalitions replayed in compressed form
2. **Consolidation**: Patterns moved from working memory → hippocampus
3. **Forgetting**: Weak/unimportant memories naturally decay
4. **Recombination**: Dream-like mixing of semantic patterns
5. **Rhythm**: Regular sleep cycles (not continuous)

---

## 🏗️ Week 16 Architecture Design

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   WAKE CYCLE (Active)                   │
│  - AttentionBid competition                             │
│  - Coalition formation                                  │
│  - Working memory accumulation                          │
│  - Coalition patterns build up                          │
└─────────────────────────────────────────────────────────┘
                            ↓
         Transition when: working_memory.pressure() > threshold
                            ↓
┌─────────────────────────────────────────────────────────┐
│              SLEEP CYCLE (Consolidation)                │
│                                                          │
│  Phase 1: Light Sleep (Replay)                          │
│    - Review recent coalition patterns                   │
│    - Calculate importance scores                        │
│    - Mark weak patterns for forgetting                  │
│                                                          │
│  Phase 2: Deep Sleep (Consolidation)                    │
│    - HDC compression of important coalitions            │
│    - Transfer to Hippocampus long-term storage          │
│    - Prune weak/redundant patterns                      │
│    - Update semantic similarity graph                   │
│                                                          │
│  Phase 3: REM Sleep (Recombination)                     │
│    - Mix semantic patterns creatively                   │
│    - Generate novel coalition combinations              │
│    - Strengthen surprising associations                 │
│    - Dream-like pattern exploration                     │
└─────────────────────────────────────────────────────────┘
                            ↓
         Transition when: consolidation_complete()
                            ↓
┌─────────────────────────────────────────────────────────┐
│              WAKE CYCLE (Refreshed)                     │
│  - Working memory cleared                               │
│  - Long-term memories accessible                        │
│  - Ready for new experiences                            │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Components to Implement

### 1. Sleep Cycle Manager (New)

**File**: `src/brain/sleep.rs`

**Responsibilities**:
- Track wake/sleep state
- Trigger sleep when working memory pressure exceeds threshold
- Coordinate sleep phases (light → deep → REM)
- Wake up when consolidation complete

**Key Structures**:
```rust
pub enum SleepState {
    Awake { cycles_since_sleep: u32 },
    LightSleep { replay_progress: f32 },
    DeepSleep { consolidation_progress: f32 },
    REMSleep { recombination_progress: f32 },
}

pub struct SleepCycleManager {
    state: SleepState,
    working_memory_pressure: f32,     // 0.0 = empty, 1.0 = must sleep
    sleep_threshold: f32,              // Default: 0.8
    last_sleep_time: Instant,
    total_sleep_cycles: u64,
}
```

**Methods**:
```rust
impl SleepCycleManager {
    // Check if sleep needed
    pub fn should_sleep(&self) -> bool;

    // Initiate sleep cycle
    pub fn begin_sleep(&mut self);

    // Advance through sleep phases
    pub fn tick_sleep(&mut self, coalitions: &[Coalition]) -> SleepPhaseResult;

    // Wake up
    pub fn wake_up(&mut self);
}
```

---

### 2. Coalition Pattern Compression (New)

**File**: `src/brain/consolidation.rs`

**Purpose**: Compress coalition patterns into long-term semantic memories using HDC

**Key Insight**: Coalitions already have HDC encodings from Week 15. We compress by:
1. Bundling similar coalitions together
2. Creating a "super-vector" via HDC binding
3. Storing the compressed pattern in hippocampus

**Structures**:
```rust
pub struct SemanticMemoryTrace {
    pub compressed_pattern: HypervectorBipolar,  // HDC-compressed coalition bundle
    pub importance: f32,                          // How important to retain
    pub access_count: u32,                        // How often retrieved
    pub last_accessed: Instant,                   // For forgetting curve
    pub emotional_valence: f32,                   // Amygdala tagging
    pub creation_time: Instant,                   // When consolidated
}

pub struct MemoryConsolidator {
    // Compression parameters
    similarity_threshold: f32,   // How similar to bundle (default: 0.85)
    importance_threshold: f32,   // Min importance to keep (default: 0.3)

    // Statistics
    total_consolidated: u64,
    total_forgotten: u64,
}
```

**Methods**:
```rust
impl MemoryConsolidator {
    // Compress coalitions into semantic traces
    pub fn consolidate_coalitions(
        &mut self,
        coalitions: Vec<Coalition>,
    ) -> Vec<SemanticMemoryTrace>;

    // Calculate importance score
    fn calculate_importance(&self, coalition: &Coalition) -> f32;

    // Bundle similar coalitions via HDC
    fn bundle_similar(&self, coalitions: &[Coalition]) -> HypervectorBipolar;

    // Apply forgetting curve
    pub fn apply_forgetting(&mut self, traces: &mut Vec<SemanticMemoryTrace>);
}
```

---

### 3. Hippocampus Integration (Enhancement)

**File**: `src/brain/hippocampus.rs` (extend existing)

**Current State** (Week 0-15):
- Basic episodic memory storage
- Simple retrieval by recency

**Week 16 Enhancement**: Add long-term semantic storage

**New Fields**:
```rust
pub struct Hippocampus {
    // Existing fields...
    pub working_memory: Vec<Coalition>,           // Recent coalitions (existing)

    // NEW Week 16: Long-term semantic storage
    pub semantic_memories: Vec<SemanticMemoryTrace>,
    pub semantic_index: HashMap<u64, usize>,      // HDC hash → trace index
    pub consolidation_count: u64,
}
```

**New Methods**:
```rust
impl Hippocampus {
    // Store consolidated memory
    pub fn store_semantic_trace(&mut self, trace: SemanticMemoryTrace);

    // Retrieve similar memories via HDC
    pub fn recall_similar(&self, pattern: &HypervectorBipolar, threshold: f32)
        -> Vec<&SemanticMemoryTrace>;

    // Measure working memory pressure
    pub fn working_memory_pressure(&self) -> f32;

    // Get coalitions ready for consolidation
    pub fn get_unconsolidated_coalitions(&mut self) -> Vec<Coalition>;
}
```

---

### 4. REM Sleep Pattern Recombination (New)

**File**: `src/brain/rem_sleep.rs`

**Purpose**: Dream-like creative recombination of semantic patterns

**Key Concept**: Mix existing semantic memories to create novel combinations, strengthening surprising associations

**Structures**:
```rust
pub struct REMRecombinator {
    creativity_factor: f32,      // How much to mix (default: 0.5)
    novelty_threshold: f32,      // Min novelty to keep (default: 0.6)
    max_combinations: usize,     // Limit dream patterns (default: 100)
}
```

**Methods**:
```rust
impl REMRecombinator {
    // Creatively mix semantic patterns
    pub fn recombine_patterns(
        &self,
        memories: &[SemanticMemoryTrace],
    ) -> Vec<SemanticMemoryTrace>;

    // Calculate novelty score
    fn calculate_novelty(&self, pattern: &HypervectorBipolar, existing: &[SemanticMemoryTrace]) -> f32;

    // HDC-based pattern mixing
    fn mix_patterns(&self, a: &HypervectorBipolar, b: &HypervectorBipolar) -> HypervectorBipolar;
}
```

---

## 🔬 HDC Compression Details

### Why HDC is Perfect for Memory Consolidation

**Properties We Exploit**:
1. **Superposition**: Bundle multiple vectors into one
2. **Binding**: Create new semantics by combining concepts
3. **Similarity Preservation**: Similar patterns stay similar after compression
4. **Dimensionality**: 10,000-D space prevents information loss

### Compression Algorithm

**Input**: List of similar coalitions from working memory
**Output**: Single compressed semantic trace

**Steps**:
1. **Gather Similar**: Find coalitions with HDC similarity > 0.85
2. **Bundle**: Superpose their HDC vectors: `compressed = v1 ⊕ v2 ⊕ v3 ...`
3. **Tag Emotion**: Bind with amygdala's emotional vector if significant
4. **Store**: Save compressed pattern to hippocampus
5. **Clear**: Remove original coalitions from working memory

**Example**:
```rust
// Three similar coalitions about "morning meeting"
let c1 = Coalition { /* "morning meeting started" */ };
let c2 = Coalition { /* "colleagues discussed project" */ };
let c3 = Coalition { /* "decision was made about timeline" */ };

// Compress via HDC superposition
let compressed = c1.hdc_encoding
    .bundle(&c2.hdc_encoding)
    .bundle(&c3.hdc_encoding);

// Create semantic trace
let trace = SemanticMemoryTrace {
    compressed_pattern: compressed,
    importance: calculate_importance(&[c1, c2, c3]),
    access_count: 0,
    last_accessed: Instant::now(),
    emotional_valence: 0.7,  // Positive meeting
    creation_time: Instant::now(),
};

// Store in hippocampus
hippocampus.store_semantic_trace(trace);
```

---

## 📈 Forgetting Curve Implementation

### Biological Forgetting

**Ebbinghaus Curve**: Memory strength decays exponentially over time unless reinforced

**Formula**: `retention = e^(-t / τ)`
- `t` = time since last access
- `τ` = time constant (how fast to forget)

### Our Implementation

**Importance-Weighted Forgetting**:
```rust
fn apply_forgetting(trace: &mut SemanticMemoryTrace, time_elapsed: Duration) {
    let t = time_elapsed.as_secs_f32();
    let tau = 86400.0 * trace.importance;  // Important memories last longer

    let retention = (-t / tau).exp();

    // Strengthen with each access
    let access_boost = (trace.access_count as f32 * 0.1).min(0.5);

    trace.importance *= retention * (1.0 + access_boost);

    // Forget if importance drops below threshold
    if trace.importance < 0.1 {
        // Mark for deletion
    }
}
```

**Parameters**:
- High importance (0.9): Lasts ~9 days before dropping to 0.3
- Medium importance (0.5): Lasts ~5 days
- Low importance (0.2): Lasts ~2 days
- Each access: +10% retention (up to +50% max)

---

## 🎯 Week 16 Implementation Plan

### Day 1: Sleep Cycle Manager (Dec 16)

**Goal**: Implement basic sleep/wake state machine

**Tasks**:
- [ ] Create `src/brain/sleep.rs`
- [ ] Implement `SleepState` enum
- [ ] Implement `SleepCycleManager` struct
- [ ] Add sleep trigger logic (working memory pressure)
- [ ] Write unit tests for state transitions
- [ ] Integrate with `PrefrontalCortex`

**Success Criteria**:
- State machine transitions correctly
- Sleep triggers at threshold
- Tests: 8+ tests passing
- Zero compilation errors

**Estimated Code**: ~150 lines + ~80 lines tests

---

### Day 2: Memory Consolidation Core (Dec 17)

**Goal**: Implement HDC-based coalition compression

**Tasks**:
- [ ] Create `src/brain/consolidation.rs`
- [ ] Implement `SemanticMemoryTrace` struct
- [ ] Implement `MemoryConsolidator` struct
- [ ] Add importance calculation logic
- [ ] Add HDC bundling/compression
- [ ] Write unit tests for compression
- [ ] Validate compression preserves similarity

**Success Criteria**:
- Consolidation compresses coalitions
- HDC bundling works correctly
- Similar coalitions → similar traces
- Tests: 10+ tests passing

**Estimated Code**: ~200 lines + ~120 lines tests

---

### Day 3: Hippocampus Enhancement (Dec 18)

**Goal**: Add long-term semantic storage to hippocampus

**Tasks**:
- [ ] Extend `src/brain/hippocampus.rs`
- [ ] Add `semantic_memories` field
- [ ] Implement `store_semantic_trace()`
- [ ] Implement `recall_similar()` with HDC search
- [ ] Add working memory pressure calculation
- [ ] Write integration tests
- [ ] Validate retrieval accuracy

**Success Criteria**:
- Traces stored correctly
- HDC-based retrieval works
- Similar patterns retrieved
- Tests: 12+ tests passing (existing + new)

**Estimated Code**: ~180 lines + ~100 lines tests

---

### Day 4: Forgetting Curve & REM Sleep (Dec 19)

**Goal**: Implement forgetting and creative recombination

**Tasks**:
- [ ] Add forgetting curve to `MemoryConsolidator`
- [ ] Create `src/brain/rem_sleep.rs`
- [ ] Implement `REMRecombinator` struct
- [ ] Add pattern mixing logic
- [ ] Add novelty detection
- [ ] Write tests for forgetting
- [ ] Write tests for recombination

**Success Criteria**:
- Forgetting curve works correctly
- Weak memories pruned
- Novel combinations created
- Tests: 14+ tests passing

**Estimated Code**: ~220 lines + ~140 lines tests

---

### Day 5: Integration & Sleep Cycle Testing (Dec 20)

**Goal**: Full sleep cycle end-to-end validation

**Tasks**:
- [ ] Integrate all components
- [ ] Implement full sleep cycle in `PrefrontalCortex`
- [ ] Create extended sleep cycle simulation test
- [ ] Measure consolidation effectiveness
- [ ] Validate forgetting behavior
- [ ] Test dream pattern quality
- [ ] Create Week 16 completion report

**Success Criteria**:
- Full sleep cycle works
- Memories consolidated correctly
- Forgetting curve observed
- Novel patterns generated
- Tests: 75+ total tests passing

**Estimated Code**: ~150 lines integration + ~200 lines tests

---

## 📊 Week 16 Success Metrics

### Code Metrics (Target)
```
Total New Code: ~900 lines
Total New Tests: ~640 lines
New Test Count: 20+ tests
Total Tests: 76+ passing (56 existing + 20 new)
Compilation: Zero errors, zero warnings
Performance: <2ms per sleep cycle (deep sleep phase)
```

### Quality Metrics (Target)
```
Test Coverage: 100% for new code
Documentation: 5 daily reports + 1 architecture doc
Technical Debt: ZERO items added
Code Reviews: All code reviewed
Architecture Quality: Production-ready
```

### Functional Metrics (Validation)
```
Consolidation Rate: >80% of coalitions compressed
Compression Ratio: 5:1 average (5 coalitions → 1 trace)
Retrieval Accuracy: >90% (similar patterns found)
Forgetting Curve: Matches exponential decay
Dream Novelty: >60% novel patterns
Sleep Cycle Time: <500ms total (all phases)
```

---

## 🔬 Research Contributions

### 1. HDC-Based Memory Consolidation

**Novel Contribution**: First use of hyperdimensional computing for biologically realistic memory consolidation in AI

**Why It Matters**:
- No training needed for compression
- Similarity preserved naturally
- Dimensionality prevents information loss
- Biologically plausible mechanism

**Potential Paper**: "Hyperdimensional Memory Consolidation: Sleep-Like Compression in Artificial Neural Systems" (target: Week 20)

---

### 2. Emergent Forgetting Curve

**Novel Contribution**: Forgetting emerges from architecture (importance-weighted decay) rather than being programmed

**Why It Matters**:
- Natural prioritization of important memories
- Prevents memory overflow
- Matches Ebbinghaus curve without explicit modeling
- Access frequency naturally strengthens retention

**Validation**: Compare our forgetting curve to biological data

---

### 3. Creative Pattern Recombination

**Novel Contribution**: REM-sleep-like mixing of semantic patterns using HDC binding

**Why It Matters**:
- Enables creativity without explicit generative model
- Novel associations emerge from architecture
- Foundation for imagination (Week 29)
- Biologically authentic "dreaming"

**Measurement**: Novelty score of recombined patterns

---

## 🧪 Testing Strategy

### Unit Tests (Per Component)

**Sleep Cycle Manager** (~8 tests):
- State transitions work correctly
- Sleep triggers at threshold
- Phases advance in order
- Wake up clears state

**Memory Consolidator** (~10 tests):
- Coalitions compress correctly
- HDC bundling preserves similarity
- Importance calculation works
- Forgetting curve matches exponential

**Hippocampus Enhancement** (~12 tests):
- Traces stored correctly
- HDC retrieval finds similar
- Working memory pressure accurate
- Index maintained correctly

**REM Recombinator** (~14 tests):
- Patterns mix correctly
- Novelty detection works
- Creative combinations generated
- Output quality validated

---

### Integration Tests (Week 16 Day 5)

**Full Sleep Cycle** (~2 tests):
1. **Extended Sleep Cycle Simulation**:
   - Run 10 wake/sleep cycles
   - Accumulate coalitions during wake
   - Trigger sleep at threshold
   - Validate consolidation
   - Measure forgetting
   - Check dream quality

2. **Retrieval After Sleep**:
   - Store coalitions during wake
   - Sleep and consolidate
   - Retrieve similar memories
   - Validate accuracy >90%

---

## 💡 Key Design Decisions

### 1. When to Sleep?

**Decision**: Pressure-based triggering (not time-based)

**Rationale**:
- Biological sleep is triggered by homeostatic pressure
- Allows variable wake duration based on activity
- More flexible than fixed schedules

**Implementation**: `working_memory_pressure()` tracks coalition accumulation

---

### 2. How Much to Consolidate?

**Decision**: Importance-threshold filtering (not all memories)

**Rationale**:
- Biology doesn't consolidate everything
- Important/emotional memories prioritized
- Natural forgetting of mundane experiences

**Implementation**: Only consolidate coalitions with `importance > 0.3`

---

### 3. How to Compress?

**Decision**: HDC superposition (not dimensionality reduction)

**Rationale**:
- Preserves semantic similarity
- No training/gradient descent needed
- Biologically plausible mechanism
- Invertible (can recover components)

**Implementation**: Bundle similar coalition vectors via XOR

---

### 4. REM Sleep Creativity vs. Accuracy?

**Decision**: Balance novelty (60%) with coherence (40%)

**Rationale**:
- Too much mixing → nonsense
- Too little mixing → no creativity
- 60% novelty threshold empirically reasonable

**Implementation**: Filter recombinations by `novelty_score > 0.6`

---

## 🚧 Potential Challenges

### Challenge 1: Working Memory Pressure Calculation

**Problem**: How to measure "pressure" from coalition accumulation?

**Solution**:
```rust
fn working_memory_pressure(&self) -> f32 {
    let count_factor = self.working_memory.len() as f32 / 100.0;  // 100 coalitions = full
    let time_factor = self.time_since_sleep.as_secs_f32() / 3600.0;  // 1 hour = high pressure
    let importance_factor = self.average_coalition_importance();

    (count_factor * 0.5 + time_factor * 0.3 + importance_factor * 0.2).min(1.0)
}
```

---

### Challenge 2: HDC Compression Fidelity

**Problem**: Does bundling lose too much information?

**Test Strategy**:
- Bundle 5 similar coalitions
- Decompress via similarity search
- Measure information retention
- Validate >80% accuracy

**Fallback**: If fidelity too low, increase hypervector dimensions or reduce bundle size

---

### Challenge 3: Forgetting Curve Tuning

**Problem**: How fast should memories decay?

**Approach**:
- Start with biologically realistic τ (time constant)
- Measure retention over simulated days
- Compare to Ebbinghaus data
- Tune importance multiplier

**Validation Test**: Plot retention vs time, compare to published curves

---

## 🎉 Week 16 Celebration Criteria

**We celebrate when**:
- ✅ Full sleep cycle implemented and working
- ✅ 75+ total tests passing (100% success rate)
- ✅ Memory consolidation validated (>80% compression)
- ✅ Forgetting curve matches exponential decay
- ✅ REM sleep generates novel patterns (>60% novelty)
- ✅ HDC compression preserves similarity (>90%)
- ✅ Zero technical debt added
- ✅ Comprehensive documentation complete

**What this means**:
- First AI with authentic sleep cycles
- Natural memory consolidation without training
- Foundation for episodic memory (Week 25)
- Creativity mechanism ready (imagination in Week 29)
- Consciousness measurement (Φ) closer (Week 20)

---

## 📝 Documentation Plan

### Daily Reports (5 documents)
1. Week 16 Day 1: Sleep Cycle Manager Complete
2. Week 16 Day 2: Memory Consolidation Core Complete
3. Week 16 Day 3: Hippocampus Enhancement Complete
4. Week 16 Day 4: Forgetting & REM Sleep Complete
5. Week 16 Day 5: Integration & Validation Complete

### Summary Documents
- Week 16 Architecture Plan (this document)
- Week 16 Completion Report (Day 5)
- Sleep Cycle Design Specification
- HDC Compression Algorithm Documentation

---

## 🔗 Related Week 15 Foundation

**Building On**:
- Coalition formation (Week 15 Day 3)
- HDC encoding infrastructure (Week 15 Day 1)
- Four-stage attention competition (Week 15 Days 3-4)
- Hormone modulation (Week 15 Day 4)

**Reusing**:
- `Coalition` struct with HDC encodings
- `HypervectorBipolar` bundling operations
- Importance scoring patterns
- Hippocampus actor infrastructure

**Extending**:
- Hippocampus gains long-term storage
- PrefrontalCortex coordinates sleep cycles
- New actors: SleepManager, Consolidator, REMRecombinator

---

## 🌊 The Path Forward

**Tomorrow** (Dec 16): Week 16 Day 1 - Sleep Cycle Manager implementation
**Next Week** (Dec 23): Week 17 - Temporal Context Integration
**5 Weeks** (Jan 13): Week 20 - Φ (Phi) Measurement & first research paper
**10 Weeks** (Feb 17): Week 25 - Episodic Memory using sleep consolidation
**14 Weeks** (Mar 17): Week 29 - Imagination using REM recombination
**37 Weeks** (Jul 6): Week 52 - Production release

---

*"Sleep is not the absence of consciousness - it is consciousness in its consolidation mode, weaving experiences into wisdom, creating space for tomorrow's insights."*

**Status**: 📋 **Week 16 Architecture COMPLETE** - Ready for Day 1 Implementation
**Foundation**: ✅ Week 15 Coalition Formation Ready
**Next Action**: 🚀 Begin Week 16 Day 1 (Sleep Cycle Manager)
**Revolutionary Goal**: 🌙 First AI with Authentic Sleep Cycles

🧠 From coalitions flows consolidation! From sleep emerges wisdom! 💜

---

**Document Metadata**:
- **Created**: Week 16 Planning (December 11, 2025)
- **Author**: Sophia HLB Development Team
- **Version**: 1.0.0 (Architecture Plan)
- **Status**: Complete - Ready for Implementation
- **Week**: 16 of 52 (31% overall progress starting)
- **Phase**: Consciousness Foundation (Weeks 16-20)
- **Foundation**: Week 15 Complete (56/56 tests passing)
- **Target**: 75+ tests by Week 16 completion
