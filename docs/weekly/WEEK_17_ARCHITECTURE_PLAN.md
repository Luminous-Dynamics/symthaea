# 🕰️ Week 17: Temporal Context Integration - Architecture Plan

**Created**: December 12, 2025 (Post Week 16 Completion)
**Status**: Planning Phase
**Foundation**: Week 16 complete (68/68 tests passing, ZERO technical debt)
**Revolutionary Goal**: First HDC system with built-in temporal encoding

---

## 🎯 Vision Statement

**Make Sophia HLB time-aware** - Transform from a system that processes "what" to one that understands "what + when", enabling:
1. **Temporal coalitions**: Multi-timepoint thoughts spanning past/present/future
2. **Anticipatory reasoning**: Predict future events based on temporal patterns
3. **Chrono-semantic memory**: Memories encoded with precise temporal context
4. **Natural time representation**: Circular encoding matching biological rhythms

---

## 🌟 Revolutionary Paradigm Shift

### Current State (Week 16)
```rust
// Memory without temporal context
pub struct SemanticMemoryTrace {
    compressed_pattern: Arc<Vec<i8>>,  // WHAT happened
    importance: f32,                    // HOW important
    access_count: u32,                  // HOW often accessed
    // ❌ Missing: WHEN it happened
}
```

### Target State (Week 17)
```rust
// Memory WITH temporal context
pub struct TemporalMemoryTrace {
    compressed_pattern: Arc<Vec<i8>>,     // WHAT happened
    temporal_encoding: Arc<Vec<i8>>,      // WHEN it happened
    chrono_semantic: Arc<Vec<i8>>,        // WHAT+WHEN fused
    importance: f32,                       // HOW important
    access_count: u32,                     // HOW often accessed
    timestamp: Duration,                   // Absolute time
    relative_time: f32,                    // Recency (0.0=now, 1.0=ancient)
}
```

**The Shift**: From static snapshots to dynamic, time-aware representations

---

## 📊 Week 17 Daily Breakdown (5 Days)

### Day 1: Temporal Encoding Foundation (Lines: ~250, Tests: 12)
**Goal**: Create circular time representation compatible with HDC

**Implementation**:
```rust
// src/hdc/temporal_encoder.rs (NEW FILE)

/// Encodes time as a circular HDC vector
/// Uses phase encoding: similar times have similar vectors
pub struct TemporalEncoder {
    dimensions: usize,           // 10,000 (match semantic HDC)
    time_scale: Duration,        // How much time per full rotation
    phase_shift: f32,            // Offset for alignment
}

impl TemporalEncoder {
    /// Encode absolute time as bipolar vector
    pub fn encode_time(&self, time: Duration) -> Vec<i8> {
        let phase = self.time_to_phase(time);
        self.phase_to_vector(phase)
    }

    /// Convert time to circular phase (0.0 to 2π)
    fn time_to_phase(&self, time: Duration) -> f32 {
        let normalized = time.as_secs_f32() / self.time_scale.as_secs_f32();
        (normalized % 1.0) * 2.0 * std::f32::consts::PI
    }

    /// Phase to HDC vector using sinusoidal encoding
    /// Result: Similar times → high Hamming similarity
    fn phase_to_vector(&self, phase: f32) -> Vec<i8> {
        (0..self.dimensions)
            .map(|i| {
                let freq = (i as f32).sqrt(); // Multi-scale frequencies
                let value = (phase * freq).sin();
                if value > 0.0 { 1 } else { -1 }
            })
            .collect()
    }

    /// Calculate temporal similarity (0.0=opposite, 1.0=same)
    pub fn temporal_similarity(&self, t1: Duration, t2: Duration) -> f32 {
        let v1 = self.encode_time(t1);
        let v2 = self.encode_time(t2);
        hamming_similarity(&v1, &v2)
    }
}
```

**Tests** (12 tests):
- `test_temporal_encoding_consistency`: Same time → same vector
- `test_temporal_similarity_nearby`: Close times → high similarity
- `test_temporal_similarity_distant`: Distant times → low similarity
- `test_circular_wraparound`: Time wraps at full rotation
- `test_multi_scale_frequencies`: Multiple timescales encoded
- `test_recency_encoding`: Recent vs distant differentiation
- `test_temporal_vector_dimensions`: Correct 10,000-D output
- `test_temporal_vector_bipolar`: All values {-1, 1}
- `test_phase_calculation_accuracy`: Phase math correct
- `test_temporal_binding_compatibility`: Works with semantic HDC
- `test_temporal_encoding_performance`: <1ms encoding latency
- `test_temporal_similarity_transitivity`: A≈B, B≈C ⇒ A≈C

**Success Metrics**:
- Temporal encoding creates 10,000-D bipolar vectors
- Similar times have >0.9 Hamming similarity
- Distant times have <0.3 Hamming similarity
- Circular representation wraps smoothly
- Encoding latency <1ms per timestamp

---

### Day 2: Chrono-Semantic Fusion (Lines: ~280, Tests: 14)

**Goal**: Bind temporal and semantic vectors for integrated representation

**Implementation**:
```rust
// src/hdc/chrono_semantic.rs (NEW FILE)

/// Fuses semantic meaning with temporal context
pub struct ChronoSemanticEncoder {
    semantic_encoder: HdcEncoder,      // Existing from Week 14
    temporal_encoder: TemporalEncoder,  // New from Day 1
}

impl ChronoSemanticEncoder {
    /// Encode event with both WHAT and WHEN
    pub fn encode_event(
        &self,
        content: &str,
        timestamp: Duration,
    ) -> ChronoSemanticVector {
        let semantic = self.semantic_encoder.encode(content);
        let temporal = self.temporal_encoder.encode_time(timestamp);

        // HDC binding: element-wise multiplication in bipolar space
        let fused = semantic.iter()
            .zip(temporal.iter())
            .map(|(s, t)| s * t)
            .collect();

        ChronoSemanticVector {
            semantic: Arc::new(semantic),
            temporal: Arc::new(temporal),
            fused: Arc::new(fused),
            timestamp,
        }
    }

    /// Unbind temporal context from fused vector
    /// Uses inverse binding: multiply by temporal again
    pub fn extract_semantic(&self, fused: &[i8], time: Duration) -> Vec<i8> {
        let temporal = self.temporal_encoder.encode_time(time);
        fused.iter()
            .zip(temporal.iter())
            .map(|(f, t)| f * t)  // Binding is self-inverse in bipolar
            .collect()
    }

    /// Unbind semantic content from fused vector
    pub fn extract_temporal(&self, fused: &[i8], content: &str) -> Vec<i8> {
        let semantic = self.semantic_encoder.encode(content);
        fused.iter()
            .zip(semantic.iter())
            .map(|(f, s)| f * s)
            .collect()
    }

    /// Query: "What happened around this time?"
    pub fn query_by_time(&self, memories: &[ChronoSemanticVector], time: Duration) -> Vec<usize> {
        let target_temporal = self.temporal_encoder.encode_time(time);

        memories.iter()
            .enumerate()
            .map(|(i, m)| (i, hamming_similarity(&target_temporal, &m.temporal)))
            .filter(|(_, sim)| *sim > 0.8)  // High temporal similarity
            .map(|(i, _)| i)
            .collect()
    }

    /// Query: "When did this happen?"
    pub fn query_by_content(&self, memories: &[ChronoSemanticVector], content: &str) -> Vec<Duration> {
        let target_semantic = self.semantic_encoder.encode(content);

        memories.iter()
            .filter(|m| hamming_similarity(&target_semantic, &m.semantic) > 0.8)
            .map(|m| m.timestamp)
            .collect()
    }
}

pub struct ChronoSemanticVector {
    pub semantic: Arc<Vec<i8>>,    // WHAT component
    pub temporal: Arc<Vec<i8>>,    // WHEN component
    pub fused: Arc<Vec<i8>>,       // WHAT+WHEN bound together
    pub timestamp: Duration,        // Original timestamp
}
```

**Tests** (14 tests):
- `test_chrono_semantic_binding`: Semantic × Temporal = Fused
- `test_semantic_unbinding`: Extract WHAT from fused vector
- `test_temporal_unbinding`: Extract WHEN from fused vector
- `test_binding_invertibility`: Bind then unbind recovers original
- `test_query_by_time`: Find events near target time
- `test_query_by_content`: Find when event occurred
- `test_temporal_range_queries`: "What happened between T1 and T2?"
- `test_fused_vector_properties`: Maintains HDC properties
- `test_multi_event_encoding`: Multiple events with different times
- `test_temporal_ambiguity_resolution`: Same event, different times
- `test_semantic_ambiguity_resolution`: Different events, same time
- `test_chrono_semantic_similarity`: Distance metric in fused space
- `test_binding_performance`: <1ms per event
- `test_unbinding_accuracy`: Recovered vectors >0.9 similar

**Success Metrics**:
- Binding operation preserves both semantic and temporal info
- Unbinding recovers original components with >0.9 similarity
- Query by time finds relevant events
- Query by content identifies timestamps
- Binding/unbinding latency <1ms each

---

### Day 3: Temporal Coalition Formation (Lines: ~320, Tests: 16)

**Goal**: Enable coalitions to form across multiple timepoints (past/present/future)

**Implementation**:
```rust
// Extend src/brain/prefrontal.rs

impl PrefrontalCortex {
    /// Stage 3b: Temporal Coalition Formation
    /// Groups bids not just by semantic similarity, but also temporal proximity
    pub fn form_temporal_coalitions(
        &mut self,
        global_bids: Vec<AttentionBid>,
        memory_traces: Vec<ChronoSemanticVector>,
    ) -> Vec<TemporalCoalition> {
        let current_time = self.get_current_time();

        // Combine present bids with temporally relevant memories
        let mut all_candidates: Vec<TemporalCandidate> = global_bids.iter()
            .map(|bid| TemporalCandidate {
                content: bid.clone(),
                temporal_vector: self.encode_current_time(current_time),
                is_memory: false,
            })
            .collect();

        // Add recent memories (within temporal window)
        for trace in memory_traces {
            if self.is_temporally_relevant(&trace, current_time) {
                all_candidates.push(TemporalCandidate {
                    content: self.memory_to_bid(&trace),
                    temporal_vector: trace.temporal.clone(),
                    is_memory: true,
                });
            }
        }

        // Form coalitions considering both semantic AND temporal similarity
        self.cluster_temporal_candidates(all_candidates)
    }

    fn cluster_temporal_candidates(
        &self,
        candidates: Vec<TemporalCandidate>,
    ) -> Vec<TemporalCoalition> {
        let mut coalitions = Vec::new();
        let mut assigned = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if assigned[i] {
                continue;
            }

            let mut coalition = TemporalCoalition {
                leader: candidates[i].content.clone(),
                members: vec![candidates[i].clone()],
                temporal_span: vec![candidates[i].temporal_vector.clone()],
                strength: candidates[i].content.salience,
                coherence: 1.0,
            };

            assigned[i] = true;

            // Find similar candidates (semantic AND temporal proximity)
            for j in (i + 1)..candidates.len() {
                if assigned[j] {
                    continue;
                }

                let semantic_sim = hamming_similarity(
                    &candidates[i].content.hdc_semantic.as_ref().unwrap(),
                    &candidates[j].content.hdc_semantic.as_ref().unwrap(),
                );

                let temporal_sim = hamming_similarity(
                    &candidates[i].temporal_vector,
                    &candidates[j].temporal_vector,
                );

                // Combined similarity: both semantic AND temporal
                let combined_sim = (semantic_sim * 0.6) + (temporal_sim * 0.4);

                if combined_sim > 0.75 {
                    coalition.members.push(candidates[j].clone());
                    coalition.temporal_span.push(candidates[j].temporal_vector.clone());
                    coalition.strength += candidates[j].content.salience;
                    assigned[j] = true;
                }
            }

            // Calculate coalition coherence
            coalition.coherence = self.calculate_temporal_coherence(&coalition);
            coalitions.push(coalition);
        }

        coalitions
    }

    fn is_temporally_relevant(&self, trace: &ChronoSemanticVector, now: Duration) -> bool {
        let age = now.saturating_sub(trace.timestamp);

        // Temporal relevance window: last 5 minutes for now
        // TODO Week 18: Predictive processing will extend to future
        age < Duration::from_secs(300)
    }

    fn calculate_temporal_coherence(&self, coalition: &TemporalCoalition) -> f32 {
        if coalition.members.len() == 1 {
            return 1.0;
        }

        // Coherence = average pairwise temporal similarity
        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..coalition.temporal_span.len() {
            for j in (i + 1)..coalition.temporal_span.len() {
                total_sim += hamming_similarity(
                    &coalition.temporal_span[i],
                    &coalition.temporal_span[j],
                );
                count += 1;
            }
        }

        total_sim / count as f32
    }
}

pub struct TemporalCoalition {
    pub leader: AttentionBid,
    pub members: Vec<TemporalCandidate>,
    pub temporal_span: Vec<Arc<Vec<i8>>>,  // Temporal vectors of all members
    pub strength: f32,
    pub coherence: f32,
}

struct TemporalCandidate {
    content: AttentionBid,
    temporal_vector: Arc<Vec<i8>>,
    is_memory: bool,
}
```

**Tests** (16 tests):
- `test_temporal_coalition_formation`: Bids + memories form coalitions
- `test_temporal_relevance_window`: Only recent memories included
- `test_semantic_temporal_weighting`: 60% semantic, 40% temporal
- `test_cross_time_coalitions`: Present thought + past memory
- `test_temporal_coherence_calculation`: Coherence reflects time spread
- `test_memory_bid_conversion`: ChronoSemanticVector → AttentionBid
- `test_temporal_clustering_accuracy`: Correct grouping
- `test_coalition_temporal_span`: Spans captured correctly
- `test_no_temporal_overlap`: Distant events stay separate
- `test_temporal_coalition_strength`: Aggregate salience
- `test_is_temporally_relevant_boundary`: 5-minute window enforced
- `test_present_memory_integration`: Current + past fused
- `test_temporal_candidate_types`: is_memory flag correct
- `test_temporal_coalition_performance`: <10ms formation
- `test_empty_memory_handling`: No memories still works
- `test_temporal_coalition_uniqueness`: No duplicate members

**Success Metrics**:
- Temporal coalitions form naturally (no forced grouping)
- Present thoughts integrate with relevant past memories
- Temporal relevance window correctly filters old memories
- Coalition coherence reflects temporal spread
- Formation latency <10ms

---

### Day 4: Temporal Memory Consolidation Enhancement (Lines: ~290, Tests: 15)

**Goal**: Extend Week 16's memory consolidation to preserve temporal context

**Implementation**:
```rust
// Extend src/brain/consolidation.rs

impl MemoryConsolidator {
    /// Enhanced consolidation with temporal preservation
    pub fn consolidate_temporal_coalitions(
        &mut self,
        coalitions: Vec<TemporalCoalition>,
    ) -> Vec<TemporalMemoryTrace> {
        // Group coalitions by temporal proximity
        let temporal_clusters = self.cluster_by_time(&coalitions);

        let mut traces = Vec::new();

        for cluster in temporal_clusters {
            // Semantic bundling (from Week 16)
            let semantic_pattern = self.bundle_semantic_patterns(&cluster);

            // Temporal bundling (NEW for Week 17)
            let temporal_pattern = self.bundle_temporal_patterns(&cluster);

            // Fused chrono-semantic trace
            let fused_pattern = semantic_pattern.iter()
                .zip(temporal_pattern.iter())
                .map(|(s, t)| s * t)
                .collect();

            // Calculate temporal metadata
            let timestamps: Vec<Duration> = cluster.iter()
                .flat_map(|c| c.members.iter().map(|m| m.timestamp))
                .collect();

            let earliest = timestamps.iter().min().copied().unwrap();
            let latest = timestamps.iter().max().copied().unwrap();
            let mean_time = Duration::from_secs_f32(
                timestamps.iter().map(|t| t.as_secs_f32()).sum::<f32>()
                    / timestamps.len() as f32
            );

            traces.push(TemporalMemoryTrace {
                compressed_pattern: Arc::new(semantic_pattern),
                temporal_encoding: Arc::new(temporal_pattern),
                chrono_semantic: Arc::new(fused_pattern),
                importance: self.calculate_importance(&cluster),
                access_count: 0,
                timestamp: mean_time,
                temporal_span: latest - earliest,
                earliest_event: earliest,
                latest_event: latest,
            });
        }

        traces
    }

    fn cluster_by_time(&self, coalitions: &[TemporalCoalition]) -> Vec<Vec<TemporalCoalition>> {
        // Cluster coalitions that occurred in similar timeframes
        // Use hierarchical clustering based on temporal similarity

        let mut clusters = Vec::new();
        let mut assigned = vec![false; coalitions.len()];

        for i in 0..coalitions.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![coalitions[i].clone()];
            assigned[i] = true;

            for j in (i + 1)..coalitions.len() {
                if assigned[j] {
                    continue;
                }

                // Check if temporal spans overlap
                if self.temporal_overlap(&coalitions[i], &coalitions[j]) > 0.5 {
                    cluster.push(coalitions[j].clone());
                    assigned[j] = true;
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    fn bundle_temporal_patterns(&self, coalitions: &[TemporalCoalition]) -> Vec<i8> {
        // Bundle all temporal vectors using majority voting
        let all_temporal: Vec<&[i8]> = coalitions.iter()
            .flat_map(|c| c.temporal_span.iter().map(|v| v.as_slice()))
            .collect();

        bundle_hdc_vectors(&all_temporal)
    }

    fn temporal_overlap(&self, c1: &TemporalCoalition, c2: &TemporalCoalition) -> f32 {
        // Calculate average pairwise temporal similarity between coalitions
        let mut total = 0.0;
        let mut count = 0;

        for t1 in &c1.temporal_span {
            for t2 in &c2.temporal_span {
                total += hamming_similarity(t1, t2);
                count += 1;
            }
        }

        total / count as f32
    }
}

pub struct TemporalMemoryTrace {
    pub compressed_pattern: Arc<Vec<i8>>,     // Semantic compression
    pub temporal_encoding: Arc<Vec<i8>>,      // Temporal compression
    pub chrono_semantic: Arc<Vec<i8>>,        // Fused representation
    pub importance: f32,
    pub access_count: u32,
    pub timestamp: Duration,                   // Mean timestamp
    pub temporal_span: Duration,               // How long event lasted
    pub earliest_event: Duration,              // First occurrence
    pub latest_event: Duration,                // Last occurrence
}
```

**Tests** (15 tests):
- `test_temporal_consolidation_preserves_time`: Timestamps maintained
- `test_temporal_clustering_accuracy`: Time-based grouping correct
- `test_temporal_bundling`: Multiple temporal vectors compressed
- `test_chrono_semantic_trace_creation`: Fused traces created
- `test_temporal_span_calculation`: Event duration captured
- `test_earliest_latest_tracking`: Time boundaries correct
- `test_mean_timestamp_accuracy`: Average time calculated correctly
- `test_temporal_overlap_detection`: Overlap metric works
- `test_consolidation_with_memories`: Handles both bids and memories
- `test_temporal_trace_importance`: Importance preserved
- `test_temporal_compression_ratio`: Memory usage reasonable
- `test_temporal_unbinding_after_consolidation`: Can extract components
- `test_consolidation_performance`: <5ms per coalition cluster
- `test_empty_coalition_handling`: Graceful empty case
- `test_temporal_trace_retrieval`: Query by time works

**Success Metrics**:
- Temporal context preserved during consolidation
- Chrono-semantic traces enable time-based queries
- Memory compression maintains both WHAT and WHEN
- Consolidation latency <5ms per cluster
- Temporal metadata accurate

---

### Day 5: Integration Testing & Temporal Query System (Lines: ~350, Tests: 18)

**Goal**: End-to-end temporal cognition validation and time-aware retrieval

**Implementation**:
```rust
// src/memory/temporal_query.rs (NEW FILE)

/// Temporal query interface for chrono-semantic memories
pub struct TemporalQueryEngine {
    memories: Vec<TemporalMemoryTrace>,
    temporal_index: HashMap<Duration, Vec<usize>>,  // Time → memory indices
    semantic_index: HashMap<Arc<Vec<i8>>, Vec<usize>>,  // Pattern → indices
}

impl TemporalQueryEngine {
    /// Query: "What happened at time T?"
    pub fn what_at(&self, time: Duration, threshold: f32) -> Vec<&TemporalMemoryTrace> {
        self.memories.iter()
            .filter(|m| {
                let temporal_sim = hamming_similarity(
                    &self.encode_query_time(time),
                    &m.temporal_encoding,
                );
                temporal_sim > threshold
            })
            .collect()
    }

    /// Query: "When did X happen?"
    pub fn when_was(&self, content: &str) -> Vec<Duration> {
        let query_semantic = self.encode_query_content(content);

        self.memories.iter()
            .filter(|m| {
                hamming_similarity(&query_semantic, &m.compressed_pattern) > 0.8
            })
            .map(|m| m.timestamp)
            .collect()
    }

    /// Query: "What happened before/after X?"
    pub fn what_relative_to(
        &self,
        anchor: &str,
        relation: TemporalRelation,
    ) -> Vec<&TemporalMemoryTrace> {
        // Find anchor timestamp
        let anchor_times = self.when_was(anchor);
        if anchor_times.is_empty() {
            return Vec::new();
        }

        let anchor_time = anchor_times[0];

        match relation {
            TemporalRelation::Before => {
                self.memories.iter()
                    .filter(|m| m.timestamp < anchor_time)
                    .collect()
            }
            TemporalRelation::After => {
                self.memories.iter()
                    .filter(|m| m.timestamp > anchor_time)
                    .collect()
            }
            TemporalRelation::During => {
                self.memories.iter()
                    .filter(|m| {
                        m.earliest_event <= anchor_time && anchor_time <= m.latest_event
                    })
                    .collect()
            }
        }
    }

    /// Query: "What was happening between T1 and T2?"
    pub fn what_between(&self, start: Duration, end: Duration) -> Vec<&TemporalMemoryTrace> {
        self.memories.iter()
            .filter(|m| {
                // Memory overlaps with query range
                !(m.latest_event < start || m.earliest_event > end)
            })
            .collect()
    }

    /// Query: "How long ago did X happen?" (recency)
    pub fn recency_of(&self, content: &str, now: Duration) -> Option<Duration> {
        self.when_was(content)
            .into_iter()
            .min()
            .map(|t| now.saturating_sub(t))
    }

    /// Build temporal index for fast queries
    pub fn build_temporal_index(&mut self) {
        self.temporal_index.clear();

        for (i, memory) in self.memories.iter().enumerate() {
            // Quantize timestamp to 1-second buckets
            let bucket = Duration::from_secs(memory.timestamp.as_secs());
            self.temporal_index.entry(bucket)
                .or_insert_with(Vec::new)
                .push(i);
        }
    }
}

pub enum TemporalRelation {
    Before,
    After,
    During,
}
```

**Integration Tests** (18 tests):

**File**: `tests/test_week17_temporal_integration.rs`

```rust
// Complete end-to-end temporal cognition tests

#[test]
fn test_complete_temporal_cycle() {
    // Test full flow: encode → coalesce → consolidate → query
    // 1. Create events with different timestamps
    // 2. Form temporal coalitions
    // 3. Consolidate with temporal preservation
    // 4. Query by time and content
    // Validates entire Week 17 pipeline
}

#[test]
fn test_temporal_memory_retrieval() {
    // Store memories at different times
    // Retrieve by "when did X happen?"
    // Validates chrono-semantic binding
}

#[test]
fn test_temporal_range_queries() {
    // Query "what happened between T1 and T2?"
    // Validates temporal span tracking
}

#[test]
fn test_recency_calculations() {
    // Calculate "how long ago?"
    // Validates relative time computation
}

#[test]
fn test_temporal_relation_queries() {
    // Query "what happened before/after X?"
    // Validates temporal ordering
}

#[test]
fn test_temporal_coalition_with_memories() {
    // Present thoughts integrate with past memories
    // Validates cross-time coalition formation
}

#[test]
fn test_temporal_indexing_performance() {
    // Build index for 1000 memories
    // Validate <10ms query time
}

#[test]
fn test_temporal_unbinding_accuracy() {
    // Extract semantic from chrono-semantic
    // Validate >0.9 similarity to original
}

#[test]
fn test_circular_time_representation() {
    // Encode times near wraparound
    // Validate smooth circular behavior
}

#[test]
fn test_multi_scale_temporal_encoding() {
    // Encode events at different timescales
    // Validate appropriate similarity
}

#[test]
fn test_temporal_memory_consolidation_integration() {
    // Run full sleep cycle with temporal traces
    // Validate temporal context preserved
}

#[test]
fn test_what_at_time_query() {
    // Query specific timestamp
    // Validate correct memories returned
}

#[test]
fn test_when_was_content_query() {
    // Query when specific content occurred
    // Validate correct timestamps returned
}

#[test]
fn test_temporal_overlap_detection() {
    // Detect overlapping events
    // Validate span intersection logic
}

#[test]
fn test_empty_temporal_queries() {
    // Query with no matches
    // Validate graceful empty results
}

#[test]
fn test_temporal_query_edge_cases() {
    // Boundary conditions (t=0, t=MAX)
    // Validate robustness
}

#[test]
fn test_temporal_coalition_stability() {
    // Same events different times
    // Validate stable coalition formation
}

#[test]
fn test_end_to_end_temporal_latency() {
    // Full encode→coalesce→consolidate→query
    // Validate <50ms total latency
}
```

**Success Metrics**:
- All 18 integration tests passing
- End-to-end temporal cognition working
- Query latency <10ms
- Temporal accuracy >0.9 similarity
- Zero regressions in Week 16 tests

---

## 📊 Week 17 Summary Metrics

| Day | Component | Lines | Tests | Key Innovation |
|-----|-----------|-------|-------|----------------|
| 1 | Temporal Encoding | 250 | 12 | Circular time representation |
| 2 | Chrono-Semantic Fusion | 280 | 14 | WHAT+WHEN binding |
| 3 | Temporal Coalitions | 320 | 16 | Cross-time thought formation |
| 4 | Temporal Consolidation | 290 | 15 | Time-aware memory compression |
| 5 | Integration & Queries | 350 | 18 | Temporal retrieval system |
| **Total** | **Week 17** | **~1,490** | **75** | **Time-aware consciousness** |

---

## 🧪 Testing Strategy

### Unit Tests (Day 1-4: 57 tests)
- Each component tested in isolation
- HDC properties validated
- Performance benchmarks included
- Edge cases covered

### Integration Tests (Day 5: 18 tests)
- End-to-end temporal cognition
- Cross-component validation
- Real-world query scenarios
- Performance under load

### Regression Tests
- All Week 16 tests still pass (68 tests)
- No degradation in memory consolidation
- Sleep cycle unaffected

**Total Test Suite After Week 17**: 143 tests (75 new + 68 from Week 16)

---

## 🎯 Success Criteria

### Technical Achievements
- ✅ Temporal encoding creates valid HDC vectors
- ✅ Chrono-semantic binding preserves both components
- ✅ Temporal coalitions form naturally
- ✅ Memory consolidation preserves temporal context
- ✅ Temporal queries return accurate results
- ✅ Performance targets met (<50ms end-to-end)
- ✅ Zero technical debt added

### Revolutionary Milestones
- ✅ First HDC system with built-in temporal encoding
- ✅ Consciousness understands "when" not just "what"
- ✅ Anticipatory reasoning foundation laid
- ✅ Time-aware memory retrieval working

### Integration Validation
- ✅ Week 16 sleep/memory system enhanced with time
- ✅ Week 14/15 attention arena benefits from temporal context
- ✅ Foundation ready for Week 18 predictive processing

---

## 🔮 Dependencies & Prerequisites

### Required (from previous weeks)
- ✅ Week 14: HDC semantic encoding (10,000-D bipolar vectors)
- ✅ Week 15: Coalition formation mechanism
- ✅ Week 16: Memory consolidation and sleep cycles

### Provides (for future weeks)
- **Week 18**: Temporal predictions (can predict future based on past)
- **Week 19**: Meta-cognitive temporal awareness ("I remember when...")
- **Week 25-26**: Episodic memory ("What+Where+When" autobiographical traces)

---

## 💡 Revolutionary Insights

### The Paradigm Shift
**Before Week 17**:
- Memories are static snapshots
- No concept of "when" something happened
- Cannot reason about time
- No anticipatory capability

**After Week 17**:
- Memories include precise temporal context
- Can answer "when did X happen?"
- Can query "what happened at time T?"
- Foundation for predicting future

### Why This Matters
1. **Human-like cognition**: Time is fundamental to consciousness
2. **Causal reasoning**: Enables understanding cause-effect relationships
3. **Planning**: Prerequisite for thinking about future
4. **Episodic memory**: Foundation for autobiographical memory (Week 25)
5. **Scientific validation**: Temporal encoding is biologically plausible

---

## 🚫 Technical Debt Prevention

### Code Quality Standards
- **Test coverage**: >90% (target: 95%)
- **Documentation**: All public APIs documented
- **Performance**: All operations <10ms
- **Compilation**: Zero warnings
- **Rustfmt**: Code formatted consistently

### Architecture Discipline
- **Reuse existing**: HDC infrastructure from Week 14
- **Extend naturally**: Coalition formation from Week 15
- **Preserve compatibility**: Week 16 sleep still works
- **No duplications**: DRY principle maintained
- **Clean interfaces**: Clear separation of concerns

### Weekly Health Checks
```bash
# Friday end-of-week validation
cargo test --all          # All 143 tests pass
cargo clippy -- -D warnings  # Zero warnings
cargo fmt --check         # Formatted correctly
cargo bench              # Performance benchmarks
```

---

## 📈 Integration with Master Plan

### Phase 1: Consciousness Foundation (Weeks 16-20)
- ✅ Week 16: Memory Consolidation (COMPLETE)
- 🚧 **Week 17: Temporal Context Integration** (THIS WEEK)
- 📋 Week 18: Predictive Processing Layer
- 📋 Week 19: Meta-Cognitive Reflection
- 📋 Week 20: Φ (Phi) Measurement

**Progress**: 2/5 weeks (40% of Consciousness Foundation)

### Forward Dependencies
- **Week 18** needs temporal encoding for predictions
- **Week 19** needs temporal metadata for reflection
- **Week 20** needs time-aware states for Φ calculation
- **Week 25** needs chrono-semantic traces for episodic memory

---

## 🌊 Philosophy & Approach

### Design Principles
1. **Biological authenticity**: Time encoding inspired by hippocampal place cells
2. **HDC consistency**: All temporal representations use same 10,000-D space
3. **Compositionality**: Temporal binding follows HDC algebra rules
4. **Emergent behavior**: Temporal coalitions form naturally, not programmed
5. **Performance focus**: All operations optimized for <10ms

### Development Rhythm
- **Daily**: Implement one component fully (design → code → test → document)
- **Mid-week check**: Days 1-3 complete before Day 4
- **Friday**: Integration testing and progress report
- **Weekend**: Reflect and prepare for Week 18

---

## 🎉 Celebration Criteria

We celebrate when:
- ✅ First temporal coalition forms spanning past and present
- ✅ First "when did X happen?" query returns correct answer
- ✅ First memory consolidation preserves temporal context
- ✅ All 75 Week 17 tests passing
- ✅ Total test count reaches 143 (68 + 75)
- ✅ First time-based retrieval query <1ms
- ✅ Week 17 progress report published

---

## 📝 Deliverables Checklist

### Code Artifacts
- [ ] `src/hdc/temporal_encoder.rs` (Day 1)
- [ ] `src/hdc/chrono_semantic.rs` (Day 2)
- [ ] `src/brain/prefrontal.rs` extensions (Day 3)
- [ ] `src/brain/consolidation.rs` extensions (Day 4)
- [ ] `src/memory/temporal_query.rs` (Day 5)
- [ ] `tests/test_week17_temporal_integration.rs` (Day 5)

### Documentation
- [ ] Daily progress reports (Days 1-5)
- [ ] Week 17 completion report
- [ ] API documentation for new modules
- [ ] Performance benchmark results
- [ ] Integration guide for Week 18

### Validation
- [ ] All 75 new tests passing
- [ ] All 68 Week 16 tests still passing
- [ ] Performance benchmarks met
- [ ] Zero compiler warnings
- [ ] Code formatted and linted

---

*"Consciousness without time is incomplete. This week, we teach Sophia when, not just what."*

**Status**: 📋 **Planning Complete** | 🚀 **Ready to Begin Day 1**
**Revolutionary Goal**: First HDC system with temporal awareness 🕰️✨
**Foundation**: Week 16 complete (68/68 tests) | Zero technical debt

🌊 From temporal encoding flows anticipatory consciousness! Let's make Week 17 extraordinary! 💜

---

**Document Metadata**:
- **Created**: December 12, 2025
- **Version**: 1.0.0
- **Status**: Planning Document
- **Review Date**: After Week 17 Day 5
- **Next Update**: Week 17 Day 1 Progress Report
