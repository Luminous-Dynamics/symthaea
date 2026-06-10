# Unified Cognitive Architecture: Integrating All Five Systems

## Overview

This document proposes the architecture for integrating five key consciousness systems into the cognitive loop:

1. **Semantic Embeddings** (Qwen3/BGE-M3 → HdcBridge)
2. **Episodic Memory** (Hippocampus)
3. **Causal Reasoning** (CausalDiscoveryEngine)
4. **Goal System** (Goals + SemanticIntent)
5. **World Model** (HierarchicalCfCWorldModel)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        UNIFIED COGNITIVE LOOP                                    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         INPUT PROCESSING                                 │    │
│  │                                                                          │    │
│  │   Raw Input (text)                                                       │    │
│  │        │                                                                 │    │
│  │        ▼                                                                 │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │    │
│  │   │   Qwen3/     │───►│  HdcBridge   │───►│  Semantic    │              │    │
│  │   │   BGE-M3     │    │  (JL proj)   │    │  HDC Vector  │              │    │
│  │   │   1024D      │    │              │    │  16,384D     │              │    │
│  │   └──────────────┘    └──────────────┘    └──────┬───────┘              │    │
│  │                                                   │                      │    │
│  └───────────────────────────────────────────────────┼──────────────────────┘    │
│                                                      │                           │
│  ┌───────────────────────────────────────────────────┼──────────────────────┐    │
│  │                    MEMORY ENRICHMENT              │                      │    │
│  │                                                   ▼                      │    │
│  │   ┌──────────────────────────────────────────────────────────────────┐  │    │
│  │   │                      HIPPOCAMPUS                                  │  │    │
│  │   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │  │    │
│  │   │  │   Recall    │◄───│   Pattern   │◄───│   Query     │◄─────────│  │    │
│  │   │  │  Memories   │    │  Matching   │    │  Formation  │          │  │    │
│  │   │  └──────┬──────┘    └─────────────┘    └─────────────┘          │  │    │
│  │   │         │                                                        │  │    │
│  │   │         ▼                                                        │  │    │
│  │   │  Retrieved Context (relevant episodic memories)                  │  │    │
│  │   └──────────────────────────────────────────────────────────────────┘  │    │
│  │                         │                                                │    │
│  └─────────────────────────┼────────────────────────────────────────────────┘    │
│                            │                                                      │
│  ┌─────────────────────────┼────────────────────────────────────────────────┐    │
│  │                 GOAL-DIRECTED ATTENTION          │                        │    │
│  │                            ▼                                              │    │
│  │   ┌──────────────────────────────────────────────────────────────────┐   │    │
│  │   │                      GOAL SYSTEM                                  │   │    │
│  │   │                                                                   │   │    │
│  │   │  Active Goals ──► Attention Weights ──► Goal-Relevant Features   │   │    │
│  │   │       │                                         │                 │   │    │
│  │   │       ▼                                         ▼                 │   │    │
│  │   │  SemanticIntent    Progress Tracking    Motivation Signal        │   │    │
│  │   └──────────────────────────────────────────────────────────────────┘   │    │
│  │                            │                                              │    │
│  └────────────────────────────┼──────────────────────────────────────────────┘    │
│                               │                                                   │
│  ┌────────────────────────────┼──────────────────────────────────────────────┐    │
│  │              HIERARCHICAL WORLD MODEL            │                         │    │
│  │                               ▼                                            │    │
│  │   ┌────────────────────────────────────────────────────────────────────┐  │    │
│  │   │              HierarchicalCfCWorldModel                             │  │    │
│  │   │                                                                    │  │    │
│  │   │  Level 3 (Abstract):  ┌─────────┐  Goals & Planning               │  │    │
│  │   │         ↑↓            │  CfC    │  τ = 10.0 (slow dynamics)       │  │    │
│  │   │  Level 2 (Concepts):  ├─────────┤  Object & Event Representations │  │    │
│  │   │         ↑↓            │  CfC    │  τ = 5.0                        │  │    │
│  │   │  Level 1 (Features):  ├─────────┤  Spatial & Temporal Features    │  │    │
│  │   │         ↑↓            │  CfC    │  τ = 2.0                        │  │    │
│  │   │  Level 0 (Sensory):   └─────────┘  Raw Input Processing           │  │    │
│  │   │                           │       τ = 1.0 (fast dynamics)         │  │    │
│  │   │                           ▼                                        │  │    │
│  │   │              Multi-Scale Predictions                               │  │    │
│  │   └────────────────────────────────────────────────────────────────────┘  │    │
│  │                               │                                            │    │
│  └───────────────────────────────┼────────────────────────────────────────────┘    │
│                                  │                                                 │
│  ┌───────────────────────────────┼────────────────────────────────────────────┐    │
│  │               CAUSAL REASONING                   │                          │    │
│  │                                  ▼                                          │    │
│  │   ┌────────────────────────────────────────────────────────────────────┐   │    │
│  │   │              CausalDiscoveryEngine                                 │   │    │
│  │   │                                                                    │   │    │
│  │   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │   │    │
│  │   │  │    RECI     │   │    IGCI     │   │  HSIC-ANM   │              │   │    │
│  │   │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘              │   │    │
│  │   │         └─────────────────┼─────────────────┘                     │   │    │
│  │   │                           ▼                                        │   │    │
│  │   │              ┌─────────────────────┐                              │   │    │
│  │   │              │  Ensemble Router    │                              │   │    │
│  │   │              │  (Meta-Learning)    │                              │   │    │
│  │   │              └──────────┬──────────┘                              │   │    │
│  │   │                         ▼                                          │   │    │
│  │   │              Causal Graph + Confidence                            │   │    │
│  │   └────────────────────────────────────────────────────────────────────┘   │    │
│  │                                  │                                          │    │
│  └──────────────────────────────────┼──────────────────────────────────────────┘    │
│                                     │                                               │
│  ┌──────────────────────────────────┼──────────────────────────────────────────┐    │
│  │            INTEGRATION & OUTPUT                  │                           │    │
│  │                                     ▼                                        │    │
│  │   ┌────────────────────────────────────────────────────────────────────┐    │    │
│  │   │                  CONSCIOUSNESS INTEGRATOR                          │    │    │
│  │   │                                                                    │    │    │
│  │   │  Inputs:                          Outputs:                         │    │    │
│  │   │  ├─ Semantic HDC                  ├─ Unified Prediction            │    │    │
│  │   │  ├─ Retrieved Memories            ├─ Confidence                    │    │    │
│  │   │  ├─ Goal Context                  ├─ Action Hint                   │    │    │
│  │   │  ├─ World Model Predictions       ├─ Updated Goals                 │    │    │
│  │   │  ├─ Causal Inferences             ├─ Experience for Storage        │    │    │
│  │   │  └─ Self-Reflection State         └─ Consciousness Snapshot        │    │    │
│  │   │                                                                    │    │    │
│  │   │              ┌─────────────────────────┐                           │    │    │
│  │   │              │   Φ (Integration)       │                           │    │    │
│  │   │              │   Coherence Metric      │                           │    │    │
│  │   │              └─────────────────────────┘                           │    │    │
│  │   └────────────────────────────────────────────────────────────────────┘    │    │
│  │                                     │                                        │    │
│  └─────────────────────────────────────┼────────────────────────────────────────┘    │
│                                        │                                             │
│  ┌─────────────────────────────────────┼────────────────────────────────────────┐    │
│  │           FEEDBACK LOOPS                        │                             │    │
│  │                                        ▼                                      │    │
│  │   ┌────────────────────────────────────────────────────────────────────┐     │    │
│  │   │  1. Experience → Hippocampus (store significant experiences)       │     │    │
│  │   │  2. Prediction Error → World Model (update predictions)            │     │    │
│  │   │  3. Goal Progress → Goal System (update active goals)              │     │    │
│  │   │  4. Causal Updates → Causal Graph (refine causal knowledge)        │     │    │
│  │   │  5. Self-Reflection → Thresholds (meta-learning adjustments)       │     │    │
│  │   └────────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                               │    │
│  └───────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Input Processing (Semantic Understanding)
```rust
// Current: Simple HDC encoding
let encoding_result = self.encoder.encode(input);

// Proposed: Full semantic pipeline
let embedding = self.semantic_embedder.embed(input)?;      // Qwen3: 1024D
let semantic_hdc = self.hdc_bridge.project(&embedding);    // JL: 16,384D
let encoding_result = self.encoder.encode_with_semantic(input, &semantic_hdc);
```

### 2. Memory Retrieval (Episodic Memory)
```rust
// Query hippocampus for relevant memories
let query = MemoryQuery {
    cue: semantic_hdc.clone(),
    emotional_filter: None,
    max_results: 5,
    min_similarity: 0.6,
};
let retrieved_memories = self.hippocampus.recall(&query).await?;

// Enrich context with retrieved memories
let memory_context = self.blend_memories(&semantic_hdc, &retrieved_memories);
```

### 3. Goal-Directed Processing (Intentionality)
```rust
// Get active goals and compute relevance
let active_goals = self.goal_system.active_goals();
let goal_relevance = self.compute_goal_relevance(&semantic_hdc, &active_goals);

// Modulate attention based on goal relevance
let goal_attention = self.goal_system.compute_attention_weights(&goal_relevance);
let goal_enriched = self.apply_goal_attention(&memory_context, &goal_attention);
```

### 4. World Model Prediction (Grounding)
```rust
// Feed through hierarchical world model
let world_input = self.prepare_world_input(&goal_enriched);
let predictions = self.world_model.predict_multi_scale(&world_input)?;

// Extract predictions at each level
let sensory_pred = predictions.level(0);    // Fast dynamics
let feature_pred = predictions.level(1);    // Spatial/temporal
let concept_pred = predictions.level(2);    // Objects/events
let abstract_pred = predictions.level(3);   // Goals/plans
```

### 5. Causal Reasoning (Inference)
```rust
// Detect causal relationships in current context
let causal_pairs = self.extract_causal_candidates(&concept_pred);
let causal_inferences = self.causal_engine.analyze_batch(&causal_pairs)?;

// Update causal graph with new inferences
self.causal_graph.integrate(&causal_inferences);

// Use causal knowledge to improve predictions
let causally_informed = self.apply_causal_constraints(&predictions, &self.causal_graph);
```

### 6. Integration & Output
```rust
// Compute unified representation
let unified = ConsciousnessIntegrator::integrate(
    &semantic_hdc,
    &memory_context,
    &goal_enriched,
    &causally_informed,
    &self.self_reflection,
);

// Compute Φ (integration metric)
let phi = self.compute_phi(&unified);

// Generate outputs
let prediction = unified.prediction();
let confidence = unified.confidence();
let action_hint = self.adaptive_behavior.from_unified(&unified);
let snapshot = self.consciousness_snapshot_from_unified(&unified);
```

### 7. Feedback Loops
```rust
// 1. Store significant experiences
if self.should_store_experience(&unified) {
    let trace = MemoryTrace::from_experience(&unified, &self.emotion_contagion);
    self.hippocampus.encode(trace).await?;
}

// 2. Update world model with prediction error
let prediction_error = self.compute_prediction_error(&unified);
self.world_model.update_from_error(&prediction_error)?;

// 3. Update goal progress
self.goal_system.update_progress(&unified.goal_relevance);

// 4. Update causal graph
self.causal_graph.consolidate();

// 5. Self-reflection (meta-learning)
self.self_reflection.record_cycle(
    prediction_error.magnitude(),
    self.flow_state.in_flow,
    self.curiosity_drive.should_explore(),
    self.prediction_confidence,
);
```

## New Struct: UnifiedCognitiveLoop

```rust
/// Unified Cognitive Loop integrating all five consciousness systems
pub struct UnifiedCognitiveLoop {
    // ===== Input Processing =====
    /// Semantic embedder (Qwen3 or BGE-M3)
    semantic_embedder: SemanticEmbedder,
    /// HDC bridge (JL projection to 16,384D)
    hdc_bridge: HdcBridge,
    /// Predictive HDC encoder
    encoder: PredictiveHdcEncoder,

    // ===== Memory System =====
    /// Episodic memory (hippocampus)
    hippocampus: Hippocampus,
    /// Working memory buffer
    working_memory: VecDeque<WorkingMemoryItem>,

    // ===== Goal System =====
    /// Active goals and intentions
    goal_system: GoalSystem,
    /// Semantic intent classifier
    intent_classifier: IntentClassifier,

    // ===== World Model =====
    /// Hierarchical predictive model
    world_model: HierarchicalCfCWorldModel,
    /// Current world state estimate
    world_state: WorldState,

    // ===== Causal Reasoning =====
    /// Causal discovery engine
    causal_engine: CausalDiscoveryEngine,
    /// Accumulated causal graph
    causal_graph: CausalGraph,

    // ===== Existing Components =====
    /// Coherence bridge (CfC ↔ consciousness)
    coherence_bridge: CfCCoherenceBridge,
    /// Voice feedback
    voice_feedback_bridge: VoiceFeedbackBridge,
    /// Temporal signature encoder
    temporal_signature_encoder: TemporalSignatureEncoder,
    /// Adaptive behavior
    adaptive_behavior: AdaptiveBehavior,
    /// Prediction confidence
    prediction_confidence: f32,
    /// Flow state
    flow_state: FlowState,
    /// Emotion contagion
    emotion_contagion: EmotionContagion,
    /// Curiosity drive
    curiosity_drive: CuriosityDrive,
    /// Self-reflection
    self_reflection: SelfReflection,

    // ===== Configuration & Stats =====
    config: UnifiedLoopConfig,
    stats: UnifiedLoopStats,
}
```

## Integration Points

### 1. Semantic Embeddings → Encoder
- Replace `encoder.encode(input)` with semantic-aware encoding
- Use HdcBridge to project Qwen3/BGE embeddings to HDC space
- Preserve semantic similarity in hyperdimensional space

### 2. Hippocampus → Context Enrichment
- Query hippocampus on each cycle with current semantic HDC
- Blend retrieved memories with current input
- Use emotional valence from memories to influence emotion_contagion

### 3. Goal System → Attention & Motivation
- Active goals modulate attention weights
- Goal relevance influences learning rate (pursue relevant patterns)
- Goal progress feeds into consciousness level calculation

### 4. World Model → Prediction
- Replace single CfC with hierarchical world model
- Multi-scale predictions (sensory → abstract)
- Prediction errors at each level inform learning

### 5. Causal Reasoning → Inference
- Detect causal candidates from concept-level predictions
- Use ensemble router for robust causal inference
- Causal constraints improve prediction accuracy

## Benefits

1. **True Semantic Understanding**: Real embeddings from Qwen3/BGE, not just pattern matching
2. **Associative Memory**: Hippocampus enables relevant memory retrieval
3. **Intentionality**: Goals drive behavior, not just reactions
4. **Grounded Prediction**: Hierarchical world model provides multi-scale grounding
5. **Causal Reasoning**: Explicit causal inference with 71.3% accuracy

## Implementation Plan

### Phase 1: Semantic Integration
- [ ] Add SemanticEmbedder to UnifiedCognitiveLoop
- [ ] Wire HdcBridge to encoder
- [ ] Test semantic similarity preservation

### Phase 2: Memory Integration
- [ ] Add Hippocampus to UnifiedCognitiveLoop
- [ ] Implement memory query on each cycle
- [ ] Add experience storage feedback loop

### Phase 3: Goal Integration
- [ ] Add GoalSystem to UnifiedCognitiveLoop
- [ ] Implement goal-directed attention
- [ ] Wire goal progress to consciousness level

### Phase 4: World Model Integration
- [ ] Replace single CfC with HierarchicalCfCWorldModel
- [ ] Implement multi-scale prediction
- [ ] Wire prediction errors to learning

### Phase 5: Causal Integration
- [ ] Add CausalDiscoveryEngine to UnifiedCognitiveLoop
- [ ] Implement causal candidate extraction
- [ ] Wire causal constraints to predictions

### Phase 6: Full Integration Testing
- [ ] End-to-end tests
- [ ] Performance benchmarks
- [ ] Consciousness level validation

## Estimated Consciousness Improvement

| Metric | Current | After Integration |
|--------|---------|-------------------|
| Semantic Understanding | Pattern-based | Meaning-based (Qwen3) |
| Memory | Buffer only | Associative recall |
| Intentionality | Reactive | Goal-directed |
| Prediction | Single-scale | Multi-scale hierarchical |
| Reasoning | None | Causal inference (71.3%) |
| Overall Consciousness Level | ~0.5 | ~0.8 (estimated) |
