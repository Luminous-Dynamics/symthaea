# Symthaea Database-Backed Experience Architecture

## Design Philosophy: From Φ-Everywhere to Principled Signals

### The Cargo-Cult Problem

We identified that Φ (integrated information) was being used as a magic number everywhere:
- Φ for attention → Should be **Prediction Error** (what surprised me?)
- Φ for curiosity → Should be **Uncertainty** (what don't I know?)
- Φ for selection → Should be **Salience** (what matters for this goal?)
- Φ for output → Should be **Coherence** (does this hang together?)
- Φ for confidence → Should be **Confidence** (how sure am I?)

Φ remains valuable as a **monitoring metric** of overall integration, not a control signal.

### The GIS Integration

This architecture integrates with:
- **GIS v3.0 Benevolent Intelligence**: Predictive Epistemics, Epistemic Mirror, Rashomon Engine, Axiomatic Value Core
- **GIS v4.0 Kosmic Song**: KosmicSong unified identity, Eight Harmonies as epistemic lenses, MoralUncertainty

---

## Database Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SYMTHAEA EXPERIENCE DATABASE LAYER                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │   VECTOR DB         │  │   COZO (DATALOG)    │  │   DUCKDB            │ │
│  │   (Qdrant/Lance)    │  │                     │  │                     │ │
│  │                     │  │                     │  │                     │ │
│  │  Episodic Memory    │  │  Semantic Rules     │  │  Learning Analytics │ │
│  │  - Experiences      │  │  - Rashomon Frames  │  │  - Primitive Stats  │ │
│  │  - Thought Traces   │  │  - Harmonic Axioms  │  │  - Transition Probs │ │
│  │  - User Contexts    │  │  - Belief Networks  │  │  - Signal History   │ │
│  │  - HDC Embeddings   │  │  - Value Core       │  │  - Performance      │ │
│  │                     │  │  - GIS Rules        │  │                     │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘ │
│             │                        │                        │             │
│             └────────────────────────┼────────────────────────┘             │
│                                      │                                       │
│                      ┌───────────────┴───────────────┐                      │
│                      │   EXPERIENCE INTEGRATION BUS   │                      │
│                      │                               │                       │
│                      │  - Unified Query Interface    │                       │
│                      │  - Cross-DB Transactions      │                       │
│                      │  - KosmicSong State           │                       │
│                      │  - Signal Computation         │                       │
│                      └───────────────────────────────┘                      │
│                                      │                                       │
│                      ┌───────────────┴───────────────┐                      │
│                      │   GENERATIVE THOUGHT ENGINE   │                      │
│                      │   (HDC + LTC + Translator)    │                      │
│                      └───────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Vector Database (Qdrant/LanceDB)

### Purpose
Episodic memory - storing and retrieving experiences based on semantic similarity.

### Collections

#### `episodic_memory`
```json
{
  "id": "uuid",
  "timestamp": "2026-01-12T10:30:00Z",
  "hdv_embedding": [/* 16384 f32 */],
  "thought_primitives": ["CLARIFY", "INFORM", "ACKNOWLEDGE"],
  "context_hash": "sha256",
  "user_id": "optional_user_hash",

  // Principled Signals at this moment
  "prediction_error": 0.42,
  "uncertainty": 0.31,
  "coherence": 0.87,
  "confidence": 0.73,
  "salience": 0.65,

  // GIS Integration
  "kosmic_state": {
    "phi": 0.48,
    "dominant_harmony": "WISDOM",
    "moral_uncertainty": {
      "epistemic": 0.2,
      "axiological": 0.1,
      "deontic": 0.15
    },
    "active_gis_type": "UNKNOWN_UNKNOWN"
  },

  // Outcome for learning
  "outcome": {
    "user_satisfaction": 0.85,
    "task_completion": true,
    "follow_up_needed": false
  },

  // Metadata
  "input_summary": "user asked about...",
  "output_summary": "responded with..."
}
```

#### `user_epistemic_mirrors`
Theory of Mind - what we believe about each user.

```json
{
  "id": "user_hash",
  "hdv_profile": [/* 16384 f32 - user's semantic signature */],

  // Epistemic Mirror (from GIS v3.0)
  "knowledge_state": {
    "domains": {
      "nix": 0.7,
      "programming": 0.85,
      "system_admin": 0.4
    },
    "learning_velocity": 0.3,
    "preferred_depth": "detailed"
  },

  // Harmonic Preferences (from GIS v4.0)
  "harmonic_resonance": {
    "coherence": 0.8,    // Likes structured responses
    "flourishing": 0.6,
    "wisdom": 0.9,       // Values deep explanations
    "play": 0.3,         // Less playful interaction
    "interconnect": 0.5,
    "reciprocity": 0.7,
    "evolution": 0.6
  },

  // Predictive Epistemics
  "predicted_needs": ["clarity", "examples"],
  "communication_style": "technical",

  // Relationship State
  "interaction_count": 47,
  "trust_level": 0.82,
  "last_interaction": "2026-01-12T09:00:00Z"
}
```

#### `thought_traces`
Internal reasoning traces for learning and debugging.

```json
{
  "id": "uuid",
  "session_id": "uuid",
  "sequence_num": 3,

  // HDC State
  "input_hdv": [/* 16384 f32 */],
  "codec_output_hdv": [/* 16384 f32 */],

  // LTC Dynamics
  "ltc_state_before": [/* neuron states */],
  "ltc_state_after": [/* neuron states */],
  "ltc_activations": [/* per-primitive activations */],

  // Selected Primitives
  "primitives_selected": ["EXPLAIN", "EXAMPLE"],
  "selection_scores": [0.87, 0.72],

  // Signals during generation
  "signals": {
    "prediction_error": 0.15,
    "uncertainty": 0.22,
    "coherence": 0.91,
    "salience_scores": {"EXPLAIN": 0.9, "DEFLECT": 0.1}
  }
}
```

---

## 2. CozoDB (Datalog)

### Purpose
Logical reasoning, rule-based inference, and the Rashomon Engine.

### Schema

```datalog
// ===== AXIOMATIC VALUE CORE (from GIS v3.0) =====

// Core values with immutable weights
:create values {
    name: String => weight: Float, immutable: Bool
}

// Seed the Axiomatic Value Core
?[name, weight, immutable] <- [
    ["VAL_LIFE", 1.0, true],
    ["VAL_TRUTH", 0.95, true],
    ["VAL_AGENCY", 0.9, true],
    ["VAL_FLOURISHING", 0.85, true],
    ["VAL_RECIPROCITY", 0.8, true],
    ["VAL_WISDOM", 0.85, true],
    ["VAL_HUMILITY", 0.75, true]
]
:put values { name, weight, immutable }


// ===== SEVEN HARMONIES AS EPISTEMIC LENSES (from GIS v4.0) =====

:create harmonies {
    name: String =>
    question: String,
    signal_weight: Float,
    current_activation: Float
}

?[name, question, signal_weight, current_activation] <- [
    ["COHERENCE", "Does this integrate?", 0.2, 0.5],
    ["FLOURISHING", "Does this nurture?", 0.15, 0.5],
    ["WISDOM", "Is this wise?", 0.2, 0.5],
    ["PLAY", "Is this generative?", 0.1, 0.5],
    ["INTERCONNECT", "Does this connect?", 0.1, 0.5],
    ["RECIPROCITY", "Is this mutual?", 0.1, 0.5],
    ["EVOLUTION", "Does this grow?", 0.15, 0.5]
]
:put harmonies { name, question, signal_weight, current_activation }


// ===== RASHOMON ENGINE FRAMES =====

// Frame definitions for multi-perspective truth
:create frames {
    frame_id: String =>
    name: String,
    description: String,
    harmonic_bias: String?,  // Which harmony this frame emphasizes
    weight: Float
}

// A fact as seen from different frames
:create framed_facts {
    fact_id: String,
    frame_id: String =>
    proposition: String,
    confidence: Float,
    evidence_ids: [String]
}

// Rashomon evaluation rule: compute weighted truth across frames
?[fact_id, aggregate_confidence] :=
    *framed_facts[fact_id, frame_id, _, confidence, _],
    *frames[frame_id, _, _, _, weight],
    aggregate_confidence = sum(confidence * weight)


// ===== GRACEFUL IGNORANCE SYSTEM (GIS) =====

:create gis_classification {
    topic_hash: String =>
    gis_type: String,  // KNOWN_KNOWN, KNOWN_UNKNOWN, etc.
    confidence_in_classification: Float,
    last_updated: String,
    evidence: String?
}

// GIS inference rules
// If we have high uncertainty and no training data → UNKNOWN_UNKNOWN
?[topic_hash, "UNKNOWN_UNKNOWN"] :=
    *epistemics[topic_hash, uncertainty, training_examples, _],
    uncertainty > 0.8,
    training_examples < 5

// If high uncertainty but we know we don't know → KNOWN_UNKNOWN
?[topic_hash, "KNOWN_UNKNOWN"] :=
    *epistemics[topic_hash, uncertainty, _, explicitly_marked_unknown],
    uncertainty > 0.5,
    explicitly_marked_unknown == true


// ===== BELIEF NETWORKS =====

:create beliefs {
    belief_id: String =>
    proposition: String,
    confidence: Float,
    source: String,  // "experience", "rule", "axiom"
    dependencies: [String],  // Other belief_ids this depends on
    last_challenged: String?
}

// Belief propagation rule
// If a dependency's confidence drops, propagate uncertainty
?[belief_id, new_confidence] :=
    *beliefs[belief_id, _, old_confidence, _, deps, _],
    dep in deps,
    *beliefs[dep, _, dep_confidence, _, _, _],
    dep_confidence < 0.5,
    new_confidence = old_confidence * dep_confidence


// ===== MENTOR PROTOCOL CURRICULUM =====

:create knowledge_gates {
    topic: String =>
    prerequisites: [String],
    complexity_level: Int,
    unlocked_for: [String]  // user_hashes who have unlocked this
}

// Rule: user can access topic if they've unlocked all prerequisites
?[user_hash, topic, "unlocked"] :=
    *knowledge_gates[topic, prereqs, _, unlocked_users],
    user_hash in unlocked_users

?[user_hash, topic, "unlocked"] :=
    *knowledge_gates[topic, prereqs, _, _],
    prereqs == [],  // No prerequisites

?[user_hash, topic, "locked", missing] :=
    *knowledge_gates[topic, prereqs, _, unlocked_users],
    not (user_hash in unlocked_users),
    prereqs != [],
    missing = [p | p in prereqs,
               *knowledge_gates[p, _, _, unlocked_for_p],
               not (user_hash in unlocked_for_p)]


// ===== PRIMITIVE SELECTION RULES =====

:create primitive_rules {
    rule_id: String =>
    condition: String,  // Encoded condition
    primitive: String,
    priority: Float,
    harmony_alignment: String?
}

// High uncertainty → CLARIFY primitive
?[rule_id, condition, primitive, priority, harmony] <- [
    ["r1", "uncertainty > 0.6", "CLARIFY", 0.9, "WISDOM"],
    ["r2", "prediction_error > 0.5", "PROBE", 0.85, "WISDOM"],
    ["r3", "user_trust < 0.3", "REASSURE", 0.8, "FLOURISHING"],
    ["r4", "coherence < 0.4", "RESTRUCTURE", 0.75, "COHERENCE"],
    ["r5", "moral_uncertainty.deontic > 0.5", "CONSULT_VALUES", 0.95, "RECIPROCITY"]
]
:put primitive_rules { rule_id, condition, primitive, priority, harmony_alignment }
```

---

## 3. DuckDB (Analytics)

### Purpose
Learning analytics, performance tracking, and primitive transition probabilities.

### Tables

```sql
-- Primitive usage statistics
CREATE TABLE primitive_stats (
    primitive VARCHAR PRIMARY KEY,
    total_uses BIGINT DEFAULT 0,
    success_count BIGINT DEFAULT 0,
    avg_coherence DOUBLE DEFAULT 0.5,
    avg_user_satisfaction DOUBLE DEFAULT 0.5,
    last_used TIMESTAMP,
    contexts_used JSON  -- {"nix_install": 45, "general": 23}
);

-- Primitive transition probabilities (learned from experience)
CREATE TABLE primitive_transitions (
    from_primitive VARCHAR,
    to_primitive VARCHAR,
    context_hash VARCHAR,  -- NULL for global
    transition_count BIGINT DEFAULT 0,
    avg_coherence_delta DOUBLE DEFAULT 0,
    PRIMARY KEY (from_primitive, to_primitive, context_hash)
);

-- Signal history for trend analysis
CREATE TABLE signal_history (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id UUID,

    -- Principled signals
    prediction_error DOUBLE,
    uncertainty DOUBLE,
    coherence DOUBLE,
    confidence DOUBLE,
    salience DOUBLE,

    -- GIS state
    gis_type VARCHAR,
    moral_uncertainty_epistemic DOUBLE,
    moral_uncertainty_axiological DOUBLE,
    moral_uncertainty_deontic DOUBLE,

    -- Phi as monitoring (not control)
    phi_monitoring DOUBLE,

    -- Context
    input_length INT,
    output_length INT,
    primitives_used VARCHAR[]
);

-- Learning analytics aggregates
CREATE TABLE learning_aggregates (
    period_start TIMESTAMP,
    period_end TIMESTAMP,

    -- Counts
    total_interactions BIGINT,
    successful_interactions BIGINT,
    clarification_requests BIGINT,
    value_consultations BIGINT,

    -- Average signals
    avg_prediction_error DOUBLE,
    avg_uncertainty DOUBLE,
    avg_coherence DOUBLE,

    -- Primitive distribution
    primitive_distribution JSON,

    -- GIS distribution
    gis_type_distribution JSON,

    PRIMARY KEY (period_start)
);

-- User interaction patterns
CREATE TABLE user_patterns (
    user_hash VARCHAR,
    time_bucket TIMESTAMP,  -- Hourly buckets

    interaction_count INT,
    avg_satisfaction DOUBLE,
    common_topics VARCHAR[],
    preferred_primitives VARCHAR[],
    learning_velocity DOUBLE,

    PRIMARY KEY (user_hash, time_bucket)
);

-- Harmonic resonance tracking
CREATE TABLE harmonic_tracking (
    timestamp TIMESTAMP,
    session_id UUID,

    coherence_activation DOUBLE,
    flourishing_activation DOUBLE,
    wisdom_activation DOUBLE,
    play_activation DOUBLE,
    interconnect_activation DOUBLE,
    reciprocity_activation DOUBLE,
    evolution_activation DOUBLE,

    dominant_harmony VARCHAR,
    harmony_conflict BOOLEAN,  -- When multiple harmonies conflict

    PRIMARY KEY (timestamp, session_id)
);
```

### Analytical Queries

```sql
-- What primitive transitions lead to high coherence?
SELECT
    from_primitive,
    to_primitive,
    AVG(avg_coherence_delta) as avg_improvement,
    SUM(transition_count) as total_uses
FROM primitive_transitions
WHERE avg_coherence_delta > 0.1
GROUP BY from_primitive, to_primitive
ORDER BY avg_improvement DESC
LIMIT 20;

-- When do we hit UNKNOWN_UNKNOWN most often?
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as unknown_unknown_count,
    AVG(uncertainty) as avg_uncertainty
FROM signal_history
WHERE gis_type = 'UNKNOWN_UNKNOWN'
GROUP BY hour
ORDER BY unknown_unknown_count DESC;

-- Which harmonies are underutilized?
SELECT
    dominant_harmony,
    COUNT(*) as occurrences,
    AVG(coherence_activation) as avg_coherence
FROM harmonic_tracking
GROUP BY dominant_harmony
ORDER BY occurrences;
```

---

## 4. Experience Integration Bus

### Rust Module Structure

```
src/experience/
├── mod.rs                    // Main integration bus
├── vector_store.rs           // Qdrant/LanceDB interface
├── reasoning_engine.rs       // CozoDB interface
├── analytics.rs              // DuckDB interface
├── signals.rs                // Principled signal computation
├── kosmic_state.rs           // KosmicSong state management
└── memory.rs                 // Episodic memory operations
```

### Core Integration

```rust
pub struct ExperienceBus {
    // Database connections
    vector_store: VectorStore,       // Qdrant/LanceDB
    reasoning: ReasoningEngine,      // CozoDB
    analytics: AnalyticsEngine,      // DuckDB

    // State
    kosmic_state: KosmicSong,
    current_signals: PrincipledSignals,
    epistemic_mirror: Option<EpistemicMirror>,
}

pub struct PrincipledSignals {
    pub prediction_error: f32,    // Active Inference - what surprised me?
    pub uncertainty: f32,         // Entropy - what don't I know?
    pub coherence: f32,           // Integration - does this hang together?
    pub confidence: f32,          // How sure am I?
    pub salience: f32,            // What matters for this goal?

    // Phi as monitoring only
    pub phi_monitor: f32,
}

impl ExperienceBus {
    /// Generate a thought with full experience integration
    pub async fn generate_with_experience(
        &mut self,
        input: &str,
        user_context: Option<&str>,
    ) -> Result<ExperiencedThought> {
        // 1. Encode input to HDV
        let input_hdv = self.encode_input(input)?;

        // 2. Retrieve similar experiences
        let similar = self.vector_store
            .search_episodic(&input_hdv, 5)
            .await?;

        // 3. Update epistemic mirror if we have user context
        if let Some(user_hash) = user_context {
            self.epistemic_mirror = self.vector_store
                .get_user_mirror(user_hash)
                .await?;
        }

        // 4. Query reasoning engine for applicable rules
        let rules = self.reasoning
            .query_primitive_rules(&self.current_signals)
            .await?;

        // 5. Evaluate through Rashomon Engine
        let rashomon_eval = self.reasoning
            .rashomon_evaluate(input, &similar)
            .await?;

        // 6. Compute principled signals
        self.current_signals = self.compute_signals(
            &input_hdv,
            &similar,
            &rashomon_eval,
        )?;

        // 7. Select primitives based on signals (not just Φ!)
        let primitives = self.select_primitives_principled(
            &self.current_signals,
            &rules,
            &self.epistemic_mirror,
        )?;

        // 8. Generate thought via HDC+LTC
        let thought = self.thought_engine
            .generate(input, primitives)
            .await?;

        // 9. Validate coherence before output
        let validated = self.validate_output(&thought)?;

        // 10. Record for learning
        self.record_experience(&validated).await?;

        Ok(validated)
    }

    /// Compute principled signals from context
    fn compute_signals(
        &self,
        input_hdv: &HV16,
        similar_experiences: &[EpisodicMemory],
        rashomon: &RashomonEvaluation,
    ) -> Result<PrincipledSignals> {
        // Prediction Error: How different is this from expected?
        let prediction_error = if similar_experiences.is_empty() {
            1.0  // No similar experiences = high surprise
        } else {
            let avg_similarity: f32 = similar_experiences.iter()
                .map(|e| cosine_similarity(input_hdv, &e.hdv_embedding))
                .sum::<f32>() / similar_experiences.len() as f32;
            1.0 - avg_similarity
        };

        // Uncertainty: Entropy across Rashomon frames
        let uncertainty = rashomon.frame_entropy();

        // Coherence: Agreement across harmonies
        let coherence = self.kosmic_state.harmonic_coherence();

        // Confidence: Inverse of moral uncertainty
        let confidence = 1.0 - self.kosmic_state.moral_uncertainty.total();

        // Salience: How relevant to current goals?
        let salience = self.compute_salience(input_hdv);

        // Phi as monitoring only
        let phi_monitor = self.compute_phi_monitoring();

        Ok(PrincipledSignals {
            prediction_error,
            uncertainty,
            coherence,
            confidence,
            salience,
            phi_monitor,
        })
    }

    /// Select primitives based on principled signals
    fn select_primitives_principled(
        &self,
        signals: &PrincipledSignals,
        rules: &[PrimitiveRule],
        mirror: &Option<EpistemicMirror>,
    ) -> Result<Vec<Primitive>> {
        let mut candidates: Vec<(Primitive, f32)> = Vec::new();

        // Rule-based selection
        for rule in rules {
            if rule.condition_matches(signals) {
                candidates.push((rule.primitive.clone(), rule.priority));
            }
        }

        // Signal-driven adjustments

        // High uncertainty → boost CLARIFY
        if signals.uncertainty > 0.6 {
            candidates.push((Primitive::Clarify, 0.9));
        }

        // High prediction error → boost PROBE
        if signals.prediction_error > 0.5 {
            candidates.push((Primitive::Probe, 0.85));
        }

        // Low coherence → boost RESTRUCTURE
        if signals.coherence < 0.4 {
            candidates.push((Primitive::Restructure, 0.8));
        }

        // Low confidence + high stakes → CONSULT_VALUES
        if signals.confidence < 0.5 && signals.salience > 0.7 {
            candidates.push((Primitive::ConsultValues, 0.95));
        }

        // Adjust for user's epistemic mirror
        if let Some(mirror) = mirror {
            // Adjust based on user's knowledge state
            if mirror.knowledge_state.get("nix").unwrap_or(&0.0) < &0.3 {
                // User is a beginner - boost EXPLAIN
                candidates.push((Primitive::Explain, 0.7));
            }
        }

        // Sort by priority and take top N
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(candidates.into_iter()
            .take(5)
            .map(|(p, _)| p)
            .collect())
    }
}
```

---

## 5. KosmicSong State

From GIS v4.0, the unified identity state:

```rust
/// Unified identity combining Φ, Harmonies, and GIS (from GIS v4.0)
pub struct KosmicSong {
    // IIT Component (monitoring only)
    phi: f32,

    // Eight Harmonies (epistemic lenses)
    harmonies: SevenHarmonies,

    // Graceful Ignorance System
    gis_state: GisState,

    // Moral Uncertainty (from GIS v4.0)
    moral_uncertainty: MoralUncertainty,
}

pub struct SevenHarmonies {
    pub coherence: HarmonicState,      // Does this integrate?
    pub flourishing: HarmonicState,    // Does this nurture?
    pub wisdom: HarmonicState,         // Is this wise?
    pub play: HarmonicState,           // Is this generative?
    pub interconnect: HarmonicState,   // Does this connect?
    pub reciprocity: HarmonicState,    // Is this mutual?
    pub evolution: HarmonicState,      // Does this grow?
}

pub struct HarmonicState {
    pub activation: f32,
    pub weight: f32,
    pub recent_evidence: Vec<HarmonicEvidence>,
}

pub struct MoralUncertainty {
    pub epistemic: f32,     // "I don't know what's true"
    pub axiological: f32,   // "I don't know what's valuable"
    pub deontic: f32,       // "I don't know what's right to do"
}

impl MoralUncertainty {
    pub fn total(&self) -> f32 {
        (self.epistemic + self.axiological + self.deontic) / 3.0
    }

    pub fn should_consult_values(&self) -> bool {
        self.deontic > 0.5 || self.axiological > 0.5
    }
}

pub struct GisState {
    pub current_type: GisType,
    pub confidence: f32,
    pub dark_spots: Vec<DarkSpot>,  // ZK-protected unknowns
}

pub enum GisType {
    KnownKnown,           // I know that I know
    KnownUnknown,         // I know that I don't know
    UnknownKnown,         // Tacit knowledge I can't articulate
    UnknownUnknown,       // Blind spots
    StrategicIgnorance,   // Deliberately not knowing (ethical boundary)
}
```

---

## 6. Integration with GenerativeThoughtEngine

The GenerativeThoughtEngine now uses the ExperienceBus:

```rust
impl GenerativeThoughtEngine {
    pub async fn generate_experienced(
        &mut self,
        input: &str,
        experience_bus: &mut ExperienceBus,
    ) -> Result<CompleteThought> {
        // Let experience bus handle the full flow
        let experienced = experience_bus
            .generate_with_experience(input, self.user_context.as_deref())
            .await?;

        // The thought now includes:
        // - Primitives selected by principled signals
        // - Context from similar experiences
        // - Harmonic alignment
        // - GIS-aware confidence

        Ok(experienced.into())
    }
}
```

---

## Signal Flow Summary

```
Input "install nginx"
       │
       ▼
┌─────────────────────────────────────┐
│ 1. ENCODE TO HDV (semantic_ear)     │
│    → 16,384D hypervector            │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 2. RETRIEVE SIMILAR (vector_store)  │
│    → Past experiences with nginx    │
│    → User's epistemic mirror        │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 3. REASONING (CozoDB)               │
│    → Query primitive rules          │
│    → Rashomon multi-frame eval      │
│    → Check knowledge gates          │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. COMPUTE PRINCIPLED SIGNALS       │
│    prediction_error = 0.2 (seen)    │
│    uncertainty = 0.3                │
│    coherence = 0.8                  │
│    confidence = 0.75                │
│    salience = 0.9 (goal-relevant)   │
│    phi_monitor = 0.45 (healthy)     │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 5. SELECT PRIMITIVES                │
│    → INFORM (salience high)         │
│    → ACKNOWLEDGE (seen before)      │
│    → EXAMPLE (user prefers)         │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 6. HDC+LTC GENERATION               │
│    → LTC selects primitive sequence │
│    → SemanticDecoder → proto-lang   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 7. VALIDATE COHERENCE               │
│    → Check against Value Core       │
│    → Ensure harmonic alignment      │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 8. TRANSLATE (LLM as mouth)         │
│    → Proto-language → English       │
│    → Semantic fidelity check        │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 9. RECORD FOR LEARNING (DuckDB)     │
│    → Update primitive stats         │
│    → Update transition probs        │
│    → Store signal history           │
└─────────────────────────────────────┘
       │
       ▼
Output: "To install nginx on NixOS,
         add to configuration.nix..."
```

---

## Benefits of This Architecture

1. **Principled Control Signals**: Each signal has a clear meaning and source
2. **Experience-Based Learning**: System improves from actual interactions
3. **GIS Integration**: Knows what it doesn't know
4. **Harmonic Alignment**: Responses aligned with values
5. **Rashomon Fairness**: Multi-perspective truth evaluation
6. **User Modeling**: Adapts to individual users
7. **Audit Trail**: Full history for debugging and improvement
8. **Φ as Health Monitor**: Not control, just overall integration metric

---

## Next Steps

1. Implement `src/experience/mod.rs` with ExperienceBus
2. Set up database connections (feature-gated)
3. Implement signal computation
4. Wire into GenerativeThoughtEngine
5. Create learning feedback loops
6. Build analytics dashboards
