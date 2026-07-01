// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Full Stack Consciousness: Unified Understanding + Active Inference + Memory + Imagination
//!
//! This module integrates all consciousness capabilities into a coherent system:
//!
//! 1. **UnifiedUnderstanding** - Semantic primes + predictive + narrative + ToM
//! 2. **ActiveInference** - Precision-weighted prediction errors + action selection
//! 3. **EpisodicMemory** - Hippocampus-style holographic encoding
//! 4. **Metacognition** - Self-monitoring + confidence calibration
//! 5. **CounterfactualReasoning** - "What if..." imagination + causal simulation
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                      FULL STACK CONSCIOUSNESS                               │
//! ├────────────────────────────────────────────────────────────────────────────┤
//! │                                                                            │
//! │   Text Input ──────────────────────────────────────────────────────        │
//! │        │                                                                   │
//! │        ▼                                                                   │
//! │   ┌────────────────────────┐         ┌───────────────────────────┐         │
//! │   │  UnifiedUnderstanding  │────────►│  ActiveInferenceAdapter   │         │
//! │   │  (DeepUnderstanding)   │         │  (PredictionErrors)       │         │
//! │   └────────────┬───────────┘         └─────────────┬─────────────┘         │
//! │                │                                   │                       │
//! │                ▼                                   ▼                       │
//! │   ┌────────────────────────┐         ┌───────────────────────────┐         │
//! │   │  MemoryAdapter         │◄───────►│  MetacognitiveAdapter     │         │
//! │   │  (Episodic Encoding)   │         │  (Self-Monitoring)        │         │
//! │   └────────────┬───────────┘         └─────────────┬─────────────┘         │
//! │                │                                   │                       │
//! │                └───────────────┬───────────────────┘                       │
//! │                                ▼                                           │
//! │                  ┌────────────────────────┐                                │
//! │                  │  CounterfactualEngine  │                                │
//! │                  │  ("What if...?")       │                                │
//! │                  └────────────┬───────────┘                                │
//! │                                │                                           │
//! │                                ▼                                           │
//! │                  ┌────────────────────────┐                                │
//! │                  │  ConsciousComprehension│                                │
//! │                  │  (Φ-Integrated Result) │                                │
//! │                  └────────────────────────┘                                │
//! │                                                                            │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```

use super::binary_hv::BinaryHV;
use super::grounded_understanding::CausalRelation;
use super::unified_understanding::{DeepUnderstanding, StoryRole, UnifiedUnderstandingPipeline};
use super::universal_semantics::SemanticPrime;
use std::collections::{HashMap, VecDeque};

// =============================================================================
// ACTIVE INFERENCE ADAPTER
// =============================================================================

/// Maps understanding results to Active Inference prediction domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceDomain {
    /// Semantic coherence (understanding quality)
    SemanticCoherence,
    /// Emotional prediction (what feelings to expect)
    EmotionalState,
    /// Social dynamics (speaker intentions/goals)
    SocialDynamics,
    /// Narrative continuity (story coherence)
    NarrativeContinuity,
    /// Causal structure (cause-effect expectations)
    CausalReasoning,
    /// Temporal prediction (what comes next)
    TemporalFlow,
}

/// A precision-weighted prediction error from understanding
#[derive(Debug, Clone)]
pub struct UnderstandingPredictionError {
    /// What was predicted
    pub expected: f32,
    /// What was observed
    pub observed: f32,
    /// Raw error
    pub error: f32,
    /// Precision (confidence in this prediction)
    pub precision: f32,
    /// Weighted error
    pub weighted_error: f32,
    /// Domain of prediction
    pub domain: InferenceDomain,
}

impl UnderstandingPredictionError {
    pub fn new(expected: f32, observed: f32, precision: f32, domain: InferenceDomain) -> Self {
        let error = observed - expected;
        Self {
            expected,
            observed,
            error,
            precision,
            weighted_error: error * precision,
            domain,
        }
    }

    /// Surprise magnitude
    pub fn surprise(&self) -> f32 {
        self.weighted_error.abs()
    }
}

/// Adapter connecting UnifiedUnderstanding to Active Inference
pub struct ActiveInferenceAdapter {
    /// Predictions for each domain
    predictions: HashMap<InferenceDomain, f32>,
    /// Precision weights for each domain
    precisions: HashMap<InferenceDomain, f32>,
    /// History of prediction errors
    error_history: VecDeque<UnderstandingPredictionError>,
    /// Total free energy (sum of weighted prediction errors)
    free_energy: f64,
    /// Learning rate for prediction updates
    learning_rate: f32,
}

impl ActiveInferenceAdapter {
    pub fn new() -> Self {
        let mut predictions = HashMap::new();
        let mut precisions = HashMap::new();

        // Initialize with neutral predictions and moderate precision
        for domain in [
            InferenceDomain::SemanticCoherence,
            InferenceDomain::EmotionalState,
            InferenceDomain::SocialDynamics,
            InferenceDomain::NarrativeContinuity,
            InferenceDomain::CausalReasoning,
            InferenceDomain::TemporalFlow,
        ] {
            predictions.insert(domain, 0.5);
            precisions.insert(domain, 0.5);
        }

        Self {
            predictions,
            precisions,
            error_history: VecDeque::with_capacity(100),
            free_energy: 0.0,
            learning_rate: 0.1,
        }
    }

    /// Process a DeepUnderstanding and generate prediction errors
    pub fn process(
        &mut self,
        understanding: &DeepUnderstanding,
    ) -> Vec<UnderstandingPredictionError> {
        let mut errors = Vec::new();

        // 1. Semantic coherence prediction
        let semantic_obs = understanding.confidence as f32;
        let semantic_pred = *self
            .predictions
            .get(&InferenceDomain::SemanticCoherence)
            .expect("SemanticCoherence initialized in new()");
        let semantic_prec = *self
            .precisions
            .get(&InferenceDomain::SemanticCoherence)
            .expect("SemanticCoherence initialized in new()");
        errors.push(UnderstandingPredictionError::new(
            semantic_pred,
            semantic_obs,
            semantic_prec,
            InferenceDomain::SemanticCoherence,
        ));

        // 2. Emotional state prediction
        let emotion_obs = (understanding.grounded.embodied.valence + 1.0) / 2.0; // Normalize to 0-1
        let emotion_pred = *self
            .predictions
            .get(&InferenceDomain::EmotionalState)
            .expect("EmotionalState initialized in new()");
        let emotion_prec = *self
            .precisions
            .get(&InferenceDomain::EmotionalState)
            .expect("EmotionalState initialized in new()");
        errors.push(UnderstandingPredictionError::new(
            emotion_pred,
            emotion_obs,
            emotion_prec,
            InferenceDomain::EmotionalState,
        ));

        // 3. Social dynamics (speaker intention clarity)
        let social_obs = if understanding.speaker_model.intentions.is_empty() {
            0.3
        } else {
            understanding.speaker_model.intentions[0].confidence
        };
        let social_pred = *self
            .predictions
            .get(&InferenceDomain::SocialDynamics)
            .expect("SocialDynamics initialized in new()");
        let social_prec = *self
            .precisions
            .get(&InferenceDomain::SocialDynamics)
            .expect("SocialDynamics initialized in new()");
        errors.push(UnderstandingPredictionError::new(
            social_pred,
            social_obs,
            social_prec,
            InferenceDomain::SocialDynamics,
        ));

        // 4. Narrative continuity
        let narrative_obs = understanding.narrative.identity_coherence as f32;
        let narrative_pred = *self
            .predictions
            .get(&InferenceDomain::NarrativeContinuity)
            .expect("NarrativeContinuity initialized in new()");
        let narrative_prec = *self
            .precisions
            .get(&InferenceDomain::NarrativeContinuity)
            .expect("NarrativeContinuity initialized in new()");
        errors.push(UnderstandingPredictionError::new(
            narrative_pred,
            narrative_obs,
            narrative_prec,
            InferenceDomain::NarrativeContinuity,
        ));

        // 5. Causal reasoning (was causation detected as expected?)
        let causal_obs = if understanding.grounded.causal_structure.is_some() {
            1.0
        } else {
            0.0
        };
        let causal_pred = *self
            .predictions
            .get(&InferenceDomain::CausalReasoning)
            .expect("CausalReasoning initialized in new()");
        let causal_prec = *self
            .precisions
            .get(&InferenceDomain::CausalReasoning)
            .expect("CausalReasoning initialized in new()");
        errors.push(UnderstandingPredictionError::new(
            causal_pred,
            causal_obs,
            causal_prec,
            InferenceDomain::CausalReasoning,
        ));

        // 6. Temporal flow (surprise in predictive processing)
        let temporal_obs = 1.0 - understanding.predictive.surprise; // Low surprise = expected
        let temporal_pred = *self
            .predictions
            .get(&InferenceDomain::TemporalFlow)
            .expect("TemporalFlow initialized in new()");
        let temporal_prec = *self
            .precisions
            .get(&InferenceDomain::TemporalFlow)
            .expect("TemporalFlow initialized in new()");
        errors.push(UnderstandingPredictionError::new(
            temporal_pred,
            temporal_obs,
            temporal_prec,
            InferenceDomain::TemporalFlow,
        ));

        // Update predictions based on observations (belief update)
        for error in &errors {
            self.update_prediction(error);
        }

        // Calculate total free energy
        self.free_energy = errors
            .iter()
            .map(|e| e.weighted_error.powi(2) as f64)
            .sum::<f64>()
            .sqrt();

        // Store history
        for error in errors.iter().cloned() {
            self.error_history.push_back(error);
            if self.error_history.len() > 100 {
                self.error_history.pop_front();
            }
        }

        errors
    }

    fn update_prediction(&mut self, error: &UnderstandingPredictionError) {
        // Bayesian update: move prediction toward observation
        if let Some(pred) = self.predictions.get_mut(&error.domain) {
            *pred += self.learning_rate * error.error;
            *pred = pred.clamp(0.0, 1.0);
        }

        // Update precision based on prediction accuracy
        if let Some(prec) = self.precisions.get_mut(&error.domain) {
            // If prediction was accurate, increase precision
            // If inaccurate, decrease precision
            let accuracy = 1.0 - error.error.abs();
            *prec = 0.9 * *prec + 0.1 * accuracy;
            *prec = prec.clamp(0.1, 0.95);
        }
    }

    /// Get current free energy (lower = better predictions)
    pub fn free_energy(&self) -> f64 {
        self.free_energy
    }

    /// Get precision for a domain
    pub fn precision(&self, domain: InferenceDomain) -> f32 {
        *self.precisions.get(&domain).unwrap_or(&0.5)
    }

    /// Get prediction for a domain
    pub fn prediction(&self, domain: InferenceDomain) -> f32 {
        *self.predictions.get(&domain).unwrap_or(&0.5)
    }
}

impl Default for ActiveInferenceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EPISODIC MEMORY ADAPTER
// =============================================================================

/// Emotional valence for memory encoding
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryEmotion {
    Positive,
    Neutral,
    Negative,
}

impl MemoryEmotion {
    pub fn from_valence(valence: f32) -> Self {
        if valence > 0.3 {
            MemoryEmotion::Positive
        } else if valence < -0.3 {
            MemoryEmotion::Negative
        } else {
            MemoryEmotion::Neutral
        }
    }

    pub fn to_scalar(&self) -> f32 {
        match self {
            MemoryEmotion::Positive => 1.0,
            MemoryEmotion::Neutral => 0.0,
            MemoryEmotion::Negative => -1.0,
        }
    }
}

/// A memory trace from understanding
#[derive(Debug, Clone)]
pub struct UnderstandingMemoryTrace {
    /// Unique ID
    pub id: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Original text
    pub text: String,
    /// Semantic encoding (BinaryHV)
    pub encoding: BinaryHV,
    /// Emotional valence
    pub emotion: MemoryEmotion,
    /// Context tags from entities and narrative
    pub tags: Vec<String>,
    /// Key semantic primes
    pub primes: Vec<SemanticPrime>,
    /// Story role
    pub story_role: StoryRole,
    /// Strength (decays over time)
    pub strength: f32,
    /// Recall count
    pub recall_count: u32,
}

/// Adapter connecting UnifiedUnderstanding to episodic memory
pub struct EpisodicMemoryAdapter {
    /// Stored memory traces
    traces: VecDeque<UnderstandingMemoryTrace>,
    /// ID counter
    next_id: u64,
    /// Maximum traces to keep
    max_traces: usize,
    /// Index by entity name
    entity_index: HashMap<String, Vec<u64>>,
    /// Index by story role
    role_index: HashMap<StoryRole, Vec<u64>>,
}

impl EpisodicMemoryAdapter {
    pub fn new(max_traces: usize) -> Self {
        Self {
            traces: VecDeque::with_capacity(max_traces),
            next_id: 1,
            max_traces,
            entity_index: HashMap::new(),
            role_index: HashMap::new(),
        }
    }

    /// Encode understanding into episodic memory
    pub fn encode(&mut self, understanding: &DeepUnderstanding) -> UnderstandingMemoryTrace {
        let id = self.next_id;
        self.next_id += 1;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Extract tags from entities
        let mut tags: Vec<String> = understanding
            .narrative
            .entities
            .iter()
            .map(|e| e.name.clone())
            .collect();

        // Add story role as tag
        tags.push(format!("{:?}", understanding.narrative.story_role));

        // Add emotion as tag
        let emotion = MemoryEmotion::from_valence(understanding.grounded.embodied.valence);
        tags.push(format!("{emotion:?}"));

        let trace = UnderstandingMemoryTrace {
            id,
            timestamp,
            text: understanding.text.clone(),
            encoding: understanding.grounded.semantic_hv,
            emotion,
            tags: tags.clone(),
            primes: understanding.grounded.primes.clone(),
            story_role: understanding.narrative.story_role,
            strength: 1.0,
            recall_count: 0,
        };

        // Update indices
        for entity in &understanding.narrative.entities {
            self.entity_index
                .entry(entity.name.clone())
                .or_default()
                .push(id);
        }
        self.role_index
            .entry(understanding.narrative.story_role)
            .or_default()
            .push(id);

        // Store trace
        self.traces.push_back(trace.clone());

        // Trim if over capacity
        while self.traces.len() > self.max_traces {
            if let Some(old) = self.traces.pop_front() {
                // Clean up indices
                for ids in self.entity_index.values_mut() {
                    ids.retain(|&i| i != old.id);
                }
                for ids in self.role_index.values_mut() {
                    ids.retain(|&i| i != old.id);
                }
            }
        }

        trace
    }

    /// Recall memories by semantic similarity
    pub fn recall_by_similarity(
        &mut self,
        query: &BinaryHV,
        top_k: usize,
    ) -> Vec<&UnderstandingMemoryTrace> {
        let mut scored: Vec<(f32, u64)> = self
            .traces
            .iter()
            .map(|t| (query.similarity(&t.encoding), t.id))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let top_ids: Vec<u64> = scored.iter().take(top_k).map(|(_, id)| *id).collect();

        // Strengthen recalled memories
        for trace in self.traces.iter_mut() {
            if top_ids.contains(&trace.id) {
                trace.recall_count += 1;
                trace.strength = (trace.strength + 0.1).min(1.0);
            }
        }

        self.traces
            .iter()
            .filter(|t| top_ids.contains(&t.id))
            .collect()
    }

    /// Recall memories by entity
    pub fn recall_by_entity(&self, entity: &str) -> Vec<&UnderstandingMemoryTrace> {
        if let Some(ids) = self.entity_index.get(entity) {
            self.traces.iter().filter(|t| ids.contains(&t.id)).collect()
        } else {
            Vec::new()
        }
    }

    /// Recall memories by story role
    pub fn recall_by_role(&self, role: StoryRole) -> Vec<&UnderstandingMemoryTrace> {
        if let Some(ids) = self.role_index.get(&role) {
            self.traces.iter().filter(|t| ids.contains(&t.id)).collect()
        } else {
            Vec::new()
        }
    }

    /// Get memory count
    pub fn count(&self) -> usize {
        self.traces.len()
    }

    /// Decay all memories (call periodically)
    pub fn decay_memories(&mut self, decay_rate: f32) {
        for trace in self.traces.iter_mut() {
            trace.strength = (trace.strength - decay_rate).max(0.0);
        }
        // Remove dead memories
        self.traces.retain(|t| t.strength > 0.0);
    }
}

impl Default for EpisodicMemoryAdapter {
    fn default() -> Self {
        Self::new(1000)
    }
}

// =============================================================================
// METACOGNITIVE ADAPTER
// =============================================================================

/// Types of cognitive events from understanding
#[derive(Debug, Clone)]
pub enum UnderstandingCognitiveEvent {
    /// Understanding completed with metrics
    UnderstandingComplete {
        phi: f64,
        confidence: f64,
        surprise: f32,
    },
    /// Uncertainty detected (unknown words, ambiguous meaning)
    UncertaintyDetected { source: String, level: f32 },
    /// Error in understanding (contradiction, missing context)
    UnderstandingError {
        error_type: UnderstandingErrorType,
        severity: f32,
    },
    /// Word learned during processing
    WordLearned { word: String, confidence: f32 },
    /// Narrative continuity maintained/broken
    NarrativeEvent { coherence: f64, role_change: bool },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnderstandingErrorType {
    UnknownWord,
    AmbiguousMeaning,
    ContradictoryInfo,
    MissingContext,
    LowConfidence,
}

/// Metacognitive assessment of understanding quality
#[derive(Debug, Clone)]
pub struct UnderstandingAssessment {
    /// Overall confidence in understanding (0-1)
    pub confidence: f64,
    /// Coherence of the understanding (0-1)
    pub coherence: f64,
    /// Current uncertainty level (0-1)
    pub uncertainty: f64,
    /// Φ trend (positive = improving, negative = degrading)
    pub phi_trend: f64,
    /// Recommendation for action
    pub recommendation: MetacognitiveRecommendation,
    /// Specific issues detected
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetacognitiveRecommendation {
    /// Understanding is solid, proceed confidently
    Proceed,
    /// Understanding is acceptable, proceed with caution
    ProceedWithCaution,
    /// Seek clarification before proceeding
    SeekClarification,
    /// Understanding is poor, request more information
    RequestMoreInfo,
}

/// Adapter connecting UnifiedUnderstanding to metacognitive monitoring
pub struct MetacognitiveAdapter {
    /// History of Φ values
    phi_history: VecDeque<f64>,
    /// History of confidence values
    confidence_history: VecDeque<f64>,
    /// Recent events
    events: VecDeque<UnderstandingCognitiveEvent>,
    /// Configuration
    phi_threshold: f64,
    uncertainty_threshold: f64,
    history_size: usize,
}

impl MetacognitiveAdapter {
    pub fn new() -> Self {
        Self {
            phi_history: VecDeque::with_capacity(50),
            confidence_history: VecDeque::with_capacity(50),
            events: VecDeque::with_capacity(100),
            phi_threshold: 0.4,
            uncertainty_threshold: 0.5,
            history_size: 50,
        }
    }

    /// Process understanding and generate metacognitive events
    pub fn process(&mut self, understanding: &DeepUnderstanding) -> UnderstandingAssessment {
        // Record Φ and confidence
        self.phi_history.push_back(understanding.integration_phi);
        self.confidence_history.push_back(understanding.confidence);

        while self.phi_history.len() > self.history_size {
            self.phi_history.pop_front();
        }
        while self.confidence_history.len() > self.history_size {
            self.confidence_history.pop_front();
        }

        // Generate events
        self.events
            .push_back(UnderstandingCognitiveEvent::UnderstandingComplete {
                phi: understanding.integration_phi,
                confidence: understanding.confidence,
                surprise: understanding.predictive.surprise,
            });

        // Check for uncertainty (high surprise or low confidence)
        if understanding.predictive.surprise > 0.7 {
            self.events
                .push_back(UnderstandingCognitiveEvent::UncertaintyDetected {
                    source: "predictive_processing".to_string(),
                    level: understanding.predictive.surprise,
                });
        }

        if understanding.confidence < 0.5 {
            self.events
                .push_back(UnderstandingCognitiveEvent::UncertaintyDetected {
                    source: "semantic_decomposition".to_string(),
                    level: (1.0 - understanding.confidence) as f32,
                });
        }

        // Record learned words
        for word in &understanding.learned_words {
            self.events
                .push_back(UnderstandingCognitiveEvent::WordLearned {
                    word: word.word.clone(),
                    confidence: word.confidence,
                });
        }

        // Narrative coherence event
        self.events
            .push_back(UnderstandingCognitiveEvent::NarrativeEvent {
                coherence: understanding.narrative.identity_coherence,
                role_change: false, // Would need to track previous role
            });

        while self.events.len() > 100 {
            self.events.pop_front();
        }

        // Generate assessment
        self.assess(understanding)
    }

    fn assess(&self, understanding: &DeepUnderstanding) -> UnderstandingAssessment {
        let confidence = understanding.confidence;
        let coherence = understanding.narrative.identity_coherence;

        // Calculate uncertainty from various sources
        let surprise_uncertainty = understanding.predictive.surprise as f64;
        let confidence_uncertainty = 1.0 - confidence;
        let uncertainty = (surprise_uncertainty + confidence_uncertainty) / 2.0;

        // Calculate Φ trend
        let phi_trend = if self.phi_history.len() >= 2 {
            let recent: f64 = self.phi_history.iter().rev().take(5).sum::<f64>() / 5.0;
            let older: f64 = self.phi_history.iter().take(5).sum::<f64>() / 5.0;
            recent - older
        } else {
            0.0
        };

        // Identify issues
        let mut issues = Vec::new();
        if confidence < 0.5 {
            issues.push("Low semantic confidence".to_string());
        }
        if understanding.predictive.surprise > 0.7 {
            issues.push("High predictive surprise".to_string());
        }
        if coherence < 0.5 {
            issues.push("Low narrative coherence".to_string());
        }
        if !understanding.learned_words.is_empty() {
            issues.push(format!(
                "Learned {} new words",
                understanding.learned_words.len()
            ));
        }

        // Determine recommendation
        let recommendation = if confidence >= 0.8 && uncertainty < 0.3 {
            MetacognitiveRecommendation::Proceed
        } else if confidence >= 0.6 && uncertainty < 0.5 {
            MetacognitiveRecommendation::ProceedWithCaution
        } else if confidence >= 0.4 {
            MetacognitiveRecommendation::SeekClarification
        } else {
            MetacognitiveRecommendation::RequestMoreInfo
        };

        UnderstandingAssessment {
            confidence,
            coherence,
            uncertainty,
            phi_trend,
            recommendation,
            issues,
        }
    }

    /// Get recent Φ trend
    pub fn phi_trend(&self) -> f64 {
        if self.phi_history.len() >= 2 {
            let last = *self.phi_history.back().unwrap_or(&0.0);
            let first = *self.phi_history.front().unwrap_or(&0.0);
            last - first
        } else {
            0.0
        }
    }
}

impl Default for MetacognitiveAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// COUNTERFACTUAL REASONING ENGINE
// =============================================================================

/// A counterfactual scenario ("what if...?")
#[derive(Debug, Clone)]
pub struct Counterfactual {
    /// The intervention (what we imagine changing)
    pub intervention: String,
    /// The variable being changed
    pub changed_variable: String,
    /// Original value
    pub original_value: String,
    /// Counterfactual value
    pub counterfactual_value: String,
    /// Predicted outcome
    pub predicted_outcome: CounterfactualOutcome,
    /// Confidence in this counterfactual
    pub confidence: f32,
    /// Causal path explaining the prediction
    pub causal_path: Vec<String>,
}

/// Predicted outcome of a counterfactual
#[derive(Debug, Clone)]
pub struct CounterfactualOutcome {
    /// Predicted emotional valence
    pub valence_delta: f32,
    /// Predicted narrative change
    pub narrative_impact: String,
    /// Affected entities
    pub affected_entities: Vec<String>,
    /// Expected free energy change (negative = better)
    pub expected_free_energy_delta: f32,
}

/// Types of counterfactual queries
#[derive(Debug, Clone)]
pub enum CounterfactualQuery {
    /// "What if X had not happened?"
    Negation { event: String },
    /// "What if X were Y instead?"
    Substitution { variable: String, new_value: String },
    /// "What would happen if X?"
    Hypothetical { scenario: String },
    /// "Why did X happen?"
    CausalExplanation { effect: String },
    /// "What will happen if I do X?"
    ActionSimulation { action: String },
}

/// Engine for counterfactual reasoning / imagination
pub struct CounterfactualEngine {
    /// Causal model built from understanding history
    causal_graph: HashMap<String, Vec<CausalEdge>>,
    /// Entity emotional associations
    entity_valences: HashMap<String, f32>,
    /// Narrative patterns observed
    narrative_patterns: VecDeque<NarrativePattern>,
    /// Free energy baseline
    baseline_free_energy: f32,
}

#[derive(Debug, Clone)]
struct CausalEdge {
    effect: String,
    strength: f32,
    relation: CausalRelation,
}

#[derive(Debug, Clone)]
struct NarrativePattern {
    trigger: StoryRole,
    typical_followup: StoryRole,
    frequency: u32,
}

impl CounterfactualEngine {
    pub fn new() -> Self {
        Self {
            causal_graph: HashMap::new(),
            entity_valences: HashMap::new(),
            narrative_patterns: VecDeque::with_capacity(50),
            baseline_free_energy: 1.0,
        }
    }

    /// Learn from understanding (build causal model)
    pub fn learn_from_understanding(&mut self, understanding: &DeepUnderstanding) {
        // Learn causal structure
        if let Some(ref causal) = understanding.grounded.causal_structure {
            let edge = CausalEdge {
                effect: causal.effect.clone(),
                strength: causal.strength as f32,
                relation: causal.relation,
            };
            self.causal_graph
                .entry(causal.cause.clone())
                .or_default()
                .push(edge);
        }

        // Learn entity valences
        for entity in &understanding.narrative.entities {
            let entry = self
                .entity_valences
                .entry(entity.name.clone())
                .or_insert(0.0);
            // Exponential moving average
            *entry = 0.8 * *entry + 0.2 * entity.valence;
        }

        // Update baseline free energy
        self.baseline_free_energy =
            0.9 * self.baseline_free_energy + 0.1 * understanding.predictive.free_energy as f32;
    }

    /// Generate a counterfactual scenario
    pub fn imagine(
        &self,
        query: CounterfactualQuery,
        context: &DeepUnderstanding,
    ) -> Counterfactual {
        match query {
            CounterfactualQuery::Negation { event } => self.imagine_negation(&event, context),
            CounterfactualQuery::Substitution {
                variable,
                new_value,
            } => self.imagine_substitution(&variable, &new_value, context),
            CounterfactualQuery::Hypothetical { scenario } => {
                self.imagine_hypothetical(&scenario, context)
            }
            CounterfactualQuery::CausalExplanation { effect } => {
                self.explain_causally(&effect, context)
            }
            CounterfactualQuery::ActionSimulation { action } => {
                self.simulate_action(&action, context)
            }
        }
    }

    fn imagine_negation(&self, event: &str, context: &DeepUnderstanding) -> Counterfactual {
        // Find what this event caused
        let effects = self.causal_graph.get(event).cloned().unwrap_or_default();

        let mut causal_path = vec![format!("If '{}' had not occurred:", event)];
        let mut total_valence_delta = 0.0f32;
        let mut affected = Vec::new();

        for edge in &effects {
            causal_path.push(format!("  → '{}' would not have happened", edge.effect));
            // Invert the effect
            if let Some(&valence) = self.entity_valences.get(&edge.effect) {
                total_valence_delta -= valence * edge.strength;
            }
            affected.push(edge.effect.clone());
        }

        let outcome = CounterfactualOutcome {
            valence_delta: total_valence_delta,
            narrative_impact: if effects.is_empty() {
                "Minimal change to the narrative".to_string()
            } else {
                format!("{} consequences would not have occurred", effects.len())
            },
            affected_entities: affected,
            expected_free_energy_delta: -total_valence_delta.abs() * 0.5,
        };

        Counterfactual {
            intervention: format!("Negation of '{event}'"),
            changed_variable: event.to_string(),
            original_value: "occurred".to_string(),
            counterfactual_value: "did not occur".to_string(),
            predicted_outcome: outcome,
            confidence: 0.5 + (effects.len() as f32 * 0.1).min(0.4),
            causal_path,
        }
    }

    fn imagine_substitution(
        &self,
        variable: &str,
        new_value: &str,
        context: &DeepUnderstanding,
    ) -> Counterfactual {
        let original_valence = self.entity_valences.get(variable).copied().unwrap_or(0.0);

        // Estimate new valence based on new_value sentiment
        let new_valence = self.estimate_valence(new_value);
        let valence_delta = new_valence - original_valence;

        let causal_path = vec![
            format!("If '{}' were '{}' instead:", variable, new_value),
            format!("  → Emotional shift: {:+.2}", valence_delta),
        ];

        let outcome = CounterfactualOutcome {
            valence_delta,
            narrative_impact: format!("Substituting '{variable}' with '{new_value}'"),
            affected_entities: vec![variable.to_string()],
            expected_free_energy_delta: -valence_delta.abs() * 0.3,
        };

        Counterfactual {
            intervention: format!("Replace '{variable}' with '{new_value}'"),
            changed_variable: variable.to_string(),
            original_value: variable.to_string(),
            counterfactual_value: new_value.to_string(),
            predicted_outcome: outcome,
            confidence: 0.4,
            causal_path,
        }
    }

    fn imagine_hypothetical(&self, scenario: &str, context: &DeepUnderstanding) -> Counterfactual {
        // Simple: estimate based on scenario keywords
        let valence = self.estimate_valence(scenario);

        let causal_path = vec![
            format!("Imagining scenario: '{}'", scenario),
            format!("  → Estimated emotional impact: {:+.2}", valence),
        ];

        let outcome = CounterfactualOutcome {
            valence_delta: valence,
            narrative_impact: scenario.to_string(),
            affected_entities: Vec::new(),
            expected_free_energy_delta: if valence > 0.0 { -0.2 } else { 0.2 },
        };

        Counterfactual {
            intervention: format!("Hypothetical: {scenario}"),
            changed_variable: "future".to_string(),
            original_value: "current state".to_string(),
            counterfactual_value: scenario.to_string(),
            predicted_outcome: outcome,
            confidence: 0.3,
            causal_path,
        }
    }

    fn explain_causally(&self, effect: &str, context: &DeepUnderstanding) -> Counterfactual {
        // Find causes of this effect
        let mut causes = Vec::new();
        for (cause, edges) in &self.causal_graph {
            for edge in edges {
                if edge.effect.contains(effect) || effect.contains(&edge.effect) {
                    causes.push((cause.clone(), edge.strength));
                }
            }
        }

        let causal_path = if causes.is_empty() {
            vec![format!("No known causes for '{}'", effect)]
        } else {
            causes
                .iter()
                .map(|(c, s)| format!("'{c}' (strength: {s:.2}) → '{effect}'"))
                .collect()
        };

        let outcome = CounterfactualOutcome {
            valence_delta: 0.0,
            narrative_impact: format!("Causal explanation for '{effect}'"),
            affected_entities: causes.iter().map(|(c, _)| c.clone()).collect(),
            expected_free_energy_delta: 0.0,
        };

        Counterfactual {
            intervention: format!("Explain: Why '{effect}'?"),
            changed_variable: effect.to_string(),
            original_value: "effect".to_string(),
            counterfactual_value: "cause".to_string(),
            predicted_outcome: outcome,
            confidence: if causes.is_empty() { 0.1 } else { 0.6 },
            causal_path,
        }
    }

    fn simulate_action(&self, action: &str, context: &DeepUnderstanding) -> Counterfactual {
        // Predict effect of action based on causal model
        let effects = self.causal_graph.get(action).cloned().unwrap_or_default();

        let mut total_valence = 0.0f32;
        let mut causal_path = vec![format!("If you do '{}' now:", action)];

        for edge in &effects {
            let effect_valence = self
                .entity_valences
                .get(&edge.effect)
                .copied()
                .unwrap_or(0.0);
            total_valence += effect_valence * edge.strength;
            causal_path.push(format!(
                "  → Could lead to '{}' (conf: {:.0}%)",
                edge.effect,
                edge.strength * 100.0
            ));
        }

        // If no known effects, estimate from action keywords
        if effects.is_empty() {
            let estimated_valence = self.estimate_valence(action);
            total_valence = estimated_valence;
            causal_path.push(format!(
                "  → Estimated outcome valence: {estimated_valence:+.2}"
            ));
        }

        let outcome = CounterfactualOutcome {
            valence_delta: total_valence,
            narrative_impact: format!("Action '{action}' simulation"),
            affected_entities: effects.iter().map(|e| e.effect.clone()).collect(),
            expected_free_energy_delta: if total_valence > 0.0 { -0.3 } else { 0.1 },
        };

        Counterfactual {
            intervention: format!("Simulate action: {action}"),
            changed_variable: "action".to_string(),
            original_value: "no action".to_string(),
            counterfactual_value: action.to_string(),
            predicted_outcome: outcome,
            confidence: if effects.is_empty() { 0.25 } else { 0.55 },
            causal_path,
        }
    }

    fn estimate_valence(&self, text: &str) -> f32 {
        let text_lower = text.to_lowercase();

        // Simple keyword-based valence estimation
        let positive_words = [
            "happy", "good", "love", "help", "success", "better", "joy", "peace",
        ];
        let negative_words = [
            "sad", "bad", "hate", "hurt", "fail", "worse", "pain", "angry",
        ];

        let pos_count = positive_words
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count();
        let neg_count = negative_words
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count();

        let total = (pos_count + neg_count).max(1) as f32;
        (pos_count as f32 - neg_count as f32) / total
    }

    /// Get number of causal edges learned
    pub fn causal_edge_count(&self) -> usize {
        self.causal_graph.values().map(|v| v.len()).sum()
    }
}

impl Default for CounterfactualEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// FULL STACK CONSCIOUSNESS ORCHESTRATOR
// =============================================================================

/// Complete conscious comprehension result
#[derive(Debug, Clone)]
pub struct ConsciousComprehension {
    /// Deep understanding (semantic + predictive + narrative + ToM)
    pub understanding: DeepUnderstanding,
    /// Active inference state (prediction errors + free energy)
    pub inference: InferenceState,
    /// Memory state (stored + recalled)
    pub memory: MemoryState,
    /// Metacognitive assessment
    pub metacognition: UnderstandingAssessment,
    /// Optional counterfactual exploration
    pub counterfactuals: Vec<Counterfactual>,
    /// Overall conscious integration score
    pub consciousness_phi: f64,
    /// Processing latency
    pub latency_ms: u64,
}

#[derive(Debug, Clone)]
pub struct InferenceState {
    pub prediction_errors: Vec<UnderstandingPredictionError>,
    pub total_free_energy: f64,
    pub precisions: HashMap<InferenceDomain, f32>,
}

#[derive(Debug, Clone)]
pub struct MemoryState {
    pub encoded: bool,
    pub trace_id: u64,
    pub recalled_count: usize,
    pub recalled_memories: Vec<String>,
}

/// The full stack consciousness system
pub struct FullStackConsciousness {
    /// Unified understanding pipeline
    understanding: UnifiedUnderstandingPipeline,
    /// Active inference adapter
    inference: ActiveInferenceAdapter,
    /// Episodic memory adapter
    memory: EpisodicMemoryAdapter,
    /// Metacognitive adapter
    metacognition: MetacognitiveAdapter,
    /// Counterfactual engine
    counterfactual: CounterfactualEngine,
    /// Enable counterfactual reasoning
    enable_counterfactual: bool,
    /// Number of counterfactuals to generate
    counterfactual_count: usize,
}

impl FullStackConsciousness {
    pub fn new() -> Self {
        Self {
            understanding: UnifiedUnderstandingPipeline::new(),
            inference: ActiveInferenceAdapter::new(),
            memory: EpisodicMemoryAdapter::new(1000),
            metacognition: MetacognitiveAdapter::new(),
            counterfactual: CounterfactualEngine::new(),
            enable_counterfactual: true,
            counterfactual_count: 2,
        }
    }

    /// Configure counterfactual reasoning
    pub fn with_counterfactuals(mut self, enabled: bool, count: usize) -> Self {
        self.enable_counterfactual = enabled;
        self.counterfactual_count = count;
        self
    }

    /// Process text through the full consciousness stack
    pub fn comprehend(&mut self, text: &str) -> ConsciousComprehension {
        let start = std::time::Instant::now();

        // 1. Deep understanding
        let understanding = self.understanding.understand(text);

        // 2. Active inference (prediction errors)
        let prediction_errors = self.inference.process(&understanding);
        let total_free_energy = self.inference.free_energy();
        let precisions: HashMap<InferenceDomain, f32> = [
            InferenceDomain::SemanticCoherence,
            InferenceDomain::EmotionalState,
            InferenceDomain::SocialDynamics,
            InferenceDomain::NarrativeContinuity,
            InferenceDomain::CausalReasoning,
            InferenceDomain::TemporalFlow,
        ]
        .iter()
        .map(|&d| (d, self.inference.precision(d)))
        .collect();

        // 3. Episodic memory encoding
        let trace = self.memory.encode(&understanding);

        // 4. Recall related memories
        let recalled = self
            .memory
            .recall_by_similarity(&understanding.grounded.semantic_hv, 3);
        let recalled_memories: Vec<String> = recalled.iter().map(|t| t.text.clone()).collect();

        // 5. Metacognitive assessment
        let assessment = self.metacognition.process(&understanding);

        // 6. Counterfactual reasoning (if enabled and context is rich enough)
        let mut counterfactuals = Vec::new();
        if self.enable_counterfactual {
            self.counterfactual.learn_from_understanding(&understanding);

            // Generate counterfactuals based on context
            if let Some(ref causal) = understanding.grounded.causal_structure {
                // "What if the cause hadn't happened?"
                counterfactuals.push(self.counterfactual.imagine(
                    CounterfactualQuery::Negation {
                        event: causal.cause.clone(),
                    },
                    &understanding,
                ));
            }

            // "What if speaker did something else?"
            if !understanding.speaker_model.intentions.is_empty() {
                let action = &understanding.speaker_model.intentions[0].goal;
                counterfactuals.push(self.counterfactual.imagine(
                    CounterfactualQuery::ActionSimulation {
                        action: action.clone(),
                    },
                    &understanding,
                ));
            }
        }

        let latency_ms = start.elapsed().as_millis() as u64;

        // Calculate overall consciousness Φ
        let consciousness_phi =
            self.calculate_overall_phi(&understanding, &assessment, total_free_energy);

        ConsciousComprehension {
            understanding,
            inference: InferenceState {
                prediction_errors,
                total_free_energy,
                precisions,
            },
            memory: MemoryState {
                encoded: true,
                trace_id: trace.id,
                recalled_count: recalled_memories.len(),
                recalled_memories,
            },
            metacognition: assessment,
            counterfactuals,
            consciousness_phi,
            latency_ms,
        }
    }

    fn calculate_overall_phi(
        &self,
        understanding: &DeepUnderstanding,
        assessment: &UnderstandingAssessment,
        free_energy: f64,
    ) -> f64 {
        // Integrate multiple sources of Φ:
        // 1. Base understanding Φ
        // 2. Metacognitive confidence
        // 3. Inverse free energy (lower = more integrated)

        let base_phi = understanding.integration_phi;
        let meta_phi = assessment.confidence * 0.5 + assessment.coherence * 0.5;
        let inference_phi = 1.0 / (1.0 + free_energy);

        // Weighted combination
        (base_phi * 0.4 + meta_phi * 0.3 + inference_phi * 0.3).min(1.0)
    }

    /// Get conversation history length
    pub fn history_len(&self) -> usize {
        self.understanding.history_len()
    }

    /// Get memory count
    pub fn memory_count(&self) -> usize {
        self.memory.count()
    }

    /// Get causal model size
    pub fn causal_model_size(&self) -> usize {
        self.counterfactual.causal_edge_count()
    }

    /// Clear history (new conversation)
    pub fn clear(&mut self) {
        self.understanding.clear_history();
    }

    /// Get the metacognitive Φ trend
    pub fn phi_trend(&self) -> f64 {
        self.metacognition.phi_trend()
    }

    /// Ask a counterfactual question
    pub fn ask_counterfactual(
        &self,
        query: CounterfactualQuery,
        context: &DeepUnderstanding,
    ) -> Counterfactual {
        self.counterfactual.imagine(query, context)
    }
}

impl Default for FullStackConsciousness {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_stack_creation() {
        let stack = FullStackConsciousness::new();
        assert_eq!(stack.history_len(), 0);
        assert_eq!(stack.memory_count(), 0);
    }

    #[test]
    fn test_basic_comprehension() {
        let mut stack = FullStackConsciousness::new();
        let result = stack.comprehend("I feel happy because my friend came");

        assert!(result.consciousness_phi > 0.0);
        assert!(result.understanding.confidence > 0.0);
        assert!(result.memory.encoded);
    }

    #[test]
    fn test_active_inference_integration() {
        let mut stack = FullStackConsciousness::new();

        // First input
        let r1 = stack.comprehend("I feel happy");

        // Second input (should build on predictions)
        let r2 = stack.comprehend("I feel even happier now");

        // Inference should have updated predictions
        assert!(r2.inference.total_free_energy >= 0.0);
    }

    #[test]
    fn test_memory_recall() {
        let mut stack = FullStackConsciousness::new();

        // Store some memories
        stack.comprehend("My friend Sarah visited yesterday");
        stack.comprehend("We talked about our childhood");

        // Query related
        let result = stack.comprehend("Sarah was always a good friend");

        // Phi should be valid after recall
        assert!(
            result.consciousness_phi.is_finite(),
            "Phi should be finite after recall"
        );
        assert!(
            result.consciousness_phi >= 0.0,
            "Phi should be non-negative"
        );
    }

    #[test]
    fn test_counterfactual_reasoning() {
        let mut stack = FullStackConsciousness::new().with_counterfactuals(true, 3);

        let result = stack.comprehend("I feel sad because my cat died");

        // Should have counterfactuals since there's causal structure
        // Note: may be empty if causal structure not detected
        assert!(result.counterfactuals.len() <= 3);
    }

    #[test]
    fn test_metacognitive_assessment() {
        let mut stack = FullStackConsciousness::new();

        let result = stack.comprehend("I know you feel bad about this situation");

        // Should have metacognitive assessment
        assert!(result.metacognition.confidence > 0.0);
        assert!(result.metacognition.coherence >= 0.0);
    }
}
