//! # Enactivist Sensorimotor Contingencies
//!
//! **PARADIGM SHIFT**: Perception is NOT passive reception of information -
//! it is ACTIVE KNOWLEDGE of sensorimotor contingencies!
//!
//! Based on O'Regan & Noe's Sensorimotor Contingency Theory (2001) and
//! enactivist philosophy from Varela, Thompson & Rosch.
//!
//! ## Core Insight
//!
//! "Seeing red" is not having a red quale in your head - it is:
//! - Knowing how red surfaces change appearance as you move
//! - Knowing that red reflects long wavelengths
//! - Knowing how lighting affects red things
//! - Having implicit mastery of all these action-perception laws
//!
//! Perception = Implicit knowledge of sensorimotor contingencies
//!
//! ## The Enactivist Revolution
//!
//! Traditional View (Representationalism):
//! ```text
//! World → Sensors → Brain (representations) → Motor → World
//! ```
//!
//! Enactivist View:
//! ```text
//! ┌──────────────────────────────────────┐
//! │     Organism-Environment Coupling    │
//! │                                      │
//! │   Action ──────► Sensory Change      │
//! │      ▲               │               │
//! │      │               │               │
//! │      └─── Contingency ◄──────┘       │
//! │           Knowledge                  │
//! │                                      │
//! └──────────────────────────────────────┘
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                 SENSORIMOTOR CONTINGENCY SYSTEM                      │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  ┌───────────────────┐    ┌────────────────────┐                    │
//! │  │ SensorimotorContin│    │ ContingencyLearner │                    │
//! │  │      gency        │◄───│                    │                    │
//! │  │                   │    │ - Records action-  │                    │
//! │  │ (action, context) │    │   outcome pairs    │                    │
//! │  │       ↓           │    │ - Builds model     │                    │
//! │  │ expected_change   │    │ - Detects surprise │                    │
//! │  └───────────────────┘    └────────────────────┘                    │
//! │            │                        │                               │
//! │            ▼                        ▼                               │
//! │  ┌───────────────────────────────────────────────────────┐         │
//! │  │              EnactivistPerception                      │         │
//! │  │                                                        │         │
//! │  │  "Perceiving X" = Knowing how X behaves under action   │         │
//! │  │                                                        │         │
//! │  │  Visual: How it changes with eye/head movement         │         │
//! │  │  Tactile: How it responds to touch/pressure            │         │
//! │  │  Auditory: How sound changes with listener movement    │         │
//! │  └───────────────────────────────────────────────────────┘         │
//! │            │                                                        │
//! │            ▼                                                        │
//! │  ┌───────────────────────────────────────────────────────┐         │
//! │  │              ActionAffordances                         │         │
//! │  │                                                        │         │
//! │  │  What actions are possible given current contingencies │         │
//! │  │  Gibson-style: Environment offers action possibilities │         │
//! │  │  Depends on body capabilities AND learned contingencies│         │
//! │  └───────────────────────────────────────────────────────┘         │
//! │                                                                      │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Integration Points
//!
//! - **Motor Cortex** (src/action/): Actions that generate sensory predictions
//! - **Prediction Error**: Contingency violations feed free energy minimization
//! - **Consciousness Equation**: SMC mastery contributes to consciousness level
//! - **HDC Encoding**: Efficient lookup via hyperdimensional computing

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ============================================================================
// HDC ENCODING FOR EFFICIENT CONTINGENCY LOOKUP
// ============================================================================

/// Lightweight HDC-style encoding for sensorimotor patterns
/// Uses 256-bit vectors for efficient similarity search
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContingencyHV {
    /// Binary hypervector (256 bits = 32 bytes)
    bits: [u64; 4],
}

impl ContingencyHV {
    /// Create a random hypervector from a seed
    pub fn random(seed: u64) -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut bits = [0u64; 4];
        for i in 0..4 {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            (i as u64).hash(&mut hasher);
            bits[i] = hasher.finish();
        }
        Self { bits }
    }

    /// Create from a string identifier (deterministic)
    pub fn from_id(id: &str) -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        Self::random(hasher.finish())
    }

    /// Bind two vectors (XOR - creates unique combination)
    pub fn bind(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] ^ other.bits[0],
                self.bits[1] ^ other.bits[1],
                self.bits[2] ^ other.bits[2],
                self.bits[3] ^ other.bits[3],
            ],
        }
    }

    /// Bundle vectors (majority voting for superposition)
    pub fn bundle(vectors: &[Self]) -> Self {
        if vectors.is_empty() {
            return Self::random(0);
        }
        if vectors.len() == 1 {
            return vectors[0].clone();
        }

        let mut counts = [0i32; 256];
        for v in vectors {
            for (i, &word) in v.bits.iter().enumerate() {
                for bit in 0..64 {
                    if (word >> bit) & 1 == 1 {
                        counts[i * 64 + bit] += 1;
                    }
                }
            }
        }

        let threshold = vectors.len() as i32 / 2;
        let mut result = [0u64; 4];
        for (i, chunk) in counts.chunks(64).enumerate() {
            for (bit, &count) in chunk.iter().enumerate() {
                if count > threshold {
                    result[i] |= 1 << bit;
                }
            }
        }

        Self { bits: result }
    }

    /// Compute Hamming similarity (0.0 to 1.0)
    pub fn similarity(&self, other: &Self) -> f64 {
        let mut same_bits = 0u32;
        for i in 0..4 {
            same_bits += (!(self.bits[i] ^ other.bits[i])).count_ones();
        }
        same_bits as f64 / 256.0
    }

    /// Permute (circular shift for sequence encoding)
    pub fn permute(&self, shift: usize) -> Self {
        let shift = shift % 256;
        if shift == 0 {
            return self.clone();
        }

        // Treat as 256-bit number and rotate
        let word_shift = shift / 64;
        let bit_shift = shift % 64;

        let mut result = [0u64; 4];
        for i in 0..4 {
            let src_idx = (i + 4 - word_shift) % 4;
            let prev_idx = (src_idx + 3) % 4;

            if bit_shift == 0 {
                result[i] = self.bits[src_idx];
            } else {
                result[i] = (self.bits[src_idx] << bit_shift)
                    | (self.bits[prev_idx] >> (64 - bit_shift));
            }
        }

        Self { bits: result }
    }
}

impl Default for ContingencyHV {
    fn default() -> Self {
        Self { bits: [0; 4] }
    }
}

// ============================================================================
// SENSORIMOTOR CONTINGENCY - Core Data Structure
// ============================================================================

/// A sensorimotor contingency: the lawful relation between action and sensation
///
/// This is the fundamental unit of enactivist perception:
/// "If I do X in context C, sensory input will change by Y"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorimotorContingency {
    /// Unique identifier for this contingency
    pub id: String,

    /// The action that causes the sensory change
    pub action: ActionDescriptor,

    /// The context in which this contingency holds
    pub context: ContextDescriptor,

    /// The expected sensory change
    pub expected_change: SensoryChange,

    /// HDC encoding for efficient lookup
    pub encoding: ContingencyHV,

    /// Confidence in this contingency (0.0 to 1.0)
    pub confidence: f64,

    /// Number of times this contingency has been observed
    pub observation_count: u64,

    /// Variance in observed outcomes (uncertainty)
    pub outcome_variance: f64,

    /// When this contingency was last activated
    #[serde(skip, default = "Instant::now")]
    pub last_activated: Instant,

    /// Contexts where this contingency has been verified
    pub verified_contexts: Vec<String>,
}

impl SensorimotorContingency {
    /// Create a new contingency from an action-outcome pair
    pub fn new(
        action: ActionDescriptor,
        context: ContextDescriptor,
        expected_change: SensoryChange,
    ) -> Self {
        // Generate encoding from action + context
        let action_hv = ContingencyHV::from_id(&format!("{:?}", action.action_type));
        let context_hv = ContingencyHV::from_id(&context.id);
        let encoding = action_hv.bind(&context_hv);

        let id = format!(
            "smc_{:?}_{}_{}",
            action.action_type,
            context.id,
            rand::random::<u32>()
        );

        Self {
            id,
            action,
            context,
            expected_change,
            encoding,
            confidence: 0.5,  // Start with medium confidence
            observation_count: 1,
            outcome_variance: 0.3,
            last_activated: Instant::now(),
            verified_contexts: Vec::new(),
        }
    }

    /// Update the contingency with a new observation
    pub fn update_with_observation(&mut self, actual_change: &SensoryChange) {
        self.observation_count += 1;
        self.last_activated = Instant::now();

        // Calculate prediction error
        let error = self.expected_change.distance(actual_change);

        // Update confidence based on prediction accuracy
        let accuracy = 1.0 - error.min(1.0);
        let learning_rate = 0.1;
        self.confidence = self.confidence * (1.0 - learning_rate) + accuracy * learning_rate;

        // Update expected change (moving average)
        self.expected_change.update_toward(actual_change, learning_rate);

        // Update variance estimate
        self.outcome_variance = self.outcome_variance * 0.9 + error * error * 0.1;
    }

    /// Check if this contingency matches an action-context pair
    pub fn matches(&self, action: &ActionDescriptor, context: &ContextDescriptor) -> bool {
        self.action.action_type == action.action_type
            && self.context.similarity(context) > 0.7
    }

    /// Predict sensory change for this contingency
    pub fn predict(&self) -> PredictedChange {
        PredictedChange {
            expected: self.expected_change.clone(),
            confidence: self.confidence,
            variance: self.outcome_variance,
        }
    }
}

/// Descriptor for an action in sensorimotor terms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionDescriptor {
    /// Type of action
    pub action_type: ActionType,

    /// Parameters of the action (e.g., direction, magnitude)
    pub parameters: HashMap<String, f64>,

    /// Effector(s) involved
    pub effectors: Vec<String>,

    /// Duration in milliseconds
    pub duration_ms: u64,
}

impl ActionDescriptor {
    pub fn new(action_type: ActionType) -> Self {
        Self {
            action_type,
            parameters: HashMap::new(),
            effectors: Vec::new(),
            duration_ms: 100,
        }
    }

    pub fn with_parameter(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    pub fn with_effector(mut self, effector: &str) -> Self {
        self.effectors.push(effector.to_string());
        self
    }
}

/// Types of actions in the sensorimotor vocabulary
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    // Visual exploration
    SaccadeLeft,
    SaccadeRight,
    SaccadeUp,
    SaccadeDown,
    Fixate,
    Blink,

    // Head movement
    HeadTurnLeft,
    HeadTurnRight,
    HeadTiltUp,
    HeadTiltDown,

    // Locomotion
    StepForward,
    StepBackward,
    TurnLeft,
    TurnRight,

    // Manipulation
    Reach,
    Grasp,
    Release,
    Push,
    Pull,
    Rotate,

    // Exploration
    Touch,
    Press,
    Tap,
    Stroke,

    // Communication
    Vocalize,
    Gesture,

    // Internal
    Attend,
    Remember,
    Imagine,

    // Custom action type
    Custom,
}

/// Descriptor for the context in which a contingency holds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDescriptor {
    /// Unique identifier
    pub id: String,

    /// Semantic features of the context
    pub features: HashMap<String, f64>,

    /// Object present in the scene
    pub objects: Vec<String>,

    /// Current modality being engaged
    pub modality: SensoryModality,

    /// HDC encoding of the context
    pub encoding: ContingencyHV,
}

impl ContextDescriptor {
    pub fn new(id: &str, modality: SensoryModality) -> Self {
        Self {
            id: id.to_string(),
            features: HashMap::new(),
            objects: Vec::new(),
            modality,
            encoding: ContingencyHV::from_id(id),
        }
    }

    pub fn with_feature(mut self, key: &str, value: f64) -> Self {
        self.features.insert(key.to_string(), value);
        self
    }

    pub fn with_object(mut self, object: &str) -> Self {
        self.objects.push(object.to_string());
        self
    }

    /// Compute similarity to another context
    pub fn similarity(&self, other: &ContextDescriptor) -> f64 {
        // HDC encoding similarity
        let hdc_sim = self.encoding.similarity(&other.encoding);

        // Feature overlap
        let mut feature_sim = 0.0;
        let mut count = 0.0;
        for (key, &val) in &self.features {
            if let Some(&other_val) = other.features.get(key) {
                feature_sim += 1.0 - (val - other_val).abs().min(1.0);
                count += 1.0;
            }
        }
        let feature_sim = if count > 0.0 { feature_sim / count } else { 0.5 };

        // Combine
        hdc_sim * 0.6 + feature_sim * 0.4
    }
}

/// Sensory modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensoryModality {
    Visual,
    Auditory,
    Tactile,
    Proprioceptive,
    Vestibular,
    Olfactory,
    Gustatory,
    Interoceptive,
    MultiModal,
}

/// Description of a sensory change (the "output" of a contingency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryChange {
    /// Which modality changes
    pub modality: SensoryModality,

    /// Direction of change (-1.0 to 1.0 for each dimension)
    pub change_vector: Vec<f64>,

    /// Magnitude of change (0.0 to 1.0)
    pub magnitude: f64,

    /// Temporal profile (when does change occur relative to action?)
    pub onset_delay_ms: u64,
    pub duration_ms: u64,

    /// Qualitative description
    pub description: String,
}

impl SensoryChange {
    pub fn new(modality: SensoryModality) -> Self {
        Self {
            modality,
            change_vector: vec![0.0; 4],  // Default 4D change space
            magnitude: 0.0,
            onset_delay_ms: 50,
            duration_ms: 200,
            description: String::new(),
        }
    }

    pub fn with_vector(mut self, vector: Vec<f64>) -> Self {
        self.magnitude = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.change_vector = vector;
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Compute distance to another sensory change
    pub fn distance(&self, other: &SensoryChange) -> f64 {
        if self.modality != other.modality {
            return 1.0;  // Different modalities = maximum distance
        }

        let len = self.change_vector.len().min(other.change_vector.len());
        if len == 0 {
            return (self.magnitude - other.magnitude).abs();
        }

        let mut sum_sq = 0.0;
        for i in 0..len {
            let diff = self.change_vector[i] - other.change_vector[i];
            sum_sq += diff * diff;
        }

        (sum_sq / len as f64).sqrt()
    }

    /// Update this change toward another (for learning)
    pub fn update_toward(&mut self, target: &SensoryChange, rate: f64) {
        let len = self.change_vector.len().min(target.change_vector.len());
        for i in 0..len {
            self.change_vector[i] = self.change_vector[i] * (1.0 - rate)
                + target.change_vector[i] * rate;
        }
        self.magnitude = self.change_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    }
}

/// A predicted sensory change with uncertainty
#[derive(Debug, Clone)]
pub struct PredictedChange {
    pub expected: SensoryChange,
    pub confidence: f64,
    pub variance: f64,
}

// ============================================================================
// CONTINGENCY LEARNER - Experience-Based Learning
// ============================================================================

/// Learns sensorimotor contingencies from experience
///
/// The core learning system that builds the contingency model through:
/// 1. Recording action-outcome pairs
/// 2. Detecting patterns (action X in context C → change Y)
/// 3. Building and updating contingency representations
/// 4. Detecting violations (surprising outcomes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContingencyLearner {
    /// Learned contingencies indexed by action type
    contingencies: HashMap<ActionType, Vec<SensorimotorContingency>>,

    /// Recent action-outcome pairs for pattern detection
    recent_experiences: VecDeque<Experience>,

    /// Configuration
    config: LearnerConfig,

    /// Statistics
    stats: LearnerStats,
}

/// A single experience (action-outcome pair)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// The action taken
    pub action: ActionDescriptor,

    /// The context at action time
    pub context: ContextDescriptor,

    /// The sensory change that resulted
    pub outcome: SensoryChange,

    /// When this experience occurred
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,

    /// Was this outcome surprising?
    pub surprise: f64,
}

/// Configuration for the contingency learner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerConfig {
    /// Maximum experiences to remember
    pub max_experiences: usize,

    /// Minimum observations before creating contingency
    pub min_observations: u64,

    /// Learning rate for contingency updates
    pub learning_rate: f64,

    /// Threshold for considering outcome "surprising"
    pub surprise_threshold: f64,

    /// Context similarity threshold for matching
    pub context_match_threshold: f64,

    /// Maximum contingencies per action type
    pub max_contingencies_per_action: usize,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            max_experiences: 1000,
            min_observations: 3,
            learning_rate: 0.1,
            surprise_threshold: 0.5,
            context_match_threshold: 0.7,
            max_contingencies_per_action: 50,
        }
    }
}

/// Statistics for the contingency learner
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearnerStats {
    /// Total experiences recorded
    pub total_experiences: u64,

    /// Total contingencies learned
    pub total_contingencies: usize,

    /// Predictions made
    pub predictions_made: u64,

    /// Accurate predictions (within threshold)
    pub accurate_predictions: u64,

    /// Violations detected (surprising outcomes)
    pub violations_detected: u64,

    /// Average prediction error
    pub avg_prediction_error: f64,

    /// Current surprise level
    pub current_surprise: f64,
}

impl ContingencyLearner {
    /// Create a new contingency learner
    pub fn new() -> Self {
        Self::with_config(LearnerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: LearnerConfig) -> Self {
        Self {
            contingencies: HashMap::new(),
            recent_experiences: VecDeque::with_capacity(config.max_experiences),
            config,
            stats: LearnerStats::default(),
        }
    }

    /// Record an experience and learn from it
    pub fn learn(&mut self, action: ActionDescriptor, context: ContextDescriptor, outcome: SensoryChange) -> LearnResult {
        self.stats.total_experiences += 1;

        // Predict before recording
        let prediction = self.predict(&action, &context);

        // Calculate surprise
        let surprise = if let Some(ref pred) = prediction {
            self.stats.predictions_made += 1;
            let error = pred.expected.distance(&outcome);

            if error < self.config.surprise_threshold {
                self.stats.accurate_predictions += 1;
            }

            // Update running average
            self.stats.avg_prediction_error =
                self.stats.avg_prediction_error * 0.99 + error * 0.01;

            error
        } else {
            0.5  // Unknown = medium surprise
        };

        self.stats.current_surprise = surprise;

        let is_violation = surprise > self.config.surprise_threshold;
        if is_violation {
            self.stats.violations_detected += 1;
        }

        // Record experience
        let experience = Experience {
            action: action.clone(),
            context: context.clone(),
            outcome: outcome.clone(),
            timestamp: Instant::now(),
            surprise,
        };

        self.recent_experiences.push_back(experience);
        while self.recent_experiences.len() > self.config.max_experiences {
            self.recent_experiences.pop_front();
        }

        // Update or create contingency
        let contingency_updated = self.update_or_create_contingency(action, context, outcome);

        LearnResult {
            surprise,
            is_violation,
            prediction_error: prediction.map(|p| p.expected.distance(&contingency_updated.expected_change)),
            contingency_updated: Some(contingency_updated.id),
        }
    }

    /// Update an existing contingency or create a new one
    fn update_or_create_contingency(
        &mut self,
        action: ActionDescriptor,
        context: ContextDescriptor,
        outcome: SensoryChange,
    ) -> SensorimotorContingency {
        let contingencies = self.contingencies
            .entry(action.action_type)
            .or_insert_with(Vec::new);

        // Find matching contingency
        let matching = contingencies.iter_mut()
            .find(|c| c.matches(&action, &context));

        if let Some(contingency) = matching {
            contingency.update_with_observation(&outcome);
            contingency.clone()
        } else {
            // Create new contingency
            let contingency = SensorimotorContingency::new(action.clone(), context, outcome);

            // Limit contingencies per action
            if contingencies.len() >= self.config.max_contingencies_per_action {
                // Remove lowest confidence
                if let Some(idx) = contingencies.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)|
                        a.confidence.partial_cmp(&b.confidence).unwrap())
                    .map(|(i, _)| i)
                {
                    contingencies.remove(idx);
                }
            }

            contingencies.push(contingency.clone());
            self.stats.total_contingencies = self.contingencies.values()
                .map(|v| v.len())
                .sum();

            contingency
        }
    }

    /// Predict sensory change for an action in context
    pub fn predict(&self, action: &ActionDescriptor, context: &ContextDescriptor) -> Option<PredictedChange> {
        let contingencies = self.contingencies.get(&action.action_type)?;

        // Find best matching contingency
        let best = contingencies.iter()
            .filter(|c| c.context.similarity(context) >= self.config.context_match_threshold)
            .max_by(|a, b| {
                let sim_a = a.context.similarity(context) * a.confidence;
                let sim_b = b.context.similarity(context) * b.confidence;
                sim_a.partial_cmp(&sim_b).unwrap()
            })?;

        Some(best.predict())
    }

    /// Get all contingencies for an action type
    pub fn get_contingencies(&self, action_type: ActionType) -> Option<&[SensorimotorContingency]> {
        self.contingencies.get(&action_type).map(|v| v.as_slice())
    }

    /// Get statistics
    pub fn stats(&self) -> &LearnerStats {
        &self.stats
    }

    /// Get prediction accuracy (0.0 to 1.0)
    pub fn prediction_accuracy(&self) -> f64 {
        if self.stats.predictions_made == 0 {
            return 0.5;
        }
        self.stats.accurate_predictions as f64 / self.stats.predictions_made as f64
    }

    /// Get current surprise level
    pub fn current_surprise(&self) -> f64 {
        self.stats.current_surprise
    }

    /// Get total number of learned contingencies
    pub fn contingency_count(&self) -> usize {
        self.stats.total_contingencies
    }
}

impl Default for ContingencyLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a learning step
#[derive(Debug, Clone)]
pub struct LearnResult {
    /// How surprising was the outcome?
    pub surprise: f64,

    /// Was this a contingency violation?
    pub is_violation: bool,

    /// Prediction error (if prediction was made)
    pub prediction_error: Option<f64>,

    /// ID of the contingency that was updated/created
    pub contingency_updated: Option<String>,
}

// ============================================================================
// ENACTIVIST PERCEPTION - Perception as Contingency Knowledge
// ============================================================================

/// Enactivist perception: perception IS implicit knowledge of contingencies
///
/// "Seeing red" is not having a red representation - it is:
/// - Knowing red surfaces look darker in shadow
/// - Knowing red shifts toward yellow in incandescent light
/// - Knowing red things reflect more at 700nm
/// - Having motor programs ready for red-related actions
///
/// This struct implements perception as contingency mastery.
#[derive(Debug, Clone)]
pub struct EnactivistPerception {
    /// The contingency learner providing the knowledge base
    learner: ContingencyLearner,

    /// Current perceptual state (what contingencies are "active")
    active_contingencies: Vec<String>,

    /// Perceptual readiness (what actions are primed)
    motor_readiness: HashMap<ActionType, f64>,

    /// Configuration
    config: PerceptionConfig,

    /// Statistics
    stats: PerceptionStats,
}

/// Configuration for enactivist perception
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionConfig {
    /// Number of contingencies to keep active
    pub max_active_contingencies: usize,

    /// Decay rate for motor readiness
    pub readiness_decay: f64,

    /// Threshold for contingency activation
    pub activation_threshold: f64,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            max_active_contingencies: 20,
            readiness_decay: 0.05,
            activation_threshold: 0.6,
        }
    }
}

/// Statistics for enactivist perception
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerceptionStats {
    /// Total perceptual cycles
    pub total_cycles: u64,

    /// Active contingencies over time (moving average)
    pub avg_active_contingencies: f64,

    /// Motor readiness (how ready to act)
    pub total_readiness: f64,

    /// Perceptual mastery (contingency coverage)
    pub mastery: f64,
}

impl EnactivistPerception {
    /// Create new enactivist perception system
    pub fn new() -> Self {
        Self::with_config(PerceptionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PerceptionConfig) -> Self {
        Self {
            learner: ContingencyLearner::new(),
            active_contingencies: Vec::new(),
            motor_readiness: HashMap::new(),
            config,
            stats: PerceptionStats::default(),
        }
    }

    /// Perceive through action: take an action and process the outcome
    ///
    /// This is the core enactivist loop:
    /// 1. Action is taken
    /// 2. Sensory change is observed
    /// 3. Contingency is learned/updated
    /// 4. Motor readiness is updated
    /// 5. Perception emerges from contingency knowledge
    pub fn perceive_through_action(
        &mut self,
        action: ActionDescriptor,
        context: ContextDescriptor,
        outcome: SensoryChange,
    ) -> PerceptionResult {
        self.stats.total_cycles += 1;

        // Learn from this action-outcome pair
        let learn_result = self.learner.learn(action.clone(), context.clone(), outcome.clone());

        // Update active contingencies
        self.update_active_contingencies(&action.action_type, &context);

        // Update motor readiness
        self.update_motor_readiness(&action.action_type, learn_result.surprise);

        // Calculate perceptual clarity (mastery of current context)
        let clarity = self.calculate_perceptual_clarity(&context);

        // Update stats
        self.stats.avg_active_contingencies =
            self.stats.avg_active_contingencies * 0.95
            + self.active_contingencies.len() as f64 * 0.05;
        self.stats.total_readiness = self.motor_readiness.values().sum();
        self.stats.mastery = self.learner.prediction_accuracy();

        PerceptionResult {
            clarity,
            surprise: learn_result.surprise,
            is_violation: learn_result.is_violation,
            active_contingencies: self.active_contingencies.len(),
            ready_actions: self.get_ready_actions(),
            prediction_error: learn_result.prediction_error,
        }
    }

    /// Update which contingencies are currently active
    fn update_active_contingencies(&mut self, action_type: &ActionType, context: &ContextDescriptor) {
        // Get relevant contingencies
        if let Some(contingencies) = self.learner.get_contingencies(*action_type) {
            self.active_contingencies = contingencies.iter()
                .filter(|c| {
                    c.confidence >= self.config.activation_threshold
                        && c.context.similarity(context) > 0.5
                })
                .take(self.config.max_active_contingencies)
                .map(|c| c.id.clone())
                .collect();
        }
    }

    /// Update motor readiness based on experience
    fn update_motor_readiness(&mut self, action_type: &ActionType, surprise: f64) {
        // Decay all readiness
        for value in self.motor_readiness.values_mut() {
            *value *= 1.0 - self.config.readiness_decay;
        }

        // Boost readiness for the performed action (successful prediction = more ready)
        let boost = 1.0 - surprise;
        let entry = self.motor_readiness.entry(*action_type).or_insert(0.0);
        *entry = (*entry + boost * 0.2).min(1.0);

        // Also boost related actions
        let related = self.get_related_actions(*action_type);
        for related_action in related {
            let entry = self.motor_readiness.entry(related_action).or_insert(0.0);
            *entry = (*entry + boost * 0.1).min(1.0);
        }
    }

    /// Get actions related to a given action
    fn get_related_actions(&self, action: ActionType) -> Vec<ActionType> {
        match action {
            ActionType::SaccadeLeft => vec![ActionType::SaccadeRight, ActionType::Fixate],
            ActionType::SaccadeRight => vec![ActionType::SaccadeLeft, ActionType::Fixate],
            ActionType::HeadTurnLeft => vec![ActionType::HeadTurnRight],
            ActionType::HeadTurnRight => vec![ActionType::HeadTurnLeft],
            ActionType::Reach => vec![ActionType::Grasp, ActionType::Touch],
            ActionType::Grasp => vec![ActionType::Release, ActionType::Reach],
            ActionType::Push => vec![ActionType::Pull],
            ActionType::Pull => vec![ActionType::Push],
            _ => vec![],
        }
    }

    /// Calculate perceptual clarity (how well we know the current context)
    fn calculate_perceptual_clarity(&self, context: &ContextDescriptor) -> f64 {
        // Clarity = coverage of contingencies for this context
        let mut total_confidence = 0.0;
        let mut count = 0.0;

        for contingencies in self.learner.contingencies.values() {
            for c in contingencies {
                if c.context.similarity(context) > 0.5 {
                    total_confidence += c.confidence;
                    count += 1.0;
                }
            }
        }

        if count > 0.0 {
            (total_confidence / count) * (count / 10.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Get actions that are ready to be executed
    pub fn get_ready_actions(&self) -> Vec<(ActionType, f64)> {
        let mut actions: Vec<_> = self.motor_readiness.iter()
            .filter(|(_, &readiness)| readiness > 0.3)
            .map(|(&action, &readiness)| (action, readiness))
            .collect();

        actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        actions
    }

    /// Predict what would happen if we took an action
    pub fn imagine_action(&self, action: &ActionDescriptor, context: &ContextDescriptor) -> Option<PredictedChange> {
        self.learner.predict(action, context)
    }

    /// Get the contingency learner
    pub fn learner(&self) -> &ContingencyLearner {
        &self.learner
    }

    /// Get mutable access to learner
    pub fn learner_mut(&mut self) -> &mut ContingencyLearner {
        &mut self.learner
    }

    /// Get statistics
    pub fn stats(&self) -> &PerceptionStats {
        &self.stats
    }

    /// Get perceptual mastery (overall contingency mastery)
    pub fn mastery(&self) -> f64 {
        self.stats.mastery
    }
}

impl Default for EnactivistPerception {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a perceptual cycle
#[derive(Debug, Clone)]
pub struct PerceptionResult {
    /// Perceptual clarity (0.0 to 1.0)
    pub clarity: f64,

    /// How surprising was the outcome
    pub surprise: f64,

    /// Was this a contingency violation
    pub is_violation: bool,

    /// Number of active contingencies
    pub active_contingencies: usize,

    /// Actions ready to execute
    pub ready_actions: Vec<(ActionType, f64)>,

    /// Prediction error if prediction was made
    pub prediction_error: Option<f64>,
}

// ============================================================================
// ACTION AFFORDANCES - Gibson-Style Possibilities
// ============================================================================

/// Gibson-style affordances: action possibilities the environment offers
///
/// Affordances are NOT properties of objects alone, but relations between:
/// - Object properties
/// - Agent capabilities
/// - Context
/// - Learned contingencies
///
/// A chair affords sitting to humans, but not to ants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionAffordance {
    /// What action is afforded
    pub action: ActionType,

    /// What object/feature affords it
    pub source: String,

    /// How salient/obvious is this affordance (0.0 to 1.0)
    pub salience: f64,

    /// Predicted outcome if action is taken
    pub predicted_outcome: Option<SensoryChange>,

    /// Confidence in this affordance
    pub confidence: f64,

    /// Is this affordance currently reachable?
    pub reachable: bool,

    /// Effort required (0.0 = easy, 1.0 = maximum effort)
    pub effort: f64,

    /// Risk level (0.0 = safe, 1.0 = dangerous)
    pub risk: f64,
}

impl ActionAffordance {
    /// Calculate net attractiveness of this affordance
    pub fn attractiveness(&self) -> f64 {
        let value = self.salience * self.confidence;
        let cost = self.effort * 0.3 + self.risk * 0.5;
        let reachability = if self.reachable { 1.0 } else { 0.3 };

        (value - cost) * reachability
    }
}

/// Detects affordances based on context and contingency knowledge
#[derive(Debug, Clone)]
pub struct AffordanceDetector {
    /// The perception system providing contingency knowledge
    perception: EnactivistPerception,

    /// Currently detected affordances
    current_affordances: Vec<ActionAffordance>,

    /// Configuration
    config: AffordanceConfig,
}

/// Configuration for affordance detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffordanceConfig {
    /// Maximum affordances to track
    pub max_affordances: usize,

    /// Minimum confidence for affordance detection
    pub min_confidence: f64,

    /// Salience threshold for including affordance
    pub salience_threshold: f64,
}

impl Default for AffordanceConfig {
    fn default() -> Self {
        Self {
            max_affordances: 10,
            min_confidence: 0.4,
            salience_threshold: 0.3,
        }
    }
}

impl AffordanceDetector {
    /// Create new affordance detector
    pub fn new() -> Self {
        Self::with_config(AffordanceConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AffordanceConfig) -> Self {
        Self {
            perception: EnactivistPerception::new(),
            current_affordances: Vec::new(),
            config,
        }
    }

    /// Create with existing perception system
    pub fn with_perception(perception: EnactivistPerception) -> Self {
        Self {
            perception,
            current_affordances: Vec::new(),
            config: AffordanceConfig::default(),
        }
    }

    /// Detect affordances for the current context
    pub fn detect(&mut self, context: &ContextDescriptor) -> Vec<ActionAffordance> {
        self.current_affordances.clear();

        // Get ready actions from perception
        let ready_actions = self.perception.get_ready_actions();

        // For each ready action, check if we have contingencies in this context
        for (action_type, readiness) in ready_actions {
            let action = ActionDescriptor::new(action_type);

            if let Some(prediction) = self.perception.imagine_action(&action, context) {
                if prediction.confidence >= self.config.min_confidence {
                    let affordance = ActionAffordance {
                        action: action_type,
                        source: context.id.clone(),
                        salience: readiness * prediction.confidence,
                        predicted_outcome: Some(prediction.expected),
                        confidence: prediction.confidence,
                        reachable: true,
                        effort: 0.3,  // Could be computed from action parameters
                        risk: prediction.variance,  // Higher variance = higher risk
                    };

                    if affordance.salience >= self.config.salience_threshold {
                        self.current_affordances.push(affordance);
                    }
                }
            }
        }

        // Sort by attractiveness
        self.current_affordances.sort_by(|a, b|
            b.attractiveness().partial_cmp(&a.attractiveness()).unwrap());

        // Limit
        self.current_affordances.truncate(self.config.max_affordances);

        self.current_affordances.clone()
    }

    /// Get the most attractive affordance
    pub fn most_attractive(&self) -> Option<&ActionAffordance> {
        self.current_affordances.first()
    }

    /// Get all current affordances
    pub fn affordances(&self) -> &[ActionAffordance] {
        &self.current_affordances
    }

    /// Process an action and update the system
    pub fn process_action(
        &mut self,
        action: ActionDescriptor,
        context: ContextDescriptor,
        outcome: SensoryChange,
    ) -> PerceptionResult {
        self.perception.perceive_through_action(action, context, outcome)
    }

    /// Get the underlying perception system
    pub fn perception(&self) -> &EnactivistPerception {
        &self.perception
    }

    /// Get mutable perception system
    pub fn perception_mut(&mut self) -> &mut EnactivistPerception {
        &mut self.perception
    }
}

impl Default for AffordanceDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// INTEGRATION: Consciousness Contribution
// ============================================================================

/// Contribution to the consciousness equation from SMC mastery
///
/// Enactivist insight: Consciousness requires MASTERY of contingencies.
/// A being that perfectly predicts all sensory consequences of its actions
/// has higher conscious awareness than one with poor prediction.
#[derive(Debug, Clone)]
pub struct ContingencyConsciousnessContribution {
    /// Overall mastery level (0.0 to 1.0)
    pub mastery: f64,

    /// Prediction accuracy
    pub prediction_accuracy: f64,

    /// Current surprise/violation level
    pub surprise: f64,

    /// Number of learned contingencies
    pub contingency_count: usize,

    /// Motor readiness (action preparedness)
    pub motor_readiness: f64,

    /// Perceptual clarity
    pub perceptual_clarity: f64,
}

impl ContingencyConsciousnessContribution {
    /// Compute from an affordance detector
    pub fn from_detector(detector: &AffordanceDetector, context: &ContextDescriptor) -> Self {
        let perception = detector.perception();
        let learner = perception.learner();

        // Calculate perceptual clarity for current context
        let mut clarity = 0.0;
        let mut count = 0.0;
        for contingencies in learner.contingencies.values() {
            for c in contingencies {
                if c.context.similarity(context) > 0.5 {
                    clarity += c.confidence;
                    count += 1.0;
                }
            }
        }
        let perceptual_clarity = if count > 0.0 {
            (clarity / count) * (count / 10.0).min(1.0)
        } else {
            0.0
        };

        Self {
            mastery: learner.prediction_accuracy(),
            prediction_accuracy: learner.prediction_accuracy(),
            surprise: learner.current_surprise(),
            contingency_count: learner.contingency_count(),
            motor_readiness: perception.stats().total_readiness,
            perceptual_clarity,
        }
    }

    /// Compute consciousness contribution (for Master Consciousness Equation)
    ///
    /// M_smc = mastery × (1 - surprise) × (1 + motor_readiness × 0.2)
    pub fn consciousness_contribution(&self) -> f64 {
        let base = self.mastery * (1.0 - self.surprise);
        let motor_boost = 1.0 + self.motor_readiness.min(1.0) * 0.2;
        (base * motor_boost).clamp(0.0, 1.0)
    }

    /// Get prediction error for feeding to free energy minimization
    pub fn prediction_error(&self) -> f64 {
        self.surprise
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contingency_hv_operations() {
        let a = ContingencyHV::from_id("action_reach");
        let b = ContingencyHV::from_id("context_object");

        // Binding should create unique combination
        let bound = a.bind(&b);
        assert!(a.similarity(&bound) < 0.7);
        assert!(b.similarity(&bound) < 0.7);

        // Self-similarity should be 1.0
        assert!((a.similarity(&a) - 1.0).abs() < 0.001);

        // Unbinding should recover
        let recovered = bound.bind(&b);
        assert!(a.similarity(&recovered) > 0.99);
    }

    #[test]
    fn test_contingency_learning() {
        let mut learner = ContingencyLearner::new();

        let action = ActionDescriptor::new(ActionType::Reach)
            .with_parameter("distance", 0.5);
        let context = ContextDescriptor::new("test_context", SensoryModality::Visual)
            .with_feature("object_present", 1.0);
        let outcome = SensoryChange::new(SensoryModality::Tactile)
            .with_vector(vec![0.8, 0.2, 0.0, 0.0])
            .with_description("contact made");

        // Learn the contingency
        let result = learner.learn(action.clone(), context.clone(), outcome.clone());

        assert!(result.contingency_updated.is_some());
        assert_eq!(learner.contingency_count(), 1);

        // Predict should now work
        let prediction = learner.predict(&action, &context);
        assert!(prediction.is_some());

        // Learn same thing again - should increase confidence
        let _ = learner.learn(action.clone(), context.clone(), outcome.clone());
        let _ = learner.learn(action.clone(), context.clone(), outcome.clone());

        let prediction2 = learner.predict(&action, &context).unwrap();
        assert!(prediction2.confidence > 0.5);
    }

    #[test]
    fn test_surprise_detection() {
        let mut learner = ContingencyLearner::new();

        let action = ActionDescriptor::new(ActionType::Push);
        let context = ContextDescriptor::new("normal_object", SensoryModality::Tactile);

        // Learn normal outcome
        let normal_outcome = SensoryChange::new(SensoryModality::Visual)
            .with_vector(vec![0.5, 0.0, 0.0, 0.0]);

        for _ in 0..5 {
            learner.learn(action.clone(), context.clone(), normal_outcome.clone());
        }

        // Now give surprising outcome
        let surprising_outcome = SensoryChange::new(SensoryModality::Visual)
            .with_vector(vec![-0.5, 1.0, 0.0, 0.0]);  // Opposite!

        let result = learner.learn(action, context, surprising_outcome);

        assert!(result.surprise > 0.3, "Outcome should be surprising");
    }

    #[test]
    fn test_enactivist_perception() {
        let mut perception = EnactivistPerception::new();

        let action = ActionDescriptor::new(ActionType::SaccadeLeft);
        let context = ContextDescriptor::new("visual_scene", SensoryModality::Visual);
        let outcome = SensoryChange::new(SensoryModality::Visual)
            .with_vector(vec![0.3, 0.0, 0.0, 0.0])
            .with_description("scene shifted right");

        // Perceive through action
        let result = perception.perceive_through_action(action.clone(), context.clone(), outcome);

        assert!(result.clarity >= 0.0);

        // After learning, motor readiness should increase
        for _ in 0..5 {
            perception.perceive_through_action(
                action.clone(),
                context.clone(),
                SensoryChange::new(SensoryModality::Visual).with_vector(vec![0.3, 0.0, 0.0, 0.0])
            );
        }

        let ready = perception.get_ready_actions();
        assert!(!ready.is_empty());
    }

    #[test]
    fn test_affordance_detection() {
        let mut detector = AffordanceDetector::new();

        let action = ActionDescriptor::new(ActionType::Grasp);
        let context = ContextDescriptor::new("graspable_object", SensoryModality::Visual)
            .with_feature("size", 0.3)
            .with_object("cup");
        let outcome = SensoryChange::new(SensoryModality::Tactile)
            .with_vector(vec![0.9, 0.5, 0.0, 0.0]);

        // Train the system
        for _ in 0..10 {
            detector.process_action(action.clone(), context.clone(), outcome.clone());
        }

        // Detect affordances
        let affordances = detector.detect(&context);

        // Should detect grasp affordance
        let grasp_affordance = affordances.iter()
            .find(|a| a.action == ActionType::Grasp);
        assert!(grasp_affordance.is_some());
    }

    #[test]
    fn test_consciousness_contribution() {
        let mut detector = AffordanceDetector::new();

        let action = ActionDescriptor::new(ActionType::Touch);
        let context = ContextDescriptor::new("test", SensoryModality::Tactile);
        let outcome = SensoryChange::new(SensoryModality::Tactile)
            .with_vector(vec![0.7, 0.0, 0.0, 0.0]);

        // Train
        for _ in 0..20 {
            detector.process_action(action.clone(), context.clone(), outcome.clone());
        }

        let contribution = ContingencyConsciousnessContribution::from_detector(&detector, &context);

        assert!(contribution.mastery > 0.0);
        assert!(contribution.consciousness_contribution() > 0.0);
    }

    #[test]
    fn test_sensory_change_distance() {
        let a = SensoryChange::new(SensoryModality::Visual)
            .with_vector(vec![1.0, 0.0, 0.0, 0.0]);
        let b = SensoryChange::new(SensoryModality::Visual)
            .with_vector(vec![0.0, 1.0, 0.0, 0.0]);
        let c = SensoryChange::new(SensoryModality::Visual)
            .with_vector(vec![1.0, 0.0, 0.0, 0.0]);

        // Same vectors should have 0 distance
        assert!(a.distance(&c) < 0.001);

        // Different vectors should have non-zero distance
        assert!(a.distance(&b) > 0.5);

        // Different modalities = max distance
        let d = SensoryChange::new(SensoryModality::Auditory)
            .with_vector(vec![1.0, 0.0, 0.0, 0.0]);
        assert!((a.distance(&d) - 1.0).abs() < 0.001);
    }
}
