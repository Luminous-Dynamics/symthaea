// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Enactivist Sensorimotor Contingencies (Core Implementation)
//!
//! This is the symthaea-core implementation of sensorimotor contingency theory.
//! Perception is NOT passive reception - it is ACTIVE knowledge of how actions
//! affect sensation (O'Regan & Noe, 2001).
//!
//! This module integrates with the existing BinaryHV hypervector infrastructure
//! for efficient contingency encoding and lookup.

use super::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ============================================================================
// SENSORIMOTOR CONTINGENCY - Core Data Structure
// ============================================================================

/// A sensorimotor contingency: the lawful relation between action and sensation
///
/// This is the fundamental unit of enactivist perception:
/// "If I do X in context C, sensory input will change by Y"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorimotorContingency {
    /// Unique identifier
    pub id: String,

    /// The action that causes the sensory change
    pub action: ActionDescriptor,

    /// The context in which this contingency holds
    pub context: ContextDescriptor,

    /// Expected sensory change
    pub expected_change: SensoryChange,

    /// HDC encoding for efficient similarity-based lookup
    pub encoding: BinaryHV,

    /// Confidence (0.0 to 1.0)
    pub confidence: f64,

    /// Number of observations
    pub observation_count: u64,

    /// Outcome variance (uncertainty)
    pub outcome_variance: f64,

    /// Last activation time
    #[serde(skip, default = "Instant::now")]
    pub last_activated: Instant,
}

impl SensorimotorContingency {
    /// Create new contingency from action-outcome pair
    pub fn new(
        action: ActionDescriptor,
        context: ContextDescriptor,
        expected_change: SensoryChange,
    ) -> Self {
        // Generate HDC encoding by binding action and context encodings
        let encoding = action.encoding.bind(&context.encoding);

        let id = format!("smc_{:?}_{}", action.action_type, rand::random::<u32>());

        Self {
            id,
            action,
            context,
            expected_change,
            encoding,
            confidence: 0.5,
            observation_count: 1,
            outcome_variance: 0.3,
            last_activated: Instant::now(),
        }
    }

    /// Update contingency with new observation
    pub fn update_with_observation(&mut self, actual_change: &SensoryChange) {
        self.observation_count += 1;
        self.last_activated = Instant::now();

        // Calculate prediction error
        let error = self.expected_change.distance(actual_change);

        // Update confidence
        let accuracy = 1.0 - error.min(1.0);
        let lr = 0.1;
        self.confidence = self.confidence * (1.0 - lr) + accuracy * lr;

        // Update expected change
        self.expected_change.update_toward(actual_change, lr);

        // Update variance
        self.outcome_variance = self.outcome_variance * 0.9 + error * error * 0.1;
    }

    /// Check if matches action-context pair
    pub fn matches(&self, action: &ActionDescriptor, context: &ContextDescriptor) -> bool {
        self.action.action_type == action.action_type && self.context.similarity(context) > 0.7
    }

    /// Predict sensory change
    pub fn predict(&self) -> PredictedChange {
        PredictedChange {
            expected: self.expected_change.clone(),
            confidence: self.confidence,
            variance: self.outcome_variance,
        }
    }
}

/// Action descriptor for sensorimotor vocabulary
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionDescriptor {
    /// Type of action
    pub action_type: ActionType,

    /// Parameters (e.g., magnitude, direction)
    pub parameters: HashMap<String, f64>,

    /// Effectors involved
    pub effectors: Vec<String>,

    /// HDC encoding
    pub encoding: BinaryHV,
}

impl ActionDescriptor {
    /// Create new action descriptor
    pub fn new(action_type: ActionType) -> Self {
        // Generate encoding from action type
        let seed = action_type as u64 * 12345;
        let encoding = BinaryHV::random(seed);

        Self {
            action_type,
            parameters: HashMap::new(),
            effectors: Vec::new(),
            encoding,
        }
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        // Update encoding to incorporate parameter
        let param_hv = BinaryHV::random((value * 1000.0) as u64);
        self.encoding = self.encoding.bind(&param_hv);
        self
    }

    /// Add effector
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

    // Custom
    Custom,
}

/// Context descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextDescriptor {
    /// Unique identifier
    pub id: String,

    /// Semantic features
    pub features: HashMap<String, f64>,

    /// Objects present
    pub objects: Vec<String>,

    /// Modality
    pub modality: SensoryModality,

    /// HDC encoding
    pub encoding: BinaryHV,
}

impl ContextDescriptor {
    /// Create new context descriptor
    pub fn new(id: &str, modality: SensoryModality) -> Self {
        // Generate encoding from id
        let mut seed: u64 = 0;
        for (i, c) in id.bytes().enumerate() {
            seed = seed.wrapping_add((c as u64) << (i % 8 * 8));
        }

        Self {
            id: id.to_string(),
            features: HashMap::new(),
            objects: Vec::new(),
            modality,
            encoding: BinaryHV::random(seed),
        }
    }

    /// Add feature
    pub fn with_feature(mut self, key: &str, value: f64) -> Self {
        self.features.insert(key.to_string(), value);
        self
    }

    /// Add object
    pub fn with_object(mut self, object: &str) -> Self {
        self.objects.push(object.to_string());
        self
    }

    /// Compute similarity to another context
    pub fn similarity(&self, other: &ContextDescriptor) -> f64 {
        // Use HDC similarity
        let hdc_sim = self.encoding.similarity(&other.encoding) as f64;

        // Also consider feature overlap
        let mut feature_sim = 0.0;
        let mut count = 0.0;
        for (key, &val) in &self.features {
            if let Some(&other_val) = other.features.get(key) {
                feature_sim += 1.0 - (val - other_val).abs().min(1.0);
                count += 1.0;
            }
        }
        let feature_sim = if count > 0.0 {
            feature_sim / count
        } else {
            0.5
        };

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

/// Sensory change description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryChange {
    /// Modality
    pub modality: SensoryModality,

    /// Change vector (encoded as BinaryHV for efficiency)
    pub encoding: BinaryHV,

    /// Scalar magnitude
    pub magnitude: f64,

    /// Onset delay (ms)
    pub onset_delay_ms: u64,

    /// Duration (ms)
    pub duration_ms: u64,

    /// Description
    pub description: String,
}

impl SensoryChange {
    /// Create new sensory change
    pub fn new(modality: SensoryModality) -> Self {
        Self {
            modality,
            encoding: BinaryHV::random(0),
            magnitude: 0.0,
            onset_delay_ms: 50,
            duration_ms: 200,
            description: String::new(),
        }
    }

    /// Set encoding from change vector
    pub fn with_vector(mut self, vector: Vec<f64>) -> Self {
        // Encode vector as BinaryHV by quantizing and combining position-encoded components
        let mut components: Vec<BinaryHV> = Vec::new();
        for (i, &val) in vector.iter().enumerate() {
            let position_hv = BinaryHV::random(i as u64 * 9999);
            let value_hv = BinaryHV::random((val * 10000.0) as u64);
            components.push(position_hv.bind(&value_hv));
        }
        self.encoding = if components.is_empty() {
            BinaryHV::random(0)
        } else {
            BinaryHV::bundle(&components)
        };
        self.magnitude = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Compute distance to another sensory change
    pub fn distance(&self, other: &SensoryChange) -> f64 {
        if self.modality != other.modality {
            return 1.0;
        }

        // Use HDC similarity (inverted)
        let sim = self.encoding.similarity(&other.encoding) as f64;
        1.0 - sim
    }

    /// Update toward another sensory change
    pub fn update_toward(&mut self, target: &SensoryChange, rate: f64) {
        // Blend encodings using bundle
        if rate > 0.5 {
            // More weight to target - bundle both
            self.encoding = BinaryHV::bundle(&[target.encoding, self.encoding]);
        }
        // For low rates, keep current encoding (no HV interpolation in binary)

        self.magnitude = self.magnitude * (1.0 - rate) + target.magnitude * rate;
    }
}

/// Predicted sensory change with uncertainty
#[derive(Debug, Clone)]
pub struct PredictedChange {
    pub expected: SensoryChange,
    pub confidence: f64,
    pub variance: f64,
}

// ============================================================================
// CONTINGENCY LEARNER
// ============================================================================

/// Learns sensorimotor contingencies from experience
#[derive(Debug, Clone)]
pub struct ContingencyLearner {
    /// Contingencies by action type
    contingencies: HashMap<ActionType, Vec<SensorimotorContingency>>,

    /// Recent experiences
    recent_experiences: VecDeque<Experience>,

    /// Configuration
    config: LearnerConfig,

    /// Statistics
    stats: LearnerStats,
}

/// A single experience (action-outcome pair)
#[derive(Debug, Clone)]
pub struct Experience {
    pub action: ActionDescriptor,
    pub context: ContextDescriptor,
    pub outcome: SensoryChange,
    pub timestamp: Instant,
    pub surprise: f64,
}

/// Learner configuration
#[derive(Debug, Clone)]
pub struct LearnerConfig {
    pub max_experiences: usize,
    pub min_observations: u64,
    pub learning_rate: f64,
    pub surprise_threshold: f64,
    pub context_match_threshold: f64,
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

/// Learner statistics
#[derive(Debug, Clone, Default)]
pub struct LearnerStats {
    pub total_experiences: u64,
    pub total_contingencies: usize,
    pub predictions_made: u64,
    pub accurate_predictions: u64,
    pub violations_detected: u64,
    pub avg_prediction_error: f64,
    pub current_surprise: f64,
}

impl ContingencyLearner {
    /// Create new learner
    pub fn new() -> Self {
        Self::with_config(LearnerConfig::default())
    }

    /// Create with config
    pub fn with_config(config: LearnerConfig) -> Self {
        Self {
            contingencies: HashMap::new(),
            recent_experiences: VecDeque::with_capacity(config.max_experiences),
            config,
            stats: LearnerStats::default(),
        }
    }

    /// Learn from experience
    pub fn learn(
        &mut self,
        action: ActionDescriptor,
        context: ContextDescriptor,
        outcome: SensoryChange,
    ) -> LearnResult {
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

            self.stats.avg_prediction_error = self.stats.avg_prediction_error * 0.99 + error * 0.01;

            error
        } else {
            0.5
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
        let contingency = self.update_or_create_contingency(action, context, outcome);

        LearnResult {
            surprise,
            is_violation,
            prediction_error: prediction.map(|p| p.expected.distance(&contingency.expected_change)),
            contingency_id: Some(contingency.id),
        }
    }

    fn update_or_create_contingency(
        &mut self,
        action: ActionDescriptor,
        context: ContextDescriptor,
        outcome: SensoryChange,
    ) -> SensorimotorContingency {
        let contingencies = self.contingencies.entry(action.action_type).or_default();

        // Find matching
        let matching = contingencies
            .iter_mut()
            .find(|c| c.matches(&action, &context));

        if let Some(c) = matching {
            c.update_with_observation(&outcome);
            c.clone()
        } else {
            let contingency = SensorimotorContingency::new(action.clone(), context, outcome);

            if contingencies.len() >= self.config.max_contingencies_per_action
                && let Some(idx) = contingencies
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.confidence.total_cmp(&b.confidence))
                    .map(|(i, _)| i)
            {
                contingencies.remove(idx);
            }

            contingencies.push(contingency.clone());
            self.stats.total_contingencies = self.contingencies.values().map(|v| v.len()).sum();

            contingency
        }
    }

    /// Predict sensory change
    pub fn predict(
        &self,
        action: &ActionDescriptor,
        context: &ContextDescriptor,
    ) -> Option<PredictedChange> {
        let contingencies = self.contingencies.get(&action.action_type)?;

        let best = contingencies
            .iter()
            .filter(|c| c.context.similarity(context) >= self.config.context_match_threshold)
            .max_by(|a, b| {
                let sim_a = a.context.similarity(context) * a.confidence;
                let sim_b = b.context.similarity(context) * b.confidence;
                sim_a.total_cmp(&sim_b)
            })?;

        Some(best.predict())
    }

    /// Get contingencies for action type
    pub fn get_contingencies(&self, action_type: ActionType) -> Option<&[SensorimotorContingency]> {
        self.contingencies.get(&action_type).map(|v| v.as_slice())
    }

    /// Get statistics
    pub fn stats(&self) -> &LearnerStats {
        &self.stats
    }

    /// Prediction accuracy
    pub fn prediction_accuracy(&self) -> f64 {
        if self.stats.predictions_made == 0 {
            return 0.5;
        }
        self.stats.accurate_predictions as f64 / self.stats.predictions_made as f64
    }

    /// Current surprise level
    pub fn current_surprise(&self) -> f64 {
        self.stats.current_surprise
    }

    /// Total contingencies
    pub fn contingency_count(&self) -> usize {
        self.stats.total_contingencies
    }
}

impl Default for ContingencyLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of learning step
#[derive(Debug, Clone)]
pub struct LearnResult {
    pub surprise: f64,
    pub is_violation: bool,
    pub prediction_error: Option<f64>,
    pub contingency_id: Option<String>,
}

// ============================================================================
// ENACTIVIST PERCEPTION
// ============================================================================

/// Enactivist perception: perception IS implicit knowledge of contingencies
#[derive(Debug, Clone)]
pub struct EnactivistPerception {
    /// Contingency learner
    learner: ContingencyLearner,

    /// Active contingencies
    active_contingencies: Vec<String>,

    /// Motor readiness
    motor_readiness: HashMap<ActionType, f64>,

    /// Configuration
    config: PerceptionConfig,

    /// Statistics
    stats: PerceptionStats,
}

/// Perception configuration
#[derive(Debug, Clone)]
pub struct PerceptionConfig {
    pub max_active_contingencies: usize,
    pub readiness_decay: f64,
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

/// Perception statistics
#[derive(Debug, Clone, Default)]
pub struct PerceptionStats {
    pub total_cycles: u64,
    pub avg_active_contingencies: f64,
    pub total_readiness: f64,
    pub mastery: f64,
}

impl EnactivistPerception {
    /// Create new
    pub fn new() -> Self {
        Self::with_config(PerceptionConfig::default())
    }

    /// Create with config
    pub fn with_config(config: PerceptionConfig) -> Self {
        Self {
            learner: ContingencyLearner::new(),
            active_contingencies: Vec::new(),
            motor_readiness: HashMap::new(),
            config,
            stats: PerceptionStats::default(),
        }
    }

    /// Perceive through action
    pub fn perceive_through_action(
        &mut self,
        action: ActionDescriptor,
        context: ContextDescriptor,
        outcome: SensoryChange,
    ) -> PerceptionResult {
        self.stats.total_cycles += 1;

        // Learn
        let learn_result = self.learner.learn(action.clone(), context.clone(), outcome);

        // Update active contingencies
        self.update_active_contingencies(&action.action_type, &context);

        // Update motor readiness
        self.update_motor_readiness(&action.action_type, learn_result.surprise);

        // Calculate clarity
        let clarity = self.calculate_clarity(&context);

        // Update stats
        self.stats.avg_active_contingencies = self.stats.avg_active_contingencies * 0.95
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

    fn update_active_contingencies(
        &mut self,
        action_type: &ActionType,
        context: &ContextDescriptor,
    ) {
        if let Some(contingencies) = self.learner.get_contingencies(*action_type) {
            self.active_contingencies = contingencies
                .iter()
                .filter(|c| {
                    c.confidence >= self.config.activation_threshold
                        && c.context.similarity(context) > 0.5
                })
                .take(self.config.max_active_contingencies)
                .map(|c| c.id.clone())
                .collect();
        }
    }

    fn update_motor_readiness(&mut self, action_type: &ActionType, surprise: f64) {
        for value in self.motor_readiness.values_mut() {
            *value *= 1.0 - self.config.readiness_decay;
        }

        let boost = 1.0 - surprise;
        let entry = self.motor_readiness.entry(*action_type).or_insert(0.0);
        *entry = (*entry + boost * 0.2).min(1.0);
    }

    fn calculate_clarity(&self, context: &ContextDescriptor) -> f64 {
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

    /// Get ready actions
    pub fn get_ready_actions(&self) -> Vec<(ActionType, f64)> {
        let mut actions: Vec<_> = self
            .motor_readiness
            .iter()
            .filter(|&(_, &r)| r > 0.3)
            .map(|(&a, &r)| (a, r))
            .collect();

        actions.sort_by(|a, b| b.1.total_cmp(&a.1));
        actions
    }

    /// Imagine action
    pub fn imagine_action(
        &self,
        action: &ActionDescriptor,
        context: &ContextDescriptor,
    ) -> Option<PredictedChange> {
        self.learner.predict(action, context)
    }

    /// Get learner
    pub fn learner(&self) -> &ContingencyLearner {
        &self.learner
    }

    /// Get stats
    pub fn stats(&self) -> &PerceptionStats {
        &self.stats
    }

    /// Mastery level
    pub fn mastery(&self) -> f64 {
        self.stats.mastery
    }
}

impl Default for EnactivistPerception {
    fn default() -> Self {
        Self::new()
    }
}

/// Perception result
#[derive(Debug, Clone)]
pub struct PerceptionResult {
    pub clarity: f64,
    pub surprise: f64,
    pub is_violation: bool,
    pub active_contingencies: usize,
    pub ready_actions: Vec<(ActionType, f64)>,
    pub prediction_error: Option<f64>,
}

// ============================================================================
// ACTION AFFORDANCES
// ============================================================================

/// Gibson-style affordance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionAffordance {
    pub action: ActionType,
    pub source: String,
    pub salience: f64,
    pub confidence: f64,
    pub reachable: bool,
    pub effort: f64,
    pub risk: f64,
}

impl ActionAffordance {
    /// Attractiveness
    pub fn attractiveness(&self) -> f64 {
        let value = self.salience * self.confidence;
        let cost = self.effort * 0.3 + self.risk * 0.5;
        let reach = if self.reachable { 1.0 } else { 0.3 };
        (value - cost) * reach
    }
}

/// Affordance detector
#[derive(Debug, Clone)]
pub struct AffordanceDetector {
    perception: EnactivistPerception,
    current_affordances: Vec<ActionAffordance>,
    config: AffordanceConfig,
}

/// Affordance config
#[derive(Debug, Clone)]
pub struct AffordanceConfig {
    pub max_affordances: usize,
    pub min_confidence: f64,
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
    /// Create new
    pub fn new() -> Self {
        Self {
            perception: EnactivistPerception::new(),
            current_affordances: Vec::new(),
            config: AffordanceConfig::default(),
        }
    }

    /// Detect affordances
    pub fn detect(&mut self, context: &ContextDescriptor) -> Vec<ActionAffordance> {
        self.current_affordances.clear();

        let ready_actions = self.perception.get_ready_actions();

        for (action_type, readiness) in ready_actions {
            let action = ActionDescriptor::new(action_type);

            if let Some(prediction) = self.perception.imagine_action(&action, context)
                && prediction.confidence >= self.config.min_confidence
            {
                let affordance = ActionAffordance {
                    action: action_type,
                    source: context.id.clone(),
                    salience: readiness * prediction.confidence,
                    confidence: prediction.confidence,
                    reachable: true,
                    effort: 0.3,
                    risk: prediction.variance,
                };

                if affordance.salience >= self.config.salience_threshold {
                    self.current_affordances.push(affordance);
                }
            }
        }

        self.current_affordances
            .sort_by(|a, b| b.attractiveness().total_cmp(&a.attractiveness()));

        self.current_affordances
            .truncate(self.config.max_affordances);

        self.current_affordances.clone()
    }

    /// Most attractive affordance
    pub fn most_attractive(&self) -> Option<&ActionAffordance> {
        self.current_affordances.first()
    }

    /// Process action
    pub fn process_action(
        &mut self,
        action: ActionDescriptor,
        context: ContextDescriptor,
        outcome: SensoryChange,
    ) -> PerceptionResult {
        self.perception
            .perceive_through_action(action, context, outcome)
    }

    /// Get perception
    pub fn perception(&self) -> &EnactivistPerception {
        &self.perception
    }
}

impl Default for AffordanceDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CONSCIOUSNESS INTEGRATION
// ============================================================================

/// Contribution to consciousness from SMC mastery
#[derive(Debug, Clone)]
pub struct ContingencyConsciousnessContribution {
    pub mastery: f64,
    pub prediction_accuracy: f64,
    pub surprise: f64,
    pub contingency_count: usize,
    pub motor_readiness: f64,
    pub perceptual_clarity: f64,
}

impl ContingencyConsciousnessContribution {
    /// Compute from detector
    pub fn from_detector(detector: &AffordanceDetector, context: &ContextDescriptor) -> Self {
        let perception = detector.perception();
        let learner = perception.learner();

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

    /// Consciousness contribution (for Master Equation)
    pub fn consciousness_contribution(&self) -> f64 {
        let base = self.mastery * (1.0 - self.surprise);
        let motor_boost = 1.0 + self.motor_readiness.min(1.0) * 0.2;
        (base * motor_boost).clamp(0.0, 1.0)
    }

    /// Prediction error for free energy
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
    fn test_contingency_learning() {
        let mut learner = ContingencyLearner::new();

        let action = ActionDescriptor::new(ActionType::Reach);
        let context = ContextDescriptor::new("test", SensoryModality::Visual);
        let outcome =
            SensoryChange::new(SensoryModality::Tactile).with_vector(vec![0.8, 0.2, 0.0, 0.0]);

        let result = learner.learn(action.clone(), context.clone(), outcome);

        assert!(result.contingency_id.is_some());
        assert_eq!(learner.contingency_count(), 1);

        // Learn more
        for _ in 0..5 {
            learner.learn(
                action.clone(),
                context.clone(),
                SensoryChange::new(SensoryModality::Tactile).with_vector(vec![0.8, 0.2, 0.0, 0.0]),
            );
        }

        assert!(learner.prediction_accuracy() > 0.5);
    }

    #[test]
    fn test_surprise_detection() {
        let mut learner = ContingencyLearner::new();

        let action = ActionDescriptor::new(ActionType::Push);
        let context = ContextDescriptor::new("ball", SensoryModality::Visual);
        let normal =
            SensoryChange::new(SensoryModality::Visual).with_vector(vec![0.5, 0.0, 0.0, 0.0]);

        for _ in 0..10 {
            learner.learn(action.clone(), context.clone(), normal.clone());
        }

        // Surprising outcome - use different modality for maximum surprise
        let surprising =
            SensoryChange::new(SensoryModality::Auditory).with_vector(vec![-0.5, 1.0, 0.0, 0.0]);

        let result = learner.learn(action, context, surprising);
        // With different modality, distance should be 1.0 (max surprise)
        assert!(
            result.surprise > 0.2,
            "Expected surprise > 0.2, got {}",
            result.surprise
        );
    }

    #[test]
    fn test_enactivist_perception() {
        let mut perception = EnactivistPerception::new();

        let action = ActionDescriptor::new(ActionType::SaccadeLeft);
        let context = ContextDescriptor::new("scene", SensoryModality::Visual);
        let outcome =
            SensoryChange::new(SensoryModality::Visual).with_vector(vec![0.3, 0.0, 0.0, 0.0]);

        for _ in 0..10 {
            perception.perceive_through_action(action.clone(), context.clone(), outcome.clone());
        }

        assert!(perception.mastery() > 0.0);
        assert!(!perception.get_ready_actions().is_empty());
    }

    #[test]
    fn test_affordance_detection() {
        let mut detector = AffordanceDetector::new();

        let action = ActionDescriptor::new(ActionType::Grasp);
        let context = ContextDescriptor::new("object", SensoryModality::Visual);
        let outcome =
            SensoryChange::new(SensoryModality::Tactile).with_vector(vec![0.9, 0.5, 0.0, 0.0]);

        for _ in 0..15 {
            detector.process_action(action.clone(), context.clone(), outcome.clone());
        }

        let affordances = detector.detect(&context);
        assert!(!affordances.is_empty());
    }

    #[test]
    fn test_consciousness_contribution() {
        let mut detector = AffordanceDetector::new();

        let action = ActionDescriptor::new(ActionType::Touch);
        let context = ContextDescriptor::new("test", SensoryModality::Tactile);
        let outcome =
            SensoryChange::new(SensoryModality::Tactile).with_vector(vec![0.7, 0.0, 0.0, 0.0]);

        for _ in 0..20 {
            detector.process_action(action.clone(), context.clone(), outcome.clone());
        }

        let contribution = ContingencyConsciousnessContribution::from_detector(&detector, &context);
        assert!(contribution.consciousness_contribution() > 0.0);
    }
}
