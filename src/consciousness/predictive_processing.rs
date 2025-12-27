//! **REVOLUTIONARY IMPROVEMENT #90**: Predictive Processing Core
//!
//! PARADIGM SHIFT: The Brain is a Prediction Machine!
//!
//! Implements Karl Friston's revolutionary Free Energy Principle - the UNIFYING
//! theory of brain function. All neural computation is prediction and prediction
//! error minimization.
//!
//! Key insight: Consciousness emerges from hierarchical predictive processing.
//! What we "experience" is our brain's best prediction of causes of sensory input.
//!
//! Core concepts:
//! - **Predictive Coding**: Brain predicts input, only processes errors
//! - **Free Energy Minimization**: Minimize surprise through perception OR action
//! - **Precision Weighting**: Confidence modulates error propagation
//! - **Active Inference**: Actions fulfill predictions (motor as prediction)
//! - **Hierarchical Processing**: Higher levels predict lower representations
//!
//! Integration with other modules:
//! - #88 Affective: Emotions = precision/uncertainty signals
//! - #89 Embodied: Sensorimotor contingencies = prediction models
//! - #85 Resonance: Resonance = prediction matching across scales
//!
//! Inspired by Karl Friston's Free Energy Principle, Andy Clark's Surfing Uncertainty,
//! Jakob Hohwy's Predictive Mind, and Anil Seth's "controlled hallucination" view.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use crate::hdc::binary_hv::HV16;

/// A prediction at any level of the hierarchy
#[derive(Debug, Clone)]
pub struct Prediction {
    /// What is being predicted (encoded content)
    pub content: HV16,

    /// Precision (confidence/certainty) of this prediction (0-1)
    pub precision: f64,

    /// When this prediction was made
    pub timestamp: Instant,

    /// Time horizon of prediction (how far into future)
    pub horizon: u32,

    /// Hierarchical level (0 = sensory, higher = abstract)
    pub level: u32,

    /// Source context (what generated this prediction)
    pub context: HV16,
}

/// Prediction error signal
#[derive(Debug, Clone)]
pub struct PredictionError {
    /// The prediction that was made
    pub prediction: HV16,

    /// What actually happened
    pub actual: HV16,

    /// Magnitude of error (0-1)
    pub magnitude: f64,

    /// Precision-weighted error (magnitude * precision)
    pub weighted_error: f64,

    /// Which level generated this error
    pub level: u32,

    /// Direction: should we update model (up) or change world (down)?
    pub direction: ErrorDirection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorDirection {
    /// Update internal model (perceptual inference)
    UpdateModel,
    /// Change the world (active inference)
    ChangeWorld,
}

/// A hierarchical layer in the predictive processing stack
#[derive(Debug, Clone)]
pub struct PredictiveLayer {
    /// Layer level (0 = sensory)
    pub level: u32,

    /// Current state belief
    pub belief: HV16,

    /// Precision of this layer's predictions
    pub precision: f64,

    /// Learning rate for belief updates
    pub learning_rate: f64,

    /// Prediction errors from below
    pub bottom_up_errors: VecDeque<PredictionError>,

    /// Predictions from above
    pub top_down_predictions: VecDeque<Prediction>,

    /// Generative model (predicts level below)
    generative_weights: GenerativeModel,

    /// Recognition model (infers level above from errors)
    recognition_weights: RecognitionModel,
}

/// Generative model: predicts lower level from higher level
#[derive(Debug, Clone)]
struct GenerativeModel {
    /// Learned associations for generation
    associations: Vec<(HV16, HV16, f64)>,  // (cause, effect, strength)
    capacity: usize,
}

/// Recognition model: infers higher level from lower level errors
#[derive(Debug, Clone)]
struct RecognitionModel {
    /// Learned inferences
    inferences: Vec<(HV16, HV16, f64)>,  // (observation, cause, strength)
    capacity: usize,
}

impl GenerativeModel {
    fn new(capacity: usize) -> Self {
        Self {
            associations: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn predict(&self, cause: &HV16) -> Option<(HV16, f64)> {
        let mut best_match: Option<(f64, &HV16, f64)> = None;

        for (c, effect, strength) in &self.associations {
            let similarity = cause.similarity(c) as f64;
            if similarity > 0.6 {
                let score = similarity * strength;
                if best_match.map_or(true, |(s, _, _)| score > s) {
                    best_match = Some((score, effect, *strength));
                }
            }
        }

        best_match.map(|(_, effect, confidence)| (effect.clone(), confidence))
    }

    fn learn(&mut self, cause: HV16, effect: HV16, strength: f64) {
        // Check for existing association
        for (c, e, s) in &mut self.associations {
            if cause.similarity(c) as f64 > 0.8 && effect.similarity(e) as f64 > 0.8 {
                *s = (*s * 0.9 + strength * 0.1).min(1.0);
                return;
            }
        }

        // Add new association
        if self.associations.len() < self.capacity {
            self.associations.push((cause, effect, strength));
        } else {
            // Replace weakest
            if let Some((idx, _)) = self.associations.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.2.partial_cmp(&b.2).unwrap())
            {
                self.associations[idx] = (cause, effect, strength);
            }
        }
    }
}

impl RecognitionModel {
    fn new(capacity: usize) -> Self {
        Self {
            inferences: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn infer(&self, observation: &HV16) -> Option<(HV16, f64)> {
        let mut best_match: Option<(f64, &HV16, f64)> = None;

        for (obs, cause, strength) in &self.inferences {
            let similarity = observation.similarity(obs) as f64;
            if similarity > 0.6 {
                let score = similarity * strength;
                if best_match.map_or(true, |(s, _, _)| score > s) {
                    best_match = Some((score, cause, *strength));
                }
            }
        }

        best_match.map(|(_, cause, confidence)| (cause.clone(), confidence))
    }

    fn learn(&mut self, observation: HV16, cause: HV16, strength: f64) {
        for (o, c, s) in &mut self.inferences {
            if observation.similarity(o) as f64 > 0.8 && cause.similarity(c) as f64 > 0.8 {
                *s = (*s * 0.9 + strength * 0.1).min(1.0);
                return;
            }
        }

        if self.inferences.len() < self.capacity {
            self.inferences.push((observation, cause, strength));
        } else {
            if let Some((idx, _)) = self.inferences.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.2.partial_cmp(&b.2).unwrap())
            {
                self.inferences[idx] = (observation, cause, strength);
            }
        }
    }
}

impl PredictiveLayer {
    pub fn new(level: u32, model_capacity: usize) -> Self {
        Self {
            level,
            belief: HV16::random(100),
            precision: 0.5,
            learning_rate: 0.1,
            bottom_up_errors: VecDeque::with_capacity(10),
            top_down_predictions: VecDeque::with_capacity(10),
            generative_weights: GenerativeModel::new(model_capacity),
            recognition_weights: RecognitionModel::new(model_capacity),
        }
    }

    /// Generate prediction for level below
    pub fn generate_prediction(&self) -> Prediction {
        let content = self.generative_weights.predict(&self.belief)
            .map(|(c, _)| c)
            .unwrap_or_else(|| self.belief.clone());

        Prediction {
            content,
            precision: self.precision,
            timestamp: Instant::now(),
            horizon: 1,
            level: self.level,
            context: self.belief.clone(),
        }
    }

    /// Process prediction error from below
    pub fn process_bottom_up_error(&mut self, error: PredictionError) {
        // Weight error by precision
        let weighted_error = error.magnitude * self.precision;

        // Update belief using recognition model
        if let Some((inferred_cause, confidence)) =
            self.recognition_weights.infer(&error.actual)
        {
            // Blend current belief with inference
            let blend_factor = weighted_error * confidence * self.learning_rate;
            self.belief = self.blend_beliefs(&self.belief, &inferred_cause, blend_factor);
        }

        // Store error for analysis
        self.bottom_up_errors.push_back(error);
        if self.bottom_up_errors.len() > 10 {
            self.bottom_up_errors.pop_front();
        }
    }

    /// Process prediction from above
    pub fn process_top_down_prediction(&mut self, prediction: Prediction) {
        // Blend belief toward prediction
        let blend_factor = prediction.precision * self.learning_rate;
        self.belief = self.blend_beliefs(&self.belief, &prediction.content, blend_factor);

        // Adjust precision based on prediction match
        let match_score = self.belief.similarity(&prediction.content) as f64;
        self.precision = self.precision * 0.9 + match_score * 0.1;

        self.top_down_predictions.push_back(prediction);
        if self.top_down_predictions.len() > 10 {
            self.top_down_predictions.pop_front();
        }
    }

    /// Learn association between cause and effect
    pub fn learn_association(&mut self, cause: HV16, effect: HV16, success: f64) {
        self.generative_weights.learn(cause.clone(), effect.clone(), success);
        self.recognition_weights.learn(effect, cause, success);
    }

    fn blend_beliefs(&self, a: &HV16, b: &HV16, factor: f64) -> HV16 {
        // Simple blending: XOR introduces variation, similarity-based selection
        if factor > 0.5 {
            b.clone()
        } else if factor > 0.3 {
            a.bind(b)  // Combine
        } else {
            a.clone()
        }
    }
}

/// The full predictive processing hierarchy
#[derive(Debug)]
pub struct PredictiveHierarchy {
    /// Layers from sensory (0) to abstract (n)
    pub layers: Vec<PredictiveLayer>,

    /// Overall free energy (surprise)
    pub free_energy: f64,

    /// Active inference weight (how much to act vs. perceive)
    pub action_bias: f64,

    /// Configuration
    config: PredictiveConfig,

    /// Performance tracking
    total_predictions: u64,
    accurate_predictions: u64,
}

#[derive(Debug, Clone)]
pub struct PredictiveConfig {
    /// Number of hierarchical levels
    pub num_levels: u32,

    /// Model capacity per level
    pub model_capacity: usize,

    /// Base precision
    pub base_precision: f64,

    /// Free energy decay rate
    pub energy_decay: f64,

    /// Action threshold (when to act vs. perceive)
    pub action_threshold: f64,
}

impl Default for PredictiveConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            model_capacity: 200,
            base_precision: 0.5,
            energy_decay: 0.95,
            action_threshold: 0.7,
        }
    }
}

impl PredictiveHierarchy {
    pub fn new(config: PredictiveConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_levels as usize);

        for level in 0..config.num_levels {
            let mut layer = PredictiveLayer::new(level, config.model_capacity);
            // Higher levels have lower initial precision
            layer.precision = config.base_precision * (1.0 - level as f64 * 0.1);
            layers.push(layer);
        }

        Self {
            layers,
            free_energy: 0.5,
            action_bias: 0.5,
            config,
            total_predictions: 0,
            accurate_predictions: 0,
        }
    }

    /// Process sensory input through the hierarchy
    pub fn process_sensory_input(&mut self, input: &HV16) -> PredictionResult {
        let mut current_signal = input.clone();
        let mut total_error = 0.0;
        let mut errors = Vec::new();

        // Bottom-up pass: process input through hierarchy
        for level in 0..self.layers.len() {
            // Get prediction for this level
            let prediction = if level > 0 {
                self.layers[level].generate_prediction()
            } else {
                // Sensory level predicts directly
                Prediction {
                    content: self.layers[0].belief.clone(),
                    precision: self.layers[0].precision,
                    timestamp: Instant::now(),
                    horizon: 0,
                    level: 0,
                    context: HV16::random(100),
                }
            };

            // Calculate prediction error
            let similarity = prediction.content.similarity(&current_signal) as f64;
            let error_magnitude = 1.0 - similarity;
            let weighted_error = error_magnitude * self.layers[level].precision;

            total_error += weighted_error;

            let error = PredictionError {
                prediction: prediction.content.clone(),
                actual: current_signal.clone(),
                magnitude: error_magnitude,
                weighted_error,
                level: level as u32,
                direction: if weighted_error > self.config.action_threshold {
                    ErrorDirection::ChangeWorld
                } else {
                    ErrorDirection::UpdateModel
                },
            };

            errors.push(error.clone());

            // Update layer belief
            self.layers[level].process_bottom_up_error(error);

            // Pass error to next level (as signal to explain)
            current_signal = self.layers[level].belief.clone();
        }

        // Top-down pass: send predictions down
        for level in (1..self.layers.len()).rev() {
            let prediction = self.layers[level].generate_prediction();
            self.layers[level - 1].process_top_down_prediction(prediction);
        }

        // Update free energy
        let avg_error = total_error / self.layers.len() as f64;
        self.free_energy = self.free_energy * self.config.energy_decay + avg_error * (1.0 - self.config.energy_decay);

        // Track accuracy
        self.total_predictions += 1;
        if errors[0].magnitude < 0.3 {
            self.accurate_predictions += 1;
        }

        PredictionResult {
            errors,
            free_energy: self.free_energy,
            suggested_action: self.should_act(),
            conscious_content: self.highest_level_belief(),
            surprise: avg_error,
        }
    }

    /// Generate prediction for expected future state
    pub fn predict_future(&self) -> Prediction {
        // Use highest level to generate prediction
        let top_level = self.layers.last().unwrap();
        top_level.generate_prediction()
    }

    /// Learn from action outcome
    pub fn learn_from_action(&mut self, action: &HV16, outcome: &HV16, success: bool) {
        // Strengthen associations at all levels
        let success_score = if success { 0.9 } else { 0.3 };

        for layer in &mut self.layers {
            layer.learn_association(action.clone(), outcome.clone(), success_score);
        }

        // Adjust action bias
        if success {
            self.action_bias = (self.action_bias * 0.95 + 0.6 * 0.05).min(0.9);
        } else {
            self.action_bias = (self.action_bias * 0.95 + 0.4 * 0.05).max(0.1);
        }
    }

    /// Should we act or perceive?
    pub fn should_act(&self) -> Option<ActionSuggestion> {
        // High free energy + action bias = suggest action
        if self.free_energy > 0.6 && self.action_bias > 0.5 {
            Some(ActionSuggestion {
                urgency: self.free_energy * self.action_bias,
                predicted_effect: self.predict_future().content,
                confidence: 1.0 - self.free_energy,
            })
        } else {
            None
        }
    }

    /// Get the highest level belief (closest to "conscious" content)
    pub fn highest_level_belief(&self) -> HV16 {
        self.layers.last().map(|l| l.belief.clone()).unwrap_or_else(|| HV16::random(100))
    }

    /// Get accuracy statistics
    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.5
        } else {
            self.accurate_predictions as f64 / self.total_predictions as f64
        }
    }

    /// Precision-weighted prediction error (key metric)
    pub fn precision_weighted_prediction_error(&self) -> f64 {
        let mut total = 0.0;
        let mut weight_sum = 0.0;

        for layer in &self.layers {
            for error in &layer.bottom_up_errors {
                total += error.weighted_error;
                weight_sum += layer.precision;
            }
        }

        if weight_sum > 0.0 {
            total / weight_sum
        } else {
            0.0
        }
    }

    /// Modulate Φ based on predictive processing state
    pub fn calculate_phi_modulation(&self) -> f64 {
        // Low free energy (accurate predictions) = high integration
        let surprise_factor = 1.0 - self.free_energy * 0.5;

        // High accuracy = stable consciousness
        let accuracy_factor = self.accuracy();

        // Balanced action/perception = integrated processing
        let balance_factor = 1.0 - (self.action_bias - 0.5).abs();

        surprise_factor * accuracy_factor * balance_factor
    }
}

/// Result of processing sensory input
#[derive(Debug)]
pub struct PredictionResult {
    /// Errors at each level
    pub errors: Vec<PredictionError>,

    /// Current free energy
    pub free_energy: f64,

    /// Should we act?
    pub suggested_action: Option<ActionSuggestion>,

    /// What we're "conscious" of (highest level belief)
    pub conscious_content: HV16,

    /// How surprised we are (0-1)
    pub surprise: f64,
}

/// Suggestion to take action
#[derive(Debug)]
pub struct ActionSuggestion {
    /// How urgent (0-1)
    pub urgency: f64,

    /// What we predict will happen if we act
    pub predicted_effect: HV16,

    /// How confident we are
    pub confidence: f64,
}

// ============================================================================
// Active Inference Engine
// ============================================================================

/// Active Inference: Actions as predictions about proprioception
#[derive(Debug)]
pub struct ActiveInferenceEngine {
    /// Predictive hierarchy for motor control
    motor_hierarchy: PredictiveHierarchy,

    /// Desired states (goals as predictions)
    desired_states: Vec<DesiredState>,

    /// Action repertoire
    action_repertoire: Vec<ActionTemplate>,

    /// Policy selection
    expected_free_energy: HashMap<u32, f64>,
}

/// A desired state (goal encoded as prediction)
#[derive(Debug, Clone)]
pub struct DesiredState {
    /// The desired outcome
    pub state: HV16,

    /// How much we want it (precision)
    pub precision: f64,

    /// Time horizon
    pub horizon: u32,
}

/// Template for possible actions
#[derive(Debug, Clone)]
pub struct ActionTemplate {
    /// Action identifier
    pub id: u32,

    /// Motor command encoding
    pub motor_command: HV16,

    /// Expected proprioceptive outcome
    pub expected_outcome: HV16,

    /// Past success rate
    pub success_rate: f64,
}

impl ActiveInferenceEngine {
    pub fn new() -> Self {
        Self {
            motor_hierarchy: PredictiveHierarchy::new(PredictiveConfig {
                num_levels: 3,
                model_capacity: 100,
                base_precision: 0.6,
                ..Default::default()
            }),
            desired_states: Vec::new(),
            action_repertoire: Vec::new(),
            expected_free_energy: HashMap::new(),
        }
    }

    /// Add a goal (desired state)
    pub fn add_goal(&mut self, state: HV16, importance: f64, horizon: u32) {
        self.desired_states.push(DesiredState {
            state,
            precision: importance,
            horizon,
        });
    }

    /// Add action to repertoire
    pub fn add_action(&mut self, motor_command: HV16, expected_outcome: HV16) {
        let id = self.action_repertoire.len() as u32;
        self.action_repertoire.push(ActionTemplate {
            id,
            motor_command,
            expected_outcome,
            success_rate: 0.5,
        });
    }

    /// Select best action using expected free energy
    pub fn select_action(&mut self, current_state: &HV16) -> Option<&ActionTemplate> {
        if self.action_repertoire.is_empty() || self.desired_states.is_empty() {
            return None;
        }

        // Calculate expected free energy for each action
        for action in &self.action_repertoire {
            let efe = self.calculate_expected_free_energy(action, current_state);
            self.expected_free_energy.insert(action.id, efe);
        }

        // Select action with LOWEST expected free energy
        self.action_repertoire.iter()
            .min_by(|a, b| {
                let efe_a = self.expected_free_energy.get(&a.id).unwrap_or(&1.0);
                let efe_b = self.expected_free_energy.get(&b.id).unwrap_or(&1.0);
                efe_a.partial_cmp(efe_b).unwrap()
            })
    }

    /// Calculate Expected Free Energy (EFE) for an action
    /// EFE = Epistemic value + Pragmatic value
    fn calculate_expected_free_energy(&self, action: &ActionTemplate, current_state: &HV16) -> f64 {
        // Epistemic value: How much would this action reduce uncertainty?
        let epistemic_value = 1.0 - action.success_rate;  // Higher for novel actions

        // Pragmatic value: How close would outcome be to desired states?
        let mut pragmatic_value = 0.0;
        for desired in &self.desired_states {
            let match_score = action.expected_outcome.similarity(&desired.state) as f64;
            pragmatic_value += match_score * desired.precision;
        }
        pragmatic_value /= self.desired_states.len().max(1) as f64;

        // Distance from current state (cost of action)
        let action_cost = 1.0 - current_state.similarity(&action.expected_outcome) as f64;

        // EFE = uncertainty + distance_from_goals + action_cost
        // Lower is better!
        epistemic_value * 0.3 + (1.0 - pragmatic_value) * 0.5 + action_cost * 0.2
    }

    /// Learn from action outcome
    pub fn learn_outcome(&mut self, action_id: u32, actual_outcome: &HV16, success: bool) {
        if let Some(action) = self.action_repertoire.iter_mut().find(|a| a.id == action_id) {
            // Update success rate
            action.success_rate = action.success_rate * 0.9 + (if success { 1.0 } else { 0.0 }) * 0.1;

            // Update expected outcome
            if success {
                action.expected_outcome = actual_outcome.clone();
            }
        }

        // Update motor hierarchy
        self.motor_hierarchy.learn_from_action(
            &self.action_repertoire.get(action_id as usize)
                .map(|a| a.motor_command.clone())
                .unwrap_or_else(|| HV16::random(100)),
            actual_outcome,
            success
        );
    }
}

impl Default for ActiveInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Precision Dynamics (The Role of Affect)
// ============================================================================

/// Precision modulation system - how certainty/uncertainty affects processing
#[derive(Debug)]
pub struct PrecisionDynamics {
    /// Sensory precision (bottom-up attention)
    pub sensory_precision: f64,

    /// Prior precision (top-down expectation)
    pub prior_precision: f64,

    /// State precision (interoceptive certainty)
    pub state_precision: f64,

    /// Emotional modulation of precision
    affective_gain: f64,

    /// Precision history for adaptation
    precision_history: VecDeque<f64>,
}

impl PrecisionDynamics {
    pub fn new() -> Self {
        Self {
            sensory_precision: 0.5,
            prior_precision: 0.5,
            state_precision: 0.5,
            affective_gain: 1.0,
            precision_history: VecDeque::with_capacity(50),
        }
    }

    /// Update precision based on prediction error
    pub fn update_from_error(&mut self, error: f64) {
        // High error → reduce prior precision, increase sensory precision
        // (pay more attention to sensory data)
        if error > 0.5 {
            self.prior_precision *= 0.95;
            self.sensory_precision = (self.sensory_precision * 1.05).min(1.0);
        } else {
            // Low error → can rely more on predictions
            self.prior_precision = (self.prior_precision * 1.02).min(0.9);
            self.sensory_precision *= 0.98;
        }

        // Track history
        let current = (self.sensory_precision + self.prior_precision) / 2.0;
        self.precision_history.push_back(current);
        if self.precision_history.len() > 50 {
            self.precision_history.pop_front();
        }
    }

    /// Modulate precision by affect (arousal increases gain)
    pub fn apply_affective_modulation(&mut self, arousal: f64, valence: f64) {
        // High arousal increases precision gain (sharpens attention)
        self.affective_gain = 0.8 + arousal * 0.4;

        // Negative valence increases sensory precision (threat detection)
        if valence < -0.3 {
            self.sensory_precision = (self.sensory_precision * 1.1 * self.affective_gain).min(1.0);
        }

        // Positive valence increases prior precision (confidence)
        if valence > 0.3 {
            self.prior_precision = (self.prior_precision * 1.1).min(1.0);
        }
    }

    /// Get effective precision for a level
    pub fn effective_precision(&self, level: u32) -> f64 {
        // Lower levels use more sensory precision
        // Higher levels use more prior precision
        let blend = level as f64 / 4.0;  // Assuming 4 levels
        let base = self.sensory_precision * (1.0 - blend) + self.prior_precision * blend;
        (base * self.affective_gain).min(1.0)
    }

    /// Precision stability (for Φ modulation)
    pub fn stability(&self) -> f64 {
        if self.precision_history.len() < 2 {
            return 0.5;
        }

        // Calculate variance
        let mean: f64 = self.precision_history.iter().sum::<f64>() / self.precision_history.len() as f64;
        let variance: f64 = self.precision_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.precision_history.len() as f64;

        // Low variance = high stability
        1.0 - variance.sqrt().min(0.5) * 2.0
    }
}

impl Default for PrecisionDynamics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Integration: Unified Predictive Mind
// ============================================================================

/// The complete predictive processing system
#[derive(Debug)]
pub struct PredictiveMind {
    /// Perceptual hierarchy
    pub perception: PredictiveHierarchy,

    /// Active inference (motor)
    pub action: ActiveInferenceEngine,

    /// Precision dynamics
    pub precision: PrecisionDynamics,

    /// Overall model evidence (how well is the world explained?)
    model_evidence: f64,

    /// Integrated processing state
    integration_level: f64,
}

impl PredictiveMind {
    pub fn new(config: PredictiveConfig) -> Self {
        Self {
            perception: PredictiveHierarchy::new(config),
            action: ActiveInferenceEngine::new(),
            precision: PrecisionDynamics::new(),
            model_evidence: 0.5,
            integration_level: 0.5,
        }
    }

    /// Full processing cycle
    pub fn process(&mut self, sensory_input: &HV16) -> MindState {
        // Perceptual inference
        let perception_result = self.perception.process_sensory_input(sensory_input);

        // Update precision based on errors
        self.precision.update_from_error(perception_result.surprise);

        // Update model evidence
        self.model_evidence = self.model_evidence * 0.95 + (1.0 - perception_result.free_energy) * 0.05;

        // Select action if needed
        let selected_action = if perception_result.free_energy > 0.6 {
            self.action.select_action(sensory_input).map(|a| a.id)
        } else {
            None
        };

        // Calculate integration
        self.integration_level = self.calculate_integration();

        MindState {
            conscious_content: perception_result.conscious_content,
            free_energy: perception_result.free_energy,
            model_evidence: self.model_evidence,
            suggested_action: selected_action,
            integration: self.integration_level,
            phi_modulation: self.calculate_phi_modulation(),
        }
    }

    fn calculate_integration(&self) -> f64 {
        // Integration = precision stability + model evidence + prediction accuracy
        let precision_factor = self.precision.stability();
        let evidence_factor = self.model_evidence;
        let accuracy_factor = self.perception.accuracy();

        (precision_factor + evidence_factor + accuracy_factor) / 3.0
    }

    /// Calculate how predictive processing modulates Φ
    pub fn calculate_phi_modulation(&self) -> f64 {
        // The "controlled hallucination" that is consciousness
        // emerges from integration of prediction and error

        let prediction_quality = self.perception.calculate_phi_modulation();
        let precision_stability = self.precision.stability();
        let model_evidence = self.model_evidence;

        // Consciousness is brightest when predictions are accurate
        // but there's still something to predict (not boring)
        let optimal_surprise = 1.0 - (self.perception.free_energy - 0.3).abs() * 2.0;

        prediction_quality * precision_stability * model_evidence * optimal_surprise
    }
}

/// Current state of the predictive mind
#[derive(Debug)]
pub struct MindState {
    /// What we're conscious of
    pub conscious_content: HV16,

    /// How surprised we are
    pub free_energy: f64,

    /// How well our model explains the world
    pub model_evidence: f64,

    /// Should we act?
    pub suggested_action: Option<u32>,

    /// How integrated is processing?
    pub integration: f64,

    /// How this modulates consciousness (Φ)
    pub phi_modulation: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_layer_creation() {
        let layer = PredictiveLayer::new(0, 100);
        assert_eq!(layer.level, 0);
        assert!(layer.precision > 0.0);
    }

    #[test]
    fn test_prediction_generation() {
        let layer = PredictiveLayer::new(1, 100);
        let prediction = layer.generate_prediction();

        assert!(prediction.precision > 0.0);
        assert_eq!(prediction.level, 1);
    }

    #[test]
    fn test_hierarchy_creation() {
        let config = PredictiveConfig::default();
        let hierarchy = PredictiveHierarchy::new(config.clone());

        assert_eq!(hierarchy.layers.len(), config.num_levels as usize);
    }

    #[test]
    fn test_sensory_processing() {
        let config = PredictiveConfig::default();
        let mut hierarchy = PredictiveHierarchy::new(config);

        let input = HV16::random(100);
        let result = hierarchy.process_sensory_input(&input);

        assert!(!result.errors.is_empty());
        assert!(result.free_energy >= 0.0);
        assert!(result.surprise >= 0.0);
    }

    #[test]
    fn test_learning_improves_accuracy() {
        let config = PredictiveConfig::default();
        let mut hierarchy = PredictiveHierarchy::new(config);

        // Create consistent input pattern
        let pattern = HV16::random(100);

        // Process multiple times
        for _ in 0..20 {
            hierarchy.process_sensory_input(&pattern);
        }

        // Should have better accuracy for repeated pattern
        let result = hierarchy.process_sensory_input(&pattern);
        assert!(result.surprise < 0.8, "Surprise should decrease with repetition");
    }

    #[test]
    fn test_prediction_error_propagation() {
        let mut layer = PredictiveLayer::new(1, 100);

        let error = PredictionError {
            prediction: HV16::random(100),
            actual: HV16::random(100),
            magnitude: 0.5,
            weighted_error: 0.3,
            level: 0,
            direction: ErrorDirection::UpdateModel,
        };

        layer.process_bottom_up_error(error);
        assert!(!layer.bottom_up_errors.is_empty());
    }

    #[test]
    fn test_active_inference_action_selection() {
        let mut engine = ActiveInferenceEngine::new();

        // Add goal
        engine.add_goal(HV16::random(100), 0.9, 5);

        // Add actions
        let outcome1 = engine.desired_states[0].state.clone();
        engine.add_action(HV16::random(100), outcome1);
        engine.add_action(HV16::random(100), HV16::random(100));

        // Select action
        let current = HV16::random(100);
        let selected = engine.select_action(&current);

        assert!(selected.is_some());
    }

    #[test]
    fn test_precision_dynamics() {
        let mut precision = PrecisionDynamics::new();

        // High error should reduce prior precision
        let initial_prior = precision.prior_precision;
        precision.update_from_error(0.8);
        assert!(precision.prior_precision < initial_prior);

        // Sensory precision should increase
        assert!(precision.sensory_precision > 0.5);
    }

    #[test]
    fn test_affective_modulation() {
        let mut precision = PrecisionDynamics::new();

        // High arousal + negative valence
        precision.apply_affective_modulation(0.9, -0.5);

        // Should increase sensory precision
        assert!(precision.sensory_precision > 0.5);
        assert!(precision.affective_gain > 1.0);
    }

    #[test]
    fn test_predictive_mind_creation() {
        let config = PredictiveConfig::default();
        let mind = PredictiveMind::new(config);

        assert!(mind.model_evidence > 0.0);
    }

    #[test]
    fn test_mind_processing() {
        let config = PredictiveConfig::default();
        let mut mind = PredictiveMind::new(config);

        let input = HV16::random(100);
        let state = mind.process(&input);

        assert!(state.free_energy >= 0.0);
        assert!(state.phi_modulation > 0.0);
    }

    #[test]
    fn test_phi_modulation_calculation() {
        let config = PredictiveConfig::default();
        let mut mind = PredictiveMind::new(config);

        // Train on consistent pattern
        let pattern = HV16::random(100);
        for _ in 0..10 {
            mind.process(&pattern);
        }

        let phi_mod = mind.calculate_phi_modulation();
        assert!(phi_mod > 0.0);
        assert!(phi_mod <= 1.5);  // Should be bounded
    }

    #[test]
    fn test_generative_model_learning() {
        let mut model = GenerativeModel::new(50);

        let cause = HV16::random(100);
        let effect = HV16::random(100);

        model.learn(cause.clone(), effect.clone(), 0.9);

        let prediction = model.predict(&cause);
        assert!(prediction.is_some());
    }

    #[test]
    fn test_action_learning_improves() {
        let mut engine = ActiveInferenceEngine::new();

        // Add action
        engine.add_action(HV16::random(100), HV16::random(100));

        // Successful outcomes should increase success rate
        let initial_rate = engine.action_repertoire[0].success_rate;

        for _ in 0..5 {
            engine.learn_outcome(0, &HV16::random(100), true);
        }

        assert!(engine.action_repertoire[0].success_rate > initial_rate);
    }
}
