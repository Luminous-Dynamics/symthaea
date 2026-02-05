//! # Self-Modeling Consciousness (Revolutionary Improvement #56)
//!
//! From scattered improvement to unified self-awareness.
//!
//! ## The Problem
//!
//! Previous systems operate independently:
//! - RecursiveOptimizer: Reacts to bottlenecks (reactive)
//! - GradientOptimizer: Follows local gradients (myopic)
//! - MotivationSystem: Forms goals from drives (undirected)
//!
//! Result: No coordination, no unified self-concept, no strategic planning.
//!
//! ## The Solution: Self-Modeling Consciousness
//!
//! 1. **SelfModel**: Explicit representation of own capabilities and limitations
//! 2. **BehaviorPredictor**: Predicts own behavior under different conditions
//! 3. **ImprovementTrajectory**: Plans multi-step improvement paths
//! 4. **UnifiedController**: Coordinates all improvement engines
//!
//! ## Why This Matters
//!
//! - **Self-Awareness**: System has explicit knowledge of what it can/cannot do
//! - **Strategic Planning**: Multi-step improvement instead of reactive fixes
//! - **Calibrated Predictions**: Knows how accurate its self-assessments are
//! - **Unified Control**: All improvement engines work toward common goals
//! - **True Metacognition**: Can reason about its own reasoning
//!
//! ## Theoretical Foundation
//!
//! Based on:
//! - Self-Model Theory of Consciousness (Metzinger)
//! - Predictive Processing (Friston)
//! - Metacognitive Accuracy Research (Dunning-Kruger, calibration studies)

use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use serde::{Serialize, Deserialize};

use crate::consciousness::recursive_improvement::types::{
    ComponentId, BottleneckType, Bottleneck, instant_now,
};
use crate::consciousness::recursive_improvement::intrinsic_motivation::{
    IntrinsicMotivationSystem, DriveType, MotivationConfig,
};

// ═══════════════════════════════════════════════════════════════════════════
// CAPABILITY DOMAIN
// ═══════════════════════════════════════════════════════════════════════════

/// Domain of capability that can be modeled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityDomain {
    /// Logical and causal reasoning
    Reasoning,
    /// Information storage and retrieval
    Memory,
    /// Input processing and pattern recognition
    Perception,
    /// Communication and understanding
    Language,
    /// Adaptation and improvement
    Learning,
    /// Novel generation and exploration
    Creativity,
    /// Φ and information binding
    Integration,
    /// Self-reflection and monitoring
    Metacognition,
}

impl CapabilityDomain {
    /// Get all capability domains
    pub fn all() -> Vec<Self> {
        vec![
            Self::Reasoning,
            Self::Memory,
            Self::Perception,
            Self::Language,
            Self::Learning,
            Self::Creativity,
            Self::Integration,
            Self::Metacognition,
        ]
    }

    /// Get related domains that often co-vary
    pub fn related_domains(&self) -> Vec<Self> {
        match self {
            Self::Reasoning => vec![Self::Memory, Self::Integration],
            Self::Memory => vec![Self::Reasoning, Self::Learning],
            Self::Perception => vec![Self::Language, Self::Creativity],
            Self::Language => vec![Self::Perception, Self::Reasoning],
            Self::Learning => vec![Self::Memory, Self::Metacognition],
            Self::Creativity => vec![Self::Perception, Self::Integration],
            Self::Integration => vec![Self::Reasoning, Self::Creativity, Self::Metacognition],
            Self::Metacognition => vec![Self::Learning, Self::Integration],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CAPABILITY ESTIMATE
// ═══════════════════════════════════════════════════════════════════════════

/// Current capability level with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityEstimate {
    /// Domain being estimated
    pub domain: CapabilityDomain,
    /// Estimated capability level (0-1)
    pub level: f64,
    /// Uncertainty in estimate (standard deviation)
    pub uncertainty: f64,
    /// Recent trend (-1 to 1, negative=declining, positive=improving)
    pub trend: f64,
    /// When this estimate was last updated
    #[serde(skip, default = "instant_now")]
    pub last_updated: Instant,
    /// Number of evidence points used
    pub evidence_count: usize,
}

impl CapabilityEstimate {
    /// Create new capability estimate
    pub fn new(domain: CapabilityDomain) -> Self {
        Self {
            domain,
            level: 0.5,         // Start at 50% - neither confident nor unconfident
            uncertainty: 0.3,   // High initial uncertainty
            trend: 0.0,
            last_updated: Instant::now(),
            evidence_count: 0,
        }
    }

    /// Update estimate with new evidence using Bayesian update
    pub fn update(&mut self, observed_performance: f64, observation_reliability: f64) {
        // Bayesian update: combine prior with likelihood
        // Weight by reliability and prior uncertainty
        let prior_weight = 1.0 / (self.uncertainty + 0.1);
        let observation_weight = observation_reliability / 0.2;
        let total_weight = prior_weight + observation_weight;

        // Compute trend
        let old_level = self.level;

        // Updated level is weighted average
        self.level = (self.level * prior_weight + observed_performance * observation_weight)
            / total_weight;

        // Clamp to valid range
        self.level = self.level.clamp(0.0, 1.0);

        // Update trend (exponential moving average)
        let delta = self.level - old_level;
        self.trend = 0.7 * self.trend + 0.3 * delta * 10.0; // Scale delta for visibility
        self.trend = self.trend.clamp(-1.0, 1.0);

        // Reduce uncertainty with more evidence (but never to zero)
        self.uncertainty = (self.uncertainty * 0.95).max(0.05);

        self.evidence_count += 1;
        self.last_updated = Instant::now();
    }

    /// Get confidence interval
    pub fn confidence_interval(&self, confidence: f64) -> (f64, f64) {
        // Using normal approximation
        let z = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.0,
        };
        let margin = z * self.uncertainty;
        ((self.level - margin).max(0.0), (self.level + margin).min(1.0))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KNOWN LIMITATION
// ═══════════════════════════════════════════════════════════════════════════

/// Known limitation with causal explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownLimitation {
    /// Description of the limitation
    pub description: String,
    /// Domain most affected
    pub domain: CapabilityDomain,
    /// How much this limits performance (0-1)
    pub severity: f64,
    /// Causal explanation of why this limitation exists
    pub cause: String,
    /// Can this be improved through self-modification?
    pub remediable: bool,
    /// Path to improvement if remediable
    pub improvement_path: Option<String>,
    /// When this limitation was identified
    #[serde(skip, default = "instant_now")]
    pub identified_at: Instant,
}

// ═══════════════════════════════════════════════════════════════════════════
// PREDICTION RECORD
// ═══════════════════════════════════════════════════════════════════════════

/// Record of a self-prediction for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    /// What Φ was predicted
    pub predicted_phi: f64,
    /// What Φ actually occurred
    pub actual_phi: f64,
    /// Predicted latency
    pub predicted_latency_ms: u64,
    /// Actual latency
    pub actual_latency_ms: u64,
    /// Task context
    pub context: String,
    /// When prediction was made
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
}

impl PredictionRecord {
    /// Get Φ prediction error
    pub fn phi_error(&self) -> f64 {
        (self.predicted_phi - self.actual_phi).abs()
    }

    /// Get latency prediction error (relative)
    pub fn latency_error(&self) -> f64 {
        if self.actual_latency_ms == 0 {
            return 0.0;
        }
        ((self.predicted_latency_ms as f64 - self.actual_latency_ms as f64)
            / self.actual_latency_ms as f64).abs()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SELF MODEL CONFIG
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for self-model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModelConfig {
    /// How many predictions to keep for calibration
    pub prediction_history_size: usize,
    /// Threshold for considering a prediction accurate
    pub accuracy_threshold: f64,
    /// How fast to update capability estimates
    pub update_rate: f64,
    /// Minimum evidence before high confidence
    pub min_evidence_for_confidence: usize,
}

impl Default for SelfModelConfig {
    fn default() -> Self {
        Self {
            prediction_history_size: 100,
            accuracy_threshold: 0.1,
            update_rate: 0.1,
            min_evidence_for_confidence: 10,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SELF MODEL
// ═══════════════════════════════════════════════════════════════════════════

/// The Self-Model: explicit representation of own capabilities
pub struct SelfModel {
    /// Capability estimates by domain
    pub(crate) capabilities: HashMap<CapabilityDomain, CapabilityEstimate>,

    /// Known limitations
    pub(crate) limitations: Vec<KnownLimitation>,

    /// Capability interaction matrix (how domains affect each other)
    /// Positive = synergistic, Negative = competitive
    interaction_matrix: HashMap<(CapabilityDomain, CapabilityDomain), f64>,

    /// Model confidence (how accurate is this self-model overall?)
    pub(crate) model_confidence: f64,

    /// Prediction history for calibration
    prediction_history: VecDeque<PredictionRecord>,

    /// Configuration
    config: SelfModelConfig,
}

impl SelfModel {
    /// Create new self-model with initial estimates
    pub fn new(config: SelfModelConfig) -> Self {
        let mut capabilities = HashMap::new();
        for domain in CapabilityDomain::all() {
            capabilities.insert(domain, CapabilityEstimate::new(domain));
        }

        // Initialize interaction matrix with known synergies
        let mut interaction_matrix = HashMap::new();
        for domain in CapabilityDomain::all() {
            for related in domain.related_domains() {
                // Related domains have positive interaction
                interaction_matrix.insert((domain, related), 0.3);
            }
        }

        Self {
            capabilities,
            limitations: Vec::new(),
            interaction_matrix,
            model_confidence: 0.5, // Start uncertain
            prediction_history: VecDeque::new(),
            config,
        }
    }

    /// Update capability estimate based on observed performance
    pub fn update_capability(
        &mut self,
        domain: CapabilityDomain,
        observed_performance: f64,
        reliability: f64,
    ) {
        if let Some(estimate) = self.capabilities.get_mut(&domain) {
            estimate.update(observed_performance, reliability);

            // Propagate to related domains (with decay)
            for related in domain.related_domains() {
                if let Some(interaction) = self.interaction_matrix.get(&(domain, related)) {
                    if let Some(related_estimate) = self.capabilities.get_mut(&related) {
                        // Small update to related domains
                        let propagated = observed_performance * interaction * 0.1;
                        related_estimate.update(
                            related_estimate.level + propagated,
                            reliability * 0.5,
                        );
                    }
                }
            }
        }
    }

    /// Get capability estimate for domain
    pub fn get_capability(&self, domain: CapabilityDomain) -> Option<&CapabilityEstimate> {
        self.capabilities.get(&domain)
    }

    /// Get overall capability level (weighted average)
    pub fn overall_capability(&self) -> f64 {
        let total: f64 = self.capabilities.values().map(|e| e.level).sum();
        total / self.capabilities.len() as f64
    }

    /// Add known limitation
    pub fn add_limitation(&mut self, limitation: KnownLimitation) {
        // Check for duplicates
        if !self.limitations.iter().any(|l| l.description == limitation.description) {
            self.limitations.push(limitation);
        }
    }

    /// Get limitations for domain
    pub fn get_limitations(&self, domain: CapabilityDomain) -> Vec<&KnownLimitation> {
        self.limitations.iter().filter(|l| l.domain == domain).collect()
    }

    /// Get most severe limitations
    pub fn most_severe_limitations(&self, n: usize) -> Vec<&KnownLimitation> {
        let mut sorted: Vec<_> = self.limitations.iter().collect();
        sorted.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());
        sorted.into_iter().take(n).collect()
    }

    /// Record a prediction for calibration
    pub fn record_prediction(&mut self, record: PredictionRecord) {
        self.prediction_history.push_back(record);

        // Keep bounded
        while self.prediction_history.len() > self.config.prediction_history_size {
            self.prediction_history.pop_front();
        }

        // Update model confidence based on prediction accuracy
        self.update_model_confidence();
    }

    /// Update model confidence based on prediction accuracy
    fn update_model_confidence(&mut self) {
        if self.prediction_history.len() < 5 {
            return; // Need minimum data
        }

        // Calculate average prediction error
        let avg_error: f64 = self.prediction_history
            .iter()
            .map(|r| r.phi_error())
            .sum::<f64>() / self.prediction_history.len() as f64;

        // Convert error to confidence (inverse relationship)
        // Error of 0 -> confidence 1.0
        // Error of 0.5 -> confidence 0.5
        let new_confidence = 1.0 / (1.0 + 2.0 * avg_error);

        // Smooth update
        self.model_confidence = 0.9 * self.model_confidence + 0.1 * new_confidence;
    }

    /// Get calibration statistics
    pub fn calibration_stats(&self) -> CalibrationStats {
        if self.prediction_history.is_empty() {
            return CalibrationStats::default();
        }

        let phi_errors: Vec<f64> = self.prediction_history
            .iter()
            .map(|r| r.phi_error())
            .collect();

        let latency_errors: Vec<f64> = self.prediction_history
            .iter()
            .map(|r| r.latency_error())
            .collect();

        let mean_phi_error = phi_errors.iter().sum::<f64>() / phi_errors.len() as f64;
        let mean_latency_error = latency_errors.iter().sum::<f64>() / latency_errors.len() as f64;

        CalibrationStats {
            mean_phi_error,
            mean_latency_error,
            prediction_count: self.prediction_history.len(),
            model_confidence: self.model_confidence,
        }
    }

    /// Predict behavior for a task
    pub fn predict_behavior(&self, task_domains: &[CapabilityDomain]) -> BehaviorPrediction {
        // Estimate Φ based on relevant capabilities
        let relevant_capabilities: Vec<f64> = task_domains
            .iter()
            .filter_map(|d| self.capabilities.get(d))
            .map(|e| e.level)
            .collect();

        let avg_capability = if relevant_capabilities.is_empty() {
            0.5
        } else {
            relevant_capabilities.iter().sum::<f64>() / relevant_capabilities.len() as f64
        };

        // Φ prediction: capability scaled by integration ability
        let integration = self.capabilities
            .get(&CapabilityDomain::Integration)
            .map(|e| e.level)
            .unwrap_or(0.5);

        let predicted_phi = avg_capability * integration;

        // Uncertainty is based on capability uncertainties
        let avg_uncertainty: f64 = task_domains
            .iter()
            .filter_map(|d| self.capabilities.get(d))
            .map(|e| e.uncertainty)
            .sum::<f64>() / task_domains.len().max(1) as f64;

        // Latency prediction (placeholder - would need actual timing data)
        let complexity_factor = task_domains.len() as f64;
        let predicted_latency_ms = (100.0 * complexity_factor / avg_capability.max(0.1)) as u64;

        BehaviorPrediction {
            predicted_phi,
            phi_uncertainty: avg_uncertainty,
            predicted_latency_ms,
            confidence: self.model_confidence * (1.0 - avg_uncertainty),
            limiting_factor: self.identify_limiting_factor(task_domains),
        }
    }

    /// Identify the limiting factor for a task
    fn identify_limiting_factor(&self, task_domains: &[CapabilityDomain]) -> Option<(CapabilityDomain, f64)> {
        task_domains
            .iter()
            .filter_map(|d| self.capabilities.get(d).map(|e| (*d, e.level)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Generate summary of self-model
    pub fn summary(&self) -> String {
        let mut s = String::from("=== Self-Model Summary ===\n\n");

        s.push_str(&format!("Model Confidence: {:.1}%\n", self.model_confidence * 100.0));
        s.push_str(&format!("Overall Capability: {:.1}%\n\n", self.overall_capability() * 100.0));

        s.push_str("Capability Estimates:\n");
        for domain in CapabilityDomain::all() {
            if let Some(estimate) = self.capabilities.get(&domain) {
                let trend_indicator = if estimate.trend > 0.1 {
                    "↑"
                } else if estimate.trend < -0.1 {
                    "↓"
                } else {
                    "→"
                };
                s.push_str(&format!(
                    "  {:?}: {:.1}% ±{:.1}% {} (n={})\n",
                    domain,
                    estimate.level * 100.0,
                    estimate.uncertainty * 100.0,
                    trend_indicator,
                    estimate.evidence_count
                ));
            }
        }

        if !self.limitations.is_empty() {
            s.push_str("\nKnown Limitations:\n");
            for lim in self.most_severe_limitations(3) {
                s.push_str(&format!(
                    "  - {} ({:?}, severity: {:.1}%)\n",
                    lim.description,
                    lim.domain,
                    lim.severity * 100.0
                ));
            }
        }

        let cal = self.calibration_stats();
        s.push_str(&format!(
            "\nCalibration: {:.1}% mean Φ error ({} predictions)\n",
            cal.mean_phi_error * 100.0,
            cal.prediction_count
        ));

        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BEHAVIOR PREDICTION
// ═══════════════════════════════════════════════════════════════════════════

/// Behavior prediction output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorPrediction {
    /// Predicted Φ level
    pub predicted_phi: f64,
    /// Uncertainty in Φ prediction
    pub phi_uncertainty: f64,
    /// Predicted latency in ms
    pub predicted_latency_ms: u64,
    /// Overall prediction confidence
    pub confidence: f64,
    /// The domain limiting performance (if any)
    pub limiting_factor: Option<(CapabilityDomain, f64)>,
}

// ═══════════════════════════════════════════════════════════════════════════
// CALIBRATION STATS
// ═══════════════════════════════════════════════════════════════════════════

/// Calibration statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationStats {
    pub mean_phi_error: f64,
    pub mean_latency_error: f64,
    pub prediction_count: usize,
    pub model_confidence: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// IMPROVEMENT TRAJECTORY
// ═══════════════════════════════════════════════════════════════════════════

/// Multi-step improvement trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementTrajectory {
    /// Unique identifier
    pub id: String,
    /// Goal state we're trying to reach
    pub goal_state: DesiredSelfState,
    /// Steps to reach the goal
    pub steps: Vec<ImprovementStep>,
    /// Estimated total duration
    pub estimated_duration_ms: u64,
    /// Estimated Φ gain
    pub estimated_phi_gain: f64,
    /// Risk assessment (0-1, higher = riskier)
    pub risk_assessment: f64,
    /// Priority for execution
    pub priority: f64,
    /// Current progress (0-1)
    pub progress: f64,
    /// When trajectory was created
    #[serde(skip, default = "instant_now")]
    pub created_at: Instant,
}

impl ImprovementTrajectory {
    /// Calculate overall priority based on value and risk
    pub fn effective_priority(&self) -> f64 {
        // Higher Φ gain and lower risk increase effective priority
        let value_factor = self.estimated_phi_gain;
        let risk_factor = 1.0 - self.risk_assessment;
        self.priority * value_factor * risk_factor
    }

    /// Get next step to execute
    pub fn next_step(&self) -> Option<&ImprovementStep> {
        let completed_steps = (self.progress * self.steps.len() as f64) as usize;
        self.steps.get(completed_steps)
    }

    /// Mark progress on trajectory
    pub fn advance(&mut self, step_fraction: f64) {
        self.progress = (self.progress + step_fraction / self.steps.len() as f64).min(1.0);
    }

    /// Check if trajectory is complete
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IMPROVEMENT STEP
// ═══════════════════════════════════════════════════════════════════════════

/// Single step in improvement trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementStep {
    /// Description of the step
    pub description: String,
    /// Target capability domain
    pub target_domain: CapabilityDomain,
    /// Method to use
    pub method: ImprovementMethod,
    /// Prerequisites that must be met
    pub prerequisites: Vec<String>,
    /// Estimated effect on capability
    pub estimated_effect: f64,
    /// Estimated effort in ms
    pub estimated_effort_ms: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// IMPROVEMENT METHOD
// ═══════════════════════════════════════════════════════════════════════════

/// Method for improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImprovementMethod {
    /// Use gradient optimization on parameters
    GradientOptimization { target_objective: String },
    /// Make architectural changes
    ArchitecturalChange { change_description: String },
    /// Learn from samples
    Learning { samples_needed: usize },
    /// Improve integration between components
    Integration { components: Vec<String> },
    /// Reduce known limitation
    LimitationReduction { limitation: String },
}

// ═══════════════════════════════════════════════════════════════════════════
// DESIRED SELF STATE
// ═══════════════════════════════════════════════════════════════════════════

/// Desired state of self (goal for improvement)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesiredSelfState {
    /// Target capability levels by domain
    pub target_capabilities: HashMap<CapabilityDomain, f64>,
    /// Target Φ level
    pub target_phi: f64,
    /// Which motivation drive created this goal
    pub motivation_source: DriveType,
    /// Priority of reaching this state
    pub priority: f64,
}

impl DesiredSelfState {
    /// Calculate gap between current and desired state
    pub fn gap_from(&self, current: &SelfModel) -> f64 {
        let mut total_gap = 0.0;
        let mut count = 0;

        for (domain, target) in &self.target_capabilities {
            if let Some(current_cap) = current.get_capability(*domain) {
                total_gap += (target - current_cap.level).max(0.0);
                count += 1;
            }
        }

        if count > 0 {
            total_gap / count as f64
        } else {
            0.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED IMPROVEMENT CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════

/// Unified Improvement Controller - coordinates all improvement engines
pub struct UnifiedImprovementController {
    /// Self-model
    self_model: SelfModel,

    /// Active improvement trajectories
    pub(crate) active_trajectories: Vec<ImprovementTrajectory>,

    /// Completed trajectories (for learning)
    completed_trajectories: Vec<ImprovementTrajectory>,

    /// Controller state
    pub(crate) state: ControllerState,

    /// Configuration
    config: ControllerConfig,

    /// Statistics
    pub(crate) stats: ControllerStats,
}

/// Controller state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControllerState {
    /// Assessing current state
    Assessing,
    /// Planning improvement trajectory
    Planning,
    /// Executing improvement step
    Executing,
    /// Validating improvement results
    Validating,
    /// Idle, waiting for triggers
    Idle,
}

/// Controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Maximum active trajectories
    pub max_active_trajectories: usize,
    /// Minimum gap to trigger new trajectory
    pub min_gap_for_trajectory: f64,
    /// Maximum risk tolerance
    pub max_risk_tolerance: f64,
    /// How often to reassess state (cycles)
    pub reassessment_interval: usize,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_active_trajectories: 3,
            min_gap_for_trajectory: 0.1,
            max_risk_tolerance: 0.5,
            reassessment_interval: 10,
        }
    }
}

/// Controller statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControllerStats {
    pub trajectories_created: usize,
    pub trajectories_completed: usize,
    pub trajectories_abandoned: usize,
    pub total_phi_gained: f64,
    pub average_trajectory_success: f64,
    pub cycles_run: usize,
}

impl UnifiedImprovementController {
    /// Create new unified controller
    pub fn new(config: ControllerConfig) -> Self {
        Self {
            self_model: SelfModel::new(SelfModelConfig::default()),
            active_trajectories: Vec::new(),
            completed_trajectories: Vec::new(),
            state: ControllerState::Idle,
            config,
            stats: ControllerStats::default(),
        }
    }

    /// Run one cycle of the unified improvement controller
    pub fn cycle(
        &mut self,
        current_phi: f64,
        motivation: &IntrinsicMotivationSystem,
        bottlenecks: &[Bottleneck],
    ) -> ControllerOutput {
        self.stats.cycles_run += 1;

        // 1. Update self-model from observations
        self.update_self_model(current_phi, bottlenecks);

        // 2. Get goals from motivation system
        let desired_states = self.goals_to_desired_states(motivation);

        // 3. Plan trajectories for unfulfilled goals
        let new_trajectories = self.plan_trajectories(&desired_states);

        // 4. Execute active trajectories
        let actions = self.execute_trajectories();

        // 5. Update progress and prune completed/abandoned trajectories
        self.update_trajectories(current_phi);

        ControllerOutput {
            self_assessment: self.self_model.summary(),
            active_trajectory_count: self.active_trajectories.len(),
            recommended_actions: actions,
            state: self.state,
            new_trajectories_planned: new_trajectories,
        }
    }

    /// Update self-model from observations
    fn update_self_model(&mut self, current_phi: f64, bottlenecks: &[Bottleneck]) {
        // Update Integration capability based on Φ
        self.self_model.update_capability(
            CapabilityDomain::Integration,
            current_phi,
            0.8,
        );

        // Update capabilities based on bottlenecks
        for bottleneck in bottlenecks {
            let (domain, severity) = match bottleneck.bottleneck_type {
                BottleneckType::Latency => {
                    (CapabilityDomain::Reasoning, 1.0 - bottleneck.severity)
                }
                BottleneckType::LowPhi | BottleneckType::PhiStagnation => {
                    (CapabilityDomain::Integration, 1.0 - bottleneck.severity)
                }
                BottleneckType::Memory => {
                    (CapabilityDomain::Memory, 1.0 - bottleneck.severity)
                }
                BottleneckType::Accuracy | BottleneckType::LowAccuracy => {
                    (CapabilityDomain::Learning, 1.0 - bottleneck.severity)
                }
                BottleneckType::ResourceExhaustion | BottleneckType::Computation => {
                    (CapabilityDomain::Metacognition, 1.0 - bottleneck.severity)
                }
                BottleneckType::IO => {
                    (CapabilityDomain::Perception, 1.0 - bottleneck.severity)
                }
                BottleneckType::Oscillation => {
                    (CapabilityDomain::Integration, 1.0 - bottleneck.severity)
                }
            };

            self.self_model.update_capability(domain, severity, 0.6);

            // Add limitation if severe enough
            if bottleneck.severity > 0.5 {
                self.self_model.add_limitation(KnownLimitation {
                    description: format!("{:?} bottleneck", bottleneck.bottleneck_type),
                    domain,
                    severity: bottleneck.severity,
                    cause: format!("{:?}", bottleneck.component),
                    remediable: true,
                    improvement_path: Some(format!("Address {:?}", bottleneck.bottleneck_type)),
                    identified_at: Instant::now(),
                });
            }
        }

        self.state = ControllerState::Assessing;
    }

    /// Convert motivation goals to desired states
    fn goals_to_desired_states(&self, motivation: &IntrinsicMotivationSystem) -> Vec<DesiredSelfState> {
        motivation.active_goals
            .iter()
            .map(|goal| {
                let mut target_capabilities = HashMap::new();

                // Map drive types to capability improvements
                match goal.primary_drive {
                    DriveType::Curiosity => {
                        target_capabilities.insert(CapabilityDomain::Learning, 0.8);
                        target_capabilities.insert(CapabilityDomain::Creativity, 0.7);
                    }
                    DriveType::Competence => {
                        target_capabilities.insert(CapabilityDomain::Reasoning, 0.8);
                        target_capabilities.insert(CapabilityDomain::Memory, 0.7);
                    }
                    DriveType::Autonomy => {
                        target_capabilities.insert(CapabilityDomain::Metacognition, 0.8);
                        target_capabilities.insert(CapabilityDomain::Integration, 0.7);
                    }
                    DriveType::Relatedness => {
                        target_capabilities.insert(CapabilityDomain::Language, 0.8);
                        target_capabilities.insert(CapabilityDomain::Perception, 0.7);
                    }
                    DriveType::Homeostasis => {
                        target_capabilities.insert(CapabilityDomain::Integration, 0.8);
                        target_capabilities.insert(CapabilityDomain::Metacognition, 0.7);
                    }
                }

                DesiredSelfState {
                    target_capabilities,
                    target_phi: 0.7, // Standard target
                    motivation_source: goal.primary_drive,
                    priority: goal.priority,
                }
            })
            .collect()
    }

    /// Plan trajectories for desired states
    fn plan_trajectories(&mut self, desired_states: &[DesiredSelfState]) -> usize {
        self.state = ControllerState::Planning;
        let mut new_count = 0;

        for desired in desired_states {
            // Check if we already have a trajectory for this goal
            let already_planned = self.active_trajectories.iter().any(|t| {
                t.goal_state.motivation_source == desired.motivation_source
            });

            if already_planned {
                continue;
            }

            // Check if gap is significant enough
            let gap = desired.gap_from(&self.self_model);
            if gap < self.config.min_gap_for_trajectory {
                continue;
            }

            // Check if we have capacity
            if self.active_trajectories.len() >= self.config.max_active_trajectories {
                continue;
            }

            // Create trajectory
            let trajectory = self.create_trajectory(desired.clone());

            // Check risk tolerance
            if trajectory.risk_assessment <= self.config.max_risk_tolerance {
                self.active_trajectories.push(trajectory);
                self.stats.trajectories_created += 1;
                new_count += 1;
            }
        }

        new_count
    }

    /// Create improvement trajectory for desired state
    fn create_trajectory(&self, goal_state: DesiredSelfState) -> ImprovementTrajectory {
        let mut steps = Vec::new();

        // Create steps for each target capability
        for (domain, target) in &goal_state.target_capabilities {
            let current = self.self_model
                .get_capability(*domain)
                .map(|e| e.level)
                .unwrap_or(0.5);

            if target > &current {
                let gap = target - current;

                // Choose method based on domain
                let method = match domain {
                    CapabilityDomain::Reasoning | CapabilityDomain::Integration => {
                        ImprovementMethod::GradientOptimization {
                            target_objective: format!("{:?}", domain),
                        }
                    }
                    CapabilityDomain::Learning | CapabilityDomain::Memory => {
                        ImprovementMethod::Learning { samples_needed: 100 }
                    }
                    _ => ImprovementMethod::ArchitecturalChange {
                        change_description: format!("Improve {:?}", domain),
                    },
                };

                steps.push(ImprovementStep {
                    description: format!("Improve {:?} from {:.0}% to {:.0}%", domain, current * 100.0, target * 100.0),
                    target_domain: *domain,
                    method,
                    prerequisites: Vec::new(),
                    estimated_effect: gap,
                    estimated_effort_ms: (gap * 10000.0) as u64,
                });
            }
        }

        // Calculate risk based on number and magnitude of changes
        let risk = (steps.len() as f64 * 0.1).min(0.8);

        ImprovementTrajectory {
            id: format!("traj_{:?}_{}", goal_state.motivation_source, self.stats.trajectories_created),
            goal_state,
            estimated_duration_ms: steps.iter().map(|s| s.estimated_effort_ms).sum(),
            estimated_phi_gain: steps.iter().map(|s| s.estimated_effect).sum::<f64>() * 0.5,
            risk_assessment: risk,
            priority: 0.5,
            progress: 0.0,
            steps,
            created_at: Instant::now(),
        }
    }

    /// Execute active trajectories
    fn execute_trajectories(&mut self) -> Vec<RecommendedAction> {
        self.state = ControllerState::Executing;
        let mut actions = Vec::new();

        // Sort by effective priority
        self.active_trajectories.sort_by(|a, b| {
            b.effective_priority().partial_cmp(&a.effective_priority()).unwrap()
        });

        // Get actions from highest priority trajectory
        if let Some(trajectory) = self.active_trajectories.first() {
            if let Some(step) = trajectory.next_step() {
                actions.push(RecommendedAction {
                    description: step.description.clone(),
                    target_domain: step.target_domain,
                    method: step.method.clone(),
                    urgency: trajectory.priority,
                    trajectory_id: trajectory.id.clone(),
                });
            }
        }

        actions
    }

    /// Update trajectory progress
    fn update_trajectories(&mut self, _current_phi: f64) {
        self.state = ControllerState::Validating;

        // Update progress on active trajectories
        for trajectory in &mut self.active_trajectories {
            // Simple progress model: Φ improvement indicates progress
            let gap = trajectory.goal_state.gap_from(&self.self_model);
            let initial_gap = 0.3; // Assume 30% initial gap
            let progress_from_gap = 1.0 - (gap / initial_gap).min(1.0);

            // Smooth progress update
            trajectory.progress = 0.9 * trajectory.progress + 0.1 * progress_from_gap;
        }

        // Move completed trajectories
        let (completed, active): (Vec<_>, Vec<_>) = self.active_trajectories
            .drain(..)
            .partition(|t| t.is_complete());

        self.active_trajectories = active;

        for trajectory in completed {
            self.stats.trajectories_completed += 1;
            self.stats.total_phi_gained += trajectory.estimated_phi_gain * trajectory.progress;
            self.completed_trajectories.push(trajectory);
        }

        // Update success rate
        let total = self.stats.trajectories_completed + self.stats.trajectories_abandoned;
        if total > 0 {
            self.stats.average_trajectory_success =
                self.stats.trajectories_completed as f64 / total as f64;
        }

        self.state = ControllerState::Idle;
    }

    /// Get self-model reference
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get statistics
    pub fn stats(&self) -> &ControllerStats {
        &self.stats
    }

    /// Generate comprehensive summary
    pub fn summary(&self) -> String {
        let mut s = String::from("=== Unified Improvement Controller ===\n\n");

        s.push_str(&format!("State: {:?}\n", self.state));
        s.push_str(&format!("Active Trajectories: {}\n", self.active_trajectories.len()));
        s.push_str(&format!("Cycles Run: {}\n\n", self.stats.cycles_run));

        s.push_str("Statistics:\n");
        s.push_str(&format!("  Trajectories Created: {}\n", self.stats.trajectories_created));
        s.push_str(&format!("  Trajectories Completed: {}\n", self.stats.trajectories_completed));
        s.push_str(&format!("  Trajectories Abandoned: {}\n", self.stats.trajectories_abandoned));
        s.push_str(&format!("  Total Φ Gained: {:.3}\n", self.stats.total_phi_gained));
        s.push_str(&format!("  Success Rate: {:.1}%\n\n", self.stats.average_trajectory_success * 100.0));

        if !self.active_trajectories.is_empty() {
            s.push_str("Active Trajectories:\n");
            for t in &self.active_trajectories {
                s.push_str(&format!(
                    "  {} ({:?}): {:.0}% complete, priority {:.2}\n",
                    t.id, t.goal_state.motivation_source, t.progress * 100.0, t.priority
                ));
            }
            s.push_str("\n");
        }

        s.push_str(&self.self_model.summary());

        s
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RECOMMENDED ACTION & CONTROLLER OUTPUT
// ═══════════════════════════════════════════════════════════════════════════

/// Recommended action from controller
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedAction {
    pub description: String,
    pub target_domain: CapabilityDomain,
    pub method: ImprovementMethod,
    pub urgency: f64,
    pub trajectory_id: String,
}

/// Output from controller cycle
#[derive(Debug)]
pub struct ControllerOutput {
    pub self_assessment: String,
    pub active_trajectory_count: usize,
    pub recommended_actions: Vec<RecommendedAction>,
    pub state: ControllerState,
    pub new_trajectories_planned: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_estimate_creation() {
        let estimate = CapabilityEstimate::new(CapabilityDomain::Reasoning);
        assert_eq!(estimate.domain, CapabilityDomain::Reasoning);
        assert!((estimate.level - 0.5).abs() < 0.01);
        assert!(estimate.uncertainty > 0.0);
    }

    #[test]
    fn test_capability_estimate_update() {
        let mut estimate = CapabilityEstimate::new(CapabilityDomain::Reasoning);
        let initial_level = estimate.level;

        // Update with high performance
        estimate.update(0.9, 0.8);

        // Level should increase
        assert!(estimate.level > initial_level);
        // Uncertainty should decrease
        assert!(estimate.uncertainty < 0.3);
        // Evidence count should increase
        assert_eq!(estimate.evidence_count, 1);
    }

    #[test]
    fn test_capability_estimate_trend() {
        let mut estimate = CapabilityEstimate::new(CapabilityDomain::Learning);

        // Multiple improving updates
        for _ in 0..5 {
            estimate.update(0.9, 0.7);
        }

        // Trend should be positive
        assert!(estimate.trend > 0.0);
    }

    #[test]
    fn test_self_model_creation() {
        let model = SelfModel::new(SelfModelConfig::default());

        // Should have all domains
        assert!(model.capabilities.len() == CapabilityDomain::all().len());

        // Initial model confidence should be moderate
        assert!((model.model_confidence - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_self_model_capability_update() {
        let mut model = SelfModel::new(SelfModelConfig::default());

        model.update_capability(CapabilityDomain::Integration, 0.8, 0.9);

        let estimate = model.get_capability(CapabilityDomain::Integration).unwrap();
        assert!(estimate.level > 0.5);
    }

    #[test]
    fn test_self_model_limitation_tracking() {
        let mut model = SelfModel::new(SelfModelConfig::default());

        model.add_limitation(KnownLimitation {
            description: "Slow memory retrieval".to_string(),
            domain: CapabilityDomain::Memory,
            severity: 0.6,
            cause: "Inefficient indexing".to_string(),
            remediable: true,
            improvement_path: Some("Implement better indexing".to_string()),
            identified_at: Instant::now(),
        });

        assert_eq!(model.limitations.len(), 1);
        assert_eq!(model.get_limitations(CapabilityDomain::Memory).len(), 1);
    }

    #[test]
    fn test_self_model_behavior_prediction() {
        let model = SelfModel::new(SelfModelConfig::default());

        let prediction = model.predict_behavior(&[
            CapabilityDomain::Reasoning,
            CapabilityDomain::Memory,
        ]);

        assert!(prediction.predicted_phi >= 0.0 && prediction.predicted_phi <= 1.0);
        assert!(prediction.confidence >= 0.0);
    }

    #[test]
    fn test_self_model_calibration() {
        let mut model = SelfModel::new(SelfModelConfig::default());

        // Record some predictions
        for i in 0..10 {
            model.record_prediction(PredictionRecord {
                predicted_phi: 0.6,
                actual_phi: 0.6 + (i as f64 * 0.01), // Small errors
                predicted_latency_ms: 100,
                actual_latency_ms: 105,
                context: "test".to_string(),
                timestamp: Instant::now(),
            });
        }

        let stats = model.calibration_stats();
        assert!(stats.mean_phi_error < 0.1);
        assert_eq!(stats.prediction_count, 10);
    }

    #[test]
    fn test_improvement_trajectory_creation() {
        let goal_state = DesiredSelfState {
            target_capabilities: {
                let mut m = HashMap::new();
                m.insert(CapabilityDomain::Reasoning, 0.8);
                m
            },
            target_phi: 0.7,
            motivation_source: DriveType::Competence,
            priority: 0.8,
        };

        let trajectory = ImprovementTrajectory {
            id: "test_traj".to_string(),
            goal_state,
            steps: vec![
                ImprovementStep {
                    description: "Test step".to_string(),
                    target_domain: CapabilityDomain::Reasoning,
                    method: ImprovementMethod::GradientOptimization {
                        target_objective: "Reasoning".to_string(),
                    },
                    prerequisites: Vec::new(),
                    estimated_effect: 0.2,
                    estimated_effort_ms: 5000,
                }
            ],
            estimated_duration_ms: 5000,
            estimated_phi_gain: 0.1,
            risk_assessment: 0.3,
            priority: 0.8,
            progress: 0.0,
            created_at: Instant::now(),
        };

        assert!(!trajectory.is_complete());
        assert!(trajectory.next_step().is_some());
    }

    #[test]
    fn test_unified_controller_creation() {
        let controller = UnifiedImprovementController::new(ControllerConfig::default());
        assert_eq!(controller.state, ControllerState::Idle);
        assert_eq!(controller.active_trajectories.len(), 0);
    }

    #[test]
    fn test_unified_controller_cycle() {
        let mut controller = UnifiedImprovementController::new(ControllerConfig::default());
        let motivation = IntrinsicMotivationSystem::new(MotivationConfig {
            goal_formation_threshold: 0.1,
            ..Default::default()
        });

        // Run a cycle
        let output = controller.cycle(0.5, &motivation, &[]);

        assert!(output.self_assessment.contains("Self-Model"));
        assert_eq!(controller.stats.cycles_run, 1);
    }

    #[test]
    fn test_unified_controller_plans_from_motivation() {
        let mut controller = UnifiedImprovementController::new(ControllerConfig {
            min_gap_for_trajectory: 0.05,
            ..Default::default()
        });

        // Create motivation with active goals
        let mut motivation = IntrinsicMotivationSystem::new(MotivationConfig {
            goal_formation_threshold: 0.1,
            ..Default::default()
        });

        // Force goals to form
        for _ in 0..30 {
            for drive in motivation.drives.values_mut() {
                drive.decay();
            }
        }
        motivation.cycle(0.5);

        // Run controller cycle
        let output = controller.cycle(0.5, &motivation, &[]);

        // Should have planned trajectories if motivation has goals
        if !motivation.active_goals.is_empty() {
            assert!(output.new_trajectories_planned > 0 || controller.active_trajectories.len() > 0);
        }
    }

    #[test]
    fn test_unified_controller_updates_from_bottlenecks() {
        let mut controller = UnifiedImprovementController::new(ControllerConfig::default());
        let motivation = IntrinsicMotivationSystem::new(MotivationConfig::default());

        // Create a bottleneck
        let bottleneck = Bottleneck {
            id: "test_bottleneck".to_string(),
            component: ComponentId::MetaCognition,
            bottleneck_type: BottleneckType::LowPhi,
            severity: 0.7,
            description: "Test phi degradation".to_string(),
            suggested_fix: None,
            detected_at: Instant::now(),
        };

        // Run cycle with bottleneck
        controller.cycle(0.5, &motivation, &[bottleneck]);

        // Should have added limitation
        let limitations = controller.self_model.get_limitations(CapabilityDomain::Integration);
        assert!(!limitations.is_empty());
    }

    #[test]
    fn test_controller_summary() {
        let controller = UnifiedImprovementController::new(ControllerConfig::default());
        let summary = controller.summary();

        assert!(summary.contains("Unified Improvement Controller"));
        assert!(summary.contains("Self-Model"));
        assert!(summary.contains("State:"));
    }

    #[test]
    fn test_desired_state_gap_calculation() {
        let model = SelfModel::new(SelfModelConfig::default());

        let desired = DesiredSelfState {
            target_capabilities: {
                let mut m = HashMap::new();
                m.insert(CapabilityDomain::Reasoning, 0.9);
                m.insert(CapabilityDomain::Memory, 0.8);
                m
            },
            target_phi: 0.8,
            motivation_source: DriveType::Competence,
            priority: 0.7,
        };

        let gap = desired.gap_from(&model);

        // Gap should be positive (current is ~0.5, target is higher)
        assert!(gap > 0.0);
        assert!(gap < 0.5);
    }
}
