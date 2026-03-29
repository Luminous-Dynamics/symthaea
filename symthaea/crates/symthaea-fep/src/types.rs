// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core data types for the FEP active inference system.

use serde::{Deserialize, Serialize};

// =============================================================================
// OBSERVATION MODEL
// =============================================================================

/// Observation from the environment/internal state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Raw observation vector
    pub values: Vec<f64>,
    /// Observation precision (inverse variance, confidence in observation)
    pub precision: f64,
    /// Timestamp (monotonic counter)
    pub timestamp: u64,
    /// Modality (e.g., "visual", "interoceptive", "cognitive")
    pub modality: String,
}

impl Observation {
    /// Create new observation
    pub fn new(values: Vec<f64>, precision: f64, modality: &str) -> Self {
        Self {
            values,
            precision,
            timestamp: 0,
            modality: modality.to_string(),
        }
    }

    /// Create from consciousness state observables
    pub fn from_consciousness_state(
        phi: f64,
        integration: f64,
        coherence: f64,
        attention: f64,
    ) -> Self {
        Self {
            values: vec![phi, integration, coherence, attention],
            precision: 1.0,
            timestamp: 0,
            modality: "consciousness".to_string(),
        }
    }

    /// Dimension of observation
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

// =============================================================================
// HIDDEN STATE (Beliefs)
// =============================================================================

/// Hidden state representation (beliefs about the world)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    /// Mean of belief distribution (expected hidden state)
    pub mean: Vec<f64>,
    /// Precision (inverse variance) for each dimension
    pub precision: Vec<f64>,
    /// Mode probabilities for discrete states (if applicable)
    pub mode_probs: Vec<f64>,
    /// Current mode (for discrete state space)
    pub current_mode: usize,
}

impl HiddenState {
    /// Create new hidden state with given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.5; dim],
            precision: vec![1.0; dim],
            mode_probs: vec![1.0],
            current_mode: 0,
        }
    }

    /// Create with discrete modes
    pub fn with_modes(continuous_dim: usize, num_modes: usize) -> Self {
        let mode_probs = vec![1.0 / num_modes as f64; num_modes];
        Self {
            mean: vec![0.5; continuous_dim],
            precision: vec![1.0; continuous_dim],
            mode_probs,
            current_mode: 0,
        }
    }

    /// Get variance (inverse of precision)
    pub fn variance(&self) -> Vec<f64> {
        self.precision.iter().map(|p| 1.0 / p.max(0.001)).collect()
    }

    /// Compute entropy of the continuous belief (Gaussian)
    pub fn entropy(&self) -> f64 {
        use std::f64::consts::PI;
        let dim = self.mean.len() as f64;
        // Entropy of multivariate Gaussian: 0.5 * (d + d*ln(2π) + ln|Σ|)
        let log_det: f64 = self.precision.iter().map(|p| -p.max(0.001).ln()).sum();
        0.5 * (dim + dim * (2.0 * PI).ln() + log_det)
    }

    /// Compute discrete entropy over modes
    pub fn mode_entropy(&self) -> f64 {
        -self
            .mode_probs
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>()
    }

    /// Total uncertainty (continuous + discrete)
    pub fn total_uncertainty(&self) -> f64 {
        self.entropy() + self.mode_entropy()
    }

    /// Confidence (inverse of uncertainty, normalized)
    pub fn confidence(&self) -> f64 {
        let avg_precision = self.precision.iter().sum::<f64>() / self.precision.len() as f64;
        let max_mode_prob = self.mode_probs.iter().cloned().fold(0.0, f64::max);
        (avg_precision * max_mode_prob).min(1.0)
    }
}

// =============================================================================
// FREE ENERGY COMPONENTS
// =============================================================================

/// Components of free energy computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyComponents {
    /// Total variational free energy
    pub total: f64,
    /// Accuracy term (expected log likelihood)
    pub accuracy: f64,
    /// Complexity term (KL divergence from prior)
    pub complexity: f64,
    /// Surprise (negative log evidence)
    pub surprise: f64,
    /// Prediction error magnitude
    pub prediction_error: f64,
}

// =============================================================================
// PRECISION SNAPSHOT
// =============================================================================

/// Snapshot of precision values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSnapshot {
    pub sensory: f64,
    pub prior: f64,
    pub state: f64,
    pub action: f64,
    pub timestamp: u64,
}

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result of perception step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionResult {
    /// Updated belief state
    pub updated_belief: HiddenState,
    /// Free energy components
    pub free_energy: FreeEnergyComponents,
    /// Current precision
    pub precision: f64,
    /// Total belief change
    pub belief_change: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Result of action selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSelectionResult {
    /// Selected action
    pub action: usize,
    /// Expected free energy of selected action
    pub expected_free_energy: f64,
    /// Probability distribution over actions
    pub action_probabilities: Vec<f64>,
    /// Whether this is an exploratory action
    pub is_exploratory: bool,
    /// Pragmatic value component
    pub pragmatic_value: f64,
    /// Epistemic value component
    pub epistemic_value: f64,
    /// Predicted state after action
    pub predicted_state: HiddenState,
}

/// Outcome of action execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// Action taken
    pub action: usize,
    /// Predicted next state
    pub predicted_next_state: HiddenState,
    /// Expected observation
    pub expected_observation: Vec<f64>,
    /// Timestamp
    pub timestamp: u64,
}

/// Summary of active inference agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceSummary {
    /// Current belief mean
    pub belief_mean: Vec<f64>,
    /// Belief confidence
    pub belief_confidence: f64,
    /// Current free energy
    pub free_energy: f64,
    /// Current precision
    pub precision: f64,
    /// Exploration rate
    pub exploration_rate: f64,
    /// Total perception cycles
    pub total_cycles: u64,
}

/// Statistics for Active Inference Agent
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveInferenceAgentStats {
    /// Total perception cycles
    pub perception_cycles: u64,
    /// Total actions taken
    pub actions_taken: u64,
    /// Average free energy
    pub avg_free_energy: f64,
    /// Average prediction error
    pub avg_prediction_error: f64,
    /// Average precision
    pub avg_precision: f64,
    /// Exploration rate (epistemic actions / total)
    pub exploration_rate: f64,
    /// Model learning updates
    pub model_updates: u64,
    /// Epistemic actions taken
    pub(crate) epistemic_actions: u64,
    /// TD learning updates
    pub td_updates: u64,
    /// Average TD error
    pub avg_td_error: f64,
    /// Transition model accuracy
    pub transition_accuracy: f64,
}

// =============================================================================
// COGNITIVE LOOP FEP RESULT
// =============================================================================

/// Result from cognitive loop FEP processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoopFEPResult {
    /// Current free energy
    pub free_energy: f64,
    /// Raw prediction error
    pub prediction_error: f64,
    /// Precision-weighted prediction error
    pub precision_weighted_error: f64,
    /// Recommended action (index)
    pub recommended_action: usize,
    /// Whether agent is surprised
    pub is_surprised: bool,
    /// Learning rate modulation factor
    pub learning_rate_modulation: f64,
    /// Should learning occur this cycle?
    pub should_learn: bool,
    /// Is agent in exploration mode?
    pub exploration_mode: bool,
    /// Confidence in current beliefs
    pub belief_confidence: f64,
    /// Epistemic value of current state
    pub epistemic_value: f64,
    /// Pragmatic value of current state
    pub pragmatic_value: f64,
    /// Temporal difference error (average)
    pub td_error: f64,
    /// Model confidence (average of transition and likelihood confidence)
    pub model_confidence: f64,
}

// =============================================================================
// EXPECTED FREE ENERGY RESULT
// =============================================================================

/// Result of expected free energy computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFreeEnergyResult {
    /// Action evaluated
    pub action: usize,
    /// Total expected free energy (lower is better)
    pub total: f64,
    /// Pragmatic component (goal-directedness)
    pub pragmatic: f64,
    /// Epistemic component (uncertainty reduction)
    pub epistemic: f64,
    /// Novelty bonus
    pub novelty: f64,
    /// Predicted state after action
    pub predicted_state: HiddenState,
    /// Expected observation after action
    pub expected_observation: Vec<f64>,
}

// =============================================================================
// MOTOR COMMAND TYPES
// =============================================================================

/// Motor command types for embodied action.
///
/// These represent the 8 possible motor outputs from the active inference system.
/// The FEP bridge selects a command type based on which action minimizes expected
/// free energy. In a cognitive architecture, these translate to changes in attention,
/// learning parameters, or actual motor commands in an embodied system.
///
/// # Command Selection
///
/// The active inference agent evaluates expected free energy for each action:
///
/// ```text
/// G(a) = E_q[ln q(s') - ln p(o',s') | a]
///      = Pragmatic Value + Epistemic Value
/// ```
///
/// The action with lowest G(a) is selected and mapped to the corresponding
/// `MotorCommandType` via [`MotorCommandType::from_action_index`].
///
/// # When Each Command Fires
///
/// | Command | Typical Trigger Condition |
/// |---------|---------------------------|
/// | `AttentionShift` | High precision error in specific sensory modality |
/// | `LearningRateAdjust` | Model confidence changing rapidly |
/// | `ExplorationTrigger` | Low epistemic value, high state uncertainty |
/// | `ReflectionInitiate` | High free energy but stable belief state |
/// | `MemoryConsolidate` | High confidence, consistently low prediction error |
/// | `ExpectationReset` | Persistent high prediction error (model mismatch) |
/// | `MotorOutput` | Pragmatic goals require external action |
/// | `NoOp` | System near equilibrium, minimal free energy |
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::consciousness::fep_active_inference::MotorCommandType;
///
/// let action_index = 2; // From FEP action selection
/// let cmd = MotorCommandType::from_action_index(action_index);
/// assert_eq!(cmd, MotorCommandType::ExplorationTrigger);
///
/// // Convert back
/// assert_eq!(cmd.to_action_index(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotorCommandType {
    /// Modulate attention focus (shift attention to different inputs).
    ///
    /// Fires when precision-weighted error is high in a specific modality,
    /// indicating that attention resources should be redirected.
    /// Updates the proprioceptive attention dimension.
    AttentionShift,

    /// Adjust learning rate based on precision dynamics.
    ///
    /// Fires when model confidence is changing, either increasing
    /// (reduce learning rate) or decreasing (increase learning rate).
    /// Implements precision-weighted learning modulation.
    LearningRateAdjust,

    /// Trigger exploration to reduce state uncertainty.
    ///
    /// Fires when epistemic value is high (information gain potential)
    /// but pragmatic value is low. Causes increased variability in
    /// proprioceptive state to gather diverse observations.
    ExplorationTrigger,

    /// Initiate metacognitive reflection.
    ///
    /// Fires when free energy is high but beliefs are relatively stable,
    /// suggesting the need for higher-order reasoning about the current
    /// situation rather than immediate action.
    ReflectionInitiate,

    /// Consolidate current representations into long-term memory.
    ///
    /// Fires when model confidence is high and prediction error is
    /// consistently low, indicating stable learned patterns that
    /// should be strengthened.
    MemoryConsolidate,

    /// Reset prediction expectations.
    ///
    /// Fires when prediction error remains persistently high despite
    /// learning, suggesting fundamental model mismatch that requires
    /// clearing cached predictions rather than incremental updates.
    ExpectationReset,

    /// Execute external motor action.
    ///
    /// Fires when pragmatic goals require physical/external action.
    /// In embodied systems, this translates to actual motor commands.
    /// In cognitive systems, may trigger shell commands or API calls.
    MotorOutput,

    /// No operation - maintain current state.
    ///
    /// Fires when the system is near equilibrium with minimal free energy.
    /// Indicates the current policy is adequate and no change is needed.
    NoOp,
}

impl MotorCommandType {
    /// Convert action index to motor command type
    pub fn from_action_index(action: usize) -> Self {
        match action {
            0 => MotorCommandType::AttentionShift,
            1 => MotorCommandType::LearningRateAdjust,
            2 => MotorCommandType::ExplorationTrigger,
            3 => MotorCommandType::ReflectionInitiate,
            4 => MotorCommandType::MemoryConsolidate,
            5 => MotorCommandType::ExpectationReset,
            6 => MotorCommandType::MotorOutput,
            _ => MotorCommandType::NoOp,
        }
    }

    /// Get action index from motor command type
    pub fn to_action_index(&self) -> usize {
        match self {
            MotorCommandType::AttentionShift => 0,
            MotorCommandType::LearningRateAdjust => 1,
            MotorCommandType::ExplorationTrigger => 2,
            MotorCommandType::ReflectionInitiate => 3,
            MotorCommandType::MemoryConsolidate => 4,
            MotorCommandType::ExpectationReset => 5,
            MotorCommandType::MotorOutput => 6,
            MotorCommandType::NoOp => 7,
        }
    }
}

/// A motor command with parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorCommand {
    /// Type of motor command
    pub command_type: MotorCommandType,
    /// Intensity of the command (0.0-1.0)
    pub intensity: f64,
    /// Directional parameters (e.g., attention shift direction)
    pub parameters: Vec<f64>,
    /// Confidence in this command
    pub confidence: f64,
    /// Expected precision of the command outcome
    pub expected_precision: f64,
    /// Predicted outcome (in observation space)
    pub predicted_outcome: Option<Vec<f64>>,
}

impl MotorCommand {
    /// Create a new motor command
    pub fn new(command_type: MotorCommandType, intensity: f64) -> Self {
        Self {
            command_type,
            intensity: intensity.clamp(0.0, 1.0),
            parameters: Vec::new(),
            confidence: 0.5,
            expected_precision: 0.5,
            predicted_outcome: None,
        }
    }

    /// Add parameters to the command
    pub fn with_parameters(mut self, params: Vec<f64>) -> Self {
        self.parameters = params;
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set expected precision
    pub fn with_expected_precision(mut self, precision: f64) -> Self {
        self.expected_precision = precision.clamp(0.0, 1.0);
        self
    }

    /// Set predicted outcome
    pub fn with_predicted_outcome(mut self, outcome: Vec<f64>) -> Self {
        self.predicted_outcome = Some(outcome);
        self
    }

    /// Check if this is a meaningful command (not NoOp with low intensity)
    pub fn is_meaningful(&self) -> bool {
        self.command_type != MotorCommandType::NoOp || self.intensity > 0.5
    }
}

/// Outcome of motor command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorOutcome {
    /// Type of command that was executed
    pub command_type: MotorCommandType,
    /// Actual intensity of execution
    pub executed_intensity: f64,
    /// Whether execution was successful
    pub success: bool,
    /// Proprioceptive feedback after execution
    pub proprioceptive_feedback: Vec<f64>,
    /// Prediction error (if outcome was predicted)
    pub prediction_error: f64,
}

/// Motor command statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorCommandStats {
    /// Total commands executed
    pub total_commands: usize,
    /// Number of meaningful commands
    pub meaningful_commands: usize,
    /// Average intensity
    pub avg_intensity: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Average prediction error
    pub avg_prediction_error: f64,
}

/// Result from enhanced FEP cycle
#[derive(Debug, Clone)]
pub struct EnhancedFEPCycleResult {
    /// Core FEP result
    pub fep_result: CognitiveLoopFEPResult,
    /// Motor command that was issued
    pub motor_command: MotorCommand,
    /// Outcome of motor execution
    pub motor_outcome: MotorOutcome,
    /// Learning signal for downstream systems
    pub learning_signal: f64,
    /// Whether learning should occur this cycle
    pub should_learn: bool,
    /// Action-outcome coupling quality
    pub action_outcome_coupling: f64,
}
