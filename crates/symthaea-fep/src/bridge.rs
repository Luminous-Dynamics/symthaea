// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognitive loop integration bridges for FEP active inference.

use std::collections::VecDeque;

use super::agent::{ActiveInferenceAgent, ActiveInferenceAgentConfig};
use super::markov_blanket::{MarkovBoundaryOperator, MarkovPartition, PermeabilityInputs};
use super::motor::{MotorSystem, rand_f64};
use super::td_learning::TemporalDifferenceLearningStats;
use super::types::{
    CognitiveLoopFEPResult, EnhancedFEPCycleResult, MotorCommand, MotorCommandType, Observation,
};

// =============================================================================
// COGNITIVE LOOP FEP BRIDGE
// =============================================================================

/// Integration adapter for cognitive loop
///
/// This provides the interface between the active inference agent and
/// the existing cognitive loop's prediction error system.
#[derive(Debug, Clone)]
pub struct CognitiveLoopFEPBridge {
    /// Active inference agent
    pub agent: ActiveInferenceAgent,
    /// Whether to modulate learning rate based on precision
    pub precision_modulated_learning: bool,
    /// Precision threshold for learning
    pub learning_precision_threshold: f64,
    /// Previous consciousness state for TD learning
    previous_consciousness_state: Option<(f64, f64, f64, f64)>,
    /// Last recommended action
    last_action: Option<usize>,
}

impl CognitiveLoopFEPBridge {
    /// Create new bridge
    pub fn new(config: ActiveInferenceAgentConfig) -> Self {
        Self {
            agent: ActiveInferenceAgent::new(config),
            precision_modulated_learning: true,
            learning_precision_threshold: 0.5,
            previous_consciousness_state: None,
            last_action: None,
        }
    }

    /// Process cognitive loop state with temporal difference learning
    pub fn process(
        &mut self,
        phi: f64,
        integration: f64,
        coherence: f64,
        attention: f64,
    ) -> CognitiveLoopFEPResult {
        // Create observation from consciousness state
        let observation =
            Observation::from_consciousness_state(phi, integration, coherence, attention);

        // Run perception (this now includes TD learning internally)
        let perception = self.agent.perceive(&observation);

        // Select action
        let action_selection = self.agent.select_action();

        // Track action for next TD update
        self.agent.act(action_selection.action);
        self.last_action = Some(action_selection.action);

        // Store current state for next iteration
        self.previous_consciousness_state = Some((phi, integration, coherence, attention));

        // Compute learning rate modulation
        let learning_rate_mod = if self.precision_modulated_learning {
            self.compute_learning_modulation()
        } else {
            1.0
        };

        // Should learning occur?
        let should_learn = perception.precision > self.learning_precision_threshold
            && perception.free_energy.prediction_error < 0.8;

        // Get TD error if available
        let td_error = self.agent.td_stats().map(|s| s.avg_td_error).unwrap_or(0.0);

        CognitiveLoopFEPResult {
            free_energy: perception.free_energy.total,
            prediction_error: perception.free_energy.prediction_error,
            precision_weighted_error: self
                .agent
                .precision
                .weight_error(perception.free_energy.prediction_error),
            recommended_action: action_selection.action,
            is_surprised: self.agent.is_surprised(),
            learning_rate_modulation: learning_rate_mod,
            should_learn,
            exploration_mode: action_selection.is_exploratory,
            belief_confidence: perception.updated_belief.confidence(),
            epistemic_value: action_selection.epistemic_value,
            pragmatic_value: action_selection.pragmatic_value,
            td_error,
            model_confidence: self
                .agent
                .td_stats()
                .map(|s| (s.avg_transition_confidence + s.avg_likelihood_confidence) / 2.0)
                .unwrap_or(0.5),
        }
    }

    /// Process with explicit action feedback
    ///
    /// Call this when you know what action was actually taken
    /// (useful for closed-loop control)
    pub fn process_with_action(
        &mut self,
        phi: f64,
        integration: f64,
        coherence: f64,
        attention: f64,
        executed_action: usize,
    ) -> CognitiveLoopFEPResult {
        // Track the executed action before processing
        self.agent.act(executed_action);
        self.last_action = Some(executed_action);

        // Now process the observation
        self.process(phi, integration, coherence, attention)
    }

    /// Signal end of episode (e.g., conversation turn, task completion)
    pub fn end_episode(&mut self) {
        self.agent.end_episode();
        self.previous_consciousness_state = None;
        self.last_action = None;
    }

    /// Compute learning rate modulation based on free energy
    fn compute_learning_modulation(&self) -> f64 {
        let precision = self.agent.precision.perceptual_precision();
        let stability = self.agent.precision.stability();

        // Also factor in TD learning confidence
        let td_confidence = self
            .agent
            .td_stats()
            .map(|s| s.avg_prediction_accuracy)
            .unwrap_or(0.5);

        // High precision + high stability + high TD confidence = boost learning
        // Low values = reduce learning
        ((precision * stability * td_confidence).powf(1.0 / 3.0))
            .max(0.1)
            .min(2.0)
    }

    /// Set goals for the agent
    pub fn set_goals(
        &mut self,
        preferred_phi: f64,
        preferred_integration: f64,
        preferred_coherence: f64,
        preferred_attention: f64,
    ) {
        self.agent.set_goals(
            vec![
                preferred_phi,
                preferred_integration,
                preferred_coherence,
                preferred_attention,
            ],
            2.0,
        );
    }

    /// Get temporal difference learning statistics
    pub fn td_stats(&self) -> Option<TemporalDifferenceLearningStats> {
        self.agent.td_stats()
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.agent.reset();
        self.previous_consciousness_state = None;
        self.last_action = None;
    }
}

// =============================================================================
// ENHANCED FEP BRIDGE
// =============================================================================

/// Enhanced FEP bridge with motor system integration.
///
/// `EnhancedFEPBridge` is the primary interface for integrating Free Energy Principle
/// active inference into the Symthaea cognitive loop. It combines:
///
/// - **Perception**: Processing consciousness observations (phi, integration, coherence, attention)
/// - **Action Selection**: Choosing motor commands that minimize expected free energy
/// - **Motor Execution**: Executing commands through the [`MotorSystem`]
/// - **Learning**: Updating the generative model based on prediction errors
///
/// # Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────────┐
/// │                   EnhancedFEPBridge                             │
/// ├─────────────────────────────────────────────────────────────────┤
/// │                                                                 │
/// │  ┌──────────────────────┐    ┌──────────────────────┐          │
/// │  │  CognitiveLoopFEP    │    │    MotorSystem       │          │
/// │  │  Bridge (core)       │    │                      │          │
/// │  │                      │    │  • execute()         │          │
/// │  │  • process()         │───▶│  • proprioception    │          │
/// │  │  • select_action()   │    │  • command_history   │          │
/// │  │  • TD learning       │    │                      │          │
/// │  └──────────────────────┘    └──────────────────────┘          │
/// │                                      │                         │
/// │                    ┌─────────────────┘                         │
/// │                    ▼                                           │
/// │           ┌───────────────────┐                                │
/// │           │  Learning Signal  │                                │
/// │           │  Computation      │                                │
/// │           │                   │                                │
/// │           │  TD error × 0.4   │                                │
/// │           │  Motor err × 0.3  │                                │
/// │           │  FE × 0.3         │                                │
/// │           └───────────────────┘                                │
/// └─────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Precision-Gated Learning
///
/// Learning only occurs when model confidence exceeds the threshold (default 0.4).
/// This prevents learning from noisy or uncertain observations:
///
/// ```rust,ignore
/// let mut bridge = EnhancedFEPBridge::new(config, 4);
///
/// // Customize precision gating
/// bridge.set_precision_gated_learning(true, 0.5); // Stricter threshold
///
/// let result = bridge.cycle(phi, integration, coherence, attention);
/// if result.should_learn {
///     // Model confidence > 0.5, safe to learn
///     downstream_learner.apply_gradient(result.learning_signal);
/// }
/// ```
///
/// # Complete Cycle Example
///
/// ```rust,ignore
/// use symthaea::consciousness::fep_active_inference::{
///     EnhancedFEPBridge, ActiveInferenceAgentConfig, MotorCommandType
/// };
///
/// let config = ActiveInferenceAgentConfig::default();
/// let mut bridge = EnhancedFEPBridge::new(config, 4); // 4D motor state
///
/// // Main cognitive loop
/// loop {
///     // Get consciousness metrics from upstream
///     let (phi, integration, coherence, attention) = get_consciousness_state();
///
///     // Full perception-action-learning cycle
///     let result = bridge.cycle(phi, integration, coherence, attention);
///
///     // Handle motor command
///     match result.motor_command.command_type {
///         MotorCommandType::AttentionShift => {
///             // Redirect attention based on parameters
///             let direction = &result.motor_command.parameters;
///             shift_attention(direction);
///         }
///         MotorCommandType::ExplorationTrigger => {
///             // Increase state variability
///             enable_exploration_mode();
///         }
///         MotorCommandType::MemoryConsolidate => {
///             // Strengthen current representations
///             consolidate_working_memory();
///         }
///         _ => {}
///     }
///
///     // Apply learning if appropriate
///     if result.should_learn {
///         apply_learning(result.learning_signal);
///     }
///
///     // Monitor action-outcome coupling
///     println!("Coupling quality: {:.2}", result.action_outcome_coupling);
/// }
/// ```
///
/// # Episode Boundaries
///
/// Call [`end_episode`](Self::end_episode) at natural task boundaries to:
/// - Reset eligibility traces in the TD learner
/// - Clear motor command history
/// - Reset action-outcome coupling tracker
///
/// This is important for episodic tasks where the future shouldn't bootstrap
/// from the past across episode boundaries.
#[derive(Debug, Clone)]
pub struct EnhancedFEPBridge {
    /// Core FEP bridge containing the generative model and active inference agent.
    pub core: CognitiveLoopFEPBridge,

    /// Motor system for command execution and proprioceptive feedback.
    pub motor: MotorSystem,

    /// Markov Boundary Operator — dynamic permeability of the sensory/active blanket.
    /// Science: Friston (2013), Kirchhoff et al. (2018).
    pub blanket: MarkovBoundaryOperator,

    /// Learning signal output (0.0-1.0) for downstream systems.
    /// Combines TD error, motor prediction error, and free energy.
    learning_signal: f64,

    /// Whether to gate learning based on model precision/confidence.
    /// When true, learning only occurs if confidence > threshold.
    precision_gated_learning: bool,

    /// Minimum precision/confidence required for learning (default 0.4).
    learning_precision_threshold: f64,

    /// History of (action, outcome_error) pairs for computing action-outcome coupling.
    /// High coupling (low avg error) indicates the model predicts action effects well.
    pub(crate) action_outcome_history: VecDeque<(MotorCommandType, f64)>,
}

impl EnhancedFEPBridge {
    /// Create a new enhanced FEP bridge
    pub fn new(config: ActiveInferenceAgentConfig, motor_state_dim: usize) -> Self {
        let state_dim = config.state_dim;
        Self {
            core: CognitiveLoopFEPBridge::new(config),
            motor: MotorSystem::new(motor_state_dim),
            blanket: MarkovBoundaryOperator::new(MarkovPartition {
                internal_dim: state_dim,
                sensory_dim: 4,
                active_dim: 8,
            }),
            learning_signal: 0.0,
            precision_gated_learning: true,
            learning_precision_threshold: 0.4,
            action_outcome_history: VecDeque::with_capacity(100),
        }
    }

    /// Update the blanket permeability from neuromodulator state.
    pub fn update_blanket_permeability(&mut self, inputs: &PermeabilityInputs) {
        self.blanket.compute_permeability(inputs);
    }

    /// Full perception-action-learning cycle
    pub fn cycle(
        &mut self,
        phi: f64,
        integration: f64,
        coherence: f64,
        attention: f64,
    ) -> EnhancedFEPCycleResult {
        // 1. Gate observation through Markov blanket
        let raw_obs = Observation::from_consciousness_state(phi, integration, coherence, attention);
        let gated_obs = self
            .blanket
            .gate_observation(&raw_obs, &self.core.agent.belief);
        let gated_phi = gated_obs.values.first().copied().unwrap_or(phi);
        let gated_integration = gated_obs.values.get(1).copied().unwrap_or(integration);
        let gated_coherence = gated_obs.values.get(2).copied().unwrap_or(coherence);
        let gated_attention = gated_obs.values.get(3).copied().unwrap_or(attention);

        // 2. Process gated observation through FEP
        let fep_result = self.core.process(
            gated_phi,
            gated_integration,
            gated_coherence,
            gated_attention,
        );

        // 2. Generate motor command from action
        let command_type = MotorCommandType::from_action_index(fep_result.recommended_action);
        let command = MotorCommand::new(command_type, fep_result.belief_confidence)
            .with_confidence(fep_result.model_confidence)
            .with_expected_precision(1.0 - fep_result.precision_weighted_error)
            .with_predicted_outcome(vec![
                phi + (rand_f64() - 0.5) * 0.1,
                integration,
                coherence,
                attention,
            ]);

        // 3. Execute motor command
        let motor_outcome = self.motor.execute(command.clone());

        // 4. Update action-outcome history for causal learning
        if self.action_outcome_history.len() >= 100 {
            self.action_outcome_history.pop_front();
        }
        self.action_outcome_history
            .push_back((command_type, motor_outcome.prediction_error));

        // 5. Compute learning signal, modulated by blanket permeability
        let raw_learning_signal = self.compute_learning_signal(&fep_result, &motor_outcome);
        self.learning_signal = self.blanket.modulate_learning_rate(raw_learning_signal);

        // 6. Determine if learning should occur
        let should_learn = if self.precision_gated_learning {
            fep_result.model_confidence > self.learning_precision_threshold
                && fep_result.should_learn
        } else {
            fep_result.should_learn
        };

        EnhancedFEPCycleResult {
            fep_result,
            motor_command: command,
            motor_outcome,
            learning_signal: self.learning_signal,
            should_learn,
            action_outcome_coupling: self.action_outcome_coupling(),
        }
    }

    /// Compute learning signal from FEP and motor results
    fn compute_learning_signal(
        &self,
        fep: &CognitiveLoopFEPResult,
        motor: &super::types::MotorOutcome,
    ) -> f64 {
        // Learning signal combines:
        // 1. TD error (temporal prediction error)
        // 2. Motor prediction error
        // 3. Free energy (surprise)

        let td_weight = 0.4;
        let motor_weight = 0.3;
        let fe_weight = 0.3;

        let td_signal = fep.td_error.abs().min(1.0);
        let motor_signal = motor.prediction_error.min(1.0);
        let fe_signal = (fep.free_energy.abs() / 10.0).min(1.0);

        // Learning should increase with surprise/error, but be gated by precision
        let raw_signal =
            td_weight * td_signal + motor_weight * motor_signal + fe_weight * fe_signal;

        // Precision-weight the learning signal
        raw_signal * fep.learning_rate_modulation
    }

    /// Compute action-outcome coupling (how well actions predict outcomes)
    fn action_outcome_coupling(&self) -> f64 {
        if self.action_outcome_history.is_empty() {
            return 0.5; // Neutral
        }

        // Lower average error = better coupling
        let avg_error: f64 = self
            .action_outcome_history
            .iter()
            .map(|(_, e)| *e)
            .sum::<f64>()
            / self.action_outcome_history.len() as f64;

        (1.0 - avg_error).clamp(0.0, 1.0)
    }

    /// Get the current learning signal
    pub fn learning_signal(&self) -> f64 {
        self.learning_signal
    }

    /// Set precision-gated learning
    pub fn set_precision_gated_learning(&mut self, enabled: bool, threshold: f64) {
        self.precision_gated_learning = enabled;
        self.learning_precision_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Signal end of episode
    pub fn end_episode(&mut self) {
        self.core.end_episode();
        self.motor.reset();
        self.action_outcome_history.clear();
        self.learning_signal = 0.0;
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.core.reset();
        self.motor.reset();
        self.action_outcome_history.clear();
        self.learning_signal = 0.0;
        let partition = self.blanket.partition().clone();
        self.blanket = MarkovBoundaryOperator::new(partition);
    }
}
