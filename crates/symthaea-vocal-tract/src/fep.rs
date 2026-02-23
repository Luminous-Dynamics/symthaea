//! FEP active inference agent for vocal tract control.
//!
//! Wraps `ActiveInferenceAgent` from `symthaea-fep` to modulate the vocal tract
//! controller's time constants and learning rate based on voice output quality.
//!
//! This module is decoupled from the main crate's `VoiceOutputMetrics` — it uses
//! [`VocalTractObservation`] instead, a 6D struct that captures the same quality
//! dimensions without pulling in the full voice feedback system.
//!
//! # Action Space (6 actions)
//!
//! | Action | Effect | When |
//! |--------|--------|------|
//! | DropTau | Faster formant transitions | High prediction error |
//! | RaiseTau | Smoother sustained vowels | Low prediction error |
//! | BoostLR | Faster learning | Initial adaptation |
//! | ReduceLR | Fine-tuning | Converged |
//! | ShiftEmphasis | Increase emphasis | Low articulation |
//! | ExplorationBurst | Random perturbation | Stuck in local minimum |
//!
//! # Observation Space (6D from VocalTractObservation)
//!
//! articulation, formant_accuracy, pitch_stability, coarticulation,
//! duration_accuracy, energy_consistency

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, TemporalDifferenceLearningConfig,
};

/// Vocal tract quality observation for the FEP agent (6D).
///
/// Decouples the FEP agent from the main crate's `VoiceOutputMetrics`.
#[derive(Debug, Clone, Default)]
pub struct VocalTractObservation {
    pub articulation_score: f64,
    pub formant_accuracy: f64,
    pub pitch_stability: f64,
    pub coarticulation_smoothness: f64,
    pub duration_accuracy: f64,
    pub energy_consistency: f64,
}

/// Actions the FEP agent can take to modulate the vocal tract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocalAction {
    /// Decrease tau -> faster formant transitions (high surprise).
    DropTau = 0,
    /// Increase tau -> smoother sustained vowels (low surprise).
    RaiseTau = 1,
    /// Increase learning rate -> faster adaptation.
    BoostLR = 2,
    /// Decrease learning rate -> fine-tuning.
    ReduceLR = 3,
    /// Shift emphasis -> more assertive articulation.
    ShiftEmphasis = 4,
    /// Exploration burst -> random perturbation to escape local minima.
    ExplorationBurst = 5,
}

impl VocalAction {
    /// Convert from action index.
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::DropTau,
            1 => Self::RaiseTau,
            2 => Self::BoostLR,
            3 => Self::ReduceLR,
            4 => Self::ShiftEmphasis,
            _ => Self::ExplorationBurst,
        }
    }
}

/// Result of a FEP tick: modulation factors for the vocal tract controller.
#[derive(Debug, Clone)]
pub struct VocalTractFepResult {
    /// Tau modulation factor (< 1.0 = faster, > 1.0 = slower).
    pub tau_factor: f32,
    /// Learning rate modulation factor.
    pub learning_rate_factor: f32,
    /// Emphasis modulation factor.
    pub emphasis_factor: f32,
    /// Current free energy estimate.
    pub free_energy: f64,
    /// Current prediction error.
    pub prediction_error: f64,
    /// Selected action.
    pub action: VocalAction,
}

impl Default for VocalTractFepResult {
    fn default() -> Self {
        Self {
            tau_factor: 1.0,
            learning_rate_factor: 1.0,
            emphasis_factor: 1.0,
            free_energy: 0.0,
            prediction_error: 0.0,
            action: VocalAction::RaiseTau,
        }
    }
}

/// FEP active inference agent for vocal tract modulation.
pub struct VocalTractFepAgent {
    agent: ActiveInferenceAgent,
    tick_count: u64,
    /// Last action taken (for learn_from_outcome closure).
    last_action: Option<usize>,
}

impl VocalTractFepAgent {
    /// Create a new FEP agent for vocal tract control.
    pub fn new() -> Self {
        let config = ActiveInferenceAgentConfig {
            state_dim: 6,
            obs_dim: 6,
            num_actions: 6,
            inference_iterations: 5,
            belief_learning_rate: 0.1,
            planning_horizon: 3,
            action_temperature: 1.0,
            enable_model_learning: true,
            enable_td_learning: true,
            td_config: TemporalDifferenceLearningConfig {
                initial_learning_rate: 0.05,
                gamma: 0.95,
                trace_decay: 0.8,
                ..Default::default()
            },
        };

        Self {
            agent: ActiveInferenceAgent::new(config),
            tick_count: 0,
            last_action: None,
        }
    }

    /// Run one FEP tick: observe voice quality -> select action -> return modulation.
    ///
    /// Call at 10Hz (every 20 motor frames at 200Hz).
    pub fn tick(&mut self, obs: &VocalTractObservation) -> VocalTractFepResult {
        self.tick_count += 1;

        // Construct 6D observation from VocalTractObservation
        let observation = Observation {
            values: vec![
                obs.articulation_score,
                obs.formant_accuracy,
                obs.pitch_stability,
                obs.coarticulation_smoothness,
                obs.duration_accuracy,
                obs.energy_consistency,
            ],
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "vocal_tract".to_string(),
        };

        // Perceive -> update belief
        let perception = self.agent.perceive(&observation);

        // Select action
        let action_result = self.agent.select_action();
        let action_idx = action_result.action;
        let action = VocalAction::from_index(action_idx);

        // Execute action (updates model)
        let _outcome = self.agent.act(action_idx);
        self.last_action = Some(action_idx);

        // Convert action to modulation factors
        let (tau_factor, lr_factor, emphasis_factor) = match action {
            VocalAction::DropTau => (0.8, 1.0, 1.0),
            VocalAction::RaiseTau => (1.2, 1.0, 1.0),
            VocalAction::BoostLR => (1.0, 1.5, 1.0),
            VocalAction::ReduceLR => (1.0, 0.7, 1.0),
            VocalAction::ShiftEmphasis => (1.0, 1.0, 1.3),
            VocalAction::ExplorationBurst => (0.9, 1.2, 1.1),
        };

        VocalTractFepResult {
            tau_factor,
            learning_rate_factor: lr_factor,
            emphasis_factor,
            free_energy: perception.free_energy.total,
            prediction_error: perception.free_energy.prediction_error,
            action,
        }
    }

    /// Learn from the outcome of the previous action.
    ///
    /// Closes the active inference loop: after observing post-action voice quality,
    /// feeds the outcome back to the agent's TD learner via `learn_from_outcome()`.
    pub fn learn(&mut self, obs: &VocalTractObservation) {
        if let Some(action) = self.last_action {
            let observation = Observation {
                values: vec![
                    obs.articulation_score,
                    obs.formant_accuracy,
                    obs.pitch_stability,
                    obs.coarticulation_smoothness,
                    obs.duration_accuracy,
                    obs.energy_consistency,
                ],
                precision: 1.0,
                timestamp: self.tick_count,
                modality: "vocal_tract_outcome".to_string(),
            };
            self.agent.learn_from_outcome(action, &observation);
        }
    }

    /// Get current free energy.
    pub fn free_energy(&self) -> Option<f64> {
        self.agent.last_fe_components.as_ref().map(|fe| fe.total)
    }

    /// Get tick count.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Reset agent state.
    pub fn reset(&mut self) {
        self.agent.reset();
        self.tick_count = 0;
        self.last_action = None;
    }

    /// Access the underlying agent stats (for testing/inspection).
    pub fn stats(&self) -> &symthaea_fep::ActiveInferenceAgentStats {
        &self.agent.stats
    }
}

impl Default for VocalTractFepAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_action_selection() {
        let mut agent = VocalTractFepAgent::new();
        let obs = VocalTractObservation {
            articulation_score: 0.8,
            formant_accuracy: 0.7,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.8,
            duration_accuracy: 0.7,
            energy_consistency: 0.8,
        };

        let result = agent.tick(&obs);

        // Should produce valid modulation factors
        assert!(result.tau_factor > 0.0);
        assert!(result.learning_rate_factor > 0.0);
        assert!(result.emphasis_factor > 0.0);
        assert!(result.free_energy.is_finite());
    }

    #[test]
    fn test_fep_repeated_ticks() {
        let mut agent = VocalTractFepAgent::new();
        let obs = VocalTractObservation {
            articulation_score: 0.9,
            formant_accuracy: 0.9,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.9,
            duration_accuracy: 0.9,
            energy_consistency: 0.9,
        };

        // Run several ticks -- should not panic
        for _ in 0..20 {
            let result = agent.tick(&obs);
            assert!(result.free_energy.is_finite());
        }

        assert_eq!(agent.tick_count(), 20);
    }

    #[test]
    fn test_fep_reset() {
        let mut agent = VocalTractFepAgent::new();
        let obs = VocalTractObservation::default();

        agent.tick(&obs);
        agent.tick(&obs);
        assert_eq!(agent.tick_count(), 2);

        agent.reset();
        assert_eq!(agent.tick_count(), 0);
    }

    #[test]
    fn test_vocal_action_from_index() {
        assert_eq!(VocalAction::from_index(0), VocalAction::DropTau);
        assert_eq!(VocalAction::from_index(1), VocalAction::RaiseTau);
        assert_eq!(VocalAction::from_index(2), VocalAction::BoostLR);
        assert_eq!(VocalAction::from_index(3), VocalAction::ReduceLR);
        assert_eq!(VocalAction::from_index(4), VocalAction::ShiftEmphasis);
        assert_eq!(VocalAction::from_index(5), VocalAction::ExplorationBurst);
        // Out of range wraps to ExplorationBurst
        assert_eq!(VocalAction::from_index(99), VocalAction::ExplorationBurst);
    }

    #[test]
    fn test_fep_result_default() {
        let result = VocalTractFepResult::default();
        assert!((result.tau_factor - 1.0).abs() < 0.01);
        assert!((result.learning_rate_factor - 1.0).abs() < 0.01);
        assert!((result.emphasis_factor - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fep_learning_loop() {
        let mut agent = VocalTractFepAgent::new();
        let obs = VocalTractObservation {
            articulation_score: 0.6,
            formant_accuracy: 0.5,
            pitch_stability: 0.7,
            coarticulation_smoothness: 0.6,
            duration_accuracy: 0.5,
            energy_consistency: 0.6,
        };

        // Run several tick+learn cycles
        for _ in 0..5 {
            let _result = agent.tick(&obs);
            agent.learn(&obs);
        }

        // TD learning should have received updates
        assert!(
            agent.stats().td_updates > 0,
            "TD learner should have received updates"
        );
        assert_eq!(agent.tick_count(), 5);
    }
}
