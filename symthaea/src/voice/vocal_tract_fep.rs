//! FEP active inference agent for vocal tract control.
//!
//! Wraps `ActiveInferenceAgent` from `symthaea-fep` to modulate the vocal tract
//! controller's time constants and learning rate based on voice output quality.
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
//! # Observation Space (6D from VoiceOutputMetrics)
//!
//! articulation, formant_accuracy, pitch_stability, coarticulation,
//! duration_accuracy, energy_consistency

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation,
    TemporalDifferenceLearningConfig,
};

use crate::voice::voice_feedback::VoiceOutputMetrics;

/// Actions the FEP agent can take to modulate the vocal tract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocalAction {
    /// Decrease tau → faster formant transitions (high surprise).
    DropTau = 0,
    /// Increase tau → smoother sustained vowels (low surprise).
    RaiseTau = 1,
    /// Increase learning rate → faster adaptation.
    BoostLR = 2,
    /// Decrease learning rate → fine-tuning.
    ReduceLR = 3,
    /// Shift emphasis → more assertive articulation.
    ShiftEmphasis = 4,
    /// Exploration burst → random perturbation to escape local minima.
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

    /// Run one FEP tick: observe voice quality → select action → return modulation.
    ///
    /// Call at 10Hz (every 20 motor frames at 200Hz).
    pub fn tick(&mut self, metrics: &VoiceOutputMetrics) -> VocalTractFepResult {
        self.tick_count += 1;

        // Construct 6D observation from VoiceOutputMetrics
        let obs = Observation {
            values: vec![
                metrics.articulation_score as f64,
                metrics.formant_accuracy as f64,
                metrics.pitch_stability as f64,
                metrics.coarticulation_smoothness as f64,
                metrics.duration_accuracy as f64,
                metrics.energy_consistency as f64,
            ],
            precision: 1.0,
            timestamp: self.tick_count,
            modality: "vocal_tract".to_string(),
        };

        // Perceive → update belief
        let perception = self.agent.perceive(&obs);

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
    pub fn learn(&mut self, metrics: &VoiceOutputMetrics) {
        if let Some(action) = self.last_action {
            let obs = Observation {
                values: vec![
                    metrics.articulation_score as f64,
                    metrics.formant_accuracy as f64,
                    metrics.pitch_stability as f64,
                    metrics.coarticulation_smoothness as f64,
                    metrics.duration_accuracy as f64,
                    metrics.energy_consistency as f64,
                ],
                precision: 1.0,
                timestamp: self.tick_count,
                modality: "vocal_tract_outcome".to_string(),
            };
            self.agent.learn_from_outcome(action, &obs);
        }
    }

    /// Get current free energy.
    pub fn free_energy(&self) -> Option<f64> {
        self.agent
            .last_fe_components
            .as_ref()
            .map(|fe| fe.total)
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

/// Complete vocal tract pipeline: encoder → controller → FEP agent.
///
/// Manages dual-rate processing:
/// - 200Hz motor: controller.forward() → FormantFrame → vocoder
/// - 10Hz cognitive: encoder.encode() → update cached HV; fep_agent.tick() → modulate
#[cfg(feature = "vocal-tract")]
pub struct VocalTractPipeline {
    /// HDC encoder: cognitive state → 16,384D ContinuousHV
    pub encoder: super::vocal_tract_encoder::VocalTractHdcEncoder,
    /// LTC controller: ContinuousHV → FormantFrame
    pub controller: super::vocal_tract_controller::VocalTractController,
    /// FEP active inference agent: voice metrics → tau/LR modulation
    pub fep_agent: VocalTractFepAgent,
    /// Cached cognitive HV (updated at 10Hz, used at 200Hz)
    cached_hv: symthaea_core::hdc::ContinuousHV,
    /// Counter for dual-rate scheduling (every 20 motor frames = 1 cognitive tick)
    motor_frame_count: usize,
    /// Motor frames per cognitive tick (200Hz / 10Hz = 20)
    frames_per_cognitive_tick: usize,
    /// Cumulative time in seconds (set on each FormantFrame).
    cumulative_time: f32,
}

#[cfg(feature = "vocal-tract")]
impl VocalTractPipeline {
    /// Create a new pipeline from a genesis seed.
    pub fn new(genesis: &symthaea_core::genesis::GenesisSeed) -> Self {
        use super::vocal_tract_controller::VocalTractConfig;
        use super::vocal_tract_encoder::VocalTractHdcEncoder;
        use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

        Self {
            encoder: VocalTractHdcEncoder::new(genesis, 32),
            controller: super::vocal_tract_controller::VocalTractController::new(
                genesis,
                &VocalTractConfig::default(),
            ),
            fep_agent: VocalTractFepAgent::new(),
            cached_hv: ContinuousHV::zero(HDC_DIMENSION),
            motor_frame_count: 0,
            frames_per_cognitive_tick: 20,
            cumulative_time: 0.0,
        }
    }

    /// Run one motor frame (200Hz).
    ///
    /// - Every `frames_per_cognitive_tick` frames: re-encode cognitive state,
    ///   optionally run FEP tick if metrics provided.
    /// - Every frame: evolve controller with cached HV → produce FormantFrame.
    pub fn tick(
        &mut self,
        cognitive_state: &super::vocal_tract_encoder::VoiceCognitiveState,
        metrics: Option<&VoiceOutputMetrics>,
        dt: f32,
    ) -> super::FormantFrame {
        // Cognitive tick (10Hz)
        if self.motor_frame_count % self.frames_per_cognitive_tick == 0 {
            self.cached_hv = self.encoder.encode(cognitive_state);

            // FEP modulation if we have metrics
            if let Some(m) = metrics {
                // Learn from previous action's outcome before selecting new action
                self.fep_agent.learn(m);
                let fep_result = self.fep_agent.tick(m);
                self.controller.modulate_tau(fep_result.tau_factor);
                let current_lr = self.controller.learning_rate();
                self.controller
                    .set_learning_rate(current_lr * fep_result.learning_rate_factor);
            }
        }

        self.motor_frame_count += 1;

        // Motor tick (200Hz): evolve network + produce formants
        let mut frame = self.controller.forward(&self.cached_hv, dt);
        frame.time = self.cumulative_time;
        self.cumulative_time += dt;
        frame
    }

    /// Reset the entire pipeline.
    pub fn reset(&mut self) {
        self.encoder.reset();
        self.controller.reset();
        self.fep_agent.reset();
        self.cached_hv = symthaea_core::hdc::ContinuousHV::zero(symthaea_core::hdc::HDC_DIMENSION);
        self.motor_frame_count = 0;
        self.cumulative_time = 0.0;
    }

    /// Get current cumulative time in seconds.
    pub fn cumulative_time(&self) -> f32 {
        self.cumulative_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_action_selection() {
        let mut agent = VocalTractFepAgent::new();
        let metrics = VoiceOutputMetrics {
            articulation_score: 0.8,
            formant_accuracy: 0.7,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.8,
            duration_accuracy: 0.7,
            energy_consistency: 0.8,
            ..Default::default()
        };

        let result = agent.tick(&metrics);

        // Should produce valid modulation factors
        assert!(result.tau_factor > 0.0);
        assert!(result.learning_rate_factor > 0.0);
        assert!(result.emphasis_factor > 0.0);
        assert!(result.free_energy.is_finite());
    }

    #[test]
    fn test_fep_repeated_ticks() {
        let mut agent = VocalTractFepAgent::new();
        let good_metrics = VoiceOutputMetrics {
            articulation_score: 0.9,
            formant_accuracy: 0.9,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.9,
            duration_accuracy: 0.9,
            energy_consistency: 0.9,
            ..Default::default()
        };

        // Run several ticks — should not panic
        for _ in 0..20 {
            let result = agent.tick(&good_metrics);
            assert!(result.free_energy.is_finite());
        }

        assert_eq!(agent.tick_count(), 20);
    }

    #[test]
    fn test_fep_reset() {
        let mut agent = VocalTractFepAgent::new();
        let metrics = VoiceOutputMetrics::default();

        agent.tick(&metrics);
        agent.tick(&metrics);
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
        let metrics = VoiceOutputMetrics {
            articulation_score: 0.6,
            formant_accuracy: 0.5,
            pitch_stability: 0.7,
            coarticulation_smoothness: 0.6,
            duration_accuracy: 0.5,
            energy_consistency: 0.6,
            ..Default::default()
        };

        // Run several tick+learn cycles
        for _ in 0..5 {
            let _result = agent.tick(&metrics);
            agent.learn(&metrics);
        }

        // TD learning should have received updates
        assert!(agent.stats().td_updates > 0, "TD learner should have received updates");
        assert_eq!(agent.tick_count(), 5);
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_pipeline_time_tracking() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-time-tracking");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();
        let dt = 0.005; // 200Hz

        // Run 40 frames
        let mut last_frame = pipeline.tick(&state, None, dt);
        for _ in 1..40 {
            last_frame = pipeline.tick(&state, None, dt);
        }

        // Last frame should have time = 39 * 0.005 = 0.195
        let expected_time = 39.0 * dt;
        assert!(
            (last_frame.time - expected_time).abs() < 1e-4,
            "Expected time ~{}, got {}",
            expected_time,
            last_frame.time
        );

        // Cumulative time should be 40 * 0.005 = 0.200
        assert!(
            (pipeline.cumulative_time() - 40.0 * dt).abs() < 1e-4,
            "Expected cumulative ~{}, got {}",
            40.0 * dt,
            pipeline.cumulative_time()
        );

        // Reset should clear time
        pipeline.reset();
        assert!((pipeline.cumulative_time()).abs() < 1e-6);
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_pipeline_fep_learning() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-fep-learning");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state = VoiceCognitiveState::default();
        let metrics = VoiceOutputMetrics {
            articulation_score: 0.6,
            formant_accuracy: 0.5,
            pitch_stability: 0.7,
            coarticulation_smoothness: 0.6,
            duration_accuracy: 0.5,
            energy_consistency: 0.6,
            ..Default::default()
        };

        // Run 60 frames (3 cognitive ticks with FEP feedback)
        for _ in 0..60 {
            pipeline.tick(&state, Some(&metrics), 0.005);
        }

        // FEP agent should have ticked 3 times
        assert_eq!(pipeline.fep_agent.tick_count(), 3);
        // TD learning should have been triggered (learn is called before tick on 2nd+ cognitive tick)
        assert!(
            pipeline.fep_agent.stats().td_updates > 0,
            "TD learner should have updates after multiple cognitive ticks"
        );
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_pipeline_end_to_end() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-vocal-pipeline");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        // Run 40 frames (2 cognitive ticks)
        for _ in 0..40 {
            let frame = pipeline.tick(&state, None, 0.005);
            assert!(frame.f1 >= 200.0 && frame.f1 <= 1000.0);
            assert!(frame.energy >= 0.0 && frame.energy <= 1.0);
        }
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_pipeline_dual_rate() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-dual-rate");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state1 = VoiceCognitiveState {
            emotional_arousal: 0.2,
            ..Default::default()
        };
        let state2 = VoiceCognitiveState {
            emotional_arousal: 0.9,
            ..Default::default()
        };

        // First cognitive tick (frame 0) with state1
        let frame_a = pipeline.tick(&state1, None, 0.005);

        // Frames 1-19 still use cached HV from state1
        for _ in 1..20 {
            pipeline.tick(&state1, None, 0.005);
        }

        // Frame 20: new cognitive tick with state2 (different arousal)
        let frame_b = pipeline.tick(&state2, None, 0.005);

        // Frame 21: still using state2's cached HV
        let frame_c = pipeline.tick(&state2, None, 0.005);

        // frame_a and frame_b used different cognitive inputs at re-encode boundaries
        // frame_b and frame_c used the same cognitive HV (frame_c at motor-only tick)
        assert!(frame_a.f1.is_finite());
        assert!(frame_b.f1.is_finite());
        assert!(frame_c.f1.is_finite());
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_pipeline_with_fep_feedback() {
        use super::super::vocal_tract_encoder::VoiceCognitiveState;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-fep-feedback");
        let mut pipeline = VocalTractPipeline::new(&genesis);

        let state = VoiceCognitiveState::default();
        let metrics = VoiceOutputMetrics {
            articulation_score: 0.8,
            formant_accuracy: 0.7,
            pitch_stability: 0.9,
            coarticulation_smoothness: 0.8,
            duration_accuracy: 0.7,
            energy_consistency: 0.8,
            ..Default::default()
        };

        // Run with FEP feedback
        for _ in 0..40 {
            let frame = pipeline.tick(&state, Some(&metrics), 0.005);
            assert!(frame.f1.is_finite());
        }

        // FEP agent should have ticked twice (at frames 0 and 20)
        assert_eq!(pipeline.fep_agent.tick_count(), 2);
    }
}
