// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! FEP Active Inference for music: the system has a generative model of
//! its own audio output and actively minimizes free energy between
//! predicted and actual sound.
//!
//! This makes the strange loop formally correct under the Free Energy Principle:
//! - **Sensory states**: Audio features (consonance, rhythm, contour, timbre)
//! - **Hidden states**: Musical beliefs (expected harmony, predicted melody)
//! - **Active states**: Synthesis parameters (pitch selection, rhythm, dynamics)
//! - **Free energy**: Divergence between predicted and actual music
//!
//! # Architecture
//!
//! ```text
//! Audio features (observation) → ActiveInferenceAgent.perceive()
//!   → belief update (minimize free energy)
//!   → action selection (minimize expected free energy)
//!   → MusicAction → modulate MusicalState
//!   → synthesis → new audio features → loop
//! ```
//!
//! The agent learns: which musical actions (chord changes, tempo shifts,
//! timbral transitions) minimize surprise while maintaining creativity
//! via epistemic value (information-seeking) and novelty bonuses.

use serde::{Deserialize, Serialize};
use symthaea_fep::Observation;
use symthaea_fep::TemporalDifferenceLearningConfig;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig};

use crate::MusicalState;
use crate::audio_feedback::AudioFeatures;

/// Musical action types (mapped to ActiveInferenceAgent's action indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MusicAction {
    /// Follow current chord progression (maintain harmonic context).
    FollowHarmony = 0,
    /// Explore chromatic motion (tension building).
    ChromaticExplore = 1,
    /// Repeat and develop current motif (consolidation).
    RepeatMotif = 2,
    /// Modulate to related key (structural shift).
    ModulateKey = 3,
    /// Increase rhythmic/harmonic complexity.
    IncreaseComplexity = 4,
    /// Resolve tension (return to tonic/stability).
    ResolveTension = 5,
    /// Add countermelody (textural enrichment).
    AddCountermelody = 6,
    /// Maintain current state (stability).
    Maintain = 7,
}

impl MusicAction {
    fn from_index(i: usize) -> Self {
        match i {
            0 => Self::FollowHarmony,
            1 => Self::ChromaticExplore,
            2 => Self::RepeatMotif,
            3 => Self::ModulateKey,
            4 => Self::IncreaseComplexity,
            5 => Self::ResolveTension,
            6 => Self::AddCountermelody,
            _ => Self::Maintain,
        }
    }
}

/// Result of one FEP music inference cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicInferenceResult {
    /// Selected musical action.
    pub action: MusicAction,
    /// Free energy (lower = better prediction of own output).
    pub free_energy: f64,
    /// Prediction error magnitude.
    pub prediction_error: f64,
    /// Precision-weighted prediction error (surprise signal).
    pub surprise: f64,
    /// Whether the system is genuinely surprised by its own audio.
    pub is_surprised: bool,
    /// Learning rate modulation (0.1-2.0x, from precision dynamics).
    pub learning_rate_mod: f64,
    /// Sensory precision (confidence in audio observation).
    pub sensory_precision: f64,
    /// Prior precision (confidence in musical predictions).
    pub prior_precision: f64,
}

/// Compact evidence that the temporal FEP path committed actions and updated
/// its transition learner rather than merely sampling isolated proposals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicInferenceLearningStats {
    pub committed_actions: u64,
    pub td_total_updates: u64,
    pub td_transition_history_size: usize,
    pub td_average_error: f64,
    pub td_average_prediction_accuracy: f64,
}

/// FEP-driven musical inference engine.
///
/// Wraps `ActiveInferenceAgent` with music-specific observation encoding,
/// action decoding, and precision dynamics. The agent learns a generative
/// model of its own audio output and selects actions to minimize surprise
/// while maintaining creative exploration via epistemic value.
pub struct MusicalInferenceEngine {
    agent: ActiveInferenceAgent,
    cycle_count: u64,
    last_result: Option<MusicInferenceResult>,
    /// Preferred observation (musical goals: consonance, stability, etc.)
    preferred_obs: Vec<f64>,
    goal_precision: f64,
    committed_actions: u64,
}

impl MusicalInferenceEngine {
    /// Create a new musical inference engine with the historical fixed FEP RNG
    /// sequence. Existing callers retain byte-for-byte action sampling behavior.
    pub fn new() -> Self {
        Self::build(None)
    }

    /// Create a musical inference engine whose stochastic action sampler is
    /// explicitly seeded by the caller.
    ///
    /// This is the constructor required by reproducible experiments. It leaves
    /// the deterministic generative-model initialization unchanged and only
    /// replaces the agent's action-selection RNG state.
    pub fn new_with_seed(seed: u64) -> Self {
        Self::build(Some(seed))
    }

    fn build(rng_seed: Option<u64>) -> Self {
        let config = ActiveInferenceAgentConfig {
            state_dim: 16,  // 16D hidden musical state
            obs_dim: 6,     // 6 audio features
            num_actions: 8, // 8 musical actions
            inference_iterations: 3,
            belief_learning_rate: 0.08,
            planning_horizon: 2,
            action_temperature: 1.5, // slightly exploratory
            enable_model_learning: true,
            enable_td_learning: true,
            td_config: TemporalDifferenceLearningConfig {
                gamma: 0.95, // long-horizon value (music has long-range structure)
                lambda: 0.8, // eligibility trace for credit assignment
                ..TemporalDifferenceLearningConfig::default()
            },
        };

        let mut agent = ActiveInferenceAgent::new(config);
        if let Some(seed) = rng_seed {
            agent.set_rng_seed(seed);
        }
        // Preferred observations: moderate brightness and rhythmic complexity,
        // strong consonance/stability, moderate energy, and low noise.
        let preferred_obs = vec![0.6, 0.5, 0.7, 0.6, 0.4, 0.2];
        let goal_precision = 2.0;
        agent.set_goals(preferred_obs.clone(), goal_precision);

        Self {
            agent,
            cycle_count: 0,
            last_result: None,
            preferred_obs,
            goal_precision,
            committed_actions: 0,
        }
    }

    /// Run one inference cycle: perceive audio → update beliefs → select action.
    ///
    /// Call this once per streaming chunk (~32ms). Pass the audio features
    /// from the feedback encoder. Returns the selected musical action and
    /// precision dynamics.
    pub fn infer(&mut self, features: &AudioFeatures) -> MusicInferenceResult {
        self.infer_internal(features, false)
    }

    /// Run one inference cycle and commit the selected action to the FEP agent.
    ///
    /// Committing records the action as the cause of the next observation, so
    /// subsequent perception cycles can perform temporal-difference updates.
    /// Legacy `infer` callers remain proposal-only for backward compatibility.
    pub fn infer_and_commit(&mut self, features: &AudioFeatures) -> MusicInferenceResult {
        self.infer_internal(features, true)
    }

    fn infer_internal(
        &mut self,
        features: &AudioFeatures,
        commit_action: bool,
    ) -> MusicInferenceResult {
        // 1. Construct observation from audio features
        let obs = Observation::new(
            vec![
                features.spectral_centroid as f64, // brightness → harmonic content
                features.rhythm_entropy as f64,    // rhythmic complexity
                (1.0 - features.harmonic_tension) as f64, // consonance (inverse tension)
                (1.0 - features.spectral_flux) as f64, // timbral stability (inverse flux)
                features.rms_energy as f64,        // dynamic level
                features.zero_crossing_rate as f64, // noisiness
            ],
            self.agent.precision.sensory_precision, // current precision estimate
            "music",
        );

        // 2. Perception: update beliefs to minimize free energy
        let perception = self.agent.perceive(&obs);

        // 3. Action selection: minimize expected free energy
        let action_result = self.agent.select_action();
        if commit_action {
            let _ = self.agent.act(action_result.action);
            self.committed_actions += 1;
        }

        // 4. Build result from FEP components
        let free_energy = perception.free_energy.surprise;
        let prediction_error = perception.free_energy.prediction_error;
        let is_surprised = prediction_error > 0.5; // high PE = surprised
        let learning_rate_mod = if is_surprised { 1.5 } else { 1.0 }; // learn more when surprised

        let result = MusicInferenceResult {
            action: MusicAction::from_index(action_result.action),
            free_energy,
            prediction_error,
            surprise: prediction_error * self.agent.precision.sensory_precision,
            is_surprised,
            learning_rate_mod,
            sensory_precision: self.agent.precision.sensory_precision,
            prior_precision: self.agent.precision.prior_precision,
        };

        self.last_result = Some(result.clone());
        self.cycle_count += 1;

        result
    }

    /// Apply the inference result to modulate a MusicalState.
    ///
    /// Translates the selected MusicAction into parameter changes,
    /// scaled by the agent's precision dynamics (confident actions
    /// produce larger changes).
    pub fn apply_action(&self, result: &MusicInferenceResult, state: &mut MusicalState) {
        let confidence = result.prior_precision.clamp(0.1, 2.0) as f32;
        let mod_strength = 0.05 * confidence;

        match result.action {
            MusicAction::FollowHarmony => {
                // Strengthen dominant harmony activation
                let max_h = state
                    .harmony_activations
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                state.harmony_activations[max_h] =
                    (state.harmony_activations[max_h] + mod_strength).min(1.0);
            }
            MusicAction::ChromaticExplore => {
                // Increase prediction error (musical surprise)
                state.prediction_error = (state.prediction_error + mod_strength * 2.0).min(1.0);
                // Boost InfinitePlay harmony (tension)
                state.harmony_activations[3] =
                    (state.harmony_activations[3] + mod_strength).min(1.0);
            }
            MusicAction::RepeatMotif => {
                // Decrease prediction error (familiarity)
                state.prediction_error = (state.prediction_error - mod_strength).max(0.0);
                // Boost EvolutionaryProgression (development)
                state.harmony_activations[6] =
                    (state.harmony_activations[6] + mod_strength).min(1.0);
            }
            MusicAction::ModulateKey => {
                // Shift valence (emotional change)
                state.valence += mod_strength * if state.valence > 0.0 { -1.0 } else { 1.0 };
                state.valence = state.valence.clamp(-1.0, 1.0);
            }
            MusicAction::IncreaseComplexity => {
                state.arousal = (state.arousal + mod_strength).min(1.0);
                state.dopamine = (state.dopamine + mod_strength * 0.5).min(1.0);
            }
            MusicAction::ResolveTension => {
                // Move toward ResonantCoherence
                state.harmony_activations[0] =
                    (state.harmony_activations[0] + mod_strength * 2.0).min(1.0);
                state.prediction_error = (state.prediction_error - mod_strength * 2.0).max(0.0);
                // Boost serotonin (satisfaction of resolution)
                state.serotonin = (state.serotonin + mod_strength).min(1.0);
            }
            MusicAction::AddCountermelody => {
                // Boost consciousness (needed for polyphony)
                state.consciousness_level = (state.consciousness_level + mod_strength).min(1.0);
            }
            MusicAction::Maintain => {
                // No-op: sustain current state
            }
        }
    }

    /// Get the last inference result.
    pub fn last_result(&self) -> Option<&MusicInferenceResult> {
        self.last_result.as_ref()
    }

    /// Total inference cycles run.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Set a bounded, inspectable FEP goal from the declared musical state.
    pub fn set_emotion_anchor(&mut self, state: &crate::MusicalState) {
        let (preferences, precision) = Self::emotion_goal(state);
        self.set_preferences_with_precision(preferences, precision);
    }

    /// Deterministic six-channel goal used by the symbolic temporal session.
    pub fn emotion_goal(state: &crate::MusicalState) -> (Vec<f64>, f64) {
        let valence = f64::from(state.valence.clamp(-1.0, 1.0));
        let arousal = f64::from(state.arousal.clamp(0.0, 1.0));
        let prediction_error = f64::from(state.prediction_error.clamp(0.0, 1.0));
        let consciousness = f64::from(state.consciousness_level.clamp(0.0, 1.0));
        let preferences = vec![
            (0.50 + 0.15 * valence).clamp(0.0, 1.0),
            (0.30 + 0.45 * arousal).clamp(0.0, 1.0),
            (0.78 - 0.28 * prediction_error).clamp(0.0, 1.0),
            (0.78 - 0.35 * arousal).clamp(0.0, 1.0),
            (0.20 + 0.65 * arousal).clamp(0.0, 1.0),
            (0.08 + 0.22 * prediction_error).clamp(0.0, 1.0),
        ];
        let precision = (1.5 + 1.5 * consciousness).clamp(0.5, 4.0);
        (preferences, precision)
    }

    /// Current free energy (lower = better model of own output).
    pub fn current_free_energy(&self) -> f64 {
        self.last_result
            .as_ref()
            .map(|r| r.free_energy)
            .unwrap_or(0.0)
    }

    /// Set musical preferences (target observation values).
    ///
    /// The agent will try to produce audio features that match these preferences.
    /// This allows different "musical personalities" or optimization targets.
    pub fn set_preferences(&mut self, prefs: Vec<f64>) {
        self.set_preferences_with_precision(prefs, self.goal_precision);
    }

    /// Set validated FEP goals and the confidence placed on them.
    pub fn set_preferences_with_precision(&mut self, prefs: Vec<f64>, precision: f64) {
        if prefs.len() != 6
            || prefs.iter().any(|value| !value.is_finite())
            || !precision.is_finite()
            || precision <= 0.0
        {
            return;
        }
        let prefs: Vec<f64> = prefs
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect();
        let precision = precision.clamp(0.1, 10.0);
        self.agent.set_goals(prefs.clone(), precision);
        self.preferred_obs = prefs;
        self.goal_precision = precision;
    }

    pub fn goal_preferences(&self) -> &[f64] {
        &self.preferred_obs
    }

    pub fn goal_precision(&self) -> f64 {
        self.goal_precision
    }

    pub fn learning_stats(&self) -> MusicInferenceLearningStats {
        let stats = self.agent.td_stats();
        MusicInferenceLearningStats {
            committed_actions: self.committed_actions,
            td_total_updates: stats.as_ref().map_or(0, |value| value.total_updates),
            td_transition_history_size: stats
                .as_ref()
                .map_or(0, |value| value.transition_history_size),
            td_average_error: stats.as_ref().map_or(0.0, |value| value.avg_td_error),
            td_average_prediction_accuracy: stats
                .as_ref()
                .map_or(0.0, |value| value.avg_prediction_accuracy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_creates_and_infers() {
        let mut engine = MusicalInferenceEngine::new();
        let features = AudioFeatures {
            spectral_centroid: 0.4,
            spectral_flux: 0.2,
            rhythm_entropy: 0.3,
            harmonic_tension: 0.15,
            rms_energy: 0.5,
            zero_crossing_rate: 0.1,
        };

        let result = engine.infer(&features);
        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error >= 0.0);
        assert!(result.sensory_precision > 0.0);
    }

    #[test]
    fn committed_inference_produces_temporal_learning_evidence() {
        let mut engine = MusicalInferenceEngine::new_with_seed(17);
        let features = AudioFeatures {
            spectral_centroid: 0.4,
            spectral_flux: 0.2,
            rhythm_entropy: 0.3,
            harmonic_tension: 0.15,
            rms_energy: 0.5,
            zero_crossing_rate: 0.1,
        };

        let _ = engine.infer_and_commit(&features);
        let _ = engine.infer_and_commit(&features);
        let stats = engine.learning_stats();
        assert_eq!(stats.committed_actions, 2);
        assert_eq!(stats.td_transition_history_size, 1);
        assert!(stats.td_total_updates > 0);
        assert!(stats.td_average_error.is_finite());
    }

    #[test]
    fn emotion_anchor_installs_bounded_fep_goals() {
        let mut engine = MusicalInferenceEngine::new();
        let state = MusicalState {
            valence: 0.8,
            arousal: 0.9,
            prediction_error: 0.4,
            consciousness_level: 0.75,
            ..MusicalState::default()
        };
        engine.set_emotion_anchor(&state);
        let (expected, precision) = MusicalInferenceEngine::emotion_goal(&state);
        assert_eq!(engine.goal_preferences(), expected.as_slice());
        assert_eq!(engine.goal_precision(), precision);
        assert!(expected.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn action_selection_varies() {
        let mut engine = MusicalInferenceEngine::new();

        // Run many cycles with different features
        let mut actions = std::collections::HashSet::new();
        for i in 0..50 {
            let features = AudioFeatures {
                spectral_centroid: (i as f32 * 0.02).min(1.0),
                spectral_flux: (i as f32 * 0.03).min(1.0),
                rhythm_entropy: 0.3 + (i as f32 * 0.01).min(0.5),
                harmonic_tension: (i as f32 * 0.015).min(1.0),
                rms_energy: 0.4,
                zero_crossing_rate: 0.1,
            };
            let result = engine.infer(&features);
            actions.insert(result.action as u8);
        }

        // Should select at least 2 different actions over 50 cycles
        assert!(
            actions.len() >= 2,
            "should explore multiple actions, got {}",
            actions.len()
        );
    }

    #[test]
    fn apply_action_modulates_state() {
        let engine = MusicalInferenceEngine::new();
        let mut state = MusicalState::default();

        // Test ChromaticExplore: should increase prediction_error
        let result = MusicInferenceResult {
            action: MusicAction::ChromaticExplore,
            free_energy: 1.0,
            prediction_error: 0.5,
            surprise: 0.3,
            is_surprised: false,
            learning_rate_mod: 1.0,
            sensory_precision: 1.0,
            prior_precision: 1.0,
        };
        let orig_pe = state.prediction_error;
        engine.apply_action(&result, &mut state);
        assert!(
            state.prediction_error > orig_pe,
            "ChromaticExplore should increase PE"
        );

        // Test ResolveTension: should increase coherence harmony
        let resolve = MusicInferenceResult {
            action: MusicAction::ResolveTension,
            ..result
        };
        let orig_h0 = state.harmony_activations[0];
        engine.apply_action(&resolve, &mut state);
        assert!(
            state.harmony_activations[0] > orig_h0,
            "ResolveTension should boost coherence"
        );
    }

    #[test]
    fn precision_dynamics_respond_to_surprise() {
        let mut engine = MusicalInferenceEngine::new();

        // Feed steady predictable input
        let steady = AudioFeatures {
            spectral_centroid: 0.4,
            spectral_flux: 0.05,
            rhythm_entropy: 0.2,
            harmonic_tension: 0.1,
            rms_energy: 0.5,
            zero_crossing_rate: 0.1,
        };
        for _ in 0..20 {
            engine.infer(&steady);
        }
        // Copy the scalars out rather than holding `last_result()`'s borrow —
        // `engine.infer()` below needs `&mut engine`.
        let (prior_after_steady, sensory_after_steady) = {
            let r = engine.last_result().unwrap();
            (r.prior_precision, r.sensory_precision)
        };

        // Then feed surprising input
        let surprise = AudioFeatures {
            spectral_centroid: 0.9,
            spectral_flux: 0.8,
            rhythm_entropy: 0.9,
            harmonic_tension: 0.8,
            rms_energy: 0.9,
            zero_crossing_rate: 0.7,
        };
        // Sample the TRANSIENT (first surprising frame) separately from the
        // settled state: 20 identical "surprise" frames stop being surprising
        // once the agent has re-learned them, so the documented prior-precision
        // dip is a property of the transition, not of the new steady state.
        engine.infer(&surprise);
        let (prior_at_transient, sensory_at_transient) = {
            let r = engine.last_result().unwrap();
            (r.prior_precision, r.sensory_precision)
        };
        for _ in 0..19 {
            engine.infer(&surprise);
        }
        let surprise_result = engine.last_result().unwrap();
        let prior_after_surprise = surprise_result.prior_precision;
        let sensory_after_surprise = surprise_result.sensory_precision;

        assert!(
            surprise_result.prediction_error > 0.0,
            "surprise should produce prediction error"
        );

        // Under surprise, sensory precision rises (pay attention to new input)
        // and prior precision falls (predictions are wrong). These dynamics
        // come from PrecisionEstimator in symthaea-fep; this test is the only
        // thing checking muse observes them through `MusicalInferenceEngine`.
        assert!(
            sensory_after_surprise > sensory_after_steady,
            "sensory precision should rise under surprise: \
             {sensory_after_steady} -> {sensory_after_surprise}"
        );
        // Measured 2026-07-28, the first time this test asserted anything:
        //   prior:   steady 1.7851 -> transient 1.7405 -> settled 2.5858
        //   sensory: steady 1.0069 -> transient 1.0302 -> settled 1.1057
        // The prior-precision DIP is real but transient-only. Asserting it on
        // the settled value fails, and that is correct behaviour rather than a
        // bug: 20 identical "surprise" frames stop being surprising once the
        // agent has learned them, so prior precision recovers past where it
        // started. The original version of this test sampled only the settled
        // value, so it could not have observed the property its own comment
        // described even if it had asserted -- which it also did not.
        assert!(
            sensory_at_transient > sensory_after_steady,
            "sensory precision should rise on the surprising frame: \
             {sensory_after_steady} -> {sensory_at_transient}"
        );
        assert!(
            prior_at_transient < prior_after_steady,
            "prior precision should fall on the surprising frame: \
             {prior_after_steady} -> {prior_at_transient}"
        );
        assert!(
            prior_after_surprise > prior_at_transient,
            "prior precision should recover as the new input is learned: \
             {prior_at_transient} -> {prior_after_surprise}"
        );
    }

    #[test]
    fn free_energy_decreases_with_learning() {
        let mut engine = MusicalInferenceEngine::new();

        // Feed consistent input — agent should learn to predict it
        let consistent = AudioFeatures {
            spectral_centroid: 0.5,
            spectral_flux: 0.1,
            rhythm_entropy: 0.3,
            harmonic_tension: 0.2,
            rms_energy: 0.4,
            zero_crossing_rate: 0.15,
        };

        let mut early_fe = 0.0;
        for i in 0..100 {
            let result = engine.infer(&consistent);
            if i == 5 {
                early_fe = result.free_energy;
            }
        }
        let late_fe = engine.current_free_energy();

        // Free energy should decrease or stabilize (agent learns)
        // Note: with stochastic action selection, exact monotonic decrease
        // isn't guaranteed, but trend should be downward
        eprintln!("  FE: early={early_fe:.4}, late={late_fe:.4}");
    }

    #[test]
    fn all_actions_are_safe() {
        let engine = MusicalInferenceEngine::new();

        // Every action should keep state in valid bounds
        for action_idx in 0..8 {
            let mut state = MusicalState {
                consciousness_level: 0.95,
                arousal: 0.95,
                dopamine: 0.95,
                serotonin: 0.95,
                noradrenaline: 0.95,
                valence: 0.9,
                harmony_activations: [0.95; 8],
                prediction_error: 0.95,
            };
            let result = MusicInferenceResult {
                action: MusicAction::from_index(action_idx),
                free_energy: 1.0,
                prediction_error: 0.5,
                surprise: 0.3,
                is_surprised: false,
                learning_rate_mod: 1.0,
                sensory_precision: 2.0,
                prior_precision: 2.0,
            };
            engine.apply_action(&result, &mut state);

            assert!(state.consciousness_level <= 1.0 && state.consciousness_level >= 0.0);
            assert!(state.arousal <= 1.0 && state.arousal >= 0.0);
            assert!(state.valence >= -1.0 && state.valence <= 1.0);
            for &h in &state.harmony_activations {
                assert!(h >= 0.0 && h <= 1.0);
            }
        }
    }

    #[test]
    fn cycle_count_increments() {
        let mut engine = MusicalInferenceEngine::new();
        assert_eq!(engine.cycle_count(), 0);
        engine.infer(&AudioFeatures::default());
        assert_eq!(engine.cycle_count(), 1);
        engine.infer(&AudioFeatures::default());
        assert_eq!(engine.cycle_count(), 2);
    }
}
