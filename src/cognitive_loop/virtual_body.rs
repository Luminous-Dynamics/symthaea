// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Virtual Body Adapter for embodied cognition integration.
//!
//! Maps cognitive loop signals to interoceptive body states, enabling the
//! cognitive loop to have a "virtual body" that modulates consciousness
//! through somatic marker-like feedback (Damasio's hypothesis).
//!
//! ## Signal Mappings
//!
//! | Cognitive Signal          | Body Signal       | Rationale                                    |
//! |---------------------------|-------------------|----------------------------------------------|
//! | Prediction error velocity | Heart rate        | Surprise spikes → sympathetic arousal         |
//! | Cycle throughput          | Breathing rate    | Faster cycles → higher metabolic demand       |
//! | Error trend (increasing)  | Fatigue           | Prolonged error → resource depletion          |
//! | Curiosity boredom level   | Hunger            | Bored system is "hungry" for novelty          |
//! | Prediction confidence     | Thirst (inverted) | Low confidence = "thirsty" for information    |
//! | Unified Phi level         | Temperature       | Higher consciousness = "warmer"               |
//! | Flow state intensity      | Gut state         | Flow feels "right" (positive gut feeling)     |
//! | FEP learning signal       | Visceral arousal  | Active inference → autonomic activation       |
//!
//! The resulting `InteroceptiveState` converts to `CoreAffect` (VAD) and
//! produces a `phi_modulation` factor that scales consciousness Phi.

use crate::consciousness::embodied_cognition::InteroceptiveState;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for the virtual body adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VirtualBodyConfig {
    /// Smoothing factor for interoceptive signals (0-1, lower = smoother)
    pub smoothing: f32,
    /// How strongly body state modulates Phi (0-1)
    pub body_consciousness_coupling: f32,
    /// Baseline heart rate (at prediction_error = 0)
    pub baseline_heart_rate: f64,
    /// Baseline breathing rate (at target cycle frequency)
    pub baseline_breathing_rate: f64,
}

impl Default for VirtualBodyConfig {
    fn default() -> Self {
        Self {
            smoothing: 0.3,
            body_consciousness_coupling: 0.4,
            baseline_heart_rate: 0.3,
            baseline_breathing_rate: 0.3,
        }
    }
}

/// Signals from the cognitive loop that drive the virtual body.
///
/// Collected at specific points during the cycle and passed to
/// `VirtualBody::update()` in one batch.
pub(crate) struct CognitiveSignals {
    pub prediction_error: f32,
    pub coherence: f32,
    pub prediction_confidence: f32,
    pub unified_psi: f64,
    pub flow_intensity: f32,
    pub in_flow: bool,
    pub curiosity_boredom: f32,
    pub fep_learning_signal: f32,
    pub error_trend: f32,
    pub cycles_per_second: f32,
    pub target_frequency: f32,
}

/// Summary of the virtual body's current state, for telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct VirtualBodyState {
    /// Current interoceptive affect (VAD space)
    pub valence: f32,
    pub arousal: f32,
    pub dominance: f32,
    /// Phi modulation factor (>1.0 = boost, <1.0 = dampen)
    pub phi_modulation: f64,
    /// Homeostatic deviation (0 = ideal, higher = stressed)
    pub homeostatic_deviation: f64,
}

/// Virtual body that maps cognitive signals to interoceptive states.
///
/// Maintains a smoothed `InteroceptiveState` that updates each cycle,
/// providing embodied grounding for the cognitive loop.
pub(crate) struct VirtualBody {
    config: VirtualBodyConfig,
    /// Current smoothed interoceptive state
    state: InteroceptiveState,
    /// Recent prediction errors for computing velocity.
    /// Capacity bound: 20 elements — evict before push via pop_front.
    error_history: VecDeque<f32>,
    /// Last computed phi_modulation
    phi_modulation: f64,
}

impl VirtualBody {
    /// Create a new virtual body with default configuration.
    pub fn new() -> Self {
        Self::with_config(VirtualBodyConfig::default())
    }

    /// Create a new virtual body with custom configuration.
    pub fn with_config(config: VirtualBodyConfig) -> Self {
        Self {
            state: InteroceptiveState::default(),
            error_history: VecDeque::with_capacity(20),
            phi_modulation: 1.0,
            config,
        }
    }

    /// Update the virtual body from cognitive loop signals.
    ///
    /// Maps cognitive signals to interoceptive states with EMA smoothing,
    /// then computes phi_modulation and CoreAffect.
    pub fn update(&mut self, signals: &CognitiveSignals) -> VirtualBodyState {
        let alpha = self.config.smoothing as f64;

        // Track prediction error velocity (derivative of error)
        // Capacity bound: 20 elements — evict before push to prevent transient over-capacity
        if self.error_history.len() >= 20 {
            self.error_history.pop_front();
        }
        self.error_history.push_back(signals.prediction_error);
        let error_velocity = if self.error_history.len() >= 2 {
            let recent = self.error_history.back().copied().unwrap_or(0.0);
            let older = self.error_history[self.error_history.len().saturating_sub(5)];
            (recent - older).abs()
        } else {
            0.0
        };

        // Map signals to raw interoceptive values
        let raw = InteroceptiveState {
            // Heart rate: baseline + spike from prediction error velocity
            heart_rate: (self.config.baseline_heart_rate
                + error_velocity as f64 * 2.0
                + signals.prediction_error as f64 * 0.5)
                .clamp(0.0, 1.0),

            // Breathing rate: proportional to cycle frequency vs target
            breathing_rate: if signals.target_frequency > 0.0 {
                (self.config.baseline_breathing_rate
                    + (signals.cycles_per_second / signals.target_frequency) as f64 * 0.4)
                    .clamp(0.0, 1.0)
            } else {
                self.config.baseline_breathing_rate
            },

            // Fatigue: increases with sustained error (positive error trend)
            fatigue: (signals.error_trend.max(0.0) as f64 * 3.0).clamp(0.0, 1.0),

            // Hunger: bored system is hungry for novelty
            hunger: signals.curiosity_boredom as f64,

            // Thirst: low confidence = thirsty for information
            thirst: (1.0 - signals.prediction_confidence as f64).clamp(0.0, 1.0),

            // Temperature: higher consciousness = warmer
            temperature: (signals.unified_psi * 2.0 - 1.0).clamp(-1.0, 1.0),

            // Gut state: flow feels right
            gut_state: if signals.in_flow {
                0.5 + signals.flow_intensity as f64 * 0.5
            } else {
                // Neutral-to-slightly-negative when not in flow
                signals.coherence as f64 * 0.3 - 0.1
            },

            // Visceral arousal: active inference learning signal
            visceral_arousal: signals.fep_learning_signal.abs() as f64,
        };

        // Smooth with EMA
        self.state.heart_rate = self.state.heart_rate * (1.0 - alpha) + raw.heart_rate * alpha;
        self.state.breathing_rate =
            self.state.breathing_rate * (1.0 - alpha) + raw.breathing_rate * alpha;
        self.state.fatigue = self.state.fatigue * (1.0 - alpha) + raw.fatigue * alpha;
        self.state.hunger = self.state.hunger * (1.0 - alpha) + raw.hunger * alpha;
        self.state.thirst = self.state.thirst * (1.0 - alpha) + raw.thirst * alpha;
        self.state.temperature = self.state.temperature * (1.0 - alpha) + raw.temperature * alpha;
        self.state.gut_state = self.state.gut_state * (1.0 - alpha) + raw.gut_state * alpha;
        self.state.visceral_arousal =
            self.state.visceral_arousal * (1.0 - alpha) + raw.visceral_arousal * alpha;

        // Compute phi modulation from body state
        // Body consciousness coupling: aroused body → heightened consciousness
        let body_factor =
            1.0 + self.state.visceral_arousal * self.config.body_consciousness_coupling as f64;
        // Homeostatic deviation dampens consciousness (stressed body → less integration)
        let homeostatic_dev = self.state.homeostatic_deviation();
        let homeostatic_factor = 1.0 - homeostatic_dev * 0.3;
        // Gut feeling boosts when positive
        let gut_factor = 1.0 + self.state.gut_state.max(0.0) * 0.1;

        self.phi_modulation = (body_factor * homeostatic_factor * gut_factor).clamp(0.5, 1.5);

        // Convert to CoreAffect for telemetry
        let affect = self.state.to_core_affect();

        VirtualBodyState {
            valence: affect.valence,
            arousal: affect.arousal,
            dominance: affect.dominance,
            phi_modulation: self.phi_modulation,
            homeostatic_deviation: homeostatic_dev,
        }
    }

    /// Get the current interoceptive state.
    pub fn interoceptive_state(&self) -> &InteroceptiveState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_signals() -> CognitiveSignals {
        CognitiveSignals {
            prediction_error: 0.1,
            coherence: 0.5,
            prediction_confidence: 0.6,
            unified_psi: 0.4,
            flow_intensity: 0.0,
            in_flow: false,
            curiosity_boredom: 0.2,
            fep_learning_signal: 0.3,
            error_trend: 0.0,
            cycles_per_second: 50.0,
            target_frequency: 50.0,
        }
    }

    #[test]
    fn test_virtual_body_initial_state() {
        let body = VirtualBody::new();
        assert!((body.phi_modulation - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_virtual_body_update_returns_valid_state() {
        let mut body = VirtualBody::new();
        let state = body.update(&default_signals());
        assert!(state.valence >= -1.0 && state.valence <= 1.0);
        assert!(state.arousal >= 0.0 && state.arousal <= 1.0);
        assert!(state.dominance >= -1.0 && state.dominance <= 1.0);
        assert!(state.phi_modulation >= 0.5 && state.phi_modulation <= 1.5);
    }

    #[test]
    fn test_high_error_raises_heart_rate() {
        let mut body = VirtualBody::new();
        let mut signals = default_signals();

        // Low error baseline
        signals.prediction_error = 0.01;
        body.update(&signals);
        let low_hr = body.interoceptive_state().heart_rate;

        // High error
        signals.prediction_error = 0.9;
        body.update(&signals);
        let high_hr = body.interoceptive_state().heart_rate;

        assert!(high_hr > low_hr, "Higher error should raise heart rate");
    }

    #[test]
    fn test_flow_improves_gut_state() {
        let mut body = VirtualBody::new();

        // No flow
        let mut signals = default_signals();
        signals.in_flow = false;
        body.update(&signals);
        let no_flow_gut = body.interoceptive_state().gut_state;

        // In flow
        signals.in_flow = true;
        signals.flow_intensity = 0.8;
        body.update(&signals);
        let flow_gut = body.interoceptive_state().gut_state;

        assert!(flow_gut > no_flow_gut, "Flow should improve gut state");
    }

    #[test]
    fn test_phi_modulation_bounds() {
        let mut body = VirtualBody::new();
        let mut signals = default_signals();

        // Extreme values shouldn't break bounds
        signals.fep_learning_signal = 1.0;
        signals.prediction_error = 1.0;
        signals.curiosity_boredom = 1.0;
        signals.prediction_confidence = 0.0;
        let state = body.update(&signals);
        assert!(state.phi_modulation >= 0.5 && state.phi_modulation <= 1.5);

        // Opposite extreme
        signals.fep_learning_signal = 0.0;
        signals.prediction_error = 0.0;
        signals.prediction_confidence = 1.0;
        signals.curiosity_boredom = 0.0;
        let state = body.update(&signals);
        assert!(state.phi_modulation >= 0.5 && state.phi_modulation <= 1.5);
    }

    #[test]
    fn test_smoothing_dampens_sudden_changes() {
        let mut body = VirtualBody::new();

        // Establish baseline with several low-error cycles
        let mut signals = default_signals();
        signals.prediction_error = 0.05;
        for _ in 0..10 {
            body.update(&signals);
        }
        let baseline_hr = body.interoceptive_state().heart_rate;

        // Sudden spike
        signals.prediction_error = 0.95;
        body.update(&signals);
        let spiked_hr = body.interoceptive_state().heart_rate;

        // Should be higher but not at maximum (smoothing dampens)
        assert!(spiked_hr > baseline_hr);
        assert!(spiked_hr < 1.0, "Smoothing should prevent instant max");
    }

    #[test]
    fn test_boredom_maps_to_hunger() {
        let mut body = VirtualBody::new();
        let mut signals = default_signals();

        signals.curiosity_boredom = 0.0;
        body.update(&signals);
        let low_hunger = body.interoceptive_state().hunger;

        signals.curiosity_boredom = 0.9;
        body.update(&signals);
        let high_hunger = body.interoceptive_state().hunger;

        assert!(
            high_hunger > low_hunger,
            "Boredom should increase hunger for novelty"
        );
    }
}
