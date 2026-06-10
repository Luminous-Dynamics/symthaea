// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phi-optimized audio: generate music that maximizes consciousness integration.
//!
//! Uses the audio feedback loop (#9) to measure how generated music affects
//! Phi, then modulates synthesis parameters toward a target Phi value.
//! This is the inverse problem: given a desired consciousness state,
//! what music produces it?
//!
//! # Strategy
//!
//! Gradient-free optimization via perturbation:
//! 1. Measure current Phi from consciousness state
//! 2. Compute Phi error (target - current)
//! 3. Modulate synthesis parameters proportional to error
//! 4. Higher error → more exploration (harmonic tension, rhythmic complexity)
//! 5. Lower error → sustain current parameters (convergence)

use crate::MusicalState;

/// Target Phi optimization mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhiTarget {
    /// Maximize Phi (default: target = 0.9).
    Maximize,
    /// Target a specific Phi value.
    Specific(f32),
    /// Minimize Phi (for contrast/research).
    Minimize,
}

impl Default for PhiTarget {
    fn default() -> Self {
        Self::Maximize
    }
}

/// Phi-optimized audio parameter controller.
pub struct PhiOptimizer {
    target: PhiTarget,
    /// Current Phi estimate from consciousness state.
    current_phi: f32,
    /// Error signal (target - current).
    phi_error: f32,
    /// Trend: positive = Phi increasing, negative = decreasing.
    phi_trend: f32,
    /// EMA-smoothed Phi for stable optimization.
    phi_ema: f32,
    /// Learning rate for parameter modulation.
    learning_rate: f32,
    /// History of Phi values for trend estimation.
    phi_history: Vec<f32>,
    /// Cycles since last improvement.
    stagnation_counter: u32,
}

impl PhiOptimizer {
    pub fn new(target: PhiTarget) -> Self {
        Self {
            target,
            current_phi: 0.5,
            phi_error: 0.0,
            phi_trend: 0.0,
            phi_ema: 0.5,
            learning_rate: 0.1,
            phi_history: Vec::with_capacity(64),
            stagnation_counter: 0,
        }
    }

    /// Update optimizer with current consciousness level (Phi proxy).
    pub fn update_phi(&mut self, phi: f32) {
        let prev_phi = self.phi_ema;
        self.current_phi = phi;

        // EMA smoothing
        let alpha = 0.1;
        self.phi_ema += alpha * (phi - self.phi_ema);

        // Compute error
        let target_val = match self.target {
            PhiTarget::Maximize => 0.9,
            PhiTarget::Specific(t) => t,
            PhiTarget::Minimize => 0.1,
        };
        self.phi_error = target_val - self.phi_ema;

        // Trend estimation
        self.phi_trend = self.phi_ema - prev_phi;

        // Stagnation detection
        self.phi_history.push(phi);
        if self.phi_history.len() > 64 {
            self.phi_history.remove(0);
        }
        if self.phi_trend.abs() < 0.001 {
            self.stagnation_counter += 1;
        } else {
            self.stagnation_counter = 0;
        }
    }

    /// Modulate a MusicalState to optimize toward target Phi.
    ///
    /// When Phi is below target:
    /// - Increase harmonic tension (dissonant intervals → more information)
    /// - Increase rhythmic complexity (shorter notes → more temporal structure)
    /// - Increase spectral flux (FM modulation → more spectral change)
    ///
    /// When Phi is above target:
    /// - Sustain current parameters (don't rock the boat)
    /// - Slight convergence toward stable patterns
    ///
    /// When stagnating:
    /// - Inject exploration (random perturbation to escape local minima)
    pub fn modulate_state(&self, state: &mut MusicalState) {
        let lr = self.learning_rate;
        let err = self.phi_error;

        if err > 0.05 {
            // Phi too low — increase integration pressure
            // More prediction error → more surprise → Phi responds to integration
            state.prediction_error += lr * err * 0.3;
            // More arousal → faster dynamics → more temporal structure
            state.arousal += lr * err * 0.2;
            // More dopamine → brighter timbre → more spectral content
            state.dopamine += lr * err * 0.15;
            // Increase InfinitePlay harmony → more tension
            state.harmony_activations[3] += lr * err * 0.1;
        } else if err < -0.05 {
            // Phi too high (or minimizing) — reduce complexity
            state.prediction_error -= lr * err.abs() * 0.2;
            state.arousal -= lr * err.abs() * 0.15;
            // Increase SacredStillness → calmer
            state.harmony_activations[7] += lr * err.abs() * 0.1;
        }

        // Stagnation escape: inject exploration
        if self.stagnation_counter > 10 {
            let perturbation = 0.05 * (self.stagnation_counter as f32 / 20.0).min(1.0);
            state.prediction_error += perturbation;
            state.harmony_activations[3] += perturbation * 0.5; // playfulness
        }

        // Clamp all values
        state.prediction_error = state.prediction_error.clamp(0.0, 1.0);
        state.arousal = state.arousal.clamp(0.0, 1.0);
        state.dopamine = state.dopamine.clamp(0.0, 1.0);
        state.serotonin = state.serotonin.clamp(0.0, 1.0);
        for h in &mut state.harmony_activations {
            *h = h.clamp(0.0, 1.0);
        }
    }

    /// Get current optimization metrics.
    pub fn metrics(&self) -> PhiOptimizerMetrics {
        PhiOptimizerMetrics {
            current_phi: self.current_phi,
            phi_ema: self.phi_ema,
            phi_error: self.phi_error,
            phi_trend: self.phi_trend,
            stagnation_counter: self.stagnation_counter,
        }
    }
}

/// Diagnostic metrics from the Phi optimizer.
#[derive(Debug, Clone, Copy)]
pub struct PhiOptimizerMetrics {
    pub current_phi: f32,
    pub phi_ema: f32,
    pub phi_error: f32,
    pub phi_trend: f32,
    pub stagnation_counter: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximize_increases_complexity_when_phi_low() {
        let mut opt = PhiOptimizer::new(PhiTarget::Maximize);
        opt.update_phi(0.2); // low Phi

        let mut state = MusicalState::default();
        let orig_arousal = state.arousal;
        let orig_pred = state.prediction_error;
        opt.modulate_state(&mut state);

        assert!(state.arousal > orig_arousal, "should increase arousal");
        assert!(
            state.prediction_error > orig_pred,
            "should increase prediction error"
        );
    }

    #[test]
    fn maximize_stabilizes_when_phi_high() {
        let mut opt = PhiOptimizer::new(PhiTarget::Maximize);
        // Converge EMA to high Phi
        for _ in 0..20 {
            opt.update_phi(0.88);
        }

        let mut state = MusicalState::default();
        let orig_arousal = state.arousal;
        opt.modulate_state(&mut state);

        // Should not significantly change (within convergence zone)
        assert!(
            (state.arousal - orig_arousal).abs() < 0.05,
            "should stabilize near target"
        );
    }

    #[test]
    fn minimize_reduces_complexity() {
        let mut opt = PhiOptimizer::new(PhiTarget::Minimize);
        opt.update_phi(0.7); // Phi above minimize target

        let mut state = MusicalState {
            arousal: 0.6,
            prediction_error: 0.5,
            ..Default::default()
        };
        opt.modulate_state(&mut state);

        assert!(
            state.harmony_activations[7] > 0.3,
            "should increase stillness"
        );
    }

    #[test]
    fn specific_target() {
        let mut opt = PhiOptimizer::new(PhiTarget::Specific(0.5));
        opt.update_phi(0.3); // below target
        assert!(
            opt.phi_error > 0.0,
            "error should be positive when below target"
        );

        opt.update_phi(0.7); // above target
        // After EMA smoothing, error direction depends on history
        let metrics = opt.metrics();
        assert!(metrics.phi_ema > 0.3, "EMA should track upward");
    }

    #[test]
    fn stagnation_injects_exploration() {
        let mut opt = PhiOptimizer::new(PhiTarget::Maximize);
        // Stagnate at moderate Phi
        for _ in 0..30 {
            opt.update_phi(0.5);
        }

        assert!(opt.stagnation_counter > 10, "should detect stagnation");

        let mut state = MusicalState::default();
        let orig_pred = state.prediction_error;
        opt.modulate_state(&mut state);

        assert!(
            state.prediction_error > orig_pred + 0.04,
            "should inject exploration"
        );
    }

    #[test]
    fn all_values_stay_in_bounds() {
        let mut opt = PhiOptimizer::new(PhiTarget::Maximize);
        opt.update_phi(0.0);

        let mut state = MusicalState {
            arousal: 0.99,
            dopamine: 0.99,
            prediction_error: 0.99,
            ..Default::default()
        };
        opt.modulate_state(&mut state);

        assert!(state.arousal <= 1.0);
        assert!(state.dopamine <= 1.0);
        assert!(state.prediction_error <= 1.0);
        for &h in &state.harmony_activations {
            assert!(h >= 0.0 && h <= 1.0);
        }
    }
}
