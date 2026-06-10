// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metacognitive Subsystem
//!
//! Standalone implementation of metacognitive monitoring as a `ConsciousnessSubsystem`.
//! Tracks processing quality, confidence, and coherence metrics,
//! updating the consciousness state accordingly.

use super::binary_hv::BinaryHV;
use super::consciousness_integration::ConsciousnessState;
use super::consciousness_subsystem::{ConsciousnessSubsystem, SubsystemError};

/// Metacognitive monitoring subsystem.
///
/// Tracks confidence and coherence based on state trajectories,
/// predicting future Phi and detecting processing degradation.
pub struct MetacognitiveSubsystem {
    enabled: bool,
    /// Rolling history of Phi values (up to `window_size`)
    phi_history: Vec<f64>,
    /// Window for trend estimation
    window_size: usize,
    /// Number of cycles processed
    cycle_count: u64,
}

impl MetacognitiveSubsystem {
    /// Create a new metacognitive subsystem.
    pub fn new() -> Self {
        Self {
            enabled: true,
            phi_history: Vec::new(),
            window_size: 10,
            cycle_count: 0,
        }
    }

    /// Create with a custom history window.
    pub fn with_window(window_size: usize) -> Self {
        Self {
            window_size,
            ..Self::new()
        }
    }

    /// Get the estimated Phi trend (positive = improving).
    pub fn phi_trend(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.0;
        }
        let recent = self
            .phi_history
            .last()
            .expect("phi_history has >= 2 elements");
        let oldest = self
            .phi_history
            .first()
            .expect("phi_history has >= 2 elements");
        recent - oldest
    }
}

impl Default for MetacognitiveSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessSubsystem for MetacognitiveSubsystem {
    fn name(&self) -> &str {
        "metacognitive"
    }

    fn process_cycle(
        &mut self,
        state: &mut ConsciousnessState,
        _inputs: &[BinaryHV],
    ) -> Result<(), SubsystemError> {
        self.cycle_count += 1;

        // Record Phi history
        self.phi_history.push(state.phi);
        if self.phi_history.len() > self.window_size {
            self.phi_history.remove(0);
        }

        // Estimate confidence from Phi stability (low variance = high confidence)
        if self.phi_history.len() >= 3 {
            let mean: f64 = self.phi_history.iter().sum::<f64>() / self.phi_history.len() as f64;
            let variance: f64 = self
                .phi_history
                .iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>()
                / self.phi_history.len() as f64;

            // Confidence = inverse of variance (bounded to [0, 1])
            let confidence = 1.0 / (1.0 + variance * 10.0);
            state.metacognitive_confidence = confidence.clamp(0.0, 1.0);

            // Coherence = how well Phi tracks with consciousness_level
            let level_diff = (state.phi - state.consciousness_level).abs();
            state.metacognitive_coherence = (1.0 - level_diff).clamp(0.0, 1.0);

            // Predict next Phi from trend
            let trend = self.phi_trend();
            state.predicted_phi = Some((state.phi + trend * 0.5).clamp(0.0, 1.0));
            state.phi_trend = trend;
        }

        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// =============================================================================
// ENGINE-BACKED VARIANT: Wraps the real MetaConsciousness engine
// =============================================================================

use super::meta_consciousness::{MetaConfig, MetaConsciousness, MetaConsciousnessState};

/// Engine-backed meta-consciousness subsystem.
///
/// Wraps the full `MetaConsciousness` engine (Φ-of-Φ, strange loops)
/// in the `ConsciousnessSubsystem` trait for pluggable registration.
pub struct MetaConsciousnessWrapped {
    engine: MetaConsciousness,
    latest_state: Option<MetaConsciousnessState>,
}

impl MetaConsciousnessWrapped {
    /// Create with default config and the given number of components.
    pub fn new(num_components: usize) -> Self {
        Self {
            engine: MetaConsciousness::new(num_components, MetaConfig::default()),
            latest_state: None,
        }
    }

    /// Create with custom config.
    pub fn with_config(num_components: usize, config: MetaConfig) -> Self {
        Self {
            engine: MetaConsciousness::new(num_components, config),
            latest_state: None,
        }
    }

    /// Get the latest meta-consciousness state (after at least one cycle).
    pub fn latest_state(&self) -> Option<&MetaConsciousnessState> {
        self.latest_state.as_ref()
    }
}

impl ConsciousnessSubsystem for MetaConsciousnessWrapped {
    fn name(&self) -> &str {
        "meta_consciousness"
    }

    fn process_cycle(
        &mut self,
        state: &mut ConsciousnessState,
        inputs: &[BinaryHV],
    ) -> Result<(), SubsystemError> {
        let meta_state = self.engine.meta_reflect(inputs);
        state.metacognitive_confidence = meta_state.metacognitive_confidence;
        self.latest_state = Some(meta_state);
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metacognitive_subsystem_creation() {
        let sub = MetacognitiveSubsystem::new();
        assert!(sub.is_enabled());
        assert_eq!(sub.name(), "metacognitive");
        assert_eq!(sub.phi_trend(), 0.0);
    }

    #[test]
    fn test_metacognitive_processes_state() {
        let mut sub = MetacognitiveSubsystem::new();
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        // Run several cycles with increasing Phi
        for i in 0..5 {
            state.phi = 0.3 + i as f64 * 0.1;
            sub.process_cycle(&mut state, &inputs).unwrap();
        }

        // After 5 cycles, should have valid metacognitive metrics
        assert!(state.metacognitive_confidence >= 0.0 && state.metacognitive_confidence <= 1.0);
        assert!(state.metacognitive_coherence >= 0.0 && state.metacognitive_coherence <= 1.0);
        assert!(state.predicted_phi.is_some());
    }

    #[test]
    fn test_metacognitive_trend_detection() {
        let mut sub = MetacognitiveSubsystem::new();
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        // Rising Phi
        for i in 0..5 {
            state.phi = 0.2 + i as f64 * 0.1;
            sub.process_cycle(&mut state, &inputs).unwrap();
        }

        assert!(
            sub.phi_trend() > 0.0,
            "Trend should be positive for rising Phi"
        );
    }

    #[test]
    fn test_metacognitive_with_custom_window() {
        let mut sub = MetacognitiveSubsystem::with_window(3);
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        for i in 0..10 {
            state.phi = 0.5;
            sub.process_cycle(&mut state, &inputs).unwrap();
        }

        // Only last 3 values should be in history
        assert_eq!(sub.phi_history.len(), 3);
    }
}
