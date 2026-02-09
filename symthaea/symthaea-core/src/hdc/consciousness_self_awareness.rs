//! Self-Awareness Subsystem
//!
//! Standalone implementation of recursive self-modeling as a `ConsciousnessSubsystem`.
//! Tracks self-model confidence, accuracy, and introspective depth.

use super::binary_hv::BinaryHV;
use super::consciousness_integration::ConsciousnessState;
use super::consciousness_subsystem::ConsciousnessSubsystem;

/// Self-awareness subsystem for recursive self-modeling.
///
/// Maintains a simple self-model that predicts its own state
/// and measures prediction accuracy over time.
pub struct SelfAwarenessSubsystem {
    enabled: bool,
    /// Predicted Phi from last cycle
    predicted_phi: f64,
    /// Running prediction error (EMA)
    prediction_error_ema: f64,
    /// Self-awareness level (increases with accurate predictions)
    awareness_level: f64,
    /// Cycle count
    cycle_count: u64,
    /// EMA alpha for prediction error smoothing
    ema_alpha: f64,
}

impl SelfAwarenessSubsystem {
    /// Create a new self-awareness subsystem.
    pub fn new() -> Self {
        Self {
            enabled: true,
            predicted_phi: 0.5,
            prediction_error_ema: 0.5,
            awareness_level: 0.0,
            cycle_count: 0,
            ema_alpha: 0.2,
        }
    }

    /// Get the current self-awareness level.
    pub fn awareness_level(&self) -> f64 {
        self.awareness_level
    }
}

impl Default for SelfAwarenessSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessSubsystem for SelfAwarenessSubsystem {
    fn name(&self) -> &str {
        "self_awareness"
    }

    fn process_cycle(&mut self, state: &mut ConsciousnessState, _inputs: &[BinaryHV]) {
        self.cycle_count += 1;

        // Compute prediction error
        let error = (state.phi - self.predicted_phi).abs();
        self.prediction_error_ema = self.ema_alpha * error + (1.0 - self.ema_alpha) * self.prediction_error_ema;

        // Prediction accuracy = inverse of error
        let accuracy = (1.0 - self.prediction_error_ema).clamp(0.0, 1.0);

        // Self-awareness grows with accurate predictions (slow convergence)
        self.awareness_level = (self.awareness_level * 0.95 + accuracy * 0.05).clamp(0.0, 1.0);

        // Update state
        state.self_model_confidence = accuracy;
        state.self_model_accuracy = accuracy;

        // Simple prediction for next cycle: EMA of recent Phi
        self.predicted_phi = self.ema_alpha * state.phi + (1.0 - self.ema_alpha) * self.predicted_phi;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_awareness_creation() {
        let sub = SelfAwarenessSubsystem::new();
        assert!(sub.is_enabled());
        assert_eq!(sub.name(), "self_awareness");
        assert_eq!(sub.awareness_level(), 0.0);
    }

    #[test]
    fn test_self_awareness_improves_with_stable_phi() {
        let mut sub = SelfAwarenessSubsystem::new();
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        // Stable Phi = easy to predict = awareness increases
        for _ in 0..50 {
            state.phi = 0.5;
            sub.process_cycle(&mut state, &inputs);
        }

        assert!(sub.awareness_level() > 0.0, "Awareness should increase with stable Phi");
        assert!(state.self_model_confidence > 0.5, "Confidence should be high with stable input");
    }

    #[test]
    fn test_self_awareness_bounded() {
        let mut sub = SelfAwarenessSubsystem::new();
        let mut state = ConsciousnessState::default();
        let inputs = vec![BinaryHV::random(42)];

        for i in 0..100 {
            state.phi = (i as f64 * 0.37).sin().abs(); // Chaotic-ish pattern
            sub.process_cycle(&mut state, &inputs);
        }

        assert!(sub.awareness_level() >= 0.0 && sub.awareness_level() <= 1.0);
        assert!(state.self_model_confidence >= 0.0 && state.self_model_confidence <= 1.0);
        assert!(state.self_model_accuracy >= 0.0 && state.self_model_accuracy <= 1.0);
    }
}
