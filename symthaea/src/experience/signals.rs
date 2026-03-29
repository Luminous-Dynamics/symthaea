// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Principled Signals Module
//!
//! This module replaces the "Φ-everywhere" cargo cult with proper,
//! meaningful control signals derived from Active Inference and
//! information theory.
//!
//! ## Signal Definitions
//!
//! | Signal | Source | Meaning | Used For |
//! |--------|--------|---------|----------|
//! | Prediction Error | Active Inference | What surprised me? | Attention, learning |
//! | Uncertainty | Entropy | What don't I know? | Question generation |
//! | Coherence | Integration | Does this hang together? | Output validation |
//! | Confidence | Inverse MU | How sure am I? | Response vs ask |
//! | Salience | Goal-relevance | What matters now? | Primitive selection |
//!
//! Φ (phi) remains as a **monitoring metric** for overall system integration,
//! not as a control signal.

use serde::{Deserialize, Serialize};

/// The five principled signals that drive Symthaea's behavior
///
/// These replace the cargo-cult use of Φ everywhere with signals
/// that have clear, operational definitions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrincipledSignals {
    /// **Prediction Error** (Active Inference)
    ///
    /// "What surprised me?"
    ///
    /// High when:
    /// - Input is unlike past experiences
    /// - Expected outcome didn't occur
    /// - User behaves unexpectedly
    ///
    /// Used for:
    /// - Directing attention
    /// - Triggering learning
    /// - Updating beliefs
    pub prediction_error: f32,

    /// **Uncertainty** (Entropy)
    ///
    /// "What don't I know?"
    ///
    /// High when:
    /// - GIS type is UNKNOWN_UNKNOWN
    /// - High variance in past outcomes
    /// - Conflicting information
    ///
    /// Used for:
    /// - Generating clarifying questions
    /// - Hedging responses
    /// - Requesting more information
    pub uncertainty: f32,

    /// **Coherence** (Integration)
    ///
    /// "Does this hang together?"
    ///
    /// High when:
    /// - Response aligns with values
    /// - Harmonies agree
    /// - No internal contradictions
    ///
    /// Used for:
    /// - Output validation
    /// - Detecting hallucinations
    /// - Ensuring consistency
    pub coherence: f32,

    /// **Confidence** (Inverse Moral Uncertainty)
    ///
    /// "How sure am I?"
    ///
    /// High when:
    /// - Low moral uncertainty (epistemic + axiological + deontic)
    /// - Clear value alignment
    /// - High coherence
    ///
    /// Used for:
    /// - Deciding to respond vs ask
    /// - Tone of response
    /// - Risk assessment
    pub confidence: f32,

    /// **Salience** (Goal-Relevance)
    ///
    /// "What matters for this goal?"
    ///
    /// High when:
    /// - Input matches current goals
    /// - High harmonic activation
    /// - User explicitly prioritizes
    ///
    /// Used for:
    /// - Primitive selection
    /// - Attention allocation
    /// - Resource prioritization
    pub salience: f32,

    /// **Φ (Phi) - MONITORING ONLY**
    ///
    /// The integrated information measure from IIT.
    ///
    /// NOT a control signal - used only for:
    /// - Overall system health monitoring
    /// - Research/logging purposes
    /// - Debugging integration issues
    ///
    /// If you find yourself using this for decisions,
    /// you're probably cargo-culting. Use the signals above.
    pub phi_monitor: f32,
}

impl PrincipledSignals {
    /// Create new signals with default (neutral) values
    pub fn new() -> Self {
        Self {
            prediction_error: 0.5,
            uncertainty: 0.5,
            coherence: 0.5,
            confidence: 0.5,
            salience: 0.5,
            phi_monitor: 0.0,
        }
    }

    /// Create signals indicating high surprise/novelty
    pub fn novel() -> Self {
        Self {
            prediction_error: 0.9,
            uncertainty: 0.7,
            coherence: 0.5,
            confidence: 0.3,
            salience: 0.8,
            phi_monitor: 0.0,
        }
    }

    /// Create signals indicating familiar territory
    pub fn familiar() -> Self {
        Self {
            prediction_error: 0.1,
            uncertainty: 0.2,
            coherence: 0.8,
            confidence: 0.8,
            salience: 0.5,
            phi_monitor: 0.0,
        }
    }

    /// Create signals indicating confusion/need for help
    pub fn confused() -> Self {
        Self {
            prediction_error: 0.6,
            uncertainty: 0.9,
            coherence: 0.3,
            confidence: 0.2,
            salience: 0.7,
            phi_monitor: 0.0,
        }
    }

    /// Check if we should ask a clarifying question
    pub fn should_clarify(&self) -> bool {
        self.uncertainty > 0.6 || (self.prediction_error > 0.5 && self.confidence < 0.4)
    }

    /// Check if we should consult values
    pub fn should_consult_values(&self) -> bool {
        self.confidence < 0.5 && self.salience > 0.7
    }

    /// Check if response coherence is acceptable
    pub fn coherence_acceptable(&self) -> bool {
        self.coherence >= 0.4
    }

    /// Compute a weighted decision score
    ///
    /// Higher = more ready to act
    /// Lower = more need for reflection/clarification
    pub fn action_readiness(&self) -> f32 {
        // Weighted combination favoring confidence and coherence
        let readiness = (self.confidence * 0.3)
            + (self.coherence * 0.3)
            + ((1.0 - self.uncertainty) * 0.2)
            + ((1.0 - self.prediction_error) * 0.2);

        readiness.clamp(0.0, 1.0)
    }

    /// Compute a weighted learning signal
    ///
    /// Higher = more to learn from this interaction
    pub fn learning_potential(&self) -> f32 {
        // High prediction error + high salience = valuable learning
        let potential =
            (self.prediction_error * 0.4) + (self.salience * 0.3) + (self.uncertainty * 0.3);

        potential.clamp(0.0, 1.0)
    }

    /// Get dominant signal name
    pub fn dominant_signal(&self) -> &'static str {
        let signals = [
            (self.prediction_error, "prediction_error"),
            (self.uncertainty, "uncertainty"),
            (self.coherence, "coherence"),
            (self.confidence, "confidence"),
            (self.salience, "salience"),
        ];

        signals
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, name)| *name)
            .unwrap_or("unknown")
    }

    /// Format as human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "PE:{:.2} U:{:.2} C:{:.2} Cf:{:.2} S:{:.2} [Φ:{:.2}]",
            self.prediction_error,
            self.uncertainty,
            self.coherence,
            self.confidence,
            self.salience,
            self.phi_monitor
        )
    }
}

/// Signal computation utilities
pub struct SignalComputer {
    /// Exponential moving average decay for temporal smoothing
    pub ema_alpha: f32,

    /// Previous signals for smoothing
    previous: Option<PrincipledSignals>,
}

impl Default for SignalComputer {
    fn default() -> Self {
        Self {
            ema_alpha: 0.3,
            previous: None,
        }
    }
}

impl SignalComputer {
    pub fn new(ema_alpha: f32) -> Self {
        Self {
            ema_alpha,
            previous: None,
        }
    }

    /// Smooth signals using exponential moving average
    pub fn smooth(&mut self, current: PrincipledSignals) -> PrincipledSignals {
        let smoothed = match &self.previous {
            None => current.clone(),
            Some(prev) => PrincipledSignals {
                prediction_error: self.ema(prev.prediction_error, current.prediction_error),
                uncertainty: self.ema(prev.uncertainty, current.uncertainty),
                coherence: self.ema(prev.coherence, current.coherence),
                confidence: self.ema(prev.confidence, current.confidence),
                salience: self.ema(prev.salience, current.salience),
                phi_monitor: self.ema(prev.phi_monitor, current.phi_monitor),
            },
        };

        self.previous = Some(smoothed.clone());
        smoothed
    }

    fn ema(&self, previous: f32, current: f32) -> f32 {
        symthaea_core::math::ema_update(previous, current, self.ema_alpha)
    }

    /// Compute prediction error from similarity scores
    pub fn compute_prediction_error(similarities: &[f32]) -> f32 {
        if similarities.is_empty() {
            return 1.0; // No data = high surprise
        }

        let avg: f32 = similarities.iter().sum::<f32>() / similarities.len() as f32;
        1.0 - avg
    }

    /// Compute uncertainty from outcome variance
    pub fn compute_uncertainty_from_outcomes(outcomes: &[bool]) -> f32 {
        if outcomes.len() < 2 {
            return 0.5; // Not enough data
        }

        let successes = outcomes.iter().filter(|&&x| x).count() as f32;
        let n = outcomes.len() as f32;
        let p = successes / n;

        // Bernoulli variance: p * (1 - p)
        // Max variance (0.25) at p = 0.5
        let variance = p * (1.0 - p);

        // Normalize to [0, 1] - max variance maps to 1.0
        (variance / 0.25).min(1.0)
    }

    /// Compute coherence from harmonic agreement
    pub fn compute_coherence_from_harmonics(activations: &[f32]) -> f32 {
        if activations.is_empty() {
            return 0.5;
        }

        let mean = activations.iter().sum::<f32>() / activations.len() as f32;
        let variance: f32 =
            activations.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / activations.len() as f32;

        // Low variance = high coherence (harmonies agree)
        1.0 - variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_clarify() {
        let signals = PrincipledSignals::confused();
        assert!(signals.should_clarify());

        let signals = PrincipledSignals::familiar();
        assert!(!signals.should_clarify());
    }

    #[test]
    fn test_action_readiness() {
        let confident = PrincipledSignals::familiar();
        let confused = PrincipledSignals::confused();

        assert!(confident.action_readiness() > confused.action_readiness());
    }

    #[test]
    fn test_learning_potential() {
        let novel = PrincipledSignals::novel();
        let familiar = PrincipledSignals::familiar();

        assert!(novel.learning_potential() > familiar.learning_potential());
    }

    #[test]
    fn test_signal_smoothing() {
        let mut computer = SignalComputer::new(0.5);

        let s1 = PrincipledSignals {
            prediction_error: 0.2,
            ..Default::default()
        };
        let s2 = PrincipledSignals {
            prediction_error: 0.8,
            ..Default::default()
        };

        let smoothed1 = computer.smooth(s1);
        assert_eq!(smoothed1.prediction_error, 0.2);

        let smoothed2 = computer.smooth(s2);
        // EMA: 0.5 * 0.8 + 0.5 * 0.2 = 0.5
        assert!((smoothed2.prediction_error - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prediction_error_computation() {
        let high_sim = vec![0.9, 0.85, 0.95];
        let pe = SignalComputer::compute_prediction_error(&high_sim);
        assert!(pe < 0.2); // Low error when similar

        let low_sim = vec![0.1, 0.2, 0.15];
        let pe = SignalComputer::compute_prediction_error(&low_sim);
        assert!(pe > 0.8); // High error when dissimilar
    }

    #[test]
    fn test_uncertainty_from_outcomes() {
        // All successes = low uncertainty
        let all_success = vec![true, true, true, true];
        let u = SignalComputer::compute_uncertainty_from_outcomes(&all_success);
        assert!(u < 0.1);

        // Mixed = high uncertainty
        let mixed = vec![true, false, true, false];
        let u = SignalComputer::compute_uncertainty_from_outcomes(&mixed);
        assert!(u > 0.9);
    }

    #[test]
    fn test_coherence_from_harmonics() {
        // All same = high coherence
        let uniform = vec![0.5, 0.5, 0.5, 0.5];
        let c = SignalComputer::compute_coherence_from_harmonics(&uniform);
        assert!(c > 0.99);

        // Varied = lower coherence
        let varied = vec![0.1, 0.9, 0.2, 0.8];
        let c = SignalComputer::compute_coherence_from_harmonics(&varied);
        assert!(c < 0.7);
    }
}
