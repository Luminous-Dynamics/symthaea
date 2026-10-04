// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Closed-loop speech realization feedback contracts.
//!
//! This module stays independent of any particular synthesizer. Broca declares
//! a target; a downstream realization system reports measured consequences;
//! the error remains typed rather than being collapsed into one quality score.

use serde::{Deserialize, Serialize};

use crate::speech_plan::{EpistemicDelivery, ProsodicIntent, SpeechPlan};

pub const SPEECH_FEEDBACK_VERSION: &str = "broca-speech-feedback-v1";

/// Desired sensory target emitted by Broca for a downstream realization layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeechSensoryTarget {
    /// Normalized pitch-range target.
    pub pitch_range: f32,
    /// Normalized relative energy/prominence target.
    pub prominence: f32,
    /// Relative speech-rate target.
    pub rate: f32,
    /// Relative pause target.
    pub pause_weight: f32,
    pub epistemic_delivery: EpistemicDelivery,
}

impl SpeechSensoryTarget {
    /// Construct a target from prosody while preserving the caller's epistemic state.
    pub fn from_prosody(
        prosody: ProsodicIntent,
        epistemic_delivery: EpistemicDelivery,
    ) -> Self {
        Self {
            pitch_range: prosody.pitch_range,
            prominence: prosody.prominence,
            rate: prosody.rate,
            pause_weight: prosody.pause_weight,
            epistemic_delivery,
        }
    }

    pub fn from_plan(plan: &SpeechPlan) -> Self {
        Self {
            pitch_range: plan.prosody.pitch_range,
            prominence: plan.prosody.prominence,
            rate: plan.prosody.rate,
            pause_weight: plan.prosody.pause_weight,
            epistemic_delivery: plan.epistemic_delivery,
        }
    }
}

/// Speaker- and backend-neutral observations returned by a realization layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeechSensoryObservation {
    pub pitch_range: Option<f32>,
    pub prominence: Option<f32>,
    pub rate: Option<f32>,
    pub pause_weight: Option<f32>,
}

/// Per-feature absolute discrepancies between intended and observed realization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeechFeedbackError {
    pub pitch_range: Option<f32>,
    pub prominence: Option<f32>,
    pub rate: Option<f32>,
    pub pause_weight: Option<f32>,
    /// Mean of the available normalized errors.
    pub mean_error: f32,
    pub compared_features: u8,
}

impl SpeechFeedbackError {
    /// Compute feature-wise normalized absolute error.
    ///
    /// Non-finite observations are treated as missing, not as extreme errors.
    pub fn compare(
        target: SpeechSensoryTarget,
        observation: SpeechSensoryObservation,
    ) -> Self {
        let pitch_range = abs_error(target.pitch_range, observation.pitch_range);
        let prominence = abs_error(target.prominence, observation.prominence);
        let rate = relative_error(target.rate, observation.rate);
        let pause_weight = abs_error(target.pause_weight, observation.pause_weight);

        let values = [pitch_range, prominence, rate, pause_weight];
        let mut total = 0.0;
        let mut compared = 0u8;
        for value in values.into_iter().flatten() {
            total += value;
            compared += 1;
        }

        Self {
            pitch_range,
            prominence,
            rate,
            pause_weight,
            mean_error: if compared == 0 {
                0.0
            } else {
                total / f32::from(compared)
            },
            compared_features: compared,
        }
    }

    pub fn exceeds(&self, threshold: f32) -> bool {
        self.compared_features > 0
            && self.mean_error.is_finite()
            && self.mean_error > threshold.max(0.0)
    }
}

/// End-to-end closed-loop evidence linking a plan, observation, and discrepancy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechFeedbackReceipt {
    pub version: String,
    pub plan_surface: String,
    pub target: SpeechSensoryTarget,
    pub observation: SpeechSensoryObservation,
    pub error: SpeechFeedbackError,
}

impl SpeechFeedbackReceipt {
    pub fn new(plan: &SpeechPlan, observation: SpeechSensoryObservation) -> Self {
        let target = SpeechSensoryTarget::from_plan(plan);
        let error = SpeechFeedbackError::compare(target, observation);
        Self {
            version: SPEECH_FEEDBACK_VERSION.to_string(),
            plan_surface: plan.grounding_surface(),
            target,
            observation,
            error,
        }
    }

    /// True when the realization layer provided no usable measurements.
    pub fn is_unobserved(&self) -> bool {
        self.error.compared_features == 0
    }
}

fn finite(value: Option<f32>) -> Option<f32> {
    value.filter(|v| v.is_finite())
}

fn abs_error(target: f32, observed: Option<f32>) -> Option<f32> {
    finite(observed).map(|value| (value - target).abs().clamp(0.0, 1.0))
}

fn relative_error(target: f32, observed: Option<f32>) -> Option<f32> {
    finite(observed).map(|value| {
        let denom = target.abs().max(0.1);
        ((value - target).abs() / denom).clamp(0.0, 1.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::StructuredDecoder;
    use crate::encoder::ThoughtChannels;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn receipt_binds_target_to_exact_plan_surface() {
        let genesis = GenesisSeed::from_phrase("broca-feedback-receipt");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        let plan = SpeechPlan::from_readout(&channels, &readout);

        let observation = SpeechSensoryObservation {
            pitch_range: Some(plan.prosody.pitch_range),
            prominence: Some(plan.prosody.prominence),
            rate: Some(plan.prosody.rate),
            pause_weight: Some(plan.prosody.pause_weight),
        };
        let receipt = SpeechFeedbackReceipt::new(&plan, observation);

        assert_eq!(receipt.version, SPEECH_FEEDBACK_VERSION);
        assert_eq!(receipt.plan_surface, plan.grounding_surface());
        assert_eq!(receipt.error.compared_features, 4);
        assert!(receipt.error.mean_error < f32::EPSILON);
        assert!(!receipt.is_unobserved());
    }

    #[test]
    fn missing_observations_do_not_become_false_zero_errors() {
        let genesis = GenesisSeed::from_phrase("broca-feedback-missing");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        let plan = SpeechPlan::from_readout(&channels, &readout);

        let receipt = SpeechFeedbackReceipt::new(
            &plan,
            SpeechSensoryObservation {
                rate: Some(plan.prosody.rate * 1.25),
                ..Default::default()
            },
        );

        assert_eq!(receipt.error.compared_features, 1);
        assert!(receipt.error.pitch_range.is_none());
        assert!(receipt.error.prominence.is_none());
        assert!(receipt.error.pause_weight.is_none());
        assert!(receipt.error.mean_error > 0.0);
    }

    #[test]
    fn large_rate_deviation_triggers_threshold() {
        let target = SpeechSensoryTarget {
            pitch_range: 1.0,
            prominence: 0.5,
            rate: 1.0,
            pause_weight: 0.5,
            epistemic_delivery: EpistemicDelivery::Assertive,
        };
        let observation = SpeechSensoryObservation {
            rate: Some(0.5),
            ..Default::default()
        };
        let error = SpeechFeedbackError::compare(target, observation);

        assert_eq!(error.compared_features, 1);
        assert!(error.exceeds(0.4));
        assert!(!error.exceeds(0.6));
    }

    #[test]
    fn non_finite_observation_is_unobserved_for_that_feature() {
        let target = SpeechSensoryTarget {
            pitch_range: 1.0,
            prominence: 0.5,
            rate: 1.0,
            pause_weight: 0.5,
            epistemic_delivery: EpistemicDelivery::Qualified,
        };
        let observation = SpeechSensoryObservation {
            pitch_range: Some(f32::NAN),
            prominence: Some(f32::INFINITY),
            rate: None,
            pause_weight: Some(0.5),
        };
        let error = SpeechFeedbackError::compare(target, observation);

        assert_eq!(error.compared_features, 1);
        assert!(error.pause_weight.is_some());
        assert!(error.pitch_range.is_none());
        assert!(error.prominence.is_none());
    }
}
