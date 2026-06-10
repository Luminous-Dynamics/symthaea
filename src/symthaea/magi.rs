// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MAGI loop calibration methods.
//!
//! All items in this module are gated behind `#[cfg(feature = "magi_loop")]`.

use crate::consciousness::recursive_improvement::{CalibrationSummary, PredictionDomain};
use crate::mind::SemanticIntent;
use crate::mind::structured_thought::EpistemicStatus;

use super::Symthaea;

impl Symthaea {
    /// Get the current calibration summary.
    ///
    /// Returns global and per-domain Brier scores, ECE, accuracy, and
    /// whether the system is currently well-calibrated.
    pub fn calibration_summary(&self) -> CalibrationSummary {
        self.calibration.calibration_summary()
    }

    /// Map EpistemicStatus to a confidence float for calibration tracking.
    ///
    /// These values represent the system's belief about being correct:
    /// - Certain: 0.95 (very high confidence)
    /// - Probable: 0.75 (moderate-high confidence)
    /// - Uncertain: 0.45 (moderate-low confidence)
    /// - Unknown: 0.15 (very low confidence)
    /// - OutOfDomain: 0.10 (minimal confidence)
    pub(super) fn epistemic_to_confidence(status: &EpistemicStatus) -> f64 {
        match status {
            EpistemicStatus::Certain => 0.95,
            EpistemicStatus::Probable => 0.75,
            EpistemicStatus::Uncertain => 0.45,
            EpistemicStatus::Unknown => 0.15,
            EpistemicStatus::OutOfDomain => 0.10,
        }
    }

    /// Map a confidence float back to EpistemicStatus.
    ///
    /// Inverse of `epistemic_to_confidence`, using midpoint thresholds.
    pub(super) fn confidence_to_epistemic(confidence: f64) -> EpistemicStatus {
        if confidence >= 0.85 {
            EpistemicStatus::Certain
        } else if confidence >= 0.60 {
            EpistemicStatus::Probable
        } else if confidence >= 0.30 {
            EpistemicStatus::Uncertain
        } else if confidence >= 0.12 {
            EpistemicStatus::Unknown
        } else {
            EpistemicStatus::OutOfDomain
        }
    }

    /// Map SemanticIntent to a PredictionDomain for calibration tracking.
    ///
    /// Groups different intents into calibration domains:
    /// - Answer, Clarify -> Factual (knowledge-based predictions)
    /// - ProposeAction -> ToolUse (action outcome predictions)
    /// - Acknowledge, Continue -> UserBehavior (social interaction predictions)
    /// - Reflect -> SystemState (introspective predictions)
    /// - ExpressUncertainty, Unknown -> Factual (default calibration domain)
    pub(super) fn map_intent_to_domain(intent: &SemanticIntent) -> PredictionDomain {
        match intent {
            SemanticIntent::Answer | SemanticIntent::Clarify => PredictionDomain::Factual,
            SemanticIntent::ProposeAction => PredictionDomain::ToolUse,
            SemanticIntent::Acknowledge | SemanticIntent::Continue => {
                PredictionDomain::UserBehavior
            }
            SemanticIntent::Reflect => PredictionDomain::SystemState,
            SemanticIntent::ExpressUncertainty | SemanticIntent::Unknown => {
                PredictionDomain::Factual
            }
        }
    }
}
