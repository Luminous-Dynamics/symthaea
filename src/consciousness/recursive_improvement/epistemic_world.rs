// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic typing for replay, prediction, and imagination.
//!
//! The purpose of these types is to prevent generated counterfactuals from being
//! promoted to empirical evidence merely because they are useful or coherent.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldEvidenceKind {
    /// Directly observed in an executed environment.
    Recorded,
    /// Derived only by selecting/traversing recorded transitions.
    ReplayDerived,
    /// Interpolation between supported observations.
    Interpolated,
    /// Output of a learned predictive model.
    ModelPredicted,
    /// Explicit alternative to what actually occurred.
    Counterfactual,
    /// Prediction outside demonstrated support.
    Extrapolated,
    /// Deliberately generated stress/adversarial world.
    AdversarialGenerated,
}

impl WorldEvidenceKind {
    /// Whether this class can count as empirical outcome evidence without a
    /// separate validation step.
    pub fn is_empirical(self) -> bool {
        matches!(self, Self::Recorded | Self::ReplayDerived)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicWorldRecord {
    pub kind: WorldEvidenceKind,
    pub provenance_digest: String,
    pub model_version: Option<String>,
    pub confidence: Option<f64>,
    /// Distance from demonstrated support, if defined by the producer.
    pub support_distance: Option<f64>,
    pub causal_assumptions: Vec<String>,
    /// True only after an independent real/recorded observation confirms the claim.
    pub empirically_validated: bool,
}

impl EpistemicWorldRecord {
    pub fn may_promote_confidence(&self) -> bool {
        self.kind.is_empirical() || self.empirically_validated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counterfactual_is_not_empirical_by_default() {
        let record = EpistemicWorldRecord {
            kind: WorldEvidenceKind::Counterfactual,
            provenance_digest: "dream-1".into(),
            model_version: Some("wm-v1".into()),
            confidence: Some(0.9),
            support_distance: Some(0.1),
            causal_assumptions: vec!["do(action=x)".into()],
            empirically_validated: false,
        };
        assert!(!record.may_promote_confidence());
    }

    #[test]
    fn validated_counterfactual_can_promote_confidence() {
        let mut record = EpistemicWorldRecord {
            kind: WorldEvidenceKind::Counterfactual,
            provenance_digest: "dream-1".into(),
            model_version: Some("wm-v1".into()),
            confidence: Some(0.9),
            support_distance: Some(0.1),
            causal_assumptions: vec![],
            empirically_validated: false,
        };
        record.empirically_validated = true;
        assert!(record.may_promote_confidence());
    }
}
