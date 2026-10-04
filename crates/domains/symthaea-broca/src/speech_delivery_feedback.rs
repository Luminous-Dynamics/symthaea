// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic delivery feedback for speech production.
//!
//! Acoustic feedback answers "did the voice sound like the target?"
//! This module answers the complementary question "did the realized delivery
//! preserve the intended linguistic stance and information structure?"

use serde::{Deserialize, Serialize};

use crate::speech_plan::{ClauseMode, EpistemicDelivery, SpeechPlan};

pub const SPEECH_DELIVERY_FEEDBACK_VERSION: &str = "broca-speech-delivery-feedback-v1";

/// Intended semantic-delivery target extracted from a SpeechPlan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechDeliveryTarget {
    pub intent: String,
    pub clause_mode: ClauseMode,
    pub epistemic_delivery: EpistemicDelivery,
    pub focus_role: Option<String>,
}

/// Observed semantic-delivery state reported by a downstream formatter/realizer.
///
/// Each field is optional because not every realization backend can recover every
/// linguistic attribute from its output. Missing evidence is not interpreted as success.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SpeechDeliveryObservation {
    pub intent: Option<String>,
    pub clause_mode: Option<ClauseMode>,
    pub epistemic_delivery: Option<EpistemicDelivery>,
    pub focus: Option<ObservedFocus>,
}

/// Explicit distinction between observed focus on a role and an observed absence of focus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservedFocus {
    Role(String),
    None,
}

/// Exact-match discrepancies at the semantic delivery boundary.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SpeechDeliveryError {
    pub intent_match: Option<bool>,
    pub clause_mode_match: Option<bool>,
    pub epistemic_delivery_match: Option<bool>,
    pub focus_match: Option<bool>,
    pub compared_features: u8,
    pub mismatched_features: u8,
}

impl SpeechDeliveryError {
    pub fn compare(
        target: SpeechDeliveryTarget,
        observation: SpeechDeliveryObservation,
    ) -> Self {
        let intent_match = compare_optional_str(
            &target.intent,
            observation.intent.as_deref(),
        );
        let clause_mode_match =
            observation.clause_mode.map(|value| value == target.clause_mode);
        let epistemic_delivery_match = observation
            .epistemic_delivery
            .map(|value| value == target.epistemic_delivery);
        let focus_match = observation.focus.map(|observed| match observed {
            ObservedFocus::Role(role) => target.focus_role.as_deref() == Some(role.as_str()),
            ObservedFocus::None => target.focus_role.is_none(),
        });

        let values = [
            intent_match,
            clause_mode_match,
            epistemic_delivery_match,
            focus_match,
        ];
        let compared_features = values.iter().filter(|value| value.is_some()).count() as u8;
        let mismatched_features = values
            .iter()
            .filter(|value| matches!(value, Some(false)))
            .count() as u8;

        Self {
            intent_match,
            clause_mode_match,
            epistemic_delivery_match,
            focus_match,
            compared_features,
            mismatched_features,
        }
    }

    pub fn has_mismatch(&self) -> bool {
        self.mismatched_features > 0
    }

    pub fn is_unobserved(&self) -> bool {
        self.compared_features == 0
    }

    /// True when every observed semantic feature matches the target.
    ///
    /// This is an observation-consistency check and permits partial observations.
    pub fn is_consistent(&self) -> bool {
        self.compared_features > 0 && self.mismatched_features == 0
    }

    /// A strict delivery gate requiring complete observation of all four semantic features.
    ///
    /// Missing observations therefore cannot silently become a promotion/evidence pass.
    pub fn passes(&self) -> bool {
        self.compared_features == 4 && self.mismatched_features == 0
    }
}

/// End-to-end semantic-delivery evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechDeliveryReceipt {
    pub version: String,
    pub plan_surface: String,
    pub target: SpeechDeliveryTarget,
    pub observation: SpeechDeliveryObservation,
    pub error: SpeechDeliveryError,
}

impl SpeechDeliveryReceipt {
    pub fn new(plan: &SpeechPlan, observation: SpeechDeliveryObservation) -> Self {
        let target = SpeechDeliveryTarget::from_plan(plan);
        let error = SpeechDeliveryError::compare(target.clone(), observation.clone());
        Self {
            version: SPEECH_DELIVERY_FEEDBACK_VERSION.to_string(),
            plan_surface: plan.grounding_surface(),
            target,
            observation,
            error,
        }
    }

    pub fn validate(&self) -> Result<(), SpeechDeliveryReceiptError> {
        if self.version != SPEECH_DELIVERY_FEEDBACK_VERSION {
            return Err(SpeechDeliveryReceiptError::InvalidVersion);
        }
        if self.plan_surface.trim().is_empty() {
            return Err(SpeechDeliveryReceiptError::EmptyPlanSurface);
        }
        if self.target.intent.trim().is_empty() {
            return Err(SpeechDeliveryReceiptError::EmptyIntent);
        }

        let expected = SpeechDeliveryError::compare(self.target.clone(), self.observation.clone());
        if expected != self.error {
            return Err(SpeechDeliveryReceiptError::ErrorMismatch);
        }
        Ok(())
    }

    pub fn validate_against_plan(
        &self,
        plan: &SpeechPlan,
    ) -> Result<(), SpeechDeliveryReceiptError> {
        self.validate()?;
        let expected_target = SpeechDeliveryTarget::from_plan(plan);
        if self.plan_surface != plan.grounding_surface() || self.target != expected_target {
            return Err(SpeechDeliveryReceiptError::PlanMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechDeliveryReceiptError {
    InvalidVersion,
    EmptyPlanSurface,
    EmptyIntent,
    ErrorMismatch,
    PlanMismatch,
}

impl std::fmt::Display for SpeechDeliveryReceiptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "speech delivery receipt version is unsupported"),
            Self::EmptyPlanSurface => write!(f, "speech delivery receipt requires a plan grounding surface"),
            Self::EmptyIntent => write!(f, "speech delivery receipt target intent must be non-empty"),
            Self::ErrorMismatch => write!(f, "speech delivery receipt error does not match target and observation"),
            Self::PlanMismatch => write!(f, "speech delivery receipt is not bound to the supplied speech plan"),
        }
    }
}

impl std::error::Error for SpeechDeliveryReceiptError {}

impl SpeechDeliveryTarget {
    pub fn from_plan(plan: &SpeechPlan) -> Self {
        Self {
            intent: plan.intent.clone(),
            clause_mode: plan.clause_mode,
            epistemic_delivery: plan.epistemic_delivery,
            focus_role: plan.focus_role.clone(),
        }
    }
}

fn compare_optional_str(target: &str, observation: Option<&str>) -> Option<bool> {
    observation.map(|value| value == target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SpeechPlan, StructuredDecoder, ThoughtChannels};
    use symthaea_core::genesis::GenesisSeed;

    fn plan() -> SpeechPlan {
        let genesis = GenesisSeed::from_phrase("broca-delivery-feedback-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(3);
        let readout = decoder.decode(&channels);
        SpeechPlan::from_readout(&channels, &readout)
    }

    #[test]
    fn receipt_lineage_matches_exact_plan() {
        let plan = plan();
        let receipt = SpeechDeliveryReceipt::new(&plan, SpeechDeliveryObservation::default());
        assert!(receipt.validate_against_plan(&plan).is_ok());
    }

    #[test]
    fn receipt_lineage_rejects_different_plan() {
        let plan = plan();
        let other = {
            let genesis = GenesisSeed::from_phrase("broca-delivery-feedback-other");
            let decoder = StructuredDecoder::new(&genesis);
            let channels = ThoughtChannels::with_intent(4);
            let readout = decoder.decode(&channels);
            SpeechPlan::from_readout(&channels, &readout)
        };
        let receipt = SpeechDeliveryReceipt::new(&plan, SpeechDeliveryObservation::default());
        assert_eq!(
            receipt.validate_against_plan(&other).expect_err("wrong upstream plan"),
            SpeechDeliveryReceiptError::PlanMismatch
        );
    }

    #[test]
    fn exact_observation_passes() {
        let plan = plan();
        let observation = SpeechDeliveryObservation {
            intent: Some(plan.intent.clone()),
            clause_mode: Some(plan.clause_mode),
            epistemic_delivery: Some(plan.epistemic_delivery),
            focus: Some(match &plan.focus_role {
                Some(role) => ObservedFocus::Role(role.clone()),
                None => ObservedFocus::None,
            }),
        };

        let receipt = SpeechDeliveryReceipt::new(&plan, observation);
        assert_eq!(receipt.error.compared_features, 4);
        assert_eq!(receipt.error.mismatched_features, 0);
        assert!(receipt.error.passes());
    }

    #[test]
    fn semantic_mismatch_is_not_hidden_by_acoustic_success() {
        let plan = plan();
        let observation = SpeechDeliveryObservation {
            intent: Some("explain".to_string()),
            clause_mode: Some(ClauseMode::Question),
            ..Default::default()
        };

        let error = SpeechDeliveryError::compare(
            SpeechDeliveryTarget::from_plan(&plan),
            observation,
        );

        assert_eq!(error.compared_features, 2);
        assert!(error.clause_mode_match == Some(false));
        assert!(error.has_mismatch());
        assert!(!error.passes());
    }

    #[test]
    fn observed_absence_of_focus_is_distinct_from_missing_evidence() {
        let mut plan = plan();
        plan.focus_role = None;

        let target = SpeechDeliveryTarget::from_plan(&plan);
        let explicit_none = SpeechDeliveryError::compare(
            target.clone(),
            SpeechDeliveryObservation {
                focus: Some(ObservedFocus::None),
                ..Default::default()
            },
        );
        let missing = SpeechDeliveryError::compare(
            target,
            SpeechDeliveryObservation::default(),
        );

        assert_eq!(explicit_none.focus_match, Some(true));
        assert!(missing.focus_match.is_none());
        assert!(explicit_none.is_consistent());
        assert!(!explicit_none.passes());
        assert!(!missing.is_consistent());
        assert!(!missing.passes());
    }

    #[test]
    fn partial_matching_cannot_pass_the_complete_delivery_gate() {
        let plan = plan();
        let error = SpeechDeliveryError::compare(
            SpeechDeliveryTarget::from_plan(&plan),
            SpeechDeliveryObservation {
                intent: Some(plan.intent.clone()),
                ..Default::default()
            },
        );

        assert!(error.is_consistent());
        assert!(!error.passes());
        assert_eq!(error.compared_features, 1);
    }

    #[test]
    fn qualified_delivery_cannot_be_observed_as_assertive_without_failure() {
        let mut plan = plan();
        plan.epistemic_delivery = EpistemicDelivery::Qualified;

        let error = SpeechDeliveryError::compare(
            SpeechDeliveryTarget::from_plan(&plan),
            SpeechDeliveryObservation {
                epistemic_delivery: Some(EpistemicDelivery::Assertive),
                ..Default::default()
            },
        );

        assert_eq!(error.epistemic_delivery_match, Some(false));
        assert!(error.has_mismatch());
        assert!(!error.passes());
    }
}
