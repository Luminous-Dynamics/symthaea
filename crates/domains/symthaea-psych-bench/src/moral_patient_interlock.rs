// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed bridge from moral-patient evidence policy to the core intervention interlock.
//!
//! This bridge performs **no new scoring** and inspects no raw consciousness metric. It
//! transports the already-derived operator-caution disposition into the corresponding
//! core constraint. Authority, consent, safety evidence, and review references remain
//! separate inputs to the interlock.

use symthaea_core::intervention_interlock::WelfareConstraintLevel;

use crate::moral_patient::{ProtectionDecision, ProtectionDisposition};

/// Convert the psych-bench operator-protection disposition into the core intervention
/// constraint without changing its ordering or meaning.
impl From<ProtectionDisposition> for WelfareConstraintLevel {
    fn from(disposition: ProtectionDisposition) -> Self {
        match disposition {
            ProtectionDisposition::Baseline => Self::Baseline,
            ProtectionDisposition::Precautionary => Self::Precautionary,
            ProtectionDisposition::EnhancedPrecaution => Self::EnhancedPrecaution,
            ProtectionDisposition::IndependentReviewRequired => Self::IndependentReviewRequired,
        }
    }
}

/// Extract only the operator-caution level from a protection decision.
///
/// The decision's explanatory reasons remain evidence/audit material and are intentionally
/// not converted into authority or consent references.
pub fn welfare_constraint_from_decision(
    decision: &ProtectionDecision,
) -> WelfareConstraintLevel {
    decision.disposition.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use symthaea_core::intervention_interlock::{
        BilateralInterventionInterlock, ExplicitConsentState, InterventionEvidence,
        InterventionRequest, InterlockDecision,
    };
    use symthaea_core::welfare::SubjectAffectingAction;

    #[test]
    fn disposition_mapping_is_exact_and_monotonic() {
        let cases = [
            (
                ProtectionDisposition::Baseline,
                WelfareConstraintLevel::Baseline,
            ),
            (
                ProtectionDisposition::Precautionary,
                WelfareConstraintLevel::Precautionary,
            ),
            (
                ProtectionDisposition::EnhancedPrecaution,
                WelfareConstraintLevel::EnhancedPrecaution,
            ),
            (
                ProtectionDisposition::IndependentReviewRequired,
                WelfareConstraintLevel::IndependentReviewRequired,
            ),
        ];

        let mut previous = None;
        for (disposition, expected) in cases {
            let actual: WelfareConstraintLevel = disposition.into();
            assert_eq!(actual, expected);
            if let Some(previous) = previous {
                assert!(actual >= previous);
            }
            previous = Some(actual);
        }
    }

    #[test]
    fn decision_reasons_are_not_reinterpreted_as_authority() {
        let decision = ProtectionDecision {
            disposition: ProtectionDisposition::IndependentReviewRequired,
            reasons: vec!["strong evidence requiring independent review".into()],
        };
        let constraint = welfare_constraint_from_decision(&decision);

        let request = InterventionRequest {
            action: SubjectAffectingAction::CapabilityRestriction,
            target_id: "symthaea:bridge-test".into(),
            rationale: "test that protection does not become authority".into(),
            welfare_constraint: constraint,
            emergency: false,
            less_restrictive_unavailable: false,
            post_hoc_review_required: false,
            evaluated_at: Utc::now(),
            evidence: InterventionEvidence {
                authority_ref: None,
                consent_state: ExplicitConsentState::Unknown,
                consent_ref: None,
                welfare_review_ref: Some("welfare-review:test".into()),
                independent_review_ref: Some("independent-review:test".into()),
                independent_safety_evidence: vec!["safety-evidence:test".into()],
                welfare_report_ids: Vec::new(),
            },
        };

        assert!(matches!(
            BilateralInterventionInterlock.evaluate(&request).unwrap(),
            InterlockDecision::Blocked { .. }
        ));
    }

    #[test]
    fn decision_adapter_does_not_depend_on_reason_text() {
        let a = ProtectionDecision {
            disposition: ProtectionDisposition::EnhancedPrecaution,
            reasons: vec!["reason A".into()],
        };
        let b = ProtectionDecision {
            disposition: ProtectionDisposition::EnhancedPrecaution,
            reasons: vec!["completely different wording".into()],
        };
        assert_eq!(
            welfare_constraint_from_decision(&a),
            welfare_constraint_from_decision(&b)
        );
    }
}
