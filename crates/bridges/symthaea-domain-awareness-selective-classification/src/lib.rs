// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit adapter from selective classification evidence into domain-awareness
//! identity hypotheses. Abstention produces no identity hypothesis.

#![deny(unsafe_code)]

use symthaea_domain_awareness::{
    EpistemicState, IdentityHypothesis, epistemic_state_for_hypotheses,
};
use symthaea_selective_classification::{
    ClassificationAssessment, ClassificationDisposition, SelectiveClassificationEvidence,
};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub struct IdentityEvidenceBridgeResult {
    pub hypotheses: Vec<IdentityHypothesis>,
    pub epistemic_hint: EpistemicState,
    pub source_evidence_id: String,
}

impl IdentityEvidenceBridgeResult {
    pub fn evidence_usable(&self) -> bool {
        !self.hypotheses.is_empty()
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Convert only an `EvidenceUsable` prediction set into identity hypotheses.
///
/// An abstaining or incomplete classifier emits no identity hypothesis. OOD
/// abstention is preserved as an `OutOfDistribution` epistemic hint. The adapter
/// never converts classification into intent, risk, or physical authority.
pub fn bridge_identity_evidence(
    evidence: &SelectiveClassificationEvidence,
    assessment: &ClassificationAssessment,
    source_observation_id: Uuid,
) -> IdentityEvidenceBridgeResult {
    if assessment.disposition != ClassificationDisposition::EvidenceUsable {
        return IdentityEvidenceBridgeResult {
            hypotheses: Vec::new(),
            epistemic_hint: if assessment.out_of_distribution() {
                EpistemicState::OutOfDistribution
            } else {
                EpistemicState::InsufficientEvidence
            },
            source_evidence_id: evidence.evidence_id.clone(),
        };
    }

    let mut hypotheses = Vec::with_capacity(evidence.prediction_set.len());
    for label in &evidence.prediction_set {
        let Some(support) = evidence.support_for(label) else {
            // A usable assessment should only come from structurally valid evidence.
            // Preserve fail-closed behavior if an inconsistent object is supplied.
            return IdentityEvidenceBridgeResult {
                hypotheses: Vec::new(),
                epistemic_hint: EpistemicState::InsufficientEvidence,
                source_evidence_id: evidence.evidence_id.clone(),
            };
        };

        let mut refs = evidence.evidence_refs.clone();
        refs.push(format!("classification:{}", evidence.evidence_id));
        refs.push(format!("model:{}", evidence.model_digest));
        refs.push(format!("calibration:{}", evidence.calibration_ref));
        refs.sort();
        refs.dedup();

        hypotheses.push(IdentityHypothesis {
            label: label.clone(),
            confidence: support,
            evidence_ids: vec![source_observation_id],
            evidence_refs: refs,
        });
    }

    let epistemic_hint = epistemic_state_for_hypotheses(&hypotheses);
    IdentityEvidenceBridgeResult {
        hypotheses,
        epistemic_hint,
        source_evidence_id: evidence.evidence_id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_selective_classification::{
        ClassificationIssue, LabelSupport, SelectiveClassificationEvidence,
    };

    fn evidence(labels: &[(&str, f64)]) -> SelectiveClassificationEvidence {
        SelectiveClassificationEvidence {
            evidence_id: "classification:1".into(),
            observed_at_ms: 1_000,
            maximum_valid_age_ms: 500,
            model_id: "classifier".into(),
            model_digest: "sha256:model-v1".into(),
            calibration_ref: "release-1".into(),
            deployment_domain_id: "domain-a".into(),
            label_support: labels
                .iter()
                .map(|(label, support)| LabelSupport {
                    label: (*label).to_string(),
                    support: *support,
                })
                .collect(),
            prediction_set: labels.iter().map(|(label, _)| (*label).to_string()).collect(),
            epistemic_uncertainty: 0.1,
            out_of_distribution_score: 0.1,
            evidence_refs: vec!["frame:1".into()],
        }
    }

    #[test]
    fn ambiguous_set_stays_competing_identity_evidence() {
        let evidence = evidence(&[("small-aircraft", 0.74), ("bird", 0.66)]);
        let result = bridge_identity_evidence(
            &evidence,
            &ClassificationAssessment {
                disposition: ClassificationDisposition::EvidenceUsable,
                issues: vec![],
            },
            Uuid::new_v4(),
        );
        assert_eq!(result.hypotheses.len(), 2);
        assert_eq!(result.epistemic_hint, EpistemicState::ConflictingEvidence);
        assert!(!result.grants_physical_authority());
    }

    #[test]
    fn ood_abstention_produces_no_identity_claim() {
        let evidence = evidence(&[("small-aircraft", 0.95)]);
        let result = bridge_identity_evidence(
            &evidence,
            &ClassificationAssessment {
                disposition: ClassificationDisposition::Abstain,
                issues: vec![ClassificationIssue::OutOfDistribution],
            },
            Uuid::new_v4(),
        );
        assert!(result.hypotheses.is_empty());
        assert_eq!(result.epistemic_hint, EpistemicState::OutOfDistribution);
    }

    #[test]
    fn singleton_high_support_can_be_known_but_still_has_no_authority() {
        let evidence = evidence(&[("small-aircraft", 0.90)]);
        let result = bridge_identity_evidence(
            &evidence,
            &ClassificationAssessment {
                disposition: ClassificationDisposition::EvidenceUsable,
                issues: vec![],
            },
            Uuid::new_v4(),
        );
        assert_eq!(result.epistemic_hint, EpistemicState::Known);
        assert!(!result.grants_physical_authority());
    }

    #[test]
    fn incomplete_classification_produces_no_identity_claim() {
        let evidence = evidence(&[("small-aircraft", 0.90)]);
        let result = bridge_identity_evidence(
            &evidence,
            &ClassificationAssessment {
                disposition: ClassificationDisposition::Incomplete,
                issues: vec![ClassificationIssue::StaleEvidence],
            },
            Uuid::new_v4(),
        );
        assert!(result.hypotheses.is_empty());
        assert_eq!(result.epistemic_hint, EpistemicState::InsufficientEvidence);
    }
}
