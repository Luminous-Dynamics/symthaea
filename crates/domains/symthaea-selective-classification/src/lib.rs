// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Selective classification evidence with explicit abstention, calibration
//! provenance, immutable model identity, and out-of-distribution handling.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct LabelSupport {
    pub label: String,
    /// Calibrated support/confidence in `[0, 1]`. This is evidence, not authority.
    pub support: f64,
}

impl LabelSupport {
    pub fn validate(&self) -> bool {
        !self.label.trim().is_empty()
            && self.support.is_finite()
            && (0.0..=1.0).contains(&self.support)
    }
}

/// Output from one qualified classifier invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectiveClassificationEvidence {
    pub evidence_id: String,
    pub observed_at_ms: u64,
    pub maximum_valid_age_ms: u64,
    pub model_id: String,
    /// Immutable content/configuration digest of the deployed model artifact.
    pub model_digest: String,
    /// Exact calibration evidence/release used for this model.
    pub calibration_ref: String,
    /// Deployment domain against which calibration/validation was qualified.
    pub deployment_domain_id: String,
    pub label_support: Vec<LabelSupport>,
    /// Set-valued prediction. Multiple labels are allowed and preserve ambiguity.
    pub prediction_set: Vec<String>,
    /// Epistemic uncertainty estimate in `[0, 1]`.
    pub epistemic_uncertainty: f64,
    /// OOD score in `[0, 1]`, where larger means farther from the qualified domain.
    pub out_of_distribution_score: f64,
    pub evidence_refs: Vec<String>,
}

impl SelectiveClassificationEvidence {
    pub fn validate(&self) -> bool {
        if self.evidence_id.trim().is_empty()
            || self.maximum_valid_age_ms == 0
            || self.model_id.trim().is_empty()
            || self.model_digest.trim().is_empty()
            || self.calibration_ref.trim().is_empty()
            || self.deployment_domain_id.trim().is_empty()
            || !self.epistemic_uncertainty.is_finite()
            || !(0.0..=1.0).contains(&self.epistemic_uncertainty)
            || !self.out_of_distribution_score.is_finite()
            || !(0.0..=1.0).contains(&self.out_of_distribution_score)
            || self.label_support.iter().any(|label| !label.validate())
            || self.evidence_refs.is_empty()
            || self.evidence_refs.iter().any(|value| value.trim().is_empty())
        {
            return false;
        }

        let support_labels = self
            .label_support
            .iter()
            .map(|entry| entry.label.as_str())
            .collect::<BTreeSet<_>>();
        if support_labels.len() != self.label_support.len() {
            return false;
        }

        let prediction_labels = self
            .prediction_set
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        prediction_labels.len() == self.prediction_set.len()
            && prediction_labels
                .iter()
                .all(|label| support_labels.contains(label))
    }

    pub fn is_fresh_at(&self, now_ms: u64) -> bool {
        self.validate()
            && now_ms >= self.observed_at_ms
            && now_ms.saturating_sub(self.observed_at_ms) <= self.maximum_valid_age_ms
    }

    pub fn support_for(&self, label: &str) -> Option<f64> {
        self.label_support
            .iter()
            .find(|entry| entry.label == label)
            .map(|entry| entry.support)
    }
}

/// Deployment-reviewed acceptance/abstention policy. There are intentionally no
/// safety-critical defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectiveClassificationPolicy {
    pub expected_model_digest: String,
    pub expected_calibration_ref: String,
    pub expected_deployment_domain_id: String,
    pub maximum_epistemic_uncertainty: f64,
    pub maximum_out_of_distribution_score: f64,
    pub minimum_prediction_support: f64,
    pub maximum_prediction_set_size: usize,
}

impl SelectiveClassificationPolicy {
    pub fn validate(&self) -> bool {
        !self.expected_model_digest.trim().is_empty()
            && !self.expected_calibration_ref.trim().is_empty()
            && !self.expected_deployment_domain_id.trim().is_empty()
            && self.maximum_epistemic_uncertainty.is_finite()
            && (0.0..=1.0).contains(&self.maximum_epistemic_uncertainty)
            && self.maximum_out_of_distribution_score.is_finite()
            && (0.0..=1.0).contains(&self.maximum_out_of_distribution_score)
            && self.minimum_prediction_support.is_finite()
            && (0.0..=1.0).contains(&self.minimum_prediction_support)
            && self.maximum_prediction_set_size > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationDisposition {
    /// Evidence is qualified for downstream epistemic reasoning. It may still be ambiguous.
    EvidenceUsable,
    /// Evidence is valid but the classifier should explicitly decline a classification claim.
    Abstain,
    /// The evidence case itself is invalid/stale/mismatched and cannot be used.
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassificationIssue {
    InvalidPolicy,
    InvalidEvidence,
    StaleEvidence,
    ModelDigestMismatch,
    CalibrationMismatch,
    DeploymentDomainMismatch,
    PredictionSetEmpty,
    PredictionSetTooLarge,
    PredictionSupportBelowMinimum(String),
    EpistemicUncertaintyTooHigh,
    OutOfDistribution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassificationAssessment {
    pub disposition: ClassificationDisposition,
    pub issues: Vec<ClassificationIssue>,
}

impl ClassificationAssessment {
    pub fn out_of_distribution(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| matches!(issue, ClassificationIssue::OutOfDistribution))
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn assess_classification(
    evidence: &SelectiveClassificationEvidence,
    policy: &SelectiveClassificationPolicy,
    now_ms: u64,
) -> ClassificationAssessment {
    if !policy.validate() {
        return ClassificationAssessment {
            disposition: ClassificationDisposition::Incomplete,
            issues: vec![ClassificationIssue::InvalidPolicy],
        };
    }
    if !evidence.validate() {
        return ClassificationAssessment {
            disposition: ClassificationDisposition::Incomplete,
            issues: vec![ClassificationIssue::InvalidEvidence],
        };
    }

    let mut incomplete = Vec::new();
    if !evidence.is_fresh_at(now_ms) {
        incomplete.push(ClassificationIssue::StaleEvidence);
    }
    if evidence.model_digest != policy.expected_model_digest {
        incomplete.push(ClassificationIssue::ModelDigestMismatch);
    }
    if evidence.calibration_ref != policy.expected_calibration_ref {
        incomplete.push(ClassificationIssue::CalibrationMismatch);
    }
    if evidence.deployment_domain_id != policy.expected_deployment_domain_id {
        incomplete.push(ClassificationIssue::DeploymentDomainMismatch);
    }
    if !incomplete.is_empty() {
        return ClassificationAssessment {
            disposition: ClassificationDisposition::Incomplete,
            issues: incomplete,
        };
    }

    let mut abstain = Vec::new();
    if evidence.prediction_set.is_empty() {
        abstain.push(ClassificationIssue::PredictionSetEmpty);
    }
    if evidence.prediction_set.len() > policy.maximum_prediction_set_size {
        abstain.push(ClassificationIssue::PredictionSetTooLarge);
    }
    for label in &evidence.prediction_set {
        if evidence
            .support_for(label)
            .is_none_or(|support| support < policy.minimum_prediction_support)
        {
            abstain.push(ClassificationIssue::PredictionSupportBelowMinimum(
                label.clone(),
            ));
        }
    }
    if evidence.epistemic_uncertainty > policy.maximum_epistemic_uncertainty {
        abstain.push(ClassificationIssue::EpistemicUncertaintyTooHigh);
    }
    if evidence.out_of_distribution_score > policy.maximum_out_of_distribution_score {
        abstain.push(ClassificationIssue::OutOfDistribution);
    }

    ClassificationAssessment {
        disposition: if abstain.is_empty() {
            ClassificationDisposition::EvidenceUsable
        } else {
            ClassificationDisposition::Abstain
        },
        issues: abstain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence() -> SelectiveClassificationEvidence {
        SelectiveClassificationEvidence {
            evidence_id: "classification:1".into(),
            observed_at_ms: 1_000,
            maximum_valid_age_ms: 500,
            model_id: "visual-air-object-classifier".into(),
            model_digest: "sha256:model-v1".into(),
            calibration_ref: "calibration:release-1".into(),
            deployment_domain_id: "maritime-daylight-v1".into(),
            label_support: vec![
                LabelSupport {
                    label: "small-aircraft".into(),
                    support: 0.74,
                },
                LabelSupport {
                    label: "bird".into(),
                    support: 0.66,
                },
            ],
            prediction_set: vec!["small-aircraft".into(), "bird".into()],
            epistemic_uncertainty: 0.12,
            out_of_distribution_score: 0.10,
            evidence_refs: vec!["frame:1".into(), "model-receipt:1".into()],
        }
    }

    fn policy() -> SelectiveClassificationPolicy {
        SelectiveClassificationPolicy {
            expected_model_digest: "sha256:model-v1".into(),
            expected_calibration_ref: "calibration:release-1".into(),
            expected_deployment_domain_id: "maritime-daylight-v1".into(),
            maximum_epistemic_uncertainty: 0.20,
            maximum_out_of_distribution_score: 0.25,
            minimum_prediction_support: 0.60,
            maximum_prediction_set_size: 3,
        }
    }

    #[test]
    fn ambiguous_prediction_set_can_be_usable_without_forcing_a_winner() {
        let result = assess_classification(&evidence(), &policy(), 1_100);
        assert_eq!(result.disposition, ClassificationDisposition::EvidenceUsable);
        assert!(result.issues.is_empty());
        assert!(!result.grants_physical_authority());
    }

    #[test]
    fn out_of_distribution_requires_abstention() {
        let mut sample = evidence();
        sample.out_of_distribution_score = 0.9;
        let result = assess_classification(&sample, &policy(), 1_100);
        assert_eq!(result.disposition, ClassificationDisposition::Abstain);
        assert!(result.out_of_distribution());
    }

    #[test]
    fn high_epistemic_uncertainty_requires_abstention() {
        let mut sample = evidence();
        sample.epistemic_uncertainty = 0.8;
        let result = assess_classification(&sample, &policy(), 1_100);
        assert_eq!(result.disposition, ClassificationDisposition::Abstain);
        assert!(result
            .issues
            .contains(&ClassificationIssue::EpistemicUncertaintyTooHigh));
    }

    #[test]
    fn stale_evidence_is_incomplete_not_a_classification() {
        let result = assess_classification(&evidence(), &policy(), 2_000);
        assert_eq!(result.disposition, ClassificationDisposition::Incomplete);
        assert!(result.issues.contains(&ClassificationIssue::StaleEvidence));
    }

    #[test]
    fn model_or_calibration_drift_fails_closed() {
        let mut sample = evidence();
        sample.model_digest = "sha256:unreviewed".into();
        sample.calibration_ref = "calibration:other".into();
        let result = assess_classification(&sample, &policy(), 1_100);
        assert_eq!(result.disposition, ClassificationDisposition::Incomplete);
        assert!(result
            .issues
            .contains(&ClassificationIssue::ModelDigestMismatch));
        assert!(result
            .issues
            .contains(&ClassificationIssue::CalibrationMismatch));
    }

    #[test]
    fn weak_prediction_set_member_requires_abstention() {
        let mut sample = evidence();
        sample.label_support[1].support = 0.2;
        let result = assess_classification(&sample, &policy(), 1_100);
        assert_eq!(result.disposition, ClassificationDisposition::Abstain);
        assert!(result.issues.iter().any(|issue| matches!(
            issue,
            ClassificationIssue::PredictionSupportBelowMinimum(label) if label == "bird"
        )));
    }
}
