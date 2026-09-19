// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public identity gate for qualification-eligibility manifests.
//!
//! The lower-level manifest evaluator composes content-addressed receipts. This
//! facade additionally binds those receipts back to the exact polarity-aware
//! adjudicated claim, preventing a valid semantic-binding/coverage stack for one
//! claim from being relabeled as eligibility evidence for another claim.

use crate::{
    AdjudicatedScientificClaim, ClaimRelationBindingReport, EvidenceCoverageAuthorityReport,
    EvidenceDecisionBindingReport, FrozenQualificationEligibilityProfile,
    QualificationEligibilityInputs, QualificationEligibilityManifest, ScientificClaim,
    qualification_eligibility::evaluate_qualification_eligibility,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationEligibilityBindingIssue {
    AdjudicatedClaimIdentityMismatch,
    RelationBindingAdjudicationMismatch,
}

pub fn evaluate_bound_qualification_eligibility(
    claim: &ScientificClaim,
    adjudicated: &AdjudicatedScientificClaim,
    relation_binding: &ClaimRelationBindingReport,
    evidence_coverage: &EvidenceCoverageAuthorityReport,
    evidence_decision_binding: &EvidenceDecisionBindingReport,
    frozen_profile: &FrozenQualificationEligibilityProfile,
    inputs: QualificationEligibilityInputs<'_>,
) -> Result<QualificationEligibilityManifest, Vec<QualificationEligibilityBindingIssue>> {
    let mut issues = Vec::new();
    if adjudicated.claim_id() != claim.claim_id()
        || adjudicated.subject_sha256() != claim.subject_sha256()
    {
        issues.push(QualificationEligibilityBindingIssue::AdjudicatedClaimIdentityMismatch);
    }
    if relation_binding.adjudication_sha256() != adjudicated.adjudication_sha256() {
        issues.push(QualificationEligibilityBindingIssue::RelationBindingAdjudicationMismatch);
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    Ok(evaluate_qualification_eligibility(
        claim,
        relation_binding,
        evidence_coverage,
        evidence_decision_binding,
        frozen_profile,
        inputs,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_issue_types_are_not_qualification_states() {
        let issue = QualificationEligibilityBindingIssue::AdjudicatedClaimIdentityMismatch;
        assert!(matches!(
            issue,
            QualificationEligibilityBindingIssue::AdjudicatedClaimIdentityMismatch
        ));
    }
}
