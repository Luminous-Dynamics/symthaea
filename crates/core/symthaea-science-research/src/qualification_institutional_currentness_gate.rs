// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Public anti-circularity and temporal-scope gate for institutional currentness.
//!
//! The internal capability records the authenticated decision. The public wrapper
//! deliberately names its temporal scope: it establishes institutional
//! currentness only at the exact authenticated evaluation interval that produced
//! the underlying readiness receipt. Retaining the artifact does not establish
//! that the claim remains current at a later time.

use serde::Serialize;
use symthaea_trust_core::{
    AuthorizedTrustRoleAttestation, RootRoleQuorumProof, Sha256Digest as TrustSha256Digest,
    TrustedTime,
};

use crate::{
    QualificationCurrentnessError, QualificationCurrentnessStatement,
    QualificationValidityReadinessAssessment, QualifiedScientificClaim, ResearchId,
    Sha256Digest,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstitutionallyCurrentAtEvaluationQualifiedScientificClaim {
    inner: crate::qualification_institutional_currentness::InstitutionallyCurrentQualifiedScientificClaim,
}

impl InstitutionallyCurrentAtEvaluationQualifiedScientificClaim {
    pub fn qualification_sha256(&self) -> &Sha256Digest { self.inner.qualification_sha256() }
    pub fn claim_id(&self) -> &ResearchId { self.inner.claim_id() }
    pub fn subject_sha256(&self) -> &Sha256Digest { self.inner.subject_sha256() }
    pub fn readiness_assessment_sha256(&self) -> &TrustSha256Digest {
        self.inner.readiness_assessment_sha256()
    }
    pub fn currentness_statement_sha256(&self) -> &TrustSha256Digest {
        self.inner.currentness_statement_sha256()
    }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest {
        self.inner.lifecycle_event_sha256()
    }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest {
        self.inner.root_authority_sha256()
    }
    pub fn trust_snapshot_authority_sha256(&self) -> &TrustSha256Digest {
        self.inner.trust_snapshot_authority_sha256()
    }
    pub fn lifecycle_role_authority_sha256(&self) -> &TrustSha256Digest {
        self.inner.lifecycle_role_authority_sha256()
    }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        self.inner.evaluation_time_authority_sha256()
    }
    pub fn evaluation_interval(&self) -> (u64, u64) { self.inner.evaluation_interval() }
    pub fn currentness_sha256(&self) -> &TrustSha256Digest { self.inner.currentness_sha256() }

    pub const fn institutional_currentness_at_evaluation_established(&self) -> bool { true }
    pub const fn currentness_at_later_time_established(&self) -> bool { false }
    pub const fn globally_latest_head_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn global_evidence_exhaustiveness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn authenticate_institutional_currentness_at_evaluation(
    qualified: &QualifiedScientificClaim,
    readiness: &QualificationValidityReadinessAssessment,
    evaluation_time: &TrustedTime,
    statement: &QualificationCurrentnessStatement,
    lifecycle_authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> Result<InstitutionallyCurrentAtEvaluationQualifiedScientificClaim, QualificationCurrentnessError> {
    if evaluation_time.bindings().iter().any(|binding| {
        binding.source_artifact_sha256() == readiness.assessment_sha256()
            || binding.source_artifact_sha256() == statement.statement_sha256()
            || binding.source_artifact_sha256() == lifecycle_authority.authority_sha256()
            || binding.source_artifact_sha256() == role_proof.proof_sha256()
    }) {
        return Err(QualificationCurrentnessError::TimeDependsOnCurrentnessDecision);
    }

    let inner = crate::qualification_institutional_currentness::authenticate_institutional_currentness(
        qualified,
        readiness,
        evaluation_time,
        statement,
        lifecycle_authority,
        role_proof,
    )?;
    Ok(InstitutionallyCurrentAtEvaluationQualifiedScientificClaim { inner })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_currentness_name_preserves_temporal_scope() {
        fn _assert_api(value: &InstitutionallyCurrentAtEvaluationQualifiedScientificClaim) {
            if value.institutional_currentness_at_evaluation_established() {
                assert!(!value.currentness_at_later_time_established());
                assert!(!value.globally_latest_head_established());
                assert!(!value.scientific_truth_established());
            }
        }
    }
}
