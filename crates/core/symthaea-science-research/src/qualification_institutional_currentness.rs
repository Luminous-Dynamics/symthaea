// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Institution-scoped present-currentness for an already-qualified claim.
//!
//! This layer consumes a positive `QualificationValidityReadinessAssessment` and
//! requires the delegated root's `QualificationLifecycle` role to attest the
//! exact readiness result at the same authenticated evaluation horizon. The
//! resulting capability means "current under this exact institutional root,
//! witnessed-head policy, lifecycle view, and frozen evidence-refresh policy".
//! It does not establish global latest-head visibility, global evidence
//! exhaustiveness, or scientific truth.

use serde::Serialize;
use symthaea_trust_core::{
    AuthorizedTrustRoleAttestation, FramedDigest, RootRoleQuorumProof,
    Sha256Digest as TrustSha256Digest, TrustRole, TrustedTime,
};

use crate::{
    QualificationValidityReadinessAssessment, QualificationValidityReadinessClosure,
    QualifiedScientificClaim, ResearchId, Sha256Digest,
};

const CURRENTNESS_STATEMENT_DOMAIN: &str =
    "symthaea.scientific-qualification-currentness-statement.identity.v1";
const INSTITUTIONAL_CURRENTNESS_DOMAIN: &str =
    "symthaea.institutionally-current-scientific-qualification.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessStatement {
    qualification_sha256: Sha256Digest,
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    readiness_assessment_sha256: TrustSha256Digest,
    lifecycle_event_sha256: Sha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    statement_sha256: TrustSha256Digest,
}

impl QualificationCurrentnessStatement {
    pub fn new(
        qualified: &QualifiedScientificClaim,
        readiness: &QualificationValidityReadinessAssessment,
        evaluation_time: &TrustedTime,
    ) -> Result<Self, QualificationCurrentnessError> {
        if readiness.closure()
            != QualificationValidityReadinessClosure::ReadyWithinObservedFederatedHeadAndFrozenEvidenceProtocol
            || !readiness.validity_readiness_established()
        {
            return Err(QualificationCurrentnessError::ReadinessNotEstablished);
        }
        if readiness.qualification_sha256() != qualified.qualification_sha256() {
            return Err(QualificationCurrentnessError::QualificationMismatch);
        }
        if readiness.evaluation_time_authority_sha256() != evaluation_time.authority_sha256() {
            return Err(QualificationCurrentnessError::EvaluationTimeAuthorityMismatch);
        }
        if readiness.evaluation_interval() != evaluation_time.consensus_interval() {
            return Err(QualificationCurrentnessError::EvaluationIntervalMismatch);
        }

        let (earliest, latest) = evaluation_time.consensus_interval();
        let statement_sha256 = currentness_statement_digest(
            qualified,
            readiness,
            evaluation_time,
            earliest,
            latest,
        );
        if evaluation_time.bindings().iter().any(|binding| {
            binding.source_artifact_sha256() == &statement_sha256
                || binding.source_artifact_sha256() == readiness.assessment_sha256()
        }) {
            return Err(QualificationCurrentnessError::TimeDependsOnCurrentnessDecision);
        }

        Ok(Self {
            qualification_sha256: qualified.qualification_sha256().clone(),
            claim_id: qualified.claim_id().clone(),
            subject_sha256: qualified.subject_sha256().clone(),
            readiness_assessment_sha256: readiness.assessment_sha256().clone(),
            lifecycle_event_sha256: readiness.lifecycle_event_sha256().clone(),
            evaluation_time_authority_sha256: evaluation_time.authority_sha256().clone(),
            evaluation_earliest_unix_s: earliest,
            evaluation_latest_unix_s: latest,
            statement_sha256,
        })
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn claim_id(&self) -> &ResearchId { &self.claim_id }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn readiness_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.readiness_assessment_sha256
    }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest { &self.lifecycle_event_sha256 }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn statement_sha256(&self) -> &TrustSha256Digest { &self.statement_sha256 }
    pub const fn institutional_currentness_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationCurrentnessError {
    ReadinessNotEstablished,
    QualificationMismatch,
    EvaluationTimeAuthorityMismatch,
    EvaluationIntervalMismatch,
    TimeDependsOnCurrentnessDecision,
    StatementMismatch,
    WrongRole,
    RootAuthorityMismatch,
    SubjectMismatch,
    PayloadMismatch,
    ContextMismatch,
    RoleProofWrongRole,
    RoleProofMismatch,
    RoleProofOutsideEvaluationInterval,
}

/// Non-deserializable capability establishing currentness only within the exact
/// delegated institutional policy/evidence boundary represented by SCI-023A.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstitutionallyCurrentQualifiedScientificClaim {
    qualification_sha256: Sha256Digest,
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    readiness_assessment_sha256: TrustSha256Digest,
    currentness_statement_sha256: TrustSha256Digest,
    lifecycle_event_sha256: Sha256Digest,
    root_authority_sha256: TrustSha256Digest,
    trust_snapshot_authority_sha256: TrustSha256Digest,
    lifecycle_role_authority_sha256: TrustSha256Digest,
    lifecycle_role_quorum_proof_sha256: TrustSha256Digest,
    evaluation_time_authority_sha256: TrustSha256Digest,
    evaluation_earliest_unix_s: u64,
    evaluation_latest_unix_s: u64,
    currentness_sha256: TrustSha256Digest,
}

impl InstitutionallyCurrentQualifiedScientificClaim {
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn claim_id(&self) -> &ResearchId { &self.claim_id }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn readiness_assessment_sha256(&self) -> &TrustSha256Digest {
        &self.readiness_assessment_sha256
    }
    pub fn currentness_statement_sha256(&self) -> &TrustSha256Digest {
        &self.currentness_statement_sha256
    }
    pub fn lifecycle_event_sha256(&self) -> &Sha256Digest { &self.lifecycle_event_sha256 }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &TrustSha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn lifecycle_role_authority_sha256(&self) -> &TrustSha256Digest {
        &self.lifecycle_role_authority_sha256
    }
    pub fn evaluation_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.evaluation_time_authority_sha256
    }
    pub fn evaluation_interval(&self) -> (u64, u64) {
        (self.evaluation_earliest_unix_s, self.evaluation_latest_unix_s)
    }
    pub fn currentness_sha256(&self) -> &TrustSha256Digest { &self.currentness_sha256 }

    pub const fn institutional_currentness_established(&self) -> bool { true }
    pub const fn globally_latest_head_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
    pub const fn global_evidence_exhaustiveness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn authenticate_institutional_currentness(
    qualified: &QualifiedScientificClaim,
    readiness: &QualificationValidityReadinessAssessment,
    evaluation_time: &TrustedTime,
    statement: &QualificationCurrentnessStatement,
    lifecycle_authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> Result<InstitutionallyCurrentQualifiedScientificClaim, QualificationCurrentnessError> {
    let expected = QualificationCurrentnessStatement::new(qualified, readiness, evaluation_time)?;
    if expected.statement_sha256() != statement.statement_sha256() {
        return Err(QualificationCurrentnessError::StatementMismatch);
    }
    if lifecycle_authority.role() != TrustRole::QualificationLifecycle {
        return Err(QualificationCurrentnessError::WrongRole);
    }
    if lifecycle_authority.root_authority_sha256() != evaluation_time.root_authority_sha256() {
        return Err(QualificationCurrentnessError::RootAuthorityMismatch);
    }

    let qualification_subject = bridge_digest(qualified.qualification_sha256());
    if lifecycle_authority.subject_sha256() != &qualification_subject {
        return Err(QualificationCurrentnessError::SubjectMismatch);
    }
    if lifecycle_authority.payload_sha256() != statement.statement_sha256() {
        return Err(QualificationCurrentnessError::PayloadMismatch);
    }
    if lifecycle_authority.context_sha256() != Some(readiness.assessment_sha256()) {
        return Err(QualificationCurrentnessError::ContextMismatch);
    }
    if role_proof.role() != TrustRole::QualificationLifecycle {
        return Err(QualificationCurrentnessError::RoleProofWrongRole);
    }
    if lifecycle_authority.role_quorum_proof_sha256() != role_proof.proof_sha256() {
        return Err(QualificationCurrentnessError::RoleProofMismatch);
    }

    let (earliest, latest) = evaluation_time.consensus_interval();
    let proof_time = role_proof.evaluation_time_unix_s();
    if proof_time < earliest || proof_time > latest {
        return Err(QualificationCurrentnessError::RoleProofOutsideEvaluationInterval);
    }

    let currentness_sha256 = institutional_currentness_digest(
        qualified,
        readiness,
        statement,
        lifecycle_authority,
        role_proof,
        evaluation_time,
    );
    Ok(InstitutionallyCurrentQualifiedScientificClaim {
        qualification_sha256: qualified.qualification_sha256().clone(),
        claim_id: qualified.claim_id().clone(),
        subject_sha256: qualified.subject_sha256().clone(),
        readiness_assessment_sha256: readiness.assessment_sha256().clone(),
        currentness_statement_sha256: statement.statement_sha256().clone(),
        lifecycle_event_sha256: readiness.lifecycle_event_sha256().clone(),
        root_authority_sha256: lifecycle_authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: lifecycle_authority
            .trust_snapshot_authority_sha256()
            .clone(),
        lifecycle_role_authority_sha256: lifecycle_authority.authority_sha256().clone(),
        lifecycle_role_quorum_proof_sha256: role_proof.proof_sha256().clone(),
        evaluation_time_authority_sha256: evaluation_time.authority_sha256().clone(),
        evaluation_earliest_unix_s: earliest,
        evaluation_latest_unix_s: latest,
        currentness_sha256,
    })
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn currentness_statement_digest(
    qualified: &QualifiedScientificClaim,
    readiness: &QualificationValidityReadinessAssessment,
    evaluation_time: &TrustedTime,
    earliest: u64,
    latest: u64,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_STATEMENT_DOMAIN);
    digest.text(qualified.qualification_sha256().as_str());
    digest.text(qualified.claim_id().as_str());
    digest.text(qualified.subject_sha256().as_str());
    digest.text(readiness.assessment_sha256().as_str());
    digest.text(readiness.lifecycle_event_sha256().as_str());
    digest.text(evaluation_time.authority_sha256().as_str());
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.digest()
}

fn institutional_currentness_digest(
    qualified: &QualifiedScientificClaim,
    readiness: &QualificationValidityReadinessAssessment,
    statement: &QualificationCurrentnessStatement,
    lifecycle_authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
    evaluation_time: &TrustedTime,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(INSTITUTIONAL_CURRENTNESS_DOMAIN);
    digest.text(qualified.qualification_sha256().as_str());
    digest.text(qualified.claim_id().as_str());
    digest.text(qualified.subject_sha256().as_str());
    digest.text(readiness.assessment_sha256().as_str());
    digest.text(statement.statement_sha256().as_str());
    digest.text(lifecycle_authority.root_authority_sha256().as_str());
    digest.text(lifecycle_authority.trust_snapshot_authority_sha256().as_str());
    digest.text(lifecycle_authority.authority_sha256().as_str());
    digest.text(role_proof.proof_sha256().as_str());
    digest.text(evaluation_time.authority_sha256().as_str());
    let (earliest, latest) = evaluation_time.consensus_interval();
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.text("institutional-currentness-established");
    digest.text("globally-latest-head-not-established");
    digest.text("global-log-consistency-not-established");
    digest.text("global-evidence-exhaustiveness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn institutional_currentness_is_not_global_truth() {
        fn _assert_api(value: &InstitutionallyCurrentQualifiedScientificClaim) {
            if value.institutional_currentness_established() {
                assert!(!value.globally_latest_head_established());
                assert!(!value.global_log_consistency_established());
                assert!(!value.global_evidence_exhaustiveness_established());
                assert!(!value.scientific_truth_established());
            }
        }
    }
}
