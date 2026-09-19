// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Authenticated cadence policy for qualification currentness.
//!
//! Currentness extension must not be controlled by caller-selected age windows or
//! witness thresholds. This module freezes a per-qualification cadence policy and
//! requires the delegated root's `QualificationLifecycle` role to authorize the
//! exact policy digest. Policy changes are append-only successors: a successor
//! policy commits the exact predecessor policy digest and the role attestation
//! must bind that predecessor as context.
//!
//! An authorized policy is not automatically the latest policy and does not by
//! itself establish currentness. Publication/latest-policy proof remains a
//! separate layer.

use serde::Serialize;
use symthaea_trust_core::{
    AuthorizedTrustRoleAttestation, FramedDigest, RootRoleQuorumProof,
    Sha256Digest as TrustSha256Digest, TrustRole,
};

use crate::{QualifiedScientificClaim, Sha256Digest};

const CURRENTNESS_CADENCE_POLICY_DOMAIN: &str =
    "symthaea.scientific-qualification-currentness-cadence-policy.identity.v1";
const AUTHORIZED_CURRENTNESS_CADENCE_POLICY_DOMAIN: &str =
    "symthaea.authorized-scientific-currentness-cadence-policy.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessCadenceLimits {
    /// Maximum time from the positive institutional-currentness evaluation to a
    /// later currentness lease evaluation.
    pub maximum_currentness_age_s: u64,
    /// Maximum age permitted by the head-federation policy used at the later
    /// evaluation.
    pub maximum_head_age_s: u64,
    /// Maximum checkpoint age permitted by the lifecycle-head freshness policy
    /// that underwrote the positive currentness decision.
    pub maximum_lifecycle_checkpoint_age_s: u64,
    /// Maximum search lag permitted by the evidence-freshness policy that
    /// underwrote the positive currentness decision.
    pub maximum_evidence_search_lag_ms: u64,
    /// Minimum geometry required for a later fresh-head federation.
    pub minimum_converged_views: usize,
    pub minimum_distinct_quorums: usize,
    pub minimum_distinct_principals: usize,
    pub minimum_distinct_organizations: usize,
    pub minimum_distinct_regions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationCurrentnessCadencePolicyIssue {
    VersionOverflow,
    ConvergedViewsBelowFederationMinimum,
    QuorumsBelowFederationMinimum,
    ZeroPrincipalThreshold,
    OrganizationThresholdExceedsPrincipals,
    RegionThresholdExceedsPrincipals,
}

impl QualificationCurrentnessCadenceLimits {
    pub fn validate(&self) -> Result<(), Vec<QualificationCurrentnessCadencePolicyIssue>> {
        let mut issues = Vec::new();
        if self.minimum_converged_views < 2 {
            issues.push(
                QualificationCurrentnessCadencePolicyIssue::ConvergedViewsBelowFederationMinimum,
            );
        }
        if self.minimum_distinct_quorums < 2 {
            issues.push(QualificationCurrentnessCadencePolicyIssue::QuorumsBelowFederationMinimum);
        }
        if self.minimum_distinct_principals == 0 {
            issues.push(QualificationCurrentnessCadencePolicyIssue::ZeroPrincipalThreshold);
        }
        if self.minimum_distinct_organizations > self.minimum_distinct_principals {
            issues.push(
                QualificationCurrentnessCadencePolicyIssue::OrganizationThresholdExceedsPrincipals,
            );
        }
        if self.minimum_distinct_regions > self.minimum_distinct_principals {
            issues.push(QualificationCurrentnessCadencePolicyIssue::RegionThresholdExceedsPrincipals);
        }
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }
}

/// Frozen, content-addressed cadence policy for one exact scientific
/// qualification. Serializable for retained evidence, not caller-constructible
/// without passing the policy constructors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationCurrentnessCadencePolicy {
    qualification_sha256: Sha256Digest,
    version: u64,
    predecessor_policy_sha256: Option<TrustSha256Digest>,
    limits: QualificationCurrentnessCadenceLimits,
    policy_sha256: TrustSha256Digest,
}

impl QualificationCurrentnessCadencePolicy {
    pub fn genesis(
        qualified: &QualifiedScientificClaim,
        limits: QualificationCurrentnessCadenceLimits,
    ) -> Result<Self, Vec<QualificationCurrentnessCadencePolicyIssue>> {
        limits.validate()?;
        Ok(Self::build(
            qualified.qualification_sha256().clone(),
            1,
            None,
            limits,
        ))
    }

    pub fn successor(
        previous: &QualificationCurrentnessCadencePolicy,
        limits: QualificationCurrentnessCadenceLimits,
    ) -> Result<Self, Vec<QualificationCurrentnessCadencePolicyIssue>> {
        limits.validate()?;
        let Some(version) = previous.version.checked_add(1) else {
            return Err(vec![QualificationCurrentnessCadencePolicyIssue::VersionOverflow]);
        };
        Ok(Self::build(
            previous.qualification_sha256.clone(),
            version,
            Some(previous.policy_sha256.clone()),
            limits,
        ))
    }

    fn build(
        qualification_sha256: Sha256Digest,
        version: u64,
        predecessor_policy_sha256: Option<TrustSha256Digest>,
        limits: QualificationCurrentnessCadenceLimits,
    ) -> Self {
        let policy_sha256 = cadence_policy_digest(
            &qualification_sha256,
            version,
            predecessor_policy_sha256.as_ref(),
            &limits,
        );
        Self {
            qualification_sha256,
            version,
            predecessor_policy_sha256,
            limits,
            policy_sha256,
        }
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn version(&self) -> u64 { self.version }
    pub fn predecessor_policy_sha256(&self) -> Option<&TrustSha256Digest> {
        self.predecessor_policy_sha256.as_ref()
    }
    pub fn limits(&self) -> &QualificationCurrentnessCadenceLimits { &self.limits }
    pub fn policy_sha256(&self) -> &TrustSha256Digest { &self.policy_sha256 }
    pub const fn institutional_authority_established(&self) -> bool { false }
    pub const fn current_policy_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationCurrentnessCadenceAuthorityError {
    WrongRole,
    QualificationSubjectMismatch,
    PolicyPayloadMismatch,
    PolicyPredecessorContextMismatch,
    RoleProofWrongRole,
    RoleProofMismatch,
    RoleProofSubjectMismatch,
    RoleProofPayloadMismatch,
    RoleProofContextMismatch,
}

/// Non-deserializable institutional authority over one exact cadence policy.
///
/// This capability establishes that the policy was authorized under one exact
/// delegated root/snapshot. It does not establish that this is the latest policy
/// for the qualification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedQualificationCurrentnessCadencePolicy {
    policy: QualificationCurrentnessCadencePolicy,
    root_authority_sha256: TrustSha256Digest,
    trust_snapshot_authority_sha256: TrustSha256Digest,
    lifecycle_role_authority_sha256: TrustSha256Digest,
    lifecycle_role_quorum_proof_sha256: TrustSha256Digest,
    authorized_at_unix_s: u64,
    authority_sha256: TrustSha256Digest,
}

impl AuthorizedQualificationCurrentnessCadencePolicy {
    pub fn policy(&self) -> &QualificationCurrentnessCadencePolicy { &self.policy }
    pub fn qualification_sha256(&self) -> &Sha256Digest { self.policy.qualification_sha256() }
    pub fn version(&self) -> u64 { self.policy.version() }
    pub fn policy_sha256(&self) -> &TrustSha256Digest { self.policy.policy_sha256() }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &TrustSha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn lifecycle_role_authority_sha256(&self) -> &TrustSha256Digest {
        &self.lifecycle_role_authority_sha256
    }
    pub fn lifecycle_role_quorum_proof_sha256(&self) -> &TrustSha256Digest {
        &self.lifecycle_role_quorum_proof_sha256
    }
    pub fn authorized_at_unix_s(&self) -> u64 { self.authorized_at_unix_s }
    pub fn authority_sha256(&self) -> &TrustSha256Digest { &self.authority_sha256 }

    pub const fn policy_authority_established(&self) -> bool { true }
    pub const fn current_policy_established(&self) -> bool { false }
    pub const fn currentness_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn authorize_currentness_cadence_policy(
    policy: &QualificationCurrentnessCadencePolicy,
    lifecycle_authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> Result<AuthorizedQualificationCurrentnessCadencePolicy, QualificationCurrentnessCadenceAuthorityError> {
    if lifecycle_authority.role() != TrustRole::QualificationLifecycle {
        return Err(QualificationCurrentnessCadenceAuthorityError::WrongRole);
    }
    let subject = bridge_digest(policy.qualification_sha256());
    if lifecycle_authority.subject_sha256() != &subject {
        return Err(QualificationCurrentnessCadenceAuthorityError::QualificationSubjectMismatch);
    }
    if lifecycle_authority.payload_sha256() != policy.policy_sha256() {
        return Err(QualificationCurrentnessCadenceAuthorityError::PolicyPayloadMismatch);
    }
    if lifecycle_authority.context_sha256() != policy.predecessor_policy_sha256() {
        return Err(
            QualificationCurrentnessCadenceAuthorityError::PolicyPredecessorContextMismatch,
        );
    }
    if role_proof.role() != TrustRole::QualificationLifecycle {
        return Err(QualificationCurrentnessCadenceAuthorityError::RoleProofWrongRole);
    }
    if lifecycle_authority.role_quorum_proof_sha256() != role_proof.proof_sha256() {
        return Err(QualificationCurrentnessCadenceAuthorityError::RoleProofMismatch);
    }
    if role_proof.subject_sha256() != &subject {
        return Err(QualificationCurrentnessCadenceAuthorityError::RoleProofSubjectMismatch);
    }
    if role_proof.payload_sha256() != policy.policy_sha256() {
        return Err(QualificationCurrentnessCadenceAuthorityError::RoleProofPayloadMismatch);
    }
    if role_proof.context_sha256() != policy.predecessor_policy_sha256() {
        return Err(QualificationCurrentnessCadenceAuthorityError::RoleProofContextMismatch);
    }

    let authority_sha256 = cadence_policy_authority_digest(
        policy,
        lifecycle_authority,
        role_proof,
    );
    Ok(AuthorizedQualificationCurrentnessCadencePolicy {
        policy: policy.clone(),
        root_authority_sha256: lifecycle_authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: lifecycle_authority
            .trust_snapshot_authority_sha256()
            .clone(),
        lifecycle_role_authority_sha256: lifecycle_authority.authority_sha256().clone(),
        lifecycle_role_quorum_proof_sha256: role_proof.proof_sha256().clone(),
        authorized_at_unix_s: role_proof.evaluation_time_unix_s(),
        authority_sha256,
    })
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn cadence_policy_digest(
    qualification_sha256: &Sha256Digest,
    version: u64,
    predecessor_policy_sha256: Option<&TrustSha256Digest>,
    limits: &QualificationCurrentnessCadenceLimits,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(CURRENTNESS_CADENCE_POLICY_DOMAIN);
    digest.text(qualification_sha256.as_str());
    digest.text(&version.to_string());
    digest.optional_sha(predecessor_policy_sha256);
    digest.text(&limits.maximum_currentness_age_s.to_string());
    digest.text(&limits.maximum_head_age_s.to_string());
    digest.text(&limits.maximum_lifecycle_checkpoint_age_s.to_string());
    digest.text(&limits.maximum_evidence_search_lag_ms.to_string());
    digest.text(&limits.minimum_converged_views.to_string());
    digest.text(&limits.minimum_distinct_quorums.to_string());
    digest.text(&limits.minimum_distinct_principals.to_string());
    digest.text(&limits.minimum_distinct_organizations.to_string());
    digest.text(&limits.minimum_distinct_regions.to_string());
    digest.digest()
}

fn cadence_policy_authority_digest(
    policy: &QualificationCurrentnessCadencePolicy,
    lifecycle_authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(AUTHORIZED_CURRENTNESS_CADENCE_POLICY_DOMAIN);
    digest.text(policy.policy_sha256().as_str());
    digest.text(policy.qualification_sha256().as_str());
    digest.text(&policy.version().to_string());
    digest.optional_sha(policy.predecessor_policy_sha256());
    digest.text(lifecycle_authority.root_authority_sha256().as_str());
    digest.text(lifecycle_authority.trust_snapshot_authority_sha256().as_str());
    digest.text(lifecycle_authority.authority_sha256().as_str());
    digest.text(role_proof.proof_sha256().as_str());
    digest.text(&role_proof.evaluation_time_unix_s().to_string());
    digest.text("policy-authority-established");
    digest.text("current-policy-not-established");
    digest.text("currentness-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn federation_geometry_requires_real_plurality() {
        let limits = QualificationCurrentnessCadenceLimits {
            maximum_currentness_age_s: 0,
            maximum_head_age_s: 0,
            maximum_lifecycle_checkpoint_age_s: 0,
            maximum_evidence_search_lag_ms: 0,
            minimum_converged_views: 1,
            minimum_distinct_quorums: 1,
            minimum_distinct_principals: 1,
            minimum_distinct_organizations: 1,
            minimum_distinct_regions: 1,
        };
        let issues = limits.validate().unwrap_err();
        assert!(issues.contains(
            &QualificationCurrentnessCadencePolicyIssue::ConvergedViewsBelowFederationMinimum,
        ));
        assert!(issues.contains(
            &QualificationCurrentnessCadencePolicyIssue::QuorumsBelowFederationMinimum,
        ));
    }
}
