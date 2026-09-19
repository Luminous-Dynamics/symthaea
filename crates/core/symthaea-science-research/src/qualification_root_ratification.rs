// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Delegated-root ratification of historical scientific qualification.
//!
//! Existing `QualifiedScientificClaim` capability predates the delegated root
//! role architecture. This module does not rewrite or relabel that history.
//! Instead, the current institutional root may explicitly ratify (1) the exact
//! historical qualification profile plus its legacy authority grant and then
//! (2) the exact historical qualification in the context of that profile
//! ratification. The resulting capability says "root-ratified now", not
//! "originally root-authorized then".

use serde::Serialize;
use symthaea_trust_core::{
    AuthorizedTrustRoleAttestation, FramedDigest, RootRoleQuorumProof,
    Sha256Digest as TrustSha256Digest, TrustRole,
};

use crate::{QualifiedScientificClaim, ResearchId, Sha256Digest};

const PROFILE_RATIFICATION_DOMAIN: &str =
    "symthaea.science-qualification-profile-root-ratification.identity.v1";
const QUALIFICATION_RATIFICATION_DOMAIN: &str =
    "symthaea.scientific-qualification-root-ratification.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationProfileRatificationError {
    WrongRole,
    RoleProofWrongRole,
    RoleProofMismatch,
    SubjectMismatch,
    ProfilePayloadMismatch,
    LegacyAuthorityContextMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RootRatifiedQualificationProfile {
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    profile_sha256: Sha256Digest,
    legacy_profile_authority_sha256: TrustSha256Digest,
    root_authority_sha256: TrustSha256Digest,
    trust_snapshot_authority_sha256: TrustSha256Digest,
    root_role_authority_sha256: TrustSha256Digest,
    ratified_at_unix_s: u64,
    ratification_sha256: TrustSha256Digest,
}

impl RootRatifiedQualificationProfile {
    pub fn claim_id(&self) -> &ResearchId { &self.claim_id }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn profile_sha256(&self) -> &Sha256Digest { &self.profile_sha256 }
    pub fn legacy_profile_authority_sha256(&self) -> &TrustSha256Digest {
        &self.legacy_profile_authority_sha256
    }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &TrustSha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn root_role_authority_sha256(&self) -> &TrustSha256Digest {
        &self.root_role_authority_sha256
    }
    pub fn ratified_at_unix_s(&self) -> u64 { self.ratified_at_unix_s }
    pub fn ratification_sha256(&self) -> &TrustSha256Digest { &self.ratification_sha256 }
    pub const fn profile_root_ratification_established(&self) -> bool { true }
    pub const fn claim_qualification_established(&self) -> bool { false }
}

pub fn ratify_qualification_profile_under_root(
    qualified: &QualifiedScientificClaim,
    authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> Result<RootRatifiedQualificationProfile, QualificationProfileRatificationError> {
    if authority.role() != TrustRole::QualificationProfile {
        return Err(QualificationProfileRatificationError::WrongRole);
    }
    if role_proof.role() != TrustRole::QualificationProfile {
        return Err(QualificationProfileRatificationError::RoleProofWrongRole);
    }
    if authority.role_quorum_proof_sha256() != role_proof.proof_sha256() {
        return Err(QualificationProfileRatificationError::RoleProofMismatch);
    }
    if authority.subject_sha256() != &bridge_digest(qualified.subject_sha256()) {
        return Err(QualificationProfileRatificationError::SubjectMismatch);
    }
    if authority.payload_sha256() != &bridge_digest(qualified.qualification_profile_sha256()) {
        return Err(QualificationProfileRatificationError::ProfilePayloadMismatch);
    }
    if authority.context_sha256() != Some(qualified.profile_authority_sha256()) {
        return Err(QualificationProfileRatificationError::LegacyAuthorityContextMismatch);
    }

    let ratification_sha256 = profile_ratification_digest(
        qualified,
        authority,
        role_proof.evaluation_time_unix_s(),
    );
    Ok(RootRatifiedQualificationProfile {
        claim_id: qualified.claim_id().clone(),
        subject_sha256: qualified.subject_sha256().clone(),
        profile_sha256: qualified.qualification_profile_sha256().clone(),
        legacy_profile_authority_sha256: qualified.profile_authority_sha256().clone(),
        root_authority_sha256: authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: authority.trust_snapshot_authority_sha256().clone(),
        root_role_authority_sha256: authority.authority_sha256().clone(),
        ratified_at_unix_s: role_proof.evaluation_time_unix_s(),
        ratification_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificQualificationRatificationError {
    WrongRole,
    RoleProofWrongRole,
    RoleProofMismatch,
    RootAuthorityMismatch,
    SubjectMismatch,
    QualificationPayloadMismatch,
    ProfileRatificationContextMismatch,
    DecisionPredatesProfileRatification,
}

/// Institutional adoption of one exact historical scientific qualification.
///
/// Serializable for retained evidence and intentionally not deserializable into
/// authority. This capability does not mutate `QualifiedScientificClaim`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RootRatifiedScientificQualification {
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    qualification_sha256: Sha256Digest,
    profile_ratification_sha256: TrustSha256Digest,
    root_authority_sha256: TrustSha256Digest,
    trust_snapshot_authority_sha256: TrustSha256Digest,
    decision_role_authority_sha256: TrustSha256Digest,
    ratified_at_unix_s: u64,
    ratification_sha256: TrustSha256Digest,
}

impl RootRatifiedScientificQualification {
    pub fn claim_id(&self) -> &ResearchId { &self.claim_id }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn profile_ratification_sha256(&self) -> &TrustSha256Digest {
        &self.profile_ratification_sha256
    }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &TrustSha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn decision_role_authority_sha256(&self) -> &TrustSha256Digest {
        &self.decision_role_authority_sha256
    }
    pub fn ratified_at_unix_s(&self) -> u64 { self.ratified_at_unix_s }
    pub fn ratification_sha256(&self) -> &TrustSha256Digest { &self.ratification_sha256 }

    pub const fn institutional_root_ratification_established(&self) -> bool { true }
    /// Ratification is intentionally not rewritten as a claim about the original
    /// historical authority architecture.
    pub const fn originally_root_authorized_established(&self) -> bool { false }
    pub const fn current_validity_established(&self) -> bool { false }
    pub const fn scientific_truth_established(&self) -> bool { false }
}

pub fn ratify_scientific_qualification_under_root(
    qualified: &QualifiedScientificClaim,
    profile_ratification: &RootRatifiedQualificationProfile,
    authority: &AuthorizedTrustRoleAttestation,
    role_proof: &RootRoleQuorumProof,
) -> Result<RootRatifiedScientificQualification, ScientificQualificationRatificationError> {
    if authority.role() != TrustRole::QualificationDecision {
        return Err(ScientificQualificationRatificationError::WrongRole);
    }
    if role_proof.role() != TrustRole::QualificationDecision {
        return Err(ScientificQualificationRatificationError::RoleProofWrongRole);
    }
    if authority.role_quorum_proof_sha256() != role_proof.proof_sha256() {
        return Err(ScientificQualificationRatificationError::RoleProofMismatch);
    }
    if authority.root_authority_sha256() != profile_ratification.root_authority_sha256() {
        return Err(ScientificQualificationRatificationError::RootAuthorityMismatch);
    }
    if authority.subject_sha256() != &bridge_digest(qualified.subject_sha256()) {
        return Err(ScientificQualificationRatificationError::SubjectMismatch);
    }
    if authority.payload_sha256() != &bridge_digest(qualified.qualification_sha256()) {
        return Err(ScientificQualificationRatificationError::QualificationPayloadMismatch);
    }
    if authority.context_sha256() != Some(profile_ratification.ratification_sha256()) {
        return Err(ScientificQualificationRatificationError::ProfileRatificationContextMismatch);
    }
    if role_proof.evaluation_time_unix_s() < profile_ratification.ratified_at_unix_s() {
        return Err(ScientificQualificationRatificationError::DecisionPredatesProfileRatification);
    }

    let ratification_sha256 = qualification_ratification_digest(
        qualified,
        profile_ratification,
        authority,
        role_proof.evaluation_time_unix_s(),
    );
    Ok(RootRatifiedScientificQualification {
        claim_id: qualified.claim_id().clone(),
        subject_sha256: qualified.subject_sha256().clone(),
        qualification_sha256: qualified.qualification_sha256().clone(),
        profile_ratification_sha256: profile_ratification.ratification_sha256().clone(),
        root_authority_sha256: authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: authority.trust_snapshot_authority_sha256().clone(),
        decision_role_authority_sha256: authority.authority_sha256().clone(),
        ratified_at_unix_s: role_proof.evaluation_time_unix_s(),
        ratification_sha256,
    })
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

fn profile_ratification_digest(
    qualified: &QualifiedScientificClaim,
    authority: &AuthorizedTrustRoleAttestation,
    ratified_at_unix_s: u64,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(PROFILE_RATIFICATION_DOMAIN);
    digest.text(qualified.claim_id().as_str());
    digest.text(qualified.subject_sha256().as_str());
    digest.text(qualified.qualification_profile_sha256().as_str());
    digest.text(qualified.profile_authority_sha256().as_str());
    digest.text(authority.root_authority_sha256().as_str());
    digest.text(authority.trust_snapshot_authority_sha256().as_str());
    digest.text(authority.authority_sha256().as_str());
    digest.text(&ratified_at_unix_s.to_string());
    digest.text("historical-profile-ratified-not-originally-root-authorized");
    digest.digest()
}

fn qualification_ratification_digest(
    qualified: &QualifiedScientificClaim,
    profile_ratification: &RootRatifiedQualificationProfile,
    authority: &AuthorizedTrustRoleAttestation,
    ratified_at_unix_s: u64,
) -> TrustSha256Digest {
    let mut digest = FramedDigest::new(QUALIFICATION_RATIFICATION_DOMAIN);
    digest.text(qualified.claim_id().as_str());
    digest.text(qualified.subject_sha256().as_str());
    digest.text(qualified.qualification_sha256().as_str());
    digest.text(profile_ratification.ratification_sha256().as_str());
    digest.text(authority.root_authority_sha256().as_str());
    digest.text(authority.trust_snapshot_authority_sha256().as_str());
    digest.text(authority.authority_sha256().as_str());
    digest.text(&ratified_at_unix_s.to_string());
    digest.text("institutional-root-ratification-established");
    digest.text("original-root-authorization-not-established");
    digest.text("current-validity-not-established");
    digest.text("scientific-truth-not-established");
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratification_and_original_authorization_are_distinct_semantics() {
        // Kept as a compile-time semantic guard: the capability exposes both
        // statements separately rather than aliasing ratification to history.
        fn _assert_api(value: &RootRatifiedScientificQualification) {
            assert!(value.institutional_root_ratification_established());
            assert!(!value.originally_root_authorized_established());
            assert!(!value.current_validity_established());
            assert!(!value.scientific_truth_established());
        }
    }
}
