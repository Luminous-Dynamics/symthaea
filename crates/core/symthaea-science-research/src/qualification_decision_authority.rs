// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lifecycle-authenticated claim qualification decisions.
//!
//! This is intentionally distinct from qualification-profile authority. A claim
//! decision may be authenticated only after a review-ready capability exists and
//! the exact eligibility profile has already crossed its own authorization gate.
//! The decision signature binds the review-ready capability as payload and the
//! exact profile *authority grant* as context, including the policy/trust lineage
//! that authorized that profile.
//!
//! This module still does not expose a `QualifiedScientificClaim`; that final
//! capability is minted separately from a verified decision.

use serde::Serialize;
use symthaea_trust_core::{
    AttestationEnvelope, AttestationExpectation, AttestationPolicy, AttestationSignatureVerifier,
    AttestationTrustContext, AttestationVerificationReport, Sha256Digest as TrustSha256Digest,
    TrustUsage, VerifiedAttestation, verify_attestation_authority,
};

use crate::{
    AuthorizedQualificationProfile, QualificationReviewReadyClaim, Sha256Digest,
};

pub const SCIENCE_QUALIFICATION_DECISION_USAGE: &str = "science.qualification-decision";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationDecisionAuthorityError {
    ProfileMismatch,
    ReviewReadyAlreadyClaimsQualification,
    ProfileAuthorityAlreadyClaimsQualification,
    DecisionPredatesProfileAuthorization {
        profile_authorized_at_unix_s: u64,
        decision_evaluated_at_unix_s: u64,
    },
    AttestationRejected(AttestationVerificationReport),
}

/// Authenticated decision over one exact review-ready claim and one exact,
/// separately authorized profile.
///
/// Serializable for retained evidence but intentionally not deserializable into
/// authority. Rehydration must rerun both profile and decision trust gates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerifiedQualificationDecision {
    review_ready: QualificationReviewReadyClaim,
    authorized_profile: AuthorizedQualificationProfile,
    verified_attestation: VerifiedAttestation,
}

impl VerifiedQualificationDecision {
    pub fn review_ready(&self) -> &QualificationReviewReadyClaim {
        &self.review_ready
    }

    pub fn authorized_profile(&self) -> &AuthorizedQualificationProfile {
        &self.authorized_profile
    }

    pub fn verified_attestation(&self) -> &VerifiedAttestation {
        &self.verified_attestation
    }

    /// Raw signed-envelope identity only.
    pub fn decision_attestation_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.attestation_sha256()
    }

    /// Full decision authority identity: envelope + exact signature policy +
    /// trust snapshot + evaluation time.
    pub fn decision_authority_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.authority_sha256()
    }

    pub fn decision_policy_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.policy_sha256()
    }

    pub fn decision_trust_snapshot_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.trust_snapshot_sha256()
    }

    pub const fn decision_authenticated(&self) -> bool {
        true
    }

    /// Authentication of the decision is still kept separate from the final
    /// qualified-claim capability constructor.
    pub const fn qualification_established(&self) -> bool {
        false
    }
}

pub fn verify_qualification_decision_authority(
    review_ready: QualificationReviewReadyClaim,
    authorized_profile: AuthorizedQualificationProfile,
    envelope: AttestationEnvelope,
    policy: &AttestationPolicy,
    verifier: &dyn AttestationSignatureVerifier,
    trust: AttestationTrustContext<'_>,
) -> Result<VerifiedQualificationDecision, QualificationDecisionAuthorityError> {
    if review_ready.profile_sha256() != authorized_profile.profile_sha256() {
        return Err(QualificationDecisionAuthorityError::ProfileMismatch);
    }
    if review_ready.qualification_established() {
        return Err(QualificationDecisionAuthorityError::ReviewReadyAlreadyClaimsQualification);
    }
    if authorized_profile.qualification_established() {
        return Err(
            QualificationDecisionAuthorityError::ProfileAuthorityAlreadyClaimsQualification,
        );
    }

    let profile_authorized_at_unix_s = authorized_profile
        .verified_attestation()
        .evaluation_time_unix_s();
    if trust.evaluation_time_unix_s < profile_authorized_at_unix_s {
        return Err(
            QualificationDecisionAuthorityError::DecisionPredatesProfileAuthorization {
                profile_authorized_at_unix_s,
                decision_evaluated_at_unix_s: trust.evaluation_time_unix_s,
            },
        );
    }

    let purpose = qualification_decision_usage();
    let subject_sha256 = bridge_digest(
        &authorized_profile.profile().profile().subject_sha256,
    );
    let payload_sha256 = bridge_digest(review_ready.readiness_sha256());
    let context_sha256 = authorized_profile.authorization_authority_sha256().clone();

    let verified_attestation = verify_attestation_authority(
        envelope,
        AttestationExpectation {
            purpose: &purpose,
            subject_sha256: &subject_sha256,
            payload_sha256: &payload_sha256,
            context_sha256: Some(&context_sha256),
        },
        policy,
        verifier,
        trust,
    )
    .map_err(QualificationDecisionAuthorityError::AttestationRejected)?;

    Ok(VerifiedQualificationDecision {
        review_ready,
        authorized_profile,
        verified_attestation,
    })
}

fn qualification_decision_usage() -> TrustUsage {
    TrustUsage::parse(SCIENCE_QUALIFICATION_DECISION_USAGE)
        .expect("static science qualification-decision usage is canonical")
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_usage_is_canonical_and_distinct_from_profile_authority() {
        let usage = qualification_decision_usage();
        assert_eq!(usage.as_str(), SCIENCE_QUALIFICATION_DECISION_USAGE);
        assert_ne!(usage.as_str(), crate::SCIENCE_QUALIFICATION_PROFILE_USAGE);
    }
}
