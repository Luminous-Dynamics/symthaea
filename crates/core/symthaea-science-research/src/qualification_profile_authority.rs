// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lifecycle-authenticated authority for scientific qualification profiles.
//!
//! A frozen eligibility profile is caller-declared policy. This module does not
//! change that fact by inspection or naming. Instead, it requires a detached
//! attestation whose signer is cryptographically verified, lifecycle-valid, and
//! explicitly authorized for the `science.qualification-profile` purpose under
//! a fresh generic trust snapshot.
//!
//! Authorizing a profile still does not qualify any claim.

use serde::Serialize;
use symthaea_trust_core::{
    AttestationEnvelope, AttestationExpectation, AttestationPolicy, AttestationSignatureVerifier,
    AttestationTrustContext, AttestationVerificationReport, Sha256Digest as TrustSha256Digest,
    TrustUsage, VerifiedAttestation, verify_attestation_authority,
};

use crate::{FrozenQualificationEligibilityProfile, Sha256Digest};

pub const SCIENCE_QUALIFICATION_PROFILE_USAGE: &str = "science.qualification-profile";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationProfileAuthorityError {
    AttestationRejected(AttestationVerificationReport),
}

/// Non-forgeable authorization of one exact frozen qualification profile.
///
/// Serializable for retained evidence but intentionally not deserializable into
/// authority. Rehydration must rerun generic attestation verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedQualificationProfile {
    profile: FrozenQualificationEligibilityProfile,
    verified_attestation: VerifiedAttestation,
}

impl AuthorizedQualificationProfile {
    pub fn profile(&self) -> &FrozenQualificationEligibilityProfile {
        &self.profile
    }

    pub fn verified_attestation(&self) -> &VerifiedAttestation {
        &self.verified_attestation
    }

    pub fn profile_sha256(&self) -> &Sha256Digest {
        self.profile.profile_sha256()
    }

    /// Identity of the raw signed envelope only. Downstream authority
    /// composition should normally bind `authorization_authority_sha256()`.
    pub fn authorization_attestation_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.attestation_sha256()
    }

    /// Identity of the actual authority grant: signed envelope + exact signature
    /// policy + exact trust snapshot + evaluation time.
    pub fn authorization_authority_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.authority_sha256()
    }

    pub fn authorization_policy_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.policy_sha256()
    }

    pub fn authorization_trust_snapshot_sha256(&self) -> &TrustSha256Digest {
        self.verified_attestation.trust_snapshot_sha256()
    }

    /// This capability authorizes policy only; it never qualifies a claim.
    pub const fn qualification_established(&self) -> bool {
        false
    }
}

pub fn verify_qualification_profile_authority(
    profile: &FrozenQualificationEligibilityProfile,
    envelope: AttestationEnvelope,
    policy: &AttestationPolicy,
    verifier: &dyn AttestationSignatureVerifier,
    trust: AttestationTrustContext<'_>,
) -> Result<AuthorizedQualificationProfile, QualificationProfileAuthorityError> {
    let purpose = qualification_profile_usage();
    let subject_sha256 = bridge_digest(&profile.profile().subject_sha256);
    let payload_sha256 = bridge_digest(profile.profile_sha256());

    let verified_attestation = verify_attestation_authority(
        envelope,
        AttestationExpectation {
            purpose: &purpose,
            subject_sha256: &subject_sha256,
            payload_sha256: &payload_sha256,
            context_sha256: None,
        },
        policy,
        verifier,
        trust,
    )
    .map_err(QualificationProfileAuthorityError::AttestationRejected)?;

    Ok(AuthorizedQualificationProfile {
        profile: profile.clone(),
        verified_attestation,
    })
}

fn qualification_profile_usage() -> TrustUsage {
    TrustUsage::parse(SCIENCE_QUALIFICATION_PROFILE_USAGE)
        .expect("static science qualification-profile usage is canonical")
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_authority_usage_is_canonical_and_distinct() {
        let usage = qualification_profile_usage();
        assert_eq!(usage.as_str(), SCIENCE_QUALIFICATION_PROFILE_USAGE);
        assert_ne!(usage.as_str(), "science.qualification-decision");
    }
}
