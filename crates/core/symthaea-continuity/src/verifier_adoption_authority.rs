// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authenticated verifier-profile adoption and rollback-resistant current-head proof.
//!
//! `VerifierProfile != AdoptionTransition != AuthenticatedAdoption
//!  != CurrentAuthorizedVerifierProfile`.
//!
//! The existing `profile_adoption` types remain transport/configuration. This module
//! adds two opaque authority boundaries without changing their identities:
//!
//! 1. an externally authenticated adoption transition admitted against an already
//!    trusted adoption-authority root; and
//! 2. a fresh rollback-resistant current-head attestation proving that exact admitted
//!    transition has not been superseded.
//! 
//! A validity interval is necessary but never sufficient for currentness. Supersession
//! is established by exact predecessor lineage plus a fresh protected head, not by TTL.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution_journal_anchor::{
    ExecutionJournalAnchorError, ExecutionJournalAnchorProfileId,
    ExecutionJournalAnchorProfileV1,
};
use crate::profile_adoption::{
    VerifierProfileAdoptionError, VerifierProfileAdoptionPredecessorV1,
    VerifierProfileAdoptionSubjectId, VerifierProfileAdoptionSubjectV1,
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
use crate::verifier::{VerificationAdmissionError, VerifierProfileId, VerifierProfileV1};
use crate::witness::EvidenceClass;

pub const VERIFIER_ADOPTION_AUTH_PURPOSE: &str =
    "symthaea.continuity.verifier-adoption-authority.v1";
pub const VERIFIER_ADOPTION_CURRENTNESS_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-verifier-adoption-currentness-claim-v1";
pub const VERIFIER_ADOPTION_CURRENTNESS_AUTH_PURPOSE: &str =
    "symthaea.continuity.verifier-adoption-currentness.v1";

const ADOPTION_AUTH_WIRE_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-adoption-authority-wire.v1\0";
const AUTHENTICATED_ADOPTION_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-verifier-adoption.v1\0";
const QUALIFIED_ADOPTION_DOMAIN: &[u8] =
    b"symthaea.continuity.qualified-verifier-adoption.v1\0";
const CURRENTNESS_CLAIM_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-adoption-currentness-claim.v1\0";
const CURRENTNESS_WIRE_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-adoption-currentness-wire.v1\0";
const CURRENTNESS_AUTH_DOMAIN: &[u8] =
    b"symthaea.continuity.authenticated-verifier-adoption-currentness.v1\0";
const CURRENT_AUTHORIZED_DOMAIN: &[u8] =
    b"symthaea.continuity.current-authorized-verifier-profile.v1\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name([u8; 32]);
        impl $name { pub fn as_bytes(&self) -> &[u8; 32] { &self.0 } }
    };
}

digest_id!(TrustedVerifierAdoptionAuthorityRootId);
digest_id!(AuthenticatedVerifierAdoptionId);
digest_id!(QualifiedVerifierProfileAdoptionId);
digest_id!(VerifierAdoptionCurrentnessClaimId);
digest_id!(AuthenticatedVerifierAdoptionCurrentnessId);
digest_id!(CurrentAuthorizedVerifierProfileId);

/// Provisioned adoption-authority root. This is intentionally non-Serde and has no
/// public constructor. A future Xenia/provisioning adapter must create it only after
/// the organizational adoption root has independently earned trust.
#[derive(Debug)]
pub struct TrustedVerifierAdoptionAuthorityRootV1 {
    root_id: TrustedVerifierAdoptionAuthorityRootId,
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
}

impl TrustedVerifierAdoptionAuthorityRootV1 {
    #[cfg(test)]
    pub(crate) fn provision_for_test(
        authority_subject: impl Into<String>,
        authority_root_id: impl Into<String>,
        authority_root_digest: [u8; 32],
    ) -> Result<Self, VerifierAdoptionAuthorityError> {
        let authority_subject = checked_text("authority_subject", authority_subject.into())?;
        let authority_root_id = checked_text("authority_root_id", authority_root_id.into())?;
        require_nonzero(authority_root_digest, VerifierAdoptionAuthorityError::ZeroAuthorityRootDigest)?;
        let root_id = TrustedVerifierAdoptionAuthorityRootId(domain_hash_parts(
            b"symthaea.continuity.trusted-verifier-adoption-authority-root.v1\0",
            &[authority_subject.as_bytes(), authority_root_id.as_bytes(), &authority_root_digest],
        ));
        Ok(Self { root_id, authority_subject, authority_root_id, authority_root_digest })
    }

    pub fn id(&self) -> TrustedVerifierAdoptionAuthorityRootId { self.root_id }
    pub fn authority_subject(&self) -> &str { &self.authority_subject }
    pub fn authority_root_id(&self) -> &str { &self.authority_root_id }
    pub fn authority_root_digest(&self) -> [u8; 32] { self.authority_root_digest }
}

/// Domain-separated bytes authenticated by the external verifier-adoption authority.
pub fn canonical_verifier_adoption_authentication_bytes(
    transition: &VerifierProfileAdoptionTransitionV1,
) -> Result<Vec<u8>, VerifierAdoptionAuthorityError> {
    transition.validate()?;
    let transition_bytes = transition.canonical_signing_bytes()?;
    let mut out = Vec::with_capacity(transition_bytes.len() + 96);
    out.extend_from_slice(ADOPTION_AUTH_WIRE_DOMAIN);
    out.extend_from_slice(&(transition_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&transition_bytes);
    out.extend_from_slice(transition.transition_digest()?.as_bytes());
    Ok(out)
}

pub fn canonical_verifier_adoption_authentication_digest(
    transition: &VerifierProfileAdoptionTransitionV1,
) -> Result<[u8; 32], VerifierAdoptionAuthorityError> {
    Ok(*blake3::hash(&canonical_verifier_adoption_authentication_bytes(transition)?).as_bytes())
}

/// Result of external cryptographic authentication under the exact provisioned
/// adoption-authority root. Constructor remains crate-owned.
#[derive(Debug)]
pub(crate) struct AuthenticatedVerifierAdoptionV1 {
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    trusted_root_id: TrustedVerifierAdoptionAuthorityRootId,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedVerifierAdoptionId,
}

impl AuthenticatedVerifierAdoptionV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        transition: &VerifierProfileAdoptionTransitionV1,
        trusted_root: &TrustedVerifierAdoptionAuthorityRootV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, VerifierAdoptionAuthorityError> {
        transition.validate()?;
        require_root_matches_subject(trusted_root, transition.subject())?;
        require_nonzero(authentication_evidence_digest, VerifierAdoptionAuthorityError::ZeroAuthenticationEvidenceDigest)?;
        let transition_digest = transition.transition_digest()?;
        let auth_payload_digest = canonical_verifier_adoption_authentication_digest(transition)?;
        let evidence_id = AuthenticatedVerifierAdoptionId(domain_hash_parts(
            AUTHENTICATED_ADOPTION_DOMAIN,
            &[transition_digest.as_bytes(), trusted_root.id().as_bytes(),
                &auth_payload_digest, &authentication_evidence_digest],
        ));
        Ok(Self { transition_digest, trusted_root_id: trusted_root.id(), authentication_evidence_digest, evidence_id })
    }
}

/// Non-Serde proof that one exact adoption transition was authenticated by the exact
/// provisioned adoption authority and admitted against its exact verifier profile.
#[derive(Debug)]
pub struct QualifiedVerifierProfileAdoptionV1 {
    adoption_id: QualifiedVerifierProfileAdoptionId,
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    predecessor: VerifierProfileAdoptionPredecessorV1,
    subject: VerifierProfileAdoptionSubjectV1,
    verifier_profile: VerifierProfileV1,
    trusted_root_id: TrustedVerifierAdoptionAuthorityRootId,
    authenticated_evidence_id: AuthenticatedVerifierAdoptionId,
    admitted_at_unix_ms: u64,
}

impl QualifiedVerifierProfileAdoptionV1 {
    pub(crate) fn qualify(
        transition: &VerifierProfileAdoptionTransitionV1,
        verifier_profile: &VerifierProfileV1,
        trusted_root: &TrustedVerifierAdoptionAuthorityRootV1,
        authenticated: &AuthenticatedVerifierAdoptionV1,
        admitted_at_unix_ms: u64,
    ) -> Result<Self, VerifierAdoptionAuthorityError> {
        transition.validate()?;
        verifier_profile.validate()?;
        transition.subject().validate_against_profile(verifier_profile)?;
        require_root_matches_subject(trusted_root, transition.subject())?;
        if admitted_at_unix_ms == 0 {
            return Err(VerifierAdoptionAuthorityError::ZeroAdmissionTime);
        }
        if admitted_at_unix_ms < transition.subject().valid_from_unix_ms()
            || admitted_at_unix_ms >= transition.subject().valid_until_unix_ms()
        {
            return Err(VerifierAdoptionAuthorityError::AdmissionOutsideValidity);
        }
        let transition_digest = transition.transition_digest()?;
        if authenticated.transition_digest != transition_digest
            || authenticated.trusted_root_id != trusted_root.id()
        {
            return Err(VerifierAdoptionAuthorityError::AuthenticatedTransitionMismatch);
        }
        let adoption_id = QualifiedVerifierProfileAdoptionId(domain_hash_parts(
            QUALIFIED_ADOPTION_DOMAIN,
            &[transition_digest.as_bytes(), transition.subject().id().as_bytes(),
                verifier_profile.id().as_bytes(), trusted_root.id().as_bytes(),
                authenticated.evidence_id.as_bytes(), &admitted_at_unix_ms.to_le_bytes()],
        ));
        Ok(Self {
            adoption_id,
            transition_digest,
            predecessor: transition.predecessor(),
            subject: transition.subject().clone(),
            verifier_profile: verifier_profile.clone(),
            trusted_root_id: trusted_root.id(),
            authenticated_evidence_id: authenticated.evidence_id,
            admitted_at_unix_ms,
        })
    }

    pub fn id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest { self.transition_digest }
    pub fn predecessor(&self) -> VerifierProfileAdoptionPredecessorV1 { self.predecessor }
    pub fn subject(&self) -> &VerifierProfileAdoptionSubjectV1 { &self.subject }
    pub fn verifier_profile(&self) -> &VerifierProfileV1 { &self.verifier_profile }
    pub fn verifier_profile_id(&self) -> VerifierProfileId { self.verifier_profile.id() }
    pub fn generation(&self) -> u64 { self.subject.generation() }
    pub fn trusted_root_id(&self) -> TrustedVerifierAdoptionAuthorityRootId { self.trusted_root_id }
    pub fn admitted_at_unix_ms(&self) -> u64 { self.admitted_at_unix_ms }
}

/// Serializable fresh protected-head claim. It is evidence, not current authority,
/// until authenticated and qualified against the exact admitted transition and exact
/// predecessor currentness proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierAdoptionCurrentnessClaimV1 {
    schema_version: String,
    platform_profile_id: ExecutionJournalAnchorProfileId,
    platform_root_epoch: u64,
    anchor_sequence: u64,
    predecessor_currentness_id: Option<CurrentAuthorizedVerifierProfileId>,
    qualified_adoption_id: QualifiedVerifierProfileAdoptionId,
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    adoption_subject_id: VerifierProfileAdoptionSubjectId,
    verifier_profile_id: VerifierProfileId,
    adoption_generation: u64,
    predecessor_transition_digest: Option<VerifierProfileAdoptionTransitionDigest>,
    adoption_authority_root_digest: [u8; 32],
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
    claim_id: VerifierAdoptionCurrentnessClaimId,
}

impl VerifierAdoptionCurrentnessClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        platform_profile: &ExecutionJournalAnchorProfileV1,
        adoption: &QualifiedVerifierProfileAdoptionV1,
        anchor_sequence: u64,
        predecessor_currentness_id: Option<CurrentAuthorizedVerifierProfileId>,
        freshness_challenge_digest: [u8; 32],
        boot_instance_digest: [u8; 32],
        boot_counter: u64,
        monotonic_counter: u64,
        anchored_at_unix_ms: u64,
        raw_anchor_evidence_digest: [u8; 32],
    ) -> Result<Self, VerifierAdoptionAuthorityError> {
        platform_profile.validate()?;
        validate_currentness_material(
            anchor_sequence, freshness_challenge_digest, boot_instance_digest, boot_counter,
            monotonic_counter, anchored_at_unix_ms, raw_anchor_evidence_digest,
        )?;
        if anchored_at_unix_ms < adoption.admitted_at_unix_ms()
            || anchored_at_unix_ms < adoption.subject().valid_from_unix_ms()
            || anchored_at_unix_ms >= adoption.subject().valid_until_unix_ms()
        {
            return Err(VerifierAdoptionAuthorityError::CurrentnessOutsideValidity);
        }
        let predecessor_transition_digest = match adoption.predecessor() {
            VerifierProfileAdoptionPredecessorV1::Bootstrap => None,
            VerifierProfileAdoptionPredecessorV1::Previous(digest) => Some(digest),
        };
        let claim_id = VerifierAdoptionCurrentnessClaimId(hash_currentness_claim(
            platform_profile.id(), platform_profile.root_epoch(), anchor_sequence,
            predecessor_currentness_id, adoption.id(), adoption.transition_digest(),
            adoption.subject().id(), adoption.verifier_profile_id(), adoption.generation(),
            predecessor_transition_digest, adoption.subject().authority_root_digest(),
            adoption.subject().valid_from_unix_ms(), adoption.subject().valid_until_unix_ms(),
            freshness_challenge_digest, boot_instance_digest, boot_counter, monotonic_counter,
            anchored_at_unix_ms, raw_anchor_evidence_digest,
        ));
        Ok(Self {
            schema_version: VERIFIER_ADOPTION_CURRENTNESS_CLAIM_SCHEMA_V1.to_owned(),
            platform_profile_id: platform_profile.id(),
            platform_root_epoch: platform_profile.root_epoch(),
            anchor_sequence,
            predecessor_currentness_id,
            qualified_adoption_id: adoption.id(),
            transition_digest: adoption.transition_digest(),
            adoption_subject_id: adoption.subject().id(),
            verifier_profile_id: adoption.verifier_profile_id(),
            adoption_generation: adoption.generation(),
            predecessor_transition_digest,
            adoption_authority_root_digest: adoption.subject().authority_root_digest(),
            valid_from_unix_ms: adoption.subject().valid_from_unix_ms(),
            valid_until_unix_ms: adoption.subject().valid_until_unix_ms(),
            freshness_challenge_digest,
            boot_instance_digest,
            boot_counter,
            monotonic_counter,
            anchored_at_unix_ms,
            raw_anchor_evidence_digest,
            claim_id,
        })
    }

    pub fn validate(&self) -> Result<(), VerifierAdoptionAuthorityError> {
        if self.schema_version != VERIFIER_ADOPTION_CURRENTNESS_CLAIM_SCHEMA_V1 {
            return Err(VerifierAdoptionAuthorityError::UnsupportedCurrentnessSchema(self.schema_version.clone()));
        }
        if self.platform_root_epoch == 0 || self.adoption_generation == 0
            || self.valid_from_unix_ms == 0 || self.valid_until_unix_ms == 0
            || self.valid_from_unix_ms >= self.valid_until_unix_ms
            || self.adoption_authority_root_digest == [0; 32]
        {
            return Err(VerifierAdoptionAuthorityError::InvalidCurrentnessContext);
        }
        validate_currentness_material(
            self.anchor_sequence, self.freshness_challenge_digest, self.boot_instance_digest,
            self.boot_counter, self.monotonic_counter, self.anchored_at_unix_ms,
            self.raw_anchor_evidence_digest,
        )?;
        if self.anchored_at_unix_ms < self.valid_from_unix_ms
            || self.anchored_at_unix_ms >= self.valid_until_unix_ms
        {
            return Err(VerifierAdoptionAuthorityError::CurrentnessOutsideValidity);
        }
        let expected = VerifierAdoptionCurrentnessClaimId(hash_currentness_claim(
            self.platform_profile_id, self.platform_root_epoch, self.anchor_sequence,
            self.predecessor_currentness_id, self.qualified_adoption_id, self.transition_digest,
            self.adoption_subject_id, self.verifier_profile_id, self.adoption_generation,
            self.predecessor_transition_digest, self.adoption_authority_root_digest,
            self.valid_from_unix_ms, self.valid_until_unix_ms, self.freshness_challenge_digest,
            self.boot_instance_digest, self.boot_counter, self.monotonic_counter,
            self.anchored_at_unix_ms, self.raw_anchor_evidence_digest,
        ));
        if expected != self.claim_id {
            return Err(VerifierAdoptionAuthorityError::CurrentnessClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> VerifierAdoptionCurrentnessClaimId { self.claim_id }
}

pub fn canonical_verifier_adoption_currentness_claim_bytes(
    claim: &VerifierAdoptionCurrentnessClaimV1,
) -> Result<Vec<u8>, VerifierAdoptionAuthorityError> {
    claim.validate()?;
    let mut out = Vec::with_capacity(768);
    out.extend_from_slice(CURRENTNESS_WIRE_DOMAIN);
    out.extend_from_slice(claim.platform_profile_id.as_bytes());
    out.extend_from_slice(&claim.platform_root_epoch.to_le_bytes());
    out.extend_from_slice(&claim.anchor_sequence.to_le_bytes());
    encode_optional_id(&mut out, claim.predecessor_currentness_id.map(|v| *v.as_bytes()));
    out.extend_from_slice(claim.qualified_adoption_id.as_bytes());
    out.extend_from_slice(claim.transition_digest.as_bytes());
    out.extend_from_slice(claim.adoption_subject_id.as_bytes());
    out.extend_from_slice(claim.verifier_profile_id.as_bytes());
    out.extend_from_slice(&claim.adoption_generation.to_le_bytes());
    encode_optional_id(&mut out, claim.predecessor_transition_digest.map(|v| *v.as_bytes()));
    out.extend_from_slice(&claim.adoption_authority_root_digest);
    out.extend_from_slice(&claim.valid_from_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.valid_until_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.freshness_challenge_digest);
    out.extend_from_slice(&claim.boot_instance_digest);
    out.extend_from_slice(&claim.boot_counter.to_le_bytes());
    out.extend_from_slice(&claim.monotonic_counter.to_le_bytes());
    out.extend_from_slice(&claim.anchored_at_unix_ms.to_le_bytes());
    out.extend_from_slice(&claim.raw_anchor_evidence_digest);
    out.extend_from_slice(claim.claim_id.as_bytes());
    Ok(out)
}

pub fn canonical_verifier_adoption_currentness_claim_digest(
    claim: &VerifierAdoptionCurrentnessClaimV1,
) -> Result<[u8; 32], VerifierAdoptionAuthorityError> {
    Ok(*blake3::hash(&canonical_verifier_adoption_currentness_claim_bytes(claim)?).as_bytes())
}

#[derive(Debug)]
pub(crate) struct AuthenticatedVerifierAdoptionCurrentnessV1 {
    claim: VerifierAdoptionCurrentnessClaimV1,
    platform_profile: ExecutionJournalAnchorProfileV1,
    authentication_evidence_digest: [u8; 32],
    evidence_id: AuthenticatedVerifierAdoptionCurrentnessId,
}

impl AuthenticatedVerifierAdoptionCurrentnessV1 {
    #[cfg(test)]
    pub(crate) fn authenticate_for_test(
        claim: VerifierAdoptionCurrentnessClaimV1,
        platform_profile: ExecutionJournalAnchorProfileV1,
        authentication_evidence_digest: [u8; 32],
    ) -> Result<Self, VerifierAdoptionAuthorityError> {
        claim.validate()?;
        platform_profile.validate()?;
        if claim.platform_profile_id != platform_profile.id()
            || claim.platform_root_epoch != platform_profile.root_epoch()
        {
            return Err(VerifierAdoptionAuthorityError::CurrentnessPlatformRootMismatch);
        }
        require_nonzero(authentication_evidence_digest, VerifierAdoptionAuthorityError::ZeroAuthenticationEvidenceDigest)?;
        let evidence_id = AuthenticatedVerifierAdoptionCurrentnessId(domain_hash_parts(
            CURRENTNESS_AUTH_DOMAIN,
            &[claim.id().as_bytes(), platform_profile.id().as_bytes(),
                &platform_profile.root_epoch().to_le_bytes(), &authentication_evidence_digest],
        ));
        Ok(Self { claim, platform_profile, authentication_evidence_digest, evidence_id })
    }
}

/// Fresh, rollback-resistant, non-Serde proof that the exact admitted adoption is
/// still the current verifier-role head.
#[derive(Debug)]
pub struct CurrentAuthorizedVerifierProfileV1 {
    current_id: CurrentAuthorizedVerifierProfileId,
    adoption_id: QualifiedVerifierProfileAdoptionId,
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    adoption_subject: VerifierProfileAdoptionSubjectV1,
    verifier_profile: VerifierProfileV1,
    platform_profile_id: ExecutionJournalAnchorProfileId,
    platform_root_epoch: u64,
    anchor_sequence: u64,
    predecessor_currentness_id: Option<CurrentAuthorizedVerifierProfileId>,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    authenticated_evidence_id: AuthenticatedVerifierAdoptionCurrentnessId,
}

impl CurrentAuthorizedVerifierProfileV1 {
    pub(crate) fn qualify(
        adoption: &QualifiedVerifierProfileAdoptionV1,
        authenticated: &AuthenticatedVerifierAdoptionCurrentnessV1,
        expected_freshness_challenge: [u8; 32],
        previous: Option<&CurrentAuthorizedVerifierProfileV1>,
    ) -> Result<Self, VerifierAdoptionAuthorityError> {
        authenticated.claim.validate()?;
        authenticated.platform_profile.validate()?;
        require_nonzero(expected_freshness_challenge, VerifierAdoptionAuthorityError::ZeroFreshnessChallenge)?;
        let claim = &authenticated.claim;
        if claim.freshness_challenge_digest != expected_freshness_challenge
            || claim.qualified_adoption_id != adoption.id()
            || claim.transition_digest != adoption.transition_digest()
            || claim.adoption_subject_id != adoption.subject().id()
            || claim.verifier_profile_id != adoption.verifier_profile_id()
            || claim.adoption_generation != adoption.generation()
            || claim.adoption_authority_root_digest != adoption.subject().authority_root_digest()
            || claim.valid_from_unix_ms != adoption.subject().valid_from_unix_ms()
            || claim.valid_until_unix_ms != adoption.subject().valid_until_unix_ms()
            || claim.anchored_at_unix_ms < adoption.admitted_at_unix_ms()
        {
            return Err(VerifierAdoptionAuthorityError::CurrentnessAdoptionMismatch);
        }
        let expected_predecessor = match adoption.predecessor() {
            VerifierProfileAdoptionPredecessorV1::Bootstrap => None,
            VerifierProfileAdoptionPredecessorV1::Previous(digest) => Some(digest),
        };
        if claim.predecessor_transition_digest != expected_predecessor {
            return Err(VerifierAdoptionAuthorityError::CurrentnessAdoptionMismatch);
        }
        validate_currentness_progression(previous, adoption, claim,
            authenticated.platform_profile.id(), authenticated.platform_profile.root_epoch())?;

        let current_id = CurrentAuthorizedVerifierProfileId(domain_hash_parts(
            CURRENT_AUTHORIZED_DOMAIN,
            &[adoption.id().as_bytes(), adoption.transition_digest().as_bytes(),
                adoption.verifier_profile_id().as_bytes(), claim.id().as_bytes(),
                authenticated.evidence_id.as_bytes(), &expected_freshness_challenge,
                &claim.anchored_at_unix_ms.to_le_bytes()],
        ));
        Ok(Self {
            current_id,
            adoption_id: adoption.id(),
            transition_digest: adoption.transition_digest(),
            adoption_subject: adoption.subject().clone(),
            verifier_profile: adoption.verifier_profile().clone(),
            platform_profile_id: authenticated.platform_profile.id(),
            platform_root_epoch: authenticated.platform_profile.root_epoch(),
            anchor_sequence: claim.anchor_sequence,
            predecessor_currentness_id: claim.predecessor_currentness_id,
            freshness_challenge_digest: expected_freshness_challenge,
            boot_instance_digest: claim.boot_instance_digest,
            boot_counter: claim.boot_counter,
            monotonic_counter: claim.monotonic_counter,
            anchored_at_unix_ms: claim.anchored_at_unix_ms,
            authenticated_evidence_id: authenticated.evidence_id,
        })
    }

    pub fn id(&self) -> CurrentAuthorizedVerifierProfileId { self.current_id }
    pub fn adoption_id(&self) -> QualifiedVerifierProfileAdoptionId { self.adoption_id }
    pub fn transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest { self.transition_digest }
    pub fn subject(&self) -> &VerifierProfileAdoptionSubjectV1 { &self.adoption_subject }
    pub fn verifier_profile(&self) -> &VerifierProfileV1 { &self.verifier_profile }
    pub fn verifier_profile_id(&self) -> VerifierProfileId { self.verifier_profile.id() }
    pub fn generation(&self) -> u64 { self.adoption_subject.generation() }
    pub fn evidence_class_ceiling(&self) -> EvidenceClass { self.adoption_subject.evidence_class_ceiling() }
    pub fn platform_profile_id(&self) -> ExecutionJournalAnchorProfileId { self.platform_profile_id }
    pub fn platform_root_epoch(&self) -> u64 { self.platform_root_epoch }
    pub fn anchor_sequence(&self) -> u64 { self.anchor_sequence }
    pub fn freshness_challenge_digest(&self) -> [u8; 32] { self.freshness_challenge_digest }
    pub fn anchored_at_unix_ms(&self) -> u64 { self.anchored_at_unix_ms }
}

fn validate_currentness_progression(
    previous: Option<&CurrentAuthorizedVerifierProfileV1>,
    adoption: &QualifiedVerifierProfileAdoptionV1,
    claim: &VerifierAdoptionCurrentnessClaimV1,
    platform_profile_id: ExecutionJournalAnchorProfileId,
    platform_root_epoch: u64,
) -> Result<(), VerifierAdoptionAuthorityError> {
    match previous {
        None => {
            if adoption.generation() != 1
                || adoption.predecessor() != VerifierProfileAdoptionPredecessorV1::Bootstrap
                || claim.anchor_sequence != 1
                || claim.predecessor_currentness_id.is_some()
            {
                return Err(VerifierAdoptionAuthorityError::InvalidInitialCurrentness);
            }
        }
        Some(previous) => {
            if previous.platform_profile_id != platform_profile_id
                || previous.platform_root_epoch != platform_root_epoch
            {
                return Err(VerifierAdoptionAuthorityError::CurrentnessPlatformRootMismatch);
            }
            let expected_sequence = previous.anchor_sequence.checked_add(1)
                .ok_or(VerifierAdoptionAuthorityError::AnchorSequenceOverflow)?;
            if claim.anchor_sequence != expected_sequence
                || claim.predecessor_currentness_id != Some(previous.id())
            {
                return Err(VerifierAdoptionAuthorityError::CurrentnessSequenceMismatch);
            }

            if adoption.transition_digest() == previous.transition_digest {
                if adoption.generation() != previous.generation()
                    || adoption.verifier_profile_id() != previous.verifier_profile_id()
                    || adoption.subject().id() != previous.subject().id()
                {
                    return Err(VerifierAdoptionAuthorityError::SameHeadDrift);
                }
            } else {
                let expected_generation = previous.generation().checked_add(1)
                    .ok_or(VerifierAdoptionAuthorityError::AdoptionGenerationOverflow)?;
                if adoption.generation() != expected_generation
                    || adoption.predecessor()
                        != VerifierProfileAdoptionPredecessorV1::Previous(previous.transition_digest)
                    || adoption.subject().authority_subject() != previous.subject().authority_subject()
                    || adoption.subject().authority_root_id() != previous.subject().authority_root_id()
                    || adoption.subject().authority_root_digest() != previous.subject().authority_root_digest()
                    || adoption.subject().verifier_role_id() != previous.subject().verifier_role_id()
                {
                    return Err(VerifierAdoptionAuthorityError::AdoptionHeadNotExactSuccessor);
                }
            }

            if claim.boot_counter < previous.boot_counter {
                return Err(VerifierAdoptionAuthorityError::BootCounterRollback);
            }
            if claim.boot_counter == previous.boot_counter {
                if claim.boot_instance_digest != previous.boot_instance_digest {
                    return Err(VerifierAdoptionAuthorityError::BootIdentityDriftWithoutCounterAdvance);
                }
                if claim.monotonic_counter <= previous.monotonic_counter {
                    return Err(VerifierAdoptionAuthorityError::MonotonicCounterRollback);
                }
            }
            if claim.anchored_at_unix_ms < previous.anchored_at_unix_ms {
                return Err(VerifierAdoptionAuthorityError::AnchorTimeRollback);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierAdoptionAuthorityError {
    #[error(transparent)]
    Adoption(#[from] VerifierProfileAdoptionError),
    #[error(transparent)]
    Verifier(#[from] VerificationAdmissionError),
    #[error(transparent)]
    PlatformAnchor(#[from] ExecutionJournalAnchorError),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("trusted verifier-adoption authority root digest must be non-zero")]
    ZeroAuthorityRootDigest,
    #[error("trusted verifier-adoption authority root does not match adoption subject")]
    AdoptionAuthorityRootMismatch,
    #[error("verifier-adoption authentication evidence digest must be non-zero")]
    ZeroAuthenticationEvidenceDigest,
    #[error("verifier adoption admission time must be non-zero")]
    ZeroAdmissionTime,
    #[error("verifier adoption was admitted outside its exact validity interval")]
    AdmissionOutsideValidity,
    #[error("authenticated verifier adoption does not match the exact transition/root")]
    AuthenticatedTransitionMismatch,
    #[error("unsupported verifier-adoption currentness claim schema: {0}")]
    UnsupportedCurrentnessSchema(String),
    #[error("verifier-adoption currentness context is structurally invalid")]
    InvalidCurrentnessContext,
    #[error("verifier-adoption currentness anchor lies outside adoption validity")]
    CurrentnessOutsideValidity,
    #[error("verifier-adoption currentness claim identity mismatch")]
    CurrentnessClaimIdentityMismatch,
    #[error("verifier-adoption currentness platform root/profile mismatch")]
    CurrentnessPlatformRootMismatch,
    #[error("verifier-adoption freshness challenge must be non-zero")]
    ZeroFreshnessChallenge,
    #[error("fresh verifier-adoption currentness does not bind the exact admitted adoption")]
    CurrentnessAdoptionMismatch,
    #[error("initial verifier-adoption currentness must start at generation/sequence 1 with bootstrap lineage")]
    InvalidInitialCurrentness,
    #[error("verifier-adoption currentness sequence/predecessor is not exact successor")]
    CurrentnessSequenceMismatch,
    #[error("same verifier-adoption head re-attestation drifted in generation/profile/subject")]
    SameHeadDrift,
    #[error("new verifier-adoption head is not the exact one-generation successor")]
    AdoptionHeadNotExactSuccessor,
    #[error("verifier-adoption currentness anchor sequence overflow")]
    AnchorSequenceOverflow,
    #[error("verifier-adoption generation overflow")]
    AdoptionGenerationOverflow,
    #[error("verifier-adoption currentness boot counter rolled back")]
    BootCounterRollback,
    #[error("verifier-adoption currentness boot identity changed without counter advance")]
    BootIdentityDriftWithoutCounterAdvance,
    #[error("verifier-adoption currentness monotonic counter did not strictly advance")]
    MonotonicCounterRollback,
    #[error("verifier-adoption currentness anchor time rolled back")]
    AnchorTimeRollback,
}

fn require_root_matches_subject(
    root: &TrustedVerifierAdoptionAuthorityRootV1,
    subject: &VerifierProfileAdoptionSubjectV1,
) -> Result<(), VerifierAdoptionAuthorityError> {
    if root.authority_subject != subject.authority_subject()
        || root.authority_root_id != subject.authority_root_id()
        || root.authority_root_digest != subject.authority_root_digest()
    {
        return Err(VerifierAdoptionAuthorityError::AdoptionAuthorityRootMismatch);
    }
    Ok(())
}

fn validate_currentness_material(
    anchor_sequence: u64,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
) -> Result<(), VerifierAdoptionAuthorityError> {
    if anchor_sequence == 0 || boot_counter == 0 || monotonic_counter == 0 || anchored_at_unix_ms == 0
        || freshness_challenge_digest == [0; 32] || boot_instance_digest == [0; 32]
        || raw_anchor_evidence_digest == [0; 32]
    {
        return Err(VerifierAdoptionAuthorityError::InvalidCurrentnessContext);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn hash_currentness_claim(
    platform_profile_id: ExecutionJournalAnchorProfileId,
    platform_root_epoch: u64,
    anchor_sequence: u64,
    predecessor_currentness_id: Option<CurrentAuthorizedVerifierProfileId>,
    qualified_adoption_id: QualifiedVerifierProfileAdoptionId,
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    adoption_subject_id: VerifierProfileAdoptionSubjectId,
    verifier_profile_id: VerifierProfileId,
    adoption_generation: u64,
    predecessor_transition_digest: Option<VerifierProfileAdoptionTransitionDigest>,
    adoption_authority_root_digest: [u8; 32],
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    freshness_challenge_digest: [u8; 32],
    boot_instance_digest: [u8; 32],
    boot_counter: u64,
    monotonic_counter: u64,
    anchored_at_unix_ms: u64,
    raw_anchor_evidence_digest: [u8; 32],
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(CURRENTNESS_CLAIM_DOMAIN);
    put_id(&mut h, platform_profile_id.as_bytes());
    put_u64(&mut h, platform_root_epoch);
    put_u64(&mut h, anchor_sequence);
    put_optional_id(&mut h, predecessor_currentness_id.map(|v| *v.as_bytes()));
    put_id(&mut h, qualified_adoption_id.as_bytes());
    put_id(&mut h, transition_digest.as_bytes());
    put_id(&mut h, adoption_subject_id.as_bytes());
    put_id(&mut h, verifier_profile_id.as_bytes());
    put_u64(&mut h, adoption_generation);
    put_optional_id(&mut h, predecessor_transition_digest.map(|v| *v.as_bytes()));
    put_id(&mut h, &adoption_authority_root_digest);
    put_u64(&mut h, valid_from_unix_ms);
    put_u64(&mut h, valid_until_unix_ms);
    put_id(&mut h, &freshness_challenge_digest);
    put_id(&mut h, &boot_instance_digest);
    put_u64(&mut h, boot_counter);
    put_u64(&mut h, monotonic_counter);
    put_u64(&mut h, anchored_at_unix_ms);
    put_id(&mut h, &raw_anchor_evidence_digest);
    *h.finalize().as_bytes()
}

fn put_id(h: &mut blake3::Hasher, bytes: &[u8]) {
    h.update(&(bytes.len() as u64).to_le_bytes());
    h.update(bytes);
}
fn put_u64(h: &mut blake3::Hasher, value: u64) { put_id(h, &value.to_le_bytes()); }
fn put_optional_id(h: &mut blake3::Hasher, value: Option<[u8; 32]>) {
    match value { Some(v) => { h.update(&[1]); put_id(h, &v); }, None => h.update(&[0]), }
}
fn encode_optional_id(out: &mut Vec<u8>, value: Option<[u8; 32]>) {
    match value { Some(v) => { out.push(1); out.extend_from_slice(&v); }, None => out.push(0), }
}
fn require_nonzero(
    digest: [u8; 32],
    error: VerifierAdoptionAuthorityError,
) -> Result<(), VerifierAdoptionAuthorityError> {
    if digest == [0; 32] { Err(error) } else { Ok(()) }
}
fn checked_text(field: &'static str, value: String) -> Result<String, VerifierAdoptionAuthorityError> {
    let trimmed = value.trim();
    if trimmed.is_empty() { return Err(VerifierAdoptionAuthorityError::BlankText { field }); }
    if trimmed.len() > 1024 { return Err(VerifierAdoptionAuthorityError::TextTooLong { field }); }
    if trimmed.chars().any(char::is_control) { return Err(VerifierAdoptionAuthorityError::ControlCharacters { field }); }
    Ok(trimmed.to_owned())
}
fn domain_hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    for part in parts { put_id(&mut h, part); }
    *h.finalize().as_bytes()
}
