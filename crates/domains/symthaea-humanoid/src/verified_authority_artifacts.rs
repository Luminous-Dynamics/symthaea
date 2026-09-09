// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed verification wrappers for canonical authority evidence artifacts.
//!
//! These wrappers close the gap between "a signed claim labeled Physical" and
//! "the signed claim is exactly the canonical physical-health artifact produced by
//! this runtime". The same invariant is enforced for epistemic state evidence.

use crate::authority_evidence_artifacts::{
    HumanoidEpistemicAuthorityEvidenceArtifact, HumanoidPhysicalAuthorityEvidenceArtifact,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::verified_authority_source::{
    HumanoidAuthoritySourceClaim, HumanoidAuthoritySourceVerificationFailure,
    HumanoidAuthoritySourceVerifier, HumanoidVerifiedAuthorityKind,
    HumanoidVerifiedAuthoritySource, verify_humanoid_authority_source,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidCanonicalAuthorityVerificationFailure {
    ClaimKindMismatch,
    EvidenceDigestMismatch,
    AuthorityScaleMismatch,
    EvaluationTimeMismatch,
    ExpiryMismatch,
    Verification(HumanoidAuthoritySourceVerificationFailure),
}

pub struct HumanoidVerifiedPhysicalAuthorityEvidence {
    inner: HumanoidVerifiedAuthoritySource,
}

impl std::fmt::Debug for HumanoidVerifiedPhysicalAuthorityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidVerifiedPhysicalAuthorityEvidence")
            .field("verification_digest", &self.inner.verification_digest())
            .field("evidence_digest", &self.inner.evidence_digest())
            .field("scale", &self.inner.scale())
            .field("valid_until_s", &self.inner.valid_until_s())
            .finish()
    }
}

impl HumanoidVerifiedPhysicalAuthorityEvidence {
    pub fn verify(
        subject: &HumanoidQualificationSubject,
        artifact: &HumanoidPhysicalAuthorityEvidenceArtifact,
        claim: &HumanoidAuthoritySourceClaim,
        verifier: &dyn HumanoidAuthoritySourceVerifier,
        now_s: f64,
    ) -> Result<Self, HumanoidCanonicalAuthorityVerificationFailure> {
        validate_claim_binding(
            claim,
            HumanoidVerifiedAuthorityKind::Physical,
            artifact.artifact_digest(),
            artifact.authority_scale(),
            artifact.evaluated_at_s(),
            artifact.valid_until_s(),
        )?;
        let inner = verify_humanoid_authority_source(
            subject,
            HumanoidVerifiedAuthorityKind::Physical,
            claim,
            verifier,
            now_s,
        )
        .map_err(HumanoidCanonicalAuthorityVerificationFailure::Verification)?;
        Ok(Self { inner })
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject, now_s: f64) -> bool {
        self.inner
            .validate_for(subject, HumanoidVerifiedAuthorityKind::Physical, now_s)
    }

    pub const fn verification_digest(&self) -> crate::evidence_digest::HumanoidEvidenceDigest {
        self.inner.verification_digest()
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.inner.valid_until_s()
    }

    pub(crate) fn source_snapshot(&self) -> crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot {
        self.inner.source_snapshot()
    }
}

pub struct HumanoidVerifiedEpistemicAuthorityEvidence {
    inner: HumanoidVerifiedAuthoritySource,
}

impl std::fmt::Debug for HumanoidVerifiedEpistemicAuthorityEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidVerifiedEpistemicAuthorityEvidence")
            .field("verification_digest", &self.inner.verification_digest())
            .field("evidence_digest", &self.inner.evidence_digest())
            .field("scale", &self.inner.scale())
            .field("valid_until_s", &self.inner.valid_until_s())
            .finish()
    }
}

impl HumanoidVerifiedEpistemicAuthorityEvidence {
    pub fn verify(
        subject: &HumanoidQualificationSubject,
        artifact: &HumanoidEpistemicAuthorityEvidenceArtifact,
        claim: &HumanoidAuthoritySourceClaim,
        verifier: &dyn HumanoidAuthoritySourceVerifier,
        now_s: f64,
    ) -> Result<Self, HumanoidCanonicalAuthorityVerificationFailure> {
        validate_claim_binding(
            claim,
            HumanoidVerifiedAuthorityKind::Epistemic,
            artifact.artifact_digest(),
            artifact.authority_scale(),
            artifact.evaluated_at_s(),
            artifact.valid_until_s(),
        )?;
        let inner = verify_humanoid_authority_source(
            subject,
            HumanoidVerifiedAuthorityKind::Epistemic,
            claim,
            verifier,
            now_s,
        )
        .map_err(HumanoidCanonicalAuthorityVerificationFailure::Verification)?;
        Ok(Self { inner })
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject, now_s: f64) -> bool {
        self.inner
            .validate_for(subject, HumanoidVerifiedAuthorityKind::Epistemic, now_s)
    }

    pub const fn verification_digest(&self) -> crate::evidence_digest::HumanoidEvidenceDigest {
        self.inner.verification_digest()
    }

    pub const fn valid_until_s(&self) -> f64 {
        self.inner.valid_until_s()
    }

    pub(crate) fn source_snapshot(&self) -> crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot {
        self.inner.source_snapshot()
    }
}

fn validate_claim_binding(
    claim: &HumanoidAuthoritySourceClaim,
    expected_kind: HumanoidVerifiedAuthorityKind,
    evidence_digest: crate::evidence_digest::HumanoidEvidenceDigest,
    authority_scale: f32,
    evaluated_at_s: f64,
    valid_until_s: f64,
) -> Result<(), HumanoidCanonicalAuthorityVerificationFailure> {
    if claim.source_kind() != expected_kind {
        return Err(HumanoidCanonicalAuthorityVerificationFailure::ClaimKindMismatch);
    }
    if claim.evidence_digest() != evidence_digest {
        return Err(HumanoidCanonicalAuthorityVerificationFailure::EvidenceDigestMismatch);
    }
    if claim.scale().to_bits() != authority_scale.to_bits() {
        return Err(HumanoidCanonicalAuthorityVerificationFailure::AuthorityScaleMismatch);
    }
    if claim.evaluated_at_s().to_bits() != evaluated_at_s.to_bits() {
        return Err(HumanoidCanonicalAuthorityVerificationFailure::EvaluationTimeMismatch);
    }
    if claim.valid_until_s().to_bits() != valid_until_s.to_bits() {
        return Err(HumanoidCanonicalAuthorityVerificationFailure::ExpiryMismatch);
    }
    Ok(())
}
