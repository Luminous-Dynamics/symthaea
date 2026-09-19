// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! EKM-081 — unified verifier trust/currentness review for EKM-080.
//!
//! EKM-080 proves that one stable verifier profile accepted all nine activation
//! phase-evidence claims. EKM-081 separately asks whether that profile is allowed
//! by local policy and remains monotonic relative to a caller-held trust
//! checkpoint. Passing this review still does not independently establish
//! verifier honesty, a global current head, epoch issuance, mutation authority,
//! or activation authority.

use super::{
    ActivationEvidenceValidationError, ActivationEvidenceVerifierProfileV1,
    ActivationExecutionEvidenceVerifierV1, VerifiedActivationEvidenceReceiptDigestV1,
    VerifiedActivationEvidenceReceiptV1,
};
use super::super::{
    RestartAuthorityEpochIssuanceRecordDigestV1, RestartAuthorityEpochIssuanceRecordV1,
};
use crate::knowledge::epistemic_restart_verifier_provenance::RestartVerifierProfileV1;
use std::error::Error;
use std::fmt;

const MAX_ID_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationVerifierTrustReviewVersion {
    V1,
}

macro_rules! digest_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name([u8; 32]);
        impl $name {
            pub fn as_bytes(self) -> [u8; 32] { self.0 }
            pub fn to_hex(self) -> String { hex32(self.0) }
        }
    };
}

digest_type!(CanonicalVerifierTrustProfileDigestV1);
digest_type!(ActivationVerifierTrustPolicyDigestV1);
digest_type!(CallerTrustedVerifierCheckpointDigestV1);
digest_type!(ActivationVerifierTrustReviewDigestV1);

/// Strict common verifier identity used for comparison only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalVerifierTrustProfileV1 {
    verifier_id: String,
    implementation_id: String,
    implementation_version: String,
    trust_snapshot_digest: [u8; 32],
    trust_snapshot_sequence: u64,
    configuration_digest: [u8; 32],
    valid_from_cycle: u64,
    valid_until_cycle: u64,
}

impl CanonicalVerifierTrustProfileV1 {
    pub fn from_restart_profile(
        profile: &RestartVerifierProfileV1,
    ) -> Result<Self, ActivationVerifierTrustReviewError> {
        Self::new(
            profile.verifier_id(),
            profile.implementation_id(),
            profile.implementation_version(),
            profile.trust_snapshot_digest(),
            profile.trust_snapshot_sequence(),
            profile.configuration_digest(),
            profile.trust_valid_from_cycle(),
            profile.trust_valid_until_cycle(),
        )
    }

    pub fn from_activation_profile(
        profile: &ActivationEvidenceVerifierProfileV1,
    ) -> Result<Self, ActivationVerifierTrustReviewError> {
        Self::new(
            profile.verifier_id(),
            profile.implementation_id(),
            profile.implementation_version(),
            profile.trust_snapshot_digest(),
            profile.trust_snapshot_sequence(),
            profile.configuration_digest(),
            profile.valid_from_cycle(),
            profile.valid_until_cycle(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        verifier_id: &str,
        implementation_id: &str,
        implementation_version: &str,
        trust_snapshot_digest: [u8; 32],
        trust_snapshot_sequence: u64,
        configuration_digest: [u8; 32],
        valid_from_cycle: u64,
        valid_until_cycle: u64,
    ) -> Result<Self, ActivationVerifierTrustReviewError> {
        validate_id("verifier", verifier_id)?;
        validate_id("implementation", implementation_id)?;
        validate_id("implementation-version", implementation_version)?;
        if trust_snapshot_digest == [0; 32] {
            return Err(ActivationVerifierTrustReviewError::ZeroTrustSnapshotDigest);
        }
        if trust_snapshot_sequence == 0 {
            return Err(ActivationVerifierTrustReviewError::InvalidTrustSnapshotSequence);
        }
        if configuration_digest == [0; 32] {
            return Err(ActivationVerifierTrustReviewError::ZeroConfigurationDigest);
        }
        if valid_from_cycle >= valid_until_cycle {
            return Err(ActivationVerifierTrustReviewError::InvalidValidityWindow);
        }
        Ok(Self {
            verifier_id: verifier_id.to_owned(),
            implementation_id: implementation_id.to_owned(),
            implementation_version: implementation_version.to_owned(),
            trust_snapshot_digest,
            trust_snapshot_sequence,
            configuration_digest,
            valid_from_cycle,
            valid_until_cycle,
        })
    }

    pub fn verifier_id(&self) -> &str { &self.verifier_id }
    pub fn implementation_id(&self) -> &str { &self.implementation_id }
    pub fn implementation_version(&self) -> &str { &self.implementation_version }
    pub fn trust_snapshot_digest(&self) -> [u8; 32] { self.trust_snapshot_digest }
    pub fn trust_snapshot_sequence(&self) -> u64 { self.trust_snapshot_sequence }
    pub fn configuration_digest(&self) -> [u8; 32] { self.configuration_digest }
    pub fn valid_from_cycle(&self) -> u64 { self.valid_from_cycle }
    pub fn valid_until_cycle(&self) -> u64 { self.valid_until_cycle }
    pub fn is_fresh_at(&self, cycle: u64) -> bool {
        cycle >= self.valid_from_cycle && cycle < self.valid_until_cycle
    }
    pub fn profile_digest(
        &self,
    ) -> Result<CanonicalVerifierTrustProfileDigestV1, ActivationVerifierTrustReviewError> {
        digest_profile(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationVerifierTrustPolicyV1 {
    verifier_id: String,
    implementation_id: String,
    implementation_version: String,
    configuration_digest: [u8; 32],
    minimum_trust_snapshot_sequence: u64,
    maximum_validity_span_cycles: u64,
}

impl ActivationVerifierTrustPolicyV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verifier_id: impl Into<String>,
        implementation_id: impl Into<String>,
        implementation_version: impl Into<String>,
        configuration_digest: [u8; 32],
        minimum_trust_snapshot_sequence: u64,
        maximum_validity_span_cycles: u64,
    ) -> Result<Self, ActivationVerifierTrustReviewError> {
        let out = Self {
            verifier_id: verifier_id.into(),
            implementation_id: implementation_id.into(),
            implementation_version: implementation_version.into(),
            configuration_digest,
            minimum_trust_snapshot_sequence,
            maximum_validity_span_cycles,
        };
        out.validate()?;
        Ok(out)
    }

    fn validate(&self) -> Result<(), ActivationVerifierTrustReviewError> {
        validate_id("policy-verifier", &self.verifier_id)?;
        validate_id("policy-implementation", &self.implementation_id)?;
        validate_id("policy-implementation-version", &self.implementation_version)?;
        if self.configuration_digest == [0; 32] {
            return Err(ActivationVerifierTrustReviewError::ZeroPolicyConfigurationDigest);
        }
        if self.minimum_trust_snapshot_sequence == 0 {
            return Err(ActivationVerifierTrustReviewError::InvalidMinimumTrustSnapshotSequence);
        }
        if self.maximum_validity_span_cycles == 0 {
            return Err(ActivationVerifierTrustReviewError::InvalidMaximumValiditySpan);
        }
        Ok(())
    }

    fn accepts(
        &self,
        profile: &CanonicalVerifierTrustProfileV1,
    ) -> Result<(), ActivationVerifierTrustReviewError> {
        self.validate()?;
        if profile.verifier_id != self.verifier_id {
            return Err(ActivationVerifierTrustReviewError::VerifierNotAllowed);
        }
        if profile.implementation_id != self.implementation_id {
            return Err(ActivationVerifierTrustReviewError::ImplementationNotAllowed);
        }
        if profile.implementation_version != self.implementation_version {
            return Err(ActivationVerifierTrustReviewError::ImplementationVersionNotAllowed);
        }
        if profile.configuration_digest != self.configuration_digest {
            return Err(ActivationVerifierTrustReviewError::ConfigurationNotAllowed);
        }
        if profile.trust_snapshot_sequence < self.minimum_trust_snapshot_sequence {
            return Err(ActivationVerifierTrustReviewError::TrustSnapshotBelowPolicyMinimum {
                minimum: self.minimum_trust_snapshot_sequence,
                actual: profile.trust_snapshot_sequence,
            });
        }
        let span = profile.valid_until_cycle.checked_sub(profile.valid_from_cycle)
            .ok_or(ActivationVerifierTrustReviewError::InvalidValidityWindow)?;
        if span > self.maximum_validity_span_cycles {
            return Err(ActivationVerifierTrustReviewError::ValiditySpanExceedsPolicy {
                maximum: self.maximum_validity_span_cycles,
                actual: span,
            });
        }
        Ok(())
    }

    pub fn policy_digest(
        &self,
    ) -> Result<ActivationVerifierTrustPolicyDigestV1, ActivationVerifierTrustReviewError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-ekm-activation-verifier-trust-policy-v1");
        hash_string(&mut hasher, &self.verifier_id)?;
        hash_string(&mut hasher, &self.implementation_id)?;
        hash_string(&mut hasher, &self.implementation_version)?;
        hasher.update(&self.configuration_digest);
        hasher.update(&self.minimum_trust_snapshot_sequence.to_le_bytes());
        hasher.update(&self.maximum_validity_span_cycles.to_le_bytes());
        Ok(ActivationVerifierTrustPolicyDigestV1(*hasher.finalize().as_bytes()))
    }
}

/// Caller-designated trust checkpoint. It records a trust assertion; it does not
/// independently prove that the profile is globally trustworthy or current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallerTrustedActivationVerifierCheckpointV1 {
    profile: CanonicalVerifierTrustProfileV1,
    trusted_at_cycle: u64,
    external_trust_anchor_digest: [u8; 32],
    checkpoint_digest: CallerTrustedVerifierCheckpointDigestV1,
}

impl CallerTrustedActivationVerifierCheckpointV1 {
    pub fn from_profile(
        profile: CanonicalVerifierTrustProfileV1,
        trusted_at_cycle: u64,
        external_trust_anchor_digest: [u8; 32],
    ) -> Result<Self, ActivationVerifierTrustReviewError> {
        if external_trust_anchor_digest == [0; 32] {
            return Err(ActivationVerifierTrustReviewError::ZeroExternalTrustAnchorDigest);
        }
        if !profile.is_fresh_at(trusted_at_cycle) {
            return Err(ActivationVerifierTrustReviewError::ReferenceProfileNotFresh {
                observed_at_cycle: trusted_at_cycle,
                valid_from_cycle: profile.valid_from_cycle,
                valid_until_cycle: profile.valid_until_cycle,
            });
        }
        let mut out = Self {
            profile,
            trusted_at_cycle,
            external_trust_anchor_digest,
            checkpoint_digest: CallerTrustedVerifierCheckpointDigestV1([0; 32]),
        };
        out.checkpoint_digest = digest_checkpoint(&out)?;
        Ok(out)
    }

    pub fn profile(&self) -> &CanonicalVerifierTrustProfileV1 { &self.profile }
    pub fn trusted_at_cycle(&self) -> u64 { self.trusted_at_cycle }
    pub fn external_trust_anchor_digest(&self) -> [u8; 32] { self.external_trust_anchor_digest }
    pub fn checkpoint_digest(&self) -> CallerTrustedVerifierCheckpointDigestV1 { self.checkpoint_digest }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationVerifierTrustContinuityDispositionV1 {
    StableTrustReuse,
    TrustSnapshotAdvance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationVerifierTrustReviewReceiptV1 {
    version: ActivationVerifierTrustReviewVersion,
    reviewed_at_cycle: u64,
    activation_evidence_receipt_digest: VerifiedActivationEvidenceReceiptDigestV1,
    issuance_record_digest: RestartAuthorityEpochIssuanceRecordDigestV1,
    candidate_profile: CanonicalVerifierTrustProfileV1,
    candidate_profile_digest: CanonicalVerifierTrustProfileDigestV1,
    trust_policy_digest: ActivationVerifierTrustPolicyDigestV1,
    reference_checkpoint_digest: CallerTrustedVerifierCheckpointDigestV1,
    continuity_disposition: ActivationVerifierTrustContinuityDispositionV1,
    local_policy_accepted: bool,
    continuity_relative_to_caller_checkpoint_proven: bool,
    profile_fresh_at_review_cycle: bool,
    provider_trust_independently_established: bool,
    global_current_head_independently_proven: bool,
    epoch_issuance_chain_verified: bool,
    epoch_issuance_authorized: bool,
    mutation_authority: bool,
    activation_authorized: bool,
    review_digest: ActivationVerifierTrustReviewDigestV1,
}

impl ActivationVerifierTrustReviewReceiptV1 {
    pub fn version(&self) -> ActivationVerifierTrustReviewVersion { self.version }
    pub fn reviewed_at_cycle(&self) -> u64 { self.reviewed_at_cycle }
    pub fn activation_evidence_receipt_digest(&self) -> VerifiedActivationEvidenceReceiptDigestV1 { self.activation_evidence_receipt_digest }
    pub fn issuance_record_digest(&self) -> RestartAuthorityEpochIssuanceRecordDigestV1 { self.issuance_record_digest }
    pub fn candidate_profile(&self) -> &CanonicalVerifierTrustProfileV1 { &self.candidate_profile }
    pub fn candidate_profile_digest(&self) -> CanonicalVerifierTrustProfileDigestV1 { self.candidate_profile_digest }
    pub fn trust_policy_digest(&self) -> ActivationVerifierTrustPolicyDigestV1 { self.trust_policy_digest }
    pub fn reference_checkpoint_digest(&self) -> CallerTrustedVerifierCheckpointDigestV1 { self.reference_checkpoint_digest }
    pub fn continuity_disposition(&self) -> ActivationVerifierTrustContinuityDispositionV1 { self.continuity_disposition }
    pub fn local_policy_accepted(&self) -> bool { self.local_policy_accepted }
    pub fn continuity_relative_to_caller_checkpoint_proven(&self) -> bool { self.continuity_relative_to_caller_checkpoint_proven }
    pub fn profile_fresh_at_review_cycle(&self) -> bool { self.profile_fresh_at_review_cycle }
    pub fn provider_trust_independently_established(&self) -> bool { self.provider_trust_independently_established }
    pub fn global_current_head_independently_proven(&self) -> bool { self.global_current_head_independently_proven }
    pub fn epoch_issuance_chain_verified(&self) -> bool { self.epoch_issuance_chain_verified }
    pub fn epoch_issuance_authorized(&self) -> bool { self.epoch_issuance_authorized }
    pub fn mutation_authority(&self) -> bool { self.mutation_authority }
    pub fn activation_authorized(&self) -> bool { self.activation_authorized }
    pub fn review_digest(&self) -> ActivationVerifierTrustReviewDigestV1 { self.review_digest }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_against(
        &self,
        record: &RestartAuthorityEpochIssuanceRecordV1,
        evidence_receipt: &VerifiedActivationEvidenceReceiptV1,
        verifier: &dyn ActivationExecutionEvidenceVerifierV1,
        policy: &ActivationVerifierTrustPolicyV1,
        reference: &CallerTrustedActivationVerifierCheckpointV1,
    ) -> Result<(), ActivationVerifierTrustReviewError> {
        let rebuilt = review_activation_evidence_verifier_trust(
            record,
            evidence_receipt,
            verifier,
            policy,
            reference,
            self.reviewed_at_cycle,
        )?;
        if rebuilt != *self {
            return Err(ActivationVerifierTrustReviewError::ReceiptMismatch);
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn review_activation_evidence_verifier_trust(
    record: &RestartAuthorityEpochIssuanceRecordV1,
    evidence_receipt: &VerifiedActivationEvidenceReceiptV1,
    verifier: &dyn ActivationExecutionEvidenceVerifierV1,
    policy: &ActivationVerifierTrustPolicyV1,
    reference: &CallerTrustedActivationVerifierCheckpointV1,
    reviewed_at_cycle: u64,
) -> Result<ActivationVerifierTrustReviewReceiptV1, ActivationVerifierTrustReviewError> {
    evidence_receipt.verify_against(record, verifier)
        .map_err(ActivationVerifierTrustReviewError::ActivationEvidenceInvalid)?;
    if reviewed_at_cycle < evidence_receipt.verified_at_cycle() {
        return Err(ActivationVerifierTrustReviewError::ReviewPredatesEvidenceValidation {
            reviewed_at_cycle,
            evidence_verified_at_cycle: evidence_receipt.verified_at_cycle(),
        });
    }
    if reviewed_at_cycle < reference.trusted_at_cycle {
        return Err(ActivationVerifierTrustReviewError::ReviewPredatesReferenceCheckpoint {
            reviewed_at_cycle,
            reference_trusted_at_cycle: reference.trusted_at_cycle,
        });
    }

    let candidate = CanonicalVerifierTrustProfileV1::from_activation_profile(
        evidence_receipt.verifier_profile(),
    )?;
    let disposition = review_candidate_profile(&candidate, policy, reference, reviewed_at_cycle)?;
    let candidate_profile_digest = candidate.profile_digest()?;
    let trust_policy_digest = policy.policy_digest()?;
    let mut out = ActivationVerifierTrustReviewReceiptV1 {
        version: ActivationVerifierTrustReviewVersion::V1,
        reviewed_at_cycle,
        activation_evidence_receipt_digest: evidence_receipt.receipt_digest(),
        issuance_record_digest: record.record_digest(),
        candidate_profile: candidate,
        candidate_profile_digest,
        trust_policy_digest,
        reference_checkpoint_digest: reference.checkpoint_digest,
        continuity_disposition: disposition,
        local_policy_accepted: true,
        continuity_relative_to_caller_checkpoint_proven: true,
        profile_fresh_at_review_cycle: true,
        provider_trust_independently_established: false,
        global_current_head_independently_proven: false,
        epoch_issuance_chain_verified: false,
        epoch_issuance_authorized: false,
        mutation_authority: false,
        activation_authorized: false,
        review_digest: ActivationVerifierTrustReviewDigestV1([0; 32]),
    };
    out.review_digest = digest_review_receipt(&out)?;
    Ok(out)
}

fn review_candidate_profile(
    candidate: &CanonicalVerifierTrustProfileV1,
    policy: &ActivationVerifierTrustPolicyV1,
    reference: &CallerTrustedActivationVerifierCheckpointV1,
    reviewed_at_cycle: u64,
) -> Result<ActivationVerifierTrustContinuityDispositionV1, ActivationVerifierTrustReviewError> {
    if reference.checkpoint_digest != digest_checkpoint(reference)? {
        return Err(ActivationVerifierTrustReviewError::ReferenceCheckpointDigestMismatch);
    }
    if !candidate.is_fresh_at(reviewed_at_cycle) {
        return Err(ActivationVerifierTrustReviewError::CandidateProfileNotFresh {
            observed_at_cycle: reviewed_at_cycle,
            valid_from_cycle: candidate.valid_from_cycle,
            valid_until_cycle: candidate.valid_until_cycle,
        });
    }
    policy.accepts(candidate)?;
    let trusted = reference.profile();
    if candidate.verifier_id != trusted.verifier_id {
        return Err(ActivationVerifierTrustReviewError::VerifierIdentityChanged);
    }
    if candidate.implementation_id != trusted.implementation_id {
        return Err(ActivationVerifierTrustReviewError::ImplementationIdentityChanged);
    }
    if candidate.implementation_version != trusted.implementation_version {
        return Err(ActivationVerifierTrustReviewError::ImplementationVersionChanged);
    }
    if candidate.configuration_digest != trusted.configuration_digest {
        return Err(ActivationVerifierTrustReviewError::ConfigurationChanged);
    }
    if candidate.trust_snapshot_sequence < trusted.trust_snapshot_sequence {
        return Err(ActivationVerifierTrustReviewError::TrustSnapshotRollback {
            trusted: trusted.trust_snapshot_sequence,
            candidate: candidate.trust_snapshot_sequence,
        });
    }
    if candidate.trust_snapshot_sequence == trusted.trust_snapshot_sequence
        && candidate.trust_snapshot_digest != trusted.trust_snapshot_digest
    {
        return Err(ActivationVerifierTrustReviewError::SameSequenceSnapshotSubstitution);
    }
    Ok(if candidate.trust_snapshot_sequence == trusted.trust_snapshot_sequence {
        ActivationVerifierTrustContinuityDispositionV1::StableTrustReuse
    } else {
        ActivationVerifierTrustContinuityDispositionV1::TrustSnapshotAdvance
    })
}

fn digest_profile(
    profile: &CanonicalVerifierTrustProfileV1,
) -> Result<CanonicalVerifierTrustProfileDigestV1, ActivationVerifierTrustReviewError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-canonical-verifier-trust-profile-v1");
    hash_string(&mut hasher, &profile.verifier_id)?;
    hash_string(&mut hasher, &profile.implementation_id)?;
    hash_string(&mut hasher, &profile.implementation_version)?;
    hasher.update(&profile.trust_snapshot_digest);
    hasher.update(&profile.trust_snapshot_sequence.to_le_bytes());
    hasher.update(&profile.configuration_digest);
    hasher.update(&profile.valid_from_cycle.to_le_bytes());
    hasher.update(&profile.valid_until_cycle.to_le_bytes());
    Ok(CanonicalVerifierTrustProfileDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_checkpoint(
    checkpoint: &CallerTrustedActivationVerifierCheckpointV1,
) -> Result<CallerTrustedVerifierCheckpointDigestV1, ActivationVerifierTrustReviewError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-caller-trusted-activation-verifier-checkpoint-v1");
    hasher.update(&checkpoint.profile.profile_digest()?.as_bytes());
    hasher.update(&checkpoint.trusted_at_cycle.to_le_bytes());
    hasher.update(&checkpoint.external_trust_anchor_digest);
    Ok(CallerTrustedVerifierCheckpointDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_review_receipt(
    receipt: &ActivationVerifierTrustReviewReceiptV1,
) -> Result<ActivationVerifierTrustReviewDigestV1, ActivationVerifierTrustReviewError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-activation-verifier-trust-review-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.reviewed_at_cycle.to_le_bytes());
    hasher.update(&receipt.activation_evidence_receipt_digest.as_bytes());
    hasher.update(&receipt.issuance_record_digest.as_bytes());
    hasher.update(&receipt.candidate_profile_digest.as_bytes());
    hasher.update(&receipt.trust_policy_digest.as_bytes());
    hasher.update(&receipt.reference_checkpoint_digest.as_bytes());
    hasher.update(&[match receipt.continuity_disposition {
        ActivationVerifierTrustContinuityDispositionV1::StableTrustReuse => 1,
        ActivationVerifierTrustContinuityDispositionV1::TrustSnapshotAdvance => 2,
    }]);
    for value in [
        receipt.local_policy_accepted,
        receipt.continuity_relative_to_caller_checkpoint_proven,
        receipt.profile_fresh_at_review_cycle,
        receipt.provider_trust_independently_established,
        receipt.global_current_head_independently_proven,
        receipt.epoch_issuance_chain_verified,
        receipt.epoch_issuance_authorized,
        receipt.mutation_authority,
        receipt.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    Ok(ActivationVerifierTrustReviewDigestV1(*hasher.finalize().as_bytes()))
}

fn validate_id(
    label: &'static str,
    value: &str,
) -> Result<(), ActivationVerifierTrustReviewError> {
    if value.is_empty() || value != value.trim() {
        return Err(ActivationVerifierTrustReviewError::InvalidIdentifier(label));
    }
    if value.len() > MAX_ID_BYTES {
        return Err(ActivationVerifierTrustReviewError::IdentifierTooLarge {
            label,
            actual: value.len(),
            maximum: MAX_ID_BYTES,
        });
    }
    Ok(())
}

fn hash_string(
    hasher: &mut blake3::Hasher,
    value: &str,
) -> Result<(), ActivationVerifierTrustReviewError> {
    let len = u64::try_from(value.len())
        .map_err(|_| ActivationVerifierTrustReviewError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[derive(Debug)]
pub enum ActivationVerifierTrustReviewError {
    ActivationEvidenceInvalid(ActivationEvidenceValidationError),
    InvalidIdentifier(&'static str),
    IdentifierTooLarge { label: &'static str, actual: usize, maximum: usize },
    ZeroTrustSnapshotDigest,
    InvalidTrustSnapshotSequence,
    ZeroConfigurationDigest,
    InvalidValidityWindow,
    ZeroPolicyConfigurationDigest,
    InvalidMinimumTrustSnapshotSequence,
    InvalidMaximumValiditySpan,
    VerifierNotAllowed,
    ImplementationNotAllowed,
    ImplementationVersionNotAllowed,
    ConfigurationNotAllowed,
    TrustSnapshotBelowPolicyMinimum { minimum: u64, actual: u64 },
    ValiditySpanExceedsPolicy { maximum: u64, actual: u64 },
    ZeroExternalTrustAnchorDigest,
    ReferenceProfileNotFresh { observed_at_cycle: u64, valid_from_cycle: u64, valid_until_cycle: u64 },
    ReferenceCheckpointDigestMismatch,
    ReviewPredatesEvidenceValidation { reviewed_at_cycle: u64, evidence_verified_at_cycle: u64 },
    ReviewPredatesReferenceCheckpoint { reviewed_at_cycle: u64, reference_trusted_at_cycle: u64 },
    CandidateProfileNotFresh { observed_at_cycle: u64, valid_from_cycle: u64, valid_until_cycle: u64 },
    VerifierIdentityChanged,
    ImplementationIdentityChanged,
    ImplementationVersionChanged,
    ConfigurationChanged,
    TrustSnapshotRollback { trusted: u64, candidate: u64 },
    SameSequenceSnapshotSubstitution,
    LengthOverflow,
    ReceiptMismatch,
}

impl fmt::Display for ActivationVerifierTrustReviewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "activation verifier trust review failed: {self:?}")
    }
}
impl Error for ActivationVerifierTrustReviewError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn activation_profile(sequence: u64, snapshot: [u8; 32]) -> ActivationEvidenceVerifierProfileV1 {
        ActivationEvidenceVerifierProfileV1::new(
            "verifier-a", "impl-a", "1", snapshot, sequence, [7; 32], 10, 200,
        ).unwrap()
    }

    fn policy() -> ActivationVerifierTrustPolicyV1 {
        ActivationVerifierTrustPolicyV1::new(
            "verifier-a", "impl-a", "1", [7; 32], 3, 200,
        ).unwrap()
    }

    #[test]
    fn restart_and_activation_profiles_normalize_to_same_identity() {
        let restart = RestartVerifierProfileV1::new(
            "verifier-a", "impl-a", "1", [8; 32], 3, 10, 200, [7; 32],
        ).unwrap();
        let activation = activation_profile(3, [8; 32]);
        let a = CanonicalVerifierTrustProfileV1::from_restart_profile(&restart).unwrap();
        let b = CanonicalVerifierTrustProfileV1::from_activation_profile(&activation).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.profile_digest().unwrap(), b.profile_digest().unwrap());
    }

    #[test]
    fn trust_snapshot_rollback_fails_closed() {
        let trusted = CanonicalVerifierTrustProfileV1::from_activation_profile(
            &activation_profile(4, [9; 32]),
        ).unwrap();
        let checkpoint = CallerTrustedActivationVerifierCheckpointV1::from_profile(
            trusted, 105, [6; 32],
        ).unwrap();
        let candidate = CanonicalVerifierTrustProfileV1::from_activation_profile(
            &activation_profile(3, [8; 32]),
        ).unwrap();
        assert!(matches!(
            review_candidate_profile(&candidate, &policy(), &checkpoint, 120),
            Err(ActivationVerifierTrustReviewError::TrustSnapshotRollback { .. })
        ));
    }
}
