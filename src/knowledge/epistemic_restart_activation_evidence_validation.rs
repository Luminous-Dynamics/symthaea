// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent provider validation for EKM-079 activation-phase evidence.
//!
//! EKM-079 proves only that nine non-zero evidence digests are present in the
//! canonical EKM-072 phase order. EKM-080 adds a phase-specific runtime/external
//! verification-provider boundary. The provider must accept each claimed phase
//! digest against the complete activation context while its verifier profile
//! remains stable for the whole validation call.
//!
//! Provider acceptance is still not epoch issuance authority. This module does
//! not establish that the provider itself is trustworthy, mint an authority
//! epoch, construct operational receipts, or expose mutation/activation power.

use super::{
    ActivationCommitReceiptDigestV1, AtomicActivationPhaseEvidenceV1,
    RestartAuthorityEpochDigestV1, RestartAuthorityEpochIssuanceRecordDigestV1,
    RestartAuthorityEpochIssuanceRecordError, RestartAuthorityEpochIssuanceRecordV1,
};
use crate::knowledge::epistemic_restart_continuity::activation_transaction_contract::AtomicActivationPhaseV1;
use std::error::Error;
use std::fmt;

const MAX_PROFILE_ID_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationEvidenceValidationVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivationEvidenceVerifierProfileDigestV1([u8; 32]);

impl ActivationEvidenceVerifierProfileDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerifiedActivationEvidenceReceiptDigestV1([u8; 32]);

impl VerifiedActivationEvidenceReceiptDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

/// Stable description of the external/runtime verifier that accepted EKM-079
/// phase evidence. This is provenance, not proof that the verifier is honest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationEvidenceVerifierProfileV1 {
    verifier_id: String,
    implementation_id: String,
    implementation_version: String,
    trust_snapshot_digest: [u8; 32],
    trust_snapshot_sequence: u64,
    configuration_digest: [u8; 32],
    valid_from_cycle: u64,
    valid_until_cycle: u64,
}

impl ActivationEvidenceVerifierProfileV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        verifier_id: impl Into<String>,
        implementation_id: impl Into<String>,
        implementation_version: impl Into<String>,
        trust_snapshot_digest: [u8; 32],
        trust_snapshot_sequence: u64,
        configuration_digest: [u8; 32],
        valid_from_cycle: u64,
        valid_until_cycle: u64,
    ) -> Result<Self, ActivationEvidenceVerifierProfileError> {
        let profile = Self {
            verifier_id: verifier_id.into(),
            implementation_id: implementation_id.into(),
            implementation_version: implementation_version.into(),
            trust_snapshot_digest,
            trust_snapshot_sequence,
            configuration_digest,
            valid_from_cycle,
            valid_until_cycle,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn verifier_id(&self) -> &str {
        &self.verifier_id
    }

    pub fn implementation_id(&self) -> &str {
        &self.implementation_id
    }

    pub fn implementation_version(&self) -> &str {
        &self.implementation_version
    }

    pub fn trust_snapshot_digest(&self) -> [u8; 32] {
        self.trust_snapshot_digest
    }

    pub fn trust_snapshot_sequence(&self) -> u64 {
        self.trust_snapshot_sequence
    }

    pub fn configuration_digest(&self) -> [u8; 32] {
        self.configuration_digest
    }

    pub fn valid_from_cycle(&self) -> u64 {
        self.valid_from_cycle
    }

    pub fn valid_until_cycle(&self) -> u64 {
        self.valid_until_cycle
    }

    pub fn profile_digest(
        &self,
    ) -> Result<ActivationEvidenceVerifierProfileDigestV1, ActivationEvidenceVerifierProfileError>
    {
        self.validate()?;
        digest_profile(self)
    }

    fn validate(&self) -> Result<(), ActivationEvidenceVerifierProfileError> {
        validate_profile_id("verifier", &self.verifier_id)?;
        validate_profile_id("implementation", &self.implementation_id)?;
        validate_profile_id("implementation-version", &self.implementation_version)?;
        if self.trust_snapshot_digest == [0; 32] {
            return Err(ActivationEvidenceVerifierProfileError::ZeroTrustSnapshotDigest);
        }
        if self.trust_snapshot_sequence == 0 {
            return Err(ActivationEvidenceVerifierProfileError::InvalidTrustSnapshotSequence);
        }
        if self.configuration_digest == [0; 32] {
            return Err(ActivationEvidenceVerifierProfileError::ZeroConfigurationDigest);
        }
        if self.valid_from_cycle >= self.valid_until_cycle {
            return Err(ActivationEvidenceVerifierProfileError::InvalidValidityWindow);
        }
        Ok(())
    }
}

/// Complete immutable context against which every phase-specific provider check
/// must evaluate its claimed evidence digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationEvidenceContextV1 {
    deployment_id: String,
    trust_domain_id: String,
    epoch_sequence: u64,
    previous_epoch_digest: Option<[u8; 32]>,
    activated_restart_v2_digest: [u8; 32],
    activation_review_receipt_digest: [u8; 32],
    transaction_contract_digest: [u8; 32],
    expected_live_generation: u64,
    committed_live_generation: u64,
    rollback_bundle_digest: [u8; 32],
    candidate_bundle_digest: [u8; 32],
    installed_state_digest: [u8; 32],
    activated_at_cycle: u64,
    activation_commit_receipt_digest: [u8; 32],
    authority_epoch_digest: [u8; 32],
    issuance_record_digest: [u8; 32],
}

impl ActivationEvidenceContextV1 {
    fn from_record(record: &RestartAuthorityEpochIssuanceRecordV1) -> Self {
        Self {
            deployment_id: record.deployment_id().to_owned(),
            trust_domain_id: record.trust_domain_id().to_owned(),
            epoch_sequence: record.epoch_sequence(),
            previous_epoch_digest: record.previous_epoch_digest().map(|digest| digest.as_bytes()),
            activated_restart_v2_digest: record.activated_restart_v2_digest(),
            activation_review_receipt_digest: record.activation_review_receipt_digest(),
            transaction_contract_digest: record.transaction_contract_digest(),
            expected_live_generation: record.expected_live_generation(),
            committed_live_generation: record.committed_live_generation(),
            rollback_bundle_digest: record.rollback_bundle_digest(),
            candidate_bundle_digest: record.candidate_bundle_digest(),
            installed_state_digest: record.installed_state_digest(),
            activated_at_cycle: record.activated_at_cycle(),
            activation_commit_receipt_digest: record.activation_commit_receipt_digest().as_bytes(),
            authority_epoch_digest: record.authority_epoch_digest().as_bytes(),
            issuance_record_digest: record.record_digest().as_bytes(),
        }
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }
    pub fn trust_domain_id(&self) -> &str {
        &self.trust_domain_id
    }
    pub fn epoch_sequence(&self) -> u64 {
        self.epoch_sequence
    }
    pub fn previous_epoch_digest(&self) -> Option<[u8; 32]> {
        self.previous_epoch_digest
    }
    pub fn activated_restart_v2_digest(&self) -> [u8; 32] {
        self.activated_restart_v2_digest
    }
    pub fn activation_review_receipt_digest(&self) -> [u8; 32] {
        self.activation_review_receipt_digest
    }
    pub fn transaction_contract_digest(&self) -> [u8; 32] {
        self.transaction_contract_digest
    }
    pub fn expected_live_generation(&self) -> u64 {
        self.expected_live_generation
    }
    pub fn committed_live_generation(&self) -> u64 {
        self.committed_live_generation
    }
    pub fn rollback_bundle_digest(&self) -> [u8; 32] {
        self.rollback_bundle_digest
    }
    pub fn candidate_bundle_digest(&self) -> [u8; 32] {
        self.candidate_bundle_digest
    }
    pub fn installed_state_digest(&self) -> [u8; 32] {
        self.installed_state_digest
    }
    pub fn activated_at_cycle(&self) -> u64 {
        self.activated_at_cycle
    }
    pub fn activation_commit_receipt_digest(&self) -> [u8; 32] {
        self.activation_commit_receipt_digest
    }
    pub fn authority_epoch_digest(&self) -> [u8; 32] {
        self.authority_epoch_digest
    }
    pub fn issuance_record_digest(&self) -> [u8; 32] {
        self.issuance_record_digest
    }
}

/// Phase-specific external/runtime evidence verifier.
///
/// Implementations are expected to validate genuine runtime, durable-storage,
/// transparency, hardware, or other deployment evidence appropriate to each
/// named phase. Returning `true` is only provider acceptance; EKM-080 does not
/// establish that the provider itself is trustworthy.
pub trait ActivationExecutionEvidenceVerifierV1 {
    fn profile(&self) -> Result<ActivationEvidenceVerifierProfileV1, String>;

    fn verify_acquire_exclusive_live_epoch_guard(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_activation_review_under_guard(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_expected_live_epoch_comparison(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_rollback_bundle_capture(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_complete_candidate_bundle_staged(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_atomic_live_state_swap(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_installed_state_under_guard(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_trusted_checkpoint_commit(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;

    fn verify_live_epoch_guard_release(
        &self,
        context: &ActivationEvidenceContextV1,
        evidence_digest: [u8; 32],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedActivationPhaseEvidenceV1 {
    phase: AtomicActivationPhaseV1,
    claimed_evidence_digest: [u8; 32],
    acceptance_digest: [u8; 32],
}

impl VerifiedActivationPhaseEvidenceV1 {
    pub fn phase(self) -> AtomicActivationPhaseV1 {
        self.phase
    }
    pub fn claimed_evidence_digest(self) -> [u8; 32] {
        self.claimed_evidence_digest
    }
    pub fn acceptance_digest(self) -> [u8; 32] {
        self.acceptance_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedActivationEvidenceReceiptV1 {
    version: ActivationEvidenceValidationVersion,
    verified_at_cycle: u64,
    issuance_record_digest: RestartAuthorityEpochIssuanceRecordDigestV1,
    activation_commit_receipt_digest: ActivationCommitReceiptDigestV1,
    authority_epoch_digest: RestartAuthorityEpochDigestV1,
    verifier_profile: ActivationEvidenceVerifierProfileV1,
    verifier_profile_digest: ActivationEvidenceVerifierProfileDigestV1,
    phase_acceptances: Vec<VerifiedActivationPhaseEvidenceV1>,
    all_phase_evidence_provider_accepted: bool,
    verifier_profile_stable_during_validation: bool,
    provider_trust_independently_established: bool,
    epoch_issuance_chain_verified: bool,
    epoch_issuance_authorized: bool,
    mutation_authority: bool,
    activation_authorized: bool,
    receipt_digest: VerifiedActivationEvidenceReceiptDigestV1,
}

impl VerifiedActivationEvidenceReceiptV1 {
    pub fn version(&self) -> ActivationEvidenceValidationVersion {
        self.version
    }
    pub fn verified_at_cycle(&self) -> u64 {
        self.verified_at_cycle
    }
    pub fn issuance_record_digest(&self) -> RestartAuthorityEpochIssuanceRecordDigestV1 {
        self.issuance_record_digest
    }
    pub fn activation_commit_receipt_digest(&self) -> ActivationCommitReceiptDigestV1 {
        self.activation_commit_receipt_digest
    }
    pub fn authority_epoch_digest(&self) -> RestartAuthorityEpochDigestV1 {
        self.authority_epoch_digest
    }
    pub fn verifier_profile(&self) -> &ActivationEvidenceVerifierProfileV1 {
        &self.verifier_profile
    }
    pub fn verifier_profile_digest(&self) -> ActivationEvidenceVerifierProfileDigestV1 {
        self.verifier_profile_digest
    }
    pub fn phase_acceptances(&self) -> &[VerifiedActivationPhaseEvidenceV1] {
        &self.phase_acceptances
    }
    pub fn all_phase_evidence_provider_accepted(&self) -> bool {
        self.all_phase_evidence_provider_accepted
    }
    pub fn verifier_profile_stable_during_validation(&self) -> bool {
        self.verifier_profile_stable_during_validation
    }
    pub fn provider_trust_independently_established(&self) -> bool {
        self.provider_trust_independently_established
    }
    pub fn epoch_issuance_chain_verified(&self) -> bool {
        self.epoch_issuance_chain_verified
    }
    pub fn epoch_issuance_authorized(&self) -> bool {
        self.epoch_issuance_authorized
    }
    pub fn mutation_authority(&self) -> bool {
        self.mutation_authority
    }
    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
    pub fn receipt_digest(&self) -> VerifiedActivationEvidenceReceiptDigestV1 {
        self.receipt_digest
    }

    pub fn verify_against(
        &self,
        record: &RestartAuthorityEpochIssuanceRecordV1,
        verifier: &dyn ActivationExecutionEvidenceVerifierV1,
    ) -> Result<(), ActivationEvidenceValidationError> {
        let rebuilt = validate_activation_evidence(record, verifier, self.verified_at_cycle)?;
        if rebuilt != *self {
            return Err(ActivationEvidenceValidationError::ReceiptMismatch);
        }
        Ok(())
    }
}

pub fn validate_activation_evidence(
    record: &RestartAuthorityEpochIssuanceRecordV1,
    verifier: &dyn ActivationExecutionEvidenceVerifierV1,
    verified_at_cycle: u64,
) -> Result<VerifiedActivationEvidenceReceiptV1, ActivationEvidenceValidationError> {
    record
        .verify()
        .map_err(ActivationEvidenceValidationError::IssuanceRecordInvalid)?;

    if verified_at_cycle < record.activated_at_cycle() {
        return Err(ActivationEvidenceValidationError::VerificationPredatesActivation {
            verified_at_cycle,
            activated_at_cycle: record.activated_at_cycle(),
        });
    }

    let profile_before = verifier
        .profile()
        .map_err(ActivationEvidenceValidationError::VerificationProvider)?;
    validate_profile_at(&profile_before, verified_at_cycle)?;
    let profile_digest = profile_before
        .profile_digest()
        .map_err(ActivationEvidenceValidationError::VerifierProfileInvalid)?;

    let context = ActivationEvidenceContextV1::from_record(record);
    let mut phase_acceptances = Vec::with_capacity(record.phase_evidence().len());

    for evidence in record.phase_evidence() {
        let accepted = dispatch_phase_verification(verifier, &context, *evidence)
            .map_err(ActivationEvidenceValidationError::VerificationProvider)?;
        if !accepted {
            return Err(ActivationEvidenceValidationError::PhaseRejected(evidence.phase()));
        }
        phase_acceptances.push(VerifiedActivationPhaseEvidenceV1 {
            phase: evidence.phase(),
            claimed_evidence_digest: evidence.evidence_digest(),
            acceptance_digest: digest_phase_acceptance(
                record.record_digest(),
                profile_digest,
                evidence.phase(),
                evidence.evidence_digest(),
                verified_at_cycle,
            ),
        });
    }

    let profile_after = verifier
        .profile()
        .map_err(ActivationEvidenceValidationError::VerificationProvider)?;
    validate_profile_at(&profile_after, verified_at_cycle)?;
    if profile_after != profile_before {
        return Err(ActivationEvidenceValidationError::VerifierProfileChangedDuringValidation);
    }

    let mut receipt = VerifiedActivationEvidenceReceiptV1 {
        version: ActivationEvidenceValidationVersion::V1,
        verified_at_cycle,
        issuance_record_digest: record.record_digest(),
        activation_commit_receipt_digest: record.activation_commit_receipt_digest(),
        authority_epoch_digest: record.authority_epoch_digest(),
        verifier_profile: profile_before,
        verifier_profile_digest: profile_digest,
        phase_acceptances,
        all_phase_evidence_provider_accepted: true,
        verifier_profile_stable_during_validation: true,
        provider_trust_independently_established: false,
        epoch_issuance_chain_verified: false,
        epoch_issuance_authorized: false,
        mutation_authority: false,
        activation_authorized: false,
        receipt_digest: VerifiedActivationEvidenceReceiptDigestV1([0; 32]),
    };
    receipt.receipt_digest = digest_validation_receipt(&receipt)?;
    Ok(receipt)
}

fn dispatch_phase_verification(
    verifier: &dyn ActivationExecutionEvidenceVerifierV1,
    context: &ActivationEvidenceContextV1,
    evidence: AtomicActivationPhaseEvidenceV1,
) -> Result<bool, String> {
    match evidence.phase() {
        AtomicActivationPhaseV1::AcquireExclusiveLiveEpochGuard => verifier
            .verify_acquire_exclusive_live_epoch_guard(context, evidence.evidence_digest()),
        AtomicActivationPhaseV1::ReverifyActivationReviewUnderGuard => verifier
            .verify_activation_review_under_guard(context, evidence.evidence_digest()),
        AtomicActivationPhaseV1::CompareExpectedLiveEpoch => verifier
            .verify_expected_live_epoch_comparison(context, evidence.evidence_digest()),
        AtomicActivationPhaseV1::CaptureRollbackBundle => verifier
            .verify_rollback_bundle_capture(context, evidence.evidence_digest()),
        AtomicActivationPhaseV1::StageCompleteCandidateBundle => verifier
            .verify_complete_candidate_bundle_staged(context, evidence.evidence_digest()),
        AtomicActivationPhaseV1::AtomicLiveStateSwap => {
            verifier.verify_atomic_live_state_swap(context, evidence.evidence_digest())
        }
        AtomicActivationPhaseV1::VerifyInstalledStateUnderGuard => verifier
            .verify_installed_state_under_guard(context, evidence.evidence_digest()),
        AtomicActivationPhaseV1::CommitTrustedCheckpoints => verifier
            .verify_trusted_checkpoint_commit(context, evidence.evidence_digest()),
        AtomicActivationPhaseV1::ReleaseLiveEpochGuard => {
            verifier.verify_live_epoch_guard_release(context, evidence.evidence_digest())
        }
    }
}

fn validate_profile_at(
    profile: &ActivationEvidenceVerifierProfileV1,
    observed_at_cycle: u64,
) -> Result<(), ActivationEvidenceValidationError> {
    profile
        .validate()
        .map_err(ActivationEvidenceValidationError::VerifierProfileInvalid)?;
    if observed_at_cycle < profile.valid_from_cycle {
        return Err(ActivationEvidenceValidationError::VerifierProfileNotYetValid {
            observed_at_cycle,
            valid_from_cycle: profile.valid_from_cycle,
        });
    }
    if observed_at_cycle >= profile.valid_until_cycle {
        return Err(ActivationEvidenceValidationError::VerifierProfileExpired {
            observed_at_cycle,
            valid_until_cycle: profile.valid_until_cycle,
        });
    }
    Ok(())
}

fn digest_profile(
    profile: &ActivationEvidenceVerifierProfileV1,
) -> Result<ActivationEvidenceVerifierProfileDigestV1, ActivationEvidenceVerifierProfileError> {
    profile.validate()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-activation-evidence-verifier-profile-v1");
    hash_profile_string(&mut hasher, &profile.verifier_id)?;
    hash_profile_string(&mut hasher, &profile.implementation_id)?;
    hash_profile_string(&mut hasher, &profile.implementation_version)?;
    hasher.update(&profile.trust_snapshot_digest);
    hasher.update(&profile.trust_snapshot_sequence.to_le_bytes());
    hasher.update(&profile.configuration_digest);
    hasher.update(&profile.valid_from_cycle.to_le_bytes());
    hasher.update(&profile.valid_until_cycle.to_le_bytes());
    Ok(ActivationEvidenceVerifierProfileDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn digest_phase_acceptance(
    record_digest: RestartAuthorityEpochIssuanceRecordDigestV1,
    profile_digest: ActivationEvidenceVerifierProfileDigestV1,
    phase: AtomicActivationPhaseV1,
    evidence_digest: [u8; 32],
    verified_at_cycle: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-activation-phase-evidence-acceptance-v1");
    hasher.update(&record_digest.as_bytes());
    hasher.update(&profile_digest.as_bytes());
    hasher.update(&[phase_tag(phase)]);
    hasher.update(&evidence_digest);
    hasher.update(&verified_at_cycle.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn digest_validation_receipt(
    receipt: &VerifiedActivationEvidenceReceiptV1,
) -> Result<VerifiedActivationEvidenceReceiptDigestV1, ActivationEvidenceValidationError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-verified-activation-evidence-receipt-v1");
    hasher.update(&[1]);
    hasher.update(&receipt.verified_at_cycle.to_le_bytes());
    hasher.update(&receipt.issuance_record_digest.as_bytes());
    hasher.update(&receipt.activation_commit_receipt_digest.as_bytes());
    hasher.update(&receipt.authority_epoch_digest.as_bytes());
    hasher.update(&receipt.verifier_profile_digest.as_bytes());
    let count = u64::try_from(receipt.phase_acceptances.len())
        .map_err(|_| ActivationEvidenceValidationError::LengthOverflow)?;
    hasher.update(&count.to_le_bytes());
    for acceptance in &receipt.phase_acceptances {
        hasher.update(&[phase_tag(acceptance.phase)]);
        hasher.update(&acceptance.claimed_evidence_digest);
        hasher.update(&acceptance.acceptance_digest);
    }
    for value in [
        receipt.all_phase_evidence_provider_accepted,
        receipt.verifier_profile_stable_during_validation,
        receipt.provider_trust_independently_established,
        receipt.epoch_issuance_chain_verified,
        receipt.epoch_issuance_authorized,
        receipt.mutation_authority,
        receipt.activation_authorized,
    ] {
        hasher.update(&[u8::from(value)]);
    }
    Ok(VerifiedActivationEvidenceReceiptDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn validate_profile_id(
    label: &'static str,
    value: &str,
) -> Result<(), ActivationEvidenceVerifierProfileError> {
    if value.is_empty() || value != value.trim() {
        return Err(ActivationEvidenceVerifierProfileError::InvalidIdentifier(label));
    }
    if value.len() > MAX_PROFILE_ID_BYTES {
        return Err(ActivationEvidenceVerifierProfileError::IdentifierTooLarge {
            label,
            actual: value.len(),
            maximum: MAX_PROFILE_ID_BYTES,
        });
    }
    Ok(())
}

fn hash_profile_string(
    hasher: &mut blake3::Hasher,
    value: &str,
) -> Result<(), ActivationEvidenceVerifierProfileError> {
    let len = u64::try_from(value.len())
        .map_err(|_| ActivationEvidenceVerifierProfileError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(value.as_bytes());
    Ok(())
}

fn phase_tag(phase: AtomicActivationPhaseV1) -> u8 {
    match phase {
        AtomicActivationPhaseV1::AcquireExclusiveLiveEpochGuard => 1,
        AtomicActivationPhaseV1::ReverifyActivationReviewUnderGuard => 2,
        AtomicActivationPhaseV1::CompareExpectedLiveEpoch => 3,
        AtomicActivationPhaseV1::CaptureRollbackBundle => 4,
        AtomicActivationPhaseV1::StageCompleteCandidateBundle => 5,
        AtomicActivationPhaseV1::AtomicLiveStateSwap => 6,
        AtomicActivationPhaseV1::VerifyInstalledStateUnderGuard => 7,
        AtomicActivationPhaseV1::CommitTrustedCheckpoints => 8,
        AtomicActivationPhaseV1::ReleaseLiveEpochGuard => 9,
    }
}

fn hex32(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationEvidenceVerifierProfileError {
    InvalidIdentifier(&'static str),
    IdentifierTooLarge {
        label: &'static str,
        actual: usize,
        maximum: usize,
    },
    ZeroTrustSnapshotDigest,
    InvalidTrustSnapshotSequence,
    ZeroConfigurationDigest,
    InvalidValidityWindow,
    LengthOverflow,
}

impl fmt::Display for ActivationEvidenceVerifierProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "activation evidence verifier profile invalid: {self:?}")
    }
}

impl Error for ActivationEvidenceVerifierProfileError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationEvidenceValidationError {
    IssuanceRecordInvalid(RestartAuthorityEpochIssuanceRecordError),
    VerifierProfileInvalid(ActivationEvidenceVerifierProfileError),
    VerificationPredatesActivation {
        verified_at_cycle: u64,
        activated_at_cycle: u64,
    },
    VerifierProfileNotYetValid {
        observed_at_cycle: u64,
        valid_from_cycle: u64,
    },
    VerifierProfileExpired {
        observed_at_cycle: u64,
        valid_until_cycle: u64,
    },
    VerificationProvider(String),
    PhaseRejected(AtomicActivationPhaseV1),
    VerifierProfileChangedDuringValidation,
    LengthOverflow,
    ReceiptMismatch,
}

impl fmt::Display for ActivationEvidenceValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "activation evidence validation failed: {self:?}")
    }
}

impl Error for ActivationEvidenceValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::epistemic_restart_continuity::activation_transaction_contract::AtomicActivationTransactionContractV1;
    use std::cell::Cell;

    fn synthetic_record(sequence: u64) -> RestartAuthorityEpochIssuanceRecordV1 {
        let contract = AtomicActivationTransactionContractV1::canonical();
        let phase_evidence = contract
            .required_phases()
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                let mut digest = [0u8; 32];
                digest[0] = u8::try_from(index + 1).unwrap();
                AtomicActivationPhaseEvidenceV1 {
                    phase: *phase,
                    evidence_digest: digest,
                }
            })
            .collect();

        let mut record = RestartAuthorityEpochIssuanceRecordV1 {
            version: super::super::RestartAuthorityEpochIssuanceRecordVersion::V1,
            deployment_id: "deployment-a".into(),
            trust_domain_id: "trust-a".into(),
            epoch_sequence: sequence,
            previous_epoch_digest: if sequence == 1 {
                None
            } else {
                Some(RestartAuthorityEpochDigestV1([9; 32]))
            },
            activated_restart_v2_digest: [1; 32],
            activation_review_receipt_digest: [2; 32],
            transaction_contract_digest: contract.contract_digest().as_bytes(),
            expected_live_generation: 40,
            committed_live_generation: 41,
            rollback_bundle_digest: [3; 32],
            candidate_bundle_digest: [4; 32],
            installed_state_digest: [5; 32],
            phase_evidence,
            activated_at_cycle: 100,
            activation_commit_receipt_digest: ActivationCommitReceiptDigestV1([0; 32]),
            authority_epoch_digest: RestartAuthorityEpochDigestV1([0; 32]),
            record_digest: RestartAuthorityEpochIssuanceRecordDigestV1([0; 32]),
        };
        record.activation_commit_receipt_digest = super::super::digest_activation_commit(&record).unwrap();
        record.authority_epoch_digest = super::super::digest_authority_epoch(&record).unwrap();
        record.record_digest = super::super::digest_record(&record).unwrap();
        record
    }

    fn profile(sequence: u64) -> ActivationEvidenceVerifierProfileV1 {
        ActivationEvidenceVerifierProfileV1::new(
            "runtime-verifier",
            "test-implementation",
            "v1",
            [8; 32],
            sequence,
            [7; 32],
            90,
            200,
        )
        .unwrap()
    }

    struct TestVerifier {
        reject: Option<AtomicActivationPhaseV1>,
        profile_calls: Cell<u64>,
        mutate_profile: bool,
    }

    impl TestVerifier {
        fn accept_all() -> Self {
            Self {
                reject: None,
                profile_calls: Cell::new(0),
                mutate_profile: false,
            }
        }

        fn phase_result(
            &self,
            phase: AtomicActivationPhaseV1,
            context: &ActivationEvidenceContextV1,
            digest: [u8; 32],
        ) -> Result<bool, String> {
            if context.committed_live_generation() != context.expected_live_generation() + 1 {
                return Ok(false);
            }
            if digest == [0; 32] {
                return Ok(false);
            }
            Ok(self.reject != Some(phase))
        }
    }

    impl ActivationExecutionEvidenceVerifierV1 for TestVerifier {
        fn profile(&self) -> Result<ActivationEvidenceVerifierProfileV1, String> {
            let call = self.profile_calls.get();
            self.profile_calls.set(call + 1);
            Ok(profile(if self.mutate_profile && call > 0 { 2 } else { 1 }))
        }

        fn verify_acquire_exclusive_live_epoch_guard(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::AcquireExclusiveLiveEpochGuard, c, d)
        }
        fn verify_activation_review_under_guard(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::ReverifyActivationReviewUnderGuard, c, d)
        }
        fn verify_expected_live_epoch_comparison(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::CompareExpectedLiveEpoch, c, d)
        }
        fn verify_rollback_bundle_capture(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::CaptureRollbackBundle, c, d)
        }
        fn verify_complete_candidate_bundle_staged(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::StageCompleteCandidateBundle, c, d)
        }
        fn verify_atomic_live_state_swap(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::AtomicLiveStateSwap, c, d)
        }
        fn verify_installed_state_under_guard(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::VerifyInstalledStateUnderGuard, c, d)
        }
        fn verify_trusted_checkpoint_commit(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::CommitTrustedCheckpoints, c, d)
        }
        fn verify_live_epoch_guard_release(&self, c: &ActivationEvidenceContextV1, d: [u8; 32]) -> Result<bool, String> {
            self.phase_result(AtomicActivationPhaseV1::ReleaseLiveEpochGuard, c, d)
        }
    }

    #[test]
    fn all_nine_phases_must_be_accepted_under_one_stable_profile() {
        let record = synthetic_record(1);
        let verifier = TestVerifier::accept_all();
        let receipt = validate_activation_evidence(&record, &verifier, 110).unwrap();
        assert_eq!(receipt.phase_acceptances().len(), 9);
        assert!(receipt.all_phase_evidence_provider_accepted());
        assert!(receipt.verifier_profile_stable_during_validation());
        assert!(!receipt.provider_trust_independently_established());
        assert!(!receipt.epoch_issuance_authorized());
        assert!(!receipt.mutation_authority());
        assert!(!receipt.activation_authorized());
    }

    #[test]
    fn one_rejected_phase_fails_closed() {
        let record = synthetic_record(1);
        let verifier = TestVerifier {
            reject: Some(AtomicActivationPhaseV1::AtomicLiveStateSwap),
            profile_calls: Cell::new(0),
            mutate_profile: false,
        };
        assert_eq!(
            validate_activation_evidence(&record, &verifier, 110).unwrap_err(),
            ActivationEvidenceValidationError::PhaseRejected(
                AtomicActivationPhaseV1::AtomicLiveStateSwap
            )
        );
    }

    #[test]
    fn verifier_profile_swap_during_validation_fails_closed() {
        let record = synthetic_record(1);
        let verifier = TestVerifier {
            reject: None,
            profile_calls: Cell::new(0),
            mutate_profile: true,
        };
        assert_eq!(
            validate_activation_evidence(&record, &verifier, 110).unwrap_err(),
            ActivationEvidenceValidationError::VerifierProfileChangedDuringValidation
        );
    }

    #[test]
    fn receipt_is_replay_verifiable_against_same_provider_profile() {
        let record = synthetic_record(2);
        let verifier = TestVerifier::accept_all();
        let receipt = validate_activation_evidence(&record, &verifier, 110).unwrap();
        let replay_verifier = TestVerifier::accept_all();
        receipt.verify_against(&record, &replay_verifier).unwrap();
    }
}
