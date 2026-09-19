// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Passive data representation for a future successful activation commit and the
//! restart-authority epoch identity derived from it.
//!
//! EKM-072 defines the required atomic activation phases. EKM-074 requires a
//! successful activation commit to be the only epoch-issuance boundary. This
//! module defines the exact non-circular digest structure needed to represent
//! that result, but deliberately provides no production constructor, executor,
//! validator against a live transaction, epoch issuer, or mutation authority.

use crate::knowledge::epistemic_restart_continuity::activation_transaction_contract::{
    AtomicActivationPhaseV1, AtomicActivationTransactionContractError,
    AtomicActivationTransactionContractV1,
};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const MAX_ID_BYTES: usize = 16 * 1024;
const PHASE_COUNT: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartAuthorityEpochIssuanceRecordVersion {
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivationCommitReceiptDigestV1([u8; 32]);

impl ActivationCommitReceiptDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartAuthorityEpochDigestV1([u8; 32]);

impl RestartAuthorityEpochDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartAuthorityEpochIssuanceRecordDigestV1([u8; 32]);

impl RestartAuthorityEpochIssuanceRecordDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex32(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomicActivationPhaseEvidenceV1 {
    phase: AtomicActivationPhaseV1,
    evidence_digest: [u8; 32],
}

impl AtomicActivationPhaseEvidenceV1 {
    pub fn phase(self) -> AtomicActivationPhaseV1 {
        self.phase
    }

    pub fn evidence_digest(self) -> [u8; 32] {
        self.evidence_digest
    }
}

/// Passive representation of the evidence that a future activation executor
/// would need to emit before an authority epoch could be issued.
///
/// There is intentionally no production constructor in EKM-079. A later tranche
/// must populate this only from an actually executed, successfully verified atomic
/// activation transaction. Shape/digest validity alone is not proof that any
/// activation event occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartAuthorityEpochIssuanceRecordV1 {
    version: RestartAuthorityEpochIssuanceRecordVersion,
    deployment_id: String,
    trust_domain_id: String,
    epoch_sequence: u64,
    previous_epoch_digest: Option<RestartAuthorityEpochDigestV1>,
    activated_restart_v2_digest: [u8; 32],
    activation_review_receipt_digest: [u8; 32],
    transaction_contract_digest: [u8; 32],
    expected_live_generation: u64,
    committed_live_generation: u64,
    rollback_bundle_digest: [u8; 32],
    candidate_bundle_digest: [u8; 32],
    installed_state_digest: [u8; 32],
    phase_evidence: Vec<AtomicActivationPhaseEvidenceV1>,
    activated_at_cycle: u64,
    activation_commit_receipt_digest: ActivationCommitReceiptDigestV1,
    authority_epoch_digest: RestartAuthorityEpochDigestV1,
    record_digest: RestartAuthorityEpochIssuanceRecordDigestV1,
}

impl RestartAuthorityEpochIssuanceRecordV1 {
    pub fn version(&self) -> RestartAuthorityEpochIssuanceRecordVersion {
        self.version
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

    pub fn previous_epoch_digest(&self) -> Option<RestartAuthorityEpochDigestV1> {
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

    pub fn phase_evidence(&self) -> &[AtomicActivationPhaseEvidenceV1] {
        &self.phase_evidence
    }

    pub fn activated_at_cycle(&self) -> u64 {
        self.activated_at_cycle
    }

    pub fn activation_commit_receipt_digest(&self) -> ActivationCommitReceiptDigestV1 {
        self.activation_commit_receipt_digest
    }

    pub fn authority_epoch_digest(&self) -> RestartAuthorityEpochDigestV1 {
        self.authority_epoch_digest
    }

    pub fn record_digest(&self) -> RestartAuthorityEpochIssuanceRecordDigestV1 {
        self.record_digest
    }

    /// EKM-079 verifies only passive shape and canonical digest relationships.
    pub fn successful_activation_event_independently_verified(&self) -> bool {
        false
    }

    pub fn trusted_checkpoint_commit_independently_verified(&self) -> bool {
        false
    }

    pub fn epoch_issuance_authorized(&self) -> bool {
        false
    }

    pub fn mutation_authority(&self) -> bool {
        false
    }

    pub fn activation_authorized(&self) -> bool {
        false
    }

    pub fn verify(&self) -> Result<(), RestartAuthorityEpochIssuanceRecordError> {
        validate_record(self)?;

        let expected_commit = digest_activation_commit(self)?;
        if self.activation_commit_receipt_digest != expected_commit {
            return Err(RestartAuthorityEpochIssuanceRecordError::ActivationCommitDigestMismatch);
        }

        let expected_epoch = digest_authority_epoch(self)?;
        if self.authority_epoch_digest != expected_epoch {
            return Err(RestartAuthorityEpochIssuanceRecordError::AuthorityEpochDigestMismatch);
        }

        let expected_record = digest_record(self)?;
        if self.record_digest != expected_record {
            return Err(RestartAuthorityEpochIssuanceRecordError::RecordDigestMismatch);
        }
        Ok(())
    }
}

fn validate_record(
    record: &RestartAuthorityEpochIssuanceRecordV1,
) -> Result<(), RestartAuthorityEpochIssuanceRecordError> {
    if record.version != RestartAuthorityEpochIssuanceRecordVersion::V1 {
        return Err(RestartAuthorityEpochIssuanceRecordError::UnsupportedVersion);
    }
    validate_id("deployment", &record.deployment_id)?;
    validate_id("trust-domain", &record.trust_domain_id)?;

    if record.epoch_sequence == 0 {
        return Err(RestartAuthorityEpochIssuanceRecordError::InvalidEpochSequence);
    }
    match (record.epoch_sequence, record.previous_epoch_digest) {
        (1, None) => {}
        (1, Some(_)) => {
            return Err(RestartAuthorityEpochIssuanceRecordError::GenesisHasPredecessor)
        }
        (_, Some(digest)) if digest.0 != [0; 32] => {}
        _ => return Err(RestartAuthorityEpochIssuanceRecordError::MissingPreviousEpoch),
    }

    for (name, digest) in [
        ("activated-restart", record.activated_restart_v2_digest),
        ("activation-review", record.activation_review_receipt_digest),
        ("transaction-contract", record.transaction_contract_digest),
        ("rollback-bundle", record.rollback_bundle_digest),
        ("candidate-bundle", record.candidate_bundle_digest),
        ("installed-state", record.installed_state_digest),
    ] {
        if digest == [0; 32] {
            return Err(RestartAuthorityEpochIssuanceRecordError::ZeroDigest(name));
        }
    }

    let canonical_contract = AtomicActivationTransactionContractV1::canonical();
    canonical_contract
        .verify()
        .map_err(RestartAuthorityEpochIssuanceRecordError::TransactionContractInvalid)?;
    if record.transaction_contract_digest != canonical_contract.contract_digest().as_bytes() {
        return Err(RestartAuthorityEpochIssuanceRecordError::TransactionContractMismatch);
    }

    let expected_generation = record
        .expected_live_generation
        .checked_add(1)
        .ok_or(RestartAuthorityEpochIssuanceRecordError::GenerationOverflow)?;
    if record.committed_live_generation != expected_generation {
        return Err(RestartAuthorityEpochIssuanceRecordError::CommittedGenerationMismatch {
            expected: expected_generation,
            actual: record.committed_live_generation,
        });
    }
    if record.activated_at_cycle == 0 {
        return Err(RestartAuthorityEpochIssuanceRecordError::InvalidActivationCycle);
    }

    if record.phase_evidence.len() != PHASE_COUNT {
        return Err(RestartAuthorityEpochIssuanceRecordError::PhaseEvidenceCountMismatch {
            expected: PHASE_COUNT,
            actual: record.phase_evidence.len(),
        });
    }
    if record.phase_evidence.len() != canonical_contract.required_phases().len() {
        return Err(RestartAuthorityEpochIssuanceRecordError::ContractPhaseCountMismatch);
    }

    let mut seen = HashSet::with_capacity(record.phase_evidence.len());
    for (actual, expected_phase) in record
        .phase_evidence
        .iter()
        .zip(canonical_contract.required_phases())
    {
        if actual.phase != *expected_phase {
            return Err(RestartAuthorityEpochIssuanceRecordError::PhaseOrderingMismatch);
        }
        if actual.evidence_digest == [0; 32] {
            return Err(RestartAuthorityEpochIssuanceRecordError::ZeroPhaseEvidenceDigest(
                actual.phase,
            ));
        }
        if !seen.insert(actual.evidence_digest) {
            return Err(RestartAuthorityEpochIssuanceRecordError::ReusedPhaseEvidenceDigest);
        }
    }

    Ok(())
}

fn digest_activation_commit(
    record: &RestartAuthorityEpochIssuanceRecordV1,
) -> Result<ActivationCommitReceiptDigestV1, RestartAuthorityEpochIssuanceRecordError> {
    validate_record(record)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-activation-commit-receipt-v1");
    hasher.update(&[1]);
    hash_string(&mut hasher, &record.deployment_id)?;
    hash_string(&mut hasher, &record.trust_domain_id)?;
    hasher.update(&record.activated_restart_v2_digest);
    hasher.update(&record.activation_review_receipt_digest);
    hasher.update(&record.transaction_contract_digest);
    hasher.update(&record.expected_live_generation.to_le_bytes());
    hasher.update(&record.committed_live_generation.to_le_bytes());
    hasher.update(&record.rollback_bundle_digest);
    hasher.update(&record.candidate_bundle_digest);
    hasher.update(&record.installed_state_digest);
    hasher.update(&record.activated_at_cycle.to_le_bytes());
    hasher.update(&(record.phase_evidence.len() as u64).to_le_bytes());
    for evidence in &record.phase_evidence {
        hasher.update(&[phase_tag(evidence.phase)]);
        hasher.update(&evidence.evidence_digest);
    }
    Ok(ActivationCommitReceiptDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_authority_epoch(
    record: &RestartAuthorityEpochIssuanceRecordV1,
) -> Result<RestartAuthorityEpochDigestV1, RestartAuthorityEpochIssuanceRecordError> {
    validate_record(record)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-authority-epoch-v1");
    hasher.update(&[1]);
    hash_string(&mut hasher, &record.deployment_id)?;
    hash_string(&mut hasher, &record.trust_domain_id)?;
    hasher.update(&record.epoch_sequence.to_le_bytes());
    match record.previous_epoch_digest {
        Some(digest) => {
            hasher.update(&[1]);
            hasher.update(&digest.0);
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&record.activated_restart_v2_digest);
    hasher.update(&record.activation_commit_receipt_digest.0);
    hasher.update(&record.committed_live_generation.to_le_bytes());
    hasher.update(&record.activated_at_cycle.to_le_bytes());
    Ok(RestartAuthorityEpochDigestV1(*hasher.finalize().as_bytes()))
}

fn digest_record(
    record: &RestartAuthorityEpochIssuanceRecordV1,
) -> Result<RestartAuthorityEpochIssuanceRecordDigestV1, RestartAuthorityEpochIssuanceRecordError> {
    validate_record(record)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-authority-epoch-issuance-record-v1");
    hasher.update(&[1]);
    hasher.update(&record.activation_commit_receipt_digest.0);
    hasher.update(&record.authority_epoch_digest.0);
    hasher.update(&record.activated_restart_v2_digest);
    hasher.update(&record.committed_live_generation.to_le_bytes());
    hasher.update(&record.activated_at_cycle.to_le_bytes());
    Ok(RestartAuthorityEpochIssuanceRecordDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn validate_id(
    label: &'static str,
    value: &str,
) -> Result<(), RestartAuthorityEpochIssuanceRecordError> {
    if value.is_empty() {
        return Err(RestartAuthorityEpochIssuanceRecordError::EmptyIdentifier(label));
    }
    if value.len() > MAX_ID_BYTES {
        return Err(RestartAuthorityEpochIssuanceRecordError::IdentifierTooLarge {
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
) -> Result<(), RestartAuthorityEpochIssuanceRecordError> {
    let len = u64::try_from(value.len())
        .map_err(|_| RestartAuthorityEpochIssuanceRecordError::LengthOverflow)?;
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
pub enum RestartAuthorityEpochIssuanceRecordError {
    UnsupportedVersion,
    EmptyIdentifier(&'static str),
    IdentifierTooLarge {
        label: &'static str,
        actual: usize,
        maximum: usize,
    },
    InvalidEpochSequence,
    GenesisHasPredecessor,
    MissingPreviousEpoch,
    ZeroDigest(&'static str),
    TransactionContractInvalid(AtomicActivationTransactionContractError),
    TransactionContractMismatch,
    GenerationOverflow,
    CommittedGenerationMismatch { expected: u64, actual: u64 },
    InvalidActivationCycle,
    PhaseEvidenceCountMismatch { expected: usize, actual: usize },
    ContractPhaseCountMismatch,
    PhaseOrderingMismatch,
    ZeroPhaseEvidenceDigest(AtomicActivationPhaseV1),
    ReusedPhaseEvidenceDigest,
    LengthOverflow,
    ActivationCommitDigestMismatch,
    AuthorityEpochDigestMismatch,
    RecordDigestMismatch,
}

impl fmt::Display for RestartAuthorityEpochIssuanceRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart authority epoch issuance record invalid: {self:?}")
    }
}

impl Error for RestartAuthorityEpochIssuanceRecordError {}

#[cfg(test)]
mod tests {
    use super::*;

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
            version: RestartAuthorityEpochIssuanceRecordVersion::V1,
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
        record.activation_commit_receipt_digest = digest_activation_commit(&record).unwrap();
        record.authority_epoch_digest = digest_authority_epoch(&record).unwrap();
        record.record_digest = digest_record(&record).unwrap();
        record
    }

    #[test]
    fn genesis_epoch_uses_no_magic_predecessor_digest() {
        let record = synthetic_record(1);
        record.verify().unwrap();
        assert_eq!(record.previous_epoch_digest(), None);
        assert!(!record.successful_activation_event_independently_verified());
        assert!(!record.epoch_issuance_authorized());
    }

    #[test]
    fn successor_epoch_requires_predecessor_digest() {
        let mut record = synthetic_record(2);
        record.previous_epoch_digest = None;
        assert!(matches!(
            record.verify(),
            Err(RestartAuthorityEpochIssuanceRecordError::MissingPreviousEpoch)
        ));
    }

    #[test]
    fn exact_phase_order_and_unique_evidence_are_required() {
        let mut record = synthetic_record(1);
        record.phase_evidence.swap(0, 1);
        assert!(matches!(
            record.verify(),
            Err(RestartAuthorityEpochIssuanceRecordError::PhaseOrderingMismatch)
        ));

        let mut record = synthetic_record(1);
        record.phase_evidence[1].evidence_digest = record.phase_evidence[0].evidence_digest;
        assert!(matches!(
            record.verify(),
            Err(RestartAuthorityEpochIssuanceRecordError::ReusedPhaseEvidenceDigest)
        ));
    }

    #[test]
    fn commit_and_epoch_digests_are_non_circular_and_independently_checked() {
        let mut record = synthetic_record(2);
        record.verify().unwrap();

        record.committed_live_generation = 42;
        assert!(matches!(
            record.verify(),
            Err(RestartAuthorityEpochIssuanceRecordError::CommittedGenerationMismatch { .. })
        ));
    }
}
