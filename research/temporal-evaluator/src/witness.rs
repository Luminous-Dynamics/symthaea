//! Signed ledger checkpoints and independent witness-quorum verification.
//!
//! This module does not implement networking or a TPM backend. It defines the
//! canonical checkpoint/attestation encodings and the strict verifier that a
//! deployment adapter must call before accepting a restored evaluator ledger.
//! A local hash chain alone cannot detect rollback of both the chain and its
//! head. A quorum of independent durable witnesses can detect that rollback
//! only while the witness state remains outside the evaluator's rollback domain.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

const CHECKPOINT_DOMAIN: &[u8] = b"SYMTHAEA-LEDGER-CHECKPOINT-V1\0";
const WITNESS_DOMAIN: &[u8] = b"SYMTHAEA-CHECKPOINT-WITNESS-V1\0";
const SIGNATURE_BYTES: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerCheckpointPayload {
    pub schema_version: u16,
    pub evaluator_id: [u8; 16],
    pub evaluator_key_id: [u8; 16],
    pub epoch: u64,
    pub highest_receipt_sequence: u64,
    pub highest_monotonic_counter: u64,
    pub receipt_count: u64,
    pub previous_checkpoint_sha256: [u8; 32],
    pub receipt_chain_head_sha256: [u8; 32],
    pub issued_at_unix_ns: u64,
}

impl LedgerCheckpointPayload {
    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.schema_version != 1
            || self.evaluator_id == [0; 16]
            || self.epoch == 0
            || self.highest_receipt_sequence == 0
            || self.highest_monotonic_counter == 0
            || self.receipt_count == 0
            || self.receipt_chain_head_sha256 == [0; 32]
            || self.issued_at_unix_ns == 0
        {
            return Err(CheckpointError::InvalidPayload);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CheckpointError> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(200);
        bytes.extend_from_slice(CHECKPOINT_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_le_bytes());
        bytes.extend_from_slice(&self.evaluator_id);
        bytes.extend_from_slice(&self.evaluator_key_id);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.highest_receipt_sequence.to_le_bytes());
        bytes.extend_from_slice(&self.highest_monotonic_counter.to_le_bytes());
        bytes.extend_from_slice(&self.receipt_count.to_le_bytes());
        bytes.extend_from_slice(&self.previous_checkpoint_sha256);
        bytes.extend_from_slice(&self.receipt_chain_head_sha256);
        bytes.extend_from_slice(&self.issued_at_unix_ns.to_le_bytes());
        Ok(bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedLedgerCheckpoint {
    pub payload: LedgerCheckpointPayload,
    pub signature: [u8; SIGNATURE_BYTES],
}

impl SignedLedgerCheckpoint {
    pub fn canonical_envelope(&self) -> Result<Vec<u8>, CheckpointError> {
        let mut bytes = self.payload.canonical_bytes()?;
        bytes.extend_from_slice(&self.signature);
        Ok(bytes)
    }

    pub fn digest(&self) -> Result<[u8; 32], CheckpointError> {
        Ok(Sha256::digest(self.canonical_envelope()?).into())
    }
}

pub fn sign_checkpoint(
    mut payload: LedgerCheckpointPayload,
    signing_key: &SigningKey,
) -> Result<SignedLedgerCheckpoint, CheckpointError> {
    payload.evaluator_key_id = public_key_id(&signing_key.verifying_key());
    let message = payload.canonical_bytes()?;
    let signature = signing_key.sign(&message).to_bytes();
    Ok(SignedLedgerCheckpoint { payload, signature })
}

pub fn verify_checkpoint(
    checkpoint: &SignedLedgerCheckpoint,
    evaluator_key: &VerifyingKey,
) -> Result<[u8; 32], CheckpointError> {
    if checkpoint.payload.evaluator_key_id != public_key_id(evaluator_key) {
        return Err(CheckpointError::EvaluatorKeyIdMismatch);
    }
    let message = checkpoint.payload.canonical_bytes()?;
    let signature = Signature::from_bytes(&checkpoint.signature);
    evaluator_key
        .verify_strict(&message, &signature)
        .map_err(|_| CheckpointError::CheckpointSignatureInvalid)?;
    checkpoint.digest()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessAttestationPayload {
    pub witness_id: [u8; 16],
    pub evaluator_id: [u8; 16],
    pub checkpoint_epoch: u64,
    pub checkpoint_sha256: [u8; 32],
    pub previous_checkpoint_sha256: [u8; 32],
    pub observed_at_unix_ns: u64,
}

impl WitnessAttestationPayload {
    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.witness_id == [0; 16]
            || self.evaluator_id == [0; 16]
            || self.checkpoint_epoch == 0
            || self.checkpoint_sha256 == [0; 32]
            || self.observed_at_unix_ns == 0
        {
            return Err(CheckpointError::InvalidAttestation);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CheckpointError> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(150);
        bytes.extend_from_slice(WITNESS_DOMAIN);
        bytes.extend_from_slice(&self.witness_id);
        bytes.extend_from_slice(&self.evaluator_id);
        bytes.extend_from_slice(&self.checkpoint_epoch.to_le_bytes());
        bytes.extend_from_slice(&self.checkpoint_sha256);
        bytes.extend_from_slice(&self.previous_checkpoint_sha256);
        bytes.extend_from_slice(&self.observed_at_unix_ns.to_le_bytes());
        Ok(bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedWitnessAttestation {
    pub payload: WitnessAttestationPayload,
    pub signature: [u8; SIGNATURE_BYTES],
}

pub fn sign_witness_attestation(
    checkpoint: &SignedLedgerCheckpoint,
    evaluator_id: [u8; 16],
    observed_at_unix_ns: u64,
    witness_key: &SigningKey,
) -> Result<SignedWitnessAttestation, CheckpointError> {
    let payload = WitnessAttestationPayload {
        witness_id: public_key_id(&witness_key.verifying_key()),
        evaluator_id,
        checkpoint_epoch: checkpoint.payload.epoch,
        checkpoint_sha256: checkpoint.digest()?,
        previous_checkpoint_sha256: checkpoint.payload.previous_checkpoint_sha256,
        observed_at_unix_ns,
    };
    let message = payload.canonical_bytes()?;
    let signature = witness_key.sign(&message).to_bytes();
    Ok(SignedWitnessAttestation { payload, signature })
}

#[derive(Clone)]
pub struct TrustedWitness {
    pub witness_id: [u8; 16],
    pub verifying_key: VerifyingKey,
}

impl TrustedWitness {
    pub fn from_key(verifying_key: VerifyingKey) -> Self {
        Self {
            witness_id: public_key_id(&verifying_key),
            verifying_key,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WitnessQuorumPolicy {
    pub total_witnesses: u16,
    pub threshold: u16,
}

impl WitnessQuorumPolicy {
    pub fn validate(self) -> Result<(), CheckpointError> {
        if self.total_witnesses == 0
            || self.threshold == 0
            || self.threshold > self.total_witnesses
            || self.threshold <= self.total_witnesses / 2
        {
            return Err(CheckpointError::InvalidQuorumPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessQuorumDecision {
    pub evaluator_id: [u8; 16],
    pub checkpoint_epoch: u64,
    pub checkpoint_sha256: [u8; 32],
    pub confirming_witnesses: u16,
    pub stale_witnesses: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointError {
    InvalidPayload,
    InvalidAttestation,
    InvalidQuorumPolicy,
    DuplicateTrustedWitness,
    UnknownWitness,
    DuplicateWitnessResponse,
    WitnessSignatureInvalid,
    EvaluatorKeyIdMismatch,
    CheckpointSignatureInvalid,
    LocalLedgerBehindWitness { local_epoch: u64, witnessed_epoch: u64 },
    WitnessesBehindLocal { local_epoch: u64, highest_witness_epoch: u64 },
    WitnessCheckpointConflict { epoch: u64 },
    InsufficientWitnessQuorum { required: u16, matching: u16, stale: u16 },
    ArithmeticOverflow,
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPayload => write!(f, "ledger checkpoint payload is invalid"),
            Self::InvalidAttestation => write!(f, "witness attestation payload is invalid"),
            Self::InvalidQuorumPolicy => write!(f, "witness quorum must be a strict majority of the configured witness set"),
            Self::DuplicateTrustedWitness => write!(f, "trusted witness registry contains duplicate identities"),
            Self::UnknownWitness => write!(f, "attestation is from an untrusted witness"),
            Self::DuplicateWitnessResponse => write!(f, "multiple responses were supplied for one witness"),
            Self::WitnessSignatureInvalid => write!(f, "witness attestation signature is invalid"),
            Self::EvaluatorKeyIdMismatch => write!(f, "checkpoint evaluator key ID does not match trusted key"),
            Self::CheckpointSignatureInvalid => write!(f, "checkpoint Ed25519 signature is invalid"),
            Self::LocalLedgerBehindWitness { local_epoch, witnessed_epoch } => write!(f, "local ledger epoch {local_epoch} is behind witnessed epoch {witnessed_epoch}; rollback or stale restore detected"),
            Self::WitnessesBehindLocal { local_epoch, highest_witness_epoch } => write!(f, "local ledger epoch {local_epoch} is ahead of highest witness epoch {highest_witness_epoch}; witness state unavailable or stale"),
            Self::WitnessCheckpointConflict { epoch } => write!(f, "trusted witnesses disagree on checkpoint digest at epoch {epoch}"),
            Self::InsufficientWitnessQuorum { required, matching, stale } => write!(f, "witness quorum unavailable: required {required}, matching {matching}, stale {stale}"),
            Self::ArithmeticOverflow => write!(f, "witness checkpoint arithmetic overflowed"),
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Verify a locally stored checkpoint against independent witness attestations.
///
/// The witness set and public keys must be provisioned outside the evaluator
/// rollback domain. A strict-majority threshold is mandatory. A valid
/// attestation claiming a newer epoch blocks local acceptance; equal-epoch
/// disagreement fails closed. Older witness responses are counted as stale,
/// and a current checkpoint is confirmed only when the threshold attests the
/// exact same signed checkpoint digest and previous-checkpoint hash.
pub fn verify_checkpoint_quorum(
    local: &SignedLedgerCheckpoint,
    evaluator_key: &VerifyingKey,
    attestations: &[SignedWitnessAttestation],
    trusted_witnesses: &[TrustedWitness],
    policy: WitnessQuorumPolicy,
) -> Result<WitnessQuorumDecision, CheckpointError> {
    policy.validate()?;
    if trusted_witnesses.len() != usize::from(policy.total_witnesses) {
        return Err(CheckpointError::InvalidQuorumPolicy);
    }
    let local_digest = verify_checkpoint(local, evaluator_key)?;
    let mut trusted = BTreeMap::new();
    for witness in trusted_witnesses {
        if witness.witness_id != public_key_id(&witness.verifying_key)
            || trusted.insert(witness.witness_id, witness.verifying_key).is_some()
        {
            return Err(CheckpointError::DuplicateTrustedWitness);
        }
    }

    let mut seen = BTreeSet::new();
    let mut current_epoch_matching = 0_u16;
    let mut stale_witnesses = 0_u16;
    let mut highest_witness_epoch = 0_u64;

    for attestation in attestations {
        let payload = &attestation.payload;
        let trusted_key = trusted.get(&payload.witness_id).ok_or(CheckpointError::UnknownWitness)?;
        if !seen.insert(payload.witness_id) {
            return Err(CheckpointError::DuplicateWitnessResponse);
        }
        let message = payload.canonical_bytes()?;
        let signature = Signature::from_bytes(&attestation.signature);
        trusted_key
            .verify_strict(&message, &signature)
            .map_err(|_| CheckpointError::WitnessSignatureInvalid)?;

        if payload.evaluator_id != local.payload.evaluator_id {
            continue;
        }
        highest_witness_epoch = highest_witness_epoch.max(payload.checkpoint_epoch);
        if payload.checkpoint_epoch > local.payload.epoch {
            return Err(CheckpointError::LocalLedgerBehindWitness {
                local_epoch: local.payload.epoch,
                witnessed_epoch: payload.checkpoint_epoch,
            });
        }
        if payload.checkpoint_epoch < local.payload.epoch {
            stale_witnesses = stale_witnesses
                .checked_add(1)
                .ok_or(CheckpointError::ArithmeticOverflow)?;
            continue;
        }
        if payload.checkpoint_sha256 != local_digest
            || payload.previous_checkpoint_sha256 != local.payload.previous_checkpoint_sha256
        {
            return Err(CheckpointError::WitnessCheckpointConflict {
                epoch: local.payload.epoch,
            });
        }
        current_epoch_matching = current_epoch_matching
            .checked_add(1)
            .ok_or(CheckpointError::ArithmeticOverflow)?;
    }

    if highest_witness_epoch < local.payload.epoch {
        return Err(CheckpointError::WitnessesBehindLocal {
            local_epoch: local.payload.epoch,
            highest_witness_epoch,
        });
    }
    if current_epoch_matching < policy.threshold {
        return Err(CheckpointError::InsufficientWitnessQuorum {
            required: policy.threshold,
            matching: current_epoch_matching,
            stale: stale_witnesses,
        });
    }

    Ok(WitnessQuorumDecision {
        evaluator_id: local.payload.evaluator_id,
        checkpoint_epoch: local.payload.epoch,
        checkpoint_sha256: local_digest,
        confirming_witnesses: current_epoch_matching,
        stale_witnesses,
    })
}

fn public_key_id(key: &VerifyingKey) -> [u8; 16] {
    let digest = Sha256::digest(key.to_bytes());
    let mut id = [0_u8; 16];
    id.copy_from_slice(&digest[..16]);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(key: &SigningKey, epoch: u64, previous: [u8; 32]) -> SignedLedgerCheckpoint {
        sign_checkpoint(
            LedgerCheckpointPayload {
                schema_version: 1,
                evaluator_id: [1; 16],
                evaluator_key_id: [0; 16],
                epoch,
                highest_receipt_sequence: epoch * 10,
                highest_monotonic_counter: epoch * 10,
                receipt_count: epoch * 10,
                previous_checkpoint_sha256: previous,
                receipt_chain_head_sha256: [u8::try_from(epoch).unwrap(); 32],
                issued_at_unix_ns: 1_800_000_000_000_000_000 + epoch,
            },
            key,
        ).unwrap()
    }

    fn attest(
        checkpoint: &SignedLedgerCheckpoint,
        witness_key: &SigningKey,
        time: u64,
    ) -> SignedWitnessAttestation {
        sign_witness_attestation(checkpoint, [1; 16], time, witness_key).unwrap()
    }

    fn witnesses(keys: &[SigningKey]) -> Vec<TrustedWitness> {
        keys.iter().map(|key| TrustedWitness::from_key(key.verifying_key())).collect()
    }

    #[test]
    fn strict_majority_confirms_exact_current_checkpoint() {
        let evaluator_key = SigningKey::from_bytes(&[1; 32]);
        let witness_keys = [
            SigningKey::from_bytes(&[2; 32]),
            SigningKey::from_bytes(&[3; 32]),
            SigningKey::from_bytes(&[4; 32]),
        ];
        let checkpoint = checkpoint(&evaluator_key, 3, [9; 32]);
        let attestations = vec![
            attest(&checkpoint, &witness_keys[0], 10),
            attest(&checkpoint, &witness_keys[1], 11),
        ];
        let result = verify_checkpoint_quorum(
            &checkpoint,
            &evaluator_key.verifying_key(),
            &attestations,
            &witnesses(&witness_keys),
            WitnessQuorumPolicy { total_witnesses: 3, threshold: 2 },
        ).unwrap();
        assert_eq!(result.confirming_witnesses, 2);
        assert_eq!(result.stale_witnesses, 0);
    }

    #[test]
    fn older_witnesses_are_reported_but_current_majority_can_confirm() {
        let evaluator_key = SigningKey::from_bytes(&[5; 32]);
        let witness_keys = [
            SigningKey::from_bytes(&[6; 32]),
            SigningKey::from_bytes(&[7; 32]),
            SigningKey::from_bytes(&[8; 32]),
        ];
        let current = checkpoint(&evaluator_key, 4, [9; 32]);
        let old = checkpoint(&evaluator_key, 3, [7; 32]);
        let attestations = vec![
            attest(&current, &witness_keys[0], 20),
            attest(&current, &witness_keys[1], 21),
            attest(&old, &witness_keys[2], 22),
        ];
        let result = verify_checkpoint_quorum(
            &current,
            &evaluator_key.verifying_key(),
            &attestations,
            &witnesses(&witness_keys),
            WitnessQuorumPolicy { total_witnesses: 3, threshold: 2 },
        ).unwrap();
        assert_eq!(result.confirming_witnesses, 2);
        assert_eq!(result.stale_witnesses, 1);
    }

    #[test]
    fn older_local_snapshot_is_rejected_when_witness_has_newer_epoch() {
        let evaluator_key = SigningKey::from_bytes(&[9; 32]);
        let witness_keys = [
            SigningKey::from_bytes(&[10; 32]),
            SigningKey::from_bytes(&[11; 32]),
            SigningKey::from_bytes(&[12; 32]),
        ];
        let local = checkpoint(&evaluator_key, 2, [1; 32]);
        let newer = checkpoint(&evaluator_key, 3, [2; 32]);
        let attestations = vec![
            attest(&newer, &witness_keys[0], 30),
            attest(&newer, &witness_keys[1], 31),
        ];
        assert_eq!(
            verify_checkpoint_quorum(
                &local,
                &evaluator_key.verifying_key(),
                &attestations,
                &witnesses(&witness_keys),
                WitnessQuorumPolicy { total_witnesses: 3, threshold: 2 },
            ),
            Err(CheckpointError::LocalLedgerBehindWitness { local_epoch: 2, witnessed_epoch: 3 })
        );
    }

    #[test]
    fn equal_epoch_conflicting_checkpoint_fails_closed() {
        let evaluator_key = SigningKey::from_bytes(&[13; 32]);
        let witness_keys = [
            SigningKey::from_bytes(&[14; 32]),
            SigningKey::from_bytes(&[15; 32]),
            SigningKey::from_bytes(&[16; 32]),
        ];
        let local = checkpoint(&evaluator_key, 4, [1; 32]);
        let conflicting = checkpoint(&evaluator_key, 4, [2; 32]);
        let attestations = vec![
            attest(&local, &witness_keys[0], 40),
            attest(&conflicting, &witness_keys[1], 41),
        ];
        assert_eq!(
            verify_checkpoint_quorum(
                &local,
                &evaluator_key.verifying_key(),
                &attestations,
                &witnesses(&witness_keys),
                WitnessQuorumPolicy { total_witnesses: 3, threshold: 2 },
            ),
            Err(CheckpointError::WitnessCheckpointConflict { epoch: 4 })
        );
    }

    #[test]
    fn missing_quorum_and_non_majority_policy_are_rejected() {
        let evaluator_key = SigningKey::from_bytes(&[17; 32]);
        let witness_keys = [
            SigningKey::from_bytes(&[18; 32]),
            SigningKey::from_bytes(&[19; 32]),
            SigningKey::from_bytes(&[20; 32]),
        ];
        let checkpoint = checkpoint(&evaluator_key, 2, [9; 32]);
        let one_attestation = vec![attest(&checkpoint, &witness_keys[0], 50)];
        assert_eq!(
            verify_checkpoint_quorum(
                &checkpoint,
                &evaluator_key.verifying_key(),
                &one_attestation,
                &witnesses(&witness_keys),
                WitnessQuorumPolicy { total_witnesses: 3, threshold: 2 },
            ),
            Err(CheckpointError::InsufficientWitnessQuorum { required: 2, matching: 1, stale: 0 })
        );
        assert_eq!(
            verify_checkpoint_quorum(
                &checkpoint,
                &evaluator_key.verifying_key(),
                &one_attestation,
                &witnesses(&witness_keys),
                WitnessQuorumPolicy { total_witnesses: 3, threshold: 1 },
            ),
            Err(CheckpointError::InvalidQuorumPolicy)
        );
    }

    #[test]
    fn duplicate_unknown_and_invalid_witness_responses_are_rejected() {
        let evaluator_key = SigningKey::from_bytes(&[21; 32]);
        let witness_keys = [
            SigningKey::from_bytes(&[22; 32]),
            SigningKey::from_bytes(&[23; 32]),
            SigningKey::from_bytes(&[24; 32]),
        ];
        let checkpoint = checkpoint(&evaluator_key, 2, [9; 32]);
        let duplicate = vec![
            attest(&checkpoint, &witness_keys[0], 60),
            attest(&checkpoint, &witness_keys[0], 61),
        ];
        assert_eq!(
            verify_checkpoint_quorum(
                &checkpoint,
                &evaluator_key.verifying_key(),
                &duplicate,
                &witnesses(&witness_keys),
                WitnessQuorumPolicy { total_witnesses: 3, threshold: 2 },
            ),
            Err(CheckpointError::DuplicateWitnessResponse)
        );
        let unknown = [SigningKey::from_bytes(&[25; 32])];
        let unknown_response = vec![attest(&checkpoint, &unknown[0], 62)];
        assert_eq!(
            verify_checkpoint_quorum(
                &checkpoint,
                &evaluator_key.verifying_key(),
                &unknown_response,
                &witnesses(&witness_keys),
                WitnessQuorumPolicy { total_witnesses: 3, threshold: 2 },
            ),
            Err(CheckpointError::UnknownWitness)
        );
    }
}
