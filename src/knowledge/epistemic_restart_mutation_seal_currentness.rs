// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! External current-head attestation for protected mutation-seal checkpoints.
//!
//! EKM-059 can prove that a deployment provider accepted one exact protected
//! mutation-seal checkpoint, but a generic signature or historical checkpoint is
//! not evidence that the checkpoint is still the deployment's current head.
//! EKM-060 therefore keeps currentness false. This module introduces a separate
//! provider contract whose evidence kinds *specifically* mean monotonic/current
//! state, and binds that attestation to one exact EKM-059 checkpoint.
//!
//! Currentness is time-bounded audit evidence only. It grants no capsule
//! construction, historical replay, writable hydration, trusted-state mutation,
//! or activation authority.

use super::epistemic_restart_mutation_seal_checkpoint::{
    RestartMutationSealCheckpointError, VerifiedRestartMutationSealCheckpointV1,
};
use std::error::Error;
use std::fmt;

pub const MAX_MUTATION_SEAL_CURRENTNESS_AUTHORITY_ID_BYTES: usize = 256;
pub const MAX_MUTATION_SEAL_CURRENTNESS_PROOF_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartMutationSealCurrentnessVersion {
    V1,
}

/// Evidence classes whose contract is explicitly about current/monotonic head
/// state. A generic detached signature is intentionally absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartMutationSealCurrentnessEvidenceKindV1 {
    MonotonicProtectedState,
    HardwareMonotonicCounter,
    TransparencyLogHead,
    CurrentHeadWitnessQuorum,
}

impl RestartMutationSealCurrentnessEvidenceKindV1 {
    fn tag(self) -> u8 {
        match self {
            Self::MonotonicProtectedState => 1,
            Self::HardwareMonotonicCounter => 2,
            Self::TransparencyLogHead => 3,
            Self::CurrentHeadWitnessQuorum => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RestartMutationSealCurrentnessDigestV1([u8; 32]);

impl RestartMutationSealCurrentnessDigestV1 {
    pub fn as_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
        }
        out
    }
}

/// Exact current-head question presented to the deployment verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartMutationSealCurrentnessStatementV1 {
    version: RestartMutationSealCurrentnessVersion,
    deployment_id: String,
    trust_domain_id: String,
    checkpoint_sequence: u64,
    checkpoint_statement_digest: [u8; 32],
    restart_capture_cycle: u64,
    mutation_seal_capsule_digest: [u8; 32],
    checkpoint_verified_at_cycle: u64,
    checkpoint_expires_at_cycle: u64,
    attested_at_cycle: u64,
    expires_at_cycle: u64,
    authority_id: String,
    evidence_kind: RestartMutationSealCurrentnessEvidenceKindV1,
}

impl RestartMutationSealCurrentnessStatementV1 {
    pub fn new(
        checkpoint: &VerifiedRestartMutationSealCheckpointV1,
        attested_at_cycle: u64,
        expires_at_cycle: u64,
        authority_id: impl Into<String>,
        evidence_kind: RestartMutationSealCurrentnessEvidenceKindV1,
    ) -> Result<Self, RestartMutationSealCurrentnessError> {
        checkpoint
            .verify_internal()
            .map_err(RestartMutationSealCurrentnessError::CheckpointRejected)?;
        if !checkpoint.seal_capsule_digest_protected()
            || checkpoint.trusted_state_mutated()
            || checkpoint.capsule_construction_authorized()
            || checkpoint.writable_hydration_authorized()
            || checkpoint.activation_authorized()
        {
            return Err(RestartMutationSealCurrentnessError::UnexpectedCheckpointAuthority);
        }
        if attested_at_cycle < checkpoint.verified_at_cycle() {
            return Err(RestartMutationSealCurrentnessError::AttestationPredatesCheckpoint {
                attested_at_cycle,
                checkpoint_verified_at_cycle: checkpoint.verified_at_cycle(),
            });
        }
        if attested_at_cycle >= checkpoint.statement().expires_at_cycle() {
            return Err(RestartMutationSealCurrentnessError::CheckpointExpiredBeforeAttestation {
                attested_at_cycle,
                checkpoint_expires_at_cycle: checkpoint.statement().expires_at_cycle(),
            });
        }
        if expires_at_cycle <= attested_at_cycle
            || expires_at_cycle > checkpoint.statement().expires_at_cycle()
        {
            return Err(RestartMutationSealCurrentnessError::InvalidValidityWindow {
                attested_at_cycle,
                expires_at_cycle,
                checkpoint_expires_at_cycle: checkpoint.statement().expires_at_cycle(),
            });
        }

        let statement = Self {
            version: RestartMutationSealCurrentnessVersion::V1,
            deployment_id: checkpoint.statement().deployment_id().to_string(),
            trust_domain_id: checkpoint.statement().trust_domain_id().to_string(),
            checkpoint_sequence: checkpoint.statement().sequence(),
            checkpoint_statement_digest: checkpoint.statement_digest().as_bytes(),
            restart_capture_cycle: checkpoint.statement().restart_captured_at_cycle(),
            mutation_seal_capsule_digest: checkpoint.statement().mutation_seal_capsule_digest(),
            checkpoint_verified_at_cycle: checkpoint.verified_at_cycle(),
            checkpoint_expires_at_cycle: checkpoint.statement().expires_at_cycle(),
            attested_at_cycle,
            expires_at_cycle,
            authority_id: authority_id.into(),
            evidence_kind,
        };
        statement.validate_shape()?;
        Ok(statement)
    }

    pub fn version(&self) -> RestartMutationSealCurrentnessVersion {
        self.version
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn trust_domain_id(&self) -> &str {
        &self.trust_domain_id
    }

    pub fn checkpoint_sequence(&self) -> u64 {
        self.checkpoint_sequence
    }

    pub fn checkpoint_statement_digest(&self) -> [u8; 32] {
        self.checkpoint_statement_digest
    }

    pub fn restart_capture_cycle(&self) -> u64 {
        self.restart_capture_cycle
    }

    pub fn mutation_seal_capsule_digest(&self) -> [u8; 32] {
        self.mutation_seal_capsule_digest
    }

    pub fn checkpoint_verified_at_cycle(&self) -> u64 {
        self.checkpoint_verified_at_cycle
    }

    pub fn checkpoint_expires_at_cycle(&self) -> u64 {
        self.checkpoint_expires_at_cycle
    }

    pub fn attested_at_cycle(&self) -> u64 {
        self.attested_at_cycle
    }

    pub fn expires_at_cycle(&self) -> u64 {
        self.expires_at_cycle
    }

    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }

    pub fn evidence_kind(&self) -> RestartMutationSealCurrentnessEvidenceKindV1 {
        self.evidence_kind
    }

    fn validate_shape(&self) -> Result<(), RestartMutationSealCurrentnessError> {
        if self.checkpoint_sequence == 0 {
            return Err(RestartMutationSealCurrentnessError::InvalidCheckpointSequence);
        }
        validate_identifier(&self.deployment_id)?;
        validate_identifier(&self.trust_domain_id)?;
        if self.authority_id.trim().is_empty()
            || self.authority_id != self.authority_id.trim()
            || self.authority_id.len() > MAX_MUTATION_SEAL_CURRENTNESS_AUTHORITY_ID_BYTES
        {
            return Err(RestartMutationSealCurrentnessError::InvalidAuthorityId);
        }
        if self.attested_at_cycle < self.checkpoint_verified_at_cycle {
            return Err(RestartMutationSealCurrentnessError::AttestationPredatesCheckpoint {
                attested_at_cycle: self.attested_at_cycle,
                checkpoint_verified_at_cycle: self.checkpoint_verified_at_cycle,
            });
        }
        if self.attested_at_cycle >= self.checkpoint_expires_at_cycle {
            return Err(RestartMutationSealCurrentnessError::CheckpointExpiredBeforeAttestation {
                attested_at_cycle: self.attested_at_cycle,
                checkpoint_expires_at_cycle: self.checkpoint_expires_at_cycle,
            });
        }
        if self.expires_at_cycle <= self.attested_at_cycle
            || self.expires_at_cycle > self.checkpoint_expires_at_cycle
        {
            return Err(RestartMutationSealCurrentnessError::InvalidValidityWindow {
                attested_at_cycle: self.attested_at_cycle,
                expires_at_cycle: self.expires_at_cycle,
                checkpoint_expires_at_cycle: self.checkpoint_expires_at_cycle,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartMutationSealCurrentnessEvidenceV1 {
    statement: RestartMutationSealCurrentnessStatementV1,
    proof: Vec<u8>,
}

impl RestartMutationSealCurrentnessEvidenceV1 {
    pub fn new(
        statement: RestartMutationSealCurrentnessStatementV1,
        proof: Vec<u8>,
    ) -> Result<Self, RestartMutationSealCurrentnessError> {
        statement.validate_shape()?;
        if proof.is_empty() {
            return Err(RestartMutationSealCurrentnessError::EmptyProof);
        }
        if proof.len() > MAX_MUTATION_SEAL_CURRENTNESS_PROOF_BYTES {
            return Err(RestartMutationSealCurrentnessError::ProofTooLarge {
                actual: proof.len(),
                maximum: MAX_MUTATION_SEAL_CURRENTNESS_PROOF_BYTES,
            });
        }
        Ok(Self { statement, proof })
    }

    pub fn statement(&self) -> &RestartMutationSealCurrentnessStatementV1 {
        &self.statement
    }

    pub fn proof(&self) -> &[u8] {
        &self.proof
    }
}

/// Provider contract: returning true means the provider attests that the exact
/// checkpoint in the statement is the current monotonic/head state under the
/// selected evidence kind at `attested_at_cycle`.
pub trait RestartMutationSealCurrentnessVerifierV1 {
    fn verify_current_restart_mutation_seal_checkpoint(
        &self,
        evidence_kind: RestartMutationSealCurrentnessEvidenceKindV1,
        authority_id: &str,
        statement_digest: [u8; 32],
        proof: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRestartMutationSealCurrentnessV1 {
    statement: RestartMutationSealCurrentnessStatementV1,
    statement_digest: RestartMutationSealCurrentnessDigestV1,
    proof_digest: [u8; 32],
    verified_at_cycle: u64,
    current_head_proven: bool,
    trusted_state_mutated: bool,
    historical_replay_authorized: bool,
    writable_hydration_authorized: bool,
    activation_authorized: bool,
}

impl VerifiedRestartMutationSealCurrentnessV1 {
    pub fn statement(&self) -> &RestartMutationSealCurrentnessStatementV1 {
        &self.statement
    }

    pub fn statement_digest(&self) -> RestartMutationSealCurrentnessDigestV1 {
        self.statement_digest
    }

    pub fn proof_digest(&self) -> [u8; 32] {
        self.proof_digest
    }

    pub fn verified_at_cycle(&self) -> u64 {
        self.verified_at_cycle
    }

    pub fn current_head_proven(&self) -> bool {
        self.current_head_proven
    }

    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }

    pub fn historical_replay_authorized(&self) -> bool {
        self.historical_replay_authorized
    }

    pub fn writable_hydration_authorized(&self) -> bool {
        self.writable_hydration_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }

    pub fn verify_internal(&self) -> Result<(), RestartMutationSealCurrentnessError> {
        self.statement.validate_shape()?;
        if digest_restart_mutation_seal_currentness_statement(&self.statement)?
            != self.statement_digest
        {
            return Err(RestartMutationSealCurrentnessError::StatementDigestMismatch);
        }
        if !self.current_head_proven
            || self.trusted_state_mutated
            || self.historical_replay_authorized
            || self.writable_hydration_authorized
            || self.activation_authorized
        {
            return Err(RestartMutationSealCurrentnessError::UnexpectedAuthority);
        }
        Ok(())
    }
}

pub fn verify_restart_mutation_seal_currentness(
    evidence: &RestartMutationSealCurrentnessEvidenceV1,
    checkpoint: &VerifiedRestartMutationSealCheckpointV1,
    observed_at_cycle: u64,
    verifier: &dyn RestartMutationSealCurrentnessVerifierV1,
) -> Result<VerifiedRestartMutationSealCurrentnessV1, RestartMutationSealCurrentnessError> {
    checkpoint
        .verify_internal()
        .map_err(RestartMutationSealCurrentnessError::CheckpointRejected)?;
    evidence.statement.validate_shape()?;
    validate_statement_binding(&evidence.statement, checkpoint)?;

    if observed_at_cycle < evidence.statement.attested_at_cycle {
        return Err(RestartMutationSealCurrentnessError::ObservationPredatesAttestation {
            observed_at_cycle,
            attested_at_cycle: evidence.statement.attested_at_cycle,
        });
    }
    if observed_at_cycle >= evidence.statement.expires_at_cycle {
        return Err(RestartMutationSealCurrentnessError::CurrentnessEvidenceExpired {
            observed_at_cycle,
            expires_at_cycle: evidence.statement.expires_at_cycle,
        });
    }

    let statement_digest =
        digest_restart_mutation_seal_currentness_statement(&evidence.statement)?;
    let accepted = verifier
        .verify_current_restart_mutation_seal_checkpoint(
            evidence.statement.evidence_kind,
            &evidence.statement.authority_id,
            statement_digest.as_bytes(),
            &evidence.proof,
        )
        .map_err(RestartMutationSealCurrentnessError::VerificationProvider)?;
    if !accepted {
        return Err(RestartMutationSealCurrentnessError::ProofRejected);
    }

    let mut proof_hasher = blake3::Hasher::new();
    proof_hasher.update(b"symthaea-ekm-restart-mutation-seal-currentness-proof-v1");
    proof_hasher.update(&evidence.proof);
    let receipt = VerifiedRestartMutationSealCurrentnessV1 {
        statement: evidence.statement.clone(),
        statement_digest,
        proof_digest: *proof_hasher.finalize().as_bytes(),
        verified_at_cycle: observed_at_cycle,
        current_head_proven: true,
        trusted_state_mutated: false,
        historical_replay_authorized: false,
        writable_hydration_authorized: false,
        activation_authorized: false,
    };
    receipt.verify_internal()?;
    Ok(receipt)
}

pub fn digest_restart_mutation_seal_currentness_statement(
    statement: &RestartMutationSealCurrentnessStatementV1,
) -> Result<RestartMutationSealCurrentnessDigestV1, RestartMutationSealCurrentnessError> {
    statement.validate_shape()?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ekm-restart-mutation-seal-currentness-v1");
    hasher.update(&[1]);
    hash_bytes(&mut hasher, statement.deployment_id.as_bytes())?;
    hash_bytes(&mut hasher, statement.trust_domain_id.as_bytes())?;
    hasher.update(&statement.checkpoint_sequence.to_le_bytes());
    hasher.update(&statement.checkpoint_statement_digest);
    hasher.update(&statement.restart_capture_cycle.to_le_bytes());
    hasher.update(&statement.mutation_seal_capsule_digest);
    hasher.update(&statement.checkpoint_verified_at_cycle.to_le_bytes());
    hasher.update(&statement.checkpoint_expires_at_cycle.to_le_bytes());
    hasher.update(&statement.attested_at_cycle.to_le_bytes());
    hasher.update(&statement.expires_at_cycle.to_le_bytes());
    hash_bytes(&mut hasher, statement.authority_id.as_bytes())?;
    hasher.update(&[statement.evidence_kind.tag()]);
    Ok(RestartMutationSealCurrentnessDigestV1(
        *hasher.finalize().as_bytes(),
    ))
}

fn validate_statement_binding(
    statement: &RestartMutationSealCurrentnessStatementV1,
    checkpoint: &VerifiedRestartMutationSealCheckpointV1,
) -> Result<(), RestartMutationSealCurrentnessError> {
    if statement.deployment_id != checkpoint.statement().deployment_id()
        || statement.trust_domain_id != checkpoint.statement().trust_domain_id()
        || statement.checkpoint_sequence != checkpoint.statement().sequence()
        || statement.checkpoint_statement_digest != checkpoint.statement_digest().as_bytes()
        || statement.restart_capture_cycle != checkpoint.statement().restart_captured_at_cycle()
        || statement.mutation_seal_capsule_digest
            != checkpoint.statement().mutation_seal_capsule_digest()
        || statement.checkpoint_verified_at_cycle != checkpoint.verified_at_cycle()
        || statement.checkpoint_expires_at_cycle != checkpoint.statement().expires_at_cycle()
    {
        return Err(RestartMutationSealCurrentnessError::CheckpointBindingMismatch);
    }
    if !checkpoint.seal_capsule_digest_protected()
        || checkpoint.trusted_state_mutated()
        || checkpoint.capsule_construction_authorized()
        || checkpoint.writable_hydration_authorized()
        || checkpoint.activation_authorized()
    {
        return Err(RestartMutationSealCurrentnessError::UnexpectedCheckpointAuthority);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartMutationSealCurrentnessError {
    CheckpointRejected(RestartMutationSealCheckpointError),
    InvalidCheckpointSequence,
    InvalidIdentifier,
    InvalidAuthorityId,
    AttestationPredatesCheckpoint {
        attested_at_cycle: u64,
        checkpoint_verified_at_cycle: u64,
    },
    CheckpointExpiredBeforeAttestation {
        attested_at_cycle: u64,
        checkpoint_expires_at_cycle: u64,
    },
    InvalidValidityWindow {
        attested_at_cycle: u64,
        expires_at_cycle: u64,
        checkpoint_expires_at_cycle: u64,
    },
    EmptyProof,
    ProofTooLarge {
        actual: usize,
        maximum: usize,
    },
    CheckpointBindingMismatch,
    UnexpectedCheckpointAuthority,
    ObservationPredatesAttestation {
        observed_at_cycle: u64,
        attested_at_cycle: u64,
    },
    CurrentnessEvidenceExpired {
        observed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    VerificationProvider(String),
    ProofRejected,
    StatementDigestMismatch,
    UnexpectedAuthority,
    LengthOverflow,
}

impl fmt::Display for RestartMutationSealCurrentnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart mutation-seal currentness rejected: {self:?}")
    }
}

impl Error for RestartMutationSealCurrentnessError {}

fn validate_identifier(value: &str) -> Result<(), RestartMutationSealCurrentnessError> {
    if value.trim().is_empty() || value != value.trim() || value.len() > 256 {
        return Err(RestartMutationSealCurrentnessError::InvalidIdentifier);
    }
    Ok(())
}

fn hash_bytes(
    hasher: &mut blake3::Hasher,
    bytes: &[u8],
) -> Result<(), RestartMutationSealCurrentnessError> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| RestartMutationSealCurrentnessError::LengthOverflow)?;
    hasher.update(&len.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}
