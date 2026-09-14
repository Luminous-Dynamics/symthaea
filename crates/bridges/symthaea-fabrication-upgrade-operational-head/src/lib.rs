// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Witnessed, exact-evidence currentness for fabrication upgrade operational state.
//!
//! This bridge deliberately proves a bounded distributed theorem: one exact
//! `FabricationUpgradeOperationalState` is the latest publication for its exact handoff inside one
//! exact append-only transparency-log view, and that view is covered by an interval-valid signed
//! checkpoint plus an independently verified witness quorum whose identity metadata matches one
//! exact threshold-authorized witness registry. The exact raw checkpoint and witness signature
//! bytes are independently reverified before the opaque head capability is minted.
//!
//! The theorem is intentionally not omniscient: it does not claim that no later checkpoint exists
//! outside the supplied witnessed view, nor that the supplied witness registry is globally newest.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_kernel::transparency_checkpoint::{
    MAX_TRANSPARENCY_CHECKPOINT_KEY_ID_BYTES, MAX_TRANSPARENCY_CHECKPOINT_SIGNATURE_BYTES,
    SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA, SignedTransparencyCheckpoint,
    VerifiedTransparencyCheckpoint, digest_transparency_checkpoint,
};
use symthaea_fabrication_kernel::transparency_witness::{
    MAX_TRANSPARENCY_WITNESSES, SIGNED_TRANSPARENCY_WITNESS_SCHEMA,
    TRANSPARENCY_WITNESS_SCHEMA, SignedTransparencyWitness, TransparencyWitnessPolicy,
    VerifiedTransparencyWitnessQuorum, digest_transparency_witness_statement,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_kernel::upgrade_operational_state::{
    FabricationUpgradeOperationalState, digest_upgrade_operational_state,
};
use symthaea_fabrication_witness_authority::{
    WitnessAuthorityRegistryIdV1, WitnessAuthorityRegistryV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const UPGRADE_OPERATIONAL_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.upgrade-operational-head-publication.v1";
pub const QUORUM_OBSERVED_UPGRADE_OPERATIONAL_HEAD_SCHEMA: &str =
    "symthaea.fabrication.quorum-observed-upgrade-operational-head.v1";
pub const UPGRADE_OPERATIONAL_HEAD_LOG_KIND_PREFIX: &str = "upgrade-operational-head-v1:";
pub const MAX_OPERATIONAL_HEAD_EXACT_VERIFIERS: usize = 16;

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-operational-head-publication.v1\0";
const LOG_KIND_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-operational-head-kind.v1\0";
const SIGNED_CHECKPOINT_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-operational-head-checkpoint-evidence.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-operational-head-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-operational-head-witness-set.v1\0";
const WITNESS_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-operational-head-witness-policy.v1\0";
const EXACT_VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-operational-head-exact-verifier-set.v1\0";
const OBSERVED_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.quorum-observed-upgrade-operational-head.v1\0";
const LEGACY_WITNESS_QUORUM_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-quorum.v1\0";
const CHECKPOINT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-checkpoint-signature.v1\0";
const WITNESS_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-signature.v1\0";

/// Portable publication data. It is evidence only; live authority comes from the opaque observed
/// head capability after transparency, witness, registry, lifecycle and raw-signature checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpgradeOperationalHeadPublicationV1 {
    pub schema_version: String,
    pub handoff_digest: Sha256Digest,
    pub state_digest: Sha256Digest,
    pub generation: u64,
    pub committed_at_unix_ms: u64,
    pub previous_state_digest: Option<Sha256Digest>,
    pub probation_clearance_digest: Option<Sha256Digest>,
    pub automatic_rollback_digest: Option<Sha256Digest>,
    pub probation_sequence: Option<u64>,
    pub reauthorized_machine_count: u64,
    pub retention_policy_sequence: u64,
    pub key_snapshot_sequence: u64,
    pub clock_epoch: u64,
}

/// Runtime cryptographic verifier for the exact raw transparency evidence bytes.
pub trait ExactUpgradeOperationalHeadEvidenceVerifierV1 {
    fn provider_id(&self) -> &str;
    fn verification_policy_digest(&self) -> Sha256Digest;

    fn verify_checkpoint_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;

    fn verify_witness_signature(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactUpgradeOperationalHeadVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactUpgradeOperationalHeadVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuorumObservedUpgradeOperationalHeadIdV1(Sha256Digest);

impl QuorumObservedUpgradeOperationalHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one exact operational state is the latest publication for its handoff in one
/// exact witnessed transparency view, with governed witness identities and exact raw signature
/// bytes independently reverified.
#[derive(Debug, Clone)]
#[must_use]
pub struct QuorumObservedUpgradeOperationalHeadV1 {
    id: QuorumObservedUpgradeOperationalHeadIdV1,
    state_digest: Sha256Digest,
    state_generation: u64,
    handoff_digest: Sha256Digest,
    automatic_rollback_digest: Option<Sha256Digest>,
    probation_clearance_digest: Option<Sha256Digest>,
    probation_sequence: Option<u64>,
    reauthorized_machine_count: u64,
    retention_policy_sequence: u64,
    key_snapshot_sequence: u64,
    clock_epoch: u64,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    transparency_log_size: u64,
    transparency_root_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    signed_checkpoint_evidence_digest: Sha256Digest,
    witness_quorum_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    witness_policy_digest: Sha256Digest,
    witness_registry_id: WitnessAuthorityRegistryIdV1,
    witness_registry_digest: Sha256Digest,
    witness_registry_sequence: u64,
    exact_verifier_set_digest: Sha256Digest,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl QuorumObservedUpgradeOperationalHeadV1 {
    pub fn id(&self) -> QuorumObservedUpgradeOperationalHeadIdV1 {
        self.id
    }
    pub fn state_digest(&self) -> Sha256Digest {
        self.state_digest
    }
    pub fn state_generation(&self) -> u64 {
        self.state_generation
    }
    pub fn handoff_digest(&self) -> Sha256Digest {
        self.handoff_digest
    }
    pub fn automatic_rollback_digest(&self) -> Option<Sha256Digest> {
        self.automatic_rollback_digest
    }
    pub fn probation_clearance_digest(&self) -> Option<Sha256Digest> {
        self.probation_clearance_digest
    }
    pub fn probation_sequence(&self) -> Option<u64> {
        self.probation_sequence
    }
    pub fn reauthorized_machine_count(&self) -> u64 {
        self.reauthorized_machine_count
    }
    pub fn retention_policy_sequence(&self) -> u64 {
        self.retention_policy_sequence
    }
    pub fn key_snapshot_sequence(&self) -> u64 {
        self.key_snapshot_sequence
    }
    pub fn clock_epoch(&self) -> u64 {
        self.clock_epoch
    }
    pub fn publication_digest(&self) -> Sha256Digest {
        self.publication_digest
    }
    pub fn publication_entry_sequence(&self) -> u64 {
        self.publication_entry_sequence
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn transparency_log_size(&self) -> u64 {
        self.transparency_log_size
    }
    pub fn transparency_root_digest(&self) -> Sha256Digest {
        self.transparency_root_digest
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn signed_checkpoint_evidence_digest(&self) -> Sha256Digest {
        self.signed_checkpoint_evidence_digest
    }
    pub fn witness_quorum_digest(&self) -> Sha256Digest {
        self.witness_quorum_digest
    }
    pub fn witness_set_evidence_digest(&self) -> Sha256Digest {
        self.witness_set_evidence_digest
    }
    pub fn witness_policy_digest(&self) -> Sha256Digest {
        self.witness_policy_digest
    }
    pub fn witness_registry_id(&self) -> WitnessAuthorityRegistryIdV1 {
        self.witness_registry_id
    }
    pub fn witness_registry_digest(&self) -> Sha256Digest {
        self.witness_registry_digest
    }
    pub fn witness_registry_sequence(&self) -> u64 {
        self.witness_registry_sequence
    }
    pub fn exact_verifier_set_digest(&self) -> Sha256Digest {
        self.exact_verifier_set_digest
    }
    pub fn exact_verifier_count(&self) -> usize {
        self.exact_verifier_count
    }
    pub fn witness_count(&self) -> usize {
        self.witness_count
    }
    pub fn organization_count(&self) -> usize {
        self.organization_count
    }
    pub fn failure_domain_count(&self) -> usize {
        self.failure_domain_count
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeOperationalHeadObservationError {
    StateInvalid(String),
    InvalidPublication,
    PublicationMismatch,
    TransparencyLogInvalid(String),
    PublicationNotFound,
    PublicationNotLatestInCheckpointView,
    PublicationBeforeStateCommit,
    PublicationMayBeFuture,
    Clock(ClockGovernanceTimeError),
    TimeScaleOverflow,
    TrustSnapshotInvalid(String),
    TrustSnapshotPostdatesCheckpoint,
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    ContainmentStateInvalid(String),
    CheckpointInvalid(String),
    CheckpointVerifiedEvidenceMismatch,
    CheckpointLogMismatch,
    CheckpointPredatesLog,
    SignerUnknown(String),
    SignerNotActive(String),
    SignerUsageNotAllowed(String),
    SignerInvalidAtEvidenceTime(String),
    SignerNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    SignerCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    WitnessPolicyInvalid,
    TooManyWitnesses { actual: usize, maximum: usize },
    WitnessMalformed(String),
    WitnessCheckpointMismatch(String),
    WitnessBeforeCheckpoint(String),
    WitnessMayBeFuture(String),
    WitnessMayBeStale(String),
    DuplicateWitnessSigner(String),
    WitnessNotRegistered(String),
    WitnessOrganizationMismatch(String),
    WitnessFailureDomainMismatch(String),
    InsufficientWitnesses { actual: usize, required: usize },
    InsufficientOrganizations { actual: usize, required: usize },
    InsufficientFailureDomains { actual: usize, required: usize },
    MissingAlgorithmDiversity,
    WitnessVerifiedEvidenceMismatch,
    InvalidExactVerificationPolicy,
    InsufficientExactVerificationProviders { actual: usize, required: usize },
    TooManyExactVerificationProviders { actual: usize, maximum: usize },
    InvalidExactVerificationProvider(String),
    DuplicateExactVerificationProvider(String),
    CheckpointSignatureRejected(String),
    WitnessSignatureRejected { provider: String, key_id: String },
    VerificationProviderError { provider: String, reason: String },
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct WitnessPolicyCommitment {
    minimum_distinct_witnesses: usize,
    minimum_distinct_organizations: usize,
    minimum_distinct_regions: usize,
    maximum_observation_age_s: u64,
    maximum_witnesses: usize,
    require_algorithm_diversity: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ExactVerifierCommitment {
    provider_id: String,
    verification_policy_digest: String,
}

#[derive(Debug, Clone, Serialize)]
struct ObservedOperationalHeadCommitment {
    schema: &'static str,
    state_digest: String,
    state_generation: u64,
    handoff_digest: String,
    automatic_rollback_digest: Option<String>,
    probation_clearance_digest: Option<String>,
    probation_sequence: Option<u64>,
    reauthorized_machine_count: u64,
    retention_policy_sequence: u64,
    key_snapshot_sequence: u64,
    clock_epoch: u64,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    transparency_log_size: u64,
    transparency_root_digest: String,
    checkpoint_digest: String,
    signed_checkpoint_evidence_digest: String,
    witness_quorum_digest: String,
    witness_set_evidence_digest: String,
    witness_policy_digest: String,
    witness_registry_id: String,
    witness_registry_digest: String,
    witness_registry_sequence: u64,
    exact_verifier_set_digest: String,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
}

pub fn build_upgrade_operational_head_publication_v1(
    state: &FabricationUpgradeOperationalState,
) -> Result<UpgradeOperationalHeadPublicationV1, UpgradeOperationalHeadObservationError> {
    state
        .validate_shape()
        .map_err(|error| UpgradeOperationalHeadObservationError::StateInvalid(format!("{error:?}")))?;
    let state_digest = digest_upgrade_operational_state(state)
        .map_err(|error| UpgradeOperationalHeadObservationError::StateInvalid(format!("{error:?}")))?;
    Ok(UpgradeOperationalHeadPublicationV1 {
        schema_version: UPGRADE_OPERATIONAL_HEAD_PUBLICATION_SCHEMA.into(),
        handoff_digest: state.handoff_digest,
        state_digest,
        generation: state.generation,
        committed_at_unix_ms: state.committed_at_unix_ms,
        previous_state_digest: state.previous_state_digest,
        probation_clearance_digest: state.evidence.probation_clearance_digest,
        automatic_rollback_digest: state.evidence.automatic_rollback_digest,
        probation_sequence: state.evidence.probation_sequence,
        reauthorized_machine_count: state.evidence.reauthorized_machine_count,
        retention_policy_sequence: state.evidence.retention_policy_sequence,
        key_snapshot_sequence: state.evidence.key_snapshot_sequence,
        clock_epoch: state.evidence.clock_epoch,
    })
}

pub fn digest_upgrade_operational_head_publication_v1(
    publication: &UpgradeOperationalHeadPublicationV1,
) -> Result<Sha256Digest, UpgradeOperationalHeadObservationError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

pub fn upgrade_operational_head_log_kind(
    handoff_digest: Sha256Digest,
) -> Result<String, UpgradeOperationalHeadObservationError> {
    if handoff_digest == Sha256Digest([0; 32]) {
        return Err(UpgradeOperationalHeadObservationError::InvalidPublication);
    }
    let mut hasher = Sha256::new();
    hasher.update(LOG_KIND_DOMAIN);
    hasher.update(&handoff_digest.0);
    Ok(format!(
        "{UPGRADE_OPERATIONAL_HEAD_LOG_KIND_PREFIX}{}",
        hasher.finalize().to_hex()
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_quorum_observed_upgrade_operational_head_v1(
    state: &FabricationUpgradeOperationalState,
    publication: &UpgradeOperationalHeadPublicationV1,
    log: &TransparencyLog,
    signed_checkpoint: &SignedTransparencyCheckpoint,
    verified_checkpoint: &VerifiedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    verified_witness_quorum: &VerifiedTransparencyWitnessQuorum,
    witness_policy: &TransparencyWitnessPolicy,
    witness_registry: &WitnessAuthorityRegistryV1,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    current_basis: &OperationalClockBasisV1,
    exact_verification_policy: &ExactUpgradeOperationalHeadVerificationPolicyV1,
    exact_verification_providers: &[&dyn ExactUpgradeOperationalHeadEvidenceVerifierV1],
) -> Result<QuorumObservedUpgradeOperationalHeadV1, Vec<UpgradeOperationalHeadObservationError>> {
    let mut violations = Vec::new();

    if let Err(error) = state.validate_shape() {
        violations.push(UpgradeOperationalHeadObservationError::StateInvalid(format!(
            "{error:?}"
        )));
    }
    let expected_publication = match build_upgrade_operational_head_publication_v1(state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if publication != &expected_publication {
        violations.push(UpgradeOperationalHeadObservationError::PublicationMismatch);
    }
    let publication_digest = match digest_upgrade_operational_head_publication_v1(publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };

    let clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeOperationalHeadObservationError::Clock(error));
            return Err(violations);
        }
    };

    if let Err(error) = log.validate() {
        violations.push(UpgradeOperationalHeadObservationError::TransparencyLogInvalid(
            format!("{error:?}"),
        ));
    }
    let log_kind = match upgrade_operational_head_log_kind(state.handoff_digest) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    let matching = log
        .entries
        .iter()
        .filter(|entry| entry.kind == log_kind)
        .collect::<Vec<_>>();
    let Some(publication_entry) = matching.last().copied() else {
        violations.push(UpgradeOperationalHeadObservationError::PublicationNotFound);
        return Err(violations);
    };
    if publication_entry.subject_digest != publication_digest {
        violations.push(
            UpgradeOperationalHeadObservationError::PublicationNotLatestInCheckpointView,
        );
    }
    let publication_recorded_at_ms = match seconds_to_millis(publication_entry.recorded_at_unix_s) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if publication_recorded_at_ms < state.committed_at_unix_ms {
        violations.push(UpgradeOperationalHeadObservationError::PublicationBeforeStateCommit);
    }
    if publication_recorded_at_ms > clock.lower_unix_ms() {
        violations.push(UpgradeOperationalHeadObservationError::PublicationMayBeFuture);
    }

    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeOperationalHeadObservationError::TrustSnapshotInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(UpgradeOperationalHeadObservationError::TrustSnapshotInvalid(
            format!("{error:?}"),
        ));
    }
    if trust_snapshot.issued_at_unix_s > signed_checkpoint.checkpoint.issued_at_unix_s {
        violations.push(UpgradeOperationalHeadObservationError::TrustSnapshotPostdatesCheckpoint);
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            UpgradeOperationalHeadObservationError::TrustSnapshotNotValidAcrossEnvelope(reason),
        );
    }

    if let Err(error) = containment_state.validate() {
        violations.push(UpgradeOperationalHeadObservationError::ContainmentStateInvalid(
            format!("{error:?}"),
        ));
    }
    let containment_state_digest = match digest_containment_state(containment_state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeOperationalHeadObservationError::ContainmentStateInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };
    let compromise_tracker_digest = match digest_signer_compromise_tracker(
        &containment_state.signer_compromise_tracker,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeOperationalHeadObservationError::ContainmentStateInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };

    if signed_checkpoint.schema_version != SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA {
        violations.push(UpgradeOperationalHeadObservationError::CheckpointInvalid(
            "unsupported signed checkpoint schema".into(),
        ));
    }
    if let Err(error) = signed_checkpoint.checkpoint.validate() {
        violations.push(UpgradeOperationalHeadObservationError::CheckpointInvalid(
            format!("{error:?}"),
        ));
    }
    match digest_transparency_checkpoint(&signed_checkpoint.checkpoint) {
        Ok(value) if value == signed_checkpoint.checkpoint_digest => {}
        Ok(_) => violations.push(UpgradeOperationalHeadObservationError::CheckpointInvalid(
            "checkpoint digest mismatch".into(),
        )),
        Err(error) => violations.push(UpgradeOperationalHeadObservationError::CheckpointInvalid(
            format!("{error:?}"),
        )),
    }
    if verified_checkpoint.checkpoint() != &signed_checkpoint.checkpoint
        || verified_checkpoint.checkpoint_digest() != signed_checkpoint.checkpoint_digest
        || verified_checkpoint.signer()
            != &(
                signed_checkpoint.signature.algorithm.clone(),
                signed_checkpoint.signature.key_id.clone(),
            )
    {
        violations.push(
            UpgradeOperationalHeadObservationError::CheckpointVerifiedEvidenceMismatch,
        );
    }
    let log_root = match log.root() {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeOperationalHeadObservationError::TransparencyLogInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };
    if signed_checkpoint.checkpoint.log_size != log.entries.len() as u64
        || signed_checkpoint.checkpoint.root_digest != log_root
    {
        violations.push(UpgradeOperationalHeadObservationError::CheckpointLogMismatch);
    }
    if log
        .entries
        .last()
        .is_some_and(|entry| entry.recorded_at_unix_s > signed_checkpoint.checkpoint.issued_at_unix_s)
    {
        violations.push(UpgradeOperationalHeadObservationError::CheckpointPredatesLog);
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        signed_checkpoint.checkpoint.issued_at_unix_s,
        signed_checkpoint.checkpoint.expires_at_unix_s,
    ) {
        violations.push(UpgradeOperationalHeadObservationError::CheckpointInvalid(format!(
            "{reason:?}"
        )));
    }
    if !signed_checkpoint.signature.algorithm.is_canonical()
        || invalid_identifier_with_limit(
            &signed_checkpoint.signature.key_id,
            MAX_TRANSPARENCY_CHECKPOINT_KEY_ID_BYTES,
        )
        || signed_checkpoint.signature.signature.is_empty()
        || signed_checkpoint.signature.signature.len()
            > MAX_TRANSPARENCY_CHECKPOINT_SIGNATURE_BYTES
    {
        violations.push(UpgradeOperationalHeadObservationError::CheckpointInvalid(
            "invalid checkpoint signer or signature bytes".into(),
        ));
    }
    requalify_signer(
        &signed_checkpoint.signature.algorithm,
        &signed_checkpoint.signature.key_id,
        KeyUsage::TransparencyLog,
        signed_checkpoint.checkpoint.issued_at_unix_s,
        trust_snapshot,
        containment_state,
        &clock,
        &mut violations,
    );

    if !valid_witness_policy(witness_policy) {
        violations.push(UpgradeOperationalHeadObservationError::WitnessPolicyInvalid);
    }
    if signed_witnesses.len() > witness_policy.maximum_witnesses
        || signed_witnesses.len() > MAX_TRANSPARENCY_WITNESSES
    {
        violations.push(UpgradeOperationalHeadObservationError::TooManyWitnesses {
            actual: signed_witnesses.len(),
            maximum: witness_policy
                .maximum_witnesses
                .min(MAX_TRANSPARENCY_WITNESSES),
        });
        return Err(violations);
    }

    let witness_policy_digest = match digest_witness_policy(witness_policy) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };
    let mut signer_ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut failure_domains = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut witness_identities = Vec::with_capacity(signed_witnesses.len());
    let mut statement_digests = Vec::with_capacity(signed_witnesses.len());
    let mut witness_evidence_digests = Vec::with_capacity(signed_witnesses.len());

    for signed in signed_witnesses {
        let key_id = signed.signature.key_id.clone();
        if signed.schema_version != SIGNED_TRANSPARENCY_WITNESS_SCHEMA
            || signed.statement.schema_version != TRANSPARENCY_WITNESS_SCHEMA
            || invalid_identifier(&signed.statement.witness_organization)
            || invalid_identifier(&signed.statement.witness_region)
            || signed.statement.checkpoint_log_size == 0
            || !signed.signature.algorithm.is_canonical()
            || invalid_identifier(&signed.signature.key_id)
            || signed.signature.signature.is_empty()
            || signed.signature.signature.len() > 64 * 1024
        {
            violations.push(UpgradeOperationalHeadObservationError::WitnessMalformed(key_id));
            continue;
        }
        if signed.statement.checkpoint_digest != signed_checkpoint.checkpoint_digest
            || signed.statement.checkpoint_log_size != signed_checkpoint.checkpoint.log_size
            || signed.statement.checkpoint_root_digest != signed_checkpoint.checkpoint.root_digest
        {
            violations.push(UpgradeOperationalHeadObservationError::WitnessCheckpointMismatch(
                key_id,
            ));
            continue;
        }
        match digest_transparency_witness_statement(&signed.statement) {
            Ok(value) if value == signed.statement_digest => {}
            Ok(_) => {
                violations.push(UpgradeOperationalHeadObservationError::WitnessMalformed(
                    key_id,
                ));
                continue;
            }
            Err(error) => {
                violations.push(UpgradeOperationalHeadObservationError::WitnessMalformed(
                    format!("{}: {error:?}", signed.signature.key_id),
                ));
                continue;
            }
        }
        if signed.statement.observed_at_unix_s < signed_checkpoint.checkpoint.issued_at_unix_s {
            violations.push(UpgradeOperationalHeadObservationError::WitnessBeforeCheckpoint(
                signed.signature.key_id.clone(),
            ));
        }
        let observed_at_ms = match seconds_to_millis(signed.statement.observed_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if observed_at_ms > clock.lower_unix_ms() {
            violations.push(UpgradeOperationalHeadObservationError::WitnessMayBeFuture(
                signed.signature.key_id.clone(),
            ));
        }
        let max_age_ms = match seconds_to_millis(witness_policy.maximum_observation_age_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        let freshness_deadline = match observed_at_ms.checked_add(max_age_ms) {
            Some(value) => value,
            None => {
                violations.push(UpgradeOperationalHeadObservationError::TimeScaleOverflow);
                continue;
            }
        };
        if clock.upper_unix_ms() > freshness_deadline {
            violations.push(UpgradeOperationalHeadObservationError::WitnessMayBeStale(
                signed.signature.key_id.clone(),
            ));
        }
        let signer = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signer_ids.insert(signer) {
            violations.push(UpgradeOperationalHeadObservationError::DuplicateWitnessSigner(
                signed.signature.key_id.clone(),
            ));
            continue;
        }
        requalify_signer(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            KeyUsage::TransparencyWitness,
            signed.statement.observed_at_unix_s,
            trust_snapshot,
            containment_state,
            &clock,
            &mut violations,
        );

        let Some(profile) = witness_registry.profile(
            &signed.signature.algorithm,
            &signed.signature.key_id,
        ) else {
            violations.push(UpgradeOperationalHeadObservationError::WitnessNotRegistered(
                signed.signature.key_id.clone(),
            ));
            continue;
        };
        if profile.organization != signed.statement.witness_organization {
            violations.push(
                UpgradeOperationalHeadObservationError::WitnessOrganizationMismatch(
                    signed.signature.key_id.clone(),
                ),
            );
        }
        if profile.failure_domain != signed.statement.witness_region {
            violations.push(
                UpgradeOperationalHeadObservationError::WitnessFailureDomainMismatch(
                    signed.signature.key_id.clone(),
                ),
            );
        }

        algorithms.insert(signed.signature.algorithm.clone());
        organizations.insert(profile.organization.clone());
        failure_domains.insert(profile.failure_domain.clone());
        witness_identities.push((
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
            signed.statement.witness_organization.clone(),
            signed.statement.witness_region.clone(),
        ));
        statement_digests.push(signed.statement_digest);
        match hash_serializable(SIGNED_WITNESS_EVIDENCE_DOMAIN, signed) {
            Ok(value) => witness_evidence_digests.push(value),
            Err(error) => violations.push(error),
        }
    }

    if witness_identities.len() < witness_policy.minimum_distinct_witnesses {
        violations.push(UpgradeOperationalHeadObservationError::InsufficientWitnesses {
            actual: witness_identities.len(),
            required: witness_policy.minimum_distinct_witnesses,
        });
    }
    if organizations.len() < witness_policy.minimum_distinct_organizations {
        violations.push(
            UpgradeOperationalHeadObservationError::InsufficientOrganizations {
                actual: organizations.len(),
                required: witness_policy.minimum_distinct_organizations,
            },
        );
    }
    if failure_domains.len() < witness_policy.minimum_distinct_regions {
        violations.push(
            UpgradeOperationalHeadObservationError::InsufficientFailureDomains {
                actual: failure_domains.len(),
                required: witness_policy.minimum_distinct_regions,
            },
        );
    }
    if witness_policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(UpgradeOperationalHeadObservationError::MissingAlgorithmDiversity);
    }

    witness_identities.sort();
    statement_digests.sort();
    let witness_quorum_digest = digest_legacy_witness_quorum(
        signed_checkpoint.checkpoint_digest,
        &statement_digests,
    );
    if verified_witness_quorum.checkpoint_digest() != signed_checkpoint.checkpoint_digest
        || verified_witness_quorum.witness_quorum_digest() != witness_quorum_digest
        || verified_witness_quorum.trust_snapshot_digest() != trust_snapshot_digest
        || verified_witness_quorum.witnesses() != witness_identities.as_slice()
    {
        violations.push(
            UpgradeOperationalHeadObservationError::WitnessVerifiedEvidenceMismatch,
        );
    }

    if !valid_exact_verification_policy(exact_verification_policy) {
        violations.push(
            UpgradeOperationalHeadObservationError::InvalidExactVerificationPolicy,
        );
    }
    if exact_verification_providers.len()
        < exact_verification_policy.minimum_distinct_providers
    {
        violations.push(
            UpgradeOperationalHeadObservationError::InsufficientExactVerificationProviders {
                actual: exact_verification_providers.len(),
                required: exact_verification_policy.minimum_distinct_providers,
            },
        );
    }
    if exact_verification_providers.len() > exact_verification_policy.maximum_providers
        || exact_verification_providers.len() > MAX_OPERATIONAL_HEAD_EXACT_VERIFIERS
    {
        violations.push(
            UpgradeOperationalHeadObservationError::TooManyExactVerificationProviders {
                actual: exact_verification_providers.len(),
                maximum: exact_verification_policy
                    .maximum_providers
                    .min(MAX_OPERATIONAL_HEAD_EXACT_VERIFIERS),
            },
        );
        return Err(violations);
    }

    let mut exact_verifier_commitments = Vec::with_capacity(exact_verification_providers.len());
    let mut seen_exact_providers = BTreeSet::new();
    for provider in exact_verification_providers {
        let provider_id = provider.provider_id().to_string();
        let verification_policy_digest = provider.verification_policy_digest();
        if invalid_identifier(&provider_id)
            || verification_policy_digest == Sha256Digest([0; 32])
        {
            violations.push(
                UpgradeOperationalHeadObservationError::InvalidExactVerificationProvider(
                    provider_id,
                ),
            );
            continue;
        }
        if !seen_exact_providers.insert(provider_id.clone()) {
            violations.push(
                UpgradeOperationalHeadObservationError::DuplicateExactVerificationProvider(
                    provider_id,
                ),
            );
            continue;
        }
        exact_verifier_commitments.push(ExactVerifierCommitment {
            provider_id: provider_id.clone(),
            verification_policy_digest: verification_policy_digest.to_hex(),
        });

        let checkpoint_message = checkpoint_signature_message(signed_checkpoint.checkpoint_digest);
        match provider.verify_checkpoint_signature(
            &signed_checkpoint.signature.algorithm,
            &signed_checkpoint.signature.key_id,
            &checkpoint_message,
            &signed_checkpoint.signature.signature,
        ) {
            Ok(true) => {}
            Ok(false) => violations.push(
                UpgradeOperationalHeadObservationError::CheckpointSignatureRejected(
                    provider_id.clone(),
                ),
            ),
            Err(reason) => violations.push(
                UpgradeOperationalHeadObservationError::VerificationProviderError {
                    provider: provider_id.clone(),
                    reason,
                },
            ),
        }

        for witness in signed_witnesses {
            let message = witness_signature_message(witness.statement_digest);
            match provider.verify_witness_signature(
                &witness.signature.algorithm,
                &witness.signature.key_id,
                &message,
                &witness.signature.signature,
            ) {
                Ok(true) => {}
                Ok(false) => violations.push(
                    UpgradeOperationalHeadObservationError::WitnessSignatureRejected {
                        provider: provider_id.clone(),
                        key_id: witness.signature.key_id.clone(),
                    },
                ),
                Err(reason) => violations.push(
                    UpgradeOperationalHeadObservationError::VerificationProviderError {
                        provider: provider_id.clone(),
                        reason: format!("{}: {reason}", witness.signature.key_id),
                    },
                ),
            }
        }
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let transparency_log_digest = digest_transparency_log(log).map_err(|error| {
        vec![UpgradeOperationalHeadObservationError::TransparencyLogInvalid(
            format!("{error:?}"),
        )]
    })?;
    let signed_checkpoint_evidence_digest = hash_serializable(
        SIGNED_CHECKPOINT_EVIDENCE_DOMAIN,
        signed_checkpoint,
    )
    .map_err(|error| vec![error])?;
    witness_evidence_digests.sort();
    let witness_set_evidence_digest = digest_witness_evidence_set(&witness_evidence_digests);
    exact_verifier_commitments.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    let exact_verifier_set_digest = hash_serializable(
        EXACT_VERIFIER_SET_DOMAIN,
        &exact_verifier_commitments,
    )
    .map_err(|error| vec![error])?;

    let commitment = ObservedOperationalHeadCommitment {
        schema: QUORUM_OBSERVED_UPGRADE_OPERATIONAL_HEAD_SCHEMA,
        state_digest: publication.state_digest.to_hex(),
        state_generation: state.generation,
        handoff_digest: state.handoff_digest.to_hex(),
        automatic_rollback_digest: state
            .evidence
            .automatic_rollback_digest
            .map(Sha256Digest::to_hex),
        probation_clearance_digest: state
            .evidence
            .probation_clearance_digest
            .map(Sha256Digest::to_hex),
        probation_sequence: state.evidence.probation_sequence,
        reauthorized_machine_count: state.evidence.reauthorized_machine_count,
        retention_policy_sequence: state.evidence.retention_policy_sequence,
        key_snapshot_sequence: state.evidence.key_snapshot_sequence,
        clock_epoch: state.evidence.clock_epoch,
        publication_digest: publication_digest.to_hex(),
        publication_entry_sequence: publication_entry.sequence,
        transparency_log_digest: transparency_log_digest.to_hex(),
        transparency_log_size: log.entries.len() as u64,
        transparency_root_digest: log_root.to_hex(),
        checkpoint_digest: signed_checkpoint.checkpoint_digest.to_hex(),
        signed_checkpoint_evidence_digest: signed_checkpoint_evidence_digest.to_hex(),
        witness_quorum_digest: witness_quorum_digest.to_hex(),
        witness_set_evidence_digest: witness_set_evidence_digest.to_hex(),
        witness_policy_digest: witness_policy_digest.to_hex(),
        witness_registry_id: witness_registry.id().to_hex(),
        witness_registry_digest: witness_registry.registry_digest().to_hex(),
        witness_registry_sequence: witness_registry.sequence(),
        exact_verifier_set_digest: exact_verifier_set_digest.to_hex(),
        exact_verifier_count: exact_verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        containment_state_digest: containment_state_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        clock_envelope_id: clock.id().to_hex(),
        operational_basis_id: current_basis.id().to_hex(),
    };
    let id = QuorumObservedUpgradeOperationalHeadIdV1(
        hash_serializable(OBSERVED_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(QuorumObservedUpgradeOperationalHeadV1 {
        id,
        state_digest: publication.state_digest,
        state_generation: state.generation,
        handoff_digest: state.handoff_digest,
        automatic_rollback_digest: state.evidence.automatic_rollback_digest,
        probation_clearance_digest: state.evidence.probation_clearance_digest,
        probation_sequence: state.evidence.probation_sequence,
        reauthorized_machine_count: state.evidence.reauthorized_machine_count,
        retention_policy_sequence: state.evidence.retention_policy_sequence,
        key_snapshot_sequence: state.evidence.key_snapshot_sequence,
        clock_epoch: state.evidence.clock_epoch,
        publication_digest,
        publication_entry_sequence: publication_entry.sequence,
        transparency_log_digest,
        transparency_log_size: log.entries.len() as u64,
        transparency_root_digest: log_root,
        checkpoint_digest: signed_checkpoint.checkpoint_digest,
        signed_checkpoint_evidence_digest,
        witness_quorum_digest,
        witness_set_evidence_digest,
        witness_policy_digest,
        witness_registry_id: witness_registry.id(),
        witness_registry_digest: witness_registry.registry_digest(),
        witness_registry_sequence: witness_registry.sequence(),
        exact_verifier_set_digest,
        exact_verifier_count: exact_verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        clock_envelope_id: clock.id(),
        operational_basis_id: current_basis.id(),
    })
}

fn validate_publication(
    publication: &UpgradeOperationalHeadPublicationV1,
) -> Result<(), UpgradeOperationalHeadObservationError> {
    if publication.schema_version != UPGRADE_OPERATIONAL_HEAD_PUBLICATION_SCHEMA
        || publication.handoff_digest == Sha256Digest([0; 32])
        || publication.state_digest == Sha256Digest([0; 32])
        || publication.generation == 0
        || publication.retention_policy_sequence == 0
        || publication.key_snapshot_sequence == 0
        || publication.clock_epoch == 0
        || publication
            .probation_sequence
            .is_some_and(|sequence| sequence == 0)
        || publication
            .probation_clearance_digest
            .is_some_and(|digest| digest == Sha256Digest([0; 32]))
        || publication
            .automatic_rollback_digest
            .is_some_and(|digest| digest == Sha256Digest([0; 32]))
        || (publication.probation_clearance_digest.is_some()
            != publication.probation_sequence.is_some())
    {
        return Err(UpgradeOperationalHeadObservationError::InvalidPublication);
    }
    Ok(())
}

fn valid_witness_policy(policy: &TransparencyWitnessPolicy) -> bool {
    policy.minimum_distinct_witnesses > 0
        && policy.minimum_distinct_organizations > 0
        && policy.minimum_distinct_regions > 0
        && policy.maximum_observation_age_s > 0
        && policy.maximum_witnesses > 0
        && policy.maximum_witnesses <= MAX_TRANSPARENCY_WITNESSES
        && policy.minimum_distinct_witnesses <= policy.maximum_witnesses
        && policy.minimum_distinct_organizations <= policy.maximum_witnesses
        && policy.minimum_distinct_regions <= policy.maximum_witnesses
}

fn digest_witness_policy(
    policy: &TransparencyWitnessPolicy,
) -> Result<Sha256Digest, UpgradeOperationalHeadObservationError> {
    if !valid_witness_policy(policy) {
        return Err(UpgradeOperationalHeadObservationError::WitnessPolicyInvalid);
    }
    hash_serializable(
        WITNESS_POLICY_DOMAIN,
        &WitnessPolicyCommitment {
            minimum_distinct_witnesses: policy.minimum_distinct_witnesses,
            minimum_distinct_organizations: policy.minimum_distinct_organizations,
            minimum_distinct_regions: policy.minimum_distinct_regions,
            maximum_observation_age_s: policy.maximum_observation_age_s,
            maximum_witnesses: policy.maximum_witnesses,
            require_algorithm_diversity: policy.require_algorithm_diversity,
        },
    )
}

fn valid_exact_verification_policy(
    policy: &ExactUpgradeOperationalHeadVerificationPolicyV1,
) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_OPERATIONAL_HEAD_EXACT_VERIFIERS
}

#[allow(clippy::too_many_arguments)]
fn requalify_signer(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    usage: KeyUsage,
    evidence_time_unix_s: u64,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<UpgradeOperationalHeadObservationError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(UpgradeOperationalHeadObservationError::SignerUnknown(
            key_id.to_string(),
        ));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(UpgradeOperationalHeadObservationError::SignerNotActive(
            key_id.to_string(),
        ));
    }
    if !record.usages.contains(&usage) {
        violations.push(UpgradeOperationalHeadObservationError::SignerUsageNotAllowed(
            key_id.to_string(),
        ));
    }
    if record.not_before_unix_s > evidence_time_unix_s
        || record
            .not_after_unix_s
            .is_some_and(|not_after| evidence_time_unix_s >= not_after)
    {
        violations.push(
            UpgradeOperationalHeadObservationError::SignerInvalidAtEvidenceTime(
                key_id.to_string(),
            ),
        );
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(
            UpgradeOperationalHeadObservationError::SignerNotValidAcrossEnvelope {
                key_id: key_id.to_string(),
                reason,
            },
        );
    }
    for compromise in containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|compromise| {
            &compromise.signer.algorithm == algorithm
                && compromise.signer.key_id == key_id
                && compromise.affected_usages.contains(&usage)
        })
    {
        match clock.require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s) {
            Ok(()) => {}
            Err(ClockGovernanceTimeError::EventMayAlreadyBeEffective) => violations.push(
                UpgradeOperationalHeadObservationError::SignerCompromisedAcrossEnvelope(
                    key_id.to_string(),
                ),
            ),
            Err(reason) => violations.push(
                UpgradeOperationalHeadObservationError::CompromiseTimeInvalid {
                    key_id: key_id.to_string(),
                    reason,
                },
            ),
        }
    }
}

fn invalid_identifier(value: &str) -> bool {
    invalid_identifier_with_limit(value, 256)
}

fn invalid_identifier_with_limit(value: &str, maximum: usize) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
}

fn seconds_to_millis(
    value: u64,
) -> Result<u64, UpgradeOperationalHeadObservationError> {
    value
        .checked_mul(1_000)
        .ok_or(UpgradeOperationalHeadObservationError::TimeScaleOverflow)
}

fn checkpoint_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = CHECKPOINT_SIGNATURE_DOMAIN.to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn witness_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = WITNESS_SIGNATURE_DOMAIN.to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn digest_legacy_witness_quorum(
    checkpoint_digest: Sha256Digest,
    statement_digests: &[Sha256Digest],
) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(LEGACY_WITNESS_QUORUM_DOMAIN);
    hasher.update(&checkpoint_digest.0);
    for digest in statement_digests {
        hasher.update(&digest.0);
    }
    hasher.finalize()
}

fn digest_witness_evidence_set(witness_digests: &[Sha256Digest]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(WITNESS_SET_EVIDENCE_DOMAIN);
    hasher.update(&(witness_digests.len() as u64).to_le_bytes());
    for digest in witness_digests {
        hasher.update(&digest.0);
    }
    hasher.finalize()
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, UpgradeOperationalHeadObservationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| UpgradeOperationalHeadObservationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::crypto_digest::sha256;
    use symthaea_fabrication_kernel::upgrade_operational_state::UpgradeOperationalEvidenceDigests;

    fn evidence() -> UpgradeOperationalEvidenceDigests {
        UpgradeOperationalEvidenceDigests {
            upgrade_state_digest: sha256(b"upgrade"),
            probation_tracker_digest: sha256(b"probation"),
            hardware_reauthorization_tracker_digest: sha256(b"hardware"),
            retention_policy_digest: sha256(b"retention"),
            key_continuity_digest: sha256(b"keys"),
            clock_continuity_digest: sha256(b"clock"),
            probation_clearance_digest: None,
            automatic_rollback_digest: None,
            probation_sequence: None,
            reauthorized_machine_count: 0,
            retention_policy_sequence: 1,
            key_snapshot_sequence: 1,
            clock_epoch: 1,
        }
    }

    #[test]
    fn publication_identity_changes_when_rollback_becomes_durable() {
        let clean = FabricationUpgradeOperationalState::genesis(
            1_000,
            sha256(b"handoff"),
            evidence(),
        )
        .unwrap();
        let mut rolled_back_evidence = evidence();
        rolled_back_evidence.automatic_rollback_digest = Some(sha256(b"rollback"));
        let rolled_back = FabricationUpgradeOperationalState::genesis(
            1_000,
            sha256(b"handoff"),
            rolled_back_evidence,
        )
        .unwrap();
        assert_ne!(
            digest_upgrade_operational_head_publication_v1(
                &build_upgrade_operational_head_publication_v1(&clean).unwrap()
            )
            .unwrap(),
            digest_upgrade_operational_head_publication_v1(
                &build_upgrade_operational_head_publication_v1(&rolled_back).unwrap()
            )
            .unwrap()
        );
    }

    #[test]
    fn log_kind_is_handoff_specific() {
        assert_ne!(
            upgrade_operational_head_log_kind(sha256(b"handoff-a")).unwrap(),
            upgrade_operational_head_log_kind(sha256(b"handoff-b")).unwrap()
        );
    }

    #[test]
    fn exact_verification_defaults_to_two_providers() {
        let policy = ExactUpgradeOperationalHeadVerificationPolicyV1::default();
        assert_eq!(policy.minimum_distinct_providers, 2);
        assert!(valid_exact_verification_policy(&policy));
    }
}
