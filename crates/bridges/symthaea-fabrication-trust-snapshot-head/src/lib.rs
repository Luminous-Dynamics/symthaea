// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Witnessed current-head authority for activated fabrication trust snapshots.
//!
//! A legitimate rotation and activation permit establish that one successor trust snapshot may be
//! active. They do not prove that no later activated snapshot is already authoritative. This bridge
//! adds a bounded distributed-currentness theorem over one exact append-only transparency view.
//!
//! Trust-head log kinds expose the snapshot sequence so the observer can reject sequence regression,
//! duplicate/equivocated sequences and any higher published sequence inside the checkpointed view.
//! The checkpoint and witness signatures are verified directly over their exact raw bytes; the
//! legacy scalar-time `VerifiedTransparencyCheckpoint` / `VerifiedTransparencyWitnessQuorum`
//! wrappers are intentionally not part of this live authority path.

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
    digest_transparency_checkpoint,
};
use symthaea_fabrication_kernel::transparency_witness::{
    MAX_TRANSPARENCY_WITNESSES, SIGNED_TRANSPARENCY_WITNESS_SCHEMA,
    TRANSPARENCY_WITNESS_SCHEMA, SignedTransparencyWitness, TransparencyWitnessPolicy,
    digest_transparency_witness_statement,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_trust_rotation_authority::{
    ClockGovernedTrustRotationIdV1, ClockGovernedTrustRotationV1,
};
use symthaea_fabrication_trust_snapshot_activation::{
    ClockGovernedTrustSnapshotActivationPermitIdV1,
    ClockGovernedTrustSnapshotActivationPermitV1,
};
use symthaea_fabrication_witness_authority::{
    WitnessAuthorityRegistryIdV1, WitnessAuthorityRegistryV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const ACTIVATED_TRUST_SNAPSHOT_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.activated-trust-snapshot-head-publication.v1";
pub const QUORUM_OBSERVED_TRUST_SNAPSHOT_HEAD_SCHEMA: &str =
    "symthaea.fabrication.quorum-observed-trust-snapshot-head.v1";
pub const TRUST_SNAPSHOT_HEAD_LOG_KIND_PREFIX: &str = "trust-snapshot-head-v1:";
pub const MAX_TRUST_SNAPSHOT_HEAD_CLOCK_HOPS: usize = 4096;
pub const MAX_TRUST_SNAPSHOT_HEAD_EXACT_VERIFIERS: usize = 16;

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.activated-trust-snapshot-head-publication.v1\0";
const SIGNED_CHECKPOINT_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-head-checkpoint-evidence.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-head-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-head-witness-set.v1\0";
const WITNESS_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-head-witness-policy.v1\0";
const EXACT_VERIFICATION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-head-exact-verification-policy.v1\0";
const EXACT_VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-head-exact-verifier-set.v1\0";
const OBSERVATION_CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.trust-snapshot-head-observation-clock-lineage.v1\0";
const OBSERVED_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.quorum-observed-trust-snapshot-head.v1\0";
const CHECKPOINT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-checkpoint-signature.v1\0";
const WITNESS_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-signature.v1\0";

/// Portable publication evidence. Serialization does not confer authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivatedTrustSnapshotHeadPublicationV1 {
    pub schema_version: String,
    pub snapshot_digest: Sha256Digest,
    pub snapshot_sequence: u64,
    pub predecessor_snapshot_digest: Sha256Digest,
    pub rotation_id: String,
    pub activation_permit_id: String,
    pub activates_at_unix_ms: u64,
    pub snapshot_issued_at_unix_s: u64,
    pub snapshot_expires_at_unix_s: u64,
    pub activation_clock_envelope_id: String,
    pub activation_operational_basis_id: String,
}

pub trait ExactTrustSnapshotHeadEvidenceVerifierV1 {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExactTrustSnapshotHeadVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactTrustSnapshotHeadVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuorumObservedTrustSnapshotHeadIdV1(Sha256Digest);

impl QuorumObservedTrustSnapshotHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one exact legitimately activated snapshot is the highest strictly-monotonic
/// trust-snapshot sequence published inside one exact witnessed transparency-checkpoint view.
#[derive(Debug, Clone)]
#[must_use]
pub struct QuorumObservedTrustSnapshotHeadV1 {
    id: QuorumObservedTrustSnapshotHeadIdV1,
    snapshot_digest: Sha256Digest,
    snapshot_sequence: u64,
    predecessor_snapshot_digest: Sha256Digest,
    rotation_id: ClockGovernedTrustRotationIdV1,
    activation_permit_id: ClockGovernedTrustSnapshotActivationPermitIdV1,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    transparency_log_size: u64,
    transparency_root_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    signed_checkpoint_evidence_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    witness_policy_digest: Sha256Digest,
    witness_registry_id: WitnessAuthorityRegistryIdV1,
    witness_registry_digest: Sha256Digest,
    witness_registry_sequence: u64,
    exact_verification_policy_digest: Sha256Digest,
    exact_verifier_set_digest: Sha256Digest,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    activation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_lineage_digest: Sha256Digest,
    observation_clock_hop_count: usize,
}

impl QuorumObservedTrustSnapshotHeadV1 {
    pub fn id(&self) -> QuorumObservedTrustSnapshotHeadIdV1 {
        self.id
    }
    pub fn snapshot_digest(&self) -> Sha256Digest {
        self.snapshot_digest
    }
    pub fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }
    pub fn predecessor_snapshot_digest(&self) -> Sha256Digest {
        self.predecessor_snapshot_digest
    }
    pub fn rotation_id(&self) -> ClockGovernedTrustRotationIdV1 {
        self.rotation_id
    }
    pub fn activation_permit_id(&self) -> ClockGovernedTrustSnapshotActivationPermitIdV1 {
        self.activation_permit_id
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
    pub fn exact_verification_policy_digest(&self) -> Sha256Digest {
        self.exact_verification_policy_digest
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
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn activation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.activation_clock_envelope_id
    }
    pub fn activation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.activation_operational_basis_id
    }
    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.observation_clock_envelope_id
    }
    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.observation_operational_basis_id
    }
    pub fn observation_clock_lineage_digest(&self) -> Sha256Digest {
        self.observation_clock_lineage_digest
    }
    pub fn observation_clock_hop_count(&self) -> usize {
        self.observation_clock_hop_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustSnapshotHeadObservationError {
    RotationActivationMismatch,
    SnapshotDigestMismatch,
    InvalidPublication,
    PublicationMismatch,
    TransparencyLogInvalid(String),
    MalformedTrustSnapshotHeadKind(String),
    TrustSnapshotSequenceRegressed { previous: u64, current: u64 },
    DuplicateTrustSnapshotSequence(u64),
    PublicationNotFound,
    HigherSnapshotSequencePublished { candidate: u64, latest: u64 },
    PublicationDigestMismatch,
    PublicationBeforeActivation,
    PublicationMayBeFuture,
    ActivationBasisMismatch,
    ActivationEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    SnapshotNotValidAcrossObservationEnvelope(ClockGovernanceTimeError),
    ContainmentStateInvalid(String),
    CheckpointInvalid(String),
    CheckpointLogMismatch,
    CheckpointPredatesLog,
    CheckpointPredatesPublication,
    CheckpointPredatesSnapshot,
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
    InvalidExactVerificationPolicy,
    InsufficientExactVerificationProviders { actual: usize, required: usize },
    TooManyExactVerificationProviders { actual: usize, maximum: usize },
    InvalidExactVerificationProvider(String),
    DuplicateExactVerificationProvider(String),
    CheckpointSignatureRejected(String),
    WitnessSignatureRejected { provider: String, key_id: String },
    VerificationProviderError { provider: String, reason: String },
    TimeScaleOverflow,
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
struct ObservedTrustSnapshotHeadCommitment {
    schema: &'static str,
    snapshot_digest: String,
    snapshot_sequence: u64,
    predecessor_snapshot_digest: String,
    rotation_id: String,
    activation_permit_id: String,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    transparency_log_size: u64,
    transparency_root_digest: String,
    checkpoint_digest: String,
    signed_checkpoint_evidence_digest: String,
    witness_set_evidence_digest: String,
    witness_policy_digest: String,
    witness_registry_id: String,
    witness_registry_digest: String,
    witness_registry_sequence: u64,
    exact_verification_policy_digest: String,
    exact_verifier_set_digest: String,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    activation_clock_envelope_id: String,
    activation_operational_basis_id: String,
    observation_clock_envelope_id: String,
    observation_operational_basis_id: String,
    observation_clock_lineage_digest: String,
    observation_clock_hop_count: usize,
}

pub fn build_activated_trust_snapshot_head_publication_v1(
    rotation: &ClockGovernedTrustRotationV1,
    activation: &ClockGovernedTrustSnapshotActivationPermitV1,
) -> Result<ActivatedTrustSnapshotHeadPublicationV1, TrustSnapshotHeadObservationError> {
    require_rotation_activation_match(rotation, activation)?;
    let snapshot = rotation.proposed_snapshot();
    let snapshot_digest = digest_trust_snapshot(snapshot).map_err(|error| {
        TrustSnapshotHeadObservationError::Encoding(format!("trust snapshot digest: {error:?}"))
    })?;
    if snapshot_digest != rotation.proposed_snapshot_digest() {
        return Err(TrustSnapshotHeadObservationError::SnapshotDigestMismatch);
    }
    Ok(ActivatedTrustSnapshotHeadPublicationV1 {
        schema_version: ACTIVATED_TRUST_SNAPSHOT_HEAD_PUBLICATION_SCHEMA.into(),
        snapshot_digest,
        snapshot_sequence: snapshot.sequence,
        predecessor_snapshot_digest: rotation.current_snapshot_digest(),
        rotation_id: rotation.id().to_hex(),
        activation_permit_id: activation.id().to_hex(),
        activates_at_unix_ms: activation.activates_at_unix_ms(),
        snapshot_issued_at_unix_s: snapshot.issued_at_unix_s,
        snapshot_expires_at_unix_s: snapshot.expires_at_unix_s,
        activation_clock_envelope_id: activation.current_clock_envelope_id().to_hex(),
        activation_operational_basis_id: activation.current_operational_basis_id().to_hex(),
    })
}

pub fn digest_activated_trust_snapshot_head_publication_v1(
    publication: &ActivatedTrustSnapshotHeadPublicationV1,
) -> Result<Sha256Digest, TrustSnapshotHeadObservationError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

pub fn trust_snapshot_head_log_kind(
    snapshot_sequence: u64,
) -> Result<String, TrustSnapshotHeadObservationError> {
    if snapshot_sequence == 0 {
        return Err(TrustSnapshotHeadObservationError::InvalidPublication);
    }
    Ok(format!("{TRUST_SNAPSHOT_HEAD_LOG_KIND_PREFIX}{snapshot_sequence}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_quorum_observed_trust_snapshot_head_v1(
    rotation: &ClockGovernedTrustRotationV1,
    activation: &ClockGovernedTrustSnapshotActivationPermitV1,
    activation_basis: &OperationalClockBasisV1,
    observation_clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    publication: &ActivatedTrustSnapshotHeadPublicationV1,
    log: &TransparencyLog,
    signed_checkpoint: &SignedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    witness_policy: &TransparencyWitnessPolicy,
    witness_registry: &WitnessAuthorityRegistryV1,
    containment_state: &FabricationContainmentState,
    exact_verification_policy: &ExactTrustSnapshotHeadVerificationPolicyV1,
    exact_verification_providers: &[&dyn ExactTrustSnapshotHeadEvidenceVerifierV1],
) -> Result<QuorumObservedTrustSnapshotHeadV1, Vec<TrustSnapshotHeadObservationError>> {
    let mut violations = Vec::new();

    if let Err(error) = require_rotation_activation_match(rotation, activation) {
        violations.push(error);
    }
    let expected_publication = match build_activated_trust_snapshot_head_publication_v1(
        rotation,
        activation,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if publication != &expected_publication {
        violations.push(TrustSnapshotHeadObservationError::PublicationMismatch);
    }
    let publication_digest = match digest_activated_trust_snapshot_head_publication_v1(publication)
    {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };

    if activation_basis.id() != activation.current_operational_basis_id() {
        violations.push(TrustSnapshotHeadObservationError::ActivationBasisMismatch);
    }
    let activation_clock = match derive_clock_governance_evaluation_envelope_v1(activation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(TrustSnapshotHeadObservationError::Clock(error));
            return Err(violations);
        }
    };
    if activation_clock.id() != activation.current_clock_envelope_id() {
        violations.push(TrustSnapshotHeadObservationError::ActivationEnvelopeMismatch);
    }
    if observation_clock_bridge.len() > MAX_TRUST_SNAPSHOT_HEAD_CLOCK_HOPS {
        violations.push(TrustSnapshotHeadObservationError::TooManyClockHops {
            actual: observation_clock_bridge.len(),
            maximum: MAX_TRUST_SNAPSHOT_HEAD_CLOCK_HOPS,
        });
        return Err(violations);
    }
    if let Err(error) = verify_clock_lineage(
        activation_basis.id(),
        observation_clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis)
    {
        Ok(value) => value,
        Err(error) => {
            violations.push(TrustSnapshotHeadObservationError::Clock(error));
            return Err(violations);
        }
    };

    let snapshot = rotation.proposed_snapshot();
    let snapshot_digest = match digest_trust_snapshot(snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(TrustSnapshotHeadObservationError::Encoding(format!(
                "trust snapshot digest: {error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if snapshot_digest != rotation.proposed_snapshot_digest()
        || snapshot_digest != activation.proposed_snapshot_digest()
        || snapshot.sequence != activation.proposed_snapshot_sequence()
    {
        violations.push(TrustSnapshotHeadObservationError::SnapshotDigestMismatch);
    }
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        snapshot.issued_at_unix_s,
        snapshot.expires_at_unix_s,
    ) {
        violations.push(
            TrustSnapshotHeadObservationError::SnapshotNotValidAcrossObservationEnvelope(reason),
        );
    }

    if let Err(error) = log.validate() {
        violations.push(TrustSnapshotHeadObservationError::TransparencyLogInvalid(
            format!("{error:?}"),
        ));
    }
    let (latest_published_sequence, publication_entry) = match inspect_trust_head_log(
        log,
        publication.snapshot_sequence,
    ) {
        Ok(value) => value,
        Err(errors) => {
            violations.extend(errors);
            (None, None)
        }
    };
    if let Some(latest) = latest_published_sequence {
        if latest > publication.snapshot_sequence {
            violations.push(
                TrustSnapshotHeadObservationError::HigherSnapshotSequencePublished {
                    candidate: publication.snapshot_sequence,
                    latest,
                },
            );
        }
    }
    let Some(publication_entry) = publication_entry else {
        violations.push(TrustSnapshotHeadObservationError::PublicationNotFound);
        return Err(violations);
    };
    if publication_entry.1 != publication_digest {
        violations.push(TrustSnapshotHeadObservationError::PublicationDigestMismatch);
    }
    let publication_recorded_at_ms = match seconds_to_millis(publication_entry.2) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if publication_recorded_at_ms < activation.activates_at_unix_ms() {
        violations.push(TrustSnapshotHeadObservationError::PublicationBeforeActivation);
    }
    if publication_recorded_at_ms > observation_clock.lower_unix_ms() {
        violations.push(TrustSnapshotHeadObservationError::PublicationMayBeFuture);
    }

    if let Err(error) = containment_state.validate() {
        violations.push(TrustSnapshotHeadObservationError::ContainmentStateInvalid(
            format!("{error:?}"),
        ));
    }
    let containment_state_digest = match digest_containment_state(containment_state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(TrustSnapshotHeadObservationError::ContainmentStateInvalid(
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
            violations.push(TrustSnapshotHeadObservationError::ContainmentStateInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };

    if signed_checkpoint.schema_version != SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA {
        violations.push(TrustSnapshotHeadObservationError::CheckpointInvalid(
            "unsupported signed checkpoint schema".into(),
        ));
    }
    if let Err(error) = signed_checkpoint.checkpoint.validate() {
        violations.push(TrustSnapshotHeadObservationError::CheckpointInvalid(format!(
            "{error:?}"
        )));
    }
    match digest_transparency_checkpoint(&signed_checkpoint.checkpoint) {
        Ok(value) if value == signed_checkpoint.checkpoint_digest => {}
        Ok(_) => violations.push(TrustSnapshotHeadObservationError::CheckpointInvalid(
            "checkpoint digest mismatch".into(),
        )),
        Err(error) => violations.push(TrustSnapshotHeadObservationError::CheckpointInvalid(
            format!("{error:?}"),
        )),
    }
    let log_root = match log.root() {
        Ok(value) => value,
        Err(error) => {
            violations.push(TrustSnapshotHeadObservationError::TransparencyLogInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };
    if signed_checkpoint.checkpoint.log_size != log.entries.len() as u64
        || signed_checkpoint.checkpoint.root_digest != log_root
    {
        violations.push(TrustSnapshotHeadObservationError::CheckpointLogMismatch);
    }
    if log
        .entries
        .last()
        .is_some_and(|entry| entry.recorded_at_unix_s > signed_checkpoint.checkpoint.issued_at_unix_s)
    {
        violations.push(TrustSnapshotHeadObservationError::CheckpointPredatesLog);
    }
    if publication_entry.2 > signed_checkpoint.checkpoint.issued_at_unix_s {
        violations.push(TrustSnapshotHeadObservationError::CheckpointPredatesPublication);
    }
    if snapshot.issued_at_unix_s > signed_checkpoint.checkpoint.issued_at_unix_s {
        violations.push(TrustSnapshotHeadObservationError::CheckpointPredatesSnapshot);
    }
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        signed_checkpoint.checkpoint.issued_at_unix_s,
        signed_checkpoint.checkpoint.expires_at_unix_s,
    ) {
        violations.push(TrustSnapshotHeadObservationError::CheckpointInvalid(format!(
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
        violations.push(TrustSnapshotHeadObservationError::CheckpointInvalid(
            "invalid checkpoint signer or signature bytes".into(),
        ));
    }
    requalify_signer(
        &signed_checkpoint.signature.algorithm,
        &signed_checkpoint.signature.key_id,
        KeyUsage::TransparencyLog,
        signed_checkpoint.checkpoint.issued_at_unix_s,
        snapshot,
        containment_state,
        &observation_clock,
        &mut violations,
    );

    if !valid_witness_policy(witness_policy) {
        violations.push(TrustSnapshotHeadObservationError::WitnessPolicyInvalid);
    }
    if signed_witnesses.len() > witness_policy.maximum_witnesses
        || signed_witnesses.len() > MAX_TRANSPARENCY_WITNESSES
    {
        violations.push(TrustSnapshotHeadObservationError::TooManyWitnesses {
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
            violations.push(TrustSnapshotHeadObservationError::WitnessMalformed(key_id));
            continue;
        }
        if signed.statement.checkpoint_digest != signed_checkpoint.checkpoint_digest
            || signed.statement.checkpoint_log_size != signed_checkpoint.checkpoint.log_size
            || signed.statement.checkpoint_root_digest != signed_checkpoint.checkpoint.root_digest
        {
            violations.push(TrustSnapshotHeadObservationError::WitnessCheckpointMismatch(
                key_id,
            ));
            continue;
        }
        match digest_transparency_witness_statement(&signed.statement) {
            Ok(value) if value == signed.statement_digest => {}
            Ok(_) => {
                violations.push(TrustSnapshotHeadObservationError::WitnessMalformed(key_id));
                continue;
            }
            Err(error) => {
                violations.push(TrustSnapshotHeadObservationError::WitnessMalformed(format!(
                    "{}: {error:?}",
                    signed.signature.key_id
                )));
                continue;
            }
        }
        if signed.statement.observed_at_unix_s < signed_checkpoint.checkpoint.issued_at_unix_s {
            violations.push(TrustSnapshotHeadObservationError::WitnessBeforeCheckpoint(
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
        if observed_at_ms > observation_clock.lower_unix_ms() {
            violations.push(TrustSnapshotHeadObservationError::WitnessMayBeFuture(
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
                violations.push(TrustSnapshotHeadObservationError::TimeScaleOverflow);
                continue;
            }
        };
        if observation_clock.upper_unix_ms() > freshness_deadline {
            violations.push(TrustSnapshotHeadObservationError::WitnessMayBeStale(
                signed.signature.key_id.clone(),
            ));
        }
        let signer = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signer_ids.insert(signer) {
            violations.push(TrustSnapshotHeadObservationError::DuplicateWitnessSigner(
                signed.signature.key_id.clone(),
            ));
            continue;
        }
        requalify_signer(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            KeyUsage::TransparencyWitness,
            signed.statement.observed_at_unix_s,
            snapshot,
            containment_state,
            &observation_clock,
            &mut violations,
        );

        let Some(profile) = witness_registry.profile(
            &signed.signature.algorithm,
            &signed.signature.key_id,
        ) else {
            violations.push(TrustSnapshotHeadObservationError::WitnessNotRegistered(
                signed.signature.key_id.clone(),
            ));
            continue;
        };
        if profile.organization != signed.statement.witness_organization {
            violations.push(TrustSnapshotHeadObservationError::WitnessOrganizationMismatch(
                signed.signature.key_id.clone(),
            ));
        }
        if profile.failure_domain != signed.statement.witness_region {
            violations.push(TrustSnapshotHeadObservationError::WitnessFailureDomainMismatch(
                signed.signature.key_id.clone(),
            ));
        }

        algorithms.insert(signed.signature.algorithm.clone());
        organizations.insert(profile.organization.clone());
        failure_domains.insert(profile.failure_domain.clone());
        match hash_serializable(SIGNED_WITNESS_EVIDENCE_DOMAIN, signed) {
            Ok(value) => witness_evidence_digests.push(value),
            Err(error) => violations.push(error),
        }
    }

    if signer_ids.len() < witness_policy.minimum_distinct_witnesses {
        violations.push(TrustSnapshotHeadObservationError::InsufficientWitnesses {
            actual: signer_ids.len(),
            required: witness_policy.minimum_distinct_witnesses,
        });
    }
    if organizations.len() < witness_policy.minimum_distinct_organizations {
        violations.push(TrustSnapshotHeadObservationError::InsufficientOrganizations {
            actual: organizations.len(),
            required: witness_policy.minimum_distinct_organizations,
        });
    }
    if failure_domains.len() < witness_policy.minimum_distinct_regions {
        violations.push(TrustSnapshotHeadObservationError::InsufficientFailureDomains {
            actual: failure_domains.len(),
            required: witness_policy.minimum_distinct_regions,
        });
    }
    if witness_policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(TrustSnapshotHeadObservationError::MissingAlgorithmDiversity);
    }

    if !valid_exact_verification_policy(exact_verification_policy) {
        violations.push(TrustSnapshotHeadObservationError::InvalidExactVerificationPolicy);
    }
    if exact_verification_providers.len() < exact_verification_policy.minimum_distinct_providers {
        violations.push(
            TrustSnapshotHeadObservationError::InsufficientExactVerificationProviders {
                actual: exact_verification_providers.len(),
                required: exact_verification_policy.minimum_distinct_providers,
            },
        );
    }
    if exact_verification_providers.len() > exact_verification_policy.maximum_providers
        || exact_verification_providers.len() > MAX_TRUST_SNAPSHOT_HEAD_EXACT_VERIFIERS
    {
        violations.push(
            TrustSnapshotHeadObservationError::TooManyExactVerificationProviders {
                actual: exact_verification_providers.len(),
                maximum: exact_verification_policy
                    .maximum_providers
                    .min(MAX_TRUST_SNAPSHOT_HEAD_EXACT_VERIFIERS),
            },
        );
        return Err(violations);
    }

    let mut verifier_commitments = Vec::with_capacity(exact_verification_providers.len());
    let mut seen_providers = BTreeSet::new();
    for provider in exact_verification_providers {
        let provider_id = provider.provider_id().to_string();
        let verification_policy_digest = provider.verification_policy_digest();
        if invalid_identifier(&provider_id)
            || verification_policy_digest == Sha256Digest([0; 32])
        {
            violations.push(TrustSnapshotHeadObservationError::InvalidExactVerificationProvider(
                provider_id,
            ));
            continue;
        }
        if !seen_providers.insert(provider_id.clone()) {
            violations.push(
                TrustSnapshotHeadObservationError::DuplicateExactVerificationProvider(provider_id),
            );
            continue;
        }
        verifier_commitments.push(ExactVerifierCommitment {
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
                TrustSnapshotHeadObservationError::CheckpointSignatureRejected(provider_id.clone()),
            ),
            Err(reason) => violations.push(
                TrustSnapshotHeadObservationError::VerificationProviderError {
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
                    TrustSnapshotHeadObservationError::WitnessSignatureRejected {
                        provider: provider_id.clone(),
                        key_id: witness.signature.key_id.clone(),
                    },
                ),
                Err(reason) => violations.push(
                    TrustSnapshotHeadObservationError::VerificationProviderError {
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
        vec![TrustSnapshotHeadObservationError::TransparencyLogInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let signed_checkpoint_evidence_digest = hash_serializable(
        SIGNED_CHECKPOINT_EVIDENCE_DOMAIN,
        signed_checkpoint,
    )
    .map_err(|error| vec![error])?;
    witness_evidence_digests.sort();
    let witness_set_evidence_digest = digest_witness_evidence_set(&witness_evidence_digests);
    verifier_commitments.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
    let exact_verification_policy_digest = hash_serializable(
        EXACT_VERIFICATION_POLICY_DOMAIN,
        exact_verification_policy,
    )
    .map_err(|error| vec![error])?;
    let exact_verifier_set_digest = hash_serializable(
        EXACT_VERIFIER_SET_DOMAIN,
        &verifier_commitments,
    )
    .map_err(|error| vec![error])?;

    let (observation_clock_lineage_digest, observation_clock_hop_count) =
        digest_observation_clock_lineage(
            activation_basis,
            observation_clock_bridge,
            observation_basis,
        )
        .map_err(|error| vec![error])?;

    let commitment = ObservedTrustSnapshotHeadCommitment {
        schema: QUORUM_OBSERVED_TRUST_SNAPSHOT_HEAD_SCHEMA,
        snapshot_digest: snapshot_digest.to_hex(),
        snapshot_sequence: snapshot.sequence,
        predecessor_snapshot_digest: rotation.current_snapshot_digest().to_hex(),
        rotation_id: rotation.id().to_hex(),
        activation_permit_id: activation.id().to_hex(),
        publication_digest: publication_digest.to_hex(),
        publication_entry_sequence: publication_entry.0,
        transparency_log_digest: transparency_log_digest.to_hex(),
        transparency_log_size: log.entries.len() as u64,
        transparency_root_digest: log_root.to_hex(),
        checkpoint_digest: signed_checkpoint.checkpoint_digest.to_hex(),
        signed_checkpoint_evidence_digest: signed_checkpoint_evidence_digest.to_hex(),
        witness_set_evidence_digest: witness_set_evidence_digest.to_hex(),
        witness_policy_digest: witness_policy_digest.to_hex(),
        witness_registry_id: witness_registry.id().to_hex(),
        witness_registry_digest: witness_registry.registry_digest().to_hex(),
        witness_registry_sequence: witness_registry.sequence(),
        exact_verification_policy_digest: exact_verification_policy_digest.to_hex(),
        exact_verifier_set_digest: exact_verifier_set_digest.to_hex(),
        exact_verifier_count: verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        containment_state_digest: containment_state_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        activation_clock_envelope_id: activation_clock.id().to_hex(),
        activation_operational_basis_id: activation_basis.id().to_hex(),
        observation_clock_envelope_id: observation_clock.id().to_hex(),
        observation_operational_basis_id: observation_basis.id().to_hex(),
        observation_clock_lineage_digest: observation_clock_lineage_digest.to_hex(),
        observation_clock_hop_count,
    };
    let id = QuorumObservedTrustSnapshotHeadIdV1(
        hash_serializable(OBSERVED_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(QuorumObservedTrustSnapshotHeadV1 {
        id,
        snapshot_digest,
        snapshot_sequence: snapshot.sequence,
        predecessor_snapshot_digest: rotation.current_snapshot_digest(),
        rotation_id: rotation.id(),
        activation_permit_id: activation.id(),
        publication_digest,
        publication_entry_sequence: publication_entry.0,
        transparency_log_digest,
        transparency_log_size: log.entries.len() as u64,
        transparency_root_digest: log_root,
        checkpoint_digest: signed_checkpoint.checkpoint_digest,
        signed_checkpoint_evidence_digest,
        witness_set_evidence_digest,
        witness_policy_digest,
        witness_registry_id: witness_registry.id(),
        witness_registry_digest: witness_registry.registry_digest(),
        witness_registry_sequence: witness_registry.sequence(),
        exact_verification_policy_digest,
        exact_verifier_set_digest,
        exact_verifier_count: verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        containment_state_digest,
        compromise_tracker_digest,
        activation_clock_envelope_id: activation_clock.id(),
        activation_operational_basis_id: activation_basis.id(),
        observation_clock_envelope_id: observation_clock.id(),
        observation_operational_basis_id: observation_basis.id(),
        observation_clock_lineage_digest,
        observation_clock_hop_count,
    })
}

fn require_rotation_activation_match(
    rotation: &ClockGovernedTrustRotationV1,
    activation: &ClockGovernedTrustSnapshotActivationPermitV1,
) -> Result<(), TrustSnapshotHeadObservationError> {
    if activation.rotation_id() != rotation.id()
        || activation.proposed_snapshot_digest() != rotation.proposed_snapshot_digest()
        || activation.proposed_snapshot_sequence() != rotation.proposed_snapshot().sequence
        || activation.activates_at_unix_ms() != rotation.activates_at_unix_ms()
    {
        return Err(TrustSnapshotHeadObservationError::RotationActivationMismatch);
    }
    Ok(())
}

fn validate_publication(
    publication: &ActivatedTrustSnapshotHeadPublicationV1,
) -> Result<(), TrustSnapshotHeadObservationError> {
    if publication.schema_version != ACTIVATED_TRUST_SNAPSHOT_HEAD_PUBLICATION_SCHEMA
        || publication.snapshot_digest == Sha256Digest([0; 32])
        || publication.predecessor_snapshot_digest == Sha256Digest([0; 32])
        || publication.snapshot_sequence == 0
        || publication.rotation_id.len() != 64
        || publication.activation_permit_id.len() != 64
        || publication.activation_clock_envelope_id.len() != 64
        || publication.activation_operational_basis_id.len() != 64
        || publication.snapshot_issued_at_unix_s >= publication.snapshot_expires_at_unix_s
    {
        return Err(TrustSnapshotHeadObservationError::InvalidPublication);
    }
    Ok(())
}

fn inspect_trust_head_log(
    log: &TransparencyLog,
    candidate_sequence: u64,
) -> Result<
    (Option<u64>, Option<(u64, Sha256Digest, u64)>),
    Vec<TrustSnapshotHeadObservationError>,
> {
    let mut violations = Vec::new();
    let mut previous_sequence = None;
    let mut latest_sequence = None;
    let mut candidate_entry = None;
    for entry in &log.entries {
        let sequence = match parse_trust_snapshot_head_sequence(&entry.kind) {
            Ok(Some(value)) => value,
            Ok(None) => continue,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if let Some(previous) = previous_sequence {
            if sequence <= previous {
                if sequence == previous {
                    violations.push(
                        TrustSnapshotHeadObservationError::DuplicateTrustSnapshotSequence(sequence),
                    );
                } else {
                    violations.push(
                        TrustSnapshotHeadObservationError::TrustSnapshotSequenceRegressed {
                            previous,
                            current: sequence,
                        },
                    );
                }
            }
        }
        previous_sequence = Some(sequence);
        latest_sequence = Some(latest_sequence.map_or(sequence, |latest: u64| latest.max(sequence)));
        if sequence == candidate_sequence {
            if candidate_entry.is_some() {
                violations.push(
                    TrustSnapshotHeadObservationError::DuplicateTrustSnapshotSequence(sequence),
                );
            } else {
                candidate_entry = Some((
                    entry.sequence,
                    entry.subject_digest,
                    entry.recorded_at_unix_s,
                ));
            }
        }
    }
    if violations.is_empty() {
        Ok((latest_sequence, candidate_entry))
    } else {
        Err(violations)
    }
}

fn parse_trust_snapshot_head_sequence(
    kind: &str,
) -> Result<Option<u64>, TrustSnapshotHeadObservationError> {
    let Some(suffix) = kind.strip_prefix(TRUST_SNAPSHOT_HEAD_LOG_KIND_PREFIX) else {
        return Ok(None);
    };
    let sequence = suffix.parse::<u64>().map_err(|_| {
        TrustSnapshotHeadObservationError::MalformedTrustSnapshotHeadKind(kind.to_string())
    })?;
    if sequence == 0 || suffix != sequence.to_string() {
        return Err(TrustSnapshotHeadObservationError::MalformedTrustSnapshotHeadKind(
            kind.to_string(),
        ));
    }
    Ok(Some(sequence))
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), TrustSnapshotHeadObservationError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(TrustSnapshotHeadObservationError::BrokenClockLineage {
            hop: 1,
            expected_predecessor: prior_basis_id.to_hex(),
            actual_predecessor: bridge[0]
                .predecessor_operational_basis_id()
                .map(|value| value.to_hex()),
        });
    }
    let mut expected = prior_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(TrustSnapshotHeadObservationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(TrustSnapshotHeadObservationError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn digest_observation_clock_lineage(
    activation_basis: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
) -> Result<(Sha256Digest, usize), TrustSnapshotHeadObservationError> {
    let mut lineage = Vec::with_capacity(bridge.len() + 2);
    lineage.push(activation_basis.id().to_hex());
    lineage.extend(bridge.iter().map(|basis| basis.id().to_hex()));
    if observation_basis.id() != activation_basis.id() {
        lineage.push(observation_basis.id().to_hex());
    }
    let digest = hash_serializable(OBSERVATION_CLOCK_LINEAGE_DOMAIN, &lineage)?;
    let hops = if observation_basis.id() == activation_basis.id() {
        0
    } else {
        bridge.len() + 1
    };
    Ok((digest, hops))
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
) -> Result<Sha256Digest, TrustSnapshotHeadObservationError> {
    if !valid_witness_policy(policy) {
        return Err(TrustSnapshotHeadObservationError::WitnessPolicyInvalid);
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
    policy: &ExactTrustSnapshotHeadVerificationPolicyV1,
) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_TRUST_SNAPSHOT_HEAD_EXACT_VERIFIERS
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
    violations: &mut Vec<TrustSnapshotHeadObservationError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(TrustSnapshotHeadObservationError::SignerUnknown(
            key_id.to_string(),
        ));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(TrustSnapshotHeadObservationError::SignerNotActive(
            key_id.to_string(),
        ));
    }
    if !record.usages.contains(&usage) {
        violations.push(TrustSnapshotHeadObservationError::SignerUsageNotAllowed(
            key_id.to_string(),
        ));
    }
    if record.not_before_unix_s > evidence_time_unix_s
        || record
            .not_after_unix_s
            .is_some_and(|not_after| evidence_time_unix_s >= not_after)
    {
        violations.push(TrustSnapshotHeadObservationError::SignerInvalidAtEvidenceTime(
            key_id.to_string(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(TrustSnapshotHeadObservationError::SignerNotValidAcrossEnvelope {
            key_id: key_id.to_string(),
            reason,
        });
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
                TrustSnapshotHeadObservationError::SignerCompromisedAcrossEnvelope(
                    key_id.to_string(),
                ),
            ),
            Err(reason) => violations.push(
                TrustSnapshotHeadObservationError::CompromiseTimeInvalid {
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

fn seconds_to_millis(value: u64) -> Result<u64, TrustSnapshotHeadObservationError> {
    value
        .checked_mul(1_000)
        .ok_or(TrustSnapshotHeadObservationError::TimeScaleOverflow)
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
) -> Result<Sha256Digest, TrustSnapshotHeadObservationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| TrustSnapshotHeadObservationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fabrication_kernel::crypto_digest::sha256;

    #[test]
    fn log_kind_round_trips_canonical_sequence() {
        let kind = trust_snapshot_head_log_kind(42).unwrap();
        assert_eq!(parse_trust_snapshot_head_sequence(&kind).unwrap(), Some(42));
        assert!(parse_trust_snapshot_head_sequence("trust-snapshot-head-v1:042").is_err());
        assert!(parse_trust_snapshot_head_sequence("trust-snapshot-head-v1:0").is_err());
    }

    #[test]
    fn trust_head_log_rejects_sequence_regression() {
        let mut log = TransparencyLog::default();
        log.append(100, trust_snapshot_head_log_kind(2).unwrap(), sha256(b"two"))
            .unwrap();
        log.append(101, trust_snapshot_head_log_kind(1).unwrap(), sha256(b"one"))
            .unwrap();
        let errors = inspect_trust_head_log(&log, 2).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            TrustSnapshotHeadObservationError::TrustSnapshotSequenceRegressed {
                previous: 2,
                current: 1
            }
        )));
    }

    #[test]
    fn trust_head_log_rejects_duplicate_sequence() {
        let mut log = TransparencyLog::default();
        log.append(100, trust_snapshot_head_log_kind(3).unwrap(), sha256(b"a"))
            .unwrap();
        log.append(101, trust_snapshot_head_log_kind(3).unwrap(), sha256(b"b"))
            .unwrap();
        let errors = inspect_trust_head_log(&log, 3).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            TrustSnapshotHeadObservationError::DuplicateTrustSnapshotSequence(3)
        )));
    }

    #[test]
    fn exact_verification_defaults_to_two_providers() {
        let policy = ExactTrustSnapshotHeadVerificationPolicyV1::default();
        assert_eq!(policy.minimum_distinct_providers, 2);
        assert!(valid_exact_verification_policy(&policy));
    }
}
