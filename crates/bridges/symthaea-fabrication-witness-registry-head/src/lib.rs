// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Witnessed monotonic currentness for activated transparency-witness registries.
//!
//! Registry creation and activation are deliberately separate from distributed currentness. This
//! bridge accepts only an opaque `ActivatedWitnessAuthorityRegistryV1`, publishes its canonical
//! sequence into transparency after activation, and uses that already-live registry to govern the
//! independent witnesses that observe the resulting checkpoint.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_fabrication_containment_state_authority::{
    ClockGovernedContainmentStateIdV1, ClockGovernedContainmentStateV1,
};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
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
use symthaea_fabrication_witness_registry_activation::{
    ActivatedWitnessAuthorityRegistryIdV1, ActivatedWitnessAuthorityRegistryV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const ACTIVATED_WITNESS_REGISTRY_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.activated-witness-registry-head-publication.v1";
pub const QUORUM_OBSERVED_WITNESS_REGISTRY_HEAD_SCHEMA: &str =
    "symthaea.fabrication.quorum-observed-witness-registry-head.v1";
pub const WITNESS_REGISTRY_HEAD_LOG_KIND_PREFIX: &str =
    "witness-authority-registry-head-v1:";
pub const MAX_WITNESS_REGISTRY_HEAD_CLOCK_HOPS: usize = 4096;
pub const MAX_WITNESS_REGISTRY_HEAD_CONTAINMENT_HOPS: usize = 4096;
pub const MAX_WITNESS_REGISTRY_HEAD_EXACT_VERIFIERS: usize = 16;

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.activated-witness-registry-head-publication.v1\0";
const SIGNED_CHECKPOINT_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-checkpoint-evidence.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-witness-set.v1\0";
const WITNESS_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-witness-policy.v1\0";
const EXACT_VERIFICATION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-exact-verification-policy.v1\0";
const EXACT_VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-exact-verifier-set.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-clock-lineage.v1\0";
const CONTAINMENT_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-registry-head-containment-lineage.v1\0";
const OBSERVED_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.quorum-observed-witness-registry-head.v1\0";
const CHECKPOINT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-checkpoint-signature.v1\0";
const WITNESS_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-signature.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivatedWitnessRegistryHeadPublicationV1 {
    pub schema_version: String,
    pub activated_registry_id: String,
    pub registry_digest: Sha256Digest,
    pub sequence: u64,
    pub previous_registry_id: String,
    pub previous_registry_digest: Sha256Digest,
    pub previous_sequence: u64,
    pub transition_id: String,
    pub activates_at_unix_ms: u64,
    pub activation_clock_envelope_id: String,
    pub activation_operational_basis_id: String,
    pub activation_containment_authority_id: String,
    pub activation_containment_state_digest: Sha256Digest,
    pub activation_containment_generation: u64,
    pub activation_compromise_tracker_digest: Sha256Digest,
    pub activation_trust_head_id: String,
    pub activation_trust_snapshot_digest: Sha256Digest,
    pub activation_trust_snapshot_sequence: u64,
    pub activation_containment_head_id: String,
}

pub trait ExactWitnessRegistryHeadEvidenceVerifierV1 {
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
pub struct ExactWitnessRegistryHeadVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactWitnessRegistryHeadVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuorumObservedWitnessRegistryHeadIdV1(Sha256Digest);

impl QuorumObservedWitnessRegistryHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct QuorumObservedWitnessRegistryHeadV1 {
    id: QuorumObservedWitnessRegistryHeadIdV1,
    activated_registry_id: ActivatedWitnessAuthorityRegistryIdV1,
    registry_digest: Sha256Digest,
    sequence: u64,
    previous_registry_digest: Sha256Digest,
    previous_sequence: u64,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    transparency_log_size: u64,
    transparency_root_digest: Sha256Digest,
    checkpoint_digest: Sha256Digest,
    signed_checkpoint_evidence_digest: Sha256Digest,
    witness_set_evidence_digest: Sha256Digest,
    witness_policy_digest: Sha256Digest,
    exact_verification_policy_digest: Sha256Digest,
    exact_verifier_set_digest: Sha256Digest,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    activation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
    activation_containment_authority_id: ClockGovernedContainmentStateIdV1,
    observation_containment_authority_id: ClockGovernedContainmentStateIdV1,
    observation_containment_state_digest: Sha256Digest,
    observation_containment_generation: u64,
    observation_compromise_tracker_digest: Sha256Digest,
    containment_lineage_digest: Sha256Digest,
    containment_hop_count: usize,
}

impl QuorumObservedWitnessRegistryHeadV1 {
    pub fn id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 {
        self.id
    }
    pub fn activated_registry_id(&self) -> ActivatedWitnessAuthorityRegistryIdV1 {
        self.activated_registry_id
    }
    pub fn registry_digest(&self) -> Sha256Digest {
        self.registry_digest
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn previous_registry_digest(&self) -> Sha256Digest {
        self.previous_registry_digest
    }
    pub fn previous_sequence(&self) -> u64 {
        self.previous_sequence
    }
    pub fn publication_digest(&self) -> Sha256Digest {
        self.publication_digest
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn trust_snapshot_sequence(&self) -> u64 {
        self.trust_snapshot_sequence
    }
    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.observation_clock_envelope_id
    }
    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.observation_operational_basis_id
    }
    pub fn observation_containment_authority_id(&self) -> ClockGovernedContainmentStateIdV1 {
        self.observation_containment_authority_id
    }
    pub fn observation_containment_state_digest(&self) -> Sha256Digest {
        self.observation_containment_state_digest
    }
    pub fn observation_containment_generation(&self) -> u64 {
        self.observation_containment_generation
    }
    pub fn observation_compromise_tracker_digest(&self) -> Sha256Digest {
        self.observation_compromise_tracker_digest
    }
    pub fn clock_lineage_digest(&self) -> Sha256Digest {
        self.clock_lineage_digest
    }
    pub fn containment_lineage_digest(&self) -> Sha256Digest {
        self.containment_lineage_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessRegistryHeadObservationError {
    InvalidPublication,
    PublicationMismatch,
    TrustSnapshotInvalid(String),
    TrustSnapshotMismatch,
    ActivationBasisMismatch,
    ActivationEnvelopeMismatch,
    ActivationContainmentMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    TooManyContainmentHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    BrokenContainmentLineage {
        hop: usize,
        expected_previous_authority: String,
        actual_previous_authority: Option<String>,
    },
    BrokenContainmentStateLineage { hop: usize },
    ContainmentGenerationOverflow,
    ContainmentGenerationNotAdjacent { previous: u64, current: u64 },
    Clock(ClockGovernanceTimeError),
    TrustSnapshotNotValidAcrossObservationEnvelope(ClockGovernanceTimeError),
    TransparencyLogInvalid(String),
    MalformedRegistryHeadKind(String),
    RegistrySequenceRegressed { previous: u64, current: u64 },
    DuplicateRegistrySequence(u64),
    PublicationNotFound,
    HigherRegistrySequencePublished { candidate: u64, latest: u64 },
    PublicationDigestMismatch,
    PublicationBeforeActivation,
    PublicationMayBeFuture,
    CheckpointInvalid(String),
    CheckpointLogMismatch,
    CheckpointPredatesLog,
    CheckpointPredatesPublication,
    SignerUnknown(String),
    SignerNotActive(String),
    SignerUsageNotAllowed(String),
    SignerInvalidAtEvidenceTime(String),
    SignerNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    SignerCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid { key_id: String, reason: ClockGovernanceTimeError },
    WitnessPolicyInvalid,
    TooManyWitnesses { actual: usize, maximum: usize },
    WitnessMalformed(String),
    WitnessCheckpointMismatch(String),
    WitnessBeforeCheckpoint(String),
    WitnessMayBeFuture(String),
    WitnessMayBeStale(String),
    DuplicateWitnessSigner(String),
    WitnessNotInActivatedRegistry(String),
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
struct ObservedRegistryHeadCommitment {
    schema: &'static str,
    activated_registry_id: String,
    registry_digest: String,
    sequence: u64,
    previous_registry_digest: String,
    previous_sequence: u64,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    transparency_log_size: u64,
    transparency_root_digest: String,
    checkpoint_digest: String,
    signed_checkpoint_evidence_digest: String,
    witness_set_evidence_digest: String,
    witness_policy_digest: String,
    exact_verification_policy_digest: String,
    exact_verifier_set_digest: String,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    trust_snapshot_digest: String,
    trust_snapshot_sequence: u64,
    activation_clock_envelope_id: String,
    activation_operational_basis_id: String,
    observation_clock_envelope_id: String,
    observation_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
    activation_containment_authority_id: String,
    observation_containment_authority_id: String,
    observation_containment_state_digest: String,
    observation_containment_generation: u64,
    observation_compromise_tracker_digest: String,
    containment_lineage_digest: String,
    containment_hop_count: usize,
}

pub fn build_activated_witness_registry_head_publication_v1(
    registry: &ActivatedWitnessAuthorityRegistryV1,
) -> Result<ActivatedWitnessRegistryHeadPublicationV1, WitnessRegistryHeadObservationError> {
    let publication = ActivatedWitnessRegistryHeadPublicationV1 {
        schema_version: ACTIVATED_WITNESS_REGISTRY_HEAD_PUBLICATION_SCHEMA.into(),
        activated_registry_id: registry.id().to_hex(),
        registry_digest: registry.registry_digest(),
        sequence: registry.sequence(),
        previous_registry_id: registry.previous_registry_id().to_hex(),
        previous_registry_digest: registry.previous_registry_digest(),
        previous_sequence: registry.previous_sequence(),
        transition_id: registry.transition_id().to_hex(),
        activates_at_unix_ms: registry.activates_at_unix_ms(),
        activation_clock_envelope_id: registry.activation_clock_envelope_id().to_hex(),
        activation_operational_basis_id: registry.activation_operational_basis_id().to_hex(),
        activation_containment_authority_id: registry.activation_containment_authority_id().to_hex(),
        activation_containment_state_digest: registry.activation_containment_state_digest(),
        activation_containment_generation: registry.activation_containment_generation(),
        activation_compromise_tracker_digest: registry.activation_compromise_tracker_digest(),
        activation_trust_head_id: registry.trust_head_id().to_hex(),
        activation_trust_snapshot_digest: registry.trust_snapshot_digest(),
        activation_trust_snapshot_sequence: registry.trust_snapshot_sequence(),
        activation_containment_head_id: registry.containment_head_id().to_hex(),
    };
    validate_publication(&publication)?;
    Ok(publication)
}

pub fn digest_activated_witness_registry_head_publication_v1(
    publication: &ActivatedWitnessRegistryHeadPublicationV1,
) -> Result<Sha256Digest, WitnessRegistryHeadObservationError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

pub fn witness_registry_head_log_kind(
    sequence: u64,
) -> Result<String, WitnessRegistryHeadObservationError> {
    if sequence == 0 {
        return Err(WitnessRegistryHeadObservationError::InvalidPublication);
    }
    Ok(format!("{WITNESS_REGISTRY_HEAD_LOG_KIND_PREFIX}{sequence}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_quorum_observed_witness_registry_head_v1(
    registry: &ActivatedWitnessAuthorityRegistryV1,
    activation_basis: &OperationalClockBasisV1,
    clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    activation_containment_authority: &ClockGovernedContainmentStateV1,
    containment_authority_bridge: &[ClockGovernedContainmentStateV1],
    observation_containment_authority: &ClockGovernedContainmentStateV1,
    trust_snapshot: &TrustSnapshot,
    publication: &ActivatedWitnessRegistryHeadPublicationV1,
    log: &TransparencyLog,
    signed_checkpoint: &SignedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    witness_policy: &TransparencyWitnessPolicy,
    exact_verification_policy: &ExactWitnessRegistryHeadVerificationPolicyV1,
    exact_verification_providers: &[&dyn ExactWitnessRegistryHeadEvidenceVerifierV1],
) -> Result<QuorumObservedWitnessRegistryHeadV1, Vec<WitnessRegistryHeadObservationError>> {
    let mut violations = Vec::new();

    let expected_publication = match build_activated_witness_registry_head_publication_v1(registry) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if publication != &expected_publication {
        violations.push(WitnessRegistryHeadObservationError::PublicationMismatch);
    }
    let publication_digest = match digest_activated_witness_registry_head_publication_v1(publication)
    {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };

    if activation_basis.id() != registry.activation_operational_basis_id() {
        violations.push(WitnessRegistryHeadObservationError::ActivationBasisMismatch);
    }
    let activation_clock = match derive_clock_governance_evaluation_envelope_v1(activation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryHeadObservationError::Clock(error));
            return Err(violations);
        }
    };
    if activation_clock.id() != registry.activation_clock_envelope_id() {
        violations.push(WitnessRegistryHeadObservationError::ActivationEnvelopeMismatch);
    }
    if activation_containment_authority.id() != registry.activation_containment_authority_id()
        || activation_containment_authority.state_digest()
            != registry.activation_containment_state_digest()
        || activation_containment_authority.generation()
            != registry.activation_containment_generation()
        || activation_containment_authority.compromise_tracker_digest()
            != registry.activation_compromise_tracker_digest()
    {
        violations.push(WitnessRegistryHeadObservationError::ActivationContainmentMismatch);
    }

    if clock_bridge.len() > MAX_WITNESS_REGISTRY_HEAD_CLOCK_HOPS {
        violations.push(WitnessRegistryHeadObservationError::TooManyClockHops {
            actual: clock_bridge.len(),
            maximum: MAX_WITNESS_REGISTRY_HEAD_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        activation_basis.id(),
        clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }
    if containment_authority_bridge.len() > MAX_WITNESS_REGISTRY_HEAD_CONTAINMENT_HOPS {
        violations.push(WitnessRegistryHeadObservationError::TooManyContainmentHops {
            actual: containment_authority_bridge.len(),
            maximum: MAX_WITNESS_REGISTRY_HEAD_CONTAINMENT_HOPS,
        });
    } else if let Err(error) = verify_containment_lineage(
        activation_containment_authority,
        containment_authority_bridge,
        observation_containment_authority,
    ) {
        violations.push(error);
    }

    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryHeadObservationError::Clock(error));
            return Err(violations);
        }
    };
    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryHeadObservationError::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(WitnessRegistryHeadObservationError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if trust_snapshot_digest != registry.trust_snapshot_digest()
        || trust_snapshot.sequence != registry.trust_snapshot_sequence()
    {
        violations.push(WitnessRegistryHeadObservationError::TrustSnapshotMismatch);
    }
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            WitnessRegistryHeadObservationError::TrustSnapshotNotValidAcrossObservationEnvelope(
                reason,
            ),
        );
    }

    if let Err(error) = log.validate() {
        violations.push(WitnessRegistryHeadObservationError::TransparencyLogInvalid(format!(
            "{error:?}"
        )));
    }
    let (latest_sequence, publication_entry) = match inspect_registry_head_log(log, registry.sequence())
    {
        Ok(value) => value,
        Err(errors) => {
            violations.extend(errors);
            (None, None)
        }
    };
    if let Some(latest) = latest_sequence {
        if latest > registry.sequence() {
            violations.push(WitnessRegistryHeadObservationError::HigherRegistrySequencePublished {
                candidate: registry.sequence(),
                latest,
            });
        }
    }
    let Some(publication_entry) = publication_entry else {
        violations.push(WitnessRegistryHeadObservationError::PublicationNotFound);
        return Err(violations);
    };
    if publication_entry.1 != publication_digest {
        violations.push(WitnessRegistryHeadObservationError::PublicationDigestMismatch);
    }
    let publication_recorded_at_ms = match seconds_to_millis(publication_entry.2) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if publication_recorded_at_ms < registry.activates_at_unix_ms() {
        violations.push(WitnessRegistryHeadObservationError::PublicationBeforeActivation);
    }
    if publication_recorded_at_ms > observation_clock.lower_unix_ms() {
        violations.push(WitnessRegistryHeadObservationError::PublicationMayBeFuture);
    }

    if signed_checkpoint.schema_version != SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA {
        violations.push(WitnessRegistryHeadObservationError::CheckpointInvalid(
            "unsupported signed checkpoint schema".into(),
        ));
    }
    if let Err(error) = signed_checkpoint.checkpoint.validate() {
        violations.push(WitnessRegistryHeadObservationError::CheckpointInvalid(format!(
            "{error:?}"
        )));
    }
    match digest_transparency_checkpoint(&signed_checkpoint.checkpoint) {
        Ok(value) if value == signed_checkpoint.checkpoint_digest => {}
        Ok(_) => violations.push(WitnessRegistryHeadObservationError::CheckpointInvalid(
            "checkpoint digest mismatch".into(),
        )),
        Err(error) => violations.push(WitnessRegistryHeadObservationError::CheckpointInvalid(
            format!("{error:?}"),
        )),
    }
    let log_root = match log.root() {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryHeadObservationError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if signed_checkpoint.checkpoint.log_size != log.entries.len() as u64
        || signed_checkpoint.checkpoint.root_digest != log_root
    {
        violations.push(WitnessRegistryHeadObservationError::CheckpointLogMismatch);
    }
    if log
        .entries
        .last()
        .is_some_and(|entry| entry.recorded_at_unix_s > signed_checkpoint.checkpoint.issued_at_unix_s)
    {
        violations.push(WitnessRegistryHeadObservationError::CheckpointPredatesLog);
    }
    if publication_entry.2 > signed_checkpoint.checkpoint.issued_at_unix_s {
        violations.push(WitnessRegistryHeadObservationError::CheckpointPredatesPublication);
    }
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        signed_checkpoint.checkpoint.issued_at_unix_s,
        signed_checkpoint.checkpoint.expires_at_unix_s,
    ) {
        violations.push(WitnessRegistryHeadObservationError::CheckpointInvalid(format!(
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
        violations.push(WitnessRegistryHeadObservationError::CheckpointInvalid(
            "invalid checkpoint signer or signature bytes".into(),
        ));
    }
    requalify_signer(
        &signed_checkpoint.signature.algorithm,
        &signed_checkpoint.signature.key_id,
        KeyUsage::TransparencyLog,
        signed_checkpoint.checkpoint.issued_at_unix_s,
        trust_snapshot,
        observation_containment_authority,
        &observation_clock,
        &mut violations,
    );

    if !valid_witness_policy(witness_policy) {
        violations.push(WitnessRegistryHeadObservationError::WitnessPolicyInvalid);
    }
    if signed_witnesses.len() > witness_policy.maximum_witnesses
        || signed_witnesses.len() > MAX_TRANSPARENCY_WITNESSES
    {
        violations.push(WitnessRegistryHeadObservationError::TooManyWitnesses {
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
            violations.push(WitnessRegistryHeadObservationError::WitnessMalformed(key_id));
            continue;
        }
        if signed.statement.checkpoint_digest != signed_checkpoint.checkpoint_digest
            || signed.statement.checkpoint_log_size != signed_checkpoint.checkpoint.log_size
            || signed.statement.checkpoint_root_digest != signed_checkpoint.checkpoint.root_digest
        {
            violations.push(WitnessRegistryHeadObservationError::WitnessCheckpointMismatch(
                key_id,
            ));
            continue;
        }
        match digest_transparency_witness_statement(&signed.statement) {
            Ok(value) if value == signed.statement_digest => {}
            Ok(_) => {
                violations.push(WitnessRegistryHeadObservationError::WitnessMalformed(key_id));
                continue;
            }
            Err(error) => {
                violations.push(WitnessRegistryHeadObservationError::WitnessMalformed(format!(
                    "{}: {error:?}",
                    signed.signature.key_id
                )));
                continue;
            }
        }
        if signed.statement.observed_at_unix_s < signed_checkpoint.checkpoint.issued_at_unix_s {
            violations.push(WitnessRegistryHeadObservationError::WitnessBeforeCheckpoint(
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
            violations.push(WitnessRegistryHeadObservationError::WitnessMayBeFuture(
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
                violations.push(WitnessRegistryHeadObservationError::TimeScaleOverflow);
                continue;
            }
        };
        if observation_clock.upper_unix_ms() > freshness_deadline {
            violations.push(WitnessRegistryHeadObservationError::WitnessMayBeStale(
                signed.signature.key_id.clone(),
            ));
        }
        let signer = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signer_ids.insert(signer) {
            violations.push(WitnessRegistryHeadObservationError::DuplicateWitnessSigner(
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
            observation_containment_authority,
            &observation_clock,
            &mut violations,
        );

        let Some(profile) = registry.profile(
            &signed.signature.algorithm,
            &signed.signature.key_id,
        ) else {
            violations.push(WitnessRegistryHeadObservationError::WitnessNotInActivatedRegistry(
                signed.signature.key_id.clone(),
            ));
            continue;
        };
        if profile.organization != signed.statement.witness_organization {
            violations.push(WitnessRegistryHeadObservationError::WitnessOrganizationMismatch(
                signed.signature.key_id.clone(),
            ));
        }
        if profile.failure_domain != signed.statement.witness_region {
            violations.push(WitnessRegistryHeadObservationError::WitnessFailureDomainMismatch(
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
        violations.push(WitnessRegistryHeadObservationError::InsufficientWitnesses {
            actual: signer_ids.len(),
            required: witness_policy.minimum_distinct_witnesses,
        });
    }
    if organizations.len() < witness_policy.minimum_distinct_organizations {
        violations.push(WitnessRegistryHeadObservationError::InsufficientOrganizations {
            actual: organizations.len(),
            required: witness_policy.minimum_distinct_organizations,
        });
    }
    if failure_domains.len() < witness_policy.minimum_distinct_regions {
        violations.push(WitnessRegistryHeadObservationError::InsufficientFailureDomains {
            actual: failure_domains.len(),
            required: witness_policy.minimum_distinct_regions,
        });
    }
    if witness_policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(WitnessRegistryHeadObservationError::MissingAlgorithmDiversity);
    }

    if !valid_exact_verification_policy(exact_verification_policy) {
        violations.push(WitnessRegistryHeadObservationError::InvalidExactVerificationPolicy);
    }
    if exact_verification_providers.len() < exact_verification_policy.minimum_distinct_providers {
        violations.push(
            WitnessRegistryHeadObservationError::InsufficientExactVerificationProviders {
                actual: exact_verification_providers.len(),
                required: exact_verification_policy.minimum_distinct_providers,
            },
        );
    }
    if exact_verification_providers.len() > exact_verification_policy.maximum_providers
        || exact_verification_providers.len() > MAX_WITNESS_REGISTRY_HEAD_EXACT_VERIFIERS
    {
        violations.push(WitnessRegistryHeadObservationError::TooManyExactVerificationProviders {
            actual: exact_verification_providers.len(),
            maximum: exact_verification_policy
                .maximum_providers
                .min(MAX_WITNESS_REGISTRY_HEAD_EXACT_VERIFIERS),
        });
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
            violations.push(WitnessRegistryHeadObservationError::InvalidExactVerificationProvider(
                provider_id,
            ));
            continue;
        }
        if !seen_providers.insert(provider_id.clone()) {
            violations.push(WitnessRegistryHeadObservationError::DuplicateExactVerificationProvider(
                provider_id,
            ));
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
                WitnessRegistryHeadObservationError::CheckpointSignatureRejected(
                    provider_id.clone(),
                ),
            ),
            Err(reason) => violations.push(
                WitnessRegistryHeadObservationError::VerificationProviderError {
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
                    WitnessRegistryHeadObservationError::WitnessSignatureRejected {
                        provider: provider_id.clone(),
                        key_id: witness.signature.key_id.clone(),
                    },
                ),
                Err(reason) => violations.push(
                    WitnessRegistryHeadObservationError::VerificationProviderError {
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
        vec![WitnessRegistryHeadObservationError::TransparencyLogInvalid(format!(
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
    let witness_policy_digest = digest_witness_policy(witness_policy).map_err(|error| vec![error])?;
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
    let (clock_lineage_digest, clock_hop_count) = digest_clock_lineage(
        activation_basis,
        clock_bridge,
        observation_basis,
    )
    .map_err(|error| vec![error])?;
    let (containment_lineage_digest, containment_hop_count) = digest_containment_lineage(
        activation_containment_authority,
        containment_authority_bridge,
        observation_containment_authority,
    )
    .map_err(|error| vec![error])?;

    let commitment = ObservedRegistryHeadCommitment {
        schema: QUORUM_OBSERVED_WITNESS_REGISTRY_HEAD_SCHEMA,
        activated_registry_id: registry.id().to_hex(),
        registry_digest: registry.registry_digest().to_hex(),
        sequence: registry.sequence(),
        previous_registry_digest: registry.previous_registry_digest().to_hex(),
        previous_sequence: registry.previous_sequence(),
        publication_digest: publication_digest.to_hex(),
        publication_entry_sequence: publication_entry.0,
        transparency_log_digest: transparency_log_digest.to_hex(),
        transparency_log_size: log.entries.len() as u64,
        transparency_root_digest: log_root.to_hex(),
        checkpoint_digest: signed_checkpoint.checkpoint_digest.to_hex(),
        signed_checkpoint_evidence_digest: signed_checkpoint_evidence_digest.to_hex(),
        witness_set_evidence_digest: witness_set_evidence_digest.to_hex(),
        witness_policy_digest: witness_policy_digest.to_hex(),
        exact_verification_policy_digest: exact_verification_policy_digest.to_hex(),
        exact_verifier_set_digest: exact_verifier_set_digest.to_hex(),
        exact_verifier_count: verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        trust_snapshot_sequence: trust_snapshot.sequence,
        activation_clock_envelope_id: activation_clock.id().to_hex(),
        activation_operational_basis_id: activation_basis.id().to_hex(),
        observation_clock_envelope_id: observation_clock.id().to_hex(),
        observation_operational_basis_id: observation_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count,
        activation_containment_authority_id: activation_containment_authority.id().to_hex(),
        observation_containment_authority_id: observation_containment_authority.id().to_hex(),
        observation_containment_state_digest: observation_containment_authority.state_digest().to_hex(),
        observation_containment_generation: observation_containment_authority.generation(),
        observation_compromise_tracker_digest: observation_containment_authority
            .compromise_tracker_digest()
            .to_hex(),
        containment_lineage_digest: containment_lineage_digest.to_hex(),
        containment_hop_count,
    };
    let id = QuorumObservedWitnessRegistryHeadIdV1(
        hash_serializable(OBSERVED_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(QuorumObservedWitnessRegistryHeadV1 {
        id,
        activated_registry_id: registry.id(),
        registry_digest: registry.registry_digest(),
        sequence: registry.sequence(),
        previous_registry_digest: registry.previous_registry_digest(),
        previous_sequence: registry.previous_sequence(),
        publication_digest,
        publication_entry_sequence: publication_entry.0,
        transparency_log_digest,
        transparency_log_size: log.entries.len() as u64,
        transparency_root_digest: log_root,
        checkpoint_digest: signed_checkpoint.checkpoint_digest,
        signed_checkpoint_evidence_digest,
        witness_set_evidence_digest,
        witness_policy_digest,
        exact_verification_policy_digest,
        exact_verifier_set_digest,
        exact_verifier_count: verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        trust_snapshot_digest,
        trust_snapshot_sequence: trust_snapshot.sequence,
        activation_clock_envelope_id: activation_clock.id(),
        activation_operational_basis_id: activation_basis.id(),
        observation_clock_envelope_id: observation_clock.id(),
        observation_operational_basis_id: observation_basis.id(),
        clock_lineage_digest,
        clock_hop_count,
        activation_containment_authority_id: activation_containment_authority.id(),
        observation_containment_authority_id: observation_containment_authority.id(),
        observation_containment_state_digest: observation_containment_authority.state_digest(),
        observation_containment_generation: observation_containment_authority.generation(),
        observation_compromise_tracker_digest: observation_containment_authority
            .compromise_tracker_digest(),
        containment_lineage_digest,
        containment_hop_count,
    })
}

fn validate_publication(
    publication: &ActivatedWitnessRegistryHeadPublicationV1,
) -> Result<(), WitnessRegistryHeadObservationError> {
    if publication.schema_version != ACTIVATED_WITNESS_REGISTRY_HEAD_PUBLICATION_SCHEMA
        || publication.activated_registry_id.len() != 64
        || publication.registry_digest == Sha256Digest([0; 32])
        || publication.sequence == 0
        || publication.previous_registry_id.len() != 64
        || publication.previous_registry_digest == Sha256Digest([0; 32])
        || publication.previous_sequence == 0
        || publication.previous_sequence.checked_add(1) != Some(publication.sequence)
        || publication.transition_id.len() != 64
        || publication.activates_at_unix_ms == 0
        || publication.activation_clock_envelope_id.len() != 64
        || publication.activation_operational_basis_id.len() != 64
        || publication.activation_containment_authority_id.len() != 64
        || publication.activation_containment_state_digest == Sha256Digest([0; 32])
        || publication.activation_containment_generation == 0
        || publication.activation_compromise_tracker_digest == Sha256Digest([0; 32])
        || publication.activation_trust_head_id.len() != 64
        || publication.activation_trust_snapshot_digest == Sha256Digest([0; 32])
        || publication.activation_trust_snapshot_sequence == 0
        || publication.activation_containment_head_id.len() != 64
    {
        return Err(WitnessRegistryHeadObservationError::InvalidPublication);
    }
    Ok(())
}

fn inspect_registry_head_log(
    log: &TransparencyLog,
    candidate_sequence: u64,
) -> Result<
    (Option<u64>, Option<(u64, Sha256Digest, u64)>),
    Vec<WitnessRegistryHeadObservationError>,
> {
    let mut violations = Vec::new();
    let mut previous_sequence = None;
    let mut latest_sequence = None;
    let mut candidate_entry = None;
    for entry in &log.entries {
        let sequence = match parse_registry_head_sequence(&entry.kind) {
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
                    violations.push(WitnessRegistryHeadObservationError::DuplicateRegistrySequence(
                        sequence,
                    ));
                } else {
                    violations.push(WitnessRegistryHeadObservationError::RegistrySequenceRegressed {
                        previous,
                        current: sequence,
                    });
                }
            }
        }
        previous_sequence = Some(sequence);
        latest_sequence = Some(latest_sequence.map_or(sequence, |latest: u64| latest.max(sequence)));
        if sequence == candidate_sequence {
            if candidate_entry.is_some() {
                violations.push(WitnessRegistryHeadObservationError::DuplicateRegistrySequence(
                    sequence,
                ));
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

fn parse_registry_head_sequence(
    kind: &str,
) -> Result<Option<u64>, WitnessRegistryHeadObservationError> {
    let Some(suffix) = kind.strip_prefix(WITNESS_REGISTRY_HEAD_LOG_KIND_PREFIX) else {
        return Ok(None);
    };
    let sequence = suffix.parse::<u64>().map_err(|_| {
        WitnessRegistryHeadObservationError::MalformedRegistryHeadKind(kind.to_string())
    })?;
    if sequence == 0 || suffix != sequence.to_string() {
        return Err(WitnessRegistryHeadObservationError::MalformedRegistryHeadKind(
            kind.to_string(),
        ));
    }
    Ok(Some(sequence))
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), WitnessRegistryHeadObservationError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(WitnessRegistryHeadObservationError::BrokenClockLineage {
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
            return Err(WitnessRegistryHeadObservationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(WitnessRegistryHeadObservationError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn verify_containment_lineage(
    start: &ClockGovernedContainmentStateV1,
    bridge: &[ClockGovernedContainmentStateV1],
    current: &ClockGovernedContainmentStateV1,
) -> Result<(), WitnessRegistryHeadObservationError> {
    if current.id() == start.id() {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(WitnessRegistryHeadObservationError::BrokenContainmentLineage {
            hop: 1,
            expected_previous_authority: start.id().to_hex(),
            actual_previous_authority: bridge[0]
                .previous_authority_id()
                .map(|value| value.to_hex()),
        });
    }
    let mut previous = start;
    for (index, candidate) in bridge.iter().enumerate() {
        verify_one_containment_hop(previous, candidate, index + 1)?;
        previous = candidate;
    }
    verify_one_containment_hop(previous, current, bridge.len() + 1)
}

fn verify_one_containment_hop(
    previous: &ClockGovernedContainmentStateV1,
    current: &ClockGovernedContainmentStateV1,
    hop: usize,
) -> Result<(), WitnessRegistryHeadObservationError> {
    if current.previous_authority_id() != Some(previous.id()) {
        return Err(WitnessRegistryHeadObservationError::BrokenContainmentLineage {
            hop,
            expected_previous_authority: previous.id().to_hex(),
            actual_previous_authority: current
                .previous_authority_id()
                .map(|value| value.to_hex()),
        });
    }
    if current.previous_state_digest() != Some(previous.state_digest()) {
        return Err(WitnessRegistryHeadObservationError::BrokenContainmentStateLineage { hop });
    }
    let expected_generation = previous
        .generation()
        .checked_add(1)
        .ok_or(WitnessRegistryHeadObservationError::ContainmentGenerationOverflow)?;
    if current.generation() != expected_generation {
        return Err(WitnessRegistryHeadObservationError::ContainmentGenerationNotAdjacent {
            previous: previous.generation(),
            current: current.generation(),
        });
    }
    Ok(())
}

fn digest_clock_lineage(
    start: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    end: &OperationalClockBasisV1,
) -> Result<(Sha256Digest, usize), WitnessRegistryHeadObservationError> {
    let mut ids = Vec::with_capacity(bridge.len() + 2);
    ids.push(start.id().to_hex());
    ids.extend(bridge.iter().map(|basis| basis.id().to_hex()));
    if end.id() != start.id() {
        ids.push(end.id().to_hex());
    }
    let digest = hash_serializable(CLOCK_LINEAGE_DOMAIN, &ids)?;
    let hops = if end.id() == start.id() {
        0
    } else {
        bridge.len() + 1
    };
    Ok((digest, hops))
}

fn digest_containment_lineage(
    start: &ClockGovernedContainmentStateV1,
    bridge: &[ClockGovernedContainmentStateV1],
    end: &ClockGovernedContainmentStateV1,
) -> Result<(Sha256Digest, usize), WitnessRegistryHeadObservationError> {
    let mut ids = Vec::with_capacity(bridge.len() + 2);
    ids.push(start.id().to_hex());
    ids.extend(bridge.iter().map(|state| state.id().to_hex()));
    if end.id() != start.id() {
        ids.push(end.id().to_hex());
    }
    let digest = hash_serializable(CONTAINMENT_LINEAGE_DOMAIN, &ids)?;
    let hops = if end.id() == start.id() {
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
) -> Result<Sha256Digest, WitnessRegistryHeadObservationError> {
    if !valid_witness_policy(policy) {
        return Err(WitnessRegistryHeadObservationError::WitnessPolicyInvalid);
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
    policy: &ExactWitnessRegistryHeadVerificationPolicyV1,
) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_WITNESS_REGISTRY_HEAD_EXACT_VERIFIERS
}

#[allow(clippy::too_many_arguments)]
fn requalify_signer(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    usage: KeyUsage,
    evidence_time_unix_s: u64,
    trust_snapshot: &TrustSnapshot,
    containment_authority: &ClockGovernedContainmentStateV1,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<WitnessRegistryHeadObservationError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(WitnessRegistryHeadObservationError::SignerUnknown(
            key_id.to_string(),
        ));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(WitnessRegistryHeadObservationError::SignerNotActive(
            key_id.to_string(),
        ));
    }
    if !record.usages.contains(&usage) {
        violations.push(WitnessRegistryHeadObservationError::SignerUsageNotAllowed(
            key_id.to_string(),
        ));
    }
    if record.not_before_unix_s > evidence_time_unix_s
        || record
            .not_after_unix_s
            .is_some_and(|not_after| evidence_time_unix_s >= not_after)
    {
        violations.push(WitnessRegistryHeadObservationError::SignerInvalidAtEvidenceTime(
            key_id.to_string(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(WitnessRegistryHeadObservationError::SignerNotValidAcrossEnvelope {
            key_id: key_id.to_string(),
            reason,
        });
    }
    for compromise in containment_authority
        .compromise_tracker()
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
                WitnessRegistryHeadObservationError::SignerCompromisedAcrossEnvelope(
                    key_id.to_string(),
                ),
            ),
            Err(reason) => violations.push(WitnessRegistryHeadObservationError::CompromiseTimeInvalid {
                key_id: key_id.to_string(),
                reason,
            }),
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

fn seconds_to_millis(value: u64) -> Result<u64, WitnessRegistryHeadObservationError> {
    value
        .checked_mul(1_000)
        .ok_or(WitnessRegistryHeadObservationError::TimeScaleOverflow)
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
) -> Result<Sha256Digest, WitnessRegistryHeadObservationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| WitnessRegistryHeadObservationError::Encoding(error.to_string()))?;
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
    fn registry_log_kind_round_trips_canonical_sequence() {
        let kind = witness_registry_head_log_kind(2).unwrap();
        assert_eq!(parse_registry_head_sequence(&kind).unwrap(), Some(2));
        assert!(parse_registry_head_sequence("witness-authority-registry-head-v1:02").is_err());
        assert!(parse_registry_head_sequence("witness-authority-registry-head-v1:0").is_err());
    }

    #[test]
    fn registry_log_rejects_sequence_regression() {
        let mut log = TransparencyLog::default();
        log.append(100, witness_registry_head_log_kind(3).unwrap(), sha256(b"three"))
            .unwrap();
        log.append(101, witness_registry_head_log_kind(2).unwrap(), sha256(b"two"))
            .unwrap();
        let errors = inspect_registry_head_log(&log, 3).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            WitnessRegistryHeadObservationError::RegistrySequenceRegressed {
                previous: 3,
                current: 2
            }
        )));
    }

    #[test]
    fn registry_log_rejects_duplicate_sequence() {
        let mut log = TransparencyLog::default();
        log.append(100, witness_registry_head_log_kind(2).unwrap(), sha256(b"a"))
            .unwrap();
        log.append(101, witness_registry_head_log_kind(2).unwrap(), sha256(b"b"))
            .unwrap();
        let errors = inspect_registry_head_log(&log, 2).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            WitnessRegistryHeadObservationError::DuplicateRegistrySequence(2)
        )));
    }

    #[test]
    fn exact_verification_defaults_to_two_providers() {
        let policy = ExactWitnessRegistryHeadVerificationPolicyV1::default();
        assert_eq!(policy.minimum_distinct_providers, 2);
        assert!(valid_exact_verification_policy(&policy));
    }
}
