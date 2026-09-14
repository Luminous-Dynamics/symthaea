// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Witnessed monotonic currentness for threshold-authorized containment state.
//!
//! This layer never promotes raw `FabricationContainmentState` bytes to live authority. The subject
//! is an opaque `ClockGovernedContainmentStateV1` from the non-circular authority lineage. Its
//! generation is published in a canonical transparency-log kind so rollback, duplicate-generation
//! equivocation and stale re-append attempts are all detectable inside the exact checkpoint view.

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
use symthaea_fabrication_trust_snapshot_head::{
    QuorumObservedTrustSnapshotHeadIdV1, QuorumObservedTrustSnapshotHeadV1,
};
use symthaea_fabrication_witness_authority::{
    WitnessAuthorityRegistryIdV1, WitnessAuthorityRegistryV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const AUTHORIZED_CONTAINMENT_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.authorized-containment-head-publication.v1";
pub const QUORUM_OBSERVED_CONTAINMENT_HEAD_SCHEMA: &str =
    "symthaea.fabrication.quorum-observed-containment-head.v1";
pub const CONTAINMENT_HEAD_LOG_KIND_PREFIX: &str = "containment-state-head-v1:";
pub const MAX_CONTAINMENT_HEAD_CLOCK_HOPS: usize = 4096;
pub const MAX_CONTAINMENT_HEAD_EXACT_VERIFIERS: usize = 16;

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.authorized-containment-head-publication.v1\0";
const SIGNED_CHECKPOINT_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-checkpoint-evidence.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-witness-set.v1\0";
const WITNESS_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-witness-policy.v1\0";
const EXACT_VERIFICATION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-exact-verification-policy.v1\0";
const EXACT_VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-exact-verifier-set.v1\0";
const AUTHORIZATION_TO_TRUST_CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-authorization-to-trust-clock-lineage.v1\0";
const TRUST_TO_OBSERVATION_CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-head-trust-to-observation-clock-lineage.v1\0";
const OBSERVED_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.quorum-observed-containment-head.v1\0";
const CHECKPOINT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-checkpoint-signature.v1\0";
const WITNESS_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-signature.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedContainmentHeadPublicationV1 {
    pub schema_version: String,
    pub authority_id: String,
    pub state_digest: Sha256Digest,
    pub generation: u64,
    pub previous_authority_id: Option<String>,
    pub previous_state_digest: Option<Sha256Digest>,
    pub compromise_tracker_digest: Sha256Digest,
    pub release_resilience_generation: u64,
    pub release_resilience_state_digest: Sha256Digest,
    pub authorization_trust_head_id: String,
    pub authorization_trust_snapshot_digest: Sha256Digest,
    pub authorization_trust_snapshot_sequence: u64,
    pub ceremony_id: String,
    pub ceremony_digest: Sha256Digest,
    pub authorization_clock_envelope_id: String,
    pub authorization_operational_basis_id: String,
}

pub trait ExactContainmentHeadEvidenceVerifierV1 {
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
pub struct ExactContainmentHeadVerificationPolicyV1 {
    pub minimum_distinct_providers: usize,
    pub maximum_providers: usize,
}

impl Default for ExactContainmentHeadVerificationPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_distinct_providers: 2,
            maximum_providers: 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ContainmentHeadTrustAnchorModeV1 {
    AuthorizationHead,
    StateBoundRefresh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuorumObservedContainmentHeadIdV1(Sha256Digest);

impl QuorumObservedContainmentHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct QuorumObservedContainmentHeadV1 {
    id: QuorumObservedContainmentHeadIdV1,
    authority_id: ClockGovernedContainmentStateIdV1,
    state_digest: Sha256Digest,
    generation: u64,
    compromise_tracker_digest: Sha256Digest,
    release_resilience_generation: u64,
    release_resilience_state_digest: Sha256Digest,
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
    trust_head_id: QuorumObservedTrustSnapshotHeadIdV1,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    trust_anchor_mode: ContainmentHeadTrustAnchorModeV1,
    exact_verification_policy_digest: Sha256Digest,
    exact_verifier_set_digest: Sha256Digest,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    trust_head_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    trust_head_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    authorization_to_trust_clock_lineage_digest: Sha256Digest,
    authorization_to_trust_clock_hop_count: usize,
    trust_to_observation_clock_lineage_digest: Sha256Digest,
    trust_to_observation_clock_hop_count: usize,
}

impl QuorumObservedContainmentHeadV1 {
    pub fn id(&self) -> QuorumObservedContainmentHeadIdV1 {
        self.id
    }
    pub fn authority_id(&self) -> ClockGovernedContainmentStateIdV1 {
        self.authority_id
    }
    pub fn state_digest(&self) -> Sha256Digest {
        self.state_digest
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn release_resilience_generation(&self) -> u64 {
        self.release_resilience_generation
    }
    pub fn release_resilience_state_digest(&self) -> Sha256Digest {
        self.release_resilience_state_digest
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
    pub fn trust_head_id(&self) -> QuorumObservedTrustSnapshotHeadIdV1 {
        self.trust_head_id
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn trust_snapshot_sequence(&self) -> u64 {
        self.trust_snapshot_sequence
    }
    pub fn trust_anchor_mode(&self) -> ContainmentHeadTrustAnchorModeV1 {
        self.trust_anchor_mode
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
    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.observation_clock_envelope_id
    }
    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.observation_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainmentHeadObservationError {
    AuthorityPublicationMismatch,
    InvalidPublication,
    TrustSnapshotInvalid(String),
    TrustSnapshotHeadMismatch,
    TrustSnapshotSequenceRollback,
    TrustSnapshotSameSequenceSubstitution,
    TrustHeadNotAnchoredToAuthority,
    AuthorizationBasisMismatch,
    AuthorizationEnvelopeMismatch,
    TrustHeadBasisMismatch,
    TrustHeadEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    TrustSnapshotNotValidAcrossObservationEnvelope(ClockGovernanceTimeError),
    TransparencyLogInvalid(String),
    MalformedContainmentHeadKind(String),
    ContainmentGenerationRegressed { previous: u64, current: u64 },
    DuplicateContainmentGeneration(u64),
    PublicationNotFound,
    HigherContainmentGenerationPublished { candidate: u64, latest: u64 },
    PublicationDigestMismatch,
    PublicationBeforeAuthorization,
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
struct ObservedContainmentHeadCommitment {
    schema: &'static str,
    authority_id: String,
    state_digest: String,
    generation: u64,
    compromise_tracker_digest: String,
    release_resilience_generation: u64,
    release_resilience_state_digest: String,
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
    trust_head_id: String,
    trust_snapshot_digest: String,
    trust_snapshot_sequence: u64,
    trust_anchor_mode: ContainmentHeadTrustAnchorModeV1,
    exact_verification_policy_digest: String,
    exact_verifier_set_digest: String,
    exact_verifier_count: usize,
    witness_count: usize,
    organization_count: usize,
    failure_domain_count: usize,
    authorization_clock_envelope_id: String,
    authorization_operational_basis_id: String,
    trust_head_clock_envelope_id: String,
    trust_head_operational_basis_id: String,
    observation_clock_envelope_id: String,
    observation_operational_basis_id: String,
    authorization_to_trust_clock_lineage_digest: String,
    authorization_to_trust_clock_hop_count: usize,
    trust_to_observation_clock_lineage_digest: String,
    trust_to_observation_clock_hop_count: usize,
}

pub fn build_authorized_containment_head_publication_v1(
    authority: &ClockGovernedContainmentStateV1,
) -> Result<AuthorizedContainmentHeadPublicationV1, ContainmentHeadObservationError> {
    let state = authority.state();
    let publication = AuthorizedContainmentHeadPublicationV1 {
        schema_version: AUTHORIZED_CONTAINMENT_HEAD_PUBLICATION_SCHEMA.into(),
        authority_id: authority.id().to_hex(),
        state_digest: authority.state_digest(),
        generation: authority.generation(),
        previous_authority_id: authority.previous_authority_id().map(|value| value.to_hex()),
        previous_state_digest: authority.previous_state_digest(),
        compromise_tracker_digest: authority.compromise_tracker_digest(),
        release_resilience_generation: state.release_resilience_generation,
        release_resilience_state_digest: state.release_resilience_state_digest,
        authorization_trust_head_id: authority.trust_head_id().to_hex(),
        authorization_trust_snapshot_digest: authority.trust_snapshot_digest(),
        authorization_trust_snapshot_sequence: authority.trust_snapshot_sequence(),
        ceremony_id: authority.ceremony_id().to_hex(),
        ceremony_digest: authority.ceremony_digest(),
        authorization_clock_envelope_id: authority.clock_envelope_id().to_hex(),
        authorization_operational_basis_id: authority.operational_basis_id().to_hex(),
    };
    validate_publication(&publication)?;
    Ok(publication)
}

pub fn digest_authorized_containment_head_publication_v1(
    publication: &AuthorizedContainmentHeadPublicationV1,
) -> Result<Sha256Digest, ContainmentHeadObservationError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

pub fn containment_head_log_kind(
    generation: u64,
) -> Result<String, ContainmentHeadObservationError> {
    if generation == 0 {
        return Err(ContainmentHeadObservationError::InvalidPublication);
    }
    Ok(format!("{CONTAINMENT_HEAD_LOG_KIND_PREFIX}{generation}"))
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_quorum_observed_containment_head_v1(
    authority: &ClockGovernedContainmentStateV1,
    authorization_basis: &OperationalClockBasisV1,
    authorization_to_trust_clock_bridge: &[OperationalClockBasisV1],
    trust_head: &QuorumObservedTrustSnapshotHeadV1,
    trust_snapshot: &TrustSnapshot,
    trust_head_basis: &OperationalClockBasisV1,
    trust_to_observation_clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    publication: &AuthorizedContainmentHeadPublicationV1,
    log: &TransparencyLog,
    signed_checkpoint: &SignedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    witness_policy: &TransparencyWitnessPolicy,
    witness_registry: &WitnessAuthorityRegistryV1,
    exact_verification_policy: &ExactContainmentHeadVerificationPolicyV1,
    exact_verification_providers: &[&dyn ExactContainmentHeadEvidenceVerifierV1],
) -> Result<QuorumObservedContainmentHeadV1, Vec<ContainmentHeadObservationError>> {
    let mut violations = Vec::new();

    let expected_publication = match build_authorized_containment_head_publication_v1(authority) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    if publication != &expected_publication {
        violations.push(ContainmentHeadObservationError::AuthorityPublicationMismatch);
    }
    let publication_digest = match digest_authorized_containment_head_publication_v1(publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };

    let authorization_clock = match require_exact_basis(
        authorization_basis,
        authority.operational_basis_id(),
        authority.clock_envelope_id(),
        ContainmentHeadObservationError::AuthorizationBasisMismatch,
        ContainmentHeadObservationError::AuthorizationEnvelopeMismatch,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    let trust_head_clock = match require_exact_basis(
        trust_head_basis,
        trust_head.observation_operational_basis_id(),
        trust_head.observation_clock_envelope_id(),
        ContainmentHeadObservationError::TrustHeadBasisMismatch,
        ContainmentHeadObservationError::TrustHeadEnvelopeMismatch,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };

    if authorization_to_trust_clock_bridge.len() > MAX_CONTAINMENT_HEAD_CLOCK_HOPS
        || trust_to_observation_clock_bridge.len() > MAX_CONTAINMENT_HEAD_CLOCK_HOPS
    {
        violations.push(ContainmentHeadObservationError::TooManyClockHops {
            actual: authorization_to_trust_clock_bridge
                .len()
                .max(trust_to_observation_clock_bridge.len()),
            maximum: MAX_CONTAINMENT_HEAD_CLOCK_HOPS,
        });
        return Err(violations);
    }
    if let Err(error) = verify_clock_lineage(
        authorization_basis.id(),
        authorization_to_trust_clock_bridge,
        trust_head_basis,
    ) {
        violations.push(error);
    }
    if let Err(error) = verify_clock_lineage(
        trust_head_basis.id(),
        trust_to_observation_clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis)
    {
        Ok(value) => value,
        Err(error) => {
            violations.push(ContainmentHeadObservationError::Clock(error));
            return Err(violations);
        }
    };

    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ContainmentHeadObservationError::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(ContainmentHeadObservationError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if trust_snapshot_digest != trust_head.snapshot_digest()
        || trust_snapshot.sequence != trust_head.snapshot_sequence()
    {
        violations.push(ContainmentHeadObservationError::TrustSnapshotHeadMismatch);
    }
    if trust_head.snapshot_sequence() < authority.trust_snapshot_sequence() {
        violations.push(ContainmentHeadObservationError::TrustSnapshotSequenceRollback);
    }
    if trust_head.snapshot_sequence() == authority.trust_snapshot_sequence()
        && trust_head.snapshot_digest() != authority.trust_snapshot_digest()
    {
        violations.push(ContainmentHeadObservationError::TrustSnapshotSameSequenceSubstitution);
    }
    let trust_anchor_mode = if trust_head.id() == authority.trust_head_id()
        && trust_head.snapshot_digest() == authority.trust_snapshot_digest()
        && trust_head.snapshot_sequence() == authority.trust_snapshot_sequence()
    {
        ContainmentHeadTrustAnchorModeV1::AuthorizationHead
    } else if trust_head.snapshot_sequence() >= authority.trust_snapshot_sequence()
        && trust_head.containment_state_digest() == authority.state_digest()
        && trust_head.compromise_tracker_digest() == authority.compromise_tracker_digest()
    {
        ContainmentHeadTrustAnchorModeV1::StateBoundRefresh
    } else {
        violations.push(ContainmentHeadObservationError::TrustHeadNotAnchoredToAuthority);
        ContainmentHeadTrustAnchorModeV1::AuthorizationHead
    };
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            ContainmentHeadObservationError::TrustSnapshotNotValidAcrossObservationEnvelope(reason),
        );
    }

    if let Err(error) = log.validate() {
        violations.push(ContainmentHeadObservationError::TransparencyLogInvalid(format!(
            "{error:?}"
        )));
    }
    let (latest_generation, publication_entry) = match inspect_containment_head_log(
        log,
        authority.generation(),
    ) {
        Ok(value) => value,
        Err(errors) => {
            violations.extend(errors);
            (None, None)
        }
    };
    if let Some(latest) = latest_generation {
        if latest > authority.generation() {
            violations.push(
                ContainmentHeadObservationError::HigherContainmentGenerationPublished {
                    candidate: authority.generation(),
                    latest,
                },
            );
        }
    }
    let Some(publication_entry) = publication_entry else {
        violations.push(ContainmentHeadObservationError::PublicationNotFound);
        return Err(violations);
    };
    if publication_entry.1 != publication_digest {
        violations.push(ContainmentHeadObservationError::PublicationDigestMismatch);
    }
    let publication_recorded_at_ms = match seconds_to_millis(publication_entry.2) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if publication_recorded_at_ms < authorization_clock.upper_unix_ms() {
        violations.push(ContainmentHeadObservationError::PublicationBeforeAuthorization);
    }
    if publication_recorded_at_ms > observation_clock.lower_unix_ms() {
        violations.push(ContainmentHeadObservationError::PublicationMayBeFuture);
    }

    if signed_checkpoint.schema_version != SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA {
        violations.push(ContainmentHeadObservationError::CheckpointInvalid(
            "unsupported signed checkpoint schema".into(),
        ));
    }
    if let Err(error) = signed_checkpoint.checkpoint.validate() {
        violations.push(ContainmentHeadObservationError::CheckpointInvalid(format!(
            "{error:?}"
        )));
    }
    match digest_transparency_checkpoint(&signed_checkpoint.checkpoint) {
        Ok(value) if value == signed_checkpoint.checkpoint_digest => {}
        Ok(_) => violations.push(ContainmentHeadObservationError::CheckpointInvalid(
            "checkpoint digest mismatch".into(),
        )),
        Err(error) => violations.push(ContainmentHeadObservationError::CheckpointInvalid(
            format!("{error:?}"),
        )),
    }
    let log_root = match log.root() {
        Ok(value) => value,
        Err(error) => {
            violations.push(ContainmentHeadObservationError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if signed_checkpoint.checkpoint.log_size != log.entries.len() as u64
        || signed_checkpoint.checkpoint.root_digest != log_root
    {
        violations.push(ContainmentHeadObservationError::CheckpointLogMismatch);
    }
    if log
        .entries
        .last()
        .is_some_and(|entry| entry.recorded_at_unix_s > signed_checkpoint.checkpoint.issued_at_unix_s)
    {
        violations.push(ContainmentHeadObservationError::CheckpointPredatesLog);
    }
    if publication_entry.2 > signed_checkpoint.checkpoint.issued_at_unix_s {
        violations.push(ContainmentHeadObservationError::CheckpointPredatesPublication);
    }
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        signed_checkpoint.checkpoint.issued_at_unix_s,
        signed_checkpoint.checkpoint.expires_at_unix_s,
    ) {
        violations.push(ContainmentHeadObservationError::CheckpointInvalid(format!(
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
        violations.push(ContainmentHeadObservationError::CheckpointInvalid(
            "invalid checkpoint signer or signature bytes".into(),
        ));
    }
    requalify_signer(
        &signed_checkpoint.signature.algorithm,
        &signed_checkpoint.signature.key_id,
        KeyUsage::TransparencyLog,
        signed_checkpoint.checkpoint.issued_at_unix_s,
        trust_snapshot,
        authority,
        &observation_clock,
        &mut violations,
    );

    if !valid_witness_policy(witness_policy) {
        violations.push(ContainmentHeadObservationError::WitnessPolicyInvalid);
    }
    if signed_witnesses.len() > witness_policy.maximum_witnesses
        || signed_witnesses.len() > MAX_TRANSPARENCY_WITNESSES
    {
        violations.push(ContainmentHeadObservationError::TooManyWitnesses {
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
            violations.push(ContainmentHeadObservationError::WitnessMalformed(key_id));
            continue;
        }
        if signed.statement.checkpoint_digest != signed_checkpoint.checkpoint_digest
            || signed.statement.checkpoint_log_size != signed_checkpoint.checkpoint.log_size
            || signed.statement.checkpoint_root_digest != signed_checkpoint.checkpoint.root_digest
        {
            violations.push(ContainmentHeadObservationError::WitnessCheckpointMismatch(
                key_id,
            ));
            continue;
        }
        match digest_transparency_witness_statement(&signed.statement) {
            Ok(value) if value == signed.statement_digest => {}
            Ok(_) => {
                violations.push(ContainmentHeadObservationError::WitnessMalformed(key_id));
                continue;
            }
            Err(error) => {
                violations.push(ContainmentHeadObservationError::WitnessMalformed(format!(
                    "{}: {error:?}",
                    signed.signature.key_id
                )));
                continue;
            }
        }
        if signed.statement.observed_at_unix_s < signed_checkpoint.checkpoint.issued_at_unix_s {
            violations.push(ContainmentHeadObservationError::WitnessBeforeCheckpoint(
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
            violations.push(ContainmentHeadObservationError::WitnessMayBeFuture(
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
                violations.push(ContainmentHeadObservationError::TimeScaleOverflow);
                continue;
            }
        };
        if observation_clock.upper_unix_ms() > freshness_deadline {
            violations.push(ContainmentHeadObservationError::WitnessMayBeStale(
                signed.signature.key_id.clone(),
            ));
        }
        let signer = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signer_ids.insert(signer) {
            violations.push(ContainmentHeadObservationError::DuplicateWitnessSigner(
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
            authority,
            &observation_clock,
            &mut violations,
        );

        let Some(profile) = witness_registry.profile(
            &signed.signature.algorithm,
            &signed.signature.key_id,
        ) else {
            violations.push(ContainmentHeadObservationError::WitnessNotRegistered(
                signed.signature.key_id.clone(),
            ));
            continue;
        };
        if profile.organization != signed.statement.witness_organization {
            violations.push(ContainmentHeadObservationError::WitnessOrganizationMismatch(
                signed.signature.key_id.clone(),
            ));
        }
        if profile.failure_domain != signed.statement.witness_region {
            violations.push(ContainmentHeadObservationError::WitnessFailureDomainMismatch(
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
        violations.push(ContainmentHeadObservationError::InsufficientWitnesses {
            actual: signer_ids.len(),
            required: witness_policy.minimum_distinct_witnesses,
        });
    }
    if organizations.len() < witness_policy.minimum_distinct_organizations {
        violations.push(ContainmentHeadObservationError::InsufficientOrganizations {
            actual: organizations.len(),
            required: witness_policy.minimum_distinct_organizations,
        });
    }
    if failure_domains.len() < witness_policy.minimum_distinct_regions {
        violations.push(ContainmentHeadObservationError::InsufficientFailureDomains {
            actual: failure_domains.len(),
            required: witness_policy.minimum_distinct_regions,
        });
    }
    if witness_policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(ContainmentHeadObservationError::MissingAlgorithmDiversity);
    }

    if !valid_exact_verification_policy(exact_verification_policy) {
        violations.push(ContainmentHeadObservationError::InvalidExactVerificationPolicy);
    }
    if exact_verification_providers.len() < exact_verification_policy.minimum_distinct_providers {
        violations.push(
            ContainmentHeadObservationError::InsufficientExactVerificationProviders {
                actual: exact_verification_providers.len(),
                required: exact_verification_policy.minimum_distinct_providers,
            },
        );
    }
    if exact_verification_providers.len() > exact_verification_policy.maximum_providers
        || exact_verification_providers.len() > MAX_CONTAINMENT_HEAD_EXACT_VERIFIERS
    {
        violations.push(
            ContainmentHeadObservationError::TooManyExactVerificationProviders {
                actual: exact_verification_providers.len(),
                maximum: exact_verification_policy
                    .maximum_providers
                    .min(MAX_CONTAINMENT_HEAD_EXACT_VERIFIERS),
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
            violations.push(ContainmentHeadObservationError::InvalidExactVerificationProvider(
                provider_id,
            ));
            continue;
        }
        if !seen_providers.insert(provider_id.clone()) {
            violations.push(ContainmentHeadObservationError::DuplicateExactVerificationProvider(
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
                ContainmentHeadObservationError::CheckpointSignatureRejected(provider_id.clone()),
            ),
            Err(reason) => violations.push(
                ContainmentHeadObservationError::VerificationProviderError {
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
                    ContainmentHeadObservationError::WitnessSignatureRejected {
                        provider: provider_id.clone(),
                        key_id: witness.signature.key_id.clone(),
                    },
                ),
                Err(reason) => violations.push(
                    ContainmentHeadObservationError::VerificationProviderError {
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
        vec![ContainmentHeadObservationError::TransparencyLogInvalid(format!(
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
    let (authorization_to_trust_clock_lineage_digest, authorization_to_trust_clock_hop_count) =
        digest_clock_lineage(
            AUTHORIZATION_TO_TRUST_CLOCK_LINEAGE_DOMAIN,
            authorization_basis,
            authorization_to_trust_clock_bridge,
            trust_head_basis,
        )
        .map_err(|error| vec![error])?;
    let (trust_to_observation_clock_lineage_digest, trust_to_observation_clock_hop_count) =
        digest_clock_lineage(
            TRUST_TO_OBSERVATION_CLOCK_LINEAGE_DOMAIN,
            trust_head_basis,
            trust_to_observation_clock_bridge,
            observation_basis,
        )
        .map_err(|error| vec![error])?;

    let commitment = ObservedContainmentHeadCommitment {
        schema: QUORUM_OBSERVED_CONTAINMENT_HEAD_SCHEMA,
        authority_id: authority.id().to_hex(),
        state_digest: authority.state_digest().to_hex(),
        generation: authority.generation(),
        compromise_tracker_digest: authority.compromise_tracker_digest().to_hex(),
        release_resilience_generation: authority.state().release_resilience_generation,
        release_resilience_state_digest: authority.state().release_resilience_state_digest.to_hex(),
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
        trust_head_id: trust_head.id().to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        trust_snapshot_sequence: trust_snapshot.sequence,
        trust_anchor_mode,
        exact_verification_policy_digest: exact_verification_policy_digest.to_hex(),
        exact_verifier_set_digest: exact_verifier_set_digest.to_hex(),
        exact_verifier_count: verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        authorization_clock_envelope_id: authorization_clock.id().to_hex(),
        authorization_operational_basis_id: authorization_basis.id().to_hex(),
        trust_head_clock_envelope_id: trust_head_clock.id().to_hex(),
        trust_head_operational_basis_id: trust_head_basis.id().to_hex(),
        observation_clock_envelope_id: observation_clock.id().to_hex(),
        observation_operational_basis_id: observation_basis.id().to_hex(),
        authorization_to_trust_clock_lineage_digest:
            authorization_to_trust_clock_lineage_digest.to_hex(),
        authorization_to_trust_clock_hop_count,
        trust_to_observation_clock_lineage_digest:
            trust_to_observation_clock_lineage_digest.to_hex(),
        trust_to_observation_clock_hop_count,
    };
    let id = QuorumObservedContainmentHeadIdV1(
        hash_serializable(OBSERVED_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(QuorumObservedContainmentHeadV1 {
        id,
        authority_id: authority.id(),
        state_digest: authority.state_digest(),
        generation: authority.generation(),
        compromise_tracker_digest: authority.compromise_tracker_digest(),
        release_resilience_generation: authority.state().release_resilience_generation,
        release_resilience_state_digest: authority.state().release_resilience_state_digest,
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
        trust_head_id: trust_head.id(),
        trust_snapshot_digest,
        trust_snapshot_sequence: trust_snapshot.sequence,
        trust_anchor_mode,
        exact_verification_policy_digest,
        exact_verifier_set_digest,
        exact_verifier_count: verifier_commitments.len(),
        witness_count: signed_witnesses.len(),
        organization_count: organizations.len(),
        failure_domain_count: failure_domains.len(),
        authorization_clock_envelope_id: authorization_clock.id(),
        authorization_operational_basis_id: authorization_basis.id(),
        trust_head_clock_envelope_id: trust_head_clock.id(),
        trust_head_operational_basis_id: trust_head_basis.id(),
        observation_clock_envelope_id: observation_clock.id(),
        observation_operational_basis_id: observation_basis.id(),
        authorization_to_trust_clock_lineage_digest,
        authorization_to_trust_clock_hop_count,
        trust_to_observation_clock_lineage_digest,
        trust_to_observation_clock_hop_count,
    })
}

fn validate_publication(
    publication: &AuthorizedContainmentHeadPublicationV1,
) -> Result<(), ContainmentHeadObservationError> {
    if publication.schema_version != AUTHORIZED_CONTAINMENT_HEAD_PUBLICATION_SCHEMA
        || publication.authority_id.len() != 64
        || publication.state_digest == Sha256Digest([0; 32])
        || publication.generation == 0
        || publication.compromise_tracker_digest == Sha256Digest([0; 32])
        || publication.release_resilience_generation == 0
        || publication.release_resilience_state_digest == Sha256Digest([0; 32])
        || publication.authorization_trust_head_id.len() != 64
        || publication.authorization_trust_snapshot_digest == Sha256Digest([0; 32])
        || publication.authorization_trust_snapshot_sequence == 0
        || publication.ceremony_id.len() != 64
        || publication.ceremony_digest == Sha256Digest([0; 32])
        || publication.authorization_clock_envelope_id.len() != 64
        || publication.authorization_operational_basis_id.len() != 64
        || (publication.generation == 1)
            != (publication.previous_authority_id.is_none()
                && publication.previous_state_digest.is_none())
        || publication
            .previous_authority_id
            .as_ref()
            .is_some_and(|value| value.len() != 64)
        || publication
            .previous_state_digest
            .is_some_and(|value| value == Sha256Digest([0; 32]))
    {
        return Err(ContainmentHeadObservationError::InvalidPublication);
    }
    Ok(())
}

fn inspect_containment_head_log(
    log: &TransparencyLog,
    candidate_generation: u64,
) -> Result<
    (Option<u64>, Option<(u64, Sha256Digest, u64)>),
    Vec<ContainmentHeadObservationError>,
> {
    let mut violations = Vec::new();
    let mut previous_generation = None;
    let mut latest_generation = None;
    let mut candidate_entry = None;
    for entry in &log.entries {
        let generation = match parse_containment_head_generation(&entry.kind) {
            Ok(Some(value)) => value,
            Ok(None) => continue,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if let Some(previous) = previous_generation {
            if generation <= previous {
                if generation == previous {
                    violations.push(
                        ContainmentHeadObservationError::DuplicateContainmentGeneration(generation),
                    );
                } else {
                    violations.push(
                        ContainmentHeadObservationError::ContainmentGenerationRegressed {
                            previous,
                            current: generation,
                        },
                    );
                }
            }
        }
        previous_generation = Some(generation);
        latest_generation = Some(
            latest_generation.map_or(generation, |latest: u64| latest.max(generation)),
        );
        if generation == candidate_generation {
            if candidate_entry.is_some() {
                violations.push(
                    ContainmentHeadObservationError::DuplicateContainmentGeneration(generation),
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
        Ok((latest_generation, candidate_entry))
    } else {
        Err(violations)
    }
}

fn parse_containment_head_generation(
    kind: &str,
) -> Result<Option<u64>, ContainmentHeadObservationError> {
    let Some(suffix) = kind.strip_prefix(CONTAINMENT_HEAD_LOG_KIND_PREFIX) else {
        return Ok(None);
    };
    let generation = suffix.parse::<u64>().map_err(|_| {
        ContainmentHeadObservationError::MalformedContainmentHeadKind(kind.to_string())
    })?;
    if generation == 0 || suffix != generation.to_string() {
        return Err(ContainmentHeadObservationError::MalformedContainmentHeadKind(
            kind.to_string(),
        ));
    }
    Ok(Some(generation))
}

fn require_exact_basis(
    basis: &OperationalClockBasisV1,
    expected_basis_id: OperationalClockBasisIdV1,
    expected_clock_id: ClockGovernanceEvaluationEnvelopeIdV1,
    basis_error: ContainmentHeadObservationError,
    envelope_error: ContainmentHeadObservationError,
) -> Result<ClockGovernanceEvaluationEnvelopeV1, ContainmentHeadObservationError> {
    if basis.id() != expected_basis_id {
        return Err(basis_error);
    }
    let clock = derive_clock_governance_evaluation_envelope_v1(basis)
        .map_err(ContainmentHeadObservationError::Clock)?;
    if clock.id() != expected_clock_id {
        return Err(envelope_error);
    }
    Ok(clock)
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), ContainmentHeadObservationError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(ContainmentHeadObservationError::BrokenClockLineage {
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
            return Err(ContainmentHeadObservationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(ContainmentHeadObservationError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn digest_clock_lineage(
    domain: &[u8],
    start: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    end: &OperationalClockBasisV1,
) -> Result<(Sha256Digest, usize), ContainmentHeadObservationError> {
    let mut lineage = Vec::with_capacity(bridge.len() + 2);
    lineage.push(start.id().to_hex());
    lineage.extend(bridge.iter().map(|basis| basis.id().to_hex()));
    if end.id() != start.id() {
        lineage.push(end.id().to_hex());
    }
    let digest = hash_serializable(domain, &lineage)?;
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
) -> Result<Sha256Digest, ContainmentHeadObservationError> {
    if !valid_witness_policy(policy) {
        return Err(ContainmentHeadObservationError::WitnessPolicyInvalid);
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

fn valid_exact_verification_policy(policy: &ExactContainmentHeadVerificationPolicyV1) -> bool {
    policy.minimum_distinct_providers > 0
        && policy.maximum_providers > 0
        && policy.minimum_distinct_providers <= policy.maximum_providers
        && policy.maximum_providers <= MAX_CONTAINMENT_HEAD_EXACT_VERIFIERS
}

#[allow(clippy::too_many_arguments)]
fn requalify_signer(
    algorithm: &SignatureAlgorithm,
    key_id: &str,
    usage: KeyUsage,
    evidence_time_unix_s: u64,
    trust_snapshot: &TrustSnapshot,
    authority: &ClockGovernedContainmentStateV1,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<ContainmentHeadObservationError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(ContainmentHeadObservationError::SignerUnknown(key_id.to_string()));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(ContainmentHeadObservationError::SignerNotActive(key_id.to_string()));
    }
    if !record.usages.contains(&usage) {
        violations.push(ContainmentHeadObservationError::SignerUsageNotAllowed(
            key_id.to_string(),
        ));
    }
    if record.not_before_unix_s > evidence_time_unix_s
        || record
            .not_after_unix_s
            .is_some_and(|not_after| evidence_time_unix_s >= not_after)
    {
        violations.push(ContainmentHeadObservationError::SignerInvalidAtEvidenceTime(
            key_id.to_string(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(ContainmentHeadObservationError::SignerNotValidAcrossEnvelope {
            key_id: key_id.to_string(),
            reason,
        });
    }
    for compromise in authority
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
                ContainmentHeadObservationError::SignerCompromisedAcrossEnvelope(
                    key_id.to_string(),
                ),
            ),
            Err(reason) => violations.push(ContainmentHeadObservationError::CompromiseTimeInvalid {
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

fn seconds_to_millis(value: u64) -> Result<u64, ContainmentHeadObservationError> {
    value
        .checked_mul(1_000)
        .ok_or(ContainmentHeadObservationError::TimeScaleOverflow)
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
) -> Result<Sha256Digest, ContainmentHeadObservationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ContainmentHeadObservationError::Encoding(error.to_string()))?;
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
    fn log_kind_round_trips_canonical_generation() {
        let kind = containment_head_log_kind(7).unwrap();
        assert_eq!(parse_containment_head_generation(&kind).unwrap(), Some(7));
        assert!(parse_containment_head_generation("containment-state-head-v1:007").is_err());
        assert!(parse_containment_head_generation("containment-state-head-v1:0").is_err());
    }

    #[test]
    fn containment_log_rejects_generation_regression() {
        let mut log = TransparencyLog::default();
        log.append(100, containment_head_log_kind(2).unwrap(), sha256(b"two"))
            .unwrap();
        log.append(101, containment_head_log_kind(1).unwrap(), sha256(b"one"))
            .unwrap();
        let errors = inspect_containment_head_log(&log, 2).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ContainmentHeadObservationError::ContainmentGenerationRegressed {
                previous: 2,
                current: 1
            }
        )));
    }

    #[test]
    fn containment_log_rejects_duplicate_generation() {
        let mut log = TransparencyLog::default();
        log.append(100, containment_head_log_kind(3).unwrap(), sha256(b"a"))
            .unwrap();
        log.append(101, containment_head_log_kind(3).unwrap(), sha256(b"b"))
            .unwrap();
        let errors = inspect_containment_head_log(&log, 3).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ContainmentHeadObservationError::DuplicateContainmentGeneration(3)
        )));
    }

    #[test]
    fn exact_verification_defaults_to_two_providers() {
        let policy = ExactContainmentHeadVerificationPolicyV1::default();
        assert_eq!(policy.minimum_distinct_providers, 2);
        assert!(valid_exact_verification_policy(&policy));
    }
}
