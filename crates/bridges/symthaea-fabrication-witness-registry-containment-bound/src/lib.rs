// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Same-checkpoint containment-current rebinding for witnessed registry heads.
//!
//! `QuorumObservedWitnessRegistryHeadV1` proves registry-sequence currentness but deliberately does
//! not prove that the containment authority used for compromise checks is the highest containment
//! generation in that checkpoint. This bridge upgrades the theorem inside the exact same log and
//! checkpoint, then independently re-authenticates raw checkpoint/witness evidence against that
//! highest authorized containment generation.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_containment_head::{
    AuthorizedContainmentHeadPublicationV1, CONTAINMENT_HEAD_LOG_KIND_PREFIX,
    build_authorized_containment_head_publication_v1,
    digest_authorized_containment_head_publication_v1,
};
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
use symthaea_fabrication_witness_registry_activation::ActivatedWitnessAuthorityRegistryV1;
use symthaea_fabrication_witness_registry_head::{
    ExactWitnessRegistryHeadEvidenceVerifierV1, ExactWitnessRegistryHeadVerificationPolicyV1,
    QuorumObservedWitnessRegistryHeadIdV1, QuorumObservedWitnessRegistryHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const CONTAINMENT_CURRENT_WITNESS_REGISTRY_HEAD_SCHEMA: &str =
    "symthaea.fabrication.containment-current-witness-registry-head.v1";
pub const MAX_REGISTRY_CONTAINMENT_REBIND_HOPS: usize = 4096;
pub const MAX_REGISTRY_CONTAINMENT_REBIND_CLOCK_HOPS: usize = 4096;

const SIGNED_CHECKPOINT_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-checkpoint-evidence.v1\0";
const SIGNED_WITNESS_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-witness-evidence.v1\0";
const WITNESS_SET_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-witness-set.v1\0";
const WITNESS_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-witness-policy.v1\0";
const EXACT_VERIFICATION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-exact-verification-policy.v1\0";
const EXACT_VERIFIER_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-exact-verifier-set.v1\0";
const CONTAINMENT_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-containment-lineage.v1\0";
const CONTAINMENT_TO_OBSERVATION_CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.registry-containment-bound-clock-lineage.v1\0";
const CAPABILITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.containment-current-witness-registry-head.v1\0";
const CHECKPOINT_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-checkpoint-signature.v1\0";
const WITNESS_SIGNATURE_DOMAIN: &[u8] =
    b"symthaea.fabrication.transparency-witness-signature.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContainmentCurrentWitnessRegistryHeadIdV1(Sha256Digest);

impl ContainmentCurrentWitnessRegistryHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct ContainmentCurrentWitnessRegistryHeadV1 {
    id: ContainmentCurrentWitnessRegistryHeadIdV1,
    registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    activated_registry_id: String,
    registry_digest: Sha256Digest,
    registry_sequence: u64,
    containment_authority_id: ClockGovernedContainmentStateIdV1,
    containment_state_digest: Sha256Digest,
    containment_generation: u64,
    compromise_tracker_digest: Sha256Digest,
    containment_publication_digest: Sha256Digest,
    containment_publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
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
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    containment_authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    containment_authorization_operational_basis_id: OperationalClockBasisIdV1,
    containment_lineage_digest: Sha256Digest,
    containment_hop_count: usize,
    containment_to_observation_clock_lineage_digest: Sha256Digest,
    containment_to_observation_clock_hop_count: usize,
}

impl ContainmentCurrentWitnessRegistryHeadV1 {
    pub fn id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 {
        self.id
    }
    pub fn registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 {
        self.registry_head_id
    }
    pub fn registry_digest(&self) -> Sha256Digest {
        self.registry_digest
    }
    pub fn registry_sequence(&self) -> u64 {
        self.registry_sequence
    }
    pub fn containment_authority_id(&self) -> ClockGovernedContainmentStateIdV1 {
        self.containment_authority_id
    }
    pub fn containment_state_digest(&self) -> Sha256Digest {
        self.containment_state_digest
    }
    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
    }
    pub fn transparency_log_digest(&self) -> Sha256Digest {
        self.transparency_log_digest
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.observation_clock_envelope_id
    }
    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.observation_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryContainmentBindingError {
    RegistryHeadMismatch,
    RegistryMismatch,
    TrustSnapshotInvalid(String),
    TrustSnapshotMismatch,
    ObservationBasisMismatch,
    ObservationEnvelopeMismatch,
    BaseContainmentMismatch,
    TooManyContainmentHops { actual: usize, maximum: usize },
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenContainmentLineage {
        hop: usize,
        expected_previous_authority: String,
        actual_previous_authority: Option<String>,
    },
    BrokenContainmentStateLineage { hop: usize },
    ContainmentGenerationOverflow,
    ContainmentGenerationNotAdjacent { previous: u64, current: u64 },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    ContainmentAuthorizationBasisMismatch,
    ContainmentAuthorizationEnvelopeMismatch,
    TransparencyLogInvalid(String),
    TransparencyLogMismatch,
    MalformedContainmentHeadKind(String),
    ContainmentGenerationRegressed { previous: u64, current: u64 },
    DuplicateContainmentGeneration(u64),
    ContainmentPublicationNotFound,
    HigherContainmentGenerationPublished { candidate: u64, latest: u64 },
    ContainmentPublicationMismatch,
    ContainmentPublicationDigestMismatch,
    ContainmentPublicationBeforeAuthorization,
    ContainmentPublicationMayBeFuture,
    CheckpointInvalid(String),
    CheckpointMismatch,
    CheckpointLogMismatch,
    CheckpointPredatesLog,
    CheckpointPredatesContainmentPublication,
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
    Clock(ClockGovernanceTimeError),
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
struct CapabilityCommitment {
    schema: &'static str,
    registry_head_id: String,
    activated_registry_id: String,
    registry_digest: String,
    registry_sequence: u64,
    containment_authority_id: String,
    containment_state_digest: String,
    containment_generation: u64,
    compromise_tracker_digest: String,
    containment_publication_digest: String,
    containment_publication_entry_sequence: u64,
    transparency_log_digest: String,
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
    observation_clock_envelope_id: String,
    observation_operational_basis_id: String,
    containment_authorization_clock_envelope_id: String,
    containment_authorization_operational_basis_id: String,
    containment_lineage_digest: String,
    containment_hop_count: usize,
    containment_to_observation_clock_lineage_digest: String,
    containment_to_observation_clock_hop_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn bind_registry_head_to_current_containment_v1(
    registry_head: &QuorumObservedWitnessRegistryHeadV1,
    activated_registry: &ActivatedWitnessAuthorityRegistryV1,
    base_containment_authority: &ClockGovernedContainmentStateV1,
    containment_authority_bridge: &[ClockGovernedContainmentStateV1],
    current_containment_authority: &ClockGovernedContainmentStateV1,
    current_containment_authorization_basis: &OperationalClockBasisV1,
    containment_to_observation_clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    trust_snapshot: &TrustSnapshot,
    containment_publication: &AuthorizedContainmentHeadPublicationV1,
    log: &TransparencyLog,
    signed_checkpoint: &SignedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    witness_policy: &TransparencyWitnessPolicy,
    exact_verification_policy: &ExactWitnessRegistryHeadVerificationPolicyV1,
    exact_verification_providers: &[&dyn ExactWitnessRegistryHeadEvidenceVerifierV1],
) -> Result<ContainmentCurrentWitnessRegistryHeadV1, Vec<RegistryContainmentBindingError>> {
    let mut violations = Vec::new();

    if registry_head.activated_registry_id() != activated_registry.id()
        || registry_head.registry_digest() != activated_registry.registry_digest()
        || registry_head.sequence() != activated_registry.sequence()
    {
        violations.push(RegistryContainmentBindingError::RegistryMismatch);
    }

    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(RegistryContainmentBindingError::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(RegistryContainmentBindingError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if trust_snapshot_digest != registry_head.trust_snapshot_digest()
        || trust_snapshot.sequence != registry_head.trust_snapshot_sequence()
        || trust_snapshot_digest != activated_registry.trust_snapshot_digest()
        || trust_snapshot.sequence != activated_registry.trust_snapshot_sequence()
    {
        violations.push(RegistryContainmentBindingError::TrustSnapshotMismatch);
    }

    if observation_basis.id() != registry_head.observation_operational_basis_id() {
        violations.push(RegistryContainmentBindingError::ObservationBasisMismatch);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(RegistryContainmentBindingError::Clock(error));
            return Err(violations);
        }
    };
    if observation_clock.id() != registry_head.observation_clock_envelope_id() {
        violations.push(RegistryContainmentBindingError::ObservationEnvelopeMismatch);
    }
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(RegistryContainmentBindingError::TrustSnapshotInvalid(format!(
            "{reason:?}"
        )));
    }

    if base_containment_authority.id() != registry_head.observation_containment_authority_id()
        || base_containment_authority.state_digest()
            != registry_head.observation_containment_state_digest()
        || base_containment_authority.generation()
            != registry_head.observation_containment_generation()
        || base_containment_authority.compromise_tracker_digest()
            != registry_head.observation_compromise_tracker_digest()
    {
        violations.push(RegistryContainmentBindingError::BaseContainmentMismatch);
    }
    if containment_authority_bridge.len() > MAX_REGISTRY_CONTAINMENT_REBIND_HOPS {
        violations.push(RegistryContainmentBindingError::TooManyContainmentHops {
            actual: containment_authority_bridge.len(),
            maximum: MAX_REGISTRY_CONTAINMENT_REBIND_HOPS,
        });
    } else if let Err(error) = verify_containment_lineage(
        base_containment_authority,
        containment_authority_bridge,
        current_containment_authority,
    ) {
        violations.push(error);
    }

    if current_containment_authorization_basis.id()
        != current_containment_authority.operational_basis_id()
    {
        violations.push(RegistryContainmentBindingError::ContainmentAuthorizationBasisMismatch);
    }
    let containment_authorization_clock = match derive_clock_governance_evaluation_envelope_v1(
        current_containment_authorization_basis,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(RegistryContainmentBindingError::Clock(error));
            return Err(violations);
        }
    };
    if containment_authorization_clock.id() != current_containment_authority.clock_envelope_id() {
        violations.push(RegistryContainmentBindingError::ContainmentAuthorizationEnvelopeMismatch);
    }
    if containment_to_observation_clock_bridge.len()
        > MAX_REGISTRY_CONTAINMENT_REBIND_CLOCK_HOPS
    {
        violations.push(RegistryContainmentBindingError::TooManyClockHops {
            actual: containment_to_observation_clock_bridge.len(),
            maximum: MAX_REGISTRY_CONTAINMENT_REBIND_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        current_containment_authorization_basis.id(),
        containment_to_observation_clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }

    let expected_containment_publication = match build_authorized_containment_head_publication_v1(
        current_containment_authority,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(RegistryContainmentBindingError::ContainmentPublicationMismatch);
            violations.push(RegistryContainmentBindingError::Encoding(format!("{error:?}")));
            return Err(violations);
        }
    };
    if containment_publication != &expected_containment_publication {
        violations.push(RegistryContainmentBindingError::ContainmentPublicationMismatch);
    }
    let containment_publication_digest = match digest_authorized_containment_head_publication_v1(
        containment_publication,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(RegistryContainmentBindingError::Encoding(format!("{error:?}")));
            return Err(violations);
        }
    };

    let transparency_log_digest = match digest_transparency_log(log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(RegistryContainmentBindingError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if transparency_log_digest != registry_head.transparency_log_digest() {
        violations.push(RegistryContainmentBindingError::TransparencyLogMismatch);
    }
    let (latest_generation, containment_entry) = match inspect_containment_head_log(
        log,
        current_containment_authority.generation(),
    ) {
        Ok(value) => value,
        Err(errors) => {
            violations.extend(errors);
            (None, None)
        }
    };
    if let Some(latest) = latest_generation {
        if latest > current_containment_authority.generation() {
            violations.push(
                RegistryContainmentBindingError::HigherContainmentGenerationPublished {
                    candidate: current_containment_authority.generation(),
                    latest,
                },
            );
        }
    }
    let Some(containment_entry) = containment_entry else {
        violations.push(RegistryContainmentBindingError::ContainmentPublicationNotFound);
        return Err(violations);
    };
    if containment_entry.1 != containment_publication_digest {
        violations.push(RegistryContainmentBindingError::ContainmentPublicationDigestMismatch);
    }
    let containment_recorded_at_ms = match seconds_to_millis(containment_entry.2) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if containment_recorded_at_ms < containment_authorization_clock.upper_unix_ms() {
        violations.push(RegistryContainmentBindingError::ContainmentPublicationBeforeAuthorization);
    }
    if containment_recorded_at_ms > observation_clock.lower_unix_ms() {
        violations.push(RegistryContainmentBindingError::ContainmentPublicationMayBeFuture);
    }

    if signed_checkpoint.checkpoint_digest != registry_head.checkpoint_digest() {
        violations.push(RegistryContainmentBindingError::CheckpointMismatch);
    }
    if signed_checkpoint.schema_version != SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA {
        violations.push(RegistryContainmentBindingError::CheckpointInvalid(
            "unsupported signed checkpoint schema".into(),
        ));
    }
    if let Err(error) = signed_checkpoint.checkpoint.validate() {
        violations.push(RegistryContainmentBindingError::CheckpointInvalid(format!(
            "{error:?}"
        )));
    }
    match digest_transparency_checkpoint(&signed_checkpoint.checkpoint) {
        Ok(value) if value == signed_checkpoint.checkpoint_digest => {}
        Ok(_) => violations.push(RegistryContainmentBindingError::CheckpointInvalid(
            "checkpoint digest mismatch".into(),
        )),
        Err(error) => violations.push(RegistryContainmentBindingError::CheckpointInvalid(
            format!("{error:?}"),
        )),
    }
    let log_root = match log.root() {
        Ok(value) => value,
        Err(error) => {
            violations.push(RegistryContainmentBindingError::TransparencyLogInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if signed_checkpoint.checkpoint.log_size != log.entries.len() as u64
        || signed_checkpoint.checkpoint.root_digest != log_root
    {
        violations.push(RegistryContainmentBindingError::CheckpointLogMismatch);
    }
    if log
        .entries
        .last()
        .is_some_and(|entry| entry.recorded_at_unix_s > signed_checkpoint.checkpoint.issued_at_unix_s)
    {
        violations.push(RegistryContainmentBindingError::CheckpointPredatesLog);
    }
    if containment_entry.2 > signed_checkpoint.checkpoint.issued_at_unix_s {
        violations.push(RegistryContainmentBindingError::CheckpointPredatesContainmentPublication);
    }
    if let Err(reason) = observation_clock.require_valid_across_seconds_window(
        signed_checkpoint.checkpoint.issued_at_unix_s,
        signed_checkpoint.checkpoint.expires_at_unix_s,
    ) {
        violations.push(RegistryContainmentBindingError::CheckpointInvalid(format!(
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
        violations.push(RegistryContainmentBindingError::CheckpointInvalid(
            "invalid checkpoint signer or signature bytes".into(),
        ));
    }
    requalify_signer(
        &signed_checkpoint.signature.algorithm,
        &signed_checkpoint.signature.key_id,
        KeyUsage::TransparencyLog,
        signed_checkpoint.checkpoint.issued_at_unix_s,
        trust_snapshot,
        current_containment_authority,
        &observation_clock,
        &mut violations,
    );

    if !valid_witness_policy(witness_policy) {
        violations.push(RegistryContainmentBindingError::WitnessPolicyInvalid);
    }
    if signed_witnesses.len() > witness_policy.maximum_witnesses
        || signed_witnesses.len() > MAX_TRANSPARENCY_WITNESSES
    {
        violations.push(RegistryContainmentBindingError::TooManyWitnesses {
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
            violations.push(RegistryContainmentBindingError::WitnessMalformed(key_id));
            continue;
        }
        if signed.statement.checkpoint_digest != signed_checkpoint.checkpoint_digest
            || signed.statement.checkpoint_log_size != signed_checkpoint.checkpoint.log_size
            || signed.statement.checkpoint_root_digest != signed_checkpoint.checkpoint.root_digest
        {
            violations.push(RegistryContainmentBindingError::WitnessCheckpointMismatch(key_id));
            continue;
        }
        match digest_transparency_witness_statement(&signed.statement) {
            Ok(value) if value == signed.statement_digest => {}
            Ok(_) => {
                violations.push(RegistryContainmentBindingError::WitnessMalformed(key_id));
                continue;
            }
            Err(error) => {
                violations.push(RegistryContainmentBindingError::WitnessMalformed(format!(
                    "{}: {error:?}",
                    signed.signature.key_id
                )));
                continue;
            }
        }
        if signed.statement.observed_at_unix_s < signed_checkpoint.checkpoint.issued_at_unix_s {
            violations.push(RegistryContainmentBindingError::WitnessBeforeCheckpoint(
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
            violations.push(RegistryContainmentBindingError::WitnessMayBeFuture(
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
                violations.push(RegistryContainmentBindingError::TimeScaleOverflow);
                continue;
            }
        };
        if observation_clock.upper_unix_ms() > freshness_deadline {
            violations.push(RegistryContainmentBindingError::WitnessMayBeStale(
                signed.signature.key_id.clone(),
            ));
        }
        let signer = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signer_ids.insert(signer) {
            violations.push(RegistryContainmentBindingError::DuplicateWitnessSigner(
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
            current_containment_authority,
            &observation_clock,
            &mut violations,
        );
        let Some(profile) = activated_registry.profile(
            &signed.signature.algorithm,
            &signed.signature.key_id,
        ) else {
            violations.push(RegistryContainmentBindingError::WitnessNotInActivatedRegistry(
                signed.signature.key_id.clone(),
            ));
            continue;
        };
        if profile.organization != signed.statement.witness_organization {
            violations.push(RegistryContainmentBindingError::WitnessOrganizationMismatch(
                signed.signature.key_id.clone(),
            ));
        }
        if profile.failure_domain != signed.statement.witness_region {
            violations.push(RegistryContainmentBindingError::WitnessFailureDomainMismatch(
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
        violations.push(RegistryContainmentBindingError::InsufficientWitnesses {
            actual: signer_ids.len(),
            required: witness_policy.minimum_distinct_witnesses,
        });
    }
    if organizations.len() < witness_policy.minimum_distinct_organizations {
        violations.push(RegistryContainmentBindingError::InsufficientOrganizations {
            actual: organizations.len(),
            required: witness_policy.minimum_distinct_organizations,
        });
    }
    if failure_domains.len() < witness_policy.minimum_distinct_regions {
        violations.push(RegistryContainmentBindingError::InsufficientFailureDomains {
            actual: failure_domains.len(),
            required: witness_policy.minimum_distinct_regions,
        });
    }
    if witness_policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(RegistryContainmentBindingError::MissingAlgorithmDiversity);
    }

    if !valid_exact_verification_policy(exact_verification_policy) {
        violations.push(RegistryContainmentBindingError::InvalidExactVerificationPolicy);
    }
    if exact_verification_providers.len() < exact_verification_policy.minimum_distinct_providers {
        violations.push(
            RegistryContainmentBindingError::InsufficientExactVerificationProviders {
                actual: exact_verification_providers.len(),
                required: exact_verification_policy.minimum_distinct_providers,
            },
        );
    }
    if exact_verification_providers.len() > exact_verification_policy.maximum_providers {
        violations.push(RegistryContainmentBindingError::TooManyExactVerificationProviders {
            actual: exact_verification_providers.len(),
            maximum: exact_verification_policy.maximum_providers,
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
            violations.push(RegistryContainmentBindingError::InvalidExactVerificationProvider(
                provider_id,
            ));
            continue;
        }
        if !seen_providers.insert(provider_id.clone()) {
            violations.push(RegistryContainmentBindingError::DuplicateExactVerificationProvider(
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
                RegistryContainmentBindingError::CheckpointSignatureRejected(provider_id.clone()),
            ),
            Err(reason) => violations.push(
                RegistryContainmentBindingError::VerificationProviderError {
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
                    RegistryContainmentBindingError::WitnessSignatureRejected {
                        provider: provider_id.clone(),
                        key_id: witness.signature.key_id.clone(),
                    },
                ),
                Err(reason) => violations.push(
                    RegistryContainmentBindingError::VerificationProviderError {
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
    let (containment_lineage_digest, containment_hop_count) = digest_containment_lineage(
        base_containment_authority,
        containment_authority_bridge,
        current_containment_authority,
    )
    .map_err(|error| vec![error])?;
    let (
        containment_to_observation_clock_lineage_digest,
        containment_to_observation_clock_hop_count,
    ) = digest_clock_lineage(
        current_containment_authorization_basis,
        containment_to_observation_clock_bridge,
        observation_basis,
    )
    .map_err(|error| vec![error])?;

    let commitment = CapabilityCommitment {
        schema: CONTAINMENT_CURRENT_WITNESS_REGISTRY_HEAD_SCHEMA,
        registry_head_id: registry_head.id().to_hex(),
        activated_registry_id: activated_registry.id().to_hex(),
        registry_digest: activated_registry.registry_digest().to_hex(),
        registry_sequence: activated_registry.sequence(),
        containment_authority_id: current_containment_authority.id().to_hex(),
        containment_state_digest: current_containment_authority.state_digest().to_hex(),
        containment_generation: current_containment_authority.generation(),
        compromise_tracker_digest: current_containment_authority.compromise_tracker_digest().to_hex(),
        containment_publication_digest: containment_publication_digest.to_hex(),
        containment_publication_entry_sequence: containment_entry.0,
        transparency_log_digest: transparency_log_digest.to_hex(),
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
        observation_clock_envelope_id: observation_clock.id().to_hex(),
        observation_operational_basis_id: observation_basis.id().to_hex(),
        containment_authorization_clock_envelope_id: containment_authorization_clock.id().to_hex(),
        containment_authorization_operational_basis_id: current_containment_authorization_basis
            .id()
            .to_hex(),
        containment_lineage_digest: containment_lineage_digest.to_hex(),
        containment_hop_count,
        containment_to_observation_clock_lineage_digest:
            containment_to_observation_clock_lineage_digest.to_hex(),
        containment_to_observation_clock_hop_count,
    };
    let id = ContainmentCurrentWitnessRegistryHeadIdV1(
        hash_serializable(CAPABILITY_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(ContainmentCurrentWitnessRegistryHeadV1 {
        id,
        registry_head_id: registry_head.id(),
        activated_registry_id: activated_registry.id().to_hex(),
        registry_digest: activated_registry.registry_digest(),
        registry_sequence: activated_registry.sequence(),
        containment_authority_id: current_containment_authority.id(),
        containment_state_digest: current_containment_authority.state_digest(),
        containment_generation: current_containment_authority.generation(),
        compromise_tracker_digest: current_containment_authority.compromise_tracker_digest(),
        containment_publication_digest,
        containment_publication_entry_sequence: containment_entry.0,
        transparency_log_digest,
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
        observation_clock_envelope_id: observation_clock.id(),
        observation_operational_basis_id: observation_basis.id(),
        containment_authorization_clock_envelope_id: containment_authorization_clock.id(),
        containment_authorization_operational_basis_id: current_containment_authorization_basis.id(),
        containment_lineage_digest,
        containment_hop_count,
        containment_to_observation_clock_lineage_digest,
        containment_to_observation_clock_hop_count,
    })
}

fn inspect_containment_head_log(
    log: &TransparencyLog,
    candidate_generation: u64,
) -> Result<
    (Option<u64>, Option<(u64, Sha256Digest, u64)>),
    Vec<RegistryContainmentBindingError>,
> {
    let mut violations = Vec::new();
    let mut previous_generation = None;
    let mut latest_generation = None;
    let mut candidate_entry = None;
    for entry in &log.entries {
        let generation = match parse_containment_generation(&entry.kind) {
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
                    violations.push(RegistryContainmentBindingError::DuplicateContainmentGeneration(
                        generation,
                    ));
                } else {
                    violations.push(RegistryContainmentBindingError::ContainmentGenerationRegressed {
                        previous,
                        current: generation,
                    });
                }
            }
        }
        previous_generation = Some(generation);
        latest_generation = Some(latest_generation.map_or(generation, |latest: u64| latest.max(generation)));
        if generation == candidate_generation {
            if candidate_entry.is_some() {
                violations.push(RegistryContainmentBindingError::DuplicateContainmentGeneration(
                    generation,
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
        Ok((latest_generation, candidate_entry))
    } else {
        Err(violations)
    }
}

fn parse_containment_generation(
    kind: &str,
) -> Result<Option<u64>, RegistryContainmentBindingError> {
    let Some(suffix) = kind.strip_prefix(CONTAINMENT_HEAD_LOG_KIND_PREFIX) else {
        return Ok(None);
    };
    let generation = suffix.parse::<u64>().map_err(|_| {
        RegistryContainmentBindingError::MalformedContainmentHeadKind(kind.to_string())
    })?;
    if generation == 0 || suffix != generation.to_string() {
        return Err(RegistryContainmentBindingError::MalformedContainmentHeadKind(
            kind.to_string(),
        ));
    }
    Ok(Some(generation))
}

fn verify_containment_lineage(
    start: &ClockGovernedContainmentStateV1,
    bridge: &[ClockGovernedContainmentStateV1],
    current: &ClockGovernedContainmentStateV1,
) -> Result<(), RegistryContainmentBindingError> {
    if current.id() == start.id() {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(RegistryContainmentBindingError::BrokenContainmentLineage {
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
) -> Result<(), RegistryContainmentBindingError> {
    if current.previous_authority_id() != Some(previous.id()) {
        return Err(RegistryContainmentBindingError::BrokenContainmentLineage {
            hop,
            expected_previous_authority: previous.id().to_hex(),
            actual_previous_authority: current
                .previous_authority_id()
                .map(|value| value.to_hex()),
        });
    }
    if current.previous_state_digest() != Some(previous.state_digest()) {
        return Err(RegistryContainmentBindingError::BrokenContainmentStateLineage { hop });
    }
    let expected_generation = previous
        .generation()
        .checked_add(1)
        .ok_or(RegistryContainmentBindingError::ContainmentGenerationOverflow)?;
    if current.generation() != expected_generation {
        return Err(RegistryContainmentBindingError::ContainmentGenerationNotAdjacent {
            previous: previous.generation(),
            current: current.generation(),
        });
    }
    Ok(())
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), RegistryContainmentBindingError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(RegistryContainmentBindingError::BrokenClockLineage {
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
            return Err(RegistryContainmentBindingError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(RegistryContainmentBindingError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn digest_containment_lineage(
    start: &ClockGovernedContainmentStateV1,
    bridge: &[ClockGovernedContainmentStateV1],
    end: &ClockGovernedContainmentStateV1,
) -> Result<(Sha256Digest, usize), RegistryContainmentBindingError> {
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

fn digest_clock_lineage(
    start: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    end: &OperationalClockBasisV1,
) -> Result<(Sha256Digest, usize), RegistryContainmentBindingError> {
    let mut ids = Vec::with_capacity(bridge.len() + 2);
    ids.push(start.id().to_hex());
    ids.extend(bridge.iter().map(|basis| basis.id().to_hex()));
    if end.id() != start.id() {
        ids.push(end.id().to_hex());
    }
    let digest = hash_serializable(CONTAINMENT_TO_OBSERVATION_CLOCK_LINEAGE_DOMAIN, &ids)?;
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
) -> Result<Sha256Digest, RegistryContainmentBindingError> {
    if !valid_witness_policy(policy) {
        return Err(RegistryContainmentBindingError::WitnessPolicyInvalid);
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
    violations: &mut Vec<RegistryContainmentBindingError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| &record.algorithm == algorithm && record.key_id == key_id)
    else {
        violations.push(RegistryContainmentBindingError::SignerUnknown(key_id.to_string()));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(RegistryContainmentBindingError::SignerNotActive(key_id.to_string()));
    }
    if !record.usages.contains(&usage) {
        violations.push(RegistryContainmentBindingError::SignerUsageNotAllowed(
            key_id.to_string(),
        ));
    }
    if record.not_before_unix_s > evidence_time_unix_s
        || record
            .not_after_unix_s
            .is_some_and(|not_after| evidence_time_unix_s >= not_after)
    {
        violations.push(RegistryContainmentBindingError::SignerInvalidAtEvidenceTime(
            key_id.to_string(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(RegistryContainmentBindingError::SignerNotValidAcrossEnvelope {
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
                RegistryContainmentBindingError::SignerCompromisedAcrossEnvelope(
                    key_id.to_string(),
                ),
            ),
            Err(reason) => violations.push(RegistryContainmentBindingError::CompromiseTimeInvalid {
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

fn seconds_to_millis(value: u64) -> Result<u64, RegistryContainmentBindingError> {
    value
        .checked_mul(1_000)
        .ok_or(RegistryContainmentBindingError::TimeScaleOverflow)
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
) -> Result<Sha256Digest, RegistryContainmentBindingError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| RegistryContainmentBindingError::Encoding(error.to_string()))?;
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
    fn containment_log_rejects_regression() {
        let mut log = TransparencyLog::default();
        log.append(100, format!("{CONTAINMENT_HEAD_LOG_KIND_PREFIX}3"), sha256(b"three"))
            .unwrap();
        log.append(101, format!("{CONTAINMENT_HEAD_LOG_KIND_PREFIX}2"), sha256(b"two"))
            .unwrap();
        let errors = inspect_containment_head_log(&log, 3).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            RegistryContainmentBindingError::ContainmentGenerationRegressed {
                previous: 3,
                current: 2
            }
        )));
    }

    #[test]
    fn containment_log_rejects_duplicate_generation() {
        let mut log = TransparencyLog::default();
        log.append(100, format!("{CONTAINMENT_HEAD_LOG_KIND_PREFIX}2"), sha256(b"a"))
            .unwrap();
        log.append(101, format!("{CONTAINMENT_HEAD_LOG_KIND_PREFIX}2"), sha256(b"b"))
            .unwrap();
        let errors = inspect_containment_head_log(&log, 2).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            RegistryContainmentBindingError::DuplicateContainmentGeneration(2)
        )));
    }
}
