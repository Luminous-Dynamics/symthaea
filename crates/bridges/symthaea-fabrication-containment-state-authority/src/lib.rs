// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold-authorized live lineage for fabrication containment state.
//!
//! `FabricationContainmentState` is durable evidence data, but it is serializable and therefore is
//! not by itself live authority. This bridge establishes an opaque authority chain whose genesis is
//! canonical and whose successors must pass the kernel's strict hash-linked successor theorem.
//!
//! Successor authorization is deliberately non-circular: the threshold ceremony must have been
//! qualified against the *previous* authorized compromise tracker, while the current witnessed
//! trust-snapshot head must also commit that exact previous containment-state/tracker digest. The
//! proposed successor cannot omit a prior compromise in order to make its own signers eligible.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state, verify_containment_state_successor,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::signer_compromise_tracker::{
    SignerCompromiseTracker, digest_signer_compromise_tracker,
};
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::KeyUsage;
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_fabrication_trust_snapshot_head::{
    QuorumObservedTrustSnapshotHeadIdV1, QuorumObservedTrustSnapshotHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const CONTAINMENT_STATE_GENESIS_PURPOSE: &str = "containment-state-genesis-v1";
pub const CONTAINMENT_STATE_SUCCESSOR_PURPOSE: &str = "containment-state-successor-v1";
pub const PREPARED_CONTAINMENT_STATE_AUTHORITY_SCHEMA: &str =
    "symthaea.fabrication.prepared-containment-state-authority.v1";
pub const CLOCK_GOVERNED_CONTAINMENT_STATE_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-containment-state.v1";

const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const PREPARED_AUTHORITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-containment-state-authority.v1\0";
const AUTHORIZED_STATE_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-containment-state.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedContainmentStateAuthorityIdV1(Sha256Digest);

impl PreparedContainmentStateAuthorityIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedContainmentStateIdV1(Sha256Digest);

impl ClockGovernedContainmentStateIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proposal for either canonical containment genesis or one exact strict successor.
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedContainmentStateAuthorityV1 {
    id: PreparedContainmentStateAuthorityIdV1,
    purpose: String,
    proposed_state: FabricationContainmentState,
    proposed_state_digest: Sha256Digest,
    previous_authority_id: Option<ClockGovernedContainmentStateIdV1>,
    previous_state_digest: Option<Sha256Digest>,
    previous_compromise_tracker_digest: Sha256Digest,
    proposed_compromise_tracker_digest: Sha256Digest,
    trust_head_id: QuorumObservedTrustSnapshotHeadIdV1,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    threshold_policy_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    clock_lower_unix_ms: u64,
    clock_upper_unix_ms: u64,
}

impl PreparedContainmentStateAuthorityV1 {
    pub fn id(&self) -> PreparedContainmentStateAuthorityIdV1 {
        self.id
    }
    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }
    pub fn purpose(&self) -> &str {
        &self.purpose
    }
    pub fn proposed_state_digest(&self) -> Sha256Digest {
        self.proposed_state_digest
    }
    pub fn proposed_generation(&self) -> u64 {
        self.proposed_state.generation
    }
    pub fn previous_authority_id(&self) -> Option<ClockGovernedContainmentStateIdV1> {
        self.previous_authority_id
    }
    pub fn previous_state_digest(&self) -> Option<Sha256Digest> {
        self.previous_state_digest
    }
    pub fn previous_compromise_tracker_digest(&self) -> Sha256Digest {
        self.previous_compromise_tracker_digest
    }
    pub fn proposed_compromise_tracker_digest(&self) -> Sha256Digest {
        self.proposed_compromise_tracker_digest
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
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

/// Opaque live authority for one exact containment-state generation.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedContainmentStateV1 {
    id: ClockGovernedContainmentStateIdV1,
    state: FabricationContainmentState,
    state_digest: Sha256Digest,
    previous_authority_id: Option<ClockGovernedContainmentStateIdV1>,
    previous_state_digest: Option<Sha256Digest>,
    authorization_previous_compromise_tracker_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    trust_head_id: QuorumObservedTrustSnapshotHeadIdV1,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    threshold_policy_digest: Sha256Digest,
    ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    ceremony_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl ClockGovernedContainmentStateV1 {
    pub fn id(&self) -> ClockGovernedContainmentStateIdV1 {
        self.id
    }
    pub fn state(&self) -> &FabricationContainmentState {
        &self.state
    }
    pub fn state_digest(&self) -> Sha256Digest {
        self.state_digest
    }
    pub fn generation(&self) -> u64 {
        self.state.generation
    }
    pub fn previous_authority_id(&self) -> Option<ClockGovernedContainmentStateIdV1> {
        self.previous_authority_id
    }
    pub fn previous_state_digest(&self) -> Option<Sha256Digest> {
        self.previous_state_digest
    }
    pub fn compromise_tracker(&self) -> &SignerCompromiseTracker {
        &self.state.signer_compromise_tracker
    }
    pub fn authorization_previous_compromise_tracker_digest(&self) -> Sha256Digest {
        self.authorization_previous_compromise_tracker_digest
    }
    pub fn compromise_tracker_digest(&self) -> Sha256Digest {
        self.compromise_tracker_digest
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
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }
    pub fn ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.ceremony_id
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }
    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainmentStateAuthorityError {
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    StateInvalid(String),
    NonCanonicalGenesis,
    InvalidSuccessor(String),
    TrustHeadContainmentMismatch,
    TrustHeadCompromiseTrackerMismatch,
    TrustSnapshotSequenceRollback { previous: u64, proposed: u64 },
    TrustSnapshotSameSequenceSubstitution,
    ObservationBasisMismatch,
    ObservationEnvelopeMismatch,
    TimeScaleOverflow,
    Encoding(String),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    CeremonySignerInvalidatedByProposedState(String),
}

#[derive(Serialize)]
struct ThresholdPolicyCommitment<'a> {
    minimum_distinct_signers: usize,
    maximum_approvals: usize,
    require_algorithm_diversity: bool,
    required_algorithms: &'a BTreeSet<SignatureAlgorithm>,
    allowed_key_ids: &'a Option<BTreeSet<String>>,
    key_usage: KeyUsage,
}

#[derive(Serialize)]
struct PreparedAuthorityCommitment {
    schema: &'static str,
    purpose: String,
    proposed_state_digest: String,
    proposed_generation: u64,
    previous_authority_id: Option<String>,
    previous_state_digest: Option<String>,
    previous_compromise_tracker_digest: String,
    proposed_compromise_tracker_digest: String,
    trust_head_id: String,
    trust_snapshot_digest: String,
    trust_snapshot_sequence: u64,
    threshold_policy_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
    clock_lower_unix_ms: u64,
    clock_upper_unix_ms: u64,
}

#[derive(Serialize)]
struct AuthorizedStateCommitment {
    schema: &'static str,
    prepared_id: String,
    ceremony_id: String,
    ceremony_digest: String,
}

pub fn prepare_clock_governed_containment_genesis_v1(
    state: &FabricationContainmentState,
    trust_head: &QuorumObservedTrustSnapshotHeadV1,
    threshold_policy: &ThresholdCeremonyPolicy,
    observation_basis: &OperationalClockBasisV1,
) -> Result<PreparedContainmentStateAuthorityV1, Vec<ContainmentStateAuthorityError>> {
    let mut violations = Vec::new();
    validate_common_inputs(
        state,
        trust_head,
        threshold_policy,
        observation_basis,
        &mut violations,
    );
    if !is_canonical_genesis(state) {
        violations.push(ContainmentStateAuthorityError::NonCanonicalGenesis);
    }
    if !violations.is_empty() {
        return Err(violations);
    }
    build_prepared(
        CONTAINMENT_STATE_GENESIS_PURPOSE,
        state,
        None,
        None,
        digest_signer_compromise_tracker(&state.signer_compromise_tracker).map_err(|error| {
            vec![ContainmentStateAuthorityError::StateInvalid(format!("{error:?}"))]
        })?,
        trust_head,
        threshold_policy,
        observation_basis,
    )
    .map_err(|error| vec![error])
}

pub fn prepare_clock_governed_containment_successor_v1(
    previous: &ClockGovernedContainmentStateV1,
    proposed: &FabricationContainmentState,
    trust_head: &QuorumObservedTrustSnapshotHeadV1,
    threshold_policy: &ThresholdCeremonyPolicy,
    observation_basis: &OperationalClockBasisV1,
) -> Result<PreparedContainmentStateAuthorityV1, Vec<ContainmentStateAuthorityError>> {
    let mut violations = Vec::new();
    validate_common_inputs(
        proposed,
        trust_head,
        threshold_policy,
        observation_basis,
        &mut violations,
    );
    if let Err(error) = verify_containment_state_successor(previous.state(), proposed) {
        violations.push(ContainmentStateAuthorityError::InvalidSuccessor(format!(
            "{error:?}"
        )));
    }
    if trust_head.containment_state_digest() != previous.state_digest() {
        violations.push(ContainmentStateAuthorityError::TrustHeadContainmentMismatch);
    }
    if trust_head.compromise_tracker_digest() != previous.compromise_tracker_digest() {
        violations.push(ContainmentStateAuthorityError::TrustHeadCompromiseTrackerMismatch);
    }
    if trust_head.snapshot_sequence() < previous.trust_snapshot_sequence() {
        violations.push(ContainmentStateAuthorityError::TrustSnapshotSequenceRollback {
            previous: previous.trust_snapshot_sequence(),
            proposed: trust_head.snapshot_sequence(),
        });
    }
    if trust_head.snapshot_sequence() == previous.trust_snapshot_sequence()
        && trust_head.snapshot_digest() != previous.trust_snapshot_digest()
    {
        violations.push(ContainmentStateAuthorityError::TrustSnapshotSameSequenceSubstitution);
    }
    if !violations.is_empty() {
        return Err(violations);
    }
    build_prepared(
        CONTAINMENT_STATE_SUCCESSOR_PURPOSE,
        proposed,
        Some(previous.id()),
        Some(previous.state_digest()),
        previous.compromise_tracker_digest(),
        trust_head,
        threshold_policy,
        observation_basis,
    )
    .map_err(|error| vec![error])
}

pub fn authorize_clock_governed_containment_state_v1(
    prepared: PreparedContainmentStateAuthorityV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<ClockGovernedContainmentStateV1, ContainmentStateAuthorityError> {
    if ceremony.purpose() != prepared.purpose {
        return Err(ContainmentStateAuthorityError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(ContainmentStateAuthorityError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(ContainmentStateAuthorityError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(ContainmentStateAuthorityError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.previous_compromise_tracker_digest {
        return Err(ContainmentStateAuthorityError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(ContainmentStateAuthorityError::ClockEnvelopeMismatch);
    }

    for (algorithm, key_id) in ceremony.signers() {
        for compromise in prepared
            .proposed_state
            .signer_compromise_tracker
            .records()
            .iter()
            .filter(|record| {
                &record.signer.algorithm == algorithm
                    && record.signer.key_id == *key_id
                    && record.affected_usages.contains(&KeyUsage::ThresholdCeremony)
            })
        {
            let effective_at_ms = seconds_to_millis(compromise.effective_at_unix_s)?;
            if effective_at_ms <= prepared.clock_upper_unix_ms {
                return Err(
                    ContainmentStateAuthorityError::CeremonySignerInvalidatedByProposedState(
                        key_id.clone(),
                    ),
                );
            }
        }
    }

    let id = ClockGovernedContainmentStateIdV1(hash_serializable(
        AUTHORIZED_STATE_DOMAIN,
        &AuthorizedStateCommitment {
            schema: CLOCK_GOVERNED_CONTAINMENT_STATE_SCHEMA,
            prepared_id: prepared.id.to_hex(),
            ceremony_id: ceremony.id().to_hex(),
            ceremony_digest: ceremony.ceremony_digest().to_hex(),
        },
    )?);

    Ok(ClockGovernedContainmentStateV1 {
        id,
        state: prepared.proposed_state,
        state_digest: prepared.proposed_state_digest,
        previous_authority_id: prepared.previous_authority_id,
        previous_state_digest: prepared.previous_state_digest,
        authorization_previous_compromise_tracker_digest: prepared.previous_compromise_tracker_digest,
        compromise_tracker_digest: prepared.proposed_compromise_tracker_digest,
        trust_head_id: prepared.trust_head_id,
        trust_snapshot_digest: prepared.trust_snapshot_digest,
        trust_snapshot_sequence: prepared.trust_snapshot_sequence,
        threshold_policy_digest: prepared.threshold_policy_digest,
        ceremony_id: ceremony.id(),
        ceremony_digest: ceremony.ceremony_digest(),
        clock_envelope_id: prepared.clock_envelope_id,
        operational_basis_id: prepared.operational_basis_id,
    })
}

fn validate_common_inputs(
    state: &FabricationContainmentState,
    trust_head: &QuorumObservedTrustSnapshotHeadV1,
    threshold_policy: &ThresholdCeremonyPolicy,
    observation_basis: &OperationalClockBasisV1,
    violations: &mut Vec<ContainmentStateAuthorityError>,
) {
    if !valid_threshold_policy(threshold_policy) {
        violations.push(ContainmentStateAuthorityError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::ThresholdCeremony {
        violations.push(ContainmentStateAuthorityError::ThresholdPolicyUsageMismatch);
    }
    if let Err(error) = state.validate() {
        violations.push(ContainmentStateAuthorityError::StateInvalid(format!(
            "{error:?}"
        )));
    }
    if observation_basis.id() != trust_head.observation_operational_basis_id() {
        violations.push(ContainmentStateAuthorityError::ObservationBasisMismatch);
    }
    match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(clock) if clock.id() == trust_head.observation_clock_envelope_id() => {}
        Ok(_) => violations.push(ContainmentStateAuthorityError::ObservationEnvelopeMismatch),
        Err(error) => violations.push(ContainmentStateAuthorityError::StateInvalid(format!(
            "clock: {error:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_prepared(
    purpose: &str,
    state: &FabricationContainmentState,
    previous_authority_id: Option<ClockGovernedContainmentStateIdV1>,
    previous_state_digest: Option<Sha256Digest>,
    previous_compromise_tracker_digest: Sha256Digest,
    trust_head: &QuorumObservedTrustSnapshotHeadV1,
    threshold_policy: &ThresholdCeremonyPolicy,
    observation_basis: &OperationalClockBasisV1,
) -> Result<PreparedContainmentStateAuthorityV1, ContainmentStateAuthorityError> {
    let proposed_state_digest = digest_containment_state(state)
        .map_err(|error| ContainmentStateAuthorityError::StateInvalid(format!("{error:?}")))?;
    let proposed_compromise_tracker_digest =
        digest_signer_compromise_tracker(&state.signer_compromise_tracker)
            .map_err(|error| ContainmentStateAuthorityError::StateInvalid(format!("{error:?}")))?;
    let threshold_policy_digest = digest_threshold_policy(threshold_policy)?;
    let clock = derive_clock_governance_evaluation_envelope_v1(observation_basis)
        .map_err(|error| ContainmentStateAuthorityError::StateInvalid(format!("clock: {error:?}")))?;
    let commitment = PreparedAuthorityCommitment {
        schema: PREPARED_CONTAINMENT_STATE_AUTHORITY_SCHEMA,
        purpose: purpose.to_string(),
        proposed_state_digest: proposed_state_digest.to_hex(),
        proposed_generation: state.generation,
        previous_authority_id: previous_authority_id.map(|value| value.to_hex()),
        previous_state_digest: previous_state_digest.map(Sha256Digest::to_hex),
        previous_compromise_tracker_digest: previous_compromise_tracker_digest.to_hex(),
        proposed_compromise_tracker_digest: proposed_compromise_tracker_digest.to_hex(),
        trust_head_id: trust_head.id().to_hex(),
        trust_snapshot_digest: trust_head.snapshot_digest().to_hex(),
        trust_snapshot_sequence: trust_head.snapshot_sequence(),
        threshold_policy_digest: threshold_policy_digest.to_hex(),
        clock_envelope_id: clock.id().to_hex(),
        operational_basis_id: observation_basis.id().to_hex(),
        clock_lower_unix_ms: clock.lower_unix_ms(),
        clock_upper_unix_ms: clock.upper_unix_ms(),
    };
    let id = PreparedContainmentStateAuthorityIdV1(hash_serializable(
        PREPARED_AUTHORITY_DOMAIN,
        &commitment,
    )?);
    Ok(PreparedContainmentStateAuthorityV1 {
        id,
        purpose: purpose.to_string(),
        proposed_state: state.clone(),
        proposed_state_digest,
        previous_authority_id,
        previous_state_digest,
        previous_compromise_tracker_digest,
        proposed_compromise_tracker_digest,
        trust_head_id: trust_head.id(),
        trust_snapshot_digest: trust_head.snapshot_digest(),
        trust_snapshot_sequence: trust_head.snapshot_sequence(),
        threshold_policy_digest,
        clock_envelope_id: clock.id(),
        operational_basis_id: observation_basis.id(),
        clock_lower_unix_ms: clock.lower_unix_ms(),
        clock_upper_unix_ms: clock.upper_unix_ms(),
    })
}

fn is_canonical_genesis(state: &FabricationContainmentState) -> bool {
    FabricationContainmentState::genesis(
        state.release_resilience_generation,
        state.release_resilience_state_digest,
    )
    .is_ok_and(|canonical| canonical == *state)
}

fn valid_threshold_policy(policy: &ThresholdCeremonyPolicy) -> bool {
    policy.minimum_distinct_signers > 0
        && policy.maximum_approvals > 0
        && policy.minimum_distinct_signers <= policy.maximum_approvals
        && policy.maximum_approvals <= MAX_THRESHOLD_APPROVALS
        && policy.required_algorithms.iter().all(SignatureAlgorithm::is_canonical)
        && policy.allowed_key_ids.as_ref().is_none_or(|ids| {
            ids.iter().all(|id| {
                !invalid_identifier(id) && id.len() <= MAX_THRESHOLD_KEY_ID_BYTES
            })
        })
}

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, ContainmentStateAuthorityError> {
    if !valid_threshold_policy(policy) {
        return Err(ContainmentStateAuthorityError::InvalidThresholdPolicy);
    }
    hash_serializable(
        THRESHOLD_POLICY_DOMAIN,
        &ThresholdPolicyCommitment {
            minimum_distinct_signers: policy.minimum_distinct_signers,
            maximum_approvals: policy.maximum_approvals,
            require_algorithm_diversity: policy.require_algorithm_diversity,
            required_algorithms: &policy.required_algorithms,
            allowed_key_ids: &policy.allowed_key_ids,
            key_usage: policy.key_usage,
        },
    )
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
}

fn seconds_to_millis(value: u64) -> Result<u64, ContainmentStateAuthorityError> {
    value
        .checked_mul(1_000)
        .ok_or(ContainmentStateAuthorityError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ContainmentStateAuthorityError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ContainmentStateAuthorityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_kernel_genesis_shape_is_accepted_as_canonical_genesis() {
        let state = FabricationContainmentState::genesis(1, Sha256Digest([1; 32])).unwrap();
        assert!(is_canonical_genesis(&state));
        let mut changed = state.clone();
        changed.latest_containment_replay_digest = Some(Sha256Digest([2; 32]));
        assert!(!is_canonical_genesis(&changed));
    }

    #[test]
    fn containment_purposes_are_distinct() {
        assert_ne!(CONTAINMENT_STATE_GENESIS_PURPOSE, CONTAINMENT_STATE_SUCCESSOR_PURPOSE);
    }
}
