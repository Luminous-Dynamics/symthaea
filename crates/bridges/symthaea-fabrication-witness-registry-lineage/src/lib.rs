// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-circular successor authorization for transparency-witness authority registries.
//!
//! The original `WitnessAuthorityRegistryV1` is a sound opaque genesis authority, but it has no
//! successor transition. This bridge adopts that opaque genesis as the root of a new live lineage
//! and authorizes an exact future successor only from a coherent witnessed trust + containment view
//! that was itself observed under the registry being superseded.
//!
//! Authorization is deliberately not activation. The resulting transition remains future authority
//! until a later theorem proves the scheduled activation has definitely arrived under a descendant
//! trusted-clock lineage and requalifies the proposed witness keys at that time.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_fabrication_containment_head::{
    QuorumObservedContainmentHeadIdV1, QuorumObservedContainmentHeadV1,
};
use symthaea_fabrication_containment_state_authority::{
    ClockGovernedContainmentStateIdV1, ClockGovernedContainmentStateV1,
};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_fabrication_trust_snapshot_head::{
    QuorumObservedTrustSnapshotHeadIdV1, QuorumObservedTrustSnapshotHeadV1,
};
use symthaea_fabrication_witness_authority::{
    MAX_WITNESS_AUTHORITY_PROFILES, WitnessAuthorityProfileV1, WitnessAuthorityRegistryIdV1,
    WitnessAuthorityRegistryV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const WITNESS_REGISTRY_TRANSITION_PURPOSE: &str = "witness-authority-registry-transition-v1";
pub const GOVERNED_WITNESS_REGISTRY_SCHEMA: &str =
    "symthaea.fabrication.governed-witness-authority-registry.v1";
pub const PREPARED_WITNESS_REGISTRY_TRANSITION_SCHEMA: &str =
    "symthaea.fabrication.prepared-witness-authority-registry-transition.v1";
pub const AUTHORIZED_WITNESS_REGISTRY_TRANSITION_SCHEMA: &str =
    "symthaea.fabrication.authorized-witness-authority-registry-transition.v1";

const PROFILE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-authority-registry-digest.v1\0";
const ROOT_DOMAIN: &[u8] =
    b"symthaea.fabrication.governed-witness-authority-registry-root.v1\0";
const TRANSITION_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-authority-registry-transition-policy.v1\0";
const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const PREPARED_TRANSITION_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-witness-authority-registry-transition.v1\0";
const AUTHORIZED_TRANSITION_DOMAIN: &[u8] =
    b"symthaea.fabrication.authorized-witness-authority-registry-transition.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessRegistryTransitionPolicyV1 {
    pub maximum_activation_delay_s: u64,
    pub minimum_profiles: usize,
}

impl Default for WitnessRegistryTransitionPolicyV1 {
    fn default() -> Self {
        Self {
            maximum_activation_delay_s: 300,
            minimum_profiles: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GovernedWitnessAuthorityRegistryIdV1(Sha256Digest);

impl GovernedWitnessAuthorityRegistryIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Active root/current registry lineage. In this tranche only the legacy opaque genesis can be
/// adopted into this type. Authorized successors remain separate until a later activation theorem.
#[derive(Debug, Clone)]
#[must_use]
pub struct GovernedWitnessAuthorityRegistryV1 {
    id: GovernedWitnessAuthorityRegistryIdV1,
    sequence: u64,
    profiles: Vec<WitnessAuthorityProfileV1>,
    registry_digest: Sha256Digest,
    legacy_genesis_registry_id: WitnessAuthorityRegistryIdV1,
    legacy_genesis_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
}

impl GovernedWitnessAuthorityRegistryV1 {
    pub fn id(&self) -> GovernedWitnessAuthorityRegistryIdV1 {
        self.id
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn profiles(&self) -> &[WitnessAuthorityProfileV1] {
        &self.profiles
    }
    pub fn registry_digest(&self) -> Sha256Digest {
        self.registry_digest
    }
    pub fn legacy_genesis_registry_id(&self) -> WitnessAuthorityRegistryIdV1 {
        self.legacy_genesis_registry_id
    }
    pub fn legacy_genesis_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.legacy_genesis_ceremony_id
    }
    pub fn profile(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
    ) -> Option<&WitnessAuthorityProfileV1> {
        self.profiles
            .iter()
            .find(|profile| &profile.algorithm == algorithm && profile.key_id == key_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedWitnessRegistryTransitionIdV1(Sha256Digest);

impl PreparedWitnessRegistryTransitionIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedWitnessRegistryTransitionV1 {
    id: PreparedWitnessRegistryTransitionIdV1,
    previous_registry_id: GovernedWitnessAuthorityRegistryIdV1,
    previous_registry_digest: Sha256Digest,
    previous_sequence: u64,
    proposed_profiles: Vec<WitnessAuthorityProfileV1>,
    proposed_registry_digest: Sha256Digest,
    proposed_sequence: u64,
    transition_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_head_id: QuorumObservedTrustSnapshotHeadIdV1,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    containment_head_id: QuorumObservedContainmentHeadIdV1,
    containment_authority_id: ClockGovernedContainmentStateIdV1,
    containment_state_digest: Sha256Digest,
    containment_generation: u64,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    activates_at_unix_ms: u64,
}

impl PreparedWitnessRegistryTransitionV1 {
    pub fn id(&self) -> PreparedWitnessRegistryTransitionIdV1 {
        self.id
    }
    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }
    pub fn previous_registry_id(&self) -> GovernedWitnessAuthorityRegistryIdV1 {
        self.previous_registry_id
    }
    pub fn previous_registry_digest(&self) -> Sha256Digest {
        self.previous_registry_digest
    }
    pub fn previous_sequence(&self) -> u64 {
        self.previous_sequence
    }
    pub fn proposed_profiles(&self) -> &[WitnessAuthorityProfileV1] {
        &self.proposed_profiles
    }
    pub fn proposed_registry_digest(&self) -> Sha256Digest {
        self.proposed_registry_digest
    }
    pub fn proposed_sequence(&self) -> u64 {
        self.proposed_sequence
    }
    pub fn transition_policy_digest(&self) -> Sha256Digest {
        self.transition_policy_digest
    }
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
    }
    pub fn trust_head_id(&self) -> QuorumObservedTrustSnapshotHeadIdV1 {
        self.trust_head_id
    }
    pub fn containment_head_id(&self) -> QuorumObservedContainmentHeadIdV1 {
        self.containment_head_id
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
    pub fn activates_at_unix_ms(&self) -> u64 {
        self.activates_at_unix_ms
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AuthorizedWitnessRegistryTransitionIdV1(Sha256Digest);

impl AuthorizedWitnessRegistryTransitionIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque authority for one future registry successor. This is **not** the active successor registry.
#[derive(Debug, Clone)]
#[must_use]
pub struct AuthorizedWitnessRegistryTransitionV1 {
    id: AuthorizedWitnessRegistryTransitionIdV1,
    prepared_id: PreparedWitnessRegistryTransitionIdV1,
    previous_registry_id: GovernedWitnessAuthorityRegistryIdV1,
    previous_registry_digest: Sha256Digest,
    previous_sequence: u64,
    proposed_profiles: Vec<WitnessAuthorityProfileV1>,
    proposed_registry_digest: Sha256Digest,
    proposed_sequence: u64,
    transition_policy_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_head_id: QuorumObservedTrustSnapshotHeadIdV1,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    containment_head_id: QuorumObservedContainmentHeadIdV1,
    containment_authority_id: ClockGovernedContainmentStateIdV1,
    containment_state_digest: Sha256Digest,
    containment_generation: u64,
    compromise_tracker_digest: Sha256Digest,
    ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    ceremony_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    activates_at_unix_ms: u64,
}

impl AuthorizedWitnessRegistryTransitionV1 {
    pub fn id(&self) -> AuthorizedWitnessRegistryTransitionIdV1 {
        self.id
    }
    pub fn prepared_id(&self) -> PreparedWitnessRegistryTransitionIdV1 {
        self.prepared_id
    }
    pub fn previous_registry_id(&self) -> GovernedWitnessAuthorityRegistryIdV1 {
        self.previous_registry_id
    }
    pub fn previous_registry_digest(&self) -> Sha256Digest {
        self.previous_registry_digest
    }
    pub fn previous_sequence(&self) -> u64 {
        self.previous_sequence
    }
    pub fn proposed_profiles(&self) -> &[WitnessAuthorityProfileV1] {
        &self.proposed_profiles
    }
    pub fn proposed_registry_digest(&self) -> Sha256Digest {
        self.proposed_registry_digest
    }
    pub fn proposed_sequence(&self) -> u64 {
        self.proposed_sequence
    }
    pub fn transition_policy_digest(&self) -> Sha256Digest {
        self.transition_policy_digest
    }
    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
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
    pub fn containment_head_id(&self) -> QuorumObservedContainmentHeadIdV1 {
        self.containment_head_id
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
    pub fn activates_at_unix_ms(&self) -> u64 {
        self.activates_at_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessRegistryLineageError {
    InvalidLegacyGenesis,
    RegistryDigestMismatch,
    InvalidTransitionPolicy,
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    SequenceOverflow,
    EmptyRegistry,
    TooFewProfiles { actual: usize, required: usize },
    TooManyProfiles { actual: usize, maximum: usize },
    InvalidProfile(String),
    DuplicateProfileSigner(String),
    NoRegistryChange,
    TrustSnapshotInvalid(String),
    TrustSnapshotHeadMismatch,
    TrustHeadObservedUnderDifferentRegistry,
    ContainmentHeadObservedUnderDifferentRegistry,
    ContainmentTrustHeadMismatch,
    ContainmentAuthorityMismatch,
    ContainmentStateMismatch,
    ContainmentTrackerMismatch,
    OperationalBasisMismatch,
    ClockEnvelopeMismatch,
    Clock(ClockGovernanceTimeError),
    TrustSnapshotNotValidAcrossEnvelope(ClockGovernanceTimeError),
    WitnessKeyUnknown(String),
    WitnessKeyNotActive(String),
    WitnessKeyUsageNotAllowed(String),
    WitnessKeyNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    WitnessKeyCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid { key_id: String, reason: ClockGovernanceTimeError },
    TimeScaleOverflow,
    ActivationMayBePast,
    ActivationMayBeTooLate,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    CeremonyTrustSnapshotMismatch,
    CeremonyCompromiseTrackerMismatch,
    CeremonyClockEnvelopeMismatch,
    Encoding(String),
}

#[derive(Serialize)]
struct RootCommitment {
    schema: &'static str,
    legacy_registry_id: String,
    legacy_registry_digest: String,
    legacy_registry_sequence: u64,
    legacy_genesis_ceremony_id: String,
}

#[derive(Serialize)]
struct TransitionPolicyCommitment {
    maximum_activation_delay_s: u64,
    minimum_profiles: usize,
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
struct PreparedTransitionCommitment {
    schema: &'static str,
    previous_registry_id: String,
    previous_registry_digest: String,
    previous_sequence: u64,
    proposed_registry_digest: String,
    proposed_sequence: u64,
    transition_policy_digest: String,
    threshold_policy_digest: String,
    trust_head_id: String,
    trust_snapshot_digest: String,
    trust_snapshot_sequence: u64,
    containment_head_id: String,
    containment_authority_id: String,
    containment_state_digest: String,
    containment_generation: u64,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
    activates_at_unix_ms: u64,
}

#[derive(Serialize)]
struct AuthorizedTransitionCommitment {
    schema: &'static str,
    prepared_id: String,
    ceremony_id: String,
    ceremony_digest: String,
}

pub fn adopt_legacy_witness_authority_registry_genesis_v1(
    registry: &WitnessAuthorityRegistryV1,
) -> Result<GovernedWitnessAuthorityRegistryV1, WitnessRegistryLineageError> {
    if registry.sequence() != 1 || registry.profiles().is_empty() {
        return Err(WitnessRegistryLineageError::InvalidLegacyGenesis);
    }
    let profiles = canonical_profiles(registry.profiles().to_vec())?;
    let registry_digest = digest_profiles(&profiles)?;
    if registry_digest != registry.registry_digest() {
        return Err(WitnessRegistryLineageError::RegistryDigestMismatch);
    }
    let commitment = RootCommitment {
        schema: GOVERNED_WITNESS_REGISTRY_SCHEMA,
        legacy_registry_id: registry.id().to_hex(),
        legacy_registry_digest: registry_digest.to_hex(),
        legacy_registry_sequence: registry.sequence(),
        legacy_genesis_ceremony_id: registry.genesis_ceremony_id().to_hex(),
    };
    let id = GovernedWitnessAuthorityRegistryIdV1(hash_serializable(ROOT_DOMAIN, &commitment)?);
    Ok(GovernedWitnessAuthorityRegistryV1 {
        id,
        sequence: registry.sequence(),
        profiles,
        registry_digest,
        legacy_genesis_registry_id: registry.id(),
        legacy_genesis_ceremony_id: registry.genesis_ceremony_id(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_witness_authority_registry_transition_v1(
    previous: &GovernedWitnessAuthorityRegistryV1,
    proposed_profiles: Vec<WitnessAuthorityProfileV1>,
    activates_at_unix_s: u64,
    transition_policy: &WitnessRegistryTransitionPolicyV1,
    threshold_policy: &ThresholdCeremonyPolicy,
    trust_head: &QuorumObservedTrustSnapshotHeadV1,
    containment_head: &QuorumObservedContainmentHeadV1,
    containment_authority: &ClockGovernedContainmentStateV1,
    trust_snapshot: &TrustSnapshot,
    operational_basis: &OperationalClockBasisV1,
) -> Result<PreparedWitnessRegistryTransitionV1, Vec<WitnessRegistryLineageError>> {
    let mut violations = Vec::new();

    if !valid_transition_policy(transition_policy) {
        violations.push(WitnessRegistryLineageError::InvalidTransitionPolicy);
    }
    if !valid_threshold_policy(threshold_policy) {
        violations.push(WitnessRegistryLineageError::InvalidThresholdPolicy);
    }
    if threshold_policy.key_usage != KeyUsage::ThresholdCeremony {
        violations.push(WitnessRegistryLineageError::ThresholdPolicyUsageMismatch);
    }

    let proposed_sequence = match previous.sequence.checked_add(1) {
        Some(value) => value,
        None => {
            violations.push(WitnessRegistryLineageError::SequenceOverflow);
            0
        }
    };
    let profiles = match canonical_profiles(proposed_profiles) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Vec::new()
        }
    };
    if profiles.is_empty() {
        violations.push(WitnessRegistryLineageError::EmptyRegistry);
    }
    if profiles.len() < transition_policy.minimum_profiles {
        violations.push(WitnessRegistryLineageError::TooFewProfiles {
            actual: profiles.len(),
            required: transition_policy.minimum_profiles,
        });
    }
    if profiles.len() > MAX_WITNESS_AUTHORITY_PROFILES {
        violations.push(WitnessRegistryLineageError::TooManyProfiles {
            actual: profiles.len(),
            maximum: MAX_WITNESS_AUTHORITY_PROFILES,
        });
    }
    let proposed_registry_digest = match digest_profiles(&profiles) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };
    if proposed_registry_digest == previous.registry_digest {
        violations.push(WitnessRegistryLineageError::NoRegistryChange);
    }

    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryLineageError::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(WitnessRegistryLineageError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if trust_snapshot_digest != trust_head.snapshot_digest()
        || trust_snapshot.sequence != trust_head.snapshot_sequence()
    {
        violations.push(WitnessRegistryLineageError::TrustSnapshotHeadMismatch);
    }
    if trust_head.witness_registry_digest() != previous.registry_digest
        || trust_head.witness_registry_sequence() != previous.sequence
    {
        violations.push(WitnessRegistryLineageError::TrustHeadObservedUnderDifferentRegistry);
    }
    if containment_head.witness_registry_digest() != previous.registry_digest
        || containment_head.witness_registry_sequence() != previous.sequence
    {
        violations.push(
            WitnessRegistryLineageError::ContainmentHeadObservedUnderDifferentRegistry,
        );
    }
    if containment_head.trust_head_id() != trust_head.id()
        || containment_head.trust_snapshot_digest() != trust_head.snapshot_digest()
        || containment_head.trust_snapshot_sequence() != trust_head.snapshot_sequence()
    {
        violations.push(WitnessRegistryLineageError::ContainmentTrustHeadMismatch);
    }
    if containment_head.authority_id() != containment_authority.id() {
        violations.push(WitnessRegistryLineageError::ContainmentAuthorityMismatch);
    }
    if containment_head.state_digest() != containment_authority.state_digest()
        || containment_head.generation() != containment_authority.generation()
    {
        violations.push(WitnessRegistryLineageError::ContainmentStateMismatch);
    }
    if containment_head.compromise_tracker_digest() != containment_authority.compromise_tracker_digest()
    {
        violations.push(WitnessRegistryLineageError::ContainmentTrackerMismatch);
    }
    if operational_basis.id() != containment_head.observation_operational_basis_id() {
        violations.push(WitnessRegistryLineageError::OperationalBasisMismatch);
    }
    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryLineageError::Clock(error));
            return Err(violations);
        }
    };
    if clock.id() != containment_head.observation_clock_envelope_id() {
        violations.push(WitnessRegistryLineageError::ClockEnvelopeMismatch);
    }
    if let Err(reason) = clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(WitnessRegistryLineageError::TrustSnapshotNotValidAcrossEnvelope(
            reason,
        ));
    }

    for profile in &profiles {
        requalify_proposed_witness_key(
            profile,
            trust_snapshot,
            containment_authority,
            &clock,
            &mut violations,
        );
    }

    let activates_at_unix_ms = match seconds_to_millis(activates_at_unix_s) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    if activates_at_unix_ms < clock.upper_unix_ms() {
        violations.push(WitnessRegistryLineageError::ActivationMayBePast);
    }
    let maximum_delay_ms = match seconds_to_millis(transition_policy.maximum_activation_delay_s) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            0
        }
    };
    let latest_activation = match clock.lower_unix_ms().checked_add(maximum_delay_ms) {
        Some(value) => value,
        None => {
            violations.push(WitnessRegistryLineageError::TimeScaleOverflow);
            0
        }
    };
    if activates_at_unix_ms > latest_activation {
        violations.push(WitnessRegistryLineageError::ActivationMayBeTooLate);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let transition_policy_digest = digest_transition_policy(transition_policy)
        .map_err(|error| vec![error])?;
    let threshold_policy_digest = digest_threshold_policy(threshold_policy)
        .map_err(|error| vec![error])?;
    let commitment = PreparedTransitionCommitment {
        schema: PREPARED_WITNESS_REGISTRY_TRANSITION_SCHEMA,
        previous_registry_id: previous.id.to_hex(),
        previous_registry_digest: previous.registry_digest.to_hex(),
        previous_sequence: previous.sequence,
        proposed_registry_digest: proposed_registry_digest.to_hex(),
        proposed_sequence,
        transition_policy_digest: transition_policy_digest.to_hex(),
        threshold_policy_digest: threshold_policy_digest.to_hex(),
        trust_head_id: trust_head.id().to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        trust_snapshot_sequence: trust_snapshot.sequence,
        containment_head_id: containment_head.id().to_hex(),
        containment_authority_id: containment_authority.id().to_hex(),
        containment_state_digest: containment_authority.state_digest().to_hex(),
        containment_generation: containment_authority.generation(),
        compromise_tracker_digest: containment_authority.compromise_tracker_digest().to_hex(),
        clock_envelope_id: clock.id().to_hex(),
        operational_basis_id: operational_basis.id().to_hex(),
        activates_at_unix_ms,
    };
    let id = PreparedWitnessRegistryTransitionIdV1(hash_serializable(
        PREPARED_TRANSITION_DOMAIN,
        &commitment,
    ).map_err(|error| vec![error])?);

    Ok(PreparedWitnessRegistryTransitionV1 {
        id,
        previous_registry_id: previous.id,
        previous_registry_digest: previous.registry_digest,
        previous_sequence: previous.sequence,
        proposed_profiles: profiles,
        proposed_registry_digest,
        proposed_sequence,
        transition_policy_digest,
        threshold_policy_digest,
        trust_head_id: trust_head.id(),
        trust_snapshot_digest,
        trust_snapshot_sequence: trust_snapshot.sequence,
        containment_head_id: containment_head.id(),
        containment_authority_id: containment_authority.id(),
        containment_state_digest: containment_authority.state_digest(),
        containment_generation: containment_authority.generation(),
        compromise_tracker_digest: containment_authority.compromise_tracker_digest(),
        clock_envelope_id: clock.id(),
        operational_basis_id: operational_basis.id(),
        activates_at_unix_ms,
    })
}

pub fn authorize_witness_authority_registry_transition_v1(
    prepared: PreparedWitnessRegistryTransitionV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<AuthorizedWitnessRegistryTransitionV1, WitnessRegistryLineageError> {
    if ceremony.purpose() != WITNESS_REGISTRY_TRANSITION_PURPOSE {
        return Err(WitnessRegistryLineageError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(WitnessRegistryLineageError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(WitnessRegistryLineageError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(WitnessRegistryLineageError::CeremonyTrustSnapshotMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest {
        return Err(WitnessRegistryLineageError::CeremonyCompromiseTrackerMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(WitnessRegistryLineageError::CeremonyClockEnvelopeMismatch);
    }
    let id = AuthorizedWitnessRegistryTransitionIdV1(hash_serializable(
        AUTHORIZED_TRANSITION_DOMAIN,
        &AuthorizedTransitionCommitment {
            schema: AUTHORIZED_WITNESS_REGISTRY_TRANSITION_SCHEMA,
            prepared_id: prepared.id.to_hex(),
            ceremony_id: ceremony.id().to_hex(),
            ceremony_digest: ceremony.ceremony_digest().to_hex(),
        },
    )?);
    Ok(AuthorizedWitnessRegistryTransitionV1 {
        id,
        prepared_id: prepared.id,
        previous_registry_id: prepared.previous_registry_id,
        previous_registry_digest: prepared.previous_registry_digest,
        previous_sequence: prepared.previous_sequence,
        proposed_profiles: prepared.proposed_profiles,
        proposed_registry_digest: prepared.proposed_registry_digest,
        proposed_sequence: prepared.proposed_sequence,
        transition_policy_digest: prepared.transition_policy_digest,
        threshold_policy_digest: prepared.threshold_policy_digest,
        trust_head_id: prepared.trust_head_id,
        trust_snapshot_digest: prepared.trust_snapshot_digest,
        trust_snapshot_sequence: prepared.trust_snapshot_sequence,
        containment_head_id: prepared.containment_head_id,
        containment_authority_id: prepared.containment_authority_id,
        containment_state_digest: prepared.containment_state_digest,
        containment_generation: prepared.containment_generation,
        compromise_tracker_digest: prepared.compromise_tracker_digest,
        ceremony_id: ceremony.id(),
        ceremony_digest: ceremony.ceremony_digest(),
        clock_envelope_id: prepared.clock_envelope_id,
        operational_basis_id: prepared.operational_basis_id,
        activates_at_unix_ms: prepared.activates_at_unix_ms,
    })
}

fn canonical_profiles(
    mut profiles: Vec<WitnessAuthorityProfileV1>,
) -> Result<Vec<WitnessAuthorityProfileV1>, WitnessRegistryLineageError> {
    if profiles.len() > MAX_WITNESS_AUTHORITY_PROFILES {
        return Err(WitnessRegistryLineageError::TooManyProfiles {
            actual: profiles.len(),
            maximum: MAX_WITNESS_AUTHORITY_PROFILES,
        });
    }
    profiles.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str()).cmp(&(&right.algorithm, right.key_id.as_str()))
    });
    let mut seen = BTreeSet::new();
    for profile in &profiles {
        if !profile.algorithm.is_canonical()
            || invalid_identifier(&profile.key_id)
            || invalid_identifier(&profile.organization)
            || invalid_identifier(&profile.failure_domain)
        {
            return Err(WitnessRegistryLineageError::InvalidProfile(profile.key_id.clone()));
        }
        if !seen.insert((profile.algorithm.clone(), profile.key_id.clone())) {
            return Err(WitnessRegistryLineageError::DuplicateProfileSigner(
                profile.key_id.clone(),
            ));
        }
    }
    Ok(profiles)
}

fn requalify_proposed_witness_key(
    profile: &WitnessAuthorityProfileV1,
    trust_snapshot: &TrustSnapshot,
    containment_authority: &ClockGovernedContainmentStateV1,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<WitnessRegistryLineageError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| record.algorithm == profile.algorithm && record.key_id == profile.key_id)
    else {
        violations.push(WitnessRegistryLineageError::WitnessKeyUnknown(
            profile.key_id.clone(),
        ));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(WitnessRegistryLineageError::WitnessKeyNotActive(
            profile.key_id.clone(),
        ));
    }
    if !record.usages.contains(&KeyUsage::TransparencyWitness) {
        violations.push(WitnessRegistryLineageError::WitnessKeyUsageNotAllowed(
            profile.key_id.clone(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(WitnessRegistryLineageError::WitnessKeyNotValidAcrossEnvelope {
            key_id: profile.key_id.clone(),
            reason,
        });
    }
    for compromise in containment_authority
        .compromise_tracker()
        .records()
        .iter()
        .filter(|compromise| {
            compromise.signer.algorithm == profile.algorithm
                && compromise.signer.key_id == profile.key_id
                && compromise
                    .affected_usages
                    .contains(&KeyUsage::TransparencyWitness)
        })
    {
        match clock.require_effective_time_after_envelope_seconds(compromise.effective_at_unix_s) {
            Ok(()) => {}
            Err(ClockGovernanceTimeError::EventMayAlreadyBeEffective) => violations.push(
                WitnessRegistryLineageError::WitnessKeyCompromisedAcrossEnvelope(
                    profile.key_id.clone(),
                ),
            ),
            Err(reason) => violations.push(WitnessRegistryLineageError::CompromiseTimeInvalid {
                key_id: profile.key_id.clone(),
                reason,
            }),
        }
    }
}

fn valid_transition_policy(policy: &WitnessRegistryTransitionPolicyV1) -> bool {
    policy.maximum_activation_delay_s > 0
        && policy.minimum_profiles > 0
        && policy.minimum_profiles <= MAX_WITNESS_AUTHORITY_PROFILES
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

fn digest_profiles(
    profiles: &[WitnessAuthorityProfileV1],
) -> Result<Sha256Digest, WitnessRegistryLineageError> {
    hash_serializable(PROFILE_DIGEST_DOMAIN, profiles)
}

fn digest_transition_policy(
    policy: &WitnessRegistryTransitionPolicyV1,
) -> Result<Sha256Digest, WitnessRegistryLineageError> {
    if !valid_transition_policy(policy) {
        return Err(WitnessRegistryLineageError::InvalidTransitionPolicy);
    }
    hash_serializable(
        TRANSITION_POLICY_DOMAIN,
        &TransitionPolicyCommitment {
            maximum_activation_delay_s: policy.maximum_activation_delay_s,
            minimum_profiles: policy.minimum_profiles,
        },
    )
}

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, WitnessRegistryLineageError> {
    if !valid_threshold_policy(policy) {
        return Err(WitnessRegistryLineageError::InvalidThresholdPolicy);
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

fn seconds_to_millis(value: u64) -> Result<u64, WitnessRegistryLineageError> {
    value
        .checked_mul(1_000)
        .ok_or(WitnessRegistryLineageError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, WitnessRegistryLineageError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| WitnessRegistryLineageError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_policy_defaults_are_bounded() {
        let policy = WitnessRegistryTransitionPolicyV1::default();
        assert!(valid_transition_policy(&policy));
        assert!(policy.maximum_activation_delay_s > 0);
        assert!(policy.minimum_profiles > 0);
    }
}
