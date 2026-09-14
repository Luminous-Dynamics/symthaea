// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh-authority activation for future transparency-witness registry transitions.
//!
//! Authorization is not activation. This bridge consumes an opaque authorized transition, proves
//! that trusted time definitely reached its scheduled activation, proves that the current
//! containment authority descends from the exact containment authority used at authorization, and
//! requalifies every proposed witness key under the fresh witnessed trust + containment view.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_containment_head::{
    QuorumObservedContainmentHeadIdV1, QuorumObservedContainmentHeadV1,
};
use symthaea_fabrication_containment_state_authority::{
    ClockGovernedContainmentStateIdV1, ClockGovernedContainmentStateV1,
};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_fabrication_trust_snapshot_head::{
    QuorumObservedTrustSnapshotHeadIdV1, QuorumObservedTrustSnapshotHeadV1,
};
use symthaea_fabrication_witness_authority::{
    MAX_WITNESS_AUTHORITY_PROFILES, WitnessAuthorityProfileV1,
};
use symthaea_fabrication_witness_registry_lineage::{
    AuthorizedWitnessRegistryTransitionIdV1, AuthorizedWitnessRegistryTransitionV1,
    GovernedWitnessAuthorityRegistryIdV1, GovernedWitnessAuthorityRegistryV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceEvaluationEnvelopeV1,
    ClockGovernanceTimeError, OperationalClockBasisIdV1, OperationalClockBasisV1,
    derive_clock_governance_evaluation_envelope_v1,
};

pub const ACTIVATED_WITNESS_AUTHORITY_REGISTRY_SCHEMA: &str =
    "symthaea.fabrication.activated-witness-authority-registry.v1";
pub const MAX_WITNESS_REGISTRY_ACTIVATION_CLOCK_HOPS: usize = 4096;
pub const MAX_WITNESS_REGISTRY_ACTIVATION_CONTAINMENT_HOPS: usize = 4096;

const PROFILE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-authority-registry-digest.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-authority-registry-activation-clock-lineage.v1\0";
const CONTAINMENT_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.witness-authority-registry-activation-containment-lineage.v1\0";
const ACTIVATED_REGISTRY_DOMAIN: &[u8] =
    b"symthaea.fabrication.activated-witness-authority-registry.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivatedWitnessAuthorityRegistryIdV1(Sha256Digest);

impl ActivatedWitnessAuthorityRegistryIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque active successor registry. It cannot be constructed from portable profile data alone.
#[derive(Debug, Clone)]
#[must_use]
pub struct ActivatedWitnessAuthorityRegistryV1 {
    id: ActivatedWitnessAuthorityRegistryIdV1,
    transition_id: AuthorizedWitnessRegistryTransitionIdV1,
    previous_registry_id: GovernedWitnessAuthorityRegistryIdV1,
    previous_registry_digest: Sha256Digest,
    previous_sequence: u64,
    sequence: u64,
    profiles: Vec<WitnessAuthorityProfileV1>,
    registry_digest: Sha256Digest,
    authorization_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    authorization_operational_basis_id: OperationalClockBasisIdV1,
    activation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
    authorization_containment_authority_id: ClockGovernedContainmentStateIdV1,
    activation_containment_authority_id: ClockGovernedContainmentStateIdV1,
    activation_containment_state_digest: Sha256Digest,
    activation_containment_generation: u64,
    activation_compromise_tracker_digest: Sha256Digest,
    containment_lineage_digest: Sha256Digest,
    containment_hop_count: usize,
    trust_head_id: QuorumObservedTrustSnapshotHeadIdV1,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    containment_head_id: QuorumObservedContainmentHeadIdV1,
    activates_at_unix_ms: u64,
}

impl ActivatedWitnessAuthorityRegistryV1 {
    pub fn id(&self) -> ActivatedWitnessAuthorityRegistryIdV1 {
        self.id
    }
    pub fn transition_id(&self) -> AuthorizedWitnessRegistryTransitionIdV1 {
        self.transition_id
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
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn profiles(&self) -> &[WitnessAuthorityProfileV1] {
        &self.profiles
    }
    pub fn registry_digest(&self) -> Sha256Digest {
        self.registry_digest
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
    pub fn activation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.activation_clock_envelope_id
    }
    pub fn activation_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.activation_operational_basis_id
    }
    pub fn clock_lineage_digest(&self) -> Sha256Digest {
        self.clock_lineage_digest
    }
    pub fn clock_hop_count(&self) -> usize {
        self.clock_hop_count
    }
    pub fn activation_containment_authority_id(&self) -> ClockGovernedContainmentStateIdV1 {
        self.activation_containment_authority_id
    }
    pub fn activation_containment_state_digest(&self) -> Sha256Digest {
        self.activation_containment_state_digest
    }
    pub fn activation_containment_generation(&self) -> u64 {
        self.activation_containment_generation
    }
    pub fn activation_compromise_tracker_digest(&self) -> Sha256Digest {
        self.activation_compromise_tracker_digest
    }
    pub fn containment_lineage_digest(&self) -> Sha256Digest {
        self.containment_lineage_digest
    }
    pub fn containment_hop_count(&self) -> usize {
        self.containment_hop_count
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
    pub fn activates_at_unix_ms(&self) -> u64 {
        self.activates_at_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessRegistryActivationError {
    PreviousRegistryMismatch,
    ProposedSequenceMismatch,
    ProposedRegistryDigestMismatch,
    AuthorizationBasisMismatch,
    AuthorizationEnvelopeMismatch,
    AuthorizationContainmentMismatch,
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
    CurrentContainmentHeadMismatch,
    CurrentTrustContainmentMismatch,
    CurrentHeadsObservedUnderDifferentRegistry,
    TrustSnapshotInvalid(String),
    TrustSnapshotHeadMismatch,
    TrustSnapshotSequenceRollback,
    TrustSnapshotSameSequenceSubstitution,
    CurrentBasisMismatch,
    CurrentEnvelopeMismatch,
    Clock(ClockGovernanceTimeError),
    ActivationNotYetCertain,
    TrustSnapshotNotValidAcrossCurrentEnvelope(ClockGovernanceTimeError),
    TooManyProfiles { actual: usize, maximum: usize },
    InvalidProfile(String),
    DuplicateProfileSigner(String),
    WitnessKeyUnknown(String),
    WitnessKeyNotActive(String),
    WitnessKeyUsageNotAllowed(String),
    WitnessKeyNotValidAcrossEnvelope {
        key_id: String,
        reason: ClockGovernanceTimeError,
    },
    WitnessKeyCompromisedAcrossEnvelope(String),
    CompromiseTimeInvalid { key_id: String, reason: ClockGovernanceTimeError },
    Encoding(String),
}

#[derive(Serialize)]
struct ActivatedRegistryCommitment {
    schema: &'static str,
    transition_id: String,
    previous_registry_id: String,
    previous_registry_digest: String,
    previous_sequence: u64,
    sequence: u64,
    registry_digest: String,
    authorization_clock_envelope_id: String,
    authorization_operational_basis_id: String,
    activation_clock_envelope_id: String,
    activation_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
    authorization_containment_authority_id: String,
    activation_containment_authority_id: String,
    activation_containment_state_digest: String,
    activation_containment_generation: u64,
    activation_compromise_tracker_digest: String,
    containment_lineage_digest: String,
    containment_hop_count: usize,
    trust_head_id: String,
    trust_snapshot_digest: String,
    trust_snapshot_sequence: u64,
    containment_head_id: String,
    activates_at_unix_ms: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn activate_authorized_witness_registry_transition_v1(
    transition: &AuthorizedWitnessRegistryTransitionV1,
    previous_registry: &GovernedWitnessAuthorityRegistryV1,
    authorization_basis: &OperationalClockBasisV1,
    authorization_containment_authority: &ClockGovernedContainmentStateV1,
    containment_authority_bridge: &[ClockGovernedContainmentStateV1],
    current_containment_authority: &ClockGovernedContainmentStateV1,
    trust_head: &QuorumObservedTrustSnapshotHeadV1,
    containment_head: &QuorumObservedContainmentHeadV1,
    trust_snapshot: &TrustSnapshot,
    clock_bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<ActivatedWitnessAuthorityRegistryV1, Vec<WitnessRegistryActivationError>> {
    let mut violations = Vec::new();

    if transition.previous_registry_id() != previous_registry.id()
        || transition.previous_registry_digest() != previous_registry.registry_digest()
        || transition.previous_sequence() != previous_registry.sequence()
    {
        violations.push(WitnessRegistryActivationError::PreviousRegistryMismatch);
    }
    let expected_sequence = match previous_registry.sequence().checked_add(1) {
        Some(value) => value,
        None => {
            violations.push(WitnessRegistryActivationError::ProposedSequenceMismatch);
            0
        }
    };
    if transition.proposed_sequence() != expected_sequence {
        violations.push(WitnessRegistryActivationError::ProposedSequenceMismatch);
    }
    let profiles = match canonical_profiles(transition.proposed_profiles().to_vec()) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Vec::new()
        }
    };
    let registry_digest = match digest_profiles(&profiles) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };
    if registry_digest != transition.proposed_registry_digest() {
        violations.push(WitnessRegistryActivationError::ProposedRegistryDigestMismatch);
    }

    if authorization_basis.id() != transition.operational_basis_id() {
        violations.push(WitnessRegistryActivationError::AuthorizationBasisMismatch);
    }
    let authorization_clock = match derive_clock_governance_evaluation_envelope_v1(authorization_basis)
    {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryActivationError::Clock(error));
            return Err(violations);
        }
    };
    if authorization_clock.id() != transition.clock_envelope_id() {
        violations.push(WitnessRegistryActivationError::AuthorizationEnvelopeMismatch);
    }

    if authorization_containment_authority.id() != transition.containment_authority_id()
        || authorization_containment_authority.state_digest() != transition.containment_state_digest()
        || authorization_containment_authority.generation() != transition.containment_generation()
        || authorization_containment_authority.compromise_tracker_digest()
            != transition.compromise_tracker_digest()
    {
        violations.push(WitnessRegistryActivationError::AuthorizationContainmentMismatch);
    }

    if clock_bridge.len() > MAX_WITNESS_REGISTRY_ACTIVATION_CLOCK_HOPS {
        violations.push(WitnessRegistryActivationError::TooManyClockHops {
            actual: clock_bridge.len(),
            maximum: MAX_WITNESS_REGISTRY_ACTIVATION_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        authorization_basis.id(),
        clock_bridge,
        current_basis,
    ) {
        violations.push(error);
    }
    if containment_authority_bridge.len() > MAX_WITNESS_REGISTRY_ACTIVATION_CONTAINMENT_HOPS {
        violations.push(WitnessRegistryActivationError::TooManyContainmentHops {
            actual: containment_authority_bridge.len(),
            maximum: MAX_WITNESS_REGISTRY_ACTIVATION_CONTAINMENT_HOPS,
        });
    } else if let Err(error) = verify_containment_lineage(
        authorization_containment_authority,
        containment_authority_bridge,
        current_containment_authority,
    ) {
        violations.push(error);
    }

    if containment_head.authority_id() != current_containment_authority.id()
        || containment_head.state_digest() != current_containment_authority.state_digest()
        || containment_head.generation() != current_containment_authority.generation()
        || containment_head.compromise_tracker_digest()
            != current_containment_authority.compromise_tracker_digest()
    {
        violations.push(WitnessRegistryActivationError::CurrentContainmentHeadMismatch);
    }
    if containment_head.trust_head_id() != trust_head.id()
        || containment_head.trust_snapshot_digest() != trust_head.snapshot_digest()
        || containment_head.trust_snapshot_sequence() != trust_head.snapshot_sequence()
    {
        violations.push(WitnessRegistryActivationError::CurrentTrustContainmentMismatch);
    }
    if trust_head.witness_registry_digest() != previous_registry.registry_digest()
        || trust_head.witness_registry_sequence() != previous_registry.sequence()
        || containment_head.witness_registry_digest() != previous_registry.registry_digest()
        || containment_head.witness_registry_sequence() != previous_registry.sequence()
    {
        violations.push(WitnessRegistryActivationError::CurrentHeadsObservedUnderDifferentRegistry);
    }

    let trust_snapshot_digest = match digest_trust_snapshot(trust_snapshot) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryActivationError::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if let Err(error) = trust_snapshot.validate() {
        violations.push(WitnessRegistryActivationError::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if trust_snapshot_digest != trust_head.snapshot_digest()
        || trust_snapshot.sequence != trust_head.snapshot_sequence()
    {
        violations.push(WitnessRegistryActivationError::TrustSnapshotHeadMismatch);
    }
    if trust_head.snapshot_sequence() < transition.trust_snapshot_sequence() {
        violations.push(WitnessRegistryActivationError::TrustSnapshotSequenceRollback);
    }
    if trust_head.snapshot_sequence() == transition.trust_snapshot_sequence()
        && trust_head.snapshot_digest() != transition.trust_snapshot_digest()
    {
        violations.push(WitnessRegistryActivationError::TrustSnapshotSameSequenceSubstitution);
    }

    if current_basis.id() != containment_head.observation_operational_basis_id() {
        violations.push(WitnessRegistryActivationError::CurrentBasisMismatch);
    }
    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(WitnessRegistryActivationError::Clock(error));
            return Err(violations);
        }
    };
    if current_clock.id() != containment_head.observation_clock_envelope_id() {
        violations.push(WitnessRegistryActivationError::CurrentEnvelopeMismatch);
    }
    if transition.activates_at_unix_ms() > current_clock.lower_unix_ms() {
        violations.push(WitnessRegistryActivationError::ActivationNotYetCertain);
    }
    if let Err(reason) = current_clock.require_valid_across_seconds_window(
        trust_snapshot.issued_at_unix_s,
        trust_snapshot.expires_at_unix_s,
    ) {
        violations.push(
            WitnessRegistryActivationError::TrustSnapshotNotValidAcrossCurrentEnvelope(reason),
        );
    }

    for profile in &profiles {
        requalify_witness_key(
            profile,
            trust_snapshot,
            current_containment_authority,
            &current_clock,
            &mut violations,
        );
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let (clock_lineage_digest, clock_hop_count) = digest_clock_lineage(
        authorization_basis,
        clock_bridge,
        current_basis,
    ).map_err(|error| vec![error])?;
    let (containment_lineage_digest, containment_hop_count) = digest_containment_lineage(
        authorization_containment_authority,
        containment_authority_bridge,
        current_containment_authority,
    ).map_err(|error| vec![error])?;

    let commitment = ActivatedRegistryCommitment {
        schema: ACTIVATED_WITNESS_AUTHORITY_REGISTRY_SCHEMA,
        transition_id: transition.id().to_hex(),
        previous_registry_id: previous_registry.id().to_hex(),
        previous_registry_digest: previous_registry.registry_digest().to_hex(),
        previous_sequence: previous_registry.sequence(),
        sequence: transition.proposed_sequence(),
        registry_digest: registry_digest.to_hex(),
        authorization_clock_envelope_id: authorization_clock.id().to_hex(),
        authorization_operational_basis_id: authorization_basis.id().to_hex(),
        activation_clock_envelope_id: current_clock.id().to_hex(),
        activation_operational_basis_id: current_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count,
        authorization_containment_authority_id: authorization_containment_authority.id().to_hex(),
        activation_containment_authority_id: current_containment_authority.id().to_hex(),
        activation_containment_state_digest: current_containment_authority.state_digest().to_hex(),
        activation_containment_generation: current_containment_authority.generation(),
        activation_compromise_tracker_digest: current_containment_authority
            .compromise_tracker_digest()
            .to_hex(),
        containment_lineage_digest: containment_lineage_digest.to_hex(),
        containment_hop_count,
        trust_head_id: trust_head.id().to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        trust_snapshot_sequence: trust_snapshot.sequence,
        containment_head_id: containment_head.id().to_hex(),
        activates_at_unix_ms: transition.activates_at_unix_ms(),
    };
    let id = ActivatedWitnessAuthorityRegistryIdV1(hash_serializable(
        ACTIVATED_REGISTRY_DOMAIN,
        &commitment,
    ).map_err(|error| vec![error])?);

    Ok(ActivatedWitnessAuthorityRegistryV1 {
        id,
        transition_id: transition.id(),
        previous_registry_id: previous_registry.id(),
        previous_registry_digest: previous_registry.registry_digest(),
        previous_sequence: previous_registry.sequence(),
        sequence: transition.proposed_sequence(),
        profiles,
        registry_digest,
        authorization_clock_envelope_id: authorization_clock.id(),
        authorization_operational_basis_id: authorization_basis.id(),
        activation_clock_envelope_id: current_clock.id(),
        activation_operational_basis_id: current_basis.id(),
        clock_lineage_digest,
        clock_hop_count,
        authorization_containment_authority_id: authorization_containment_authority.id(),
        activation_containment_authority_id: current_containment_authority.id(),
        activation_containment_state_digest: current_containment_authority.state_digest(),
        activation_containment_generation: current_containment_authority.generation(),
        activation_compromise_tracker_digest: current_containment_authority
            .compromise_tracker_digest(),
        containment_lineage_digest,
        containment_hop_count,
        trust_head_id: trust_head.id(),
        trust_snapshot_digest,
        trust_snapshot_sequence: trust_snapshot.sequence,
        containment_head_id: containment_head.id(),
        activates_at_unix_ms: transition.activates_at_unix_ms(),
    })
}

fn canonical_profiles(
    mut profiles: Vec<WitnessAuthorityProfileV1>,
) -> Result<Vec<WitnessAuthorityProfileV1>, WitnessRegistryActivationError> {
    if profiles.len() > MAX_WITNESS_AUTHORITY_PROFILES {
        return Err(WitnessRegistryActivationError::TooManyProfiles {
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
            return Err(WitnessRegistryActivationError::InvalidProfile(
                profile.key_id.clone(),
            ));
        }
        if !seen.insert((profile.algorithm.clone(), profile.key_id.clone())) {
            return Err(WitnessRegistryActivationError::DuplicateProfileSigner(
                profile.key_id.clone(),
            ));
        }
    }
    Ok(profiles)
}

fn requalify_witness_key(
    profile: &WitnessAuthorityProfileV1,
    trust_snapshot: &TrustSnapshot,
    containment_authority: &ClockGovernedContainmentStateV1,
    clock: &ClockGovernanceEvaluationEnvelopeV1,
    violations: &mut Vec<WitnessRegistryActivationError>,
) {
    let Some(record) = trust_snapshot
        .keys
        .iter()
        .find(|record| record.algorithm == profile.algorithm && record.key_id == profile.key_id)
    else {
        violations.push(WitnessRegistryActivationError::WitnessKeyUnknown(
            profile.key_id.clone(),
        ));
        return;
    };
    if record.status != KeyLifecycleStatus::Active {
        violations.push(WitnessRegistryActivationError::WitnessKeyNotActive(
            profile.key_id.clone(),
        ));
    }
    if !record.usages.contains(&KeyUsage::TransparencyWitness) {
        violations.push(WitnessRegistryActivationError::WitnessKeyUsageNotAllowed(
            profile.key_id.clone(),
        ));
    }
    if let Err(reason) = clock.require_valid_across_optional_seconds_window(
        record.not_before_unix_s,
        record.not_after_unix_s,
    ) {
        violations.push(WitnessRegistryActivationError::WitnessKeyNotValidAcrossEnvelope {
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
                WitnessRegistryActivationError::WitnessKeyCompromisedAcrossEnvelope(
                    profile.key_id.clone(),
                ),
            ),
            Err(reason) => violations.push(WitnessRegistryActivationError::CompromiseTimeInvalid {
                key_id: profile.key_id.clone(),
                reason,
            }),
        }
    }
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), WitnessRegistryActivationError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(WitnessRegistryActivationError::BrokenClockLineage {
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
            return Err(WitnessRegistryActivationError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(WitnessRegistryActivationError::BrokenClockLineage {
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
) -> Result<(), WitnessRegistryActivationError> {
    if current.id() == start.id() {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(WitnessRegistryActivationError::BrokenContainmentLineage {
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
) -> Result<(), WitnessRegistryActivationError> {
    if current.previous_authority_id() != Some(previous.id()) {
        return Err(WitnessRegistryActivationError::BrokenContainmentLineage {
            hop,
            expected_previous_authority: previous.id().to_hex(),
            actual_previous_authority: current
                .previous_authority_id()
                .map(|value| value.to_hex()),
        });
    }
    if current.previous_state_digest() != Some(previous.state_digest()) {
        return Err(WitnessRegistryActivationError::BrokenContainmentStateLineage { hop });
    }
    let expected_generation = previous
        .generation()
        .checked_add(1)
        .ok_or(WitnessRegistryActivationError::ContainmentGenerationOverflow)?;
    if current.generation() != expected_generation {
        return Err(WitnessRegistryActivationError::ContainmentGenerationNotAdjacent {
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
) -> Result<(Sha256Digest, usize), WitnessRegistryActivationError> {
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
) -> Result<(Sha256Digest, usize), WitnessRegistryActivationError> {
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

fn digest_profiles(
    profiles: &[WitnessAuthorityProfileV1],
) -> Result<Sha256Digest, WitnessRegistryActivationError> {
    hash_serializable(PROFILE_DIGEST_DOMAIN, profiles)
}

fn invalid_identifier(value: &str) -> bool {
    value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, WitnessRegistryActivationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| WitnessRegistryActivationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_resource_bounds_are_explicit() {
        assert_eq!(MAX_WITNESS_REGISTRY_ACTIVATION_CLOCK_HOPS, 4096);
        assert_eq!(MAX_WITNESS_REGISTRY_ACTIVATION_CONTAINMENT_HOPS, 4096);
    }
}
