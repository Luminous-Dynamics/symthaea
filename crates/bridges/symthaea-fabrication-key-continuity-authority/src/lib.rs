// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interval-safe opaque key continuity across fabrication trust-snapshot transitions.
//!
//! The kernel's serializable `VerifiedKeyContinuity` remains portable evidence. This bridge derives
//! live continuity directly from the two raw trust snapshots, a trusted operational-clock interval,
//! the exact continuity policy, and persistent compromise state. No caller-selected scalar
//! transition timestamp is accepted.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::key_continuity::KeyContinuityPolicy;
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_KEY_CONTINUITY_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-key-continuity.v1";

const KEY_CONTINUITY_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-key-continuity-policy.v1\0";
const BRIDGE_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-key-continuity-bridge-set.v1\0";
const KEY_CONTINUITY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-key-continuity.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedKeyContinuityIdV1(Sha256Digest);

impl ClockGovernedKeyContinuityIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }
    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one exact successor trust snapshot has sufficient bridge/successor authority
/// across every possible transition time in one trusted clock envelope and for the entire required
/// overlap horizon after the latest possible transition.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedKeyContinuityV1 {
    id: ClockGovernedKeyContinuityIdV1,
    previous_snapshot_digest: Sha256Digest,
    successor_snapshot_digest: Sha256Digest,
    previous_snapshot_sequence: u64,
    successor_snapshot_sequence: u64,
    policy_digest: Sha256Digest,
    bridge_set_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
    overlap_required_until_unix_ms: u64,
    bridge_keys_by_usage: BTreeMap<KeyUsage, Vec<(SignatureAlgorithm, String)>>,
    successor_algorithms_by_usage: BTreeMap<KeyUsage, Vec<SignatureAlgorithm>>,
}

impl ClockGovernedKeyContinuityV1 {
    pub fn id(&self) -> ClockGovernedKeyContinuityIdV1 {
        self.id
    }
    pub fn previous_snapshot_digest(&self) -> Sha256Digest {
        self.previous_snapshot_digest
    }
    pub fn successor_snapshot_digest(&self) -> Sha256Digest {
        self.successor_snapshot_digest
    }
    pub fn previous_snapshot_sequence(&self) -> u64 {
        self.previous_snapshot_sequence
    }
    pub fn successor_snapshot_sequence(&self) -> u64 {
        self.successor_snapshot_sequence
    }
    pub fn policy_digest(&self) -> Sha256Digest {
        self.policy_digest
    }
    pub fn bridge_set_digest(&self) -> Sha256Digest {
        self.bridge_set_digest
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
    pub fn overlap_required_until_unix_ms(&self) -> u64 {
        self.overlap_required_until_unix_ms
    }
    pub fn bridge_keys_by_usage(
        &self,
    ) -> &BTreeMap<KeyUsage, Vec<(SignatureAlgorithm, String)>> {
        &self.bridge_keys_by_usage
    }
    pub fn successor_algorithms_by_usage(&self) -> &BTreeMap<KeyUsage, Vec<SignatureAlgorithm>> {
        &self.successor_algorithms_by_usage
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedKeyContinuityError {
    InvalidPolicy,
    PreviousSnapshotInvalid(String),
    SuccessorSnapshotInvalid(String),
    SequenceNotAdvanced,
    Clock(ClockGovernanceTimeError),
    TimeScaleOverflow,
    PreviousSnapshotNotAvailableAcrossTransition,
    SuccessorSnapshotNotAvailableAcrossTransition,
    PreviousSnapshotInsufficientOverlap,
    SuccessorSnapshotInsufficientOverlap,
    MissingBridgeKeys(KeyUsage),
    InsufficientSuccessorKeys(KeyUsage),
    MissingAlgorithmDiversity(KeyUsage),
    ContainmentStateInvalid(String),
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct KeyContinuityPolicyCommitment<'a> {
    required_usages: &'a BTreeSet<KeyUsage>,
    minimum_bridge_keys_per_usage: usize,
    minimum_successor_keys_per_usage: usize,
    minimum_overlap_s: u64,
    require_successor_algorithm_diversity: bool,
}

#[derive(Debug, Clone, Serialize)]
struct BridgeSetCommitment<'a> {
    bridge_keys_by_usage: &'a BTreeMap<KeyUsage, Vec<(SignatureAlgorithm, String)>>,
    successor_algorithms_by_usage: &'a BTreeMap<KeyUsage, Vec<SignatureAlgorithm>>,
}

#[derive(Debug, Clone, Serialize)]
struct KeyContinuityCommitment {
    schema: &'static str,
    previous_snapshot_digest: String,
    successor_snapshot_digest: String,
    previous_snapshot_sequence: u64,
    successor_snapshot_sequence: u64,
    policy_digest: String,
    bridge_set_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    clock_envelope_id: String,
    operational_basis_id: String,
    overlap_required_until_unix_ms: u64,
}

pub fn derive_clock_governed_key_continuity_v1(
    previous: &TrustSnapshot,
    successor: &TrustSnapshot,
    policy: &KeyContinuityPolicy,
    containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
) -> Result<ClockGovernedKeyContinuityV1, Vec<ClockGovernedKeyContinuityError>> {
    let mut violations = Vec::new();

    if !valid_policy(policy) {
        violations.push(ClockGovernedKeyContinuityError::InvalidPolicy);
    }
    if let Err(error) = previous.validate() {
        violations.push(ClockGovernedKeyContinuityError::PreviousSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if let Err(error) = successor.validate() {
        violations.push(ClockGovernedKeyContinuityError::SuccessorSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if successor.sequence <= previous.sequence {
        violations.push(ClockGovernedKeyContinuityError::SequenceNotAdvanced);
    }
    if let Err(error) = containment_state.validate() {
        violations.push(ClockGovernedKeyContinuityError::ContainmentStateInvalid(format!(
            "{error:?}"
        )));
    }

    let clock = match derive_clock_governance_evaluation_envelope_v1(operational_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(ClockGovernedKeyContinuityError::Clock(error));
            return Err(violations);
        }
    };
    let overlap_ms = match policy.minimum_overlap_s.checked_mul(1_000) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedKeyContinuityError::TimeScaleOverflow);
            0
        }
    };
    let overlap_required_until_unix_ms = match clock.upper_unix_ms().checked_add(overlap_ms) {
        Some(value) => value,
        None => {
            violations.push(ClockGovernedKeyContinuityError::TimeScaleOverflow);
            clock.upper_unix_ms()
        }
    };

    validate_snapshot_transition_window(
        previous,
        clock.lower_unix_ms(),
        overlap_required_until_unix_ms,
        true,
        &mut violations,
    );
    validate_snapshot_transition_window(
        successor,
        clock.lower_unix_ms(),
        overlap_required_until_unix_ms,
        false,
        &mut violations,
    );

    let mut bridge_keys_by_usage = BTreeMap::new();
    let mut successor_algorithms_by_usage = BTreeMap::new();
    for usage in &policy.required_usages {
        let previous_keys = eligible_identities(
            previous,
            *usage,
            clock.lower_unix_ms(),
            overlap_required_until_unix_ms,
            containment_state,
        );
        let successor_keys = eligible_identities(
            successor,
            *usage,
            clock.lower_unix_ms(),
            overlap_required_until_unix_ms,
            containment_state,
        );
        let bridge = previous_keys
            .intersection(&successor_keys)
            .cloned()
            .collect::<Vec<_>>();
        if bridge.len() < policy.minimum_bridge_keys_per_usage {
            violations.push(ClockGovernedKeyContinuityError::MissingBridgeKeys(*usage));
        }
        if successor_keys.len() < policy.minimum_successor_keys_per_usage {
            violations.push(ClockGovernedKeyContinuityError::InsufficientSuccessorKeys(*usage));
        }
        let algorithms = successor_keys
            .iter()
            .map(|(algorithm, _)| algorithm.clone())
            .collect::<BTreeSet<_>>();
        if policy.require_successor_algorithm_diversity && algorithms.len() < 2 {
            violations.push(ClockGovernedKeyContinuityError::MissingAlgorithmDiversity(*usage));
        }
        bridge_keys_by_usage.insert(*usage, bridge);
        successor_algorithms_by_usage.insert(*usage, algorithms.into_iter().collect());
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let previous_snapshot_digest = digest_trust_snapshot(previous).map_err(|error| {
        vec![ClockGovernedKeyContinuityError::PreviousSnapshotInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let successor_snapshot_digest = digest_trust_snapshot(successor).map_err(|error| {
        vec![ClockGovernedKeyContinuityError::SuccessorSnapshotInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let policy_digest = digest_policy(policy).map_err(|error| vec![error])?;
    let bridge_set_digest = hash_serializable(
        BRIDGE_SET_DOMAIN,
        &BridgeSetCommitment {
            bridge_keys_by_usage: &bridge_keys_by_usage,
            successor_algorithms_by_usage: &successor_algorithms_by_usage,
        },
    )
    .map_err(|error| vec![error])?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        vec![ClockGovernedKeyContinuityError::ContainmentStateInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| {
                vec![ClockGovernedKeyContinuityError::ContainmentStateInvalid(format!(
                    "{error:?}"
                ))]
            },
        )?;

    let commitment = KeyContinuityCommitment {
        schema: CLOCK_GOVERNED_KEY_CONTINUITY_SCHEMA,
        previous_snapshot_digest: previous_snapshot_digest.to_hex(),
        successor_snapshot_digest: successor_snapshot_digest.to_hex(),
        previous_snapshot_sequence: previous.sequence,
        successor_snapshot_sequence: successor.sequence,
        policy_digest: policy_digest.to_hex(),
        bridge_set_digest: bridge_set_digest.to_hex(),
        containment_state_digest: containment_state_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        clock_envelope_id: clock.id().to_hex(),
        operational_basis_id: operational_basis.id().to_hex(),
        overlap_required_until_unix_ms,
    };
    let id = ClockGovernedKeyContinuityIdV1(
        hash_serializable(KEY_CONTINUITY_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(ClockGovernedKeyContinuityV1 {
        id,
        previous_snapshot_digest,
        successor_snapshot_digest,
        previous_snapshot_sequence: previous.sequence,
        successor_snapshot_sequence: successor.sequence,
        policy_digest,
        bridge_set_digest,
        containment_state_digest,
        compromise_tracker_digest,
        clock_envelope_id: clock.id(),
        operational_basis_id: operational_basis.id(),
        overlap_required_until_unix_ms,
        bridge_keys_by_usage,
        successor_algorithms_by_usage,
    })
}

fn validate_snapshot_transition_window(
    snapshot: &TrustSnapshot,
    transition_lower_unix_ms: u64,
    overlap_required_until_unix_ms: u64,
    previous: bool,
    violations: &mut Vec<ClockGovernedKeyContinuityError>,
) {
    let issued_ms = snapshot.issued_at_unix_s.checked_mul(1_000);
    let expires_ms = snapshot.expires_at_unix_s.checked_mul(1_000);
    if issued_ms.is_none() || expires_ms.is_none() {
        violations.push(ClockGovernedKeyContinuityError::TimeScaleOverflow);
        return;
    }
    if issued_ms.is_some_and(|value| value > transition_lower_unix_ms) {
        violations.push(if previous {
            ClockGovernedKeyContinuityError::PreviousSnapshotNotAvailableAcrossTransition
        } else {
            ClockGovernedKeyContinuityError::SuccessorSnapshotNotAvailableAcrossTransition
        });
    }
    if expires_ms.is_some_and(|value| value < overlap_required_until_unix_ms) {
        violations.push(if previous {
            ClockGovernedKeyContinuityError::PreviousSnapshotInsufficientOverlap
        } else {
            ClockGovernedKeyContinuityError::SuccessorSnapshotInsufficientOverlap
        });
    }
}

fn eligible_identities(
    snapshot: &TrustSnapshot,
    usage: KeyUsage,
    transition_lower_unix_ms: u64,
    overlap_required_until_unix_ms: u64,
    containment_state: &FabricationContainmentState,
) -> BTreeSet<(SignatureAlgorithm, String)> {
    snapshot
        .keys
        .iter()
        .filter(|record| {
            key_covers_interval(
                record,
                usage,
                transition_lower_unix_ms,
                overlap_required_until_unix_ms,
            ) && !compromised_before_overlap_end(
                record,
                usage,
                overlap_required_until_unix_ms,
                containment_state,
            )
        })
        .map(|record| (record.algorithm.clone(), record.key_id.clone()))
        .collect()
}

fn key_covers_interval(
    record: &KeyTrustRecord,
    usage: KeyUsage,
    transition_lower_unix_ms: u64,
    overlap_required_until_unix_ms: u64,
) -> bool {
    if record.status != KeyLifecycleStatus::Active || !record.usages.contains(&usage) {
        return false;
    }
    let Some(not_before_ms) = record.not_before_unix_s.checked_mul(1_000) else {
        return false;
    };
    if not_before_ms > transition_lower_unix_ms {
        return false;
    }
    match record.not_after_unix_s {
        Some(value) => value
            .checked_mul(1_000)
            .is_some_and(|not_after_ms| not_after_ms >= overlap_required_until_unix_ms),
        None => true,
    }
}

fn compromised_before_overlap_end(
    record: &KeyTrustRecord,
    usage: KeyUsage,
    overlap_required_until_unix_ms: u64,
    containment_state: &FabricationContainmentState,
) -> bool {
    containment_state
        .signer_compromise_tracker
        .records()
        .iter()
        .filter(|compromise| {
            compromise.signer.algorithm == record.algorithm
                && compromise.signer.key_id == record.key_id
                && compromise.affected_usages.contains(&usage)
        })
        .any(|compromise| {
            compromise
                .effective_at_unix_s
                .checked_mul(1_000)
                .is_none_or(|effective_ms| effective_ms < overlap_required_until_unix_ms)
        })
}

fn valid_policy(policy: &KeyContinuityPolicy) -> bool {
    !policy.required_usages.is_empty()
        && policy.minimum_bridge_keys_per_usage > 0
        && policy.minimum_successor_keys_per_usage > 0
        && policy.minimum_overlap_s > 0
}

fn digest_policy(
    policy: &KeyContinuityPolicy,
) -> Result<Sha256Digest, ClockGovernedKeyContinuityError> {
    if !valid_policy(policy) {
        return Err(ClockGovernedKeyContinuityError::InvalidPolicy);
    }
    hash_serializable(
        KEY_CONTINUITY_POLICY_DOMAIN,
        &KeyContinuityPolicyCommitment {
            required_usages: &policy.required_usages,
            minimum_bridge_keys_per_usage: policy.minimum_bridge_keys_per_usage,
            minimum_successor_keys_per_usage: policy.minimum_successor_keys_per_usage,
            minimum_overlap_s: policy.minimum_overlap_s,
            require_successor_algorithm_diversity: policy.require_successor_algorithm_diversity,
        },
    )
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ClockGovernedKeyContinuityError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ClockGovernedKeyContinuityError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
