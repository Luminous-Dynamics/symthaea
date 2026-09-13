// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable current-policy and active-waiver lineage for clock-governed fabrication governance.
//!
//! This crate deliberately separates portable migration history from live currentness authority.
//! A lineage begins only after an interval-qualified threshold quorum authorizes an exact genesis
//! policy under an exact operational-clock basis. Every later transition requires both the opaque
//! migration authorization and its opaque runtime activation permit, plus proof that governance
//! clock authority has not forked or regressed since the previous lineage state.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_fabrication_governance_bridge::{
    ClockGovernedPolicyMigrationIdV1, ClockGovernedPolicyMigrationV1,
};
use symthaea_fabrication_governance_runtime::{
    ClockGovernedPolicyActivationPermitIdV1, ClockGovernedPolicyActivationPermitV1,
};
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::containment_state::{
    FabricationContainmentState, digest_containment_state,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::policy_migration::{
    PolicyBinding, PolicyInvariantDisposition, PolicyMigrationError, digest_policy_binding,
};
use symthaea_fabrication_kernel::signer_compromise_tracker::digest_signer_compromise_tracker;
use symthaea_fabrication_kernel::threshold::{
    MAX_THRESHOLD_APPROVALS, MAX_THRESHOLD_KEY_ID_BYTES, ThresholdCeremonyPolicy,
};
use symthaea_fabrication_kernel::trust::{KeyUsage, TrustSnapshot, digest_trust_snapshot};
use symthaea_fabrication_trust_bridge::{
    ClockGovernedThresholdCeremonyIdV1, ClockGovernedThresholdCeremonyV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE: &str =
    "clock-governed-policy-lineage-genesis-v1";
pub const PREPARED_POLICY_LINEAGE_GENESIS_SCHEMA: &str =
    "symthaea.fabrication.prepared-policy-lineage-genesis.v1";
pub const CLOCK_GOVERNED_POLICY_LINEAGE_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-policy-lineage.v1";
pub const MAX_POLICY_LINEAGE_CLOCK_BRIDGE_HOPS: usize = 4096;

const THRESHOLD_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-threshold-policy.v1\0";
const PREPARED_GENESIS_DOMAIN: &[u8] =
    b"symthaea.fabrication.prepared-policy-lineage-genesis.v1\0";
const POLICY_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-policy-lineage.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedPolicyLineageGenesisIdV1(Sha256Digest);

impl PreparedPolicyLineageGenesisIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedPolicyLineageIdV1(Sha256Digest);

impl ClockGovernedPolicyLineageIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// One currently unresolved policy waiver.
///
/// `predecessor_digest` is retained so a later successor policy can resolve the waiver only by
/// restoring the exact waived invariant. A different digest requires a separate explicit
/// migration theorem rather than being silently treated as restoration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivePolicyWaiverV1 {
    invariant: String,
    predecessor_digest: Sha256Digest,
    incident_digest: Sha256Digest,
    expires_at_unix_s: u64,
    originating_migration_id: ClockGovernedPolicyMigrationIdV1,
}

impl ActivePolicyWaiverV1 {
    pub fn invariant(&self) -> &str {
        &self.invariant
    }

    pub fn predecessor_digest(&self) -> Sha256Digest {
        self.predecessor_digest
    }

    pub fn incident_digest(&self) -> Sha256Digest {
        self.incident_digest
    }

    pub fn expires_at_unix_s(&self) -> u64 {
        self.expires_at_unix_s
    }

    pub fn originating_migration_id(&self) -> ClockGovernedPolicyMigrationIdV1 {
        self.originating_migration_id
    }
}

/// Opaque proposal for the first durable current-policy root.
///
/// Genesis is intentionally threshold-authorized rather than accepting a bare `PolicyBinding` as
/// currentness authority. The threshold payload commits the exact policy, threshold policy, trust
/// snapshot, containment state, compromise tracker, and operational-clock envelope.
#[derive(Debug, Clone)]
#[must_use]
pub struct PreparedPolicyLineageGenesisV1 {
    id: PreparedPolicyLineageGenesisIdV1,
    initial_policy: PolicyBinding,
    initial_policy_binding_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
}

impl PreparedPolicyLineageGenesisV1 {
    pub fn id(&self) -> PreparedPolicyLineageGenesisIdV1 {
        self.id
    }

    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }

    pub fn initial_policy(&self) -> &PolicyBinding {
        &self.initial_policy
    }

    pub fn initial_policy_binding_digest(&self) -> Sha256Digest {
        self.initial_policy_binding_digest
    }

    pub fn threshold_policy_digest(&self) -> Sha256Digest {
        self.threshold_policy_digest
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

    pub fn containment_generation(&self) -> u64 {
        self.containment_generation
    }

    pub fn clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.clock_envelope_id
    }

    pub fn operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.operational_basis_id
    }
}

/// Opaque live current-policy state for one policy domain.
///
/// The state is hash-linked by `previous_lineage_id`. It stores only *currently unresolved*
/// waivers, never every historical waiver. Historical transitions remain committed recursively in
/// the prior state ID without being misinterpreted as current exceptions.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedPolicyLineageV1 {
    id: ClockGovernedPolicyLineageIdV1,
    sequence: u64,
    domain: String,
    current_policy: PolicyBinding,
    current_policy_binding_digest: Sha256Digest,
    active_waivers: Vec<ActivePolicyWaiverV1>,
    previous_lineage_id: Option<ClockGovernedPolicyLineageIdV1>,
    genesis_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    last_migration_id: Option<ClockGovernedPolicyMigrationIdV1>,
    last_activation_permit_id: Option<ClockGovernedPolicyActivationPermitIdV1>,
    latest_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    latest_operational_basis_id: OperationalClockBasisIdV1,
}

impl ClockGovernedPolicyLineageV1 {
    pub fn id(&self) -> ClockGovernedPolicyLineageIdV1 {
        self.id
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn domain(&self) -> &str {
        &self.domain
    }

    pub fn current_policy(&self) -> &PolicyBinding {
        &self.current_policy
    }

    pub fn current_policy_binding_digest(&self) -> Sha256Digest {
        self.current_policy_binding_digest
    }

    pub fn active_waivers(&self) -> &[ActivePolicyWaiverV1] {
        &self.active_waivers
    }

    pub fn previous_lineage_id(&self) -> Option<ClockGovernedPolicyLineageIdV1> {
        self.previous_lineage_id
    }

    pub fn genesis_ceremony_id(&self) -> ClockGovernedThresholdCeremonyIdV1 {
        self.genesis_ceremony_id
    }

    pub fn last_migration_id(&self) -> Option<ClockGovernedPolicyMigrationIdV1> {
        self.last_migration_id
    }

    pub fn last_activation_permit_id(&self) -> Option<ClockGovernedPolicyActivationPermitIdV1> {
        self.last_activation_permit_id
    }

    pub fn latest_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.latest_clock_envelope_id
    }

    pub fn latest_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.latest_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedPolicyLineageError {
    InvalidThresholdPolicy,
    ThresholdPolicyUsageMismatch,
    Policy(PolicyMigrationError),
    TrustSnapshotInvalid(String),
    ContainmentStateInvalid(String),
    CompromiseTrackerInvalid(String),
    Clock(ClockGovernanceTimeError),
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    ThresholdPolicyDigestMismatch,
    TrustSnapshotDigestMismatch,
    CompromiseTrackerDigestMismatch,
    ClockEnvelopeMismatch,
    LineageStateInvalid(String),
    MigrationPermitMismatch,
    ActivationBasisMismatch,
    ActivationEnvelopeMismatch,
    ActivationWindowMismatch,
    TooManyClockBridgeHops {
        actual: usize,
        maximum: usize,
    },
    BrokenCurrentClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    DomainMismatch,
    PredecessorPolicyMismatch,
    SequenceOverflow,
    ActiveWaiverConflictsCurrentPolicy(String),
    ActiveWaiverMayBeExpired {
        invariant: String,
        expires_at_unix_ms: u64,
        current_upper_unix_ms: u64,
    },
    WaiverRestorationDigestMismatch(String),
    DuplicateActiveWaiver(String),
    TimeScaleOverflow,
    Encoding(String),
}

/// Prepare a threshold-authorized genesis root for one policy domain.
///
/// Unlike the legacy migration tracker, a bare policy value cannot initialize currentness. The
/// operational clock basis is also committed directly, so every later currentness transition can
/// prove that its execution clock descends from this exact governance-time root.
pub fn prepare_clock_governed_policy_lineage_genesis_v1(
    initial_policy: PolicyBinding,
    threshold_policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    containment_state: &FabricationContainmentState,
    operational_basis: &OperationalClockBasisV1,
) -> Result<PreparedPolicyLineageGenesisV1, ClockGovernedPolicyLineageError> {
    initial_policy
        .validate()
        .map_err(ClockGovernedPolicyLineageError::Policy)?;
    validate_threshold_policy(threshold_policy)?;
    if threshold_policy.key_usage != KeyUsage::PolicyMigration {
        return Err(ClockGovernedPolicyLineageError::ThresholdPolicyUsageMismatch);
    }
    trust_snapshot.validate().map_err(|error| {
        ClockGovernedPolicyLineageError::TrustSnapshotInvalid(format!("{error:?}"))
    })?;
    containment_state.validate().map_err(|error| {
        ClockGovernedPolicyLineageError::ContainmentStateInvalid(format!("{error:?}"))
    })?;
    containment_state
        .signer_compromise_tracker
        .validate()
        .map_err(|error| {
            ClockGovernedPolicyLineageError::CompromiseTrackerInvalid(format!("{error:?}"))
        })?;

    let clock = derive_clock_governance_evaluation_envelope_v1(operational_basis)
        .map_err(ClockGovernedPolicyLineageError::Clock)?;
    let initial_policy_binding_digest =
        digest_policy_binding(&initial_policy).map_err(ClockGovernedPolicyLineageError::Policy)?;
    let threshold_policy_digest = digest_threshold_policy(threshold_policy)?;
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        ClockGovernedPolicyLineageError::TrustSnapshotInvalid(format!("{error:?}"))
    })?;
    let containment_state_digest = digest_containment_state(containment_state).map_err(|error| {
        ClockGovernedPolicyLineageError::ContainmentStateInvalid(format!("{error:?}"))
    })?;
    let compromise_tracker_digest =
        digest_signer_compromise_tracker(&containment_state.signer_compromise_tracker).map_err(
            |error| {
                ClockGovernedPolicyLineageError::CompromiseTrackerInvalid(format!("{error:?}"))
            },
        )?;

    let id = PreparedPolicyLineageGenesisIdV1(digest_prepared_genesis(
        initial_policy_binding_digest,
        threshold_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        containment_state.generation,
        clock.id(),
        operational_basis.id(),
    )?);

    Ok(PreparedPolicyLineageGenesisV1 {
        id,
        initial_policy,
        initial_policy_binding_digest,
        threshold_policy_digest,
        trust_snapshot_digest,
        containment_state_digest,
        compromise_tracker_digest,
        containment_generation: containment_state.generation,
        clock_envelope_id: clock.id(),
        operational_basis_id: operational_basis.id(),
    })
}

/// Mint the first opaque current-policy state after an interval-qualified threshold quorum approves
/// the complete genesis context.
pub fn authorize_clock_governed_policy_lineage_genesis_v1(
    prepared: PreparedPolicyLineageGenesisV1,
    ceremony: &ClockGovernedThresholdCeremonyV1,
) -> Result<ClockGovernedPolicyLineageV1, ClockGovernedPolicyLineageError> {
    if ceremony.purpose() != CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE {
        return Err(ClockGovernedPolicyLineageError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != prepared.signing_payload_digest() {
        return Err(ClockGovernedPolicyLineageError::CeremonyPayloadMismatch);
    }
    if ceremony.policy_digest() != prepared.threshold_policy_digest {
        return Err(ClockGovernedPolicyLineageError::ThresholdPolicyDigestMismatch);
    }
    if ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest {
        return Err(ClockGovernedPolicyLineageError::TrustSnapshotDigestMismatch);
    }
    if ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest {
        return Err(ClockGovernedPolicyLineageError::CompromiseTrackerDigestMismatch);
    }
    if ceremony.clock_envelope_id() != prepared.clock_envelope_id {
        return Err(ClockGovernedPolicyLineageError::ClockEnvelopeMismatch);
    }

    let sequence = 1;
    let domain = prepared.initial_policy.domain.clone();
    let active_waivers = Vec::new();
    let id = ClockGovernedPolicyLineageIdV1(digest_lineage_state(
        sequence,
        &domain,
        prepared.initial_policy_binding_digest,
        &active_waivers,
        None,
        ceremony.id(),
        None,
        None,
        prepared.clock_envelope_id,
        prepared.operational_basis_id,
    )?);

    Ok(ClockGovernedPolicyLineageV1 {
        id,
        sequence,
        domain,
        current_policy: prepared.initial_policy,
        current_policy_binding_digest: prepared.initial_policy_binding_digest,
        active_waivers,
        previous_lineage_id: None,
        genesis_ceremony_id: ceremony.id(),
        last_migration_id: None,
        last_activation_permit_id: None,
        latest_clock_envelope_id: prepared.clock_envelope_id,
        latest_operational_basis_id: prepared.operational_basis_id,
    })
}

/// Advance one domain's durable current-policy state.
///
/// `clock_bridge` contains any intermediate operational bases strictly between the lineage state's
/// latest basis and `activation_basis`. The function proves that governance time has not forked or
/// regressed since the previous policy transition, independently of the activation permit's own
/// authorization-to-activation ancestry proof.
pub fn advance_clock_governed_policy_lineage_v1(
    current: &ClockGovernedPolicyLineageV1,
    migration: &ClockGovernedPolicyMigrationV1,
    activation_permit: &ClockGovernedPolicyActivationPermitV1,
    clock_bridge: &[OperationalClockBasisV1],
    activation_basis: &OperationalClockBasisV1,
) -> Result<ClockGovernedPolicyLineageV1, ClockGovernedPolicyLineageError> {
    validate_lineage_state(current)?;
    if activation_permit.migration_id() != migration.id() {
        return Err(ClockGovernedPolicyLineageError::MigrationPermitMismatch);
    }
    if activation_permit.activation_operational_basis_id() != activation_basis.id() {
        return Err(ClockGovernedPolicyLineageError::ActivationBasisMismatch);
    }
    if activation_permit.authorization_clock_envelope_id() != migration.clock_envelope_id() {
        return Err(ClockGovernedPolicyLineageError::ClockEnvelopeMismatch);
    }
    if activation_permit.activates_at_unix_s() != migration.plan().activates_at_unix_s
        || activation_permit.rollback_deadline_unix_s()
            != migration.plan().rollback_deadline_unix_s
    {
        return Err(ClockGovernedPolicyLineageError::ActivationWindowMismatch);
    }
    if clock_bridge.len() > MAX_POLICY_LINEAGE_CLOCK_BRIDGE_HOPS {
        return Err(ClockGovernedPolicyLineageError::TooManyClockBridgeHops {
            actual: clock_bridge.len(),
            maximum: MAX_POLICY_LINEAGE_CLOCK_BRIDGE_HOPS,
        });
    }
    verify_current_clock_lineage(current.latest_operational_basis_id, clock_bridge, activation_basis)?;

    let activation_envelope = derive_clock_governance_evaluation_envelope_v1(activation_basis)
        .map_err(ClockGovernedPolicyLineageError::Clock)?;
    if activation_envelope.id() != activation_permit.activation_clock_envelope_id() {
        return Err(ClockGovernedPolicyLineageError::ActivationEnvelopeMismatch);
    }

    let plan = migration.plan();
    if plan.predecessor.domain != current.domain || plan.successor.domain != current.domain {
        return Err(ClockGovernedPolicyLineageError::DomainMismatch);
    }
    if plan.predecessor != current.current_policy {
        return Err(ClockGovernedPolicyLineageError::PredecessorPolicyMismatch);
    }

    let successor_invariants = plan.successor.invariant_map();
    let mut next_waivers = BTreeMap::<String, ActivePolicyWaiverV1>::new();

    for waiver in &current.active_waivers {
        if current
            .current_policy
            .invariants
            .iter()
            .any(|invariant| invariant.name == waiver.invariant)
        {
            return Err(ClockGovernedPolicyLineageError::ActiveWaiverConflictsCurrentPolicy(
                waiver.invariant.clone(),
            ));
        }
        if let Some(restored_digest) = successor_invariants.get(waiver.invariant.as_str()) {
            if *restored_digest != waiver.predecessor_digest {
                return Err(
                    ClockGovernedPolicyLineageError::WaiverRestorationDigestMismatch(
                        waiver.invariant.clone(),
                    ),
                );
            }
            continue;
        }
        require_waiver_valid_across_upper(waiver, activation_envelope.upper_unix_ms())?;
        if next_waivers
            .insert(waiver.invariant.clone(), waiver.clone())
            .is_some()
        {
            return Err(ClockGovernedPolicyLineageError::DuplicateActiveWaiver(
                waiver.invariant.clone(),
            ));
        }
    }

    for item in &plan.migrations {
        if let PolicyInvariantDisposition::Waived {
            incident_digest,
            expires_at_unix_s,
        } = &item.disposition
        {
            let waiver = ActivePolicyWaiverV1 {
                invariant: item.name.clone(),
                predecessor_digest: item.predecessor_digest,
                incident_digest: *incident_digest,
                expires_at_unix_s: *expires_at_unix_s,
                originating_migration_id: migration.id(),
            };
            require_waiver_valid_across_upper(&waiver, activation_envelope.upper_unix_ms())?;
            if next_waivers.insert(item.name.clone(), waiver).is_some() {
                return Err(ClockGovernedPolicyLineageError::DuplicateActiveWaiver(
                    item.name.clone(),
                ));
            }
        }
    }

    let active_waivers = next_waivers.into_values().collect::<Vec<_>>();
    let current_policy = plan.successor.clone();
    let current_policy_binding_digest =
        digest_policy_binding(&current_policy).map_err(ClockGovernedPolicyLineageError::Policy)?;
    let sequence = current
        .sequence
        .checked_add(1)
        .ok_or(ClockGovernedPolicyLineageError::SequenceOverflow)?;
    let previous_lineage_id = Some(current.id);
    let last_migration_id = Some(migration.id());
    let last_activation_permit_id = Some(activation_permit.id());
    let id = ClockGovernedPolicyLineageIdV1(digest_lineage_state(
        sequence,
        &current.domain,
        current_policy_binding_digest,
        &active_waivers,
        previous_lineage_id,
        current.genesis_ceremony_id,
        last_migration_id,
        last_activation_permit_id,
        activation_envelope.id(),
        activation_basis.id(),
    )?);

    Ok(ClockGovernedPolicyLineageV1 {
        id,
        sequence,
        domain: current.domain.clone(),
        current_policy,
        current_policy_binding_digest,
        active_waivers,
        previous_lineage_id,
        genesis_ceremony_id: current.genesis_ceremony_id,
        last_migration_id,
        last_activation_permit_id,
        latest_clock_envelope_id: activation_envelope.id(),
        latest_operational_basis_id: activation_basis.id(),
    })
}

fn validate_lineage_state(
    state: &ClockGovernedPolicyLineageV1,
) -> Result<(), ClockGovernedPolicyLineageError> {
    if state.sequence == 0 {
        return Err(ClockGovernedPolicyLineageError::LineageStateInvalid(
            "sequence is zero".into(),
        ));
    }
    state
        .current_policy
        .validate()
        .map_err(ClockGovernedPolicyLineageError::Policy)?;
    if state.current_policy.domain != state.domain {
        return Err(ClockGovernedPolicyLineageError::LineageStateInvalid(
            "domain does not match current policy".into(),
        ));
    }
    let expected_policy_digest = digest_policy_binding(&state.current_policy)
        .map_err(ClockGovernedPolicyLineageError::Policy)?;
    if expected_policy_digest != state.current_policy_binding_digest {
        return Err(ClockGovernedPolicyLineageError::LineageStateInvalid(
            "current policy binding digest mismatch".into(),
        ));
    }
    if state.sequence == 1 {
        if state.previous_lineage_id.is_some()
            || state.last_migration_id.is_some()
            || state.last_activation_permit_id.is_some()
        {
            return Err(ClockGovernedPolicyLineageError::LineageStateInvalid(
                "genesis contains transition authority".into(),
            ));
        }
    } else if state.previous_lineage_id.is_none()
        || state.last_migration_id.is_none()
        || state.last_activation_permit_id.is_none()
    {
        return Err(ClockGovernedPolicyLineageError::LineageStateInvalid(
            "non-genesis state is missing transition authority".into(),
        ));
    }

    let current_invariants = state.current_policy.invariant_map();
    let mut previous_name = None::<&str>;
    for waiver in &state.active_waivers {
        if waiver.invariant.trim().is_empty()
            || waiver.invariant != waiver.invariant.trim()
            || waiver.invariant.chars().any(char::is_control)
            || waiver.predecessor_digest.0 == [0; 32]
            || waiver.incident_digest.0 == [0; 32]
            || waiver.expires_at_unix_s == 0
        {
            return Err(ClockGovernedPolicyLineageError::LineageStateInvalid(format!(
                "invalid active waiver {}",
                waiver.invariant
            )));
        }
        if current_invariants.contains_key(waiver.invariant.as_str()) {
            return Err(ClockGovernedPolicyLineageError::ActiveWaiverConflictsCurrentPolicy(
                waiver.invariant.clone(),
            ));
        }
        if previous_name.is_some_and(|name| name >= waiver.invariant.as_str()) {
            return Err(ClockGovernedPolicyLineageError::DuplicateActiveWaiver(
                waiver.invariant.clone(),
            ));
        }
        previous_name = Some(waiver.invariant.as_str());
    }

    let expected_id = ClockGovernedPolicyLineageIdV1(digest_lineage_state(
        state.sequence,
        &state.domain,
        state.current_policy_binding_digest,
        &state.active_waivers,
        state.previous_lineage_id,
        state.genesis_ceremony_id,
        state.last_migration_id,
        state.last_activation_permit_id,
        state.latest_clock_envelope_id,
        state.latest_operational_basis_id,
    )?);
    if expected_id != state.id {
        return Err(ClockGovernedPolicyLineageError::LineageStateInvalid(
            "lineage identity mismatch".into(),
        ));
    }
    Ok(())
}

fn verify_current_clock_lineage(
    current_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    activation_basis: &OperationalClockBasisV1,
) -> Result<(), ClockGovernedPolicyLineageError> {
    if activation_basis.id() == current_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(ClockGovernedPolicyLineageError::BrokenCurrentClockLineage {
            hop: 1,
            expected_predecessor: current_basis_id.to_hex(),
            actual_predecessor: bridge[0]
                .predecessor_operational_basis_id()
                .map(|value| value.to_hex()),
        });
    }

    let mut expected = current_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(ClockGovernedPolicyLineageError::BrokenCurrentClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = activation_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(ClockGovernedPolicyLineageError::BrokenCurrentClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn require_waiver_valid_across_upper(
    waiver: &ActivePolicyWaiverV1,
    current_upper_unix_ms: u64,
) -> Result<(), ClockGovernedPolicyLineageError> {
    let expires_at_unix_ms = waiver
        .expires_at_unix_s
        .checked_mul(1_000)
        .ok_or(ClockGovernedPolicyLineageError::TimeScaleOverflow)?;
    if expires_at_unix_ms <= current_upper_unix_ms {
        return Err(ClockGovernedPolicyLineageError::ActiveWaiverMayBeExpired {
            invariant: waiver.invariant.clone(),
            expires_at_unix_ms,
            current_upper_unix_ms,
        });
    }
    Ok(())
}

fn validate_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<(), ClockGovernedPolicyLineageError> {
    let valid = policy.minimum_distinct_signers > 0
        && policy.maximum_approvals > 0
        && policy.minimum_distinct_signers <= policy.maximum_approvals
        && policy.maximum_approvals <= MAX_THRESHOLD_APPROVALS
        && policy
            .required_algorithms
            .iter()
            .all(SignatureAlgorithm::is_canonical)
        && policy.allowed_key_ids.as_ref().is_none_or(|ids| {
            ids.iter().all(|id| {
                !id.trim().is_empty()
                    && id == id.trim()
                    && id.len() <= MAX_THRESHOLD_KEY_ID_BYTES
                    && !id.chars().any(char::is_control)
            })
        });
    if valid {
        Ok(())
    } else {
        Err(ClockGovernedPolicyLineageError::InvalidThresholdPolicy)
    }
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

fn digest_threshold_policy(
    policy: &ThresholdCeremonyPolicy,
) -> Result<Sha256Digest, ClockGovernedPolicyLineageError> {
    let bytes = serde_json::to_vec(&ThresholdPolicyCommitment {
        minimum_distinct_signers: policy.minimum_distinct_signers,
        maximum_approvals: policy.maximum_approvals,
        require_algorithm_diversity: policy.require_algorithm_diversity,
        required_algorithms: &policy.required_algorithms,
        allowed_key_ids: &policy.allowed_key_ids,
        key_usage: policy.key_usage,
    })
    .map_err(|error| ClockGovernedPolicyLineageError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(THRESHOLD_POLICY_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct PreparedGenesisCommitment {
    schema: &'static str,
    purpose: &'static str,
    initial_policy_binding_digest: String,
    threshold_policy_digest: String,
    trust_snapshot_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    containment_generation: u64,
    clock_envelope_id: String,
    operational_basis_id: String,
}

#[allow(clippy::too_many_arguments)]
fn digest_prepared_genesis(
    initial_policy_binding_digest: Sha256Digest,
    threshold_policy_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    operational_basis_id: OperationalClockBasisIdV1,
) -> Result<Sha256Digest, ClockGovernedPolicyLineageError> {
    let bytes = serde_json::to_vec(&PreparedGenesisCommitment {
        schema: PREPARED_POLICY_LINEAGE_GENESIS_SCHEMA,
        purpose: CLOCK_GOVERNED_POLICY_LINEAGE_GENESIS_PURPOSE,
        initial_policy_binding_digest: initial_policy_binding_digest.to_hex(),
        threshold_policy_digest: threshold_policy_digest.to_hex(),
        trust_snapshot_digest: trust_snapshot_digest.to_hex(),
        containment_state_digest: containment_state_digest.to_hex(),
        compromise_tracker_digest: compromise_tracker_digest.to_hex(),
        containment_generation,
        clock_envelope_id: clock_envelope_id.to_hex(),
        operational_basis_id: operational_basis_id.to_hex(),
    })
    .map_err(|error| ClockGovernedPolicyLineageError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(PREPARED_GENESIS_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[derive(Serialize)]
struct ActiveWaiverCommitment {
    invariant: String,
    predecessor_digest: String,
    incident_digest: String,
    expires_at_unix_s: u64,
    originating_migration_id: String,
}

#[derive(Serialize)]
struct PolicyLineageCommitment {
    schema: &'static str,
    sequence: u64,
    domain: String,
    current_policy_binding_digest: String,
    active_waivers: Vec<ActiveWaiverCommitment>,
    previous_lineage_id: Option<String>,
    genesis_ceremony_id: String,
    last_migration_id: Option<String>,
    last_activation_permit_id: Option<String>,
    latest_clock_envelope_id: String,
    latest_operational_basis_id: String,
}

#[allow(clippy::too_many_arguments)]
fn digest_lineage_state(
    sequence: u64,
    domain: &str,
    current_policy_binding_digest: Sha256Digest,
    active_waivers: &[ActivePolicyWaiverV1],
    previous_lineage_id: Option<ClockGovernedPolicyLineageIdV1>,
    genesis_ceremony_id: ClockGovernedThresholdCeremonyIdV1,
    last_migration_id: Option<ClockGovernedPolicyMigrationIdV1>,
    last_activation_permit_id: Option<ClockGovernedPolicyActivationPermitIdV1>,
    latest_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    latest_operational_basis_id: OperationalClockBasisIdV1,
) -> Result<Sha256Digest, ClockGovernedPolicyLineageError> {
    let bytes = serde_json::to_vec(&PolicyLineageCommitment {
        schema: CLOCK_GOVERNED_POLICY_LINEAGE_SCHEMA,
        sequence,
        domain: domain.to_string(),
        current_policy_binding_digest: current_policy_binding_digest.to_hex(),
        active_waivers: active_waivers
            .iter()
            .map(|waiver| ActiveWaiverCommitment {
                invariant: waiver.invariant.clone(),
                predecessor_digest: waiver.predecessor_digest.to_hex(),
                incident_digest: waiver.incident_digest.to_hex(),
                expires_at_unix_s: waiver.expires_at_unix_s,
                originating_migration_id: waiver.originating_migration_id.to_hex(),
            })
            .collect(),
        previous_lineage_id: previous_lineage_id.map(|value| value.to_hex()),
        genesis_ceremony_id: genesis_ceremony_id.to_hex(),
        last_migration_id: last_migration_id.map(|value| value.to_hex()),
        last_activation_permit_id: last_activation_permit_id.map(|value| value.to_hex()),
        latest_clock_envelope_id: latest_clock_envelope_id.to_hex(),
        latest_operational_basis_id: latest_operational_basis_id.to_hex(),
    })
    .map_err(|error| ClockGovernedPolicyLineageError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(POLICY_LINEAGE_DOMAIN);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
