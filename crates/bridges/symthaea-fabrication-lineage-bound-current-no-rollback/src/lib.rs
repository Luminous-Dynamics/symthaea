// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Same-checkpoint full operational lineage and no-rollback authority for lineage-bound upgrades.
//!
//! The durable kernel state/publication schemas remain unchanged. This bridge defines the exact
//! hardened meaning of their generic evidence digests for the lineage-bound upgrade path, then proves
//! the complete published operational chain is rollback-free in one exact current governance view.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_evidence_retention_head::{
    CurrentEvidenceRetentionHeadIdV1, CurrentEvidenceRetentionHeadV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_kernel::upgrade_operational_state::{
    FabricationUpgradeOperationalState, UpgradeOperationalEvidenceDigests,
    digest_upgrade_operational_state, verify_upgrade_operational_state_successor,
};
use symthaea_fabrication_key_continuity_authority::{
    ClockGovernedKeyContinuityIdV1, ClockGovernedKeyContinuityV1,
};
use symthaea_fabrication_lineage_bound_hardware::LineageBoundHardwareReauthorizationV1;
use symthaea_fabrication_lineage_bound_probation_telemetry::{
    LineageBoundTelemetryProbationClearanceIdV1, LineageBoundTelemetryProbationClearanceV1,
};
use symthaea_fabrication_lineage_bound_upgrade_handoff::{
    LineageBoundClockGovernedUpgradeHandoffIdV1, LineageBoundClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_lineage_bound_upgrade_probation::{
    LineageBoundUpgradeProbationClearanceIdV1, LineageBoundUpgradeProbationClearanceV1,
};
use symthaea_fabrication_lineage_bound_upgrade_runtime::{
    LineageBoundUpgradeActivationPermitIdV1, LineageBoundUpgradeActivationPermitV1,
};
use symthaea_fabrication_upgrade_operational_head::{
    UpgradeOperationalHeadPublicationV1, build_upgrade_operational_head_publication_v1,
    digest_upgrade_operational_head_publication_v1, upgrade_operational_head_log_kind,
};
use symthaea_fabrication_witness_registry_containment_bound::{
    ContainmentCurrentWitnessRegistryHeadIdV1, ContainmentCurrentWitnessRegistryHeadV1,
};
use symthaea_fabrication_witness_registry_head::{
    QuorumObservedWitnessRegistryHeadIdV1, QuorumObservedWitnessRegistryHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const LINEAGE_BOUND_OPERATIONAL_EVIDENCE_BINDING_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-operational-evidence-binding.v1";
pub const LINEAGE_BOUND_CURRENT_NO_ROLLBACK_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-current-no-rollback.v1";
pub const MAX_LINEAGE_BOUND_OPERATIONAL_STATES: usize = 65_536;
pub const MAX_LINEAGE_BOUND_HARDWARE_AUTHORITIES: usize = 65_536;
pub const MAX_LINEAGE_BOUND_NO_ROLLBACK_CLOCK_HOPS: usize = 4096;

const PROBATION_TRACKER_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-probation-tracker.v1\0";
const HARDWARE_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-hardware-authority-set.v1\0";
const OBSERVATION_CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-operational-observation-clock-lineage.v1\0";
const BINDING_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-operational-evidence-binding.v1\0";
const OPERATIONAL_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-current-no-rollback-operational-lineage.v1\0";
const CURRENT_NO_ROLLBACK_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-current-no-rollback.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundOperationalEvidenceBindingIdV1(Sha256Digest);

impl LineageBoundOperationalEvidenceBindingIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundOperationalEvidenceBindingV1 {
    id: LineageBoundOperationalEvidenceBindingIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    lineage_handoff_id: LineageBoundClockGovernedUpgradeHandoffIdV1,
    activation_permit_id: LineageBoundUpgradeActivationPermitIdV1,
    probation_clearance_id: LineageBoundUpgradeProbationClearanceIdV1,
    telemetry_clearance_id: LineageBoundTelemetryProbationClearanceIdV1,
    predecessor_root_digest: Sha256Digest,
    predecessor_current_head_digest: Sha256Digest,
    predecessor_governance_view_digest: Sha256Digest,
    predecessor_registry_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    probation_tracker_digest: Sha256Digest,
    hardware_authority_set_digest: Sha256Digest,
    hardware_authority_count: usize,
    machine_ids: Vec<String>,
    retention_policy_digest: Sha256Digest,
    retention_policy_sequence: u64,
    key_continuity_id: ClockGovernedKeyContinuityIdV1,
    key_snapshot_sequence: u64,
    durable_clock_continuity_digest: Sha256Digest,
    durable_clock_epoch: u64,
    observation_clock_lineage_digest: Sha256Digest,
    activation_operational_basis_id: OperationalClockBasisIdV1,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    governance_checkpoint_digest: Sha256Digest,
    transparency_log_digest: Sha256Digest,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
    evidence_ready_at_unix_ms: u64,
}

impl LineageBoundOperationalEvidenceBindingV1 {
    pub fn id(&self) -> LineageBoundOperationalEvidenceBindingIdV1 { self.id }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.governance_view_id }
    pub fn registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 { self.registry_head_id }
    pub fn retention_head_id(&self) -> CurrentEvidenceRetentionHeadIdV1 { self.retention_head_id }
    pub fn lineage_handoff_id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 { self.lineage_handoff_id }
    pub fn activation_permit_id(&self) -> LineageBoundUpgradeActivationPermitIdV1 { self.activation_permit_id }
    pub fn probation_clearance_id(&self) -> LineageBoundUpgradeProbationClearanceIdV1 { self.probation_clearance_id }
    pub fn telemetry_clearance_id(&self) -> LineageBoundTelemetryProbationClearanceIdV1 { self.telemetry_clearance_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn predecessor_current_head_digest(&self) -> Sha256Digest { self.predecessor_current_head_digest }
    pub fn predecessor_governance_view_digest(&self) -> Sha256Digest { self.predecessor_governance_view_digest }
    pub fn predecessor_registry_head_digest(&self) -> Sha256Digest { self.predecessor_registry_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn upgrade_cycle_sequence(&self) -> u64 { self.upgrade_cycle_sequence }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn probation_tracker_digest(&self) -> Sha256Digest { self.probation_tracker_digest }
    pub fn hardware_authority_set_digest(&self) -> Sha256Digest { self.hardware_authority_set_digest }
    pub fn hardware_authority_count(&self) -> usize { self.hardware_authority_count }
    pub fn machine_ids(&self) -> &[String] { &self.machine_ids }
    pub fn retention_policy_digest(&self) -> Sha256Digest { self.retention_policy_digest }
    pub fn retention_policy_sequence(&self) -> u64 { self.retention_policy_sequence }
    pub fn key_continuity_id(&self) -> ClockGovernedKeyContinuityIdV1 { self.key_continuity_id }
    pub fn key_snapshot_sequence(&self) -> u64 { self.key_snapshot_sequence }
    pub fn durable_clock_continuity_digest(&self) -> Sha256Digest { self.durable_clock_continuity_digest }
    pub fn durable_clock_epoch(&self) -> u64 { self.durable_clock_epoch }
    pub fn observation_clock_lineage_digest(&self) -> Sha256Digest { self.observation_clock_lineage_digest }
    pub fn activation_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.activation_operational_basis_id }
    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.observation_operational_basis_id }
    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.observation_clock_envelope_id }
    pub fn governance_checkpoint_digest(&self) -> Sha256Digest { self.governance_checkpoint_digest }
    pub fn transparency_log_digest(&self) -> Sha256Digest { self.transparency_log_digest }
    pub fn activates_at_unix_ms(&self) -> u64 { self.activates_at_unix_ms }
    pub fn finalization_deadline_unix_ms(&self) -> u64 { self.finalization_deadline_unix_ms }
    pub fn evidence_ready_at_unix_ms(&self) -> u64 { self.evidence_ready_at_unix_ms }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundCurrentNoRollbackIdV1(Sha256Digest);

impl LineageBoundCurrentNoRollbackIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundCurrentNoRollbackV1 {
    id: LineageBoundCurrentNoRollbackIdV1,
    evidence_binding_id: LineageBoundOperationalEvidenceBindingIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    lineage_handoff_id: LineageBoundClockGovernedUpgradeHandoffIdV1,
    activation_permit_id: LineageBoundUpgradeActivationPermitIdV1,
    probation_clearance_id: LineageBoundUpgradeProbationClearanceIdV1,
    telemetry_clearance_id: LineageBoundTelemetryProbationClearanceIdV1,
    predecessor_root_digest: Sha256Digest,
    predecessor_current_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    state_digest: Sha256Digest,
    state_generation: u64,
    publication_digest: Sha256Digest,
    publication_entry_sequence: u64,
    transparency_log_digest: Sha256Digest,
    operational_lineage_digest: Sha256Digest,
    operational_state_count: usize,
    hardware_authority_set_digest: Sha256Digest,
    hardware_authority_count: usize,
    key_continuity_id: ClockGovernedKeyContinuityIdV1,
    observation_clock_lineage_digest: Sha256Digest,
    observation_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
}

impl LineageBoundCurrentNoRollbackV1 {
    pub fn id(&self) -> LineageBoundCurrentNoRollbackIdV1 { self.id }
    pub fn evidence_binding_id(&self) -> LineageBoundOperationalEvidenceBindingIdV1 { self.evidence_binding_id }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.governance_view_id }
    pub fn registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 { self.registry_head_id }
    pub fn retention_head_id(&self) -> CurrentEvidenceRetentionHeadIdV1 { self.retention_head_id }
    pub fn lineage_handoff_id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 { self.lineage_handoff_id }
    pub fn activation_permit_id(&self) -> LineageBoundUpgradeActivationPermitIdV1 { self.activation_permit_id }
    pub fn probation_clearance_id(&self) -> LineageBoundUpgradeProbationClearanceIdV1 { self.probation_clearance_id }
    pub fn telemetry_clearance_id(&self) -> LineageBoundTelemetryProbationClearanceIdV1 { self.telemetry_clearance_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn predecessor_current_head_digest(&self) -> Sha256Digest { self.predecessor_current_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn upgrade_cycle_sequence(&self) -> u64 { self.upgrade_cycle_sequence }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn state_digest(&self) -> Sha256Digest { self.state_digest }
    pub fn state_generation(&self) -> u64 { self.state_generation }
    pub fn publication_digest(&self) -> Sha256Digest { self.publication_digest }
    pub fn publication_entry_sequence(&self) -> u64 { self.publication_entry_sequence }
    pub fn transparency_log_digest(&self) -> Sha256Digest { self.transparency_log_digest }
    pub fn operational_lineage_digest(&self) -> Sha256Digest { self.operational_lineage_digest }
    pub fn operational_state_count(&self) -> usize { self.operational_state_count }
    pub fn hardware_authority_set_digest(&self) -> Sha256Digest { self.hardware_authority_set_digest }
    pub fn hardware_authority_count(&self) -> usize { self.hardware_authority_count }
    pub fn key_continuity_id(&self) -> ClockGovernedKeyContinuityIdV1 { self.key_continuity_id }
    pub fn observation_clock_lineage_digest(&self) -> Sha256Digest { self.observation_clock_lineage_digest }
    pub fn observation_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.observation_operational_basis_id }
    pub fn observation_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.observation_clock_envelope_id }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundCurrentNoRollbackError {
    GovernanceRetentionMismatch,
    GovernanceRegistryMismatch,
    LineageMismatch,
    CurrentTrustMismatch,
    CurrentContainmentMismatch,
    KeyContinuityMismatch,
    ActivationBasisMismatch,
    ActivationEnvelopeMismatch,
    ObservationBasisMismatch,
    ObservationEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage { hop: usize, expected_predecessor: String, actual_predecessor: Option<String> },
    Clock(ClockGovernanceTimeError),
    FinalizationMayBeClosed,
    EmptyHardwareAuthorities,
    TooManyHardwareAuthorities { actual: usize, maximum: usize },
    DuplicateHardwareMachine(String),
    HardwareLineageMismatch(String),
    HardwareTrustMismatch(String),
    HardwareContainmentMismatch(String),
    HardwareClockMismatch(String),
    HardwareExpired(String),
    TimeScaleOverflow,
    UpgradeCycleSequenceOverflow,
    HardwareCountOverflow,
    ZeroUpgradeStateDigest,
    EmptyOperationalLineage,
    TooManyOperationalStates { actual: usize, maximum: usize },
    OperationalInputCountMismatch { states: usize, publications: usize, log_entries: usize },
    OperationalStateInvalid { index: usize, reason: String },
    OperationalGenesisInvalid,
    HandoffDigestMismatch { index: usize },
    StateBeforeActivation { index: usize },
    OperationalSuccessorInvalid { index: usize, reason: String },
    RollbackObserved { generation: u64, digest: Sha256Digest },
    PublicationMismatch { index: usize },
    PublicationDigestMismatch { index: usize },
    PublicationBeforeStateCommit { index: usize },
    PublicationMayBeFuture { index: usize },
    TransparencyLogInvalid(String),
    TransparencyLogMismatch,
    CandidateEvidenceMismatch(&'static str),
    CandidateCommittedBeforeEvidence,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct ProbationTrackerCommitment {
    probation_clearance_id: String,
    telemetry_clearance_id: String,
    observation_set_digest: String,
    telemetry_binding_set_digest: String,
}

#[derive(Debug, Clone, Serialize)]
struct HardwareAuthorityCommitment {
    authority_id: String,
    machine_id: String,
    reauthorization_sequence: u64,
    statement_digest: String,
    signed_evidence_digest: String,
    hardware_identity_digest: String,
    machine_profile_digest: String,
    firmware_digest: String,
    calibration_digest: String,
    capability_digest: String,
    expires_at_unix_s: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ClockLineageCommitment {
    activation_basis_id: String,
    bridge_basis_ids: Vec<String>,
    observation_basis_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceBindingCommitment {
    schema: &'static str,
    governance_view_id: String,
    registry_head_id: String,
    retention_head_id: String,
    lineage_handoff_id: String,
    activation_permit_id: String,
    probation_clearance_id: String,
    telemetry_clearance_id: String,
    predecessor_root_digest: String,
    predecessor_current_head_digest: String,
    predecessor_governance_view_digest: String,
    predecessor_registry_head_digest: String,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    handoff_plan_digest: String,
    probation_tracker_digest: String,
    hardware_authority_set_digest: String,
    hardware_authority_count: usize,
    machine_ids: Vec<String>,
    retention_policy_digest: String,
    retention_policy_sequence: u64,
    key_continuity_id: String,
    key_snapshot_sequence: u64,
    durable_clock_continuity_digest: String,
    durable_clock_epoch: u64,
    observation_clock_lineage_digest: String,
    activation_operational_basis_id: String,
    observation_operational_basis_id: String,
    observation_clock_envelope_id: String,
    governance_checkpoint_digest: String,
    transparency_log_digest: String,
    activates_at_unix_ms: u64,
    finalization_deadline_unix_ms: u64,
    evidence_ready_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OperationalLineageEntryCommitment {
    generation: u64,
    state_digest: String,
    publication_digest: String,
    publication_entry_sequence: u64,
    publication_recorded_at_unix_s: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CurrentNoRollbackCommitment {
    schema: &'static str,
    evidence_binding_id: String,
    governance_view_id: String,
    registry_head_id: String,
    retention_head_id: String,
    lineage_handoff_id: String,
    activation_permit_id: String,
    probation_clearance_id: String,
    telemetry_clearance_id: String,
    predecessor_root_digest: String,
    predecessor_current_head_digest: String,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    handoff_plan_digest: String,
    state_digest: String,
    state_generation: u64,
    publication_digest: String,
    publication_entry_sequence: u64,
    transparency_log_digest: String,
    operational_lineage_digest: String,
    operational_state_count: usize,
    hardware_authority_set_digest: String,
    hardware_authority_count: usize,
    key_continuity_id: String,
    observation_clock_lineage_digest: String,
    observation_operational_basis_id: String,
    observation_clock_envelope_id: String,
}

#[allow(clippy::too_many_arguments)]
pub fn derive_lineage_bound_operational_evidence_binding_v1(
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    registry_head: &QuorumObservedWitnessRegistryHeadV1,
    retention_head: &CurrentEvidenceRetentionHeadV1,
    handoff: &LineageBoundClockGovernedUpgradeHandoffV1,
    activation: &LineageBoundUpgradeActivationPermitV1,
    probation: &LineageBoundUpgradeProbationClearanceV1,
    telemetry: &LineageBoundTelemetryProbationClearanceV1,
    hardware_authorities: &[LineageBoundHardwareReauthorizationV1],
    key_continuity: &ClockGovernedKeyContinuityV1,
    activation_basis: &OperationalClockBasisV1,
    activation_to_observation_clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
) -> Result<LineageBoundOperationalEvidenceBindingV1, Vec<LineageBoundCurrentNoRollbackError>> {
    let mut violations = Vec::new();

    if retention_head.governance_view_id() != governance_view.id()
        || retention_head.registry_head_id() != registry_head.id()
        || retention_head.governance_checkpoint_digest() != governance_view.checkpoint_digest()
        || retention_head.transparency_log_digest() != governance_view.transparency_log_digest()
    {
        violations.push(LineageBoundCurrentNoRollbackError::GovernanceRetentionMismatch);
    }
    if governance_view.registry_head_id() != registry_head.id()
        || governance_view.registry_digest() != registry_head.registry_digest()
        || governance_view.registry_sequence() != registry_head.sequence()
    {
        violations.push(LineageBoundCurrentNoRollbackError::GovernanceRegistryMismatch);
    }

    if activation.lineage_handoff_id() != handoff.id()
        || activation.handoff_plan_digest() != handoff.plan_digest()
        || probation.lineage_handoff_id() != handoff.id()
        || probation.activation_permit_id() != activation.id()
        || probation.handoff_plan_digest() != handoff.plan_digest()
        || telemetry.clearance_id() != probation.id()
        || telemetry.activation_permit_digest() != activation.id().as_digest()
        || telemetry.handoff_plan_digest() != handoff.plan_digest()
        || probation.predecessor_root_digest() != handoff.predecessor_root_id().as_digest()
        || activation.predecessor_root_digest() != probation.predecessor_root_digest()
        || telemetry.predecessor_root_digest() != probation.predecessor_root_digest()
        || activation.predecessor_finalization_sequence() != handoff.predecessor_finalization_sequence()
        || probation.predecessor_finalization_sequence() != handoff.predecessor_finalization_sequence()
        || telemetry.predecessor_finalization_sequence() != handoff.predecessor_finalization_sequence()
    {
        violations.push(LineageBoundCurrentNoRollbackError::LineageMismatch);
    }

    if probation.trust_snapshot_digest() != registry_head.trust_snapshot_digest() {
        violations.push(LineageBoundCurrentNoRollbackError::CurrentTrustMismatch);
    }
    if probation.containment_state_digest() != governance_view.containment_state_digest()
        || probation.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
        || telemetry.containment_state_digest() != governance_view.containment_state_digest()
        || telemetry.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
    {
        violations.push(LineageBoundCurrentNoRollbackError::CurrentContainmentMismatch);
    }
    if key_continuity.successor_snapshot_digest() != registry_head.trust_snapshot_digest()
        || key_continuity.successor_snapshot_sequence() != registry_head.trust_snapshot_sequence()
    {
        violations.push(LineageBoundCurrentNoRollbackError::KeyContinuityMismatch);
    }

    if activation_basis.id() != activation.current_operational_basis_id() {
        violations.push(LineageBoundCurrentNoRollbackError::ActivationBasisMismatch);
    }
    let activation_clock = match derive_clock_governance_evaluation_envelope_v1(activation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundCurrentNoRollbackError::Clock(error));
            return Err(violations);
        }
    };
    if activation_clock.id() != activation.current_clock_envelope_id() {
        violations.push(LineageBoundCurrentNoRollbackError::ActivationEnvelopeMismatch);
    }
    if observation_basis.id() != governance_view.observation_operational_basis_id()
        || observation_basis.id() != retention_head.observation_operational_basis_id()
    {
        violations.push(LineageBoundCurrentNoRollbackError::ObservationBasisMismatch);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundCurrentNoRollbackError::Clock(error));
            return Err(violations);
        }
    };
    if observation_clock.id() != governance_view.observation_clock_envelope_id()
        || observation_clock.id() != retention_head.observation_clock_envelope_id()
    {
        violations.push(LineageBoundCurrentNoRollbackError::ObservationEnvelopeMismatch);
    }
    if activation_to_observation_clock_bridge.len() > MAX_LINEAGE_BOUND_NO_ROLLBACK_CLOCK_HOPS {
        violations.push(LineageBoundCurrentNoRollbackError::TooManyClockHops {
            actual: activation_to_observation_clock_bridge.len(),
            maximum: MAX_LINEAGE_BOUND_NO_ROLLBACK_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        activation_basis.id(),
        activation_to_observation_clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }
    if observation_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms {
        violations.push(LineageBoundCurrentNoRollbackError::FinalizationMayBeClosed);
    }

    let upgrade_cycle_sequence = match handoff.predecessor_finalization_sequence().checked_add(1) {
        Some(value) => value,
        None => {
            violations.push(LineageBoundCurrentNoRollbackError::UpgradeCycleSequenceOverflow);
            0
        }
    };

    if hardware_authorities.is_empty() {
        violations.push(LineageBoundCurrentNoRollbackError::EmptyHardwareAuthorities);
    }
    if hardware_authorities.len() > MAX_LINEAGE_BOUND_HARDWARE_AUTHORITIES {
        violations.push(LineageBoundCurrentNoRollbackError::TooManyHardwareAuthorities {
            actual: hardware_authorities.len(),
            maximum: MAX_LINEAGE_BOUND_HARDWARE_AUTHORITIES,
        });
        return Err(violations);
    }

    let mut seen_machines = BTreeSet::new();
    let mut hardware_commitments = Vec::with_capacity(hardware_authorities.len());
    let mut evidence_ready_at_unix_ms = probation.observation_ended_at_unix_ms();
    for authority in hardware_authorities {
        let machine_id = authority.statement().machine_id.clone();
        if !seen_machines.insert(machine_id.clone()) {
            violations.push(LineageBoundCurrentNoRollbackError::DuplicateHardwareMachine(machine_id));
            continue;
        }
        if authority.lineage_handoff_id() != handoff.id()
            || authority.probation_clearance_id() != probation.id()
            || authority.telemetry_clearance_id() != telemetry.id()
            || authority.predecessor_root_digest() != probation.predecessor_root_digest()
            || authority.current_head_digest() != probation.current_head_digest()
            || authority.governance_view_digest() != probation.governance_view_digest()
            || authority.registry_head_digest() != probation.registry_head_digest()
            || authority.predecessor_finalization_sequence() != handoff.predecessor_finalization_sequence()
            || authority.handoff_plan_digest() != handoff.plan_digest()
        {
            violations.push(LineageBoundCurrentNoRollbackError::HardwareLineageMismatch(machine_id.clone()));
        }
        if authority.trust_snapshot_digest() != registry_head.trust_snapshot_digest() {
            violations.push(LineageBoundCurrentNoRollbackError::HardwareTrustMismatch(machine_id.clone()));
        }
        if authority.containment_state_digest() != governance_view.containment_state_digest()
            || authority.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
        {
            violations.push(LineageBoundCurrentNoRollbackError::HardwareContainmentMismatch(machine_id.clone()));
        }
        if authority.current_operational_basis_id() != observation_basis.id()
            || authority.current_clock_envelope_id() != observation_clock.id()
        {
            violations.push(LineageBoundCurrentNoRollbackError::HardwareClockMismatch(machine_id.clone()));
        }
        let expires_at_unix_ms = match seconds_to_millis(authority.statement().expires_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if expires_at_unix_ms <= observation_clock.upper_unix_ms() {
            violations.push(LineageBoundCurrentNoRollbackError::HardwareExpired(machine_id.clone()));
        }
        let issued_at_unix_ms = match seconds_to_millis(authority.statement().issued_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        evidence_ready_at_unix_ms = evidence_ready_at_unix_ms.max(issued_at_unix_ms);
        hardware_commitments.push(HardwareAuthorityCommitment {
            authority_id: authority.id().to_hex(),
            machine_id,
            reauthorization_sequence: authority.statement().reauthorization_sequence,
            statement_digest: authority.statement_digest().to_hex(),
            signed_evidence_digest: authority.signed_evidence_digest().to_hex(),
            hardware_identity_digest: authority.statement().hardware_identity_digest.to_hex(),
            machine_profile_digest: authority.statement().machine_profile_digest.to_hex(),
            firmware_digest: authority.statement().firmware_digest.to_hex(),
            calibration_digest: authority.statement().calibration_digest.to_hex(),
            capability_digest: authority.statement().capability_digest.to_hex(),
            expires_at_unix_s: authority.statement().expires_at_unix_s,
        });
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    hardware_commitments.sort_by(|left, right| {
        left.machine_id.cmp(&right.machine_id).then(left.authority_id.cmp(&right.authority_id))
    });
    let machine_ids = hardware_commitments
        .iter()
        .map(|commitment| commitment.machine_id.clone())
        .collect::<Vec<_>>();
    let hardware_authority_set_digest =
        hash_serializable(HARDWARE_SET_DOMAIN, &hardware_commitments).map_err(|error| vec![error])?;
    let probation_tracker_digest = hash_serializable(
        PROBATION_TRACKER_DOMAIN,
        &ProbationTrackerCommitment {
            probation_clearance_id: probation.id().to_hex(),
            telemetry_clearance_id: telemetry.id().to_hex(),
            observation_set_digest: probation.observation_set_digest().to_hex(),
            telemetry_binding_set_digest: telemetry.telemetry_binding_set_digest().to_hex(),
        },
    )
    .map_err(|error| vec![error])?;
    let observation_clock_lineage_digest = digest_observation_clock_lineage(
        activation_basis,
        activation_to_observation_clock_bridge,
        observation_basis,
    )
    .map_err(|error| vec![error])?;

    let commitment = EvidenceBindingCommitment {
        schema: LINEAGE_BOUND_OPERATIONAL_EVIDENCE_BINDING_SCHEMA,
        governance_view_id: governance_view.id().to_hex(),
        registry_head_id: registry_head.id().to_hex(),
        retention_head_id: retention_head.id().to_hex(),
        lineage_handoff_id: handoff.id().to_hex(),
        activation_permit_id: activation.id().to_hex(),
        probation_clearance_id: probation.id().to_hex(),
        telemetry_clearance_id: telemetry.id().to_hex(),
        predecessor_root_digest: probation.predecessor_root_digest().to_hex(),
        predecessor_current_head_digest: probation.current_head_digest().to_hex(),
        predecessor_governance_view_digest: probation.governance_view_digest().to_hex(),
        predecessor_registry_head_digest: probation.registry_head_digest().to_hex(),
        predecessor_finalization_sequence: handoff.predecessor_finalization_sequence(),
        upgrade_cycle_sequence,
        handoff_plan_digest: handoff.plan_digest().to_hex(),
        probation_tracker_digest: probation_tracker_digest.to_hex(),
        hardware_authority_set_digest: hardware_authority_set_digest.to_hex(),
        hardware_authority_count: hardware_commitments.len(),
        machine_ids: machine_ids.clone(),
        retention_policy_digest: retention_head.policy_digest().to_hex(),
        retention_policy_sequence: retention_head.sequence(),
        key_continuity_id: key_continuity.id().to_hex(),
        key_snapshot_sequence: registry_head.trust_snapshot_sequence(),
        durable_clock_continuity_digest: activation.clock_lineage_digest().to_hex(),
        durable_clock_epoch: activation_basis.epoch(),
        observation_clock_lineage_digest: observation_clock_lineage_digest.to_hex(),
        activation_operational_basis_id: activation_basis.id().to_hex(),
        observation_operational_basis_id: observation_basis.id().to_hex(),
        observation_clock_envelope_id: observation_clock.id().to_hex(),
        governance_checkpoint_digest: governance_view.checkpoint_digest().to_hex(),
        transparency_log_digest: governance_view.transparency_log_digest().to_hex(),
        activates_at_unix_ms: activation.activates_at_unix_ms(),
        finalization_deadline_unix_ms: activation.finalization_deadline_unix_ms(),
        evidence_ready_at_unix_ms,
    };
    let id = LineageBoundOperationalEvidenceBindingIdV1(
        hash_serializable(BINDING_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(LineageBoundOperationalEvidenceBindingV1 {
        id,
        governance_view_id: governance_view.id(),
        registry_head_id: registry_head.id(),
        retention_head_id: retention_head.id(),
        lineage_handoff_id: handoff.id(),
        activation_permit_id: activation.id(),
        probation_clearance_id: probation.id(),
        telemetry_clearance_id: telemetry.id(),
        predecessor_root_digest: probation.predecessor_root_digest(),
        predecessor_current_head_digest: probation.current_head_digest(),
        predecessor_governance_view_digest: probation.governance_view_digest(),
        predecessor_registry_head_digest: probation.registry_head_digest(),
        predecessor_finalization_sequence: handoff.predecessor_finalization_sequence(),
        upgrade_cycle_sequence,
        handoff_plan_digest: handoff.plan_digest(),
        probation_tracker_digest,
        hardware_authority_set_digest,
        hardware_authority_count: hardware_commitments.len(),
        machine_ids,
        retention_policy_digest: retention_head.policy_digest(),
        retention_policy_sequence: retention_head.sequence(),
        key_continuity_id: key_continuity.id(),
        key_snapshot_sequence: registry_head.trust_snapshot_sequence(),
        durable_clock_continuity_digest: activation.clock_lineage_digest(),
        durable_clock_epoch: activation_basis.epoch(),
        observation_clock_lineage_digest,
        activation_operational_basis_id: activation_basis.id(),
        observation_operational_basis_id: observation_basis.id(),
        observation_clock_envelope_id: observation_clock.id(),
        governance_checkpoint_digest: governance_view.checkpoint_digest(),
        transparency_log_digest: governance_view.transparency_log_digest(),
        activates_at_unix_ms: activation.activates_at_unix_ms(),
        finalization_deadline_unix_ms: activation.finalization_deadline_unix_ms(),
        evidence_ready_at_unix_ms,
    })
}

pub fn build_lineage_bound_no_rollback_operational_evidence_v1(
    binding: &LineageBoundOperationalEvidenceBindingV1,
    upgrade_state_digest: Sha256Digest,
) -> Result<UpgradeOperationalEvidenceDigests, LineageBoundCurrentNoRollbackError> {
    if upgrade_state_digest.0 == [0; 32] {
        return Err(LineageBoundCurrentNoRollbackError::ZeroUpgradeStateDigest);
    }
    let reauthorized_machine_count = u64::try_from(binding.hardware_authority_count)
        .map_err(|_| LineageBoundCurrentNoRollbackError::HardwareCountOverflow)?;
    Ok(UpgradeOperationalEvidenceDigests {
        upgrade_state_digest,
        probation_tracker_digest: binding.probation_tracker_digest,
        hardware_reauthorization_tracker_digest: binding.hardware_authority_set_digest,
        retention_policy_digest: binding.retention_policy_digest,
        key_continuity_digest: binding.key_continuity_id.as_digest(),
        clock_continuity_digest: binding.durable_clock_continuity_digest,
        probation_clearance_digest: Some(binding.probation_clearance_id.as_digest()),
        automatic_rollback_digest: None,
        probation_sequence: Some(binding.upgrade_cycle_sequence),
        reauthorized_machine_count,
        retention_policy_sequence: binding.retention_policy_sequence,
        key_snapshot_sequence: binding.key_snapshot_sequence,
        clock_epoch: binding.durable_clock_epoch,
    })
}

pub fn derive_lineage_bound_current_no_rollback_v1(
    binding: &LineageBoundOperationalEvidenceBindingV1,
    observation_basis: &OperationalClockBasisV1,
    states: &[FabricationUpgradeOperationalState],
    publications: &[UpgradeOperationalHeadPublicationV1],
    log: &TransparencyLog,
) -> Result<LineageBoundCurrentNoRollbackV1, Vec<LineageBoundCurrentNoRollbackError>> {
    let mut violations = Vec::new();
    if observation_basis.id() != binding.observation_operational_basis_id {
        violations.push(LineageBoundCurrentNoRollbackError::ObservationBasisMismatch);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundCurrentNoRollbackError::Clock(error));
            return Err(violations);
        }
    };
    if observation_clock.id() != binding.observation_clock_envelope_id {
        violations.push(LineageBoundCurrentNoRollbackError::ObservationEnvelopeMismatch);
    }

    if states.is_empty() {
        violations.push(LineageBoundCurrentNoRollbackError::EmptyOperationalLineage);
        return Err(violations);
    }
    if states.len() > MAX_LINEAGE_BOUND_OPERATIONAL_STATES {
        violations.push(LineageBoundCurrentNoRollbackError::TooManyOperationalStates {
            actual: states.len(), maximum: MAX_LINEAGE_BOUND_OPERATIONAL_STATES,
        });
        return Err(violations);
    }
    if let Err(error) = log.validate() {
        violations.push(LineageBoundCurrentNoRollbackError::TransparencyLogInvalid(format!("{error:?}")));
    }
    let transparency_log_digest = match digest_transparency_log(log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundCurrentNoRollbackError::TransparencyLogInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    if transparency_log_digest != binding.transparency_log_digest {
        violations.push(LineageBoundCurrentNoRollbackError::TransparencyLogMismatch);
    }

    let log_kind = match upgrade_operational_head_log_kind(binding.handoff_plan_digest) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundCurrentNoRollbackError::Encoding(format!("{error:?}")));
            return Err(violations);
        }
    };
    let matching_entries = log.entries.iter().filter(|entry| entry.kind == log_kind).collect::<Vec<_>>();
    if states.len() != publications.len() || states.len() != matching_entries.len() {
        violations.push(LineageBoundCurrentNoRollbackError::OperationalInputCountMismatch {
            states: states.len(), publications: publications.len(), log_entries: matching_entries.len(),
        });
        return Err(violations);
    }

    let mut lineage_commitments = Vec::with_capacity(states.len());
    let mut state_digests = Vec::with_capacity(states.len());
    let mut publication_digests = Vec::with_capacity(states.len());
    for (index, ((state, publication), entry)) in states.iter().zip(publications).zip(&matching_entries).enumerate() {
        if let Err(error) = state.validate_shape() {
            violations.push(LineageBoundCurrentNoRollbackError::OperationalStateInvalid {
                index, reason: format!("{error:?}"),
            });
            continue;
        }
        if index == 0 && (state.generation != 1 || state.previous_state_digest.is_some()) {
            violations.push(LineageBoundCurrentNoRollbackError::OperationalGenesisInvalid);
        }
        if state.handoff_digest != binding.handoff_plan_digest {
            violations.push(LineageBoundCurrentNoRollbackError::HandoffDigestMismatch { index });
        }
        if state.committed_at_unix_ms < binding.activates_at_unix_ms {
            violations.push(LineageBoundCurrentNoRollbackError::StateBeforeActivation { index });
        }
        if index > 0 {
            if let Err(error) = verify_upgrade_operational_state_successor(&states[index - 1], state) {
                violations.push(LineageBoundCurrentNoRollbackError::OperationalSuccessorInvalid {
                    index, reason: format!("{error:?}"),
                });
            }
        }
        if let Some(digest) = state.evidence.automatic_rollback_digest {
            violations.push(LineageBoundCurrentNoRollbackError::RollbackObserved {
                generation: state.generation, digest,
            });
        }
        let state_digest = match digest_upgrade_operational_state(state) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundCurrentNoRollbackError::OperationalStateInvalid {
                    index, reason: format!("{error:?}"),
                });
                continue;
            }
        };
        let expected_publication = match build_upgrade_operational_head_publication_v1(state) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundCurrentNoRollbackError::Encoding(format!("{error:?}")));
                continue;
            }
        };
        if publication != &expected_publication {
            violations.push(LineageBoundCurrentNoRollbackError::PublicationMismatch { index });
        }
        let publication_digest = match digest_upgrade_operational_head_publication_v1(publication) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundCurrentNoRollbackError::Encoding(format!("{error:?}")));
                continue;
            }
        };
        if entry.subject_digest != publication_digest {
            violations.push(LineageBoundCurrentNoRollbackError::PublicationDigestMismatch { index });
        }
        let publication_recorded_at_ms = match seconds_to_millis(entry.recorded_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if publication_recorded_at_ms < state.committed_at_unix_ms {
            violations.push(LineageBoundCurrentNoRollbackError::PublicationBeforeStateCommit { index });
        }
        if publication_recorded_at_ms > observation_clock.lower_unix_ms() {
            violations.push(LineageBoundCurrentNoRollbackError::PublicationMayBeFuture { index });
        }
        state_digests.push(state_digest);
        publication_digests.push(publication_digest);
        lineage_commitments.push(OperationalLineageEntryCommitment {
            generation: state.generation,
            state_digest: state_digest.to_hex(),
            publication_digest: publication_digest.to_hex(),
            publication_entry_sequence: entry.sequence,
            publication_recorded_at_unix_s: entry.recorded_at_unix_s,
        });
    }

    let Some(candidate) = states.last() else {
        return Err(vec![LineageBoundCurrentNoRollbackError::EmptyOperationalLineage]);
    };
    let expected_machine_count = match u64::try_from(binding.hardware_authority_count) {
        Ok(value) => value,
        Err(_) => {
            violations.push(LineageBoundCurrentNoRollbackError::HardwareCountOverflow);
            0
        }
    };
    for (matches, name) in [
        (candidate.evidence.probation_tracker_digest == binding.probation_tracker_digest, "probation_tracker_digest"),
        (candidate.evidence.hardware_reauthorization_tracker_digest == binding.hardware_authority_set_digest, "hardware_reauthorization_tracker_digest"),
        (candidate.evidence.retention_policy_digest == binding.retention_policy_digest, "retention_policy_digest"),
        (candidate.evidence.key_continuity_digest == binding.key_continuity_id.as_digest(), "key_continuity_digest"),
        (candidate.evidence.clock_continuity_digest == binding.durable_clock_continuity_digest, "clock_continuity_digest"),
        (candidate.evidence.probation_clearance_digest == Some(binding.probation_clearance_id.as_digest()), "probation_clearance_digest"),
        (candidate.evidence.probation_sequence == Some(binding.upgrade_cycle_sequence), "probation_sequence"),
        (candidate.evidence.reauthorized_machine_count == expected_machine_count, "reauthorized_machine_count"),
        (candidate.evidence.retention_policy_sequence == binding.retention_policy_sequence, "retention_policy_sequence"),
        (candidate.evidence.key_snapshot_sequence == binding.key_snapshot_sequence, "key_snapshot_sequence"),
        (candidate.evidence.clock_epoch == binding.durable_clock_epoch, "clock_epoch"),
    ] {
        if !matches {
            violations.push(LineageBoundCurrentNoRollbackError::CandidateEvidenceMismatch(name));
        }
    }
    if candidate.committed_at_unix_ms < binding.evidence_ready_at_unix_ms {
        violations.push(LineageBoundCurrentNoRollbackError::CandidateCommittedBeforeEvidence);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let operational_lineage_digest =
        hash_serializable(OPERATIONAL_LINEAGE_DOMAIN, &lineage_commitments).map_err(|error| vec![error])?;
    let Some(state_digest) = state_digests.last().copied() else {
        return Err(vec![LineageBoundCurrentNoRollbackError::EmptyOperationalLineage]);
    };
    let Some(publication_digest) = publication_digests.last().copied() else {
        return Err(vec![LineageBoundCurrentNoRollbackError::EmptyOperationalLineage]);
    };
    let Some(last_entry) = matching_entries.last() else {
        return Err(vec![LineageBoundCurrentNoRollbackError::EmptyOperationalLineage]);
    };

    let commitment = CurrentNoRollbackCommitment {
        schema: LINEAGE_BOUND_CURRENT_NO_ROLLBACK_SCHEMA,
        evidence_binding_id: binding.id.to_hex(),
        governance_view_id: binding.governance_view_id.to_hex(),
        registry_head_id: binding.registry_head_id.to_hex(),
        retention_head_id: binding.retention_head_id.to_hex(),
        lineage_handoff_id: binding.lineage_handoff_id.to_hex(),
        activation_permit_id: binding.activation_permit_id.to_hex(),
        probation_clearance_id: binding.probation_clearance_id.to_hex(),
        telemetry_clearance_id: binding.telemetry_clearance_id.to_hex(),
        predecessor_root_digest: binding.predecessor_root_digest.to_hex(),
        predecessor_current_head_digest: binding.predecessor_current_head_digest.to_hex(),
        predecessor_finalization_sequence: binding.predecessor_finalization_sequence,
        upgrade_cycle_sequence: binding.upgrade_cycle_sequence,
        handoff_plan_digest: binding.handoff_plan_digest.to_hex(),
        state_digest: state_digest.to_hex(),
        state_generation: candidate.generation,
        publication_digest: publication_digest.to_hex(),
        publication_entry_sequence: last_entry.sequence,
        transparency_log_digest: transparency_log_digest.to_hex(),
        operational_lineage_digest: operational_lineage_digest.to_hex(),
        operational_state_count: states.len(),
        hardware_authority_set_digest: binding.hardware_authority_set_digest.to_hex(),
        hardware_authority_count: binding.hardware_authority_count,
        key_continuity_id: binding.key_continuity_id.to_hex(),
        observation_clock_lineage_digest: binding.observation_clock_lineage_digest.to_hex(),
        observation_operational_basis_id: binding.observation_operational_basis_id.to_hex(),
        observation_clock_envelope_id: binding.observation_clock_envelope_id.to_hex(),
    };
    let id = LineageBoundCurrentNoRollbackIdV1(
        hash_serializable(CURRENT_NO_ROLLBACK_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(LineageBoundCurrentNoRollbackV1 {
        id,
        evidence_binding_id: binding.id,
        governance_view_id: binding.governance_view_id,
        registry_head_id: binding.registry_head_id,
        retention_head_id: binding.retention_head_id,
        lineage_handoff_id: binding.lineage_handoff_id,
        activation_permit_id: binding.activation_permit_id,
        probation_clearance_id: binding.probation_clearance_id,
        telemetry_clearance_id: binding.telemetry_clearance_id,
        predecessor_root_digest: binding.predecessor_root_digest,
        predecessor_current_head_digest: binding.predecessor_current_head_digest,
        predecessor_finalization_sequence: binding.predecessor_finalization_sequence,
        upgrade_cycle_sequence: binding.upgrade_cycle_sequence,
        handoff_plan_digest: binding.handoff_plan_digest,
        state_digest,
        state_generation: candidate.generation,
        publication_digest,
        publication_entry_sequence: last_entry.sequence,
        transparency_log_digest,
        operational_lineage_digest,
        operational_state_count: states.len(),
        hardware_authority_set_digest: binding.hardware_authority_set_digest,
        hardware_authority_count: binding.hardware_authority_count,
        key_continuity_id: binding.key_continuity_id,
        observation_clock_lineage_digest: binding.observation_clock_lineage_digest,
        observation_operational_basis_id: binding.observation_operational_basis_id,
        observation_clock_envelope_id: binding.observation_clock_envelope_id,
    })
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), LineageBoundCurrentNoRollbackError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() { return Ok(()); }
        return Err(LineageBoundCurrentNoRollbackError::BrokenClockLineage {
            hop: 1,
            expected_predecessor: prior_basis_id.to_hex(),
            actual_predecessor: bridge[0].predecessor_operational_basis_id().map(|value| value.to_hex()),
        });
    }
    let mut expected = prior_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(LineageBoundCurrentNoRollbackError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(LineageBoundCurrentNoRollbackError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn digest_observation_clock_lineage(
    start: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    end: &OperationalClockBasisV1,
) -> Result<Sha256Digest, LineageBoundCurrentNoRollbackError> {
    let commitment = ClockLineageCommitment {
        activation_basis_id: start.id().to_hex(),
        bridge_basis_ids: bridge.iter().map(|basis| basis.id().to_hex()).collect(),
        observation_basis_id: end.id().to_hex(),
    };
    hash_serializable(OBSERVATION_CLOCK_LINEAGE_DOMAIN, &commitment)
}

fn seconds_to_millis(value: u64) -> Result<u64, LineageBoundCurrentNoRollbackError> {
    value.checked_mul(1_000).ok_or(LineageBoundCurrentNoRollbackError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundCurrentNoRollbackError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundCurrentNoRollbackError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
