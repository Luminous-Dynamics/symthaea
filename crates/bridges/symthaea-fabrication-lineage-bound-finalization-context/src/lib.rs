// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Single-view pre-finalization context that preserves the global upgrade predecessor lineage.
//!
//! This crate does not authorize or execute finalization. It proves that the complete hardened
//! lineage-bound upgrade authority stack converges on one exact current governance view and one exact
//! hardware set. Its opaque ID is the payload a later finalization quorum may authorize.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_evidence_retention_head::{
    CurrentEvidenceRetentionHeadIdV1, CurrentEvidenceRetentionHeadV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_key_continuity_authority::{
    ClockGovernedKeyContinuityIdV1, ClockGovernedKeyContinuityV1,
};
use symthaea_fabrication_lineage_bound_current_no_rollback::{
    LineageBoundCurrentNoRollbackIdV1, LineageBoundCurrentNoRollbackV1,
    LineageBoundOperationalEvidenceBindingIdV1, LineageBoundOperationalEvidenceBindingV1,
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

pub const LINEAGE_BOUND_FINALIZATION_CONTEXT_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-finalization-context.v1";
pub const MAX_LINEAGE_BOUND_FINALIZATION_HARDWARE_AUTHORITIES: usize = 65_536;

const CONTEXT_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalization-context-policy.v1\0";
// Must exactly match #3268's canonical hardware-set commitment. Recomputing it here is deliberate:
// this is an independent exact-set rebind, not a caller assertion.
const HARDWARE_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-hardware-authority-set.v1\0";
const CONTEXT_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalization-context.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LineageBoundFinalizationContextPolicyV1 {
    pub minimum_reauthorized_machines: usize,
    pub maximum_reauthorized_machines: usize,
}

impl Default for LineageBoundFinalizationContextPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_reauthorized_machines: 1,
            maximum_reauthorized_machines: MAX_LINEAGE_BOUND_FINALIZATION_HARDWARE_AUTHORITIES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundUpgradeFinalizationContextIdV1(Sha256Digest);

impl LineageBoundUpgradeFinalizationContextIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundUpgradeFinalizationContextV1 {
    id: LineageBoundUpgradeFinalizationContextIdV1,
    context_policy_digest: Sha256Digest,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    operational_evidence_binding_id: LineageBoundOperationalEvidenceBindingIdV1,
    no_rollback_id: LineageBoundCurrentNoRollbackIdV1,
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
    operational_state_digest: Sha256Digest,
    operational_state_generation: u64,
    operational_lineage_digest: Sha256Digest,
    hardware_authority_set_digest: Sha256Digest,
    hardware_authority_count: usize,
    hardware_authority_ids: Vec<String>,
    machine_ids: Vec<String>,
    key_continuity_id: ClockGovernedKeyContinuityIdV1,
    current_trust_snapshot_digest: Sha256Digest,
    current_trust_snapshot_sequence: u64,
    current_containment_state_digest: Sha256Digest,
    current_compromise_tracker_digest: Sha256Digest,
    current_containment_generation: u64,
    current_transparency_log_digest: Sha256Digest,
    current_checkpoint_digest: Sha256Digest,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    observation_clock_lineage_digest: Sha256Digest,
    finalization_deadline_unix_ms: u64,
    probation_clearance_expires_at_unix_ms: u64,
    earliest_hardware_expiry_unix_ms: u64,
}

impl LineageBoundUpgradeFinalizationContextV1 {
    pub fn id(&self) -> LineageBoundUpgradeFinalizationContextIdV1 { self.id }
    pub fn signing_payload_digest(&self) -> Sha256Digest { self.id.as_digest() }
    pub fn context_policy_digest(&self) -> Sha256Digest { self.context_policy_digest }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.governance_view_id }
    pub fn registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 { self.registry_head_id }
    pub fn retention_head_id(&self) -> CurrentEvidenceRetentionHeadIdV1 { self.retention_head_id }
    pub fn operational_evidence_binding_id(&self) -> LineageBoundOperationalEvidenceBindingIdV1 { self.operational_evidence_binding_id }
    pub fn no_rollback_id(&self) -> LineageBoundCurrentNoRollbackIdV1 { self.no_rollback_id }
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
    pub fn operational_state_digest(&self) -> Sha256Digest { self.operational_state_digest }
    pub fn operational_state_generation(&self) -> u64 { self.operational_state_generation }
    pub fn operational_lineage_digest(&self) -> Sha256Digest { self.operational_lineage_digest }
    pub fn hardware_authority_set_digest(&self) -> Sha256Digest { self.hardware_authority_set_digest }
    pub fn hardware_authority_count(&self) -> usize { self.hardware_authority_count }
    pub fn hardware_authority_ids(&self) -> &[String] { &self.hardware_authority_ids }
    pub fn machine_ids(&self) -> &[String] { &self.machine_ids }
    pub fn key_continuity_id(&self) -> ClockGovernedKeyContinuityIdV1 { self.key_continuity_id }
    pub fn current_trust_snapshot_digest(&self) -> Sha256Digest { self.current_trust_snapshot_digest }
    pub fn current_trust_snapshot_sequence(&self) -> u64 { self.current_trust_snapshot_sequence }
    pub fn current_containment_state_digest(&self) -> Sha256Digest { self.current_containment_state_digest }
    pub fn current_compromise_tracker_digest(&self) -> Sha256Digest { self.current_compromise_tracker_digest }
    pub fn current_containment_generation(&self) -> u64 { self.current_containment_generation }
    pub fn current_transparency_log_digest(&self) -> Sha256Digest { self.current_transparency_log_digest }
    pub fn current_checkpoint_digest(&self) -> Sha256Digest { self.current_checkpoint_digest }
    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.current_clock_envelope_id }
    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.current_operational_basis_id }
    pub fn observation_clock_lineage_digest(&self) -> Sha256Digest { self.observation_clock_lineage_digest }
    pub fn finalization_deadline_unix_ms(&self) -> u64 { self.finalization_deadline_unix_ms }
    pub fn probation_clearance_expires_at_unix_ms(&self) -> u64 { self.probation_clearance_expires_at_unix_ms }
    pub fn earliest_hardware_expiry_unix_ms(&self) -> u64 { self.earliest_hardware_expiry_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundFinalizationContextError {
    InvalidPolicy,
    CurrentGovernanceMismatch,
    CurrentRetentionMismatch,
    OperationalBindingMismatch,
    CurrentNoRollbackMismatch,
    PredecessorProvenanceMismatch,
    LineageAuthorityMismatch,
    KeyContinuityMismatch,
    KeyContinuityContainmentMismatch,
    CurrentBasisMismatch,
    CurrentEnvelopeMismatch,
    Clock(ClockGovernanceTimeError),
    FinalizationMayBeClosed,
    ProbationMayBeExpired,
    InsufficientHardwareAuthorities { actual: usize, required: usize },
    TooManyHardwareAuthorities { actual: usize, maximum: usize },
    HardwareCountMismatch { supplied: usize, bound: usize },
    DuplicateHardwareMachine(String),
    DuplicateHardwareAuthority(String),
    HardwareLineageMismatch(String),
    HardwareTrustMismatch(String),
    HardwareContainmentMismatch(String),
    HardwareClockMismatch(String),
    HardwareStatementMismatch(String),
    HardwareStatementMayBeFuture(String),
    HardwareMayExpire(String),
    HardwareSetDigestMismatch,
    HardwareMachineSetMismatch,
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct ContextPolicyCommitment {
    minimum_reauthorized_machines: usize,
    maximum_reauthorized_machines: usize,
}

// Field order and domain intentionally mirror #3268's exact canonical set computation.
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
struct FinalizationContextCommitment {
    schema: &'static str,
    context_policy_digest: String,
    governance_view_id: String,
    registry_head_id: String,
    retention_head_id: String,
    operational_evidence_binding_id: String,
    no_rollback_id: String,
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
    operational_state_digest: String,
    operational_state_generation: u64,
    operational_lineage_digest: String,
    hardware_authority_set_digest: String,
    hardware_authority_count: usize,
    hardware_authority_ids: Vec<String>,
    machine_ids: Vec<String>,
    key_continuity_id: String,
    current_trust_snapshot_digest: String,
    current_trust_snapshot_sequence: u64,
    current_containment_state_digest: String,
    current_compromise_tracker_digest: String,
    current_containment_generation: u64,
    current_transparency_log_digest: String,
    current_checkpoint_digest: String,
    current_clock_envelope_id: String,
    current_operational_basis_id: String,
    observation_clock_lineage_digest: String,
    finalization_deadline_unix_ms: u64,
    probation_clearance_expires_at_unix_ms: u64,
    earliest_hardware_expiry_unix_ms: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_lineage_bound_upgrade_finalization_context_v1(
    policy: &LineageBoundFinalizationContextPolicyV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    registry_head: &QuorumObservedWitnessRegistryHeadV1,
    retention_head: &CurrentEvidenceRetentionHeadV1,
    evidence_binding: &LineageBoundOperationalEvidenceBindingV1,
    no_rollback: &LineageBoundCurrentNoRollbackV1,
    handoff: &LineageBoundClockGovernedUpgradeHandoffV1,
    activation: &LineageBoundUpgradeActivationPermitV1,
    probation: &LineageBoundUpgradeProbationClearanceV1,
    telemetry: &LineageBoundTelemetryProbationClearanceV1,
    hardware_authorities: &[LineageBoundHardwareReauthorizationV1],
    key_continuity: &ClockGovernedKeyContinuityV1,
    current_basis: &OperationalClockBasisV1,
) -> Result<LineageBoundUpgradeFinalizationContextV1, Vec<LineageBoundFinalizationContextError>> {
    let mut violations = Vec::new();
    if !valid_policy(policy) {
        violations.push(LineageBoundFinalizationContextError::InvalidPolicy);
    }

    if governance_view.registry_head_id() != registry_head.id()
        || governance_view.registry_digest() != registry_head.registry_digest()
        || governance_view.registry_sequence() != registry_head.sequence()
    {
        violations.push(LineageBoundFinalizationContextError::CurrentGovernanceMismatch);
    }
    if retention_head.governance_view_id() != governance_view.id()
        || retention_head.registry_head_id() != registry_head.id()
        || retention_head.governance_checkpoint_digest() != governance_view.checkpoint_digest()
        || retention_head.transparency_log_digest() != governance_view.transparency_log_digest()
        || retention_head.containment_state_digest() != governance_view.containment_state_digest()
        || retention_head.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
        || retention_head.containment_generation() != governance_view.containment_generation()
    {
        violations.push(LineageBoundFinalizationContextError::CurrentRetentionMismatch);
    }

    if evidence_binding.governance_view_id() != governance_view.id()
        || evidence_binding.registry_head_id() != registry_head.id()
        || evidence_binding.retention_head_id() != retention_head.id()
        || evidence_binding.governance_checkpoint_digest() != governance_view.checkpoint_digest()
        || evidence_binding.transparency_log_digest() != governance_view.transparency_log_digest()
    {
        violations.push(LineageBoundFinalizationContextError::OperationalBindingMismatch);
    }
    if no_rollback.evidence_binding_id() != evidence_binding.id()
        || no_rollback.governance_view_id() != governance_view.id()
        || no_rollback.registry_head_id() != registry_head.id()
        || no_rollback.retention_head_id() != retention_head.id()
        || no_rollback.transparency_log_digest() != governance_view.transparency_log_digest()
        || no_rollback.observation_operational_basis_id() != evidence_binding.observation_operational_basis_id()
        || no_rollback.observation_clock_envelope_id() != evidence_binding.observation_clock_envelope_id()
        || no_rollback.observation_clock_lineage_digest() != evidence_binding.observation_clock_lineage_digest()
    {
        violations.push(LineageBoundFinalizationContextError::CurrentNoRollbackMismatch);
    }

    let predecessor_root_digest = handoff.predecessor_root_id().as_digest();
    let predecessor_current_head_digest = handoff.current_head_id().as_digest();
    let predecessor_governance_view_digest = handoff.governance_view_id().as_digest();
    let predecessor_registry_head_digest = handoff.registry_head_id().as_digest();
    let predecessor_finalization_sequence = handoff.predecessor_finalization_sequence();
    let predecessor_matches = activation.predecessor_root_digest() == predecessor_root_digest
        && probation.predecessor_root_digest() == predecessor_root_digest
        && telemetry.predecessor_root_digest() == predecessor_root_digest
        && evidence_binding.predecessor_root_digest() == predecessor_root_digest
        && no_rollback.predecessor_root_digest() == predecessor_root_digest
        && activation.current_head_digest() == predecessor_current_head_digest
        && probation.current_head_digest() == predecessor_current_head_digest
        && telemetry.current_head_digest() == predecessor_current_head_digest
        && evidence_binding.predecessor_current_head_digest() == predecessor_current_head_digest
        && no_rollback.predecessor_current_head_digest() == predecessor_current_head_digest
        && activation.governance_view_digest() == predecessor_governance_view_digest
        && probation.governance_view_digest() == predecessor_governance_view_digest
        && telemetry.governance_view_digest() == predecessor_governance_view_digest
        && evidence_binding.predecessor_governance_view_digest() == predecessor_governance_view_digest
        && activation.registry_head_digest() == predecessor_registry_head_digest
        && probation.registry_head_digest() == predecessor_registry_head_digest
        && telemetry.registry_head_digest() == predecessor_registry_head_digest
        && evidence_binding.predecessor_registry_head_digest() == predecessor_registry_head_digest
        && activation.predecessor_finalization_sequence() == predecessor_finalization_sequence
        && probation.predecessor_finalization_sequence() == predecessor_finalization_sequence
        && telemetry.predecessor_finalization_sequence() == predecessor_finalization_sequence
        && evidence_binding.predecessor_finalization_sequence() == predecessor_finalization_sequence
        && no_rollback.predecessor_finalization_sequence() == predecessor_finalization_sequence;
    if !predecessor_matches {
        violations.push(LineageBoundFinalizationContextError::PredecessorProvenanceMismatch);
    }

    if activation.lineage_handoff_id() != handoff.id()
        || probation.lineage_handoff_id() != handoff.id()
        || probation.activation_permit_id() != activation.id()
        || telemetry.clearance_id() != probation.id()
        || telemetry.activation_permit_digest() != activation.id().as_digest()
        || evidence_binding.lineage_handoff_id() != handoff.id()
        || evidence_binding.activation_permit_id() != activation.id()
        || evidence_binding.probation_clearance_id() != probation.id()
        || evidence_binding.telemetry_clearance_id() != telemetry.id()
        || no_rollback.lineage_handoff_id() != handoff.id()
        || no_rollback.activation_permit_id() != activation.id()
        || no_rollback.probation_clearance_id() != probation.id()
        || no_rollback.telemetry_clearance_id() != telemetry.id()
        || activation.handoff_plan_digest() != handoff.plan_digest()
        || probation.handoff_plan_digest() != handoff.plan_digest()
        || telemetry.handoff_plan_digest() != handoff.plan_digest()
        || evidence_binding.handoff_plan_digest() != handoff.plan_digest()
        || no_rollback.handoff_plan_digest() != handoff.plan_digest()
        || evidence_binding.upgrade_cycle_sequence() != no_rollback.upgrade_cycle_sequence()
    {
        violations.push(LineageBoundFinalizationContextError::LineageAuthorityMismatch);
    }

    if evidence_binding.key_continuity_id() != key_continuity.id()
        || no_rollback.key_continuity_id() != key_continuity.id()
        || key_continuity.successor_snapshot_digest() != registry_head.trust_snapshot_digest()
        || key_continuity.successor_snapshot_sequence() != registry_head.trust_snapshot_sequence()
    {
        violations.push(LineageBoundFinalizationContextError::KeyContinuityMismatch);
    }
    if key_continuity.containment_state_digest() != governance_view.containment_state_digest()
        || key_continuity.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
    {
        violations.push(LineageBoundFinalizationContextError::KeyContinuityContainmentMismatch);
    }

    if current_basis.id() != governance_view.observation_operational_basis_id()
        || current_basis.id() != retention_head.observation_operational_basis_id()
        || current_basis.id() != evidence_binding.observation_operational_basis_id()
        || current_basis.id() != no_rollback.observation_operational_basis_id()
    {
        violations.push(LineageBoundFinalizationContextError::CurrentBasisMismatch);
    }
    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationContextError::Clock(error));
            return Err(violations);
        }
    };
    if current_clock.id() != governance_view.observation_clock_envelope_id()
        || current_clock.id() != retention_head.observation_clock_envelope_id()
        || current_clock.id() != evidence_binding.observation_clock_envelope_id()
        || current_clock.id() != no_rollback.observation_clock_envelope_id()
    {
        violations.push(LineageBoundFinalizationContextError::CurrentEnvelopeMismatch);
    }

    if current_clock.upper_unix_ms() >= activation.finalization_deadline_unix_ms() {
        violations.push(LineageBoundFinalizationContextError::FinalizationMayBeClosed);
    }
    if current_clock.upper_unix_ms() >= probation.clearance_expires_at_unix_ms() {
        violations.push(LineageBoundFinalizationContextError::ProbationMayBeExpired);
    }

    if hardware_authorities.len() < policy.minimum_reauthorized_machines {
        violations.push(LineageBoundFinalizationContextError::InsufficientHardwareAuthorities {
            actual: hardware_authorities.len(), required: policy.minimum_reauthorized_machines,
        });
    }
    if hardware_authorities.len() > policy.maximum_reauthorized_machines
        || hardware_authorities.len() > MAX_LINEAGE_BOUND_FINALIZATION_HARDWARE_AUTHORITIES
    {
        violations.push(LineageBoundFinalizationContextError::TooManyHardwareAuthorities {
            actual: hardware_authorities.len(),
            maximum: policy.maximum_reauthorized_machines.min(MAX_LINEAGE_BOUND_FINALIZATION_HARDWARE_AUTHORITIES),
        });
        return Err(violations);
    }
    if hardware_authorities.len() != evidence_binding.hardware_authority_count()
        || hardware_authorities.len() != no_rollback.hardware_authority_count()
    {
        violations.push(LineageBoundFinalizationContextError::HardwareCountMismatch {
            supplied: hardware_authorities.len(), bound: evidence_binding.hardware_authority_count(),
        });
    }

    let mut seen_machines = BTreeSet::new();
    let mut seen_authorities = BTreeSet::new();
    let mut hardware_commitments = Vec::with_capacity(hardware_authorities.len());
    let mut earliest_hardware_expiry_unix_ms = u64::MAX;
    for authority in hardware_authorities {
        let machine_id = authority.statement().machine_id.clone();
        let authority_id = authority.id().to_hex();
        if !seen_machines.insert(machine_id.clone()) {
            violations.push(LineageBoundFinalizationContextError::DuplicateHardwareMachine(machine_id));
            continue;
        }
        if !seen_authorities.insert(authority_id.clone()) {
            violations.push(LineageBoundFinalizationContextError::DuplicateHardwareAuthority(authority_id));
            continue;
        }
        if authority.lineage_handoff_id() != handoff.id()
            || authority.probation_clearance_id() != probation.id()
            || authority.telemetry_clearance_id() != telemetry.id()
            || authority.predecessor_root_digest() != predecessor_root_digest
            || authority.current_head_digest() != predecessor_current_head_digest
            || authority.governance_view_digest() != predecessor_governance_view_digest
            || authority.registry_head_digest() != predecessor_registry_head_digest
            || authority.predecessor_finalization_sequence() != predecessor_finalization_sequence
            || authority.handoff_plan_digest() != handoff.plan_digest()
        {
            violations.push(LineageBoundFinalizationContextError::HardwareLineageMismatch(machine_id.clone()));
        }
        if authority.trust_snapshot_digest() != registry_head.trust_snapshot_digest() {
            violations.push(LineageBoundFinalizationContextError::HardwareTrustMismatch(machine_id.clone()));
        }
        if authority.containment_state_digest() != governance_view.containment_state_digest()
            || authority.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
        {
            violations.push(LineageBoundFinalizationContextError::HardwareContainmentMismatch(machine_id.clone()));
        }
        if authority.current_operational_basis_id() != current_basis.id()
            || authority.current_clock_envelope_id() != current_clock.id()
        {
            violations.push(LineageBoundFinalizationContextError::HardwareClockMismatch(machine_id.clone()));
        }
        if authority.statement().handoff_digest != handoff.plan_digest()
            || authority.statement().successor_source_tree_digest != handoff.plan().successor.source_tree_digest
            || authority.statement().successor_executable_digest != handoff.plan().successor.executable_digest
        {
            violations.push(LineageBoundFinalizationContextError::HardwareStatementMismatch(machine_id.clone()));
        }
        let issued_at_unix_ms = match seconds_to_millis(authority.statement().issued_at_unix_s) {
            Ok(value) => value,
            Err(error) => { violations.push(error); continue; }
        };
        let expires_at_unix_ms = match seconds_to_millis(authority.statement().expires_at_unix_s) {
            Ok(value) => value,
            Err(error) => { violations.push(error); continue; }
        };
        if issued_at_unix_ms > current_clock.lower_unix_ms() {
            violations.push(LineageBoundFinalizationContextError::HardwareStatementMayBeFuture(machine_id.clone()));
        }
        if expires_at_unix_ms <= current_clock.upper_unix_ms() {
            violations.push(LineageBoundFinalizationContextError::HardwareMayExpire(machine_id.clone()));
        }
        earliest_hardware_expiry_unix_ms = earliest_hardware_expiry_unix_ms.min(expires_at_unix_ms);
        hardware_commitments.push(HardwareAuthorityCommitment {
            authority_id,
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

    hardware_commitments.sort_by(|left, right| {
        left.machine_id.cmp(&right.machine_id).then(left.authority_id.cmp(&right.authority_id))
    });
    let hardware_authority_ids = hardware_commitments.iter().map(|entry| entry.authority_id.clone()).collect::<Vec<_>>();
    let machine_ids = hardware_commitments.iter().map(|entry| entry.machine_id.clone()).collect::<Vec<_>>();
    let hardware_authority_set_digest = match hash_serializable(HARDWARE_SET_DOMAIN, &hardware_commitments) {
        Ok(value) => value,
        Err(error) => { violations.push(error); Sha256Digest([0; 32]) }
    };
    if hardware_authority_set_digest != evidence_binding.hardware_authority_set_digest()
        || hardware_authority_set_digest != no_rollback.hardware_authority_set_digest()
    {
        violations.push(LineageBoundFinalizationContextError::HardwareSetDigestMismatch);
    }
    if machine_ids.as_slice() != evidence_binding.machine_ids() {
        violations.push(LineageBoundFinalizationContextError::HardwareMachineSetMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let context_policy_digest = hash_serializable(
        CONTEXT_POLICY_DOMAIN,
        &ContextPolicyCommitment {
            minimum_reauthorized_machines: policy.minimum_reauthorized_machines,
            maximum_reauthorized_machines: policy.maximum_reauthorized_machines,
        },
    ).map_err(|error| vec![error])?;

    let commitment = FinalizationContextCommitment {
        schema: LINEAGE_BOUND_FINALIZATION_CONTEXT_SCHEMA,
        context_policy_digest: context_policy_digest.to_hex(),
        governance_view_id: governance_view.id().to_hex(),
        registry_head_id: registry_head.id().to_hex(),
        retention_head_id: retention_head.id().to_hex(),
        operational_evidence_binding_id: evidence_binding.id().to_hex(),
        no_rollback_id: no_rollback.id().to_hex(),
        lineage_handoff_id: handoff.id().to_hex(),
        activation_permit_id: activation.id().to_hex(),
        probation_clearance_id: probation.id().to_hex(),
        telemetry_clearance_id: telemetry.id().to_hex(),
        predecessor_root_digest: predecessor_root_digest.to_hex(),
        predecessor_current_head_digest: predecessor_current_head_digest.to_hex(),
        predecessor_governance_view_digest: predecessor_governance_view_digest.to_hex(),
        predecessor_registry_head_digest: predecessor_registry_head_digest.to_hex(),
        predecessor_finalization_sequence,
        upgrade_cycle_sequence: evidence_binding.upgrade_cycle_sequence(),
        handoff_plan_digest: handoff.plan_digest().to_hex(),
        operational_state_digest: no_rollback.state_digest().to_hex(),
        operational_state_generation: no_rollback.state_generation(),
        operational_lineage_digest: no_rollback.operational_lineage_digest().to_hex(),
        hardware_authority_set_digest: hardware_authority_set_digest.to_hex(),
        hardware_authority_count: hardware_commitments.len(),
        hardware_authority_ids: hardware_authority_ids.clone(),
        machine_ids: machine_ids.clone(),
        key_continuity_id: key_continuity.id().to_hex(),
        current_trust_snapshot_digest: registry_head.trust_snapshot_digest().to_hex(),
        current_trust_snapshot_sequence: registry_head.trust_snapshot_sequence(),
        current_containment_state_digest: governance_view.containment_state_digest().to_hex(),
        current_compromise_tracker_digest: governance_view.compromise_tracker_digest().to_hex(),
        current_containment_generation: governance_view.containment_generation(),
        current_transparency_log_digest: governance_view.transparency_log_digest().to_hex(),
        current_checkpoint_digest: governance_view.checkpoint_digest().to_hex(),
        current_clock_envelope_id: current_clock.id().to_hex(),
        current_operational_basis_id: current_basis.id().to_hex(),
        observation_clock_lineage_digest: no_rollback.observation_clock_lineage_digest().to_hex(),
        finalization_deadline_unix_ms: activation.finalization_deadline_unix_ms(),
        probation_clearance_expires_at_unix_ms: probation.clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    };
    let id = LineageBoundUpgradeFinalizationContextIdV1(
        hash_serializable(CONTEXT_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(LineageBoundUpgradeFinalizationContextV1 {
        id,
        context_policy_digest,
        governance_view_id: governance_view.id(),
        registry_head_id: registry_head.id(),
        retention_head_id: retention_head.id(),
        operational_evidence_binding_id: evidence_binding.id(),
        no_rollback_id: no_rollback.id(),
        lineage_handoff_id: handoff.id(),
        activation_permit_id: activation.id(),
        probation_clearance_id: probation.id(),
        telemetry_clearance_id: telemetry.id(),
        predecessor_root_digest,
        predecessor_current_head_digest,
        predecessor_governance_view_digest,
        predecessor_registry_head_digest,
        predecessor_finalization_sequence,
        upgrade_cycle_sequence: evidence_binding.upgrade_cycle_sequence(),
        handoff_plan_digest: handoff.plan_digest(),
        operational_state_digest: no_rollback.state_digest(),
        operational_state_generation: no_rollback.state_generation(),
        operational_lineage_digest: no_rollback.operational_lineage_digest(),
        hardware_authority_set_digest,
        hardware_authority_count: hardware_commitments.len(),
        hardware_authority_ids,
        machine_ids,
        key_continuity_id: key_continuity.id(),
        current_trust_snapshot_digest: registry_head.trust_snapshot_digest(),
        current_trust_snapshot_sequence: registry_head.trust_snapshot_sequence(),
        current_containment_state_digest: governance_view.containment_state_digest(),
        current_compromise_tracker_digest: governance_view.compromise_tracker_digest(),
        current_containment_generation: governance_view.containment_generation(),
        current_transparency_log_digest: governance_view.transparency_log_digest(),
        current_checkpoint_digest: governance_view.checkpoint_digest(),
        current_clock_envelope_id: current_clock.id(),
        current_operational_basis_id: current_basis.id(),
        observation_clock_lineage_digest: no_rollback.observation_clock_lineage_digest(),
        finalization_deadline_unix_ms: activation.finalization_deadline_unix_ms(),
        probation_clearance_expires_at_unix_ms: probation.clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    })
}

fn valid_policy(policy: &LineageBoundFinalizationContextPolicyV1) -> bool {
    policy.minimum_reauthorized_machines > 0
        && policy.maximum_reauthorized_machines > 0
        && policy.minimum_reauthorized_machines <= policy.maximum_reauthorized_machines
        && policy.maximum_reauthorized_machines <= MAX_LINEAGE_BOUND_FINALIZATION_HARDWARE_AUTHORITIES
}

fn seconds_to_millis(value: u64) -> Result<u64, LineageBoundFinalizationContextError> {
    value.checked_mul(1_000).ok_or(LineageBoundFinalizationContextError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundFinalizationContextError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundFinalizationContextError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
