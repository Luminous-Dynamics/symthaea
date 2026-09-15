// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh execution permission for threshold-authorized lineage-bound upgrade finalization.
//!
//! Authorization is not execution authority. This bridge requires a strict append-only descendant
//! transparency view, definitely later trusted time, unchanged registry/trust/containment/retention
//! semantics, no newly published operational state for the authorized handoff, no pre-existing
//! finalized-head claim for that handoff, and fresh requalification of the exact same hardware
//! statements before minting an opaque execution permit. It performs no mutation itself.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_evidence_retention_head::{
    CurrentEvidenceRetentionHeadIdV1, CurrentEvidenceRetentionHeadV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_lineage_bound_current_no_rollback::{
    LineageBoundCurrentNoRollbackIdV1, LineageBoundCurrentNoRollbackV1,
};
use symthaea_fabrication_lineage_bound_finalization_authorization::{
    AuthorizedLineageBoundFinalizationIdV1, AuthorizedLineageBoundFinalizationV1,
};
use symthaea_fabrication_lineage_bound_finalization_context::LineageBoundUpgradeFinalizationContextIdV1;
use symthaea_fabrication_lineage_bound_hardware::LineageBoundHardwareReauthorizationV1;
use symthaea_fabrication_upgrade_finalized_head::finalized_upgrade_head_log_kind;
use symthaea_fabrication_upgrade_operational_head::upgrade_operational_head_log_kind;
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

pub const LINEAGE_BOUND_FINALIZATION_EXECUTION_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-finalization-execution.v1";
pub const MAX_LINEAGE_BOUND_FINALIZATION_EXECUTION_CLOCK_HOPS: usize = 4096;
pub const MAX_LINEAGE_BOUND_FINALIZATION_EXECUTION_HARDWARE: usize = 65_536;

const HARDWARE_REFRESH_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalization-hardware-refresh-set.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalization-execution-clock-lineage.v1\0";
const EXECUTION_PERMIT_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalization-execution.v1\0";

/// Pair one exact hardware capability admitted by the authorized context with a fresh temporal
/// requalification of the exact same signed statement under the later execution interval.
pub struct LineageBoundFinalizationHardwareRefreshInputV1<'a> {
    pub authorized: &'a LineageBoundHardwareReauthorizationV1,
    pub fresh: &'a LineageBoundHardwareReauthorizationV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundFinalizationExecutionPermitIdV1(Sha256Digest);

impl LineageBoundFinalizationExecutionPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundFinalizationExecutionPermitV1 {
    id: LineageBoundFinalizationExecutionPermitIdV1,
    authorization_id: AuthorizedLineageBoundFinalizationIdV1,
    context_id: LineageBoundUpgradeFinalizationContextIdV1,
    predecessor_root_digest: Sha256Digest,
    predecessor_current_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    authorized_governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    fresh_governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    authorized_registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    fresh_registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    authorized_retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    fresh_retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    authorized_no_rollback_id: LineageBoundCurrentNoRollbackIdV1,
    authorized_checkpoint_digest: Sha256Digest,
    fresh_checkpoint_digest: Sha256Digest,
    authorized_transparency_log_digest: Sha256Digest,
    fresh_transparency_log_digest: Sha256Digest,
    authorized_log_size: usize,
    fresh_log_size: usize,
    appended_entry_count: usize,
    registry_digest: Sha256Digest,
    registry_sequence: u64,
    trust_snapshot_digest: Sha256Digest,
    trust_snapshot_sequence: u64,
    containment_authority_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    retention_authority_id: String,
    retention_policy_digest: Sha256Digest,
    retention_policy_sequence: u64,
    operational_state_digest: Sha256Digest,
    operational_state_generation: u64,
    operational_lineage_digest: Sha256Digest,
    authorized_hardware_set_digest: Sha256Digest,
    hardware_refresh_set_digest: Sha256Digest,
    hardware_authority_count: usize,
    machine_ids: Vec<String>,
    authorized_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    fresh_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    authorized_operational_basis_id: OperationalClockBasisIdV1,
    fresh_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
    finalization_deadline_unix_ms: u64,
    probation_clearance_expires_at_unix_ms: u64,
    earliest_hardware_expiry_unix_ms: u64,
}

impl LineageBoundFinalizationExecutionPermitV1 {
    pub fn id(&self) -> LineageBoundFinalizationExecutionPermitIdV1 { self.id }
    pub fn authorization_id(&self) -> AuthorizedLineageBoundFinalizationIdV1 { self.authorization_id }
    pub fn context_id(&self) -> LineageBoundUpgradeFinalizationContextIdV1 { self.context_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn predecessor_current_head_digest(&self) -> Sha256Digest { self.predecessor_current_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn upgrade_cycle_sequence(&self) -> u64 { self.upgrade_cycle_sequence }
    pub fn fresh_governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.fresh_governance_view_id }
    pub fn fresh_registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 { self.fresh_registry_head_id }
    pub fn fresh_retention_head_id(&self) -> CurrentEvidenceRetentionHeadIdV1 { self.fresh_retention_head_id }
    pub fn authorized_no_rollback_id(&self) -> LineageBoundCurrentNoRollbackIdV1 { self.authorized_no_rollback_id }
    pub fn fresh_checkpoint_digest(&self) -> Sha256Digest { self.fresh_checkpoint_digest }
    pub fn fresh_transparency_log_digest(&self) -> Sha256Digest { self.fresh_transparency_log_digest }
    pub fn appended_entry_count(&self) -> usize { self.appended_entry_count }
    pub fn operational_state_digest(&self) -> Sha256Digest { self.operational_state_digest }
    pub fn operational_state_generation(&self) -> u64 { self.operational_state_generation }
    pub fn operational_lineage_digest(&self) -> Sha256Digest { self.operational_lineage_digest }
    pub fn authorized_hardware_set_digest(&self) -> Sha256Digest { self.authorized_hardware_set_digest }
    pub fn hardware_refresh_set_digest(&self) -> Sha256Digest { self.hardware_refresh_set_digest }
    pub fn hardware_authority_count(&self) -> usize { self.hardware_authority_count }
    pub fn machine_ids(&self) -> &[String] { &self.machine_ids }
    pub fn fresh_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.fresh_clock_envelope_id }
    pub fn fresh_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.fresh_operational_basis_id }
    pub fn clock_lineage_digest(&self) -> Sha256Digest { self.clock_lineage_digest }
    pub fn clock_hop_count(&self) -> usize { self.clock_hop_count }
    pub fn finalization_deadline_unix_ms(&self) -> u64 { self.finalization_deadline_unix_ms }
    pub fn probation_clearance_expires_at_unix_ms(&self) -> u64 { self.probation_clearance_expires_at_unix_ms }
    pub fn earliest_hardware_expiry_unix_ms(&self) -> u64 { self.earliest_hardware_expiry_unix_ms }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundFinalizationExecutionError {
    AuthorizedContextMismatch,
    AuthorizedRegistryMismatch,
    AuthorizedRetentionMismatch,
    AuthorizedNoRollbackMismatch,
    AuthorizedBasisMismatch,
    AuthorizedEnvelopeMismatch,
    AuthorizedLogInvalid(String),
    AuthorizedLogMismatch,
    FreshRegistryMismatch,
    FreshRetentionMismatch,
    FreshBasisMismatch,
    FreshEnvelopeMismatch,
    FreshLogInvalid(String),
    FreshLogMismatch,
    RegistrySemanticsChanged,
    TrustSnapshotChanged,
    ContainmentSemanticsChanged,
    RetentionSemanticsChanged,
    FreshCheckpointNotDifferent,
    TransparencyLogNotStrictExtension,
    OperationalHeadAppended { entry_sequence: u64 },
    FinalizedHeadAlreadyPublished { entry_sequence: u64 },
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage { hop: usize, expected_predecessor: String, actual_predecessor: Option<String> },
    Clock(ClockGovernanceTimeError),
    ExecutionClockNotDefinitelyLater { authorized_upper_unix_ms: u64, fresh_lower_unix_ms: u64 },
    FinalizationMayBeClosed,
    ProbationMayBeExpired,
    HardwareInputCountMismatch { expected: usize, actual: usize },
    TooManyHardwareAuthorities { actual: usize, maximum: usize },
    DuplicateAuthorizedHardwareMachine(String),
    DuplicateFreshHardwareMachine(String),
    DuplicateAuthorizedHardwareAuthority(String),
    DuplicateFreshHardwareAuthority(String),
    AuthorizedHardwareSetMismatch,
    HardwareMachineMismatch(String),
    AuthorizedHardwareLineageMismatch(String),
    FreshHardwareLineageMismatch(String),
    HardwareStatementChanged(String),
    HardwareSignedEvidenceChanged(String),
    HardwarePolicyChanged(String),
    HardwareVerifierSetChanged(String),
    HardwareTrustChanged(String),
    HardwareContainmentChanged(String),
    AuthorizedHardwareClockMismatch(String),
    FreshHardwareClockMismatch(String),
    HardwareStatementMayBeFuture(String),
    HardwareMayExpire(String),
    EarliestHardwareExpiryMismatch,
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct HardwareRefreshCommitment {
    machine_id: String,
    authorized_authority_id: String,
    fresh_authority_id: String,
    reauthorization_sequence: u64,
    statement_digest: String,
    signed_evidence_digest: String,
    hardware_policy_digest: String,
    verifier_set_digest: String,
    hardware_identity_digest: String,
    machine_profile_digest: String,
    firmware_digest: String,
    calibration_digest: String,
    capability_digest: String,
    expires_at_unix_s: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ClockLineageCommitment {
    authorized_basis_id: String,
    bridge_basis_ids: Vec<String>,
    fresh_basis_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct ExecutionPermitCommitment {
    schema: &'static str,
    authorization_id: String,
    context_id: String,
    predecessor_root_digest: String,
    predecessor_current_head_digest: String,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    authorized_governance_view_id: String,
    fresh_governance_view_id: String,
    authorized_registry_head_id: String,
    fresh_registry_head_id: String,
    authorized_retention_head_id: String,
    fresh_retention_head_id: String,
    authorized_no_rollback_id: String,
    authorized_checkpoint_digest: String,
    fresh_checkpoint_digest: String,
    authorized_transparency_log_digest: String,
    fresh_transparency_log_digest: String,
    authorized_log_size: usize,
    fresh_log_size: usize,
    appended_entry_count: usize,
    registry_digest: String,
    registry_sequence: u64,
    trust_snapshot_digest: String,
    trust_snapshot_sequence: u64,
    containment_authority_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    containment_generation: u64,
    retention_authority_id: String,
    retention_policy_digest: String,
    retention_policy_sequence: u64,
    operational_state_digest: String,
    operational_state_generation: u64,
    operational_lineage_digest: String,
    authorized_hardware_set_digest: String,
    hardware_refresh_set_digest: String,
    hardware_authority_count: usize,
    machine_ids: Vec<String>,
    authorized_clock_envelope_id: String,
    fresh_clock_envelope_id: String,
    authorized_operational_basis_id: String,
    fresh_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
    finalization_deadline_unix_ms: u64,
    probation_clearance_expires_at_unix_ms: u64,
    earliest_hardware_expiry_unix_ms: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn derive_lineage_bound_finalization_execution_permit_v1(
    authorization: &AuthorizedLineageBoundFinalizationV1,
    authorized_governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    authorized_registry_head: &QuorumObservedWitnessRegistryHeadV1,
    authorized_retention_head: &CurrentEvidenceRetentionHeadV1,
    authorized_no_rollback: &LineageBoundCurrentNoRollbackV1,
    authorized_log: &TransparencyLog,
    authorized_basis: &OperationalClockBasisV1,
    fresh_governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    fresh_registry_head: &QuorumObservedWitnessRegistryHeadV1,
    fresh_retention_head: &CurrentEvidenceRetentionHeadV1,
    fresh_log: &TransparencyLog,
    authorized_to_fresh_clock_bridge: &[OperationalClockBasisV1],
    fresh_basis: &OperationalClockBasisV1,
    hardware_refreshes: &[LineageBoundFinalizationHardwareRefreshInputV1<'_>],
) -> Result<LineageBoundFinalizationExecutionPermitV1, Vec<LineageBoundFinalizationExecutionError>> {
    let mut violations = Vec::new();
    let context = authorization.context();

    if authorized_governance_view.id() != context.governance_view_id()
        || authorized_registry_head.id() != context.registry_head_id()
        || authorized_retention_head.id() != context.retention_head_id()
        || authorized_no_rollback.id() != context.no_rollback_id()
    {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedContextMismatch);
    }
    if authorized_governance_view.registry_head_id() != authorized_registry_head.id()
        || authorized_governance_view.registry_digest() != authorized_registry_head.registry_digest()
        || authorized_governance_view.registry_sequence() != authorized_registry_head.sequence()
        || authorized_registry_head.trust_snapshot_digest() != context.current_trust_snapshot_digest()
        || authorized_registry_head.trust_snapshot_sequence() != context.current_trust_snapshot_sequence()
        || authorized_governance_view.containment_state_digest() != context.current_containment_state_digest()
        || authorized_governance_view.compromise_tracker_digest() != context.current_compromise_tracker_digest()
        || authorized_governance_view.containment_generation() != context.current_containment_generation()
    {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedRegistryMismatch);
    }
    if authorized_retention_head.governance_view_id() != authorized_governance_view.id()
        || authorized_retention_head.registry_head_id() != authorized_registry_head.id()
        || authorized_retention_head.governance_checkpoint_digest() != context.current_checkpoint_digest()
        || authorized_retention_head.transparency_log_digest() != context.current_transparency_log_digest()
    {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedRetentionMismatch);
    }
    if authorized_no_rollback.governance_view_id() != authorized_governance_view.id()
        || authorized_no_rollback.registry_head_id() != authorized_registry_head.id()
        || authorized_no_rollback.retention_head_id() != authorized_retention_head.id()
        || authorized_no_rollback.state_digest() != context.operational_state_digest()
        || authorized_no_rollback.state_generation() != context.operational_state_generation()
        || authorized_no_rollback.operational_lineage_digest() != context.operational_lineage_digest()
        || authorized_no_rollback.hardware_authority_set_digest() != context.hardware_authority_set_digest()
    {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedNoRollbackMismatch);
    }

    if authorized_basis.id() != context.current_operational_basis_id()
        || authorized_basis.id() != authorized_governance_view.observation_operational_basis_id()
        || authorized_basis.id() != authorized_registry_head.observation_operational_basis_id()
        || authorized_basis.id() != authorized_retention_head.observation_operational_basis_id()
        || authorized_basis.id() != authorized_no_rollback.observation_operational_basis_id()
    {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedBasisMismatch);
    }
    let authorized_clock = match derive_clock_governance_evaluation_envelope_v1(authorized_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationExecutionError::Clock(error));
            return Err(violations);
        }
    };
    if authorized_clock.id() != context.current_clock_envelope_id()
        || authorized_clock.id() != authorized_governance_view.observation_clock_envelope_id()
        || authorized_clock.id() != authorized_registry_head.observation_clock_envelope_id()
        || authorized_clock.id() != authorized_retention_head.observation_clock_envelope_id()
        || authorized_clock.id() != authorized_no_rollback.observation_clock_envelope_id()
    {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedEnvelopeMismatch);
    }

    if let Err(error) = authorized_log.validate() {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedLogInvalid(format!("{error:?}")));
    }
    let authorized_log_digest = match digest_transparency_log(authorized_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationExecutionError::AuthorizedLogInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    if authorized_log_digest != context.current_transparency_log_digest()
        || authorized_log_digest != authorized_governance_view.transparency_log_digest()
        || authorized_log_digest != authorized_registry_head.transparency_log_digest()
        || authorized_log_digest != authorized_retention_head.transparency_log_digest()
        || authorized_log_digest != authorized_no_rollback.transparency_log_digest()
    {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedLogMismatch);
    }

    if fresh_governance_view.registry_head_id() != fresh_registry_head.id()
        || fresh_governance_view.registry_digest() != fresh_registry_head.registry_digest()
        || fresh_governance_view.registry_sequence() != fresh_registry_head.sequence()
        || fresh_retention_head.governance_view_id() != fresh_governance_view.id()
        || fresh_retention_head.registry_head_id() != fresh_registry_head.id()
        || fresh_retention_head.governance_checkpoint_digest() != fresh_governance_view.checkpoint_digest()
    {
        violations.push(LineageBoundFinalizationExecutionError::FreshRegistryMismatch);
    }
    if fresh_registry_head.activated_registry_id() != authorized_registry_head.activated_registry_id()
        || fresh_registry_head.registry_digest() != authorized_registry_head.registry_digest()
        || fresh_registry_head.sequence() != authorized_registry_head.sequence()
    {
        violations.push(LineageBoundFinalizationExecutionError::RegistrySemanticsChanged);
    }
    if fresh_registry_head.trust_snapshot_digest() != authorized_registry_head.trust_snapshot_digest()
        || fresh_registry_head.trust_snapshot_sequence() != authorized_registry_head.trust_snapshot_sequence()
    {
        violations.push(LineageBoundFinalizationExecutionError::TrustSnapshotChanged);
    }
    if fresh_governance_view.containment_authority_id() != authorized_governance_view.containment_authority_id()
        || fresh_governance_view.containment_state_digest() != authorized_governance_view.containment_state_digest()
        || fresh_governance_view.compromise_tracker_digest() != authorized_governance_view.compromise_tracker_digest()
        || fresh_governance_view.containment_generation() != authorized_governance_view.containment_generation()
    {
        violations.push(LineageBoundFinalizationExecutionError::ContainmentSemanticsChanged);
    }
    if fresh_retention_head.retention_authority_id() != authorized_retention_head.retention_authority_id()
        || fresh_retention_head.policy_digest() != authorized_retention_head.policy_digest()
        || fresh_retention_head.sequence() != authorized_retention_head.sequence()
        || fresh_retention_head.effective_at_unix_s() != authorized_retention_head.effective_at_unix_s()
    {
        violations.push(LineageBoundFinalizationExecutionError::RetentionSemanticsChanged);
    }
    if fresh_retention_head.containment_authority_id() != fresh_governance_view.containment_authority_id()
        || fresh_retention_head.containment_state_digest() != fresh_governance_view.containment_state_digest()
        || fresh_retention_head.compromise_tracker_digest() != fresh_governance_view.compromise_tracker_digest()
        || fresh_retention_head.containment_generation() != fresh_governance_view.containment_generation()
        || fresh_retention_head.trust_snapshot_digest() != fresh_registry_head.trust_snapshot_digest()
    {
        violations.push(LineageBoundFinalizationExecutionError::FreshRetentionMismatch);
    }

    if fresh_basis.id() != fresh_governance_view.observation_operational_basis_id()
        || fresh_basis.id() != fresh_registry_head.observation_operational_basis_id()
        || fresh_basis.id() != fresh_retention_head.observation_operational_basis_id()
    {
        violations.push(LineageBoundFinalizationExecutionError::FreshBasisMismatch);
    }
    let fresh_clock = match derive_clock_governance_evaluation_envelope_v1(fresh_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationExecutionError::Clock(error));
            return Err(violations);
        }
    };
    if fresh_clock.id() != fresh_governance_view.observation_clock_envelope_id()
        || fresh_clock.id() != fresh_registry_head.observation_clock_envelope_id()
        || fresh_clock.id() != fresh_retention_head.observation_clock_envelope_id()
    {
        violations.push(LineageBoundFinalizationExecutionError::FreshEnvelopeMismatch);
    }

    if let Err(error) = fresh_log.validate() {
        violations.push(LineageBoundFinalizationExecutionError::FreshLogInvalid(format!("{error:?}")));
    }
    let fresh_log_digest = match digest_transparency_log(fresh_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationExecutionError::FreshLogInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    if fresh_log_digest != fresh_governance_view.transparency_log_digest()
        || fresh_log_digest != fresh_registry_head.transparency_log_digest()
        || fresh_log_digest != fresh_retention_head.transparency_log_digest()
    {
        violations.push(LineageBoundFinalizationExecutionError::FreshLogMismatch);
    }
    if fresh_governance_view.checkpoint_digest() == context.current_checkpoint_digest()
        || fresh_registry_head.checkpoint_digest() == context.current_checkpoint_digest()
    {
        violations.push(LineageBoundFinalizationExecutionError::FreshCheckpointNotDifferent);
    }
    if fresh_log.entries.len() <= authorized_log.entries.len()
        || fresh_log.verify_successor_of(authorized_log).is_err()
    {
        violations.push(LineageBoundFinalizationExecutionError::TransparencyLogNotStrictExtension);
    }

    if authorized_to_fresh_clock_bridge.len() > MAX_LINEAGE_BOUND_FINALIZATION_EXECUTION_CLOCK_HOPS {
        violations.push(LineageBoundFinalizationExecutionError::TooManyClockHops {
            actual: authorized_to_fresh_clock_bridge.len(),
            maximum: MAX_LINEAGE_BOUND_FINALIZATION_EXECUTION_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        authorized_basis.id(),
        authorized_to_fresh_clock_bridge,
        fresh_basis,
    ) {
        violations.push(error);
    }
    if fresh_clock.lower_unix_ms() <= authorized_clock.upper_unix_ms() {
        violations.push(LineageBoundFinalizationExecutionError::ExecutionClockNotDefinitelyLater {
            authorized_upper_unix_ms: authorized_clock.upper_unix_ms(),
            fresh_lower_unix_ms: fresh_clock.lower_unix_ms(),
        });
    }
    if fresh_clock.upper_unix_ms() >= context.finalization_deadline_unix_ms() {
        violations.push(LineageBoundFinalizationExecutionError::FinalizationMayBeClosed);
    }
    if fresh_clock.upper_unix_ms() >= context.probation_clearance_expires_at_unix_ms() {
        violations.push(LineageBoundFinalizationExecutionError::ProbationMayBeExpired);
    }

    if fresh_log.entries.len() > authorized_log.entries.len() {
        let operational_kind = match upgrade_operational_head_log_kind(context.handoff_plan_digest()) {
            Ok(value) => value,
            Err(error) => {
                violations.push(LineageBoundFinalizationExecutionError::Encoding(format!("{error:?}")));
                String::new()
            }
        };
        if !operational_kind.is_empty() {
            for entry in &fresh_log.entries[authorized_log.entries.len()..] {
                if entry.kind == operational_kind {
                    violations.push(LineageBoundFinalizationExecutionError::OperationalHeadAppended {
                        entry_sequence: entry.sequence,
                    });
                }
            }
        }
    }
    let finalized_kind = match finalized_upgrade_head_log_kind(context.handoff_plan_digest()) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationExecutionError::Encoding(format!("{error:?}")));
            String::new()
        }
    };
    if !finalized_kind.is_empty() {
        for entry in &fresh_log.entries {
            if entry.kind == finalized_kind {
                violations.push(LineageBoundFinalizationExecutionError::FinalizedHeadAlreadyPublished {
                    entry_sequence: entry.sequence,
                });
            }
        }
    }

    if hardware_refreshes.len() != context.hardware_authority_count() {
        violations.push(LineageBoundFinalizationExecutionError::HardwareInputCountMismatch {
            expected: context.hardware_authority_count(), actual: hardware_refreshes.len(),
        });
    }
    if hardware_refreshes.len() > MAX_LINEAGE_BOUND_FINALIZATION_EXECUTION_HARDWARE {
        violations.push(LineageBoundFinalizationExecutionError::TooManyHardwareAuthorities {
            actual: hardware_refreshes.len(), maximum: MAX_LINEAGE_BOUND_FINALIZATION_EXECUTION_HARDWARE,
        });
        return Err(violations);
    }

    let mut seen_authorized_machines = BTreeSet::new();
    let mut seen_fresh_machines = BTreeSet::new();
    let mut seen_authorized_ids = BTreeSet::new();
    let mut seen_fresh_ids = BTreeSet::new();
    let mut authorized_ids = Vec::with_capacity(hardware_refreshes.len());
    let mut machine_ids = Vec::with_capacity(hardware_refreshes.len());
    let mut refresh_commitments = Vec::with_capacity(hardware_refreshes.len());
    let mut earliest_hardware_expiry_unix_ms = u64::MAX;

    for refresh in hardware_refreshes {
        let authorized = refresh.authorized;
        let fresh = refresh.fresh;
        let machine_id = authorized.statement().machine_id.clone();
        if !seen_authorized_machines.insert(machine_id.clone()) {
            violations.push(LineageBoundFinalizationExecutionError::DuplicateAuthorizedHardwareMachine(machine_id.clone()));
        }
        if !seen_fresh_machines.insert(fresh.statement().machine_id.clone()) {
            violations.push(LineageBoundFinalizationExecutionError::DuplicateFreshHardwareMachine(fresh.statement().machine_id.clone()));
        }
        if !seen_authorized_ids.insert(authorized.id().to_hex()) {
            violations.push(LineageBoundFinalizationExecutionError::DuplicateAuthorizedHardwareAuthority(authorized.id().to_hex()));
        }
        if !seen_fresh_ids.insert(fresh.id().to_hex()) {
            violations.push(LineageBoundFinalizationExecutionError::DuplicateFreshHardwareAuthority(fresh.id().to_hex()));
        }
        if fresh.statement().machine_id != machine_id {
            violations.push(LineageBoundFinalizationExecutionError::HardwareMachineMismatch(machine_id.clone()));
        }

        if authorized.predecessor_root_digest() != context.predecessor_root_digest()
            || authorized.current_head_digest() != context.predecessor_current_head_digest()
            || authorized.predecessor_finalization_sequence() != context.predecessor_finalization_sequence()
            || authorized.handoff_plan_digest() != context.handoff_plan_digest()
            || authorized.trust_snapshot_digest() != context.current_trust_snapshot_digest()
            || authorized.containment_state_digest() != context.current_containment_state_digest()
            || authorized.compromise_tracker_digest() != context.current_compromise_tracker_digest()
        {
            violations.push(LineageBoundFinalizationExecutionError::AuthorizedHardwareLineageMismatch(machine_id.clone()));
        }
        if fresh.lineage_handoff_id() != authorized.lineage_handoff_id()
            || fresh.probation_clearance_id() != authorized.probation_clearance_id()
            || fresh.telemetry_clearance_id() != authorized.telemetry_clearance_id()
            || fresh.predecessor_root_digest() != authorized.predecessor_root_digest()
            || fresh.current_head_digest() != authorized.current_head_digest()
            || fresh.governance_view_digest() != authorized.governance_view_digest()
            || fresh.registry_head_digest() != authorized.registry_head_digest()
            || fresh.predecessor_finalization_sequence() != authorized.predecessor_finalization_sequence()
            || fresh.handoff_plan_digest() != authorized.handoff_plan_digest()
        {
            violations.push(LineageBoundFinalizationExecutionError::FreshHardwareLineageMismatch(machine_id.clone()));
        }
        if fresh.statement() != authorized.statement() || fresh.statement_digest() != authorized.statement_digest() {
            violations.push(LineageBoundFinalizationExecutionError::HardwareStatementChanged(machine_id.clone()));
        }
        if fresh.signed_evidence_digest() != authorized.signed_evidence_digest() {
            violations.push(LineageBoundFinalizationExecutionError::HardwareSignedEvidenceChanged(machine_id.clone()));
        }
        if fresh.hardware_policy_digest() != authorized.hardware_policy_digest() {
            violations.push(LineageBoundFinalizationExecutionError::HardwarePolicyChanged(machine_id.clone()));
        }
        if fresh.verifier_set_digest() != authorized.verifier_set_digest() {
            violations.push(LineageBoundFinalizationExecutionError::HardwareVerifierSetChanged(machine_id.clone()));
        }
        if fresh.trust_snapshot_digest() != authorized.trust_snapshot_digest()
            || fresh.trust_snapshot_digest() != fresh_registry_head.trust_snapshot_digest()
        {
            violations.push(LineageBoundFinalizationExecutionError::HardwareTrustChanged(machine_id.clone()));
        }
        if fresh.containment_state_digest() != authorized.containment_state_digest()
            || fresh.compromise_tracker_digest() != authorized.compromise_tracker_digest()
            || fresh.containment_state_digest() != fresh_governance_view.containment_state_digest()
            || fresh.compromise_tracker_digest() != fresh_governance_view.compromise_tracker_digest()
        {
            violations.push(LineageBoundFinalizationExecutionError::HardwareContainmentChanged(machine_id.clone()));
        }
        if authorized.current_operational_basis_id() != authorized_basis.id()
            || authorized.current_clock_envelope_id() != authorized_clock.id()
        {
            violations.push(LineageBoundFinalizationExecutionError::AuthorizedHardwareClockMismatch(machine_id.clone()));
        }
        if fresh.current_operational_basis_id() != fresh_basis.id()
            || fresh.current_clock_envelope_id() != fresh_clock.id()
        {
            violations.push(LineageBoundFinalizationExecutionError::FreshHardwareClockMismatch(machine_id.clone()));
        }

        let issued_at_unix_ms = match seconds_to_millis(fresh.statement().issued_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        let expires_at_unix_ms = match seconds_to_millis(fresh.statement().expires_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if issued_at_unix_ms > fresh_clock.lower_unix_ms() {
            violations.push(LineageBoundFinalizationExecutionError::HardwareStatementMayBeFuture(machine_id.clone()));
        }
        if expires_at_unix_ms <= fresh_clock.upper_unix_ms() {
            violations.push(LineageBoundFinalizationExecutionError::HardwareMayExpire(machine_id.clone()));
        }
        earliest_hardware_expiry_unix_ms = earliest_hardware_expiry_unix_ms.min(expires_at_unix_ms);

        authorized_ids.push(authorized.id().to_hex());
        machine_ids.push(machine_id.clone());
        refresh_commitments.push(HardwareRefreshCommitment {
            machine_id,
            authorized_authority_id: authorized.id().to_hex(),
            fresh_authority_id: fresh.id().to_hex(),
            reauthorization_sequence: authorized.statement().reauthorization_sequence,
            statement_digest: authorized.statement_digest().to_hex(),
            signed_evidence_digest: authorized.signed_evidence_digest().to_hex(),
            hardware_policy_digest: authorized.hardware_policy_digest().to_hex(),
            verifier_set_digest: authorized.verifier_set_digest().to_hex(),
            hardware_identity_digest: authorized.statement().hardware_identity_digest.to_hex(),
            machine_profile_digest: authorized.statement().machine_profile_digest.to_hex(),
            firmware_digest: authorized.statement().firmware_digest.to_hex(),
            calibration_digest: authorized.statement().calibration_digest.to_hex(),
            capability_digest: authorized.statement().capability_digest.to_hex(),
            expires_at_unix_s: authorized.statement().expires_at_unix_s,
        });
    }

    authorized_ids.sort();
    let mut expected_authorized_ids = context.hardware_authority_ids().to_vec();
    expected_authorized_ids.sort();
    machine_ids.sort();
    let mut expected_machine_ids = context.machine_ids().to_vec();
    expected_machine_ids.sort();
    if authorized_ids != expected_authorized_ids || machine_ids != expected_machine_ids {
        violations.push(LineageBoundFinalizationExecutionError::AuthorizedHardwareSetMismatch);
    }
    if earliest_hardware_expiry_unix_ms != context.earliest_hardware_expiry_unix_ms() {
        violations.push(LineageBoundFinalizationExecutionError::EarliestHardwareExpiryMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    refresh_commitments.sort_by(|left, right| {
        left.machine_id
            .cmp(&right.machine_id)
            .then(left.authorized_authority_id.cmp(&right.authorized_authority_id))
            .then(left.fresh_authority_id.cmp(&right.fresh_authority_id))
    });
    let hardware_refresh_set_digest =
        hash_serializable(HARDWARE_REFRESH_SET_DOMAIN, &refresh_commitments).map_err(|error| vec![error])?;
    let (clock_lineage_digest, clock_hop_count) = digest_clock_lineage(
        authorized_basis,
        authorized_to_fresh_clock_bridge,
        fresh_basis,
    )
    .map_err(|error| vec![error])?;
    let appended_entry_count = fresh_log.entries.len() - authorized_log.entries.len();

    let commitment = ExecutionPermitCommitment {
        schema: LINEAGE_BOUND_FINALIZATION_EXECUTION_SCHEMA,
        authorization_id: authorization.id().to_hex(),
        context_id: context.id().to_hex(),
        predecessor_root_digest: context.predecessor_root_digest().to_hex(),
        predecessor_current_head_digest: context.predecessor_current_head_digest().to_hex(),
        predecessor_finalization_sequence: context.predecessor_finalization_sequence(),
        upgrade_cycle_sequence: context.upgrade_cycle_sequence(),
        authorized_governance_view_id: authorized_governance_view.id().to_hex(),
        fresh_governance_view_id: fresh_governance_view.id().to_hex(),
        authorized_registry_head_id: authorized_registry_head.id().to_hex(),
        fresh_registry_head_id: fresh_registry_head.id().to_hex(),
        authorized_retention_head_id: authorized_retention_head.id().to_hex(),
        fresh_retention_head_id: fresh_retention_head.id().to_hex(),
        authorized_no_rollback_id: authorized_no_rollback.id().to_hex(),
        authorized_checkpoint_digest: context.current_checkpoint_digest().to_hex(),
        fresh_checkpoint_digest: fresh_governance_view.checkpoint_digest().to_hex(),
        authorized_transparency_log_digest: authorized_log_digest.to_hex(),
        fresh_transparency_log_digest: fresh_log_digest.to_hex(),
        authorized_log_size: authorized_log.entries.len(),
        fresh_log_size: fresh_log.entries.len(),
        appended_entry_count,
        registry_digest: authorized_registry_head.registry_digest().to_hex(),
        registry_sequence: authorized_registry_head.sequence(),
        trust_snapshot_digest: authorized_registry_head.trust_snapshot_digest().to_hex(),
        trust_snapshot_sequence: authorized_registry_head.trust_snapshot_sequence(),
        containment_authority_digest: authorized_governance_view.containment_authority_id().as_digest().to_hex(),
        containment_state_digest: authorized_governance_view.containment_state_digest().to_hex(),
        compromise_tracker_digest: authorized_governance_view.compromise_tracker_digest().to_hex(),
        containment_generation: authorized_governance_view.containment_generation(),
        retention_authority_id: authorized_retention_head.retention_authority_id().to_hex(),
        retention_policy_digest: authorized_retention_head.policy_digest().to_hex(),
        retention_policy_sequence: authorized_retention_head.sequence(),
        operational_state_digest: context.operational_state_digest().to_hex(),
        operational_state_generation: context.operational_state_generation(),
        operational_lineage_digest: context.operational_lineage_digest().to_hex(),
        authorized_hardware_set_digest: context.hardware_authority_set_digest().to_hex(),
        hardware_refresh_set_digest: hardware_refresh_set_digest.to_hex(),
        hardware_authority_count: refresh_commitments.len(),
        machine_ids: machine_ids.clone(),
        authorized_clock_envelope_id: authorized_clock.id().to_hex(),
        fresh_clock_envelope_id: fresh_clock.id().to_hex(),
        authorized_operational_basis_id: authorized_basis.id().to_hex(),
        fresh_operational_basis_id: fresh_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count,
        finalization_deadline_unix_ms: context.finalization_deadline_unix_ms(),
        probation_clearance_expires_at_unix_ms: context.probation_clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    };
    let id = LineageBoundFinalizationExecutionPermitIdV1(
        hash_serializable(EXECUTION_PERMIT_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(LineageBoundFinalizationExecutionPermitV1 {
        id,
        authorization_id: authorization.id(),
        context_id: context.id(),
        predecessor_root_digest: context.predecessor_root_digest(),
        predecessor_current_head_digest: context.predecessor_current_head_digest(),
        predecessor_finalization_sequence: context.predecessor_finalization_sequence(),
        upgrade_cycle_sequence: context.upgrade_cycle_sequence(),
        authorized_governance_view_id: authorized_governance_view.id(),
        fresh_governance_view_id: fresh_governance_view.id(),
        authorized_registry_head_id: authorized_registry_head.id(),
        fresh_registry_head_id: fresh_registry_head.id(),
        authorized_retention_head_id: authorized_retention_head.id(),
        fresh_retention_head_id: fresh_retention_head.id(),
        authorized_no_rollback_id: authorized_no_rollback.id(),
        authorized_checkpoint_digest: context.current_checkpoint_digest(),
        fresh_checkpoint_digest: fresh_governance_view.checkpoint_digest(),
        authorized_transparency_log_digest: authorized_log_digest,
        fresh_transparency_log_digest: fresh_log_digest,
        authorized_log_size: authorized_log.entries.len(),
        fresh_log_size: fresh_log.entries.len(),
        appended_entry_count,
        registry_digest: authorized_registry_head.registry_digest(),
        registry_sequence: authorized_registry_head.sequence(),
        trust_snapshot_digest: authorized_registry_head.trust_snapshot_digest(),
        trust_snapshot_sequence: authorized_registry_head.trust_snapshot_sequence(),
        containment_authority_digest: authorized_governance_view.containment_authority_id().as_digest(),
        containment_state_digest: authorized_governance_view.containment_state_digest(),
        compromise_tracker_digest: authorized_governance_view.compromise_tracker_digest(),
        containment_generation: authorized_governance_view.containment_generation(),
        retention_authority_id: authorized_retention_head.retention_authority_id().to_hex(),
        retention_policy_digest: authorized_retention_head.policy_digest(),
        retention_policy_sequence: authorized_retention_head.sequence(),
        operational_state_digest: context.operational_state_digest(),
        operational_state_generation: context.operational_state_generation(),
        operational_lineage_digest: context.operational_lineage_digest(),
        authorized_hardware_set_digest: context.hardware_authority_set_digest(),
        hardware_refresh_set_digest,
        hardware_authority_count: refresh_commitments.len(),
        machine_ids,
        authorized_clock_envelope_id: authorized_clock.id(),
        fresh_clock_envelope_id: fresh_clock.id(),
        authorized_operational_basis_id: authorized_basis.id(),
        fresh_operational_basis_id: fresh_basis.id(),
        clock_lineage_digest,
        clock_hop_count,
        finalization_deadline_unix_ms: context.finalization_deadline_unix_ms(),
        probation_clearance_expires_at_unix_ms: context.probation_clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    })
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), LineageBoundFinalizationExecutionError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(LineageBoundFinalizationExecutionError::BrokenClockLineage {
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
            return Err(LineageBoundFinalizationExecutionError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(LineageBoundFinalizationExecutionError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn digest_clock_lineage(
    authorized_basis: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    fresh_basis: &OperationalClockBasisV1,
) -> Result<(Sha256Digest, usize), LineageBoundFinalizationExecutionError> {
    let mut ids = Vec::with_capacity(bridge.len() + 2);
    ids.push(authorized_basis.id().to_hex());
    ids.extend(bridge.iter().map(|basis| basis.id().to_hex()));
    if fresh_basis.id() != authorized_basis.id() {
        ids.push(fresh_basis.id().to_hex());
    }
    let digest = hash_serializable(CLOCK_LINEAGE_DOMAIN, &ids)?;
    let hops = if fresh_basis.id() == authorized_basis.id() { 0 } else { bridge.len() + 1 };
    Ok((digest, hops))
}

fn seconds_to_millis(value: u64) -> Result<u64, LineageBoundFinalizationExecutionError> {
    value
        .checked_mul(1_000)
        .ok_or(LineageBoundFinalizationExecutionError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundFinalizationExecutionError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundFinalizationExecutionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
