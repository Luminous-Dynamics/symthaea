// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh execution permission for threshold-authorized irreversible upgrade finalization.
//!
//! Finalization authorization is intentionally not execution authority. This crate requires a
//! strictly newer authenticated transparency/checkpoint view, a definitely later descendant
//! operational-clock interval, unchanged registry/trust/containment/retention/operational semantics,
//! and fresh requalification of every exact hardware authorization before minting an opaque
//! execution permit. Any semantic governance change forces a new finalization authorization.

#![deny(unsafe_code)]

use serde::Serialize;
use std::collections::BTreeSet;
use symthaea_fabrication_current_no_rollback::{
    CurrentNoRollbackUpgradeAuthorityIdV1, CurrentNoRollbackUpgradeAuthorityV1,
};
use symthaea_fabrication_evidence_retention_head::{
    CurrentEvidenceRetentionHeadIdV1, CurrentEvidenceRetentionHeadV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_upgrade_finalization_authorization::{
    AuthorizedClockGovernedUpgradeFinalizationIdV1,
    AuthorizedClockGovernedUpgradeFinalizationV1,
};
use symthaea_fabrication_upgrade_finalization_context::QualifiedUpgradeFinalizationContextIdV1;
use symthaea_fabrication_upgrade_hardware_authority::ClockGovernedHardwareReauthorizationV1;
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

pub const CLOCK_GOVERNED_UPGRADE_FINALIZATION_EXECUTION_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-upgrade-finalization-execution.v1";
pub const MAX_FINALIZATION_EXECUTION_CLOCK_HOPS: usize = 4096;
pub const MAX_FINALIZATION_EXECUTION_HARDWARE_AUTHORITIES: usize = 65_536;

const AUTHORIZED_HARDWARE_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-finalization-hardware-set.v1\0";
const HARDWARE_REFRESH_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-finalization-hardware-refresh-set.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-finalization-execution-clock-lineage.v1\0";
const EXECUTION_PERMIT_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-upgrade-finalization-execution.v1\0";

/// One exact hardware capability admitted by the threshold-authorized context paired with a fresh
/// requalification of the exact same signed hardware statement under the later execution clock.
pub struct FinalizationHardwareRefreshInputV1<'a> {
    pub authorized: &'a ClockGovernedHardwareReauthorizationV1,
    pub fresh: &'a ClockGovernedHardwareReauthorizationV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedUpgradeFinalizationExecutionPermitIdV1(Sha256Digest);

impl ClockGovernedUpgradeFinalizationExecutionPermitIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque permission to execute one exact threshold-authorized finalization against one strictly
/// newer authenticated governance view. This still performs no mutation itself.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedUpgradeFinalizationExecutionPermitV1 {
    id: ClockGovernedUpgradeFinalizationExecutionPermitIdV1,
    authorization_id: AuthorizedClockGovernedUpgradeFinalizationIdV1,
    context_id: QualifiedUpgradeFinalizationContextIdV1,
    authorized_governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    fresh_governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    authorized_registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    fresh_registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    authorized_retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    fresh_retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    authorized_no_rollback_id: CurrentNoRollbackUpgradeAuthorityIdV1,
    fresh_no_rollback_id: CurrentNoRollbackUpgradeAuthorityIdV1,
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
    containment_authority_id_digest: Sha256Digest,
    containment_state_digest: Sha256Digest,
    compromise_tracker_digest: Sha256Digest,
    containment_generation: u64,
    retention_authority_id_digest: Sha256Digest,
    retention_policy_digest: Sha256Digest,
    retention_policy_sequence: u64,
    operational_state_digest: Sha256Digest,
    operational_state_generation: u64,
    operational_lineage_digest: Sha256Digest,
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

impl ClockGovernedUpgradeFinalizationExecutionPermitV1 {
    pub fn id(&self) -> ClockGovernedUpgradeFinalizationExecutionPermitIdV1 {
        self.id
    }

    pub fn authorization_id(&self) -> AuthorizedClockGovernedUpgradeFinalizationIdV1 {
        self.authorization_id
    }

    pub fn context_id(&self) -> QualifiedUpgradeFinalizationContextIdV1 {
        self.context_id
    }

    pub fn fresh_governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 {
        self.fresh_governance_view_id
    }

    pub fn fresh_registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 {
        self.fresh_registry_head_id
    }

    pub fn fresh_retention_head_id(&self) -> CurrentEvidenceRetentionHeadIdV1 {
        self.fresh_retention_head_id
    }

    pub fn fresh_no_rollback_id(&self) -> CurrentNoRollbackUpgradeAuthorityIdV1 {
        self.fresh_no_rollback_id
    }

    pub fn fresh_checkpoint_digest(&self) -> Sha256Digest {
        self.fresh_checkpoint_digest
    }

    pub fn fresh_transparency_log_digest(&self) -> Sha256Digest {
        self.fresh_transparency_log_digest
    }

    pub fn appended_entry_count(&self) -> usize {
        self.appended_entry_count
    }

    pub fn hardware_refresh_set_digest(&self) -> Sha256Digest {
        self.hardware_refresh_set_digest
    }

    pub fn hardware_authority_count(&self) -> usize {
        self.hardware_authority_count
    }

    pub fn machine_ids(&self) -> &[String] {
        &self.machine_ids
    }

    pub fn fresh_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.fresh_clock_envelope_id
    }

    pub fn fresh_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.fresh_operational_basis_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeFinalizationExecutionError {
    AuthorizedContextMismatch,
    AuthorizedRegistryMismatch,
    AuthorizedRetentionMismatch,
    AuthorizedNoRollbackMismatch,
    AuthorizedBasisMismatch,
    AuthorizedEnvelopeMismatch,
    FreshRegistryMismatch,
    FreshRetentionMismatch,
    FreshNoRollbackMismatch,
    FreshBasisMismatch,
    FreshEnvelopeMismatch,
    RegistrySemanticsChanged,
    TrustSnapshotChanged,
    ContainmentSemanticsChanged,
    RetentionSemanticsChanged,
    OperationalSemanticsChanged,
    FreshCheckpointNotDifferent,
    TransparencyLogInvalid { fresh: bool, reason: String },
    TransparencyLogDigestMismatch { fresh: bool },
    TransparencyLogNotStrictExtension,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage {
        hop: usize,
        expected_predecessor: String,
        actual_predecessor: Option<String>,
    },
    Clock(ClockGovernanceTimeError),
    ExecutionClockNotDefinitelyLater {
        authorized_upper_unix_ms: u64,
        fresh_lower_unix_ms: u64,
    },
    FinalizationMayBeClosed,
    ProbationMayBeExpired,
    HardwareInputCountMismatch { expected: usize, actual: usize },
    TooManyHardwareAuthorities { actual: usize, maximum: usize },
    DuplicateAuthorizedHardwareMachine(String),
    DuplicateFreshHardwareMachine(String),
    AuthorizedHardwareSetMismatch,
    HardwareMachineMismatch(String),
    HardwareHandoffMismatch(String),
    HardwareProbationMismatch(String),
    HardwareTelemetryMismatch(String),
    HardwareStatementChanged(String),
    HardwareSignedEvidenceChanged(String),
    HardwarePolicyChanged(String),
    HardwareVerifierSetChanged(String),
    HardwareTrustChanged(String),
    HardwareContainmentChanged(String),
    AuthorizedHardwareClockMismatch(String),
    FreshHardwareClockMismatch(String),
    HardwareMayExpire(String),
    EarliestHardwareExpiryMismatch,
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct AuthorizedHardwareCommitment {
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
    expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct HardwareRefreshCommitment {
    machine_id: String,
    authorized_authority_id: String,
    fresh_authority_id: String,
    statement_digest: String,
    signed_evidence_digest: String,
    hardware_policy_digest: String,
    verifier_set_digest: String,
    hardware_identity_digest: String,
    machine_profile_digest: String,
    firmware_digest: String,
    calibration_digest: String,
    capability_digest: String,
    expires_at_unix_ms: u64,
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
    authorized_governance_view_id: String,
    fresh_governance_view_id: String,
    authorized_registry_head_id: String,
    fresh_registry_head_id: String,
    authorized_retention_head_id: String,
    fresh_retention_head_id: String,
    authorized_no_rollback_id: String,
    fresh_no_rollback_id: String,
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
    containment_authority_id_digest: String,
    containment_state_digest: String,
    compromise_tracker_digest: String,
    containment_generation: u64,
    retention_authority_id_digest: String,
    retention_policy_digest: String,
    retention_policy_sequence: u64,
    operational_state_digest: String,
    operational_state_generation: u64,
    operational_lineage_digest: String,
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
pub fn derive_clock_governed_upgrade_finalization_execution_permit_v1(
    authorization: &AuthorizedClockGovernedUpgradeFinalizationV1,
    authorized_governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    authorized_registry_head: &QuorumObservedWitnessRegistryHeadV1,
    authorized_retention_head: &CurrentEvidenceRetentionHeadV1,
    authorized_no_rollback: &CurrentNoRollbackUpgradeAuthorityV1,
    authorized_log: &TransparencyLog,
    authorized_basis: &OperationalClockBasisV1,
    fresh_governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    fresh_registry_head: &QuorumObservedWitnessRegistryHeadV1,
    fresh_retention_head: &CurrentEvidenceRetentionHeadV1,
    fresh_no_rollback: &CurrentNoRollbackUpgradeAuthorityV1,
    fresh_log: &TransparencyLog,
    authorized_to_fresh_clock_bridge: &[OperationalClockBasisV1],
    fresh_basis: &OperationalClockBasisV1,
    hardware_refreshes: &[FinalizationHardwareRefreshInputV1<'_>],
) -> Result<ClockGovernedUpgradeFinalizationExecutionPermitV1, Vec<UpgradeFinalizationExecutionError>> {
    let mut violations = Vec::new();
    let context = authorization.context();

    if authorized_governance_view.id() != context.governance_view_id()
        || authorized_registry_head.id() != context.registry_head_id()
        || authorized_retention_head.id() != context.retention_head_id()
        || authorized_no_rollback.id() != context.no_rollback_id()
    {
        violations.push(UpgradeFinalizationExecutionError::AuthorizedContextMismatch);
    }

    if authorized_governance_view.registry_head_id() != authorized_registry_head.id()
        || authorized_governance_view.registry_digest() != authorized_registry_head.registry_digest()
        || authorized_governance_view.registry_sequence() != authorized_registry_head.sequence()
        || authorized_registry_head.trust_snapshot_digest()
            != context.current_trust_snapshot_digest()
        || authorized_registry_head.trust_snapshot_sequence()
            != context.current_trust_snapshot_sequence()
    {
        violations.push(UpgradeFinalizationExecutionError::AuthorizedRegistryMismatch);
    }

    if authorized_retention_head.governance_view_id() != authorized_governance_view.id()
        || authorized_retention_head.registry_head_id() != authorized_registry_head.id()
        || authorized_retention_head.policy_digest() != authorized_no_rollback.retention_policy_digest()
        || authorized_retention_head.sequence()
            != authorized_no_rollback.retention_policy_sequence()
    {
        violations.push(UpgradeFinalizationExecutionError::AuthorizedRetentionMismatch);
    }

    if authorized_no_rollback.governance_view_id() != authorized_governance_view.id()
        || authorized_no_rollback.retention_head_id() != authorized_retention_head.id()
        || authorized_no_rollback.governance_checkpoint_digest()
            != context.governance_checkpoint_digest()
        || authorized_no_rollback.state_digest() != context.operational_state_digest()
        || authorized_no_rollback.state_generation() != context.operational_state_generation()
        || authorized_no_rollback.operational_lineage_digest()
            != context.operational_lineage_digest()
    {
        violations.push(UpgradeFinalizationExecutionError::AuthorizedNoRollbackMismatch);
    }

    if authorized_basis.id() != context.current_operational_basis_id()
        || authorized_basis.id() != authorized_governance_view.observation_operational_basis_id()
        || authorized_basis.id() != authorized_registry_head.observation_operational_basis_id()
        || authorized_basis.id() != authorized_retention_head.observation_operational_basis_id()
        || authorized_basis.id() != authorized_no_rollback.observation_operational_basis_id()
    {
        violations.push(UpgradeFinalizationExecutionError::AuthorizedBasisMismatch);
    }
    let authorized_clock = match derive_clock_governance_evaluation_envelope_v1(authorized_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeFinalizationExecutionError::Clock(error));
            return Err(violations);
        }
    };
    if authorized_clock.id() != context.current_clock_envelope_id()
        || authorized_clock.id() != authorized_governance_view.observation_clock_envelope_id()
        || authorized_clock.id() != authorized_registry_head.observation_clock_envelope_id()
        || authorized_clock.id() != authorized_retention_head.observation_clock_envelope_id()
        || authorized_clock.id() != authorized_no_rollback.observation_clock_envelope_id()
    {
        violations.push(UpgradeFinalizationExecutionError::AuthorizedEnvelopeMismatch);
    }

    if fresh_governance_view.registry_head_id() != fresh_registry_head.id()
        || fresh_governance_view.registry_digest() != fresh_registry_head.registry_digest()
        || fresh_governance_view.registry_sequence() != fresh_registry_head.sequence()
    {
        violations.push(UpgradeFinalizationExecutionError::FreshRegistryMismatch);
    }
    if fresh_retention_head.governance_view_id() != fresh_governance_view.id()
        || fresh_retention_head.registry_head_id() != fresh_registry_head.id()
        || fresh_retention_head.governance_checkpoint_digest()
            != fresh_governance_view.checkpoint_digest()
        || fresh_retention_head.transparency_log_digest()
            != fresh_governance_view.transparency_log_digest()
    {
        violations.push(UpgradeFinalizationExecutionError::FreshRetentionMismatch);
    }
    if fresh_no_rollback.governance_view_id() != fresh_governance_view.id()
        || fresh_no_rollback.retention_head_id() != fresh_retention_head.id()
        || fresh_no_rollback.governance_checkpoint_digest()
            != fresh_governance_view.checkpoint_digest()
        || fresh_no_rollback.transparency_log_digest()
            != fresh_governance_view.transparency_log_digest()
    {
        violations.push(UpgradeFinalizationExecutionError::FreshNoRollbackMismatch);
    }

    if fresh_basis.id() != fresh_governance_view.observation_operational_basis_id()
        || fresh_basis.id() != fresh_registry_head.observation_operational_basis_id()
        || fresh_basis.id() != fresh_retention_head.observation_operational_basis_id()
        || fresh_basis.id() != fresh_no_rollback.observation_operational_basis_id()
    {
        violations.push(UpgradeFinalizationExecutionError::FreshBasisMismatch);
    }
    let fresh_clock = match derive_clock_governance_evaluation_envelope_v1(fresh_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeFinalizationExecutionError::Clock(error));
            return Err(violations);
        }
    };
    if fresh_clock.id() != fresh_governance_view.observation_clock_envelope_id()
        || fresh_clock.id() != fresh_registry_head.observation_clock_envelope_id()
        || fresh_clock.id() != fresh_retention_head.observation_clock_envelope_id()
        || fresh_clock.id() != fresh_no_rollback.observation_clock_envelope_id()
    {
        violations.push(UpgradeFinalizationExecutionError::FreshEnvelopeMismatch);
    }

    if authorized_registry_head.activated_registry_id() != fresh_registry_head.activated_registry_id()
        || authorized_registry_head.registry_digest() != fresh_registry_head.registry_digest()
        || authorized_registry_head.sequence() != fresh_registry_head.sequence()
    {
        violations.push(UpgradeFinalizationExecutionError::RegistrySemanticsChanged);
    }
    if authorized_registry_head.trust_snapshot_digest() != fresh_registry_head.trust_snapshot_digest()
        || authorized_registry_head.trust_snapshot_sequence()
            != fresh_registry_head.trust_snapshot_sequence()
        || fresh_registry_head.trust_snapshot_digest() != context.current_trust_snapshot_digest()
        || fresh_registry_head.trust_snapshot_sequence() != context.current_trust_snapshot_sequence()
    {
        violations.push(UpgradeFinalizationExecutionError::TrustSnapshotChanged);
    }
    if authorized_governance_view.containment_authority_id()
            != fresh_governance_view.containment_authority_id()
        || authorized_governance_view.containment_state_digest()
            != fresh_governance_view.containment_state_digest()
        || authorized_governance_view.containment_generation()
            != fresh_governance_view.containment_generation()
        || authorized_governance_view.compromise_tracker_digest()
            != fresh_governance_view.compromise_tracker_digest()
        || fresh_governance_view.containment_state_digest()
            != context.current_containment_state_digest()
        || fresh_governance_view.containment_generation() != context.current_containment_generation()
        || fresh_governance_view.compromise_tracker_digest()
            != context.current_compromise_tracker_digest()
    {
        violations.push(UpgradeFinalizationExecutionError::ContainmentSemanticsChanged);
    }
    if authorized_retention_head.retention_authority_id()
            != fresh_retention_head.retention_authority_id()
        || authorized_retention_head.policy_digest() != fresh_retention_head.policy_digest()
        || authorized_retention_head.sequence() != fresh_retention_head.sequence()
        || authorized_retention_head.effective_at_unix_s()
            != fresh_retention_head.effective_at_unix_s()
        || fresh_retention_head.trust_snapshot_digest() != context.current_trust_snapshot_digest()
        || fresh_retention_head.containment_state_digest()
            != context.current_containment_state_digest()
        || fresh_retention_head.compromise_tracker_digest()
            != context.current_compromise_tracker_digest()
        || fresh_retention_head.containment_generation() != context.current_containment_generation()
    {
        violations.push(UpgradeFinalizationExecutionError::RetentionSemanticsChanged);
    }
    if authorized_no_rollback.state_digest() != fresh_no_rollback.state_digest()
        || authorized_no_rollback.state_generation() != fresh_no_rollback.state_generation()
        || authorized_no_rollback.operational_lineage_digest()
            != fresh_no_rollback.operational_lineage_digest()
        || authorized_no_rollback.operational_state_count()
            != fresh_no_rollback.operational_state_count()
        || authorized_no_rollback.publication_digest() != fresh_no_rollback.publication_digest()
        || authorized_no_rollback.publication_entry_sequence()
            != fresh_no_rollback.publication_entry_sequence()
        || authorized_no_rollback.handoff_id() != fresh_no_rollback.handoff_id()
        || authorized_no_rollback.handoff_plan_digest() != fresh_no_rollback.handoff_plan_digest()
        || authorized_no_rollback.activation_permit_id()
            != fresh_no_rollback.activation_permit_id()
        || authorized_no_rollback.probation_clearance_digest()
            != fresh_no_rollback.probation_clearance_digest()
        || authorized_no_rollback.probation_sequence() != fresh_no_rollback.probation_sequence()
        || authorized_no_rollback.reauthorized_machine_count()
            != fresh_no_rollback.reauthorized_machine_count()
        || authorized_no_rollback.retention_policy_digest()
            != fresh_no_rollback.retention_policy_digest()
        || authorized_no_rollback.retention_policy_sequence()
            != fresh_no_rollback.retention_policy_sequence()
        || authorized_no_rollback.key_snapshot_sequence() != fresh_no_rollback.key_snapshot_sequence()
        || authorized_no_rollback.clock_epoch() != fresh_no_rollback.clock_epoch()
    {
        violations.push(UpgradeFinalizationExecutionError::OperationalSemanticsChanged);
    }

    if fresh_governance_view.checkpoint_digest() == context.governance_checkpoint_digest() {
        violations.push(UpgradeFinalizationExecutionError::FreshCheckpointNotDifferent);
    }

    let authorized_log_digest = match validate_and_digest_log(authorized_log, false) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };
    if authorized_log_digest != authorized_governance_view.transparency_log_digest()
        || authorized_log_digest != authorized_registry_head.transparency_log_digest()
        || authorized_log_digest != authorized_retention_head.transparency_log_digest()
        || authorized_log_digest != authorized_no_rollback.transparency_log_digest()
    {
        violations.push(UpgradeFinalizationExecutionError::TransparencyLogDigestMismatch {
            fresh: false,
        });
    }
    let fresh_log_digest = match validate_and_digest_log(fresh_log, true) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };
    if fresh_log_digest != fresh_governance_view.transparency_log_digest()
        || fresh_log_digest != fresh_registry_head.transparency_log_digest()
        || fresh_log_digest != fresh_retention_head.transparency_log_digest()
        || fresh_log_digest != fresh_no_rollback.transparency_log_digest()
    {
        violations.push(UpgradeFinalizationExecutionError::TransparencyLogDigestMismatch {
            fresh: true,
        });
    }
    if fresh_log.entries.len() <= authorized_log.entries.len()
        || fresh_log.verify_successor_of(authorized_log).is_err()
    {
        violations.push(UpgradeFinalizationExecutionError::TransparencyLogNotStrictExtension);
    }

    if authorized_to_fresh_clock_bridge.len() > MAX_FINALIZATION_EXECUTION_CLOCK_HOPS {
        violations.push(UpgradeFinalizationExecutionError::TooManyClockHops {
            actual: authorized_to_fresh_clock_bridge.len(),
            maximum: MAX_FINALIZATION_EXECUTION_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        authorized_basis.id(),
        authorized_to_fresh_clock_bridge,
        fresh_basis,
    ) {
        violations.push(error);
    }
    if fresh_clock.lower_unix_ms() <= authorized_clock.upper_unix_ms() {
        violations.push(UpgradeFinalizationExecutionError::ExecutionClockNotDefinitelyLater {
            authorized_upper_unix_ms: authorized_clock.upper_unix_ms(),
            fresh_lower_unix_ms: fresh_clock.lower_unix_ms(),
        });
    }
    if fresh_clock.upper_unix_ms() >= context.finalization_deadline_unix_ms() {
        violations.push(UpgradeFinalizationExecutionError::FinalizationMayBeClosed);
    }
    if fresh_clock.upper_unix_ms() >= context.probation_clearance_expires_at_unix_ms() {
        violations.push(UpgradeFinalizationExecutionError::ProbationMayBeExpired);
    }

    if hardware_refreshes.len() != context.hardware_authority_count() {
        violations.push(UpgradeFinalizationExecutionError::HardwareInputCountMismatch {
            expected: context.hardware_authority_count(),
            actual: hardware_refreshes.len(),
        });
    }
    if hardware_refreshes.len() > MAX_FINALIZATION_EXECUTION_HARDWARE_AUTHORITIES {
        violations.push(UpgradeFinalizationExecutionError::TooManyHardwareAuthorities {
            actual: hardware_refreshes.len(),
            maximum: MAX_FINALIZATION_EXECUTION_HARDWARE_AUTHORITIES,
        });
        return Err(violations);
    }

    let mut authorized_machines = BTreeSet::new();
    let mut fresh_machines = BTreeSet::new();
    let mut authorized_hardware_commitments = Vec::with_capacity(hardware_refreshes.len());
    let mut refresh_commitments = Vec::with_capacity(hardware_refreshes.len());
    let mut earliest_hardware_expiry_unix_ms = u64::MAX;

    for pair in hardware_refreshes {
        let authorized = pair.authorized;
        let fresh = pair.fresh;
        let machine_id = authorized.statement().machine_id.clone();
        let fresh_machine_id = fresh.statement().machine_id.clone();

        if !authorized_machines.insert(machine_id.clone()) {
            violations.push(
                UpgradeFinalizationExecutionError::DuplicateAuthorizedHardwareMachine(
                    machine_id.clone(),
                ),
            );
        }
        if !fresh_machines.insert(fresh_machine_id.clone()) {
            violations.push(UpgradeFinalizationExecutionError::DuplicateFreshHardwareMachine(
                fresh_machine_id.clone(),
            ));
        }
        if machine_id != fresh_machine_id {
            violations.push(UpgradeFinalizationExecutionError::HardwareMachineMismatch(
                machine_id.clone(),
            ));
        }

        if authorized.handoff_id() != context.handoff_id()
            || fresh.handoff_id() != context.handoff_id()
        {
            violations.push(UpgradeFinalizationExecutionError::HardwareHandoffMismatch(
                machine_id.clone(),
            ));
        }
        if authorized.probation_clearance_id() != context.probation_clearance_id()
            || fresh.probation_clearance_id() != context.probation_clearance_id()
        {
            violations.push(UpgradeFinalizationExecutionError::HardwareProbationMismatch(
                machine_id.clone(),
            ));
        }
        if authorized.telemetry_bound_clearance_id() != context.telemetry_bound_clearance_id()
            || fresh.telemetry_bound_clearance_id() != context.telemetry_bound_clearance_id()
        {
            violations.push(UpgradeFinalizationExecutionError::HardwareTelemetryMismatch(
                machine_id.clone(),
            ));
        }
        if authorized.statement() != fresh.statement()
            || authorized.statement_digest() != fresh.statement_digest()
        {
            violations.push(UpgradeFinalizationExecutionError::HardwareStatementChanged(
                machine_id.clone(),
            ));
        }
        if authorized.signed_evidence_digest() != fresh.signed_evidence_digest() {
            violations.push(UpgradeFinalizationExecutionError::HardwareSignedEvidenceChanged(
                machine_id.clone(),
            ));
        }
        if authorized.hardware_policy_digest() != fresh.hardware_policy_digest() {
            violations.push(UpgradeFinalizationExecutionError::HardwarePolicyChanged(
                machine_id.clone(),
            ));
        }
        if authorized.verifier_set_digest() != fresh.verifier_set_digest() {
            violations.push(UpgradeFinalizationExecutionError::HardwareVerifierSetChanged(
                machine_id.clone(),
            ));
        }
        if authorized.trust_snapshot_digest() != context.current_trust_snapshot_digest()
            || fresh.trust_snapshot_digest() != context.current_trust_snapshot_digest()
        {
            violations.push(UpgradeFinalizationExecutionError::HardwareTrustChanged(
                machine_id.clone(),
            ));
        }
        if authorized.containment_state_digest() != context.current_containment_state_digest()
            || fresh.containment_state_digest() != context.current_containment_state_digest()
            || authorized.compromise_tracker_digest() != context.current_compromise_tracker_digest()
            || fresh.compromise_tracker_digest() != context.current_compromise_tracker_digest()
        {
            violations.push(UpgradeFinalizationExecutionError::HardwareContainmentChanged(
                machine_id.clone(),
            ));
        }
        if authorized.current_operational_basis_id() != authorized_basis.id()
            || authorized.current_clock_envelope_id() != authorized_clock.id()
        {
            violations.push(
                UpgradeFinalizationExecutionError::AuthorizedHardwareClockMismatch(
                    machine_id.clone(),
                ),
            );
        }
        if fresh.current_operational_basis_id() != fresh_basis.id()
            || fresh.current_clock_envelope_id() != fresh_clock.id()
        {
            violations.push(UpgradeFinalizationExecutionError::FreshHardwareClockMismatch(
                machine_id.clone(),
            ));
        }

        let expires_at_unix_ms = match seconds_to_millis(authorized.statement().expires_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if expires_at_unix_ms <= fresh_clock.upper_unix_ms() {
            violations.push(UpgradeFinalizationExecutionError::HardwareMayExpire(
                machine_id.clone(),
            ));
        }
        earliest_hardware_expiry_unix_ms =
            earliest_hardware_expiry_unix_ms.min(expires_at_unix_ms);

        authorized_hardware_commitments.push(AuthorizedHardwareCommitment {
            authority_id: authorized.id().to_hex(),
            machine_id: machine_id.clone(),
            reauthorization_sequence: authorized.statement().reauthorization_sequence,
            statement_digest: authorized.statement_digest().to_hex(),
            signed_evidence_digest: authorized.signed_evidence_digest().to_hex(),
            hardware_identity_digest: authorized.statement().hardware_identity_digest.to_hex(),
            machine_profile_digest: authorized.statement().machine_profile_digest.to_hex(),
            firmware_digest: authorized.statement().firmware_digest.to_hex(),
            calibration_digest: authorized.statement().calibration_digest.to_hex(),
            capability_digest: authorized.statement().capability_digest.to_hex(),
            expires_at_unix_ms,
        });
        refresh_commitments.push(HardwareRefreshCommitment {
            machine_id,
            authorized_authority_id: authorized.id().to_hex(),
            fresh_authority_id: fresh.id().to_hex(),
            statement_digest: authorized.statement_digest().to_hex(),
            signed_evidence_digest: authorized.signed_evidence_digest().to_hex(),
            hardware_policy_digest: authorized.hardware_policy_digest().to_hex(),
            verifier_set_digest: authorized.verifier_set_digest().to_hex(),
            hardware_identity_digest: authorized.statement().hardware_identity_digest.to_hex(),
            machine_profile_digest: authorized.statement().machine_profile_digest.to_hex(),
            firmware_digest: authorized.statement().firmware_digest.to_hex(),
            calibration_digest: authorized.statement().calibration_digest.to_hex(),
            capability_digest: authorized.statement().capability_digest.to_hex(),
            expires_at_unix_ms,
        });
    }

    authorized_hardware_commitments.sort_by(|left, right| {
        left.machine_id
            .cmp(&right.machine_id)
            .then(left.authority_id.cmp(&right.authority_id))
    });
    let authorized_hardware_set_digest = match hash_serializable(
        AUTHORIZED_HARDWARE_SET_DOMAIN,
        &authorized_hardware_commitments,
    ) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };
    let machine_ids = authorized_hardware_commitments
        .iter()
        .map(|commitment| commitment.machine_id.clone())
        .collect::<Vec<_>>();
    if authorized_hardware_set_digest != context.hardware_authority_set_digest()
        || machine_ids != context.machine_ids()
    {
        violations.push(UpgradeFinalizationExecutionError::AuthorizedHardwareSetMismatch);
    }
    if earliest_hardware_expiry_unix_ms != context.earliest_hardware_expiry_unix_ms() {
        violations.push(UpgradeFinalizationExecutionError::EarliestHardwareExpiryMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    refresh_commitments.sort_by(|left, right| {
        left.machine_id
            .cmp(&right.machine_id)
            .then(left.authorized_authority_id.cmp(&right.authorized_authority_id))
    });
    let hardware_refresh_set_digest =
        hash_serializable(HARDWARE_REFRESH_SET_DOMAIN, &refresh_commitments)
            .map_err(|error| vec![error])?;

    let clock_lineage_digest = hash_serializable(
        CLOCK_LINEAGE_DOMAIN,
        &ClockLineageCommitment {
            authorized_basis_id: authorized_basis.id().to_hex(),
            bridge_basis_ids: authorized_to_fresh_clock_bridge
                .iter()
                .map(|basis| basis.id().to_hex())
                .collect(),
            fresh_basis_id: fresh_basis.id().to_hex(),
        },
    )
    .map_err(|error| vec![error])?;

    let appended_entry_count = fresh_log
        .entries
        .len()
        .checked_sub(authorized_log.entries.len())
        .ok_or_else(|| vec![UpgradeFinalizationExecutionError::TransparencyLogNotStrictExtension])?;

    let commitment = ExecutionPermitCommitment {
        schema: CLOCK_GOVERNED_UPGRADE_FINALIZATION_EXECUTION_SCHEMA,
        authorization_id: authorization.id().to_hex(),
        context_id: context.id().to_hex(),
        authorized_governance_view_id: authorized_governance_view.id().to_hex(),
        fresh_governance_view_id: fresh_governance_view.id().to_hex(),
        authorized_registry_head_id: authorized_registry_head.id().to_hex(),
        fresh_registry_head_id: fresh_registry_head.id().to_hex(),
        authorized_retention_head_id: authorized_retention_head.id().to_hex(),
        fresh_retention_head_id: fresh_retention_head.id().to_hex(),
        authorized_no_rollback_id: authorized_no_rollback.id().to_hex(),
        fresh_no_rollback_id: fresh_no_rollback.id().to_hex(),
        authorized_checkpoint_digest: authorized_governance_view.checkpoint_digest().to_hex(),
        fresh_checkpoint_digest: fresh_governance_view.checkpoint_digest().to_hex(),
        authorized_transparency_log_digest: authorized_log_digest.to_hex(),
        fresh_transparency_log_digest: fresh_log_digest.to_hex(),
        authorized_log_size: authorized_log.entries.len(),
        fresh_log_size: fresh_log.entries.len(),
        appended_entry_count,
        registry_digest: fresh_registry_head.registry_digest().to_hex(),
        registry_sequence: fresh_registry_head.sequence(),
        trust_snapshot_digest: fresh_registry_head.trust_snapshot_digest().to_hex(),
        trust_snapshot_sequence: fresh_registry_head.trust_snapshot_sequence(),
        containment_authority_id_digest: fresh_governance_view
            .containment_authority_id()
            .as_digest()
            .to_hex(),
        containment_state_digest: fresh_governance_view.containment_state_digest().to_hex(),
        compromise_tracker_digest: fresh_governance_view.compromise_tracker_digest().to_hex(),
        containment_generation: fresh_governance_view.containment_generation(),
        retention_authority_id_digest: fresh_retention_head
            .retention_authority_id()
            .as_digest()
            .to_hex(),
        retention_policy_digest: fresh_retention_head.policy_digest().to_hex(),
        retention_policy_sequence: fresh_retention_head.sequence(),
        operational_state_digest: fresh_no_rollback.state_digest().to_hex(),
        operational_state_generation: fresh_no_rollback.state_generation(),
        operational_lineage_digest: fresh_no_rollback.operational_lineage_digest().to_hex(),
        hardware_refresh_set_digest: hardware_refresh_set_digest.to_hex(),
        hardware_authority_count: refresh_commitments.len(),
        machine_ids: machine_ids.clone(),
        authorized_clock_envelope_id: authorized_clock.id().to_hex(),
        fresh_clock_envelope_id: fresh_clock.id().to_hex(),
        authorized_operational_basis_id: authorized_basis.id().to_hex(),
        fresh_operational_basis_id: fresh_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count: authorized_to_fresh_clock_bridge.len(),
        finalization_deadline_unix_ms: context.finalization_deadline_unix_ms(),
        probation_clearance_expires_at_unix_ms: context.probation_clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    };
    let id = ClockGovernedUpgradeFinalizationExecutionPermitIdV1(
        hash_serializable(EXECUTION_PERMIT_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(ClockGovernedUpgradeFinalizationExecutionPermitV1 {
        id,
        authorization_id: authorization.id(),
        context_id: context.id(),
        authorized_governance_view_id: authorized_governance_view.id(),
        fresh_governance_view_id: fresh_governance_view.id(),
        authorized_registry_head_id: authorized_registry_head.id(),
        fresh_registry_head_id: fresh_registry_head.id(),
        authorized_retention_head_id: authorized_retention_head.id(),
        fresh_retention_head_id: fresh_retention_head.id(),
        authorized_no_rollback_id: authorized_no_rollback.id(),
        fresh_no_rollback_id: fresh_no_rollback.id(),
        authorized_checkpoint_digest: authorized_governance_view.checkpoint_digest(),
        fresh_checkpoint_digest: fresh_governance_view.checkpoint_digest(),
        authorized_transparency_log_digest: authorized_log_digest,
        fresh_transparency_log_digest: fresh_log_digest,
        authorized_log_size: authorized_log.entries.len(),
        fresh_log_size: fresh_log.entries.len(),
        appended_entry_count,
        registry_digest: fresh_registry_head.registry_digest(),
        registry_sequence: fresh_registry_head.sequence(),
        trust_snapshot_digest: fresh_registry_head.trust_snapshot_digest(),
        trust_snapshot_sequence: fresh_registry_head.trust_snapshot_sequence(),
        containment_authority_id_digest: fresh_governance_view
            .containment_authority_id()
            .as_digest(),
        containment_state_digest: fresh_governance_view.containment_state_digest(),
        compromise_tracker_digest: fresh_governance_view.compromise_tracker_digest(),
        containment_generation: fresh_governance_view.containment_generation(),
        retention_authority_id_digest: fresh_retention_head.retention_authority_id().as_digest(),
        retention_policy_digest: fresh_retention_head.policy_digest(),
        retention_policy_sequence: fresh_retention_head.sequence(),
        operational_state_digest: fresh_no_rollback.state_digest(),
        operational_state_generation: fresh_no_rollback.state_generation(),
        operational_lineage_digest: fresh_no_rollback.operational_lineage_digest(),
        hardware_refresh_set_digest,
        hardware_authority_count: refresh_commitments.len(),
        machine_ids,
        authorized_clock_envelope_id: authorized_clock.id(),
        fresh_clock_envelope_id: fresh_clock.id(),
        authorized_operational_basis_id: authorized_basis.id(),
        fresh_operational_basis_id: fresh_basis.id(),
        clock_lineage_digest,
        clock_hop_count: authorized_to_fresh_clock_bridge.len(),
        finalization_deadline_unix_ms: context.finalization_deadline_unix_ms(),
        probation_clearance_expires_at_unix_ms: context.probation_clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    })
}

fn validate_and_digest_log(
    log: &TransparencyLog,
    fresh: bool,
) -> Result<Sha256Digest, UpgradeFinalizationExecutionError> {
    log.validate().map_err(|error| {
        UpgradeFinalizationExecutionError::TransparencyLogInvalid {
            fresh,
            reason: format!("{error:?}"),
        }
    })?;
    digest_transparency_log(log).map_err(|error| {
        UpgradeFinalizationExecutionError::TransparencyLogInvalid {
            fresh,
            reason: format!("{error:?}"),
        }
    })
}

fn verify_clock_lineage(
    authorized_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    fresh_basis: &OperationalClockBasisV1,
) -> Result<(), UpgradeFinalizationExecutionError> {
    let mut expected = authorized_basis_id;
    for (index, basis) in bridge.iter().enumerate() {
        let actual = basis.predecessor_operational_basis_id();
        if actual != Some(expected) {
            return Err(UpgradeFinalizationExecutionError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = fresh_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(UpgradeFinalizationExecutionError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn seconds_to_millis(seconds: u64) -> Result<u64, UpgradeFinalizationExecutionError> {
    seconds
        .checked_mul(1_000)
        .ok_or(UpgradeFinalizationExecutionError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, UpgradeFinalizationExecutionError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| UpgradeFinalizationExecutionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
