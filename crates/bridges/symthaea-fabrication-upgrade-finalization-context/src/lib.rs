// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Single-view pre-finalization context for irreversible fabrication upgrade governance.
//!
//! This crate does not finalize an upgrade. It proves that the hardened live capabilities required
//! for a future finalization quorum all describe one exact handoff, one exact governance checkpoint,
//! one exact current trust/containment view, one rollback-free operational lineage, and one exact
//! trusted clock interval. The resulting opaque context ID is suitable as a future threshold payload.

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
use symthaea_fabrication_key_continuity_authority::{
    ClockGovernedKeyContinuityIdV1, ClockGovernedKeyContinuityV1,
};
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, ClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_upgrade_hardware_authority::ClockGovernedHardwareReauthorizationV1;
use symthaea_fabrication_upgrade_probation_authority::{
    ClockGovernedUpgradeProbationClearanceIdV1, ClockGovernedUpgradeProbationClearanceV1,
};
use symthaea_fabrication_upgrade_probation_telemetry::{
    TelemetryBoundUpgradeProbationClearanceIdV1, TelemetryBoundUpgradeProbationClearanceV1,
};
use symthaea_fabrication_upgrade_runtime::{
    ClockGovernedUpgradeActivationPermitIdV1, ClockGovernedUpgradeActivationPermitV1,
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

pub const QUALIFIED_UPGRADE_FINALIZATION_CONTEXT_SCHEMA: &str =
    "symthaea.fabrication.qualified-upgrade-finalization-context.v1";
pub const MAX_FINALIZATION_HARDWARE_AUTHORITIES: usize = 65_536;

const CONTEXT_POLICY_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-finalization-context-policy.v1\0";
const HARDWARE_SET_DOMAIN: &[u8] =
    b"symthaea.fabrication.upgrade-finalization-hardware-set.v1\0";
const FINALIZATION_CONTEXT_DOMAIN: &[u8] =
    b"symthaea.fabrication.qualified-upgrade-finalization-context.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UpgradeFinalizationContextPolicyV1 {
    pub minimum_reauthorized_machines: usize,
    pub maximum_reauthorized_machines: usize,
}

impl Default for UpgradeFinalizationContextPolicyV1 {
    fn default() -> Self {
        Self {
            minimum_reauthorized_machines: 1,
            maximum_reauthorized_machines: MAX_FINALIZATION_HARDWARE_AUTHORITIES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QualifiedUpgradeFinalizationContextIdV1(Sha256Digest);

impl QualifiedUpgradeFinalizationContextIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that all prerequisite authority is mutually consistent at one exact witnessed view.
/// It is a threshold-signing payload candidate, not irreversible finalization authority.
#[derive(Debug, Clone)]
#[must_use]
pub struct QualifiedUpgradeFinalizationContextV1 {
    id: QualifiedUpgradeFinalizationContextIdV1,
    context_policy_digest: Sha256Digest,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    registry_head_id: QuorumObservedWitnessRegistryHeadIdV1,
    governance_checkpoint_digest: Sha256Digest,
    retention_head_id: CurrentEvidenceRetentionHeadIdV1,
    no_rollback_id: CurrentNoRollbackUpgradeAuthorityIdV1,
    operational_state_digest: Sha256Digest,
    operational_state_generation: u64,
    operational_lineage_digest: Sha256Digest,
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
    handoff_plan_digest: Sha256Digest,
    activation_permit_id: ClockGovernedUpgradeActivationPermitIdV1,
    probation_clearance_id: ClockGovernedUpgradeProbationClearanceIdV1,
    telemetry_bound_clearance_id: TelemetryBoundUpgradeProbationClearanceIdV1,
    hardware_authority_set_digest: Sha256Digest,
    hardware_authority_count: usize,
    machine_ids: Vec<String>,
    key_continuity_id: ClockGovernedKeyContinuityIdV1,
    current_trust_snapshot_digest: Sha256Digest,
    current_trust_snapshot_sequence: u64,
    current_containment_state_digest: Sha256Digest,
    current_compromise_tracker_digest: Sha256Digest,
    current_containment_generation: u64,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    finalization_deadline_unix_ms: u64,
    probation_clearance_expires_at_unix_ms: u64,
    earliest_hardware_expiry_unix_ms: u64,
}

impl QualifiedUpgradeFinalizationContextV1 {
    pub fn id(&self) -> QualifiedUpgradeFinalizationContextIdV1 {
        self.id
    }

    pub fn signing_payload_digest(&self) -> Sha256Digest {
        self.id.as_digest()
    }

    pub fn context_policy_digest(&self) -> Sha256Digest {
        self.context_policy_digest
    }

    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 {
        self.governance_view_id
    }

    pub fn registry_head_id(&self) -> QuorumObservedWitnessRegistryHeadIdV1 {
        self.registry_head_id
    }

    pub fn governance_checkpoint_digest(&self) -> Sha256Digest {
        self.governance_checkpoint_digest
    }

    pub fn retention_head_id(&self) -> CurrentEvidenceRetentionHeadIdV1 {
        self.retention_head_id
    }

    pub fn no_rollback_id(&self) -> CurrentNoRollbackUpgradeAuthorityIdV1 {
        self.no_rollback_id
    }

    pub fn operational_state_digest(&self) -> Sha256Digest {
        self.operational_state_digest
    }

    pub fn operational_state_generation(&self) -> u64 {
        self.operational_state_generation
    }

    pub fn operational_lineage_digest(&self) -> Sha256Digest {
        self.operational_lineage_digest
    }

    pub fn handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.handoff_id
    }

    pub fn handoff_plan_digest(&self) -> Sha256Digest {
        self.handoff_plan_digest
    }

    pub fn activation_permit_id(&self) -> ClockGovernedUpgradeActivationPermitIdV1 {
        self.activation_permit_id
    }

    pub fn probation_clearance_id(&self) -> ClockGovernedUpgradeProbationClearanceIdV1 {
        self.probation_clearance_id
    }

    pub fn telemetry_bound_clearance_id(&self) -> TelemetryBoundUpgradeProbationClearanceIdV1 {
        self.telemetry_bound_clearance_id
    }

    pub fn hardware_authority_set_digest(&self) -> Sha256Digest {
        self.hardware_authority_set_digest
    }

    pub fn hardware_authority_count(&self) -> usize {
        self.hardware_authority_count
    }

    pub fn machine_ids(&self) -> &[String] {
        &self.machine_ids
    }

    pub fn key_continuity_id(&self) -> ClockGovernedKeyContinuityIdV1 {
        self.key_continuity_id
    }

    pub fn current_trust_snapshot_digest(&self) -> Sha256Digest {
        self.current_trust_snapshot_digest
    }

    pub fn current_trust_snapshot_sequence(&self) -> u64 {
        self.current_trust_snapshot_sequence
    }

    pub fn current_containment_state_digest(&self) -> Sha256Digest {
        self.current_containment_state_digest
    }

    pub fn current_compromise_tracker_digest(&self) -> Sha256Digest {
        self.current_compromise_tracker_digest
    }

    pub fn current_containment_generation(&self) -> u64 {
        self.current_containment_generation
    }

    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 {
        self.current_clock_envelope_id
    }

    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 {
        self.current_operational_basis_id
    }

    pub fn finalization_deadline_unix_ms(&self) -> u64 {
        self.finalization_deadline_unix_ms
    }

    pub fn probation_clearance_expires_at_unix_ms(&self) -> u64 {
        self.probation_clearance_expires_at_unix_ms
    }

    pub fn earliest_hardware_expiry_unix_ms(&self) -> u64 {
        self.earliest_hardware_expiry_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeFinalizationContextError {
    InvalidPolicy,
    GovernanceRegistryMismatch,
    GovernanceRetentionMismatch,
    GovernanceRollbackMismatch,
    HandoffActivationMismatch,
    ProbationMismatch,
    TelemetryMismatch,
    OperationalProbationMismatch,
    OperationalTrustMismatch,
    KeyContinuityMismatch,
    ObservationBasisMismatch,
    ObservationEnvelopeMismatch,
    Clock(ClockGovernanceTimeError),
    FinalizationMayBeClosed,
    ProbationMayBeExpired,
    InsufficientHardwareAuthorities { actual: usize, required: usize },
    TooManyHardwareAuthorities { actual: usize, maximum: usize },
    HardwareCountOverflow,
    OperationalHardwareCountMismatch { operational: u64, supplied: u64 },
    DuplicateHardwareMachine(String),
    HardwareHandoffMismatch(String),
    HardwareProbationMismatch(String),
    HardwareTelemetryMismatch(String),
    HardwareTrustMismatch(String),
    HardwareContainmentMismatch(String),
    HardwareClockMismatch(String),
    HardwareStatementMismatch(String),
    HardwareStatementMayBeFuture(String),
    HardwareMayExpire(String),
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct ContextPolicyCommitment {
    minimum_reauthorized_machines: usize,
    maximum_reauthorized_machines: usize,
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
    expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct FinalizationContextCommitment {
    schema: &'static str,
    context_policy_digest: String,
    governance_view_id: String,
    registry_head_id: String,
    governance_checkpoint_digest: String,
    retention_head_id: String,
    no_rollback_id: String,
    operational_state_digest: String,
    operational_state_generation: u64,
    operational_lineage_digest: String,
    handoff_id: String,
    handoff_plan_digest: String,
    activation_permit_id: String,
    probation_clearance_id: String,
    telemetry_bound_clearance_id: String,
    hardware_authority_set_digest: String,
    hardware_authority_count: usize,
    machine_ids: Vec<String>,
    key_continuity_id: String,
    current_trust_snapshot_digest: String,
    current_trust_snapshot_sequence: u64,
    current_containment_state_digest: String,
    current_compromise_tracker_digest: String,
    current_containment_generation: u64,
    current_clock_envelope_id: String,
    current_operational_basis_id: String,
    finalization_deadline_unix_ms: u64,
    probation_clearance_expires_at_unix_ms: u64,
    earliest_hardware_expiry_unix_ms: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_upgrade_finalization_context_v1(
    policy: &UpgradeFinalizationContextPolicyV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
    registry_head: &QuorumObservedWitnessRegistryHeadV1,
    retention_head: &CurrentEvidenceRetentionHeadV1,
    no_rollback: &CurrentNoRollbackUpgradeAuthorityV1,
    handoff: &ClockGovernedUpgradeHandoffV1,
    activation: &ClockGovernedUpgradeActivationPermitV1,
    probation: &ClockGovernedUpgradeProbationClearanceV1,
    telemetry_bound_probation: &TelemetryBoundUpgradeProbationClearanceV1,
    hardware_authorities: &[ClockGovernedHardwareReauthorizationV1],
    key_continuity: &ClockGovernedKeyContinuityV1,
    current_basis: &OperationalClockBasisV1,
) -> Result<QualifiedUpgradeFinalizationContextV1, Vec<UpgradeFinalizationContextError>> {
    let mut violations = Vec::new();

    if !valid_policy(policy) {
        violations.push(UpgradeFinalizationContextError::InvalidPolicy);
    }

    if governance_view.registry_head_id() != registry_head.id()
        || governance_view.registry_digest() != registry_head.registry_digest()
        || governance_view.registry_sequence() != registry_head.sequence()
        || governance_view.checkpoint_digest() != retention_head.governance_checkpoint_digest()
        || governance_view.transparency_log_digest() != retention_head.transparency_log_digest()
    {
        violations.push(UpgradeFinalizationContextError::GovernanceRegistryMismatch);
    }

    if retention_head.governance_view_id() != governance_view.id()
        || retention_head.registry_head_id() != registry_head.id()
        || retention_head.containment_state_digest() != governance_view.containment_state_digest()
        || retention_head.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
        || retention_head.containment_generation() != governance_view.containment_generation()
    {
        violations.push(UpgradeFinalizationContextError::GovernanceRetentionMismatch);
    }

    if no_rollback.governance_view_id() != governance_view.id()
        || no_rollback.retention_head_id() != retention_head.id()
        || no_rollback.governance_checkpoint_digest() != governance_view.checkpoint_digest()
        || no_rollback.transparency_log_digest() != governance_view.transparency_log_digest()
        || no_rollback.observation_operational_basis_id()
            != governance_view.observation_operational_basis_id()
        || no_rollback.observation_clock_envelope_id()
            != governance_view.observation_clock_envelope_id()
    {
        violations.push(UpgradeFinalizationContextError::GovernanceRollbackMismatch);
    }

    if activation.handoff_id() != handoff.id()
        || activation.handoff_plan_digest() != handoff.plan_digest()
        || no_rollback.handoff_id() != handoff.id()
        || no_rollback.handoff_plan_digest() != handoff.plan_digest()
        || no_rollback.activation_permit_id() != activation.id()
    {
        violations.push(UpgradeFinalizationContextError::HandoffActivationMismatch);
    }

    if probation.handoff_id() != handoff.id()
        || probation.activation_permit_id() != activation.id()
    {
        violations.push(UpgradeFinalizationContextError::ProbationMismatch);
    }
    if telemetry_bound_probation.clearance_id() != probation.id()
        || telemetry_bound_probation.observation_set_digest() != probation.observation_set_digest()
        || telemetry_bound_probation.observation_count() != probation.observation_count()
    {
        violations.push(UpgradeFinalizationContextError::TelemetryMismatch);
    }
    if no_rollback.probation_clearance_digest() != Some(probation.id().as_digest()) {
        violations.push(UpgradeFinalizationContextError::OperationalProbationMismatch);
    }

    if no_rollback.key_snapshot_sequence() != registry_head.trust_snapshot_sequence() {
        violations.push(UpgradeFinalizationContextError::OperationalTrustMismatch);
    }
    if key_continuity.successor_snapshot_digest() != registry_head.trust_snapshot_digest()
        || key_continuity.successor_snapshot_sequence() != registry_head.trust_snapshot_sequence()
    {
        violations.push(UpgradeFinalizationContextError::KeyContinuityMismatch);
    }

    if current_basis.id() != governance_view.observation_operational_basis_id()
        || current_basis.id() != retention_head.observation_operational_basis_id()
        || current_basis.id() != no_rollback.observation_operational_basis_id()
    {
        violations.push(UpgradeFinalizationContextError::ObservationBasisMismatch);
    }
    let current_clock = match derive_clock_governance_evaluation_envelope_v1(current_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeFinalizationContextError::Clock(error));
            return Err(violations);
        }
    };
    if current_clock.id() != governance_view.observation_clock_envelope_id()
        || current_clock.id() != retention_head.observation_clock_envelope_id()
        || current_clock.id() != no_rollback.observation_clock_envelope_id()
    {
        violations.push(UpgradeFinalizationContextError::ObservationEnvelopeMismatch);
    }

    if current_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms {
        violations.push(UpgradeFinalizationContextError::FinalizationMayBeClosed);
    }
    if current_clock.upper_unix_ms() >= probation.clearance_expires_at_unix_ms() {
        violations.push(UpgradeFinalizationContextError::ProbationMayBeExpired);
    }

    if hardware_authorities.len() < policy.minimum_reauthorized_machines {
        violations.push(UpgradeFinalizationContextError::InsufficientHardwareAuthorities {
            actual: hardware_authorities.len(),
            required: policy.minimum_reauthorized_machines,
        });
    }
    if hardware_authorities.len() > policy.maximum_reauthorized_machines
        || hardware_authorities.len() > MAX_FINALIZATION_HARDWARE_AUTHORITIES
    {
        violations.push(UpgradeFinalizationContextError::TooManyHardwareAuthorities {
            actual: hardware_authorities.len(),
            maximum: policy
                .maximum_reauthorized_machines
                .min(MAX_FINALIZATION_HARDWARE_AUTHORITIES),
        });
        return Err(violations);
    }

    let supplied_hardware_count = match u64::try_from(hardware_authorities.len()) {
        Ok(value) => value,
        Err(_) => {
            violations.push(UpgradeFinalizationContextError::HardwareCountOverflow);
            0
        }
    };
    if no_rollback.reauthorized_machine_count() != supplied_hardware_count {
        violations.push(UpgradeFinalizationContextError::OperationalHardwareCountMismatch {
            operational: no_rollback.reauthorized_machine_count(),
            supplied: supplied_hardware_count,
        });
    }

    let mut seen_machine_ids = BTreeSet::new();
    let mut hardware_commitments = Vec::with_capacity(hardware_authorities.len());
    let mut earliest_hardware_expiry_unix_ms = u64::MAX;
    for authority in hardware_authorities {
        let machine_id = authority.statement().machine_id.clone();
        if !seen_machine_ids.insert(machine_id.clone()) {
            violations.push(UpgradeFinalizationContextError::DuplicateHardwareMachine(
                machine_id,
            ));
            continue;
        }
        if authority.handoff_id() != handoff.id() {
            violations.push(UpgradeFinalizationContextError::HardwareHandoffMismatch(
                machine_id.clone(),
            ));
        }
        if authority.probation_clearance_id() != probation.id() {
            violations.push(UpgradeFinalizationContextError::HardwareProbationMismatch(
                machine_id.clone(),
            ));
        }
        if authority.telemetry_bound_clearance_id() != telemetry_bound_probation.id() {
            violations.push(UpgradeFinalizationContextError::HardwareTelemetryMismatch(
                machine_id.clone(),
            ));
        }
        if authority.trust_snapshot_digest() != registry_head.trust_snapshot_digest() {
            violations.push(UpgradeFinalizationContextError::HardwareTrustMismatch(
                machine_id.clone(),
            ));
        }
        if authority.containment_state_digest() != governance_view.containment_state_digest()
            || authority.compromise_tracker_digest() != governance_view.compromise_tracker_digest()
        {
            violations.push(UpgradeFinalizationContextError::HardwareContainmentMismatch(
                machine_id.clone(),
            ));
        }
        if authority.current_operational_basis_id() != current_basis.id()
            || authority.current_clock_envelope_id() != current_clock.id()
        {
            violations.push(UpgradeFinalizationContextError::HardwareClockMismatch(
                machine_id.clone(),
            ));
        }
        if authority.statement().handoff_digest != handoff.plan_digest()
            || authority.statement().successor_source_tree_digest
                != handoff.plan().successor.source_tree_digest
            || authority.statement().successor_executable_digest
                != handoff.plan().successor.executable_digest
        {
            violations.push(UpgradeFinalizationContextError::HardwareStatementMismatch(
                machine_id.clone(),
            ));
        }

        let issued_at_unix_ms = match seconds_to_millis(authority.statement().issued_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        let expires_at_unix_ms = match seconds_to_millis(authority.statement().expires_at_unix_s) {
            Ok(value) => value,
            Err(error) => {
                violations.push(error);
                continue;
            }
        };
        if issued_at_unix_ms > current_clock.lower_unix_ms() {
            violations.push(UpgradeFinalizationContextError::HardwareStatementMayBeFuture(
                machine_id.clone(),
            ));
        }
        if expires_at_unix_ms <= current_clock.upper_unix_ms() {
            violations.push(UpgradeFinalizationContextError::HardwareMayExpire(
                machine_id.clone(),
            ));
        }
        earliest_hardware_expiry_unix_ms =
            earliest_hardware_expiry_unix_ms.min(expires_at_unix_ms);

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
            expires_at_unix_ms,
        });
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    hardware_commitments.sort_by(|left, right| {
        left.machine_id
            .cmp(&right.machine_id)
            .then(left.authority_id.cmp(&right.authority_id))
    });
    let machine_ids = hardware_commitments
        .iter()
        .map(|commitment| commitment.machine_id.clone())
        .collect::<Vec<_>>();
    let hardware_authority_set_digest =
        hash_serializable(HARDWARE_SET_DOMAIN, &hardware_commitments).map_err(|error| vec![error])?;
    let context_policy_digest = hash_serializable(
        CONTEXT_POLICY_DOMAIN,
        &ContextPolicyCommitment {
            minimum_reauthorized_machines: policy.minimum_reauthorized_machines,
            maximum_reauthorized_machines: policy.maximum_reauthorized_machines,
        },
    )
    .map_err(|error| vec![error])?;

    let commitment = FinalizationContextCommitment {
        schema: QUALIFIED_UPGRADE_FINALIZATION_CONTEXT_SCHEMA,
        context_policy_digest: context_policy_digest.to_hex(),
        governance_view_id: governance_view.id().to_hex(),
        registry_head_id: registry_head.id().to_hex(),
        governance_checkpoint_digest: governance_view.checkpoint_digest().to_hex(),
        retention_head_id: retention_head.id().to_hex(),
        no_rollback_id: no_rollback.id().to_hex(),
        operational_state_digest: no_rollback.state_digest().to_hex(),
        operational_state_generation: no_rollback.state_generation(),
        operational_lineage_digest: no_rollback.operational_lineage_digest().to_hex(),
        handoff_id: handoff.id().to_hex(),
        handoff_plan_digest: handoff.plan_digest().to_hex(),
        activation_permit_id: activation.id().to_hex(),
        probation_clearance_id: probation.id().to_hex(),
        telemetry_bound_clearance_id: telemetry_bound_probation.id().to_hex(),
        hardware_authority_set_digest: hardware_authority_set_digest.to_hex(),
        hardware_authority_count: hardware_commitments.len(),
        machine_ids: machine_ids.clone(),
        key_continuity_id: key_continuity.id().to_hex(),
        current_trust_snapshot_digest: registry_head.trust_snapshot_digest().to_hex(),
        current_trust_snapshot_sequence: registry_head.trust_snapshot_sequence(),
        current_containment_state_digest: governance_view.containment_state_digest().to_hex(),
        current_compromise_tracker_digest: governance_view.compromise_tracker_digest().to_hex(),
        current_containment_generation: governance_view.containment_generation(),
        current_clock_envelope_id: current_clock.id().to_hex(),
        current_operational_basis_id: current_basis.id().to_hex(),
        finalization_deadline_unix_ms: handoff.plan().finalization_deadline_unix_ms,
        probation_clearance_expires_at_unix_ms: probation.clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    };
    let id = QualifiedUpgradeFinalizationContextIdV1(
        hash_serializable(FINALIZATION_CONTEXT_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(QualifiedUpgradeFinalizationContextV1 {
        id,
        context_policy_digest,
        governance_view_id: governance_view.id(),
        registry_head_id: registry_head.id(),
        governance_checkpoint_digest: governance_view.checkpoint_digest(),
        retention_head_id: retention_head.id(),
        no_rollback_id: no_rollback.id(),
        operational_state_digest: no_rollback.state_digest(),
        operational_state_generation: no_rollback.state_generation(),
        operational_lineage_digest: no_rollback.operational_lineage_digest(),
        handoff_id: handoff.id(),
        handoff_plan_digest: handoff.plan_digest(),
        activation_permit_id: activation.id(),
        probation_clearance_id: probation.id(),
        telemetry_bound_clearance_id: telemetry_bound_probation.id(),
        hardware_authority_set_digest,
        hardware_authority_count: hardware_commitments.len(),
        machine_ids,
        key_continuity_id: key_continuity.id(),
        current_trust_snapshot_digest: registry_head.trust_snapshot_digest(),
        current_trust_snapshot_sequence: registry_head.trust_snapshot_sequence(),
        current_containment_state_digest: governance_view.containment_state_digest(),
        current_compromise_tracker_digest: governance_view.compromise_tracker_digest(),
        current_containment_generation: governance_view.containment_generation(),
        current_clock_envelope_id: current_clock.id(),
        current_operational_basis_id: current_basis.id(),
        finalization_deadline_unix_ms: handoff.plan().finalization_deadline_unix_ms,
        probation_clearance_expires_at_unix_ms: probation.clearance_expires_at_unix_ms(),
        earliest_hardware_expiry_unix_ms,
    })
}

fn valid_policy(policy: &UpgradeFinalizationContextPolicyV1) -> bool {
    policy.minimum_reauthorized_machines > 0
        && policy.maximum_reauthorized_machines > 0
        && policy.minimum_reauthorized_machines <= policy.maximum_reauthorized_machines
        && policy.maximum_reauthorized_machines <= MAX_FINALIZATION_HARDWARE_AUTHORITIES
}

fn seconds_to_millis(value: u64) -> Result<u64, UpgradeFinalizationContextError> {
    value
        .checked_mul(1_000)
        .ok_or(UpgradeFinalizationContextError::TimeScaleOverflow)
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, UpgradeFinalizationContextError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| UpgradeFinalizationContextError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
