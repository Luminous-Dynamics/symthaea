// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact state binding between fresh finalization execution authority and legacy upgrade state.
//!
//! The execution permit proves fresh distributed authority, but irreversible stage mutation still
//! targets `FabricationUpgradeState`. This bridge prevents an arbitrary activated state from being
//! substituted by following the exact digest already committed by the fresh rollback-free
//! operational state.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_current_no_rollback::{
    CurrentNoRollbackUpgradeAuthorityIdV1, CurrentNoRollbackUpgradeAuthorityV1,
};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::upgrade_operational_state::{
    FabricationUpgradeOperationalState, digest_upgrade_operational_state,
};
use symthaea_fabrication_kernel::upgrade_state::{
    FabricationUpgradeState, digest_upgrade_state,
};
use symthaea_fabrication_kernel::upgrade_tracker::UpgradeStage;
use symthaea_fabrication_upgrade_finalization_execution::{
    ClockGovernedUpgradeFinalizationExecutionPermitIdV1,
    ClockGovernedUpgradeFinalizationExecutionPermitV1,
};

pub const EXECUTION_BOUND_UPGRADE_FINALIZATION_STATE_SCHEMA: &str =
    "symthaea.fabrication.execution-bound-upgrade-finalization-state.v1";
const STATE_BINDING_DOMAIN: &[u8] =
    b"symthaea.fabrication.execution-bound-upgrade-finalization-state.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionBoundUpgradeFinalizationStateIdV1(Sha256Digest);

impl ExecutionBoundUpgradeFinalizationStateIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque proof that one exact activated `FabricationUpgradeState` is the state already committed
/// by the exact fresh rollback-free operational head named by the execution permit.
#[derive(Debug, Clone)]
#[must_use]
pub struct ExecutionBoundUpgradeFinalizationStateV1 {
    id: ExecutionBoundUpgradeFinalizationStateIdV1,
    execution_permit_id: ClockGovernedUpgradeFinalizationExecutionPermitIdV1,
    fresh_no_rollback_id: CurrentNoRollbackUpgradeAuthorityIdV1,
    operational_state_digest: Sha256Digest,
    operational_state_generation: u64,
    upgrade_state_digest: Sha256Digest,
    upgrade_state_generation: u64,
    handoff_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    upgrade_tracker_digest: Sha256Digest,
    policy_migration_set_digest: Sha256Digest,
    clock_tracker_digest: Sha256Digest,
    authority_epoch_tracker_digest: Sha256Digest,
    recovery_tracker_digest: Sha256Digest,
    evidence_compaction_tracker_digest: Sha256Digest,
}

impl ExecutionBoundUpgradeFinalizationStateV1 {
    pub fn id(&self) -> ExecutionBoundUpgradeFinalizationStateIdV1 {
        self.id
    }

    pub fn execution_permit_id(&self) -> ClockGovernedUpgradeFinalizationExecutionPermitIdV1 {
        self.execution_permit_id
    }

    pub fn fresh_no_rollback_id(&self) -> CurrentNoRollbackUpgradeAuthorityIdV1 {
        self.fresh_no_rollback_id
    }

    pub fn operational_state_digest(&self) -> Sha256Digest {
        self.operational_state_digest
    }

    pub fn operational_state_generation(&self) -> u64 {
        self.operational_state_generation
    }

    pub fn upgrade_state_digest(&self) -> Sha256Digest {
        self.upgrade_state_digest
    }

    pub fn upgrade_state_generation(&self) -> u64 {
        self.upgrade_state_generation
    }

    pub fn handoff_sequence(&self) -> u64 {
        self.handoff_sequence
    }

    pub fn handoff_plan_digest(&self) -> Sha256Digest {
        self.handoff_plan_digest
    }

    pub fn upgrade_tracker_digest(&self) -> Sha256Digest {
        self.upgrade_tracker_digest
    }

    pub fn policy_migration_set_digest(&self) -> Sha256Digest {
        self.policy_migration_set_digest
    }

    pub fn clock_tracker_digest(&self) -> Sha256Digest {
        self.clock_tracker_digest
    }

    pub fn authority_epoch_tracker_digest(&self) -> Sha256Digest {
        self.authority_epoch_tracker_digest
    }

    pub fn recovery_tracker_digest(&self) -> Sha256Digest {
        self.recovery_tracker_digest
    }

    pub fn evidence_compaction_tracker_digest(&self) -> Sha256Digest {
        self.evidence_compaction_tracker_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpgradeFinalizationStateBindingError {
    NoRollbackPermitMismatch,
    OperationalStateInvalid(String),
    OperationalStateDigestMismatch,
    OperationalStateGenerationMismatch,
    OperationalHandoffMismatch,
    OperationalRollbackPresent,
    OperationalEvidenceMismatch(&'static str),
    UpgradeStateInvalid(String),
    UpgradeStateDigestMismatch,
    UpgradeStateNotActivated,
    UpgradeStateHandoffMismatch,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct StateBindingCommitment {
    schema: &'static str,
    execution_permit_id: String,
    fresh_no_rollback_id: String,
    operational_state_digest: String,
    operational_state_generation: u64,
    upgrade_state_digest: String,
    upgrade_state_generation: u64,
    handoff_sequence: u64,
    handoff_plan_digest: String,
    upgrade_tracker_digest: String,
    policy_migration_set_digest: String,
    clock_tracker_digest: String,
    authority_epoch_tracker_digest: String,
    recovery_tracker_digest: String,
    evidence_compaction_tracker_digest: String,
}

pub fn bind_upgrade_finalization_state_v1(
    execution_permit: &ClockGovernedUpgradeFinalizationExecutionPermitV1,
    fresh_no_rollback: &CurrentNoRollbackUpgradeAuthorityV1,
    operational_state: &FabricationUpgradeOperationalState,
    upgrade_state: &FabricationUpgradeState,
) -> Result<ExecutionBoundUpgradeFinalizationStateV1, Vec<UpgradeFinalizationStateBindingError>> {
    let mut violations = Vec::new();

    if execution_permit.fresh_no_rollback_id() != fresh_no_rollback.id() {
        violations.push(UpgradeFinalizationStateBindingError::NoRollbackPermitMismatch);
    }

    if let Err(error) = operational_state.validate_shape() {
        violations.push(UpgradeFinalizationStateBindingError::OperationalStateInvalid(
            format!("{error:?}"),
        ));
    }
    let operational_state_digest = match digest_upgrade_operational_state(operational_state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeFinalizationStateBindingError::OperationalStateInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };
    if operational_state_digest != fresh_no_rollback.state_digest() {
        violations.push(UpgradeFinalizationStateBindingError::OperationalStateDigestMismatch);
    }
    if operational_state.generation != fresh_no_rollback.state_generation() {
        violations.push(UpgradeFinalizationStateBindingError::OperationalStateGenerationMismatch);
    }
    if operational_state.handoff_digest != fresh_no_rollback.handoff_plan_digest() {
        violations.push(UpgradeFinalizationStateBindingError::OperationalHandoffMismatch);
    }
    if operational_state.evidence.automatic_rollback_digest.is_some() {
        violations.push(UpgradeFinalizationStateBindingError::OperationalRollbackPresent);
    }

    if operational_state.evidence.probation_clearance_digest
        != fresh_no_rollback.probation_clearance_digest()
    {
        violations.push(UpgradeFinalizationStateBindingError::OperationalEvidenceMismatch(
            "probation_clearance_digest",
        ));
    }
    if operational_state.evidence.probation_sequence != fresh_no_rollback.probation_sequence() {
        violations.push(UpgradeFinalizationStateBindingError::OperationalEvidenceMismatch(
            "probation_sequence",
        ));
    }
    if operational_state.evidence.reauthorized_machine_count
        != fresh_no_rollback.reauthorized_machine_count()
    {
        violations.push(UpgradeFinalizationStateBindingError::OperationalEvidenceMismatch(
            "reauthorized_machine_count",
        ));
    }
    if operational_state.evidence.retention_policy_digest
        != fresh_no_rollback.retention_policy_digest()
        || operational_state.evidence.retention_policy_sequence
            != fresh_no_rollback.retention_policy_sequence()
    {
        violations.push(UpgradeFinalizationStateBindingError::OperationalEvidenceMismatch(
            "retention_policy",
        ));
    }
    if operational_state.evidence.key_snapshot_sequence != fresh_no_rollback.key_snapshot_sequence()
    {
        violations.push(UpgradeFinalizationStateBindingError::OperationalEvidenceMismatch(
            "key_snapshot_sequence",
        ));
    }
    if operational_state.evidence.clock_epoch != fresh_no_rollback.clock_epoch() {
        violations.push(UpgradeFinalizationStateBindingError::OperationalEvidenceMismatch(
            "clock_epoch",
        ));
    }

    if let Err(error) = upgrade_state.validate_shape() {
        violations.push(UpgradeFinalizationStateBindingError::UpgradeStateInvalid(format!(
            "{error:?}"
        )));
    }
    let upgrade_state_digest = match digest_upgrade_state(upgrade_state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(UpgradeFinalizationStateBindingError::UpgradeStateInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if operational_state.evidence.upgrade_state_digest != upgrade_state_digest {
        violations.push(UpgradeFinalizationStateBindingError::UpgradeStateDigestMismatch);
    }
    if upgrade_state.active_stage != UpgradeStage::Activated {
        violations.push(UpgradeFinalizationStateBindingError::UpgradeStateNotActivated);
    }
    if upgrade_state.evidence.handoff_digest != fresh_no_rollback.handoff_plan_digest() {
        violations.push(UpgradeFinalizationStateBindingError::UpgradeStateHandoffMismatch);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let commitment = StateBindingCommitment {
        schema: EXECUTION_BOUND_UPGRADE_FINALIZATION_STATE_SCHEMA,
        execution_permit_id: execution_permit.id().to_hex(),
        fresh_no_rollback_id: fresh_no_rollback.id().to_hex(),
        operational_state_digest: operational_state_digest.to_hex(),
        operational_state_generation: operational_state.generation,
        upgrade_state_digest: upgrade_state_digest.to_hex(),
        upgrade_state_generation: upgrade_state.generation,
        handoff_sequence: upgrade_state.handoff_sequence,
        handoff_plan_digest: fresh_no_rollback.handoff_plan_digest().to_hex(),
        upgrade_tracker_digest: upgrade_state.evidence.upgrade_tracker_digest.to_hex(),
        policy_migration_set_digest: upgrade_state.evidence.policy_migration_set_digest.to_hex(),
        clock_tracker_digest: upgrade_state.evidence.clock_tracker_digest.to_hex(),
        authority_epoch_tracker_digest: upgrade_state.evidence.authority_epoch_tracker_digest.to_hex(),
        recovery_tracker_digest: upgrade_state.evidence.recovery_tracker_digest.to_hex(),
        evidence_compaction_tracker_digest: upgrade_state
            .evidence
            .evidence_compaction_tracker_digest
            .to_hex(),
    };
    let id = ExecutionBoundUpgradeFinalizationStateIdV1(
        hash_serializable(STATE_BINDING_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(ExecutionBoundUpgradeFinalizationStateV1 {
        id,
        execution_permit_id: execution_permit.id(),
        fresh_no_rollback_id: fresh_no_rollback.id(),
        operational_state_digest,
        operational_state_generation: operational_state.generation,
        upgrade_state_digest,
        upgrade_state_generation: upgrade_state.generation,
        handoff_sequence: upgrade_state.handoff_sequence,
        handoff_plan_digest: fresh_no_rollback.handoff_plan_digest(),
        upgrade_tracker_digest: upgrade_state.evidence.upgrade_tracker_digest,
        policy_migration_set_digest: upgrade_state.evidence.policy_migration_set_digest,
        clock_tracker_digest: upgrade_state.evidence.clock_tracker_digest,
        authority_epoch_tracker_digest: upgrade_state.evidence.authority_epoch_tracker_digest,
        recovery_tracker_digest: upgrade_state.evidence.recovery_tracker_digest,
        evidence_compaction_tracker_digest: upgrade_state
            .evidence
            .evidence_compaction_tracker_digest,
    })
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, UpgradeFinalizationStateBindingError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| UpgradeFinalizationStateBindingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
