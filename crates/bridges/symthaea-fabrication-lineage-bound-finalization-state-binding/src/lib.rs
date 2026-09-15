// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact concrete-state binding for lineage-bound irreversible upgrade finalization.
//!
//! Fresh execution authority still does not identify a concrete mutable `FabricationUpgradeState`
//! by itself. This bridge follows the exact upgrade-state digest committed by the exact rollback-free
//! operational state authorized by the lineage-bound finalization context, requires that concrete
//! state to be `Activated` for the exact current upgrade cycle, and checks activation/commit ordering.

#![deny(unsafe_code)]

use serde::Serialize;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::upgrade_operational_state::{
    FabricationUpgradeOperationalState, digest_upgrade_operational_state,
};
use symthaea_fabrication_kernel::upgrade_state::{FabricationUpgradeState, digest_upgrade_state};
use symthaea_fabrication_kernel::upgrade_tracker::UpgradeStage;
use symthaea_fabrication_lineage_bound_current_no_rollback::{
    LineageBoundCurrentNoRollbackIdV1, LineageBoundCurrentNoRollbackV1,
};
use symthaea_fabrication_lineage_bound_finalization_authorization::{
    AuthorizedLineageBoundFinalizationIdV1, AuthorizedLineageBoundFinalizationV1,
};
use symthaea_fabrication_lineage_bound_finalization_execution::{
    LineageBoundFinalizationExecutionPermitIdV1, LineageBoundFinalizationExecutionPermitV1,
};
use symthaea_fabrication_lineage_bound_upgrade_runtime::{
    LineageBoundUpgradeActivationPermitIdV1, LineageBoundUpgradeActivationPermitV1,
};

pub const LINEAGE_BOUND_FINALIZATION_STATE_BINDING_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-finalization-state-binding.v1";
const STATE_BINDING_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalization-state-binding.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundFinalizationStateBindingIdV1(Sha256Digest);

impl LineageBoundFinalizationStateBindingIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundFinalizationStateBindingV1 {
    id: LineageBoundFinalizationStateBindingIdV1,
    authorization_id: AuthorizedLineageBoundFinalizationIdV1,
    execution_permit_id: LineageBoundFinalizationExecutionPermitIdV1,
    no_rollback_id: LineageBoundCurrentNoRollbackIdV1,
    activation_permit_id: LineageBoundUpgradeActivationPermitIdV1,
    predecessor_root_digest: Sha256Digest,
    predecessor_current_head_digest: Sha256Digest,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    handoff_plan_digest: Sha256Digest,
    operational_state_digest: Sha256Digest,
    operational_state_generation: u64,
    operational_lineage_digest: Sha256Digest,
    operational_state_committed_at_unix_ms: u64,
    upgrade_state_digest: Sha256Digest,
    upgrade_state_generation: u64,
    upgrade_state_committed_at_unix_ms: u64,
    handoff_sequence: u64,
    upgrade_tracker_digest: Sha256Digest,
    policy_migration_set_digest: Sha256Digest,
    clock_tracker_digest: Sha256Digest,
    authority_epoch_tracker_digest: Sha256Digest,
    recovery_tracker_digest: Sha256Digest,
    evidence_compaction_tracker_digest: Sha256Digest,
}

impl LineageBoundFinalizationStateBindingV1 {
    pub fn id(&self) -> LineageBoundFinalizationStateBindingIdV1 { self.id }
    pub fn authorization_id(&self) -> AuthorizedLineageBoundFinalizationIdV1 { self.authorization_id }
    pub fn execution_permit_id(&self) -> LineageBoundFinalizationExecutionPermitIdV1 { self.execution_permit_id }
    pub fn no_rollback_id(&self) -> LineageBoundCurrentNoRollbackIdV1 { self.no_rollback_id }
    pub fn activation_permit_id(&self) -> LineageBoundUpgradeActivationPermitIdV1 { self.activation_permit_id }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn predecessor_current_head_digest(&self) -> Sha256Digest { self.predecessor_current_head_digest }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn upgrade_cycle_sequence(&self) -> u64 { self.upgrade_cycle_sequence }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn operational_state_digest(&self) -> Sha256Digest { self.operational_state_digest }
    pub fn operational_state_generation(&self) -> u64 { self.operational_state_generation }
    pub fn operational_lineage_digest(&self) -> Sha256Digest { self.operational_lineage_digest }
    pub fn operational_state_committed_at_unix_ms(&self) -> u64 { self.operational_state_committed_at_unix_ms }
    pub fn upgrade_state_digest(&self) -> Sha256Digest { self.upgrade_state_digest }
    pub fn upgrade_state_generation(&self) -> u64 { self.upgrade_state_generation }
    pub fn upgrade_state_committed_at_unix_ms(&self) -> u64 { self.upgrade_state_committed_at_unix_ms }
    pub fn handoff_sequence(&self) -> u64 { self.handoff_sequence }
    pub fn upgrade_tracker_digest(&self) -> Sha256Digest { self.upgrade_tracker_digest }
    pub fn policy_migration_set_digest(&self) -> Sha256Digest { self.policy_migration_set_digest }
    pub fn clock_tracker_digest(&self) -> Sha256Digest { self.clock_tracker_digest }
    pub fn authority_epoch_tracker_digest(&self) -> Sha256Digest { self.authority_epoch_tracker_digest }
    pub fn recovery_tracker_digest(&self) -> Sha256Digest { self.recovery_tracker_digest }
    pub fn evidence_compaction_tracker_digest(&self) -> Sha256Digest { self.evidence_compaction_tracker_digest }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundFinalizationStateBindingError {
    ExecutionAuthorizationMismatch,
    ExecutionContextMismatch,
    NoRollbackMismatch,
    ActivationMismatch,
    PredecessorProvenanceMismatch,
    OperationalStateInvalid(String),
    OperationalStateDigestMismatch,
    OperationalStateGenerationMismatch,
    OperationalHandoffMismatch,
    OperationalRollbackPresent,
    UpgradeStateInvalid(String),
    UpgradeStateDigestMismatch,
    UpgradeStateNotActivated,
    UpgradeStateHandoffMismatch,
    UpgradeStateSequenceMismatch { expected: u64, actual: u64 },
    UpgradeStateBeforeActivation,
    UpgradeStateAfterOperationalCommit,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct StateBindingCommitment {
    schema: &'static str,
    authorization_id: String,
    execution_permit_id: String,
    no_rollback_id: String,
    activation_permit_id: String,
    predecessor_root_digest: String,
    predecessor_current_head_digest: String,
    predecessor_finalization_sequence: u64,
    upgrade_cycle_sequence: u64,
    handoff_plan_digest: String,
    operational_state_digest: String,
    operational_state_generation: u64,
    operational_lineage_digest: String,
    operational_state_committed_at_unix_ms: u64,
    upgrade_state_digest: String,
    upgrade_state_generation: u64,
    upgrade_state_committed_at_unix_ms: u64,
    handoff_sequence: u64,
    upgrade_tracker_digest: String,
    policy_migration_set_digest: String,
    clock_tracker_digest: String,
    authority_epoch_tracker_digest: String,
    recovery_tracker_digest: String,
    evidence_compaction_tracker_digest: String,
}

pub fn bind_lineage_bound_finalization_state_v1(
    authorization: &AuthorizedLineageBoundFinalizationV1,
    execution_permit: &LineageBoundFinalizationExecutionPermitV1,
    no_rollback: &LineageBoundCurrentNoRollbackV1,
    activation: &LineageBoundUpgradeActivationPermitV1,
    operational_state: &FabricationUpgradeOperationalState,
    upgrade_state: &FabricationUpgradeState,
) -> Result<LineageBoundFinalizationStateBindingV1, Vec<LineageBoundFinalizationStateBindingError>> {
    let mut violations = Vec::new();
    let context = authorization.context();

    if execution_permit.authorization_id() != authorization.id() {
        violations.push(LineageBoundFinalizationStateBindingError::ExecutionAuthorizationMismatch);
    }
    if execution_permit.context_id() != context.id() {
        violations.push(LineageBoundFinalizationStateBindingError::ExecutionContextMismatch);
    }
    if execution_permit.authorized_no_rollback_id() != no_rollback.id()
        || context.no_rollback_id() != no_rollback.id()
        || no_rollback.state_digest() != execution_permit.operational_state_digest()
        || no_rollback.state_generation() != execution_permit.operational_state_generation()
        || no_rollback.operational_lineage_digest() != execution_permit.operational_lineage_digest()
    {
        violations.push(LineageBoundFinalizationStateBindingError::NoRollbackMismatch);
    }
    if context.activation_permit_id() != activation.id()
        || no_rollback.activation_permit_id() != activation.id()
        || activation.handoff_plan_digest() != context.handoff_plan_digest()
    {
        violations.push(LineageBoundFinalizationStateBindingError::ActivationMismatch);
    }
    if execution_permit.predecessor_root_digest() != context.predecessor_root_digest()
        || execution_permit.predecessor_current_head_digest() != context.predecessor_current_head_digest()
        || execution_permit.predecessor_finalization_sequence() != context.predecessor_finalization_sequence()
        || execution_permit.upgrade_cycle_sequence() != context.upgrade_cycle_sequence()
        || activation.predecessor_root_digest() != context.predecessor_root_digest()
        || activation.current_head_digest() != context.predecessor_current_head_digest()
        || activation.predecessor_finalization_sequence() != context.predecessor_finalization_sequence()
    {
        violations.push(LineageBoundFinalizationStateBindingError::PredecessorProvenanceMismatch);
    }

    if let Err(error) = operational_state.validate_shape() {
        violations.push(LineageBoundFinalizationStateBindingError::OperationalStateInvalid(
            format!("{error:?}"),
        ));
    }
    let operational_state_digest = match digest_upgrade_operational_state(operational_state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationStateBindingError::OperationalStateInvalid(
                format!("{error:?}"),
            ));
            Sha256Digest([0; 32])
        }
    };
    if operational_state_digest != no_rollback.state_digest()
        || operational_state_digest != context.operational_state_digest()
        || operational_state_digest != execution_permit.operational_state_digest()
    {
        violations.push(LineageBoundFinalizationStateBindingError::OperationalStateDigestMismatch);
    }
    if operational_state.generation != no_rollback.state_generation()
        || operational_state.generation != context.operational_state_generation()
        || operational_state.generation != execution_permit.operational_state_generation()
    {
        violations.push(LineageBoundFinalizationStateBindingError::OperationalStateGenerationMismatch);
    }
    if operational_state.handoff_digest != context.handoff_plan_digest()
        || operational_state.handoff_digest != no_rollback.handoff_plan_digest()
    {
        violations.push(LineageBoundFinalizationStateBindingError::OperationalHandoffMismatch);
    }
    if operational_state.evidence.automatic_rollback_digest.is_some() {
        violations.push(LineageBoundFinalizationStateBindingError::OperationalRollbackPresent);
    }

    if let Err(error) = upgrade_state.validate_shape() {
        violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateInvalid(format!(
            "{error:?}"
        )));
    }
    let upgrade_state_digest = match digest_upgrade_state(upgrade_state) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateInvalid(format!(
                "{error:?}"
            )));
            Sha256Digest([0; 32])
        }
    };
    if operational_state.evidence.upgrade_state_digest != upgrade_state_digest {
        violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateDigestMismatch);
    }
    if upgrade_state.active_stage != UpgradeStage::Activated {
        violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateNotActivated);
    }
    if upgrade_state.evidence.handoff_digest != context.handoff_plan_digest() {
        violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateHandoffMismatch);
    }
    if upgrade_state.handoff_sequence != context.upgrade_cycle_sequence() {
        violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateSequenceMismatch {
            expected: context.upgrade_cycle_sequence(), actual: upgrade_state.handoff_sequence,
        });
    }
    if upgrade_state.committed_at_unix_ms < activation.activates_at_unix_ms() {
        violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateBeforeActivation);
    }
    if upgrade_state.committed_at_unix_ms > operational_state.committed_at_unix_ms {
        violations.push(LineageBoundFinalizationStateBindingError::UpgradeStateAfterOperationalCommit);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let commitment = StateBindingCommitment {
        schema: LINEAGE_BOUND_FINALIZATION_STATE_BINDING_SCHEMA,
        authorization_id: authorization.id().to_hex(),
        execution_permit_id: execution_permit.id().to_hex(),
        no_rollback_id: no_rollback.id().to_hex(),
        activation_permit_id: activation.id().to_hex(),
        predecessor_root_digest: context.predecessor_root_digest().to_hex(),
        predecessor_current_head_digest: context.predecessor_current_head_digest().to_hex(),
        predecessor_finalization_sequence: context.predecessor_finalization_sequence(),
        upgrade_cycle_sequence: context.upgrade_cycle_sequence(),
        handoff_plan_digest: context.handoff_plan_digest().to_hex(),
        operational_state_digest: operational_state_digest.to_hex(),
        operational_state_generation: operational_state.generation,
        operational_lineage_digest: no_rollback.operational_lineage_digest().to_hex(),
        operational_state_committed_at_unix_ms: operational_state.committed_at_unix_ms,
        upgrade_state_digest: upgrade_state_digest.to_hex(),
        upgrade_state_generation: upgrade_state.generation,
        upgrade_state_committed_at_unix_ms: upgrade_state.committed_at_unix_ms,
        handoff_sequence: upgrade_state.handoff_sequence,
        upgrade_tracker_digest: upgrade_state.evidence.upgrade_tracker_digest.to_hex(),
        policy_migration_set_digest: upgrade_state.evidence.policy_migration_set_digest.to_hex(),
        clock_tracker_digest: upgrade_state.evidence.clock_tracker_digest.to_hex(),
        authority_epoch_tracker_digest: upgrade_state.evidence.authority_epoch_tracker_digest.to_hex(),
        recovery_tracker_digest: upgrade_state.evidence.recovery_tracker_digest.to_hex(),
        evidence_compaction_tracker_digest: upgrade_state.evidence.evidence_compaction_tracker_digest.to_hex(),
    };
    let id = LineageBoundFinalizationStateBindingIdV1(
        hash_serializable(STATE_BINDING_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(LineageBoundFinalizationStateBindingV1 {
        id,
        authorization_id: authorization.id(),
        execution_permit_id: execution_permit.id(),
        no_rollback_id: no_rollback.id(),
        activation_permit_id: activation.id(),
        predecessor_root_digest: context.predecessor_root_digest(),
        predecessor_current_head_digest: context.predecessor_current_head_digest(),
        predecessor_finalization_sequence: context.predecessor_finalization_sequence(),
        upgrade_cycle_sequence: context.upgrade_cycle_sequence(),
        handoff_plan_digest: context.handoff_plan_digest(),
        operational_state_digest,
        operational_state_generation: operational_state.generation,
        operational_lineage_digest: no_rollback.operational_lineage_digest(),
        operational_state_committed_at_unix_ms: operational_state.committed_at_unix_ms,
        upgrade_state_digest,
        upgrade_state_generation: upgrade_state.generation,
        upgrade_state_committed_at_unix_ms: upgrade_state.committed_at_unix_ms,
        handoff_sequence: upgrade_state.handoff_sequence,
        upgrade_tracker_digest: upgrade_state.evidence.upgrade_tracker_digest,
        policy_migration_set_digest: upgrade_state.evidence.policy_migration_set_digest,
        clock_tracker_digest: upgrade_state.evidence.clock_tracker_digest,
        authority_epoch_tracker_digest: upgrade_state.evidence.authority_epoch_tracker_digest,
        recovery_tracker_digest: upgrade_state.evidence.recovery_tracker_digest,
        evidence_compaction_tracker_digest: upgrade_state.evidence.evidence_compaction_tracker_digest,
    })
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundFinalizationStateBindingError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundFinalizationStateBindingError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
