// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic terminal authority for one fully qualified fabrication upgrade finalization.
//!
//! The legacy upgrade tracker remains portable compatibility evidence. This crate consumes only the
//! hardened finalization authorization, fresh execution permit, exact concrete-state binding and
//! exact hardened handoff. It emits a portable deterministic finalization record plus an opaque live
//! terminal capability. No caller-selected finalization timestamp participates.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::upgrade_tracker::UpgradeStage;
use symthaea_fabrication_upgrade_authority::{
    ClockGovernedUpgradeHandoffIdV1, ClockGovernedUpgradeHandoffV1,
};
use symthaea_fabrication_upgrade_finalization_authorization::{
    AuthorizedClockGovernedUpgradeFinalizationIdV1,
    AuthorizedClockGovernedUpgradeFinalizationV1,
};
use symthaea_fabrication_upgrade_finalization_execution::{
    ClockGovernedUpgradeFinalizationExecutionPermitIdV1,
    ClockGovernedUpgradeFinalizationExecutionPermitV1,
};
use symthaea_fabrication_upgrade_finalization_state_binding::{
    ExecutionBoundUpgradeFinalizationStateIdV1, ExecutionBoundUpgradeFinalizationStateV1,
};

pub const CLOCK_GOVERNED_UPGRADE_FINALIZATION_RECORD_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-upgrade-finalization-record.v1";
pub const CLOCK_GOVERNED_FINALIZED_UPGRADE_SCHEMA: &str =
    "symthaea.fabrication.clock-governed-finalized-upgrade.v1";

const FINALIZATION_RECORD_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-upgrade-finalization-record.v1\0";
const FINALIZED_UPGRADE_DOMAIN: &[u8] =
    b"symthaea.fabrication.clock-governed-finalized-upgrade.v1\0";

/// Portable deterministic audit record. This is evidence, not executable authority by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockGovernedUpgradeFinalizationRecordV1 {
    pub schema_version: String,
    pub finalization_sequence: u64,
    pub terminal_stage: UpgradeStage,
    pub predecessor_upgrade_state_digest: Sha256Digest,
    pub predecessor_upgrade_state_generation: u64,
    pub finalized_upgrade_state_generation: u64,
    pub handoff_id: String,
    pub handoff_plan_digest: Sha256Digest,
    pub successor_source_tree_digest: Sha256Digest,
    pub successor_executable_digest: Sha256Digest,
    pub successor_durable_state_digest: Sha256Digest,
    pub successor_replay_contract_digest: Sha256Digest,
    pub authorization_id: String,
    pub context_id: String,
    pub execution_permit_id: String,
    pub state_binding_id: String,
    pub fresh_checkpoint_digest: Sha256Digest,
    pub fresh_transparency_log_digest: Sha256Digest,
    pub fresh_clock_envelope_id: String,
    pub fresh_operational_basis_id: String,
    pub hardware_refresh_set_digest: Sha256Digest,
    pub hardware_authority_count: u64,
    pub machine_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockGovernedFinalizedUpgradeIdV1(Sha256Digest);

impl ClockGovernedFinalizedUpgradeIdV1 {
    pub fn as_digest(self) -> Sha256Digest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

/// Opaque terminal authority for one exact predecessor upgrade state. Re-running the constructor
/// with the exact same qualified inputs is idempotent and produces the same ID.
#[derive(Debug, Clone)]
#[must_use]
pub struct ClockGovernedFinalizedUpgradeV1 {
    id: ClockGovernedFinalizedUpgradeIdV1,
    record: ClockGovernedUpgradeFinalizationRecordV1,
    record_digest: Sha256Digest,
    authorization_id: AuthorizedClockGovernedUpgradeFinalizationIdV1,
    execution_permit_id: ClockGovernedUpgradeFinalizationExecutionPermitIdV1,
    state_binding_id: ExecutionBoundUpgradeFinalizationStateIdV1,
    handoff_id: ClockGovernedUpgradeHandoffIdV1,
}

impl ClockGovernedFinalizedUpgradeV1 {
    pub fn id(&self) -> ClockGovernedFinalizedUpgradeIdV1 {
        self.id
    }

    pub fn record(&self) -> &ClockGovernedUpgradeFinalizationRecordV1 {
        &self.record
    }

    pub fn record_digest(&self) -> Sha256Digest {
        self.record_digest
    }

    pub fn authorization_id(&self) -> AuthorizedClockGovernedUpgradeFinalizationIdV1 {
        self.authorization_id
    }

    pub fn execution_permit_id(&self) -> ClockGovernedUpgradeFinalizationExecutionPermitIdV1 {
        self.execution_permit_id
    }

    pub fn state_binding_id(&self) -> ExecutionBoundUpgradeFinalizationStateIdV1 {
        self.state_binding_id
    }

    pub fn handoff_id(&self) -> ClockGovernedUpgradeHandoffIdV1 {
        self.handoff_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockGovernedUpgradeFinalizationStateError {
    ExecutionAuthorizationMismatch,
    ExecutionContextMismatch,
    StateBindingExecutionMismatch,
    HandoffAuthorizationMismatch,
    HandoffPlanMismatch,
    GenerationOverflow,
    HardwareCountOverflow,
    EmptyMachineSet,
    MachineCountMismatch,
    InvalidSuccessorDigest(&'static str),
    InvalidRecord,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct FinalizedUpgradeCommitment {
    schema: &'static str,
    record_digest: String,
    authorization_id: String,
    execution_permit_id: String,
    state_binding_id: String,
    handoff_id: String,
}

pub fn finalize_clock_governed_upgrade_v1(
    authorization: &AuthorizedClockGovernedUpgradeFinalizationV1,
    execution_permit: &ClockGovernedUpgradeFinalizationExecutionPermitV1,
    state_binding: &ExecutionBoundUpgradeFinalizationStateV1,
    handoff: &ClockGovernedUpgradeHandoffV1,
) -> Result<ClockGovernedFinalizedUpgradeV1, ClockGovernedUpgradeFinalizationStateError> {
    if execution_permit.authorization_id() != authorization.id() {
        return Err(ClockGovernedUpgradeFinalizationStateError::ExecutionAuthorizationMismatch);
    }
    if execution_permit.context_id() != authorization.context().id() {
        return Err(ClockGovernedUpgradeFinalizationStateError::ExecutionContextMismatch);
    }
    if state_binding.execution_permit_id() != execution_permit.id() {
        return Err(ClockGovernedUpgradeFinalizationStateError::StateBindingExecutionMismatch);
    }
    if authorization.context().handoff_id() != handoff.id() {
        return Err(ClockGovernedUpgradeFinalizationStateError::HandoffAuthorizationMismatch);
    }
    if authorization.context().handoff_plan_digest() != handoff.plan_digest()
        || state_binding.handoff_plan_digest() != handoff.plan_digest()
    {
        return Err(ClockGovernedUpgradeFinalizationStateError::HandoffPlanMismatch);
    }

    let finalized_upgrade_state_generation = state_binding
        .upgrade_state_generation()
        .checked_add(1)
        .ok_or(ClockGovernedUpgradeFinalizationStateError::GenerationOverflow)?;
    let hardware_authority_count = u64::try_from(execution_permit.hardware_authority_count())
        .map_err(|_| ClockGovernedUpgradeFinalizationStateError::HardwareCountOverflow)?;
    if execution_permit.machine_ids().is_empty() {
        return Err(ClockGovernedUpgradeFinalizationStateError::EmptyMachineSet);
    }
    if execution_permit.machine_ids().len() != execution_permit.hardware_authority_count() {
        return Err(ClockGovernedUpgradeFinalizationStateError::MachineCountMismatch);
    }

    for (name, digest) in [
        ("successor_source_tree_digest", handoff.plan().successor.source_tree_digest),
        ("successor_executable_digest", handoff.plan().successor.executable_digest),
        ("successor_durable_state_digest", handoff.plan().successor.durable_state_digest),
        ("successor_replay_contract_digest", handoff.plan().successor.replay_contract_digest),
    ] {
        if digest.0 == [0; 32] {
            return Err(ClockGovernedUpgradeFinalizationStateError::InvalidSuccessorDigest(
                name,
            ));
        }
    }

    let record = ClockGovernedUpgradeFinalizationRecordV1 {
        schema_version: CLOCK_GOVERNED_UPGRADE_FINALIZATION_RECORD_SCHEMA.into(),
        finalization_sequence: state_binding.handoff_sequence(),
        terminal_stage: UpgradeStage::Finalized,
        predecessor_upgrade_state_digest: state_binding.upgrade_state_digest(),
        predecessor_upgrade_state_generation: state_binding.upgrade_state_generation(),
        finalized_upgrade_state_generation,
        handoff_id: handoff.id().to_hex(),
        handoff_plan_digest: handoff.plan_digest(),
        successor_source_tree_digest: handoff.plan().successor.source_tree_digest,
        successor_executable_digest: handoff.plan().successor.executable_digest,
        successor_durable_state_digest: handoff.plan().successor.durable_state_digest,
        successor_replay_contract_digest: handoff.plan().successor.replay_contract_digest,
        authorization_id: authorization.id().to_hex(),
        context_id: authorization.context().id().to_hex(),
        execution_permit_id: execution_permit.id().to_hex(),
        state_binding_id: state_binding.id().to_hex(),
        fresh_checkpoint_digest: execution_permit.fresh_checkpoint_digest(),
        fresh_transparency_log_digest: execution_permit.fresh_transparency_log_digest(),
        fresh_clock_envelope_id: execution_permit.fresh_clock_envelope_id().to_hex(),
        fresh_operational_basis_id: execution_permit.fresh_operational_basis_id().to_hex(),
        hardware_refresh_set_digest: execution_permit.hardware_refresh_set_digest(),
        hardware_authority_count,
        machine_ids: execution_permit.machine_ids().to_vec(),
    };
    validate_record(&record)?;
    let record_digest = digest_finalization_record_v1(&record)?;

    let commitment = FinalizedUpgradeCommitment {
        schema: CLOCK_GOVERNED_FINALIZED_UPGRADE_SCHEMA,
        record_digest: record_digest.to_hex(),
        authorization_id: authorization.id().to_hex(),
        execution_permit_id: execution_permit.id().to_hex(),
        state_binding_id: state_binding.id().to_hex(),
        handoff_id: handoff.id().to_hex(),
    };
    let id = ClockGovernedFinalizedUpgradeIdV1(hash_serializable(
        FINALIZED_UPGRADE_DOMAIN,
        &commitment,
    )?);

    Ok(ClockGovernedFinalizedUpgradeV1 {
        id,
        record,
        record_digest,
        authorization_id: authorization.id(),
        execution_permit_id: execution_permit.id(),
        state_binding_id: state_binding.id(),
        handoff_id: handoff.id(),
    })
}

pub fn digest_finalization_record_v1(
    record: &ClockGovernedUpgradeFinalizationRecordV1,
) -> Result<Sha256Digest, ClockGovernedUpgradeFinalizationStateError> {
    validate_record(record)?;
    hash_serializable(FINALIZATION_RECORD_DOMAIN, record)
}

fn validate_record(
    record: &ClockGovernedUpgradeFinalizationRecordV1,
) -> Result<(), ClockGovernedUpgradeFinalizationStateError> {
    if record.schema_version != CLOCK_GOVERNED_UPGRADE_FINALIZATION_RECORD_SCHEMA
        || record.finalization_sequence == 0
        || record.terminal_stage != UpgradeStage::Finalized
        || record.predecessor_upgrade_state_generation == 0
        || record.finalized_upgrade_state_generation
            != record
                .predecessor_upgrade_state_generation
                .checked_add(1)
                .ok_or(ClockGovernedUpgradeFinalizationStateError::GenerationOverflow)?
        || record.predecessor_upgrade_state_digest.0 == [0; 32]
        || record.handoff_plan_digest.0 == [0; 32]
        || record.fresh_checkpoint_digest.0 == [0; 32]
        || record.fresh_transparency_log_digest.0 == [0; 32]
        || record.hardware_refresh_set_digest.0 == [0; 32]
        || record.hardware_authority_count == 0
        || record.machine_ids.is_empty()
        || u64::try_from(record.machine_ids.len())
            .map_err(|_| ClockGovernedUpgradeFinalizationStateError::HardwareCountOverflow)?
            != record.hardware_authority_count
    {
        return Err(ClockGovernedUpgradeFinalizationStateError::InvalidRecord);
    }
    for digest in [
        record.successor_source_tree_digest,
        record.successor_executable_digest,
        record.successor_durable_state_digest,
        record.successor_replay_contract_digest,
    ] {
        if digest.0 == [0; 32] {
            return Err(ClockGovernedUpgradeFinalizationStateError::InvalidRecord);
        }
    }
    if record.handoff_id.trim().is_empty()
        || record.authorization_id.trim().is_empty()
        || record.context_id.trim().is_empty()
        || record.execution_permit_id.trim().is_empty()
        || record.state_binding_id.trim().is_empty()
        || record.fresh_clock_envelope_id.trim().is_empty()
        || record.fresh_operational_basis_id.trim().is_empty()
        || record.machine_ids.iter().any(|machine_id| {
            machine_id.trim().is_empty()
                || machine_id != machine_id.trim()
                || machine_id.chars().any(char::is_control)
        })
    {
        return Err(ClockGovernedUpgradeFinalizationStateError::InvalidRecord);
    }
    Ok(())
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, ClockGovernedUpgradeFinalizationStateError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ClockGovernedUpgradeFinalizationStateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
