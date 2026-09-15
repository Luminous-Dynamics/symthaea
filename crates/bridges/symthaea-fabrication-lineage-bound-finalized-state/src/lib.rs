// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic terminal authority for one lineage-bound fabrication upgrade.
//!
//! This crate consumes only the lineage-bound authorization, fresh execution permit, exact concrete
//! state binding and exact lineage-bound handoff. It emits a portable deterministic terminal record
//! plus an opaque live terminal capability. No caller-selected finalization timestamp participates.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::upgrade_handoff::{
    UpgradeEndpoint, digest_upgrade_endpoint,
};
use symthaea_fabrication_kernel::upgrade_tracker::UpgradeStage;
use symthaea_fabrication_lineage_bound_finalization_authorization::{
    AuthorizedLineageBoundFinalizationIdV1, AuthorizedLineageBoundFinalizationV1,
};
use symthaea_fabrication_lineage_bound_finalization_execution::{
    LineageBoundFinalizationExecutionPermitIdV1, LineageBoundFinalizationExecutionPermitV1,
};
use symthaea_fabrication_lineage_bound_finalization_state_binding::{
    LineageBoundFinalizationStateBindingIdV1, LineageBoundFinalizationStateBindingV1,
};
use symthaea_fabrication_lineage_bound_upgrade_handoff::{
    LineageBoundClockGovernedUpgradeHandoffIdV1, LineageBoundClockGovernedUpgradeHandoffV1,
};

pub const LINEAGE_BOUND_UPGRADE_FINALIZATION_RECORD_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-upgrade-finalization-record.v1";
pub const LINEAGE_BOUND_FINALIZED_UPGRADE_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-finalized-upgrade.v1";

const FINALIZATION_RECORD_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-upgrade-finalization-record.v1\0";
const FINALIZED_UPGRADE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalized-upgrade.v1\0";

/// Portable deterministic terminal evidence. This record is serializable audit data, not executable
/// authority by itself. Live terminal authority comes only from `LineageBoundFinalizedUpgradeV1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageBoundUpgradeFinalizationRecordV1 {
    pub schema_version: String,
    pub finalization_sequence: u64,
    pub predecessor_finalization_sequence: u64,
    pub terminal_stage: UpgradeStage,
    pub predecessor_root_digest: Sha256Digest,
    pub predecessor_current_head_digest: Sha256Digest,
    pub predecessor_upgrade_state_digest: Sha256Digest,
    pub predecessor_upgrade_state_generation: u64,
    pub finalized_upgrade_state_generation: u64,
    pub operational_state_digest: Sha256Digest,
    pub operational_lineage_digest: Sha256Digest,
    pub lineage_handoff_id: String,
    pub inner_handoff_id: String,
    pub handoff_plan_digest: Sha256Digest,
    pub predecessor_endpoint_digest: Sha256Digest,
    pub successor_endpoint_digest: Sha256Digest,
    pub successor_endpoint: UpgradeEndpoint,
    pub authorization_id: String,
    pub context_id: String,
    pub execution_permit_id: String,
    pub state_binding_id: String,
    pub no_rollback_id: String,
    pub fresh_checkpoint_digest: Sha256Digest,
    pub fresh_transparency_log_digest: Sha256Digest,
    pub fresh_clock_envelope_id: String,
    pub fresh_operational_basis_id: String,
    pub authorized_hardware_set_digest: Sha256Digest,
    pub hardware_refresh_set_digest: Sha256Digest,
    pub hardware_authority_count: u64,
    pub machine_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LineageBoundFinalizedUpgradeIdV1(Sha256Digest);

impl LineageBoundFinalizedUpgradeIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct LineageBoundFinalizedUpgradeV1 {
    id: LineageBoundFinalizedUpgradeIdV1,
    record: LineageBoundUpgradeFinalizationRecordV1,
    record_digest: Sha256Digest,
    authorization_id: AuthorizedLineageBoundFinalizationIdV1,
    execution_permit_id: LineageBoundFinalizationExecutionPermitIdV1,
    state_binding_id: LineageBoundFinalizationStateBindingIdV1,
    lineage_handoff_id: LineageBoundClockGovernedUpgradeHandoffIdV1,
}

impl LineageBoundFinalizedUpgradeV1 {
    pub fn id(&self) -> LineageBoundFinalizedUpgradeIdV1 { self.id }
    pub fn record(&self) -> &LineageBoundUpgradeFinalizationRecordV1 { &self.record }
    pub fn record_digest(&self) -> Sha256Digest { self.record_digest }
    pub fn authorization_id(&self) -> AuthorizedLineageBoundFinalizationIdV1 { self.authorization_id }
    pub fn execution_permit_id(&self) -> LineageBoundFinalizationExecutionPermitIdV1 { self.execution_permit_id }
    pub fn state_binding_id(&self) -> LineageBoundFinalizationStateBindingIdV1 { self.state_binding_id }
    pub fn lineage_handoff_id(&self) -> LineageBoundClockGovernedUpgradeHandoffIdV1 { self.lineage_handoff_id }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundFinalizedStateError {
    ExecutionAuthorizationMismatch,
    ExecutionContextMismatch,
    StateBindingAuthorizationMismatch,
    StateBindingExecutionMismatch,
    HandoffMismatch,
    HandoffPlanMismatch,
    PredecessorProvenanceMismatch,
    FinalizationSequenceOverflow,
    FinalizationSequenceMismatch { expected: u64, actual: u64 },
    GenerationOverflow,
    PredecessorEndpointInvalid(String),
    PredecessorEndpointDigestMismatch,
    SuccessorEndpointInvalid(String),
    HardwareCountOverflow,
    EmptyMachineSet,
    MachineCountMismatch,
    DuplicateMachine(String),
    InvalidRecord(&'static str),
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct FinalizedUpgradeCommitment {
    schema: &'static str,
    record_digest: String,
    authorization_id: String,
    execution_permit_id: String,
    state_binding_id: String,
    lineage_handoff_id: String,
    predecessor_root_digest: String,
    finalization_sequence: u64,
    successor_endpoint_digest: String,
}

pub fn finalize_lineage_bound_upgrade_v1(
    authorization: &AuthorizedLineageBoundFinalizationV1,
    execution_permit: &LineageBoundFinalizationExecutionPermitV1,
    state_binding: &LineageBoundFinalizationStateBindingV1,
    handoff: &LineageBoundClockGovernedUpgradeHandoffV1,
) -> Result<LineageBoundFinalizedUpgradeV1, LineageBoundFinalizedStateError> {
    let context = authorization.context();
    if execution_permit.authorization_id() != authorization.id() {
        return Err(LineageBoundFinalizedStateError::ExecutionAuthorizationMismatch);
    }
    if execution_permit.context_id() != context.id() {
        return Err(LineageBoundFinalizedStateError::ExecutionContextMismatch);
    }
    if state_binding.authorization_id() != authorization.id() {
        return Err(LineageBoundFinalizedStateError::StateBindingAuthorizationMismatch);
    }
    if state_binding.execution_permit_id() != execution_permit.id() {
        return Err(LineageBoundFinalizedStateError::StateBindingExecutionMismatch);
    }
    if context.lineage_handoff_id() != handoff.id() {
        return Err(LineageBoundFinalizedStateError::HandoffMismatch);
    }
    if context.handoff_plan_digest() != handoff.plan_digest()
        || state_binding.handoff_plan_digest() != handoff.plan_digest()
    {
        return Err(LineageBoundFinalizedStateError::HandoffPlanMismatch);
    }
    if execution_permit.predecessor_root_digest() != context.predecessor_root_digest()
        || state_binding.predecessor_root_digest() != context.predecessor_root_digest()
        || handoff.predecessor_root_id().as_digest() != context.predecessor_root_digest()
        || execution_permit.predecessor_current_head_digest() != context.predecessor_current_head_digest()
        || state_binding.predecessor_current_head_digest() != context.predecessor_current_head_digest()
        || handoff.current_head_id().as_digest() != context.predecessor_current_head_digest()
        || execution_permit.predecessor_finalization_sequence() != context.predecessor_finalization_sequence()
        || state_binding.predecessor_finalization_sequence() != context.predecessor_finalization_sequence()
        || handoff.predecessor_finalization_sequence() != context.predecessor_finalization_sequence()
    {
        return Err(LineageBoundFinalizedStateError::PredecessorProvenanceMismatch);
    }

    let expected_finalization_sequence = context
        .predecessor_finalization_sequence()
        .checked_add(1)
        .ok_or(LineageBoundFinalizedStateError::FinalizationSequenceOverflow)?;
    if context.upgrade_cycle_sequence() != expected_finalization_sequence
        || execution_permit.upgrade_cycle_sequence() != expected_finalization_sequence
        || state_binding.upgrade_cycle_sequence() != expected_finalization_sequence
        || state_binding.handoff_sequence() != expected_finalization_sequence
    {
        return Err(LineageBoundFinalizedStateError::FinalizationSequenceMismatch {
            expected: expected_finalization_sequence,
            actual: state_binding.handoff_sequence(),
        });
    }

    let predecessor_endpoint_digest = digest_upgrade_endpoint(&handoff.plan().predecessor)
        .map_err(|error| LineageBoundFinalizedStateError::PredecessorEndpointInvalid(format!("{error:?}")))?;
    if predecessor_endpoint_digest != handoff.predecessor_endpoint_digest() {
        return Err(LineageBoundFinalizedStateError::PredecessorEndpointDigestMismatch);
    }
    handoff
        .plan()
        .successor
        .validate()
        .map_err(|error| LineageBoundFinalizedStateError::SuccessorEndpointInvalid(format!("{error:?}")))?;
    let successor_endpoint_digest = digest_upgrade_endpoint(&handoff.plan().successor)
        .map_err(|error| LineageBoundFinalizedStateError::SuccessorEndpointInvalid(format!("{error:?}")))?;

    let finalized_upgrade_state_generation = state_binding
        .upgrade_state_generation()
        .checked_add(1)
        .ok_or(LineageBoundFinalizedStateError::GenerationOverflow)?;
    let hardware_authority_count = u64::try_from(execution_permit.hardware_authority_count())
        .map_err(|_| LineageBoundFinalizedStateError::HardwareCountOverflow)?;
    if execution_permit.machine_ids().is_empty() {
        return Err(LineageBoundFinalizedStateError::EmptyMachineSet);
    }
    if execution_permit.machine_ids().len() != execution_permit.hardware_authority_count() {
        return Err(LineageBoundFinalizedStateError::MachineCountMismatch);
    }
    let mut machine_ids = execution_permit.machine_ids().to_vec();
    machine_ids.sort();
    let unique = machine_ids.iter().cloned().collect::<BTreeSet<_>>();
    if unique.len() != machine_ids.len() {
        let duplicate = machine_ids
            .windows(2)
            .find(|pair| pair[0] == pair[1])
            .map(|pair| pair[0].clone())
            .unwrap_or_default();
        return Err(LineageBoundFinalizedStateError::DuplicateMachine(duplicate));
    }

    let record = LineageBoundUpgradeFinalizationRecordV1 {
        schema_version: LINEAGE_BOUND_UPGRADE_FINALIZATION_RECORD_SCHEMA.into(),
        finalization_sequence: expected_finalization_sequence,
        predecessor_finalization_sequence: context.predecessor_finalization_sequence(),
        terminal_stage: UpgradeStage::Finalized,
        predecessor_root_digest: context.predecessor_root_digest(),
        predecessor_current_head_digest: context.predecessor_current_head_digest(),
        predecessor_upgrade_state_digest: state_binding.upgrade_state_digest(),
        predecessor_upgrade_state_generation: state_binding.upgrade_state_generation(),
        finalized_upgrade_state_generation,
        operational_state_digest: state_binding.operational_state_digest(),
        operational_lineage_digest: state_binding.operational_lineage_digest(),
        lineage_handoff_id: handoff.id().to_hex(),
        inner_handoff_id: handoff.inner_handoff_id().to_hex(),
        handoff_plan_digest: handoff.plan_digest(),
        predecessor_endpoint_digest,
        successor_endpoint_digest,
        successor_endpoint: handoff.plan().successor.clone(),
        authorization_id: authorization.id().to_hex(),
        context_id: context.id().to_hex(),
        execution_permit_id: execution_permit.id().to_hex(),
        state_binding_id: state_binding.id().to_hex(),
        no_rollback_id: state_binding.no_rollback_id().to_hex(),
        fresh_checkpoint_digest: execution_permit.fresh_checkpoint_digest(),
        fresh_transparency_log_digest: execution_permit.fresh_transparency_log_digest(),
        fresh_clock_envelope_id: execution_permit.fresh_clock_envelope_id().to_hex(),
        fresh_operational_basis_id: execution_permit.fresh_operational_basis_id().to_hex(),
        authorized_hardware_set_digest: execution_permit.authorized_hardware_set_digest(),
        hardware_refresh_set_digest: execution_permit.hardware_refresh_set_digest(),
        hardware_authority_count,
        machine_ids,
    };
    validate_record(&record)?;
    let record_digest = digest_lineage_bound_finalization_record_v1(&record)?;

    let commitment = FinalizedUpgradeCommitment {
        schema: LINEAGE_BOUND_FINALIZED_UPGRADE_SCHEMA,
        record_digest: record_digest.to_hex(),
        authorization_id: authorization.id().to_hex(),
        execution_permit_id: execution_permit.id().to_hex(),
        state_binding_id: state_binding.id().to_hex(),
        lineage_handoff_id: handoff.id().to_hex(),
        predecessor_root_digest: context.predecessor_root_digest().to_hex(),
        finalization_sequence: expected_finalization_sequence,
        successor_endpoint_digest: successor_endpoint_digest.to_hex(),
    };
    let id = LineageBoundFinalizedUpgradeIdV1(hash_serializable(
        FINALIZED_UPGRADE_DOMAIN,
        &commitment,
    )?);

    Ok(LineageBoundFinalizedUpgradeV1 {
        id,
        record,
        record_digest,
        authorization_id: authorization.id(),
        execution_permit_id: execution_permit.id(),
        state_binding_id: state_binding.id(),
        lineage_handoff_id: handoff.id(),
    })
}

pub fn digest_lineage_bound_finalization_record_v1(
    record: &LineageBoundUpgradeFinalizationRecordV1,
) -> Result<Sha256Digest, LineageBoundFinalizedStateError> {
    validate_record(record)?;
    hash_serializable(FINALIZATION_RECORD_DOMAIN, record)
}

fn validate_record(
    record: &LineageBoundUpgradeFinalizationRecordV1,
) -> Result<(), LineageBoundFinalizedStateError> {
    if record.schema_version != LINEAGE_BOUND_UPGRADE_FINALIZATION_RECORD_SCHEMA {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("schema"));
    }
    if record.terminal_stage != UpgradeStage::Finalized {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("terminal_stage"));
    }
    if record.finalization_sequence == 0 || record.predecessor_finalization_sequence == 0 {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("sequence"));
    }
    if record.predecessor_finalization_sequence.checked_add(1) != Some(record.finalization_sequence) {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("sequence_adjacency"));
    }
    if record.predecessor_upgrade_state_generation == 0
        || record.predecessor_upgrade_state_generation.checked_add(1)
            != Some(record.finalized_upgrade_state_generation)
    {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("generation_adjacency"));
    }
    for (name, digest) in [
        ("predecessor_root_digest", record.predecessor_root_digest),
        ("predecessor_current_head_digest", record.predecessor_current_head_digest),
        ("predecessor_upgrade_state_digest", record.predecessor_upgrade_state_digest),
        ("operational_state_digest", record.operational_state_digest),
        ("operational_lineage_digest", record.operational_lineage_digest),
        ("handoff_plan_digest", record.handoff_plan_digest),
        ("predecessor_endpoint_digest", record.predecessor_endpoint_digest),
        ("successor_endpoint_digest", record.successor_endpoint_digest),
        ("fresh_checkpoint_digest", record.fresh_checkpoint_digest),
        ("fresh_transparency_log_digest", record.fresh_transparency_log_digest),
        ("authorized_hardware_set_digest", record.authorized_hardware_set_digest),
        ("hardware_refresh_set_digest", record.hardware_refresh_set_digest),
    ] {
        if digest.0 == [0; 32] {
            return Err(LineageBoundFinalizedStateError::InvalidRecord(name));
        }
    }
    if record.lineage_handoff_id.trim().is_empty()
        || record.inner_handoff_id.trim().is_empty()
        || record.authorization_id.trim().is_empty()
        || record.context_id.trim().is_empty()
        || record.execution_permit_id.trim().is_empty()
        || record.state_binding_id.trim().is_empty()
        || record.no_rollback_id.trim().is_empty()
        || record.fresh_clock_envelope_id.trim().is_empty()
        || record.fresh_operational_basis_id.trim().is_empty()
    {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("identifier"));
    }
    if record.hardware_authority_count == 0
        || usize::try_from(record.hardware_authority_count).ok() != Some(record.machine_ids.len())
    {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("hardware_count"));
    }
    let unique = record.machine_ids.iter().collect::<BTreeSet<_>>();
    if unique.len() != record.machine_ids.len() {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("duplicate_machine"));
    }
    record
        .successor_endpoint
        .validate()
        .map_err(|_| LineageBoundFinalizedStateError::InvalidRecord("successor_endpoint"))?;
    let successor_digest = digest_upgrade_endpoint(&record.successor_endpoint)
        .map_err(|_| LineageBoundFinalizedStateError::InvalidRecord("successor_endpoint_digest"))?;
    if successor_digest != record.successor_endpoint_digest {
        return Err(LineageBoundFinalizedStateError::InvalidRecord("successor_endpoint_digest"));
    }
    Ok(())
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundFinalizedStateError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundFinalizedStateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
