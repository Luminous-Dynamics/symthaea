// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Append-only distributed currentness for deterministic lineage-bound finalized upgrades.
//!
//! Local terminal authority is not distributed finality. This bridge publishes one exact
//! `LineageBoundFinalizedUpgradeV1` into a lineage-specific transparency namespace, requires a strict
//! append-only descendant of the exact execution log, observes it under definitely later trusted
//! time, permits idempotent duplicate publication of the exact same digest, and rejects any
//! conflicting terminal claim or any lineage-head prepublication already present before execution.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::transparency::{TransparencyLog, digest_transparency_log};
use symthaea_fabrication_lineage_bound_finalization_execution::{
    LineageBoundFinalizationExecutionPermitIdV1, LineageBoundFinalizationExecutionPermitV1,
};
use symthaea_fabrication_lineage_bound_finalized_state::{
    LineageBoundFinalizedUpgradeIdV1, LineageBoundFinalizedUpgradeV1,
};
use symthaea_fabrication_witness_registry_containment_bound::{
    ContainmentCurrentWitnessRegistryHeadIdV1, ContainmentCurrentWitnessRegistryHeadV1,
};
use symthaea_trust_kernel::{
    ClockGovernanceEvaluationEnvelopeIdV1, ClockGovernanceTimeError, OperationalClockBasisIdV1,
    OperationalClockBasisV1, derive_clock_governance_evaluation_envelope_v1,
};

pub const LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_PUBLICATION_SCHEMA: &str =
    "symthaea.fabrication.lineage-bound-finalized-upgrade-head-publication.v1";
pub const CURRENT_LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_SCHEMA: &str =
    "symthaea.fabrication.current-lineage-bound-finalized-upgrade-head.v1";
pub const LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX: &str =
    "lineage-finalized-upgrade-head-v1:";
pub const MAX_LINEAGE_BOUND_FINALIZED_HEAD_CLOCK_HOPS: usize = 4096;

const PUBLICATION_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalized-upgrade-head-publication.v1\0";
const LOG_KIND_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalized-upgrade-head-kind.v1\0";
const CLOCK_LINEAGE_DOMAIN: &[u8] =
    b"symthaea.fabrication.lineage-bound-finalized-upgrade-head-clock-lineage.v1\0";
const CURRENT_HEAD_DOMAIN: &[u8] =
    b"symthaea.fabrication.current-lineage-bound-finalized-upgrade-head.v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageBoundFinalizedUpgradeHeadPublicationV1 {
    pub schema_version: String,
    pub finalized_upgrade_id: String,
    pub record_digest: Sha256Digest,
    pub finalization_sequence: u64,
    pub predecessor_finalization_sequence: u64,
    pub predecessor_root_digest: Sha256Digest,
    pub predecessor_current_head_digest: Sha256Digest,
    pub lineage_handoff_id: String,
    pub handoff_plan_digest: Sha256Digest,
    pub predecessor_upgrade_state_digest: Sha256Digest,
    pub predecessor_upgrade_state_generation: u64,
    pub finalized_upgrade_state_generation: u64,
    pub operational_state_digest: Sha256Digest,
    pub operational_lineage_digest: Sha256Digest,
    pub successor_endpoint_digest: Sha256Digest,
    pub authorization_id: String,
    pub execution_permit_id: String,
    pub state_binding_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CurrentLineageBoundFinalizedUpgradeHeadIdV1(Sha256Digest);

impl CurrentLineageBoundFinalizedUpgradeHeadIdV1 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct CurrentLineageBoundFinalizedUpgradeHeadV1 {
    id: CurrentLineageBoundFinalizedUpgradeHeadIdV1,
    governance_view_id: ContainmentCurrentWitnessRegistryHeadIdV1,
    finalized_upgrade_id: LineageBoundFinalizedUpgradeIdV1,
    execution_permit_id: LineageBoundFinalizationExecutionPermitIdV1,
    record_digest: Sha256Digest,
    handoff_plan_digest: Sha256Digest,
    finalization_sequence: u64,
    predecessor_finalization_sequence: u64,
    predecessor_root_digest: Sha256Digest,
    predecessor_current_head_digest: Sha256Digest,
    predecessor_upgrade_state_digest: Sha256Digest,
    predecessor_upgrade_state_generation: u64,
    finalized_upgrade_state_generation: u64,
    operational_state_digest: Sha256Digest,
    operational_lineage_digest: Sha256Digest,
    successor_endpoint_digest: Sha256Digest,
    publication_digest: Sha256Digest,
    publication_count: usize,
    latest_publication_entry_sequence: u64,
    execution_transparency_log_digest: Sha256Digest,
    execution_log_size: usize,
    current_transparency_log_digest: Sha256Digest,
    current_log_size: usize,
    appended_entry_count: usize,
    execution_checkpoint_digest: Sha256Digest,
    current_checkpoint_digest: Sha256Digest,
    execution_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    current_clock_envelope_id: ClockGovernanceEvaluationEnvelopeIdV1,
    execution_operational_basis_id: OperationalClockBasisIdV1,
    current_operational_basis_id: OperationalClockBasisIdV1,
    clock_lineage_digest: Sha256Digest,
    clock_hop_count: usize,
}

impl CurrentLineageBoundFinalizedUpgradeHeadV1 {
    pub fn id(&self) -> CurrentLineageBoundFinalizedUpgradeHeadIdV1 { self.id }
    pub fn governance_view_id(&self) -> ContainmentCurrentWitnessRegistryHeadIdV1 { self.governance_view_id }
    pub fn finalized_upgrade_id(&self) -> LineageBoundFinalizedUpgradeIdV1 { self.finalized_upgrade_id }
    pub fn execution_permit_id(&self) -> LineageBoundFinalizationExecutionPermitIdV1 { self.execution_permit_id }
    pub fn record_digest(&self) -> Sha256Digest { self.record_digest }
    pub fn handoff_plan_digest(&self) -> Sha256Digest { self.handoff_plan_digest }
    pub fn finalization_sequence(&self) -> u64 { self.finalization_sequence }
    pub fn predecessor_finalization_sequence(&self) -> u64 { self.predecessor_finalization_sequence }
    pub fn predecessor_root_digest(&self) -> Sha256Digest { self.predecessor_root_digest }
    pub fn predecessor_current_head_digest(&self) -> Sha256Digest { self.predecessor_current_head_digest }
    pub fn predecessor_upgrade_state_digest(&self) -> Sha256Digest { self.predecessor_upgrade_state_digest }
    pub fn predecessor_upgrade_state_generation(&self) -> u64 { self.predecessor_upgrade_state_generation }
    pub fn finalized_upgrade_state_generation(&self) -> u64 { self.finalized_upgrade_state_generation }
    pub fn operational_state_digest(&self) -> Sha256Digest { self.operational_state_digest }
    pub fn operational_lineage_digest(&self) -> Sha256Digest { self.operational_lineage_digest }
    pub fn successor_endpoint_digest(&self) -> Sha256Digest { self.successor_endpoint_digest }
    pub fn publication_digest(&self) -> Sha256Digest { self.publication_digest }
    pub fn publication_count(&self) -> usize { self.publication_count }
    pub fn latest_publication_entry_sequence(&self) -> u64 { self.latest_publication_entry_sequence }
    pub fn execution_transparency_log_digest(&self) -> Sha256Digest { self.execution_transparency_log_digest }
    pub fn execution_log_size(&self) -> usize { self.execution_log_size }
    pub fn current_transparency_log_digest(&self) -> Sha256Digest { self.current_transparency_log_digest }
    pub fn current_log_size(&self) -> usize { self.current_log_size }
    pub fn appended_entry_count(&self) -> usize { self.appended_entry_count }
    pub fn execution_checkpoint_digest(&self) -> Sha256Digest { self.execution_checkpoint_digest }
    pub fn current_checkpoint_digest(&self) -> Sha256Digest { self.current_checkpoint_digest }
    pub fn execution_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.execution_clock_envelope_id }
    pub fn current_clock_envelope_id(&self) -> ClockGovernanceEvaluationEnvelopeIdV1 { self.current_clock_envelope_id }
    pub fn execution_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.execution_operational_basis_id }
    pub fn current_operational_basis_id(&self) -> OperationalClockBasisIdV1 { self.current_operational_basis_id }
    pub fn clock_lineage_digest(&self) -> Sha256Digest { self.clock_lineage_digest }
    pub fn clock_hop_count(&self) -> usize { self.clock_hop_count }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBoundFinalizedHeadError {
    InvalidPublication,
    PublicationMismatch,
    ExecutionPermitMismatch,
    ExecutionLogInvalid(String),
    ExecutionLogMismatch,
    CurrentLogInvalid(String),
    CurrentLogMismatch,
    LogNotStrictExtension,
    GovernanceObservationBasisMismatch,
    GovernanceObservationEnvelopeMismatch,
    ExecutionBasisMismatch,
    ExecutionEnvelopeMismatch,
    TooManyClockHops { actual: usize, maximum: usize },
    BrokenClockLineage { hop: usize, expected_predecessor: String, actual_predecessor: Option<String> },
    Clock(ClockGovernanceTimeError),
    ObservationClockNotDefinitelyLater { execution_upper_unix_ms: u64, observation_lower_unix_ms: u64 },
    CheckpointNotAdvanced,
    LineageHeadPrepublished { entry_sequence: u64 },
    PublicationNotFound,
    ConflictingFinalizationPublished { entry_sequence: u64, subject_digest: Sha256Digest },
    PublicationBeforeExecution { entry_sequence: u64 },
    PublicationMayBeFuture { entry_sequence: u64 },
    TimeScaleOverflow,
    Encoding(String),
}

#[derive(Debug, Clone, Serialize)]
struct ClockLineageCommitment {
    execution_basis_id: String,
    bridge_basis_ids: Vec<String>,
    observation_basis_id: String,
}

#[derive(Debug, Clone, Serialize)]
struct CurrentHeadCommitment {
    schema: &'static str,
    governance_view_id: String,
    finalized_upgrade_id: String,
    execution_permit_id: String,
    record_digest: String,
    handoff_plan_digest: String,
    finalization_sequence: u64,
    predecessor_finalization_sequence: u64,
    predecessor_root_digest: String,
    predecessor_current_head_digest: String,
    predecessor_upgrade_state_digest: String,
    predecessor_upgrade_state_generation: u64,
    finalized_upgrade_state_generation: u64,
    operational_state_digest: String,
    operational_lineage_digest: String,
    successor_endpoint_digest: String,
    publication_digest: String,
    publication_count: usize,
    latest_publication_entry_sequence: u64,
    execution_transparency_log_digest: String,
    execution_log_size: usize,
    current_transparency_log_digest: String,
    current_log_size: usize,
    appended_entry_count: usize,
    execution_checkpoint_digest: String,
    current_checkpoint_digest: String,
    execution_clock_envelope_id: String,
    current_clock_envelope_id: String,
    execution_operational_basis_id: String,
    current_operational_basis_id: String,
    clock_lineage_digest: String,
    clock_hop_count: usize,
}

pub fn build_lineage_bound_finalized_upgrade_head_publication_v1(
    finalized: &LineageBoundFinalizedUpgradeV1,
) -> LineageBoundFinalizedUpgradeHeadPublicationV1 {
    let record = finalized.record();
    LineageBoundFinalizedUpgradeHeadPublicationV1 {
        schema_version: LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_PUBLICATION_SCHEMA.into(),
        finalized_upgrade_id: finalized.id().to_hex(),
        record_digest: finalized.record_digest(),
        finalization_sequence: record.finalization_sequence,
        predecessor_finalization_sequence: record.predecessor_finalization_sequence,
        predecessor_root_digest: record.predecessor_root_digest,
        predecessor_current_head_digest: record.predecessor_current_head_digest,
        lineage_handoff_id: record.lineage_handoff_id.clone(),
        handoff_plan_digest: record.handoff_plan_digest,
        predecessor_upgrade_state_digest: record.predecessor_upgrade_state_digest,
        predecessor_upgrade_state_generation: record.predecessor_upgrade_state_generation,
        finalized_upgrade_state_generation: record.finalized_upgrade_state_generation,
        operational_state_digest: record.operational_state_digest,
        operational_lineage_digest: record.operational_lineage_digest,
        successor_endpoint_digest: record.successor_endpoint_digest,
        authorization_id: record.authorization_id.clone(),
        execution_permit_id: record.execution_permit_id.clone(),
        state_binding_id: record.state_binding_id.clone(),
    }
}

pub fn digest_lineage_bound_finalized_upgrade_head_publication_v1(
    publication: &LineageBoundFinalizedUpgradeHeadPublicationV1,
) -> Result<Sha256Digest, LineageBoundFinalizedHeadError> {
    validate_publication(publication)?;
    hash_serializable(PUBLICATION_DOMAIN, publication)
}

pub fn lineage_bound_finalized_upgrade_head_log_kind(
    handoff_plan_digest: Sha256Digest,
) -> Result<String, LineageBoundFinalizedHeadError> {
    if handoff_plan_digest.0 == [0; 32] {
        return Err(LineageBoundFinalizedHeadError::InvalidPublication);
    }
    let mut hasher = Sha256::new();
    hasher.update(LOG_KIND_DOMAIN);
    hasher.update(&handoff_plan_digest.0);
    Ok(format!(
        "{LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX}{}",
        hasher.finalize().to_hex()
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_current_lineage_bound_finalized_upgrade_head_v1(
    finalized: &LineageBoundFinalizedUpgradeV1,
    execution_permit: &LineageBoundFinalizationExecutionPermitV1,
    publication: &LineageBoundFinalizedUpgradeHeadPublicationV1,
    execution_log: &TransparencyLog,
    current_log: &TransparencyLog,
    execution_basis: &OperationalClockBasisV1,
    execution_to_observation_clock_bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
    governance_view: &ContainmentCurrentWitnessRegistryHeadV1,
) -> Result<CurrentLineageBoundFinalizedUpgradeHeadV1, Vec<LineageBoundFinalizedHeadError>> {
    let mut violations = Vec::new();
    let record = finalized.record();

    if finalized.execution_permit_id() != execution_permit.id()
        || record.execution_permit_id != execution_permit.id().to_hex()
        || record.fresh_checkpoint_digest != execution_permit.fresh_checkpoint_digest()
        || record.fresh_transparency_log_digest != execution_permit.fresh_transparency_log_digest()
        || record.fresh_clock_envelope_id != execution_permit.fresh_clock_envelope_id().to_hex()
        || record.fresh_operational_basis_id != execution_permit.fresh_operational_basis_id().to_hex()
    {
        violations.push(LineageBoundFinalizedHeadError::ExecutionPermitMismatch);
    }

    let expected_publication = build_lineage_bound_finalized_upgrade_head_publication_v1(finalized);
    if publication != &expected_publication {
        violations.push(LineageBoundFinalizedHeadError::PublicationMismatch);
    }
    let publication_digest = match digest_lineage_bound_finalized_upgrade_head_publication_v1(publication) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            Sha256Digest([0; 32])
        }
    };

    if let Err(error) = execution_log.validate() {
        violations.push(LineageBoundFinalizedHeadError::ExecutionLogInvalid(format!("{error:?}")));
    }
    let execution_log_digest = match digest_transparency_log(execution_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizedHeadError::ExecutionLogInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    if execution_log_digest != execution_permit.fresh_transparency_log_digest()
        || execution_log_digest != record.fresh_transparency_log_digest
    {
        violations.push(LineageBoundFinalizedHeadError::ExecutionLogMismatch);
    }

    if let Err(error) = current_log.validate() {
        violations.push(LineageBoundFinalizedHeadError::CurrentLogInvalid(format!("{error:?}")));
    }
    let current_log_digest = match digest_transparency_log(current_log) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizedHeadError::CurrentLogInvalid(format!("{error:?}")));
            Sha256Digest([0; 32])
        }
    };
    if current_log_digest != governance_view.transparency_log_digest() {
        violations.push(LineageBoundFinalizedHeadError::CurrentLogMismatch);
    }
    if current_log.entries.len() <= execution_log.entries.len()
        || current_log.verify_successor_of(execution_log).is_err()
    {
        violations.push(LineageBoundFinalizedHeadError::LogNotStrictExtension);
    }
    if governance_view.checkpoint_digest() == execution_permit.fresh_checkpoint_digest() {
        violations.push(LineageBoundFinalizedHeadError::CheckpointNotAdvanced);
    }

    if execution_basis.id() != execution_permit.fresh_operational_basis_id() {
        violations.push(LineageBoundFinalizedHeadError::ExecutionBasisMismatch);
    }
    let execution_clock = match derive_clock_governance_evaluation_envelope_v1(execution_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizedHeadError::Clock(error));
            return Err(violations);
        }
    };
    if execution_clock.id() != execution_permit.fresh_clock_envelope_id() {
        violations.push(LineageBoundFinalizedHeadError::ExecutionEnvelopeMismatch);
    }

    if observation_basis.id() != governance_view.observation_operational_basis_id() {
        violations.push(LineageBoundFinalizedHeadError::GovernanceObservationBasisMismatch);
    }
    let observation_clock = match derive_clock_governance_evaluation_envelope_v1(observation_basis) {
        Ok(value) => value,
        Err(error) => {
            violations.push(LineageBoundFinalizedHeadError::Clock(error));
            return Err(violations);
        }
    };
    if observation_clock.id() != governance_view.observation_clock_envelope_id() {
        violations.push(LineageBoundFinalizedHeadError::GovernanceObservationEnvelopeMismatch);
    }

    if execution_to_observation_clock_bridge.len() > MAX_LINEAGE_BOUND_FINALIZED_HEAD_CLOCK_HOPS {
        violations.push(LineageBoundFinalizedHeadError::TooManyClockHops {
            actual: execution_to_observation_clock_bridge.len(),
            maximum: MAX_LINEAGE_BOUND_FINALIZED_HEAD_CLOCK_HOPS,
        });
    } else if let Err(error) = verify_clock_lineage(
        execution_basis.id(),
        execution_to_observation_clock_bridge,
        observation_basis,
    ) {
        violations.push(error);
    }
    if observation_clock.lower_unix_ms() <= execution_clock.upper_unix_ms() {
        violations.push(LineageBoundFinalizedHeadError::ObservationClockNotDefinitelyLater {
            execution_upper_unix_ms: execution_clock.upper_unix_ms(),
            observation_lower_unix_ms: observation_clock.lower_unix_ms(),
        });
    }

    let log_kind = match lineage_bound_finalized_upgrade_head_log_kind(record.handoff_plan_digest) {
        Ok(value) => value,
        Err(error) => {
            violations.push(error);
            return Err(violations);
        }
    };
    for entry in &execution_log.entries {
        if entry.kind == log_kind {
            violations.push(LineageBoundFinalizedHeadError::LineageHeadPrepublished {
                entry_sequence: entry.sequence,
            });
        }
    }

    let matching_entries = current_log
        .entries
        .iter()
        .filter(|entry| entry.kind == log_kind)
        .collect::<Vec<_>>();
    if matching_entries.is_empty() {
        violations.push(LineageBoundFinalizedHeadError::PublicationNotFound);
    }
    let mut latest_publication_entry_sequence = 0u64;
    for entry in &matching_entries {
        if entry.subject_digest != publication_digest {
            violations.push(LineageBoundFinalizedHeadError::ConflictingFinalizationPublished {
                entry_sequence: entry.sequence,
                subject_digest: entry.subject_digest,
            });
        }
        if usize::try_from(entry.sequence).ok().is_none_or(|sequence| sequence <= execution_log.entries.len()) {
            violations.push(LineageBoundFinalizedHeadError::PublicationBeforeExecution {
                entry_sequence: entry.sequence,
            });
        }
        let recorded_at_unix_ms = match entry.recorded_at_unix_s.checked_mul(1_000) {
            Some(value) => value,
            None => {
                violations.push(LineageBoundFinalizedHeadError::TimeScaleOverflow);
                continue;
            }
        };
        if recorded_at_unix_ms <= execution_clock.upper_unix_ms() {
            violations.push(LineageBoundFinalizedHeadError::PublicationBeforeExecution {
                entry_sequence: entry.sequence,
            });
        }
        if recorded_at_unix_ms > observation_clock.lower_unix_ms() {
            violations.push(LineageBoundFinalizedHeadError::PublicationMayBeFuture {
                entry_sequence: entry.sequence,
            });
        }
        latest_publication_entry_sequence = latest_publication_entry_sequence.max(entry.sequence);
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    let (clock_lineage_digest, clock_hop_count) = digest_clock_lineage(
        execution_basis,
        execution_to_observation_clock_bridge,
        observation_basis,
    )
    .map_err(|error| vec![error])?;
    let appended_entry_count = current_log.entries.len() - execution_log.entries.len();
    let commitment = CurrentHeadCommitment {
        schema: CURRENT_LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_SCHEMA,
        governance_view_id: governance_view.id().to_hex(),
        finalized_upgrade_id: finalized.id().to_hex(),
        execution_permit_id: execution_permit.id().to_hex(),
        record_digest: finalized.record_digest().to_hex(),
        handoff_plan_digest: record.handoff_plan_digest.to_hex(),
        finalization_sequence: record.finalization_sequence,
        predecessor_finalization_sequence: record.predecessor_finalization_sequence,
        predecessor_root_digest: record.predecessor_root_digest.to_hex(),
        predecessor_current_head_digest: record.predecessor_current_head_digest.to_hex(),
        predecessor_upgrade_state_digest: record.predecessor_upgrade_state_digest.to_hex(),
        predecessor_upgrade_state_generation: record.predecessor_upgrade_state_generation,
        finalized_upgrade_state_generation: record.finalized_upgrade_state_generation,
        operational_state_digest: record.operational_state_digest.to_hex(),
        operational_lineage_digest: record.operational_lineage_digest.to_hex(),
        successor_endpoint_digest: record.successor_endpoint_digest.to_hex(),
        publication_digest: publication_digest.to_hex(),
        publication_count: matching_entries.len(),
        latest_publication_entry_sequence,
        execution_transparency_log_digest: execution_log_digest.to_hex(),
        execution_log_size: execution_log.entries.len(),
        current_transparency_log_digest: current_log_digest.to_hex(),
        current_log_size: current_log.entries.len(),
        appended_entry_count,
        execution_checkpoint_digest: execution_permit.fresh_checkpoint_digest().to_hex(),
        current_checkpoint_digest: governance_view.checkpoint_digest().to_hex(),
        execution_clock_envelope_id: execution_clock.id().to_hex(),
        current_clock_envelope_id: observation_clock.id().to_hex(),
        execution_operational_basis_id: execution_basis.id().to_hex(),
        current_operational_basis_id: observation_basis.id().to_hex(),
        clock_lineage_digest: clock_lineage_digest.to_hex(),
        clock_hop_count,
    };
    let id = CurrentLineageBoundFinalizedUpgradeHeadIdV1(
        hash_serializable(CURRENT_HEAD_DOMAIN, &commitment).map_err(|error| vec![error])?,
    );

    Ok(CurrentLineageBoundFinalizedUpgradeHeadV1 {
        id,
        governance_view_id: governance_view.id(),
        finalized_upgrade_id: finalized.id(),
        execution_permit_id: execution_permit.id(),
        record_digest: finalized.record_digest(),
        handoff_plan_digest: record.handoff_plan_digest,
        finalization_sequence: record.finalization_sequence,
        predecessor_finalization_sequence: record.predecessor_finalization_sequence,
        predecessor_root_digest: record.predecessor_root_digest,
        predecessor_current_head_digest: record.predecessor_current_head_digest,
        predecessor_upgrade_state_digest: record.predecessor_upgrade_state_digest,
        predecessor_upgrade_state_generation: record.predecessor_upgrade_state_generation,
        finalized_upgrade_state_generation: record.finalized_upgrade_state_generation,
        operational_state_digest: record.operational_state_digest,
        operational_lineage_digest: record.operational_lineage_digest,
        successor_endpoint_digest: record.successor_endpoint_digest,
        publication_digest,
        publication_count: matching_entries.len(),
        latest_publication_entry_sequence,
        execution_transparency_log_digest: execution_log_digest,
        execution_log_size: execution_log.entries.len(),
        current_transparency_log_digest: current_log_digest,
        current_log_size: current_log.entries.len(),
        appended_entry_count,
        execution_checkpoint_digest: execution_permit.fresh_checkpoint_digest(),
        current_checkpoint_digest: governance_view.checkpoint_digest(),
        execution_clock_envelope_id: execution_clock.id(),
        current_clock_envelope_id: observation_clock.id(),
        execution_operational_basis_id: execution_basis.id(),
        current_operational_basis_id: observation_basis.id(),
        clock_lineage_digest,
        clock_hop_count,
    })
}

fn validate_publication(
    publication: &LineageBoundFinalizedUpgradeHeadPublicationV1,
) -> Result<(), LineageBoundFinalizedHeadError> {
    if publication.schema_version != LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_PUBLICATION_SCHEMA
        || !canonical_hex_id(&publication.finalized_upgrade_id)
        || !canonical_hex_id(&publication.lineage_handoff_id)
        || !canonical_hex_id(&publication.authorization_id)
        || !canonical_hex_id(&publication.execution_permit_id)
        || !canonical_hex_id(&publication.state_binding_id)
        || publication.finalization_sequence == 0
        || publication.predecessor_finalization_sequence == 0
        || publication.predecessor_finalization_sequence.checked_add(1)
            != Some(publication.finalization_sequence)
        || publication.predecessor_upgrade_state_generation == 0
        || publication.predecessor_upgrade_state_generation.checked_add(1)
            != Some(publication.finalized_upgrade_state_generation)
    {
        return Err(LineageBoundFinalizedHeadError::InvalidPublication);
    }
    for digest in [
        publication.record_digest,
        publication.predecessor_root_digest,
        publication.predecessor_current_head_digest,
        publication.handoff_plan_digest,
        publication.predecessor_upgrade_state_digest,
        publication.operational_state_digest,
        publication.operational_lineage_digest,
        publication.successor_endpoint_digest,
    ] {
        if digest.0 == [0; 32] {
            return Err(LineageBoundFinalizedHeadError::InvalidPublication);
        }
    }
    Ok(())
}

fn verify_clock_lineage(
    prior_basis_id: OperationalClockBasisIdV1,
    bridge: &[OperationalClockBasisV1],
    current_basis: &OperationalClockBasisV1,
) -> Result<(), LineageBoundFinalizedHeadError> {
    if current_basis.id() == prior_basis_id {
        if bridge.is_empty() {
            return Ok(());
        }
        return Err(LineageBoundFinalizedHeadError::BrokenClockLineage {
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
            return Err(LineageBoundFinalizedHeadError::BrokenClockLineage {
                hop: index + 1,
                expected_predecessor: expected.to_hex(),
                actual_predecessor: actual.map(|value| value.to_hex()),
            });
        }
        expected = basis.id();
    }
    let actual = current_basis.predecessor_operational_basis_id();
    if actual != Some(expected) {
        return Err(LineageBoundFinalizedHeadError::BrokenClockLineage {
            hop: bridge.len() + 1,
            expected_predecessor: expected.to_hex(),
            actual_predecessor: actual.map(|value| value.to_hex()),
        });
    }
    Ok(())
}

fn digest_clock_lineage(
    execution_basis: &OperationalClockBasisV1,
    bridge: &[OperationalClockBasisV1],
    observation_basis: &OperationalClockBasisV1,
) -> Result<(Sha256Digest, usize), LineageBoundFinalizedHeadError> {
    let mut ids = Vec::with_capacity(bridge.len() + 2);
    ids.push(execution_basis.id().to_hex());
    ids.extend(bridge.iter().map(|basis| basis.id().to_hex()));
    if observation_basis.id() != execution_basis.id() {
        ids.push(observation_basis.id().to_hex());
    }
    let digest = hash_serializable(CLOCK_LINEAGE_DOMAIN, &ids)?;
    let hops = if observation_basis.id() == execution_basis.id() {
        0
    } else {
        bridge.len() + 1
    };
    Ok((digest, hops))
}

fn canonical_hex_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_serializable<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, LineageBoundFinalizedHeadError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| LineageBoundFinalizedHeadError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}
