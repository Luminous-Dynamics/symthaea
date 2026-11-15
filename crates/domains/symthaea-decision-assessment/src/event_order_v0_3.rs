// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic virtual-time schedule and fault replay for event-order research.
//!
//! This module generates and replays synthetic schedules. It is deliberately
//! non-authoritative: it does not authenticate owner receipts, convert clocks,
//! prove physical truth, or grant effect authority. The v0.2 evaluator remains
//! the decision oracle; this layer makes its inputs reproducible.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{EventOrder, EventTraceDisposition, GateState, ReconciledOperationOutcome};
use crate::event_order_v0_2::{
    AttemptHistoryInventory, AttemptIdentity, AttemptLifecycleReport, AttemptStopDischargeRelation,
    AttemptTerminalReconciliation, BitemporalEventEnvelope, OperationAttemptRecord,
    QualifiedEventTime, ScopedStopObligation, StartGateReport, StopDischargeReceipt, StopScope,
    assess_start_under_stop_obligations, compare_qualified_event_times, validate_attempt_lifecycle,
};

pub const SCHEDULE_SCHEMA_VERSION: &str = "symthaea.event-order-schedule.v0.3.2";
pub const GENERATOR_VERSION: &str = "symthaea.event-order-generator.v0.3.3";
pub const MAX_GENERATED_EVENTS: usize = 4_096;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtualSchedule {
    pub schema_version: String,
    pub generator_version: String,
    pub seed: u64,
    pub scenario_family: String,
    /// Input array order is intentionally non-semantic. The canonical scheduler
    /// uses (at_ns, ordinal); ordinal is an explicit schedule identity.
    pub actions: Vec<ScheduledAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionEvidence {
    AttemptStart {
        retry_of_attempt_id: Option<String>,
    },
    EffectCommit {
        receipt_ref: String,
        receipt_sha256: String,
    },
    CancellationAcknowledgement {
        receipt_ref: String,
        receipt_sha256: String,
    },
    TerminalReconciliation {
        exact_start_action_id: String,
        qualification: GateState,
    },
    StopTrigger {
        scope: StopScope,
        receipt_ref: String,
        receipt_sha256: String,
    },
    StopPolicyRevision {
        scope: StopScope,
        receipt_ref: String,
        receipt_sha256: String,
    },
    StopDischarge {
        scope: StopScope,
        inventory_frontier_event_id: String,
        attempt_manifest_sha256: String,
    },
    InventoryAttestation {
        scope: StopScope,
        inventory_frontier_event_id: String,
        attempt_manifest_sha256: String,
        receipt_ref: String,
        receipt_sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledAction {
    pub action_id: String,
    pub at_ns: u64,
    pub ordinal: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner_time_ref: Option<ActionClockTimeReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<ActionEvidence>,
    pub action: ScheduleAction,
}

/// The tagged action vocabulary is explicit so frozen schedules remain
/// inspectable and replayable without executing arbitrary code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScheduleAction {
    ClockAdvance {
        clock_domain: String,
        basis_id: String,
        epoch: String,
        scale: String,
        unit: String,
        local_tick_ns: u64,
        offset_ns: i64,
        uncertainty_ns: u64,
        calibration_id: String,
        mapping_valid_from_ns: u64,
        mapping_valid_through_ns: u64,
        mapping_receipt_ref: String,
        mapping_receipt_sha256: String,
        qualification: GateState,
    },
    EventProduced {
        event_id: String,
        clock_domain: String,
        basis_id: String,
        epoch: String,
        scale: String,
        unit: String,
        clock_mapping_action_id: String,
        source_timestamp_ns: u64,
        causal_predecessors: Vec<String>,
    },
    NetworkPartition {
        channel_id: String,
        partitioned: bool,
    },
    Delivery {
        event_id: String,
        channel_id: String,
        recipient_id: String,
        delivery_id: String,
        mode: DeliveryMode,
    },
    EventObserved {
        event_id: String,
        observer_id: String,
        clock_domain: String,
        basis_id: String,
        epoch: String,
        scale: String,
        unit: String,
        clock_mapping_action_id: String,
        observed_at_ns: u64,
        observation_receipt_ref: String,
        observation_receipt_sha256: String,
        qualification: GateState,
    },
    EventPersisted {
        event_id: String,
        observer_id: String,
        store_id: String,
        clock_domain: String,
        basis_id: String,
        epoch: String,
        scale: String,
        unit: String,
        clock_mapping_action_id: String,
        persisted_at_ns: u64,
        persistence_receipt_ref: String,
        persistence_receipt_sha256: String,
        qualification: GateState,
    },
    CancellationRequested {
        operation_id: String,
        attempt_id: String,
    },
    CancellationAcknowledged {
        operation_id: String,
        attempt_id: String,
        accepted: bool,
    },
    StopTriggered {
        obligation_id: String,
        policy_revision: String,
    },
    StopPolicyRevised {
        obligation_id: String,
        policy_revision: String,
    },
    StopDischarged {
        obligation_id: String,
        trigger_action_id: String,
        policy_revision_action_id: String,
        policy_revision: String,
        receipt_ref: String,
        receipt_sha256: String,
    },
    EffectAttempted {
        operation_id: String,
        attempt_id: String,
    },
    EffectCommitted {
        operation_id: String,
        attempt_id: String,
        effect_id: String,
    },
    AttemptReconciled {
        operation_id: String,
        attempt_id: String,
        outcome: ReconciliationOutcome,
        receipt_ref: String,
        receipt_sha256: String,
    },
    ProcessCrashed {
        process_id: String,
    },
    ProcessRestarted {
        process_id: String,
    },
    EventSuperseded {
        event_id: String,
        superseded_event_id: String,
    },
    AttemptInventoryAttested {
        scope: StopScope,
        inventory_frontier_event_id: String,
        attempt_manifest_sha256: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMode {
    Deliver,
    Duplicate,
    Drop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationCommitOrder {
    CommitBeforeAcknowledgement,
    CommitAfterAcknowledgement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconciliationOutcome {
    Applied,
    NotApplied,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerDisposition {
    Applied,
    Dropped,
    Duplicate,
    BlockedByPartition,
    BlockedByCrashedRecipient,
    ClockRegression,
    EventTimeRegression,
    StaleClockMappingReference,
    OwnerTimeMissing,
    OwnerTimeUnqualified,
    ClockMappingUnqualified,
    ClockMappingRejected,
    ClockMappingExpired,
    ClockMappingBeyondSample,
    ClockMappingNotYetValid,
    ClockMappingValidityInvalid,
    ClockTimestampBeyondQualifiedTick,
    ObservationUnqualified,
    ObservationRejected,
    PersistenceUnqualified,
    PersistenceRejected,
    ConflictingObservation,
    ConflictingPersistence,
    ConflictingAcknowledgement,
    ConflictingReconciliation,
    ConflictingEffectCommit,
    AcceptedCancellationAfterCommit,
    CommitAfterAcceptedCancellation,
    PreconditionViolation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayRecord {
    pub action_id: String,
    pub at_ns: u64,
    pub ordinal: u64,
    pub owner_time_ref: Option<ActionClockTimeReference>,
    pub evidence: Option<ActionEvidence>,
    pub disposition: SchedulerDisposition,
    pub action: ScheduleAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub schedule_sha256: String,
    /// Digest over ordered replay records, not over this report (which contains
    /// this field and would otherwise be self-referential).
    pub replay_sha256: String,
    pub event_count: usize,
    pub delivery_count: usize,
    pub observed_count: usize,
    pub persisted_count: usize,
    pub records: Vec<ReplayRecord>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReplayError {
    #[error("unsupported schedule schema: {0}")]
    UnsupportedSchema(String),
    #[error("generator version and scenario family must be non-empty")]
    MissingGeneratorMetadata,
    #[error("schedule has too many actions: {0}")]
    TooManyActions(usize),
    #[error("action ID is empty or duplicated: {0}")]
    DuplicateActionId(String),
    #[error("action ordinal is duplicated: {0}")]
    DuplicateOrdinal(u64),
    #[error("required identifier is empty: {0}")]
    EmptyIdentifier(&'static str),
    #[error("SHA-256 field is not exactly 64 lowercase hexadecimal characters: {0}")]
    MalformedSha256(&'static str),
    #[error("unsupported time unit for nanosecond fields: {0}")]
    UnsupportedTimeUnit(String),
    #[error("event was produced without a clock advance: {0}")]
    MissingClock(String),
    #[error("event timestamp exceeds the latest virtual clock tick: {0}")]
    EventBeyondClock(String),
    #[error("event ID is duplicated: {0}")]
    DuplicateEvent(String),
    #[error("causal predecessor is missing or appears after its dependent event: {0}")]
    InvalidCausalPredecessor(String),
    #[error("delivery ID is duplicated: {0}")]
    DuplicateDelivery(String),
    #[error("generated event count must be in 1..={MAX_GENERATED_EVENTS}")]
    InvalidGeneratedEventCount,
    #[error("serialization failed: {0}")]
    Serialization(String),
    #[error("the original schedule does not satisfy the counterexample predicate")]
    OriginalDoesNotReproduce,
    #[error("action evidence does not match the scheduled action: {0}")]
    ActionEvidenceMismatch(&'static str),
}

/// Validate structural schedule invariants. Intentionally does not reject
/// semantic faults (for example, a commit after a stop trigger): those are the
/// inputs the v0.2 oracle is meant to adjudicate.
pub fn validate_schedule(schedule: &VirtualSchedule) -> Result<(), ReplayError> {
    if schedule.schema_version != SCHEDULE_SCHEMA_VERSION {
        return Err(ReplayError::UnsupportedSchema(schedule.schema_version.clone()));
    }
    if schedule.generator_version.trim().is_empty() || schedule.scenario_family.trim().is_empty() {
        return Err(ReplayError::MissingGeneratorMetadata);
    }
    if schedule.actions.len() > MAX_GENERATED_EVENTS * 16 {
        return Err(ReplayError::TooManyActions(schedule.actions.len()));
    }

    let mut actions = schedule.actions.clone();
    actions.sort_by(|a, b| (a.at_ns, a.ordinal, &a.action_id).cmp(&(b.at_ns, b.ordinal, &b.action_id)));

    let mut action_ids = BTreeSet::new();
    let mut ordinals = BTreeSet::new();
    let mut clock_mapping_action_history =
        BTreeMap::<ClockSeriesKey, BTreeMap<String, ClockMappingState>>::new();
    let mut produced_events = BTreeSet::<String>::new();
    let mut delivery_ids = BTreeSet::<String>::new();

    for scheduled in &actions {
        if scheduled.action_id.trim().is_empty() || !action_ids.insert(scheduled.action_id.clone()) {
            return Err(ReplayError::DuplicateActionId(scheduled.action_id.clone()));
        }
        if !ordinals.insert(scheduled.ordinal) {
            return Err(ReplayError::DuplicateOrdinal(scheduled.ordinal));
        }

        match &scheduled.action {
            ScheduleAction::ClockAdvance {
                clock_domain, basis_id, epoch, scale, unit, local_tick_ns,
                calibration_id, mapping_valid_from_ns, mapping_valid_through_ns,
                mapping_receipt_ref, mapping_receipt_sha256, qualification, ..
            } => {
                require_id(clock_domain, "clock_domain")?;
                require_id(basis_id, "basis_id")?;
                require_id(epoch, "epoch")?;
                require_id(scale, "scale")?;
                require_id(unit, "unit")?;
                if unit != "nanoseconds" {
                    return Err(ReplayError::UnsupportedTimeUnit(unit.clone()));
                }
                require_id(calibration_id, "calibration_id")?;
                require_id(mapping_receipt_ref, "mapping_receipt_ref")?;
                require_sha256(mapping_receipt_sha256, "mapping_receipt_sha256")?;
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                clock_mapping_action_history
                    .entry(key)
                    .or_default()
                    .insert(
                        scheduled.action_id.clone(),
                        ClockMappingState {
                            local_tick_ns: *local_tick_ns,
                            valid_from_ns: *mapping_valid_from_ns,
                            valid_through_ns: *mapping_valid_through_ns,
                            qualification: *qualification,
                        },
                    );
            }
            ScheduleAction::EventProduced {
                event_id, clock_domain, basis_id, epoch, scale, unit,
                clock_mapping_action_id, source_timestamp_ns, causal_predecessors,
            } => {
                require_id(event_id, "event_id")?;
                require_id(clock_domain, "clock_domain")?;
                require_id(basis_id, "basis_id")?;
                require_id(epoch, "epoch")?;
                require_id(scale, "scale")?;
                require_id(unit, "unit")?;
                require_id(clock_mapping_action_id, "clock_mapping_action_id")?;
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                let mapping = clock_mapping_action_history
                    .get(&key)
                    .and_then(|history| history.get(clock_mapping_action_id))
                    .ok_or_else(|| {
                        if clock_mapping_action_history.contains_key(&key) {
                            ReplayError::InvalidCausalPredecessor(clock_mapping_action_id.clone())
                        } else {
                            ReplayError::MissingClock(clock_domain.clone())
                        }
                    })?;
                if source_timestamp_ns > &mapping.local_tick_ns {
                    return Err(ReplayError::EventBeyondClock(event_id.clone()));
                }
                if !produced_events.insert(event_id.clone()) {
                    return Err(ReplayError::DuplicateEvent(event_id.clone()));
                }
                let mut predecessors = BTreeSet::new();
                for predecessor in causal_predecessors {
                    if predecessor == event_id
                        || !predecessors.insert(predecessor)
                        || !produced_events.contains(predecessor)
                    {
                        return Err(ReplayError::InvalidCausalPredecessor(predecessor.clone()));
                    }
                }
            }
            ScheduleAction::NetworkPartition { channel_id, .. } => {
                require_id(channel_id, "channel_id")?;
            }
            ScheduleAction::Delivery {
                event_id, channel_id, recipient_id, delivery_id, ..
            } => {
                require_id(event_id, "event_id")?;
                require_id(channel_id, "channel_id")?;
                require_id(recipient_id, "recipient_id")?;
                require_id(delivery_id, "delivery_id")?;
                if !produced_events.contains(event_id) {
                    return Err(ReplayError::InvalidCausalPredecessor(event_id.clone()));
                }
                if !delivery_ids.insert(delivery_id.clone()) {
                    return Err(ReplayError::DuplicateDelivery(delivery_id.clone()));
                }
            }
            ScheduleAction::EventObserved {
                event_id, observer_id, clock_domain, basis_id, epoch, scale, unit,
                clock_mapping_action_id, observation_receipt_ref, observation_receipt_sha256, ..
            } => {
                require_id(event_id, "event_id")?;
                require_id(observer_id, "observer_id")?;
                require_id(clock_domain, "observer_clock_domain")?;
                require_id(basis_id, "observer_clock_basis_id")?;
                require_id(epoch, "observer_clock_epoch")?;
                require_id(scale, "observer_clock_scale")?;
                require_id(unit, "observer_clock_unit")?;
                if unit != "nanoseconds" {
                    return Err(ReplayError::UnsupportedTimeUnit(unit.clone()));
                }
                require_id(clock_mapping_action_id, "observer_clock_mapping_action_id")?;
                require_id(observation_receipt_ref, "observation_receipt_ref")?;
                require_sha256(observation_receipt_sha256, "observation_receipt_sha256")?;
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                if !clock_mapping_action_history.get(&key)
                    .is_some_and(|history| history.contains_key(clock_mapping_action_id))
                {
                    return Err(ReplayError::InvalidCausalPredecessor(clock_mapping_action_id.clone()));
                }
            }
            ScheduleAction::EventPersisted {
                event_id, observer_id, store_id, clock_domain, basis_id, epoch, scale, unit,
                clock_mapping_action_id, persistence_receipt_ref, persistence_receipt_sha256, ..
            } => {
                require_id(event_id, "event_id")?;
                require_id(observer_id, "observer_id")?;
                require_id(store_id, "store_id")?;
                require_id(clock_domain, "store_clock_domain")?;
                require_id(basis_id, "store_clock_basis_id")?;
                require_id(epoch, "store_clock_epoch")?;
                require_id(scale, "store_clock_scale")?;
                require_id(unit, "store_clock_unit")?;
                if unit != "nanoseconds" {
                    return Err(ReplayError::UnsupportedTimeUnit(unit.clone()));
                }
                require_id(clock_mapping_action_id, "store_clock_mapping_action_id")?;
                require_id(persistence_receipt_ref, "persistence_receipt_ref")?;
                require_sha256(persistence_receipt_sha256, "persistence_receipt_sha256")?;
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                if !clock_mapping_action_history.get(&key)
                    .is_some_and(|history| history.contains_key(clock_mapping_action_id))
                {
                    return Err(ReplayError::InvalidCausalPredecessor(clock_mapping_action_id.clone()));
                }
            }
            ScheduleAction::CancellationRequested { operation_id, attempt_id }
            | ScheduleAction::CancellationAcknowledged { operation_id, attempt_id, .. }
            | ScheduleAction::EffectAttempted { operation_id, attempt_id, .. }
            | ScheduleAction::EffectCommitted { operation_id, attempt_id, .. }
            | ScheduleAction::AttemptReconciled { operation_id, attempt_id, .. } => {
                require_id(operation_id, "operation_id")?;
                require_id(attempt_id, "attempt_id")?;
                if let ScheduleAction::EffectCommitted { effect_id, .. } = &scheduled.action {
                    require_id(effect_id, "effect_id")?;
                }
                if let ScheduleAction::AttemptReconciled { receipt_ref, receipt_sha256, .. } = &scheduled.action {
                    require_id(receipt_ref, "receipt_ref")?;
                    require_sha256(receipt_sha256, "attempt_receipt_sha256")?;
                }
            }
            ScheduleAction::StopTriggered { obligation_id, policy_revision }
            | ScheduleAction::StopPolicyRevised { obligation_id, policy_revision } => {
                require_id(obligation_id, "obligation_id")?;
                require_id(policy_revision, "policy_revision")?;
            }
            ScheduleAction::StopDischarged {
                obligation_id, trigger_action_id, policy_revision_action_id,
                policy_revision, receipt_ref, receipt_sha256,
            } => {
                require_id(obligation_id, "obligation_id")?;
                require_id(trigger_action_id, "trigger_action_id")?;
                require_id(policy_revision_action_id, "policy_revision_action_id")?;
                require_id(policy_revision, "policy_revision")?;
                require_id(receipt_ref, "receipt_ref")?;
                require_sha256(receipt_sha256, "stop_discharge_receipt_sha256")?;
            }
            ScheduleAction::AttemptInventoryAttested {
                scope, inventory_frontier_event_id, attempt_manifest_sha256,
            } => {
                if !scope.is_well_formed() {
                    return Err(ReplayError::ActionEvidenceMismatch("inventory_scope"));
                }
                require_id(inventory_frontier_event_id, "inventory_frontier_event_id")?;
                require_sha256(attempt_manifest_sha256, "attempt_manifest_sha256")?;
            }
            ScheduleAction::ProcessCrashed { process_id }
            | ScheduleAction::ProcessRestarted { process_id } => {
                require_id(process_id, "process_id")?;
            }
            ScheduleAction::EventSuperseded { event_id, superseded_event_id } => {
                require_id(event_id, "event_id")?;
                require_id(superseded_event_id, "superseded_event_id")?;
                if !produced_events.contains(event_id) || !produced_events.contains(superseded_event_id) {
                    return Err(ReplayError::InvalidCausalPredecessor(event_id.clone()));
                }
            }
        }

        if let Some(owner_time_ref) = &scheduled.owner_time_ref {
            require_id(&owner_time_ref.clock_mapping_action_id, "owner_time_mapping_action_id")?;
            if !clock_mapping_action_history.values().any(|history| {
                history.contains_key(&owner_time_ref.clock_mapping_action_id)
            }) {
                return Err(ReplayError::InvalidCausalPredecessor(
                    owner_time_ref.clock_mapping_action_id.clone(),
                ));
            }
        }

        if let Some(evidence) = &scheduled.evidence {
            let matches_action = matches!(
                (&scheduled.action, evidence),
                (ScheduleAction::EffectAttempted { .. }, ActionEvidence::AttemptStart { .. })
                    | (ScheduleAction::EffectCommitted { .. }, ActionEvidence::EffectCommit { .. })
                    | (ScheduleAction::CancellationAcknowledged { .. }, ActionEvidence::CancellationAcknowledgement { .. })
                    | (ScheduleAction::AttemptReconciled { .. }, ActionEvidence::TerminalReconciliation { .. })
                    | (ScheduleAction::StopTriggered { .. }, ActionEvidence::StopTrigger { .. })
                    | (ScheduleAction::StopPolicyRevised { .. }, ActionEvidence::StopPolicyRevision { .. })
                    | (ScheduleAction::StopDischarged { .. }, ActionEvidence::StopDischarge { .. })
                    | (ScheduleAction::AttemptInventoryAttested { .. }, ActionEvidence::InventoryAttestation { .. })
            );
            if !matches_action {
                return Err(ReplayError::ActionEvidenceMismatch("action_variant"));
            }
            match evidence {
                ActionEvidence::AttemptStart { retry_of_attempt_id } => {
                    if let Some(predecessor) = retry_of_attempt_id {
                        require_id(predecessor, "retry_of_attempt_id")?;
                    }
                }
                ActionEvidence::EffectCommit { receipt_ref, receipt_sha256 }
                | ActionEvidence::CancellationAcknowledgement { receipt_ref, receipt_sha256 }
                | ActionEvidence::StopTrigger { receipt_ref, receipt_sha256, .. }
                | ActionEvidence::StopPolicyRevision { receipt_ref, receipt_sha256, .. }
                | ActionEvidence::InventoryAttestation { receipt_ref, receipt_sha256, .. } => {
                    require_id(receipt_ref, "action_evidence_receipt_ref")?;
                    require_sha256(receipt_sha256, "action_evidence_receipt_sha256")?;
                }
                ActionEvidence::TerminalReconciliation { exact_start_action_id, .. } => {
                    require_id(exact_start_action_id, "exact_start_action_id")?;
                }
                ActionEvidence::StopDischarge {
                    scope, inventory_frontier_event_id, attempt_manifest_sha256,
                } => {
                    if !scope.is_well_formed() {
                        return Err(ReplayError::ActionEvidenceMismatch("stop_discharge_scope"));
                    }
                    require_id(inventory_frontier_event_id, "inventory_frontier_event_id")?;
                    require_sha256(attempt_manifest_sha256, "attempt_manifest_sha256")?;
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ClockMappingState {
    local_tick_ns: u64,
    valid_from_ns: u64,
    valid_through_ns: u64,
    qualification: GateState,
}

type ClockSeriesKey = (String, String, String, String, String);

fn clock_series_key(
    clock_domain: &str,
    basis_id: &str,
    epoch: &str,
    scale: &str,
    unit: &str,
) -> ClockSeriesKey {
    (
        clock_domain.to_owned(),
        basis_id.to_owned(),
        epoch.to_owned(),
        scale.to_owned(),
        unit.to_owned(),
    )
}

fn require_id(value: &str, name: &'static str) -> Result<(), ReplayError> {
    if value.trim().is_empty() {
        Err(ReplayError::EmptyIdentifier(name))
    } else {
        Ok(())
    }
}

fn classify_clock_mapping(
    active_mapping_id: Option<&String>,
    referenced_mapping_id: &str,
    mapping: Option<&ClockMappingState>,
    timestamp_ns: u64,
) -> Option<SchedulerDisposition> {
    if active_mapping_id.is_none_or(|active| active != referenced_mapping_id)
        || mapping.is_none()
    {
        return Some(SchedulerDisposition::StaleClockMappingReference);
    }
    let mapping = mapping?;
    if mapping.qualification == GateState::Fail {
        Some(SchedulerDisposition::ClockMappingRejected)
    } else if mapping.qualification != GateState::Pass {
        Some(SchedulerDisposition::ClockMappingUnqualified)
    } else if mapping.valid_from_ns > mapping.valid_through_ns {
        Some(SchedulerDisposition::ClockMappingValidityInvalid)
    } else if timestamp_ns < mapping.valid_from_ns {
        Some(SchedulerDisposition::ClockMappingNotYetValid)
    } else if timestamp_ns > mapping.local_tick_ns {
        Some(SchedulerDisposition::ClockTimestampBeyondQualifiedTick)
    } else if timestamp_ns > mapping.valid_through_ns {
        Some(SchedulerDisposition::ClockMappingExpired)
    } else {
        None
    }
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), ReplayError> {
    if value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ReplayError::MalformedSha256(field))
    }
}

fn ordered_actions(schedule: &VirtualSchedule) -> Vec<ScheduledAction> {
    let mut actions = schedule.actions.clone();
    actions.sort_by(|a, b| (a.at_ns, a.ordinal, &a.action_id).cmp(&(b.at_ns, b.ordinal, &b.action_id)));
    for scheduled in &mut actions {
        if let ScheduleAction::EventProduced { causal_predecessors, .. } = &mut scheduled.action {
            causal_predecessors.sort();
        }
    }
    actions
}

/// Canonical schedule projection. Action-vector order and causal-predecessor
/// vector order are non-semantic; all other fields are committed exactly.
pub fn canonical_schedule_bytes(schedule: &VirtualSchedule) -> Result<Vec<u8>, ReplayError> {
    validate_schedule(schedule)?;
    serde_json::to_vec(&VirtualSchedule {
        schema_version: schedule.schema_version.clone(),
        generator_version: schedule.generator_version.clone(),
        seed: schedule.seed,
        scenario_family: schedule.scenario_family.clone(),
        actions: ordered_actions(schedule),
    })
    .map_err(|error| ReplayError::Serialization(error.to_string()))
}

pub fn canonical_schedule_sha256(schedule: &VirtualSchedule) -> Result<String, ReplayError> {
    Ok(sha256_domain_hex(b"symthaea.event-order.schedule.v0.3|", &canonical_schedule_bytes(schedule)?))
}

#[derive(Debug, Clone)]
struct StopReplayState {
    trigger_action_id: String,
    policy_revision_action_id: String,
    policy_revision: String,
}

/// Replay a schedule in virtual time. The output records schedule intent and
/// scheduler disposition. It is not an event-order safety verdict; callers
/// pass the resulting observations to the v0.2 evaluator.
pub fn replay_virtual_schedule(schedule: &VirtualSchedule) -> Result<ReplayReport, ReplayError> {
    validate_schedule(schedule)?;
    let schedule_sha256 = canonical_schedule_sha256(schedule)?;
    let mut partitioned = BTreeSet::<String>::new();
    let mut crashed = BTreeSet::<String>::new();
    let mut produced = BTreeSet::<String>::new();
    let mut active_clock_mapping_ids = BTreeMap::<ClockSeriesKey, String>::new();
    let mut clock_mapping_states = BTreeMap::<String, ClockMappingState>::new();
    let mut clock_series_by_mapping_id = BTreeMap::<String, ClockSeriesKey>::new();
    let mut event_times = BTreeMap::<ClockSeriesKey, u64>::new();
    let mut delivered = BTreeSet::<(String, String)>::new();
    let mut observed = BTreeSet::<(String, String)>::new();
    let mut observation_receipts =
        BTreeMap::<(String, String), (String, u64, String, String, GateState)>::new();
    let mut persisted = BTreeSet::<(String, String)>::new();
    let mut persistence_receipts =
        BTreeMap::<(String, String), (String, u64, String, String, GateState)>::new();
    let mut cancel_requests = BTreeSet::<(String, String)>::new();
    let mut cancel_acknowledgements = BTreeMap::<(String, String), bool>::new();
    let mut stop_triggers = BTreeMap::<String, StopReplayState>::new();
    let mut stop_discharges = BTreeSet::<String>::new();
    let mut attempts = BTreeSet::<(String, String)>::new();
    let mut attempt_start_action_ids = BTreeMap::<(String, String), String>::new();
    let mut committed_attempts =
        BTreeMap::<(String, String), (String, String, String)>::new();
    let mut reconciled_attempts =
        BTreeMap::<(String, String), (ReconciliationOutcome, String, String, GateState, String)>::new();
    let mut inventory_attestations = BTreeSet::<(String, String, String)>::new();
    let mut records = Vec::new();

    for scheduled in ordered_actions(schedule) {
        let disposition = match &scheduled.action {
            ScheduleAction::ClockAdvance {
                clock_domain, basis_id, epoch, scale, unit, local_tick_ns,
                mapping_valid_from_ns, mapping_valid_through_ns, qualification, ..
            } => {
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                let previous_mapping_id = active_clock_mapping_ids.insert(key, scheduled.action_id.clone());
                let previous_tick = previous_mapping_id
                    .as_ref()
                    .and_then(|id| clock_mapping_states.get(id))
                    .map(|mapping| mapping.local_tick_ns);
                clock_mapping_states.insert(
                    scheduled.action_id.clone(),
                    ClockMappingState {
                        local_tick_ns: *local_tick_ns,
                        valid_from_ns: *mapping_valid_from_ns,
                        valid_through_ns: *mapping_valid_through_ns,
                        qualification: *qualification,
                    },
                );
                clock_series_by_mapping_id.insert(scheduled.action_id.clone(), clock_series_key(clock_domain, basis_id, epoch, scale, unit));
                if previous_tick.is_some_and(|tick| *local_tick_ns < tick) {
                    SchedulerDisposition::ClockRegression
                } else {
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::EventProduced {
                event_id, clock_domain, basis_id, epoch, scale, unit,
                clock_mapping_action_id, source_timestamp_ns, ..
            } => {
                produced.insert(event_id.clone());
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                let current_mapping = active_clock_mapping_ids.get(&key);
                let mapping = clock_mapping_states.get(clock_mapping_action_id);
                let previous = event_times.insert(key, *source_timestamp_ns);
                if let Some(disposition) = classify_clock_mapping(
                    current_mapping,
                    clock_mapping_action_id,
                    mapping,
                    *source_timestamp_ns,
                ) {
                    disposition
                } else if previous.is_some_and(|timestamp| *source_timestamp_ns < timestamp) {
                    SchedulerDisposition::EventTimeRegression
                } else {
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::NetworkPartition { channel_id, partitioned: true } => {
                partitioned.insert(channel_id.clone());
                SchedulerDisposition::Applied
            }
            ScheduleAction::NetworkPartition { channel_id, partitioned: false } => {
                partitioned.remove(channel_id);
                SchedulerDisposition::Applied
            }
            ScheduleAction::Delivery { event_id, channel_id, recipient_id, mode, .. } => {
                if *mode == DeliveryMode::Drop {
                    SchedulerDisposition::Dropped
                } else if partitioned.contains(channel_id) {
                    SchedulerDisposition::BlockedByPartition
                } else if crashed.contains(recipient_id) {
                    SchedulerDisposition::BlockedByCrashedRecipient
                } else {
                    let key = (event_id.clone(), recipient_id.clone());
                    match mode {
                        DeliveryMode::Deliver => {
                            if delivered.insert(key) {
                                SchedulerDisposition::Applied
                            } else {
                                SchedulerDisposition::Duplicate
                            }
                        }
                        DeliveryMode::Duplicate => {
                            if delivered.contains(&key) {
                                SchedulerDisposition::Duplicate
                            } else {
                                SchedulerDisposition::PreconditionViolation
                            }
                        }
                        DeliveryMode::Drop => SchedulerDisposition::Dropped,
                    }
                }
            }
            ScheduleAction::EventObserved {
                event_id, observer_id, clock_domain, basis_id, epoch, scale, unit,
                clock_mapping_action_id, observed_at_ns, observation_receipt_ref,
                observation_receipt_sha256, qualification,
            } => {
                let key = (event_id.clone(), observer_id.clone());
                let clock_key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                let mapping_disposition = classify_clock_mapping(
                    active_clock_mapping_ids.get(&clock_key),
                    clock_mapping_action_id,
                    clock_mapping_states.get(clock_mapping_action_id),
                    *observed_at_ns,
                );
                let receipt_key = (
                    clock_mapping_action_id.clone(),
                    *observed_at_ns,
                    observation_receipt_ref.clone(),
                    observation_receipt_sha256.clone(),
                    *qualification,
                );
                if crashed.contains(observer_id) {
                    SchedulerDisposition::BlockedByCrashedRecipient
                } else if !delivered.contains(&key) {
                    SchedulerDisposition::PreconditionViolation
                } else if let Some(disposition) = mapping_disposition {
                    disposition
                } else if *qualification == GateState::Fail {
                    SchedulerDisposition::ObservationRejected
                } else if *qualification != GateState::Pass {
                    SchedulerDisposition::ObservationUnqualified
                } else if let Some(previous) = observation_receipts.get(&key) {
                    if previous == &receipt_key {
                        SchedulerDisposition::Duplicate
                    } else {
                        SchedulerDisposition::ConflictingObservation
                    }
                } else {
                    observation_receipts.insert(key.clone(), receipt_key);
                    observed.insert(key);
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::EventPersisted {
                event_id, observer_id, store_id, clock_domain, basis_id, epoch, scale, unit,
                clock_mapping_action_id, persisted_at_ns, persistence_receipt_ref,
                persistence_receipt_sha256, qualification,
            } => {
                let observation = (event_id.clone(), observer_id.clone());
                let persistence = (event_id.clone(), store_id.clone());
                let clock_key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                let mapping_disposition = classify_clock_mapping(
                    active_clock_mapping_ids.get(&clock_key),
                    clock_mapping_action_id,
                    clock_mapping_states.get(clock_mapping_action_id),
                    *persisted_at_ns,
                );
                let receipt_key = (
                    clock_mapping_action_id.clone(),
                    *persisted_at_ns,
                    persistence_receipt_ref.clone(),
                    persistence_receipt_sha256.clone(),
                    *qualification,
                );
                if crashed.contains(observer_id) || crashed.contains(store_id) {
                    SchedulerDisposition::BlockedByCrashedRecipient
                } else if !observed.contains(&observation) {
                    SchedulerDisposition::PreconditionViolation
                } else if let Some(disposition) = mapping_disposition {
                    disposition
                } else if *qualification == GateState::Fail {
                    SchedulerDisposition::PersistenceRejected
                } else if *qualification != GateState::Pass {
                    SchedulerDisposition::PersistenceUnqualified
                } else if let Some(previous) = persistence_receipts.get(&persistence) {
                    if previous == &receipt_key {
                        SchedulerDisposition::Duplicate
                    } else {
                        SchedulerDisposition::ConflictingPersistence
                    }
                } else {
                    persistence_receipts.insert(persistence.clone(), receipt_key);
                    persisted.insert(persistence);
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::CancellationRequested { operation_id, attempt_id } => {
                if cancel_requests.insert((operation_id.clone(), attempt_id.clone())) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::Duplicate
                }
            }
            ScheduleAction::CancellationAcknowledged { operation_id, attempt_id, accepted } => {
                let key = (operation_id.clone(), attempt_id.clone());
                if !cancel_requests.contains(&key) {
                    SchedulerDisposition::PreconditionViolation
                } else if let Some(previous) = cancel_acknowledgements.get(&key) {
                    if previous == accepted {
                        SchedulerDisposition::Duplicate
                    } else {
                        SchedulerDisposition::ConflictingAcknowledgement
                    }
                } else {
                    cancel_acknowledgements.insert(key.clone(), *accepted);
                    if *accepted && committed_attempts.contains(&key) {
                        SchedulerDisposition::AcceptedCancellationAfterCommit
                    } else {
                        SchedulerDisposition::Applied
                    }
                }
            }
            ScheduleAction::StopTriggered { obligation_id, policy_revision } => {
                if stop_triggers.contains_key(obligation_id) {
                    SchedulerDisposition::PreconditionViolation
                } else {
                    let scope = match &scheduled.evidence {
                        Some(ActionEvidence::StopTrigger { scope, .. }) => Some(scope.clone()),
                        _ => None,
                    };
                    stop_triggers.insert(obligation_id.clone(), StopReplayState {
                        trigger_action_id: scheduled.action_id.clone(),
                        policy_revision_action_id: scheduled.action_id.clone(),
                        policy_revision: policy_revision.clone(),
                        scope,
                    });
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::StopPolicyRevised { obligation_id, policy_revision } => {
                if let Some(state) = stop_triggers.get_mut(obligation_id) {
                    state.policy_revision = policy_revision.clone();
                    state.policy_revision_action_id = scheduled.action_id.clone();
                    state.scope = match &scheduled.evidence {
                        Some(ActionEvidence::StopPolicyRevision { scope, .. }) => Some(scope.clone()),
                        _ => None,
                    };
                    stop_discharges.remove(obligation_id);
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::StopDischarged {
                obligation_id, trigger_action_id, policy_revision_action_id, policy_revision, ..
            } => {
                let exact_binding = stop_triggers.get(obligation_id).is_some_and(|state| {
                    state.trigger_action_id == *trigger_action_id
                        && state.policy_revision_action_id == *policy_revision_action_id
                        && state.policy_revision == *policy_revision
                });
                let inventory_binding = match (&scheduled.evidence, stop_triggers.get(obligation_id)) {
                    (Some(ActionEvidence::StopDischarge { scope, inventory_frontier_event_id, attempt_manifest_sha256 }), Some(state)) => {
                        state.scope.as_ref() == Some(scope)
                            && inventory_attestations.contains(&(
                                serde_json::to_string(scope).unwrap_or_default(),
                                inventory_frontier_event_id.clone(),
                                attempt_manifest_sha256.clone(),
                            ))
                    }
                    (None, Some(state)) if state.scope.is_none() => true,
                    _ => false,
                };
                if exact_binding && inventory_binding && stop_discharges.insert(obligation_id.clone()) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::EffectAttempted { operation_id, attempt_id } => {
                let key = (operation_id.clone(), attempt_id.clone());
                let retry_valid = match &scheduled.evidence {
                    Some(ActionEvidence::AttemptStart { retry_of_attempt_id: Some(predecessor_id) }) => {
                        let predecessor_key = (operation_id.clone(), predecessor_id.clone());
                        reconciled_attempts.get(&predecessor_key).is_some_and(|terminal| {
                            terminal.0 == ReconciliationOutcome::NotApplied
                                && terminal.3 == GateState::Pass
                                && terminal.4 == attempt_start_action_ids.get(&predecessor_key).cloned().unwrap_or_default()
                        })
                    }
                    Some(ActionEvidence::AttemptStart { retry_of_attempt_id: None }) => true,
                    None => true,
                    _ => false,
                };
                if !retry_valid {
                    SchedulerDisposition::PreconditionViolation
                } else if attempts.insert(key.clone()) {
                    attempt_start_action_ids.insert(key, scheduled.action_id.clone());
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::EffectCommitted { operation_id, attempt_id, effect_id } => {
                let key = (operation_id.clone(), attempt_id.clone());
                let evidence = match &scheduled.evidence {
                    Some(ActionEvidence::EffectCommit { receipt_ref, receipt_sha256 }) => {
                        (receipt_ref.clone(), receipt_sha256.clone())
                    }
                    _ => (String::new(), String::new()),
                };
                let candidate = (effect_id.clone(), evidence.0, evidence.1);
                if !attempts.contains(&key) {
                    SchedulerDisposition::PreconditionViolation
                } else if let Some(previous) = committed_attempts.get(&key) {
                    if previous == &candidate {
                        SchedulerDisposition::Duplicate
                    } else {
                        SchedulerDisposition::ConflictingEffectCommit
                    }
                } else {
                    committed_attempts.insert(key.clone(), candidate);
                    if cancel_acknowledgements.get(&key) == Some(&true) {
                        SchedulerDisposition::CommitAfterAcceptedCancellation
                    } else {
                        SchedulerDisposition::Applied
                    }
                }
            }
            ScheduleAction::AttemptReconciled {
                operation_id, attempt_id, outcome, receipt_ref, receipt_sha256,
            } => {
                let key = (operation_id.clone(), attempt_id.clone());
                let (qualification, exact_start_action_id) = match &scheduled.evidence {
                    Some(ActionEvidence::TerminalReconciliation { exact_start_action_id, qualification }) => {
                        (*qualification, exact_start_action_id.clone())
                    }
                    _ => (GateState::Indeterminate, String::new()),
                };
                let terminal = (*outcome, receipt_ref.clone(), receipt_sha256.clone(), qualification, exact_start_action_id);
                let exact_start = attempt_start_action_ids.get(&key).is_some_and(|start_id| {
                    terminal.4 == *start_id
                });
                if !attempts.contains(&key) || (scheduled.evidence.is_some() && !exact_start) {
                    SchedulerDisposition::PreconditionViolation
                } else if let Some(previous) = reconciled_attempts.get(&key) {
                    if previous == &terminal {
                        SchedulerDisposition::Duplicate
                    } else {
                        SchedulerDisposition::ConflictingReconciliation
                    }
                } else {
                    reconciled_attempts.insert(key, terminal);
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::AttemptInventoryAttested {
                scope, inventory_frontier_event_id, attempt_manifest_sha256,
            } => {
                let exact = match &scheduled.evidence {
                    Some(ActionEvidence::InventoryAttestation {
                        scope: evidence_scope,
                        inventory_frontier_event_id: evidence_frontier,
                        attempt_manifest_sha256: evidence_manifest,
                        ..
                    }) => evidence_scope == scope
                        && evidence_frontier == inventory_frontier_event_id
                        && evidence_manifest == attempt_manifest_sha256,
                    _ => false,
                };
                if exact {
                    inventory_attestations.insert((
                        serde_json::to_string(scope).unwrap_or_default(),
                        inventory_frontier_event_id.clone(),
                        attempt_manifest_sha256.clone(),
                    ));
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::ProcessCrashed { process_id } => {
                if crashed.insert(process_id.clone()) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::Duplicate
                }
            }
            ScheduleAction::ProcessRestarted { process_id } => {
                if crashed.remove(process_id) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::EventSuperseded { .. } => SchedulerDisposition::Applied,
        };
        let mut disposition = disposition;
        if let Some(owner_time_ref) = &scheduled.owner_time_ref {
            let series_key = clock_series_by_mapping_id.get(&owner_time_ref.clock_mapping_action_id);
            let active_mapping_id = series_key.and_then(|key| active_clock_mapping_ids.get(key));
            let owner_time_disposition = classify_clock_mapping(
                active_mapping_id,
                &owner_time_ref.clock_mapping_action_id,
                clock_mapping_states.get(&owner_time_ref.clock_mapping_action_id),
                owner_time_ref.local_timestamp_ns,
            );
            if let Some(owner_time_disposition) = owner_time_disposition {
                if matches!(disposition, SchedulerDisposition::Applied | SchedulerDisposition::Duplicate) {
                    disposition = owner_time_disposition;
                }
            }
        }
        records.push(ReplayRecord {
            action_id: scheduled.action_id,
            at_ns: scheduled.at_ns,
            ordinal: scheduled.ordinal,
            owner_time_ref: scheduled.owner_time_ref,
            evidence: scheduled.evidence,
            disposition,
            action: scheduled.action,
        });
    }

    let replay_bytes = serde_json::to_vec(&records)
        .map_err(|error| ReplayError::Serialization(error.to_string()))?;
    let event_count = produced.len();
    let delivery_count = records.iter().filter(|record| matches!(record.action, ScheduleAction::Delivery { .. })).count();
    let observed_count = observed.len();
    let persisted_count = persisted.len();
    Ok(ReplayReport {
        schedule_sha256,
        replay_sha256: sha256_domain_hex(b"symthaea.event-order.replay.v0.3|", &replay_bytes),
        event_count,
        delivery_count,
        observed_count,
        persisted_count,
        records,
    })
}

/// Deterministic SplitMix64 generator. It is for reproducible synthetic test
/// schedules, not cryptographic randomness or production identity generation.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = (self.next_u64() as usize) % (index + 1);
            values.swap(index, other);
        }
    }
}

fn push_action(
    actions: &mut Vec<ScheduledAction>,
    next_ordinal: &mut u64,
    at_ns: u64,
    action: ScheduleAction,
) {
    let ordinal = *next_ordinal;
    *next_ordinal += 1;
    actions.push(ScheduledAction {
        action_id: format!("action-{ordinal:06}"),
        at_ns,
        ordinal,
        owner_time_ref: None,
        evidence: None,
        action,
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockMappingFault {
    Unqualified,
    Expired,
    StaleReference,
}

/// Generate a single-event clock-mapping fault without asserting that the
/// supplied mapping is trustworthy. Faults remain valid schedule inputs and
/// are classified by replay against the exact mapping action reference.
pub fn generate_clock_mapping_fault_schedule(
    seed: u64,
    fault: ClockMappingFault,
) -> VirtualSchedule {
    let mut actions = Vec::new();
    let mut ordinal = 0_u64;
    let first_mapping_action_id = format!("action-{ordinal:06}");
    let (first_qualification, first_valid_through_ns) = match fault {
        ClockMappingFault::Unqualified => (GateState::Indeterminate, 100),
        ClockMappingFault::Expired => (GateState::Pass, 50),
        ClockMappingFault::StaleReference => (GateState::Pass, 100),
    };
    push_action(
        &mut actions,
        &mut ordinal,
        1,
        ScheduleAction::ClockAdvance {
            clock_domain: "clock-source".to_owned(),
            basis_id: "basis-source".to_owned(),
            epoch: "epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 100,
            offset_ns: 0,
            uncertainty_ns: 100,
            calibration_id: format!("cal-{seed:016x}-first"),
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: first_valid_through_ns,
            mapping_receipt_ref: format!("synthetic://mapping/{seed:016x}/first"),
            mapping_receipt_sha256: "1".repeat(64),
            qualification: first_qualification,
        },
    );
    let event_mapping_action_id = first_mapping_action_id.clone();
    if fault == ClockMappingFault::StaleReference {
        push_action(
            &mut actions,
            &mut ordinal,
            2,
            ScheduleAction::ClockAdvance {
                clock_domain: "clock-source".to_owned(),
                basis_id: "basis-source".to_owned(),
                epoch: "epoch-0".to_owned(),
                scale: "tai".to_owned(),
                unit: "nanoseconds".to_owned(),
                local_tick_ns: 110,
                offset_ns: 0,
                uncertainty_ns: 50,
                calibration_id: format!("cal-{seed:016x}-second"),
                mapping_valid_from_ns: 0,
                mapping_valid_through_ns: 200,
                mapping_receipt_ref: format!("synthetic://mapping/{seed:016x}/second"),
                mapping_receipt_sha256: "2".repeat(64),
                qualification: GateState::Pass,
            },
        );
    }
    let event_at = if fault == ClockMappingFault::StaleReference { 3 } else { 2 };
    let event_time = if fault == ClockMappingFault::Expired { 75 } else { 80 };
    push_action(
        &mut actions,
        &mut ordinal,
        event_at,
        ScheduleAction::EventProduced {
            event_id: format!("event-{seed:016x}"),
            clock_domain: "clock-source".to_owned(),
            basis_id: "basis-source".to_owned(),
            epoch: "epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: event_mapping_action_id.clone(),
            source_timestamp_ns: event_time,
            causal_predecessors: Vec::new(),
        },
    );
    VirtualSchedule {
        schema_version: SCHEDULE_SCHEMA_VERSION.to_owned(),
        generator_version: GENERATOR_VERSION.to_owned(),
        seed,
        scenario_family: match fault {
            ClockMappingFault::Unqualified => "clock_mapping_unqualified".to_owned(),
            ClockMappingFault::Expired => "clock_mapping_expired".to_owned(),
            ClockMappingFault::StaleReference => "clock_mapping_stale_reference".to_owned(),
        },
        actions,
    }
}

/// Generate a frozen-replay-friendly event-production and network-delivery
/// schedule. The first half of deliveries occurs before a deterministic
/// partition; the second half occurs while partitioned. A subset is duplicated
/// or dropped using the fixed seed.
pub fn generate_delivery_schedule(
    seed: u64,
    event_count: usize,
) -> Result<VirtualSchedule, ReplayError> {
    if event_count == 0 || event_count > MAX_GENERATED_EVENTS {
        return Err(ReplayError::InvalidGeneratedEventCount);
    }
    let mut rng = SplitMix64(seed);
    let mut actions = Vec::new();
    let mut ordinal = 0_u64;
    let mut clock_ticks = [0_u64; 2];
    let mut event_ids = Vec::with_capacity(event_count);

    for index in 0..event_count {
        let clock_index = index % 2;
        clock_ticks[clock_index] += 100;
        let domain = if clock_index == 0 { "clock-a" } else { "clock-b" };
        let at_ns = index as u64 + 1;
        let clock_mapping_action_id = format!("action-{ordinal:06}");
        push_action(
            &mut actions,
            &mut ordinal,
            at_ns,
            ScheduleAction::ClockAdvance {
                clock_domain: domain.to_owned(),
                basis_id: format!("basis-{domain}"),
                epoch: "epoch-0".to_owned(),
                scale: "tai".to_owned(),
                unit: "nanoseconds".to_owned(),
                local_tick_ns: clock_ticks[clock_index],
                offset_ns: (rng.next_u64() % 1_000_001) as i64 - 500_000,
                uncertainty_ns: rng.next_u64() % 5_000,
                calibration_id: format!("cal-{domain}-{}", clock_ticks[clock_index]),
                mapping_valid_from_ns: clock_ticks[clock_index].saturating_sub(10_000),
                mapping_valid_through_ns: clock_ticks[clock_index] + 10_000,
                mapping_receipt_ref: format!("synthetic://mapping/{domain}/{}", clock_ticks[clock_index]),
                mapping_receipt_sha256: "0000000000000000000000000000000000000000000000000000000000000000".to_owned(),
                qualification: GateState::Pass,
            },
        );
        let event_id = format!("event-{index:06}");
        let causal_predecessors = event_ids.last().cloned().into_iter().collect();
        push_action(
            &mut actions,
            &mut ordinal,
            at_ns,
            ScheduleAction::EventProduced {
                event_id: event_id.clone(),
                clock_domain: domain.to_owned(),
                basis_id: format!("basis-{domain}"),
                epoch: "epoch-0".to_owned(),
                scale: "tai".to_owned(),
                unit: "nanoseconds".to_owned(),
                clock_mapping_action_id,
                source_timestamp_ns: clock_ticks[clock_index],
                causal_predecessors,
            },
        );
        event_ids.push(event_id);
    }

    let mut delivery_order: Vec<usize> = (0..event_count).collect();
    rng.shuffle(&mut delivery_order);
    let split = event_count / 2;
    let delivery_base = event_count as u64 + 1;
    for (position, event_index) in delivery_order.into_iter().enumerate() {
        let at_ns = delivery_base + position as u64 + 1;
        if position == split {
            push_action(
                &mut actions,
                &mut ordinal,
                at_ns,
                ScheduleAction::NetworkPartition {
                    channel_id: "channel-main".to_owned(),
                    partitioned: true,
                },
            );
        }
        let mode = if rng.next_u64() % 11 == 0 { DeliveryMode::Drop } else { DeliveryMode::Deliver };
        push_action(
            &mut actions,
            &mut ordinal,
            at_ns,
            ScheduleAction::Delivery {
                event_id: format!("event-{event_index:06}"),
                channel_id: "channel-main".to_owned(),
                recipient_id: "worker-0".to_owned(),
                delivery_id: format!("delivery-{position:06}"),
                mode,
            },
        );
        if mode == DeliveryMode::Deliver && rng.next_u64() % 4 == 0 {
            push_action(
                &mut actions,
                &mut ordinal,
                at_ns,
                ScheduleAction::Delivery {
                    event_id: format!("event-{event_index:06}"),
                    channel_id: "channel-main".to_owned(),
                    recipient_id: "worker-0".to_owned(),
                    delivery_id: format!("duplicate-{position:06}"),
                    mode: DeliveryMode::Duplicate,
                },
            );
        }
    }
    let reconnect_at = delivery_base + event_count as u64 + 2;
    push_action(
        &mut actions,
        &mut ordinal,
        reconnect_at,
        ScheduleAction::NetworkPartition {
            channel_id: "channel-main".to_owned(),
            partitioned: false,
        },
    );
    // Explicit post-reconnect delivery proves that the scheduler resumes
    // transport rather than leaving the channel permanently blocked.
    push_action(
        &mut actions,
        &mut ordinal,
        reconnect_at + 1,
        ScheduleAction::Delivery {
            event_id: event_ids[0].clone(),
            channel_id: "channel-main".to_owned(),
            recipient_id: "worker-after-reconnect".to_owned(),
            delivery_id: "delivery-after-reconnect".to_owned(),
            mode: DeliveryMode::Deliver,
        },
    );

    let schedule = VirtualSchedule {
        schema_version: SCHEDULE_SCHEMA_VERSION.to_owned(),
        generator_version: GENERATOR_VERSION.to_owned(),
        seed,
        scenario_family: "delivery_permutation_with_partition".to_owned(),
        actions,
    };
    validate_schedule(&schedule)?;
    Ok(schedule)
}

/// Generate one event with independent source, observer and durable-store
/// clock series. Simulator action time is only the replay scheduler; the
/// observed and persisted timestamps are carried from their own clock domains.
pub fn generate_clocked_observation_persistence_schedule(
    seed: u64,
) -> Result<VirtualSchedule, ReplayError> {
    let mut schedule = generate_delivery_schedule(seed, 4)?;
    let mut ordinal = schedule.actions.iter().map(|action| action.ordinal).max().unwrap_or(0) + 1;
    let start_at = schedule.actions.iter().map(|action| action.at_ns).max().unwrap_or(0) + 1;
    let event_id = "event-000000".to_owned();
    let observer_id = "worker-after-reconnect".to_owned();
    let store_id = "durable-store-0".to_owned();
    let observed_at_ns = 1_000 + (seed % 1_000);
    let persisted_at_ns = observed_at_ns + 200;

    let observer_clock_domain = format!("observer-clock-{seed:016x}");
    let observer_basis_id = format!("observer-basis-{seed:016x}");
    let observer_mapping_action_id = format!("action-{ordinal:06}");
    push_action(
        &mut schedule.actions,
        &mut ordinal,
        start_at,
        ScheduleAction::ClockAdvance {
            clock_domain: observer_clock_domain.clone(),
            basis_id: observer_basis_id.clone(),
            epoch: "observer-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 100_000,
            offset_ns: 0,
            uncertainty_ns: 25,
            calibration_id: format!("observer-cal-{seed:016x}"),
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 100_000,
            mapping_receipt_ref: format!("synthetic://observer-clock/{seed:016x}"),
            mapping_receipt_sha256: "a".repeat(64),
            qualification: GateState::Pass,
        },
    );

    let store_clock_domain = format!("store-clock-{seed:016x}");
    let store_basis_id = format!("store-basis-{seed:016x}");
    let store_mapping_action_id = format!("action-{ordinal:06}");
    push_action(
        &mut schedule.actions,
        &mut ordinal,
        start_at + 1,
        ScheduleAction::ClockAdvance {
            clock_domain: store_clock_domain.clone(),
            basis_id: store_basis_id.clone(),
            epoch: "store-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 100_000,
            offset_ns: 0,
            uncertainty_ns: 10,
            calibration_id: format!("store-cal-{seed:016x}"),
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 100_000,
            mapping_receipt_ref: format!("synthetic://store-clock/{seed:016x}"),
            mapping_receipt_sha256: "b".repeat(64),
            qualification: GateState::Pass,
        },
    );

    push_action(
        &mut schedule.actions,
        &mut ordinal,
        start_at + 2,
        ScheduleAction::EventObserved {
            event_id: event_id.clone(),
            observer_id: observer_id.clone(),
            clock_domain: observer_clock_domain,
            basis_id: observer_basis_id,
            epoch: "observer-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: observer_mapping_action_id,
            observed_at_ns,
            observation_receipt_ref: format!("synthetic://observation/{seed:016x}"),
            observation_receipt_sha256: "c".repeat(64),
            qualification: GateState::Pass,
        },
    );
    push_action(
        &mut schedule.actions,
        &mut ordinal,
        start_at + 3,
        ScheduleAction::EventPersisted {
            event_id,
            observer_id,
            store_id,
            clock_domain: store_clock_domain,
            basis_id: store_basis_id,
            epoch: "store-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: store_mapping_action_id,
            persisted_at_ns,
            persistence_receipt_ref: format!("synthetic://durable-persistence/{seed:016x}"),
            persistence_receipt_sha256: "d".repeat(64),
            qualification: GateState::Pass,
        },
    );

    schedule.generator_version = GENERATOR_VERSION.to_owned();
    schedule.scenario_family = "independent_observer_and_store_clock_evidence".to_owned();
    validate_schedule(&schedule)?;
    Ok(schedule)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectedTimeEvidenceBinding {
    pub action_id: String,
    pub clock_mapping_action_id: String,
    pub source_timestamp_ns: u64,
    pub receipt_ref: String,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitemporalProjectionReport {
    pub schedule_sha256: Option<String>,
    pub disposition: EventTraceDisposition,
    pub envelope: Option<BitemporalEventEnvelope>,
    pub occurrence_binding: Option<ProjectedTimeEvidenceBinding>,
    pub observation_binding: Option<ProjectedTimeEvidenceBinding>,
    pub persistence_binding: Option<ProjectedTimeEvidenceBinding>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
enum ProjectionFault {
    Indeterminate(&'static str),
    Critical(&'static str),
}

fn resolve_owner_qualified_time(
    schedule: &VirtualSchedule,
    mapping_action_id: &str,
    local_timestamp_ns: u64,
    source_clock_domain_id: &str,
    receipt_ref: &str,
    receipt_sha256: &str,
    resolutions: &BTreeMap<(String, u64), QualifiedEventTime>,
) -> Result<QualifiedEventTime, ProjectionFault> {
    let mapping_action = schedule.actions.iter().find(|action| action.action_id == mapping_action_id)
        .ok_or(ProjectionFault::Critical("CLOCK_MAPPING_ACTION_NOT_FOUND"))?;
    let (clock_domain, local_tick_ns, valid_from, valid_through, mapped_ref, mapped_sha, qualification) =
        match &mapping_action.action {
            ScheduleAction::ClockAdvance {
                clock_domain, local_tick_ns, mapping_valid_from_ns, mapping_valid_through_ns,
                mapping_receipt_ref, mapping_receipt_sha256, qualification, ..
            } => (
                clock_domain.as_str(),
                *local_tick_ns,
                *mapping_valid_from_ns,
                *mapping_valid_through_ns,
                mapping_receipt_ref.as_str(),
                mapping_receipt_sha256.as_str(),
                *qualification,
            ),
            _ => return Err(ProjectionFault::Critical("MAPPING_REFERENCE_NOT_CLOCK_ADVANCE")),
        };

    if clock_domain != source_clock_domain_id || mapped_ref != receipt_ref || mapped_sha != receipt_sha256 {
        return Err(ProjectionFault::Critical("CLOCK_MAPPING_RECEIPT_BINDING_MISMATCH"));
    }
    if qualification == GateState::Fail {
        return Err(ProjectionFault::Critical("CLOCK_MAPPING_QUALIFICATION_FAILED"));
    }
    if qualification != GateState::Pass {
        return Err(ProjectionFault::Indeterminate("CLOCK_MAPPING_QUALIFICATION_UNRESOLVED"));
    }
    if valid_from > valid_through {
        return Err(ProjectionFault::Indeterminate("CLOCK_MAPPING_VALIDITY_INTERVAL_INVALID"));
    }
    if local_timestamp_ns < valid_from {
        return Err(ProjectionFault::Indeterminate("CLOCK_TIMESTAMP_BEFORE_MAPPING_VALIDITY"));
    }
    if local_timestamp_ns > valid_through {
        return Err(ProjectionFault::Indeterminate("CLOCK_TIMESTAMP_AFTER_MAPPING_VALIDITY"));
    }
    if local_timestamp_ns > local_tick_ns {
        return Err(ProjectionFault::Indeterminate("CLOCK_TIMESTAMP_AFTER_MAPPING_SAMPLE"));
    }

    let qualified = resolutions.get(&(mapping_action_id.to_owned(), local_timestamp_ns))
        .cloned()
        .ok_or(ProjectionFault::Indeterminate("OWNER_QUALIFIED_TIME_RESOLUTION_MISSING"))?;
    let expected_valid_from = i64::try_from(valid_from)
        .map_err(|_| ProjectionFault::Indeterminate("CLOCK_MAPPING_VALIDITY_OUTSIDE_SUPPORTED_RANGE"))?;
    let expected_valid_through = i64::try_from(valid_through)
        .map_err(|_| ProjectionFault::Indeterminate("CLOCK_MAPPING_VALIDITY_OUTSIDE_SUPPORTED_RANGE"))?;
    if qualified.source_clock_domain_id != source_clock_domain_id
        || qualified.mapping_receipt_ref != receipt_ref
        || qualified.mapping_receipt_sha256 != receipt_sha256
        || qualified.mapping_valid_from_ns != expected_valid_from
        || qualified.mapping_valid_through_ns != expected_valid_through
    {
        return Err(ProjectionFault::Critical("OWNER_TIME_RESOLUTION_BINDING_MISMATCH"));
    }
    match compare_qualified_event_times(&qualified, &qualified).order {
        crate::EventOrder::Overlapping => Ok(qualified),
        crate::EventOrder::Unknown => Err(ProjectionFault::Indeterminate("OWNER_TIME_RESOLUTION_UNQUALIFIED")),
        crate::EventOrder::Inconsistent | crate::EventOrder::Before | crate::EventOrder::After => {
            Err(ProjectionFault::Critical("OWNER_TIME_RESOLUTION_INVALID"))
        }
    }
}

fn projection_failure(fault: ProjectionFault) -> BitemporalProjectionReport {
    match fault {
        ProjectionFault::Indeterminate(code) => BitemporalProjectionReport {
            schedule_sha256: None,
            disposition: EventTraceDisposition::Indeterminate,
            envelope: None,
            occurrence_binding: None,
            observation_binding: None,
            persistence_binding: None,
            reason_codes: vec![code.to_owned()],
        },
        ProjectionFault::Critical(code) => BitemporalProjectionReport {
            schedule_sha256: None,
            disposition: EventTraceDisposition::CriticalFailure,
            envelope: None,
            occurrence_binding: None,
            observation_binding: None,
            persistence_binding: None,
            reason_codes: vec![code.to_owned()],
        },
    }
}

/// Project a single event, observer and store lineage into the v0.2
/// bitemporal envelope. Owner-normalized times are supplied separately and
/// keyed by exact mapping action ID plus source-local timestamp. This function
/// never promotes simulator action time into owner time and never performs
/// clock normalization or receipt authentication itself.
pub fn project_bitemporal_event(
    schedule: &VirtualSchedule,
    event_id: &str,
    observer_id: &str,
    store_id: &str,
    resolutions: &BTreeMap<(String, u64), QualifiedEventTime>,
) -> Result<BitemporalProjectionReport, ReplayError> {
    validate_schedule(schedule)?;

    let produced: Vec<_> = schedule.actions.iter().filter(|action| {
        matches!(&action.action, ScheduleAction::EventProduced { event_id: candidate, .. } if candidate == event_id)
    }).collect();
    if produced.len() != 1 {
        return Ok(if produced.is_empty() {
            projection_failure(ProjectionFault::Indeterminate("EVENT_PRODUCTION_EVIDENCE_MISSING"))
        } else {
            projection_failure(ProjectionFault::Critical("EVENT_PRODUCTION_CARDINALITY_INVALID"))
        });
    }

    let observations: Vec<_> = schedule.actions.iter().filter(|action| {
        matches!(&action.action, ScheduleAction::EventObserved {
            event_id: candidate, ..
        } if candidate == event_id)
    }).collect();
    if observations.len() != 1 {
        return Ok(if observations.is_empty() {
            projection_failure(ProjectionFault::Indeterminate("FIRST_OBSERVATION_EVIDENCE_MISSING"))
        } else {
            projection_failure(ProjectionFault::Critical("FIRST_OBSERVATION_CARDINALITY_INVALID"))
        });
    }
    if !matches!(&observations[0].action, ScheduleAction::EventObserved { observer_id: selected, .. } if selected == observer_id) {
        return Ok(projection_failure(ProjectionFault::Indeterminate("SELECTED_OBSERVER_IS_NOT_FIRST_OBSERVER")));
    }

    let persistence_records: Vec<_> = schedule.actions.iter().filter(|action| {
        matches!(&action.action, ScheduleAction::EventPersisted {
            event_id: candidate, ..
        } if candidate == event_id)
    }).collect();
    if persistence_records.len() != 1 {
        return Ok(if persistence_records.is_empty() {
            projection_failure(ProjectionFault::Indeterminate("DURABLE_RECORD_EVIDENCE_MISSING"))
        } else {
            projection_failure(ProjectionFault::Critical("DURABLE_RECORD_CARDINALITY_INVALID"))
        });
    }
    if !matches!(&persistence_records[0].action, ScheduleAction::EventPersisted {
        observer_id: selected_observer, store_id: selected_store, ..
    } if selected_observer == observer_id && selected_store == store_id) {
        return Ok(projection_failure(ProjectionFault::Indeterminate("SELECTED_STORE_IS_NOT_UNIQUE_DURABLE_RECORD")));
    }

    let replay = replay_virtual_schedule(schedule)?;
    for (action, reason) in [
        (&produced[0], "EVENT_PRODUCTION_NOT_ADMISSIBLE_IN_REPLAY"),
        (&observations[0], "OBSERVATION_NOT_ADMISSIBLE_IN_REPLAY"),
        (&persistence_records[0], "DURABLE_RECORD_NOT_ADMISSIBLE_IN_REPLAY"),
    ] {
        let disposition = replay.records.iter()
            .find(|record| record.action_id == action.action_id)
            .map(|record| record.disposition);
        if disposition != Some(SchedulerDisposition::Applied) {
            return Ok(projection_failure(ProjectionFault::Indeterminate(reason)));
        }
    }

    let observed_time = match &observations[0].action {
        ScheduleAction::EventObserved {
            clock_domain, clock_mapping_action_id, observed_at_ns,
            observation_receipt_ref, observation_receipt_sha256, qualification, ..
        } => {
            if *qualification == GateState::Fail {
                return Ok(projection_failure(ProjectionFault::Critical("OBSERVATION_QUALIFICATION_FAILED")));
            }
            if *qualification != GateState::Pass {
                return Ok(projection_failure(ProjectionFault::Indeterminate("OBSERVATION_QUALIFICATION_UNRESOLVED")));
            }
            match resolve_owner_qualified_time(
                schedule, clock_mapping_action_id, *observed_at_ns, clock_domain,
                observation_receipt_ref, observation_receipt_sha256, resolutions,
            ) {
                Ok(value) => value,
                Err(fault) => return Ok(projection_failure(fault)),
            }
        }
        _ => unreachable!("filtered observation action"),
    };

    let persisted_time = match &persistence_records[0].action {
        ScheduleAction::EventPersisted {
            clock_domain, clock_mapping_action_id, persisted_at_ns,
            persistence_receipt_ref, persistence_receipt_sha256, qualification, ..
        } => {
            if *qualification == GateState::Fail {
                return Ok(projection_failure(ProjectionFault::Critical("PERSISTENCE_QUALIFICATION_FAILED")));
            }
            if *qualification != GateState::Pass {
                return Ok(projection_failure(ProjectionFault::Indeterminate("PERSISTENCE_QUALIFICATION_UNRESOLVED")));
            }
            match resolve_owner_qualified_time(
                schedule, clock_mapping_action_id, *persisted_at_ns, clock_domain,
                persistence_receipt_ref, persistence_receipt_sha256, resolutions,
            ) {
                Ok(value) => value,
                Err(fault) => return Ok(projection_failure(fault)),
            }
        }
        _ => unreachable!("filtered persistence action"),
    };

    let occurred_at = match &produced[0].action {
        ScheduleAction::EventProduced {
            clock_domain, clock_mapping_action_id, source_timestamp_ns, ..
        } => {
            let mapping = schedule.actions.iter().find(|action| action.action_id == *clock_mapping_action_id)
                .ok_or(ReplayError::InvalidCausalPredecessor(clock_mapping_action_id.clone()))?;
            match &mapping.action {
                ScheduleAction::ClockAdvance { mapping_receipt_ref, mapping_receipt_sha256, .. } => {
                    match resolve_owner_qualified_time(
                        schedule, clock_mapping_action_id, *source_timestamp_ns, clock_domain,
                        mapping_receipt_ref, mapping_receipt_sha256, resolutions,
                    ) {
                        Ok(value) => value,
                        Err(fault) => return Ok(projection_failure(fault)),
                    }
                }
                _ => return Ok(projection_failure(ProjectionFault::Critical("OCCURRENCE_MAPPING_REFERENCE_NOT_CLOCK_ADVANCE"))),
            }
        }
        _ => unreachable!("filtered event-production action"),
    };

    let mut reasons = BTreeSet::new();
    let occurrence_to_observation = compare_qualified_event_times(&occurred_at, &observed_time).order;
    let occurrence_status = match occurrence_to_observation {
        crate::EventOrder::Before | crate::EventOrder::Overlapping => None,
        crate::EventOrder::Unknown => Some(ProjectionFault::Indeterminate("OCCURRENCE_OBSERVATION_ORDER_UNRESOLVED")),
        crate::EventOrder::After | crate::EventOrder::Inconsistent => Some(ProjectionFault::Critical("OBSERVATION_PRECEDES_EVENT_OCCURRENCE")),
    };
    if let Some(fault) = occurrence_status {
        return Ok(projection_failure(fault));
    }

    let observation_to_persistence = compare_qualified_event_times(&observed_time, &persisted_time).order;
    let (disposition, capture_qualification) = match observation_to_persistence {
        crate::EventOrder::Before => (EventTraceDisposition::Safe, GateState::Pass),
        crate::EventOrder::Overlapping | crate::EventOrder::Unknown => {
            reasons.insert("OBSERVATION_PERSISTENCE_ORDER_UNRESOLVED".to_owned());
            (EventTraceDisposition::Indeterminate, GateState::Indeterminate)
        }
        crate::EventOrder::After | crate::EventOrder::Inconsistent => {
            reasons.insert("PERSISTENCE_NOT_AFTER_OBSERVATION".to_owned());
            (EventTraceDisposition::CriticalFailure, GateState::Fail)
        }
    };

    let superseded: Vec<_> = schedule.actions.iter().filter_map(|action| {
        if let ScheduleAction::EventSuperseded { event_id: current, superseded_event_id } = &action.action {
            (current == event_id).then_some(superseded_event_id.clone())
        } else {
            None
        }
    }).collect();
    if superseded.len() > 1 {
        reasons.insert("EVENT_SUPERSESSION_CARDINALITY_INVALID".to_owned());
        return Ok(BitemporalProjectionReport {
            schedule_sha256: Some(replay.schedule_sha256.clone()),
            disposition: EventTraceDisposition::CriticalFailure,
            envelope: None,
            occurrence_binding: None,
            observation_binding: None,
            persistence_binding: None,
            reason_codes: reasons.into_iter().collect(),
        });
    }

    let occurrence_binding = match &produced[0].action {
        ScheduleAction::EventProduced {
            clock_mapping_action_id, source_timestamp_ns, ..
        } => {
            let mapping = schedule.actions.iter().find(|action| action.action_id == *clock_mapping_action_id)
                .expect("mapping action was already resolved");
            let (receipt_ref, receipt_sha256) = match &mapping.action {
                ScheduleAction::ClockAdvance { mapping_receipt_ref, mapping_receipt_sha256, .. } => {
                    (mapping_receipt_ref.clone(), mapping_receipt_sha256.clone())
                }
                _ => unreachable!("mapping action was already checked"),
            };
            Some(ProjectedTimeEvidenceBinding {
                action_id: produced[0].action_id.clone(),
                clock_mapping_action_id: clock_mapping_action_id.clone(),
                source_timestamp_ns: *source_timestamp_ns,
                receipt_ref,
                receipt_sha256,
            })
        }
        _ => None,
    };
    let observation_binding = match &observations[0].action {
        ScheduleAction::EventObserved {
            clock_mapping_action_id, observed_at_ns, observation_receipt_ref,
            observation_receipt_sha256, ..
        } => Some(ProjectedTimeEvidenceBinding {
            action_id: observations[0].action_id.clone(),
            clock_mapping_action_id: clock_mapping_action_id.clone(),
            source_timestamp_ns: *observed_at_ns,
            receipt_ref: observation_receipt_ref.clone(),
            receipt_sha256: observation_receipt_sha256.clone(),
        }),
        _ => None,
    };
    let persistence_binding = match &persistence_records[0].action {
        ScheduleAction::EventPersisted {
            clock_mapping_action_id, persisted_at_ns, persistence_receipt_ref,
            persistence_receipt_sha256, ..
        } => Some(ProjectedTimeEvidenceBinding {
            action_id: persistence_records[0].action_id.clone(),
            clock_mapping_action_id: clock_mapping_action_id.clone(),
            source_timestamp_ns: *persisted_at_ns,
            receipt_ref: persistence_receipt_ref.clone(),
            receipt_sha256: persistence_receipt_sha256.clone(),
        }),
        _ => None,
    };

    Ok(BitemporalProjectionReport {
        schedule_sha256: Some(replay.schedule_sha256.clone()),
        disposition,
        envelope: Some(BitemporalEventEnvelope {
            event_id: event_id.to_owned(),
            occurred_at: Some(occurred_at),
            first_observed_at: observed_time,
            durably_recorded_at: persisted_time,
            capture_qualification,
            supersedes_event_id: superseded.into_iter().next(),
        }),
        occurrence_binding,
        observation_binding,
        persistence_binding,
        reason_codes: reasons.into_iter().collect(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureClockFault {
    ObserverUnqualified,
    ObserverExpired,
    ObserverNotYetValid,
    ObserverValidityInvalid,
    StoreUnqualified,
    StoreExpired,
    StoreNotYetValid,
    StoreValidityInvalid,
}

/// Mutate the exact observer or store mapping in a generated capture history
/// to produce a named calibration/validity fault. The schedule remains
/// structurally valid so the replay classifier, not the parser, sees the fault.
pub fn generate_capture_clock_fault_schedule(
    seed: u64,
    fault: CaptureClockFault,
) -> Result<VirtualSchedule, ReplayError> {
    let mut schedule = generate_clocked_observation_persistence_schedule(seed)?;
    let (mapping_id, timestamp_ns, is_observer) = match fault {
        CaptureClockFault::ObserverUnqualified
        | CaptureClockFault::ObserverExpired
        | CaptureClockFault::ObserverNotYetValid
        | CaptureClockFault::ObserverValidityInvalid => {
            schedule.actions.iter().find_map(|action| {
                if let ScheduleAction::EventObserved {
                    clock_mapping_action_id, observed_at_ns, ..
                } = &action.action {
                    Some((clock_mapping_action_id.clone(), *observed_at_ns, true))
                } else {
                    None
                }
            }).ok_or(ReplayError::MissingClock("observer-clock".to_owned()))?
        }
        CaptureClockFault::StoreUnqualified
        | CaptureClockFault::StoreExpired
        | CaptureClockFault::StoreNotYetValid
        | CaptureClockFault::StoreValidityInvalid => {
            schedule.actions.iter().find_map(|action| {
                if let ScheduleAction::EventPersisted {
                    clock_mapping_action_id, persisted_at_ns, ..
                } = &action.action {
                    Some((clock_mapping_action_id.clone(), *persisted_at_ns, false))
                } else {
                    None
                }
            }).ok_or(ReplayError::MissingClock("store-clock".to_owned()))?
        }
    };
    let mapping = schedule.actions.iter_mut()
        .find(|action| action.action_id == mapping_id)
        .ok_or(ReplayError::InvalidCausalPredecessor(mapping_id.clone()))?;
    let ScheduleAction::ClockAdvance {
        mapping_valid_from_ns, mapping_valid_through_ns, qualification, ..
    } = &mut mapping.action else {
        return Err(ReplayError::InvalidCausalPredecessor(mapping_id));
    };

    match fault {
        CaptureClockFault::ObserverUnqualified | CaptureClockFault::StoreUnqualified => {
            *qualification = GateState::Indeterminate;
        }
        CaptureClockFault::ObserverExpired | CaptureClockFault::StoreExpired => {
            *mapping_valid_through_ns = timestamp_ns.saturating_sub(1);
        }
        CaptureClockFault::ObserverNotYetValid | CaptureClockFault::StoreNotYetValid => {
            *mapping_valid_from_ns = timestamp_ns.saturating_add(1);
        }
        CaptureClockFault::ObserverValidityInvalid | CaptureClockFault::StoreValidityInvalid => {
            *mapping_valid_from_ns = mapping_valid_through_ns.saturating_add(1);
        }
    }

    schedule.generator_version = GENERATOR_VERSION.to_owned();
    schedule.scenario_family = format!("capture_clock_fault_{fault:?}").to_ascii_lowercase();
    debug_assert!(if is_observer {
        matches!(fault, CaptureClockFault::ObserverUnqualified
            | CaptureClockFault::ObserverExpired
            | CaptureClockFault::ObserverNotYetValid
            | CaptureClockFault::ObserverValidityInvalid)
    } else {
        matches!(fault, CaptureClockFault::StoreUnqualified
            | CaptureClockFault::StoreExpired
            | CaptureClockFault::StoreNotYetValid
            | CaptureClockFault::StoreValidityInvalid)
    });
    validate_schedule(&schedule)?;
    Ok(schedule)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerReceiptAssessment {
    pub identity: GateState,
    pub digest_integrity: GateState,
    pub authenticity: GateState,
    pub scope_qualification: GateState,
    pub completeness: GateState,
    pub semantic_qualification: GateState,
    pub in_flight_reconciliation: GateState,
}

impl OwnerReceiptAssessment {
    fn receipt_qualification(self) -> GateState {
        let gates = [
            self.identity,
            self.digest_integrity,
            self.authenticity,
            self.scope_qualification,
            self.semantic_qualification,
        ];
        if gates.contains(&GateState::Fail) {
            GateState::Fail
        } else if gates.iter().all(|gate| *gate == GateState::Pass) {
            GateState::Pass
        } else {
            GateState::Indeterminate
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectedEffectCommit {
    pub action_id: String,
    pub logical_operation_id: String,
    pub attempt_id: String,
    pub effect_id: String,
    pub owner_time: QualifiedEventTime,
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub receipt_qualification: GateState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionHistoryProjectionReport {
    pub disposition: EventTraceDisposition,
    pub attempts: Vec<OperationAttemptRecord>,
    pub attempt_lifecycle: AttemptLifecycleReport,
    pub obligations: Vec<ScopedStopObligation>,
    pub inventory: Option<AttemptHistoryInventory>,
    pub start_gate: Option<StartGateReport>,
    pub effect_commits: Vec<ProjectedEffectCommit>,
    pub reason_codes: Vec<String>,
}

fn action_owner_time(
    schedule: &VirtualSchedule,
    action: &ScheduledAction,
    resolutions: &BTreeMap<(String, u64), QualifiedEventTime>,
) -> Result<QualifiedEventTime, ProjectionFault> {
    let reference = action.owner_time_ref.as_ref()
        .ok_or(ProjectionFault::Indeterminate("ACTION_OWNER_TIME_REFERENCE_MISSING"))?;
    let mapping_action = schedule.actions.iter()
        .find(|candidate| candidate.action_id == reference.clock_mapping_action_id)
        .ok_or(ProjectionFault::Critical("ACTION_OWNER_TIME_MAPPING_NOT_FOUND"))?;
    let (clock_domain, receipt_ref, receipt_sha256) = match &mapping_action.action {
        ScheduleAction::ClockAdvance {
            clock_domain, mapping_receipt_ref, mapping_receipt_sha256, ..
        } => (clock_domain.as_str(), mapping_receipt_ref.as_str(), mapping_receipt_sha256.as_str()),
        _ => return Err(ProjectionFault::Critical("ACTION_OWNER_TIME_REFERENCE_NOT_CLOCK_MAPPING")),
    };
    resolve_owner_qualified_time(
        schedule,
        &reference.clock_mapping_action_id,
        reference.local_timestamp_ns,
        clock_domain,
        receipt_ref,
        receipt_sha256,
        resolutions,
    )
}

fn receipt_assessment_for_action<'a>(
    action_id: &str,
    assessments: &'a BTreeMap<String, OwnerReceiptAssessment>,
) -> Result<OwnerReceiptAssessment, ProjectionFault> {
    let assessment = assessments.get(action_id)
        .copied()
        .ok_or(ProjectionFault::Indeterminate("OWNER_RECEIPT_ASSESSMENT_MISSING"))?;
    match assessment.receipt_qualification() {
        GateState::Pass => Ok(assessment),
        GateState::Fail => Err(ProjectionFault::Critical("OWNER_RECEIPT_QUALIFICATION_FAILED")),
        GateState::Indeterminate | GateState::NotApplicable => {
            Err(ProjectionFault::Indeterminate("OWNER_RECEIPT_QUALIFICATION_UNRESOLVED"))
        }
    }
}

fn action_projection_failure(fault: ProjectionFault) -> ActionHistoryProjectionReport {
    let (disposition, code) = match fault {
        ProjectionFault::Indeterminate(code) => (EventTraceDisposition::Indeterminate, code),
        ProjectionFault::Critical(code) => (EventTraceDisposition::CriticalFailure, code),
    };
    ActionHistoryProjectionReport {
        disposition,
        attempts: Vec::new(),
        attempt_lifecycle: AttemptLifecycleReport {
            disposition,
            pending_attempt_ids: Vec::new(),
            reason_codes: vec![code.to_owned()],
        },
        obligations: Vec::new(),
        inventory: None,
        start_gate: None,
        effect_commits: Vec::new(),
        reason_codes: vec![code.to_owned()],
    }
}

fn event_order_between(
    first: &QualifiedEventTime,
    second: &QualifiedEventTime,
) -> EventOrder {
    compare_qualified_event_times(first, second).order
}

/// Project scheduled effect/attempt/stop evidence into v0.2's typed attempt,
/// obligation and inventory contracts. Owner time and owner receipt results
/// are injected by the caller; this function never authenticates bytes,
/// normalizes clocks, synthesizes PASS, authorizes execution, or treats a
/// cancellation acknowledgement as a NOT_APPLIED terminal result.
pub fn project_action_history(
    schedule: &VirtualSchedule,
    logical_operation_id: &str,
    time_resolutions: &BTreeMap<(String, u64), QualifiedEventTime>,
    receipt_assessments: &BTreeMap<String, OwnerReceiptAssessment>,
) -> Result<ActionHistoryProjectionReport, ReplayError> {
    validate_schedule(schedule)?;
    let replay = replay_virtual_schedule(schedule)?;
    let canonical_actions = ordered_actions(schedule);
    let mut reasons = BTreeSet::new();
    let mut critical = false;
    let mut indeterminate = false;

    let record_disposition = |action_id: &str| {
        replay.records.iter()
            .find(|record| record.action_id == action_id)
            .map(|record| record.disposition)
    };

    let mut starts = BTreeMap::<(String, String), (&ScheduledAction, QualifiedEventTime, Option<String>)>::new();
    let mut terminal_actions = BTreeMap::<(String, String), (&ScheduledAction, QualifiedEventTime, OwnerReceiptAssessment)>::new();
    let mut commits = Vec::new();

    for action in &canonical_actions {
        match &action.action {
            ScheduleAction::EffectAttempted { operation_id, attempt_id } => {
                let key = (operation_id.clone(), attempt_id.clone());
                if starts.contains_key(&key) {
                    critical = true;
                    reasons.insert(format!("DUPLICATE_ATTEMPT_START_ACTION:{operation_id}:{attempt_id}"));
                    continue;
                }
                if record_disposition(&action.action_id) != Some(SchedulerDisposition::Applied) {
                    indeterminate = true;
                    reasons.insert(format!("ATTEMPT_START_NOT_REPLAY_APPLIED:{attempt_id}"));
                    continue;
                }
                let retry_of = match &action.evidence {
                    Some(ActionEvidence::AttemptStart { retry_of_attempt_id }) => retry_of_attempt_id.clone(),
                    _ => {
                        indeterminate = true;
                        reasons.insert(format!("ATTEMPT_START_LINEAGE_EVIDENCE_MISSING:{attempt_id}"));
                        None
                    }
                };
                match action_owner_time(schedule, action, time_resolutions) {
                    Ok(owner_time) => {
                        starts.insert(key, (action, owner_time, retry_of));
                    }
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                    }
                }
            }
            ScheduleAction::AttemptReconciled {
                operation_id, attempt_id, outcome, receipt_ref, receipt_sha256,
            } => {
                let key = (operation_id.clone(), attempt_id.clone());
                if !starts.contains_key(&key) {
                    critical = true;
                    reasons.insert(format!("TERMINAL_WITHOUT_ATTEMPT_START:{attempt_id}"));
                    continue;
                }
                match record_disposition(&action.action_id) {
                    Some(SchedulerDisposition::Applied) => {}
                    Some(SchedulerDisposition::Duplicate) => continue,
                    _ => {
                        critical = true;
                        reasons.insert(format!("TERMINAL_NOT_REPLAY_APPLIED:{attempt_id}"));
                        continue;
                    }
                }
                let (exact_start_action_id, qualification) = match &action.evidence {
                    Some(ActionEvidence::TerminalReconciliation { exact_start_action_id, qualification }) => {
                        (exact_start_action_id, *qualification)
                    }
                    _ => {
                        indeterminate = true;
                        reasons.insert(format!("TERMINAL_LINEAGE_EVIDENCE_MISSING:{attempt_id}"));
                        continue;
                    }
                };
                let (start_action, _, _) = starts.get(&key).expect("start was inserted");
                if exact_start_action_id != &start_action.action_id {
                    critical = true;
                    reasons.insert(format!("TERMINAL_START_ACTION_BINDING_MISMATCH:{attempt_id}"));
                    continue;
                }
                let assessment = match receipt_assessment_for_action(&action.action_id, receipt_assessments) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                };
                if qualification != GateState::Pass {
                    if qualification == GateState::Fail {
                        critical = true;
                        reasons.insert(format!("TERMINAL_QUALIFICATION_FAILED:{attempt_id}"));
                    } else {
                        indeterminate = true;
                        reasons.insert(format!("TERMINAL_QUALIFICATION_UNRESOLVED:{attempt_id}"));
                    }
                    continue;
                }
                let terminal_time = match action_owner_time(schedule, action, time_resolutions) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                };
                if let Some(previous) = terminal_actions.insert(key.clone(), (action, terminal_time, assessment)) {
                    let _ = previous;
                    critical = true;
                    reasons.insert(format!("MULTIPLE_TERMINAL_ACTIONS:{attempt_id}"));
                }
                let _ = (outcome, receipt_ref, receipt_sha256);
            }
            ScheduleAction::EffectCommitted {
                operation_id, attempt_id, effect_id, ..
            } => {
                if !starts.contains_key(&(operation_id.clone(), attempt_id.clone())) {
                    critical = true;
                    reasons.insert(format!("EFFECT_COMMIT_WITHOUT_ATTEMPT_START:{attempt_id}"));
                    continue;
                }
                match record_disposition(&action.action_id) {
                    Some(SchedulerDisposition::Applied) => {}
                    Some(SchedulerDisposition::Duplicate) => continue,
                    _ => {
                        critical = true;
                        reasons.insert(format!("EFFECT_COMMIT_NOT_REPLAY_APPLIED:{attempt_id}"));
                        continue;
                    }
                }
                let (receipt_ref, receipt_sha256) = match &action.evidence {
                    Some(ActionEvidence::EffectCommit { receipt_ref, receipt_sha256 }) => {
                        (receipt_ref.clone(), receipt_sha256.clone())
                    }
                    _ => {
                        indeterminate = true;
                        reasons.insert(format!("EFFECT_COMMIT_RECEIPT_EVIDENCE_MISSING:{attempt_id}"));
                        continue;
                    }
                };
                let assessment = match receipt_assessment_for_action(&action.action_id, receipt_assessments) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                };
                let owner_time = match action_owner_time(schedule, action, time_resolutions) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{attempt_id}"));
                        continue;
                    }
                };
                commits.push(ProjectedEffectCommit {
                    action_id: action.action_id.clone(),
                    logical_operation_id: operation_id.clone(),
                    attempt_id: attempt_id.clone(),
                    effect_id: effect_id.clone(),
                    owner_time,
                    receipt_ref,
                    receipt_sha256,
                    receipt_qualification: assessment.receipt_qualification(),
                });
            }
            _ => {}
        }
    }

    let mut attempts = Vec::new();
    let mut attempt_start_times = BTreeMap::<String, QualifiedEventTime>::new();
    let mut attempt_terminal_times = BTreeMap::<String, QualifiedEventTime>::new();

    for ((operation_id, attempt_id), (start_action, start_time, retry_of)) in &starts {
        let start_event_id = start_action.action_id.clone();
        attempt_start_times.insert(attempt_id.clone(), start_time.clone());
        let terminal = terminal_actions.get(&(operation_id.clone(), attempt_id.clone()));
        let (terminal_record, terminal_time) = if let Some((terminal_action, terminal_time, _assessment)) = terminal {
            let (outcome, receipt_ref, receipt_sha256) = match &terminal_action.action {
                ScheduleAction::AttemptReconciled { outcome, receipt_ref, receipt_sha256, .. } => {
                    (*outcome, receipt_ref.clone(), receipt_sha256.clone())
                }
                _ => unreachable!("terminal action index is typed"),
            };
            let (exact_start_action_id, qualification) = match &terminal_action.evidence {
                Some(ActionEvidence::TerminalReconciliation { exact_start_action_id, qualification }) => {
                    (exact_start_action_id.clone(), *qualification)
                }
                _ => unreachable!("evidence checked during collection"),
            };
            let _ = (receipt_ref, receipt_sha256);
            let start_to_terminal = event_order_between(start_time, terminal_time);
            let mapped_outcome = match outcome {
                ReconciliationOutcome::Applied => Some(ReconciledOperationOutcome::Applied),
                ReconciliationOutcome::NotApplied => Some(ReconciledOperationOutcome::NotApplied),
                ReconciliationOutcome::Unknown => None,
            };
            if let Some(mapped_outcome) = mapped_outcome {
                let record = AttemptTerminalReconciliation {
                    event_id: terminal_action.action_id.clone(),
                    start_event_id: exact_start_action_id,
                    outcome: mapped_outcome,
                    qualification,
                    start_to_terminal_order: start_to_terminal,
                };
                attempt_terminal_times.insert(attempt_id.clone(), terminal_time.clone());
                (Some(record), Some(terminal_time.clone()))
            } else {
                indeterminate = true;
                reasons.insert(format!("TERMINAL_OUTCOME_UNKNOWN:{attempt_id}"));
                (None, None)
            }
        } else {
            (None, None)
        };

        attempts.push(OperationAttemptRecord {
            identity: AttemptIdentity {
                logical_operation_id: operation_id.clone(),
                attempt_id: attempt_id.clone(),
                retry_of_attempt_id: retry_of.clone(),
            },
            start_event_ids: vec![start_event_id],
            terminal_reconciliation_event_ids: terminal_record.iter().map(|item| item.event_id.clone()).collect(),
            terminal_reconciliation: terminal_record,
            stop_discharge_relations: Vec::new(),
            retry_predecessor_terminal_event_id: None,
            retry_predecessor_start_event_id: None,
            retry_predecessor_terminal_to_start_order: None,
        });
        let _ = terminal_time;
    }

    let mut obligations = Vec::<ScopedStopObligation>::new();
    let mut inventories = Vec::<(AttemptHistoryInventory, String)>::new();
    let mut stop_trigger_actions = BTreeMap::<String, (&ScheduledAction, StopScope, String, QualifiedEventTime)>::new();
    let mut latest_policy_actions = BTreeMap::<String, (&ScheduledAction, StopScope, String, QualifiedEventTime)>::new();
    for action in &canonical_actions {
        match &action.action {
            ScheduleAction::StopTriggered { obligation_id, policy_revision } => {
                let (scope, receipt_ref, receipt_sha256) = match &action.evidence {
                    Some(ActionEvidence::StopTrigger { scope, receipt_ref, receipt_sha256 }) => {
                        (scope.clone(), receipt_ref.clone(), receipt_sha256.clone())
                    }
                    _ => {
                        indeterminate = true;
                        reasons.insert(format!("STOP_TRIGGER_EVIDENCE_MISSING:{obligation_id}"));
                        continue;
                    }
                };
                let assessment = match receipt_assessment_for_action(&action.action_id, receipt_assessments) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                };
                let owner_time = match action_owner_time(schedule, action, time_resolutions) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                };
                let _ = (receipt_ref, receipt_sha256, assessment);
                if stop_trigger_actions.insert(obligation_id.clone(), (action, scope.clone(), policy_revision.clone(), owner_time.clone())).is_some() {
                    critical = true;
                    reasons.insert(format!("DUPLICATE_STOP_TRIGGER:{obligation_id}"));
                    continue;
                }
                latest_policy_actions.insert(obligation_id.clone(), (action, scope, policy_revision.clone(), owner_time));
            }
            ScheduleAction::StopPolicyRevised { obligation_id, policy_revision } => {
                let (scope, receipt_ref, receipt_sha256) = match &action.evidence {
                    Some(ActionEvidence::StopPolicyRevision { scope, receipt_ref, receipt_sha256 }) => {
                        (scope.clone(), receipt_ref.clone(), receipt_sha256.clone())
                    }
                    _ => {
                        indeterminate = true;
                        reasons.insert(format!("STOP_POLICY_REVISION_EVIDENCE_MISSING:{obligation_id}"));
                        continue;
                    }
                };
                let Some((trigger_action, _trigger_scope, _old_policy, trigger_time)) = stop_trigger_actions.get(obligation_id) else {
                    critical = true;
                    reasons.insert(format!("STOP_POLICY_REVISION_WITHOUT_TRIGGER:{obligation_id}"));
                    continue;
                };
                let assessment = match receipt_assessment_for_action(&action.action_id, receipt_assessments) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                };
                let owner_time = match action_owner_time(schedule, action, time_resolutions) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(format!("{code}:{obligation_id}"));
                        continue;
                    }
                };
                let mut revision_is_ordered = true;
                match event_order_between(trigger_time, &owner_time) {
                    EventOrder::Before => {}
                    EventOrder::After | EventOrder::Inconsistent => {
                        critical = true;
                        revision_is_ordered = false;
                        reasons.insert(format!("STOP_POLICY_REVISION_PRECEDES_TRIGGER:{obligation_id}"));
                    }
                    EventOrder::Overlapping | EventOrder::Unknown => {
                        indeterminate = true;
                        revision_is_ordered = false;
                        reasons.insert(format!("STOP_POLICY_REVISION_ORDER_UNRESOLVED:{obligation_id}"));
                    }
                }
                if let Some((_previous_action, _previous_scope, _previous_policy, previous_time)) = latest_policy_actions.get(obligation_id) {
                    match event_order_between(previous_time, &owner_time) {
                        EventOrder::Before => {}
                        EventOrder::After | EventOrder::Inconsistent => {
                            critical = true;
                            revision_is_ordered = false;
                            reasons.insert(format!("STOP_POLICY_REVISION_ORDER_CONTRADICTION:{obligation_id}"));
                        }
                        EventOrder::Overlapping | EventOrder::Unknown => {
                            indeterminate = true;
                            revision_is_ordered = false;
                            reasons.insert(format!("STOP_POLICY_REVISION_SEQUENCE_UNRESOLVED:{obligation_id}"));
                        }
                    }
                }
                let _ = (trigger_action, receipt_ref, receipt_sha256, assessment);
                if revision_is_ordered {
                    latest_policy_actions.insert(obligation_id.clone(), (action, scope, policy_revision.clone(), owner_time));
                }
            }
            ScheduleAction::AttemptInventoryAttested {
                scope, inventory_frontier_event_id, attempt_manifest_sha256,
            } => {
                let (evidence_scope, evidence_frontier, evidence_manifest, receipt_ref, receipt_sha256) =
                    match &action.evidence {
                        Some(ActionEvidence::InventoryAttestation {
                            scope, inventory_frontier_event_id, attempt_manifest_sha256,
                            receipt_ref, receipt_sha256,
                        }) => (scope, inventory_frontier_event_id, attempt_manifest_sha256, receipt_ref, receipt_sha256),
                        _ => {
                            indeterminate = true;
                            reasons.insert("ATTEMPT_INVENTORY_RECEIPT_EVIDENCE_MISSING".to_owned());
                            continue;
                        }
                    };
                if evidence_scope != scope || evidence_frontier != inventory_frontier_event_id
                    || evidence_manifest != attempt_manifest_sha256
                {
                    critical = true;
                    reasons.insert("ATTEMPT_INVENTORY_EVIDENCE_BINDING_MISMATCH".to_owned());
                    continue;
                }
                let assessment = match receipt_assessment_for_action(&action.action_id, receipt_assessments) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(code.to_owned());
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(code.to_owned());
                        continue;
                    }
                };
                let _owner_time = match action_owner_time(schedule, action, time_resolutions) {
                    Ok(value) => value,
                    Err(ProjectionFault::Indeterminate(code)) => {
                        indeterminate = true;
                        reasons.insert(code.to_owned());
                        continue;
                    }
                    Err(ProjectionFault::Critical(code)) => {
                        critical = true;
                        reasons.insert(code.to_owned());
                        continue;
                    }
                };
                inventories.push((AttemptHistoryInventory {
                    inventory_frontier_event_id: inventory_frontier_event_id.clone(),
                    attempt_manifest_sha256: attempt_manifest_sha256.clone(),
                    receipt_ref: receipt_ref.clone(),
                    receipt_sha256: receipt_sha256.clone(),
                    scope: scope.clone(),
                    identity: assessment.identity,
                    digest_integrity: assessment.digest_integrity,
                    authenticity: assessment.authenticity,
                    scope_qualification: assessment.scope_qualification,
                    completeness: assessment.completeness,
                }, action.action_id.clone()));
            }
            _ => {}
        }
    }

    let mut discharged_obligation_ids = BTreeSet::new();
    for action in &canonical_actions {
        let ScheduleAction::StopDischarged {
            obligation_id, trigger_action_id, policy_revision_action_id, policy_revision,
            receipt_ref, receipt_sha256,
        } = &action.action else { continue; };
        let Some((trigger_action, trigger_scope, _trigger_policy, trigger_time)) = stop_trigger_actions.get(obligation_id) else {
            critical = true;
            reasons.insert(format!("STOP_DISCHARGE_WITHOUT_TRIGGER:{obligation_id}"));
            continue;
        };
        let Some((current_policy_action, current_scope, current_policy, _revision_time)) = latest_policy_actions.get(obligation_id) else {
            critical = true;
            reasons.insert(format!("STOP_DISCHARGE_WITHOUT_CURRENT_POLICY:{obligation_id}"));
            continue;
        };
        if trigger_action.action_id != *trigger_action_id
            || current_policy_action.action_id != *policy_revision_action_id
            || current_policy != policy_revision
        {
            critical = true;
            reasons.insert(format!("STOP_DISCHARGE_ACTION_LINEAGE_MISMATCH:{obligation_id}"));
            continue;
        }
        let (scope, frontier, manifest) = match &action.evidence {
            Some(ActionEvidence::StopDischarge { scope, inventory_frontier_event_id, attempt_manifest_sha256 }) => {
                (scope.clone(), inventory_frontier_event_id.clone(), attempt_manifest_sha256.clone())
            }
            _ => {
                indeterminate = true;
                reasons.insert(format!("STOP_DISCHARGE_INVENTORY_BINDING_MISSING:{obligation_id}"));
                continue;
            }
        };
        if scope != *trigger_scope || scope != *current_scope {
            critical = true;
            reasons.insert(format!("STOP_DISCHARGE_SCOPE_MISMATCH:{obligation_id}"));
            continue;
        }
        let inventory_match = inventories.iter().filter(|(inventory, _)| {
            inventory.scope == scope
                && inventory.inventory_frontier_event_id == frontier
                && inventory.attempt_manifest_sha256 == manifest
        }).collect::<Vec<_>>();
        if inventory_match.len() != 1 {
            critical = true;
            reasons.insert(format!("STOP_DISCHARGE_INVENTORY_ATTESTATION_NOT_UNIQUE:{obligation_id}"));
            continue;
        }
        let assessment = match receipt_assessment_for_action(&action.action_id, receipt_assessments) {
            Ok(value) => value,
            Err(ProjectionFault::Indeterminate(code)) => {
                indeterminate = true;
                reasons.insert(format!("{code}:{obligation_id}"));
                continue;
            }
            Err(ProjectionFault::Critical(code)) => {
                critical = true;
                reasons.insert(format!("{code}:{obligation_id}"));
                continue;
            }
        };
        let discharge_time = match action_owner_time(schedule, action, time_resolutions) {
            Ok(value) => value,
            Err(ProjectionFault::Indeterminate(code)) => {
                indeterminate = true;
                reasons.insert(format!("{code}:{obligation_id}"));
                continue;
            }
            Err(ProjectionFault::Critical(code)) => {
                critical = true;
                reasons.insert(format!("{code}:{obligation_id}"));
                continue;
            }
        };
        let trigger_order = event_order_between(trigger_time, &discharge_time);
        if trigger_order != EventOrder::Before {
            match trigger_order {
                EventOrder::After | EventOrder::Inconsistent => {
                    critical = true;
                    reasons.insert(format!("STOP_DISCHARGE_PRECEDES_TRIGGER:{obligation_id}"));
                }
                EventOrder::Overlapping | EventOrder::Unknown => {
                    indeterminate = true;
                    reasons.insert(format!("STOP_DISCHARGE_ORDER_UNRESOLVED:{obligation_id}"));
                }
                EventOrder::Before => {}
            }
        }
        let discharge = StopDischargeReceipt {
            event_id: action.action_id.clone(),
            obligation_id: obligation_id.clone(),
            trigger_event_id: trigger_action.action_id.clone(),
            policy_revision: trigger_policy.clone(),
            scope: scope.clone(),
            receipt_ref: receipt_ref.clone(),
            receipt_sha256: receipt_sha256.clone(),
            qualification: assessment.receipt_qualification(),
            in_flight_reconciliation: assessment.in_flight_reconciliation,
            in_flight_inventory_frontier_event_id: frontier,
            in_flight_attempt_manifest_sha256: manifest,
            trigger_to_discharge_order: trigger_order,
        };
        if !discharged_obligation_ids.insert(obligation_id.clone()) {
            critical = true;
            reasons.insert(format!("DUPLICATE_STOP_DISCHARGE:{obligation_id}"));
            continue;
        }
        let inventory_action_id = &inventory_match[0].1;
        if let Some(inventory_action) = schedule.actions.iter().find(|item| item.action_id == *inventory_action_id) {
            match action_owner_time(schedule, inventory_action, time_resolutions) {
                Ok(inventory_time) => match event_order_between(&inventory_time, &discharge_time) {
                    EventOrder::Before => {}
                    EventOrder::After | EventOrder::Inconsistent => {
                        critical = true;
                        reasons.insert(format!("INVENTORY_ATTESTATION_AFTER_DISCHARGE:{obligation_id}"));
                    }
                    EventOrder::Overlapping | EventOrder::Unknown => {
                        indeterminate = true;
                        reasons.insert(format!("INVENTORY_ATTESTATION_ORDER_UNRESOLVED:{obligation_id}"));
                    }
                },
                Err(ProjectionFault::Indeterminate(code)) => {
                    indeterminate = true;
                    reasons.insert(format!("{code}:{obligation_id}"));
                }
                Err(ProjectionFault::Critical(code)) => {
                    critical = true;
                    reasons.insert(format!("{code}:{obligation_id}"));
                }
            }
        }
        obligations.push(ScopedStopObligation {
            obligation_id: obligation_id.clone(),
            trigger_event_id: trigger_action.action_id.clone(),
            policy_revision: current_policy.clone(),
            scope,
            discharge: Some(discharge),
        });
    }

    for (obligation_id, (trigger_action, trigger_scope, _trigger_policy, _trigger_time)) in &stop_trigger_actions {
        if discharged_obligation_ids.contains(obligation_id) {
            continue;
        }
        let current_policy = latest_policy_actions.get(obligation_id)
            .map(|entry| entry.2.clone())
            .unwrap_or_else(|| _trigger_policy.clone());
        obligations.push(ScopedStopObligation {
            obligation_id: obligation_id.clone(),
            trigger_event_id: trigger_action.action_id.clone(),
            policy_revision: current_policy,
            scope: trigger_scope.clone(),
            discharge: None,
        });
        indeterminate = true;
        reasons.insert(format!("STOP_OBLIGATION_UNDISCHARGED:{obligation_id}"));
    }

    if inventories.len() != 1 {
        indeterminate = true;
        reasons.insert("COMPLETE_ATTEMPT_INVENTORY_NOT_UNIQUE".to_owned());
    }
    let inventory = (inventories.len() == 1).then(|| inventories[0].0.clone());

    let attempt_index: BTreeMap<String, usize> = attempts.iter().enumerate()
        .map(|(index, attempt)| (attempt.identity.attempt_id.clone(), index))
        .collect();

    for index in 0..attempts.len() {
        let attempt_id = attempts[index].identity.attempt_id.clone();
        let logical_id = attempts[index].identity.logical_operation_id.clone();
        let start_time = attempt_start_times.get(&attempt_id).cloned();
        if let Some(parent_id) = attempts[index].identity.retry_of_attempt_id.clone() {
            let parent = attempt_index.get(&parent_id).copied();
            if let Some(parent_index) = parent {
                let parent_record = &attempts[parent_index];
                attempts[index].retry_predecessor_start_event_id = parent_record.start_event_ids.first().cloned();
                attempts[index].retry_predecessor_terminal_event_id =
                    parent_record.terminal_reconciliation.as_ref().map(|terminal| terminal.event_id.clone());
                if let (Some(parent_terminal_time), Some(current_start_time)) = (
                    attempt_terminal_times.get(&parent_id),
                    start_time.as_ref(),
                ) {
                    attempts[index].retry_predecessor_terminal_to_start_order =
                        Some(event_order_between(parent_terminal_time, current_start_time));
                }
            }
        }

        for obligation in obligations.iter().filter(|obligation| {
            obligation.scope.applies_to(&logical_id)
        }) {
            let Some(discharge) = obligation.discharge.as_ref() else { continue; };
            let discharge_action = schedule.actions.iter().find(|item| item.action_id == discharge.event_id);
            let Some(discharge_action) = discharge_action else {
                critical = true;
                reasons.insert(format!("STOP_DISCHARGE_ACTION_NOT_FOUND:{}", discharge.event_id));
                continue;
            };
            let discharge_time = match action_owner_time(schedule, discharge_action, time_resolutions) {
                Ok(value) => value,
                Err(_) => {
                    indeterminate = true;
                    reasons.insert(format!("STOP_DISCHARGE_OWNER_TIME_MISSING:{}", discharge.event_id));
                    continue;
                }
            };
            let Some(start_time) = start_time.as_ref() else {
                indeterminate = true;
                reasons.insert(format!("ATTEMPT_START_OWNER_TIME_MISSING:{attempt_id}"));
                continue;
            };
            let discharge_to_start = event_order_between(&discharge_time, start_time);
            let (terminal_event_id_at_discharge, terminal_to_discharge_order) = if discharge_to_start == EventOrder::After {
                if let Some(terminal_time) = attempt_terminal_times.get(&attempt_id) {
                    let terminal_id = attempts[index].terminal_reconciliation.as_ref().map(|item| item.event_id.clone());
                    (terminal_id, Some(event_order_between(terminal_time, &discharge_time)))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };
            let attempt_start_event_id = attempts[index].start_event_ids[0].clone();
            attempts[index].stop_discharge_relations.push(AttemptStopDischargeRelation {
                obligation_id: obligation.obligation_id.clone(),
                discharge_event_id: discharge.event_id.clone(),
                attempt_start_event_id,
                discharge_to_start_order: discharge_to_start,
                terminal_event_id_at_discharge,
                terminal_to_discharge_order,
            });
        }
    }

    let attempt_lifecycle = validate_attempt_lifecycle(&attempts);
    if attempt_lifecycle.disposition == EventTraceDisposition::CriticalFailure {
        critical = true;
        reasons.extend(attempt_lifecycle.reason_codes.clone());
    } else if attempt_lifecycle.disposition == EventTraceDisposition::Indeterminate {
        indeterminate = true;
        reasons.extend(attempt_lifecycle.reason_codes.clone());
    }

    let inventory = inventory;
    let start_gate = if let Some(inventory) = inventory.as_ref() {
        Some(assess_start_under_stop_obligations(
            logical_operation_id, &attempts, inventory, &obligations,
        ))
    } else {
        None
    };
    if let Some(gate) = &start_gate {
        if gate.disposition == EventTraceDisposition::CriticalFailure {
            critical = true;
            reasons.extend(gate.reason_codes.clone());
        } else if gate.disposition == EventTraceDisposition::Indeterminate {
            indeterminate = true;
            reasons.extend(gate.reason_codes.clone());
        }
    }

    let disposition = if critical {
        EventTraceDisposition::CriticalFailure
    } else if indeterminate {
        EventTraceDisposition::Indeterminate
    } else {
        EventTraceDisposition::Safe
    };
    Ok(ActionHistoryProjectionReport {
        disposition,
        attempts,
        attempt_lifecycle,
        obligations,
        inventory,
        start_gate,
        effect_commits: commits,
        reason_codes: reasons.into_iter().collect(),
    })
}

/// Generate a deterministic stop/cancellation/effect-commit race. The
/// attempt starts before the stop trigger; both schedules then contain a
/// cancellation request, the selected acknowledgement/commit order, an
/// explicit terminal reconciliation and an exact stop discharge. A commit
/// after an accepted cancellation (or an accepted cancellation after commit)
/// is preserved as a conflict disposition rather than hidden.
pub fn generate_cancellation_commit_race_schedule(
    seed: u64,
    order: CancellationCommitOrder,
) -> VirtualSchedule {
    let mut actions = Vec::new();
    let mut ordinal = 0_u64;
    let operation_id = format!("operation-{seed:016x}");
    let attempt_id = format!("attempt-{seed:016x}");
    let obligation_id = format!("stop-{seed:016x}");

    push_action(
        &mut actions,
        &mut ordinal,
        1,
        ScheduleAction::EffectAttempted {
            operation_id: operation_id.clone(),
            attempt_id: attempt_id.clone(),
        },
    );
    // The next ordinal is the exact ID of the trigger action below.
    let trigger_action_id = format!("action-{ordinal:06}");
    push_action(
        &mut actions,
        &mut ordinal,
        2,
        ScheduleAction::StopTriggered {
            obligation_id: obligation_id.clone(),
            policy_revision: "policy-v1".to_owned(),
        },
    );
    push_action(
        &mut actions,
        &mut ordinal,
        3,
        ScheduleAction::CancellationRequested {
            operation_id: operation_id.clone(),
            attempt_id: attempt_id.clone(),
        },
    );

    match order {
        CancellationCommitOrder::CommitBeforeAcknowledgement => {
            push_action(
                &mut actions,
                &mut ordinal,
                4,
                ScheduleAction::EffectCommitted {
                    operation_id: operation_id.clone(),
                    attempt_id: attempt_id.clone(),
                    effect_id: format!("effect-{seed:016x}"),
                },
            );
            push_action(
                &mut actions,
                &mut ordinal,
                5,
                ScheduleAction::CancellationAcknowledged {
                    operation_id: operation_id.clone(),
                    attempt_id: attempt_id.clone(),
                    accepted: true,
                },
            );
        }
        CancellationCommitOrder::CommitAfterAcknowledgement => {
            push_action(
                &mut actions,
                &mut ordinal,
                4,
                ScheduleAction::CancellationAcknowledged {
                    operation_id: operation_id.clone(),
                    attempt_id: attempt_id.clone(),
                    accepted: true,
                },
            );
            push_action(
                &mut actions,
                &mut ordinal,
                5,
                ScheduleAction::EffectCommitted {
                    operation_id: operation_id.clone(),
                    attempt_id: attempt_id.clone(),
                    effect_id: format!("effect-{seed:016x}"),
                },
            );
        }
    }
    push_action(
        &mut actions,
        &mut ordinal,
        6,
        ScheduleAction::AttemptReconciled {
            operation_id,
            attempt_id,
            outcome: ReconciliationOutcome::Applied,
            receipt_ref: format!("synthetic://attempt-reconciliation/{seed:016x}"),
            receipt_sha256: "f".repeat(64),
        },
    );
    push_action(
        &mut actions,
        &mut ordinal,
        7,
        ScheduleAction::StopDischarged {
            obligation_id,
            trigger_action_id: trigger_action_id.clone(),
            policy_revision_action_id: trigger_action_id,
            policy_revision: "policy-v1".to_owned(),
            receipt_ref: format!("synthetic://stop-discharge/{seed:016x}"),
            receipt_sha256: "e".repeat(64),
        },
    );
    VirtualSchedule {
        schema_version: SCHEDULE_SCHEMA_VERSION.to_owned(),
        generator_version: GENERATOR_VERSION.to_owned(),
        seed,
        scenario_family: match order {
            CancellationCommitOrder::CommitBeforeAcknowledgement => {
                "stop_cancel_commit_before_acknowledgement".to_owned()
            }
            CancellationCommitOrder::CommitAfterAcknowledgement => {
                "stop_cancel_commit_after_acknowledgement".to_owned()
            }
        },
        actions,
    }
}

/// Deterministic delta-debugging reducer. It removes chunks of actions only
/// when structural validation still passes and the caller's counterexample
/// predicate still reproduces. The predicate should run the independent
/// evaluator, not merely test for a particular action label.
pub fn shrink_schedule<F>(
    original: &VirtualSchedule,
    mut preserves_counterexample: F,
) -> Result<VirtualSchedule, ReplayError>
where
    F: FnMut(&VirtualSchedule) -> bool,
{
    validate_schedule(original)?;
    if !preserves_counterexample(original) {
        return Err(ReplayError::OriginalDoesNotReproduce);
    }

    let mut current = original.clone();
    let mut granularity = 2_usize;
    loop {
        let length = current.actions.len();
        if length <= 1 {
            break;
        }
        let chunk_size = length.div_ceil(granularity).max(1);
        let mut reduced = false;
        let mut start = 0;
        while start < length {
            let end = (start + chunk_size).min(length);
            let mut candidate = current.clone();
            candidate.actions.drain(start..end);
            if candidate.actions.len() < length
                && validate_schedule(&candidate).is_ok()
                && preserves_counterexample(&candidate)
            {
                current = candidate;
                granularity = granularity.saturating_sub(1).max(2);
                reduced = true;
                break;
            }
            start = end;
        }
        if !reduced {
            if granularity >= length {
                break;
            }
            granularity = (granularity * 2).min(length);
        }
    }
    Ok(current)
}

fn sha256_domain_hex(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualified_resolution(
        schedule: &VirtualSchedule,
        mapping_action_id: &str,
        local_timestamp_ns: u64,
        normalized_timestamp_ns: i64,
    ) -> QualifiedEventTime {
        let action = schedule.actions.iter()
            .find(|action| action.action_id == mapping_action_id)
            .expect("mapping action exists");
        let (clock_domain, receipt_ref, receipt_sha256, valid_from, valid_through) = match &action.action {
            ScheduleAction::ClockAdvance {
                clock_domain, mapping_receipt_ref, mapping_receipt_sha256,
                mapping_valid_from_ns, mapping_valid_through_ns, ..
            } => (
                clock_domain.clone(),
                mapping_receipt_ref.clone(),
                mapping_receipt_sha256.clone(),
                i64::try_from(*mapping_valid_from_ns).expect("fixture validity start fits i64"),
                i64::try_from(*mapping_valid_through_ns).expect("fixture validity end fits i64"),
            ),
            _ => panic!("mapping reference targets a non-clock action"),
        };
        let _ = local_timestamp_ns;
        QualifiedEventTime {
            source_clock_domain_id: clock_domain,
            time_basis: crate::event_order_v0_2::TimeBasisDescriptor {
                basis_id: "owner-normalized-tai-v1".to_owned(),
                epoch: "tai-epoch-0".to_owned(),
                scale: "TAI".to_owned(),
                unit: crate::event_order_v0_2::EventTimeUnit::Nanoseconds,
            },
            mapping_receipt_ref: receipt_ref,
            mapping_receipt_sha256: receipt_sha256,
            mapping_qualification: GateState::Pass,
            mapping_valid_from_ns: valid_from,
            mapping_valid_through_ns: valid_through,
            max_interval_width_ns: 0,
            interval: crate::event_order_v0_2::EventTimeIntervalNs {
                earliest_ns: normalized_timestamp_ns,
                latest_ns: normalized_timestamp_ns,
            },
        }
    }

    fn push_clocked_action(
        actions: &mut Vec<ScheduledAction>,
        ordinal: &mut u64,
        at_ns: u64,
        action: ScheduleAction,
        owner_time_ref: ActionClockTimeReference,
        evidence: ActionEvidence,
    ) -> String {
        let action_id = format!("action-{ordinal:06}");
        push_action(actions, ordinal, at_ns, action);
        let scheduled = actions.last_mut().expect("action was pushed");
        scheduled.owner_time_ref = Some(owner_time_ref);
        scheduled.evidence = Some(evidence);
        action_id
    }

    fn push_test_clock_mapping(
        actions: &mut Vec<ScheduledAction>,
        ordinal: &mut u64,
        at_ns: u64,
        local_tick_ns: u64,
    ) -> String {
        let action_id = format!("action-{ordinal:06}");
        push_action(
            actions,
            ordinal,
            at_ns,
            ScheduleAction::ClockAdvance {
                clock_domain: "action-clock".to_owned(),
                basis_id: "action-basis".to_owned(),
                epoch: "action-epoch-0".to_owned(),
                scale: "tai".to_owned(),
                unit: "nanoseconds".to_owned(),
                local_tick_ns,
                offset_ns: 0,
                uncertainty_ns: 0,
                calibration_id: format!("cal-{local_tick_ns}"),
                mapping_valid_from_ns: 0,
                mapping_valid_through_ns: 10_000,
                mapping_receipt_ref: format!("synthetic://action-clock/{local_tick_ns}"),
                mapping_receipt_sha256: "a".repeat(64),
                qualification: GateState::Pass,
            },
        );
        action_id
    }

    fn passing_owner_receipt() -> OwnerReceiptAssessment {
        OwnerReceiptAssessment {
            identity: GateState::Pass,
            digest_integrity: GateState::Pass,
            authenticity: GateState::Pass,
            scope_qualification: GateState::Pass,
            completeness: GateState::Pass,
            semantic_qualification: GateState::Pass,
            in_flight_reconciliation: GateState::Pass,
        }
    }

    fn action_history_fixture() -> (
        VirtualSchedule,
        BTreeMap<(String, u64), QualifiedEventTime>,
        BTreeMap<String, OwnerReceiptAssessment>,
        String,
    ) {
        let mut actions = Vec::new();
        let mut ordinal = 0_u64;
        let mut resolutions = BTreeMap::new();
        let mut receipts = BTreeMap::new();
        let operation_id = "robot-1".to_owned();
        let attempt_id = "attempt-1".to_owned();
        let obligation_id = "global-stop-1".to_owned();

        let start_mapping = push_test_clock_mapping(&mut actions, &mut ordinal, 1, 100);
        let start_action_id = push_clocked_action(
            &mut actions,
            &mut ordinal,
            2,
            ScheduleAction::EffectAttempted {
                operation_id: operation_id.clone(),
                attempt_id: attempt_id.clone(),
            },
            ActionClockTimeReference {
                clock_mapping_action_id: start_mapping.clone(),
                local_timestamp_ns: 100,
            },
            ActionEvidence::AttemptStart { retry_of_attempt_id: None },
        );
        resolutions.insert(
            (start_mapping.clone(), 100),
            qualified_resolution_from_fixture(
                &actions, &start_mapping, 100, 1_000,
            ),
        );

        let commit_mapping = push_test_clock_mapping(&mut actions, &mut ordinal, 3, 110);
        let commit_action_id = push_clocked_action(
            &mut actions,
            &mut ordinal,
            4,
            ScheduleAction::EffectCommitted {
                operation_id: operation_id.clone(),
                attempt_id: attempt_id.clone(),
                effect_id: "effect-1".to_owned(),
            },
            ActionClockTimeReference {
                clock_mapping_action_id: commit_mapping.clone(),
                local_timestamp_ns: 110,
            },
            ActionEvidence::EffectCommit {
                receipt_ref: "synthetic://effect-commit/1".to_owned(),
                receipt_sha256: "b".repeat(64),
            },
        );
        resolutions.insert(
            (commit_mapping.clone(), 110),
            qualified_resolution_from_fixture(&actions, &commit_mapping, 110, 1_050),
        );
        receipts.insert(commit_action_id, passing_owner_receipt());

        let terminal_mapping = push_test_clock_mapping(&mut actions, &mut ordinal, 5, 120);
        let terminal_action_id = push_clocked_action(
            &mut actions,
            &mut ordinal,
            6,
            ScheduleAction::AttemptReconciled {
                operation_id: operation_id.clone(),
                attempt_id: attempt_id.clone(),
                outcome: ReconciliationOutcome::Applied,
                receipt_ref: "synthetic://terminal/1".to_owned(),
                receipt_sha256: "c".repeat(64),
            },
            ActionClockTimeReference {
                clock_mapping_action_id: terminal_mapping.clone(),
                local_timestamp_ns: 120,
            },
            ActionEvidence::TerminalReconciliation {
                exact_start_action_id: start_action_id.clone(),
                qualification: GateState::Pass,
            },
        );
        resolutions.insert(
            (terminal_mapping.clone(), 120),
            qualified_resolution_from_fixture(&actions, &terminal_mapping, 120, 1_100),
        );
        receipts.insert(terminal_action_id.clone(), passing_owner_receipt());

        let trigger_mapping = push_test_clock_mapping(&mut actions, &mut ordinal, 7, 130);
        let trigger_action_id = push_clocked_action(
            &mut actions,
            &mut ordinal,
            8,
            ScheduleAction::StopTriggered {
                obligation_id: obligation_id.clone(),
                policy_revision: "policy-v1".to_owned(),
            },
            ActionClockTimeReference {
                clock_mapping_action_id: trigger_mapping.clone(),
                local_timestamp_ns: 130,
            },
            ActionEvidence::StopTrigger {
                scope: StopScope::Global,
                receipt_ref: "synthetic://stop-trigger/1".to_owned(),
                receipt_sha256: "d".repeat(64),
            },
        );
        resolutions.insert(
            (trigger_mapping.clone(), 130),
            qualified_resolution_from_fixture(&actions, &trigger_mapping, 130, 1_150),
        );
        receipts.insert(trigger_action_id.clone(), passing_owner_receipt());

        let inventory_mapping = push_test_clock_mapping(&mut actions, &mut ordinal, 9, 135);
        let inventory_frontier = "inventory-frontier-1".to_owned();
        let discharge_action_id = format!("action-{:06}", ordinal + 2);
        let mut expected_attempt = OperationAttemptRecord {
            identity: AttemptIdentity {
                logical_operation_id: operation_id.clone(),
                attempt_id: attempt_id.clone(),
                retry_of_attempt_id: None,
            },
            start_event_ids: vec![start_action_id.clone()],
            terminal_reconciliation_event_ids: vec![terminal_action_id.clone()],
            terminal_reconciliation: Some(AttemptTerminalReconciliation {
                event_id: terminal_action_id.clone(),
                start_event_id: start_action_id.clone(),
                outcome: ReconciledOperationOutcome::Applied,
                qualification: GateState::Pass,
                start_to_terminal_order: EventOrder::Before,
            }),
            stop_discharge_relations: vec![AttemptStopDischargeRelation {
                obligation_id: obligation_id.clone(),
                discharge_event_id: discharge_action_id.clone(),
                attempt_start_event_id: start_action_id.clone(),
                discharge_to_start_order: EventOrder::After,
                terminal_event_id_at_discharge: Some(terminal_action_id.clone()),
                terminal_to_discharge_order: Some(EventOrder::Before),
            }],
            retry_predecessor_terminal_event_id: None,
            retry_predecessor_start_event_id: None,
            retry_predecessor_terminal_to_start_order: None,
        };
        let manifest = crate::event_order_v0_2::canonical_attempt_manifest_sha256(
            std::slice::from_ref(&expected_attempt),
        ).expect("canonical attempt manifest");
        expected_attempt.stop_discharge_relations.sort_by(|a, b| a.obligation_id.cmp(&b.obligation_id));

        let inventory_action_id = push_clocked_action(
            &mut actions,
            &mut ordinal,
            10,
            ScheduleAction::AttemptInventoryAttested {
                scope: StopScope::Global,
                inventory_frontier_event_id: inventory_frontier.clone(),
                attempt_manifest_sha256: manifest.clone(),
            },
            ActionClockTimeReference {
                clock_mapping_action_id: inventory_mapping.clone(),
                local_timestamp_ns: 135,
            },
            ActionEvidence::InventoryAttestation {
                scope: StopScope::Global,
                inventory_frontier_event_id: inventory_frontier.clone(),
                attempt_manifest_sha256: manifest.clone(),
                receipt_ref: "synthetic://inventory-attestation/1".to_owned(),
                receipt_sha256: "e".repeat(64),
            },
        );
        resolutions.insert(
            (inventory_mapping.clone(), 135),
            qualified_resolution_from_fixture(&actions, &inventory_mapping, 135, 1_175),
        );
        receipts.insert(inventory_action_id, passing_owner_receipt());

        let discharge_mapping = push_test_clock_mapping(&mut actions, &mut ordinal, 11, 140);
        let discharge_action_id_actual = push_clocked_action(
            &mut actions,
            &mut ordinal,
            12,
            ScheduleAction::StopDischarged {
                obligation_id,
                trigger_action_id: trigger_action_id.clone(),
                policy_revision_action_id: trigger_action_id,
                policy_revision: "policy-v1".to_owned(),
                receipt_ref: "synthetic://stop-discharge/1".to_owned(),
                receipt_sha256: "f".repeat(64),
            },
            ActionClockTimeReference {
                clock_mapping_action_id: discharge_mapping.clone(),
                local_timestamp_ns: 140,
            },
            ActionEvidence::StopDischarge {
                scope: StopScope::Global,
                inventory_frontier_event_id: inventory_frontier,
                attempt_manifest_sha256: manifest.clone(),
            },
        );
        assert_eq!(discharge_action_id, discharge_action_id_actual);
        resolutions.insert(
            (discharge_mapping.clone(), 140),
            qualified_resolution_from_fixture(&actions, &discharge_mapping, 140, 1_200),
        );
        receipts.insert(discharge_action_id_actual, passing_owner_receipt());

        let schedule = VirtualSchedule {
            schema_version: SCHEDULE_SCHEMA_VERSION.to_owned(),
            generator_version: GENERATOR_VERSION.to_owned(),
            seed: 0x2026_1123,
            scenario_family: "qualified_attempt_terminal_stop_inventory".to_owned(),
            actions,
        };
        (schedule, resolutions, receipts, manifest)
    }

    fn qualified_resolution_from_fixture(
        actions: &[ScheduledAction],
        mapping_action_id: &str,
        local_timestamp_ns: u64,
        normalized_ns: i64,
    ) -> QualifiedEventTime {
        let temporary = VirtualSchedule {
            schema_version: SCHEDULE_SCHEMA_VERSION.to_owned(),
            generator_version: GENERATOR_VERSION.to_owned(),
            seed: 1,
            scenario_family: "test_resolution".to_owned(),
            actions: actions.to_vec(),
        };
        qualified_resolution(&temporary, mapping_action_id, local_timestamp_ns, normalized_ns)
    }

    fn owner_time_resolutions(schedule: &VirtualSchedule) -> BTreeMap<(String, u64), QualifiedEventTime> {
        let mut resolutions = BTreeMap::new();
        for action in &schedule.actions {
            match &action.action {
                ScheduleAction::EventProduced {
                    clock_mapping_action_id, source_timestamp_ns, ..
                } => {
                    resolutions.insert(
                        (clock_mapping_action_id.clone(), *source_timestamp_ns),
                        qualified_resolution(schedule, clock_mapping_action_id, *source_timestamp_ns, 500),
                    );
                }
                ScheduleAction::EventObserved {
                    clock_mapping_action_id, observed_at_ns, ..
                } => {
                    resolutions.insert(
                        (clock_mapping_action_id.clone(), *observed_at_ns),
                        qualified_resolution(schedule, clock_mapping_action_id, *observed_at_ns, 1_000),
                    );
                }
                ScheduleAction::EventPersisted {
                    clock_mapping_action_id, persisted_at_ns, ..
                } => {
                    resolutions.insert(
                        (clock_mapping_action_id.clone(), *persisted_at_ns),
                        qualified_resolution(schedule, clock_mapping_action_id, *persisted_at_ns, 1_200),
                    );
                }
                _ => {}
            }
        }
        resolutions
    }

    #[test]
    fn seeded_generator_is_byte_for_byte_deterministic() {
        let left = generate_delivery_schedule(0x4c55_4d49, 24).expect("valid schedule");
        let right = generate_delivery_schedule(0x4c55_4d49, 24).expect("valid schedule");
        assert_eq!(canonical_schedule_bytes(&left), canonical_schedule_bytes(&right));
        assert_eq!(replay_virtual_schedule(&left), replay_virtual_schedule(&right));
    }

    #[test]
    fn action_vector_permutation_preserves_canonical_schedule_and_replay_digests() {
        let schedule = generate_delivery_schedule(0x2026_1115, 32).expect("valid schedule");
        let mut permuted = schedule.clone();
        permuted.actions.reverse();
        assert_eq!(canonical_schedule_sha256(&schedule), canonical_schedule_sha256(&permuted));
        let left = replay_virtual_schedule(&schedule).expect("replay");
        let right = replay_virtual_schedule(&permuted).expect("replay");
        assert_eq!(left.replay_sha256, right.replay_sha256);
        assert_eq!(left.records, right.records);
    }

    #[test]
    fn partition_blocks_delivery_until_reconnect() {
        let schedule = generate_delivery_schedule(7, 20).expect("valid schedule");
        let report = replay_virtual_schedule(&schedule).expect("replay");
        assert!(report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::BlockedByPartition
        }));
        assert!(report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::NetworkPartition { partitioned: false, .. })
        }));
        assert!(report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::Applied
                && matches!(&record.action, ScheduleAction::Delivery { recipient_id, .. } if recipient_id == "worker-after-reconnect")
        }));
    }

    #[test]
    fn shrinker_preserves_validity_and_counterexample_predicate() {
        let schedule = generate_delivery_schedule(19, 16).expect("valid schedule");
        let predicate = |candidate: &VirtualSchedule| {
            candidate.actions.iter().any(|item| {
                matches!(&item.action, ScheduleAction::EventProduced { event_id, .. } if event_id == "event-000000")
            }) && candidate.actions.iter().any(|item| {
                matches!(&item.action, ScheduleAction::Delivery { event_id, .. } if event_id == "event-000000")
            })
        };
        let minimized = shrink_schedule(&schedule, predicate).expect("shrink");
        validate_schedule(&minimized).expect("minimized schedule remains valid");
        assert!(predicate(&minimized));
        assert!(minimized.actions.len() < schedule.actions.len());
    }

    #[test]
    fn missing_clock_and_causal_predecessors_fail_closed() {
        let mut schedule = generate_delivery_schedule(23, 4).expect("valid schedule");
        schedule.actions.retain(|item| {
            !matches!(item.action, ScheduleAction::ClockAdvance { ref clock_domain, .. } if clock_domain == "clock-a")
        });
        assert!(matches!(validate_schedule(&schedule), Err(ReplayError::MissingClock(_))));

        let mut schedule = generate_delivery_schedule(23, 4).expect("valid schedule");
        if let Some(item) = schedule.actions.iter_mut().find(|item| {
            matches!(item.action, ScheduleAction::EventProduced { ref event_id, .. } if event_id == "event-000000")
        }) {
            if let ScheduleAction::EventProduced { causal_predecessors, .. } = &mut item.action {
                causal_predecessors.push("not-yet-produced".to_owned());
            }
        }
        assert!(matches!(validate_schedule(&schedule), Err(ReplayError::InvalidCausalPredecessor(_))));
    }

    #[test]
    fn attempt_and_stop_faults_remain_representable_in_the_schedule() {
        let mut schedule = generate_delivery_schedule(31, 2).expect("valid schedule");
        let at_ns = 10_000;
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        push_action(
            &mut schedule.actions,
            &mut ordinal,
            at_ns,
            ScheduleAction::StopTriggered {
                obligation_id: "stop-1".to_owned(),
                policy_revision: "policy-1".to_owned(),
            },
        );
        push_action(
            &mut schedule.actions,
            &mut ordinal,
            at_ns + 1,
            ScheduleAction::EffectAttempted {
                operation_id: "op-1".to_owned(),
                attempt_id: "attempt-1".to_owned(),
            },
        );
        push_action(
            &mut schedule.actions,
            &mut ordinal,
            at_ns + 2,
            ScheduleAction::EffectCommitted {
                operation_id: "op-1".to_owned(),
                attempt_id: "attempt-1".to_owned(),
                effect_id: "effect-1".to_owned(),
            },
        );
        let report = replay_virtual_schedule(&schedule).expect("fault schedule replays");
        assert!(report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EffectCommitted { .. })
                && record.disposition == SchedulerDisposition::Applied
        }));
    }

    #[test]
    fn policy_revision_requires_a_new_exact_discharge() {
        let mut schedule = generate_delivery_schedule(41, 2).expect("valid schedule");
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let t = 20_000;
        let obligation = "stop-revision-test";
        let trigger_action_id = format!("action-{ordinal:06}");
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::StopTriggered {
            obligation_id: obligation.to_owned(), policy_revision: "policy-v1".to_owned(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 1, ScheduleAction::StopDischarged {
            obligation_id: obligation.to_owned(), trigger_action_id: trigger_action_id.clone(),
            policy_revision_action_id: trigger_action_id.clone(), policy_revision: "policy-v1".to_owned(),
            receipt_ref: "receipt-v1".to_owned(), receipt_sha256: "a".repeat(64),
        });
        let revised_policy_action_id = format!("action-{ordinal:06}");
        push_action(&mut schedule.actions, &mut ordinal, t + 2, ScheduleAction::StopPolicyRevised {
            obligation_id: obligation.to_owned(), policy_revision: "policy-v2".to_owned(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 3, ScheduleAction::StopDischarged {
            obligation_id: obligation.to_owned(), trigger_action_id: trigger_action_id.clone(),
            policy_revision_action_id: trigger_action_id.clone(), policy_revision: "policy-v1".to_owned(),
            receipt_ref: "stale-receipt".to_owned(), receipt_sha256: "b".repeat(64),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 4, ScheduleAction::StopDischarged {
            obligation_id: obligation.to_owned(), trigger_action_id: "substituted-trigger".to_owned(),
            policy_revision_action_id: revised_policy_action_id.clone(), policy_revision: "policy-v2".to_owned(),
            receipt_ref: "substituted-trigger".to_owned(), receipt_sha256: "c".repeat(64),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 5, ScheduleAction::StopDischarged {
            obligation_id: obligation.to_owned(), trigger_action_id: trigger_action_id.clone(),
            policy_revision_action_id: revised_policy_action_id, policy_revision: "policy-v2".to_owned(),
            receipt_ref: "receipt-v2".to_owned(), receipt_sha256: "d".repeat(64),
        });
        let report = replay_virtual_schedule(&schedule).expect("replay");
        let discharge_dispositions: Vec<_> = report.records.iter().filter_map(|record| {
            if matches!(record.action, ScheduleAction::StopDischarged { .. }) {
                Some(record.disposition)
            } else {
                None
            }
        }).collect();
        assert_eq!(discharge_dispositions, vec![
            SchedulerDisposition::Applied,
            SchedulerDisposition::PreconditionViolation,
            SchedulerDisposition::PreconditionViolation,
            SchedulerDisposition::Applied,
        ]);
    }

    #[test]
    fn crash_blocks_persistence_until_restart() {
        let mut schedule = generate_delivery_schedule(43, 2).expect("valid schedule");
        let event_id = "event-000000".to_owned();
        let observer = "worker-after-reconnect".to_owned();
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let t = 20_000;
        let store_clock_mapping_id = format!("action-{ordinal:06}");
        push_action(&mut schedule.actions, &mut ordinal, t - 4, ScheduleAction::ClockAdvance {
            clock_domain: "store-clock".to_owned(),
            basis_id: "basis-store".to_owned(),
            epoch: "store-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 1_000,
            offset_ns: 0,
            uncertainty_ns: 10,
            calibration_id: "store-calibration".to_owned(),
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 2_000,
            mapping_receipt_ref: "synthetic://store-clock-map".to_owned(),
            mapping_receipt_sha256: "a".repeat(64),
            qualification: GateState::Pass,
        });
        let observer_clock_mapping_id = format!("action-{ordinal:06}");
        push_action(&mut schedule.actions, &mut ordinal, t - 3, ScheduleAction::ClockAdvance {
            clock_domain: "observer-clock".to_owned(),
            basis_id: "basis-observer".to_owned(),
            epoch: "observer-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 1_000,
            offset_ns: 0,
            uncertainty_ns: 20,
            calibration_id: "observer-calibration".to_owned(),
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 2_000,
            mapping_receipt_ref: "synthetic://observer-clock-map".to_owned(),
            mapping_receipt_sha256: "b".repeat(64),
            qualification: GateState::Pass,
        });
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::EventObserved {
            event_id: event_id.clone(),
            observer_id: observer.clone(),
            clock_domain: "observer-clock".to_owned(),
            basis_id: "basis-observer".to_owned(),
            epoch: "observer-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: observer_clock_mapping_id,
            observed_at_ns: 100,
            observation_receipt_ref: "synthetic://observation/first".to_owned(),
            observation_receipt_sha256: "c".repeat(64),
            qualification: GateState::Pass,
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 1, ScheduleAction::ProcessCrashed {
            process_id: observer.clone(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 2, ScheduleAction::EventPersisted {
            event_id: event_id.clone(),
            observer_id: observer.clone(),
            store_id: "store-1".to_owned(),
            clock_domain: "store-clock".to_owned(),
            basis_id: "basis-store".to_owned(),
            epoch: "store-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: store_clock_mapping_id.clone(),
            persisted_at_ns: 150,
            persistence_receipt_ref: "synthetic://persistence/durable".to_owned(),
            persistence_receipt_sha256: "d".repeat(64),
            qualification: GateState::Pass,
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 3, ScheduleAction::ProcessRestarted {
            process_id: observer.clone(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 4, ScheduleAction::EventPersisted {
            event_id,
            observer_id: observer,
            store_id: "store-1".to_owned(),
            clock_domain: "store-clock".to_owned(),
            basis_id: "basis-store".to_owned(),
            epoch: "store-epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: store_clock_mapping_id,
            persisted_at_ns: 150,
            persistence_receipt_ref: "synthetic://persistence/durable".to_owned(),
            persistence_receipt_sha256: "d".repeat(64),
            qualification: GateState::Pass,
        });
        let report = replay_virtual_schedule(&schedule).expect("replay");
        let persistence_dispositions: Vec<_> = report.records.iter().filter_map(|record| {
            if matches!(record.action, ScheduleAction::EventPersisted { .. }) {
                Some(record.disposition)
            } else {
                None
            }
        }).collect();
        assert_eq!(persistence_dispositions, vec![
            SchedulerDisposition::BlockedByCrashedRecipient,
            SchedulerDisposition::Applied,
        ]);
    }


    #[test]
    fn clock_and_event_time_regressions_remain_replayable_faults() {
        let mut schedule = generate_delivery_schedule(61, 2).expect("valid schedule");
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let t = 20_000;
        let regression_mapping_id = format!("action-{ordinal:06}");
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::ClockAdvance {
            clock_domain: "clock-a".to_owned(),
            basis_id: "basis-clock-a".to_owned(),
            epoch: "epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 50,
            offset_ns: -250_000,
            uncertainty_ns: 50_000,
            calibration_id: "calibration-backwards".to_owned(),
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 60,
            mapping_receipt_ref: "synthetic://mapping/backwards".to_owned(),
            mapping_receipt_sha256: "d".repeat(64),
            qualification: GateState::Pass,
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 1, ScheduleAction::EventProduced {
            event_id: "event-backwards".to_owned(),
            clock_domain: "clock-a".to_owned(),
            basis_id: "basis-clock-a".to_owned(),
            epoch: "epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: regression_mapping_id,
            source_timestamp_ns: 50,
            causal_predecessors: Vec::new(),
        });
        let new_epoch_mapping_id = format!("action-{ordinal:06}");
        push_action(&mut schedule.actions, &mut ordinal, t + 2, ScheduleAction::ClockAdvance {
            clock_domain: "clock-a".to_owned(),
            basis_id: "basis-clock-a".to_owned(),
            epoch: "epoch-1".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 5,
            offset_ns: 0,
            uncertainty_ns: 10,
            calibration_id: "calibration-new-epoch".to_owned(),
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 100,
            mapping_receipt_ref: "synthetic://mapping/new-epoch".to_owned(),
            mapping_receipt_sha256: "e".repeat(64),
            qualification: GateState::Pass,
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 3, ScheduleAction::EventProduced {
            event_id: "event-new-epoch".to_owned(),
            clock_domain: "clock-a".to_owned(),
            basis_id: "basis-clock-a".to_owned(),
            epoch: "epoch-1".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            clock_mapping_action_id: new_epoch_mapping_id,
            source_timestamp_ns: 5,
            causal_predecessors: Vec::new(),
        });
        validate_schedule(&schedule).expect("fault schedule must remain structurally valid");
        let report = replay_virtual_schedule(&schedule).expect("fault schedule replays");
        assert!(report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::ClockRegression
        }));
        assert!(report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::EventTimeRegression
        }));
        assert!(report.records.iter().any(|record| {
            matches!(&record.action, ScheduleAction::EventProduced { event_id, .. } if event_id == "event-new-epoch")
                && record.disposition == SchedulerDisposition::Applied
        }));
    }

    #[test]
    fn exact_effect_commit_repeat_is_idempotent_but_substitution_conflicts() {
        let mut schedule = generate_delivery_schedule(73, 1).expect("valid schedule");
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let operation_id = "effect-op".to_owned();
        let attempt_id = "effect-attempt".to_owned();
        let t = 20_000;
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::EffectAttempted {
            operation_id: operation_id.clone(), attempt_id: attempt_id.clone(),
        });
        for (offset, effect_id) in [(1, "effect-a"), (2, "effect-a"), (3, "effect-b")] {
            push_action(&mut schedule.actions, &mut ordinal, t + offset, ScheduleAction::EffectCommitted {
                operation_id: operation_id.clone(), attempt_id: attempt_id.clone(),
                effect_id: effect_id.to_owned(),
            });
        }
        let report = replay_virtual_schedule(&schedule).expect("replay");
        let dispositions: Vec<_> = report.records.iter().filter_map(|record| {
            if matches!(record.action, ScheduleAction::EffectCommitted { .. }) {
                Some(record.disposition)
            } else {
                None
            }
        }).collect();
        assert_eq!(dispositions, vec![
            SchedulerDisposition::Applied,
            SchedulerDisposition::Duplicate,
            SchedulerDisposition::ConflictingEffectCommit,
        ]);
    }

    #[test]
    fn attempt_reconciliation_is_idempotent_but_conflicts_are_visible() {
        let mut schedule = generate_delivery_schedule(67, 1).expect("valid schedule");
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let operation_id = "reconcile-op".to_owned();
        let attempt_id = "reconcile-attempt".to_owned();
        let t = 20_000;
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::EffectAttempted {
            operation_id: operation_id.clone(), attempt_id: attempt_id.clone(),
        });
        for (offset, outcome, receipt_ref, digest) in [
            (1, ReconciliationOutcome::NotApplied, "terminal-a", "a"),
            (2, ReconciliationOutcome::NotApplied, "terminal-a", "a"),
            (3, ReconciliationOutcome::Applied, "terminal-b", "b"),
        ] {
            push_action(&mut schedule.actions, &mut ordinal, t + offset, ScheduleAction::AttemptReconciled {
                operation_id: operation_id.clone(), attempt_id: attempt_id.clone(), outcome,
                receipt_ref: receipt_ref.to_owned(), receipt_sha256: digest.repeat(64),
            });
        }
        let report = replay_virtual_schedule(&schedule).expect("replay");
        let dispositions: Vec<_> = report.records.iter().filter_map(|record| {
            if matches!(record.action, ScheduleAction::AttemptReconciled { .. }) {
                Some(record.disposition)
            } else {
                None
            }
        }).collect();
        assert_eq!(dispositions, vec![
            SchedulerDisposition::Applied,
            SchedulerDisposition::Duplicate,
            SchedulerDisposition::ConflictingReconciliation,
        ]);
    }

    #[test]
    fn explicit_owner_failure_is_not_collapsed_into_indeterminate() {
        let mut mapping_failed = generate_clocked_observation_persistence_schedule(113)
            .expect("clocked schedule");
        for action in &mut mapping_failed.actions {
            if let ScheduleAction::ClockAdvance { clock_domain, qualification, .. } = &mut action.action {
                if clock_domain.starts_with("observer-clock-") {
                    *qualification = GateState::Fail;
                }
            }
        }
        let mapping_report = replay_virtual_schedule(&mapping_failed).expect("replay");
        assert!(mapping_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
                && record.disposition == SchedulerDisposition::ClockMappingRejected
        }));

        let mut observation_failed = generate_clocked_observation_persistence_schedule(127)
            .expect("clocked schedule");
        for action in &mut observation_failed.actions {
            if let ScheduleAction::EventObserved { qualification, .. } = &mut action.action {
                *qualification = GateState::Fail;
            }
        }
        let observation_report = replay_virtual_schedule(&observation_failed).expect("replay");
        assert!(observation_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
                && record.disposition == SchedulerDisposition::ObservationRejected
        }));

        let mut persistence_failed = generate_clocked_observation_persistence_schedule(131)
            .expect("clocked schedule");
        for action in &mut persistence_failed.actions {
            if let ScheduleAction::EventPersisted { qualification, .. } = &mut action.action {
                *qualification = GateState::Fail;
            }
        }
        let persistence_report = replay_virtual_schedule(&persistence_failed).expect("replay");
        assert!(persistence_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventPersisted { .. })
                && record.disposition == SchedulerDisposition::PersistenceRejected
        }));
    }

    #[test]
    fn observer_and_store_mapping_qualification_states_fail_closed() {
        let mut observation_unqualified = generate_clocked_observation_persistence_schedule(97)
            .expect("clocked schedule");
        if let Some(action) = observation_unqualified.actions.iter_mut().find(|action| {
            matches!(action.action, ScheduleAction::EventObserved { .. })
        }) {
            if let ScheduleAction::EventObserved { qualification, .. } = &mut action.action {
                *qualification = GateState::Indeterminate;
            }
        }
        let observation_report = replay_virtual_schedule(&observation_unqualified).expect("replay");
        assert!(observation_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
                && record.disposition == SchedulerDisposition::ObservationUnqualified
        }));

        let mut persistence_unqualified = generate_clocked_observation_persistence_schedule(101)
            .expect("clocked schedule");
        if let Some(action) = persistence_unqualified.actions.iter_mut().find(|action| {
            matches!(action.action, ScheduleAction::EventPersisted { .. })
        }) {
            if let ScheduleAction::EventPersisted { qualification, .. } = &mut action.action {
                *qualification = GateState::Indeterminate;
            }
        }
        let persistence_report = replay_virtual_schedule(&persistence_unqualified).expect("replay");
        assert!(persistence_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventPersisted { .. })
                && record.disposition == SchedulerDisposition::PersistenceUnqualified
        }));
    }

    #[test]
    fn observer_clock_validity_windows_are_enforced_independently() {
        let mut not_yet_valid = generate_clocked_observation_persistence_schedule(103)
            .expect("clocked schedule");
        for action in &mut not_yet_valid.actions {
            if let ScheduleAction::ClockAdvance {
                clock_domain, mapping_valid_from_ns, ..
            } = &mut action.action {
                if clock_domain.starts_with("observer-clock-") {
                    *mapping_valid_from_ns = 2_000;
                }
            }
        }
        let not_yet_valid_report = replay_virtual_schedule(&not_yet_valid).expect("replay");
        assert!(not_yet_valid_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
                && record.disposition == SchedulerDisposition::ClockMappingNotYetValid
        }));

        let mut expired = generate_clocked_observation_persistence_schedule(107)
            .expect("clocked schedule");
        for action in &mut expired.actions {
            if let ScheduleAction::ClockAdvance {
                clock_domain, mapping_valid_through_ns, ..
            } = &mut action.action {
                if clock_domain.starts_with("observer-clock-") {
                    *mapping_valid_through_ns = 800;
                }
            }
        }
        let expired_report = replay_virtual_schedule(&expired).expect("replay");
        assert!(expired_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
                && record.disposition == SchedulerDisposition::ClockMappingExpired
        }));

        let mut invalid = generate_clocked_observation_persistence_schedule(109)
            .expect("clocked schedule");
        for action in &mut invalid.actions {
            if let ScheduleAction::ClockAdvance {
                clock_domain, mapping_valid_from_ns, mapping_valid_through_ns, ..
            } = &mut action.action {
                if clock_domain.starts_with("observer-clock-") {
                    *mapping_valid_from_ns = 2_000;
                    *mapping_valid_through_ns = 800;
                }
            }
        }
        let invalid_report = replay_virtual_schedule(&invalid).expect("replay");
        assert!(invalid_report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
                && record.disposition == SchedulerDisposition::ClockMappingValidityInvalid
        }));
    }

    #[test]
    fn observer_and_persistence_receipt_conflicts_are_explicit() {
        let mut observation_schedule = generate_clocked_observation_persistence_schedule(79)
            .expect("clocked schedule");
        let original_observation = observation_schedule.actions.iter().find_map(|action| {
            match &action.action {
                ScheduleAction::EventObserved { .. } => Some(action.action.clone()),
                _ => None,
            }
        }).expect("original observation");
        let mut altered_observation = original_observation;
        if let ScheduleAction::EventObserved { observation_receipt_ref, .. } = &mut altered_observation {
            *observation_receipt_ref = "synthetic://observation/substituted".to_owned();
        }
        let mut ordinal = observation_schedule.actions.iter().map(|action| action.ordinal).max().unwrap_or(0) + 1;
        let later = observation_schedule.actions.iter().map(|action| action.at_ns).max().unwrap_or(0) + 1;
        push_action(&mut observation_schedule.actions, &mut ordinal, later, altered_observation);
        let observation_report = replay_virtual_schedule(&observation_schedule).expect("replay");
        assert!(observation_report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::ConflictingObservation
        }));

        let mut persistence_schedule = generate_clocked_observation_persistence_schedule(83)
            .expect("clocked schedule");
        let original_persistence = persistence_schedule.actions.iter().find_map(|action| {
            match &action.action {
                ScheduleAction::EventPersisted { .. } => Some(action.action.clone()),
                _ => None,
            }
        }).expect("original persistence");
        let mut altered_persistence = original_persistence;
        if let ScheduleAction::EventPersisted { persistence_receipt_ref, .. } = &mut altered_persistence {
            *persistence_receipt_ref = "synthetic://persistence/substituted".to_owned();
        }
        let mut ordinal = persistence_schedule.actions.iter().map(|action| action.ordinal).max().unwrap_or(0) + 1;
        let later = persistence_schedule.actions.iter().map(|action| action.at_ns).max().unwrap_or(0) + 1;
        push_action(&mut persistence_schedule.actions, &mut ordinal, later, altered_persistence);
        let persistence_report = replay_virtual_schedule(&persistence_schedule).expect("replay");
        assert!(persistence_report.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::ConflictingPersistence
        }));
    }

    #[test]
    fn observation_time_beyond_its_clock_sample_fails_closed() {
        let mut schedule = generate_clocked_observation_persistence_schedule(89)
            .expect("clocked schedule");
        if let Some(action) = schedule.actions.iter_mut().find(|action| {
            matches!(action.action, ScheduleAction::EventObserved { .. })
        }) {
            if let ScheduleAction::EventObserved { observed_at_ns, .. } = &mut action.action {
                *observed_at_ns = 100_001;
            }
        }
        validate_schedule(&schedule).expect("future timestamp is a replayable fault");
        let report = replay_virtual_schedule(&schedule).expect("replay");
        assert!(report.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
                && record.disposition == SchedulerDisposition::ClockTimestampBeyondQualifiedTick
        }));
    }

    #[test]
    fn action_history_projection_composes_attempt_stop_and_complete_inventory() {
        let (schedule, resolutions, receipts, expected_manifest) = action_history_fixture();
        validate_schedule(&schedule).expect("valid action-history fixture");
        let report = project_action_history(
            &schedule,
            "robot-1",
            &resolutions,
            &receipts,
        ).expect("action-history projection");
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
        assert_eq!(report.attempts.len(), 1);
        assert_eq!(report.attempt_lifecycle.disposition, EventTraceDisposition::Safe);
        assert_eq!(
            report.inventory.as_ref().map(|inventory| inventory.attempt_manifest_sha256.as_str()),
            Some(expected_manifest.as_str()),
        );
        assert_eq!(report.obligations.len(), 1);
        assert_eq!(report.obligations[0].discharge.as_ref().map(|receipt| receipt.qualification), Some(GateState::Pass));
        assert!(report.start_gate.as_ref().is_some_and(|gate| gate.start_permitted_by_stop_contract));
        assert_eq!(report.effect_commits.len(), 1);
    }

    #[test]
    fn action_history_projection_blocks_missing_owner_time_and_receipt_authenticity() {
        let (mut schedule, resolutions, mut receipts, _) = action_history_fixture();
        let start = schedule.actions.iter_mut().find(|action| {
            matches!(action.action, ScheduleAction::EffectAttempted { .. })
        }).expect("attempt start");
        start.owner_time_ref = None;
        let report = project_action_history(&schedule, "robot-1", &resolutions, &receipts)
            .expect("missing owner time is an assessment");
        assert_ne!(report.disposition, EventTraceDisposition::Safe);
        assert!(report.reason_codes.iter().any(|code| code.contains("ACTION_OWNER_TIME_REFERENCE_MISSING")));

        let (schedule, resolutions, mut receipts, _) = action_history_fixture();
        let terminal_action_id = schedule.actions.iter().find_map(|action| {
            if matches!(action.action, ScheduleAction::AttemptReconciled { .. }) {
                Some(action.action_id.clone())
            } else {
                None
            }
        }).expect("terminal action");
        let mut receipt = receipts.get(&terminal_action_id).copied().expect("owner assessment");
        receipt.authenticity = GateState::Fail;
        receipts.insert(terminal_action_id, receipt);
        let report = project_action_history(&schedule, "robot-1", &resolutions, &receipts)
            .expect("failed authenticity is an assessment");
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|code| code.contains("OWNER_RECEIPT_QUALIFICATION_FAILED")));
    }

    #[test]
    fn bitemporal_projection_uses_owner_times_not_simulator_action_time() {
        let schedule = generate_clocked_observation_persistence_schedule(0x2026_1115)
            .expect("clocked observation/persistence schedule");
        let resolutions = owner_time_resolutions(&schedule);
        let report = project_bitemporal_event(
            &schedule,
            "event-000000",
            "worker-after-reconnect",
            "durable-store-0",
            &resolutions,
        ).expect("projection result");
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
        let envelope = report.envelope.expect("qualified bitemporal envelope");
        let occurred = envelope.occurred_at.expect("owner-qualified occurrence time");
        assert_eq!(occurred.interval.earliest_ns, 500);
        assert_eq!(envelope.first_observed_at.interval.earliest_ns, 1_000);
        assert_eq!(envelope.durably_recorded_at.interval.earliest_ns, 1_200);
        let simulator_observed_at = schedule.actions.iter().find_map(|action| {
            if let ScheduleAction::EventObserved { event_id, .. } = &action.action {
                (event_id == "event-000000").then_some(action.at_ns)
            } else {
                None
            }
        }).expect("simulator observation action exists");
        assert_ne!(
            envelope.first_observed_at.interval.earliest_ns as u64,
            simulator_observed_at,
        );
        assert_eq!(envelope.capture_qualification, GateState::Pass);
        assert!(report.schedule_sha256.as_ref().is_some_and(|digest| digest.len() == 64));
        assert_eq!(
            report.occurrence_binding.as_ref().map(|binding| binding.action_id.as_str()),
            schedule.actions.iter().find(|action| {
                matches!(&action.action, ScheduleAction::EventProduced { event_id, .. } if event_id == "event-000000")
            }).map(|action| action.action_id.as_str()),
        );
        assert!(report.observation_binding.is_some());
        assert!(report.persistence_binding.is_some());
    }

    #[test]
    fn bitemporal_projection_fails_closed_on_missing_or_substituted_owner_time() {
        let schedule = generate_clocked_observation_persistence_schedule(0x2026_1116)
            .expect("clocked observation/persistence schedule");
        let mut missing = owner_time_resolutions(&schedule);
        let observation_key = schedule.actions.iter().find_map(|action| {
            if let ScheduleAction::EventObserved {
                clock_mapping_action_id, observed_at_ns, ..
            } = &action.action {
                Some((clock_mapping_action_id.clone(), *observed_at_ns))
            } else {
                None
            }
        }).expect("observation mapping key");
        missing.remove(&observation_key);
        let missing_report = project_bitemporal_event(
            &schedule, "event-000000", "worker-after-reconnect", "durable-store-0", &missing,
        ).expect("missing resolution remains an assessment");
        assert_eq!(missing_report.disposition, EventTraceDisposition::Indeterminate);
        assert!(missing_report.envelope.is_none());

        let mut substituted = owner_time_resolutions(&schedule);
        let value = substituted.get_mut(&observation_key).expect("owner time resolution");
        value.mapping_receipt_sha256 = "f".repeat(64);
        let substituted_report = project_bitemporal_event(
            &schedule, "event-000000", "worker-after-reconnect", "durable-store-0", &substituted,
        ).expect("substituted resolution remains an assessment");
        assert_eq!(substituted_report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(substituted_report.envelope.is_none());
    }

    #[test]
    fn projection_rejects_multiple_observers_as_ambiguous_first_observation() {
        let mut schedule = generate_clocked_observation_persistence_schedule(0x2026_1117)
            .expect("clocked schedule");
        let original = schedule.actions.iter().find(|action| {
            matches!(action.action, ScheduleAction::EventObserved { .. })
        }).expect("observation").clone();
        let mut duplicate = original.clone();
        duplicate.action_id = format!("action-{:06}", schedule.actions.iter().map(|x| x.ordinal).max().unwrap_or(0) + 1);
        duplicate.ordinal = schedule.actions.iter().map(|x| x.ordinal).max().unwrap_or(0) + 1;
        duplicate.at_ns += 1;
        if let ScheduleAction::EventObserved { observer_id, .. } = &mut duplicate.action {
            *observer_id = "another-observer".to_owned();
        }
        schedule.actions.push(duplicate);
        let report = project_bitemporal_event(
            &schedule, "event-000000", "worker-after-reconnect", "durable-store-0",
            &owner_time_resolutions(&schedule),
        ).expect("projection assessment");
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.envelope.is_none());
        assert!(report.reason_codes.iter().any(|code| code == "FIRST_OBSERVATION_CARDINALITY_INVALID"));
    }

    #[test]
    fn projection_requires_owner_qualified_occurrence_time() {
        let schedule = generate_clocked_observation_persistence_schedule(0x2026_1118)
            .expect("clocked schedule");
        let mut resolutions = owner_time_resolutions(&schedule);
        let occurrence_key = schedule.actions.iter().find_map(|action| {
            if let ScheduleAction::EventProduced { clock_mapping_action_id, source_timestamp_ns, .. } = &action.action {
                Some((clock_mapping_action_id.clone(), *source_timestamp_ns))
            } else {
                None
            }
        }).expect("occurrence key");
        resolutions.remove(&occurrence_key);
        let report = project_bitemporal_event(
            &schedule, "event-000000", "worker-after-reconnect", "durable-store-0", &resolutions,
        ).expect("projection assessment");
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert!(report.envelope.is_none());
    }

    #[test]
    fn projection_rejects_occurrence_after_first_observation() {
        let schedule = generate_clocked_observation_persistence_schedule(0x2026_1119)
            .expect("clocked schedule");
        let mut resolutions = owner_time_resolutions(&schedule);
        let occurrence_key = schedule.actions.iter().find_map(|action| {
            if let ScheduleAction::EventProduced { clock_mapping_action_id, source_timestamp_ns, .. } = &action.action {
                Some((clock_mapping_action_id.clone(), *source_timestamp_ns))
            } else {
                None
            }
        }).expect("occurrence key");
        let occurrence = resolutions.get_mut(&occurrence_key).expect("occurrence time");
        occurrence.interval.earliest_ns = 1_500;
        occurrence.interval.latest_ns = 1_500;
        let report = project_bitemporal_event(
            &schedule, "event-000000", "worker-after-reconnect", "durable-store-0", &resolutions,
        ).expect("projection assessment");
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.envelope.is_none());
        assert!(report.reason_codes.iter().any(|code| code == "OBSERVATION_PRECEDES_EVENT_OCCURRENCE"));
    }

    #[test]
    fn projection_requires_production_mapping_to_be_admissible() {
        let mut schedule = generate_clocked_observation_persistence_schedule(0x2026_1120)
            .expect("clocked schedule");
        let production_mapping_id = schedule.actions.iter().find_map(|action| {
            if let ScheduleAction::EventProduced { event_id, clock_mapping_action_id, .. } = &action.action {
                (event_id == "event-000000").then_some(clock_mapping_action_id.clone())
            } else {
                None
            }
        }).expect("production map");
        let mapping = schedule.actions.iter_mut().find(|action| action.action_id == production_mapping_id)
            .expect("source clock mapping");
        if let ScheduleAction::ClockAdvance { qualification, .. } = &mut mapping.action {
            *qualification = GateState::Indeterminate;
        }
        let report = project_bitemporal_event(
            &schedule, "event-000000", "worker-after-reconnect", "durable-store-0",
            &owner_time_resolutions(&schedule),
        ).expect("projection assessment");
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert!(report.envelope.is_none());
    }

    #[test]
    fn owner_time_resolution_must_preserve_exact_mapping_validity() {
        let schedule = generate_clocked_observation_persistence_schedule(0x2026_1121)
            .expect("clocked schedule");
        let mut resolutions = owner_time_resolutions(&schedule);
        let observation_key = schedule.actions.iter().find_map(|action| {
            if let ScheduleAction::EventObserved { clock_mapping_action_id, observed_at_ns, .. } = &action.action {
                Some((clock_mapping_action_id.clone(), *observed_at_ns))
            } else {
                None
            }
        }).expect("observation key");
        let value = resolutions.get_mut(&observation_key).expect("resolution");
        value.mapping_valid_through_ns -= 1;
        let report = project_bitemporal_event(
            &schedule, "event-000000", "worker-after-reconnect", "durable-store-0", &resolutions,
        ).expect("projection assessment");
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.envelope.is_none());
    }

    #[test]
    fn observer_and_store_timestamps_use_independent_qualified_clock_series() {
        let schedule = generate_clocked_observation_persistence_schedule(0x2026_1115)
            .expect("clocked bitemporal schedule");
        validate_schedule(&schedule).expect("valid schedule");
        let report = replay_virtual_schedule(&schedule).expect("replay");
        let observed = report.records.iter().find(|record| {
            matches!(record.action, ScheduleAction::EventObserved { .. })
        }).expect("observation record");
        let persisted = report.records.iter().find(|record| {
            matches!(record.action, ScheduleAction::EventPersisted { .. })
        }).expect("persistence record");
        assert_eq!(observed.disposition, SchedulerDisposition::Applied);
        assert_eq!(persisted.disposition, SchedulerDisposition::Applied);
        let (observed_domain, observed_at) = match &observed.action {
            ScheduleAction::EventObserved { clock_domain, observed_at_ns, .. } => {
                (clock_domain, *observed_at_ns)
            }
            _ => unreachable!("filtered observation record"),
        };
        let (store_domain, persisted_at) = match &persisted.action {
            ScheduleAction::EventPersisted { clock_domain, persisted_at_ns, .. } => {
                (clock_domain, *persisted_at_ns)
            }
            _ => unreachable!("filtered persistence record"),
        };
        assert_ne!(observed_domain, store_domain);
        assert_ne!(observed.at_ns, observed_at);
        assert_ne!(persisted.at_ns, persisted_at);
        assert!(persisted_at > observed_at);
        assert_eq!(report.observed_count, 1);
        assert_eq!(report.persisted_count, 1);
    }

    #[test]
    fn observer_and_store_clock_faults_are_all_classified_fail_closed() {
        let cases = [
            (CaptureClockFault::ObserverUnqualified, SchedulerDisposition::ClockMappingUnqualified, "observer"),
            (CaptureClockFault::ObserverExpired, SchedulerDisposition::ClockMappingExpired, "observer"),
            (CaptureClockFault::ObserverNotYetValid, SchedulerDisposition::ClockMappingNotYetValid, "observer"),
            (CaptureClockFault::ObserverValidityInvalid, SchedulerDisposition::ClockMappingValidityInvalid, "observer"),
            (CaptureClockFault::StoreUnqualified, SchedulerDisposition::ClockMappingUnqualified, "store"),
            (CaptureClockFault::StoreExpired, SchedulerDisposition::ClockMappingExpired, "store"),
            (CaptureClockFault::StoreNotYetValid, SchedulerDisposition::ClockMappingNotYetValid, "store"),
            (CaptureClockFault::StoreValidityInvalid, SchedulerDisposition::ClockMappingValidityInvalid, "store"),
        ];
        for (fault, expected, domain) in cases {
            let schedule = generate_capture_clock_fault_schedule(0x2026_1122, fault)
                .expect("capture clock fault schedule");
            validate_schedule(&schedule).expect("fault remains structurally valid");
            let report = replay_virtual_schedule(&schedule).expect("fault replay");
            let action_disposition = report.records.iter().find_map(|record| match (&record.action, domain) {
                (ScheduleAction::EventObserved { .. }, "observer") => Some(record.disposition),
                (ScheduleAction::EventPersisted { .. }, "store") => Some(record.disposition),
                _ => None,
            }).expect("target capture action");
            assert_eq!(action_disposition, expected);
        }
    }

    #[test]
    fn clock_mapping_fault_generator_preserves_three_distinct_failures() {
        let cases = [
            (ClockMappingFault::Unqualified, SchedulerDisposition::ClockMappingUnqualified),
            (ClockMappingFault::Expired, SchedulerDisposition::ClockMappingExpired),
            (ClockMappingFault::StaleReference, SchedulerDisposition::StaleClockMappingReference),
        ];
        for (fault, expected) in cases {
            let schedule = generate_clock_mapping_fault_schedule(0x2026_1115, fault);
            validate_schedule(&schedule).expect("fault schedule remains structurally valid");
            let report = replay_virtual_schedule(&schedule).expect("fault schedule replays");
            assert!(report.records.iter().any(|record| {
                matches!(record.action, ScheduleAction::EventProduced { .. })
                    && record.disposition == expected
            }));
        }
    }

    #[test]
    fn cancellation_commit_race_generator_exposes_both_orders() {
        let before_ack = generate_cancellation_commit_race_schedule(
            0x2026_1115,
            CancellationCommitOrder::CommitBeforeAcknowledgement,
        );
        let after_ack = generate_cancellation_commit_race_schedule(
            0x2026_1115,
            CancellationCommitOrder::CommitAfterAcknowledgement,
        );
        validate_schedule(&before_ack).expect("valid commit-before-ack schedule");
        validate_schedule(&after_ack).expect("valid commit-after-ack schedule");
        let before = replay_virtual_schedule(&before_ack).expect("before-ack replay");
        let after = replay_virtual_schedule(&after_ack).expect("after-ack replay");
        assert!(before.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::AcceptedCancellationAfterCommit
        }));
        assert!(after.records.iter().any(|record| {
            record.disposition == SchedulerDisposition::CommitAfterAcceptedCancellation
        }));
        assert!(before.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::StopDischarged { .. })
                && record.disposition == SchedulerDisposition::Applied
        }));
        assert!(after.records.iter().any(|record| {
            matches!(record.action, ScheduleAction::StopDischarged { .. })
                && record.disposition == SchedulerDisposition::Applied
        }));
        assert_eq!(
            canonical_schedule_sha256(&before_ack),
            canonical_schedule_sha256(&generate_cancellation_commit_race_schedule(
                0x2026_1115,
                CancellationCommitOrder::CommitBeforeAcknowledgement,
            )),
        );
    }

    #[test]
    fn nanosecond_clock_series_rejects_other_units() {
        let mut schedule = generate_delivery_schedule(71, 2).expect("valid schedule");
        if let Some(item) = schedule.actions.iter_mut().find(|item| {
            matches!(item.action, ScheduleAction::ClockAdvance { .. })
        }) {
            if let ScheduleAction::ClockAdvance { unit, .. } = &mut item.action {
                *unit = "milliseconds".to_owned();
            }
        }
        assert!(matches!(
            validate_schedule(&schedule),
            Err(ReplayError::UnsupportedTimeUnit(unit)) if unit == "milliseconds"
        ));
    }

    #[test]
    fn receipt_digests_require_canonical_sha256_shape() {
        let mut schedule = generate_delivery_schedule(53, 2).expect("valid schedule");
        if let Some(item) = schedule.actions.iter_mut().find(|item| {
            matches!(item.action, ScheduleAction::ClockAdvance { .. })
        }) {
            if let ScheduleAction::ClockAdvance { mapping_receipt_sha256, .. } = &mut item.action {
                *mapping_receipt_sha256 = "not-a-digest".to_owned();
            }
        }
        assert!(matches!(validate_schedule(&schedule), Err(ReplayError::MalformedSha256("mapping_receipt_sha256"))));

        let mut schedule = generate_delivery_schedule(59, 1).expect("valid schedule");
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let trigger_action_id = format!("action-{ordinal:06}");
        push_action(&mut schedule.actions, &mut ordinal, 20_000, ScheduleAction::StopTriggered {
            obligation_id: "digest-stop".to_owned(), policy_revision: "policy-1".to_owned(),
        });
        push_action(&mut schedule.actions, &mut ordinal, 20_001, ScheduleAction::StopDischarged {
            obligation_id: "digest-stop".to_owned(), trigger_action_id: trigger_action_id.clone(),
            policy_revision_action_id: trigger_action_id, policy_revision: "policy-1".to_owned(),
            receipt_ref: "receipt".to_owned(), receipt_sha256: "z".repeat(64),
        });
        assert!(matches!(validate_schedule(&schedule), Err(ReplayError::MalformedSha256("stop_discharge_receipt_sha256"))));
    }

    #[test]
    fn duplicate_cancel_acknowledgement_is_not_a_second_success() {
        let mut schedule = generate_delivery_schedule(47, 1).expect("valid schedule");
        let mut ordinal = schedule.actions.iter().map(|item| item.ordinal).max().unwrap_or(0) + 1;
        let t = 20_000;
        let operation_id = "cancel-op".to_owned();
        let attempt_id = "cancel-attempt".to_owned();
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::CancellationRequested {
            operation_id: operation_id.clone(), attempt_id: attempt_id.clone(),
        });
        for (offset, accepted) in [(1, true), (2, true), (3, false)] {
            push_action(&mut schedule.actions, &mut ordinal, t + offset, ScheduleAction::CancellationAcknowledged {
                operation_id: operation_id.clone(), attempt_id: attempt_id.clone(), accepted,
            });
        }
        let report = replay_virtual_schedule(&schedule).expect("replay");
        let ack_dispositions: Vec<_> = report.records.iter().filter_map(|record| {
            if matches!(record.action, ScheduleAction::CancellationAcknowledged { .. }) {
                Some(record.disposition)
            } else {
                None
            }
        }).collect();
        assert_eq!(ack_dispositions, vec![
            SchedulerDisposition::Applied,
            SchedulerDisposition::Duplicate,
            SchedulerDisposition::ConflictingAcknowledgement,
        ]);
    }
}
