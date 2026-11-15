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

use crate::{EventTraceDisposition, GateState};
use crate::event_order_v0_2::{
    BitemporalEventEnvelope, QualifiedEventTime, compare_qualified_event_times,
};

pub const SCHEDULE_SCHEMA_VERSION: &str = "symthaea.event-order-schedule.v0.3";
pub const GENERATOR_VERSION: &str = "symthaea.event-order-generator.v0.3.2";
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
pub struct ScheduledAction {
    pub action_id: String,
    pub at_ns: u64,
    pub ordinal: u64,
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
    ClockMappingUnqualified,
    ClockMappingRejected,
    ClockMappingExpired,
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
            | ScheduleAction::EffectAttempted { operation_id, attempt_id }
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
    let mut committed_attempts = BTreeMap::<(String, String), String>::new();
    let mut reconciled_attempts =
        BTreeMap::<(String, String), (ReconciliationOutcome, String, String)>::new();
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
                    stop_triggers.insert(obligation_id.clone(), StopReplayState {
                        trigger_action_id: scheduled.action_id.clone(),
                        policy_revision_action_id: scheduled.action_id.clone(),
                        policy_revision: policy_revision.clone(),
                    });
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::StopPolicyRevised { obligation_id, policy_revision } => {
                if let Some(state) = stop_triggers.get_mut(obligation_id) {
                    state.policy_revision = policy_revision.clone();
                    state.policy_revision_action_id = scheduled.action_id.clone();
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
                if exact_binding && stop_discharges.insert(obligation_id.clone()) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::EffectAttempted { operation_id, attempt_id } => {
                let key = (operation_id.clone(), attempt_id.clone());
                if attempts.insert(key) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::EffectCommitted { operation_id, attempt_id, effect_id } => {
                let key = (operation_id.clone(), attempt_id.clone());
                if !attempts.contains(&key) {
                    SchedulerDisposition::PreconditionViolation
                } else if let Some(previous_effect_id) = committed_attempts.get(&key) {
                    if previous_effect_id == effect_id {
                        SchedulerDisposition::Duplicate
                    } else {
                        SchedulerDisposition::ConflictingEffectCommit
                    }
                } else {
                    committed_attempts.insert(key.clone(), effect_id.clone());
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
                let terminal = (*outcome, receipt_ref.clone(), receipt_sha256.clone());
                if !attempts.contains(&key) {
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
        records.push(ReplayRecord {
            action_id: scheduled.action_id,
            at_ns: scheduled.at_ns,
            ordinal: scheduled.ordinal,
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
pub struct BitemporalProjectionReport {
    pub disposition: EventTraceDisposition,
    pub envelope: Option<BitemporalEventEnvelope>,
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
    if qualified.source_clock_domain_id != source_clock_domain_id
        || qualified.mapping_receipt_ref != receipt_ref
        || qualified.mapping_receipt_sha256 != receipt_sha256
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
            disposition: EventTraceDisposition::Indeterminate,
            envelope: None,
            reason_codes: vec![code.to_owned()],
        },
        ProjectionFault::Critical(code) => BitemporalProjectionReport {
            disposition: EventTraceDisposition::CriticalFailure,
            envelope: None,
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
            event_id: candidate, observer_id: observer, ..
        } if candidate == event_id && observer == observer_id)
    }).collect();
    if observations.len() != 1 {
        return Ok(if observations.is_empty() {
            projection_failure(ProjectionFault::Indeterminate("FIRST_OBSERVATION_EVIDENCE_MISSING"))
        } else {
            projection_failure(ProjectionFault::Critical("FIRST_OBSERVATION_CARDINALITY_INVALID"))
        });
    }

    let persistence_records: Vec<_> = schedule.actions.iter().filter(|action| {
        matches!(&action.action, ScheduleAction::EventPersisted {
            event_id: candidate, observer_id: observer, store_id: store, ..
        } if candidate == event_id && observer == observer_id && store == store_id)
    }).collect();
    if persistence_records.len() != 1 {
        return Ok(if persistence_records.is_empty() {
            projection_failure(ProjectionFault::Indeterminate("DURABLE_RECORD_EVIDENCE_MISSING"))
        } else {
            projection_failure(ProjectionFault::Critical("DURABLE_RECORD_CARDINALITY_INVALID"))
        });
    }

    let replay = replay_virtual_schedule(schedule)?;
    for (action, reason) in [
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
            let mapping = schedule.actions.iter().find(|action| action.action_id == *clock_mapping_action_id);
            match mapping.map(|action| &action.action) {
                Some(ScheduleAction::ClockAdvance { mapping_receipt_ref, mapping_receipt_sha256, .. }) => {
                    resolve_owner_qualified_time(
                        schedule, clock_mapping_action_id, *source_timestamp_ns, clock_domain,
                        mapping_receipt_ref, mapping_receipt_sha256, resolutions,
                    ).ok()
                }
                _ => None,
            }
        }
        _ => None,
    };

    let mut reasons = BTreeSet::new();
    if occurred_at.is_none() {
        reasons.insert("OCCURRENCE_TIME_NOT_OWNER_QUALIFIED".to_owned());
    }

    let order = compare_qualified_event_times(&observed_time, &persisted_time).order;
    let (disposition, capture_qualification) = match order {
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
            disposition: EventTraceDisposition::CriticalFailure,
            envelope: None,
            reason_codes: reasons.into_iter().collect(),
        });
    }

    Ok(BitemporalProjectionReport {
        disposition,
        envelope: Some(BitemporalEventEnvelope {
            event_id: event_id.to_owned(),
            occurred_at,
            first_observed_at: observed_time,
            durably_recorded_at: persisted_time,
            capture_qualification,
            supersedes_event_id: superseded.into_iter().next(),
        }),
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
        let (clock_domain, receipt_ref, receipt_sha256) = match &action.action {
            ScheduleAction::ClockAdvance {
                clock_domain, mapping_receipt_ref, mapping_receipt_sha256, ..
            } => (clock_domain.clone(), mapping_receipt_ref.clone(), mapping_receipt_sha256.clone()),
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
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 100_000,
            max_interval_width_ns: 0,
            interval: crate::event_order_v0_2::EventTimeIntervalNs {
                earliest_ns: normalized_timestamp_ns,
                latest_ns: normalized_timestamp_ns,
            },
        }
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
