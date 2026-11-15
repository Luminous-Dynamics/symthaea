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

use crate::GateState;

pub const SCHEDULE_SCHEMA_VERSION: &str = "symthaea.event-order-schedule.v0.3";
pub const GENERATOR_VERSION: &str = "symthaea.event-order-generator.v0.3.1";
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
    },
    EventPersisted {
        event_id: String,
        observer_id: String,
        store_id: String,
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
    ConflictingAcknowledgement,
    ConflictingReconciliation,
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
    let mut clock_ticks = BTreeMap::<ClockSeriesKey, u64>::new();
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
                calibration_id, mapping_receipt_ref, mapping_receipt_sha256, ..
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
                clock_ticks.insert(
                    clock_series_key(clock_domain, basis_id, epoch, scale, unit),
                    *local_tick_ns,
                );
            }
            ScheduleAction::EventProduced {
                event_id, clock_domain, basis_id, epoch, scale, unit,
                source_timestamp_ns, causal_predecessors,
            } => {
                require_id(event_id, "event_id")?;
                require_id(clock_domain, "clock_domain")?;
                require_id(basis_id, "basis_id")?;
                require_id(epoch, "epoch")?;
                require_id(scale, "scale")?;
                require_id(unit, "unit")?;
                let latest_tick = clock_ticks
                    .get(&clock_series_key(clock_domain, basis_id, epoch, scale, unit))
                    .ok_or_else(|| ReplayError::MissingClock(clock_domain.clone()))?;
                if source_timestamp_ns > latest_tick {
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
            ScheduleAction::EventObserved { event_id, observer_id } => {
                require_id(event_id, "event_id")?;
                require_id(observer_id, "observer_id")?;
            }
            ScheduleAction::EventPersisted { event_id, observer_id, store_id } => {
                require_id(event_id, "event_id")?;
                require_id(observer_id, "observer_id")?;
                require_id(store_id, "store_id")?;
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
    let mut clock_ticks = BTreeMap::<ClockSeriesKey, u64>::new();
    let mut event_times = BTreeMap::<ClockSeriesKey, u64>::new();
    let mut delivered = BTreeSet::<(String, String)>::new();
    let mut observed = BTreeSet::<(String, String)>::new();
    let mut persisted = BTreeSet::<(String, String)>::new();
    let mut cancel_requests = BTreeSet::<(String, String)>::new();
    let mut cancel_acknowledgements = BTreeMap::<(String, String), bool>::new();
    let mut stop_triggers = BTreeMap::<String, StopReplayState>::new();
    let mut stop_discharges = BTreeSet::<String>::new();
    let mut attempts = BTreeSet::<(String, String)>::new();
    let mut committed_attempts = BTreeSet::<(String, String)>::new();
    let mut reconciled_attempts =
        BTreeMap::<(String, String), (ReconciliationOutcome, String, String)>::new();
    let mut records = Vec::new();

    for scheduled in ordered_actions(schedule) {
        let disposition = match &scheduled.action {
            ScheduleAction::ClockAdvance {
                clock_domain, basis_id, epoch, scale, unit, local_tick_ns, ..
            } => {
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                let previous = clock_ticks.insert(key, *local_tick_ns);
                if previous.is_some_and(|tick| *local_tick_ns < tick) {
                    SchedulerDisposition::ClockRegression
                } else {
                    SchedulerDisposition::Applied
                }
            }
            ScheduleAction::EventProduced {
                event_id, clock_domain, basis_id, epoch, scale, unit, source_timestamp_ns, ..
            } => {
                produced.insert(event_id.clone());
                let key = clock_series_key(clock_domain, basis_id, epoch, scale, unit);
                let previous = event_times.insert(key, *source_timestamp_ns);
                if previous.is_some_and(|timestamp| *source_timestamp_ns < timestamp) {
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
            ScheduleAction::EventObserved { event_id, observer_id } => {
                let key = (event_id.clone(), observer_id.clone());
                if crashed.contains(observer_id) {
                    SchedulerDisposition::BlockedByCrashedRecipient
                } else if !delivered.contains(&key) {
                    SchedulerDisposition::PreconditionViolation
                } else if observed.insert(key) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::Duplicate
                }
            }
            ScheduleAction::EventPersisted { event_id, observer_id, store_id } => {
                let observation = (event_id.clone(), observer_id.clone());
                let persistence = (event_id.clone(), store_id.clone());
                if crashed.contains(observer_id) {
                    SchedulerDisposition::BlockedByCrashedRecipient
                } else if !observed.contains(&observation) {
                    SchedulerDisposition::PreconditionViolation
                } else if persisted.insert(persistence) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::Duplicate
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
            ScheduleAction::EffectCommitted { operation_id, attempt_id, .. } => {
                let key = (operation_id.clone(), attempt_id.clone());
                if !attempts.contains(&key) || !committed_attempts.insert(key.clone()) {
                    SchedulerDisposition::PreconditionViolation
                } else if cancel_acknowledgements.get(&key) == Some(&true) {
                    SchedulerDisposition::CommitAfterAcceptedCancellation
                } else {
                    SchedulerDisposition::Applied
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
    let trigger_action_id = format!("action-{ordinal:06}");
    push_action(
        &mut actions,
        &mut ordinal,
        1,
        ScheduleAction::EffectAttempted {
            operation_id: operation_id.clone(),
            attempt_id: attempt_id.clone(),
        },
    );
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

    let mut push_commit = || {
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
    };
    let mut push_ack = || {
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
    };
    match order {
        CancellationCommitOrder::CommitBeforeAcknowledgement => {
            push_commit();
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
            push_ack();
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
            trigger_action_id,
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
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::EventObserved {
            event_id: event_id.clone(), observer_id: observer.clone(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 1, ScheduleAction::ProcessCrashed {
            process_id: observer.clone(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 2, ScheduleAction::EventPersisted {
            event_id: event_id.clone(), observer_id: observer.clone(), store_id: "store-1".to_owned(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 3, ScheduleAction::ProcessRestarted {
            process_id: observer.clone(),
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 4, ScheduleAction::EventPersisted {
            event_id, observer_id: observer, store_id: "store-1".to_owned(),
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
        push_action(&mut schedule.actions, &mut ordinal, t, ScheduleAction::ClockAdvance {
            clock_domain: "clock-a".to_owned(),
            basis_id: "basis-a".to_owned(),
            epoch: "epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            local_tick_ns: 50,
            offset_ns: -250_000,
            uncertainty_ns: 50_000,
            calibration_id: "calibration-backwards".to_owned(),
            mapping_valid_through_ns: 60,
            mapping_receipt_ref: "synthetic://mapping/backwards".to_owned(),
            mapping_receipt_sha256: "d".repeat(64),
            qualification: GateState::Indeterminate,
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 1, ScheduleAction::EventProduced {
            event_id: "event-backwards".to_owned(),
            clock_domain: "clock-a".to_owned(),
            basis_id: "basis-clock-a".to_owned(),
            epoch: "epoch-0".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
            source_timestamp_ns: 50,
            causal_predecessors: Vec::new(),
        });
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
            mapping_valid_through_ns: 100,
            mapping_receipt_ref: "synthetic://mapping/new-epoch".to_owned(),
            mapping_receipt_sha256: "e".repeat(64),
            qualification: GateState::Indeterminate,
        });
        push_action(&mut schedule.actions, &mut ordinal, t + 3, ScheduleAction::EventProduced {
            event_id: "event-new-epoch".to_owned(),
            clock_domain: "clock-a".to_owned(),
            basis_id: "basis-clock-a".to_owned(),
            epoch: "epoch-1".to_owned(),
            scale: "tai".to_owned(),
            unit: "nanoseconds".to_owned(),
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
