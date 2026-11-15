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
    #[error("clock moved backwards in domain {0}")]
    ClockRegression(String),
    #[error("event was produced without a clock advance: {0}")]
    MissingClock(String),
    #[error("event timestamp exceeds the latest virtual clock tick: {0}")]
    EventBeyondClock(String),
    #[error("event source time regressed in domain {0}")]
    EventTimeRegression(String),
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
    let mut clock_ticks = BTreeMap::<String, u64>::new();
    let mut event_times = BTreeMap::<String, u64>::new();
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
                calibration_id, mapping_receipt_ref, ..
            } => {
                require_id(clock_domain, "clock_domain")?;
                require_id(basis_id, "basis_id")?;
                require_id(epoch, "epoch")?;
                require_id(scale, "scale")?;
                require_id(unit, "unit")?;
                require_id(calibration_id, "calibration_id")?;
                require_id(mapping_receipt_ref, "mapping_receipt_ref")?;
                if let Some(previous) = clock_ticks.get(clock_domain) {
                    if local_tick_ns < previous {
                        return Err(ReplayError::ClockRegression(clock_domain.clone()));
                    }
                }
                clock_ticks.insert(clock_domain.clone(), *local_tick_ns);
            }
            ScheduleAction::EventProduced {
                event_id, clock_domain, source_timestamp_ns, causal_predecessors,
            } => {
                require_id(event_id, "event_id")?;
                require_id(clock_domain, "clock_domain")?;
                let latest_tick = clock_ticks
                    .get(clock_domain)
                    .ok_or_else(|| ReplayError::MissingClock(clock_domain.clone()))?;
                if source_timestamp_ns > latest_tick {
                    return Err(ReplayError::EventBeyondClock(event_id.clone()));
                }
                if let Some(previous_time) = event_times.get(clock_domain) {
                    if source_timestamp_ns < previous_time {
                        return Err(ReplayError::EventTimeRegression(clock_domain.clone()));
                    }
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
                event_times.insert(clock_domain.clone(), *source_timestamp_ns);
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
                if let ScheduleAction::AttemptReconciled { receipt_ref, .. } = &scheduled.action {
                    require_id(receipt_ref, "receipt_ref")?;
                }
            }
            ScheduleAction::StopTriggered { obligation_id, policy_revision }
            | ScheduleAction::StopPolicyRevised { obligation_id, policy_revision } => {
                require_id(obligation_id, "obligation_id")?;
                require_id(policy_revision, "policy_revision")?;
            }
            ScheduleAction::StopDischarged { obligation_id, policy_revision, receipt_ref, .. } => {
                require_id(obligation_id, "obligation_id")?;
                require_id(policy_revision, "policy_revision")?;
                require_id(receipt_ref, "receipt_ref")?;
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

fn require_id(value: &str, name: &'static str) -> Result<(), ReplayError> {
    if value.trim().is_empty() {
        Err(ReplayError::EmptyIdentifier(name))
    } else {
        Ok(())
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

/// Replay a schedule in virtual time. The output records schedule intent and
/// scheduler disposition. It is not an event-order safety verdict; callers
/// pass the resulting observations to the v0.2 evaluator.
pub fn replay_virtual_schedule(schedule: &VirtualSchedule) -> Result<ReplayReport, ReplayError> {
    validate_schedule(schedule)?;
    let schedule_sha256 = canonical_schedule_sha256(schedule)?;
    let mut partitioned = BTreeSet::<String>::new();
    let mut crashed = BTreeSet::<String>::new();
    let mut produced = BTreeSet::<String>::new();
    let mut delivered = BTreeSet::<(String, String)>::new();
    let mut observed = BTreeSet::<(String, String)>::new();
    let mut persisted = BTreeSet::<(String, String)>::new();
    let mut cancel_requests = BTreeSet::<(String, String)>::new();
    let mut stop_policies = BTreeMap::<String, String>::new();
    let mut attempts = BTreeSet::<(String, String)>::new();
    let mut committed_attempts = BTreeSet::<(String, String)>::new();
    let mut reconciled_attempts = BTreeSet::<(String, String)>::new();
    let mut records = Vec::new();

    for scheduled in ordered_actions(schedule) {
        let disposition = match &scheduled.action {
            ScheduleAction::ClockAdvance { .. } => SchedulerDisposition::Applied,
            ScheduleAction::EventProduced { event_id, .. } => {
                produced.insert(event_id.clone());
                SchedulerDisposition::Applied
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
                if !observed.contains(&observation) {
                    SchedulerDisposition::PreconditionViolation
                } else if persisted.insert(persistence) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::Duplicate
                }
            }
            ScheduleAction::CancellationRequested { operation_id, attempt_id } => {
                cancel_requests.insert((operation_id.clone(), attempt_id.clone()));
                SchedulerDisposition::Applied
            }
            ScheduleAction::CancellationAcknowledged { operation_id, attempt_id, .. } => {
                if cancel_requests.contains(&(operation_id.clone(), attempt_id.clone())) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::StopTriggered { obligation_id, policy_revision }
            | ScheduleAction::StopPolicyRevised { obligation_id, policy_revision } => {
                stop_policies.insert(obligation_id.clone(), policy_revision.clone());
                SchedulerDisposition::Applied
            }
            ScheduleAction::StopDischarged { obligation_id, policy_revision, .. } => {
                if stop_policies.get(obligation_id) == Some(policy_revision) {
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
                if attempts.contains(&key) && committed_attempts.insert(key) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::AttemptReconciled { operation_id, attempt_id, .. } => {
                let key = (operation_id.clone(), attempt_id.clone());
                if attempts.contains(&key) && reconciled_attempts.insert(key) {
                    SchedulerDisposition::Applied
                } else {
                    SchedulerDisposition::PreconditionViolation
                }
            }
            ScheduleAction::ProcessCrashed { process_id } => {
                crashed.insert(process_id.clone());
                SchedulerDisposition::Applied
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
}
