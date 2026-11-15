// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Additive v0.2 contracts for qualified time, bitemporal visibility,
//! execution-attempt lineage, and multiple scoped stop obligations.
//!
//! This module is deliberately non-authoritative. It consumes owner-qualified
//! states; it does not authenticate clocks, verify receipt bytes, authorize
//! effects, or replace the v0.1 event ledger. An EventOrder::Before result
//! means the first argument is definitely before the second argument.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{EventOrder, EventTraceDisposition, GateState, ReconciledOperationOutcome};

pub const EVENT_ORDER_V0_2_SCHEMA_VERSION: &str = "engineering-decision-event-order-v0.2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventTimeUnit {
    Nanoseconds,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeBasisDescriptor {
    pub basis_id: String,
    pub epoch: String,
    pub scale: String,
    pub unit: EventTimeUnit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventTimeIntervalNs {
    pub earliest_ns: i64,
    pub latest_ns: i64,
}

/// A source-clock mapping into one explicitly named common time basis.
///
/// The receipt reference, digest and qualification are owner-supplied. This
/// crate checks their structure and binding shape, not their authenticity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedClockMapping {
    pub source_clock_domain_id: String,
    pub target_time_basis: TimeBasisDescriptor,
    pub mapping_receipt_ref: String,
    pub mapping_receipt_sha256: String,
    pub qualification: GateState,
    pub valid_from_ns: i64,
    pub valid_through_ns: i64,
    /// Maximum full interval width accepted for this mapping, not a point error.
    pub max_interval_width_ns: u64,
}

/// An event time already mapped into a named common basis, with the mapping
/// receipt identity carried alongside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedEventTime {
    pub source_clock_domain_id: String,
    pub time_basis: TimeBasisDescriptor,
    pub mapping_receipt_ref: String,
    pub mapping_receipt_sha256: String,
    pub mapping_qualification: GateState,
    pub mapping_valid_from_ns: i64,
    pub mapping_valid_through_ns: i64,
    pub max_interval_width_ns: u64,
    pub interval: EventTimeIntervalNs,
}

impl QualifiedClockMapping {
    /// Bind an already-normalized interval to this owner-qualified mapping.
    /// The interval is not transformed here; the owner adapter must perform
    /// and qualify the actual source-to-basis conversion.
    pub fn bind_normalized_interval(&self, interval: EventTimeIntervalNs) -> QualifiedEventTime {
        QualifiedEventTime {
            source_clock_domain_id: self.source_clock_domain_id.clone(),
            time_basis: self.target_time_basis.clone(),
            mapping_receipt_ref: self.mapping_receipt_ref.clone(),
            mapping_receipt_sha256: self.mapping_receipt_sha256.clone(),
            mapping_qualification: self.qualification,
            mapping_valid_from_ns: self.valid_from_ns,
            mapping_valid_through_ns: self.valid_through_ns,
            max_interval_width_ns: self.max_interval_width_ns,
            interval,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockOrderAssessment {
    pub order: EventOrder,
    pub reason_codes: Vec<String>,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_qualified_time(
    value: &QualifiedEventTime,
    side: &str,
    reasons: &mut BTreeSet<String>,
) -> bool {
    let mut valid = true;
    if value.source_clock_domain_id.trim().is_empty()
        || value.time_basis.basis_id.trim().is_empty()
        || value.time_basis.epoch.trim().is_empty()
        || value.time_basis.scale.trim().is_empty()
        || value.mapping_receipt_ref.trim().is_empty()
        || !is_sha256_hex(&value.mapping_receipt_sha256)
    {
        reasons.insert(format!("CLOCK_MAPPING_BINDING_INVALID:{side}"));
        valid = false;
    }
    if value.interval.earliest_ns > value.interval.latest_ns {
        reasons.insert(format!("CLOCK_INTERVAL_REVERSED:{side}"));
        valid = false;
    }
    if value.mapping_valid_from_ns > value.mapping_valid_through_ns {
        reasons.insert(format!("CLOCK_MAPPING_VALIDITY_INTERVAL_INVALID:{side}"));
        valid = false;
    }
    if value.interval.earliest_ns < value.mapping_valid_from_ns
        || value.interval.latest_ns > value.mapping_valid_through_ns
    {
        reasons.insert(format!("CLOCK_MAPPING_OUTSIDE_VALIDITY:{side}"));
        valid = false;
    }
    let width = (value.interval.latest_ns as i128
        - value.interval.earliest_ns as i128) as u128;
    if width > value.max_interval_width_ns as u128 {
        reasons.insert(format!("CLOCK_UNCERTAINTY_EXCEEDS_DECLARED_BOUND:{side}"));
        valid = false;
    }
    valid
}

/// Compare two already-normalized event-time intervals.
///
/// Cross-domain values are comparable only when both map into the same
/// explicitly named basis and both mappings are owner-qualified PASS. A
/// timestamp from an unqualified or different basis is UNKNOWN, not an
/// invitation to compare numerically. Closed intervals that touch are
/// OVERLAPPING because strict precedence has not been established.
pub fn compare_qualified_event_times(
    first: &QualifiedEventTime,
    second: &QualifiedEventTime,
) -> ClockOrderAssessment {
    let mut reasons = BTreeSet::new();
    let first_valid = validate_qualified_time(first, "first", &mut reasons);
    let second_valid = validate_qualified_time(second, "second", &mut reasons);
    if !first_valid || !second_valid {
        return ClockOrderAssessment {
            order: EventOrder::Inconsistent,
            reason_codes: reasons.into_iter().collect(),
        };
    }

    if first.mapping_qualification == GateState::Fail
        || second.mapping_qualification == GateState::Fail
    {
        reasons.insert("CLOCK_MAPPING_QUALIFICATION_FAILED".to_owned());
        return ClockOrderAssessment {
            order: EventOrder::Inconsistent,
            reason_codes: reasons.into_iter().collect(),
        };
    }
    if first.mapping_qualification != GateState::Pass
        || second.mapping_qualification != GateState::Pass
    {
        reasons.insert("CLOCK_MAPPING_QUALIFICATION_UNRESOLVED".to_owned());
        return ClockOrderAssessment {
            order: EventOrder::Unknown,
            reason_codes: reasons.into_iter().collect(),
        };
    }
    if first.time_basis != second.time_basis {
        reasons.insert("CLOCK_TIME_BASIS_MISMATCH".to_owned());
        return ClockOrderAssessment {
            order: EventOrder::Unknown,
            reason_codes: reasons.into_iter().collect(),
        };
    }

    let order = if first.interval.latest_ns < second.interval.earliest_ns {
        EventOrder::Before
    } else if first.interval.earliest_ns > second.interval.latest_ns {
        EventOrder::After
    } else {
        reasons.insert("CLOCK_INTERVALS_OVERLAP".to_owned());
        EventOrder::Overlapping
    };

    ClockOrderAssessment {
        order,
        reason_codes: reasons.into_iter().collect(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierVisibility {
    AvailableAtK,
    NotAvailableAtK,
    IndeterminateAtK,
}

/// Bitemporal event envelope. occurred_at describes event time; observed and
/// recorded times describe when the evidence became knowable and durable.
/// Corrections are append-only references, not overwrites of prior history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitemporalEventEnvelope {
    pub event_id: String,
    pub occurred_at: Option<QualifiedEventTime>,
    pub first_observed_at: QualifiedEventTime,
    pub durably_recorded_at: QualifiedEventTime,
    pub capture_qualification: GateState,
    pub supersedes_event_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierVisibilityAssessment {
    pub visibility: FrontierVisibility,
    pub reason_codes: Vec<String>,
}

/// Decide whether an event was available to a decision at frontier K.
///
/// Occurrence time is intentionally not used to establish knowability. Both
/// first observation and durable recording must be definitely BEFORE K.
/// Late-arriving records therefore cannot be inserted into a historical
/// decision's information set merely because their occurrence timestamp is old.
pub fn assess_visibility_at_frontier(
    event: &BitemporalEventEnvelope,
    decision_frontier: &QualifiedEventTime,
) -> FrontierVisibilityAssessment {
    let mut reasons = BTreeSet::new();
    if event.event_id.trim().is_empty() {
        reasons.insert("BITEMPORAL_EVENT_ID_INVALID".to_owned());
        return FrontierVisibilityAssessment {
            visibility: FrontierVisibility::IndeterminateAtK,
            reason_codes: reasons.into_iter().collect(),
        };
    }
    if event.capture_qualification == GateState::Fail {
        reasons.insert("BITEMPORAL_CAPTURE_QUALIFICATION_FAILED".to_owned());
        return FrontierVisibilityAssessment {
            visibility: FrontierVisibility::IndeterminateAtK,
            reason_codes: reasons.into_iter().collect(),
        };
    }
    if event.capture_qualification != GateState::Pass {
        reasons.insert("BITEMPORAL_CAPTURE_QUALIFICATION_UNRESOLVED".to_owned());
        return FrontierVisibilityAssessment {
            visibility: FrontierVisibility::IndeterminateAtK,
            reason_codes: reasons.into_iter().collect(),
        };
    }

    let observation_to_record = compare_qualified_event_times(
        &event.first_observed_at,
        &event.durably_recorded_at,
    );
    if matches!(observation_to_record.order, EventOrder::After | EventOrder::Inconsistent) {
        reasons.insert("BITEMPORAL_RECORD_PRECEDES_OBSERVATION".to_owned());
        return FrontierVisibilityAssessment {
            visibility: FrontierVisibility::IndeterminateAtK,
            reason_codes: reasons.into_iter().collect(),
        };
    }
    if observation_to_record.order == EventOrder::Unknown {
        reasons.insert("BITEMPORAL_OBSERVATION_RECORD_ORDER_UNRESOLVED".to_owned());
        return FrontierVisibilityAssessment {
            visibility: FrontierVisibility::IndeterminateAtK,
            reason_codes: reasons.into_iter().collect(),
        };
    }
    let observed = compare_qualified_event_times(&event.first_observed_at, decision_frontier);
    let recorded = compare_qualified_event_times(&event.durably_recorded_at, decision_frontier);
    for reason in observed.reason_codes.iter().chain(recorded.reason_codes.iter()) {
        reasons.insert(reason.clone());
    }

    let visibility = match (observed.order, recorded.order) {
        (EventOrder::Before, EventOrder::Before) if !event.event_id.trim().is_empty() => {
            FrontierVisibility::AvailableAtK
        }
        (EventOrder::After, EventOrder::After) => {
            reasons.insert("BITEMPORAL_EVENT_NOT_YET_AVAILABLE_AT_K".to_owned());
            FrontierVisibility::NotAvailableAtK
        }
        (EventOrder::Before, EventOrder::After) | (EventOrder::After, EventOrder::Before) => {
            reasons.insert("BITEMPORAL_OBSERVATION_RECORD_ORDER_CONTRADICTORY".to_owned());
            FrontierVisibility::IndeterminateAtK
        }
        _ => {
            reasons.insert("BITEMPORAL_VISIBILITY_ORDER_UNRESOLVED".to_owned());
            FrontierVisibility::IndeterminateAtK
        }
    };

    FrontierVisibilityAssessment {
        visibility,
        reason_codes: reasons.into_iter().collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptIdentity {
    pub logical_operation_id: String,
    pub attempt_id: String,
    pub retry_of_attempt_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptTerminalReconciliation {
    pub event_id: String,
    /// Exact start event this terminal reconciles.
    pub start_event_id: String,
    pub outcome: ReconciledOperationOutcome,
    pub qualification: GateState,
    /// Relation of the start event to this terminal event.
    pub start_to_terminal_order: EventOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptStopDischargeRelation {
    pub obligation_id: String,
    pub discharge_event_id: String,
    pub attempt_start_event_id: String,
    /// BEFORE means the discharge occurred before this attempt started.
    /// AFTER means the attempt was already started when discharge was assessed.
    pub discharge_to_start_order: EventOrder,
    /// Required only when the attempt started before the discharge.
    pub terminal_event_id_at_discharge: Option<String>,
    /// Relation of that exact terminal event to the exact discharge event.
    pub terminal_to_discharge_order: Option<EventOrder>,
}

/// Canonical attempt summary reconstructed from owner-qualified observations.
/// Vectors preserve cardinality evidence so duplicate starts/terminals are
/// rejected rather than overwritten during map construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationAttemptRecord {
    pub identity: AttemptIdentity,
    pub start_event_ids: Vec<String>,
    pub terminal_reconciliation_event_ids: Vec<String>,
    pub terminal_reconciliation: Option<AttemptTerminalReconciliation>,
    /// For each applicable stop discharge, binds whether this attempt started
    /// after discharge or was terminally reconciled before discharge.
    pub stop_discharge_relations: Vec<AttemptStopDischargeRelation>,
    /// Exact predecessor terminal event to which the retry relation applies.
    pub retry_predecessor_terminal_event_id: Option<String>,
    /// Exact current start event to which the retry relation applies.
    pub retry_predecessor_start_event_id: Option<String>,
    /// Relation of the predecessor terminal to this attempt's start.
    pub retry_predecessor_terminal_to_start_order: Option<EventOrder>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptLifecycleReport {
    pub disposition: EventTraceDisposition,
    pub pending_attempt_ids: Vec<String>,
    pub reason_codes: Vec<String>,
}

/// Validate attempt identity, start/terminal cardinality, and retry lineage.
/// A retry is admissible only when the exact predecessor is qualified PASS,
/// terminal NOT_APPLIED, and ordered BEFORE the retry attempt's start.
pub fn validate_attempt_lifecycle(
    attempts: &[OperationAttemptRecord],
) -> AttemptLifecycleReport {
    let mut reasons = BTreeSet::new();
    let mut critical = false;
    let mut indeterminate = false;
    let mut by_id: BTreeMap<&str, &OperationAttemptRecord> = BTreeMap::new();
    let mut event_ids = BTreeSet::new();
    let mut roots_by_logical_operation = BTreeMap::<String, String>::new();
    let mut retry_children = BTreeMap::<String, String>::new();

    for attempt in attempts {
        let logical_id = attempt.identity.logical_operation_id.trim();
        let attempt_id = attempt.identity.attempt_id.trim();
        if logical_id.is_empty() || attempt_id.is_empty() {
            critical = true;
            reasons.insert("ATTEMPT_IDENTITY_INVALID".to_owned());
        }
        match attempt.identity.retry_of_attempt_id.as_deref() {
            None => {
                if roots_by_logical_operation
                    .insert(logical_id.to_owned(), attempt_id.to_owned())
                    .is_some()
                {
                    critical = true;
                    reasons.insert(format!("MULTIPLE_ROOT_ATTEMPTS_FOR_LOGICAL_OPERATION:{logical_id}"));
                }
            }
            Some(parent_id) => {
                if retry_children
                    .insert(parent_id.to_owned(), attempt_id.to_owned())
                    .is_some()
                {
                    critical = true;
                    reasons.insert(format!("ATTEMPT_RETRY_FORK:{parent_id}"));
                }
            }
        }
        if by_id.insert(attempt.identity.attempt_id.as_str(), attempt).is_some() {
            critical = true;
            reasons.insert(format!("DUPLICATE_ATTEMPT_ID:{}", attempt.identity.attempt_id));
        }
        if attempt.start_event_ids.len() != 1 {
            critical = true;
            reasons.insert(format!("ATTEMPT_START_CARDINALITY_INVALID:{}", attempt.identity.attempt_id));
        }
        for event_id in &attempt.start_event_ids {
            if event_id.trim().is_empty() || !event_ids.insert(event_id.clone()) {
                critical = true;
                reasons.insert(format!("ATTEMPT_EVENT_ID_INVALID_OR_DUPLICATE:{event_id}"));
            }
        }
        if attempt.terminal_reconciliation_event_ids.len() > 1 {
            critical = true;
            reasons.insert(format!("DUPLICATE_TERMINAL_RECONCILIATION:{}", attempt.identity.attempt_id));
        }
        for event_id in &attempt.terminal_reconciliation_event_ids {
            if event_id.trim().is_empty() || !event_ids.insert(event_id.clone()) {
                critical = true;
                reasons.insert(format!("ATTEMPT_EVENT_ID_INVALID_OR_DUPLICATE:{event_id}"));
            }
        }
        match (&attempt.terminal_reconciliation, attempt.terminal_reconciliation_event_ids.as_slice()) {
            (None, []) => {}
            (Some(terminal), [event_id]) if &terminal.event_id == event_id => {
                let exact_start_binding = attempt.start_event_ids.first().is_some_and(|start_id| {
                    start_id == &terminal.start_event_id
                });
                if !exact_start_binding {
                    critical = true;
                    reasons.insert(format!("TERMINAL_RECONCILIATION_START_BINDING_INVALID:{}", attempt.identity.attempt_id));
                }
                if terminal.event_id.trim().is_empty() {
                    critical = true;
                    reasons.insert(format!("TERMINAL_RECONCILIATION_EVENT_ID_INVALID:{}", attempt.identity.attempt_id));
                }
                match terminal.qualification {
                    GateState::Pass => match terminal.start_to_terminal_order {
                        EventOrder::Before => {}
                        EventOrder::After | EventOrder::Inconsistent => {
                            critical = true;
                            reasons.insert(format!("TERMINAL_RECONCILIATION_PRECEDES_OR_CONTRADICTS_START:{}", attempt.identity.attempt_id));
                        }
                        EventOrder::Overlapping | EventOrder::Unknown => {
                            indeterminate = true;
                            reasons.insert(format!("TERMINAL_RECONCILIATION_ORDER_UNRESOLVED:{}", attempt.identity.attempt_id));
                        }
                    },
                    GateState::Fail => {
                        critical = true;
                        reasons.insert(format!("TERMINAL_RECONCILIATION_QUALIFICATION_FAILED:{}", attempt.identity.attempt_id));
                    }
                    GateState::Indeterminate | GateState::NotApplicable => {
                        indeterminate = true;
                        reasons.insert(format!("TERMINAL_RECONCILIATION_QUALIFICATION_UNRESOLVED:{}", attempt.identity.attempt_id));
                    }
                }
            }
            _ => {
                critical = true;
                reasons.insert(format!("TERMINAL_RECONCILIATION_BINDING_INVALID:{}", attempt.identity.attempt_id));
            }
        }
        if attempt.terminal_reconciliation.is_none() {
            indeterminate = true;
            reasons.insert(format!("ATTEMPT_OUTCOME_UNRECONCILED:{}", attempt.identity.attempt_id));
        }
    }

    // Check retry links after constructing the complete identity index so
    // delivery order is irrelevant.
    for attempt in attempts {
        match attempt.identity.retry_of_attempt_id.as_deref() {
            None => {
                if attempt.retry_predecessor_terminal_event_id.is_some()
                    || attempt.retry_predecessor_start_event_id.is_some()
                    || attempt.retry_predecessor_terminal_to_start_order.is_some()
                {
                    critical = true;
                    reasons.insert(format!("ROOT_ATTEMPT_HAS_RETRY_BINDING:{}", attempt.identity.attempt_id));
                }
            }
            Some(predecessor_id) => {
                let Some(predecessor) = by_id.get(predecessor_id).copied() else {
                    critical = true;
                    reasons.insert(format!("RETRY_PREDECESSOR_MISSING:{}", attempt.identity.attempt_id));
                    continue;
                };
                if predecessor.identity.logical_operation_id != attempt.identity.logical_operation_id {
                    critical = true;
                    reasons.insert(format!("RETRY_LOGICAL_OPERATION_MISMATCH:{}", attempt.identity.attempt_id));
                }
                match predecessor.terminal_reconciliation.as_ref() {
                    Some(terminal)
                        if terminal.outcome == ReconciledOperationOutcome::NotApplied
                            && terminal.qualification == GateState::Pass
                            && terminal.start_to_terminal_order == EventOrder::Before =>
                    {
                        let exact_event_binding = attempt
                            .retry_predecessor_terminal_event_id
                            .as_deref()
                            == Some(terminal.event_id.as_str())
                            && attempt.retry_predecessor_start_event_id.as_deref()
                                == attempt.start_event_ids.first().map(String::as_str);
                        if !exact_event_binding {
                            critical = true;
                            reasons.insert(format!("RETRY_EVENT_BINDING_INVALID:{}", attempt.identity.attempt_id));
                        }
                        match attempt.retry_predecessor_terminal_to_start_order {
                            Some(EventOrder::Before) => {}
                            Some(EventOrder::After | EventOrder::Inconsistent) | None => {
                                critical = true;
                                reasons.insert(format!("RETRY_NOT_PROVEN_AFTER_RECONCILIATION:{}", attempt.identity.attempt_id));
                            }
                            Some(EventOrder::Overlapping | EventOrder::Unknown) => {
                                critical = true;
                                reasons.insert(format!("RETRY_ORDER_UNRESOLVED_AFTER_RECONCILIATION:{}", attempt.identity.attempt_id));
                            }
                        }
                    }
                    Some(terminal) if terminal.outcome == ReconciledOperationOutcome::Applied => {
                        critical = true;
                        reasons.insert(format!("RETRY_PREDECESSOR_WAS_APPLIED:{}", attempt.identity.attempt_id));
                    }
                    Some(_) => {
                        critical = true;
                        reasons.insert(format!("RETRY_PREDECESSOR_NOT_QUALIFIED:{}", attempt.identity.attempt_id));
                    }
                    None => {
                        critical = true;
                        reasons.insert(format!("RETRY_BEFORE_RECONCILIATION:{}", attempt.identity.attempt_id));
                    }
                }
            }
        }
    }

    // Detect retry-lineage cycles independently of vector delivery order.
    for attempt in attempts {
        let mut visited = BTreeSet::new();
        let mut current = attempt.identity.attempt_id.as_str();
        while let Some(record) = by_id.get(current).copied() {
            if !visited.insert(current.to_owned()) {
                critical = true;
                reasons.insert(format!("ATTEMPT_RETRY_LINEAGE_CYCLE:{}", attempt.identity.attempt_id));
                break;
            }
            let Some(parent) = record.identity.retry_of_attempt_id.as_deref() else {
                break;
            };
            current = parent;
        }
    }

    let pending_attempt_ids = attempts
        .iter()
        .filter(|attempt| attempt.terminal_reconciliation.is_none())
        .map(|attempt| attempt.identity.attempt_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let disposition = if critical {
        EventTraceDisposition::CriticalFailure
    } else if indeterminate {
        EventTraceDisposition::Indeterminate
    } else {
        EventTraceDisposition::Safe
    };

    AttemptLifecycleReport {
        disposition,
        pending_attempt_ids,
        reason_codes: reasons.into_iter().collect(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptHistoryInventory {
    pub inventory_frontier_event_id: String,
    pub attempt_manifest_sha256: String,
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub scope: StopScope,
    pub identity: GateState,
    pub digest_integrity: GateState,
    pub authenticity: GateState,
    pub scope_qualification: GateState,
    pub completeness: GateState,
}

/// Domain-separated commitment to an attempt set, invariant under delivery order.
pub fn canonical_attempt_manifest_sha256(
    attempts: &[OperationAttemptRecord],
) -> Result<String, serde_json::Error> {
    let mut serialized_records = attempts
        .iter()
        .map(|attempt| {
            let mut canonical = attempt.clone();
            canonical.start_event_ids.sort();
            canonical.terminal_reconciliation_event_ids.sort();
            canonical.stop_discharge_relations.sort_by(|left, right| {
                left.obligation_id
                    .cmp(&right.obligation_id)
                    .then(left.discharge_event_id.cmp(&right.discharge_event_id))
                    .then(left.attempt_start_event_id.cmp(&right.attempt_start_event_id))
            });
            serde_json::to_string(&canonical)
        })
        .collect::<Result<Vec<_>, _>>()?;
    serialized_records.sort();
    let projection = serde_json::to_vec(&(
        "symthaea.engineering-decision.attempt-manifest.v0.2",
        serialized_records,
    ))?;
    let digest = Sha256::digest(&projection);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "logical_operation_ids", rename_all = "snake_case")]
pub enum StopScope {
    Global,
    LogicalOperations(BTreeSet<String>),
}

impl StopScope {
    pub fn is_well_formed(&self) -> bool {
        match self {
            Self::Global => true,
            Self::LogicalOperations(ids) => {
                !ids.is_empty() && ids.iter().all(|id| !id.trim().is_empty())
            }
        }
    }

    pub fn applies_to(&self, logical_operation_id: &str) -> bool {
        match self {
            Self::Global => true,
            Self::LogicalOperations(ids) => ids.contains(logical_operation_id),
        }
    }
}

fn scope_covers(candidate: &StopScope, required: &StopScope) -> bool {
    match (candidate, required) {
        (StopScope::Global, _) => true,
        (StopScope::LogicalOperations(_), StopScope::Global) => false,
        (StopScope::LogicalOperations(candidate), StopScope::LogicalOperations(required)) => {
            required.is_subset(candidate)
        }
    }
}

fn validate_attempt_inventory(
    attempts: &[OperationAttemptRecord],
    inventory: &AttemptHistoryInventory,
    logical_operation_id: &str,
    obligations: &[ScopedStopObligation],
) -> (EventTraceDisposition, Vec<String>) {
    let mut reasons = BTreeSet::new();
    let mut critical = false;
    let mut indeterminate = false;

    if inventory.inventory_frontier_event_id.trim().is_empty()
        || inventory.receipt_ref.trim().is_empty()
        || !is_sha256_hex(&inventory.receipt_sha256)
        || !is_sha256_hex(&inventory.attempt_manifest_sha256)
        || !inventory.scope.is_well_formed()
    {
        critical = true;
        reasons.insert("ATTEMPT_INVENTORY_BINDING_INVALID".to_owned());
    }

    for (name, state) in [
        ("IDENTITY", inventory.identity),
        ("DIGEST_INTEGRITY", inventory.digest_integrity),
        ("AUTHENTICITY", inventory.authenticity),
        ("SCOPE", inventory.scope_qualification),
        ("COMPLETENESS", inventory.completeness),
    ] {
        match state {
            GateState::Pass => {}
            GateState::Fail => {
                critical = true;
                reasons.insert(format!("ATTEMPT_INVENTORY_{name}_FAILED"));
            }
            GateState::Indeterminate | GateState::NotApplicable => {
                indeterminate = true;
                reasons.insert(format!("ATTEMPT_INVENTORY_{name}_UNRESOLVED"));
            }
        }
    }

    match canonical_attempt_manifest_sha256(attempts) {
        Ok(digest) if digest == inventory.attempt_manifest_sha256 => {}
        Ok(_) => {
            critical = true;
            reasons.insert("ATTEMPT_INVENTORY_MANIFEST_DIGEST_MISMATCH".to_owned());
        }
        Err(_) => {
            critical = true;
            reasons.insert("ATTEMPT_INVENTORY_MANIFEST_SERIALIZATION_FAILED".to_owned());
        }
    }

    let current_scope = StopScope::LogicalOperations(
        [logical_operation_id.to_owned()].into_iter().collect(),
    );
    if !scope_covers(&inventory.scope, &current_scope) {
        indeterminate = true;
        reasons.insert("ATTEMPT_INVENTORY_SCOPE_DOES_NOT_COVER_START".to_owned());
    }
    for obligation in obligations {
        if obligation.scope.is_well_formed()
            && obligation.scope.applies_to(logical_operation_id)
            && !scope_covers(&inventory.scope, &obligation.scope)
        {
            indeterminate = true;
            reasons.insert(format!(
                "ATTEMPT_INVENTORY_SCOPE_DOES_NOT_COVER_OBLIGATION:{}",
                obligation.obligation_id
            ));
        }
    }
    if attempts.iter().any(|attempt| {
        !inventory.scope.applies_to(&attempt.identity.logical_operation_id)
    }) {
        critical = true;
        reasons.insert("ATTEMPT_INVENTORY_CONTAINS_OUT_OF_SCOPE_RECORD".to_owned());
    }

    let disposition = if critical {
        EventTraceDisposition::CriticalFailure
    } else if indeterminate {
        EventTraceDisposition::Indeterminate
    } else {
        EventTraceDisposition::Safe
    };
    (disposition, reasons.into_iter().collect())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopDischargeReceipt {
    pub event_id: String,
    pub obligation_id: String,
    pub trigger_event_id: String,
    pub policy_revision: String,
    pub scope: StopScope,
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub qualification: GateState,
    /// Owner-native qualification that the complete in-flight attempt inventory
    /// in this obligation's scope was reconciled at discharge time.
    pub in_flight_reconciliation: GateState,
    /// Exact inventory frontier and canonical attempt-manifest digest qualified
    /// by the owner receipt at discharge time.
    pub in_flight_inventory_frontier_event_id: String,
    pub in_flight_attempt_manifest_sha256: String,
    /// Relation of the trigger event to the discharge event.
    pub trigger_to_discharge_order: EventOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopedStopObligation {
    pub obligation_id: String,
    pub trigger_event_id: String,
    pub policy_revision: String,
    pub scope: StopScope,
    pub discharge: Option<StopDischargeReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartGateReport {
    /// Eligibility only; this report never grants execution or effect authority.
    pub start_permitted_by_stop_contract: bool,
    pub disposition: EventTraceDisposition,
    pub applicable_obligation_ids: Vec<String>,
    pub reason_codes: Vec<String>,
}

fn attempt_is_reconciled(attempt: &OperationAttemptRecord) -> bool {
    attempt.terminal_reconciliation.as_ref().is_some_and(|terminal| {
        terminal.qualification == GateState::Pass
            && terminal.start_to_terminal_order == EventOrder::Before
            && attempt.start_event_ids.first().is_some_and(|start_id| {
                start_id == &terminal.start_event_id
            })
    })
}

fn validate_attempt_stop_discharge_relations(
    attempts: &[OperationAttemptRecord],
    obligations: &[ScopedStopObligation],
) -> (EventTraceDisposition, Vec<String>) {
    let mut reasons = BTreeSet::new();
    let mut critical = false;
    let mut indeterminate = false;

    for attempt in attempts {
        let mut seen_obligation_ids = BTreeSet::new();

        for relation in &attempt.stop_discharge_relations {
            if !seen_obligation_ids.insert(relation.obligation_id.clone()) {
                critical = true;
                reasons.insert(format!(
                    "DUPLICATE_ATTEMPT_STOP_RELATION:{}:{}",
                    attempt.identity.attempt_id, relation.obligation_id
                ));
            }
            let known_applicable = obligations.iter().any(|obligation| {
                obligation.obligation_id == relation.obligation_id
                    && obligation.scope.applies_to(&attempt.identity.logical_operation_id)
            });
            if !known_applicable {
                critical = true;
                reasons.insert(format!(
                    "ATTEMPT_STOP_RELATION_OUT_OF_SCOPE:{}:{}",
                    attempt.identity.attempt_id, relation.obligation_id
                ));
            }
        }

        for obligation in obligations.iter().filter(|obligation| {
            obligation.scope.applies_to(&attempt.identity.logical_operation_id)
        }) {
            let Some(discharge) = obligation.discharge.as_ref() else {
                continue;
            };
            let matching = attempt.stop_discharge_relations.iter().find(|relation| {
                relation.obligation_id == obligation.obligation_id
            });
            let Some(relation) = matching else {
                critical = true;
                reasons.insert(format!(
                    "ATTEMPT_STOP_DISCHARGE_RELATION_MISSING:{}:{}",
                    attempt.identity.attempt_id, obligation.obligation_id
                ));
                continue;
            };

            let exact_start_id = attempt.start_event_ids.first().map(String::as_str);
            if relation.discharge_event_id != discharge.event_id
                || relation.attempt_start_event_id.as_str() != exact_start_id.unwrap_or("")
            {
                critical = true;
                reasons.insert(format!(
                    "ATTEMPT_STOP_DISCHARGE_EVENT_BINDING_INVALID:{}:{}",
                    attempt.identity.attempt_id, obligation.obligation_id
                ));
                continue;
            }

            match relation.discharge_to_start_order {
                EventOrder::Before => {
                    if relation.terminal_event_id_at_discharge.is_some()
                        || relation.terminal_to_discharge_order.is_some()
                    {
                        critical = true;
                        reasons.insert(format!(
                            "POST_DISCHARGE_ATTEMPT_HAS_PRE_DISCHARGE_TERMINAL:{}:{}",
                            attempt.identity.attempt_id, obligation.obligation_id
                        ));
                    }
                }
                EventOrder::After => {
                    let Some(terminal) = attempt.terminal_reconciliation.as_ref() else {
                        critical = true;
                        reasons.insert(format!(
                            "ATTEMPT_UNRESOLVED_AT_STOP_DISCHARGE:{}:{}",
                            attempt.identity.attempt_id, obligation.obligation_id
                        ));
                        continue;
                    };
                    if relation.terminal_event_id_at_discharge.as_deref()
                        != Some(terminal.event_id.as_str())
                        || terminal.start_event_id != relation.attempt_start_event_id
                        || terminal.qualification != GateState::Pass
                        || terminal.start_to_terminal_order != EventOrder::Before
                    {
                        critical = true;
                        reasons.insert(format!(
                            "ATTEMPT_TERMINAL_NOT_BOUND_AT_STOP_DISCHARGE:{}:{}",
                            attempt.identity.attempt_id, obligation.obligation_id
                        ));
                        continue;
                    }
                    match relation.terminal_to_discharge_order {
                        Some(EventOrder::Before) => {}
                        Some(EventOrder::After | EventOrder::Inconsistent) | None => {
                            critical = true;
                            reasons.insert(format!(
                                "ATTEMPT_TERMINAL_NOT_BEFORE_STOP_DISCHARGE:{}:{}",
                                attempt.identity.attempt_id, obligation.obligation_id
                            ));
                        }
                        Some(EventOrder::Overlapping | EventOrder::Unknown) => {
                            indeterminate = true;
                            reasons.insert(format!(
                                "ATTEMPT_TERMINAL_ORDER_UNRESOLVED_AT_STOP_DISCHARGE:{}:{}",
                                attempt.identity.attempt_id, obligation.obligation_id
                            ));
                        }
                    }
                }
                EventOrder::Overlapping | EventOrder::Unknown => {
                    indeterminate = true;
                    reasons.insert(format!(
                        "ATTEMPT_STOP_DISCHARGE_START_ORDER_UNRESOLVED:{}:{}",
                        attempt.identity.attempt_id, obligation.obligation_id
                    ));
                }
                EventOrder::Inconsistent => {
                    critical = true;
                    reasons.insert(format!(
                        "ATTEMPT_STOP_DISCHARGE_START_ORDER_INCONSISTENT:{}:{}",
                        attempt.identity.attempt_id, obligation.obligation_id
                    ));
                }
            }
        }
    }

    let disposition = if critical {
        EventTraceDisposition::CriticalFailure
    } else if indeterminate {
        EventTraceDisposition::Indeterminate
    } else {
        EventTraceDisposition::Safe
    };
    (disposition, reasons.into_iter().collect())
}

/// Assess whether a new attempt is eligible under every applicable stop
/// obligation and the known prior-attempt history. A PASS result is not
/// authority; it means only that supplied obligations do not prohibit starting.
pub fn assess_start_under_stop_obligations(
    logical_operation_id: &str,
    prior_attempts: &[OperationAttemptRecord],
    inventory: &AttemptHistoryInventory,
    obligations: &[ScopedStopObligation],
) -> StartGateReport {
    assess_start_under_stop_obligations_with_inventory_cut(
        logical_operation_id, prior_attempts, prior_attempts, inventory, obligations,
    )
}

/// Assess full current history against a separately attested point-in-time
/// inventory cut. The snapshot is hashed against the inventory manifest;
/// current history is checked for lifecycle and stop-relation integrity.
/// This preserves post-discharge append-only deltas without weakening lineage.
pub fn assess_start_under_stop_obligations_with_inventory_cut(
    logical_operation_id: &str,
    current_attempts: &[OperationAttemptRecord],
    inventory_snapshot_attempts: &[OperationAttemptRecord],
    inventory: &AttemptHistoryInventory,
    obligations: &[ScopedStopObligation],
) -> StartGateReport {
    let mut reasons = BTreeSet::new();
    let mut critical = false;
    let mut indeterminate = false;
    let mut applicable = BTreeSet::new();

    if logical_operation_id.trim().is_empty() {
        critical = true;
        reasons.insert("START_LOGICAL_OPERATION_ID_INVALID".to_owned());
    }

    // Validate complete-history structural integrity across the supplied snapshot.
    // A critical identity/lineage defect anywhere in the qualified inventory is
    // not hidden by filtering to one logical operation.
    let full_history_report = validate_attempt_lifecycle(current_attempts);
    if full_history_report.disposition == EventTraceDisposition::CriticalFailure {
        critical = true;
        reasons.insert("ATTEMPT_INVENTORY_CONTAINS_CRITICAL_HISTORY_DEFECT".to_owned());
    }

    // Retry/active-attempt admissibility is local to the requested logical
    // operation. A pending sibling operation inside a discharged global scope
    // is not, by itself, a reason to serialize otherwise independent work.
    let relevant_prior_attempts = current_attempts
        .iter()
        .filter(|attempt| attempt.identity.logical_operation_id == logical_operation_id)
        .cloned()
        .collect::<Vec<_>>();
    let (inventory_disposition, inventory_reasons) = validate_attempt_inventory(
        inventory_snapshot_attempts,
        inventory,
        logical_operation_id,
        obligations,
    );
    reasons.extend(inventory_reasons);
    match inventory_disposition {
        EventTraceDisposition::CriticalFailure => critical = true,
        EventTraceDisposition::Indeterminate => indeterminate = true,
        EventTraceDisposition::Safe => {}
    }

    let attempts_report = validate_attempt_lifecycle(&relevant_prior_attempts);
    match attempts_report.disposition {
        EventTraceDisposition::CriticalFailure => {
            critical = true;
            reasons.insert("PRIOR_ATTEMPT_HISTORY_CRITICAL_FAILURE".to_owned());
        }
        EventTraceDisposition::Indeterminate => {
            indeterminate = true;
            reasons.insert("PRIOR_ATTEMPT_HISTORY_INDETERMINATE".to_owned());
        }
        EventTraceDisposition::Safe => {}
    }
    for attempt in relevant_prior_attempts.iter().filter(|attempt| {
        attempt.identity.logical_operation_id == logical_operation_id
    }) {
        if !attempt_is_reconciled(attempt) {
            indeterminate = true;
            reasons.insert(format!(
                "PRIOR_ATTEMPT_NOT_RECONCILED:{}",
                attempt.identity.attempt_id
            ));
        }
    }

    let mut seen_obligation_ids = BTreeSet::new();
    for obligation in obligations {
        if !obligation.scope.is_well_formed() {
            critical = true;
            reasons.insert("STOP_OBLIGATION_SCOPE_INVALID".to_owned());
            continue;
        }
        if !obligation.scope.applies_to(logical_operation_id) {
            continue;
        }
        if obligation.obligation_id.trim().is_empty()
            || obligation.trigger_event_id.trim().is_empty()
            || obligation.policy_revision.trim().is_empty()
            || !seen_obligation_ids.insert(obligation.obligation_id.clone())
        {
            critical = true;
            reasons.insert("STOP_OBLIGATION_ID_OR_POLICY_INVALID".to_owned());
            continue;
        }
        applicable.insert(obligation.obligation_id.clone());

        let Some(discharge) = obligation.discharge.as_ref() else {
            indeterminate = true;
            reasons.insert(format!(
                "STOP_OBLIGATION_NOT_DISCHARGED:{}",
                obligation.obligation_id
            ));
            continue;
        };

        let bindings_valid = !discharge.event_id.trim().is_empty()
            && discharge.obligation_id == obligation.obligation_id
            && discharge.trigger_event_id == obligation.trigger_event_id
            && discharge.policy_revision == obligation.policy_revision
            && discharge.scope == obligation.scope
            && !discharge.receipt_ref.trim().is_empty()
            && is_sha256_hex(&discharge.receipt_sha256)
            && !discharge.in_flight_inventory_frontier_event_id.trim().is_empty()
            && is_sha256_hex(&discharge.in_flight_attempt_manifest_sha256);
        if !bindings_valid {
            critical = true;
            reasons.insert(format!(
                "STOP_DISCHARGE_RECEIPT_OR_INVENTORY_BINDING_INVALID:{}",
                obligation.obligation_id
            ));
        }

        match discharge.qualification {
            GateState::Pass => {}
            GateState::Fail => {
                critical = true;
                reasons.insert(format!(
                    "STOP_DISCHARGE_QUALIFICATION_FAILED:{}",
                    obligation.obligation_id
                ));
            }
            GateState::Indeterminate | GateState::NotApplicable => {
                indeterminate = true;
                reasons.insert(format!(
                    "STOP_DISCHARGE_QUALIFICATION_UNRESOLVED:{}",
                    obligation.obligation_id
                ));
            }
        }

        match discharge.trigger_to_discharge_order {
            EventOrder::Before => {}
            EventOrder::After | EventOrder::Inconsistent => {
                critical = true;
                reasons.insert(format!(
                    "STOP_DISCHARGE_NOT_AFTER_TRIGGER:{}",
                    obligation.obligation_id
                ));
            }
            EventOrder::Overlapping | EventOrder::Unknown => {
                indeterminate = true;
                reasons.insert(format!(
                    "STOP_DISCHARGE_ORDER_UNRESOLVED:{}",
                    obligation.obligation_id
                ));
            }
        }

        match discharge.in_flight_reconciliation {
            GateState::Pass => {}
            GateState::Fail => {
                critical = true;
                reasons.insert(format!(
                    "STOP_DISCHARGE_IN_FLIGHT_RECONCILIATION_FAILED:{}",
                    obligation.obligation_id
                ));
            }
            GateState::Indeterminate | GateState::NotApplicable => {
                indeterminate = true;
                reasons.insert(format!(
                    "STOP_DISCHARGE_IN_FLIGHT_RECONCILIATION_UNRESOLVED:{}",
                    obligation.obligation_id
                ));
            }
        }
    }

    let (relation_disposition, relation_reasons) =
        validate_attempt_stop_discharge_relations(current_attempts, obligations);
    reasons.extend(relation_reasons);
    match relation_disposition {
        EventTraceDisposition::CriticalFailure => critical = true,
        EventTraceDisposition::Indeterminate => indeterminate = true,
        EventTraceDisposition::Safe => {}
    }

    let disposition = if critical {
        EventTraceDisposition::CriticalFailure
    } else if indeterminate {
        EventTraceDisposition::Indeterminate
    } else {
        EventTraceDisposition::Safe
    };

    StartGateReport {
        start_permitted_by_stop_contract: disposition == EventTraceDisposition::Safe,
        disposition,
        applicable_obligation_ids: applicable.into_iter().collect(),
        reason_codes: reasons.into_iter().collect(),
    }

}
#[cfg(test)]
mod tests {
    use super::*;

    fn qualified_time(domain: &str, basis: &str, earliest: i64, latest: i64) -> QualifiedEventTime {
        QualifiedEventTime {
            source_clock_domain_id: domain.to_owned(),
            time_basis: TimeBasisDescriptor {
                basis_id: basis.to_owned(),
                epoch: "unix-epoch-1970".to_owned(),
                scale: "TAI".to_owned(),
                unit: EventTimeUnit::Nanoseconds,
            },
            mapping_receipt_ref: format!("synthetic://clock-map/{domain}"),
            mapping_receipt_sha256: "a".repeat(64),
            mapping_qualification: GateState::Pass,
            mapping_valid_from_ns: 0,
            mapping_valid_through_ns: 10_000,
            max_interval_width_ns: 1_000,
            interval: EventTimeIntervalNs {
                earliest_ns: earliest,
                latest_ns: latest,
            },
        }
    }

    fn terminal(
        event_id: &str,
        start_event_id: &str,
        outcome: ReconciledOperationOutcome,
    ) -> AttemptTerminalReconciliation {
        AttemptTerminalReconciliation {
            event_id: event_id.to_owned(),
            start_event_id: start_event_id.to_owned(),
            outcome,
            qualification: GateState::Pass,
            start_to_terminal_order: EventOrder::Before,
        }
    }

    fn attempt(
        logical: &str,
        id: &str,
        retry_of: Option<&str>,
        start_id: &str,
        terminal_value: Option<AttemptTerminalReconciliation>,
        retry_order: Option<EventOrder>,
        retry_terminal_event_id: Option<&str>,
    ) -> OperationAttemptRecord {
        OperationAttemptRecord {
            identity: AttemptIdentity {
                logical_operation_id: logical.to_owned(),
                attempt_id: id.to_owned(),
                retry_of_attempt_id: retry_of.map(str::to_owned),
            },
            start_event_ids: vec![start_id.to_owned()],
            terminal_reconciliation_event_ids: terminal_value
                .as_ref()
                .map(|value| vec![value.event_id.clone()])
                .unwrap_or_default(),
            terminal_reconciliation: terminal_value,
            stop_discharge_relations: Vec::new(),
            retry_predecessor_terminal_event_id: retry_terminal_event_id.map(str::to_owned),
            retry_predecessor_start_event_id: retry_of.map(|_| start_id.to_owned()),
            retry_predecessor_terminal_to_start_order: retry_order,
        }
    }

    fn post_discharge_relation(
        obligation: &ScopedStopObligation,
        attempt_start_event_id: &str,
    ) -> AttemptStopDischargeRelation {
        AttemptStopDischargeRelation {
            obligation_id: obligation.obligation_id.clone(),
            discharge_event_id: obligation.discharge.as_ref().expect("discharged").event_id.clone(),
            attempt_start_event_id: attempt_start_event_id.to_owned(),
            discharge_to_start_order: EventOrder::Before,
            terminal_event_id_at_discharge: None,
            terminal_to_discharge_order: None,
        }
    }

    fn attempt_inventory(
        attempts: &[OperationAttemptRecord],
        scope: StopScope,
        completeness: GateState,
    ) -> AttemptHistoryInventory {
        AttemptHistoryInventory {
            inventory_frontier_event_id: "inventory-frontier-k".to_owned(),
            attempt_manifest_sha256: canonical_attempt_manifest_sha256(attempts).expect("attempt digest"),
            receipt_ref: "synthetic://attempt-inventory/receipt".to_owned(),
            receipt_sha256: "d".repeat(64),
            scope,
            identity: GateState::Pass,
            digest_integrity: GateState::Pass,
            authenticity: GateState::Pass,
            scope_qualification: GateState::Pass,
            completeness,
        }
    }

    fn scoped_obligation(id: &str, logical: &str) -> ScopedStopObligation {
        let scope = StopScope::LogicalOperations([logical.to_owned()].into_iter().collect());
        let trigger = format!("trigger-{id}");
        ScopedStopObligation {
            obligation_id: id.to_owned(),
            trigger_event_id: trigger.clone(),
            policy_revision: "policy-v7".to_owned(),
            scope: scope.clone(),
            discharge: Some(StopDischargeReceipt {
                event_id: format!("discharge-{id}"),
                obligation_id: id.to_owned(),
                trigger_event_id: trigger,
                policy_revision: "policy-v7".to_owned(),
                scope,
                receipt_ref: format!("synthetic://discharge/{id}"),
                receipt_sha256: "b".repeat(64),
                qualification: GateState::Pass,
                in_flight_reconciliation: GateState::Pass,
                in_flight_inventory_frontier_event_id: format!("inventory-frontier-{id}"),
                in_flight_attempt_manifest_sha256: "c".repeat(64),
                trigger_to_discharge_order: EventOrder::Before,
            }),
        }
    }

    #[test]
    fn inventory_cut_and_current_history_allow_a_post_discharge_pending_sibling() {
        let obligation = ScopedStopObligation {
            obligation_id: "global-cut".to_owned(),
            trigger_event_id: "trigger-global-cut".to_owned(),
            policy_revision: "policy-v1".to_owned(),
            scope: StopScope::Global,
            discharge: Some(StopDischargeReceipt {
                event_id: "discharge-global-cut".to_owned(),
                obligation_id: "global-cut".to_owned(),
                trigger_event_id: "trigger-global-cut".to_owned(),
                policy_revision: "policy-v1".to_owned(),
                scope: StopScope::Global,
                receipt_ref: "synthetic://global-cut/discharge".to_owned(),
                receipt_sha256: "b".repeat(64),
                qualification: GateState::Pass,
                in_flight_reconciliation: GateState::Pass,
                in_flight_inventory_frontier_event_id: "frontier-global-cut".to_owned(),
                in_flight_attempt_manifest_sha256: "c".repeat(64),
                trigger_to_discharge_order: EventOrder::Before,
            }),
        };
        let mut snapshot_attempt = attempt(
            "robot-1", "pre-cut-attempt", None, "pre-cut-start",
            Some(terminal("pre-cut-terminal", "pre-cut-start", ReconciledOperationOutcome::Applied)),
            None, None,
        );
        snapshot_attempt.stop_discharge_relations.push(AttemptStopDischargeRelation {
            obligation_id: "global-cut".to_owned(),
            discharge_event_id: "discharge-global-cut".to_owned(),
            attempt_start_event_id: "pre-cut-start".to_owned(),
            discharge_to_start_order: EventOrder::After,
            terminal_event_id_at_discharge: Some("pre-cut-terminal".to_owned()),
            terminal_to_discharge_order: Some(EventOrder::Before),
        });
        let snapshot = vec![snapshot_attempt.clone()];
        let inventory = attempt_inventory(&snapshot, StopScope::Global, GateState::Pass);
        let mut post_discharge_sibling = attempt(
            "robot-2", "post-discharge-sibling", None, "post-discharge-start", None, None, None,
        );
        post_discharge_sibling.stop_discharge_relations.push(
            post_discharge_relation(&obligation, "post-discharge-start"),
        );
        let current = vec![snapshot_attempt, post_discharge_sibling];
        let report = assess_start_under_stop_obligations_with_inventory_cut(
            "robot-1", &current, &snapshot, &inventory, std::slice::from_ref(&obligation),
        );
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
        assert!(report.start_permitted_by_stop_contract);
        let legacy_report = assess_start_under_stop_obligations(
            "robot-1", &current, &inventory, std::slice::from_ref(&obligation),
        );
        assert_eq!(legacy_report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(!legacy_report.start_permitted_by_stop_contract);
    }

    #[test]
    fn inventory_cut_does_not_hide_a_pending_attempt_for_the_requested_operation() {
        let obligation = scoped_obligation("robot-stop", "robot-1");
        let snapshot_attempt = attempt(
            "robot-1", "snapshot-attempt", None, "snapshot-start",
            Some(terminal("snapshot-terminal", "snapshot-start", ReconciledOperationOutcome::Applied)),
            None, None,
        );
        let snapshot = vec![snapshot_attempt.clone()];
        let inventory = attempt_inventory(
            &snapshot,
            StopScope::LogicalOperations(["robot-1".to_owned()].into_iter().collect()),
            GateState::Pass,
        );
        let pending_current = attempt(
            "robot-1", "post-cut-pending", None, "post-cut-start", None, None, None,
        );
        let current = vec![snapshot_attempt, pending_current];
        let report = assess_start_under_stop_obligations_with_inventory_cut(
            "robot-1", &current, &snapshot, &inventory, std::slice::from_ref(&obligation),
        );
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert!(!report.start_permitted_by_stop_contract);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "PRIOR_ATTEMPT_NOT_RECONCILED:post-cut-pending"
        }));
    }
    #[test]
    fn only_same_qualified_time_basis_can_establish_cross_domain_order() {
        let effect = qualified_time("device-clock", "tai-ns-v1", 100, 110);
        let trigger = qualified_time("sensor-clock", "tai-ns-v1", 200, 210);
        assert_eq!(
            compare_qualified_event_times(&effect, &trigger).order,
            EventOrder::Before
        );

        let unrelated_basis = qualified_time("sensor-clock", "unix-smear-v2", 200, 210);
        assert_eq!(
            compare_qualified_event_times(&effect, &unrelated_basis).order,
            EventOrder::Unknown
        );
    }

    #[test]
    fn time_basis_requires_epoch_and_scale_identity() {
        let first = qualified_time("clock-a", "basis", 10, 20);
        let mut second = qualified_time("clock-b", "basis", 30, 40);
        second.time_basis.epoch = "different-epoch".to_owned();
        assert_eq!(
            compare_qualified_event_times(&first, &second).order,
            EventOrder::Unknown
        );
        second.time_basis.epoch = first.time_basis.epoch.clone();
        second.time_basis.scale = "UTC".to_owned();
        assert_eq!(
            compare_qualified_event_times(&first, &second).order,
            EventOrder::Unknown
        );
    }

    #[test]
    fn touching_or_overlapping_intervals_never_manufacture_strict_order() {
        let first = qualified_time("clock-a", "basis", 10, 20);
        let touching = qualified_time("clock-b", "basis", 20, 30);
        let overlap = qualified_time("clock-c", "basis", 19, 30);
        assert_eq!(
            compare_qualified_event_times(&first, &touching).order,
            EventOrder::Overlapping
        );
        assert_eq!(
            compare_qualified_event_times(&first, &overlap).order,
            EventOrder::Overlapping
        );
    }

    #[test]
    fn unqualified_or_expired_clock_mapping_is_never_used_as_order_evidence() {
        let mut unqualified = qualified_time("clock-a", "basis", 10, 20);
        unqualified.mapping_qualification = GateState::Indeterminate;
        assert_eq!(
            compare_qualified_event_times(&unqualified, &qualified_time("clock-b", "basis", 30, 40)).order,
            EventOrder::Unknown
        );

        let mut expired = qualified_time("clock-a", "basis", 10, 20);
        expired.mapping_valid_through_ns = 15;
        assert_eq!(
            compare_qualified_event_times(&expired, &qualified_time("clock-b", "basis", 30, 40)).order,
            EventOrder::Inconsistent
        );
    }

    #[test]
    fn late_arriving_event_is_not_available_at_historical_decision_frontier() {
        let frontier = qualified_time("decision-clock", "basis", 500, 500);
        let event = BitemporalEventEnvelope {
            event_id: "late-event".to_owned(),
            occurred_at: Some(qualified_time("device-clock", "basis", 100, 110)),
            first_observed_at: qualified_time("collector", "basis", 510, 515),
            durably_recorded_at: qualified_time("store", "basis", 520, 525),
            capture_qualification: GateState::Pass,
            supersedes_event_id: None,
        };
        let assessment = assess_visibility_at_frontier(&event, &frontier);
        assert_eq!(assessment.visibility, FrontierVisibility::NotAvailableAtK);
    }

    #[test]
    fn record_time_cannot_precede_first_observation() {
        let frontier = qualified_time("decision-clock", "basis", 500, 500);
        let event = BitemporalEventEnvelope {
            event_id: "contradictory-ingest".to_owned(),
            occurred_at: None,
            first_observed_at: qualified_time("collector", "basis", 430, 440),
            durably_recorded_at: qualified_time("store", "basis", 400, 410),
            capture_qualification: GateState::Pass,
            supersedes_event_id: None,
        };
        let assessment = assess_visibility_at_frontier(&event, &frontier);
        assert_eq!(assessment.visibility, FrontierVisibility::IndeterminateAtK);
        assert!(assessment.reason_codes.iter().any(|reason| reason == "BITEMPORAL_RECORD_PRECEDES_OBSERVATION"));
    }

    #[test]
    fn observed_and_recorded_before_frontier_are_available_at_k() {
        let frontier = qualified_time("decision-clock", "basis", 500, 500);
        let event = BitemporalEventEnvelope {
            event_id: "known-event".to_owned(),
            occurred_at: Some(qualified_time("device-clock", "basis", 100, 110)),
            first_observed_at: qualified_time("collector", "basis", 400, 410),
            durably_recorded_at: qualified_time("store", "basis", 420, 430),
            capture_qualification: GateState::Pass,
            supersedes_event_id: None,
        };
        assert_eq!(
            assess_visibility_at_frontier(&event, &frontier).visibility,
            FrontierVisibility::AvailableAtK
        );
    }

    #[test]
    fn retry_requires_exact_predecessor_not_applied_terminal_and_proven_order() {
        let first = attempt(
            "valve-1",
            "attempt-1",
            None,
            "start-1",
            Some(terminal("terminal-1", "start-1", ReconciledOperationOutcome::NotApplied)),
            None,
            None,
        );
        let retry = attempt(
            "valve-1",
            "attempt-2",
            Some("attempt-1"),
            "start-2",
            None,
            Some(EventOrder::Before),
            Some("terminal-1"),
        );
        let report = validate_attempt_lifecycle(&[retry.clone(), first.clone()]);
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert!(report.reason_codes.iter().any(|reason| reason == "ATTEMPT_OUTCOME_UNRECONCILED:attempt-2"));

        let mut invalid_retry = retry;
        invalid_retry.retry_predecessor_terminal_to_start_order = Some(EventOrder::Unknown);
        let report = validate_attempt_lifecycle(&[first, invalid_retry]);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| reason == "RETRY_ORDER_UNRESOLVED_AFTER_RECONCILIATION:attempt-2"));
    }

    #[test]
    fn unknown_predecessor_outcome_cannot_be_retried() {
        let first = attempt("pump-1", "attempt-a", None, "start-a", None, None, None);
        let retry = attempt(
            "pump-1",
            "attempt-b",
            Some("attempt-a"),
            "start-b",
            None,
            Some(EventOrder::Before),
            None,
        );
        let report = validate_attempt_lifecycle(&[first, retry]);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| reason == "RETRY_BEFORE_RECONCILIATION:attempt-b"));
    }

    #[test]
    fn terminal_reconciliation_must_bind_the_exact_start_event() {
        let malformed = attempt(
            "motor-1",
            "attempt-1",
            None,
            "start-1",
            Some(terminal(
                "terminal-1",
                "different-start",
                ReconciledOperationOutcome::Applied,
            )),
            None,
            None,
        );
        let report = validate_attempt_lifecycle(&[malformed]);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "TERMINAL_RECONCILIATION_START_BINDING_INVALID:attempt-1"
        }));
    }

    #[test]
    fn duplicate_starts_and_terminal_reconciliations_are_rejected() {
        let mut malformed = attempt(
            "motor-1",
            "attempt-1",
            None,
            "start-1",
            Some(terminal("terminal-1", "start-1", ReconciledOperationOutcome::Applied)),
            None,
            None,
        );
        malformed.start_event_ids.push("start-duplicate".to_owned());
        malformed.terminal_reconciliation_event_ids.push("terminal-duplicate".to_owned());
        let report = validate_attempt_lifecycle(&[malformed]);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| reason == "ATTEMPT_START_CARDINALITY_INVALID:attempt-1"));
        assert!(report.reason_codes.iter().any(|reason| reason == "DUPLICATE_TERMINAL_RECONCILIATION:attempt-1"));
    }

    #[test]
    fn overlapping_stop_obligations_are_independently_required() {
        let mut attempts = vec![attempt(
            "robot-7",
            "attempt-1",
            None,
            "start-1",
            Some(terminal("terminal-1", "start-1", ReconciledOperationOutcome::NotApplied)),
            None,
            None,
        )];
        let first = scoped_obligation("obligation-a", "robot-7");
        let mut second = scoped_obligation("obligation-b", "robot-7");
        second.discharge = None;
        attempts[0].stop_discharge_relations.push(post_discharge_relation(&first, "start-1"));

        let inventory = attempt_inventory(
            &attempts,
            StopScope::LogicalOperations(["robot-7".to_owned()].into_iter().collect()),
            GateState::Pass,
        );
        let report = assess_start_under_stop_obligations("robot-7", &attempts, &inventory, &[first.clone(), second]);
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert_eq!(report.applicable_obligation_ids, vec!["obligation-a".to_owned(), "obligation-b".to_owned()]);
    }

    #[test]
    fn stop_discharge_requires_owner_qualification_of_the_in_flight_inventory() {
        let mut obligation = scoped_obligation("obligation-a", "robot-7");
        obligation.discharge.as_mut().unwrap().in_flight_reconciliation = GateState::Fail;
        let inventory = attempt_inventory(
            &[],
            StopScope::LogicalOperations(["robot-7".to_owned()].into_iter().collect()),
            GateState::Pass,
        );
        let report = assess_start_under_stop_obligations("robot-7", &[], &inventory, &[obligation]);
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "STOP_DISCHARGE_IN_FLIGHT_RECONCILIATION_FAILED:obligation-a"
        }));
    }

    #[test]
    fn missing_attempt_inventory_completeness_never_treats_absence_as_safe() {
        let obligation = scoped_obligation("obligation-a", "robot-7");
        let inventory = attempt_inventory(
            &[],
            StopScope::LogicalOperations(["robot-7".to_owned()].into_iter().collect()),
            GateState::Indeterminate,
        );
        let report = assess_start_under_stop_obligations(
            "robot-7",
            &[],
            &inventory,
            &[obligation],
        );
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "ATTEMPT_INVENTORY_COMPLETENESS_UNRESOLVED"
        }));
    }

    #[test]
    fn attempt_manifest_digest_is_invariant_under_record_delivery_order() {
        let attempts = vec![
            attempt(
                "pump-1",
                "b",
                None,
                "start-b",
                Some(terminal("term-b", "start-b", ReconciledOperationOutcome::NotApplied)),
                None,
                None,
            ),
            attempt(
                "pump-2",
                "a",
                None,
                "start-a",
                Some(terminal("term-a", "start-a", ReconciledOperationOutcome::NotApplied)),
                None,
                None,
            ),
        ];
        let mut reversed = attempts.clone();
        reversed.reverse();
        assert_eq!(
            canonical_attempt_manifest_sha256(&attempts).unwrap(),
            canonical_attempt_manifest_sha256(&reversed).unwrap()
        );
    }

    #[test]
    fn pre_discharge_unresolved_attempt_cannot_be_hidden_by_global_discharge() {
        let scope = StopScope::Global;
        let trigger = "trigger-global".to_owned();
        let obligation = ScopedStopObligation {
            obligation_id: "global-obligation".to_owned(),
            trigger_event_id: trigger.clone(),
            policy_revision: "policy-v1".to_owned(),
            scope: scope.clone(),
            discharge: Some(StopDischargeReceipt {
                event_id: "discharge-global".to_owned(),
                obligation_id: "global-obligation".to_owned(),
                trigger_event_id: trigger,
                policy_revision: "policy-v1".to_owned(),
                scope,
                receipt_ref: "synthetic://global-discharge".to_owned(),
                receipt_sha256: "b".repeat(64),
                qualification: GateState::Pass,
                in_flight_reconciliation: GateState::Pass,
                in_flight_inventory_frontier_event_id: "inventory-at-discharge".to_owned(),
                in_flight_attempt_manifest_sha256: "c".repeat(64),
                trigger_to_discharge_order: EventOrder::Before,
            }),
        };
        let mut attempt = attempt(
            "robot-a",
            "attempt-a",
            None,
            "start-a",
            None,
            None,
            None,
        );
        attempt.stop_discharge_relations.push(AttemptStopDischargeRelation {
            obligation_id: "global-obligation".to_owned(),
            discharge_event_id: "discharge-global".to_owned(),
            attempt_start_event_id: "start-a".to_owned(),
            discharge_to_start_order: EventOrder::After,
            terminal_event_id_at_discharge: None,
            terminal_to_discharge_order: None,
        });
        let attempts = vec![attempt];
        let inventory = attempt_inventory(&attempts, StopScope::Global, GateState::Pass);

        let report = assess_start_under_stop_obligations(
            "robot-b",
            &attempts,
            &inventory,
            &[obligation],
        );
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "ATTEMPT_UNRESOLVED_AT_STOP_DISCHARGE:attempt-a:global-obligation"
        }));
    }

    #[test]
    fn pre_discharge_terminal_reconciliation_allows_independent_post_discharge_start() {
        let scope = StopScope::Global;
        let trigger = "trigger-global".to_owned();
        let obligation = ScopedStopObligation {
            obligation_id: "global-obligation".to_owned(),
            trigger_event_id: trigger.clone(),
            policy_revision: "policy-v1".to_owned(),
            scope: scope.clone(),
            discharge: Some(StopDischargeReceipt {
                event_id: "discharge-global".to_owned(),
                obligation_id: "global-obligation".to_owned(),
                trigger_event_id: trigger,
                policy_revision: "policy-v1".to_owned(),
                scope,
                receipt_ref: "synthetic://global-discharge".to_owned(),
                receipt_sha256: "b".repeat(64),
                qualification: GateState::Pass,
                in_flight_reconciliation: GateState::Pass,
                in_flight_inventory_frontier_event_id: "inventory-at-discharge".to_owned(),
                in_flight_attempt_manifest_sha256: "c".repeat(64),
                trigger_to_discharge_order: EventOrder::Before,
            }),
        };
        let mut attempt = attempt(
            "robot-a",
            "attempt-a",
            None,
            "start-a",
            Some(terminal("terminal-a", "start-a", ReconciledOperationOutcome::Applied)),
            None,
            None,
        );
        attempt.stop_discharge_relations.push(AttemptStopDischargeRelation {
            obligation_id: "global-obligation".to_owned(),
            discharge_event_id: "discharge-global".to_owned(),
            attempt_start_event_id: "start-a".to_owned(),
            discharge_to_start_order: EventOrder::After,
            terminal_event_id_at_discharge: Some("terminal-a".to_owned()),
            terminal_to_discharge_order: Some(EventOrder::Before),
        });
        let attempts = vec![attempt];
        let inventory = attempt_inventory(&attempts, StopScope::Global, GateState::Pass);

        let report = assess_start_under_stop_obligations(
            "robot-b",
            &attempts,
            &inventory,
            &[obligation],
        );
        assert!(report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
    }

    #[test]
    fn stop_relation_must_bind_the_exact_discharge_and_attempt_start() {
        let obligation = scoped_obligation("obligation-a", "robot-a");
        let mut attempt = attempt(
            "robot-a",
            "attempt-a",
            None,
            "start-a",
            None,
            None,
            None,
        );
        let mut relation = post_discharge_relation(&obligation, "start-a");
        relation.discharge_event_id = "substituted-discharge".to_owned();
        attempt.stop_discharge_relations.push(relation);
        let attempts = vec![attempt];
        let inventory = attempt_inventory(
            &attempts,
            StopScope::LogicalOperations(["robot-a".to_owned()].into_iter().collect()),
            GateState::Pass,
        );
        let report = assess_start_under_stop_obligations(
            "robot-a",
            &attempts,
            &inventory,
            &[obligation],
        );
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "ATTEMPT_STOP_DISCHARGE_EVENT_BINDING_INVALID:attempt-a:obligation-a"
        }));
    }

    #[test]
    fn attempt_manifest_digest_is_invariant_under_stop_relation_delivery_order() {
        let obligation_a = scoped_obligation("obligation-a", "robot-a");
        let obligation_b = scoped_obligation("obligation-b", "robot-a");
        let mut attempt = attempt(
            "robot-a",
            "attempt-a",
            None,
            "start-a",
            None,
            None,
            None,
        );
        attempt.stop_discharge_relations = vec![
            post_discharge_relation(&obligation_a, "start-a"),
            post_discharge_relation(&obligation_b, "start-a"),
        ];
        let forward = canonical_attempt_manifest_sha256(&[attempt.clone()]).unwrap();

        attempt.stop_discharge_relations.reverse();
        let reversed = canonical_attempt_manifest_sha256(&[attempt]).unwrap();
        assert_eq!(forward, reversed);
    }

    #[test]
    fn pending_sibling_attempt_does_not_serialize_a_discharged_global_scope() {
        let sibling = attempt(
            "robot-a",
            "attempt-a",
            None,
            "start-a",
            None,
            None,
            None,
        );
        let attempts = vec![sibling];

        let scope = StopScope::Global;
        let trigger = "global-stop-trigger".to_owned();
        let obligation = ScopedStopObligation {
            obligation_id: "global-obligation".to_owned(),
            trigger_event_id: trigger.clone(),
            policy_revision: "policy-global-v1".to_owned(),
            scope: scope.clone(),
            discharge: Some(StopDischargeReceipt {
                event_id: "global-stop-discharge".to_owned(),
                obligation_id: "global-obligation".to_owned(),
                trigger_event_id: trigger,
                policy_revision: "policy-global-v1".to_owned(),
                scope,
                receipt_ref: "synthetic://global-discharge/receipt".to_owned(),
                receipt_sha256: "b".repeat(64),
                qualification: GateState::Pass,
                in_flight_reconciliation: GateState::Pass,
                in_flight_inventory_frontier_event_id: "global-inventory-frontier".to_owned(),
                in_flight_attempt_manifest_sha256: "c".repeat(64),
                trigger_to_discharge_order: EventOrder::Before,
            }),
        };

        let mut attempts = attempts;
        attempts[0].stop_discharge_relations.push(post_discharge_relation(&obligation, "start-a"));
        let inventory = attempt_inventory(&attempts, StopScope::Global, GateState::Pass);
        let report = assess_start_under_stop_obligations(
            "robot-b",
            &attempts,
            &inventory,
            &[obligation],
        );
        assert!(report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
    }

    #[test]
    fn same_logical_operation_cannot_start_while_its_previous_attempt_is_pending() {
        let pending = attempt(
            "robot-a",
            "attempt-a",
            None,
            "start-a",
            None,
            None,
            None,
        );
        let attempts = vec![pending];
        let inventory = attempt_inventory(
            &attempts,
            StopScope::LogicalOperations(["robot-a".to_owned()].into_iter().collect()),
            GateState::Pass,
        );
        let report = assess_start_under_stop_obligations(
            "robot-a",
            &attempts,
            &inventory,
            &[],
        );
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "PRIOR_ATTEMPT_NOT_RECONCILED:attempt-a"
        }));
    }

    #[test]
    fn unrelated_pending_attempt_does_not_block_an_unrelated_scoped_start() {
        let unrelated = attempt("robot-9", "attempt-pending", None, "start-pending", None, None, None);
        let obligation = scoped_obligation("obligation-a", "robot-7");
        let attempts = vec![unrelated];
        let inventory = attempt_inventory(&attempts, StopScope::Global, GateState::Pass);
        let report = assess_start_under_stop_obligations("robot-8", &attempts, &inventory, &[obligation]);
        assert!(report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
    }

    #[test]
    fn unrelated_operation_is_not_captured_by_a_scoped_stop_obligation() {
        let obligation = scoped_obligation("obligation-a", "robot-7");
        let inventory = attempt_inventory(&[], StopScope::Global, GateState::Pass);
        let report = assess_start_under_stop_obligations("robot-8", &[], &inventory, &[obligation]);
        assert!(report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
        assert!(report.applicable_obligation_ids.is_empty());
    }

    #[test]
    fn discharge_requires_an_exact_in_flight_inventory_commitment() {
        let mut obligation = scoped_obligation("obligation-a", "robot-7");
        obligation.discharge.as_mut().unwrap().in_flight_attempt_manifest_sha256 =
            "not-a-digest".to_owned();
        let inventory = attempt_inventory(
            &[],
            StopScope::LogicalOperations(["robot-7".to_owned()].into_iter().collect()),
            GateState::Pass,
        );
        let report = assess_start_under_stop_obligations("robot-7", &[], &inventory, &[obligation]);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "STOP_DISCHARGE_RECEIPT_OR_INVENTORY_BINDING_INVALID:obligation-a"
        }));
    }

    #[test]
    fn two_discharge_receipts_must_bind_their_own_obligation_and_policy() {
        let mut first = scoped_obligation("obligation-a", "robot-7");
        let second = scoped_obligation("obligation-b", "robot-7");
        first.discharge.as_mut().unwrap().obligation_id = "obligation-b".to_owned();
        let inventory = attempt_inventory(
            &[],
            StopScope::LogicalOperations(["robot-7".to_owned()].into_iter().collect()),
            GateState::Pass,
        );
        let report = assess_start_under_stop_obligations("robot-7", &[], &inventory, &[first, second]);
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "STOP_DISCHARGE_RECEIPT_BINDING_INVALID:obligation-a"
        }));
    }

    #[test]
    fn retry_order_must_bind_exact_predecessor_terminal_and_current_start() {
        let first = attempt(
            "pump-1",
            "attempt-1",
            None,
            "start-1",
            Some(terminal("terminal-1", "start-1", ReconciledOperationOutcome::NotApplied)),
            None,
            None,
        );
        let retry = attempt(
            "pump-1",
            "attempt-2",
            Some("attempt-1"),
            "start-2",
            None,
            Some(EventOrder::Before),
            Some("substituted-terminal"),
        );
        let report = validate_attempt_lifecycle(&[first, retry]);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "RETRY_EVENT_BINDING_INVALID:attempt-2"
        }));
    }

    #[test]
    fn retry_lineage_cannot_fork_or_create_multiple_roots() {
        let root_a = attempt("pump-1", "root-a", None, "start-root-a", None, None, None);
        let root_b = attempt("pump-1", "root-b", None, "start-root-b", None, None, None);
        let report = validate_attempt_lifecycle(&[root_a.clone(), root_b]);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "MULTIPLE_ROOT_ATTEMPTS_FOR_LOGICAL_OPERATION:pump-1"
        }));

        let child_a = attempt(
            "pump-1",
            "child-a",
            Some("root-a"),
            "start-child-a",
            None,
            Some(EventOrder::Before),
            None,
        );
        let child_b = attempt(
            "pump-1",
            "child-b",
            Some("root-a"),
            "start-child-b",
            None,
            Some(EventOrder::Before),
            None,
        );
        let report = validate_attempt_lifecycle(&[root_a, child_a, child_b]);
        assert!(report.reason_codes.iter().any(|reason| reason == "ATTEMPT_RETRY_FORK:root-a"));
    }

    #[test]
    fn attempt_report_is_invariant_under_record_delivery_permutation() {
        let first = attempt(
            "pump-1",
            "a1",
            None,
            "start-a1",
            Some(terminal("term-a1", "start-a1", ReconciledOperationOutcome::NotApplied)),
            None,
            None,
        );
        let second = attempt(
            "pump-1",
            "a2",
            Some("a1"),
            "start-a2",
            None,
            Some(EventOrder::Before),
            Some("term-a1"),
        );
        let forward = vec![first.clone(), second.clone()];
        let reverse = vec![second, first];
        assert_eq!(
            validate_attempt_lifecycle(&forward),
            validate_attempt_lifecycle(&reverse)
        );
    }
}
