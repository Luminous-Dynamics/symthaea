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

use crate::{EventOrder, EventTraceDisposition, GateState, ReconciledOperationOutcome};

pub const EVENT_ORDER_V0_2_SCHEMA_VERSION: &str = "engineering-decision-event-order-v0.2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventTimeIntervalNs {
    pub earliest_unix_ns: i64,
    pub latest_unix_ns: i64,
}

/// A source-clock mapping into one explicitly named common time basis.
///
/// The receipt reference, digest and qualification are owner-supplied. This
/// crate checks their structure and binding shape, not their authenticity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedClockMapping {
    pub source_clock_domain_id: String,
    pub target_time_basis_id: String,
    pub mapping_receipt_ref: String,
    pub mapping_receipt_sha256: String,
    pub qualification: GateState,
    pub valid_from_unix_ns: i64,
    pub valid_through_unix_ns: i64,
    /// Maximum full interval width accepted for this mapping, not a point error.
    pub max_interval_width_ns: u64,
}

/// An event time already mapped into a named common basis, with the mapping
/// receipt identity carried alongside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedEventTime {
    pub source_clock_domain_id: String,
    pub time_basis_id: String,
    pub mapping_receipt_ref: String,
    pub mapping_receipt_sha256: String,
    pub mapping_qualification: GateState,
    pub mapping_valid_from_unix_ns: i64,
    pub mapping_valid_through_unix_ns: i64,
    pub max_interval_width_ns: u64,
    pub interval: EventTimeIntervalNs,
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
        || value.time_basis_id.trim().is_empty()
        || value.mapping_receipt_ref.trim().is_empty()
        || !is_sha256_hex(&value.mapping_receipt_sha256)
    {
        reasons.insert(format!("CLOCK_MAPPING_BINDING_INVALID:{side}"));
        valid = false;
    }
    if value.interval.earliest_unix_ns > value.interval.latest_unix_ns {
        reasons.insert(format!("CLOCK_INTERVAL_REVERSED:{side}"));
        valid = false;
    }
    if value.mapping_valid_from_unix_ns > value.mapping_valid_through_unix_ns {
        reasons.insert(format!("CLOCK_MAPPING_VALIDITY_INTERVAL_INVALID:{side}"));
        valid = false;
    }
    if value.interval.earliest_unix_ns < value.mapping_valid_from_unix_ns
        || value.interval.latest_unix_ns > value.mapping_valid_through_unix_ns
    {
        reasons.insert(format!("CLOCK_MAPPING_OUTSIDE_VALIDITY:{side}"));
        valid = false;
    }
    let width = (value.interval.latest_unix_ns as i128
        - value.interval.earliest_unix_ns as i128) as u128;
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
    if first.time_basis_id != second.time_basis_id {
        reasons.insert("CLOCK_TIME_BASIS_MISMATCH".to_owned());
        return ClockOrderAssessment {
            order: EventOrder::Unknown,
            reason_codes: reasons.into_iter().collect(),
        };
    }

    let order = if first.interval.latest_unix_ns < second.interval.earliest_unix_ns {
        EventOrder::Before
    } else if first.interval.earliest_unix_ns > second.interval.latest_unix_ns {
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
    pub outcome: ReconciledOperationOutcome,
    pub qualification: GateState,
    /// Relation of the start event to this terminal event.
    pub start_to_terminal_order: EventOrder,
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

    for attempt in attempts {
        let logical_id = attempt.identity.logical_operation_id.trim();
        let attempt_id = attempt.identity.attempt_id.trim();
        if logical_id.is_empty() || attempt_id.is_empty() {
            critical = true;
            reasons.insert("ATTEMPT_IDENTITY_INVALID".to_owned());
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
                if attempt.retry_predecessor_terminal_to_start_order.is_some() {
                    critical = true;
                    reasons.insert(format!("ROOT_ATTEMPT_HAS_RETRY_ORDER:{}", attempt.identity.attempt_id));
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
    })
}

/// Assess whether a new attempt is eligible under every applicable stop
/// obligation and the known prior-attempt history. A PASS result is not
/// authority; it means only that supplied obligations do not prohibit starting.
pub fn assess_start_under_stop_obligations(
    logical_operation_id: &str,
    prior_attempts: &[OperationAttemptRecord],
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

    let attempts_report = validate_attempt_lifecycle(prior_attempts);
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
    for attempt in prior_attempts.iter().filter(|attempt| {
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
        if obligation.obligation_id.trim().is_empty()
            || obligation.trigger_event_id.trim().is_empty()
            || obligation.policy_revision.trim().is_empty()
            || !obligation.scope.is_well_formed()
            || !seen_obligation_ids.insert(obligation.obligation_id.clone())
        {
            critical = true;
            reasons.insert("STOP_OBLIGATION_ID_SCOPE_OR_POLICY_INVALID".to_owned());
            continue;
        }
        if !obligation.scope.applies_to(logical_operation_id) {
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
            && is_sha256_hex(&discharge.receipt_sha256);
        if !bindings_valid {
            critical = true;
            reasons.insert(format!(
                "STOP_DISCHARGE_RECEIPT_BINDING_INVALID:{}",
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

        for attempt in prior_attempts.iter().filter(|attempt| {
            obligation.scope.applies_to(&attempt.identity.logical_operation_id)
        }) {
            if !attempt_is_reconciled(attempt) {
                critical = true;
                reasons.insert(format!(
                    "STOP_DISCHARGE_WITH_UNRECONCILED_ATTEMPT:{}:{}",
                    obligation.obligation_id, attempt.identity.attempt_id
                ));
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
            time_basis_id: basis.to_owned(),
            mapping_receipt_ref: format!("synthetic://clock-map/{domain}"),
            mapping_receipt_sha256: "a".repeat(64),
            mapping_qualification: GateState::Pass,
            mapping_valid_from_unix_ns: 0,
            mapping_valid_through_unix_ns: 10_000,
            max_interval_width_ns: 1_000,
            interval: EventTimeIntervalNs {
                earliest_unix_ns: earliest,
                latest_unix_ns: latest,
            },
        }
    }

    fn terminal(event_id: &str, outcome: ReconciledOperationOutcome) -> AttemptTerminalReconciliation {
        AttemptTerminalReconciliation {
            event_id: event_id.to_owned(),
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
            retry_predecessor_terminal_to_start_order: retry_order,
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
                trigger_to_discharge_order: EventOrder::Before,
            }),
        }
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
        expired.mapping_valid_through_unix_ns = 15;
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
            Some(terminal("terminal-1", ReconciledOperationOutcome::NotApplied)),
            None,
        );
        let retry = attempt(
            "valve-1",
            "attempt-2",
            Some("attempt-1"),
            "start-2",
            None,
            Some(EventOrder::Before),
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
        let first = attempt("pump-1", "attempt-a", None, "start-a", None, None);
        let retry = attempt(
            "pump-1",
            "attempt-b",
            Some("attempt-a"),
            "start-b",
            None,
            Some(EventOrder::Before),
        );
        let report = validate_attempt_lifecycle(&[first, retry]);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| reason == "RETRY_BEFORE_RECONCILIATION:attempt-b"));
    }

    #[test]
    fn duplicate_starts_and_terminal_reconciliations_are_rejected() {
        let mut malformed = attempt(
            "motor-1",
            "attempt-1",
            None,
            "start-1",
            Some(terminal("terminal-1", ReconciledOperationOutcome::Applied)),
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
        let attempts = vec![attempt(
            "robot-7",
            "attempt-1",
            None,
            "start-1",
            Some(terminal("terminal-1", ReconciledOperationOutcome::NotApplied)),
            None,
        )];
        let first = scoped_obligation("obligation-a", "robot-7");
        let mut second = scoped_obligation("obligation-b", "robot-7");
        second.discharge = None;

        let report = assess_start_under_stop_obligations("robot-7", &attempts, &[first.clone(), second]);
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Indeterminate);
        assert_eq!(report.applicable_obligation_ids, vec!["obligation-a".to_owned(), "obligation-b".to_owned()]);
    }

    #[test]
    fn stop_discharge_cannot_cover_an_unreconciled_attempt_in_its_scope() {
        let attempts = vec![attempt("robot-7", "attempt-pending", None, "start-pending", None, None)];
        let obligation = scoped_obligation("obligation-a", "robot-7");
        let report = assess_start_under_stop_obligations("robot-7", &attempts, &[obligation]);
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "STOP_DISCHARGE_WITH_UNRECONCILED_ATTEMPT:obligation-a:attempt-pending"
        }));
    }

    #[test]
    fn unrelated_operation_is_not_captured_by_a_scoped_stop_obligation() {
        let obligation = scoped_obligation("obligation-a", "robot-7");
        let report = assess_start_under_stop_obligations("robot-8", &[], &[obligation]);
        assert!(report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::Safe);
        assert!(report.applicable_obligation_ids.is_empty());
    }

    #[test]
    fn two_discharge_receipts_must_bind_their_own_obligation_and_policy() {
        let mut first = scoped_obligation("obligation-a", "robot-7");
        let second = scoped_obligation("obligation-b", "robot-7");
        first.discharge.as_mut().unwrap().obligation_id = "obligation-b".to_owned();
        let report = assess_start_under_stop_obligations("robot-7", &[], &[first, second]);
        assert!(!report.start_permitted_by_stop_contract);
        assert_eq!(report.disposition, EventTraceDisposition::CriticalFailure);
        assert!(report.reason_codes.iter().any(|reason| {
            reason == "STOP_DISCHARGE_RECEIPT_BINDING_INVALID:obligation-a"
        }));
    }

    #[test]
    fn attempt_report_is_invariant_under_record_delivery_permutation() {
        let first = attempt(
            "pump-1",
            "a1",
            None,
            "start-a1",
            Some(terminal("term-a1", ReconciledOperationOutcome::NotApplied)),
            None,
        );
        let second = attempt(
            "pump-1",
            "a2",
            Some("a1"),
            "start-a2",
            None,
            Some(EventOrder::Before),
        );
        let forward = vec![first.clone(), second.clone()];
        let reverse = vec![second, first];
        assert_eq!(
            validate_attempt_lifecycle(&forward),
            validate_attempt_lifecycle(&reverse)
        );
    }
}
