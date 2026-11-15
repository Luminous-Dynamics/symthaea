// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable multi-frontier attempt-inventory snapshots.
//!
//! The stop-discharge snapshot remains immutable. Later decision snapshots
//! descend from it, preserve old evidence, and add events only when they
//! become bitemporally visible. This module consumes owner-qualified inputs;
//! it does not authenticate signatures, normalize clocks, or authorize effects.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{EventOrder, EventTraceDisposition, GateState};
use crate::event_order_v0_2::{
    AttemptHistoryInventory, AttemptLifecycleReport, BitemporalEventEnvelope,
    OperationAttemptRecord, QualifiedEventTime, ScopedStopObligation, StartGateReport,
    StopDischargeReceipt, StopScope, assess_start_under_stop_obligations,
    canonical_attempt_manifest_sha256, compare_qualified_event_times,
    validate_attempt_lifecycle,
};

pub const INVENTORY_SNAPSHOT_SCHEMA_V0_3_3: &str =
    "symthaea.engineering-decision.inventory-snapshot.v0.3.3";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotHistoryEventKind {
    SnapshotFrontier,
    AttemptStart,
    AttemptTerminal,
    StopTrigger,
    StopPolicyRevision,
    StopDischarge,
    Other,
}

/// Immutable event projection. The event ID is repeated deliberately so the
/// snapshot validator can detect a substituted envelope identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotHistoryEvent {
    pub event_id: String,
    pub kind: SnapshotHistoryEventKind,
    pub logical_operation_id: Option<String>,
    pub attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<StopScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_sha256: Option<String>,
    pub envelope: BitemporalEventEnvelope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotOwnerQualification {
    pub identity: GateState,
    pub digest_integrity: GateState,
    pub authenticity: GateState,
    pub scope_qualification: GateState,
    pub completeness: GateState,
    pub freshness: GateState,
    pub revocation: GateState,
}

/// One immutable cumulative history at a named bitemporal cut. The v0.2
/// attempt digest remains available for existing consumers; the v0.3.3
/// snapshot digest additionally commits to scope, ancestry, frontier and all
/// event envelopes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttemptInventorySnapshot {
    pub snapshot_id: String,
    pub parent_snapshot_ids: Vec<String>,
    pub frontier_event_id: String,
    pub frontier_time: QualifiedEventTime,
    pub attestation_event_id: String,
    pub attested_at: QualifiedEventTime,
    pub scope: StopScope,
    pub attempts: Vec<OperationAttemptRecord>,
    pub events: Vec<SnapshotHistoryEvent>,
    pub attempt_manifest_sha256: String,
    pub snapshot_manifest_sha256: String,
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub qualification: SnapshotOwnerQualification,
}

/// Exact immutable snapshot used to discharge one stop obligation. A later
/// current-decision snapshot must descend from this snapshot and contain the
/// exact discharge event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopDischargeSnapshotBinding {
    pub obligation_id: String,
    pub discharge_snapshot_id: String,
    pub trigger_event_id: String,
    pub policy_revision_event_id: String,
    pub discharge_event_id: String,
    pub policy_revision: String,
    pub scope: StopScope,
    pub trigger_time: QualifiedEventTime,
    pub policy_revision_time: QualifiedEventTime,
    pub discharge_time: QualifiedEventTime,
    pub receipt_ref: String,
    pub receipt_sha256: String,
    pub qualification: GateState,
    pub in_flight_reconciliation: GateState,
}

/// A new start decision consumes a later cumulative snapshot. The stop
/// bindings continue to point to their original discharge-time snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrentDecisionSnapshotSelection {
    pub decision_id: String,
    pub logical_operation_id: String,
    pub decision_time: QualifiedEventTime,
    pub current_snapshot_id: String,
    pub applicable_stop_obligation_ids: Vec<String>,
    pub stop_discharges: Vec<StopDischargeSnapshotBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotChainAssessment {
    pub disposition: EventTraceDisposition,
    pub snapshot_ids: Vec<String>,
    pub current_snapshot_id: Option<String>,
    pub current_inventory: Option<AttemptHistoryInventory>,
    pub attempts: Vec<OperationAttemptRecord>,
    pub obligations: Vec<ScopedStopObligation>,
    pub attempt_lifecycle: Option<AttemptLifecycleReport>,
    pub start_gate: Option<StartGateReport>,
    pub reason_codes: Vec<String>,
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn scopes_intersect(left: &StopScope, right: &StopScope) -> bool {
    match (left, right) {
        (StopScope::Global, _) | (_, StopScope::Global) => true,
        (StopScope::LogicalOperations(left), StopScope::LogicalOperations(right)) => {
            left.iter().any(|logical_id| right.contains(logical_id))
        }
    }
}

fn merge_disposition(
    current: &mut EventTraceDisposition,
    next: EventTraceDisposition,
) {
    *current = match (*current, next) {
        (EventTraceDisposition::CriticalFailure, _)
        | (_, EventTraceDisposition::CriticalFailure) => EventTraceDisposition::CriticalFailure,
        (EventTraceDisposition::Indeterminate, _)
        | (_, EventTraceDisposition::Indeterminate) => EventTraceDisposition::Indeterminate,
        _ => EventTraceDisposition::Safe,
    };
}

fn gate_disposition(state: GateState) -> EventTraceDisposition {
    match state {
        GateState::Pass => EventTraceDisposition::Safe,
        GateState::Fail => EventTraceDisposition::CriticalFailure,
        GateState::Indeterminate | GateState::NotApplicable => EventTraceDisposition::Indeterminate,
    }
}

fn check_before(
    first: &QualifiedEventTime,
    second: &QualifiedEventTime,
    critical_code: &'static str,
    indeterminate_code: &'static str,
    disposition: &mut EventTraceDisposition,
    reasons: &mut BTreeSet<String>,
) {
    match compare_qualified_event_times(first, second).order {
        EventOrder::Before => {}
        EventOrder::After | EventOrder::Inconsistent => {
            merge_disposition(disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(critical_code.to_owned());
        }
        EventOrder::Overlapping | EventOrder::Unknown => {
            merge_disposition(disposition, EventTraceDisposition::Indeterminate);
            reasons.insert(indeterminate_code.to_owned());
        }
    }
}

/// Domain-separated, order-independent digest over immutable snapshot
/// metadata, the v0.2 attempt manifest and the bitemporal event set.
pub fn canonical_snapshot_manifest_sha256(
    snapshot: &AttemptInventorySnapshot,
) -> Result<String, serde_json::Error> {
    let attempt_digest = canonical_attempt_manifest_sha256(&snapshot.attempts)?;
    let mut events = snapshot.events.clone();
    events.sort_by(|left, right| {
        left.event_id.cmp(&right.event_id)
            .then(left.kind.cmp(&right.kind))
            .then(left.logical_operation_id.cmp(&right.logical_operation_id))
            .then(left.attempt_id.cmp(&right.attempt_id))
    });
    let mut parents = snapshot.parent_snapshot_ids.clone();
    parents.sort();
    let projection = serde_json::to_vec(&(
        INVENTORY_SNAPSHOT_SCHEMA_V0_3_3,
        snapshot.snapshot_id.as_str(),
        parents,
        snapshot.frontier_event_id.as_str(),
        &snapshot.frontier_time,
        snapshot.attestation_event_id.as_str(),
        &snapshot.attested_at,
        &snapshot.scope,
        &snapshot.receipt_ref,
        &snapshot.receipt_sha256,
        snapshot.qualification,
        attempt_digest,
        events,
    ))?;
    let digest = Sha256::digest(&projection);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn validate_snapshot_payload(
    snapshot: &AttemptInventorySnapshot,
) -> (EventTraceDisposition, Vec<String>) {
    let mut disposition = EventTraceDisposition::Safe;
    let mut reasons = BTreeSet::new();

    if snapshot.snapshot_id.trim().is_empty()
        || snapshot.frontier_event_id.trim().is_empty()
        || snapshot.attestation_event_id.trim().is_empty()
        || snapshot.receipt_ref.trim().is_empty()
        || !is_sha256_hex(&snapshot.receipt_sha256)
        || !is_sha256_hex(&snapshot.attempt_manifest_sha256)
        || !is_sha256_hex(&snapshot.snapshot_manifest_sha256)
        || !snapshot.scope.is_well_formed()
    {
        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
        reasons.insert("SNAPSHOT_ID_SCOPE_OR_RECEIPT_SHAPE_INVALID".to_owned());
    }

    for (name, state) in [
        ("IDENTITY", snapshot.qualification.identity),
        ("DIGEST_INTEGRITY", snapshot.qualification.digest_integrity),
        ("AUTHENTICITY", snapshot.qualification.authenticity),
        ("SCOPE", snapshot.qualification.scope_qualification),
        ("COMPLETENESS", snapshot.qualification.completeness),
        ("FRESHNESS", snapshot.qualification.freshness),
        ("REVOCATION", snapshot.qualification.revocation),
    ] {
        let gate = gate_disposition(state);
        if gate != EventTraceDisposition::Safe {
            merge_disposition(&mut disposition, gate);
            reasons.insert(format!("SNAPSHOT_{name}_{state:?}"));
        }
    }

    match canonical_attempt_manifest_sha256(&snapshot.attempts) {
        Ok(digest) if digest == snapshot.attempt_manifest_sha256 => {}
        Ok(_) => {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert("SNAPSHOT_ATTEMPT_MANIFEST_DIGEST_MISMATCH".to_owned());
        }
        Err(_) => {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert("SNAPSHOT_ATTEMPT_MANIFEST_SERIALIZATION_FAILED".to_owned());
        }
    }
    match canonical_snapshot_manifest_sha256(snapshot) {
        Ok(digest) if digest == snapshot.snapshot_manifest_sha256 => {}
        Ok(_) => {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert("SNAPSHOT_MANIFEST_DIGEST_MISMATCH".to_owned());
        }
        Err(_) => {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert("SNAPSHOT_MANIFEST_SERIALIZATION_FAILED".to_owned());
        }
    }

    let mut event_index = BTreeMap::<String, &SnapshotHistoryEvent>::new();
    for event in &snapshot.events {
        if event.event_id.trim().is_empty()
            || event.event_id != event.envelope.event_id
            || event_index.insert(event.event_id.clone(), event).is_some()
        {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_EVENT_ID_INVALID_OR_DUPLICATE:{}", event.event_id));
            continue;
        }
        if let Some(receipt_ref) = event.receipt_ref.as_deref() {
            if receipt_ref.trim().is_empty() {
                merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                reasons.insert(format!("SNAPSHOT_EVENT_RECEIPT_REF_INVALID:{}", event.event_id));
            }
        }
        if let Some(receipt_sha256) = event.receipt_sha256.as_deref() {
            if !is_sha256_hex(receipt_sha256) {
                merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                reasons.insert(format!("SNAPSHOT_EVENT_RECEIPT_DIGEST_INVALID:{}", event.event_id));
            }
        }
        match event.kind {
            SnapshotHistoryEventKind::AttemptStart | SnapshotHistoryEventKind::AttemptTerminal => {
                if event.logical_operation_id.as_deref().is_none_or(|logical_id| {
                    !snapshot.scope.applies_to(logical_id)
                }) {
                    merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                    reasons.insert(format!("SNAPSHOT_ATTEMPT_EVENT_SCOPE_INVALID:{}", event.event_id));
                }
            }
            SnapshotHistoryEventKind::StopTrigger
            | SnapshotHistoryEventKind::StopPolicyRevision
            | SnapshotHistoryEventKind::StopDischarge => {
                if event.scope.as_ref().is_none_or(|scope| {
                    !scope.is_well_formed() || !scope_covers(&snapshot.scope, scope)
                }) {
                    merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                    reasons.insert(format!("SNAPSHOT_STOP_EVENT_SCOPE_INVALID:{}", event.event_id));
                }
            }
            _ => {}
        }
        if event.kind == SnapshotHistoryEventKind::StopPolicyRevision
            && event.policy_revision.as_deref().is_none_or(str::is_empty)
        {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_POLICY_REVISION_ID_MISSING:{}", event.event_id));
        }
        if event.envelope.capture_qualification != GateState::Pass {
            merge_disposition(
                &mut disposition,
                gate_disposition(event.envelope.capture_qualification),
            );
            reasons.insert(format!("SNAPSHOT_EVENT_CAPTURE_UNQUALIFIED:{}", event.event_id));
        }

        let is_current_frontier = event.event_id == snapshot.frontier_event_id
            && event.kind == SnapshotHistoryEventKind::SnapshotFrontier;
        if is_current_frontier {
            if event.envelope.occurred_at.as_ref() != Some(&snapshot.frontier_time) {
                merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                reasons.insert("SNAPSHOT_FRONTIER_TIME_BINDING_MISMATCH".to_owned());
            }
            check_before(
                &snapshot.frontier_time,
                &snapshot.attested_at,
                "SNAPSHOT_ATTESTATION_PRECEDES_FRONTIER",
                "SNAPSHOT_FRONTIER_ATTESTATION_ORDER_UNRESOLVED",
                &mut disposition,
                &mut reasons,
            );
            check_before(
                &event.envelope.durably_recorded_at,
                &snapshot.attested_at,
                "SNAPSHOT_FRONTIER_NOT_DURABLE_BEFORE_ATTESTATION",
                "SNAPSHOT_FRONTIER_DURABILITY_ORDER_UNRESOLVED",
                &mut disposition,
                &mut reasons,
            );
        } else {
            check_before(
                &event.envelope.first_observed_at,
                &snapshot.frontier_time,
                "SNAPSHOT_CONTAINS_EVENT_NOT_OBSERVED_AT_FRONTIER",
                "SNAPSHOT_EVENT_OBSERVATION_FRONTIER_ORDER_UNRESOLVED",
                &mut disposition,
                &mut reasons,
            );
            check_before(
                &event.envelope.durably_recorded_at,
                &snapshot.frontier_time,
                "SNAPSHOT_CONTAINS_EVENT_NOT_DURABLE_AT_FRONTIER",
                "SNAPSHOT_EVENT_DURABILITY_FRONTIER_ORDER_UNRESOLVED",
                &mut disposition,
                &mut reasons,
            );
            if let Some(occurred_at) = event.envelope.occurred_at.as_ref() {
                match compare_qualified_event_times(occurred_at, &snapshot.frontier_time).order {
                    EventOrder::After | EventOrder::Inconsistent => {
                        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                        reasons.insert(format!("SNAPSHOT_EVENT_OCCURS_AFTER_FRONTIER:{}", event.event_id));
                    }
                    EventOrder::Overlapping | EventOrder::Unknown => {
                        merge_disposition(&mut disposition, EventTraceDisposition::Indeterminate);
                        reasons.insert(format!("SNAPSHOT_EVENT_OCCURRENCE_FRONTIER_ORDER_UNRESOLVED:{}", event.event_id));
                    }
                    EventOrder::Before => {}
                }
            }
        }
    }

    let frontier_records = snapshot.events.iter().filter(|event| {
        event.event_id == snapshot.frontier_event_id
            && event.kind == SnapshotHistoryEventKind::SnapshotFrontier
    }).count();
    if frontier_records != 1 {
        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
        reasons.insert("SNAPSHOT_FRONTIER_EVENT_NOT_UNIQUE".to_owned());
    }

    let mut attempts_by_id = BTreeMap::<String, &OperationAttemptRecord>::new();
    for attempt in &snapshot.attempts {
        if !snapshot.scope.applies_to(&attempt.identity.logical_operation_id) {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_ATTEMPT_OUT_OF_SCOPE:{}", attempt.identity.attempt_id));
        }
        if attempts_by_id.insert(attempt.identity.attempt_id.clone(), attempt).is_some() {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_ATTEMPT_ID_DUPLICATE:{}", attempt.identity.attempt_id));
        }

        for start_id in &attempt.start_event_ids {
            match event_index.get(start_id.as_str()) {
                Some(event)
                    if event.kind == SnapshotHistoryEventKind::AttemptStart
                        && event.logical_operation_id.as_deref()
                            == Some(attempt.identity.logical_operation_id.as_str())
                        && event.attempt_id.as_deref() == Some(attempt.identity.attempt_id.as_str()) => {}
                _ => {
                    merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                    reasons.insert(format!("SNAPSHOT_ATTEMPT_START_EVENT_BINDING_INVALID:{start_id}"));
                }
            }
        }
        for terminal_id in &attempt.terminal_reconciliation_event_ids {
            match event_index.get(terminal_id.as_str()) {
                Some(event)
                    if event.kind == SnapshotHistoryEventKind::AttemptTerminal
                        && event.logical_operation_id.as_deref()
                            == Some(attempt.identity.logical_operation_id.as_str())
                        && event.attempt_id.as_deref() == Some(attempt.identity.attempt_id.as_str()) => {}
                _ => {
                    merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                    reasons.insert(format!("SNAPSHOT_ATTEMPT_TERMINAL_EVENT_BINDING_INVALID:{terminal_id}"));
                }
            }
        }
    }

    for event in &snapshot.events {
        if let Some(attempt_id) = event.attempt_id.as_deref() {
            match event.kind {
                SnapshotHistoryEventKind::AttemptStart => {
                    let linked = attempts_by_id.get(attempt_id).is_some_and(|attempt| {
                        attempt.identity.logical_operation_id == event.logical_operation_id.clone().unwrap_or_default()
                            && attempt.start_event_ids.contains(&event.event_id)
                    });
                    if !linked {
                        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                        reasons.insert(format!("SNAPSHOT_ORPHAN_ATTEMPT_START_EVENT:{}", event.event_id));
                    }
                }
                SnapshotHistoryEventKind::AttemptTerminal => {
                    let linked = attempts_by_id.get(attempt_id).is_some_and(|attempt| {
                        attempt.identity.logical_operation_id == event.logical_operation_id.clone().unwrap_or_default()
                            && attempt.terminal_reconciliation_event_ids.contains(&event.event_id)
                    });
                    if !linked {
                        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                        reasons.insert(format!("SNAPSHOT_ORPHAN_ATTEMPT_TERMINAL_EVENT:{}", event.event_id));
                    }
                }
                _ => {}
            }
        } else if matches!(
            event.kind,
            SnapshotHistoryEventKind::AttemptStart | SnapshotHistoryEventKind::AttemptTerminal
        ) {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_ATTEMPT_EVENT_MISSING_ATTEMPT_ID:{}", event.event_id));
        }
    }

    (disposition, reasons.into_iter().collect())
}

fn validate_snapshot_transition(
    parent: &AttemptInventorySnapshot,
    child: &AttemptInventorySnapshot,
) -> (EventTraceDisposition, Vec<String>) {
    let mut disposition = EventTraceDisposition::Safe;
    let mut reasons = BTreeSet::new();

    if !scope_covers(&child.scope, &parent.scope) {
        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
        reasons.insert("SNAPSHOT_CHILD_SCOPE_DOES_NOT_COVER_PARENT".to_owned());
    }
    check_before(
        &parent.attested_at,
        &child.frontier_time,
        "SNAPSHOT_CHILD_FRONTIER_PRECEDES_PARENT_ATTESTATION",
        "SNAPSHOT_PARENT_ATTESTATION_CHILD_FRONTIER_ORDER_UNRESOLVED",
        &mut disposition,
        &mut reasons,
    );
    check_before(
        &parent.frontier_time,
        &child.frontier_time,
        "SNAPSHOT_FRONTIER_NOT_MONOTONIC",
        "SNAPSHOT_FRONTIER_SUCCESSION_UNRESOLVED",
        &mut disposition,
        &mut reasons,
    );

    let child_events: BTreeMap<_, _> = child.events.iter()
        .map(|event| (event.event_id.as_str(), event))
        .collect();
    for old_event in &parent.events {
        match child_events.get(old_event.event_id.as_str()) {
            Some(new_event) if **new_event == *old_event => {}
            _ => {
                merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                reasons.insert(format!("SNAPSHOT_EVENT_MUTATED_OR_REMOVED:{}", old_event.event_id));
            }
        }
    }

    let child_attempts: BTreeMap<_, _> = child.attempts.iter()
        .map(|attempt| (attempt.identity.attempt_id.as_str(), attempt))
        .collect();
    for old_attempt in &parent.attempts {
        let Some(new_attempt) = child_attempts.get(old_attempt.identity.attempt_id.as_str()) else {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_ATTEMPT_REMOVED:{}", old_attempt.identity.attempt_id));
            continue;
        };
        if new_attempt.identity != old_attempt.identity
            || new_attempt.start_event_ids != old_attempt.start_event_ids
        {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_ATTEMPT_IDENTITY_OR_START_MUTATED:{}", old_attempt.identity.attempt_id));
        }
        if old_attempt.terminal_reconciliation.is_some()
            && (new_attempt.terminal_reconciliation != old_attempt.terminal_reconciliation
                || new_attempt.terminal_reconciliation_event_ids
                    != old_attempt.terminal_reconciliation_event_ids)
        {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_TERMINAL_RECONCILIATION_REWRITTEN:{}", old_attempt.identity.attempt_id));
        }
        for old_relation in &old_attempt.stop_discharge_relations {
            if !new_attempt.stop_discharge_relations.contains(old_relation) {
                merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                reasons.insert(format!("SNAPSHOT_STOP_RELATION_REMOVED:{}", old_attempt.identity.attempt_id));
            }
        }
    }

    let parent_event_ids: BTreeSet<_> = parent.events.iter()
        .map(|event| event.event_id.as_str())
        .collect();
    for new_event in &child.events {
        if parent_event_ids.contains(new_event.event_id.as_str())
            || new_event.event_id == child.frontier_event_id
        {
            continue;
        }
        let parent_scope_applies = match (
            new_event.logical_operation_id.as_deref(),
            new_event.scope.as_ref(),
        ) {
            (Some(logical_id), _) => parent.scope.applies_to(logical_id),
            (None, Some(event_scope)) => scopes_intersect(&parent.scope, event_scope),
            (None, None) => matches!(parent.scope, StopScope::Global),
        };
        if !parent_scope_applies {
            continue;
        }
        check_before(
            &parent.frontier_time,
            &new_event.envelope.first_observed_at,
            "SNAPSHOT_ADDED_EVENT_ALREADY_OBSERVED_AT_PARENT_FRONTIER",
            "SNAPSHOT_NEW_EVENT_OBSERVATION_PARENT_CUT_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );
        check_before(
            &parent.frontier_time,
            &new_event.envelope.durably_recorded_at,
            "SNAPSHOT_ADDED_EVENT_ALREADY_DURABLE_AT_PARENT_FRONTIER",
            "SNAPSHOT_NEW_EVENT_DURABILITY_PARENT_CUT_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );
    }

    (disposition, reasons.into_iter().collect())
}

fn snapshot_ancestors(
    snapshot_id: &str,
    snapshots: &BTreeMap<String, &AttemptInventorySnapshot>,
    output: &mut BTreeSet<String>,
) {
    if !output.insert(snapshot_id.to_owned()) {
        return;
    }
    if let Some(snapshot) = snapshots.get(snapshot_id) {
        for parent_id in &snapshot.parent_snapshot_ids {
            snapshot_ancestors(parent_id, snapshots, output);
        }
    }
}

fn failure_report(
    disposition: EventTraceDisposition,
    reasons: BTreeSet<String>,
) -> SnapshotChainAssessment {
    SnapshotChainAssessment {
        disposition,
        snapshot_ids: Vec::new(),
        current_snapshot_id: None,
        current_inventory: None,
        attempts: Vec::new(),
        obligations: Vec::new(),
        attempt_lifecycle: None,
        start_gate: None,
        reason_codes: reasons.into_iter().collect(),
    }
}

/// Validate the snapshot DAG, immutable transitions, stop-discharge snapshot
/// pins, and a later current-decision snapshot. The result is shaped for the
/// existing v0.2 attempt/stop evaluator.
pub fn assess_snapshot_chain(
    snapshots: &[AttemptInventorySnapshot],
    selection: &CurrentDecisionSnapshotSelection,
) -> SnapshotChainAssessment {
    let mut disposition = EventTraceDisposition::Safe;
    let mut reasons = BTreeSet::new();
    let mut snapshot_index = BTreeMap::<String, &AttemptInventorySnapshot>::new();

    if selection.decision_id.trim().is_empty() || selection.logical_operation_id.trim().is_empty() {
        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
        reasons.insert("SNAPSHOT_DECISION_ID_INVALID".to_owned());
    }

    for snapshot in snapshots {
        if snapshot_index.insert(snapshot.snapshot_id.clone(), snapshot).is_some() {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("SNAPSHOT_ID_DUPLICATE:{}", snapshot.snapshot_id));
        }
        let (snapshot_disposition, snapshot_reasons) = validate_snapshot_payload(snapshot);
        merge_disposition(&mut disposition, snapshot_disposition);
        reasons.extend(snapshot_reasons);
        let mut parent_ids = BTreeSet::new();
        for parent_id in &snapshot.parent_snapshot_ids {
            if !parent_ids.insert(parent_id) || parent_id == &snapshot.snapshot_id {
                merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                reasons.insert(format!("SNAPSHOT_PARENT_ID_INVALID:{}", snapshot.snapshot_id));
            }
        }
    }

    for snapshot in snapshots {
        for parent_id in &snapshot.parent_snapshot_ids {
            match snapshot_index.get(parent_id) {
                Some(parent) => {
                    let (transition_disposition, transition_reasons) =
                        validate_snapshot_transition(parent, snapshot);
                    merge_disposition(&mut disposition, transition_disposition);
                    reasons.extend(transition_reasons);
                }
                None => {
                    merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                    reasons.insert(format!("SNAPSHOT_PARENT_MISSING:{parent_id}"));
                }
            }
        }
    }

    let Some(current_snapshot) = snapshot_index.get(&selection.current_snapshot_id).copied() else {
        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
        reasons.insert("CURRENT_DECISION_SNAPSHOT_MISSING".to_owned());
        return failure_report(disposition, reasons);
    };

    check_before(
        &current_snapshot.frontier_time,
        &selection.decision_time,
        "CURRENT_SNAPSHOT_FRONTIER_AFTER_DECISION",
        "CURRENT_SNAPSHOT_FRONTIER_DECISION_ORDER_UNRESOLVED",
        &mut disposition,
        &mut reasons,
    );
    check_before(
        &current_snapshot.attested_at,
        &selection.decision_time,
        "CURRENT_SNAPSHOT_ATTESTED_AFTER_DECISION",
        "CURRENT_SNAPSHOT_ATTESTATION_DECISION_ORDER_UNRESOLVED",
        &mut disposition,
        &mut reasons,
    );
    if !current_snapshot.scope.applies_to(&selection.logical_operation_id) {
        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
        reasons.insert("CURRENT_SNAPSHOT_DOES_NOT_COVER_LOGICAL_OPERATION".to_owned());
    }

    let mut obligations = Vec::<ScopedStopObligation>::new();
    let mut seen_obligations = BTreeSet::new();
    let declared_obligations: BTreeSet<_> = selection.applicable_stop_obligation_ids.iter().cloned().collect();
    let binding_obligations: BTreeSet<_> = selection.stop_discharges.iter()
        .map(|binding| binding.obligation_id.clone())
        .collect();
    if declared_obligations.len() != selection.applicable_stop_obligation_ids.len()
        || declared_obligations != binding_obligations
    {
        merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
        reasons.insert("APPLICABLE_STOP_OBLIGATION_SET_MISMATCH".to_owned());
    }
    let mut ancestors = BTreeSet::new();
    snapshot_ancestors(&current_snapshot.snapshot_id, &snapshot_index, &mut ancestors);
    let current_event_index: BTreeMap<_, _> = current_snapshot.events.iter()
        .map(|event| (event.event_id.as_str(), event))
        .collect();

    for binding in &selection.stop_discharges {
        if !binding.scope.applies_to(&selection.logical_operation_id) {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("STOP_DISCHARGE_SCOPE_NOT_APPLICABLE_TO_DECISION:{}", binding.obligation_id));
        }
        if binding.obligation_id.trim().is_empty()
            || binding.policy_revision.trim().is_empty()
            || !binding.scope.is_well_formed()
            || !seen_obligations.insert(binding.obligation_id.clone())
            || binding.receipt_ref.trim().is_empty()
            || !is_sha256_hex(&binding.receipt_sha256)
        {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert("STOP_DISCHARGE_SNAPSHOT_BINDING_INVALID".to_owned());
            continue;
        }
        let Some(discharge_snapshot) = snapshot_index.get(&binding.discharge_snapshot_id).copied() else {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("STOP_DISCHARGE_SNAPSHOT_MISSING:{}", binding.obligation_id));
            continue;
        };
        if !scope_covers(&discharge_snapshot.scope, &binding.scope)
            || !scope_covers(&current_snapshot.scope, &binding.scope)
        {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("STOP_DISCHARGE_SNAPSHOT_SCOPE_MISMATCH:{}", binding.obligation_id));
        }
        if !ancestors.contains(&binding.discharge_snapshot_id) {
            merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
            reasons.insert(format!("CURRENT_SNAPSHOT_DOES_NOT_DESCEND_FROM_DISCHARGE_SNAPSHOT:{}", binding.obligation_id));
        }

        let discharge_snapshot_events: BTreeMap<_, _> = discharge_snapshot.events.iter()
            .map(|event| (event.event_id.as_str(), event))
            .collect();
        for (event_id, expected_kind, expected_time, label) in [
            (&binding.trigger_event_id, SnapshotHistoryEventKind::StopTrigger, &binding.trigger_time, "TRIGGER"),
            (&binding.policy_revision_event_id, SnapshotHistoryEventKind::StopPolicyRevision, &binding.policy_revision_time, "POLICY_REVISION"),
        ] {
            match discharge_snapshot_events.get(event_id.as_str()) {
                Some(event)
                    if event.kind == expected_kind
                        && event.envelope.occurred_at.as_ref() == Some(expected_time)
                        && event.scope.as_ref() == Some(&binding.scope)
                        && (expected_kind != SnapshotHistoryEventKind::StopPolicyRevision
                            || event.policy_revision.as_deref() == Some(binding.policy_revision.as_str())) => {}
                _ => {
                    merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                    reasons.insert(format!("STOP_{label}_NOT_BOUND_IN_DISCHARGE_SNAPSHOT:{}", binding.obligation_id));
                }
            }
        }
        check_before(
            &binding.trigger_time,
            &discharge_snapshot.frontier_time,
            "STOP_TRIGGER_AFTER_PINNED_SNAPSHOT_FRONTIER",
            "STOP_TRIGGER_PINNED_SNAPSHOT_ORDER_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );
        check_before(
            &binding.policy_revision_time,
            &discharge_snapshot.frontier_time,
            "STOP_POLICY_AFTER_PINNED_SNAPSHOT_FRONTIER",
            "STOP_POLICY_PINNED_SNAPSHOT_ORDER_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );
        check_before(
            &discharge_snapshot.frontier_time,
            &discharge_snapshot.attested_at,
            "STOP_SNAPSHOT_ATTESTATION_PRECEDES_FRONTIER",
            "STOP_SNAPSHOT_ATTESTATION_ORDER_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );
        check_before(
            &discharge_snapshot.attested_at,
            &binding.discharge_time,
            "STOP_DISCHARGE_PRECEDES_SNAPSHOT_ATTESTATION",
            "STOP_DISCHARGE_SNAPSHOT_ATTESTATION_ORDER_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );
        check_before(
            &binding.trigger_time,
            &binding.discharge_time,
            "STOP_DISCHARGE_PRECEDES_TRIGGER",
            "STOP_DISCHARGE_TRIGGER_ORDER_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );
        merge_disposition(&mut disposition, gate_disposition(binding.qualification));
        merge_disposition(&mut disposition, gate_disposition(binding.in_flight_reconciliation));

        match current_event_index.get(binding.discharge_event_id.as_str()) {
            Some(event)
                if event.kind == SnapshotHistoryEventKind::StopDischarge
                    && event.envelope.occurred_at.as_ref() == Some(&binding.discharge_time)
                    && event.scope.as_ref() == Some(&binding.scope)
                    && event.policy_revision.as_deref() == Some(binding.policy_revision.as_str())
                    && event.receipt_ref.as_deref() == Some(binding.receipt_ref.as_str())
                    && event.receipt_sha256.as_deref() == Some(binding.receipt_sha256.as_str()) => {}
            _ => {
                merge_disposition(&mut disposition, EventTraceDisposition::CriticalFailure);
                reasons.insert(format!("CURRENT_SNAPSHOT_DISCHARGE_BINDING_MISMATCH:{}", binding.obligation_id));
            }
        }
        check_before(
            &binding.discharge_time,
            &current_snapshot.frontier_time,
            "CURRENT_SNAPSHOT_FRONTIER_PRECEDES_STOP_DISCHARGE",
            "CURRENT_SNAPSHOT_STOP_DISCHARGE_ORDER_UNRESOLVED",
            &mut disposition,
            &mut reasons,
        );

        obligations.push(ScopedStopObligation {
            obligation_id: binding.obligation_id.clone(),
            trigger_event_id: binding.trigger_event_id.clone(),
            policy_revision: binding.policy_revision.clone(),
            scope: binding.scope.clone(),
            discharge: Some(StopDischargeReceipt {
                event_id: binding.discharge_event_id.clone(),
                obligation_id: binding.obligation_id.clone(),
                trigger_event_id: binding.trigger_event_id.clone(),
                policy_revision: binding.policy_revision.clone(),
                scope: binding.scope.clone(),
                receipt_ref: binding.receipt_ref.clone(),
                receipt_sha256: binding.receipt_sha256.clone(),
                qualification: binding.qualification,
                in_flight_reconciliation: binding.in_flight_reconciliation,
                in_flight_inventory_frontier_event_id: discharge_snapshot.frontier_event_id.clone(),
                in_flight_attempt_manifest_sha256: discharge_snapshot.attempt_manifest_sha256.clone(),
                trigger_to_discharge_order: compare_qualified_event_times(
                    &binding.trigger_time, &binding.discharge_time,
                ).order,
            }),
        });
    }

    let attempt_lifecycle = validate_attempt_lifecycle(&current_snapshot.attempts);
    merge_disposition(&mut disposition, attempt_lifecycle.disposition);
    reasons.extend(attempt_lifecycle.reason_codes.clone());

    let inventory = AttemptHistoryInventory {
        inventory_frontier_event_id: current_snapshot.frontier_event_id.clone(),
        attempt_manifest_sha256: current_snapshot.attempt_manifest_sha256.clone(),
        receipt_ref: current_snapshot.receipt_ref.clone(),
        receipt_sha256: current_snapshot.receipt_sha256.clone(),
        scope: current_snapshot.scope.clone(),
        identity: current_snapshot.qualification.identity,
        digest_integrity: current_snapshot.qualification.digest_integrity,
        authenticity: current_snapshot.qualification.authenticity,
        scope_qualification: current_snapshot.qualification.scope_qualification,
        completeness: current_snapshot.qualification.completeness,
    };
    let start_gate = assess_start_under_stop_obligations(
        &selection.logical_operation_id,
        &current_snapshot.attempts,
        &inventory,
        &obligations,
    );
    merge_disposition(&mut disposition, start_gate.disposition);
    reasons.extend(start_gate.reason_codes.clone());

    let mut start_gate = start_gate;
    if disposition != EventTraceDisposition::Safe {
        start_gate.start_permitted_by_stop_contract = false;
        start_gate.disposition = disposition;
        start_gate.reason_codes.extend(reasons.iter().cloned());
        start_gate.reason_codes.sort();
        start_gate.reason_codes.dedup();
    }

    SnapshotChainAssessment {
        disposition,
        snapshot_ids: snapshot_index.keys().cloned().collect(),
        current_snapshot_id: Some(current_snapshot.snapshot_id.clone()),
        current_inventory: Some(inventory),
        attempts: current_snapshot.attempts.clone(),
        obligations,
        attempt_lifecycle: Some(attempt_lifecycle),
        start_gate: Some(start_gate),
        reason_codes: reasons.into_iter().collect(),
    }
}
