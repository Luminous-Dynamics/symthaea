// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent qualification contracts for CogSec shadow evidence.
//!
//! This crate deliberately sits **outside** the runtime instrumentation under
//! test. A deterministic scenario driver states what transitions it expects;
//! the typed event ledger reports what was observed; and the generic evidence
//! plane independently reports mechanism counters. A missing hook therefore
//! cannot make itself disappear from both sides of the comparison.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_cogsec_evidence::{
    EvidenceLedgerSnapshot, ReconciliationReport, ShadowEventKind, reconcile_shadow_evidence,
};
use symthaea_evidence_plane::EvidenceCounters;

/// Expected occurrence relation for one shadow event kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventCountExpectation {
    /// The event must occur exactly `n` times.
    Exactly(u64),
    /// The event must occur at least `n` times.
    AtLeast(u64),
    /// The event must not occur.
    MustBeZero,
    /// The event count must exactly match another event kind's count.
    ExactlySameAs(ShadowEventKind),
}

/// Scenario-owned expected transition contract.
///
/// This is qualification input, not runtime instrumentation output. Production
/// qualification should source it from the deterministic test/scenario driver,
/// not infer it from the events being validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioContract {
    /// Stable human-readable scenario identifier. This is not security authority.
    pub scenario_id: String,
    /// Expected count relation for every event kind the scenario constrains.
    pub expectations: BTreeMap<ShadowEventKind, EventCountExpectation>,
}

impl ScenarioContract {
    /// Create an empty scenario contract.
    pub fn new(scenario_id: impl Into<String>) -> Self {
        Self {
            scenario_id: scenario_id.into(),
            expectations: BTreeMap::new(),
        }
    }

    /// Add or replace one event-count expectation.
    pub fn expect(mut self, kind: ShadowEventKind, expectation: EventCountExpectation) -> Self {
        self.expectations.insert(kind, expectation);
        self
    }
}

/// One mismatch between a scenario-owned expected count and observed events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventCountViolation {
    /// Event kind whose count failed its expectation.
    pub kind: ShadowEventKind,
    /// Declared independent expectation.
    pub expectation: EventCountExpectation,
    /// Event count actually present in the snapshot.
    pub observed: u64,
    /// For relational expectations, the reference count that was compared.
    pub reference_observed: Option<u64>,
}

/// Full shadow qualification result: structural/event/counter reconciliation
/// plus the independent scenario expectation check.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioQualificationReport {
    /// Scenario identifier supplied by the independent driver.
    pub scenario_id: String,
    /// Existing typed-event versus mechanism-counter reconciliation.
    pub evidence: ReconciliationReport,
    /// Scenario expectation violations.
    pub event_count_violations: Vec<EventCountViolation>,
}

impl ScenarioQualificationReport {
    /// Whether this run can claim complete coverage for the scenario's scoped paths.
    pub fn qualifies_for_full_coverage(&self) -> bool {
        self.evidence.qualifies_for_full_coverage() && self.event_count_violations.is_empty()
    }
}

/// Validate one event snapshot against an independent scenario contract.
pub fn validate_expected_event_counts(
    contract: &ScenarioContract,
    snapshot: &EvidenceLedgerSnapshot,
) -> Vec<EventCountViolation> {
    let mut counts = BTreeMap::<ShadowEventKind, u64>::new();
    for event in &snapshot.events {
        *counts.entry(event.kind).or_insert(0) += 1;
    }

    contract
        .expectations
        .iter()
        .filter_map(|(kind, expectation)| {
            let observed = counts.get(kind).copied().unwrap_or(0);
            let (satisfied, reference_observed) = match expectation {
                EventCountExpectation::Exactly(expected) => (observed == *expected, None),
                EventCountExpectation::AtLeast(minimum) => (observed >= *minimum, None),
                EventCountExpectation::MustBeZero => (observed == 0, None),
                EventCountExpectation::ExactlySameAs(reference_kind) => {
                    let reference = counts.get(reference_kind).copied().unwrap_or(0);
                    (observed == reference, Some(reference))
                }
            };

            (!satisfied).then_some(EventCountViolation {
                kind: *kind,
                expectation: *expectation,
                observed,
                reference_observed,
            })
        })
        .collect()
}

/// Run the complete first-layer shadow qualification.
///
/// The existing evidence adapter checks event structure, causal pairing, P0
/// coverage, resource-version semantics and optional mechanism counters. This
/// outer layer independently checks scenario-owned expected event counts.
pub fn qualify_shadow_scenario(
    contract: &ScenarioContract,
    snapshot: &EvidenceLedgerSnapshot,
    measured: Option<&EvidenceCounters>,
) -> ScenarioQualificationReport {
    ScenarioQualificationReport {
        scenario_id: contract.scenario_id.clone(),
        evidence: reconcile_shadow_evidence(snapshot, measured),
        event_count_violations: validate_expected_event_counts(contract, snapshot),
    }
}

/// Canonical v0 contract: one goal input while working memory has capacity.
pub fn goal_no_eviction_v0() -> ScenarioContract {
    use EventCountExpectation::{Exactly, MustBeZero};
    use ShadowEventKind::*;

    ScenarioContract::new("cogsec-shadow-v0/goal-no-eviction")
        .expect(IngressObserved, Exactly(1))
        .expect(WorkingMemoryAdmissionEvaluated, Exactly(1))
        .expect(WorkingMemoryAdmissionObserved, Exactly(1))
        .expect(WorkingMemoryEvictionObserved, MustBeZero)
        .expect(GraduationEvaluated, MustBeZero)
        .expect(GraduationObserved, MustBeZero)
        .expect(WorkingStateInfluenceEvaluated, Exactly(1))
        .expect(WorkingStateInfluenceObserved, Exactly(1))
        .expect(GoalActivationEvaluated, Exactly(1))
        .expect(GoalActivationObserved, Exactly(1))
        .expect(AffectMutationEvaluated, MustBeZero)
        .expect(AffectMutationObserved, MustBeZero)
}

/// Canonical v0 contract: one goal input that forces working-memory eviction.
pub fn goal_with_eviction_v0() -> ScenarioContract {
    use EventCountExpectation::Exactly;
    use ShadowEventKind::*;

    ScenarioContract::new("cogsec-shadow-v0/goal-with-eviction")
        .expect(IngressObserved, Exactly(1))
        .expect(WorkingMemoryAdmissionEvaluated, Exactly(1))
        .expect(WorkingMemoryAdmissionObserved, Exactly(1))
        .expect(WorkingMemoryEvictionObserved, Exactly(1))
        .expect(GraduationEvaluated, Exactly(1))
        .expect(GraduationObserved, Exactly(1))
        .expect(WorkingStateInfluenceEvaluated, Exactly(1))
        .expect(WorkingStateInfluenceObserved, Exactly(1))
        .expect(GoalActivationEvaluated, Exactly(1))
        .expect(GoalActivationObserved, Exactly(1))
}

/// Canonical v0 contract: one feedback input.
pub fn feedback_input_v0() -> ScenarioContract {
    use EventCountExpectation::{Exactly, MustBeZero};
    use ShadowEventKind::*;

    ScenarioContract::new("cogsec-shadow-v0/feedback-input")
        .expect(IngressObserved, Exactly(1))
        .expect(WorkingMemoryAdmissionEvaluated, Exactly(1))
        .expect(WorkingMemoryAdmissionObserved, Exactly(1))
        .expect(WorkingStateInfluenceEvaluated, Exactly(1))
        .expect(WorkingStateInfluenceObserved, Exactly(1))
        .expect(AffectMutationEvaluated, Exactly(1))
        .expect(AffectMutationObserved, Exactly(1))
        .expect(GoalActivationEvaluated, MustBeZero)
        .expect(GoalActivationObserved, MustBeZero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use symthaea_cogsec::Digest32;
    use symthaea_cogsec_evidence::{
        EvidenceCompleteness, EvidenceConfidentiality, IngressClass, LedgerStats,
        PrincipalContext, QualificationManifest, ShadowEvent, ShadowEventPayload,
        SHADOW_EVENT_SCHEMA_V1,
    };

    fn event(sequence: u64, kind: ShadowEventKind, payload: ShadowEventPayload) -> ShadowEvent {
        ShadowEvent {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            event_id: symthaea_cogsec_evidence::EventId {
                ledger_epoch: 1,
                sequence,
            },
            proposal_id: None,
            transaction_id: None,
            principals: PrincipalContext::default(),
            kind,
            resource: None,
            resource_version_before: None,
            resource_version_after: None,
            state_root_before: None,
            state_root_after: None,
            policy_root: None,
            policy_epoch: None,
            authorization_epoch: None,
            revocation_epoch: None,
            causal_parents: BTreeSet::new(),
            cognitive_tick: None,
            run_id: None,
            confidentiality: EvidenceConfidentiality::LocalPrivate,
            payload,
        }
    }

    fn snapshot(events: Vec<ShadowEvent>) -> EvidenceLedgerSnapshot {
        let last = events.iter().map(|e| e.event_id.sequence).max().unwrap_or(0);
        EvidenceLedgerSnapshot {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            ledger_epoch: 1,
            last_assigned_sequence: last,
            manifest: QualificationManifest::new([], []),
            completeness: EvidenceCompleteness::Complete,
            stats: LedgerStats {
                assigned_sequences: last,
                stored_events: events.len() as u64,
                ..LedgerStats::default()
            },
            events,
        }
    }

    #[test]
    fn exact_count_detects_under_instrumentation_even_when_kind_is_present() {
        let contract = ScenarioContract::new("ten-ingress")
            .expect(ShadowEventKind::IngressObserved, EventCountExpectation::Exactly(10));
        let snapshot = snapshot(vec![event(
            1,
            ShadowEventKind::IngressObserved,
            ShadowEventPayload::Ingress {
                ingress_class: IngressClass::LegacyUnclassified,
            },
        )]);

        let violations = validate_expected_event_counts(&contract, &snapshot);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].observed, 1);
        assert_eq!(
            violations[0].expectation,
            EventCountExpectation::Exactly(10)
        );
    }

    #[test]
    fn must_be_zero_catches_unexpected_stage() {
        let contract = ScenarioContract::new("no-eviction")
            .expect(ShadowEventKind::WorkingMemoryEvictionObserved, EventCountExpectation::MustBeZero);
        let snapshot = snapshot(vec![event(
            1,
            ShadowEventKind::WorkingMemoryEvictionObserved,
            ShadowEventPayload::EvictionObserved {
                evicted_item_ref: Digest32([7; 32]),
            },
        )]);

        assert_eq!(validate_expected_event_counts(&contract, &snapshot).len(), 1);
    }

    #[test]
    fn relational_count_compares_independent_stage_counts() {
        let contract = ScenarioContract::new("pair-counts").expect(
            ShadowEventKind::GoalActivationObserved,
            EventCountExpectation::ExactlySameAs(ShadowEventKind::GoalActivationEvaluated),
        );

        let snapshot = snapshot(vec![
            event(
                1,
                ShadowEventKind::IngressObserved,
                ShadowEventPayload::Ingress {
                    ingress_class: IngressClass::LegacyUnclassified,
                },
            ),
            event(
                2,
                ShadowEventKind::GoalActivationObserved,
                ShadowEventPayload::MutationObserved { applied: true },
            ),
        ]);

        let violations = validate_expected_event_counts(&contract, &snapshot);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].observed, 1);
        assert_eq!(violations[0].reference_observed, Some(0));
    }

    #[test]
    fn canonical_contracts_encode_the_documented_scenarios() {
        let no_eviction = goal_no_eviction_v0();
        assert_eq!(
            no_eviction.expectations.get(&ShadowEventKind::GoalActivationObserved),
            Some(&EventCountExpectation::Exactly(1))
        );
        assert_eq!(
            no_eviction.expectations.get(&ShadowEventKind::GraduationObserved),
            Some(&EventCountExpectation::MustBeZero)
        );

        let eviction = goal_with_eviction_v0();
        assert_eq!(
            eviction.expectations.get(&ShadowEventKind::GraduationObserved),
            Some(&EventCountExpectation::Exactly(1))
        );

        let feedback = feedback_input_v0();
        assert_eq!(
            feedback.expectations.get(&ShadowEventKind::AffectMutationObserved),
            Some(&EventCountExpectation::Exactly(1))
        );
        assert_eq!(
            feedback.expectations.get(&ShadowEventKind::GoalActivationObserved),
            Some(&EventCountExpectation::MustBeZero)
        );
    }
}
