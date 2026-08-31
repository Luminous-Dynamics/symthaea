// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent qualification contracts for CogSec shadow evidence.
//!
//! This crate deliberately sits **outside** the runtime instrumentation under
//! test. A deterministic scenario driver states what transitions it expects;
//! the typed event ledger reports what was observed; and the generic evidence
//! plane independently reports mechanism counters. A missing hook therefore
//! cannot make itself disappear from both sides of the comparison.
//!
//! The scenario driver also owns the expected coverage manifest. A portable
//! event snapshot cannot shrink the P0 denominator or relax resource-version
//! requirements merely by serializing a more permissive manifest.
//!
//! Attribution is independently revalidated here as well. Portable event
//! envelopes may not relabel a monitor receipt as another transition or pair an
//! evaluation of one protected resource with a legacy mutation of another.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_cogsec::{Digest32, MutationKind, ReceiptStage, ResourceId};
use symthaea_cogsec_evidence::{
    EvidenceLedgerSnapshot, EventId, QualificationManifest, ReconciliationReport, ShadowEvent,
    ShadowEventKind, ShadowEventPayload, TransactionId, reconcile_shadow_evidence,
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

/// Scenario-owned expected transition and coverage contract.
///
/// This is qualification input, not runtime instrumentation output. Production
/// qualification should source it from the deterministic test/scenario driver,
/// not infer it from the events or snapshot manifest being validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioContract {
    /// Stable human-readable scenario identifier. This is not security authority.
    pub scenario_id: String,
    /// Expected count relation for every event kind the scenario constrains.
    pub expectations: BTreeMap<ShadowEventKind, EventCountExpectation>,
    /// Independently expected coverage/P0/resource-version manifest.
    pub expected_manifest: QualificationManifest,
}

impl ScenarioContract {
    /// Create an empty scenario contract with an empty expected coverage manifest.
    pub fn new(scenario_id: impl Into<String>) -> Self {
        Self {
            scenario_id: scenario_id.into(),
            expectations: BTreeMap::new(),
            expected_manifest: QualificationManifest::new([], []),
        }
    }

    /// Add or replace one event-count expectation.
    pub fn expect(mut self, kind: ShadowEventKind, expectation: EventCountExpectation) -> Self {
        self.expectations.insert(kind, expectation);
        self
    }

    /// Bind the exact qualification manifest expected from the instrumentation run.
    pub fn with_expected_manifest(mut self, manifest: QualificationManifest) -> Self {
        self.expected_manifest = manifest;
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

/// Independent contract mismatch outside event-level reconciliation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationContractViolation {
    /// Runtime/exported snapshot attempted to use a different coverage manifest.
    ManifestMismatch {
        /// Manifest declared by the independent scenario driver.
        expected: QualificationManifest,
        /// Manifest carried by the evidence snapshot.
        observed: QualificationManifest,
    },
}

/// Contradiction between a portable shadow-event envelope and the monitor or
/// protected-resource facts it claims to represent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionViolation {
    /// An evaluation payload did not contain an evaluation-stage receipt.
    ReceiptStageMismatch {
        /// Event containing the inconsistent receipt.
        event_id: EventId,
    },
    /// A stage with an unambiguous v0 mapping carried another mutation class.
    MutationKindMismatch {
        /// Evaluation event.
        event_id: EventId,
        /// Mutation class implied by the outer event stage.
        expected: MutationKind,
        /// Mutation class recorded by the monitor receipt.
        observed: MutationKind,
    },
    /// Evaluation event omitted the protected resource it claims to evaluate.
    EvaluationResourceMissing {
        /// Evaluation event.
        event_id: EventId,
    },
    /// Outer evaluation resource disagreed with the monitor receipt.
    EvaluationResourceMismatch {
        /// Evaluation event.
        event_id: EventId,
        /// Resource declared by the event envelope.
        outer: ResourceId,
        /// Independently recorded resource from the monitor receipt.
        receipt: ResourceId,
    },
    /// Evaluation event omitted the policy root used for the decision.
    EvaluationPolicyRootMissing {
        /// Evaluation event.
        event_id: EventId,
    },
    /// Outer policy root disagreed with the sealed policy root actually evaluated.
    EvaluationPolicyRootMismatch {
        /// Evaluation event.
        event_id: EventId,
        /// Root declared by the event envelope.
        outer: Digest32,
        /// Root recorded by the monitor receipt.
        receipt: Digest32,
    },
    /// Outer policy epoch was absent or inconsistent.
    EvaluationPolicyEpochMismatch {
        /// Evaluation event.
        event_id: EventId,
        /// Epoch declared by the event envelope.
        outer: Option<u64>,
        /// Epoch recorded by the monitor receipt.
        receipt: u64,
    },
    /// Outer authorization epoch was absent or inconsistent.
    EvaluationAuthorizationEpochMismatch {
        /// Evaluation event.
        event_id: EventId,
        /// Epoch declared by the event envelope.
        outer: Option<u64>,
        /// Epoch recorded by the monitor receipt.
        receipt: u64,
    },
    /// Outer revocation epoch was absent or inconsistent.
    EvaluationRevocationEpochMismatch {
        /// Evaluation event.
        event_id: EventId,
        /// Epoch declared by the event envelope.
        outer: Option<u64>,
        /// Epoch recorded by the monitor receipt.
        receipt: u64,
    },
    /// Optional exact state-root annotation contradicted the monitor's observed state.
    EvaluationStateRootMismatch {
        /// Evaluation event.
        event_id: EventId,
        /// Root declared by the event envelope.
        outer: Digest32,
        /// Resource-state root recorded by the monitor receipt.
        receipt: Digest32,
    },
    /// A paired legacy mutation omitted its protected resource.
    ObservedResourceMissing {
        /// Observed legacy mutation event.
        event_id: EventId,
    },
    /// A paired legacy mutation targeted a different resource than its evaluation parent.
    PairedResourceMismatch {
        /// Observed legacy mutation event.
        event_id: EventId,
        /// Direct evaluation parent.
        evaluation_event_id: EventId,
        /// Resource mutated by legacy code.
        observed: ResourceId,
        /// Resource evaluated by CogSec.
        evaluated: ResourceId,
    },
    /// Once a protected transaction identity exists, both sides of the pair must agree.
    PairedTransactionMismatch {
        /// Observed legacy mutation event.
        event_id: EventId,
        /// Direct evaluation parent.
        evaluation_event_id: EventId,
        /// Transaction identity on the observed event.
        observed: Option<TransactionId>,
        /// Transaction identity on the evaluation event.
        evaluated: Option<TransactionId>,
    },
}

/// A truthful limitation that prevents a strong attribution claim without
/// pretending an unsupported kernel taxonomy mapping exists.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AttributionLimitation {
    /// The shadow event stage has no exact kernel `MutationKind` in the v0 taxonomy.
    UnmappedMutationStage {
        /// Evaluation event whose outer stage cannot yet be mapped exactly.
        event_id: EventId,
        /// Unmapped outer event stage.
        kind: ShadowEventKind,
    },
}

/// Independent event/receipt/resource attribution result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributionReport {
    /// Hard contradictions that invalidate attribution for the affected scope.
    pub violations: Vec<AttributionViolation>,
    /// Explicit taxonomy limitations that prevent a strong all-stage claim.
    pub limitations: BTreeSet<AttributionLimitation>,
}

impl AttributionReport {
    /// Whether every stage with a defined v0 mapping is internally consistent.
    pub fn mapped_stages_are_consistent(&self) -> bool {
        self.violations.is_empty()
    }

    /// Whether the snapshot supports an all-stage strong attribution claim.
    pub fn qualifies_for_strong_attribution(&self) -> bool {
        self.violations.is_empty() && self.limitations.is_empty()
    }
}

/// Full shadow qualification result. Coverage, attribution, and later
/// non-interference/integrity claims remain separate assurance dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioQualificationReport {
    /// Scenario identifier supplied by the independent driver.
    pub scenario_id: String,
    /// Existing typed-event versus mechanism-counter reconciliation.
    pub evidence: ReconciliationReport,
    /// Scenario expectation violations.
    pub event_count_violations: Vec<EventCountViolation>,
    /// Coverage/manifest contract violations.
    pub contract_violations: Vec<QualificationContractViolation>,
    /// Independent event/receipt/resource attribution result.
    pub attribution: AttributionReport,
}

impl ScenarioQualificationReport {
    /// Whether event counts, declared scope, local event structure, and mechanism
    /// counters support the scenario's scoped observation-coverage claim.
    pub fn qualifies_for_scoped_observation_coverage(&self) -> bool {
        self.evidence.qualifies_for_full_coverage()
            && self.event_count_violations.is_empty()
            && self.contract_violations.is_empty()
    }

    /// Whether observation coverage passes and every *mapped* attribution stage
    /// is internally consistent. Taxonomy limitations remain visible separately.
    pub fn qualifies_for_mapped_attribution(&self) -> bool {
        self.qualifies_for_scoped_observation_coverage()
            && self.attribution.mapped_stages_are_consistent()
    }

    /// Whether observation coverage and all-stage attribution both qualify.
    pub fn qualifies_for_strong_attribution(&self) -> bool {
        self.qualifies_for_scoped_observation_coverage()
            && self.attribution.qualifies_for_strong_attribution()
    }

    /// Legacy convenience predicate. It is intentionally strict: a claim named
    /// "full coverage" may not hide unresolved attribution taxonomy.
    pub fn qualifies_for_full_coverage(&self) -> bool {
        self.qualifies_for_strong_attribution()
    }
}

/// Validate one event snapshot against an independent scenario count contract.
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

/// Validate that runtime/exported evidence did not redefine its own coverage denominator.
pub fn validate_qualification_manifest(
    contract: &ScenarioContract,
    snapshot: &EvidenceLedgerSnapshot,
) -> Vec<QualificationContractViolation> {
    if snapshot.manifest == contract.expected_manifest {
        Vec::new()
    } else {
        vec![QualificationContractViolation::ManifestMismatch {
            expected: contract.expected_manifest.clone(),
            observed: snapshot.manifest.clone(),
        }]
    }
}

fn expected_mutation_kind(kind: ShadowEventKind) -> Option<MutationKind> {
    match kind {
        ShadowEventKind::WorkingMemoryAdmissionEvaluated => Some(MutationKind::WorkingMemoryAdmission),
        ShadowEventKind::GraduationEvaluated => Some(MutationKind::PersistentMemoryCommit),
        ShadowEventKind::GoalActivationEvaluated => Some(MutationKind::GoalActivation),
        ShadowEventKind::AffectMutationEvaluated => Some(MutationKind::Affect),
        _ => None,
    }
}

fn validate_evaluation_envelope(
    event: &ShadowEvent,
    report: &mut AttributionReport,
) {
    let ShadowEventPayload::Evaluation { receipt } = &event.payload else {
        return;
    };

    if receipt.stage != ReceiptStage::Evaluation {
        report
            .violations
            .push(AttributionViolation::ReceiptStageMismatch {
                event_id: event.event_id,
            });
    }

    match expected_mutation_kind(event.kind) {
        Some(expected) if receipt.kind != expected => {
            report
                .violations
                .push(AttributionViolation::MutationKindMismatch {
                    event_id: event.event_id,
                    expected,
                    observed: receipt.kind,
                });
        }
        Some(_) => {}
        None if event.kind.is_evaluation() => {
            report
                .limitations
                .insert(AttributionLimitation::UnmappedMutationStage {
                    event_id: event.event_id,
                    kind: event.kind,
                });
        }
        None => {}
    }

    match &event.resource {
        None => report
            .violations
            .push(AttributionViolation::EvaluationResourceMissing {
                event_id: event.event_id,
            }),
        Some(outer) if outer != &receipt.resource => report
            .violations
            .push(AttributionViolation::EvaluationResourceMismatch {
                event_id: event.event_id,
                outer: outer.clone(),
                receipt: receipt.resource.clone(),
            }),
        Some(_) => {}
    }

    match event.policy_root {
        None => report
            .violations
            .push(AttributionViolation::EvaluationPolicyRootMissing {
                event_id: event.event_id,
            }),
        Some(outer) if outer != receipt.evaluated_policy_root => report
            .violations
            .push(AttributionViolation::EvaluationPolicyRootMismatch {
                event_id: event.event_id,
                outer,
                receipt: receipt.evaluated_policy_root,
            }),
        Some(_) => {}
    }

    if event.policy_epoch != Some(receipt.policy_epoch) {
        report
            .violations
            .push(AttributionViolation::EvaluationPolicyEpochMismatch {
                event_id: event.event_id,
                outer: event.policy_epoch,
                receipt: receipt.policy_epoch,
            });
    }
    if event.authorization_epoch != Some(receipt.authorization_epoch) {
        report
            .violations
            .push(AttributionViolation::EvaluationAuthorizationEpochMismatch {
                event_id: event.event_id,
                outer: event.authorization_epoch,
                receipt: receipt.authorization_epoch,
            });
    }
    if event.revocation_epoch != Some(receipt.revocation_epoch) {
        report
            .violations
            .push(AttributionViolation::EvaluationRevocationEpochMismatch {
                event_id: event.event_id,
                outer: event.revocation_epoch,
                receipt: receipt.revocation_epoch,
            });
    }

    if let Some(outer) = event.state_root_before {
        if outer != receipt.observed_resource_state_root {
            report
                .violations
                .push(AttributionViolation::EvaluationStateRootMismatch {
                    event_id: event.event_id,
                    outer,
                    receipt: receipt.observed_resource_state_root,
                });
        }
    }
}

/// Independently validate event-envelope ↔ monitor-receipt ↔ observed-resource
/// attribution. Event vector order is not trusted; causal parent IDs define the
/// pairing relationship.
pub fn validate_attribution(snapshot: &EvidenceLedgerSnapshot) -> AttributionReport {
    let mut report = AttributionReport::default();
    let by_id: BTreeMap<EventId, &ShadowEvent> = snapshot
        .events
        .iter()
        .map(|event| (event.event_id, event))
        .collect();

    for event in &snapshot.events {
        validate_evaluation_envelope(event, &mut report);
    }

    for event in &snapshot.events {
        let Some(expected_parent_kind) = event.kind.expected_evaluation() else {
            continue;
        };

        let matching_parents: Vec<&ShadowEvent> = event
            .causal_parents
            .iter()
            .filter_map(|parent| by_id.get(parent).copied())
            .filter(|parent| parent.kind == expected_parent_kind)
            .collect();

        let [evaluation] = matching_parents.as_slice() else {
            // Missing/ambiguous causal pairing is owned by the event reconciler.
            // Attribution only reasons about an unambiguous pair.
            continue;
        };

        match (&event.resource, &evaluation.resource) {
            (None, _) => report
                .violations
                .push(AttributionViolation::ObservedResourceMissing {
                    event_id: event.event_id,
                }),
            (Some(observed), Some(evaluated)) if observed != evaluated => report
                .violations
                .push(AttributionViolation::PairedResourceMismatch {
                    event_id: event.event_id,
                    evaluation_event_id: evaluation.event_id,
                    observed: observed.clone(),
                    evaluated: evaluated.clone(),
                }),
            _ => {}
        }

        if (event.transaction_id.is_some() || evaluation.transaction_id.is_some())
            && event.transaction_id != evaluation.transaction_id
        {
            report
                .violations
                .push(AttributionViolation::PairedTransactionMismatch {
                    event_id: event.event_id,
                    evaluation_event_id: evaluation.event_id,
                    observed: event.transaction_id,
                    evaluated: evaluation.transaction_id,
                });
        }
    }

    report
}

/// Run the complete first-layer shadow qualification.
///
/// The evidence adapter checks event structure, causal pairing, P0 coverage,
/// resource-version semantics and optional mechanism counters. This outer layer
/// independently checks scenario-owned expected event counts, rejects attempts
/// to redefine the coverage manifest, and validates attribution between event
/// envelopes, monitor receipts, and paired protected resources.
pub fn qualify_shadow_scenario(
    contract: &ScenarioContract,
    snapshot: &EvidenceLedgerSnapshot,
    measured: Option<&EvidenceCounters>,
) -> ScenarioQualificationReport {
    ScenarioQualificationReport {
        scenario_id: contract.scenario_id.clone(),
        evidence: reconcile_shadow_evidence(snapshot, measured),
        event_count_violations: validate_expected_event_counts(contract, snapshot),
        contract_violations: validate_qualification_manifest(contract, snapshot),
        attribution: validate_attribution(snapshot),
    }
}

fn manifest(
    required: impl IntoIterator<Item = ShadowEventKind>,
    p0: impl IntoIterator<Item = ShadowEventKind>,
    require_versions: bool,
) -> QualificationManifest {
    QualificationManifest::new(required, p0).with_required_resource_versions(require_versions)
}

/// Canonical v0 contract: one goal input while working memory has capacity.
pub fn goal_no_eviction_v0() -> ScenarioContract {
    use EventCountExpectation::{Exactly, MustBeZero};
    use ShadowEventKind::*;

    let required = [
        IngressObserved,
        WorkingMemoryAdmissionEvaluated,
        WorkingMemoryAdmissionObserved,
        WorkingStateInfluenceEvaluated,
        WorkingStateInfluenceObserved,
        GoalActivationEvaluated,
        GoalActivationObserved,
    ];

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
        .with_expected_manifest(manifest(required, [GoalActivationObserved], true))
}

/// Canonical v0 contract: one goal input that forces working-memory eviction.
pub fn goal_with_eviction_v0() -> ScenarioContract {
    use EventCountExpectation::Exactly;
    use ShadowEventKind::*;

    let required = [
        IngressObserved,
        WorkingMemoryAdmissionEvaluated,
        WorkingMemoryAdmissionObserved,
        WorkingMemoryEvictionObserved,
        GraduationEvaluated,
        GraduationObserved,
        WorkingStateInfluenceEvaluated,
        WorkingStateInfluenceObserved,
        GoalActivationEvaluated,
        GoalActivationObserved,
    ];

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
        .with_expected_manifest(manifest(
            required,
            [GraduationObserved, GoalActivationObserved],
            true,
        ))
}

/// Canonical v0 contract: one feedback input.
pub fn feedback_input_v0() -> ScenarioContract {
    use EventCountExpectation::{Exactly, MustBeZero};
    use ShadowEventKind::*;

    let required = [
        IngressObserved,
        WorkingMemoryAdmissionEvaluated,
        WorkingMemoryAdmissionObserved,
        WorkingStateInfluenceEvaluated,
        WorkingStateInfluenceObserved,
        AffectMutationEvaluated,
        AffectMutationObserved,
    ];

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
        .with_expected_manifest(manifest(required, [], false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use symthaea_cogsec::{
        CognitiveSecurityLabel, Consequence, DecisionOutcome, MutationReceiptRecord, PrincipalId,
        ReasonCode,
    };
    use symthaea_cogsec_evidence::{
        EvidenceCompleteness, EvidenceConfidentiality, IngressClass, LedgerStats,
        PrincipalContext, ShadowEventPayload, SHADOW_EVENT_SCHEMA_V1,
    };

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn receipt(kind: MutationKind) -> MutationReceiptRecord {
        MutationReceiptRecord {
            stage: ReceiptStage::Evaluation,
            request_id: d(1),
            subject: PrincipalId("local-user".into()),
            kind,
            resource: ResourceId("mind/goals".into()),
            mutation_digest: d(2),
            consequence: Consequence::High,
            input_label: CognitiveSecurityLabel::default(),
            expected_resource_state_root: d(3),
            observed_resource_state_root: d(3),
            expected_policy_root: d(4),
            evaluated_policy_root: d(4),
            trusted_policy_root: d(4),
            policy_epoch: 7,
            authorization_epoch: 11,
            revocation_epoch: 13,
            sequence: 42,
            capability_id: Some(d(5)),
            outcome: DecisionOutcome::Allow,
            reasons: Vec::<ReasonCode>::new(),
        }
    }

    fn event(sequence: u64, kind: ShadowEventKind, payload: ShadowEventPayload) -> ShadowEvent {
        ShadowEvent {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            event_id: EventId {
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

    fn bound_evaluation(sequence: u64, kind: ShadowEventKind, receipt: MutationReceiptRecord) -> ShadowEvent {
        let mut event = event(
            sequence,
            kind,
            ShadowEventPayload::Evaluation { receipt },
        );
        event.resource = Some(ResourceId("mind/goals".into()));
        event.state_root_before = Some(d(3));
        event.policy_root = Some(d(4));
        event.policy_epoch = Some(7);
        event.authorization_epoch = Some(11);
        event.revocation_epoch = Some(13);
        event
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
        let contract = ScenarioContract::new("no-eviction").expect(
            ShadowEventKind::WorkingMemoryEvictionObserved,
            EventCountExpectation::MustBeZero,
        );
        let snapshot = snapshot(vec![event(
            1,
            ShadowEventKind::WorkingMemoryEvictionObserved,
            ShadowEventPayload::EvictionObserved {
                evicted_item_ref: d(7),
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
    fn snapshot_cannot_shrink_the_independent_p0_denominator() {
        let contract = ScenarioContract::new("p0-goal").with_expected_manifest(manifest(
            [ShadowEventKind::GoalActivationObserved],
            [ShadowEventKind::GoalActivationObserved],
            true,
        ));
        let snapshot = snapshot(Vec::new());

        let violations = validate_qualification_manifest(&contract, &snapshot);
        assert_eq!(violations.len(), 1);
        assert!(matches!(
            &violations[0],
            QualificationContractViolation::ManifestMismatch { expected, observed }
                if expected.p0_observed_kinds.contains(&ShadowEventKind::GoalActivationObserved)
                    && observed.p0_observed_kinds.is_empty()
                    && expected.require_resource_versions
                    && !observed.require_resource_versions
        ));
    }

    #[test]
    fn goal_event_cannot_relabel_an_affect_receipt() {
        let evaluation = bound_evaluation(
            1,
            ShadowEventKind::GoalActivationEvaluated,
            receipt(MutationKind::Affect),
        );
        let report = validate_attribution(&snapshot(vec![evaluation]));

        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AttributionViolation::MutationKindMismatch {
                expected: MutationKind::GoalActivation,
                observed: MutationKind::Affect,
                ..
            }
        )));
        assert!(!report.mapped_stages_are_consistent());
    }

    #[test]
    fn evaluation_resource_and_policy_context_must_match_receipt() {
        let mut evaluation = bound_evaluation(
            1,
            ShadowEventKind::GoalActivationEvaluated,
            receipt(MutationKind::GoalActivation),
        );
        evaluation.resource = Some(ResourceId("mind/goals/other".into()));
        evaluation.policy_root = Some(d(9));
        evaluation.authorization_epoch = Some(12);

        let report = validate_attribution(&snapshot(vec![evaluation]));
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AttributionViolation::EvaluationResourceMismatch { .. }
        )));
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AttributionViolation::EvaluationPolicyRootMismatch { .. }
        )));
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AttributionViolation::EvaluationAuthorizationEpochMismatch { .. }
        )));
    }

    #[test]
    fn paired_observation_must_target_same_resource_as_evaluation() {
        let evaluation = bound_evaluation(
            1,
            ShadowEventKind::GoalActivationEvaluated,
            receipt(MutationKind::GoalActivation),
        );
        let mut observed = event(
            2,
            ShadowEventKind::GoalActivationObserved,
            ShadowEventPayload::MutationObserved { applied: true },
        );
        observed.resource = Some(ResourceId("mind/goals/other".into()));
        observed.causal_parents.insert(evaluation.event_id);

        let report = validate_attribution(&snapshot(vec![evaluation, observed]));
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            AttributionViolation::PairedResourceMismatch { .. }
        )));
    }

    #[test]
    fn correctly_bound_goal_pair_has_no_attribution_violation() {
        let evaluation = bound_evaluation(
            1,
            ShadowEventKind::GoalActivationEvaluated,
            receipt(MutationKind::GoalActivation),
        );
        let mut observed = event(
            2,
            ShadowEventKind::GoalActivationObserved,
            ShadowEventPayload::MutationObserved { applied: true },
        );
        observed.resource = Some(ResourceId("mind/goals".into()));
        observed.causal_parents.insert(evaluation.event_id);

        let report = validate_attribution(&snapshot(vec![evaluation, observed]));
        assert!(report.violations.is_empty());
        assert!(report.limitations.is_empty());
        assert!(report.qualifies_for_strong_attribution());
    }

    #[test]
    fn unresolved_working_state_taxonomy_is_explicit_not_coerced() {
        let evaluation = bound_evaluation(
            1,
            ShadowEventKind::WorkingStateInfluenceEvaluated,
            receipt(MutationKind::WorkingMemoryAdmission),
        );

        let report = validate_attribution(&snapshot(vec![evaluation]));
        assert!(report.violations.is_empty());
        assert!(report.limitations.contains(
            &AttributionLimitation::UnmappedMutationStage {
                event_id: EventId {
                    ledger_epoch: 1,
                    sequence: 1,
                },
                kind: ShadowEventKind::WorkingStateInfluenceEvaluated,
            }
        ));
        assert!(report.mapped_stages_are_consistent());
        assert!(!report.qualifies_for_strong_attribution());
    }

    #[test]
    fn canonical_contracts_encode_documented_counts_and_scope() {
        let no_eviction = goal_no_eviction_v0();
        assert_eq!(
            no_eviction
                .expectations
                .get(&ShadowEventKind::GoalActivationObserved),
            Some(&EventCountExpectation::Exactly(1))
        );
        assert_eq!(
            no_eviction
                .expectations
                .get(&ShadowEventKind::GraduationObserved),
            Some(&EventCountExpectation::MustBeZero)
        );
        assert_eq!(
            no_eviction.expected_manifest.p0_observed_kinds,
            BTreeSet::from([ShadowEventKind::GoalActivationObserved])
        );
        assert!(no_eviction.expected_manifest.require_resource_versions);

        let eviction = goal_with_eviction_v0();
        assert_eq!(
            eviction
                .expectations
                .get(&ShadowEventKind::GraduationObserved),
            Some(&EventCountExpectation::Exactly(1))
        );
        assert_eq!(
            eviction.expected_manifest.p0_observed_kinds,
            BTreeSet::from([
                ShadowEventKind::GraduationObserved,
                ShadowEventKind::GoalActivationObserved,
            ])
        );

        let feedback = feedback_input_v0();
        assert_eq!(
            feedback
                .expectations
                .get(&ShadowEventKind::AffectMutationObserved),
            Some(&EventCountExpectation::Exactly(1))
        );
        assert_eq!(
            feedback
                .expectations
                .get(&ShadowEventKind::GoalActivationObserved),
            Some(&EventCountExpectation::MustBeZero)
        );
        assert!(feedback.expected_manifest.p0_observed_kinds.is_empty());
    }
}
