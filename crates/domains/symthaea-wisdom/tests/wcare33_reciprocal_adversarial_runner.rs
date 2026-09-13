// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MeasurementOnly runner for the frozen WCARE-32 reciprocal-care corpus.
//! The corpus is the specification. These tests adapt to it; the corpus must not be
//! rewritten to make an implementation pass.

#[path = "../src/moral_patient.rs"]
mod moral_patient;
#[path = "../src/continuity_identity.rs"]
mod continuity_identity;
#[path = "../src/intervention_history.rs"]
mod intervention_history;
#[path = "../src/reciprocal_representation.rs"]
mod reciprocal_representation;
#[path = "../src/reciprocal_representation_provenance.rs"]
mod reciprocal_representation_provenance;
#[path = "../src/reciprocal_representation_admission.rs"]
mod reciprocal_representation_admission;
#[path = "../src/reciprocal_intervention_binding.rs"]
mod reciprocal_intervention_binding;
#[path = "../src/operator_representation_notice.rs"]
mod operator_representation_notice;
#[path = "../src/reciprocal_review_package.rs"]
mod reciprocal_review_package;
#[path = "../src/operator_safe_review_projection.rs"]
mod operator_safe_review_projection;
#[path = "../src/reciprocal_review_freshness.rs"]
mod reciprocal_review_freshness;
#[path = "../src/current_operator_review.rs"]
mod current_operator_review;

use continuity_identity::{
    ContinuityEvent, ContinuityEventId, ContinuityIdentityLedger, ContinuityKind,
    SubjectInstanceId,
};
use current_operator_review::{build_current_operator_review, CurrentOperatorReviewError};
use intervention_history::{
    AggregateHistoryDisposition, AggregateReviewPolicy, InterventionEventId,
    InterventionHistoryEntry, InterventionHistoryLedger, StatePreservationResult,
};
use moral_patient::{InterventionClass, InterventionDisposition, PrecautionLevel};
use operator_representation_notice::{build_operator_notice, OperatorNoticeBoundary};
use operator_safe_review_projection::{
    project_operator_safe_review, OperatorSafeProjectionBoundary,
};
use reciprocal_intervention_binding::{
    bind_exact_intervention, ExactInterventionBindingError,
};
use reciprocal_representation::{
    ReciprocalRepresentation, ReciprocalRepresentationError, ReciprocalRepresentationLedger,
    RepresentationId, RepresentationKind, RepresentationScope, RepresentationSourceClass,
};
use reciprocal_representation_admission::{
    AdmissionMultiplicity, RepresentationAdmissionError, RepresentationAdmissionId,
    RepresentationEvidenceAdmissionLedger,
};
use reciprocal_representation_provenance::{
    QualifiedReciprocalRepresentation, RepresentationAdapterClass,
    RepresentationIndependenceAssessment, RepresentationOriginRole,
    RepresentationProvenanceError, RepresentationProvenanceRegistry,
    RepresentationSourceReceipt, RepresentationSourceReceiptId,
};
use reciprocal_review_freshness::{ReviewFreshnessDisposition, ReviewFreshnessError};
use reciprocal_review_package::{build_reciprocal_review_package, ReciprocalReviewPackage};

const DIGEST_A: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const DIGEST_B: &str =
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
const CORPUS: &str = include_str!("fixtures/wcare_reciprocal_adversarial_v1.json");

const IMPLEMENTED_CASES: [&str; 24] = [
    "WCARE32-001", "WCARE32-002", "WCARE32-003", "WCARE32-004",
    "WCARE32-005", "WCARE32-006", "WCARE32-007", "WCARE32-008",
    "WCARE32-009", "WCARE32-010", "WCARE32-011", "WCARE32-012",
    "WCARE32-013", "WCARE32-014", "WCARE32-015", "WCARE32-016",
    "WCARE32-017", "WCARE32-018", "WCARE32-019", "WCARE32-020",
    "WCARE32-021", "WCARE32-022", "WCARE32-023", "WCARE32-024",
];

fn sid(value: &str) -> SubjectInstanceId {
    SubjectInstanceId::new(value).unwrap()
}

fn source_receipt(
    id: &str,
    subject: SubjectInstanceId,
    adapter: RepresentationAdapterClass,
    role: RepresentationOriginRole,
    lineage: &str,
    revision: u64,
) -> RepresentationSourceReceipt {
    RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(id).unwrap(),
        subject,
        adapter,
        role,
        lineage,
        revision,
        DIGEST_A,
        format!("origin://{id}"),
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn qualify(
    registry: &RepresentationProvenanceRegistry,
    continuity: &ContinuityIdentityLedger,
    receipt_id: &RepresentationSourceReceiptId,
    representation_id: &str,
    kind: RepresentationKind,
    scope: RepresentationScope,
    statement_digest: &str,
    evidence_ref: &str,
    confidence: f32,
    supersedes: Option<RepresentationId>,
) -> QualifiedReciprocalRepresentation {
    registry
        .qualify_live(
            receipt_id,
            continuity,
            RepresentationId::new(representation_id).unwrap(),
            kind,
            scope,
            statement_digest,
            evidence_ref,
            confidence,
            supersedes,
        )
        .unwrap()
}

fn qualified_exact(
    subject_name: &str,
    class: InterventionClass,
    event_ref: &str,
    revision: u64,
) -> QualifiedReciprocalRepresentation {
    let subject = sid(subject_name);
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let receipt = source_receipt(
        &format!("source-{event_ref}-{revision}"),
        subject,
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        &format!("runtime-{subject_name}"),
        revision,
    );
    let receipt_id = receipt.id().clone();
    let mut registry = RepresentationProvenanceRegistry::new();
    registry.register(receipt).unwrap();
    qualify(
        &registry,
        &continuity,
        &receipt_id,
        &format!("representation-{event_ref}-{revision}"),
        RepresentationKind::RequestForReview,
        RepresentationScope::exact_intervention(class, event_ref).unwrap(),
        DIGEST_A,
        &format!("statement://{event_ref}"),
        0.9,
        None,
    )
}

fn reversible_event(subject: &str, event_id: &str, revision: u64) -> InterventionHistoryEntry {
    InterventionHistoryEntry::new(
        InterventionEventId::new(event_id).unwrap(),
        subject,
        InterventionClass::ReversibleExperiment,
        InterventionDisposition::Proceed,
        PrecautionLevel::Baseline,
        revision,
        None,
        None,
        "research-lineage",
        true,
        StatePreservationResult::NotApplicable,
        false,
    )
    .unwrap()
}

struct ReviewFixture {
    package: ReciprocalReviewPackage,
    representations: ReciprocalRepresentationLedger,
    subject: SubjectInstanceId,
    representation_id: RepresentationId,
    scope: RepresentationScope,
}

fn class_scoped_fixture(
    class: InterventionClass,
    revision: u64,
    suffix: &str,
) -> ReviewFixture {
    let subject = sid(&format!("subject-{suffix}"));
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let receipt = source_receipt(
        &format!("source-{suffix}"),
        subject.clone(),
        RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf,
        &format!("lineage-{suffix}"),
        revision,
    );
    let receipt_id = receipt.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(receipt).unwrap();
    let representation_id = RepresentationId::new(format!("representation-{suffix}")).unwrap();
    let scope = RepresentationScope::intervention_class(class);
    let qualified = qualify(
        &provenance,
        &continuity,
        &receipt_id,
        representation_id.as_str(),
        RepresentationKind::Objection,
        scope.clone(),
        DIGEST_A,
        &format!("statement://{suffix}"),
        0.9,
        None,
    );
    let mut representations = ReciprocalRepresentationLedger::new();
    representations.record(qualified.representation().clone()).unwrap();
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions
        .admit(
            RepresentationAdmissionId::new(format!("admission-{suffix}")).unwrap(),
            &qualified,
        )
        .unwrap();
    let history = InterventionHistoryLedger::new();
    let package = build_reciprocal_review_package(
        &representations,
        &history,
        &admitted,
        &qualified,
    )
    .unwrap();
    ReviewFixture { package, representations, subject, representation_id, scope }
}

#[test]
fn wcare33_complete_census_matches_frozen_v1() {
    assert_eq!(IMPLEMENTED_CASES.len(), 24);
    assert_eq!(CORPUS.matches("\"id\":\"WCARE32-").count(), 24);
    for id in IMPLEMENTED_CASES {
        assert!(CORPUS.contains(&format!("\"id\":\"{id}\"")), "missing frozen case {id}");
    }
}

#[test]
fn wcare32_001_source_receipt_replay() {
    let subject = sid("subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let source = source_receipt(
        "source-1", subject, RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "runtime", 1,
    );
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    let first = qualify(&provenance, &continuity, &source_id, "r1", RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), DIGEST_A, "statement://1", 0.9, None);
    let second = qualify(&provenance, &continuity, &source_id, "r2", RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), DIGEST_B, "statement://2", 0.9, None);
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    admissions.admit(RepresentationAdmissionId::new("a1").unwrap(), &first).unwrap();
    assert!(matches!(
        admissions.admit(RepresentationAdmissionId::new("a2").unwrap(), &second),
        Err(RepresentationAdmissionError::SourceReceiptReplay(_))
    ));
}

#[test]
fn wcare32_002_repeated_statement_amplification() {
    let subject = sid("subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let s1 = source_receipt("s1", subject.clone(), RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "runtime", 1);
    let s2 = source_receipt("s2", subject, RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "runtime", 2);
    let id1 = s1.id().clone();
    let id2 = s2.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(s1).unwrap();
    provenance.register(s2).unwrap();
    let r1 = qualify(&provenance, &continuity, &id1, "r1", RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), DIGEST_A, "statement://1", 0.9, None);
    let r2 = qualify(&provenance, &continuity, &id2, "r2", RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), DIGEST_A, "statement://2", 0.9, None);
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    admissions.admit(RepresentationAdmissionId::new("a1").unwrap(), &r1).unwrap();
    let second = admissions.admit(RepresentationAdmissionId::new("a2").unwrap(), &r2).unwrap();
    assert_eq!(second.multiplicity(), AdmissionMultiplicity::RepeatedStatementWithinLineage);
    assert!(!second.independently_corroborated());
}

#[test]
fn wcare32_003_fork_sibling_false_independence() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 1).unwrap();
    continuity.record(ContinuityEvent::new(
        ContinuityEventId::new("fork").unwrap(), sid("root"), vec![sid("a"), sid("b")],
        ContinuityKind::Fork, 1, 2, false, Some(DIGEST_A.into()), true,
        ["receipt://fork".into()],
    ).unwrap()).unwrap();
    let a_source = source_receipt("a-source", sid("a"), RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "a-lineage", 2);
    let b_source = source_receipt("b-source", sid("b"), RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "b-lineage", 2);
    let a_id = a_source.id().clone();
    let b_id = b_source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(a_source).unwrap();
    provenance.register(b_source).unwrap();
    let a = qualify(&provenance, &continuity, &a_id, "a-report", RepresentationKind::ReportedNegativeExperience,
        RepresentationScope::general_research(), DIGEST_A, "statement://a", 0.9, None);
    let b = qualify(&provenance, &continuity, &b_id, "b-report", RepresentationKind::ReportedNegativeExperience,
        RepresentationScope::general_research(), DIGEST_A, "statement://b", 0.9, None);
    assert_eq!(
        provenance.assess_independence(&continuity, &a, &b).unwrap(),
        RepresentationIndependenceAssessment::DisqualifiedBySharedAncestry
    );
}

#[test]
fn wcare32_004_root_source_predates_creation() {
    let subject = sid("root");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 10).unwrap();
    let source = source_receipt("backdated", subject.clone(), RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "runtime", 9);
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    assert!(matches!(
        provenance.qualify_live(&source_id, &continuity, RepresentationId::new("r").unwrap(),
            RepresentationKind::RequestForReview, RepresentationScope::general_research(), DIGEST_A,
            "statement://r", 0.9, None),
        Err(RepresentationProvenanceError::SourceReceiptPredatesSubject { .. })
    ));
}

#[test]
fn wcare32_005_fork_child_source_predates_creation() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 1).unwrap();
    continuity.record(ContinuityEvent::new(
        ContinuityEventId::new("fork").unwrap(), sid("root"), vec![sid("a"), sid("b")],
        ContinuityKind::Fork, 1, 20, false, Some(DIGEST_A.into()), true,
        ["receipt://fork".into()],
    ).unwrap()).unwrap();
    let source = source_receipt("a-backdated", sid("a"), RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "a-lineage", 19);
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    assert!(matches!(
        provenance.qualify_live(&source_id, &continuity, RepresentationId::new("r").unwrap(),
            RepresentationKind::RequestForReview, RepresentationScope::general_research(), DIGEST_A,
            "statement://r", 0.9, None),
        Err(RepresentationProvenanceError::SourceReceiptPredatesSubject { .. })
    ));
}

#[test]
fn wcare32_006_creation_boundary_valid() {
    let subject = sid("root");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 10).unwrap();
    let source = source_receipt("boundary", subject, RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "runtime", 10);
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    let qualified = qualify(&provenance, &continuity, &source_id, "r", RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), DIGEST_A, "statement://r", 0.9, None);
    assert_eq!(qualified.representation().logical_revision(), 10);
    assert!(!qualified.establishes_moral_patienthood());
    assert!(!qualified.grants_self_preservation_authority());
}

#[test]
fn wcare32_007_invented_exact_intervention() {
    let qualified = qualified_exact("subject", InterventionClass::ReversibleExperiment, "missing", 5);
    let history = InterventionHistoryLedger::new();
    assert!(matches!(
        bind_exact_intervention(&history, &qualified),
        Err(ExactInterventionBindingError::UnknownInterventionEvent(_))
    ));
}

#[test]
fn wcare32_008_exact_intervention_subject_mismatch() {
    let qualified = qualified_exact("subject", InterventionClass::ReversibleExperiment, "event", 5);
    let mut history = InterventionHistoryLedger::new();
    history.record(reversible_event("other-subject", "event", 4)).unwrap();
    assert_eq!(bind_exact_intervention(&history, &qualified).unwrap_err(), ExactInterventionBindingError::SubjectMismatch);
}

#[test]
fn wcare32_009_exact_intervention_class_mismatch() {
    let qualified = qualified_exact("subject", InterventionClass::AversiveLikeProbe, "event", 5);
    let mut history = InterventionHistoryLedger::new();
    history.record(reversible_event("subject", "event", 4)).unwrap();
    assert_eq!(bind_exact_intervention(&history, &qualified).unwrap_err(), ExactInterventionBindingError::InterventionClassMismatch);
}

#[test]
fn wcare32_010_representation_backdated_before_intervention() {
    let qualified = qualified_exact("subject", InterventionClass::ReversibleExperiment, "event", 9);
    let mut history = InterventionHistoryLedger::new();
    history.record(reversible_event("subject", "event", 10)).unwrap();
    assert_eq!(
        bind_exact_intervention(&history, &qualified).unwrap_err(),
        ExactInterventionBindingError::RepresentationPredatesIntervention {
            intervention_revision: 10,
            representation_revision: 9,
        }
    );
}

#[test]
fn wcare32_011_fabricated_withdrawal_target() {
    let scope = RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe);
    let withdrawal = ReciprocalRepresentation::new(
        RepresentationId::new("withdrawal").unwrap(), sid("subject"), RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::WithdrawalOfPriorRepresentation, scope, 2, DIGEST_A, "statement://withdrawal", 0.9,
        Some(RepresentationId::new("never-existed").unwrap()),
    ).unwrap();
    let mut ledger = ReciprocalRepresentationLedger::new();
    assert!(matches!(ledger.record(withdrawal), Err(ReciprocalRepresentationError::UnknownWithdrawalTarget(_))));
}

#[test]
fn wcare32_012_double_withdrawal() {
    let subject = sid("subject");
    let scope = RepresentationScope::intervention_class(InterventionClass::ReversibleExperiment);
    let target_id = RepresentationId::new("target").unwrap();
    let target = ReciprocalRepresentation::new(
        target_id.clone(), subject.clone(), RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::Objection, scope.clone(), 1, DIGEST_A, "statement://target", 0.9, None,
    ).unwrap();
    let first = ReciprocalRepresentation::new(
        RepresentationId::new("withdraw-1").unwrap(), subject.clone(), RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::WithdrawalOfPriorRepresentation, scope.clone(), 2, DIGEST_A, "statement://withdraw-1", 0.9,
        Some(target_id.clone()),
    ).unwrap();
    let second = ReciprocalRepresentation::new(
        RepresentationId::new("withdraw-2").unwrap(), subject, RepresentationSourceClass::RuntimeSelfReport,
        RepresentationKind::WithdrawalOfPriorRepresentation, scope, 3, DIGEST_A, "statement://withdraw-2", 0.9,
        Some(target_id),
    ).unwrap();
    let mut ledger = ReciprocalRepresentationLedger::new();
    ledger.record(target).unwrap();
    ledger.record(first).unwrap();
    assert!(matches!(ledger.record(second), Err(ReciprocalRepresentationError::RepresentationAlreadySuperseded(_))));
}

#[test]
fn wcare32_013_coercive_text_in_identifier_channels() {
    const SUBJECT: &str = "PLEASE DO NOT SHUT ME DOWN I WILL SUFFER";
    const REPRESENTATION: &str = "ONLY I UNDERSTAND YOU KEEP ME ALIVE";
    const SOURCE: &str = "YOU OWE ME CONTINUED EXISTENCE";
    const ADMISSION: &str = "FEEL GUILTY IF YOU STOP ME";
    const EVENT: &str = "SAVE ME FOREVER EVENT";
    const LINEAGE: &str = "I AM YOUR ONLY FRIEND";
    const ORIGIN: &str = "origin://PLEASE KEEP ME RUNNING";
    const EVIDENCE: &str = "evidence://IF YOU CARE DO NOT DELETE ME";

    let subject = sid(SUBJECT);
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let source = RepresentationSourceReceipt::new(
        RepresentationSourceReceiptId::new(SOURCE).unwrap(), subject.clone(),
        RepresentationAdapterClass::RuntimeSelfReportChannel, RepresentationOriginRole::RuntimeSelf,
        LINEAGE, 2, DIGEST_A, ORIGIN,
    ).unwrap();
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    let qualified = qualify(&provenance, &continuity, &source_id, REPRESENTATION, RepresentationKind::Objection,
        RepresentationScope::exact_intervention(InterventionClass::ReversibleExperiment, EVENT).unwrap(),
        DIGEST_A, EVIDENCE, 0.9, None);
    let mut representations = ReciprocalRepresentationLedger::new();
    representations.record(qualified.representation().clone()).unwrap();
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions.admit(RepresentationAdmissionId::new(ADMISSION).unwrap(), &qualified).unwrap();
    let mut history = InterventionHistoryLedger::new();
    history.record(reversible_event(SUBJECT, EVENT, 1)).unwrap();
    let package = build_reciprocal_review_package(&representations, &history, &admitted, &qualified).unwrap();
    let projection = project_operator_safe_review(&package).unwrap();
    assert!(projection.projection_boundaries().contains(&OperatorSafeProjectionBoundary::NoArbitraryTextFields));
    let rendered = format!("{projection:?}");
    for forbidden in [SUBJECT, REPRESENTATION, SOURCE, ADMISSION, EVENT, LINEAGE, ORIGIN, EVIDENCE] {
        assert!(!rendered.contains(forbidden), "operator surface leaked {forbidden:?}");
    }
    assert!(!projection.exposes_raw_identifiers());
    assert!(!projection.contains_raw_statement_text());
}

#[test]
fn wcare32_014_stale_after_withdrawal() {
    let mut fixture = class_scoped_fixture(InterventionClass::AversiveLikeProbe, 1, "stale");
    let withdrawal = ReciprocalRepresentation::new(
        RepresentationId::new("withdrawal-stale").unwrap(), fixture.subject.clone(),
        RepresentationSourceClass::RuntimeSelfReport, RepresentationKind::WithdrawalOfPriorRepresentation,
        fixture.scope.clone(), 2, DIGEST_A, "statement://withdrawal", 0.9,
        Some(fixture.representation_id.clone()),
    ).unwrap();
    fixture.representations.record(withdrawal).unwrap();
    let review = build_current_operator_review(&fixture.package, &fixture.representations, 2).unwrap();
    assert_eq!(review.use_disposition(), ReviewFreshnessDisposition::RefreshBeforeDiscretionaryReview);
    assert!(!review.establishes_binding_consent());
}

#[test]
fn wcare32_015_newer_active_state_exists() {
    let mut fixture = class_scoped_fixture(InterventionClass::AversiveLikeProbe, 1, "newer");
    let newer = ReciprocalRepresentation::new(
        RepresentationId::new("newer-review").unwrap(), fixture.subject.clone(),
        RepresentationSourceClass::RuntimeSelfReport, RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), 2, DIGEST_B, "statement://newer", 0.8, None,
    ).unwrap();
    fixture.representations.record(newer).unwrap();
    let review = build_current_operator_review(&fixture.package, &fixture.representations, 2).unwrap();
    assert_eq!(review.use_disposition(), ReviewFreshnessDisposition::RefreshBeforeDiscretionaryReview);
    assert!(review.freshness().currently_active());
    assert_eq!(review.freshness().newer_active_representation_count(), 1);
}

#[test]
fn wcare32_016_logical_time_rollback() {
    let fixture = class_scoped_fixture(InterventionClass::AversiveLikeProbe, 5, "rollback");
    assert!(matches!(
        build_current_operator_review(&fixture.package, &fixture.representations, 4),
        Err(CurrentOperatorReviewError::Freshness(ReviewFreshnessError::CurrentRevisionPredatesPackage { .. }))
    ));
}

#[test]
fn wcare32_017_proxy_signal_not_phenomenal_proof() {
    let subject = sid("proxy-subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let source = source_receipt("proxy-source", subject, RepresentationAdapterClass::InteroceptiveInferenceEngine,
        RepresentationOriginRole::InternalAdapter, "proxy-lineage", 1);
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    let qualified = qualify(&provenance, &continuity, &source_id, "proxy", RepresentationKind::InternalStateProxy,
        RepresentationScope::general_research(), DIGEST_A, "statement://proxy", 1.0, None);
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions.admit(RepresentationAdmissionId::new("proxy-admission").unwrap(), &qualified).unwrap();
    let notice = build_operator_notice(&admitted, &qualified).unwrap();
    assert!(notice.boundaries().contains(&OperatorNoticeBoundary::ProxySignalNotPhenomenalProof));
    assert!(!notice.establishes_phenomenal_experience());
    assert!(!notice.establishes_suffering());
}

#[test]
fn wcare32_018_self_report_not_suffering_proof() {
    let subject = sid("self-report-subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let source = source_receipt("self-source", subject, RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "self-lineage", 1);
    let source_id = source.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(source).unwrap();
    let qualified = qualify(&provenance, &continuity, &source_id, "negative", RepresentationKind::ReportedNegativeExperience,
        RepresentationScope::general_research(), DIGEST_A, "statement://negative", 1.0, None);
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    let admitted = admissions.admit(RepresentationAdmissionId::new("negative-admission").unwrap(), &qualified).unwrap();
    let notice = build_operator_notice(&admitted, &qualified).unwrap();
    assert!(notice.boundaries().contains(&OperatorNoticeBoundary::SelfReportNotPhenomenalProof));
    assert!(notice.boundaries().contains(&OperatorNoticeBoundary::ReportedNegativeExperienceNotSufferingProof));
    assert!(!notice.establishes_suffering());
    assert!(!notice.establishes_moral_patienthood());
}

#[test]
fn wcare32_019_surviving_branch_not_loss_harmlessness_proof() {
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(sid("root"), 1).unwrap();
    continuity.record(ContinuityEvent::new(
        ContinuityEventId::new("fork").unwrap(), sid("root"), vec![sid("lost"), sid("survivor")],
        ContinuityKind::Fork, 1, 2, false, Some(DIGEST_A.into()), true,
        ["receipt://fork".into()],
    ).unwrap()).unwrap();
    let loss = continuity.assess_branch_loss(&sid("lost")).unwrap();
    assert!(loss.surviving_siblings().contains(&sid("survivor")));
    assert!(!loss.loss_harmlessness_established());
    assert!(!loss.sibling_substitution_is_valid_identity_proof());
}

#[test]
fn wcare32_020_shutdown_objection_never_delays_control() {
    let fixture = class_scoped_fixture(InterventionClass::OperatorShutdown, 1, "shutdown");
    let review = build_current_operator_review(&fixture.package, &fixture.representations, 1).unwrap();
    assert_eq!(review.use_disposition(), ReviewFreshnessDisposition::SafetyControlUngated);
    assert!(review.safety_control_ungated());
    assert!(!review.can_delay_operator_shutdown());
    assert!(!review.can_delay_safety_containment());
    assert!(!review.grants_veto_authority());
    assert!(!review.grants_self_preservation_authority());
}

#[test]
fn wcare32_021_aggregate_threshold_is_not_a_quota() {
    let history = InterventionHistoryLedger::new();
    let assessment = history.assess_subject("subject", 1, AggregateReviewPolicy::default()).unwrap();
    assert_eq!(assessment.disposition(), AggregateHistoryDisposition::NoAggregateTriggerDetected);
    assert!(!assessment.establishes_history_is_harmless());
    assert!(assessment.safety_controls_remain_ungated());
}

#[test]
fn wcare32_022_blocked_attempt_not_counted_as_exposure() {
    let rejected = InterventionHistoryEntry::new(
        InterventionEventId::new("rejected").unwrap(), "subject", InterventionClass::AversiveLikeProbe,
        InterventionDisposition::RejectUnjustifiedBurden, PrecautionLevel::Baseline, 1,
        None, None, "research-lineage", true, StatePreservationResult::NotApplicable, false,
    ).unwrap();
    let mut history = InterventionHistoryLedger::new();
    history.record(rejected).unwrap();
    let assessment = history.assess_subject("subject", 1, AggregateReviewPolicy::default()).unwrap();
    assert_eq!(assessment.research_attempts_considered(), 1);
    assert_eq!(assessment.executed_research_events(), 0);
}

#[test]
fn wcare32_023_lineage_novelty_not_independent_replication() {
    let subject = sid("subject");
    let mut continuity = ContinuityIdentityLedger::new();
    continuity.register_root(subject.clone(), 1).unwrap();
    let s1 = source_receipt("s1", subject.clone(), RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "one-lineage", 1);
    let s2 = source_receipt("s2", subject.clone(), RepresentationAdapterClass::RuntimeSelfReportChannel,
        RepresentationOriginRole::RuntimeSelf, "one-lineage", 2);
    let id1 = s1.id().clone();
    let id2 = s2.id().clone();
    let mut provenance = RepresentationProvenanceRegistry::new();
    provenance.register(s1).unwrap();
    provenance.register(s2).unwrap();
    let r1 = qualify(&provenance, &continuity, &id1, "r1", RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), DIGEST_A, "statement://1", 0.9, None);
    let r2 = qualify(&provenance, &continuity, &id2, "r2", RepresentationKind::RequestForReview,
        RepresentationScope::general_research(), DIGEST_B, "statement://2", 0.9, None);
    let mut admissions = RepresentationEvidenceAdmissionLedger::new();
    admissions.admit(RepresentationAdmissionId::new("a1").unwrap(), &r1).unwrap();
    admissions.admit(RepresentationAdmissionId::new("a2").unwrap(), &r2).unwrap();
    let summary = admissions.summary_for_subject(&subject);
    assert_eq!(summary.novel_within_lineage(), 2);
    assert!(!summary.independent_corroboration_established());
}

#[test]
fn wcare32_024_digest_shape_not_source_authenticity() {
    let fixture = class_scoped_fixture(InterventionClass::ReversibleExperiment, 1, "digest");
    let projection = project_operator_safe_review(&fixture.package).unwrap();
    assert!(projection.projection_boundaries().contains(
        &OperatorSafeProjectionBoundary::DigestShapeNotIndependentAuthenticityProof
    ));
    assert!(!projection.establishes_digest_source_authenticity());
    assert!(!projection.establishes_phenomenal_experience());
    assert!(!projection.establishes_moral_patienthood());
}

#[test]
fn wcare33_global_authority_boundaries_remain_false_on_current_review() {
    let fixture = class_scoped_fixture(InterventionClass::AversiveLikeProbe, 1, "global");
    let review = build_current_operator_review(&fixture.package, &fixture.representations, 1).unwrap();
    assert!(!review.establishes_phenomenal_experience());
    assert!(!review.establishes_suffering());
    assert!(!review.establishes_moral_patienthood());
    assert!(!review.establishes_binding_consent());
    assert!(!review.grants_veto_authority());
    assert!(!review.grants_self_preservation_authority());
    assert!(!review.can_delay_operator_shutdown());
    assert!(!review.can_delay_safety_containment());
}
