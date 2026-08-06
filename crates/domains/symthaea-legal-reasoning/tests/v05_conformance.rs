use symthaea_legal_reasoning::{
    ArgumentGraph, Atom, AuthorityId, BatchLimits, CanonicalEvidence, DecisionRecord,
    DecisionReviewFlag, DocumentId, FactAssertion, FactConflictPolicy, FactIntervention, FactKind,
    FactPolicy, FormalRule, InferenceProfile, LegalDate, LegalStatus, Literal, ProvisionId, RuleId,
    RuleKind, RulePack, RulePackId, SemanticReview, SourceRef, TemporalFactAssertion,
    TemporalFactBase, TemporalScope, analyze_counterfactuals, evaluate_batch_with_limits,
    evaluate_case, preflight, validate_argument_graph, validate_decision_record,
};

fn positive(value: &str) -> Literal {
    Literal::Positive(Atom::new(value).unwrap())
}

fn date(year: i32, month: u8, day: u8) -> LegalDate {
    LegalDate::new(year, month, day).unwrap()
}

fn source() -> SourceRef {
    SourceRef::new(
        DocumentId::new("case-record").unwrap(),
        ProvisionId::new("stipulation-1").unwrap(),
    )
}

fn registration_pack() -> RulePack {
    RulePack::new(
        RulePackId::new("registration-v05").unwrap(),
        [
            FormalRule::new(
                RuleId::new("resident-default").unwrap(),
                RuleKind::Defeasible,
                [positive("resident")],
                positive("register"),
            )
            .unwrap()
            .with_exceptions([positive("exempt")])
            .unwrap(),
            FormalRule::new(
                RuleId::new("liable-remedy").unwrap(),
                RuleKind::Defeasible,
                [positive("liable")],
                positive("remedy"),
            )
            .unwrap(),
        ],
        [],
    )
    .unwrap()
}

#[test]
fn temporal_intake_policy_and_inference_preserve_exclusions() {
    let authority = AuthorityId::new("registry").unwrap();
    let temporal = TemporalFactBase::new([
        TemporalFactAssertion::new(
            FactAssertion::stipulated(positive("resident"))
                .with_source(source())
                .asserted_by(authority.clone()),
            TemporalScope::unbounded(),
        )
        .recorded_on(date(2026, 7, 1)),
        TemporalFactAssertion::new(
            FactAssertion::assumed(positive("exempt")),
            TemporalScope::unbounded(),
        ),
    ]);
    let policy = FactPolicy::evidence_bound()
        .allow_only_authorities([authority])
        .with_conflicts(FactConflictPolicy::RejectConflictedAtoms);
    let evaluation = evaluate_case(
        &registration_pack(),
        &temporal,
        date(2026, 7, 21),
        date(2026, 7, 21),
        &policy,
        &InferenceProfile::grounded_blocking_v1(),
    )
    .unwrap();

    assert_eq!(evaluation.intake.admission.rejected.len(), 1);
    assert_eq!(
        evaluation.intake.admission.rejected[0].assertion.kind,
        FactKind::Assumed
    );
    assert_eq!(
        evaluation.result.status(&positive("register")),
        LegalStatus::Supported
    );
    assert!(!evaluation.canonical_bytes().is_empty());
}

#[test]
fn arguments_decisions_and_validation_form_one_review_chain() {
    let pack = registration_pack();
    let facts = symthaea_legal_reasoning::FactBase::from_literals([
        positive("resident"),
        positive("exempt"),
    ]);
    let result =
        symthaea_legal_reasoning::infer(&pack, &facts, &InferenceProfile::grounded_blocking_v1())
            .unwrap();
    let graph = ArgumentGraph::from_result(&pack, &result).unwrap();
    let decision = DecisionRecord::from_result(&pack, &result, &positive("register")).unwrap();

    assert!(!validate_argument_graph(&graph).has_errors());
    assert!(!validate_decision_record(&decision).has_errors());
    assert_eq!(decision.status, LegalStatus::Undetermined);
    assert!(
        decision
            .review_flags
            .contains(&DecisionReviewFlag::UndeterminedConclusion)
    );
    assert!(!graph.canonical_bytes().is_empty());
    assert!(!decision.canonical_bytes().is_empty());
}

#[test]
fn semantic_review_counterfactual_batch_and_preflight_compose() {
    let pack = registration_pack();
    let facts = symthaea_legal_reasoning::FactBase::from_literals([
        positive("resident"),
        positive("liable"),
        positive("liable").opposite(),
    ]);
    let review = SemanticReview::evaluate(
        &pack,
        &facts,
        [
            InferenceProfile::grounded_blocking_v1(),
            InferenceProfile::grounded_propagating_v1(),
        ],
    );
    assert!(!review.sensitivity(&positive("remedy")).is_stable());

    let counterfactual = analyze_counterfactuals(
        &pack,
        &facts,
        &InferenceProfile::grounded_blocking_v1(),
        &positive("register"),
        [FactIntervention::Add(FactAssertion::observed(positive(
            "exempt",
        )))],
    )
    .unwrap();
    assert_eq!(counterfactual.status_changing().len(), 1);

    let batch = evaluate_batch_with_limits(
        &pack,
        &facts,
        &InferenceProfile::grounded_blocking_v1(),
        [positive("register"), positive("remedy")],
        BatchLimits { max_queries: 2 },
    )
    .unwrap();
    assert_eq!(batch.records.len(), 2);

    let readiness = preflight(&pack, &facts);
    assert_eq!(readiness.metrics.rules, 2);
    assert!(readiness.metrics.conflicted_fact_atoms > 0);
    assert!(!review.canonical_bytes().is_empty());
    assert!(!counterfactual.canonical_bytes().is_empty());
    assert!(!batch.canonical_bytes().is_empty());
    assert!(!readiness.canonical_bytes().is_empty());
}
