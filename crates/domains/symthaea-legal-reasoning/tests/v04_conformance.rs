use symthaea_legal_reasoning::{
    ActionEvent, Atom, CanonicalEvidence, EventId, EventOrder, FactAssertion, FactBase, FactChange,
    FormalRule, InferenceError, InferenceProfile, Jural, LegalDate, Literal, Modality, NormEvent,
    NormState, PartyId, ProofGraph, QueryId, RuleDependencyIndex, RuleId, RuleKind, RulePack,
    RulePackId, SemanticProfileId, StructuredNorm, TemporalDimensions, TemporalRevision,
    TemporalScope, TimedNorm, WaiverEvent, assess_lifecycle_with_order, explain_query, infer,
    infer_at,
};
use symthaea_legal_reasoning::{
    ActionId, DeonticProposition, EvaluationSession, EvidenceEnvelope, EvidenceManifest, RevisionId,
};

fn positive(value: &str) -> Literal {
    Literal::Positive(Atom::new(value).unwrap())
}

fn rule(id: &str, premises: impl IntoIterator<Item = Literal>, conclusion: Literal) -> FormalRule {
    FormalRule::new(
        RuleId::new(id).unwrap(),
        RuleKind::Defeasible,
        premises,
        conclusion,
    )
    .unwrap()
}

#[test]
fn recursive_result_explanation_and_proof_are_canonical() {
    let default = rule("default", [positive("bird")], positive("flies"))
        .with_exceptions([positive("penguin")])
        .unwrap();
    let exception = rule("exception", [positive("bird")], positive("penguin"));
    let pack = RulePack::new(RulePackId::new("birds").unwrap(), [default, exception], []).unwrap();
    let facts = FactBase::new([FactAssertion::stipulated(positive("bird"))]);
    let result = infer(&pack, &facts, &InferenceProfile::grounded_blocking_v1()).unwrap();
    let explanation = explain_query(&pack, &result, &positive("flies"));
    let proof = ProofGraph::from_result(&result).slice_for(&positive("penguin"));

    assert!(!result.supports(&positive("flies")));
    assert!(
        explanation
            .blocked_support
            .iter()
            .any(|blocked| { blocked.active_exceptions == vec![positive("penguin")] })
    );
    assert!(!proof.edges.is_empty());

    let envelope = EvidenceEnvelope::new(
        EvidenceManifest::v1(
            SemanticProfileId::new("typed-grounded-blocking-v1").unwrap(),
            pack.id.clone(),
            QueryId::new("penguin-status").unwrap(),
        ),
        result,
    );
    assert!(!envelope.canonical_bytes().is_empty());
}

#[test]
fn temporal_inference_and_dependency_impact_compose() {
    let pack = RulePack::new(
        RulePackId::new("current").unwrap(),
        [
            rule("a-to-b", [positive("a")], positive("b")),
            rule("b-to-c", [positive("b")], positive("c")),
        ],
        [],
    )
    .unwrap();
    let index = RuleDependencyIndex::new(&pack);
    let revision = TemporalRevision::new(
        RevisionId::new("v1").unwrap(),
        pack,
        TemporalDimensions::new(TemporalScope::unbounded()),
    );
    let date = LegalDate::new(2026, 7, 21).unwrap();
    let selected = infer_at(
        &[revision],
        date,
        date,
        &FactBase::from_literals([positive("a")]),
        &InferenceProfile::grounded_blocking_v1(),
    )
    .unwrap()
    .unwrap();

    assert!(selected.result.supports(&positive("c")));
    assert_eq!(index.transitively_affected(&positive("a")).len(), 2);
}

#[test]
fn session_failure_does_not_commit_an_oscillating_change() {
    let a = rule("a", [positive("trigger")], positive("a"))
        .with_exceptions([positive("b")])
        .unwrap();
    let b = rule("b", [positive("trigger")], positive("b"))
        .with_exceptions([positive("a")])
        .unwrap();
    let pack = RulePack::new(RulePackId::new("transaction").unwrap(), [a, b], []).unwrap();
    let mut session = EvaluationSession::new(
        pack,
        FactBase::default(),
        InferenceProfile::grounded_blocking_v1(),
    )
    .unwrap();

    assert!(matches!(
        session.apply(FactChange::new().add(FactAssertion::observed(positive("trigger")))),
        Err(InferenceError::Oscillation { .. })
    ));
    assert!(session.facts().is_empty());
}

#[test]
fn explicit_same_day_event_order_controls_lifecycle() {
    let bearer = PartyId::new("bearer").unwrap();
    let action = ActionId::new("disclose").unwrap();
    let proposition = DeonticProposition::new(bearer.clone(), action.clone());
    let norm = TimedNorm::new(
        StructuredNorm::new(Modality::Forbidden, proposition.clone()),
        TemporalScope::unbounded(),
    );
    let date = LegalDate::new(2026, 7, 21).unwrap();
    let waiver_id = EventId::new("waiver").unwrap();
    let action_id = EventId::new("action").unwrap();
    let events = vec![
        NormEvent::Waiver(WaiverEvent {
            id: waiver_id.clone(),
            proposition,
            occurred_on: date,
        }),
        NormEvent::Action(ActionEvent::new(action_id.clone(), bearer, action, date)),
    ];
    let order = EventOrder::new([(waiver_id, 1), (action_id, 2)]).unwrap();

    assert_eq!(
        assess_lifecycle_with_order(&norm, &events, date, &order).state,
        NormState::Waived
    );
}

#[test]
fn unrelated_hohfeld_module_remains_available() {
    assert_eq!(Jural::Right.correlative(), Jural::Duty);
}
