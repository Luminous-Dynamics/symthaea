use std::collections::BTreeSet;

use symthaea_legal_reasoning::{
    ActionEvent, ActionId, Atom, CanonicalEvidence, DeonticProposition, EventId, EvidenceEnvelope,
    EvidenceManifest, FormalRule, Jural, JuralRelation, LegalDate, LegalPositionState, LegalStatus,
    Literal, Modality, NormEvent, NormState, PartyId, PowerExercise, PriorityBasis, QueryId,
    RuleId, RuleKind, RulePack, RulePackId, SemanticProfileId, StructuredNorm, Superiority,
    TemporalScope, TimedNorm, assess_lifecycle, exercise_power, resolve_literal,
    validate_rule_pack,
};

fn positive(value: &str) -> Literal {
    Literal::Positive(Atom::new(value).unwrap())
}

fn negative(value: &str) -> Literal {
    Literal::Negative(Atom::new(value).unwrap())
}

#[test]
fn priority_resolution_is_auditable_and_evidence_bindable() {
    let general = FormalRule::new(
        RuleId::new("resident-registration").unwrap(),
        RuleKind::Defeasible,
        [positive("resident")],
        positive("must-register"),
    )
    .unwrap();
    let exception = FormalRule::new(
        RuleId::new("diplomat-exemption").unwrap(),
        RuleKind::Defeasible,
        [positive("diplomat")],
        negative("must-register"),
    )
    .unwrap();
    let pack = RulePack::new(
        RulePackId::new("registration-v2").unwrap(),
        [general, exception],
        [Superiority::new(
            RuleId::new("diplomat-exemption").unwrap(),
            RuleId::new("resident-registration").unwrap(),
            PriorityBasis::MoreSpecific,
        )],
    )
    .unwrap();
    let facts: BTreeSet<Literal> = [positive("resident"), positive("diplomat")]
        .into_iter()
        .collect();
    let resolution = resolve_literal(&pack, &facts, &positive("must-register"));

    assert_eq!(resolution.status, LegalStatus::Refuted);
    assert_eq!(resolution.defeats.len(), 1);

    let envelope = EvidenceEnvelope::new(
        EvidenceManifest::v1(
            SemanticProfileId::new("direct-priority-v1").unwrap(),
            pack.id.clone(),
            QueryId::new("registration-status").unwrap(),
        ),
        resolution,
    );
    assert!(!envelope.canonical_bytes().is_empty());
}

#[test]
fn lifecycle_violation_can_activate_a_reparative_norm() {
    let debtor = PartyId::new("debtor").unwrap();
    let creditor = PartyId::new("creditor").unwrap();
    let pay = DeonticProposition::new(debtor.clone(), ActionId::new("pay").unwrap())
        .with_beneficiary(creditor.clone());
    let interest = DeonticProposition::new(debtor, ActionId::new("pay-interest").unwrap())
        .with_beneficiary(creditor);
    let due = LegalDate::new(2026, 7, 20).unwrap();
    let primary = TimedNorm::new(
        StructuredNorm::new(Modality::Obligatory, pay),
        TemporalScope::unbounded(),
    )
    .with_deadline(due)
    .unwrap()
    .with_reparation(StructuredNorm::new(Modality::Obligatory, interest.clone()));

    let result = assess_lifecycle(&primary, &[], LegalDate::new(2026, 7, 21).unwrap());
    assert_eq!(result.state, NormState::Violated);
    assert_eq!(
        result.activated_reparation,
        Some(StructuredNorm::new(Modality::Obligatory, interest))
    );
}

#[test]
fn recorded_performance_fulfils_only_the_matching_party_bound_norm() {
    let debtor = PartyId::new("debtor").unwrap();
    let creditor = PartyId::new("creditor").unwrap();
    let proposition = DeonticProposition::new(debtor.clone(), ActionId::new("pay").unwrap())
        .with_beneficiary(creditor.clone());
    let norm = TimedNorm::new(
        StructuredNorm::new(Modality::Obligatory, proposition),
        TemporalScope::unbounded(),
    );
    let date = LegalDate::new(2026, 7, 21).unwrap();
    let event = NormEvent::Action(
        ActionEvent::new(
            EventId::new("payment-1").unwrap(),
            debtor,
            ActionId::new("pay").unwrap(),
            date,
        )
        .with_beneficiary(creditor),
    );

    assert_eq!(
        assess_lifecycle(&norm, &[event], date).state,
        NormState::Fulfilled
    );
}

#[test]
fn power_exercise_preserves_correlative_closure() {
    let power = JuralRelation::new(
        PartyId::new("court").unwrap(),
        PartyId::new("debtor").unwrap(),
        Jural::Power,
        ActionId::new("enter-judgment").unwrap(),
    );
    let right = JuralRelation::new(
        PartyId::new("creditor").unwrap(),
        PartyId::new("debtor").unwrap(),
        Jural::Right,
        ActionId::new("pay-judgment").unwrap(),
    );
    let state = LegalPositionState::new([power.clone()]).unwrap();
    let (next, _) =
        exercise_power(&state, &PowerExercise::new(power).assert(right.clone())).unwrap();

    assert!(next.contains(&right));
    assert!(next.contains(&right.correlative_relation()));
    assert!(next.is_correlatively_closed());
}

#[test]
fn validation_remains_advisory_and_does_not_change_results() {
    let allow = FormalRule::new(
        RuleId::new("allow").unwrap(),
        RuleKind::Defeasible,
        [positive("condition")],
        positive("enter"),
    )
    .unwrap();
    let pack = RulePack::new(RulePackId::new("entry").unwrap(), [allow], []).unwrap();
    let report = validate_rule_pack(&pack);
    let facts = [positive("condition")].into_iter().collect();

    assert!(!report.has_errors());
    assert_eq!(
        resolve_literal(&pack, &facts, &positive("enter")).status,
        LegalStatus::Supported
    );
}
