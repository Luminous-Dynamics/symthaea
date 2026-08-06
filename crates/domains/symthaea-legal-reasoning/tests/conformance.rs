use symthaea_legal_reasoning::{
    ActionId, DeonticProposition, Jural, JuralRelation, Modality, PartyId, PermissionStatus, Rule,
    StructuredNorm, proposition_permission_status, try_derive_with_trace, try_why_not,
};

#[test]
fn derived_exception_is_order_invariant_with_identical_trace() {
    let default = Rule::new(&["resident"], &["exempt"], "must_register");
    let exemption = Rule::new(&["diplomat"], &[], "exempt");
    let resident = Rule::new(&["diplomat"], &[], "resident");

    let left = try_derive_with_trace(
        &[default.clone(), exemption.clone(), resident.clone()],
        &["diplomat"],
    )
    .unwrap();
    let right = try_derive_with_trace(&[resident, default, exemption], &["diplomat"]).unwrap();

    assert_eq!(left, right);
    assert!(left.entails("resident"));
    assert!(left.entails("exempt"));
    assert!(!left.entails("must_register"));
}

#[test]
fn why_not_exposes_the_operative_exception() {
    let rules = vec![Rule::new(&["resident"], &["exempt"], "must_register")];
    let blocked = try_why_not(&rules, &["resident", "exempt"], "must_register").unwrap();

    assert_eq!(blocked.len(), 1);
    assert!(blocked[0].missing_conditions.is_empty());
    assert_eq!(blocked[0].active_exceptions, vec!["exempt"]);
}

#[test]
fn deontic_and_hohfeld_models_preserve_party_direction() {
    let employer = PartyId::new("employer").unwrap();
    let employee = PartyId::new("employee").unwrap();
    let pay = ActionId::new("pay_wage").unwrap();
    let proposition =
        DeonticProposition::new(employer.clone(), pay.clone()).with_beneficiary(employee.clone());
    let norms = vec![StructuredNorm::new(
        Modality::Obligatory,
        proposition.clone(),
    )];

    assert_eq!(
        proposition_permission_status(&norms, &proposition),
        PermissionStatus::ImpliedByObligation
    );

    let employee_right = JuralRelation::new(employee, employer, Jural::Right, pay);
    let employer_duty = employee_right.correlative_relation();
    assert_eq!(employer_duty.position, Jural::Duty);
    assert!(employee_right.is_correlative_of(&employer_duty));
}
