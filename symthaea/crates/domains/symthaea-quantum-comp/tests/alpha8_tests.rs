use symthaea_quantum_comp::{
    BindingProbeRunner, ClaimBoundary, IntegrationDeclaration, ReplayPlan, ReplayScope,
    audit_binding_probe, fixture_catalog, gate_local_artifact, known_schema_labels, named_fixture,
    preflight_binding_config,
};

#[test]
fn alpha10_schema_labels_are_current() {
    let labels = known_schema_labels();
    assert!(labels.iter().any(|label| label.contains("replay_plan")));
    assert!(labels.iter().all(|label| label.ends_with("alpha10")));
}

#[test]
fn fixture_catalog_is_stable() {
    let catalog = fixture_catalog();
    assert_eq!(catalog.len(), 3);
    assert!(
        catalog
            .iter()
            .any(|fixture| fixture.name == "smoke-binding")
    );
}

#[test]
fn replay_plan_mentions_claim_caveats() {
    let plan = ReplayPlan::for_scope(ReplayScope::Smoke);
    assert!(plan.to_markdown().contains("quantum advantage"));
    assert!(plan.to_text().contains("binding smoke"));
}

#[test]
fn local_gate_can_be_constructed() {
    let fixture = named_fixture("smoke-binding").unwrap();
    let preflight = preflight_binding_config(&fixture.config);
    let report = BindingProbeRunner::new(fixture.config)
        .unwrap()
        .run()
        .unwrap();
    let audit = audit_binding_probe(&report, ClaimBoundary::LocalSimulation);
    let replay = ReplayPlan::for_scope(ReplayScope::Smoke);
    let gate = gate_local_artifact(&preflight, &audit, Some(&fixture), &replay);
    assert!(gate.can_release_locally());
    assert!(gate.to_text().contains("local release gate"));
}

#[test]
fn interop_declarations_are_explicit() {
    let decl = IntegrationDeclaration::external_backend_observation("test-backend");
    assert!(decl.to_text().contains("test-backend"));
    assert!(decl.to_text().contains("quantum consciousness"));
}
