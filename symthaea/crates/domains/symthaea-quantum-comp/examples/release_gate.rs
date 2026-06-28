use symthaea_quantum_comp::{
    BindingProbeRunner, ClaimBoundary, ReplayPlan, ReplayScope, audit_binding_probe,
    gate_local_artifact, named_fixture, preflight_binding_config,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = named_fixture("smoke-binding").expect("fixture exists");
    let preflight = preflight_binding_config(&fixture.config);
    let report = BindingProbeRunner::new(fixture.config)?.run()?;
    let audit = audit_binding_probe(&report, ClaimBoundary::LocalSimulation);
    let replay = ReplayPlan::for_scope(ReplayScope::Smoke);
    let gate = gate_local_artifact(&preflight, &audit, Some(&fixture), &replay);
    println!("{}", gate.to_text());
    Ok(())
}
