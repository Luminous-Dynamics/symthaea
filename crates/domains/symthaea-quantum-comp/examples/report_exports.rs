use symthaea_quantum_comp::{
    BindingProbeConfig, BindingProbeRunner, ClaimBoundary, ReportTable, ReproducibilityRecord,
    RunEnvironment, audit_binding_probe,
};

fn main() -> symthaea_quantum_comp::Result<()> {
    let config = BindingProbeConfig {
        dimension: 512,
        trials: 12,
        noise: 0.05,
        seed: 20260622,
        topology_threshold: 0.55,
    };
    let report = BindingProbeRunner::new(config)?.run()?;
    let environment = RunEnvironment::local_unknown();
    let repro =
        ReproducibilityRecord::from_manifest_and_environment(&report.manifest, &environment);
    let audit = audit_binding_probe(&report, ClaimBoundary::LocalSimulation);

    println!("{}", environment.to_text());
    println!("{}", repro.to_text());
    println!("{}", audit.to_text());
    println!("{}", report.to_markdown());
    println!("{}", report.to_csv());
    Ok(())
}
