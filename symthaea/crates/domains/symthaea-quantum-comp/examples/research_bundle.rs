use symthaea_quantum_comp::{
    BindingProbeRunner, ClaimBoundary, ExperimentManifest, ExperimentProtocol,
    ResearchArtifactReceipt, ResearchBundle, RunEnvironment, SubstrateProfile, audit_binding_probe,
    preflight_binding_config,
};

fn main() {
    let cfg = symthaea_quantum_comp::RunPreset::Smoke.binding_config();
    let preflight = preflight_binding_config(&cfg);
    let report = BindingProbeRunner::new(cfg).unwrap().run().unwrap();
    let audit = audit_binding_probe(&report, ClaimBoundary::LocalSimulation);
    let manifest = ExperimentManifest::local_simulation(
        "alpha10-research-bundle-example",
        ExperimentProtocol::ClassicalXorBinding,
        cfg.seed,
        cfg.dimension,
        cfg.trials,
        SubstrateProfile::quantum_inspired(),
    );
    let env = RunEnvironment::local_unknown();
    let receipt = ResearchArtifactReceipt::from_manifest_report_and_environment(
        &manifest,
        &report.to_text(),
        &env,
        Some("alpha10-example".to_string()),
    );
    let bundle = ResearchBundle::new(
        "alpha10-smoke-binding",
        manifest.to_text(),
        report.to_text(),
        format!("{}\n{}", preflight.to_text(), audit.to_text()),
        receipt.to_text(),
    );
    println!("{}", bundle.to_markdown());
}
