use symthaea_quantum_comp::{
    BindingProbeConfig, BindingProbeRunner, ClaimBoundary, NegativeControlConfig,
    NegativeControlRunner, ReportTable, ReproducibilityRecord, RunEnvironment, audit_binding_probe,
    audit_negative_control, fnv1a64,
};

#[test]
fn alpha5_report_exports_are_nonempty() {
    let cfg = BindingProbeConfig {
        dimension: 128,
        trials: 4,
        noise: 0.03,
        seed: 123,
        topology_threshold: 0.55,
    };
    let report = BindingProbeRunner::new(cfg).unwrap().run().unwrap();
    assert!(report.to_csv().contains("classical_noisy"));
    assert!(report.to_markdown().contains("Binding Probe Report"));
}

#[test]
fn alpha5_reproducibility_record_is_stable() {
    let cfg = BindingProbeConfig {
        dimension: 128,
        trials: 4,
        noise: 0.03,
        seed: 123,
        topology_threshold: 0.55,
    };
    let report = BindingProbeRunner::new(cfg).unwrap().run().unwrap();
    let env = RunEnvironment::local_unknown();
    let a = ReproducibilityRecord::from_manifest_and_environment(&report.manifest, &env);
    let b = ReproducibilityRecord::from_manifest_and_environment(&report.manifest, &env);
    assert_eq!(a, b);
    assert_ne!(a.combined_fingerprint, 0);
}

#[test]
fn alpha5_audit_helpers_keep_claims_conservative() {
    let cfg = BindingProbeConfig {
        dimension: 128,
        trials: 4,
        noise: 0.03,
        seed: 123,
        topology_threshold: 0.55,
    };
    let report = BindingProbeRunner::new(cfg).unwrap().run().unwrap();
    let local = audit_binding_probe(&report, ClaimBoundary::LocalSimulation);
    assert!(local.passed());
    let external = audit_binding_probe(&report, ClaimBoundary::ExternalBackendObservation);
    assert!(external.has_warnings());
}

#[test]
fn alpha5_negative_control_audit_passes_with_gap() {
    let report = NegativeControlRunner::new(NegativeControlConfig {
        dimension: 256,
        trials: 8,
        noise: 0.02,
        seed: 99,
    })
    .unwrap()
    .run()
    .unwrap();
    let audit = audit_negative_control(&report, 0.30);
    assert!(audit.passed());
}

#[test]
fn alpha5_fnv_hash_is_stable() {
    assert_eq!(fnv1a64(b"symthaea"), fnv1a64(b"symthaea"));
    assert_ne!(fnv1a64(b"symthaea"), fnv1a64(b"mycelix"));
}
