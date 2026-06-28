use symthaea_quantum_comp::{
    ExperimentManifest, ExperimentMatrixConfig, ExperimentMatrixRunner, ExperimentProtocol,
    PairedDifferenceSummary, ResearchArtifactReceipt, RunEnvironment, SubstrateProfile,
    exact_two_sided_sign_test_p_value,
};

#[test]
fn alpha6_significance_helpers_work() {
    let p = exact_two_sided_sign_test_p_value(0, 5);
    assert!((p - 0.0625).abs() < 1e-12);
    let a = [0.9, 0.8, 0.7];
    let b = [0.8, 0.7, 0.6];
    let summary = PairedDifferenceSummary::from_pairs(&a, &b, 1e-6).unwrap();
    assert_eq!(summary.a_wins, 3);
}

#[test]
fn alpha6_matrix_runner_exports() {
    let cfg = ExperimentMatrixConfig {
        dimensions: vec![64],
        noise_levels: vec![0.0, 0.1],
        trials: 2,
        replicates: 2,
        seed: 11,
        seed_stride: 23,
        topology_threshold: 0.55,
    };
    let report = ExperimentMatrixRunner::new(cfg).unwrap().run().unwrap();
    assert_eq!(report.cells.len(), 2);
    assert!(report.to_markdown().contains("Experiment Matrix Report"));
}

#[test]
fn alpha6_research_receipt_is_stable() {
    let manifest = ExperimentManifest::local_simulation(
        "alpha6-receipt",
        ExperimentProtocol::NoiseSweep,
        1,
        64,
        2,
        SubstrateProfile::quantum_inspired(),
    );
    let env = RunEnvironment::local_unknown();
    let a = ResearchArtifactReceipt::from_manifest_report_and_environment(
        &manifest, "report", &env, None,
    );
    let b = ResearchArtifactReceipt::from_manifest_report_and_environment(
        &manifest, "report", &env, None,
    );
    assert_eq!(a.receipt_fingerprint, b.receipt_fingerprint);
    assert!(a.to_text().contains("not a cryptographic signature"));
}
