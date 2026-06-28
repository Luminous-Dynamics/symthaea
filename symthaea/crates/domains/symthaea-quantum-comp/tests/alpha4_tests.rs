use symthaea_quantum_comp::{
    ComparativeBindingConfig, ComparativeBindingRunner, NoiseRobustnessSummary, NoiseSweepConfig,
    NoiseSweepRunner, SampleSummary,
};

#[test]
fn alpha4_sample_summary_has_ci() {
    let summary = SampleSummary::from_samples(&[0.8, 0.9, 1.0]).unwrap();
    let (lo, hi) = summary.approximate_95_ci();
    assert!(lo < summary.mean);
    assert!(hi > summary.mean);
}

#[test]
fn alpha4_comparative_report_runs() {
    let mut cfg = ComparativeBindingConfig::default();
    cfg.base.dimension = 128;
    cfg.base.trials = 4;
    cfg.replicates = 3;
    let report = ComparativeBindingRunner::new(cfg).unwrap().run().unwrap();
    assert_eq!(report.classical.noisy.count, 3);
    assert!(report.to_text().contains("comparative-binding-v0.4"));
}

#[test]
fn alpha4_robustness_summary_runs() {
    let mut cfg = NoiseSweepConfig::default();
    cfg.base.dimension = 128;
    cfg.base.trials = 4;
    cfg.steps = 3;
    let sweep = NoiseSweepRunner::new(cfg).unwrap().run().unwrap();
    let summary = NoiseRobustnessSummary::from_sweep(&sweep, 0.75);
    assert!(summary.to_text().contains("similarity_floor"));
}
