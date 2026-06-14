// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use symthaea_fractal_time_lab::floquet_time_crystal::{TimeCrystalDetector, TimeCrystalSimulator};
use symthaea_fractal_time_lab::hofstadter::{HDC_DIM, HofstadterGenerator};
use symthaea_fractal_time_lab::metrics::{
    ExperimentScorecard, scorecards_to_csv, scorecards_to_json_array,
};
use symthaea_fractal_time_lab::multiscale_phi::{
    BoxCoveringCoarseGrainer, BoxDimensionEstimator, CoarseGrainer, MultiScalePhi,
    SpectralCoarseGrainer,
};
use symthaea_fractal_time_lab::null_models::NullModels;
use symthaea_fractal_time_lab::report::scorecards_to_markdown_report;
use symthaea_fractal_time_lab::run_all_benchmarks;
use symthaea_fractal_time_lab::runner::{BenchmarkConfig, run_benchmark_run};

#[test]
fn test_runner_reproducibility() {
    let config = BenchmarkConfig {
        seed: 42,
        trials: 4,
    };
    let a = run_all_benchmarks(config);
    let b = run_all_benchmarks(config);

    assert_eq!(a.len(), b.len());
    for (ca, cb) in a.iter().zip(b.iter()) {
        assert_eq!(ca.experiment, cb.experiment);
        assert_eq!(ca.passed, cb.passed);
        assert!((ca.primary_score - cb.primary_score).abs() < 1e-10);
        assert!((ca.null_mean - cb.null_mean).abs() < 1e-10);
        assert!((ca.null_std - cb.null_std).abs() < 1e-10);
    }
}

#[test]
fn test_full_benchmark_run_has_claim_boundaries() {
    let run = run_benchmark_run(BenchmarkConfig {
        seed: 42,
        trials: 3,
    });
    assert_eq!(run.scorecards.len(), 5);
    assert!(run.epistemic_status.contains("EXPLORATORY"));
    assert!(
        run.non_claims
            .iter()
            .any(|claim| claim.contains("fractal time"))
    );
}

#[test]
fn test_markdown_report_contains_scope() {
    let run = run_benchmark_run(BenchmarkConfig {
        seed: 42,
        trials: 2,
    });
    let report = scorecards_to_markdown_report(&run);
    assert!(report.contains("Claim Scope"));
    assert!(report.contains("does not prove fractal time"));
}

#[test]
fn test_scorecard_json_and_csv_are_emitted() {
    let card = ExperimentScorecard::new(
        "sanity",
        "scorecard serializes",
        1.0,
        &[0.1, 0.2, 0.3],
        3,
        42,
        1.0,
        "exploratory",
    );

    let json = scorecards_to_json_array(&[card.clone()]);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert!(parsed.is_array());

    let csv = scorecards_to_csv(&[card]);
    assert!(csv.starts_with("experiment,hypothesis"));
    assert_eq!(csv.lines().count(), 2);
}

#[test]
fn test_seeded_null_models_are_reproducible() {
    assert_eq!(
        NullModels::random_spectrum(16, 42),
        NullModels::random_spectrum(16, 42)
    );

    let g1 = NullModels::random_graph(16, 0.25, 42);
    let g2 = NullModels::random_graph(16, 0.25, 42);
    assert_eq!(g1.edge_count(), g2.edge_count());
}

#[test]
fn test_hofstadter_cross_scale_outputs_are_finite() {
    let generator = HofstadterGenerator::new(HDC_DIM);
    let spectra = vec![
        generator.generate_harper_slice(1, 13),
        generator.generate_harper_slice(2, 21),
        generator.generate_harper_slice(3, 34),
    ];

    let score = generator.average_cross_scale_similarity(&spectra, -4.0, 4.0, 128);
    assert!(score.is_finite());
    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn test_time_crystal_likeness_beats_damped_persistence_control() {
    let mut simulator = TimeCrystalSimulator::new(20);
    let signal = simulator.signal(128, 0.0, 0.05);
    let damped = NullModels::damped_oscillator(128, 0.80);

    let detector = TimeCrystalDetector;
    let dtc_score = detector.time_crystal_likeness(&signal);
    let damped_score = detector.time_crystal_likeness(&damped);

    assert!(dtc_score > damped_score);
}

#[test]
fn test_spectral_and_box_coarse_grainers_are_safe() {
    let graph = NullModels::hierarchical_graph(4, 4);
    let analyzer = MultiScalePhi;

    let spectral = SpectralCoarseGrainer;
    let spectral_coarse = spectral
        .coarse_grain(&graph)
        .expect("spectral coarse graph");
    assert!(spectral_coarse.node_count() <= graph.node_count());
    assert!(analyzer.phi_proxy(&spectral_coarse).is_finite());

    let box_cover = BoxCoveringCoarseGrainer { radius: 1 };
    let box_coarse = box_cover.coarse_grain(&graph).expect("box coarse graph");
    assert!(box_coarse.node_count() <= graph.node_count());
    assert!(analyzer.phi_proxy(&box_coarse).is_finite());
}

#[test]
fn test_box_dimension_estimate_is_serializable() {
    let graph = NullModels::binary_tree(5);
    let estimate = BoxDimensionEstimator::estimate(&graph, 4).expect("box dimension estimate");
    let json = serde_json::to_string(&estimate).expect("serialize estimate");
    assert!(json.contains("dimension"));
}
