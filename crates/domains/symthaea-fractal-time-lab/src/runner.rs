// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::floquet_time_crystal::{TimeCrystalDetector, TimeCrystalSimulator};
use crate::hofstadter::{HDC_DIM, HofstadterGenerator};
use crate::metrics::ExperimentScorecard;
use crate::multiscale_phi::{
    BoxCoveringCoarseGrainer, BoxDimensionEstimator, CoarseGrainer, DegreeBinCoarseGrainer,
    MultiScalePhi, SpectralCoarseGrainer,
};
use crate::null_models::NullModels;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub seed: u64,
    pub trials: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            trials: 32,
        }
    }
}

impl BenchmarkConfig {
    pub fn sanitized(self) -> Self {
        Self {
            seed: self.seed,
            trials: self.trials.max(1),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRun {
    pub version: String,
    pub epistemic_status: String,
    pub config: BenchmarkConfig,
    pub scorecards: Vec<ExperimentScorecard>,
    pub claims: Vec<String>,
    pub non_claims: Vec<String>,
}

impl BenchmarkRun {
    pub fn new(config: BenchmarkConfig, scorecards: Vec<ExperimentScorecard>) -> Self {
        Self {
            version: "fractal-time-lab-v0.5".to_string(),
            epistemic_status: "EXPLORATORY_BENCHMARK_NOT_PHYSICAL_PROOF".to_string(),
            config: config.sanitized(),
            scorecards,
            claims: vec![
                "Computational benchmark run completed under the given seed/trial configuration."
                    .to_string(),
                "Scorecards compare toy structural hypotheses against explicit null models."
                    .to_string(),
            ],
            non_claims: vec![
                "Does not prove fractal time.".to_string(),
                "Does not prove quantum consciousness.".to_string(),
                "Does not simulate a physical quantum many-body time crystal.".to_string(),
                "Does not compute full IIT Phi.".to_string(),
                "Does not constitute production scientific validation.".to_string(),
            ],
        }
    }

    pub fn all_passed(&self) -> bool {
        self.scorecards.iter().all(|card| card.passed)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|err| format!(r#"{{"serialization_error":"{}"}}"#, err))
    }
}

pub fn run_all_benchmarks(config: BenchmarkConfig) -> Vec<ExperimentScorecard> {
    let config = config.sanitized();

    vec![
        run_hofstadter_experiment(config),
        run_time_crystal_experiment(config),
        run_multiscale_phi_experiment(config),
        run_box_covering_experiment(config),
        run_box_dimension_experiment(config),
    ]
}

pub fn run_benchmark_run(config: BenchmarkConfig) -> BenchmarkRun {
    let config = config.sanitized();
    let scorecards = run_all_benchmarks(config);
    BenchmarkRun::new(config, scorecards)
}

pub fn run_hofstadter_experiment(config: BenchmarkConfig) -> ExperimentScorecard {
    let config = config.sanitized();
    let generator = HofstadterGenerator::new(HDC_DIM);

    let related_spectra = vec![
        generator.generate_harper_slice(1, 13),
        generator.generate_harper_slice(2, 21),
        generator.generate_harper_slice(3, 34),
    ];

    let primary_score = generator.average_cross_scale_similarity(&related_spectra, -4.0, 4.0, 128);

    let reference = &related_spectra[0];
    let hv_reference = generator.encode_spectrum(reference, -4.0, 4.0, 128);

    let mut null_scores = Vec::with_capacity(config.trials * 3);

    for trial in 0..config.trials {
        let trial_seed = config.seed + trial as u64;

        let random = NullModels::random_spectrum(reference.len(), trial_seed);
        let jittered = NullModels::jittered_spectrum(reference, 1.0, trial_seed + 10_000);
        let smooth = NullModels::sinusoidal_spectrum(reference.len(), 3.0);

        for null_spectrum in [&random, &jittered, &smooth] {
            let hv_null = generator.encode_spectrum(null_spectrum, -4.0, 4.0, 128);
            null_scores.push(generator.similarity_score(&hv_reference, &hv_null));
        }
    }

    ExperimentScorecard::new(
        "Hofstadter-HDC cross-scale similarity",
        "Related Harper slices preserve HDC similarity better than random/jittered/smooth spectra.",
        primary_score,
        &null_scores,
        config.trials,
        config.seed,
        2.0,
        "Exploratory: uses Harper slices and HDC quantization, not full experimental Hofstadter data.",
    )
}

pub fn run_time_crystal_experiment(config: BenchmarkConfig) -> ExperimentScorecard {
    let config = config.sanitized();
    let mut simulator = TimeCrystalSimulator::new(20);
    let signal = simulator.signal(256, 0.02, 0.05);

    let detector = TimeCrystalDetector;
    let primary_score = detector.time_crystal_likeness(&signal);

    let mut null_scores = Vec::with_capacity(config.trials * 2);

    for trial in 0..config.trials {
        let decay = 0.60 + 0.35 * (trial as f64 / config.trials as f64);
        let damped = NullModels::damped_oscillator(256, decay);
        let random = NullModels::random_signal(256, config.seed + trial as u64);

        null_scores.push(detector.time_crystal_likeness(&damped));
        null_scores.push(detector.time_crystal_likeness(&random));
    }

    ExperimentScorecard::new(
        "Persistent 2T response",
        "A DTC-like toy model shows persistent subharmonic response compared with damped and random controls.",
        primary_score,
        &null_scores,
        config.trials,
        config.seed,
        2.0,
        "Exploratory: classical Floquet surrogate, not a quantum many-body simulation.",
    )
}

pub fn run_multiscale_phi_experiment(config: BenchmarkConfig) -> ExperimentScorecard {
    let config = config.sanitized();
    let analyzer = MultiScalePhi;
    let spectral = SpectralCoarseGrainer;

    let hierarchical = NullModels::hierarchical_graph(4, 4);
    let coarse = spectral
        .coarse_grain(&hierarchical)
        .unwrap_or_else(|_| hierarchical.clone());

    let primary_score = analyzer.integration_survival(&hierarchical, &coarse);

    let mut null_scores = Vec::with_capacity(config.trials * 2);

    for trial in 0..config.trials {
        let random = NullModels::random_graph(16, 0.20, config.seed + trial as u64);

        if let Ok(coarse_random) = spectral.coarse_grain(&random) {
            null_scores.push(analyzer.integration_survival(&random, &coarse_random));
        }

        let degree_grainer = DegreeBinCoarseGrainer { bins: 2 };
        if let Ok(coarse_degree) = degree_grainer.coarse_grain(&random) {
            null_scores.push(analyzer.integration_survival(&random, &coarse_degree));
        }
    }

    if null_scores.is_empty() {
        null_scores.push(0.0);
    }

    ExperimentScorecard::new(
        "Multi-scale integration survival",
        "Hierarchical modular graphs preserve an EI/Phi proxy across spectral coarse-graining better than random controls.",
        primary_score,
        &null_scores,
        config.trials,
        config.seed,
        1.0,
        "Exploratory: EI/Phi proxy and spectral coarse-graining, not full IIT Phi.",
    )
}

pub fn run_box_covering_experiment(config: BenchmarkConfig) -> ExperimentScorecard {
    let config = config.sanitized();
    let analyzer = MultiScalePhi;
    let box_grainer = BoxCoveringCoarseGrainer { radius: 1 };

    let path_graph = NullModels::path_graph(24);
    let coarse_path = box_grainer
        .coarse_grain(&path_graph)
        .unwrap_or_else(|_| path_graph.clone());

    let primary_score = analyzer.integration_survival(&path_graph, &coarse_path);

    let mut null_scores = Vec::with_capacity(config.trials);

    for trial in 0..config.trials {
        let random = NullModels::random_graph(24, 0.20, config.seed + 50_000 + trial as u64);

        if let Ok(coarse_random) = box_grainer.coarse_grain(&random) {
            null_scores.push(analyzer.integration_survival(&random, &coarse_random));
        }
    }

    if null_scores.is_empty() {
        null_scores.push(0.0);
    }

    ExperimentScorecard::new(
        "Box-covering integration survival",
        "A path/tree-like graph preserves EI/Phi proxy across greedy box covering differently than random graph controls.",
        primary_score,
        &null_scores,
        config.trials,
        config.seed,
        0.5,
        "Exploratory: greedy graph-radius boxes, not optimized minimum box covering.",
    )
}

pub fn run_box_dimension_experiment(config: BenchmarkConfig) -> ExperimentScorecard {
    let config = config.sanitized();

    let tree = NullModels::binary_tree(5);
    let estimate = BoxDimensionEstimator::estimate(&tree, 4).ok();

    let primary_score = estimate
        .as_ref()
        .map(|e| e.dimension * e.r_squared)
        .unwrap_or(0.0);

    let mut null_scores = Vec::with_capacity(config.trials);

    for trial in 0..config.trials {
        let random = NullModels::random_graph(31, 0.12, config.seed + 90_000 + trial as u64);
        if let Ok(est) = BoxDimensionEstimator::estimate(&random, 4) {
            null_scores.push(est.dimension * est.r_squared);
        }
    }

    if null_scores.is_empty() {
        null_scores.push(0.0);
    }

    ExperimentScorecard::new(
        "Greedy box-dimension diagnostic",
        "Tree-like graph has a stable greedy box-count scaling signal compared with random graph controls.",
        primary_score,
        &null_scores,
        config.trials,
        config.seed,
        0.5,
        "Exploratory: greedy cover dimension is heuristic and sensitive to graph family and radius range.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_is_reproducible_for_same_seed() {
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
    fn test_benchmark_run_contains_claim_boundaries() {
        let run = run_benchmark_run(BenchmarkConfig {
            seed: 42,
            trials: 2,
        });
        assert_eq!(run.scorecards.len(), 5);
        assert!(!run.non_claims.is_empty());
        assert!(run.epistemic_status.contains("EXPLORATORY"));
    }
}
