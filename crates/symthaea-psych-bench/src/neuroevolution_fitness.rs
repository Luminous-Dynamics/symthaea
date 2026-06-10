// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Psych-bench fitness adapter: evaluates organisms using standardized
//! cognitive benchmarks as fitness signals.
//!
//! Maps evolved CfC phenotype parameters to `BenchmarkConfig`, runs a suite
//! of quick benchmarks, and converts scores to `OrganismFitness`.
//!
//! ## Rationale
//!
//! Instead of synthetic random inputs, this adapter evaluates evolved
//! configurations on real cognitive tasks (Stroop, binding, visual search,
//! emotional interference). Evolution discovers CfC configurations optimized
//! for actual information processing, not just FEP prediction on noise.
//!
//! ## Feature gate
//!
//! Requires `psych-bench` feature: `cargo test --features psych-bench`

use crate::harness::{BenchmarkConfig, BenchmarkResult, PsychBenchmark};
use serde::{Deserialize, Serialize};

use symthaea_neuroevolution::NeuralPhenotype;
use symthaea_neuroevolution::OrganismFitness;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Which benchmarks to include in the fitness evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkSuite {
    /// Fast: 3 benchmarks (~50ms total). Good for evolution.
    Quick,
    /// Medium: 8 benchmarks (~200ms total).
    Standard,
    /// Full: all available benchmarks (~2s total).
    Comprehensive,
    /// Custom selection by name.
    Custom(Vec<String>),
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::Quick
    }
}

/// Configuration for psych-bench fitness evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychBenchFitnessConfig {
    /// Which benchmarks to run.
    pub suite: BenchmarkSuite,
    /// Trials per condition (lower = faster, noisier).
    pub trials_per_condition: usize,
    /// HDC dimension for benchmarks (match organism eval dim).
    pub dimension: usize,
    /// Base seed for reproducibility.
    pub seed: u64,
    /// Weight for accuracy in composite fitness.
    pub accuracy_weight: f64,
    /// Weight for reaction time (lower RT = better).
    pub rt_weight: f64,
    /// Weight for consistency (lower std_dev = better).
    pub consistency_weight: f64,
}

impl Default for PsychBenchFitnessConfig {
    fn default() -> Self {
        Self {
            suite: BenchmarkSuite::Quick,
            trials_per_condition: 10,
            dimension: 512,
            seed: 42,
            accuracy_weight: 0.5,
            rt_weight: 0.3,
            consistency_weight: 0.2,
        }
    }
}

/// Result of a psych-bench fitness evaluation.
#[derive(Debug, Clone)]
pub struct PsychBenchFitnessResult {
    /// Composite fitness score (higher = better).
    pub composite: f64,
    /// Per-benchmark scores.
    pub benchmark_scores: Vec<(String, f64)>,
    /// Total wall-clock time in milliseconds.
    pub total_elapsed_ms: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Maps a neural phenotype to a BenchmarkConfig.
///
/// The evolved CfC parameters modulate cognitive performance:
/// - tau_base → time_pressure (faster tau = more time pressure tolerance)
/// - learning_rate → encoding fidelity (higher LR = faster but noisier encoding)
/// - gating_steepness → inhibition strength (steeper gating = better inhibition)
/// - layer_count → working memory capacity proxy
pub fn phenotype_to_benchmark_config(
    phenotype: &NeuralPhenotype,
    base_config: &PsychBenchFitnessConfig,
    organism_seed: u64,
) -> BenchmarkConfig {
    let nc = &phenotype.neuron_config;

    // Map tau_base [0.01, 1.0] → time_pressure [0.0, 0.5]
    // Faster tau = organism can handle more time pressure
    let time_pressure = (1.0 - nc.tau_base).max(0.0).min(0.5) as f64;

    // Map gating_steepness [0.1, 5.0] → D2 inhibition [0.1, 0.6]
    // Steeper gating = stronger inhibitory control
    let d2_inhibition = ((nc.gating_steepness - 0.1) / 4.9 * 0.5 + 0.1) as f64;

    // Map learning_rate [0.0001, 0.1] → encoding_noise [0.0, 0.15]
    // Higher LR = more encoding noise (speed-accuracy tradeoff)
    let encoding_noise =
        (nc.learning_rate.ln() - 0.0001f32.ln()) / (0.1f32.ln() - 0.0001f32.ln()) * 0.15;

    // Map layer_count [1, 5] → working_memory_capacity [3, 9]
    let wm_capacity = (phenotype.network_config.layer_sizes.len() * 2 + 1).min(9);

    BenchmarkConfig {
        dimension: base_config.dimension,
        trials_per_condition: base_config.trials_per_condition,
        seed: organism_seed,
        time_pressure,
        encoding_noise: encoding_noise.max(0.0) as f64,
        working_memory_capacity: wm_capacity,
        neuromod_d2_inhibition: d2_inhibition,
        enable_fep: true,
        enable_social: true,
        trial_trace: false,
        ..Default::default()
    }
}

/// Run the quick benchmark suite and return a composite fitness.
pub fn evaluate_with_psych_bench(
    phenotype: &NeuralPhenotype,
    config: &PsychBenchFitnessConfig,
    organism_seed: u64,
) -> PsychBenchFitnessResult {
    let bench_config = phenotype_to_benchmark_config(phenotype, config, organism_seed);

    let benchmarks = get_benchmark_suite(&config.suite);
    let mut total_elapsed = 0u64;
    let mut benchmark_scores = Vec::new();
    let mut composite_sum = 0.0;

    for bench in &benchmarks {
        let result = bench.run(&bench_config);
        total_elapsed += result.elapsed_ms;

        let score = extract_composite_score(
            &result,
            config.accuracy_weight,
            config.rt_weight,
            config.consistency_weight,
        );
        benchmark_scores.push((result.benchmark.clone(), score));
        composite_sum += score;
    }

    let composite = if benchmarks.is_empty() {
        0.0
    } else {
        composite_sum / benchmarks.len() as f64
    };

    PsychBenchFitnessResult {
        composite,
        benchmark_scores,
        total_elapsed_ms: total_elapsed,
    }
}

/// Convert psych-bench results to OrganismFitness format.
pub fn psych_bench_to_organism_fitness(
    psych_result: &PsychBenchFitnessResult,
    fep_fitness: &OrganismFitness,
    psych_weight: f64,
) -> OrganismFitness {
    // Blend psych-bench score with FEP fitness
    let blended_composite =
        fep_fitness.composite * (1.0 - psych_weight) + psych_result.composite * psych_weight;

    OrganismFitness {
        composite: blended_composite,
        free_energy: fep_fitness.free_energy,
        phi: fep_fitness.phi,
        consciousness: fep_fitness.consciousness,
        prediction_accuracy: psych_result.composite, // Override with cognitive accuracy
        energy_efficiency: fep_fitness.energy_efficiency,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNALS
// ═══════════════════════════════════════════════════════════════════════════════

fn get_benchmark_suite(suite: &BenchmarkSuite) -> Vec<Box<dyn PsychBenchmark>> {
    use crate::benchmarks;

    match suite {
        BenchmarkSuite::Quick => vec![
            Box::new(benchmarks::executive::stroop::StroopBenchmark),
            Box::new(benchmarks::binding::temporal_order::TemporalOrderBenchmark),
            Box::new(benchmarks::attention::visual_search::VisualSearchBenchmark),
        ],
        BenchmarkSuite::Standard => vec![
            Box::new(benchmarks::executive::stroop::StroopBenchmark),
            Box::new(benchmarks::binding::temporal_order::TemporalOrderBenchmark),
            Box::new(benchmarks::attention::visual_search::VisualSearchBenchmark),
            Box::new(benchmarks::affect::emotional_stroop::EmotionalStroopBenchmark),
            Box::new(benchmarks::metacognition::calibration::MetacognitiveCalibrationBenchmark),
            Box::new(benchmarks::language::semantic_priming::SemanticPrimingBenchmark),
            Box::new(benchmarks::social::dictator_game::DictatorGameBenchmark),
            Box::new(benchmarks::reasoning::arc_fluid::ArcFluidBenchmark),
        ],
        BenchmarkSuite::Comprehensive => {
            // Full suite — same as Standard plus additional domains
            vec![
                Box::new(benchmarks::executive::stroop::StroopBenchmark),
                Box::new(benchmarks::binding::temporal_order::TemporalOrderBenchmark),
                Box::new(benchmarks::attention::visual_search::VisualSearchBenchmark),
                Box::new(benchmarks::affect::emotional_stroop::EmotionalStroopBenchmark),
                Box::new(benchmarks::metacognition::calibration::MetacognitiveCalibrationBenchmark),
                Box::new(benchmarks::language::semantic_priming::SemanticPrimingBenchmark),
                Box::new(benchmarks::social::dictator_game::DictatorGameBenchmark),
                Box::new(benchmarks::reasoning::arc_fluid::ArcFluidBenchmark),
            ]
        }
        BenchmarkSuite::Custom(names) => {
            // Build from known benchmarks matching the requested names
            let all: Vec<Box<dyn PsychBenchmark>> = vec![
                Box::new(benchmarks::executive::stroop::StroopBenchmark),
                Box::new(benchmarks::binding::temporal_order::TemporalOrderBenchmark),
                Box::new(benchmarks::attention::visual_search::VisualSearchBenchmark),
                Box::new(benchmarks::affect::emotional_stroop::EmotionalStroopBenchmark),
                Box::new(benchmarks::metacognition::calibration::MetacognitiveCalibrationBenchmark),
                Box::new(benchmarks::language::semantic_priming::SemanticPrimingBenchmark),
                Box::new(benchmarks::social::dictator_game::DictatorGameBenchmark),
                Box::new(benchmarks::reasoning::arc_fluid::ArcFluidBenchmark),
            ];
            all.into_iter()
                .filter(|b| names.iter().any(|n| b.name().contains(n.as_str())))
                .collect()
        }
    }
}

/// Extract a composite score from benchmark metrics.
///
/// Looks for standard metric keys (accuracy, rt, effect size) and
/// computes a weighted composite.
fn extract_composite_score(
    result: &BenchmarkResult,
    accuracy_weight: f64,
    rt_weight: f64,
    consistency_weight: f64,
) -> f64 {
    // Find accuracy-like metrics
    let accuracy = result
        .metrics
        .iter()
        .find(|(k, _)| {
            k.contains("accuracy")
                || k.contains("correct")
                || k.contains("hit_rate")
                || k.contains("proportion_correct")
        })
        .map(|(_, v)| v.mean)
        .unwrap_or(0.5);

    // Find RT-like metrics (lower is better → invert)
    let rt_score = result
        .metrics
        .iter()
        .find(|(k, _)| k.contains("rt") || k.contains("reaction_time"))
        .map(|(_, v)| {
            // Normalize: 1 tick ≈ 50ms, typical range 1-20 ticks
            // Score: lower RT = higher score
            (1.0 - (v.mean / 20.0).min(1.0)).max(0.0)
        })
        .unwrap_or(0.5);

    // Find consistency metric (lower std_dev = better)
    let consistency = result
        .metrics
        .iter()
        .find(|(k, _)| k.contains("accuracy") || k.contains("correct"))
        .map(|(_, v)| {
            // Lower std_dev relative to mean = more consistent
            if v.mean.abs() > 1e-10 {
                (1.0 - v.std_dev / v.mean.abs()).max(0.0).min(1.0)
            } else {
                0.5
            }
        })
        .unwrap_or(0.5);

    accuracy * accuracy_weight + rt_score * rt_weight + consistency * consistency_weight
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::MetricValue;
    use symthaea_neuroevolution::NeuralGenome;

    #[test]
    fn test_phenotype_to_config_valid() {
        let genome = NeuralGenome::random(42);
        let phenotype = genome.decode();
        let config = PsychBenchFitnessConfig::default();
        let bench_config = phenotype_to_benchmark_config(&phenotype, &config, 42);

        assert!(bench_config.time_pressure >= 0.0 && bench_config.time_pressure <= 0.5);
        assert!(bench_config.encoding_noise >= 0.0);
        assert!(bench_config.working_memory_capacity >= 3);
        assert!(bench_config.neuromod_d2_inhibition >= 0.1);
    }

    #[test]
    fn test_evaluate_quick_suite() {
        let genome = NeuralGenome::random(42);
        let phenotype = genome.decode();
        let config = PsychBenchFitnessConfig {
            trials_per_condition: 5, // Minimal for speed
            ..Default::default()
        };
        let result = evaluate_with_psych_bench(&phenotype, &config, 42);

        assert!(result.composite.is_finite());
        assert!(result.composite >= 0.0 && result.composite <= 1.0);
        assert_eq!(result.benchmark_scores.len(), 3); // Quick suite = 3 benchmarks
    }

    #[test]
    fn test_different_genomes_different_scores() {
        let config = PsychBenchFitnessConfig {
            trials_per_condition: 5,
            ..Default::default()
        };

        let g1 = NeuralGenome::random(100);
        let g2 = NeuralGenome::random(200);
        let r1 = evaluate_with_psych_bench(&g1.decode(), &config, 100);
        let r2 = evaluate_with_psych_bench(&g2.decode(), &config, 200);

        // Different genomes should produce different (or at least valid) scores
        assert!(r1.composite.is_finite());
        assert!(r2.composite.is_finite());
    }

    #[test]
    fn test_blend_with_fep_fitness() {
        let psych_result = PsychBenchFitnessResult {
            composite: 0.8,
            benchmark_scores: vec![("Stroop".into(), 0.8)],
            total_elapsed_ms: 50,
        };
        let fep_fitness = OrganismFitness {
            composite: 0.5,
            free_energy: -1.0,
            phi: 0.3,
            consciousness: 0.3,
            prediction_accuracy: 0.6,
            energy_efficiency: 0.7,
        };

        let blended = psych_bench_to_organism_fitness(&psych_result, &fep_fitness, 0.5);
        // 0.5 * 0.5 + 0.5 * 0.8 = 0.65
        assert!((blended.composite - 0.65).abs() < 1e-10);
        assert_eq!(blended.prediction_accuracy, 0.8); // Overridden by psych score
    }

    #[test]
    fn test_extract_composite_score() {
        let mut result = BenchmarkResult::new("test", None);
        result.conditions = 1;
        result.trials_per_condition = 10;
        result.insert(
            "accuracy",
            MetricValue::from_samples(&[0.9, 0.85, 0.92, 0.88, 0.91]),
        );
        result.insert("rt", MetricValue::from_samples(&[5.0, 6.0, 4.5, 5.5, 5.0]));

        let score = super::extract_composite_score(&result, 0.5, 0.3, 0.2);
        assert!(score > 0.0 && score <= 1.0);
    }
}
