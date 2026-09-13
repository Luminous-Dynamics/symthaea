// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Preregistered learned-vs-frozen diagonal-HLS state-tracking ablation.
//!
//! This first learning experiment is deliberately **current-state only**. The
//! fixed associative decoder has a justified algebra for current one-hop queries
//! and current two-hop object-location composition, but no justified temporal
//! addressing mechanism yet. Historical recall is therefore excluded from the
//! preregistration rather than approximated with an arbitrary lag key.
//!
//! The runner creates one initial HLS cell, preserves an untouched frozen clone,
//! trains a second clone on an ordered, predeclared set of world seeds, and then
//! evaluates both clones on the exact same held-out worlds. Results are paired by
//! test seed; no best-seed selection is performed.

use crate::holographic_liquid::{HlsActivation, HlsConfig, HolographicLiquidCell};
use crate::state_tracking_benchmark::{StateTrackingBenchmark, StateTrackingBenchmarkConfig};
use crate::state_tracking_codec::StateTrackingCodec;
use crate::state_tracking_current_only::{
    CurrentOnlyTrainingError, evaluate_current_only_episode, train_current_only_episode,
};
use crate::state_tracking_exact_training::{
    AssociativeEpisodeMetrics, ExactEpisodeTrainingConfig, ExactEpisodeTrainingReport,
};
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ExactLearningAblationPlan {
    pub hls_config: HlsConfig,
    pub cell_seed: u64,
    pub codec_seed: u64,
    /// Seed in this template is ignored; each train/test seed replaces it.
    pub benchmark_template: StateTrackingBenchmarkConfig,
    pub training: ExactEpisodeTrainingConfig,
    /// Ordered training worlds. One episode-end update is applied after each.
    pub train_world_seeds: Vec<u64>,
    /// Held-out worlds. They must be disjoint from all training seeds.
    pub test_world_seeds: Vec<u64>,
}

impl ExactLearningAblationPlan {
    /// Small deterministic plan suitable only for mechanical CI/smoke execution.
    pub fn smoke() -> Self {
        Self {
            hls_config: HlsConfig {
                dim: 64,
                state_norm_limit: f32::INFINITY,
                activation: HlsActivation::Tanh,
                ..HlsConfig::default()
            },
            cell_seed: 0xA11CE,
            codec_seed: 0xC0DEC,
            benchmark_template: StateTrackingBenchmarkConfig {
                entities: 4,
                objects: 6,
                locations: 3,
                events: 40,
                query_every: 4,
                historical_query_rate: 0.0,
                ..StateTrackingBenchmarkConfig::default()
            },
            training: ExactEpisodeTrainingConfig {
                learning_rate: 1e-3,
                gradient_norm_clip: 0.25,
                parameter_abs_bound: 2.0,
                loss_epsilon: 1e-4,
            },
            train_world_seeds: vec![101, 102],
            test_world_seeds: vec![1001, 1002, 1003],
        }
    }

    /// First fixed research plan. This is intentionally exposed as code so any
    /// later change to scale/seeds/hyperparameters is reviewable in Git history.
    ///
    /// `research_v0` is a current-state-only exploratory preregistration. A
    /// historical-memory experiment requires a separate version after an explicit
    /// temporal-addressing algebra is implemented and qualified.
    pub fn research_v0() -> Self {
        Self {
            hls_config: HlsConfig {
                dim: 512,
                state_norm_limit: f32::INFINITY,
                activation: HlsActivation::Tanh,
                ..HlsConfig::default()
            },
            cell_seed: 0x484C_5330,
            codec_seed: 0x4844_4330,
            benchmark_template: StateTrackingBenchmarkConfig {
                entities: 16,
                objects: 32,
                locations: 8,
                events: 1_000,
                query_every: 10,
                min_dt: 1e-3,
                max_dt: 1e2,
                historical_query_rate: 0.0,
                ..StateTrackingBenchmarkConfig::default()
            },
            training: ExactEpisodeTrainingConfig {
                learning_rate: 1e-3,
                gradient_norm_clip: 0.5,
                parameter_abs_bound: 3.0,
                loss_epsilon: 1e-4,
            },
            train_world_seeds: (10_001..10_013).collect(),
            test_world_seeds: (20_001..20_025).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainingWorldResult {
    pub seed: u64,
    pub report: ExactEpisodeTrainingReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutWorldComparison {
    pub seed: u64,
    pub frozen: AssociativeEpisodeMetrics,
    pub trained: AssociativeEpisodeMetrics,
    /// Trained minus frozen. Negative is better for loss.
    pub mean_loss_delta: f64,
    /// Trained minus frozen. Positive is better for accuracy metrics.
    pub accuracy_delta: f64,
    pub compositional_accuracy_delta: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedEffectSummary {
    pub mean: f64,
    /// Sample standard error of the paired world-level deltas.
    pub standard_error: f64,
    pub min: f64,
    pub max: f64,
    pub positive_worlds: usize,
    pub negative_worlds: usize,
    pub zero_worlds: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactLearningAblationResult {
    pub training_worlds: Vec<TrainingWorldResult>,
    pub held_out_worlds: Vec<HeldOutWorldComparison>,
    pub loss_effect: PairedEffectSummary,
    pub accuracy_effect: PairedEffectSummary,
    pub compositional_accuracy_effect: PairedEffectSummary,
    pub initial_parameter_norm: f32,
    pub trained_parameter_norm: f32,
}

#[derive(Debug)]
pub enum ExactLearningAblationError {
    EmptyTrainingSeeds,
    EmptyTestSeeds,
    DuplicateTrainingSeed(u64),
    DuplicateTestSeed(u64),
    TrainTestSeedOverlap(u64),
    FiniteStateNormLimit,
    HistoricalQueriesEnabled(f64),
    Training(CurrentOnlyTrainingError),
    Cell(String),
    Codec(String),
    Benchmark(String),
}

impl fmt::Display for ExactLearningAblationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTrainingSeeds => write!(f, "learning ablation requires training seeds"),
            Self::EmptyTestSeeds => write!(f, "learning ablation requires held-out test seeds"),
            Self::DuplicateTrainingSeed(seed) => write!(f, "duplicate training seed {seed}"),
            Self::DuplicateTestSeed(seed) => write!(f, "duplicate test seed {seed}"),
            Self::TrainTestSeedOverlap(seed) => write!(f, "seed {seed} appears in both training and held-out sets"),
            Self::FiniteStateNormLimit => write!(f, "exact-learning ablation requires state_norm_limit = infinity"),
            Self::HistoricalQueriesEnabled(rate) => write!(
                f,
                "research_v0 is current-state-only; historical_query_rate must be exactly 0.0, got {rate}"
            ),
            Self::Training(error) => write!(f, "learning ablation training error: {error}"),
            Self::Cell(error) => write!(f, "learning ablation cell error: {error}"),
            Self::Codec(error) => write!(f, "learning ablation codec error: {error}"),
            Self::Benchmark(error) => write!(f, "learning ablation benchmark error: {error}"),
        }
    }
}

impl std::error::Error for ExactLearningAblationError {}

impl From<CurrentOnlyTrainingError> for ExactLearningAblationError {
    fn from(value: CurrentOnlyTrainingError) -> Self {
        Self::Training(value)
    }
}

/// Execute the exact learned-vs-frozen paired ablation.
pub fn run_exact_learning_ablation(
    plan: &ExactLearningAblationPlan,
) -> Result<ExactLearningAblationResult, ExactLearningAblationError> {
    // Validate the full experiment scope before constructing or mutating any HLS
    // state. In particular, historical-query drift fails here.
    validate_plan(plan)?;

    let initial = HolographicLiquidCell::try_new(plan.hls_config.clone(), plan.cell_seed)
        .map_err(|error| ExactLearningAblationError::Cell(error.to_string()))?;
    let frozen = initial.clone();
    let initial_parameter_norm = initial.parameters().l2_norm();
    let mut trained = initial;
    let mut training_worlds = Vec::with_capacity(plan.train_world_seeds.len());

    for &seed in &plan.train_world_seeds {
        let benchmark = world_from_template(&plan.benchmark_template, seed)?;
        let codec = StateTrackingCodec::from_benchmark_config(
            plan.hls_config.dim,
            &benchmark.config,
            plan.codec_seed,
        )
        .map_err(|error| ExactLearningAblationError::Codec(error.to_string()))?;
        let report = train_current_only_episode(&mut trained, &benchmark, &codec, &plan.training)?;
        training_worlds.push(TrainingWorldResult { seed, report });
    }

    let trained_parameter_norm = trained.parameters().l2_norm();
    let mut held_out_worlds = Vec::with_capacity(plan.test_world_seeds.len());
    for &seed in &plan.test_world_seeds {
        let benchmark = world_from_template(&plan.benchmark_template, seed)?;
        let codec = StateTrackingCodec::from_benchmark_config(
            plan.hls_config.dim,
            &benchmark.config,
            plan.codec_seed,
        )
        .map_err(|error| ExactLearningAblationError::Codec(error.to_string()))?;
        let frozen_metrics = evaluate_current_only_episode(
            &frozen,
            &benchmark,
            &codec,
            plan.training.loss_epsilon,
        )?;
        let trained_metrics = evaluate_current_only_episode(
            &trained,
            &benchmark,
            &codec,
            plan.training.loss_epsilon,
        )?;
        held_out_worlds.push(HeldOutWorldComparison {
            seed,
            mean_loss_delta: trained_metrics.mean_loss as f64 - frozen_metrics.mean_loss as f64,
            accuracy_delta: trained_metrics.score.accuracy() - frozen_metrics.score.accuracy(),
            compositional_accuracy_delta: trained_metrics.score.compositional_accuracy()
                - frozen_metrics.score.compositional_accuracy(),
            frozen: frozen_metrics,
            trained: trained_metrics,
        });
    }

    let loss_effect = summarize(held_out_worlds.iter().map(|row| row.mean_loss_delta));
    let accuracy_effect = summarize(held_out_worlds.iter().map(|row| row.accuracy_delta));
    let compositional_accuracy_effect = summarize(
        held_out_worlds
            .iter()
            .map(|row| row.compositional_accuracy_delta),
    );

    Ok(ExactLearningAblationResult {
        training_worlds,
        held_out_worlds,
        loss_effect,
        accuracy_effect,
        compositional_accuracy_effect,
        initial_parameter_norm,
        trained_parameter_norm,
    })
}

fn world_from_template(
    template: &StateTrackingBenchmarkConfig,
    seed: u64,
) -> Result<StateTrackingBenchmark, ExactLearningAblationError> {
    let mut config = template.clone();
    config.seed = seed;
    StateTrackingBenchmark::generate(config)
        .map_err(|error| ExactLearningAblationError::Benchmark(error.to_string()))
}

fn validate_plan(plan: &ExactLearningAblationPlan) -> Result<(), ExactLearningAblationError> {
    if plan.train_world_seeds.is_empty() {
        return Err(ExactLearningAblationError::EmptyTrainingSeeds);
    }
    if plan.test_world_seeds.is_empty() {
        return Err(ExactLearningAblationError::EmptyTestSeeds);
    }
    if plan.hls_config.state_norm_limit.is_finite() {
        return Err(ExactLearningAblationError::FiniteStateNormLimit);
    }
    if !plan.benchmark_template.historical_query_rate.is_finite()
        || plan.benchmark_template.historical_query_rate != 0.0
    {
        return Err(ExactLearningAblationError::HistoricalQueriesEnabled(
            plan.benchmark_template.historical_query_rate,
        ));
    }

    let mut train = HashSet::new();
    for &seed in &plan.train_world_seeds {
        if !train.insert(seed) {
            return Err(ExactLearningAblationError::DuplicateTrainingSeed(seed));
        }
    }
    let mut test = HashSet::new();
    for &seed in &plan.test_world_seeds {
        if !test.insert(seed) {
            return Err(ExactLearningAblationError::DuplicateTestSeed(seed));
        }
        if train.contains(&seed) {
            return Err(ExactLearningAblationError::TrainTestSeedOverlap(seed));
        }
    }
    Ok(())
}

fn summarize(values: impl Iterator<Item = f64>) -> PairedEffectSummary {
    let values = values.collect::<Vec<_>>();
    debug_assert!(!values.is_empty());
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let sample_variance = if n > 1 {
        values
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    PairedEffectSummary {
        mean,
        standard_error: (sample_variance / n as f64).sqrt(),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        positive_worlds: values.iter().filter(|value| **value > 0.0).count(),
        negative_worlds: values.iter().filter(|value| **value < 0.0).count(),
        zero_worlds: values.iter().filter(|value| **value == 0.0).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_ablation_is_deterministic_without_outcome_assumption() {
        let plan = ExactLearningAblationPlan::smoke();
        let a = run_exact_learning_ablation(&plan).unwrap();
        let b = run_exact_learning_ablation(&plan).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.training_worlds.len(), plan.train_world_seeds.len());
        assert_eq!(a.held_out_worlds.len(), plan.test_world_seeds.len());
        assert!(a.loss_effect.mean.is_finite());
        assert!(a.loss_effect.standard_error.is_finite());
        assert!(a.accuracy_effect.mean.is_finite());
        assert!(a.trained_parameter_norm.is_finite());
        // No assertion on the sign of learned-vs-frozen effects.
    }

    #[test]
    fn train_test_seed_overlap_is_rejected() {
        let mut plan = ExactLearningAblationPlan::smoke();
        plan.test_world_seeds[0] = plan.train_world_seeds[0];
        assert!(matches!(
            run_exact_learning_ablation(&plan),
            Err(ExactLearningAblationError::TrainTestSeedOverlap(_))
        ));
    }

    #[test]
    fn historical_scope_drift_is_rejected_before_execution() {
        let mut plan = ExactLearningAblationPlan::smoke();
        plan.benchmark_template.historical_query_rate = 0.5;
        assert!(matches!(
            run_exact_learning_ablation(&plan),
            Err(ExactLearningAblationError::HistoricalQueriesEnabled(rate)) if rate == 0.5
        ));
    }

    #[test]
    fn paired_summary_reports_world_signs_and_standard_error() {
        let summary = summarize([1.0, -1.0, 2.0, 0.0].into_iter());
        assert_eq!(summary.positive_worlds, 2);
        assert_eq!(summary.negative_worlds, 1);
        assert_eq!(summary.zero_worlds, 1);
        assert!((summary.mean - 0.5).abs() < 1e-12);
        assert!(summary.standard_error > 0.0);
    }
}
