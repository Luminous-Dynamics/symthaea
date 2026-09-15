// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Seed-level summaries for the preregistered validity-memory falsification surface.
//!
//! The synthetic capacity protocol contains many queries inside each generated
//! history, but those queries share the same codebooks, archive state and random
//! seed. They are therefore not independent experimental replicates.
//!
//! This module makes the frozen replicate seed the analysis unit. It summarizes
//! each preregistered case across seeds only and deliberately exposes descriptive
//! statistics rather than p-values, confidence claims or a success threshold.
//! Raw per-seed observations remain authoritative.
//!
//! Repeated axis anchors are also treated as aliases of one numeric condition,
//! not additional replication. If two labeled cases share the same
//! `(dimension, key_count, candidate_count, horizon, span_length)` tuple, then
//! their same-seed falsification observations must agree bit-for-bit on every
//! axis-independent metric before any aggregation is emitted.

use crate::validity_capacity::{ValidityCapacityCase, ValidityCapacityPlan};
use crate::validity_capacity_falsification::{
    ValidityCapacityFalsificationObservation, ValidityCapacityFalsificationResult,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeedMetricSummary {
    pub seed_count: usize,
    pub mean: f64,
    pub median: f64,
    pub sample_standard_deviation: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub positive_seed_count: usize,
    pub negative_seed_count: usize,
    pub zero_seed_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacitySeedAggregateObservation {
    pub case: ValidityCapacityCase,
    pub seed_count: usize,
    pub accuracy_residual: SeedMetricSummary,
    pub target_mean_bias: SeedMetricSummary,
    pub target_mse_ratio_to_null: SeedMetricSummary,
    pub vocabulary_distractor_mean_bias: SeedMetricSummary,
    pub vocabulary_distractor_mse_ratio_to_null: SeedMetricSummary,
    pub shadow_distractor_mean_bias: SeedMetricSummary,
    pub shadow_distractor_mse_ratio_to_null: SeedMetricSummary,
    pub vocabulary_minus_shadow_mse: SeedMetricSummary,
    pub vocabulary_minus_shadow_variance: SeedMetricSummary,
    pub mean_true_margin: SeedMetricSummary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidityCapacitySeedAggregationResult {
    pub observations: Vec<ValidityCapacitySeedAggregateObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidityCapacitySeedAggregationError {
    EmptyCases,
    EmptySeeds,
    UnexpectedObservation {
        case: ValidityCapacityCase,
        seed: u64,
    },
    DuplicateObservation {
        case: ValidityCapacityCase,
        seed: u64,
    },
    MissingObservation {
        case: ValidityCapacityCase,
        seed: u64,
    },
    ConditionAliasMismatch {
        reference_case: ValidityCapacityCase,
        alias_case: ValidityCapacityCase,
        seed: u64,
    },
    NonFiniteMetric {
        case: ValidityCapacityCase,
        seed: u64,
        metric: &'static str,
    },
}

impl fmt::Display for ValidityCapacitySeedAggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCases => write!(f, "seed aggregation requires at least one capacity case"),
            Self::EmptySeeds => write!(f, "seed aggregation requires at least one replicate seed"),
            Self::UnexpectedObservation { case, seed } => write!(
                f,
                "seed aggregation received an observation outside the frozen plan: case={case:?}, seed={seed}"
            ),
            Self::DuplicateObservation { case, seed } => write!(
                f,
                "seed aggregation received duplicate observation: case={case:?}, seed={seed}"
            ),
            Self::MissingObservation { case, seed } => write!(
                f,
                "seed aggregation is missing frozen observation: case={case:?}, seed={seed}"
            ),
            Self::ConditionAliasMismatch {
                reference_case,
                alias_case,
                seed,
            } => write!(
                f,
                "same-seed numeric-condition aliases disagree: reference={reference_case:?}, alias={alias_case:?}, seed={seed}"
            ),
            Self::NonFiniteMetric { case, seed, metric } => write!(
                f,
                "seed aggregation metric {metric} is non-finite for case={case:?}, seed={seed}"
            ),
        }
    }
}

impl std::error::Error for ValidityCapacitySeedAggregationError {}

/// Aggregate the already-measured falsification surface using the frozen seed as
/// the independent replication unit.
///
/// Output order is exactly `plan.cases` order. Within each summary, source values
/// are collected in exactly `plan.replicate_seeds` order before deterministic
/// sorting for the median. Hash-map iteration order is never an output dependency.
///
/// Before summarizing, repeated labeled cases with the same numeric condition are
/// checked as deterministic aliases. Their same-seed observations must be exactly
/// equal at the integer level and bit-equal for every floating-point field after
/// ignoring only `case.axis`.
pub fn aggregate_validity_capacity_by_seed(
    plan: &ValidityCapacityPlan,
    result: &ValidityCapacityFalsificationResult,
) -> Result<ValidityCapacitySeedAggregationResult, ValidityCapacitySeedAggregationError> {
    if plan.cases.is_empty() {
        return Err(ValidityCapacitySeedAggregationError::EmptyCases);
    }
    if plan.replicate_seeds.is_empty() {
        return Err(ValidityCapacitySeedAggregationError::EmptySeeds);
    }

    validate_observation_set(plan, result)?;
    validate_condition_aliases(result)?;

    let mut observations = Vec::with_capacity(plan.cases.len());
    for &case in &plan.cases {
        let mut seed_observations = Vec::with_capacity(plan.replicate_seeds.len());
        for &seed in &plan.replicate_seeds {
            let observation = result
                .observations
                .iter()
                .find(|observation| observation.case == case && observation.seed == seed)
                .ok_or(ValidityCapacitySeedAggregationError::MissingObservation { case, seed })?;
            seed_observations.push(observation);
        }

        observations.push(ValidityCapacitySeedAggregateObservation {
            case,
            seed_count: seed_observations.len(),
            accuracy_residual: summarize_metric(
                case,
                &seed_observations,
                "accuracy_residual",
                |observation| observation.accuracy_residual,
            )?,
            target_mean_bias: summarize_metric(
                case,
                &seed_observations,
                "target_mean_bias",
                |observation| observation.target_mean_bias,
            )?,
            target_mse_ratio_to_null: summarize_metric(
                case,
                &seed_observations,
                "target_mse_ratio_to_null",
                |observation| observation.target_mse_ratio_to_null,
            )?,
            vocabulary_distractor_mean_bias: summarize_metric(
                case,
                &seed_observations,
                "vocabulary_distractor_mean_bias",
                |observation| observation.vocabulary_distractor_mean_bias,
            )?,
            vocabulary_distractor_mse_ratio_to_null: summarize_metric(
                case,
                &seed_observations,
                "vocabulary_distractor_mse_ratio_to_null",
                |observation| observation.vocabulary_distractor_mse_ratio_to_null,
            )?,
            shadow_distractor_mean_bias: summarize_metric(
                case,
                &seed_observations,
                "shadow_distractor_mean_bias",
                |observation| observation.shadow_distractor_mean_bias,
            )?,
            shadow_distractor_mse_ratio_to_null: summarize_metric(
                case,
                &seed_observations,
                "shadow_distractor_mse_ratio_to_null",
                |observation| observation.shadow_distractor_mse_ratio_to_null,
            )?,
            vocabulary_minus_shadow_mse: summarize_metric(
                case,
                &seed_observations,
                "vocabulary_minus_shadow_mse",
                |observation| observation.vocabulary_minus_shadow_mse,
            )?,
            vocabulary_minus_shadow_variance: summarize_metric(
                case,
                &seed_observations,
                "vocabulary_minus_shadow_variance",
                |observation| observation.vocabulary_minus_shadow_variance,
            )?,
            mean_true_margin: summarize_metric(
                case,
                &seed_observations,
                "mean_true_margin",
                |observation| observation.mean_true_margin,
            )?,
        });
    }

    Ok(ValidityCapacitySeedAggregationResult { observations })
}

fn validate_observation_set(
    plan: &ValidityCapacityPlan,
    result: &ValidityCapacityFalsificationResult,
) -> Result<(), ValidityCapacitySeedAggregationError> {
    let mut seen = HashSet::with_capacity(result.observations.len());
    for observation in &result.observations {
        if !plan.cases.contains(&observation.case)
            || !plan.replicate_seeds.contains(&observation.seed)
        {
            return Err(ValidityCapacitySeedAggregationError::UnexpectedObservation {
                case: observation.case,
                seed: observation.seed,
            });
        }
        if !seen.insert((observation.case, observation.seed)) {
            return Err(ValidityCapacitySeedAggregationError::DuplicateObservation {
                case: observation.case,
                seed: observation.seed,
            });
        }
    }

    for &case in &plan.cases {
        for &seed in &plan.replicate_seeds {
            if !seen.contains(&(case, seed)) {
                return Err(ValidityCapacitySeedAggregationError::MissingObservation {
                    case,
                    seed,
                });
            }
        }
    }
    Ok(())
}

fn validate_condition_aliases(
    result: &ValidityCapacityFalsificationResult,
) -> Result<(), ValidityCapacitySeedAggregationError> {
    type NumericCondition = (usize, usize, usize, u64, u64);
    let mut first_by_condition_seed: HashMap<
        (NumericCondition, u64),
        &ValidityCapacityFalsificationObservation,
    > = HashMap::new();

    for observation in &result.observations {
        let condition = numeric_condition(observation.case);
        let key = (condition, observation.seed);
        if let Some(reference) = first_by_condition_seed.get(&key) {
            if !axis_independent_observation_bits_equal(reference, observation) {
                return Err(ValidityCapacitySeedAggregationError::ConditionAliasMismatch {
                    reference_case: reference.case,
                    alias_case: observation.case,
                    seed: observation.seed,
                });
            }
        } else {
            first_by_condition_seed.insert(key, observation);
        }
    }

    Ok(())
}

fn numeric_condition(case: ValidityCapacityCase) -> (usize, usize, usize, u64, u64) {
    (
        case.dim,
        case.key_count,
        case.candidate_count,
        case.horizon,
        case.span_length,
    )
}

fn axis_independent_observation_bits_equal(
    left: &ValidityCapacityFalsificationObservation,
    right: &ValidityCapacityFalsificationObservation,
) -> bool {
    if numeric_condition(left.case) != numeric_condition(right.case)
        || left.seed != right.seed
        || left.total_queries != right.total_queries
    {
        return false;
    }

    let left_values = [
        left.empirical_accuracy,
        left.predicted_accuracy,
        left.accuracy_residual,
        left.accuracy_integration_refinement_delta,
        left.target_mean_bias,
        left.target_mean_squared_residual,
        left.target_mse_ratio_to_null,
        left.vocabulary_distractor_mean_bias,
        left.vocabulary_distractor_mean_squared_residual,
        left.vocabulary_distractor_mse_ratio_to_null,
        left.shadow_distractor_mean_bias,
        left.shadow_distractor_mean_squared_residual,
        left.shadow_distractor_mse_ratio_to_null,
        left.max_abs_shadow_candidate_similarity,
        left.vocabulary_minus_shadow_mse,
        left.vocabulary_minus_shadow_mse_ratio,
        left.vocabulary_minus_shadow_variance,
        left.mean_true_margin,
        left.smallest_true_margin,
    ];
    let right_values = [
        right.empirical_accuracy,
        right.predicted_accuracy,
        right.accuracy_residual,
        right.accuracy_integration_refinement_delta,
        right.target_mean_bias,
        right.target_mean_squared_residual,
        right.target_mse_ratio_to_null,
        right.vocabulary_distractor_mean_bias,
        right.vocabulary_distractor_mean_squared_residual,
        right.vocabulary_distractor_mse_ratio_to_null,
        right.shadow_distractor_mean_bias,
        right.shadow_distractor_mean_squared_residual,
        right.shadow_distractor_mse_ratio_to_null,
        right.max_abs_shadow_candidate_similarity,
        right.vocabulary_minus_shadow_mse,
        right.vocabulary_minus_shadow_mse_ratio,
        right.vocabulary_minus_shadow_variance,
        right.mean_true_margin,
        right.smallest_true_margin,
    ];

    left_values
        .iter()
        .zip(right_values.iter())
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn summarize_metric<F>(
    case: ValidityCapacityCase,
    observations: &[&ValidityCapacityFalsificationObservation],
    metric: &'static str,
    extract: F,
) -> Result<SeedMetricSummary, ValidityCapacitySeedAggregationError>
where
    F: Fn(&ValidityCapacityFalsificationObservation) -> f64,
{
    let mut values = Vec::with_capacity(observations.len());
    for observation in observations {
        let value = extract(observation);
        if !value.is_finite() {
            return Err(ValidityCapacitySeedAggregationError::NonFiniteMetric {
                case,
                seed: observation.seed,
                metric,
            });
        }
        values.push(value);
    }
    Ok(summarize_values(&values))
}

fn summarize_values(values: &[f64]) -> SeedMetricSummary {
    debug_assert!(!values.is_empty());
    let seed_count = values.len();
    let mean = values.iter().sum::<f64>() / seed_count as f64;

    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let median = if seed_count % 2 == 1 {
        sorted[seed_count / 2]
    } else {
        let right = seed_count / 2;
        0.5 * (sorted[right - 1] + sorted[right])
    };

    let sample_standard_deviation = if seed_count > 1 {
        let squared_deviation_sum = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>();
        (squared_deviation_sum / (seed_count - 1) as f64).sqrt()
    } else {
        0.0
    };

    let positive_seed_count = values.iter().filter(|value| **value > 0.0).count();
    let negative_seed_count = values.iter().filter(|value| **value < 0.0).count();
    let zero_seed_count = seed_count - positive_seed_count - negative_seed_count;

    SeedMetricSummary {
        seed_count,
        mean,
        median,
        sample_standard_deviation,
        minimum: sorted[0],
        maximum: sorted[seed_count - 1],
        positive_seed_count,
        negative_seed_count,
        zero_seed_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validity_capacity::{ValidityCapacityAxis, ValidityCapacityCase};
    use crate::validity_capacity_falsification::measure_validity_capacity_falsification_surface;

    #[test]
    fn smoke_aggregation_uses_seed_as_replication_unit() {
        let plan = ValidityCapacityPlan::smoke();
        let raw = measure_validity_capacity_falsification_surface(&plan).unwrap();
        let aggregated = aggregate_validity_capacity_by_seed(&plan, &raw).unwrap();

        assert_eq!(aggregated.observations.len(), plan.cases.len());
        for (summary, case) in aggregated.observations.iter().zip(&plan.cases) {
            assert_eq!(summary.case, *case);
            assert_eq!(summary.seed_count, plan.replicate_seeds.len());
            assert_eq!(summary.accuracy_residual.seed_count, plan.replicate_seeds.len());
            assert_eq!(
                summary.accuracy_residual.positive_seed_count
                    + summary.accuracy_residual.negative_seed_count
                    + summary.accuracy_residual.zero_seed_count,
                plan.replicate_seeds.len()
            );
        }
    }

    #[test]
    fn descriptive_summary_has_known_sample_statistics() {
        let summary = summarize_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(summary.seed_count, 5);
        assert_eq!(summary.mean.to_bits(), 3.0_f64.to_bits());
        assert_eq!(summary.median.to_bits(), 3.0_f64.to_bits());
        assert!((summary.sample_standard_deviation - 2.5_f64.sqrt()).abs() < 1.0e-15);
        assert_eq!(summary.minimum.to_bits(), 1.0_f64.to_bits());
        assert_eq!(summary.maximum.to_bits(), 5.0_f64.to_bits());
        assert_eq!(summary.positive_seed_count, 5);
        assert_eq!(summary.negative_seed_count, 0);
        assert_eq!(summary.zero_seed_count, 0);
    }

    #[test]
    fn aggregation_is_independent_of_raw_observation_order() {
        let plan = ValidityCapacityPlan::smoke();
        let raw = measure_validity_capacity_falsification_surface(&plan).unwrap();
        let expected = aggregate_validity_capacity_by_seed(&plan, &raw).unwrap();

        let mut reversed = raw.clone();
        reversed.observations.reverse();
        let actual = aggregate_validity_capacity_by_seed(&plan, &reversed).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn duplicate_and_missing_seed_observations_fail_closed() {
        let plan = ValidityCapacityPlan::smoke();
        let raw = measure_validity_capacity_falsification_surface(&plan).unwrap();

        let mut duplicate = raw.clone();
        duplicate.observations.push(raw.observations[0].clone());
        assert!(matches!(
            aggregate_validity_capacity_by_seed(&plan, &duplicate),
            Err(ValidityCapacitySeedAggregationError::DuplicateObservation { .. })
        ));

        let mut missing = raw.clone();
        missing.observations.pop();
        assert!(matches!(
            aggregate_validity_capacity_by_seed(&plan, &missing),
            Err(ValidityCapacitySeedAggregationError::MissingObservation { .. })
        ));
    }

    #[test]
    fn unexpected_seed_fails_closed() {
        let plan = ValidityCapacityPlan::smoke();
        let mut raw = measure_validity_capacity_falsification_surface(&plan).unwrap();
        raw.observations[0].seed = u64::MAX;
        assert!(matches!(
            aggregate_validity_capacity_by_seed(&plan, &raw),
            Err(ValidityCapacitySeedAggregationError::UnexpectedObservation { .. })
        ));
    }

    #[test]
    fn same_seed_axis_aliases_must_match_bit_for_bit() {
        let first_case = ValidityCapacityCase {
            axis: ValidityCapacityAxis::Dimension,
            dim: 64,
            key_count: 2,
            candidate_count: 2,
            horizon: 8,
            span_length: 2,
        };
        let alias_case = ValidityCapacityCase {
            axis: ValidityCapacityAxis::Horizon,
            ..first_case
        };
        let plan = ValidityCapacityPlan {
            cases: vec![first_case, alias_case],
            replicate_seeds: vec![7],
        };
        let first = falsification_fixture(first_case, 7);
        let alias = falsification_fixture(alias_case, 7);
        let consistent = ValidityCapacityFalsificationResult {
            observations: vec![first.clone(), alias.clone()],
        };
        assert!(aggregate_validity_capacity_by_seed(&plan, &consistent).is_ok());

        let mut changed = alias;
        changed.accuracy_residual = f64::from_bits(changed.accuracy_residual.to_bits() ^ 1);
        let inconsistent = ValidityCapacityFalsificationResult {
            observations: vec![first, changed],
        };
        assert!(matches!(
            aggregate_validity_capacity_by_seed(&plan, &inconsistent),
            Err(ValidityCapacitySeedAggregationError::ConditionAliasMismatch { .. })
        ));
    }

    fn falsification_fixture(
        case: ValidityCapacityCase,
        seed: u64,
    ) -> ValidityCapacityFalsificationObservation {
        ValidityCapacityFalsificationObservation {
            case,
            seed,
            total_queries: 8,
            empirical_accuracy: 0.75,
            predicted_accuracy: 0.7,
            accuracy_residual: 0.05,
            accuracy_integration_refinement_delta: 1.0e-12,
            target_mean_bias: 0.01,
            target_mean_squared_residual: 0.2,
            target_mse_ratio_to_null: 1.1,
            vocabulary_distractor_mean_bias: -0.02,
            vocabulary_distractor_mean_squared_residual: 0.3,
            vocabulary_distractor_mse_ratio_to_null: 1.2,
            shadow_distractor_mean_bias: 0.03,
            shadow_distractor_mean_squared_residual: 0.25,
            shadow_distractor_mse_ratio_to_null: 1.0,
            max_abs_shadow_candidate_similarity: 0.04,
            vocabulary_minus_shadow_mse: 0.05,
            vocabulary_minus_shadow_mse_ratio: 1.2,
            vocabulary_minus_shadow_variance: 0.01,
            mean_true_margin: 0.4,
            smallest_true_margin: -0.1,
        }
    }
}
