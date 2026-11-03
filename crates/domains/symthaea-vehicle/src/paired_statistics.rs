// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Family-clustered paired statistics for controller comparison.
//!
//! Scenario variants are first aggregated within their declared failure family.
//! Bootstrap resampling then samples families, not individual variants, to
//! avoid treating near-duplicate siblings as independent evidence.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub const PAIRED_STATISTICS_SCHEMA: &str = "symthaea.vehicle.paired-statistics.v0.1";
pub const MIN_PAIRED_BOOTSTRAP_REPLICATES: u32 = 1_000;
pub const MAX_PAIRED_BOOTSTRAP_REPLICATES: u32 = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairedMetricDirection {
    /// Improvement is baseline minus advanced.
    LowerIsBetter,
    /// Improvement is advanced minus baseline.
    HigherIsBetter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedMetricObservation {
    pub failure_family_id: String,
    pub scenario_id: String,
    pub advanced_value: f64,
    pub baseline_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedFamilyEffect {
    pub failure_family_id: String,
    pub scenario_ids: Vec<String>,
    pub mean_advanced_value: f64,
    pub mean_baseline_value: f64,
    /// Positive values always favor the advanced controller.
    pub mean_improvement: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairedStatisticsError {
    EmptyMetricName,
    InsufficientFailureFamilies,
    EmptyFailureFamilyId,
    EmptyScenarioId,
    DuplicateScenarioId,
    NonFiniteObservation,
    InvalidBootstrapReplicateCount,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedMetricSummary {
    pub schema_version: String,
    pub metric_name: String,
    pub direction: PairedMetricDirection,
    pub seed: u64,
    pub bootstrap_replicates: u32,
    pub failure_family_count: usize,
    pub scenario_count: usize,
    /// Exact paired observations used to reconstruct family aggregates.
    pub observations: Vec<PairedMetricObservation>,
    /// Macro-average over family means; each failure family receives equal weight.
    pub mean_family_improvement: f64,
    pub median_family_improvement: f64,
    pub family_win_fraction: f64,
    pub worst_family_improvement: f64,
    pub percentile_95_low: f64,
    pub percentile_95_high: f64,
    pub family_effects: Vec<PairedFamilyEffect>,
    /// BLAKE3 over the serialized summary with this field empty; not a signature.
    pub summary_digest: String,
}

impl PairedMetricSummary {
    /// Check receipt digest and reconstruct every aggregate and bootstrap interval.
    pub fn identity_is_valid(&self) -> bool {
        if self.schema_version != PAIRED_STATISTICS_SCHEMA
            || self.metric_name.trim().is_empty()
            || self.failure_family_count != self.family_effects.len()
            || self.failure_family_count < 2
            || !(MIN_PAIRED_BOOTSTRAP_REPLICATES..=MAX_PAIRED_BOOTSTRAP_REPLICATES)
                .contains(&self.bootstrap_replicates)
            || self.summary_digest.len() != 64
            || !self.summary_digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return false;
        }

        let mut family_ids = BTreeSet::new();
        let mut scenario_ids = BTreeSet::new();
        let mut grouped: BTreeMap<String, Vec<&PairedMetricObservation>> = BTreeMap::new();
        if self.observations.len() != self.scenario_count {
            return false;
        }
        for observation in &self.observations {
            if observation.failure_family_id.trim().is_empty()
                || observation.scenario_id.trim().is_empty()
                || !observation.advanced_value.is_finite()
                || !observation.baseline_value.is_finite()
                || !scenario_ids.insert(observation.scenario_id.as_str())
            {
                return false;
            }
            grouped
                .entry(observation.failure_family_id.clone())
                .or_default()
                .push(observation);
        }
        if grouped.len() != self.failure_family_count {
            return false;
        }

        if self.observations.windows(2).any(|pair| {
            let previous_family = pair[0].failure_family_id.as_str();
            let current_family = pair[1].failure_family_id.as_str();
            let previous_scenario = pair[0].scenario_id.as_str();
            let current_scenario = pair[1].scenario_id.as_str();
            previous_family > current_family
                || (previous_family == current_family
                    && previous_scenario >= current_scenario)
        }) || self.family_effects.windows(2).any(|pair| {
            pair[0].failure_family_id.as_str() >= pair[1].failure_family_id.as_str()
        }) {
            return false;
        }

        let mut improvements = Vec::with_capacity(self.family_effects.len());
        let mut scenario_count = 0usize;
        for family in &self.family_effects {
            if family.failure_family_id.trim().is_empty()
                || !family_ids.insert(family.failure_family_id.as_str())
                || family.scenario_ids.is_empty()
                || !family.mean_advanced_value.is_finite()
                || !family.mean_baseline_value.is_finite()
                || !family.mean_improvement.is_finite()
            {
                return false;
            }
            let Some(observations) = grouped.get(&family.failure_family_id) else {
                return false;
            };
            let mut expected_scenario_ids = observations
                .iter()
                .map(|observation| observation.scenario_id.as_str())
                .collect::<Vec<_>>();
            expected_scenario_ids.sort_unstable();
            let mut declared_scenario_ids = family
                .scenario_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            declared_scenario_ids.sort_unstable();
            let mean_advanced = arithmetic_mean(
                &observations
                    .iter()
                    .map(|observation| observation.advanced_value)
                    .collect::<Vec<_>>(),
            );
            let mean_baseline = arithmetic_mean(
                &observations
                    .iter()
                    .map(|observation| observation.baseline_value)
                    .collect::<Vec<_>>(),
            );
            let expected_improvement = match self.direction {
                PairedMetricDirection::LowerIsBetter => mean_baseline - mean_advanced,
                PairedMetricDirection::HigherIsBetter => mean_advanced - mean_baseline,
            };
            if declared_scenario_ids != expected_scenario_ids
                || !approximately_equal(family.mean_advanced_value, mean_advanced)
                || !approximately_equal(family.mean_baseline_value, mean_baseline)
                || !approximately_equal(family.mean_improvement, expected_improvement)
            {
                return false;
            }
            scenario_count += family.scenario_ids.len();
            improvements.push(family.mean_improvement);
        }

        if scenario_count != self.scenario_count || family_ids.len() != grouped.len() {
            return false;
        }
        let derived_mean = arithmetic_mean(&improvements);
        let mut sorted = improvements.clone();
        sorted.sort_by(f64::total_cmp);
        let derived_median = percentile(&sorted, 0.5);
        let derived_worst = sorted[0];
        let derived_wins = improvements.iter().filter(|value| **value > 0.0).count() as f64
            / improvements.len() as f64;
        let bootstrap = bootstrap_family_means(
            &improvements,
            self.seed,
            self.bootstrap_replicates,
        );

        if !approximately_equal(self.mean_family_improvement, derived_mean)
            || !approximately_equal(self.median_family_improvement, derived_median)
            || !approximately_equal(self.family_win_fraction, derived_wins)
            || !approximately_equal(self.worst_family_improvement, derived_worst)
            || !approximately_equal(self.percentile_95_low, bootstrap.0)
            || !approximately_equal(self.percentile_95_high, bootstrap.1)
        {
            return false;
        }

        let mut summary = self.clone();
        let expected = summary.summary_digest.clone();
        summary.summary_digest.clear();
        serde_json::to_vec(&summary)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string() == expected)
            .unwrap_or(false)
    }
}

/// Summarize matched advanced-vs-baseline metric values using failure-family
/// clusters as the independent resampling units.
///
/// Duplicate scenario IDs are rejected. All scenario variants in a failure
/// family are averaged first; bootstrap resampling then draws the same number
/// of family means with replacement. The resulting interval describes
/// between-family uncertainty for this declared family corpus, not a
/// real-world fleet-risk interval or an estimate of scenario prevalence.
pub fn summarize_paired_metric(
    metric_name: impl Into<String>,
    direction: PairedMetricDirection,
    observations: &[PairedMetricObservation],
    seed: u64,
    bootstrap_replicates: u32,
) -> Result<PairedMetricSummary, PairedStatisticsError> {
    let metric_name = metric_name.into();
    if metric_name.trim().is_empty() {
        return Err(PairedStatisticsError::EmptyMetricName);
    }
    if !(MIN_PAIRED_BOOTSTRAP_REPLICATES..=MAX_PAIRED_BOOTSTRAP_REPLICATES)
        .contains(&bootstrap_replicates)
    {
        return Err(PairedStatisticsError::InvalidBootstrapReplicateCount);
    }

    let mut grouped: BTreeMap<String, Vec<(String, f64, f64)>> = BTreeMap::new();
    let mut all_scenario_ids = BTreeSet::new();
    for observation in observations {
        if observation.failure_family_id.trim().is_empty() {
            return Err(PairedStatisticsError::EmptyFailureFamilyId);
        }
        if observation.scenario_id.trim().is_empty() {
            return Err(PairedStatisticsError::EmptyScenarioId);
        }
        if !observation.advanced_value.is_finite() || !observation.baseline_value.is_finite() {
            return Err(PairedStatisticsError::NonFiniteObservation);
        }
        if !all_scenario_ids.insert(observation.scenario_id.as_str()) {
            return Err(PairedStatisticsError::DuplicateScenarioId);
        }
        grouped.entry(observation.failure_family_id.clone()).or_default().push((
            observation.scenario_id.clone(),
            observation.advanced_value,
            observation.baseline_value,
        ));
    }

    if grouped.len() < 2 {
        return Err(PairedStatisticsError::InsufficientFailureFamilies);
    }

    let mut family_effects = Vec::with_capacity(grouped.len());
    for (failure_family_id, mut family_observations) in grouped {
        family_observations.sort_by(|left, right| left.0.cmp(&right.0));
        let advanced = family_observations
            .iter()
            .map(|(_, value, _)| *value)
            .collect::<Vec<_>>();
        let baseline = family_observations
            .iter()
            .map(|(_, _, value)| *value)
            .collect::<Vec<_>>();
        let mean_advanced_value = arithmetic_mean(&advanced);
        let mean_baseline_value = arithmetic_mean(&baseline);
        let mean_improvement = match direction {
            PairedMetricDirection::LowerIsBetter => mean_baseline_value - mean_advanced_value,
            PairedMetricDirection::HigherIsBetter => mean_advanced_value - mean_baseline_value,
        };
        if !mean_advanced_value.is_finite()
            || !mean_baseline_value.is_finite()
            || !mean_improvement.is_finite()
        {
            return Err(PairedStatisticsError::NonFiniteObservation);
        }
        family_effects.push(PairedFamilyEffect {
            failure_family_id,
            scenario_ids: family_observations
                .into_iter()
                .map(|(scenario_id, _, _)| scenario_id)
                .collect(),
            mean_advanced_value,
            mean_baseline_value,
            mean_improvement,
        });
    }

    let improvements = family_effects
        .iter()
        .map(|family| family.mean_improvement)
        .collect::<Vec<_>>();
    let mut retained_observations = observations.to_vec();
    retained_observations.sort_by(|left, right| {
        left.failure_family_id
            .cmp(&right.failure_family_id)
            .then_with(|| left.scenario_id.cmp(&right.scenario_id))
    });
    let mut sorted = improvements.clone();
    sorted.sort_by(f64::total_cmp);
    let bootstrap = bootstrap_family_means(&improvements, seed, bootstrap_replicates);
    let mut summary = PairedMetricSummary {
        schema_version: PAIRED_STATISTICS_SCHEMA.into(),
        metric_name,
        direction,
        seed,
        bootstrap_replicates,
        failure_family_count: family_effects.len(),
        scenario_count: family_effects.iter().map(|family| family.scenario_ids.len()).sum(),
        observations: retained_observations,
        mean_family_improvement: arithmetic_mean(&improvements),
        median_family_improvement: percentile(&sorted, 0.5),
        family_win_fraction: improvements.iter().filter(|value| **value > 0.0).count() as f64
            / improvements.len() as f64,
        worst_family_improvement: sorted[0],
        percentile_95_low: bootstrap.0,
        percentile_95_high: bootstrap.1,
        family_effects,
        summary_digest: String::new(),
    };
    let bytes = serde_json::to_vec(&summary)
        .map_err(|_| PairedStatisticsError::NonFiniteObservation)?;
    summary.summary_digest = blake3::hash(&bytes).to_hex().to_string();
    Ok(summary)
}

fn bootstrap_family_means(values: &[f64], seed: u64, replicates: u32) -> (f64, f64) {
    let mut rng = SplitMix64(seed);
    let mut means = Vec::with_capacity(replicates as usize);
    for _ in 0..replicates {
        let mut sample = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            let index = rng.index_below(values.len() as u64) as usize;
            sample.push(values[index]);
        }
        means.push(arithmetic_mean(&sample));
    }
    means.sort_by(f64::total_cmp);
    (percentile(&means, 0.025), percentile(&means, 0.975))
}

fn arithmetic_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    // Divide before summation so a finite mean does not overflow merely
    // because the unscaled sum would exceed f64::MAX.
    let divisor = values.len() as f64;
    values.iter().map(|value| value / divisor).sum()
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = (sorted.len() - 1) as f64 * probability.clamp(0.0, 1.0);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= 1e-10 * left.abs().max(right.abs()).max(1.0)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn index_below(&mut self, upper_exclusive: u64) -> u64 {
        let limit = u64::MAX - (u64::MAX % upper_exclusive);
        loop {
            let value = self.next_u64();
            if value < limit {
                return value % upper_exclusive;
            }
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observations() -> Vec<PairedMetricObservation> {
        vec![
            PairedMetricObservation {
                failure_family_id: "crossing-occlusion".into(),
                scenario_id: "crossing-001".into(),
                advanced_value: 2.0,
                baseline_value: 4.0,
            },
            PairedMetricObservation {
                failure_family_id: "crossing-occlusion".into(),
                scenario_id: "crossing-002".into(),
                advanced_value: 4.0,
                baseline_value: 6.0,
            },
            PairedMetricObservation {
                failure_family_id: "friction-loss".into(),
                scenario_id: "friction-001".into(),
                advanced_value: 3.0,
                baseline_value: 5.0,
            },
            PairedMetricObservation {
                failure_family_id: "sensor-delay".into(),
                scenario_id: "sensor-001".into(),
                advanced_value: 7.0,
                baseline_value: 6.0,
            },
        ]
    }

    #[test]
    fn aggregates_siblings_before_bootstrapping_failure_families() {
        let summary = summarize_paired_metric(
            "rms_lateral_error_m",
            PairedMetricDirection::LowerIsBetter,
            &observations(),
            42,
            1000,
        )
        .unwrap();
        assert_eq!(summary.failure_family_count, 3);
        assert_eq!(summary.scenario_count, 4);
        assert_eq!(summary.family_effects[0].scenario_ids.len(), 2);
        assert!(summary.identity_is_valid());
    }

    #[test]
    fn positive_improvement_always_favors_advanced_controller() {
        let lower = summarize_paired_metric(
            "error",
            PairedMetricDirection::LowerIsBetter,
            &observations(),
            7,
            1000,
        )
        .unwrap();
        let higher_observations = observations()
            .into_iter()
            .map(|mut observation| {
                observation.advanced_value = -observation.advanced_value;
                observation.baseline_value = -observation.baseline_value;
                observation
            })
            .collect::<Vec<_>>();
        let higher = summarize_paired_metric(
            "utility",
            PairedMetricDirection::HigherIsBetter,
            &higher_observations,
            7,
            1000,
        )
        .unwrap();
        assert!(lower.mean_family_improvement > 0.0);
        assert!(higher.mean_family_improvement > 0.0);
    }

    #[test]
    fn bootstrap_summary_is_deterministic_for_a_frozen_seed() {
        let first = summarize_paired_metric(
            "error",
            PairedMetricDirection::LowerIsBetter,
            &observations(),
            1234,
            2000,
        )
        .unwrap();
        let second = summarize_paired_metric(
            "error",
            PairedMetricDirection::LowerIsBetter,
            &observations(),
            1234,
            2000,
        )
        .unwrap();
        assert_eq!(first.summary_digest, second.summary_digest);
        assert_eq!(first.percentile_95_low, second.percentile_95_low);
        assert_eq!(first.percentile_95_high, second.percentile_95_high);
    }

    #[test]
    fn input_order_does_not_change_canonical_summary() {
        let input = observations();
        let mut reversed = input.clone();
        reversed.reverse();
        let first = summarize_paired_metric(
            "error",
            PairedMetricDirection::LowerIsBetter,
            &input,
            77,
            1000,
        )
        .unwrap();
        let second = summarize_paired_metric(
            "error",
            PairedMetricDirection::LowerIsBetter,
            &reversed,
            77,
            1000,
        )
        .unwrap();
        assert_eq!(first.summary_digest, second.summary_digest);
        assert!(first.identity_is_valid());
        assert!(second.identity_is_valid());
    }

    #[test]
    fn duplicate_scenario_ids_are_rejected() {
        let mut input = observations();
        input[1].scenario_id = input[0].scenario_id.clone();
        assert_eq!(
            summarize_paired_metric(
                "error",
                PairedMetricDirection::LowerIsBetter,
                &input,
                1,
                1000,
            )
            .unwrap_err(),
            PairedStatisticsError::DuplicateScenarioId
        );
    }

    #[test]
    fn one_family_and_nonfinite_values_are_rejected() {
        let mut one_family = observations();
        for observation in &mut one_family {
            observation.failure_family_id = "single-family".into();
        }
        assert_eq!(
            summarize_paired_metric(
                "error",
                PairedMetricDirection::LowerIsBetter,
                &one_family,
                1,
                1000,
            )
            .unwrap_err(),
            PairedStatisticsError::InsufficientFailureFamilies
        );
        let mut nonfinite = observations();
        nonfinite[0].advanced_value = f64::NAN;
        assert_eq!(
            summarize_paired_metric(
                "error",
                PairedMetricDirection::LowerIsBetter,
                &nonfinite,
                1,
                1000,
            )
            .unwrap_err(),
            PairedStatisticsError::NonFiniteObservation
        );
    }

    #[test]
    fn finite_extreme_samples_do_not_overflow_just_while_averaging() {
        let input = vec![
            PairedMetricObservation {
                failure_family_id: "family-a".into(),
                scenario_id: "a".into(),
                advanced_value: f64::MAX,
                baseline_value: f64::MAX,
            },
            PairedMetricObservation {
                failure_family_id: "family-b".into(),
                scenario_id: "b".into(),
                advanced_value: f64::MAX,
                baseline_value: f64::MAX,
            },
        ];
        let summary = summarize_paired_metric(
            "large_finite_metric",
            PairedMetricDirection::LowerIsBetter,
            &input,
            11,
            1000,
        )
        .unwrap();
        assert_eq!(summary.mean_family_improvement, 0.0);
        assert!(summary.identity_is_valid());
    }

    #[test]
    fn finite_inputs_with_unrepresentable_paired_difference_are_rejected() {
        let input = vec![
            PairedMetricObservation {
                failure_family_id: "family-a".into(),
                scenario_id: "a".into(),
                advanced_value: -f64::MAX,
                baseline_value: f64::MAX,
            },
            PairedMetricObservation {
                failure_family_id: "family-b".into(),
                scenario_id: "b".into(),
                advanced_value: 0.0,
                baseline_value: 0.0,
            },
        ];
        assert_eq!(
            summarize_paired_metric(
                "unrepresentable_difference",
                PairedMetricDirection::LowerIsBetter,
                &input,
                12,
                1000,
            )
            .unwrap_err(),
            PairedStatisticsError::NonFiniteObservation
        );
    }

    #[test]
    fn recomputed_digest_does_not_hide_inconsistent_aggregates() {
        let mut summary = summarize_paired_metric(
            "error",
            PairedMetricDirection::LowerIsBetter,
            &observations(),
            3,
            1000,
        )
        .unwrap();
        summary.mean_family_improvement += 1.0;
        summary.family_effects[0].mean_advanced_value += 1.0;
        summary.summary_digest.clear();
        let bytes = serde_json::to_vec(&summary).unwrap();
        summary.summary_digest = blake3::hash(&bytes).to_hex().to_string();
        assert!(!summary.identity_is_valid());
    }
}
