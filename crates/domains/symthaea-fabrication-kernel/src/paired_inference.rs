// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic paired cluster-bootstrap inference for production evaluations.
//!
//! Supply one already-aggregated observation per independent unit (for example,
//! a production lot, independent shift, or site). Repeated parts, sensor samples,
//! or scenarios within one cluster must be aggregated before calling this module;
//! treating them as independent would create pseudo-replication.
//!
//! This module estimates uncertainty in a cluster-weighted mean paired difference.
//! It does not establish randomization, causal identification, measurement validity,
//! adequate statistical power, or physical provenance. Those remain protocol duties.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Minimum number of independent paired clusters before emitting a CI.
/// This is a conservative project gate, not a universal statistical theorem.
pub const MIN_INDEPENDENT_CLUSTERS: usize = 20;
/// Minimum bootstrap draws required for a report.
pub const MIN_BOOTSTRAP_REPLICATES: usize = 1_000;
/// Version for the machine-readable paired-inference report.
pub const PAIRED_INFERENCE_REPORT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

/// One pair of outcomes after aggregation within a truly independent cluster.
/// Delta is always candidate minus baseline; direction only labels better/worse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedClusterObservation {
    pub cluster_id: String,
    pub baseline: f64,
    pub candidate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedBootstrapConfig {
    pub replicates: usize,
    pub seed: u64,
    /// For example, 0.95 for a percentile 95% interval.
    pub confidence_level: f64,
}

impl Default for PairedBootstrapConfig {
    fn default() -> Self {
        Self {
            replicates: 10_000,
            seed: 0x4D46_4756_414C_5545,
            confidence_level: 0.95,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedBootstrapReport {
    /// Version of the serialized report contract.
    pub schema_version: u16,
    pub independent_cluster_count: usize,
    pub bootstrap_replicates: usize,
    pub seed: u64,
    pub confidence_level: f64,
    /// Equal-cluster-weighted mean of (candidate - baseline).
    pub mean_delta: f64,
    /// Median of the observed paired (candidate - baseline) deltas.
    pub median_delta: f64,
    /// Percentile-bootstrap confidence interval for mean_delta.
    pub interval_lower: f64,
    pub interval_upper: f64,
    pub candidate_better_cluster_count: usize,
    pub candidate_worse_cluster_count: usize,
    pub tied_cluster_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairedInferenceError {
    TooFewClusters { found: usize, minimum: usize },
    TooFewReplicates { found: usize, minimum: usize },
    InvalidConfidenceLevel,
    InvalidClusterId,
    DuplicateCluster { cluster_id: String },
    NonFiniteObservation { cluster_id: String },
    NonFiniteDelta { cluster_id: String },
    NonFiniteStatistic,
}

impl fmt::Display for PairedInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewClusters { found, minimum } => {
                write!(
                    f,
                    "found {found} independent clusters; need at least {minimum}"
                )
            }
            Self::TooFewReplicates { found, minimum } => {
                write!(
                    f,
                    "found {found} bootstrap replicates; need at least {minimum}"
                )
            }
            Self::InvalidConfidenceLevel => {
                write!(
                    f,
                    "confidence_level must be finite and strictly between 0.5 and 1.0"
                )
            }
            Self::InvalidClusterId => {
                write!(f, "cluster IDs must be non-empty and already trimmed")
            }
            Self::DuplicateCluster { cluster_id } => {
                write!(f, "cluster ID is duplicated: {cluster_id}")
            }
            Self::NonFiniteObservation { cluster_id } => {
                write!(f, "cluster {cluster_id} contains a non-finite outcome")
            }
            Self::NonFiniteDelta { cluster_id } => {
                write!(
                    f,
                    "candidate-minus-baseline delta is non-finite for cluster {cluster_id}"
                )
            }
            Self::NonFiniteStatistic => write!(f, "a statistic or bootstrap result is non-finite"),
        }
    }
}
impl Error for PairedInferenceError {}

/// Estimate a paired difference and percentile-bootstrap interval over independent
/// clusters. Inputs are sorted by cluster ID before sampling, so fixed inputs,
/// configuration, and seed produce an order-invariant report.
///
/// The report is intentionally descriptive: it emits no p-value, causal claim, or
/// automatic pass based only on whether the interval excludes zero.
pub fn analyze_paired_clusters(
    observations: &[PairedClusterObservation],
    direction: MetricDirection,
    config: PairedBootstrapConfig,
) -> Result<PairedBootstrapReport, PairedInferenceError> {
    if observations.len() < MIN_INDEPENDENT_CLUSTERS {
        return Err(PairedInferenceError::TooFewClusters {
            found: observations.len(),
            minimum: MIN_INDEPENDENT_CLUSTERS,
        });
    }
    if config.replicates < MIN_BOOTSTRAP_REPLICATES {
        return Err(PairedInferenceError::TooFewReplicates {
            found: config.replicates,
            minimum: MIN_BOOTSTRAP_REPLICATES,
        });
    }
    if !config.confidence_level.is_finite()
        || config.confidence_level <= 0.5
        || config.confidence_level >= 1.0
    {
        return Err(PairedInferenceError::InvalidConfidenceLevel);
    }

    let mut seen = BTreeSet::new();
    let mut rows = observations.to_vec();
    for row in &rows {
        if row.cluster_id.trim().is_empty() || row.cluster_id.trim() != row.cluster_id.as_str() {
            return Err(PairedInferenceError::InvalidClusterId);
        }
        if !seen.insert(row.cluster_id.as_str()) {
            return Err(PairedInferenceError::DuplicateCluster {
                cluster_id: row.cluster_id.clone(),
            });
        }
        if !row.baseline.is_finite() || !row.candidate.is_finite() {
            return Err(PairedInferenceError::NonFiniteObservation {
                cluster_id: row.cluster_id.clone(),
            });
        }
        if !(row.candidate - row.baseline).is_finite() {
            return Err(PairedInferenceError::NonFiniteDelta {
                cluster_id: row.cluster_id.clone(),
            });
        }
    }
    rows.sort_by(|a, b| a.cluster_id.cmp(&b.cluster_id));

    let deltas: Vec<f64> = rows.iter().map(|r| r.candidate - r.baseline).collect();
    let mean_delta = stable_mean(&deltas).ok_or(PairedInferenceError::NonFiniteStatistic)?;
    let mut sorted_deltas = deltas.clone();
    sorted_deltas.sort_by(f64::total_cmp);
    let median_delta =
        percentile(&sorted_deltas, 0.5).ok_or(PairedInferenceError::NonFiniteStatistic)?;

    let mut better = 0;
    let mut worse = 0;
    let mut tied = 0;
    for delta in &deltas {
        if *delta == 0.0 {
            tied += 1;
        } else {
            let is_better = match direction {
                MetricDirection::HigherIsBetter => *delta > 0.0,
                MetricDirection::LowerIsBetter => *delta < 0.0,
            };
            if is_better {
                better += 1;
            } else {
                worse += 1;
            }
        }
    }

    let mut rng = SplitMix64 { state: config.seed };
    let mut bootstrap_means = Vec::with_capacity(config.replicates);
    for _ in 0..config.replicates {
        let mut sampled_mean = 0.0;
        let mut correction = 0.0;
        for _ in 0..deltas.len() {
            let sampled = deltas[rng.index(deltas.len())];
            // Kahan-compensated summation avoids accumulating systematic
            // round-off in constant-sample known-answer cases.
            let term = sampled / deltas.len() as f64;
            let adjusted = term - correction;
            let next = sampled_mean + adjusted;
            correction = (next - sampled_mean) - adjusted;
            sampled_mean = next;
        }
        if !sampled_mean.is_finite() {
            return Err(PairedInferenceError::NonFiniteStatistic);
        }
        bootstrap_means.push(sampled_mean);
    }
    bootstrap_means.sort_by(f64::total_cmp);

    let tail_probability = (1.0 - config.confidence_level) / 2.0;
    let interval_lower = percentile(&bootstrap_means, tail_probability)
        .ok_or(PairedInferenceError::NonFiniteStatistic)?;
    let interval_upper = percentile(&bootstrap_means, 1.0 - tail_probability)
        .ok_or(PairedInferenceError::NonFiniteStatistic)?;

    Ok(PairedBootstrapReport {
        schema_version: PAIRED_INFERENCE_REPORT_SCHEMA_VERSION,
        independent_cluster_count: deltas.len(),
        bootstrap_replicates: config.replicates,
        seed: config.seed,
        confidence_level: config.confidence_level,
        mean_delta,
        median_delta,
        interval_lower,
        interval_upper,
        candidate_better_cluster_count: better,
        candidate_worse_cluster_count: worse,
        tied_cluster_count: tied,
    })
}

fn stable_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = values
        .iter()
        .fold(0.0, |acc, value| acc + value / values.len() as f64);
    mean.is_finite().then_some(mean)
}

/// Linear-interpolated quantile over a non-empty, sorted finite slice.
fn percentile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let position = probability * (sorted.len() - 1) as f64;
    let low_index = position.floor() as usize;
    let high_index = position.ceil() as usize;
    let fraction = position - low_index as f64;
    let low = sorted[low_index];
    let high = sorted[high_index];
    // Preserve exact endpoints for constant samples instead of introducing
    // floating-point noise through an algebraically redundant interpolation.
    if low == high {
        return Some(low);
    }
    let value = low * (1.0 - fraction) + high * fraction;
    value.is_finite().then_some(value)
}

/// Small specified PRNG for reproducible bootstrap sampling; not cryptographic.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (((self.next_u64() as u128) * (length as u128)) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observations(deltas: &[f64]) -> Vec<PairedClusterObservation> {
        deltas
            .iter()
            .enumerate()
            .map(|(index, delta)| PairedClusterObservation {
                cluster_id: format!("cluster-{index:03}"),
                baseline: 100.0,
                candidate: 100.0 + delta,
            })
            .collect()
    }

    fn config(seed: u64) -> PairedBootstrapConfig {
        PairedBootstrapConfig {
            replicates: 2_000,
            seed,
            confidence_level: 0.95,
        }
    }

    #[test]
    fn report_json_round_trip_is_versioned_and_deterministic() {
        let data = observations(&vec![-2.0; MIN_INDEPENDENT_CLUSTERS]);
        let report =
            analyze_paired_clusters(&data, MetricDirection::LowerIsBetter, config(7)).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let decoded: PairedBootstrapReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, decoded);
        assert_eq!(
            decoded.schema_version,
            PAIRED_INFERENCE_REPORT_SCHEMA_VERSION
        );
        let mut observation_json = serde_json::to_value(&data[0]).unwrap();
        observation_json
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_owned(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<PairedClusterObservation>(observation_json).is_err());
    }

    #[test]
    fn constant_improvement_has_exact_known_answer_interval() {
        let data = observations(&vec![-2.0; MIN_INDEPENDENT_CLUSTERS]);
        let report =
            analyze_paired_clusters(&data, MetricDirection::LowerIsBetter, config(7)).unwrap();
        assert_eq!(report.independent_cluster_count, MIN_INDEPENDENT_CLUSTERS);
        assert_eq!(report.mean_delta, -2.0);
        assert_eq!(report.median_delta, -2.0);
        assert_eq!(report.interval_lower, -2.0);
        assert_eq!(report.interval_upper, -2.0);
        assert_eq!(
            report.candidate_better_cluster_count,
            MIN_INDEPENDENT_CLUSTERS
        );
        assert_eq!(report.candidate_worse_cluster_count, 0);
        assert_eq!(report.tied_cluster_count, 0);
    }

    #[test]
    fn fixed_seed_is_deterministic_and_order_invariant() {
        let data = observations(&(-10..10).map(|n| n as f64).collect::<Vec<_>>());
        let first =
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, config(42)).unwrap();
        let mut reversed = data.clone();
        reversed.reverse();
        let second =
            analyze_paired_clusters(&reversed, MetricDirection::HigherIsBetter, config(42))
                .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn better_and_worse_counts_follow_declared_direction() {
        let data = observations(&[
            -2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 3.0, 0.5, -0.5, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0, 7.0,
            -7.0, 8.0, -8.0, 9.0,
        ]);
        let low =
            analyze_paired_clusters(&data, MetricDirection::LowerIsBetter, config(9)).unwrap();
        let high =
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, config(9)).unwrap();
        assert_eq!(
            low.candidate_better_cluster_count,
            high.candidate_worse_cluster_count
        );
        assert_eq!(
            low.candidate_worse_cluster_count,
            high.candidate_better_cluster_count
        );
        assert_eq!(low.tied_cluster_count, 1);
    }

    #[test]
    fn rejects_too_few_clusters() {
        let data = observations(&vec![1.0; MIN_INDEPENDENT_CLUSTERS - 1]);
        assert!(matches!(
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, config(1)),
            Err(PairedInferenceError::TooFewClusters { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_cluster_ids() {
        let mut data = observations(&vec![1.0; MIN_INDEPENDENT_CLUSTERS]);
        data[1].cluster_id = data[0].cluster_id.clone();
        assert!(matches!(
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, config(1)),
            Err(PairedInferenceError::DuplicateCluster { .. })
        ));
    }

    #[test]
    fn rejects_empty_or_untrimmed_cluster_ids_and_non_finite_values() {
        let mut data = observations(&vec![1.0; MIN_INDEPENDENT_CLUSTERS]);
        data[0].cluster_id = "  ".to_owned();
        assert!(matches!(
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, config(1)),
            Err(PairedInferenceError::InvalidClusterId)
        ));

        let mut data = observations(&vec![1.0; MIN_INDEPENDENT_CLUSTERS]);
        data[0].candidate = f64::NAN;
        assert!(matches!(
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, config(1)),
            Err(PairedInferenceError::NonFiniteObservation { .. })
        ));
    }

    #[test]
    fn rejects_invalid_configuration() {
        let data = observations(&vec![1.0; MIN_INDEPENDENT_CLUSTERS]);
        let too_few = PairedBootstrapConfig {
            replicates: 999,
            ..config(1)
        };
        assert!(matches!(
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, too_few),
            Err(PairedInferenceError::TooFewReplicates { .. })
        ));

        let bad_confidence = PairedBootstrapConfig {
            confidence_level: 1.0,
            ..config(1)
        };
        assert!(matches!(
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, bad_confidence),
            Err(PairedInferenceError::InvalidConfidenceLevel)
        ));
    }

    #[test]
    fn rejects_non_finite_delta_from_finite_extreme_inputs() {
        let mut data = observations(&vec![1.0; MIN_INDEPENDENT_CLUSTERS]);
        data[0].baseline = -f64::MAX;
        data[0].candidate = f64::MAX;
        assert!(matches!(
            analyze_paired_clusters(&data, MetricDirection::HigherIsBetter, config(1)),
            Err(PairedInferenceError::NonFiniteDelta { .. })
        ));
    }
}
