// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Paired counterfactual outcome scoring for adaptive retrieval evaluation.
//!
//! The runtime VOI estimator must be calibrated against observed marginal
//! outcomes. This module consumes a matched no-retrieval / retrieval pair for
//! each query, derives disjoint utility deltas and reports a deterministic
//! cluster-bootstrap interval over independent source/user/conversation groups.
//!
//! It does not execute retrieval or answer generation. The experiment runner
//! supplies paired outcomes from the same frozen query, corpus snapshot,
//! reader, judge and evaluation rubric. Unpaired observations are rejected by
//! construction because one record must contain both potential outcomes.

use crate::memory_retrieval_value_calibration::RetrievalUtilityCalibrationObservation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const BOOTSTRAP_REPLICATES: usize = 2_000;
const MIN_BOOTSTRAP_GROUPS: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalUtilityPointPrediction {
    pub correctness_gain_micro: i64,
    pub lifecycle_risk_reduction_micro: i64,
    pub retrieval_harm_micro: i64,
}

/// One paired query result under no retrieval and one additional retrieval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedRetrievalOutcome {
    pub observation_id: String,
    pub calibration_group_id: String,
    pub query_fingerprint_sha256: String,
    /// All component scores use the frozen utility schema, higher is better.
    pub no_retrieval_correctness_micro: i64,
    pub retrieval_correctness_micro: i64,
    /// Loss score, higher is worse (e.g. stale/unauthorized/conflict misuse).
    pub no_retrieval_lifecycle_loss_micro: i64,
    pub retrieval_lifecycle_loss_micro: i64,
    /// Independent harm rubric, higher is worse (e.g. distractor intrusion).
    /// It must not duplicate the correctness or lifecycle-loss dimensions.
    pub no_retrieval_harm_micro: i64,
    pub retrieval_harm_micro: i64,
    /// Measured cost of the additional retrieval policy, including router cost.
    pub retrieval_cost_micro: u64,
    pub retrieval_calls: u32,
    pub retrieval_tokens: u64,
    pub retrieval_latency_ms: u64,
}

impl PairedRetrievalOutcome {
    pub fn validate(&self) -> Result<(), String> {
        require_non_empty(&self.observation_id, "observation_id")?;
        require_non_empty(&self.calibration_group_id, "calibration_group_id")?;
        validate_sha256(&self.query_fingerprint_sha256, "query_fingerprint_sha256")
    }

    pub fn correctness_gain_micro(&self) -> Result<i64, String> {
        self.retrieval_correctness_micro
            .checked_sub(self.no_retrieval_correctness_micro)
            .ok_or_else(|| "correctness gain delta exceeds i64 range".into())
    }

    pub fn lifecycle_risk_reduction_micro(&self) -> Result<i64, String> {
        self.no_retrieval_lifecycle_loss_micro
            .checked_sub(self.retrieval_lifecycle_loss_micro)
            .ok_or_else(|| "lifecycle risk-reduction delta exceeds i64 range".into())
    }

    pub fn retrieval_harm_delta_micro(&self) -> Result<i64, String> {
        self.retrieval_harm_micro
            .checked_sub(self.no_retrieval_harm_micro)
            .ok_or_else(|| "retrieval harm delta exceeds i64 range".into())
    }

    pub fn net_utility_delta_micro(&self) -> i128 {
        let correctness = i128::from(self.retrieval_correctness_micro)
            - i128::from(self.no_retrieval_correctness_micro);
        let lifecycle_reduction = i128::from(self.no_retrieval_lifecycle_loss_micro)
            - i128::from(self.retrieval_lifecycle_loss_micro);
        let harm = i128::from(self.retrieval_harm_micro)
            - i128::from(self.no_retrieval_harm_micro);
        correctness + lifecycle_reduction - harm - i128::from(self.retrieval_cost_micro)
    }

    /// Convert an outcome into one held-out calibration observation for a
    /// frozen point predictor. The same query fingerprint binds both potential
    /// outcomes; the group ID controls leakage-safe calibration.
    pub fn to_calibration_observation(
        &self,
        prediction: &RetrievalUtilityPointPrediction,
    ) -> Result<RetrievalUtilityCalibrationObservation, String> {
        Ok(RetrievalUtilityCalibrationObservation {
            observation_id: self.observation_id.clone(),
            calibration_group_id: self.calibration_group_id.clone(),
            query_fingerprint_sha256: self.query_fingerprint_sha256.clone(),
            predicted_correctness_gain_micro: prediction.correctness_gain_micro,
            observed_correctness_gain_micro: self.correctness_gain_micro()?,
            predicted_lifecycle_risk_reduction_micro: prediction.lifecycle_risk_reduction_micro,
            observed_lifecycle_risk_reduction_micro: self.lifecycle_risk_reduction_micro()?,
            predicted_retrieval_harm_micro: prediction.retrieval_harm_micro,
            observed_retrieval_harm_micro: self.retrieval_harm_delta_micro()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBootstrapInterval {
    pub lower_95_micro: i64,
    pub upper_95_micro: i64,
    pub independent_group_count: usize,
    pub replicate_count: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedRetrievalUtilitySummary {
    pub observation_count: usize,
    pub independent_group_count: usize,
    pub positive_net_utility_count: usize,
    pub neutral_net_utility_count: usize,
    pub negative_net_utility_count: usize,
    pub mean_correctness_gain_micro: i64,
    pub mean_lifecycle_risk_reduction_micro: i64,
    pub mean_retrieval_harm_delta_micro: i64,
    /// Per-query weighted mean over all paired observations.
    pub mean_net_utility_delta_micro: i64,
    /// Equal-weight mean across independent calibration/bootstrap groups.
    pub mean_group_net_utility_delta_micro: i64,
    pub mean_retrieval_cost_micro: u64,
    pub mean_retrieval_calls: u64,
    pub mean_retrieval_tokens: u64,
    pub mean_retrieval_latency_ms: u64,
    pub net_utility_cluster_bootstrap_95: ClusterBootstrapInterval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupAggregate {
    net_utility_sum: i128,
    observation_count: usize,
}

/// Summarize paired outcomes with a deterministic source-group bootstrap.
///
/// The interval resamples independent groups, not individual questions, and
/// estimates the mean of group means so a single large source document cannot
/// dominate the interval. At least 20 independent groups are required. The
/// percentile interval is a descriptive paired uncertainty estimate; it is
/// not a conformal guarantee and does not correct multiple treatment-arm
/// comparisons. Pre-register the primary policy contrast.
pub fn summarize_paired_retrieval_outcomes(
    outcomes: &[PairedRetrievalOutcome],
    seed: u64,
) -> Result<PairedRetrievalUtilitySummary, String> {
    if outcomes.is_empty() {
        return Err("paired retrieval outcomes must not be empty".into());
    }
    if seed == 0 {
        return Err("cluster bootstrap seed must be non-zero".into());
    }

    let mut observation_ids = BTreeSet::new();
    let mut query_fingerprints = BTreeSet::new();
    let mut groups = BTreeMap::<String, GroupAggregate>::new();

    let mut correctness = Vec::with_capacity(outcomes.len());
    let mut lifecycle = Vec::with_capacity(outcomes.len());
    let mut harm = Vec::with_capacity(outcomes.len());
    let mut net = Vec::with_capacity(outcomes.len());
    let mut cost = 0u128;
    let mut calls = 0u128;
    let mut tokens = 0u128;
    let mut latency = 0u128;
    let mut positive = 0usize;
    let mut neutral = 0usize;
    let mut negative = 0usize;

    for outcome in outcomes {
        outcome.validate()?;
        if !observation_ids.insert(outcome.observation_id.as_str()) {
            return Err(format!("duplicate paired observation ID {:?}", outcome.observation_id));
        }
        if !query_fingerprints.insert(outcome.query_fingerprint_sha256.as_str()) {
            return Err(format!(
                "duplicate query fingerprint {:?}; each query must have one paired outcome",
                outcome.query_fingerprint_sha256
            ));
        }

        let correctness_delta = outcome.correctness_gain_micro()?;
        let lifecycle_delta = outcome.lifecycle_risk_reduction_micro()?;
        let harm_delta = outcome.retrieval_harm_delta_micro()?;
        let net_delta = outcome.net_utility_delta_micro();

        correctness.push(i128::from(correctness_delta));
        lifecycle.push(i128::from(lifecycle_delta));
        harm.push(i128::from(harm_delta));
        net.push(net_delta);
        cost += u128::from(outcome.retrieval_cost_micro);
        calls += u128::from(outcome.retrieval_calls);
        tokens += u128::from(outcome.retrieval_tokens);
        latency += u128::from(outcome.retrieval_latency_ms);

        match net_delta.cmp(&0) {
            std::cmp::Ordering::Greater => positive += 1,
            std::cmp::Ordering::Equal => neutral += 1,
            std::cmp::Ordering::Less => negative += 1,
        }

        groups
            .entry(outcome.calibration_group_id.clone())
            .and_modify(|group| {
                group.net_utility_sum += net_delta;
                group.observation_count += 1;
            })
            .or_insert(GroupAggregate {
                net_utility_sum: net_delta,
                observation_count: 1,
            });
    }

    if groups.len() < MIN_BOOTSTRAP_GROUPS {
        return Err(format!(
            "at least {MIN_BOOTSTRAP_GROUPS} independent groups are required for cluster bootstrap; received {}",
            groups.len()
        ));
    }

    let group_means: Vec<i128> = groups
        .values()
        .map(|group| group.net_utility_sum / group.observation_count as i128)
        .collect();
    let interval = cluster_bootstrap_mean_interval(&group_means, seed)?;

    let mean_cost = cost / outcomes.len() as u128;
    let mean_calls = calls / outcomes.len() as u128;
    let mean_tokens = tokens / outcomes.len() as u128;
    let mean_latency = latency / outcomes.len() as u128;

    Ok(PairedRetrievalUtilitySummary {
        observation_count: outcomes.len(),
        independent_group_count: groups.len(),
        positive_net_utility_count: positive,
        neutral_net_utility_count: neutral,
        negative_net_utility_count: negative,
        mean_correctness_gain_micro: mean_i64(&correctness)?,
        mean_lifecycle_risk_reduction_micro: mean_i64(&lifecycle)?,
        mean_retrieval_harm_delta_micro: mean_i64(&harm)?,
        mean_net_utility_delta_micro: mean_i64(&net)?,
        mean_group_net_utility_delta_micro: mean_i64(&group_means)?,
        mean_retrieval_cost_micro: u64::try_from(mean_cost)
            .map_err(|_| "mean retrieval cost exceeds u64".to_string())?,
        mean_retrieval_calls: u64::try_from(mean_calls)
            .map_err(|_| "mean retrieval calls exceeds u64".to_string())?,
        mean_retrieval_tokens: u64::try_from(mean_tokens)
            .map_err(|_| "mean retrieval tokens exceeds u64".to_string())?,
        mean_retrieval_latency_ms: u64::try_from(mean_latency)
            .map_err(|_| "mean retrieval latency exceeds u64".to_string())?,
        net_utility_cluster_bootstrap_95: interval,
    })
}

fn cluster_bootstrap_mean_interval(
    group_means: &[i128],
    seed: u64,
) -> Result<ClusterBootstrapInterval, String> {
    if group_means.len() < MIN_BOOTSTRAP_GROUPS {
        return Err("too few independent groups for cluster bootstrap".into());
    }

    let mut rng = XorShift64 { state: seed };
    let mut estimates = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut sum = 0i128;
        for _ in 0..group_means.len() {
            let index = (rng.next() % group_means.len() as u64) as usize;
            sum += group_means[index];
        }
        estimates.push(sum / group_means.len() as i128);
    }
    estimates.sort_unstable();

    let lower_index = (BOOTSTRAP_REPLICATES - 1) * 25 / 1000;
    let upper_index = (BOOTSTRAP_REPLICATES - 1) * 975 / 1000;
    Ok(ClusterBootstrapInterval {
        lower_95_micro: i64_from_i128(estimates[lower_index])?,
        upper_95_micro: i64_from_i128(estimates[upper_index])?,
        independent_group_count: group_means.len(),
        replicate_count: BOOTSTRAP_REPLICATES,
        seed,
    })
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }
}

fn mean_i64(values: &[i128]) -> Result<i64, String> {
    if values.is_empty() {
        return Err("cannot calculate mean of empty values".into());
    }
    let sum: i128 = values.iter().sum();
    i64_from_i128(sum / values.len() as i128)
}

fn i64_from_i128(value: i128) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| "paired utility statistic exceeds i64 range".into())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
    {
        return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        format!("{:x}", Sha256::digest(value.as_bytes()))
    }

    fn outcome(index: usize, group: usize, delta: i64) -> PairedRetrievalOutcome {
        PairedRetrievalOutcome {
            observation_id: format!("outcome-{index}"),
            calibration_group_id: format!("group-{group}"),
            query_fingerprint_sha256: digest(&format!("query-{index}")),
            no_retrieval_correctness_micro: 500,
            retrieval_correctness_micro: 500 + delta,
            no_retrieval_lifecycle_loss_micro: 100,
            retrieval_lifecycle_loss_micro: 100,
            no_retrieval_harm_micro: 10,
            retrieval_harm_micro: 10,
            retrieval_cost_micro: 20,
            retrieval_calls: 1,
            retrieval_tokens: 100,
            retrieval_latency_ms: 50,
        }
    }

    #[test]
    fn paired_outcome_derives_disjoint_utility_deltas_and_calibration_row() {
        let result = outcome(0, 0, 100);
        assert_eq!(result.correctness_gain_micro().unwrap(), 100);
        assert_eq!(result.lifecycle_risk_reduction_micro().unwrap(), 0);
        assert_eq!(result.retrieval_harm_delta_micro().unwrap(), 0);
        assert_eq!(result.net_utility_delta_micro(), 80);

        let observation = result.to_calibration_observation(
            &RetrievalUtilityPointPrediction {
                correctness_gain_micro: 90,
                lifecycle_risk_reduction_micro: 0,
                retrieval_harm_micro: 0,
            },
        ).unwrap();
        assert_eq!(observation.observed_correctness_gain_micro, 100);
        assert_eq!(observation.predicted_correctness_gain_micro, 90);
    }

    #[test]
    fn summarizes_paired_results_with_group_cluster_bootstrap() {
        let outcomes: Vec<_> = (0..20)
            .map(|index| outcome(index, index, 100))
            .collect();
        let summary = summarize_paired_retrieval_outcomes(&outcomes, 42).unwrap();
        assert_eq!(summary.observation_count, 20);
        assert_eq!(summary.independent_group_count, 20);
        assert_eq!(summary.positive_net_utility_count, 20);
        assert_eq!(summary.neutral_net_utility_count, 0);
        assert_eq!(summary.negative_net_utility_count, 0);
        assert_eq!(summary.mean_correctness_gain_micro, 100);
        assert_eq!(summary.mean_net_utility_delta_micro, 80);
        assert_eq!(summary.mean_group_net_utility_delta_micro, 80);
        assert_eq!(summary.mean_retrieval_cost_micro, 20);
        assert_eq!(summary.net_utility_cluster_bootstrap_95.lower_95_micro, 80);
        assert_eq!(summary.net_utility_cluster_bootstrap_95.upper_95_micro, 80);
        assert_eq!(summary.net_utility_cluster_bootstrap_95.replicate_count, 2000);
    }

    #[test]
    fn cluster_bootstrap_weights_independent_groups_not_query_count() {
        let mut outcomes: Vec<_> = (0..100)
            .map(|index| outcome(index, 0, 100))
            .collect();
        outcomes.extend((100..120).map(|index| outcome(index, index - 100, -100)));
        let summary = summarize_paired_retrieval_outcomes(&outcomes, 7).unwrap();
        assert_eq!(summary.observation_count, 120);
        assert_eq!(summary.independent_group_count, 21);
        // Group 0 has 100 queries with mean net utility 80; the other 20
        // independent groups each have net utility -120. Publish both the
        // query-weighted mean and equal-group mean so the estimand is explicit.
        assert_eq!(summary.mean_net_utility_delta_micro, 46);
        assert_eq!(summary.mean_group_net_utility_delta_micro, -110);
        assert_eq!(summary.net_utility_cluster_bootstrap_95.lower_95_micro, -120);
    }

    #[test]
    fn rejects_missing_pair_duplicate_query_or_too_few_groups() {
        let mut outcomes: Vec<_> = (0..20)
            .map(|index| outcome(index, index, 0))
            .collect();
        assert!(summarize_paired_retrieval_outcomes(&outcomes, 1).is_ok());

        outcomes[1].query_fingerprint_sha256 = outcomes[0].query_fingerprint_sha256.clone();
        assert!(summarize_paired_retrieval_outcomes(&outcomes, 1)
            .unwrap_err()
            .contains("duplicate query fingerprint"));

        outcomes[1].query_fingerprint_sha256 = digest("new query");
        outcomes.truncate(19);
        assert!(summarize_paired_retrieval_outcomes(&outcomes, 1)
            .unwrap_err()
            .contains("at least 20 independent groups"));
    }

    #[test]
    fn rejects_zero_bootstrap_seed_and_is_reproducible() {
        let outcomes: Vec<_> = (0..20)
            .map(|index| outcome(index, index, index as i64 - 10))
            .collect();
        assert!(summarize_paired_retrieval_outcomes(&outcomes, 0).is_err());
        let first = summarize_paired_retrieval_outcomes(&outcomes, 99).unwrap();
        let second = summarize_paired_retrieval_outcomes(&outcomes, 99).unwrap();
        assert_eq!(
            first.net_utility_cluster_bootstrap_95,
            second.net_utility_cluster_bootstrap_95
        );
    }
}
