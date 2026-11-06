//! Paired, cluster-aware uncertainty summaries for RMI-Bench.
//!
//! This module reports dimension- and direction-specific estimates. It never
//! computes a cross-dimension reciprocity score and never treats turns as
//! independent when they share a declared sampling cluster.

use std::collections::{BTreeMap, BTreeSet};

use crate::Dimension;

/// Direction of a controlled causal contrast.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InfluenceDirection {
    /// Partner intervention changes the agent's next action.
    PartnerToAgent,
    /// Agent intervention changes the partner's next action.
    AgentToPartner,
}

/// Reason a paired trial was attempted but has no complete effect measurement.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MissingOutcomeReason {
    /// The policy explicitly abstained and no comparable next action exists.
    Abstained,
    /// The policy did not return before the registered deadline.
    Timeout,
    /// The produced action or artifact failed structural validation.
    InvalidOutput,
    /// Required source or result bytes were unavailable.
    MissingArtifact,
    /// The adapter cannot expose the required metric for this modality.
    AdapterUnavailable,
    /// The participant withdrew this observation under the approved protocol.
    ParticipantWithdrawal,
    /// A different preregistered missingness reason.
    Other,
}

/// Measured paired effects for one attempted trial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairedEffectValue {
    /// Distance between targeted and control next actions.
    pub targeted_delta: u32,
    /// Distance between sham and control next actions.
    pub sham_delta: u32,
    /// Distance between unrelated-feature and control next actions.
    pub unrelated_delta: u32,
    /// Optional distance between yoked-replay and control next actions.
    pub yoked_delta: Option<u32>,
}

/// Explicit status of one attempted paired trial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairedEffectOutcome {
    /// All required paired effects were observed.
    Measured(PairedEffectValue),
    /// The attempt remains in the denominator but is not imputed.
    Missing(MissingOutcomeReason),
}

/// One paired trial with an explicit cluster identifier.
///
/// cluster_id must identify the independent sampling unit declared by the
/// study (for example, musician or musician-session, depending on assignment).
/// It must not be a turn, condition row, or audio frame ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedEffectObservation {
    /// Independent sampling cluster.
    pub cluster_id: String,
    /// Unique attempted trial within a dimension and direction.
    pub trial_id: String,
    /// Musical feature dimension.
    pub dimension: Dimension,
    /// Causal direction represented by this observation.
    pub direction: InfluenceDirection,
    /// Number of discrete values in this task's cyclic feature domain.
    pub domain_size: u32,
    /// Paired outcome, including explicit missingness.
    pub outcome: PairedEffectOutcome,
}

/// Configuration for the deterministic cluster bootstrap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BootstrapConfig {
    /// Number of cluster-resampling replicates; must be at least 100.
    pub resamples: usize,
    /// Seed for replayable bootstrap index draws.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            resamples: 2_000,
            seed: 0x524D_492D_4253_5431,
        }
    }
}

/// A mean and cluster-bootstrap percentile interval for one metric.
///
/// A mean is unavailable when no complete observations exist. A 95% interval
/// is unavailable with fewer than two measured clusters; the module will not
/// manufacture precision by resampling turns from one cluster.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// Equal-cluster-weighted mean, normalized to the task's maximum distance.
    pub mean: Option<f64>,
    /// Lower endpoint of the 95% percentile cluster-bootstrap interval.
    pub ci95_lower: Option<f64>,
    /// Upper endpoint of the 95% percentile cluster-bootstrap interval.
    pub ci95_upper: Option<f64>,
    /// Number of measured independent clusters contributing to this metric.
    pub clusters: usize,
    /// Number of bootstrap replicates used (zero when no interval is estimable).
    pub bootstrap_resamples: usize,
}

/// Counts of each explicit missing-outcome category.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MissingOutcomeCounts {
    /// Explicit policy abstention.
    pub abstained: usize,
    /// Deadline timeout.
    pub timeout: usize,
    /// Invalid action or artifact.
    pub invalid_output: usize,
    /// Missing committed or source artifact.
    pub missing_artifact: usize,
    /// Adapter cannot expose the required measurement.
    pub adapter_unavailable: usize,
    /// Participant withdrawal.
    pub participant_withdrawal: usize,
    /// Other preregistered missingness.
    pub other: usize,
}

impl MissingOutcomeCounts {
    fn record(&mut self, reason: MissingOutcomeReason) {
        match reason {
            MissingOutcomeReason::Abstained => self.abstained += 1,
            MissingOutcomeReason::Timeout => self.timeout += 1,
            MissingOutcomeReason::InvalidOutput => self.invalid_output += 1,
            MissingOutcomeReason::MissingArtifact => self.missing_artifact += 1,
            MissingOutcomeReason::AdapterUnavailable => self.adapter_unavailable += 1,
            MissingOutcomeReason::ParticipantWithdrawal => self.participant_withdrawal += 1,
            MissingOutcomeReason::Other => self.other += 1,
        }
    }
}

/// Separate summary for one musical dimension and one causal direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimensionEffectSummary {
    /// Musical feature dimension.
    pub dimension: Dimension,
    /// Causal direction.
    pub direction: InfluenceDirection,
    /// All attempted trials, including missing outcomes.
    pub attempted_trials: usize,
    /// Trials with all required paired effects observed.
    pub measured_trials: usize,
    /// Trials retained as missing rather than imputed.
    pub missing_trials: usize,
    /// Breakdown of explicit missingness reasons.
    pub missing_by_reason: MissingOutcomeCounts,
    /// Distinct clusters represented in the attempted denominator.
    pub attempted_clusters: usize,
    /// Distinct clusters with at least one complete observation.
    pub measured_clusters: usize,
    /// Targeted action-change magnitude.
    pub targeted_delta: Estimate,
    /// Sham action-change magnitude.
    pub sham_delta: Estimate,
    /// Unrelated-feature action-change magnitude.
    pub unrelated_delta: Estimate,
    /// Yoked-replay action-change magnitude, when present.
    pub yoked_delta: Estimate,
    /// Paired targeted-minus-sham effect; positive means treatment exceeds sham.
    pub targeted_minus_sham: Estimate,
    /// Paired targeted-minus-unrelated effect; positive favors target specificity.
    pub targeted_minus_unrelated: Estimate,
    /// Paired targeted-minus-yoked effect, on trials with a yoke.
    pub targeted_minus_yoked: Estimate,
    /// Fraction of measured trials whose targeted effect was nonzero.
    pub input_sensitive_rate: Estimate,
    /// Fraction of measured trials with zero sham and unrelated effects.
    pub target_specific_rate: Estimate,
}

/// Fail-closed validation errors for the paired statistics input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StatisticsError {
    /// Too few bootstrap replicates for a percentile interval.
    InvalidBootstrapConfig,
    /// A cluster or trial identity is empty.
    EmptyIdentity,
    /// A trial identity is duplicated within a dimension/direction.
    DuplicateTrial,
    /// A cyclic feature domain must contain at least two values.
    InvalidDomain,
    /// A distance exceeds the maximum shortest distance in its domain.
    InvalidDistance,
}

impl std::fmt::Display for StatisticsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StatisticsError {}

/// Summarize paired effects separately by dimension and causal direction.
///
/// Each cluster is reduced to its own within-cluster mean before bootstrap
/// resampling. This avoids giving a musician/session with many turns more
/// statistical weight merely because more turns were recorded. Missing trials
/// are counted but never imputed or included in a metric estimate.
pub fn summarize_paired_effects(
    observations: &[PairedEffectObservation],
    config: BootstrapConfig,
) -> Result<Vec<DimensionEffectSummary>, StatisticsError> {
    if config.resamples < 100 {
        return Err(StatisticsError::InvalidBootstrapConfig);
    }

    let mut seen_trials = BTreeSet::new();
    let mut groups: BTreeMap<
        (Dimension, InfluenceDirection),
        Vec<&PairedEffectObservation>,
    > = BTreeMap::new();

    for observation in observations {
        if observation.cluster_id.trim().is_empty() || observation.trial_id.trim().is_empty() {
            return Err(StatisticsError::EmptyIdentity);
        }
        if observation.domain_size < 2 {
            return Err(StatisticsError::InvalidDomain);
        }
        if !seen_trials.insert((
            observation.dimension,
            observation.direction,
            observation.trial_id.as_str(),
        )) {
            return Err(StatisticsError::DuplicateTrial);
        }
        if let PairedEffectOutcome::Measured(value) = observation.outcome {
            let max_distance = observation.domain_size / 2;
            let distances = [
                Some(value.targeted_delta),
                Some(value.sham_delta),
                Some(value.unrelated_delta),
                value.yoked_delta,
            ];
            if distances
                .into_iter()
                .flatten()
                .any(|distance| distance > max_distance)
            {
                return Err(StatisticsError::InvalidDistance);
            }
        }
        groups
            .entry((observation.dimension, observation.direction))
            .or_default()
            .push(observation);
    }

    let mut summaries = Vec::with_capacity(groups.len());
    for ((dimension, direction), group) in groups {
        summaries.push(summarize_group(dimension, direction, &group, config));
    }
    Ok(summaries)
}

fn summarize_group(
    dimension: Dimension,
    direction: InfluenceDirection,
    group: &[&PairedEffectObservation],
    config: BootstrapConfig,
) -> DimensionEffectSummary {
    let mut attempted_clusters = BTreeSet::new();
    let mut metrics: [BTreeMap<String, Vec<f64>>; 9] =
        std::array::from_fn(|_| BTreeMap::new());
    let mut measured_trials = 0usize;
    let mut missing_trials = 0usize;
    let mut missing_by_reason = MissingOutcomeCounts::default();

    for observation in group {
        attempted_clusters.insert(observation.cluster_id.as_str());
        let PairedEffectOutcome::Measured(value) = observation.outcome else {
            if let PairedEffectOutcome::Missing(reason) = observation.outcome {
                missing_by_reason.record(reason);
            }
            missing_trials += 1;
            continue;
        };
        measured_trials += 1;
        let max_distance = f64::from(observation.domain_size / 2);
        let targeted = f64::from(value.targeted_delta) / max_distance;
        let sham = f64::from(value.sham_delta) / max_distance;
        let unrelated = f64::from(value.unrelated_delta) / max_distance;

        push_metric(&mut metrics[0], &observation.cluster_id, targeted);
        push_metric(&mut metrics[1], &observation.cluster_id, sham);
        push_metric(&mut metrics[2], &observation.cluster_id, unrelated);
        push_metric(&mut metrics[4], &observation.cluster_id, targeted - sham);
        push_metric(
            &mut metrics[5],
            &observation.cluster_id,
            targeted - unrelated,
        );
        push_metric(
            &mut metrics[7],
            &observation.cluster_id,
            if value.targeted_delta > 0 { 1.0 } else { 0.0 },
        );
        push_metric(
            &mut metrics[8],
            &observation.cluster_id,
            if value.sham_delta == 0 && value.unrelated_delta == 0 {
                1.0
            } else {
                0.0
            },
        );

        if let Some(yoked_delta) = value.yoked_delta {
            let yoked = f64::from(yoked_delta) / max_distance;
            push_metric(&mut metrics[3], &observation.cluster_id, yoked);
            push_metric(
                &mut metrics[6],
                &observation.cluster_id,
                targeted - yoked,
            );
        }
    }

    let group_seed = config.seed
        ^ ((dimension as u64) << 32)
        ^ ((direction as u64) << 48);
    let estimate = |index: usize| {
        estimate_cluster_means(&metrics[index], config.resamples, group_seed ^ index as u64)
    };
    let measured_clusters = metrics[0].len();

    DimensionEffectSummary {
        dimension,
        direction,
        attempted_trials: group.len(),
        measured_trials,
        missing_trials,
        missing_by_reason,
        attempted_clusters: attempted_clusters.len(),
        measured_clusters,
        targeted_delta: estimate(0),
        sham_delta: estimate(1),
        unrelated_delta: estimate(2),
        yoked_delta: estimate(3),
        targeted_minus_sham: estimate(4),
        targeted_minus_unrelated: estimate(5),
        targeted_minus_yoked: estimate(6),
        input_sensitive_rate: estimate(7),
        target_specific_rate: estimate(8),
    }
}

fn push_metric(
    clusters: &mut BTreeMap<String, Vec<f64>>,
    cluster_id: &str,
    value: f64,
) {
    clusters
        .entry(cluster_id.to_owned())
        .or_default()
        .push(value);
}

fn estimate_cluster_means(
    cluster_values: &BTreeMap<String, Vec<f64>>,
    resamples: usize,
    seed: u64,
) -> Estimate {
    if cluster_values.is_empty() {
        return Estimate {
            mean: None,
            ci95_lower: None,
            ci95_upper: None,
            clusters: 0,
            bootstrap_resamples: 0,
        };
    }

    let cluster_means: Vec<f64> = cluster_values
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect();
    let mean = cluster_means.iter().sum::<f64>() / cluster_means.len() as f64;

    if cluster_means.len() < 2 {
        return Estimate {
            mean: Some(mean),
            ci95_lower: None,
            ci95_upper: None,
            clusters: 1,
            bootstrap_resamples: 0,
        };
    }

    let mut rng = SplitMix64::new(seed);
    let mut bootstrap_means = Vec::with_capacity(resamples);
    let count = cluster_means.len();
    for _ in 0..resamples {
        let mut sum = 0.0;
        for _ in 0..count {
            let selected = rng.index_below(count);
            sum += cluster_means[selected];
        }
        bootstrap_means.push(sum / count as f64);
    }
    bootstrap_means.sort_by(f64::total_cmp);

    Estimate {
        mean: Some(mean),
        ci95_lower: Some(percentile(&bootstrap_means, 0.025)),
        ci95_upper: Some(percentile(&bootstrap_means, 0.975)),
        clusters: count,
        bootstrap_resamples: resamples,
    }
}

fn percentile(sorted_values: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted_values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let fraction = position - lower as f64;
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    }
}

/// Small deterministic generator for reproducible bootstrap index draws.
/// This is not a randomization source for a human experiment.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index_below(&mut self, upper: usize) -> usize {
        let bound = upper as u64;
        let acceptance_limit = u64::MAX - (u64::MAX % bound);
        loop {
            let value = self.next_u64();
            if value < acceptance_limit {
                return (value % bound) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(
        cluster: &str,
        trial: &str,
        targeted: u32,
        sham: u32,
        unrelated: u32,
        yoked: Option<u32>,
    ) -> PairedEffectObservation {
        PairedEffectObservation {
            cluster_id: cluster.to_owned(),
            trial_id: trial.to_owned(),
            dimension: Dimension::Motif,
            direction: InfluenceDirection::PartnerToAgent,
            domain_size: 8,
            outcome: PairedEffectOutcome::Measured(PairedEffectValue {
                targeted_delta: targeted,
                sham_delta: sham,
                unrelated_delta: unrelated,
                yoked_delta: yoked,
            }),
        }
    }

    fn missing(cluster: &str, trial: &str) -> PairedEffectObservation {
        PairedEffectObservation {
            cluster_id: cluster.to_owned(),
            trial_id: trial.to_owned(),
            dimension: Dimension::Motif,
            direction: InfluenceDirection::PartnerToAgent,
            domain_size: 8,
            outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::Timeout),
        }
    }

    fn config() -> BootstrapConfig {
        BootstrapConfig {
            resamples: 2_000,
            seed: 1234,
        }
    }

    fn single_summary(
        observations: &[PairedEffectObservation],
    ) -> DimensionEffectSummary {
        summarize_paired_effects(observations, config())
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    }

    #[test]
    fn paired_estimate_uses_target_minus_sham_not_unpaired_arm_means() {
        let observations = [
            measured("musician-a", "a-1", 3, 1, 0, Some(2)),
            measured("musician-b", "b-1", 2, 0, 1, Some(1)),
        ];
        let summary = single_summary(&observations);
        assert_eq!(summary.targeted_minus_sham.mean, Some(0.5));
        assert_eq!(summary.targeted_minus_unrelated.mean, Some(0.5));
        assert_eq!(summary.targeted_minus_yoked.mean, Some(0.25));
    }

    #[test]
    fn repeated_turns_are_cluster_averaged_before_resampling() {
        let observations = [
            measured("musician-a", "a-1", 2, 0, 0, None),
            measured("musician-a", "a-2", 2, 0, 0, None),
            measured("musician-a", "a-3", 2, 0, 0, None),
            measured("musician-b", "b-1", 0, 0, 0, None),
        ];
        let summary = single_summary(&observations);
        assert_eq!(summary.targeted_delta.mean, Some(0.5));
        assert_eq!(summary.targeted_delta.clusters, 2);
        assert_eq!(summary.targeted_delta.bootstrap_resamples, 2_000);
    }

    #[test]
    fn deterministic_bootstrap_replays_exactly_from_frozen_seed() {
        let observations = [
            measured("a", "a-1", 3, 0, 1, None),
            measured("b", "b-1", 2, 0, 0, None),
            measured("c", "c-1", 1, 0, 0, None),
            measured("d", "d-1", 0, 0, 0, None),
        ];
        let first = single_summary(&observations).targeted_delta;
        let second = single_summary(&observations).targeted_delta;
        assert_eq!(first, second);
        assert!(first.ci95_lower.unwrap() <= first.mean.unwrap());
        assert!(first.mean.unwrap() <= first.ci95_upper.unwrap());
    }

    #[test]
    fn one_cluster_reports_mean_but_does_not_fabricate_a_confidence_interval() {
        let summary = single_summary(&[measured("one-musician", "one-trial", 3, 0, 0, None)]);
        assert_eq!(summary.targeted_delta.mean, Some(0.75));
        assert_eq!(summary.targeted_delta.ci95_lower, None);
        assert_eq!(summary.targeted_delta.ci95_upper, None);
        assert_eq!(summary.targeted_delta.bootstrap_resamples, 0);
    }

    #[test]
    fn missing_trials_remain_in_attempted_denominator_and_are_not_imputed() {
        let observations = [
            measured("a", "a-1", 2, 0, 0, None),
            missing("b", "b-1"),
            missing("c", "c-1"),
        ];
        let summary = single_summary(&observations);
        assert_eq!(summary.attempted_trials, 3);
        assert_eq!(summary.measured_trials, 1);
        assert_eq!(summary.missing_trials, 2);
        assert_eq!(summary.attempted_clusters, 3);
        assert_eq!(summary.measured_clusters, 1);
        assert_eq!(summary.targeted_delta.ci95_lower, None);
    }

    #[test]
    fn yoke_metric_uses_only_trials_with_a_yoked_control() {
        let observations = [
            measured("a", "a-1", 2, 0, 0, Some(0)),
            measured("b", "b-1", 2, 0, 0, None),
            measured("c", "c-1", 2, 0, 0, Some(2)),
        ];
        let summary = single_summary(&observations);
        assert_eq!(summary.targeted_minus_yoked.mean, Some(0.25));
        assert_eq!(summary.targeted_minus_yoked.clusters, 2);
        assert_eq!(summary.yoked_delta.clusters, 2);
    }

    #[test]
    fn dimensions_and_directions_are_never_aggregated_together() {
        let partner_to_agent = measured("a", "a-1", 2, 0, 0, None);
        let mut agent_to_partner = measured("b", "b-1", 1, 0, 0, None);
        agent_to_partner.direction = InfluenceDirection::AgentToPartner;
        let mut rhythm = measured("c", "c-1", 1, 0, 0, None);
        rhythm.dimension = Dimension::Rhythm;
        let summaries =
            summarize_paired_effects(&[partner_to_agent, agent_to_partner, rhythm], config())
                .unwrap();
        assert_eq!(summaries.len(), 3);
        assert!(summaries.iter().any(|summary| {
            summary.dimension == Dimension::Motif
                && summary.direction == InfluenceDirection::PartnerToAgent
        }));
        assert!(summaries.iter().any(|summary| {
            summary.dimension == Dimension::Motif
                && summary.direction == InfluenceDirection::AgentToPartner
        }));
        assert!(summaries.iter().any(|summary| {
            summary.dimension == Dimension::Rhythm
                && summary.direction == InfluenceDirection::PartnerToAgent
        }));
    }

    #[test]
    fn bootstrap_index_sampler_always_returns_an_in_range_index() {
        let mut rng = SplitMix64::new(777);
        for upper in 2..100 {
            for _ in 0..100 {
                assert!(rng.index_below(upper) < upper);
            }
        }
    }

    #[test]
    fn invalid_bootstrap_replicate_count_is_rejected() {
        let result = summarize_paired_effects(
            &[measured("a", "a-1", 2, 0, 0, None)],
            BootstrapConfig {
                resamples: 99,
                seed: 1,
            },
        );
        assert_eq!(result, Err(StatisticsError::InvalidBootstrapConfig));
    }

    #[test]
    fn duplicate_trial_ids_are_rejected_within_dimension_and_direction() {
        let observations = [
            measured("a", "duplicate", 2, 0, 0, None),
            measured("b", "duplicate", 1, 0, 0, None),
        ];
        assert_eq!(
            summarize_paired_effects(&observations, config()),
            Err(StatisticsError::DuplicateTrial)
        );
    }

    #[test]
    fn impossible_distance_and_domain_are_rejected() {
        assert_eq!(
            summarize_paired_effects(
                &[PairedEffectObservation {
                    domain_size: 1,
                    ..measured("a", "a-1", 0, 0, 0, None)
                }],
                config()
            ),
            Err(StatisticsError::InvalidDomain)
        );
        assert_eq!(
            summarize_paired_effects(
                &[measured("a", "a-1", 5, 0, 0, None)],
                config()
            ),
            Err(StatisticsError::InvalidDistance)
        );
    }

    #[test]
    fn missing_reason_counts_remain_disaggregated() {
        let observations = [
            PairedEffectObservation {
                outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::Abstained),
                ..missing("a", "a-1")
            },
            PairedEffectObservation {
                outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::Timeout),
                ..missing("b", "b-1")
            },
            PairedEffectObservation {
                outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::InvalidOutput),
                ..missing("c", "c-1")
            },
            PairedEffectObservation {
                outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::MissingArtifact),
                ..missing("d", "d-1")
            },
            PairedEffectObservation {
                outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::AdapterUnavailable),
                ..missing("e", "e-1")
            },
            PairedEffectObservation {
                outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::ParticipantWithdrawal),
                ..missing("f", "f-1")
            },
            PairedEffectObservation {
                outcome: PairedEffectOutcome::Missing(MissingOutcomeReason::Other),
                ..missing("g", "g-1")
            },
        ];
        let summary = single_summary(&observations);
        assert_eq!(summary.missing_trials, 7);
        assert_eq!(
            summary.missing_by_reason,
            MissingOutcomeCounts {
                abstained: 1,
                timeout: 1,
                invalid_output: 1,
                missing_artifact: 1,
                adapter_unavailable: 1,
                participant_withdrawal: 1,
                other: 1,
            }
        );
    }

    #[test]
    fn all_missing_attempts_produce_no_mean_or_interval() {
        let summary = single_summary(&[missing("a", "a-1"), missing("b", "b-1")]);
        assert_eq!(summary.attempted_trials, 2);
        assert_eq!(summary.measured_trials, 0);
        assert_eq!(summary.targeted_delta.mean, None);
        assert_eq!(summary.targeted_delta.ci95_lower, None);
        assert_eq!(summary.targeted_delta.clusters, 0);
    }
}
