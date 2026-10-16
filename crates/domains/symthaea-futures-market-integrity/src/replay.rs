//! Prequential replay evaluation for binary-market forecasts.
//!
//! The evaluator chooses the latest observation at or before a frozen target cutoff. It never
//! selects a later observation, and it excludes missing or stale forecasts rather than filling
//! them with zero or silently substituting a later quote. A forecast method identifier is only a
//! label; this module does not authenticate source data, event timestamps, or outcome provenance.

use std::collections::BTreeSet;

/// Errors found while constructing or replaying forecast records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// A required identifier was blank.
    EmptyIdentifier(&'static str),
    /// A forecast probability was non-finite or outside [0, 1].
    InvalidProbability,
    /// The forecast cutoff was not strictly earlier than the recorded resolution time.
    InvalidTargetTimes,
    /// The maximum allowed forecast age was zero.
    ZeroMaxStaleness,
    /// An event/method timestamp key occurred more than once.
    DuplicateObservation {
        /// Event whose observation key collided.
        event_id: String,
        /// Forecast method whose observation key collided.
        forecaster_id: String,
        /// Duplicated observation time in Unix seconds.
        observed_at_unix_seconds: u64,
    },
    /// The same event/horizon target was supplied more than once.
    DuplicateTarget {
        /// Event whose target was duplicated.
        event_id: String,
        /// Horizon whose target was duplicated.
        horizon_id: String,
    },
    /// No paired comparisons were supplied.
    EmptyComparisonSet,
    /// The same event/horizon pair appeared more than once in a paired comparison corpus.
    DuplicateComparison {
        /// Event whose comparison key collided.
        event_id: String,
        /// Horizon whose comparison key collided.
        horizon_id: String,
    },
    /// Bootstrap replication count was below the documented minimum.
    InvalidBootstrapReplicates,
    /// A non-finite numeric input was supplied to the comparison evaluator.
    InvalidComparisonValue,
}

/// A forecast available at one timestamp for one event and method.
///
/// A method may represent the market price, a base-rate model, a source-aware pool, or Symthaea.
/// Method names and timestamps are data supplied by the caller, not authenticated by this type.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastObservation {
    event_id: String,
    forecaster_id: String,
    observed_at_unix_seconds: u64,
    probability: f64,
}

impl ForecastObservation {
    /// Create a forecast observation after validating identifiers and its probability.
    pub fn new(
        event_id: impl Into<String>,
        forecaster_id: impl Into<String>,
        observed_at_unix_seconds: u64,
        probability: f64,
    ) -> Result<Self, ReplayError> {
        let event_id = required_identifier(event_id.into(), "event_id")?;
        let forecaster_id = required_identifier(forecaster_id.into(), "forecaster_id")?;
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ReplayError::InvalidProbability);
        }
        Ok(Self {
            event_id,
            forecaster_id,
            observed_at_unix_seconds,
            probability,
        })
    }

    /// Return the event identifier.
    pub fn event_id(&self) -> &str {
        &self.event_id
    }

    /// Return the forecast-method identifier.
    pub fn forecaster_id(&self) -> &str {
        &self.forecaster_id
    }

    /// Return the Unix-second observation time.
    pub fn observed_at_unix_seconds(&self) -> u64 {
        self.observed_at_unix_seconds
    }

    /// Return the validated event probability.
    pub fn probability(&self) -> f64 {
        self.probability
    }
}

/// A resolved event scored at a preregistered forecast cutoff.
///
/// Supply the cutoff timestamp from market/event lifecycle evidence. It is not inferred from
/// resolution time: market close, public announcement, and official resolution may differ.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastTarget {
    event_id: String,
    horizon_id: String,
    cutoff_at_unix_seconds: u64,
    resolved_at_unix_seconds: u64,
    outcome: bool,
    max_staleness_seconds: u64,
}

impl ForecastTarget {
    /// Create a target with a frozen pre-resolution cutoff and positive staleness limit.
    pub fn new(
        event_id: impl Into<String>,
        horizon_id: impl Into<String>,
        cutoff_at_unix_seconds: u64,
        resolved_at_unix_seconds: u64,
        outcome: bool,
        max_staleness_seconds: u64,
    ) -> Result<Self, ReplayError> {
        let event_id = required_identifier(event_id.into(), "event_id")?;
        let horizon_id = required_identifier(horizon_id.into(), "horizon_id")?;
        if cutoff_at_unix_seconds >= resolved_at_unix_seconds {
            return Err(ReplayError::InvalidTargetTimes);
        }
        if max_staleness_seconds == 0 {
            return Err(ReplayError::ZeroMaxStaleness);
        }
        Ok(Self {
            event_id,
            horizon_id,
            cutoff_at_unix_seconds,
            resolved_at_unix_seconds,
            outcome,
            max_staleness_seconds,
        })
    }

    /// Return the event identifier.
    pub fn event_id(&self) -> &str {
        &self.event_id
    }

    /// Return the frozen horizon identifier.
    pub fn horizon_id(&self) -> &str {
        &self.horizon_id
    }

    /// Return the forecast cutoff time in Unix seconds.
    pub fn cutoff_at_unix_seconds(&self) -> u64 {
        self.cutoff_at_unix_seconds
    }

    /// Return the event resolution time in Unix seconds.
    pub fn resolved_at_unix_seconds(&self) -> u64 {
        self.resolved_at_unix_seconds
    }

    /// Return the observed binary outcome.
    pub fn outcome(&self) -> bool {
        self.outcome
    }

    /// Return the maximum permitted forecast age in seconds.
    pub fn max_staleness_seconds(&self) -> u64 {
        self.max_staleness_seconds
    }
}

/// Why a method/horizon pair was omitted from scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayExclusionReason {
    /// No observation for this event/method existed at or before the cutoff.
    NoForecastAtOrBeforeCutoff,
    /// A pre-cutoff observation existed, but the latest one exceeded the staleness limit.
    ForecastTooStale,
}

/// An explicit exclusion; exclusions are data, not silently dropped observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayExclusion {
    /// Event being evaluated.
    pub event_id: String,
    /// Horizon being evaluated.
    pub horizon_id: String,
    /// Forecast method being evaluated.
    pub forecaster_id: String,
    /// Reason no score was emitted.
    pub reason: ReplayExclusionReason,
}

/// One forecast score with the selected observation and timing preserved.
#[derive(Debug, Clone, PartialEq)]
pub struct ForecastScore {
    /// Event being scored.
    pub event_id: String,
    /// Horizon being scored.
    pub horizon_id: String,
    /// Forecast method being scored.
    pub forecaster_id: String,
    /// Probability selected by the prequential cutoff rule.
    pub probability: f64,
    /// Realized binary outcome.
    pub outcome: bool,
    /// Unix-second time of the selected forecast.
    pub forecast_at_unix_seconds: u64,
    /// Unix-second frozen target cutoff.
    pub cutoff_at_unix_seconds: u64,
    /// Seconds between forecast observation and target cutoff.
    pub forecast_age_seconds: u64,
    /// Squared probability error; lower is better.
    pub brier_score: f64,
    /// Negative log probability assigned to the realized outcome; lower is better.
    ///
    /// This is mathematically infinite when the forecast assigns exactly zero probability to the
    /// realized outcome. No hidden epsilon floor is applied.
    pub log_loss: f64,
}

/// Per-event replay output. Metrics are not collapsed across events, so downstream inference can
/// use event-clustered uncertainty estimates rather than pretending repeated horizons are
/// independent.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayReport {
    /// Scores emitted for eligible method/event/horizon combinations.
    pub scored: Vec<ForecastScore>,
    /// Missing and stale combinations that were excluded from scoring.
    pub excluded: Vec<ReplayExclusion>,
}

/// Replay all method/event/horizon combinations using only forecasts known by each cutoff.
///
/// Every method appearing anywhere in the observation input is checked against every target.
/// For a target, its latest pre-cutoff observation is selected; if that observation is stale, it
/// is excluded rather than replaced with a post-cutoff forecast or an older, staler observation.
/// Duplicate observation keys and duplicate event/horizon targets are rejected.
pub fn replay_horizons(
    observations: &[ForecastObservation],
    targets: &[ForecastTarget],
) -> Result<ReplayReport, ReplayError> {
    let mut observation_keys = BTreeSet::new();
    for observation in observations {
        let key = (
            observation.event_id.clone(),
            observation.forecaster_id.clone(),
            observation.observed_at_unix_seconds,
        );
        if !observation_keys.insert(key) {
            return Err(ReplayError::DuplicateObservation {
                event_id: observation.event_id.clone(),
                forecaster_id: observation.forecaster_id.clone(),
                observed_at_unix_seconds: observation.observed_at_unix_seconds,
            });
        }
    }

    let mut target_keys = BTreeSet::new();
    for target in targets {
        let key = (target.event_id.clone(), target.horizon_id.clone());
        if !target_keys.insert(key) {
            return Err(ReplayError::DuplicateTarget {
                event_id: target.event_id.clone(),
                horizon_id: target.horizon_id.clone(),
            });
        }
        if target.cutoff_at_unix_seconds >= target.resolved_at_unix_seconds {
            return Err(ReplayError::InvalidTargetTimes);
        }
        if target.max_staleness_seconds == 0 {
            return Err(ReplayError::ZeroMaxStaleness);
        }
    }

    let forecasters = observations
        .iter()
        .map(|observation| observation.forecaster_id.clone())
        .collect::<BTreeSet<_>>();
    let mut report = ReplayReport {
        scored: Vec::new(),
        excluded: Vec::new(),
    };

    for target in targets {
        for forecaster_id in &forecasters {
            let selected = observations
                .iter()
                .filter(|observation| {
                    observation.event_id == target.event_id
                        && observation.forecaster_id == *forecaster_id
                        && observation.observed_at_unix_seconds <= target.cutoff_at_unix_seconds
                })
                .max_by_key(|observation| observation.observed_at_unix_seconds);

            let Some(observation) = selected else {
                report.excluded.push(ReplayExclusion {
                    event_id: target.event_id.clone(),
                    horizon_id: target.horizon_id.clone(),
                    forecaster_id: forecaster_id.clone(),
                    reason: ReplayExclusionReason::NoForecastAtOrBeforeCutoff,
                });
                continue;
            };

            let age = target.cutoff_at_unix_seconds - observation.observed_at_unix_seconds;
            if age > target.max_staleness_seconds {
                report.excluded.push(ReplayExclusion {
                    event_id: target.event_id.clone(),
                    horizon_id: target.horizon_id.clone(),
                    forecaster_id: forecaster_id.clone(),
                    reason: ReplayExclusionReason::ForecastTooStale,
                });
                continue;
            }

            let outcome_value = if target.outcome { 1.0 } else { 0.0 };
            let brier_score = (observation.probability - outcome_value).powi(2);
            let realized_probability = if target.outcome {
                observation.probability
            } else {
                1.0 - observation.probability
            };
            let log_loss = -realized_probability.ln();

            report.scored.push(ForecastScore {
                event_id: target.event_id.clone(),
                horizon_id: target.horizon_id.clone(),
                forecaster_id: forecaster_id.clone(),
                probability: observation.probability,
                outcome: target.outcome,
                forecast_at_unix_seconds: observation.observed_at_unix_seconds,
                cutoff_at_unix_seconds: target.cutoff_at_unix_seconds,
                forecast_age_seconds: age,
                brier_score,
                log_loss,
            });
        }
    }

    report.scored.sort_by(|left, right| {
        (&left.event_id, &left.horizon_id, &left.forecaster_id)
            .cmp(&(&right.event_id, &right.horizon_id, &right.forecaster_id))
    });
    report.excluded.sort_by(|left, right| {
        (&left.event_id, &left.horizon_id, &left.forecaster_id)
            .cmp(&(&right.event_id, &right.horizon_id, &right.forecaster_id))
    });
    Ok(report)
}

fn required_identifier(value: String, field: &'static str) -> Result<String, ReplayError> {
    if value.trim().is_empty() {
        return Err(ReplayError::EmptyIdentifier(field));
    }
    Ok(value.trim().to_owned())
}

/// One paired comparison between a candidate and baseline forecast.
///
/// The cluster identifier should identify the independent unit for inference (for example, a
/// parent event that generated multiple related markets). Every row from one cluster is averaged
/// within that cluster before clusters receive equal weight in the bootstrap.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedForecastComparison {
    event_id: String,
    horizon_id: String,
    cluster_id: String,
    outcome: bool,
    baseline_probability: f64,
    candidate_probability: f64,
}

impl PairedForecastComparison {
    /// Construct one paired forecast comparison with explicit event and dependence-cluster IDs.
    pub fn new(
        event_id: impl Into<String>,
        horizon_id: impl Into<String>,
        cluster_id: impl Into<String>,
        outcome: bool,
        baseline_probability: f64,
        candidate_probability: f64,
    ) -> Result<Self, ReplayError> {
        let event_id = required_identifier(event_id.into(), "event_id")?;
        let horizon_id = required_identifier(horizon_id.into(), "horizon_id")?;
        let cluster_id = required_identifier(cluster_id.into(), "cluster_id")?;
        if !baseline_probability.is_finite()
            || !candidate_probability.is_finite()
            || !(0.0..=1.0).contains(&baseline_probability)
            || !(0.0..=1.0).contains(&candidate_probability)
        {
            return Err(ReplayError::InvalidComparisonValue);
        }
        Ok(Self {
            event_id,
            horizon_id,
            cluster_id,
            outcome,
            baseline_probability,
            candidate_probability,
        })
    }
}

/// Paired, parent-event-cluster bootstrap summary for Brier-score differences.
///
/// Difference is defined as candidate Brier score minus baseline Brier score, so a negative
/// value favors the candidate. The percentile interval is a deterministic nonparametric bootstrap
/// interval, not a guarantee of frequentist coverage under arbitrary dependence or selection.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterBootstrapSummary {
    /// Number of paired event/horizon rows.
    pub comparison_count: usize,
    /// Number of independent cluster labels represented.
    pub cluster_count: usize,
    /// Number of bootstrap resamples used.
    pub bootstrap_replicates: usize,
    /// Seed used by the reproducible xorshift resampler.
    pub seed: u64,
    /// Mean cluster-weighted candidate-minus-baseline Brier score.
    pub mean_brier_difference: f64,
    /// Lower 2.5th percentile of bootstrap mean differences.
    pub lower_95_brier_difference: f64,
    /// Upper 97.5th percentile of bootstrap mean differences.
    pub upper_95_brier_difference: f64,
    /// Fraction of bootstrap means strictly below zero; not a p-value.
    pub bootstrap_share_candidate_better: f64,
}

/// Compute paired Brier differences with an equal-weight event-cluster bootstrap.
///
/// Rows sharing one cluster ID are averaged first, then clusters—not rows—are resampled with
/// replacement. This prevents repeated horizons or sibling contracts from masquerading as
/// independent observations. At least 1,000 replicates and a nonzero seed are required. The
/// custom xorshift PRNG is for deterministic research replay only, not security or cryptography.
///
/// The caller is responsible for defining valid event clusters before observing benchmark
/// results. If clusters are not independent, this interval may still be overconfident.
pub fn paired_cluster_bootstrap(
    comparisons: &[PairedForecastComparison],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<ClusterBootstrapSummary, ReplayError> {
    if comparisons.is_empty() {
        return Err(ReplayError::EmptyComparisonSet);
    }
    if bootstrap_replicates < 1_000 || seed == 0 {
        return Err(ReplayError::InvalidBootstrapReplicates);
    }

    let mut seen_pairs = BTreeSet::new();
    let mut cluster_differences: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();

    for row in comparisons {
        let key = (row.event_id.clone(), row.horizon_id.clone());
        if !seen_pairs.insert(key) {
            return Err(ReplayError::DuplicateComparison {
                event_id: row.event_id.clone(),
                horizon_id: row.horizon_id.clone(),
            });
        }
        let outcome = if row.outcome { 1.0 } else { 0.0 };
        let baseline_brier = (row.baseline_probability - outcome).powi(2);
        let candidate_brier = (row.candidate_probability - outcome).powi(2);
        cluster_differences
            .entry(row.cluster_id.clone())
            .or_default()
            .push(candidate_brier - baseline_brier);
    }

    let cluster_means = cluster_differences
        .values()
        .map(|values| values.iter().sum::<f64>() / values.len() as f64)
        .collect::<Vec<_>>();
    let cluster_count = cluster_means.len();
    let mean_brier_difference =
        cluster_means.iter().sum::<f64>() / cluster_count as f64;

    let mut rng_state = seed;
    let mut bootstrap_means = Vec::with_capacity(bootstrap_replicates);
    let mut candidate_better_count = 0usize;
    for _ in 0..bootstrap_replicates {
        let mut sum = 0.0;
        for _ in 0..cluster_count {
            let index = (next_replay_random(&mut rng_state) as usize) % cluster_count;
            sum += cluster_means[index];
        }
        let mean = sum / cluster_count as f64;
        if mean < 0.0 {
            candidate_better_count += 1;
        }
        bootstrap_means.push(mean);
    }
    bootstrap_means.sort_by(f64::total_cmp);

    let lower_index = ((bootstrap_replicates - 1) as f64 * 0.025).floor() as usize;
    let upper_index = ((bootstrap_replicates - 1) as f64 * 0.975).floor() as usize;

    Ok(ClusterBootstrapSummary {
        comparison_count: comparisons.len(),
        cluster_count,
        bootstrap_replicates,
        seed,
        mean_brier_difference,
        lower_95_brier_difference: bootstrap_means[lower_index],
        upper_95_brier_difference: bootstrap_means[upper_index],
        bootstrap_share_candidate_better: candidate_better_count as f64
            / bootstrap_replicates as f64,
    })
}

fn next_replay_random(state: &mut u64) -> u64 {
    // Xorshift64*: deterministic non-cryptographic generator. Seed zero is rejected above.
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(event: &str, method: &str, at: u64, p: f64) -> ForecastObservation {
        ForecastObservation::new(event, method, at, p).expect("valid forecast fixture")
    }

    fn target(
        event: &str,
        horizon: &str,
        cutoff: u64,
        resolved: u64,
        outcome: bool,
        max_age: u64,
    ) -> ForecastTarget {
        ForecastTarget::new(event, horizon, cutoff, resolved, outcome, max_age)
            .expect("valid target fixture")
    }


    fn paired(
        event: &str,
        horizon: &str,
        cluster: &str,
        outcome: bool,
        baseline: f64,
        candidate: f64,
    ) -> PairedForecastComparison {
        PairedForecastComparison::new(event, horizon, cluster, outcome, baseline, candidate)
            .expect("valid paired comparison fixture")
    }

    #[test]
    fn clustered_bootstrap_is_reproducible_for_identical_input_and_seed() {
        let rows = vec![
            paired("e1", "30d", "parent-a", true, 0.45, 0.60),
            paired("e1", "7d", "parent-a", true, 0.70, 0.80),
            paired("e2", "30d", "parent-b", false, 0.35, 0.20),
            paired("e3", "30d", "parent-c", true, 0.50, 0.70),
            paired("e4", "30d", "parent-d", false, 0.50, 0.30),
        ];
        let a = paired_cluster_bootstrap(&rows, 2_000, 0x5eed).expect("valid bootstrap");
        let b = paired_cluster_bootstrap(&rows, 2_000, 0x5eed).expect("same deterministic run");
        assert_eq!(a, b);
        assert_eq!(a.comparison_count, 5);
        assert_eq!(a.cluster_count, 4);
        assert_eq!(a.bootstrap_replicates, 2_000);
    }

    #[test]
    fn clustered_bootstrap_weights_clusters_not_repeated_horizons() {
        let rows = vec![
            paired("a", "h1", "parent-a", true, 0.1, 0.9),
            paired("a", "h2", "parent-a", true, 0.1, 0.9),
            paired("a", "h3", "parent-a", true, 0.1, 0.9),
            paired("a", "h4", "parent-a", true, 0.1, 0.9),
            paired("a", "h5", "parent-a", true, 0.1, 0.9),
            paired("a", "h6", "parent-a", true, 0.1, 0.9),
            paired("a", "h7", "parent-a", true, 0.1, 0.9),
            paired("a", "h8", "parent-a", true, 0.1, 0.9),
            paired("a", "h9", "parent-a", true, 0.1, 0.9),
            paired("a", "h10", "parent-a", true, 0.1, 0.9),
            paired("b", "h1", "parent-b", true, 0.9, 0.1),
        ];
        let summary = paired_cluster_bootstrap(&rows, 2_000, 123).expect("valid bootstrap");
        assert_eq!(summary.comparison_count, 11);
        assert_eq!(summary.cluster_count, 2);
        assert!(summary.mean_brier_difference.abs() < 1e-12);
    }

    #[test]
    fn clustered_bootstrap_rejects_empty_duplicate_or_invalid_input() {
        assert_eq!(
            paired_cluster_bootstrap(&[], 1_000, 1),
            Err(ReplayError::EmptyComparisonSet)
        );
        let row = paired("e1", "30d", "p1", true, 0.4, 0.6);
        assert_eq!(
            paired_cluster_bootstrap(&[row.clone(), row], 1_000, 1),
            Err(ReplayError::DuplicateComparison {
                event_id: "e1".to_owned(),
                horizon_id: "30d".to_owned(),
            })
        );
        assert_eq!(
            paired_cluster_bootstrap(&[paired("e", "h", "c", true, 0.4, 0.5)], 999, 1),
            Err(ReplayError::InvalidBootstrapReplicates)
        );
        assert_eq!(
            paired_cluster_bootstrap(&[paired("e", "h", "c", true, 0.4, 0.5)], 1_000, 0),
            Err(ReplayError::InvalidBootstrapReplicates)
        );
        assert_eq!(
            PairedForecastComparison::new("e", "h", "c", true, f64::NAN, 0.5),
            Err(ReplayError::InvalidComparisonValue)
        );
        assert_eq!(
            PairedForecastComparison::new("e", "h", "c", true, 0.4, 1.1),
            Err(ReplayError::InvalidComparisonValue)
        );
    }

    #[test]
    fn clustered_bootstrap_difference_is_candidate_minus_baseline() {
        let rows = vec![
            paired("e1", "30d", "p1", true, 0.2, 0.8),
            paired("e2", "30d", "p2", true, 0.3, 0.9),
            paired("e3", "30d", "p3", true, 0.1, 0.7),
        ];
        let summary = paired_cluster_bootstrap(&rows, 1_000, 7).expect("valid bootstrap");
        assert!(summary.mean_brier_difference > 0.0);
        assert_eq!(summary.bootstrap_share_candidate_better, 0.0);
        assert!(summary.lower_95_brier_difference > 0.0);
    }

    #[test]
    fn selects_latest_forecast_at_or_before_cutoff() {
        let observations = vec![
            observation("e1", "market", 90, 0.2),
            observation("e1", "market", 100, 0.8),
            observation("e1", "market", 101, 0.99),
        ];
        let report = replay_horizons(
            &observations,
            &[target("e1", "30d", 100, 200, true, 10)],
        )
        .expect("valid replay");
        assert_eq!(report.scored.len(), 1);
        assert_eq!(report.scored[0].probability, 0.8);
        assert_eq!(report.scored[0].forecast_at_unix_seconds, 100);
        assert_eq!(report.scored[0].forecast_age_seconds, 0);
        assert!((report.scored[0].brier_score - 0.04).abs() < 1e-12);
    }

    #[test]
    fn future_only_forecast_is_an_explicit_exclusion() {
        let observations = vec![observation("e1", "market", 101, 0.99)];
        let report = replay_horizons(&observations, &[target("e1", "30d", 100, 200, true, 10)])
            .expect("valid replay");
        assert!(report.scored.is_empty());
        assert_eq!(report.excluded.len(), 1);
        assert_eq!(
            report.excluded[0].reason,
            ReplayExclusionReason::NoForecastAtOrBeforeCutoff
        );
    }

    #[test]
    fn stale_forecast_is_not_backfilled_as_a_valid_score() {
        let observations = vec![observation("e1", "market", 80, 0.7)];
        let report = replay_horizons(
            &observations,
            &[target("e1", "30d", 100, 200, true, 5)],
        )
        .expect("valid replay");
        assert!(report.scored.is_empty());
        assert_eq!(report.excluded[0].reason, ReplayExclusionReason::ForecastTooStale);
    }

    #[test]
    fn impossible_realized_outcome_has_infinite_log_loss_not_an_epsilon_clamp() {
        let observations = vec![observation("e1", "market", 100, 0.0)];
        let report = replay_horizons(
            &observations,
            &[target("e1", "1d", 100, 200, true, 1)],
        )
        .expect("valid replay");
        assert_eq!(report.scored[0].brier_score, 1.0);
        assert!(report.scored[0].log_loss.is_infinite());
    }

    #[test]
    fn probabilities_must_be_finite_and_bounded() {
        assert_eq!(
            ForecastObservation::new("e", "m", 10, f64::NAN),
            Err(ReplayError::InvalidProbability)
        );
        assert_eq!(
            ForecastObservation::new("e", "m", 10, 1.01),
            Err(ReplayError::InvalidProbability)
        );
    }

    #[test]
    fn duplicate_observation_keys_fail_closed() {
        let observations = vec![
            observation("e1", "market", 100, 0.4),
            observation("e1", "market", 100, 0.6),
        ];
        assert_eq!(
            replay_horizons(&observations, &[]),
            Err(ReplayError::DuplicateObservation {
                event_id: "e1".to_owned(),
                forecaster_id: "market".to_owned(),
                observed_at_unix_seconds: 100,
            })
        );
    }

    #[test]
    fn duplicate_event_horizon_target_fails_closed() {
        let t = target("e1", "30d", 100, 200, true, 10);
        assert_eq!(
            replay_horizons(&[], &[t.clone(), t]),
            Err(ReplayError::DuplicateTarget {
                event_id: "e1".to_owned(),
                horizon_id: "30d".to_owned(),
            })
        );
    }

    #[test]
    fn target_must_precede_resolution_and_staleness_must_be_positive() {
        assert_eq!(
            ForecastTarget::new("e1", "close", 200, 200, true, 10),
            Err(ReplayError::InvalidTargetTimes)
        );
        assert_eq!(
            ForecastTarget::new("e1", "close", 199, 200, true, 0),
            Err(ReplayError::ZeroMaxStaleness)
        );
    }

    #[test]
    fn each_method_is_checked_for_each_horizon_without_averaging_events_together() {
        let observations = vec![
            observation("e1", "market", 90, 0.4),
            observation("e1", "symthaea", 95, 0.7),
            observation("e2", "market", 190, 0.3),
            observation("e2", "symthaea", 195, 0.6),
        ];
        let targets = vec![
            target("e1", "90d", 100, 200, true, 20),
            target("e2", "7d", 200, 300, false, 20),
        ];
        let report = replay_horizons(&observations, &targets).expect("valid replay");
        assert_eq!(report.scored.len(), 4);
        assert_eq!(report.scored[0].event_id, "e1");
        assert_eq!(report.scored[0].horizon_id, "90d");
        assert!(report
            .scored
            .iter()
            .any(|row| row.forecaster_id == "symthaea" && row.event_id == "e2"));
    }
}
