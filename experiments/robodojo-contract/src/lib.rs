//! Dependency-free fail-closed contract gate for embodied-policy experiments.
//!
//! This validates a candidate-action boundary; it is not a robot controller,
//! collision checker, or proof of physical safety. Prediction error can request
//! replanning, but environment-specific limits and collision checks must be
//! enforced by an adapter with access to the robot and simulator telemetry.
#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisLimit { pub min: f64, pub max: f64 }

impl AxisLimit {
    pub fn new(min: f64, max: f64) -> Result<Self, ConfigError> {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(ConfigError::InvalidAxisLimit);
        }
        Ok(Self { min, max })
    }
    fn contains(self, value: f64) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateConfig {
    pub state_dim: usize,
    pub action_limits: Vec<AxisLimit>,
    pub prediction_error_threshold: f64,
    pub max_consecutive_high_error: usize,
}

impl GateConfig {
    pub fn new(
        state_dim: usize,
        action_limits: Vec<AxisLimit>,
        prediction_error_threshold: f64,
        max_consecutive_high_error: usize,
    ) -> Result<Self, ConfigError> {
        if state_dim == 0 { return Err(ConfigError::EmptyState); }
        if action_limits.is_empty() { return Err(ConfigError::EmptyActionSpace); }
        if action_limits.iter().any(|x| !x.min.is_finite() || !x.max.is_finite() || x.min > x.max) {
            return Err(ConfigError::InvalidAxisLimit);
        }
        if !prediction_error_threshold.is_finite() || prediction_error_threshold < 0.0 {
            return Err(ConfigError::InvalidPredictionThreshold);
        }
        if max_consecutive_high_error == 0 { return Err(ConfigError::ZeroRecoveryBudget); }
        Ok(Self { state_dim, action_limits, prediction_error_threshold, max_consecutive_high_error })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    EmptyState, EmptyActionSpace, InvalidAxisLimit, InvalidPredictionThreshold, ZeroRecoveryBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationFrame {
    pub episode_id: String,
    pub step: u64,
    pub state_values: Vec<f64>,
    /// Caller-defined predictor error; must be calibrated for the chosen policy.
    pub prediction_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionProposal {
    pub episode_id: String,
    pub based_on_step: u64,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    NoActiveEpisode,
    EmptyEpisodeId,
    EpisodeMismatch,
    StateDimensionMismatch { expected: usize, actual: usize },
    NonFiniteObservation { index: usize },
    InvalidPredictionError,
    NonMonotonicStep { previous: u64, received: u64 },
    SustainedPredictionError { observations: usize, threshold: f64 },
    ActionDimensionMismatch { expected: usize, actual: usize },
    NonFiniteAction { index: usize },
    ActionOutOfBounds { index: usize, value: f64, min: f64, max: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObservationDecision {
    Accepted,
    RequestReplan { prediction_error: f64, consecutive_high_error: usize },
    SafeStop(StopReason),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionRejection {
    SafeStop(StopReason),
    NoObservation,
    StaleObservation { observed_step: u64, action_step: u64 },
    ReplanRequired { observed_step: u64 },
}

/// Episode-scoped guard. Invalid observations/actions latch a stop until an
/// explicit episode reset. Repeated high prediction errors stop the trace;
/// one high error requests replanning and cannot authorize a new action.
#[derive(Debug, Clone)]
pub struct EpisodeGuard {
    config: GateConfig,
    episode_id: Option<String>,
    last_step: Option<u64>,
    consecutive_high_error: usize,
    observation_permits_action: bool,
    fault: Option<StopReason>,
}

impl EpisodeGuard {
    pub fn new(config: GateConfig) -> Self {
        Self {
            config, episode_id: None, last_step: None, consecutive_high_error: 0,
            observation_permits_action: false, fault: None,
        }
    }

    /// Starts a new episode and is the only operation that clears a latched stop.
    pub fn start_episode(&mut self, episode_id: impl Into<String>) -> Result<(), StopReason> {
        let id = episode_id.into();
        if id.trim().is_empty() {
            self.episode_id = None;
            self.last_step = None;
            self.consecutive_high_error = 0;
            self.observation_permits_action = false;
            self.fault = Some(StopReason::EmptyEpisodeId);
            return Err(StopReason::EmptyEpisodeId);
        }
        self.episode_id = Some(id);
        self.last_step = None;
        self.consecutive_high_error = 0;
        self.observation_permits_action = false;
        self.fault = None;
        Ok(())
    }

    pub fn last_observed_step(&self) -> Option<u64> { self.last_step }
    pub fn consecutive_high_error(&self) -> usize { self.consecutive_high_error }
    pub fn stopped(&self) -> Option<&StopReason> { self.fault.as_ref() }

    pub fn observe(&mut self, frame: &ObservationFrame) -> ObservationDecision {
        if let Some(reason) = self.fault.clone() { return ObservationDecision::SafeStop(reason); }
        let Some(active) = self.episode_id.as_deref() else {
            return self.stop(StopReason::NoActiveEpisode);
        };
        if active != frame.episode_id { return self.stop(StopReason::EpisodeMismatch); }
        if frame.state_values.len() != self.config.state_dim {
            return self.stop(StopReason::StateDimensionMismatch {
                expected: self.config.state_dim, actual: frame.state_values.len(),
            });
        }
        if let Some(index) = frame.state_values.iter().position(|x| !x.is_finite()) {
            return self.stop(StopReason::NonFiniteObservation { index });
        }
        if !frame.prediction_error.is_finite() || frame.prediction_error < 0.0 {
            return self.stop(StopReason::InvalidPredictionError);
        }
        if let Some(previous) = self.last_step {
            if frame.step <= previous {
                return self.stop(StopReason::NonMonotonicStep { previous, received: frame.step });
            }
        }

        self.last_step = Some(frame.step);
        if frame.prediction_error > self.config.prediction_error_threshold {
            self.consecutive_high_error = self.consecutive_high_error.saturating_add(1);
            self.observation_permits_action = false;
            if self.consecutive_high_error >= self.config.max_consecutive_high_error {
                return self.stop(StopReason::SustainedPredictionError {
                    observations: self.consecutive_high_error,
                    threshold: self.config.prediction_error_threshold,
                });
            }
            ObservationDecision::RequestReplan {
                prediction_error: frame.prediction_error,
                consecutive_high_error: self.consecutive_high_error,
            }
        } else {
            self.consecutive_high_error = 0;
            self.observation_permits_action = true;
            ObservationDecision::Accepted
        }
    }

    /// Validates a candidate only; it never sends commands to a robot.
    /// Approval means only that this contract's checks passed.
    pub fn approve_action(&mut self, proposal: &ActionProposal) -> Result<Vec<f64>, ActionRejection> {
        if let Some(reason) = self.fault.clone() { return Err(ActionRejection::SafeStop(reason)); }
        let Some(active) = self.episode_id.as_deref() else {
            return self.stop_action(StopReason::NoActiveEpisode);
        };
        if active != proposal.episode_id { return self.stop_action(StopReason::EpisodeMismatch); }
        let Some(observed_step) = self.last_step else { return Err(ActionRejection::NoObservation); };
        if proposal.based_on_step != observed_step {
            return Err(ActionRejection::StaleObservation {
                observed_step, action_step: proposal.based_on_step,
            });
        }
        if !self.observation_permits_action {
            return Err(ActionRejection::ReplanRequired { observed_step });
        }
        if proposal.values.len() != self.config.action_limits.len() {
            return self.stop_action(StopReason::ActionDimensionMismatch {
                expected: self.config.action_limits.len(), actual: proposal.values.len(),
            });
        }
        for (index, (&value, limit)) in proposal.values.iter().zip(&self.config.action_limits).enumerate() {
            if !value.is_finite() { return self.stop_action(StopReason::NonFiniteAction { index }); }
            if !limit.contains(value) {
                return self.stop_action(StopReason::ActionOutOfBounds {
                    index, value, min: limit.min, max: limit.max,
                });
            }
        }
        Ok(proposal.values.clone())
    }

    fn stop(&mut self, reason: StopReason) -> ObservationDecision {
        self.observation_permits_action = false;
        if self.fault.is_none() { self.fault = Some(reason.clone()); }
        ObservationDecision::SafeStop(self.fault.clone().unwrap_or(reason))
    }

    fn stop_action<T>(&mut self, reason: StopReason) -> Result<T, ActionRejection> {
        self.observation_permits_action = false;
        if self.fault.is_none() { self.fault = Some(reason.clone()); }
        Err(ActionRejection::SafeStop(self.fault.clone().unwrap_or(reason)))
    }
}


/// One trial from an adapted or official robot benchmark. A valid partial score
/// is still a valid trial; a run without a final judge score uses progress=None
/// and remains in reliability/timeout denominators rather than disappearing.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialRecord {
    pub task_id: String,
    pub variant_id: String,
    /// Stable matched-trial key, normally a preregistered trial/seed identifier.
    pub trial_id: String,
    pub progress: Option<f64>,
    pub completed: bool,
    pub valid_run: bool,
    pub timed_out: bool,
    pub elapsed_seconds: f64,
    pub model_calls: u32,
    pub action_steps: u32,
    pub estimated_cost_usd: f64,
    pub safety_events: u32,
}

impl TrialRecord {
    pub fn validate(&self) -> Result<(), MetricError> {
        if self.task_id.trim().is_empty()
            || self.variant_id.trim().is_empty()
            || self.trial_id.trim().is_empty()
        {
            return Err(MetricError::EmptyIdentifier);
        }
        if !self.elapsed_seconds.is_finite() || self.elapsed_seconds < 0.0 {
            return Err(MetricError::InvalidDuration);
        }
        if !self.estimated_cost_usd.is_finite() || self.estimated_cost_usd < 0.0 {
            return Err(MetricError::InvalidCost);
        }
        if self.valid_run != self.progress.is_some() {
            return Err(MetricError::ValidityScoreMismatch);
        }
        match self.progress {
            Some(score) if !score.is_finite() || !(0.0..=1.0).contains(&score) => {
                Err(MetricError::InvalidProgress)
            }
            Some(score) if self.completed != (score >= 1.0) => {
                Err(MetricError::CompletionScoreMismatch)
            }
            None if self.completed => Err(MetricError::CompletionScoreMismatch),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricError {
    EmptyRecords,
    EmptyIdentifier,
    InvalidProgress,
    CompletionScoreMismatch,
    ValidityScoreMismatch,
    InvalidDuration,
    InvalidCost,
    DuplicateTrial,
    MixedVariants,
    MismatchedPairedTrials,
    TooFewBootstrapSamples,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskSummary {
    pub task_id: String,
    pub variant_id: String,
    pub total_trials: usize,
    pub valid_trials: usize,
    pub invalid_trials: usize,
    pub completions: usize,
    /// Full completions divided by all attempted trials; invalid runs count as
    /// non-completions instead of silently vanishing from the denominator.
    pub completion_rate_all_trials: f64,
    /// Full completions divided by trials with a usable final score.
    pub completion_rate_valid_trials: Option<f64>,
    pub mean_progress: Option<f64>,
    pub progress_ci95: Option<(f64, f64)>,
    pub mean_elapsed_seconds: f64,
    pub mean_model_calls: f64,
    pub mean_action_steps: f64,
    pub mean_estimated_cost_usd: f64,
    pub timeout_count: usize,
    pub safety_event_count: u64,
}

/// Summarize every task × variant independently. This does not aggregate tasks
/// with different difficulty into one score. Bootstrap intervals are
/// deterministic for a given input, replicate count, and seed.
pub fn summarize_trials(
    records: &[TrialRecord],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<Vec<TaskSummary>, MetricError> {
    if records.is_empty() { return Err(MetricError::EmptyRecords); }
    if bootstrap_replicates > 0 && bootstrap_replicates < 100 {
        return Err(MetricError::TooFewBootstrapSamples);
    }

    use std::collections::{BTreeMap, BTreeSet};
    let mut groups: BTreeMap<(String, String), Vec<&TrialRecord>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for record in records {
        record.validate()?;
        let trial_key = (record.task_id.clone(), record.variant_id.clone(), record.trial_id.clone());
        if !seen.insert(trial_key) { return Err(MetricError::DuplicateTrial); }
        groups.entry((record.task_id.clone(), record.variant_id.clone())).or_default().push(record);
    }

    let mut summaries = Vec::with_capacity(groups.len());
    for ((task_id, variant_id), trials) in groups {
        let total = trials.len();
        let valid_scores: Vec<f64> = trials.iter().filter_map(|r| r.progress).collect();
        let valid = valid_scores.len();
        let completions = trials.iter().filter(|r| r.completed).count();
        let progress_ci95 = if valid_scores.len() >= 2 && bootstrap_replicates > 0 {
            Some(bootstrap_mean_ci(&valid_scores, bootstrap_replicates, derive_seed(seed, &task_id, &variant_id)))
        } else {
            None
        };
        let sum = |f: fn(&TrialRecord) -> f64| trials.iter().map(|r| f(r)).sum::<f64>() / total as f64;
        summaries.push(TaskSummary {
            task_id,
            variant_id,
            total_trials: total,
            valid_trials: valid,
            invalid_trials: total - valid,
            completions,
            completion_rate_all_trials: completions as f64 / total as f64,
            completion_rate_valid_trials: (valid > 0).then(|| completions as f64 / valid as f64),
            mean_progress: if valid > 0 { Some(valid_scores.iter().sum::<f64>() / valid as f64) } else { None },
            progress_ci95,
            mean_elapsed_seconds: sum(|r| r.elapsed_seconds),
            mean_model_calls: sum(|r| r.model_calls as f64),
            mean_action_steps: sum(|r| r.action_steps as f64),
            mean_estimated_cost_usd: sum(|r| r.estimated_cost_usd),
            timeout_count: trials.iter().filter(|r| r.timed_out).count(),
            safety_event_count: trials.iter().map(|r| r.safety_events as u64).sum(),
        });
    }
    Ok(summaries)
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedComparison {
    pub baseline_variant: String,
    pub candidate_variant: String,
    pub total_pairs: usize,
    pub scored_pairs: usize,
    pub tasks_with_scored_pairs: usize,
    /// Candidate minus baseline; macro-averaged over tasks, each task weighted equally.
    pub mean_progress_delta: Option<f64>,
    /// Stratified paired bootstrap interval for the task-macro progress delta.
    pub progress_delta_ci95: Option<(f64, f64)>,
    /// Candidate minus baseline over all matched attempts; invalid attempts count as incomplete.
    pub completion_rate_delta: f64,
    pub valid_run_rate_delta: f64,
    pub mean_elapsed_seconds_delta: f64,
    pub mean_cost_usd_delta: f64,
    pub safety_event_count_delta: i64,
}

/// Compare two conditions only when each task/trial key is present exactly once
/// on both sides. Paired score differences use only pairs with two valid graded
/// scores, while validity/completion/runtime/cost/fault deltas retain all pairs.
pub fn compare_paired_trials(
    baseline: &[TrialRecord],
    candidate: &[TrialRecord],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<PairedComparison, MetricError> {
    use std::collections::{BTreeMap, BTreeSet};
    if baseline.is_empty() || candidate.is_empty() { return Err(MetricError::EmptyRecords); }
    if bootstrap_replicates > 0 && bootstrap_replicates < 100 {
        return Err(MetricError::TooFewBootstrapSamples);
    }

    fn index(records: &[TrialRecord]) -> Result<(String, BTreeMap<(String, String), &TrialRecord>), MetricError> {
        let mut variant: Option<String> = None;
        let mut map = BTreeMap::new();
        for r in records {
            r.validate()?;
            if let Some(v) = &variant {
                if v != &r.variant_id { return Err(MetricError::MixedVariants); }
            } else {
                variant = Some(r.variant_id.clone());
            }
            let key = (r.task_id.clone(), r.trial_id.clone());
            if map.insert(key, r).is_some() { return Err(MetricError::DuplicateTrial); }
        }
        Ok((variant.ok_or(MetricError::EmptyRecords)?, map))
    }

    let (baseline_variant, base) = index(baseline)?;
    let (candidate_variant, cand) = index(candidate)?;
    if baseline_variant == candidate_variant || base.len() != cand.len() {
        return Err(MetricError::MismatchedPairedTrials);
    }
    let base_keys: BTreeSet<_> = base.keys().cloned().collect();
    let cand_keys: BTreeSet<_> = cand.keys().cloned().collect();
    if base_keys != cand_keys { return Err(MetricError::MismatchedPairedTrials); }

    let mut per_task_deltas: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut scored_pairs = 0usize;
    let mut candidate_completions = 0usize;
    let mut baseline_completions = 0usize;
    let mut candidate_valid = 0usize;
    let mut baseline_valid = 0usize;
    let mut elapsed_delta = 0.0;
    let mut cost_delta = 0.0;
    let mut safety_delta = 0i64;
    for (key, b) in &base {
        let c = cand.get(key).expect("paired key sets were checked");
        candidate_completions += usize::from(c.completed);
        baseline_completions += usize::from(b.completed);
        candidate_valid += usize::from(c.valid_run);
        baseline_valid += usize::from(b.valid_run);
        elapsed_delta += c.elapsed_seconds - b.elapsed_seconds;
        cost_delta += c.estimated_cost_usd - b.estimated_cost_usd;
        safety_delta += c.safety_events as i64 - b.safety_events as i64;
        if let (Some(bp), Some(cp)) = (b.progress, c.progress) {
            per_task_deltas.entry(key.0.clone()).or_default().push(cp - bp);
            scored_pairs += 1;
        }
    }
    let tasks_with_scored_pairs = per_task_deltas.len();
    let mean_progress_delta = if tasks_with_scored_pairs == 0 {
        None
    } else {
        Some(per_task_deltas.values()
            .map(|d| d.iter().sum::<f64>() / d.len() as f64)
            .sum::<f64>() / tasks_with_scored_pairs as f64)
    };
    let progress_delta_ci95 = if scored_pairs >= 2 && bootstrap_replicates > 0 {
        Some(bootstrap_macro_delta_ci(&per_task_deltas, bootstrap_replicates, seed))
    } else {
        None
    };
    let n = base.len() as f64;
    Ok(PairedComparison {
        baseline_variant,
        candidate_variant,
        total_pairs: base.len(),
        scored_pairs,
        tasks_with_scored_pairs,
        mean_progress_delta,
        progress_delta_ci95,
        completion_rate_delta: (candidate_completions as f64 - baseline_completions as f64) / n,
        valid_run_rate_delta: (candidate_valid as f64 - baseline_valid as f64) / n,
        mean_elapsed_seconds_delta: elapsed_delta / n,
        mean_cost_usd_delta: cost_delta / n,
        safety_event_count_delta: safety_delta,
    })
}

fn derive_seed(seed: u64, task: &str, variant: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64 ^ seed;
    for byte in task.bytes().chain([0xff]).chain(variant.bytes()) {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    if h == 0 { 0x9e3779b97f4a7c15 } else { h }
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545f4914f6cdd1d)
}

fn percentile(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    let index = ((sorted.len() - 1) * numerator) / denominator;
    sorted[index]
}

fn bootstrap_mean_ci(values: &[f64], replicates: usize, seed: u64) -> (f64, f64) {
    let mut rng = seed;
    let n = values.len();
    let mut means = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut total = 0.0;
        for _ in 0..n {
            total += values[(next_random(&mut rng) as usize) % n];
        }
        means.push(total / n as f64);
    }
    means.sort_by(f64::total_cmp);
    (percentile(&means, 25, 1000), percentile(&means, 975, 1000))
}

fn bootstrap_macro_delta_ci(
    by_task: &std::collections::BTreeMap<String, Vec<f64>>,
    replicates: usize,
    seed: u64,
) -> (f64, f64) {
    let mut rng = seed.max(1);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut task_means = 0.0;
        for deltas in by_task.values() {
            let mut total = 0.0;
            for _ in 0..deltas.len() {
                total += deltas[(next_random(&mut rng) as usize) % deltas.len()];
            }
            task_means += total / deltas.len() as f64;
        }
        estimates.push(task_means / by_task.len() as f64);
    }
    estimates.sort_by(f64::total_cmp);
    (percentile(&estimates, 25, 1000), percentile(&estimates, 975, 1000))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> EpisodeGuard {
        EpisodeGuard::new(GateConfig::new(
            3, vec![AxisLimit::new(-1.0, 1.0).unwrap(), AxisLimit::new(0.0, 0.08).unwrap()],
            0.5, 3,
        ).unwrap())
    }
    fn obs(ep: &str, step: u64, error: f64) -> ObservationFrame {
        ObservationFrame { episode_id: ep.to_owned(), step, state_values: vec![0.1, 0.2, 0.3], prediction_error: error }
    }
    fn action(ep: &str, step: u64) -> ActionProposal {
        ActionProposal { episode_id: ep.to_owned(), based_on_step: step, values: vec![0.25, 0.04] }
    }

    #[test]
    fn valid_observation_and_bounded_action_pass() {
        let mut g = guard(); g.start_episode("a").unwrap();
        assert_eq!(g.observe(&obs("a", 1, 0.2)), ObservationDecision::Accepted);
        assert_eq!(g.approve_action(&action("a", 1)), Ok(vec![0.25, 0.04]));
    }
    #[test]
    fn stale_action_is_rejected() {
        let mut g = guard(); g.start_episode("a").unwrap();
        g.observe(&obs("a", 1, 0.1)); g.observe(&obs("a", 2, 0.1));
        assert_eq!(g.approve_action(&action("a", 1)),
            Err(ActionRejection::StaleObservation { observed_step: 2, action_step: 1 }));
    }
    #[test]
    fn wrong_episode_latches_stop_until_reset() {
        let mut g = guard(); g.start_episode("a").unwrap();
        assert_eq!(g.observe(&obs("b", 1, 0.1)), ObservationDecision::SafeStop(StopReason::EpisodeMismatch));
        assert!(matches!(g.approve_action(&action("a", 1)), Err(ActionRejection::SafeStop(_))));
        g.start_episode("c").unwrap();
        assert_eq!(g.observe(&obs("c", 0, 0.1)), ObservationDecision::Accepted);
    }
    #[test]
    fn non_finite_observation_stops() {
        let mut g = guard(); g.start_episode("a").unwrap();
        let mut frame = obs("a", 1, 0.1); frame.state_values[1] = f64::NAN;
        assert_eq!(g.observe(&frame), ObservationDecision::SafeStop(StopReason::NonFiniteObservation { index: 1 }));
    }
    #[test]
    fn repeated_or_out_of_order_steps_stop() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 7, 0.1));
        assert_eq!(g.observe(&obs("a", 7, 0.1)),
            ObservationDecision::SafeStop(StopReason::NonMonotonicStep { previous: 7, received: 7 }));
    }
    #[test]
    fn high_error_requests_replan_and_blocks_action_until_recovery() {
        let mut g = guard(); g.start_episode("a").unwrap();
        assert_eq!(g.observe(&obs("a", 1, 0.9)),
            ObservationDecision::RequestReplan { prediction_error: 0.9, consecutive_high_error: 1 });
        assert_eq!(g.approve_action(&action("a", 1)), Err(ActionRejection::ReplanRequired { observed_step: 1 }));
        assert_eq!(g.observe(&obs("a", 2, 0.1)), ObservationDecision::Accepted);
        assert_eq!(g.consecutive_high_error(), 0);
        assert!(g.approve_action(&action("a", 2)).is_ok());
    }
    #[test]
    fn sustained_high_error_latches_stop() {
        let mut g = guard(); g.start_episode("a").unwrap();
        g.observe(&obs("a", 1, 0.9)); g.observe(&obs("a", 2, 0.9));
        assert_eq!(g.observe(&obs("a", 3, 0.9)),
            ObservationDecision::SafeStop(StopReason::SustainedPredictionError { observations: 3, threshold: 0.5 }));
    }
    #[test]
    fn non_finite_action_is_never_returned() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 1, 0.1));
        let mut bad = action("a", 1); bad.values[0] = f64::INFINITY;
        assert_eq!(g.approve_action(&bad), Err(ActionRejection::SafeStop(StopReason::NonFiniteAction { index: 0 })));
    }
    #[test]
    fn out_of_range_action_is_never_returned() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 1, 0.1));
        let mut bad = action("a", 1); bad.values[1] = 0.1;
        assert!(matches!(g.approve_action(&bad), Err(ActionRejection::SafeStop(StopReason::ActionOutOfBounds { index: 1, .. }))));
    }
    #[test]
    fn episode_reset_clears_previous_state() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 10, 0.1));
        g.observe(&obs("a", 11, 0.9)); g.start_episode("b").unwrap();
        assert_eq!(g.last_observed_step(), None); assert_eq!(g.consecutive_high_error(), 0);
        assert_eq!(g.approve_action(&action("b", 0)), Err(ActionRejection::NoObservation));
    }
    #[test]
    fn invalid_configuration_is_rejected() {
        assert_eq!(AxisLimit::new(2.0, 1.0), Err(ConfigError::InvalidAxisLimit));
        assert_eq!(GateConfig::new(0, vec![AxisLimit { min: 0.0, max: 1.0 }], 0.5, 3), Err(ConfigError::EmptyState));
        assert_eq!(GateConfig::new(1, Vec::new(), 0.5, 3), Err(ConfigError::EmptyActionSpace));
        assert_eq!(GateConfig::new(1, vec![AxisLimit { min: 0.0, max: 1.0 }], f64::NAN, 3), Err(ConfigError::InvalidPredictionThreshold));
        assert_eq!(GateConfig::new(1, vec![AxisLimit { min: 0.0, max: 1.0 }], 0.5, 0), Err(ConfigError::ZeroRecoveryBudget));
    }
    #[test]
    fn same_trace_has_same_decisions() {
        fn trace() -> Vec<ObservationDecision> {
            let mut g = guard(); g.start_episode("repeatable").unwrap();
            vec![g.observe(&obs("repeatable", 0, 0.1)), g.observe(&obs("repeatable", 1, 0.7)), g.observe(&obs("repeatable", 2, 0.2))]
        }
        assert_eq!(trace(), trace());
    }

    fn trial(task: &str, variant: &str, id: &str, progress: Option<f64>, elapsed: f64) -> TrialRecord {
        TrialRecord {
            task_id: task.to_owned(),
            variant_id: variant.to_owned(),
            trial_id: id.to_owned(),
            progress,
            completed: progress == Some(1.0),
            valid_run: progress.is_some(),
            timed_out: false,
            elapsed_seconds: elapsed,
            model_calls: 12,
            action_steps: 80,
            estimated_cost_usd: 0.25,
            safety_events: 0,
        }
    }

    #[test]
    fn partial_progress_and_invalid_runs_are_reported_separately() {
        let mut invalid = trial("stack-bowls", "base", "03", None, 5.0);
        invalid.timed_out = true;
        let rows = vec![
            trial("stack-bowls", "base", "01", Some(0.5), 4.0),
            trial("stack-bowls", "base", "02", Some(1.0), 6.0),
            invalid,
        ];
        let summary = summarize_trials(&rows, 2_000, 7).unwrap();
        assert_eq!(summary.len(), 1);
        let s = &summary[0];
        assert_eq!(s.total_trials, 3);
        assert_eq!(s.valid_trials, 2);
        assert_eq!(s.invalid_trials, 1);
        assert_eq!(s.completions, 1);
        assert!((s.completion_rate_all_trials - (1.0 / 3.0)).abs() < 1e-12);
        assert_eq!(s.completion_rate_valid_trials, Some(0.5));
        assert_eq!(s.mean_progress, Some(0.75));
        assert_eq!(s.timeout_count, 1);
        assert!(s.progress_ci95.is_some());
    }

    #[test]
    fn paired_comparison_is_task_macro_averaged_and_repeatable() {
        let mut base = vec![
            trial("task-a", "baseline", "01", Some(0.25), 10.0),
            trial("task-a", "baseline", "02", Some(0.50), 10.0),
            trial("task-b", "baseline", "01", Some(0.25), 10.0),
            trial("task-b", "baseline", "02", Some(0.50), 10.0),
        ];
        let mut candidate = vec![
            trial("task-a", "sidecar", "01", Some(0.50), 9.0),
            trial("task-a", "sidecar", "02", Some(0.75), 9.0),
            trial("task-b", "sidecar", "01", Some(0.50), 9.0),
            trial("task-b", "sidecar", "02", Some(0.75), 9.0),
        ];
        for row in &mut base { row.estimated_cost_usd = 1.00; }
        for row in &mut candidate { row.estimated_cost_usd = 0.80; }
        let a = compare_paired_trials(&base, &candidate, 2_000, 99).unwrap();
        let b = compare_paired_trials(&base, &candidate, 2_000, 99).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.total_pairs, 4);
        assert_eq!(a.scored_pairs, 4);
        assert_eq!(a.tasks_with_scored_pairs, 2);
        assert_eq!(a.mean_progress_delta, Some(0.25));
        assert_eq!(a.progress_delta_ci95, Some((0.25, 0.25)));
        assert_eq!(a.mean_elapsed_seconds_delta, -1.0);
        assert!((a.mean_cost_usd_delta + 0.20).abs() < 1e-12);
    }

    #[test]
    fn duplicate_trials_are_rejected() {
        let x = trial("task-a", "baseline", "same-id", Some(0.25), 1.0);
        assert_eq!(summarize_trials(&[x.clone(), x], 0, 1), Err(MetricError::DuplicateTrial));
    }

    #[test]
    fn unpaired_baseline_and_candidate_are_rejected() {
        let base = vec![trial("task-a", "baseline", "seed-1", Some(0.25), 1.0)];
        let candidate = vec![trial("task-a", "sidecar", "seed-2", Some(0.50), 1.0)];
        assert_eq!(
            compare_paired_trials(&base, &candidate, 0, 1),
            Err(MetricError::MismatchedPairedTrials)
        );
    }

    #[test]
    fn invalid_trial_must_not_claim_a_grade_or_completion() {
        let mut bad = trial("task-a", "baseline", "bad", None, 1.0);
        bad.completed = true;
        assert_eq!(bad.validate(), Err(MetricError::CompletionScoreMismatch));
        let mut bad_score = trial("task-a", "baseline", "bad-score", Some(1.2), 1.0);
        bad_score.completed = true;
        assert_eq!(bad_score.validate(), Err(MetricError::InvalidProgress));
    }

    #[test]
    fn task_summaries_do_not_silently_drop_all_invalid_trials() {
        let mut x = trial("store-safe", "baseline", "t1", None, 12.0);
        x.timed_out = true;
        let s = summarize_trials(&[x], 0, 1).unwrap();
        assert_eq!(s[0].valid_trials, 0);
        assert_eq!(s[0].invalid_trials, 1);
        assert_eq!(s[0].mean_progress, None);
        assert_eq!(s[0].progress_ci95, None);
        assert_eq!(s[0].timeout_count, 1);
        assert_eq!(s[0].completion_rate_all_trials, 0.0);
    }

}
