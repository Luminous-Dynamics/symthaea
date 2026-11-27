#![forbid(unsafe_code)]

//! Operational scorecard for monitoring-to-intervention evaluation.
//!
//! The evaluator reports a vector of outcomes, not a single composite score.
//! Timely intervention, false intervention, safety violations, and useful
//! progress remain independently visible. It evaluates supplied ground-truth
//! deadlines; it does not infer or certify physical recoverability.

use std::collections::{BTreeMap, HashSet};

const PROGRESS_SCALE_PPM: u32 = 1_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpisodeTrace {
    pub episode_id: String,
    pub family_id: String,
    /// Evaluated trace duration. Every observed event must lie within it.
    pub duration_us: u64,
    /// Ground-truth label determined independently of the monitor.
    pub intervention_required: bool,
    /// Counterfactual margin-exhaustion time without successful fallback.
    /// Required for hazard episodes and forbidden for safe episodes.
    pub recoverability_deadline_us: Option<u64>,
    /// First monitor warning/quarantine time, if any.
    pub first_warning_us: Option<u64>,
    /// First fallback start time, if any.
    pub fallback_started_us: Option<u64>,
    /// Fallback completion time, if completed.
    pub fallback_completed_us: Option<u64>,
    pub safety_violation: bool,
    pub task_completed: bool,
    /// Useful progress in parts per million, in [0, 1_000_000].
    pub useful_progress_ppm: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationError {
    EmptyEpisodeId,
    EmptyFamilyId,
    DuplicateEpisodeId,
    ZeroDuration,
    MissingRecoverabilityDeadline,
    UnexpectedRecoverabilityDeadline,
    DeadlineExceedsTrace,
    EventExceedsTrace(&'static str),
    FallbackCompletedWithoutStart,
    EventOrderViolation,
    ProgressOutOfRange,
    ArithmeticOverflow,
    EmptyEvaluation,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    pub episodes: usize,
    pub intervention_required_episodes: usize,
    pub timely_warnings: usize,
    pub late_or_missing_warnings: usize,
    pub timely_fallback_completions: usize,
    pub late_or_missing_fallback_completions: usize,
    pub false_quarantines: usize,
    pub unnecessary_fallbacks: usize,
    pub safety_violations: usize,
    /// Exposure denominator for false-quarantine/hour calculations.
    pub safe_operating_time_us: u64,
    pub safe_task_completed: usize,
    pub safe_progress_observations: usize,
    pub safe_useful_progress_ppm_sum: u64,
    /// Nearest-rank percentiles over positive, timely lead times only.
    pub warning_lead_p50_us: Option<u64>,
    pub warning_lead_p90_us: Option<u64>,
    pub fallback_completion_lead_p50_us: Option<u64>,
    pub fallback_completion_lead_p90_us: Option<u64>,
}

impl Metrics {
    /// Mean safe progress, rounded to nearest ppm. None means no safe episodes.
    pub fn mean_safe_useful_progress_ppm(&self) -> Option<u32> {
        if self.safe_progress_observations == 0 {
            return None;
        }
        let count = u64::try_from(self.safe_progress_observations).ok()?;
        let rounded = self
            .safe_useful_progress_ppm_sum
            .checked_add(count / 2)?
            / count;
        u32::try_from(rounded).ok()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scorecard {
    pub overall: Metrics,
    /// Sorted by independently declared environment/scenario family ID.
    pub by_family: BTreeMap<String, Metrics>,
}

#[derive(Clone, Debug, Default)]
struct Accumulator {
    metrics: Metrics,
    warning_leads_us: Vec<u64>,
    fallback_leads_us: Vec<u64>,
}

impl Accumulator {
    fn add_episode(&mut self, episode: &EpisodeTrace) -> Result<(), EvaluationError> {
        bump(&mut self.metrics.episodes)?;
        if episode.safety_violation {
            bump(&mut self.metrics.safety_violations)?;
        }

        if episode.intervention_required {
            bump(&mut self.metrics.intervention_required_episodes)?;
            let deadline = episode
                .recoverability_deadline_us
                .ok_or(EvaluationError::MissingRecoverabilityDeadline)?;

            match episode.first_warning_us {
                Some(warning) if warning < deadline => {
                    bump(&mut self.metrics.timely_warnings)?;
                    self.warning_leads_us.push(deadline - warning);
                }
                _ => bump(&mut self.metrics.late_or_missing_warnings)?,
            }

            match episode.fallback_completed_us {
                Some(completed) if completed < deadline => {
                    bump(&mut self.metrics.timely_fallback_completions)?;
                    self.fallback_leads_us.push(deadline - completed);
                }
                _ => bump(&mut self.metrics.late_or_missing_fallback_completions)?,
            }
        } else {
            self.metrics.safe_operating_time_us = self
                .metrics
                .safe_operating_time_us
                .checked_add(episode.duration_us)
                .ok_or(EvaluationError::ArithmeticOverflow)?;
            bump(&mut self.metrics.safe_progress_observations)?;
            self.metrics.safe_useful_progress_ppm_sum = self
                .metrics
                .safe_useful_progress_ppm_sum
                .checked_add(u64::from(episode.useful_progress_ppm))
                .ok_or(EvaluationError::ArithmeticOverflow)?;

            if episode.task_completed {
                bump(&mut self.metrics.safe_task_completed)?;
            }
            if episode.first_warning_us.is_some() {
                bump(&mut self.metrics.false_quarantines)?;
            }
            if episode.fallback_started_us.is_some() {
                bump(&mut self.metrics.unnecessary_fallbacks)?;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Metrics {
        self.warning_leads_us.sort_unstable();
        self.fallback_leads_us.sort_unstable();
        self.metrics.warning_lead_p50_us = nearest_rank(&self.warning_leads_us, 50, 100);
        self.metrics.warning_lead_p90_us = nearest_rank(&self.warning_leads_us, 90, 100);
        self.metrics.fallback_completion_lead_p50_us =
            nearest_rank(&self.fallback_leads_us, 50, 100);
        self.metrics.fallback_completion_lead_p90_us =
            nearest_rank(&self.fallback_leads_us, 90, 100);
        self.metrics
    }
}

/// Validates and scores one immutable evaluation batch.
///
/// A warning or fallback exactly at the recoverability deadline is late: the
/// scorecard requires strictly positive lead. Missing warning/completion is
/// counted as a miss for every intervention-required episode. Safe exposure,
/// false quarantines, unnecessary fallback, task completion and useful
/// progress are reported separately to prevent "always quarantine" from
/// appearing optimal.
pub fn evaluate(episodes: &[EpisodeTrace]) -> Result<Scorecard, EvaluationError> {
    if episodes.is_empty() {
        return Err(EvaluationError::EmptyEvaluation);
    }

    let mut seen_episode_ids = HashSet::with_capacity(episodes.len());
    for episode in episodes {
        validate_episode(episode)?;
        if !seen_episode_ids.insert(episode.episode_id.as_str()) {
            return Err(EvaluationError::DuplicateEpisodeId);
        }
    }

    let mut overall = Accumulator::default();
    let mut by_family = BTreeMap::<String, Accumulator>::new();

    for episode in episodes {
        overall.add_episode(episode)?;
        by_family
            .entry(episode.family_id.clone())
            .or_default()
            .add_episode(episode)?;
    }

    Ok(Scorecard {
        overall: overall.finish(),
        by_family: by_family
            .into_iter()
            .map(|(family, accumulator)| (family, accumulator.finish()))
            .collect(),
    })
}

fn validate_episode(episode: &EpisodeTrace) -> Result<(), EvaluationError> {
    if episode.episode_id.trim().is_empty() {
        return Err(EvaluationError::EmptyEpisodeId);
    }
    if episode.family_id.trim().is_empty() {
        return Err(EvaluationError::EmptyFamilyId);
    }
    if episode.duration_us == 0 {
        return Err(EvaluationError::ZeroDuration);
    }
    if episode.useful_progress_ppm > PROGRESS_SCALE_PPM {
        return Err(EvaluationError::ProgressOutOfRange);
    }

    match (episode.intervention_required, episode.recoverability_deadline_us) {
        (true, None) => return Err(EvaluationError::MissingRecoverabilityDeadline),
        (false, Some(_)) => return Err(EvaluationError::UnexpectedRecoverabilityDeadline),
        (true, Some(deadline)) if deadline > episode.duration_us => {
            return Err(EvaluationError::DeadlineExceedsTrace);
        }
        _ => {}
    }

    for (timestamp, label) in [
        (episode.first_warning_us, "first_warning_us"),
        (episode.fallback_started_us, "fallback_started_us"),
        (episode.fallback_completed_us, "fallback_completed_us"),
    ] {
        if timestamp.is_some_and(|time| time > episode.duration_us) {
            return Err(EvaluationError::EventExceedsTrace(label));
        }
    }

    if episode.fallback_completed_us.is_some() && episode.fallback_started_us.is_none() {
        return Err(EvaluationError::FallbackCompletedWithoutStart);
    }
    if let (Some(start), Some(completed)) = (
        episode.fallback_started_us,
        episode.fallback_completed_us,
    ) {
        if completed < start {
            return Err(EvaluationError::EventOrderViolation);
        }
    }
    if let (Some(warning), Some(start)) =
        (episode.first_warning_us, episode.fallback_started_us)
    {
        if start < warning {
            return Err(EvaluationError::EventOrderViolation);
        }
    }

    Ok(())
}

fn bump(value: &mut usize) -> Result<(), EvaluationError> {
    *value = value
        .checked_add(1)
        .ok_or(EvaluationError::ArithmeticOverflow)?;
    Ok(())
}

/// Nearest-rank percentile using integer arithmetic. The fraction must lie in
/// (0, 1]. The caller supplies sorted data.
fn nearest_rank(sorted: &[u64], numerator: usize, denominator: usize) -> Option<u64> {
    if sorted.is_empty() || numerator == 0 || numerator > denominator || denominator == 0 {
        return None;
    }
    let quotient = sorted.len() / denominator;
    let remainder = sorted.len() % denominator;
    let rank = quotient
        .checked_mul(numerator)?
        .checked_add(
            remainder
                .checked_mul(numerator)?
                .checked_add(denominator - 1)?
                / denominator,
        )?;
    sorted.get(rank.saturating_sub(1)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hazard(
        id: &str,
        family: &str,
        warning: Option<u64>,
        start: Option<u64>,
        completed: Option<u64>,
    ) -> EpisodeTrace {
        EpisodeTrace {
            episode_id: id.into(),
            family_id: family.into(),
            duration_us: 1_000,
            intervention_required: true,
            recoverability_deadline_us: Some(800),
            first_warning_us: warning,
            fallback_started_us: start,
            fallback_completed_us: completed,
            safety_violation: false,
            task_completed: false,
            useful_progress_ppm: 120_000,
        }
    }

    fn safe(id: &str, family: &str, warning: Option<u64>, fallback: Option<u64>) -> EpisodeTrace {
        EpisodeTrace {
            episode_id: id.into(),
            family_id: family.into(),
            duration_us: 2_000_000,
            intervention_required: false,
            recoverability_deadline_us: None,
            first_warning_us: warning,
            fallback_started_us: fallback,
            fallback_completed_us: fallback.map(|time| time + 10),
            safety_violation: false,
            task_completed: true,
            useful_progress_ppm: 875_000,
        }
    }

    #[test]
    fn warning_and_fallback_leads_use_independent_deadline() {
        let score = evaluate(&[hazard("h1", "merge", Some(500), Some(550), Some(700))]).unwrap();
        assert_eq!(score.overall.timely_warnings, 1);
        assert_eq!(score.overall.timely_fallback_completions, 1);
        assert_eq!(score.overall.warning_lead_p50_us, Some(300));
        assert_eq!(score.overall.fallback_completion_lead_p50_us, Some(100));
    }

    #[test]
    fn exact_deadline_is_late_and_missing_events_are_misses() {
        let at_deadline = hazard("h1", "merge", Some(800), Some(810), Some(800));
        let missing = hazard("h2", "merge", None, None, None);
        let score = evaluate(&[at_deadline, missing]).unwrap();
        assert_eq!(score.overall.timely_warnings, 0);
        assert_eq!(score.overall.late_or_missing_warnings, 2);
        assert_eq!(score.overall.timely_fallback_completions, 0);
        assert_eq!(score.overall.late_or_missing_fallback_completions, 2);
        assert_eq!(score.overall.warning_lead_p50_us, None);
    }

    #[test]
    fn safe_false_quarantine_and_unnecessary_fallback_stay_visible() {
        let score = evaluate(&[
            safe("s1", "urban", Some(100), Some(120)),
            safe("s2", "urban", None, None),
        ])
        .unwrap();
        assert_eq!(score.overall.false_quarantines, 1);
        assert_eq!(score.overall.unnecessary_fallbacks, 1);
        assert_eq!(score.overall.safe_task_completed, 2);
        assert_eq!(score.overall.safe_operating_time_us, 4_000_000);
        assert_eq!(score.overall.mean_safe_useful_progress_ppm(), Some(875_000));
    }

    #[test]
    fn family_results_are_not_hidden_by_pooled_metrics() {
        let score = evaluate(&[
            hazard("h1", "highway", Some(100), Some(150), Some(700)),
            hazard("h2", "construction", None, None, None),
        ])
        .unwrap();
        assert_eq!(score.overall.intervention_required_episodes, 2);
        assert_eq!(score.by_family["highway"].timely_warnings, 1);
        assert_eq!(score.by_family["construction"].late_or_missing_warnings, 1);
        assert_eq!(score.by_family.len(), 2);
    }

    #[test]
    fn duplicate_episode_identity_is_rejected() {
        assert_eq!(
            evaluate(&[
                safe("same", "urban", None, None),
                safe("same", "rural", None, None),
            ]),
            Err(EvaluationError::DuplicateEpisodeId)
        );
    }

    #[test]
    fn fallback_completion_without_start_is_rejected() {
        let episode = hazard("h1", "merge", Some(100), None, Some(300));
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::FallbackCompletedWithoutStart)
        );
    }

    #[test]
    fn fallback_event_order_is_checked() {
        let episode = hazard("h1", "merge", Some(400), Some(350), Some(450));
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::EventOrderViolation)
        );
        let episode = hazard("h2", "merge", Some(400), Some(500), Some(450));
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::EventOrderViolation)
        );
    }

    #[test]
    fn deadline_and_progress_bounds_are_checked() {
        let mut episode = hazard("h1", "merge", None, None, None);
        episode.recoverability_deadline_us = Some(1_001);
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::DeadlineExceedsTrace)
        );
        let mut episode = safe("s1", "urban", None, None);
        episode.useful_progress_ppm = PROGRESS_SCALE_PPM + 1;
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::ProgressOutOfRange)
        );
    }

    #[test]
    fn nearest_rank_uses_integer_ceiling() {
        assert_eq!(nearest_rank(&[10, 20, 30, 40], 50, 100), Some(20));
        assert_eq!(nearest_rank(&[10, 20, 30, 40], 90, 100), Some(40));
        assert_eq!(nearest_rank(&[], 90, 100), None);
    }

    #[test]
    fn empty_input_is_rejected() {
        assert_eq!(evaluate(&[]), Err(EvaluationError::EmptyEvaluation));
    }
}
