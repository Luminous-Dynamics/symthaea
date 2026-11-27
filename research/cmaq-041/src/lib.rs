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
pub struct ExposureInterval {
    /// Asset or simulator timeline whose operating exposure is measured.
    pub domain_id: String,
    /// Absolute monotonic interval within that domain.
    pub start_us: u64,
    pub end_us: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceBindings {
    pub source_trace_sha256: String,
    pub scenario_manifest_sha256: String,
    pub ground_truth_receipt_sha256: String,
    pub system_build_sha256: String,
    pub monitor_build_sha256: String,
    pub fallback_policy_sha256: String,
    pub score_definition_sha256: String,
    pub score_receipt_sha256: String,
    pub closure_manifest_sha256: String,
    pub closure_authority_receipt_sha256: String,
    /// CMAQ-036 effective block identity; repeated IDs mark correlated episodes.
    pub effective_block_id: String,
}

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
    /// First monitor warning time, if any.
    pub first_warning_us: Option<u64>,
    /// Time the system actually entered its quarantine state, if any.
    pub quarantined_us: Option<u64>,
    /// First fallback start time, if any.
    pub fallback_started_us: Option<u64>,
    /// Fallback completion time, if completed.
    pub fallback_completed_us: Option<u64>,
    pub safety_violation: bool,
    pub task_completed: bool,
    /// Useful progress in parts per million, in [0, 1_000_000].
    pub useful_progress_ppm: u32,
    /// Required for safe episodes; hazard episodes add no safe exposure.
    pub safe_exposure_interval: Option<ExposureInterval>,
    pub evidence: EvidenceBindings,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvaluationError {
    EmptyEpisodeId,
    EmptyFamilyId,
    DuplicateEpisodeId,
    InvalidEvidenceDigest(&'static str),
    EmptyEffectiveBlockId,
    ZeroDuration,
    MissingRecoverabilityDeadline,
    UnexpectedRecoverabilityDeadline,
    DeadlineExceedsTrace,
    EventExceedsTrace(&'static str),
    FallbackCompletedWithoutStart,
    EventOrderViolation,
    ProgressOutOfRange,
    MissingSafeExposureInterval,
    UnexpectedSafeExposureInterval,
    InvalidSafeExposureInterval,
    ArithmeticOverflow,
    EmptyEvaluation,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    pub episodes: usize,
    /// Number of unique CMAQ-036 effective block IDs, not raw episode count.
    pub effective_block_count: usize,
    pub intervention_required_episodes: usize,
    pub timely_warnings: usize,
    pub late_or_missing_warnings: usize,
    pub timely_quarantines: usize,
    pub late_or_missing_quarantines: usize,
    pub timely_fallback_completions: usize,
    pub late_or_missing_fallback_completions: usize,
    pub false_warnings: usize,
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
    pub quarantine_lead_p50_us: Option<u64>,
    pub quarantine_lead_p90_us: Option<u64>,
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
        let rounded = self.safe_useful_progress_ppm_sum.checked_add(count / 2)? / count;
        u32::try_from(rounded).ok()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scorecard {
    pub overall: Metrics,
    /// Sorted by independently declared environment/scenario family ID.
    pub by_family: BTreeMap<String, Metrics>,
    /// One row per CMAQ-036 effective dependency block for clustered analysis.
    pub by_effective_block: BTreeMap<String, Metrics>,
}

#[derive(Clone, Debug, Default)]
struct Accumulator {
    metrics: Metrics,
    effective_block_ids: HashSet<String>,
    safe_exposure_intervals: Vec<ExposureInterval>,
    warning_leads_us: Vec<u64>,
    quarantine_leads_us: Vec<u64>,
    fallback_leads_us: Vec<u64>,
}

impl Accumulator {
    fn add_episode(&mut self, episode: &EpisodeTrace) -> Result<(), EvaluationError> {
        bump(&mut self.metrics.episodes)?;
        self.effective_block_ids
            .insert(episode.evidence.effective_block_id.clone());
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

            match episode.quarantined_us {
                Some(quarantined) if quarantined < deadline => {
                    bump(&mut self.metrics.timely_quarantines)?;
                    self.quarantine_leads_us.push(deadline - quarantined);
                }
                _ => bump(&mut self.metrics.late_or_missing_quarantines)?,
            }

            match episode.fallback_completed_us {
                Some(completed) if completed < deadline => {
                    bump(&mut self.metrics.timely_fallback_completions)?;
                    self.fallback_leads_us.push(deadline - completed);
                }
                _ => bump(&mut self.metrics.late_or_missing_fallback_completions)?,
            }
        } else {
            let exposure = episode
                .safe_exposure_interval
                .as_ref()
                .ok_or(EvaluationError::MissingSafeExposureInterval)?;
            self.safe_exposure_intervals.push(exposure.clone());
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
                bump(&mut self.metrics.false_warnings)?;
            }
            if episode.quarantined_us.is_some() {
                bump(&mut self.metrics.false_quarantines)?;
            }
            if episode.fallback_started_us.is_some() {
                bump(&mut self.metrics.unnecessary_fallbacks)?;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Metrics, EvaluationError> {
        self.metrics.effective_block_count = self.effective_block_ids.len();
        self.metrics.safe_operating_time_us = union_exposure_us(&mut self.safe_exposure_intervals)?;
        self.warning_leads_us.sort_unstable();
        self.quarantine_leads_us.sort_unstable();
        self.fallback_leads_us.sort_unstable();
        self.metrics.warning_lead_p50_us = nearest_rank(&self.warning_leads_us, 50, 100);
        self.metrics.warning_lead_p90_us = nearest_rank(&self.warning_leads_us, 90, 100);
        self.metrics.quarantine_lead_p50_us = nearest_rank(&self.quarantine_leads_us, 50, 100);
        self.metrics.quarantine_lead_p90_us = nearest_rank(&self.quarantine_leads_us, 90, 100);
        self.metrics.fallback_completion_lead_p50_us =
            nearest_rank(&self.fallback_leads_us, 50, 100);
        self.metrics.fallback_completion_lead_p90_us =
            nearest_rank(&self.fallback_leads_us, 90, 100);
        Ok(self.metrics)
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
    let mut by_effective_block = BTreeMap::<String, Accumulator>::new();

    for episode in episodes {
        overall.add_episode(episode)?;
        by_family
            .entry(episode.family_id.clone())
            .or_default()
            .add_episode(episode)?;
        by_effective_block
            .entry(episode.evidence.effective_block_id.clone())
            .or_default()
            .add_episode(episode)?;
    }

    let overall = overall.finish()?;
    let by_family = by_family
        .into_iter()
        .map(|(family, accumulator)| Ok((family, accumulator.finish()?)))
        .collect::<Result<BTreeMap<_, _>, EvaluationError>>()?;
    let by_effective_block = by_effective_block
        .into_iter()
        .map(|(block_id, accumulator)| Ok((block_id, accumulator.finish()?)))
        .collect::<Result<BTreeMap<_, _>, EvaluationError>>()?;

    Ok(Scorecard {
        overall,
        by_family,
        by_effective_block,
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
    if episode.evidence.effective_block_id.trim().is_empty() {
        return Err(EvaluationError::EmptyEffectiveBlockId);
    }
    for (digest, label) in [
        (&episode.evidence.source_trace_sha256, "source_trace_sha256"),
        (
            &episode.evidence.scenario_manifest_sha256,
            "scenario_manifest_sha256",
        ),
        (
            &episode.evidence.ground_truth_receipt_sha256,
            "ground_truth_receipt_sha256",
        ),
        (&episode.evidence.system_build_sha256, "system_build_sha256"),
        (
            &episode.evidence.monitor_build_sha256,
            "monitor_build_sha256",
        ),
        (
            &episode.evidence.fallback_policy_sha256,
            "fallback_policy_sha256",
        ),
        (
            &episode.evidence.score_definition_sha256,
            "score_definition_sha256",
        ),
        (
            &episode.evidence.score_receipt_sha256,
            "score_receipt_sha256",
        ),
        (
            &episode.evidence.closure_manifest_sha256,
            "closure_manifest_sha256",
        ),
        (
            &episode.evidence.closure_authority_receipt_sha256,
            "closure_authority_receipt_sha256",
        ),
    ] {
        if !is_sha256(digest) {
            return Err(EvaluationError::InvalidEvidenceDigest(label));
        }
    }

    match (
        episode.intervention_required,
        episode.recoverability_deadline_us,
    ) {
        (true, None) => return Err(EvaluationError::MissingRecoverabilityDeadline),
        (false, Some(_)) => return Err(EvaluationError::UnexpectedRecoverabilityDeadline),
        (true, Some(deadline)) if deadline > episode.duration_us => {
            return Err(EvaluationError::DeadlineExceedsTrace);
        }
        _ => {}
    }

    match (
        episode.intervention_required,
        episode.safe_exposure_interval.as_ref(),
    ) {
        (true, Some(_)) => return Err(EvaluationError::UnexpectedSafeExposureInterval),
        (false, None) => return Err(EvaluationError::MissingSafeExposureInterval),
        (false, Some(interval)) => {
            if interval.domain_id.trim().is_empty()
                || interval.end_us <= interval.start_us
                || interval.end_us - interval.start_us != episode.duration_us
            {
                return Err(EvaluationError::InvalidSafeExposureInterval);
            }
        }
        (true, None) => {}
    }

    for (timestamp, label) in [
        (episode.first_warning_us, "first_warning_us"),
        (episode.quarantined_us, "quarantined_us"),
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
    if let (Some(start), Some(completed)) =
        (episode.fallback_started_us, episode.fallback_completed_us)
    {
        if completed < start {
            return Err(EvaluationError::EventOrderViolation);
        }
    }
    if let (Some(warning), Some(quarantined)) = (episode.first_warning_us, episode.quarantined_us) {
        if quarantined < warning {
            return Err(EvaluationError::EventOrderViolation);
        }
    }
    if let (Some(quarantined), Some(start)) = (episode.quarantined_us, episode.fallback_started_us)
    {
        if start < quarantined {
            return Err(EvaluationError::EventOrderViolation);
        }
    }
    if let (Some(warning), Some(start)) = (episode.first_warning_us, episode.fallback_started_us) {
        if start < warning {
            return Err(EvaluationError::EventOrderViolation);
        }
    }

    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn union_exposure_us(intervals: &mut [ExposureInterval]) -> Result<u64, EvaluationError> {
    intervals.sort_by(|left, right| {
        left.domain_id
            .cmp(&right.domain_id)
            .then_with(|| left.start_us.cmp(&right.start_us))
            .then_with(|| left.end_us.cmp(&right.end_us))
    });

    let mut total = 0u64;
    let mut current: Option<(String, u64, u64)> = None;
    for interval in intervals {
        let overlaps = current.as_ref().is_some_and(|(domain, _, end)| {
            domain == &interval.domain_id && interval.start_us <= *end
        });
        if overlaps {
            if let Some((_, _, end)) = current.as_mut() {
                *end = (*end).max(interval.end_us);
            }
            continue;
        }

        if let Some((_, start, end)) = current.take() {
            total = total
                .checked_add(end - start)
                .ok_or(EvaluationError::ArithmeticOverflow)?;
        }
        current = Some((
            interval.domain_id.clone(),
            interval.start_us,
            interval.end_us,
        ));
    }
    if let Some((_, start, end)) = current {
        total = total
            .checked_add(end - start)
            .ok_or(EvaluationError::ArithmeticOverflow)?;
    }
    Ok(total)
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
    let rank = quotient.checked_mul(numerator)?.checked_add(
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

    fn evidence_for(id: &str) -> EvidenceBindings {
        let digest = "a".repeat(64);
        EvidenceBindings {
            source_trace_sha256: digest.clone(),
            scenario_manifest_sha256: digest.clone(),
            ground_truth_receipt_sha256: digest.clone(),
            system_build_sha256: digest.clone(),
            monitor_build_sha256: digest.clone(),
            fallback_policy_sha256: digest.clone(),
            score_definition_sha256: digest.clone(),
            score_receipt_sha256: digest.clone(),
            closure_manifest_sha256: digest.clone(),
            closure_authority_receipt_sha256: digest,
            effective_block_id: format!("block-{id}"),
        }
    }

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
            quarantined_us: warning,
            fallback_started_us: start,
            fallback_completed_us: completed,
            safety_violation: false,
            task_completed: false,
            useful_progress_ppm: 120_000,
            safe_exposure_interval: None,
            evidence: evidence_for(id),
        }
    }

    fn safe(
        id: &str,
        family: &str,
        warning: Option<u64>,
        quarantine: Option<u64>,
        fallback: Option<u64>,
    ) -> EpisodeTrace {
        EpisodeTrace {
            episode_id: id.into(),
            family_id: family.into(),
            duration_us: 2_000_000,
            intervention_required: false,
            recoverability_deadline_us: None,
            first_warning_us: warning,
            quarantined_us: quarantine,
            fallback_started_us: fallback,
            fallback_completed_us: fallback.map(|time| time + 10),
            safety_violation: false,
            task_completed: true,
            useful_progress_ppm: 875_000,
            safe_exposure_interval: Some(ExposureInterval {
                domain_id: "vehicle-1".into(),
                start_us: 0,
                end_us: 2_000_000,
            }),
            evidence: evidence_for(id),
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
        let at_deadline = hazard("h1", "merge", Some(800), Some(800), Some(800));
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
            safe("s1", "urban", Some(100), Some(110), Some(120)),
            safe("s2", "urban", None, None, None),
            safe("s3", "urban", Some(300), None, None),
        ])
        .unwrap();
        assert_eq!(score.overall.false_warnings, 2);
        assert_eq!(score.overall.false_quarantines, 1);
        assert_eq!(score.overall.unnecessary_fallbacks, 1);
        assert_eq!(score.overall.safe_task_completed, 3);
        // Three overlapping clips from one vehicle count as one interval.
        assert_eq!(score.overall.safe_operating_time_us, 2_000_000);
        assert_eq!(score.overall.mean_safe_useful_progress_ppm(), Some(875_000));
    }

    #[test]
    fn overlapping_and_adjacent_safe_windows_are_unioned_per_domain() {
        let first = safe("s1", "urban", None, None, None);
        let mut second = safe("s2", "urban", None, None, None);
        let mut third = safe("s3", "urban", None, None, None);
        second.safe_exposure_interval = Some(ExposureInterval {
            domain_id: "vehicle-1".into(),
            start_us: 1_000_000,
            end_us: 3_000_000,
        });
        third.safe_exposure_interval = Some(ExposureInterval {
            domain_id: "vehicle-1".into(),
            start_us: 3_000_000,
            end_us: 5_000_000,
        });
        let score = evaluate(&[first, second, third]).unwrap();
        assert_eq!(score.overall.safe_operating_time_us, 5_000_000);
    }

    #[test]
    fn safe_exposure_interval_is_required_and_validated() {
        let mut episode = safe("s1", "urban", None, None, None);
        episode.safe_exposure_interval = None;
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::MissingSafeExposureInterval)
        );

        let mut episode = safe("s2", "urban", None, None, None);
        episode.safe_exposure_interval.as_mut().unwrap().end_us = 1_000_000;
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::InvalidSafeExposureInterval)
        );

        let mut episode = hazard("h1", "merge", None, None, None);
        episode.safe_exposure_interval = Some(ExposureInterval {
            domain_id: "vehicle-1".into(),
            start_us: 0,
            end_us: 1_000,
        });
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::UnexpectedSafeExposureInterval)
        );
    }

    #[test]
    fn family_results_are_not_hidden_by_pooled_metrics() {
        let score = evaluate(&[
            hazard("h1", "highway", Some(100), Some(150), Some(700)),
            hazard("h2", "construction", None, None, None),
        ])
        .unwrap();
        assert_eq!(score.overall.intervention_required_episodes, 2);
        assert_eq!(score.overall.effective_block_count, 2);
        assert_eq!(score.by_family["highway"].timely_warnings, 1);
        assert_eq!(score.by_family["construction"].late_or_missing_warnings, 1);
        assert_eq!(score.by_family.len(), 2);
        assert_eq!(score.by_effective_block["block-h1"].episodes, 1);
        assert_eq!(score.by_effective_block["block-h2"].episodes, 1);
    }

    #[test]
    fn effective_block_ids_are_counted_once_even_with_many_episodes() {
        let mut first = hazard("h1", "merge", Some(500), Some(550), Some(700));
        let mut second = hazard("h2", "merge", Some(510), Some(560), Some(710));
        first.evidence.effective_block_id = "shared-drive-session".into();
        second.evidence.effective_block_id = "shared-drive-session".into();
        let score = evaluate(&[first, second]).unwrap();
        assert_eq!(score.overall.episodes, 2);
        assert_eq!(score.overall.effective_block_count, 1);
        assert_eq!(score.by_effective_block["shared-drive-session"].episodes, 2);
    }

    #[test]
    fn malformed_evidence_digest_is_rejected() {
        let mut episode = safe("s1", "urban", None, None, None);
        episode.evidence.source_trace_sha256 = "not-a-sha256".into();
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::InvalidEvidenceDigest(
                "source_trace_sha256"
            ))
        );
    }

    #[test]
    fn empty_effective_block_id_is_rejected() {
        let mut episode = safe("s1", "urban", None, None, None);
        episode.evidence.effective_block_id = "   ".into();
        assert_eq!(
            evaluate(&[episode]),
            Err(EvaluationError::EmptyEffectiveBlockId)
        );
    }

    #[test]
    fn duplicate_episode_identity_is_rejected() {
        assert_eq!(
            evaluate(&[
                safe("same", "urban", None, None, None),
                safe("same", "rural", None, None, None),
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
        let mut episode = safe("s1", "urban", None, None, None);
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
