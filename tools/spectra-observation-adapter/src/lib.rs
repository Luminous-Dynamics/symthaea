// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Candidate-side observation contract and type-1 ROC adapters for SPECTRA.
//!
//! This module preserves binary decisions, sequential ordinal confidence and
//! response latency separately. It composes shared ROC/AUC and signal-detection
//! metrics; it does not implement unequal-variance SDT parameter fitting.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_statistics::{
    RocCurve, RocObservation, SignalDetectionCounts, SignalDetectionMetrics, try_roc_curve,
};

/// Predeclared counts for the two independent auxiliary-observation routes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservationProfile {
    /// Ordinal confidence levels, ordered from low to high confidence in the
    /// chosen response. Must be at least three.
    pub confidence_levels: u8,
    /// Per-participant RT quantile bins. Must be at least three.
    pub rt_bins: u8,
}

impl ObservationProfile {
    pub fn validate(self) -> Result<Self, ObservationError> {
        if !(3..=10).contains(&self.confidence_levels)
            || !(3..=10).contains(&self.rt_bins)
        {
            return Err(ObservationError::InvalidProfile);
        }
        Ok(self)
    }
}

/// One candidate-visible trial. It contains no event-present/event-absent
/// truth label, injected model parameter, or hidden campaign identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateTrial {
    pub trial_id: u64,
    pub participant_id: u32,
    pub session_id: u32,
    /// Monotonic timestamp from the declared clock domain.
    pub stimulus_onset_ns: u64,
    /// None represents a recorded missing decision or timeout.
    pub decision_yes: Option<bool>,
    /// Timestamp of the first committed yes/no decision.
    pub decision_time_ns: Option<u64>,
    /// Optional ordinal confidence in the chosen response, 1=low, K=high.
    pub confidence_level: Option<u8>,
    /// Must be recorded after the binary decision.
    pub confidence_time_ns: Option<u64>,
    /// Resolution of the clock used for onset and decision timestamps.
    pub clock_resolution_ns: u64,
    /// Independently bounded uncertainty in the physical stimulus-onset time.
    pub onset_uncertainty_ns: u64,
    /// Independently calibrated response-device/input latency uncertainty.
    pub input_latency_uncertainty_ns: u64,
}

/// Candidate-visible records. The observation profile is frozen separately
/// from per-trial values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateView {
    pub profile: ObservationProfile,
    pub trials: Vec<CandidateTrial>,
}

/// Evaluator-only event truth. Keep it outside the candidate process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatorTruth {
    trial_ids: Vec<u64>,
    event_present: Vec<bool>,
}

/// Explicit malformed-input dispositions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationError {
    InvalidProfile,
    EmptyCampaign,
    MisalignedTruth,
    DuplicateTrialId,
    PartialDecision,
    InvalidDecisionTimestamp,
    PartialConfidence,
    ConfidenceWithoutDecision,
    InvalidConfidenceTimestamp,
    ConfidenceOutOfRange,
    InvalidClockResolution,
    TimingBoundOverflow,
    MissingEventClass,
    StatisticFailure,
}

impl fmt::Display for ObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidProfile => "confidence levels and RT bins must each be between 3 and 10",
            Self::EmptyCampaign => "candidate campaign must contain at least one trial",
            Self::MisalignedTruth => "candidate rows do not align with evaluator truth",
            Self::DuplicateTrialId => "trial identifiers must be unique",
            Self::PartialDecision => "decision and decision timestamp must both be present or both be absent",
            Self::InvalidDecisionTimestamp => "decision timestamp must be strictly after stimulus onset",
            Self::PartialConfidence => "confidence level and confidence timestamp must both be present or both be absent",
            Self::ConfidenceWithoutDecision => "confidence requires a committed binary decision",
            Self::InvalidConfidenceTimestamp => "confidence timestamp must be strictly after the binary decision",
            Self::ConfidenceOutOfRange => "confidence level is outside the predeclared ordinal range",
            Self::InvalidClockResolution => "clock resolution must be non-zero",
            Self::TimingBoundOverflow => "combined RT timing uncertainty bound overflowed",
            Self::MissingEventClass => "evaluator truth must contain both event-present and event-absent trials",
            Self::StatisticFailure => "shared ROC or signal-detection API could not evaluate observations",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ObservationError {}

/// Descriptive adapter result. A missing curve means its route did not
/// contain both event classes with usable observations; it is not a zero AUC.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationReport {
    pub total_trials: usize,
    pub decision_trials: usize,
    pub missing_decisions: usize,
    pub confidence_trials: usize,
    pub missing_confidence_after_decision: usize,
    pub rt_trials: usize,
    pub rt_usable_trials: usize,
    pub rt_resolution_limited_trials: usize,
    pub missing_rt_after_decision: usize,
    pub distinct_confidence_categories: usize,
    pub distinct_rt_categories: usize,
    pub decision_counts: SignalDetectionCounts,
    pub binary_signal_detection: Option<SignalDetectionMetrics>,
    pub confidence_type1_roc: Option<RocCurve>,
    pub rt_type1_roc: Option<RocCurve>,
    pub confidence_roc_auc: Option<f64>,
    pub rt_roc_auc: Option<f64>,
}

/// Evaluate candidate-visible records against evaluator-only event truth.
///
/// Confidence scores preserve only ordinal ordering; category spacing is not
/// treated as interval-scaled. RT categories are per-participant quantiles,
/// with equal latencies grouped into a shared mid-rank bin.
pub fn evaluate(
    candidate: &CandidateView,
    truth: &EvaluatorTruth,
) -> Result<ObservationReport, ObservationError> {
    let profile = candidate.profile.validate()?;
    let total = candidate.trials.len();
    if total == 0 {
        return Err(ObservationError::EmptyCampaign);
    }
    if truth.trial_ids.len() != total || truth.event_present.len() != total {
        return Err(ObservationError::MisalignedTruth);
    }

    let mut seen_candidate = BTreeSet::new();
    let mut seen_truth = BTreeSet::new();
    for (index, row) in candidate.trials.iter().enumerate() {
        if !seen_candidate.insert(row.trial_id) || !seen_truth.insert(truth.trial_ids[index]) {
            return Err(ObservationError::DuplicateTrialId);
        }
        if row.trial_id != truth.trial_ids[index] {
            return Err(ObservationError::MisalignedTruth);
        }
        validate_trial(row, profile)?;
    }

    let signal_count = truth.event_present.iter().filter(|value| **value).count();
    let noise_count = total - signal_count;
    if signal_count == 0 || noise_count == 0 {
        return Err(ObservationError::MissingEventClass);
    }

    let mut decision_counts = SignalDetectionCounts::default();
    let mut decision_trials = 0usize;
    let mut confidence_trials = 0usize;
    let mut rt_trials = 0usize;
    let mut confidence_observations = Vec::new();
    let mut confidence_categories = BTreeSet::new();

    for (row, is_signal) in candidate.trials.iter().zip(truth.event_present.iter().copied()) {
        let Some(decision_yes) = row.decision_yes else {
            continue;
        };
        decision_trials += 1;

        match (is_signal, decision_yes) {
            (true, true) => decision_counts.hits += 1,
            (true, false) => decision_counts.misses += 1,
            (false, true) => decision_counts.false_alarms += 1,
            (false, false) => decision_counts.correct_rejections += 1,
        }

        if let Some(level) = row.confidence_level {
            confidence_trials += 1;
            let score = confidence_score(decision_yes, level, profile.confidence_levels);
            confidence_categories.insert(score);
            confidence_observations.push(RocObservation {
                score: score as f64,
                is_signal,
            });
        }

        if row.decision_time_ns.is_some() {
            rt_trials += 1;
        }
    }

    let (rt_observations, rt_usable_trials, rt_resolution_limited_trials) =
        build_rt_observations(candidate, truth, profile)?;
    let rt_categories: BTreeSet<_> = rt_observations
        .iter()
        .map(|observation| observation.score.to_bits())
        .collect();

    let confidence_curve = optional_roc(&confidence_observations)?;
    let rt_curve = optional_roc(&rt_observations)?;
    let binary_signal_detection = decision_counts.metrics();

    Ok(ObservationReport {
        total_trials: total,
        decision_trials,
        missing_decisions: total - decision_trials,
        confidence_trials,
        missing_confidence_after_decision: decision_trials - confidence_trials,
        rt_trials,
        rt_usable_trials,
        rt_resolution_limited_trials,
        missing_rt_after_decision: decision_trials - rt_trials,
        distinct_confidence_categories: confidence_categories.len(),
        distinct_rt_categories: rt_categories.len(),
        decision_counts,
        binary_signal_detection,
        confidence_roc_auc: confidence_curve.as_ref().map(|curve| curve.auc),
        rt_roc_auc: rt_curve.as_ref().map(|curve| curve.auc),
        confidence_type1_roc: confidence_curve,
        rt_type1_roc: rt_curve,
    })
}

fn validate_trial(
    row: &CandidateTrial,
    profile: ObservationProfile,
) -> Result<(), ObservationError> {
    if row.clock_resolution_ns == 0 {
        return Err(ObservationError::InvalidClockResolution);
    }

    if row.decision_yes.is_some() != row.decision_time_ns.is_some() {
        return Err(ObservationError::PartialDecision);
    }

    if let Some(decision_time) = row.decision_time_ns {
        if decision_time <= row.stimulus_onset_ns {
            return Err(ObservationError::InvalidDecisionTimestamp);
        }
    }

    if row.confidence_level.is_some() != row.confidence_time_ns.is_some() {
        return Err(ObservationError::PartialConfidence);
    }

    if let Some(level) = row.confidence_level {
        if row.decision_yes.is_none() {
            return Err(ObservationError::ConfidenceWithoutDecision);
        }
        if level == 0 || level > profile.confidence_levels {
            return Err(ObservationError::ConfidenceOutOfRange);
        }
        let decision_time = row
            .decision_time_ns
            .ok_or(ObservationError::ConfidenceWithoutDecision)?;
        let confidence_time = row
            .confidence_time_ns
            .ok_or(ObservationError::PartialConfidence)?;
        if confidence_time <= decision_time {
            return Err(ObservationError::InvalidConfidenceTimestamp);
        }
    }

    Ok(())
}

/// Map ordinal confidence into evidence categories:
/// high-confidence no < ... < low-confidence no < low-confidence yes
/// < ... < high-confidence yes. No equal-spacing assumption is made.
fn confidence_score(decision_yes: bool, level: u8, levels: u8) -> i16 {
    if decision_yes {
        i16::from(levels) + i16::from(level)
    } else {
        i16::from(levels) - i16::from(level)
    }
}

/// Construct RT scores by participant-specific latency quantiles. Fast yes
/// responses have the highest score; slow yes responses are next; slow no
/// responses are closer to the decision boundary; fast no responses have the
/// lowest score. Equal RTs receive the same bin.
fn build_rt_observations(
    candidate: &CandidateView,
    truth: &EvaluatorTruth,
    profile: ObservationProfile,
) -> Result<(Vec<RocObservation>, usize, usize), ObservationError> {
    let mut by_participant: BTreeMap<u32, Vec<(usize, u64)>> = BTreeMap::new();
    let mut rt_usable_trials = 0usize;
    let mut rt_resolution_limited_trials = 0usize;

    for (index, row) in candidate.trials.iter().enumerate() {
        let (Some(_decision_yes), Some(decision_time)) = (row.decision_yes, row.decision_time_ns)
        else {
            continue;
        };
        let latency = decision_time
            .checked_sub(row.stimulus_onset_ns)
            .ok_or(ObservationError::InvalidDecisionTimestamp)?;
        if latency == 0 {
            return Err(ObservationError::InvalidDecisionTimestamp);
        }
        let timing_bound = row.clock_resolution_ns
            .checked_mul(2)
            .and_then(|value| value.checked_add(row.onset_uncertainty_ns))
            .and_then(|value| value.checked_add(row.input_latency_uncertainty_ns))
            .ok_or(ObservationError::TimingBoundOverflow)?;
        if latency <= timing_bound {
            rt_resolution_limited_trials += 1;
            continue;
        }
        rt_usable_trials += 1;
        by_participant.entry(row.participant_id).or_default().push((index, latency));
    }

    let mut scores_by_index = vec![None::<i16>; candidate.trials.len()];
    for observations in by_participant.values_mut() {
        observations.sort_by_key(|(index, latency)| (*latency, *index));
        let n = observations.len();
        let mut start = 0usize;

        while start < n {
            let latency = observations[start].1;
            let mut end = start + 1;
            while end < n && observations[end].1 == latency {
                end += 1;
            }

            let midrank = (start + end - 1) as f64 / 2.0;
            let percentile = (midrank + 0.5) / n as f64;
            let bin = ((percentile * f64::from(profile.rt_bins)).floor() as u8 + 1)
                .clamp(1, profile.rt_bins);
            let magnitude = i16::from(profile.rt_bins + 1 - bin);

            for (index, _) in &observations[start..end] {
                let decision_yes = candidate.trials[*index]
                    .decision_yes
                    .ok_or(ObservationError::PartialDecision)?;
                scores_by_index[*index] = Some(if decision_yes { magnitude } else { -magnitude });
            }
            start = end;
        }
    }

    let mut result = Vec::new();
    for (index, score) in scores_by_index.into_iter().enumerate() {
        if let Some(score) = score {
            result.push(RocObservation {
                score: f64::from(score),
                is_signal: truth.event_present[index],
            });
        }
    }
    Ok((result, rt_usable_trials, rt_resolution_limited_trials))
}

fn optional_roc(
    observations: &[RocObservation],
) -> Result<Option<RocCurve>, ObservationError> {
    if observations.is_empty()
        || !observations.iter().any(|item| item.is_signal)
        || !observations.iter().any(|item| !item.is_signal)
    {
        return Ok(None);
    }
    try_roc_curve(observations)
        .map(Some)
        .map_err(|_| ObservationError::StatisticFailure)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> ObservationProfile {
        ObservationProfile { confidence_levels: 3, rt_bins: 3 }
    }

    fn perfect_fixture() -> (CandidateView, EvaluatorTruth) {
        let rows = vec![
            CandidateTrial { trial_id: 0, participant_id: 0, session_id: 0, stimulus_onset_ns: 1_000, decision_yes: Some(true), decision_time_ns: Some(1_100), confidence_level: Some(3), confidence_time_ns: Some(1_200), clock_resolution_ns: 1, onset_uncertainty_ns: 1, input_latency_uncertainty_ns: 1 },
            CandidateTrial { trial_id: 1, participant_id: 0, session_id: 0, stimulus_onset_ns: 2_000, decision_yes: Some(true), decision_time_ns: Some(2_200), confidence_level: Some(2), confidence_time_ns: Some(2_300), clock_resolution_ns: 1, onset_uncertainty_ns: 1 },
            CandidateTrial { trial_id: 2, participant_id: 0, session_id: 0, stimulus_onset_ns: 3_000, decision_yes: Some(true), decision_time_ns: Some(3_300), confidence_level: Some(1), confidence_time_ns: Some(3_400), clock_resolution_ns: 1, onset_uncertainty_ns: 1 },
            CandidateTrial { trial_id: 3, participant_id: 0, session_id: 0, stimulus_onset_ns: 4_000, decision_yes: Some(false), decision_time_ns: Some(4_300), confidence_level: Some(1), confidence_time_ns: Some(4_400), clock_resolution_ns: 1, onset_uncertainty_ns: 1 },
            CandidateTrial { trial_id: 4, participant_id: 0, session_id: 0, stimulus_onset_ns: 5_000, decision_yes: Some(false), decision_time_ns: Some(5_200), confidence_level: Some(2), confidence_time_ns: Some(5_300), clock_resolution_ns: 1, onset_uncertainty_ns: 1 },
            CandidateTrial { trial_id: 5, participant_id: 0, session_id: 0, stimulus_onset_ns: 6_000, decision_yes: Some(false), decision_time_ns: Some(6_100), confidence_level: Some(3), confidence_time_ns: Some(6_200), clock_resolution_ns: 1, onset_uncertainty_ns: 1 },
        ];
        let truth = EvaluatorTruth {
            trial_ids: (0..6).collect(),
            event_present: vec![true, true, true, false, false, false],
        };
        (CandidateView { profile: profile(), trials: rows }, truth)
    }

    #[test]
    fn confidence_and_rt_routes_recover_perfect_ordering() {
        let (candidate, truth) = perfect_fixture();
        let report = evaluate(&candidate, &truth).unwrap();
        assert_eq!(report.confidence_roc_auc, Some(1.0));
        assert_eq!(report.rt_roc_auc, Some(1.0));
        assert_eq!(report.decision_counts.hits, 3);
        assert_eq!(report.decision_counts.correct_rejections, 3);
    }

    #[test]
    fn confidence_order_is_ordinal_and_preserves_response_direction() {
        assert!(confidence_score(false, 3, 3) < confidence_score(false, 1, 3));
        assert!(confidence_score(false, 1, 3) < confidence_score(true, 1, 3));
        assert!(confidence_score(true, 1, 3) < confidence_score(true, 3, 3));
    }

    #[test]
    fn missing_confidence_does_not_drop_binary_or_rt_data() {
        let (mut candidate, truth) = perfect_fixture();
        for row in &mut candidate.trials {
            row.confidence_level = None;
            row.confidence_time_ns = None;
        }
        let report = evaluate(&candidate, &truth).unwrap();
        assert_eq!(report.confidence_trials, 0);
        assert_eq!(report.missing_confidence_after_decision, 6);
        assert!(report.confidence_type1_roc.is_none());
        assert_eq!(report.rt_trials, 6);
        assert!(report.rt_type1_roc.is_some());
        assert_eq!(report.decision_trials, 6);
    }

    #[test]
    fn rt_below_timing_uncertainty_is_excluded_and_counted() {
        let (mut candidate, truth) = perfect_fixture();
        candidate.trials[0].clock_resolution_ns = 100;
        candidate.trials[0].onset_uncertainty_ns = 50;
        candidate.trials[0].input_latency_uncertainty_ns = 50;
        let report = evaluate(&candidate, &truth).unwrap();
        assert_eq!(report.rt_resolution_limited_trials, 1);
        assert_eq!(report.rt_usable_trials, 5);
        assert_eq!(report.rt_trials, 6);
    }

    #[test]
    fn equal_rt_ties_are_not_arbitrarily_ordered() {
        let profile = ObservationProfile { confidence_levels: 3, rt_bins: 3 };
        let rows = vec![
            CandidateTrial {
                trial_id: 0, participant_id: 0, session_id: 0,
                stimulus_onset_ns: 0, decision_yes: Some(true),
                decision_time_ns: Some(100), confidence_level: None,
                confidence_time_ns: None, clock_resolution_ns: 1,
                onset_uncertainty_ns: 1, input_latency_uncertainty_ns: 1,
            },
            CandidateTrial {
                trial_id: 1, participant_id: 0, session_id: 0,
                stimulus_onset_ns: 0, decision_yes: Some(true),
                decision_time_ns: Some(100), confidence_level: None,
                confidence_time_ns: None, clock_resolution_ns: 1,
                onset_uncertainty_ns: 1, input_latency_uncertainty_ns: 1,
            },
        ];
        let truth = EvaluatorTruth {
            trial_ids: vec![0, 1],
            event_present: vec![true, false],
        };
        let report = evaluate(&CandidateView { profile, trials: rows }, &truth).unwrap();
        assert_eq!(report.rt_roc_auc, Some(0.5));
    }

    #[test]
    fn confidence_must_be_recorded_after_binary_decision() {
        let (mut candidate, truth) = perfect_fixture();
        candidate.trials[0].confidence_time_ns = candidate.trials[0].decision_time_ns;
        assert_eq!(evaluate(&candidate, &truth), Err(ObservationError::InvalidConfidenceTimestamp));
    }

    #[test]
    fn non_monotonic_decision_time_is_rejected() {
        let (mut candidate, truth) = perfect_fixture();
        candidate.trials[0].decision_time_ns = Some(candidate.trials[0].stimulus_onset_ns);
        assert_eq!(evaluate(&candidate, &truth), Err(ObservationError::InvalidDecisionTimestamp));
    }

    #[test]
    fn out_of_range_confidence_is_rejected() {
        let (mut candidate, truth) = perfect_fixture();
        candidate.trials[0].confidence_level = Some(4);
        assert_eq!(evaluate(&candidate, &truth), Err(ObservationError::ConfidenceOutOfRange));
    }

    #[test]
    fn duplicate_or_misaligned_trial_ids_are_rejected() {
        let (mut candidate, truth) = perfect_fixture();
        candidate.trials[1].trial_id = candidate.trials[0].trial_id;
        assert_eq!(evaluate(&candidate, &truth), Err(ObservationError::DuplicateTrialId));

        let (candidate, mut truth) = perfect_fixture();
        truth.trial_ids[0] = 99;
        assert_eq!(evaluate(&candidate, &truth), Err(ObservationError::MisalignedTruth));
    }

    #[test]
    fn invalid_profiles_and_clock_resolution_fail_closed() {
        assert_eq!(
            ObservationProfile { confidence_levels: 2, rt_bins: 3 }.validate(),
            Err(ObservationError::InvalidProfile)
        );
        let (mut candidate, truth) = perfect_fixture();
        candidate.trials[0].clock_resolution_ns = 0;
        assert_eq!(evaluate(&candidate, &truth), Err(ObservationError::InvalidClockResolution));
    }

    #[test]
    fn binary_only_campaign_keeps_raw_counts_and_returns_no_confidence_curve() {
        let (mut candidate, truth) = perfect_fixture();
        for row in &mut candidate.trials {
            row.confidence_level = None;
            row.confidence_time_ns = None;
        }
        let report = evaluate(&candidate, &truth).unwrap();
        assert_eq!(report.decision_counts.hits, 3);
        assert_eq!(report.decision_counts.misses, 0);
        assert!(report.binary_signal_detection.is_some());
        assert!(report.confidence_type1_roc.is_none());
        assert!(report.rt_type1_roc.is_some());
    }
}
