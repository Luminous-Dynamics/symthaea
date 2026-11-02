// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic stress testing for signal-detection metric assumptions.
//! This crate composes shared metrics and adds descriptive model diagnostics;
//! it does not implement alternative inferential procedures.

use std::collections::BTreeMap;
use std::fmt;

use symthaea_statistics::{
    RocObservation, SignalDetectionCounts, SignalDetectionMetrics, correlation, mean,
    try_precision_recall_curve, try_roc_curve, variance,
};

const LABEL_STREAM: u64 = 0x4C41_4245_4C00_0001;
const EVIDENCE_STREAM: u64 = 0x4556_4944_454E_0002;
const AUXILIARY_STREAM: u64 = 0x4155_5849_4C00_0003;
const MISSING_STREAM: u64 = 0x4D49_5353_494E_0004;

const VARIANCE_RATIO_LOW: f64 = 0.67;
const VARIANCE_RATIO_HIGH: f64 = 1.50;
const SKEW_LIMIT: f64 = 0.75;
const EXCESS_KURTOSIS_LIMIT: f64 = 2.0;
const LAG_ONE_LIMIT: f64 = 0.25;
const TEMPORAL_SHIFT_LIMIT: f64 = 0.35;
const PARTICIPANT_SD_LIMIT: f64 = 0.40;
const MISSINGNESS_GAP_LIMIT: f64 = 0.10;
const AUXILIARY_AUC_LOW: f64 = 0.30;
const AUXILIARY_AUC_HIGH: f64 = 0.70;

/// Synthetic worlds. Names and truth are evaluator-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StressWorld {
    GaussianReference,
    UnequalVariance,
    HeavyTailed,
    ContaminatedMixture,
    SerialDependence,
    SensitivityDrift,
    CriterionDrift,
    ParticipantHeterogeneity,
    OutcomeDependentMissingness,
    CovariateLeakage,
}

impl StressWorld {
    pub const ALL: [Self; 10] = [
        Self::GaussianReference,
        Self::UnequalVariance,
        Self::HeavyTailed,
        Self::ContaminatedMixture,
        Self::SerialDependence,
        Self::SensitivityDrift,
        Self::CriterionDrift,
        Self::ParticipantHeterogeneity,
        Self::OutcomeDependentMissingness,
        Self::CovariateLeakage,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::GaussianReference => "gaussian_reference",
            Self::UnequalVariance => "unequal_variance",
            Self::HeavyTailed => "heavy_tailed",
            Self::ContaminatedMixture => "contaminated_mixture",
            Self::SerialDependence => "serial_dependence",
            Self::SensitivityDrift => "sensitivity_drift",
            Self::CriterionDrift => "criterion_drift",
            Self::ParticipantHeterogeneity => "participant_heterogeneity",
            Self::OutcomeDependentMissingness => "outcome_dependent_missingness",
            Self::CovariateLeakage => "covariate_leakage",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustnessError {
    InvalidDesign,
    CountOverflow,
    MisalignedRecords,
    PartialObservation,
    NonFiniteScore,
    MissingClass,
    StatisticFailure,
}

impl fmt::Display for RobustnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidDesign => "design needs >=4 participants, >=2 sessions, and >=8 even trials per session",
            Self::CountOverflow => "design exceeds supported count or identifier range",
            Self::MisalignedRecords => "candidate records and evaluator truth are not aligned",
            Self::PartialObservation => "score and response must both be present or both be absent",
            Self::NonFiniteScore => "candidate contains a non-finite score",
            Self::MissingClass => "both event-present and event-absent observed trials are required",
            Self::StatisticFailure => "shared statistics API could not evaluate the records",
        };
        f.write_str(message)
    }
}

impl std::error::Error for RobustnessError {}

/// Fixed design for one independent synthetic campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimulationDesign {
    pub participants: usize,
    pub sessions_per_participant: usize,
    pub trials_per_session: usize,
    pub seed: u64,
}

impl SimulationDesign {
    pub fn validate(self) -> Result<Self, RobustnessError> {
        if self.participants < 4
            || self.sessions_per_participant < 2
            || self.trials_per_session < 8
            || self.trials_per_session % 2 != 0
        {
            return Err(RobustnessError::InvalidDesign);
        }
        let total = self
            .participants
            .checked_mul(self.sessions_per_participant)
            .and_then(|v| v.checked_mul(self.trials_per_session))
            .ok_or(RobustnessError::CountOverflow)?;
        u64::try_from(total).map_err(|_| RobustnessError::CountOverflow)?;
        Ok(self)
    }

    pub fn total_trials(self) -> Result<usize, RobustnessError> {
        let design = self.validate()?;
        design
            .participants
            .checked_mul(design.sessions_per_participant)
            .and_then(|v| v.checked_mul(design.trials_per_session))
            .ok_or(RobustnessError::CountOverflow)
    }
}

/// One candidate-visible row. An absent score/response is a recorded missing
/// observation; the row itself remains present in the trial ledger.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateTrial {
    pub trial_id: u64,
    pub participant_id: u32,
    pub session_id: u32,
    pub period: u32,
    pub evidence_score: Option<f64>,
    pub response: Option<bool>,
    /// Independent auxiliary feature. It is noise in all but the leakage world.
    pub auxiliary_score: f64,
}

/// Candidate-visible record set. It carries no target labels, scenario, seed,
/// injected sensitivity, or hidden missingness reason.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateView {
    pub trials: Vec<CandidateTrial>,
}

/// Evaluator-only truth, separate from candidate-visible records.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorTruth {
    trial_ids: Vec<u64>,
    event_present: Vec<bool>,
    missing: Vec<bool>,
    injected_d_prime: Vec<f64>,
    world: StressWorld,
    seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticBundle {
    pub candidate: CandidateView,
    pub evaluator: EvaluatorTruth,
}

/// Configured screening flag; these are not hypothesis tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticFlag {
    UnequalClassVariance,
    NonGaussianTails,
    SerialDependence,
    TemporalSensitivityDrift,
    TemporalCriterionDrift,
    ParticipantHeterogeneity,
    DifferentialMissingness,
    AuxiliaryChannelLeakage,
}

impl DiagnosticFlag {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnequalClassVariance => "unequal_class_variance",
            Self::NonGaussianTails => "non_gaussian_tails",
            Self::SerialDependence => "serial_dependence",
            Self::TemporalSensitivityDrift => "temporal_sensitivity_drift",
            Self::TemporalCriterionDrift => "temporal_criterion_drift",
            Self::ParticipantHeterogeneity => "participant_heterogeneity",
            Self::DifferentialMissingness => "differential_missingness",
            Self::AuxiliaryChannelLeakage => "auxiliary_channel_leakage",
        }
    }
}

/// No configured concern is not proof that assumptions hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDisposition {
    NotEvaluable,
    AssumptionConcern,
    NoConfiguredConcern,
}

impl MetricDisposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotEvaluable => "not_evaluable",
            Self::AssumptionConcern => "assumption_concern",
            Self::NoConfiguredConcern => "no_configured_concern",
        }
    }
}

/// Descriptive report. No p-value, confidence interval, causal claim, or
/// adjusted d-prime is produced.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustnessReport {
    pub world: StressWorld,
    pub seed: u64,
    pub total_trials: usize,
    pub observed_trials: usize,
    pub missing_trials: usize,
    pub signal_trials_observed: usize,
    pub noise_trials_observed: usize,
    pub counts: SignalDetectionCounts,
    pub signal_detection: SignalDetectionMetrics,
    pub roc_auc: f64,
    pub average_precision: f64,
    pub positive_prevalence: f64,
    pub signal_noise_variance_ratio: f64,
    pub signal_skewness: f64,
    pub noise_skewness: f64,
    pub signal_excess_kurtosis: f64,
    pub noise_excess_kurtosis: f64,
    pub within_session_lag_one_residual_correlation: f64,
    pub early_late_d_prime_delta: Option<f64>,
    pub early_late_criterion_delta: Option<f64>,
    pub mean_participant_d_prime: Option<f64>,
    pub participant_d_prime_sd: Option<f64>,
    pub signal_missing_rate: f64,
    pub noise_missing_rate: f64,
    pub auxiliary_channel_auc: f64,
    pub flags: Vec<DiagnosticFlag>,
    pub disposition: MetricDisposition,
}

/// Generate one deterministic synthetic campaign. The PRNG is for replay,
/// not cryptographic target selection or real-world blinding.
pub fn simulate(
    world: StressWorld,
    design: SimulationDesign,
) -> Result<SyntheticBundle, RobustnessError> {
    let design = design.validate()?;
    let total = design.total_trials()?;

    let mut labels_rng = SplitMix64::new(design.seed ^ LABEL_STREAM);
    let mut evidence_rng = NormalRng::new(design.seed ^ EVIDENCE_STREAM);
    let mut auxiliary_rng = NormalRng::new(design.seed ^ AUXILIARY_STREAM);
    let mut missing_rng = SplitMix64::new(design.seed ^ MISSING_STREAM);

    let mut candidate_trials = Vec::with_capacity(total);
    let mut trial_ids = Vec::with_capacity(total);
    let mut event_present = Vec::with_capacity(total);
    let mut missing = Vec::with_capacity(total);
    let mut injected_d_prime = Vec::with_capacity(total);
    let mut trial_id = 0u64;

    for participant in 0..design.participants {
        let mut ar_state = 0.0;
        for session in 0..design.sessions_per_participant {
            let mut labels = vec![true; design.trials_per_session / 2];
            labels.extend(vec![false; design.trials_per_session / 2]);
            shuffle(&mut labels, &mut labels_rng)?;

            for (period, is_signal) in labels.into_iter().enumerate() {
                let session_fraction =
                    session as f64 / (design.sessions_per_participant - 1) as f64;

                let sensitivity = match world {
                    StressWorld::SensitivityDrift => 0.20 + 1.60 * session_fraction,
                    StressWorld::ParticipantHeterogeneity => {
                        0.15 + 1.70 * participant as f64 / (design.participants - 1) as f64
                    }
                    StressWorld::CovariateLeakage => 0.0,
                    _ => 1.0,
                };

                let criterion = match world {
                    StressWorld::CriterionDrift => -1.0 + 2.0 * session_fraction,
                    _ => 0.0,
                };

                let mut residual = match world {
                    StressWorld::HeavyTailed => evidence_rng.standardized_student_t3(),
                    StressWorld::ContaminatedMixture => {
                        if evidence_rng.uniform() < 0.08 {
                            evidence_rng.standard_normal() * 5.0
                        } else {
                            evidence_rng.standard_normal()
                        }
                    }
                    StressWorld::SerialDependence => {
                        let innovation = evidence_rng.standard_normal();
                        ar_state = 0.80 * ar_state
                            + (1.0_f64 - 0.80_f64.powi(2)).sqrt() * innovation;
                        ar_state
                    }
                    _ => evidence_rng.standard_normal(),
                };

                let class_scale = if world == StressWorld::UnequalVariance && is_signal {
                    1.80
                } else {
                    1.0
                };
                residual *= class_scale;

                let evidence_score = if world == StressWorld::CovariateLeakage {
                    residual
                } else if is_signal {
                    sensitivity / 2.0 + residual
                } else {
                    -sensitivity / 2.0 + residual
                };

                let response = evidence_score >= criterion;
                let auxiliary_score = if world == StressWorld::CovariateLeakage {
                    (if is_signal { 1.5 } else { -1.5 })
                        + auxiliary_rng.standard_normal() * 0.80
                } else {
                    auxiliary_rng.standard_normal()
                };

                let is_missing = if world == StressWorld::OutcomeDependentMissingness {
                    let probability = if is_signal && response { 0.65 } else { 0.05 };
                    missing_rng.uniform() < probability
                } else {
                    false
                };

                let participant_id =
                    u32::try_from(participant).map_err(|_| RobustnessError::CountOverflow)?;
                let session_id =
                    u32::try_from(session).map_err(|_| RobustnessError::CountOverflow)?;
                let period_id =
                    u32::try_from(period).map_err(|_| RobustnessError::CountOverflow)?;

                candidate_trials.push(CandidateTrial {
                    trial_id,
                    participant_id,
                    session_id,
                    period: period_id,
                    evidence_score: (!is_missing).then_some(evidence_score),
                    response: (!is_missing).then_some(response),
                    auxiliary_score,
                });
                trial_ids.push(trial_id);
                event_present.push(is_signal);
                missing.push(is_missing);
                injected_d_prime.push(sensitivity);
                trial_id = trial_id.checked_add(1).ok_or(RobustnessError::CountOverflow)?;
            }
        }
    }

    Ok(SyntheticBundle {
        candidate: CandidateView {
            trials: candidate_trials,
        },
        evaluator: EvaluatorTruth {
            trial_ids,
            event_present,
            missing,
            injected_d_prime,
            world,
            seed: design.seed,
        },
    })
}

/// Evaluate candidate-visible records against the evaluator-only truth sidecar.
pub fn evaluate(
    candidate: &CandidateView,
    truth: &EvaluatorTruth,
) -> Result<RobustnessReport, RobustnessError> {
    let n = candidate.trials.len();
    if n == 0
        || truth.trial_ids.len() != n
        || truth.event_present.len() != n
        || truth.missing.len() != n
        || truth.injected_d_prime.len() != n
    {
        return Err(RobustnessError::MisalignedRecords);
    }

    let mut counts = SignalDetectionCounts::default();
    let mut signal_scores = Vec::new();
    let mut noise_scores = Vec::new();
    let mut auxiliary_observations = Vec::with_capacity(n);
    let mut main_observations = Vec::new();
    let mut observed_by_class = [0usize; 2];
    let mut missing_by_class = [0usize; 2];

    for index in 0..n {
        let row = candidate.trials[index];
        let is_signal = truth.event_present[index];

        if row.trial_id != truth.trial_ids[index] {
            return Err(RobustnessError::MisalignedRecords);
        }
        if row.evidence_score.is_some() != row.response.is_some() {
            return Err(RobustnessError::PartialObservation);
        }
        if !row.auxiliary_score.is_finite()
            || row.evidence_score.is_some_and(|value| !value.is_finite())
        {
            return Err(RobustnessError::NonFiniteScore);
        }

        auxiliary_observations.push(RocObservation {
            score: row.auxiliary_score,
            is_signal,
        });

        if truth.missing[index] {
            if row.evidence_score.is_some() {
                return Err(RobustnessError::MisalignedRecords);
            }
            missing_by_class[if is_signal { 1 } else { 0 }] += 1;
            continue;
        }
        let score = row.evidence_score.ok_or(RobustnessError::MisalignedRecords)?;
        let response = row.response.ok_or(RobustnessError::MisalignedRecords)?;
        observed_by_class[if is_signal { 1 } else { 0 }] += 1;
        main_observations.push(RocObservation { score, is_signal });

        match (is_signal, response) {
            (true, true) => counts.hits += 1,
            (true, false) => counts.misses += 1,
            (false, true) => counts.false_alarms += 1,
            (false, false) => counts.correct_rejections += 1,
        }

        if is_signal {
            signal_scores.push(score);
        } else {
            noise_scores.push(score);
        }
    }

    if observed_by_class[0] == 0 || observed_by_class[1] == 0 {
        return Err(RobustnessError::MissingClass);
    }

    let signal_detection = counts.metrics().ok_or(RobustnessError::StatisticFailure)?;
    let roc = try_roc_curve(&main_observations).map_err(|_| RobustnessError::StatisticFailure)?;
    let pr = try_precision_recall_curve(&main_observations)
        .map_err(|_| RobustnessError::StatisticFailure)?;
    let aux_roc =
        try_roc_curve(&auxiliary_observations).map_err(|_| RobustnessError::StatisticFailure)?;

    let signal_variance = variance(&signal_scores).ok_or(RobustnessError::StatisticFailure)?;
    let noise_variance = variance(&noise_scores).ok_or(RobustnessError::StatisticFailure)?;
    if noise_variance <= 0.0 || !noise_variance.is_finite() {
        return Err(RobustnessError::StatisticFailure);
    }
    let variance_ratio = signal_variance / noise_variance;

    let signal_skewness = skewness(&signal_scores).ok_or(RobustnessError::StatisticFailure)?;
    let noise_skewness = skewness(&noise_scores).ok_or(RobustnessError::StatisticFailure)?;
    let signal_kurtosis = excess_kurtosis(&signal_scores).ok_or(RobustnessError::StatisticFailure)?;
    let noise_kurtosis = excess_kurtosis(&noise_scores).ok_or(RobustnessError::StatisticFailure)?;

    let residual_lag_one =
        residual_lag_one_correlation(candidate, truth, &signal_scores, &noise_scores);
    let (early_late_d_prime_delta, early_late_criterion_delta) =
        temporal_metric_deltas(candidate, truth);

    let mut participant_d_primes = Vec::new();
    let max_participant = candidate
        .trials
        .iter()
        .map(|row| row.participant_id as usize)
        .max()
        .unwrap_or(0);
    for participant in 0..=max_participant {
        let participant_counts = counts_for(candidate, truth, |row| {
            row.participant_id as usize == participant
        });
        if let Some(metrics) = participant_counts.and_then(SignalDetectionCounts::metrics) {
            participant_d_primes.push(metrics.d_prime);
        }
    }
    let participant_mean = mean(&participant_d_primes);
    let participant_sd = population_sd(&participant_d_primes);

    let signal_missing_rate =
        missing_by_class[1] as f64 / (missing_by_class[1] + observed_by_class[1]) as f64;
    let noise_missing_rate =
        missing_by_class[0] as f64 / (missing_by_class[0] + observed_by_class[0]) as f64;
    let overall_missing_rate = (missing_by_class[0] + missing_by_class[1]) as f64 / n as f64;

    let mut flags = Vec::new();
    if !(VARIANCE_RATIO_LOW..=VARIANCE_RATIO_HIGH).contains(&variance_ratio) {
        flags.push(DiagnosticFlag::UnequalClassVariance);
    }
    if signal_skewness.abs().max(noise_skewness.abs()) > SKEW_LIMIT
        || signal_kurtosis > EXCESS_KURTOSIS_LIMIT
        || noise_kurtosis > EXCESS_KURTOSIS_LIMIT
    {
        flags.push(DiagnosticFlag::NonGaussianTails);
    }
    if residual_lag_one.abs() > LAG_ONE_LIMIT {
        flags.push(DiagnosticFlag::SerialDependence);
    }
    if early_late_d_prime_delta.is_some_and(|delta| delta.abs() > TEMPORAL_SHIFT_LIMIT) {
        flags.push(DiagnosticFlag::TemporalSensitivityDrift);
    }
    if early_late_criterion_delta.is_some_and(|delta| delta.abs() > TEMPORAL_SHIFT_LIMIT) {
        flags.push(DiagnosticFlag::TemporalCriterionDrift);
    }
    if participant_sd.is_some_and(|sd| sd > PARTICIPANT_SD_LIMIT) {
        flags.push(DiagnosticFlag::ParticipantHeterogeneity);
    }
    if (signal_missing_rate - noise_missing_rate).abs() > MISSINGNESS_GAP_LIMIT {
        flags.push(DiagnosticFlag::DifferentialMissingness);
    }
    if !(AUXILIARY_AUC_LOW..=AUXILIARY_AUC_HIGH).contains(&aux_roc.auc) {
        flags.push(DiagnosticFlag::AuxiliaryChannelLeakage);
    }

    let observed_trials = observed_by_class[0] + observed_by_class[1];
    let disposition = if observed_by_class[0] < 20
        || observed_by_class[1] < 20
        || overall_missing_rate > 0.35
    {
        MetricDisposition::NotEvaluable
    } else if !flags.is_empty() {
        MetricDisposition::AssumptionConcern
    } else {
        MetricDisposition::NoConfiguredConcern
    };

    Ok(RobustnessReport {
        world: truth.world,
        seed: truth.seed,
        total_trials: n,
        observed_trials,
        missing_trials: n - observed_trials,
        signal_trials_observed: observed_by_class[1],
        noise_trials_observed: observed_by_class[0],
        counts,
        signal_detection,
        roc_auc: roc.auc,
        average_precision: pr.average_precision,
        positive_prevalence: observed_by_class[1] as f64 / observed_trials as f64,
        signal_noise_variance_ratio: variance_ratio,
        signal_skewness,
        noise_skewness,
        signal_excess_kurtosis: signal_kurtosis,
        noise_excess_kurtosis: noise_kurtosis,
        within_session_lag_one_residual_correlation: residual_lag_one,
        early_late_d_prime_delta,
        early_late_criterion_delta,
        mean_participant_d_prime: participant_mean,
        participant_d_prime_sd: participant_sd,
        signal_missing_rate,
        noise_missing_rate,
        auxiliary_channel_auc: aux_roc.auc,
        flags,
        disposition,
    })
}

fn counts_for(
    candidate: &CandidateView,
    truth: &EvaluatorTruth,
    keep: impl Fn(&CandidateTrial) -> bool,
) -> Option<SignalDetectionCounts> {
    let mut counts = SignalDetectionCounts::default();
    for (row, is_signal) in candidate.trials.iter().zip(truth.event_present.iter().copied()) {
        if !keep(row) {
            continue;
        }
        let (Some(_), Some(response)) = (row.evidence_score, row.response) else {
            continue;
        };
        match (is_signal, response) {
            (true, true) => counts.hits += 1,
            (true, false) => counts.misses += 1,
            (false, true) => counts.false_alarms += 1,
            (false, false) => counts.correct_rejections += 1,
        }
    }
    Some(counts)
}

fn temporal_metric_deltas(
    candidate: &CandidateView,
    truth: &EvaluatorTruth,
) -> (Option<f64>, Option<f64>) {
    let midpoint = candidate
        .trials
        .iter()
        .map(|row| row.session_id)
        .max()
        .unwrap_or(0)
        / 2;
    let early = counts_for(candidate, truth, |row| row.session_id <= midpoint)
        .and_then(SignalDetectionCounts::metrics);
    let late = counts_for(candidate, truth, |row| row.session_id > midpoint)
        .and_then(SignalDetectionCounts::metrics);
    match (early, late) {
        (Some(a), Some(b)) => (
            Some(b.d_prime - a.d_prime),
            Some(b.criterion_c - a.criterion_c),
        ),
        _ => (None, None),
    }
}

fn residual_lag_one_correlation(
    candidate: &CandidateView,
    truth: &EvaluatorTruth,
    signal_scores: &[f64],
    noise_scores: &[f64],
) -> f64 {
    let Some(signal_mean) = mean(signal_scores) else {
        return 0.0;
    };
    let Some(noise_mean) = mean(noise_scores) else {
        return 0.0;
    };

    let mut by_session: BTreeMap<(u32, u32), Vec<f64>> = BTreeMap::new();
    for (row, is_signal) in candidate.trials.iter().zip(truth.event_present.iter().copied()) {
        let Some(score) = row.evidence_score else {
            continue;
        };
        let residual = score - if is_signal { signal_mean } else { noise_mean };
        by_session
            .entry((row.participant_id, row.session_id))
            .or_default()
            .push(residual);
    }

    let mut previous = Vec::new();
    let mut current = Vec::new();
    for series in by_session.values() {
        for pair in series.windows(2) {
            previous.push(pair[0]);
            current.push(pair[1]);
        }
    }
    correlation(&previous, &current).unwrap_or(0.0)
}

fn skewness(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let m = mean(values)?;
    let v = variance(values)?;
    if v <= 0.0 {
        return None;
    }
    let third = values.iter().map(|x| (x - m).powi(3)).sum::<f64>() / values.len() as f64;
    Some(third / v.powf(1.5))
}

fn excess_kurtosis(values: &[f64]) -> Option<f64> {
    if values.len() < 4 {
        return None;
    }
    let m = mean(values)?;
    let v = variance(values)?;
    if v <= 0.0 {
        return None;
    }
    let fourth = values.iter().map(|x| (x - m).powi(4)).sum::<f64>() / values.len() as f64;
    Some(fourth / v.powi(2) - 3.0)
}

fn population_sd(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let m = mean(values)?;
    Some((values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64).sqrt())
}

fn shuffle<T>(items: &mut [T], rng: &mut SplitMix64) -> Result<(), RobustnessError> {
    for index in (1..items.len()).rev() {
        let upper = u64::try_from(index + 1).map_err(|_| RobustnessError::CountOverflow)?;
        let selected = rng.below(upper) as usize;
        items.swap(index, selected);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
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

    fn below(&mut self, upper: u64) -> u64 {
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let product = (self.next_u64() as u128) * (upper as u128);
            let low = product as u64;
            if low >= threshold {
                return (product >> 64) as u64;
            }
        }
    }

    fn uniform(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        ((mantissa as f64 + 0.5) / 9_007_199_254_740_992.0)
            .min(1.0 - f64::EPSILON)
    }
}

#[derive(Debug, Clone, Copy)]
struct NormalRng {
    rng: SplitMix64,
    spare: Option<f64>,
}

impl NormalRng {
    fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.rng.uniform()
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.rng.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.rng.uniform();
        let (sin, cos) = angle.sin_cos();
        self.spare = Some(radius * sin);
        radius * cos
    }

    /// Student-t with 3 degrees of freedom, rescaled to unit variance.
    fn standardized_student_t3(&mut self) -> f64 {
        let numerator = self.standard_normal();
        let a = self.standard_normal();
        let b = self.standard_normal();
        let c = self.standard_normal();
        numerator / (a * a + b * b + c * c).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(seed: u64) -> SimulationDesign {
        SimulationDesign {
            participants: 24,
            sessions_per_participant: 4,
            trials_per_session: 32,
            seed,
        }
    }

    fn report(world: StressWorld, seed: u64) -> RobustnessReport {
        let bundle = simulate(world, design(seed)).unwrap();
        evaluate(&bundle.candidate, &bundle.evaluator).unwrap()
    }

    #[test]
    fn gaussian_reference_is_replayable_and_has_no_configured_flag() {
        let bundle = simulate(StressWorld::GaussianReference, design(0x5EED_2026)).unwrap();
        let replay = simulate(StressWorld::GaussianReference, design(0x5EED_2026)).unwrap();
        assert_eq!(bundle, replay);
        let result = evaluate(&bundle.candidate, &bundle.evaluator).unwrap();
        assert_eq!(result.disposition, MetricDisposition::NoConfiguredConcern);
        assert!(result.flags.is_empty());
        assert!((result.signal_detection.d_prime - 1.0).abs() < 0.20);
        assert!((result.roc_auc - 0.76).abs() < 0.08);
    }

    #[test]
    fn variance_inequality_is_reported_without_adjusting_d_prime() {
        let result = report(StressWorld::UnequalVariance, 0x5EED_2026);
        assert!(result.flags.contains(&DiagnosticFlag::UnequalClassVariance));
        assert!(result.signal_detection.d_prime.is_finite());
    }

    #[test]
    fn heavy_tails_and_contamination_are_screened() {
        let heavy = report(StressWorld::HeavyTailed, 0x5EED_2026);
        let mixture = report(StressWorld::ContaminatedMixture, 0x5EED_2026);
        assert!(heavy.flags.contains(&DiagnosticFlag::NonGaussianTails));
        assert!(mixture.flags.contains(&DiagnosticFlag::NonGaussianTails));
    }

    #[test]
    fn autocorrelation_is_detected_at_the_declared_strength() {
        let result = report(StressWorld::SerialDependence, 0x5EED_2026);
        assert!(result.flags.contains(&DiagnosticFlag::SerialDependence));
    }

    #[test]
    fn sensitivity_and_criterion_drift_are_not_collapsed() {
        let sensitivity = report(StressWorld::SensitivityDrift, 0x5EED_2026);
        let criterion = report(StressWorld::CriterionDrift, 0x5EED_2026);
        assert!(sensitivity.flags.contains(&DiagnosticFlag::TemporalSensitivityDrift));
        assert!(!sensitivity.flags.contains(&DiagnosticFlag::TemporalCriterionDrift));
        assert!(criterion.flags.contains(&DiagnosticFlag::TemporalCriterionDrift));
        assert!(!criterion.flags.contains(&DiagnosticFlag::TemporalSensitivityDrift));
    }

    #[test]
    fn participant_heterogeneity_is_reported() {
        let result = report(StressWorld::ParticipantHeterogeneity, 0x5EED_2026);
        assert!(result.flags.contains(&DiagnosticFlag::ParticipantHeterogeneity));
    }

    #[test]
    fn outcome_dependent_missingness_is_reported_and_retained() {
        let result = report(StressWorld::OutcomeDependentMissingness, 0x5EED_2026);
        assert!(result.missing_trials > 0);
        assert!(result.flags.contains(&DiagnosticFlag::DifferentialMissingness));
    }

    #[test]
    fn auxiliary_assignment_leakage_is_separate_from_main_score() {
        let result = report(StressWorld::CovariateLeakage, 0x5EED_2026);
        assert!(result.flags.contains(&DiagnosticFlag::AuxiliaryChannelLeakage));
        assert!(result.auxiliary_channel_auc > 0.85);
        assert!((result.roc_auc - 0.5).abs() < 0.10);
    }

    #[test]
    fn malformed_records_fail_closed() {
        let mut bundle = simulate(StressWorld::GaussianReference, design(7)).unwrap();
        bundle.candidate.trials[0].evidence_score = Some(f64::NAN);
        assert_eq!(
            evaluate(&bundle.candidate, &bundle.evaluator),
            Err(RobustnessError::NonFiniteScore)
        );

        let mut bundle = simulate(StressWorld::GaussianReference, design(7)).unwrap();
        bundle.candidate.trials[0].response = None;
        assert_eq!(
            evaluate(&bundle.candidate, &bundle.evaluator),
            Err(RobustnessError::PartialObservation)
        );
    }
}
