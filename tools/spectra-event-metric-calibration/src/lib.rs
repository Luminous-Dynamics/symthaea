// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Known-truth calibration of binary signal-detection metrics.
//!
//! The generator is a mathematical reference fixture, not a physical model.
//! Signal and noise evidence follow equal-variance unit-normal distributions:
//!   noise:  N(-d'/2, 1)
//!   signal: N(+d'/2, 1)
//! A positive response is emitted when evidence >= criterion.
//!
//! Candidate-facing records contain only an opaque trial ordinal, a graded
//! evidence score and a binary response. Ground-truth labels, seed and injected
//! parameters live in a separate evaluator sidecar.

use std::fmt;

use symthaea_statistics::{
    RocObservation, SignalDetectionCounts, SignalDetectionMetrics, normal_cdf,
    try_precision_recall_curve, try_roc_curve,
};

const MAX_ABS_MODEL_PARAMETER: f64 = 8.0;
const SIGNAL_STREAM: u64 = 0x5349_474E_414C_0001;
const NOISE_STREAM: u64 = 0x4E4F_4953_4500_0002;
const SHUFFLE_STREAM: u64 = 0x5348_5546_464C_4503;

/// Invalid model, design, or evaluator input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationError {
    InvalidModel,
    EmptyClass,
    CountOverflow,
    NonFiniteScore,
    MisalignedRecords,
    StatisticFailure,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidModel => "model parameters must be finite and within the declared range",
            Self::EmptyClass => "signal and noise classes must both contain at least one trial",
            Self::CountOverflow => "trial count exceeds the supported identifier/count range",
            Self::NonFiniteScore => "candidate evidence score is not finite",
            Self::MisalignedRecords => "candidate records do not match the evaluator sidecar",
            Self::StatisticFailure => "shared statistics API could not evaluate the supplied records",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for CalibrationError {}

/// Injected parameters for the equal-variance Gaussian reference model.
///
/// Negative d-prime is permitted as an explicitly reversed-ranking control.
/// The bounded range prevents pathological values from overflowing score
/// arithmetic and is part of this fixture's versioned contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianSignalModel {
    d_prime: f64,
    criterion_c: f64,
}

impl GaussianSignalModel {
    pub fn new(d_prime: f64, criterion_c: f64) -> Result<Self, CalibrationError> {
        if !d_prime.is_finite()
            || !criterion_c.is_finite()
            || d_prime.abs() > MAX_ABS_MODEL_PARAMETER
            || criterion_c.abs() > MAX_ABS_MODEL_PARAMETER
        {
            return Err(CalibrationError::InvalidModel);
        }
        Ok(Self {
            d_prime,
            criterion_c,
        })
    }

    pub fn d_prime(self) -> f64 {
        self.d_prime
    }

    pub fn criterion_c(self) -> f64 {
        self.criterion_c
    }
}

/// Fixed class counts and seed for one independent synthetic campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimulationDesign {
    pub signal_trials: usize,
    pub noise_trials: usize,
    pub seed: u64,
}

impl SimulationDesign {
    pub fn validate(self) -> Result<Self, CalibrationError> {
        if self.signal_trials == 0 || self.noise_trials == 0 {
            return Err(CalibrationError::EmptyClass);
        }
        let total = self
            .signal_trials
            .checked_add(self.noise_trials)
            .ok_or(CalibrationError::CountOverflow)?;
        u64::try_from(total).map_err(|_| CalibrationError::CountOverflow)?;
        Ok(self)
    }
}

/// One candidate-visible trial.
///
/// This record intentionally has no target/event label, seed, generator
/// identity or injected model parameter. The evidence score is a synthetic
/// graded observer output, not a calibrated probability or physical sensor
/// reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateTrial {
    pub trial_id: u64,
    pub evidence_score: f64,
    pub response: bool,
}

/// Candidate-visible view of a synthetic campaign.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateView {
    trials: Vec<CandidateTrial>,
}

impl CandidateView {
    pub fn trials(&self) -> &[CandidateTrial] {
        &self.trials
    }
}

/// Evaluator-only truth sidecar. Do not pass this value to the candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorTruth {
    trial_ids: Vec<u64>,
    signal_labels: Vec<bool>,
    model: GaussianSignalModel,
    seed: u64,
}

impl EvaluatorTruth {
    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn model(&self) -> GaussianSignalModel {
        self.model
    }
}

/// Separate candidate and evaluator artifacts produced by the generator.
///
/// Keeping the two values distinct helps tests and callers preserve role
/// separation. It is not a security boundary against a process that receives
/// both values or can inspect the generator source.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticBundle {
    pub candidate: CandidateView,
    pub evaluator: EvaluatorTruth,
}

/// Auditable metric report for one generated campaign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationReport {
    pub seed: u64,
    pub injected_d_prime: f64,
    pub injected_criterion_c: f64,
    pub prevalence: f64,
    pub counts: SignalDetectionCounts,
    pub signal_detection: SignalDetectionMetrics,
    pub roc_auc: f64,
    pub expected_roc_auc: f64,
    pub average_precision: f64,
    /// Expected average precision of a random ranking, equal to prevalence.
    pub random_ranking_ap_baseline: f64,
    pub accuracy: f64,
    pub expected_accuracy: f64,
    pub balanced_accuracy: f64,
    pub expected_balanced_accuracy: f64,
    /// Undefined when the observer makes no positive calls.
    pub precision: Option<f64>,
    pub expected_hit_rate: f64,
    pub expected_false_alarm_rate: f64,
}

/// Generate one campaign. Identical inputs reproduce the same records in the
/// declared Rust/runtime implementation.
///
/// Common random numbers are used across model cells: for a fixed seed and
/// class counts, the underlying standard-normal draws and trial permutation do
/// not depend on d-prime or criterion. This permits paired comparisons across
/// sensitivity/criterion ladders without reusing outcomes as independent data.
pub fn simulate(
    model: GaussianSignalModel,
    design: SimulationDesign,
) -> Result<SyntheticBundle, CalibrationError> {
    let design = design.validate()?;
    let total = design
        .signal_trials
        .checked_add(design.noise_trials)
        .ok_or(CalibrationError::CountOverflow)?;

    let mut signal_rng = NormalRng::new(design.seed ^ SIGNAL_STREAM);
    let mut noise_rng = NormalRng::new(design.seed ^ NOISE_STREAM);
    let mut rows = Vec::with_capacity(total);

    for _ in 0..design.signal_trials {
        let evidence = model.d_prime / 2.0 + signal_rng.standard_normal();
        rows.push((evidence, evidence >= model.criterion_c, true));
    }
    for _ in 0..design.noise_trials {
        let evidence = -model.d_prime / 2.0 + noise_rng.standard_normal();
        rows.push((evidence, evidence >= model.criterion_c, false));
    }

    let mut shuffle_rng = SplitMix64::new(design.seed ^ SHUFFLE_STREAM);
    for index in (1..rows.len()).rev() {
        let upper = u64::try_from(index + 1).map_err(|_| CalibrationError::CountOverflow)?;
        let selected = shuffle_rng.below(upper) as usize;
        rows.swap(index, selected);
    }

    let mut candidate_trials = Vec::with_capacity(total);
    let mut trial_ids = Vec::with_capacity(total);
    let mut signal_labels = Vec::with_capacity(total);

    for (index, (evidence_score, response, is_signal)) in rows.into_iter().enumerate() {
        let trial_id = u64::try_from(index).map_err(|_| CalibrationError::CountOverflow)?;
        candidate_trials.push(CandidateTrial {
            trial_id,
            evidence_score,
            response,
        });
        trial_ids.push(trial_id);
        signal_labels.push(is_signal);
    }

    Ok(SyntheticBundle {
        candidate: CandidateView {
            trials: candidate_trials,
        },
        evaluator: EvaluatorTruth {
            trial_ids,
            signal_labels,
            model,
            seed: design.seed,
        },
    })
}

/// Evaluate candidate-visible records against the separately held truth sidecar.
///
/// ROC/AUC and average precision use the graded evidence scores. They are not
/// reconstructed from the binary response. The analytic references apply only
/// to the declared equal-variance Gaussian generator.
pub fn evaluate(
    candidate: &CandidateView,
    truth: &EvaluatorTruth,
) -> Result<CalibrationReport, CalibrationError> {
    if candidate.trials.len() != truth.signal_labels.len()
        || candidate.trials.len() != truth.trial_ids.len()
        || candidate.trials.is_empty()
    {
        return Err(CalibrationError::MisalignedRecords);
    }

    let mut counts = SignalDetectionCounts::default();
    let mut ranked = Vec::with_capacity(candidate.trials.len());

    for ((trial, expected_id), is_signal) in candidate
        .trials
        .iter()
        .zip(truth.trial_ids.iter())
        .zip(truth.signal_labels.iter().copied())
    {
        if trial.trial_id != *expected_id {
            return Err(CalibrationError::MisalignedRecords);
        }
        if !trial.evidence_score.is_finite() {
            return Err(CalibrationError::NonFiniteScore);
        }

        match (is_signal, trial.response) {
            (true, true) => counts.hits += 1,
            (true, false) => counts.misses += 1,
            (false, true) => counts.false_alarms += 1,
            (false, false) => counts.correct_rejections += 1,
        }
        ranked.push(RocObservation {
            score: trial.evidence_score,
            is_signal,
        });
    }

    let signal_detection = counts
        .metrics()
        .ok_or(CalibrationError::StatisticFailure)?;
    let roc = try_roc_curve(&ranked).map_err(|_| CalibrationError::StatisticFailure)?;
    let pr = try_precision_recall_curve(&ranked)
        .map_err(|_| CalibrationError::StatisticFailure)?;

    let total = signal_detection.signal_trials + signal_detection.noise_trials;
    let prevalence = signal_detection.signal_trials as f64 / total as f64;
    let model = truth.model;
    let expected_hit_rate = normal_cdf(model.d_prime / 2.0 - model.criterion_c);
    let expected_false_alarm_rate = normal_cdf(-model.d_prime / 2.0 - model.criterion_c);
    let expected_roc_auc = normal_cdf(model.d_prime / std::f64::consts::SQRT_2);
    let expected_accuracy = prevalence * expected_hit_rate
        + (1.0 - prevalence) * (1.0 - expected_false_alarm_rate);
    let expected_balanced_accuracy =
        0.5 * (expected_hit_rate + (1.0 - expected_false_alarm_rate));

    let correct = counts.hits + counts.correct_rejections;
    let accuracy = correct as f64 / total as f64;
    let balanced_accuracy =
        0.5 * (signal_detection.hit_rate + signal_detection.correct_rejection_rate);
    let positive_calls = counts.hits + counts.false_alarms;
    let precision = (positive_calls > 0)
        .then(|| counts.hits as f64 / positive_calls as f64);

    Ok(CalibrationReport {
        seed: truth.seed,
        injected_d_prime: model.d_prime,
        injected_criterion_c: model.criterion_c,
        prevalence,
        counts,
        signal_detection,
        roc_auc: roc.auc,
        expected_roc_auc,
        average_precision: pr.average_precision,
        random_ranking_ap_baseline: prevalence,
        accuracy,
        expected_accuracy,
        balanced_accuracy,
        expected_balanced_accuracy,
        precision,
        expected_hit_rate,
        expected_false_alarm_rate,
    })
}

/// SplitMix64 PRNG. It is deterministic, not cryptographically unpredictable.
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

    fn unit_interval_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        ((mantissa as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0))
            .min(1.0 - f64::EPSILON)
    }

    /// Exact bounded reduction using multiply-high and rejection sampling.
    fn below(&mut self, upper: u64) -> u64 {
        debug_assert!(upper > 0);
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let product = (self.next_u64() as u128) * (upper as u128);
            let low = product as u64;
            if low >= threshold {
                return (product >> 64) as u64;
            }
        }
    }
}

/// Box–Muller standard-normal generator with a separate deterministic stream.
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

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }

        let radius = (-2.0 * self.rng.unit_interval_open().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.rng.unit_interval_open();
        let (sin, cos) = angle.sin_cos();
        self.spare = Some(radius * sin);
        radius * cos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(signal_trials: usize, noise_trials: usize) -> SimulationDesign {
        SimulationDesign {
            signal_trials,
            noise_trials,
            seed: 0x5EED_2026,
        }
    }

    #[test]
    fn same_seed_replays_candidate_and_evaluator_artifacts() {
        let model = GaussianSignalModel::new(1.0, 0.25).unwrap();
        let first = simulate(model, design(2_000, 2_000)).unwrap();
        let second = simulate(model, design(2_000, 2_000)).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            evaluate(&first.candidate, &first.evaluator).unwrap(),
            evaluate(&second.candidate, &second.evaluator).unwrap()
        );
    }

    #[test]
    fn criterion_changes_binary_calls_not_score_ranking_or_auc() {
        let loose_model = GaussianSignalModel::new(1.0, -1.0).unwrap();
        let strict_model = GaussianSignalModel::new(1.0, 1.0).unwrap();
        let loose = simulate(loose_model, design(8_000, 8_000)).unwrap();
        let strict = simulate(strict_model, design(8_000, 8_000)).unwrap();

        assert_eq!(loose.candidate.trials.len(), strict.candidate.trials.len());
        assert!(loose
            .candidate
            .trials
            .iter()
            .zip(strict.candidate.trials.iter())
            .all(|(a, b)| a.trial_id == b.trial_id && a.evidence_score == b.evidence_score));
        assert!(loose
            .candidate
            .trials
            .iter()
            .zip(strict.candidate.trials.iter())
            .any(|(a, b)| a.response != b.response));

        let loose_report = evaluate(&loose.candidate, &loose.evaluator).unwrap();
        let strict_report = evaluate(&strict.candidate, &strict.evaluator).unwrap();
        assert_eq!(loose_report.roc_auc, strict_report.roc_auc);
        assert!((loose_report.signal_detection.d_prime
            - strict_report.signal_detection.d_prime)
            .abs()
            < 0.10);
        assert!((loose_report.signal_detection.criterion_c + 1.0).abs() < 0.12);
        assert!((strict_report.signal_detection.criterion_c - 1.0).abs() < 0.12);
    }

    #[test]
    fn sensitivity_ladder_is_monotone_under_common_random_numbers() {
        let mut previous_auc = 0.0;
        let mut previous_d_prime = f64::NEG_INFINITY;
        for injected in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0] {
            let model = GaussianSignalModel::new(injected, 0.0).unwrap();
            let bundle = simulate(model, design(6_000, 6_000)).unwrap();
            let report = evaluate(&bundle.candidate, &bundle.evaluator).unwrap();
            assert!(report.roc_auc > previous_auc || injected == 0.0);
            assert!(report.signal_detection.d_prime >= previous_d_prime);
            assert!((report.roc_auc - report.expected_roc_auc).abs() < 0.025);
            previous_auc = report.roc_auc;
            previous_d_prime = report.signal_detection.d_prime;
        }
    }

    #[test]
    fn analytic_operating_characteristics_match_generated_rates() {
        let model = GaussianSignalModel::new(1.0, 0.5).unwrap();
        let bundle = simulate(model, design(8_000, 8_000)).unwrap();
        let report = evaluate(&bundle.candidate, &bundle.evaluator).unwrap();

        assert!((report.signal_detection.hit_rate - report.expected_hit_rate).abs() < 0.025);
        assert!(
            (report.signal_detection.false_alarm_rate - report.expected_false_alarm_rate).abs()
                < 0.025
        );
        assert!((report.roc_auc - report.expected_roc_auc).abs() < 0.02);
        assert!((report.signal_detection.d_prime - 1.0).abs() < 0.10);
        assert!((report.signal_detection.criterion_c - 0.5).abs() < 0.10);
    }

    #[test]
    fn rare_event_track_reports_prevalence_baseline_for_random_ranking() {
        let model = GaussianSignalModel::new(0.0, 0.0).unwrap();
        let bundle = simulate(model, design(600, 11_400)).unwrap();
        let report = evaluate(&bundle.candidate, &bundle.evaluator).unwrap();

        assert!((report.prevalence - 0.05).abs() < 1e-12);
        assert!((report.average_precision - report.prevalence).abs() < 0.01);
        assert!((report.roc_auc - 0.5).abs() < 0.02);
        assert!((report.random_ranking_ap_baseline - 0.05).abs() < 1e-12);
    }

    #[test]
    fn binary_accuracy_is_prevalence_and_criterion_sensitive() {
        let design = design(2_000, 18_000);
        let balanced_model = GaussianSignalModel::new(1.0, 0.0).unwrap();
        let liberal_model = GaussianSignalModel::new(1.0, -1.0).unwrap();
        let balanced = simulate(balanced_model, design).unwrap();
        let liberal = simulate(liberal_model, design).unwrap();

        let balanced_report = evaluate(&balanced.candidate, &balanced.evaluator).unwrap();
        let liberal_report = evaluate(&liberal.candidate, &liberal.evaluator).unwrap();
        assert_eq!(balanced_report.roc_auc, liberal_report.roc_auc);
        assert_ne!(balanced_report.accuracy, liberal_report.accuracy);
        assert_ne!(
            balanced_report.signal_detection.hit_rate,
            liberal_report.signal_detection.hit_rate
        );
    }

    #[test]
    fn auc_is_prevalence_invariant_while_ap_baseline_is_not() {
        let model = GaussianSignalModel::new(1.0, 0.0).unwrap();
        let balanced = simulate(model, design(600, 600)).unwrap();
        let rare = simulate(model, design(600, 11_400)).unwrap();

        let balanced_report = evaluate(&balanced.candidate, &balanced.evaluator).unwrap();
        let rare_report = evaluate(&rare.candidate, &rare.evaluator).unwrap();

        assert!((balanced_report.roc_auc - rare_report.roc_auc).abs() < 0.02);
        assert!((balanced_report.random_ranking_ap_baseline - 0.5).abs() < 1e-12);
        assert!((rare_report.random_ranking_ap_baseline - 0.05).abs() < 1e-12);
        assert!(balanced_report.average_precision > rare_report.average_precision);
    }

    #[test]
    fn candidate_and_evaluator_records_must_remain_aligned() {
        let model = GaussianSignalModel::new(0.5, 0.0).unwrap();
        let mut bundle = simulate(model, design(16, 16)).unwrap();
        bundle.candidate.trials[0].trial_id += 1;
        assert_eq!(
            evaluate(&bundle.candidate, &bundle.evaluator),
            Err(CalibrationError::MisalignedRecords)
        );
    }

    #[test]
    fn non_finite_candidate_scores_fail_closed() {
        let model = GaussianSignalModel::new(0.5, 0.0).unwrap();
        let mut bundle = simulate(model, design(16, 16)).unwrap();
        bundle.candidate.trials[0].evidence_score = f64::NAN;
        assert_eq!(
            evaluate(&bundle.candidate, &bundle.evaluator),
            Err(CalibrationError::NonFiniteScore)
        );
    }

    #[test]
    fn invalid_models_and_empty_classes_are_rejected() {
        assert_eq!(
            GaussianSignalModel::new(f64::NAN, 0.0),
            Err(CalibrationError::InvalidModel)
        );
        assert_eq!(
            GaussianSignalModel::new(9.0, 0.0),
            Err(CalibrationError::InvalidModel)
        );
        let model = GaussianSignalModel::new(0.0, 0.0).unwrap();
        assert_eq!(
            simulate(model, design(0, 10)),
            Err(CalibrationError::EmptyClass)
        );
    }

    #[test]
    fn reversed_ranking_control_has_auc_below_chance() {
        let model = GaussianSignalModel::new(-1.0, 0.0).unwrap();
        let bundle = simulate(model, design(6_000, 6_000)).unwrap();
        let report = evaluate(&bundle.candidate, &bundle.evaluator).unwrap();
        assert!(report.roc_auc < 0.5);
        assert!(report.signal_detection.d_prime < 0.0);
    }
}
