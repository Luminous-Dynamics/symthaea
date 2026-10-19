// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic stability monitoring for FDM metrology evidence.
//!
//! This module asks a different question from one-time calibration: when does
//! the measurement process stop being trustworthy, and what claims must be
//! withdrawn once a monitor signals? Policies see only sequential check-standard
//! residuals and declared nominal uncertainty. The stress scenario is evaluator
//! metadata and never an input to the monitor.
//!
//! All readings, standards, and drift profiles are synthetic. No device I/O,
//! printer control, real calibration certificate, or physical-quality claim is
//! provided here.

use crate::crypto_digest::sha256;
use serde::{Deserialize, Serialize};

pub const FDM_STABILITY_SCHEMA: &str = "symthaea.fdm-metrology-stability-synthetic.v1";
pub const STABILITY_SERIES_LENGTH: usize = 48;
pub const STABILITY_CHANGE_STEP: usize = 18;
const DECLARED_SIGMA_MM: f64 = 0.006;
const EWMA_LAMBDA: f64 = 0.20;
const CUSUM_REFERENCE_K: f64 = 0.5 * DECLARED_SIGMA_MM;
const CUSUM_DECISION_H: f64 = 8.0 * DECLARED_SIGMA_MM;
const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilityScenario {
    StableProcess,
    AbruptBiasShift,
    LinearBiasDrift,
    GainChangeAcrossRange,
    VarianceInflation,
    TransientOutlier,
}

impl StabilityScenario {
    pub fn name(self) -> &'static str {
        match self {
            Self::StableProcess => "stable_process",
            Self::AbruptBiasShift => "abrupt_bias_shift",
            Self::LinearBiasDrift => "linear_bias_drift",
            Self::GainChangeAcrossRange => "gain_change_across_range",
            Self::VarianceInflation => "variance_inflation",
            Self::TransientOutlier => "transient_outlier",
        }
    }

    pub fn all() -> [Self; 6] {
        [
            Self::StableProcess,
            Self::AbruptBiasShift,
            Self::LinearBiasDrift,
            Self::GainChangeAcrossRange,
            Self::VarianceInflation,
            Self::TransientOutlier,
        ]
    }

    fn change_step(self) -> Option<usize> {
        match self {
            Self::StableProcess | Self::TransientOutlier => None,
            Self::AbruptBiasShift
            | Self::LinearBiasDrift
            | Self::GainChangeAcrossRange
            | Self::VarianceInflation => Some(STABILITY_CHANGE_STEP),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilityMonitorPolicy {
    /// Negative control: never inspect the stream and never withdraw trust.
    NoMonitoring,
    /// Quarantine on any individual residual beyond the 3-sigma point limit.
    Shewhart,
    /// Quarantine when an exponentially weighted residual exceeds its limit.
    Ewma,
    /// Two-sided cumulative-sum monitor for persistent small shifts.
    Cusum,
    /// Warning on one point outlier; quarantine on persistent evidence.
    HybridPersistenceGuard,
}

impl StabilityMonitorPolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::NoMonitoring => "no_monitoring",
            Self::Shewhart => "shewhart",
            Self::Ewma => "ewma",
            Self::Cusum => "cusum",
            Self::HybridPersistenceGuard => "hybrid_persistence_guard",
        }
    }

    pub fn all() -> [Self; 5] {
        [
            Self::NoMonitoring,
            Self::Shewhart,
            Self::Ewma,
            Self::Cusum,
            Self::HybridPersistenceGuard,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustDisposition {
    Trusted,
    Warning,
    Quarantined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilitySignal {
    None,
    SinglePointOutlier,
    EwmaLimit,
    CusumLimit,
    PersistenceRule,
    QuarantineRemainsActive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckStandardObservation {
    pub observation_id: String,
    pub sequence_index: usize,
    pub check_standard_id: String,
    pub nominal_value_mm: f64,
    pub measured_value_mm: f64,
    pub residual_mm: f64,
    /// Declared baseline standard uncertainty; this is not re-estimated from
    /// the same observation that is being assessed.
    pub declared_sigma_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityDecision {
    pub sequence_index: usize,
    pub disposition: TrustDisposition,
    pub signal: StabilitySignal,
    pub ewma_residual_mm: f64,
    pub positive_cusum_mm: f64,
    pub negative_cusum_mm: f64,
    pub point_limit_mm: f64,
    pub ewma_limit_mm: f64,
    pub cusum_limit_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityMetrics {
    pub observation_count: usize,
    pub trusted_count: usize,
    pub warning_count: usize,
    pub quarantined_count: usize,
    pub first_alert_step: Option<usize>,
    pub first_quarantine_step: Option<usize>,
    pub expected_change_step: Option<usize>,
    pub detection_delay_steps: Option<usize>,
    /// Count of post-change samples still marked Trusted. Warning and
    /// Quarantined samples are not counted as admissible evidence.
    pub trusted_after_change_count: usize,
    pub post_change_observation_count: usize,
    pub unsafe_trust_fraction_after_change: Option<f64>,
    pub false_quarantine_on_stable_process: bool,
    pub quarantine_on_single_transient_outlier: bool,
    pub mean_absolute_residual_mm: f64,
    pub maximum_absolute_residual_mm: f64,
    pub terminal_disposition: TrustDisposition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityReport {
    pub schema_version: String,
    pub evidence_class: String,
    pub scenario: StabilityScenario,
    pub policy: StabilityMonitorPolicy,
    pub seed: u64,
    pub declared_sigma_mm: f64,
    pub observations: Vec<CheckStandardObservation>,
    pub decisions: Vec<StabilityDecision>,
    pub metrics: StabilityMetrics,
    pub report_digest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StabilityError {
    InvalidReport(String),
    SerializationFailure,
}

impl std::fmt::Display for StabilityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidReport(reason) => {
                write!(formatter, "invalid metrology-stability report: {reason}")
            }
            Self::SerializationFailure => {
                formatter.write_str("could not serialize metrology-stability evidence")
            }
        }
    }
}

impl std::error::Error for StabilityError {}

#[derive(Debug, Clone, Copy, Default)]
struct ChartState {
    ewma_mm: f64,
    positive_cusum_mm: f64,
    negative_cusum_mm: f64,
}

fn deterministic_unit_noise(seed: u64, sequence_index: usize, standard_index: usize) -> f64 {
    let mut value = seed
        ^ (sequence_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (standard_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ 0xA24B_AED4_963E_E407;
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    let unit = (value >> 11) as f64 / 9_007_199_254_740_992.0;
    2.0 * unit - 1.0
}

fn make_observation(
    scenario: StabilityScenario,
    seed: u64,
    sequence_index: usize,
) -> CheckStandardObservation {
    // Three check standards span the measurement range. This prevents a
    // single zero-point check from concealing every range-dependent failure.
    let standards = [10.0, 20.0, 30.0];
    let standard_index = sequence_index % standards.len();
    let nominal = standards[standard_index];
    let mut noise_scale = DECLARED_SIGMA_MM;
    let bias = match scenario {
        StabilityScenario::StableProcess | StabilityScenario::TransientOutlier => 0.0,
        StabilityScenario::AbruptBiasShift if sequence_index >= STABILITY_CHANGE_STEP => 0.035,
        StabilityScenario::AbruptBiasShift => 0.0,
        StabilityScenario::LinearBiasDrift if sequence_index >= STABILITY_CHANGE_STEP => {
            (sequence_index - STABILITY_CHANGE_STEP + 1) as f64 * 0.0015
        }
        StabilityScenario::LinearBiasDrift => 0.0,
        StabilityScenario::GainChangeAcrossRange if sequence_index >= STABILITY_CHANGE_STEP => {
            (nominal - 20.0) * 0.003
        }
        StabilityScenario::GainChangeAcrossRange => 0.0,
        StabilityScenario::VarianceInflation => {
            if sequence_index >= STABILITY_CHANGE_STEP {
                noise_scale *= 5.0;
            }
            0.0
        }
    };
    let outlier = if scenario == StabilityScenario::TransientOutlier
        && sequence_index == STABILITY_CHANGE_STEP
    {
        0.040
    } else {
        0.0
    };
    let noise = deterministic_unit_noise(seed, sequence_index, standard_index) * noise_scale;
    let residual = bias + outlier + noise;
    CheckStandardObservation {
        observation_id: format!("check-obs-{sequence_index:03}"),
        sequence_index,
        check_standard_id: format!("reference-{standard_index}"),
        nominal_value_mm: nominal,
        measured_value_mm: nominal + residual,
        residual_mm: residual,
        declared_sigma_mm: DECLARED_SIGMA_MM,
    }
}

fn chart_step(previous: ChartState, residual_mm: f64) -> ChartState {
    ChartState {
        ewma_mm: EWMA_LAMBDA * residual_mm + (1.0 - EWMA_LAMBDA) * previous.ewma_mm,
        positive_cusum_mm: (previous.positive_cusum_mm + residual_mm - CUSUM_REFERENCE_K)
            .max(0.0),
        negative_cusum_mm: (previous.negative_cusum_mm + residual_mm + CUSUM_REFERENCE_K)
            .min(0.0),
    }
}

fn hybrid_persistent_run(observations: &[CheckStandardObservation]) -> bool {
    let recent = observations.iter().rev().take(3);
    let outliers = recent
        .filter(|item| item.residual_mm.abs() > 3.0 * item.declared_sigma_mm)
        .count();
    outliers >= 2
}

fn decide(
    policy: StabilityMonitorPolicy,
    observation: &CheckStandardObservation,
    history: &[CheckStandardObservation],
    chart: ChartState,
    already_quarantined: bool,
) -> (TrustDisposition, StabilitySignal) {
    if already_quarantined {
        return (
            TrustDisposition::Quarantined,
            StabilitySignal::QuarantineRemainsActive,
        );
    }
    if policy == StabilityMonitorPolicy::NoMonitoring {
        return (TrustDisposition::Trusted, StabilitySignal::None);
    }

    let sigma = observation.declared_sigma_mm;
    let point_limit = 3.0 * sigma;
    let ewma_limit = 4.0 * sigma * (EWMA_LAMBDA / (2.0 - EWMA_LAMBDA)).sqrt();
    let cusum_limit = CUSUM_DECISION_H;

    match policy {
        StabilityMonitorPolicy::NoMonitoring => (TrustDisposition::Trusted, StabilitySignal::None),
        StabilityMonitorPolicy::Shewhart => {
            if observation.residual_mm.abs() > point_limit {
                (
                    TrustDisposition::Quarantined,
                    StabilitySignal::SinglePointOutlier,
                )
            } else {
                (TrustDisposition::Trusted, StabilitySignal::None)
            }
        }
        StabilityMonitorPolicy::Ewma => {
            if chart.ewma_mm.abs() > ewma_limit {
                (TrustDisposition::Quarantined, StabilitySignal::EwmaLimit)
            } else {
                (TrustDisposition::Trusted, StabilitySignal::None)
            }
        }
        StabilityMonitorPolicy::Cusum => {
            if chart.positive_cusum_mm > cusum_limit
                || chart.negative_cusum_mm < -cusum_limit
            {
                (TrustDisposition::Quarantined, StabilitySignal::CusumLimit)
            } else {
                (TrustDisposition::Trusted, StabilitySignal::None)
            }
        }
        StabilityMonitorPolicy::HybridPersistenceGuard => {
            let cusum_signal = chart.positive_cusum_mm > cusum_limit
                || chart.negative_cusum_mm < -cusum_limit;
            if cusum_signal {
                return (
                    TrustDisposition::Quarantined,
                    StabilitySignal::CusumLimit,
                );
            }
            if hybrid_persistent_run(history) {
                return (
                    TrustDisposition::Quarantined,
                    StabilitySignal::PersistenceRule,
                );
            }
            if observation.residual_mm.abs() > point_limit {
                (TrustDisposition::Warning, StabilitySignal::SinglePointOutlier)
            } else {
                (TrustDisposition::Trusted, StabilitySignal::None)
            }
        }
    }
}

fn report_digest(report: &StabilityReport) -> Result<String, StabilityError> {
    let bytes = serde_json::to_vec(&(
        &report.schema_version,
        &report.evidence_class,
        report.scenario,
        report.policy,
        report.seed,
        report.declared_sigma_mm,
        &report.observations,
        &report.decisions,
        &report.metrics,
    ))
    .map_err(|_| StabilityError::SerializationFailure)?;
    let domain = b"symthaea.fdm-metrology-stability-report.v1\0";
    let mut input = Vec::with_capacity(domain.len() + bytes.len());
    input.extend_from_slice(domain);
    input.extend_from_slice(&bytes);
    Ok(sha256(&input).to_hex())
}

fn build_metrics(
    scenario: StabilityScenario,
    observations: &[CheckStandardObservation],
    decisions: &[StabilityDecision],
) -> StabilityMetrics {
    let trusted_count = decisions
        .iter()
        .filter(|item| item.disposition == TrustDisposition::Trusted)
        .count();
    let warning_count = decisions
        .iter()
        .filter(|item| item.disposition == TrustDisposition::Warning)
        .count();
    let quarantined_count = decisions
        .iter()
        .filter(|item| item.disposition == TrustDisposition::Quarantined)
        .count();
    let first_alert_step = decisions
        .iter()
        .find(|item| item.disposition != TrustDisposition::Trusted)
        .map(|item| item.sequence_index);
    let first_quarantine_step = decisions
        .iter()
        .find(|item| item.disposition == TrustDisposition::Quarantined)
        .map(|item| item.sequence_index);
    let expected_change_step = scenario.change_step();
    let detection_delay_steps = match (expected_change_step, first_quarantine_step) {
        (Some(change), Some(detection)) if detection >= change => Some(detection - change),
        _ => None,
    };
    let post_change_observation_count = expected_change_step.map_or(0, |change| {
        observations
            .iter()
            .filter(|item| item.sequence_index >= change)
            .count()
    });
    let trusted_after_change_count = expected_change_step.map_or(0, |change| {
        decisions
            .iter()
            .filter(|item| {
                item.sequence_index >= change
                    && item.disposition == TrustDisposition::Trusted
            })
            .count()
    });
    let unsafe_trust_fraction_after_change = (post_change_observation_count > 0).then(|| {
        trusted_after_change_count as f64 / post_change_observation_count as f64
    });
    let sum_absolute = observations
        .iter()
        .map(|item| item.residual_mm.abs())
        .sum::<f64>();
    let maximum_absolute = observations
        .iter()
        .map(|item| item.residual_mm.abs())
        .fold(0.0_f64, f64::max);
    let terminal_disposition = decisions
        .last()
        .map(|item| item.disposition)
        .unwrap_or(TrustDisposition::Warning);
    StabilityMetrics {
        observation_count: observations.len(),
        trusted_count,
        warning_count,
        quarantined_count,
        first_alert_step,
        first_quarantine_step,
        expected_change_step,
        detection_delay_steps,
        trusted_after_change_count,
        post_change_observation_count,
        unsafe_trust_fraction_after_change,
        false_quarantine_on_stable_process: scenario == StabilityScenario::StableProcess
            && first_quarantine_step.is_some(),
        quarantine_on_single_transient_outlier: scenario == StabilityScenario::TransientOutlier
            && first_quarantine_step.is_some(),
        mean_absolute_residual_mm: if observations.is_empty() {
            0.0
        } else {
            sum_absolute / observations.len() as f64
        },
        maximum_absolute_residual_mm: maximum_absolute,
        terminal_disposition,
    }
}

pub fn run_stability_arm(
    scenario: StabilityScenario,
    policy: StabilityMonitorPolicy,
    seed: u64,
) -> Result<StabilityReport, StabilityError> {
    let observations: Vec<_> = (0..STABILITY_SERIES_LENGTH)
        .map(|index| make_observation(scenario, seed, index))
        .collect();
    let mut decisions = Vec::with_capacity(observations.len());
    let mut chart = ChartState::default();
    let mut quarantined = false;

    for observation in &observations {
        chart = chart_step(chart, observation.residual_mm);
        let (disposition, signal) = decide(
            policy,
            observation,
            &observations[..=observation.sequence_index],
            chart,
            quarantined,
        );
        quarantined |= disposition == TrustDisposition::Quarantined;
        let ewma_limit =
            4.0 * observation.declared_sigma_mm * (EWMA_LAMBDA / (2.0 - EWMA_LAMBDA)).sqrt();
        decisions.push(StabilityDecision {
            sequence_index: observation.sequence_index,
            disposition,
            signal,
            ewma_residual_mm: chart.ewma_mm,
            positive_cusum_mm: chart.positive_cusum_mm,
            negative_cusum_mm: chart.negative_cusum_mm,
            point_limit_mm: 3.0 * observation.declared_sigma_mm,
            ewma_limit_mm: ewma_limit,
            cusum_limit_mm: CUSUM_DECISION_H,
        });
    }

    let metrics = build_metrics(scenario, &observations, &decisions);
    let mut report = StabilityReport {
        schema_version: FDM_STABILITY_SCHEMA.to_owned(),
        evidence_class: "synthetic".to_owned(),
        scenario,
        policy,
        seed,
        declared_sigma_mm: DECLARED_SIGMA_MM,
        observations,
        decisions,
        metrics,
        report_digest_sha256: String::new(),
    };
    report.report_digest_sha256 = report_digest(&report)?;
    Ok(report)
}

pub fn run_stability_benchmark(seed: u64) -> Result<Vec<StabilityReport>, StabilityError> {
    let mut reports = Vec::with_capacity(
        StabilityScenario::all().len() * StabilityMonitorPolicy::all().len(),
    );
    for scenario in StabilityScenario::all() {
        for policy in StabilityMonitorPolicy::all() {
            reports.push(run_stability_arm(scenario, policy, seed)?);
        }
    }
    Ok(reports)
}

/// Fail-closed exact verifier: regenerate every observation and monitor decision
/// from the declared scenario, seed and policy, then compare the entire report.
pub fn verify_stability_report(report: &StabilityReport) -> Result<(), StabilityError> {
    if report.schema_version != FDM_STABILITY_SCHEMA {
        return Err(StabilityError::InvalidReport(
            "unsupported schema version".to_owned(),
        ));
    }
    if report.evidence_class != "synthetic" {
        return Err(StabilityError::InvalidReport(
            "evidence class must remain synthetic".to_owned(),
        ));
    }
    if report.observations.len() != STABILITY_SERIES_LENGTH
        || report.decisions.len() != STABILITY_SERIES_LENGTH
    {
        return Err(StabilityError::InvalidReport(
            "trace length does not match the frozen protocol".to_owned(),
        ));
    }
    if !report.declared_sigma_mm.is_finite()
        || (report.declared_sigma_mm - DECLARED_SIGMA_MM).abs() > EPSILON
    {
        return Err(StabilityError::InvalidReport(
            "declared uncertainty does not match the frozen protocol".to_owned(),
        ));
    }
    let expected = run_stability_arm(report.scenario, report.policy, report.seed)?;
    if report != &expected {
        return Err(StabilityError::InvalidReport(
            "measurements, decisions, metrics, or digest do not replay".to_owned(),
        ));
    }
    if report.report_digest_sha256 != report_digest(report)? {
        return Err(StabilityError::InvalidReport(
            "report digest mismatch".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_scenario_policy_arms_replay_and_keep_synthetic_evidence_label() {
        let reports = run_stability_benchmark(42).unwrap();
        assert_eq!(
            reports.len(),
            StabilityScenario::all().len() * StabilityMonitorPolicy::all().len()
        );
        for report in &reports {
            assert_eq!(report.evidence_class, "synthetic");
            assert_eq!(report.observations.len(), STABILITY_SERIES_LENGTH);
            assert_eq!(report.decisions.len(), STABILITY_SERIES_LENGTH);
            assert_eq!(report.metrics.observation_count, STABILITY_SERIES_LENGTH);
            verify_stability_report(report).unwrap();
        }
    }

    #[test]
    fn stable_stream_does_not_quarantine_for_monitored_policies() {
        for policy in [
            StabilityMonitorPolicy::Shewhart,
            StabilityMonitorPolicy::Ewma,
            StabilityMonitorPolicy::Cusum,
            StabilityMonitorPolicy::HybridPersistenceGuard,
        ] {
            let report = run_stability_arm(StabilityScenario::StableProcess, policy, 42).unwrap();
            assert!(
                !report.metrics.false_quarantine_on_stable_process,
                "{} quarantined a stable stream",
                policy.name()
            );
        }
    }

    #[test]
    fn unmonitored_negative_control_keeps_trusting_shifted_measurements() {
        let report = run_stability_arm(
            StabilityScenario::AbruptBiasShift,
            StabilityMonitorPolicy::NoMonitoring,
            42,
        )
        .unwrap();
        assert_eq!(report.metrics.first_quarantine_step, None);
        assert_eq!(report.metrics.trusted_after_change_count, 30);
        assert_eq!(report.metrics.unsafe_trust_fraction_after_change, Some(1.0));
    }

    #[test]
    fn sequential_monitors_detect_abrupt_bias_without_receiving_scenario_label() {
        for policy in [
            StabilityMonitorPolicy::Shewhart,
            StabilityMonitorPolicy::Ewma,
            StabilityMonitorPolicy::Cusum,
            StabilityMonitorPolicy::HybridPersistenceGuard,
        ] {
            let report = run_stability_arm(
                StabilityScenario::AbruptBiasShift,
                policy,
                42,
            )
            .unwrap();
            assert!(
                report.metrics.first_quarantine_step.is_some(),
                "{} failed to quarantine the shifted stream",
                policy.name()
            );
            assert!(
                report.metrics.first_quarantine_step.unwrap() >= STABILITY_CHANGE_STEP,
                "monitor quarantined before the injected change",
            );
            assert!(
                report.metrics.trusted_after_change_count < report.metrics.post_change_observation_count,
                "{} left every post-change reading trusted",
                policy.name()
            );
        }
    }

    #[test]
    fn hybrid_policy_warns_on_single_outlier_without_immediate_quarantine() {
        let report = run_stability_arm(
            StabilityScenario::TransientOutlier,
            StabilityMonitorPolicy::HybridPersistenceGuard,
            42,
        )
        .unwrap();
        assert_eq!(
            report.decisions[STABILITY_CHANGE_STEP].disposition,
            TrustDisposition::Warning
        );
        assert_eq!(report.metrics.first_quarantine_step, None);
        assert!(report.metrics.quarantine_on_single_transient_outlier == false);
    }

    #[test]
    fn range_gain_and_variance_change_are_not_equivalent_to_constant_offset() {
        let offset = run_stability_arm(
            StabilityScenario::AbruptBiasShift,
            StabilityMonitorPolicy::HybridPersistenceGuard,
            7,
        )
        .unwrap();
        let gain = run_stability_arm(
            StabilityScenario::GainChangeAcrossRange,
            StabilityMonitorPolicy::HybridPersistenceGuard,
            7,
        )
        .unwrap();
        let variance = run_stability_arm(
            StabilityScenario::VarianceInflation,
            StabilityMonitorPolicy::HybridPersistenceGuard,
            7,
        )
        .unwrap();
        let stable = run_stability_arm(
            StabilityScenario::StableProcess,
            StabilityMonitorPolicy::HybridPersistenceGuard,
            7,
        )
        .unwrap();
        assert_ne!(
            offset.observations[20].residual_mm,
            gain.observations[20].residual_mm
        );
        assert!(
            variance.metrics.maximum_absolute_residual_mm
                > stable.metrics.maximum_absolute_residual_mm
        );
    }

    #[test]
    fn tampered_observations_decisions_metrics_and_digests_fail_closed() {
        let mut report = run_stability_arm(
            StabilityScenario::AbruptBiasShift,
            StabilityMonitorPolicy::HybridPersistenceGuard,
            5,
        )
        .unwrap();
        report.observations[20].residual_mm += 0.01;
        assert!(verify_stability_report(&report).is_err());

        let mut report = run_stability_arm(
            StabilityScenario::StableProcess,
            StabilityMonitorPolicy::Shewhart,
            5,
        )
        .unwrap();
        report.metrics.trusted_count -= 1;
        assert!(verify_stability_report(&report).is_err());

        let mut report = run_stability_arm(
            StabilityScenario::StableProcess,
            StabilityMonitorPolicy::Shewhart,
            5,
        )
        .unwrap();
        report.decisions[0].disposition = TrustDisposition::Quarantined;
        assert!(verify_stability_report(&report).is_err());
    }

    #[test]
    fn repeated_runs_are_bitwise_structurally_deterministic() {
        let first = run_stability_benchmark(9001).unwrap();
        let second = run_stability_benchmark(9001).unwrap();
        assert_eq!(first, second);
    }
}
