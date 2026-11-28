// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Selective-observation weighted risk control (SOWRC), fixed-horizon v1.
//!
//! This module implements the observable martingale-mixture upper boundary
//! described by Bai, Fang and Chen (Axioms 2026, 15(9), 706). It assumes:
//! predictable pre-label inclusion probabilities q_t >= q_min > 0,
//! conditional ignorability, bounded nonconformity scores, a frozen prediction
//! artifact, and deterministic fixed-horizon exposure envelopes.
//!
//! The implementation accepts externally observed labels and q_t values. It
//! does not generate audit randomness, authenticate the audit custodian,
//! verify conditional ignorability, certify deployment drift, or establish
//! an anytime-valid guarantee. Only fixed-horizon calibration-population
//! control is in scope.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const SOWRC_METRIC_VERSION: &str = "symthaea-sowrc-fixed-horizon-v1";
const ETA_FACTORS: [f64; 6] = [0.05, 0.15, 0.30, 0.50, 0.75, 0.90];
const BISECTION_STEPS: usize = 80;
const SUM_TOLERANCE: f64 = 1e-10;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SowrcObservation {
    /// Strictly ordered zero-based event index in the frozen calibration horizon.
    pub sequence_number: usize,
    /// Stable identity in the frozen prediction and audit manifests.
    pub record_id: String,
    /// Conditional probability that an independent label is observed for this
    /// event, given pre-label history. This is q_t, not merely an audit quota.
    pub inclusion_probability: f64,
    /// True when an alert or randomized audit actually produced an independent label.
    pub label_observed: bool,
    /// Nonconformity of the independently observed true label, computed from
    /// the frozen pre-label class probabilities: r_t(X_t,Y_t)=1-p_t(Y_t|X_t).
    /// It is absent precisely when the label was not observed.
    pub true_label_nonconformity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SowrcConfig {
    /// Target calibration-population weighted miscoverage alpha.
    pub target_risk: f64,
    /// Total failure probability allocated across the fixed threshold grid.
    pub delta: f64,
    /// Externally guaranteed lower bound for every pre-label q_t.
    pub q_min: f64,
    /// Frozen nested-set grid. Must be strictly increasing and terminate at 1.
    pub threshold_grid: Vec<f64>,
    /// Fixed positive betting fractions, all satisfying kappa(eta)>0.
    pub betting_fractions: Vec<f64>,
    /// Fixed positive mixture weights summing to one.
    pub mixture_weights: Vec<f64>,
    /// Reference to the custodian-controlled predictable inclusion design.
    pub design_manifest_ref: String,
    /// Reference to frozen pre-label class probabilities / prediction outputs.
    pub prediction_manifest_ref: String,
}

impl SowrcConfig {
    /// Conservative default fixed-horizon design for an externally supplied
    /// inclusion floor. The betting grid scales with q_min but is fully fixed
    /// before labels are inspected.
    pub fn for_audit_floor(
        q_min: f64,
        design_manifest_ref: impl Into<String>,
        prediction_manifest_ref: impl Into<String>,
    ) -> Self {
        let eta_max = 2.0 * q_min / (1.0 + 2.0 * q_min / 3.0);
        let betting_fractions = ETA_FACTORS
            .iter()
            .map(|factor| eta_max * factor)
            .collect::<Vec<_>>();
        let mixture_weights = vec![1.0 / betting_fractions.len() as f64; betting_fractions.len()];
        Self {
            target_risk: 0.10,
            delta: 0.10,
            q_min,
            threshold_grid: (0..=10).map(|index| index as f64 / 10.0).collect(),
            betting_fractions,
            mixture_weights,
            design_manifest_ref: design_manifest_ref.into(),
            prediction_manifest_ref: prediction_manifest_ref.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SowrcThresholdBound {
    pub lambda: f64,
    pub allocated_delta: f64,
    /// Sum_t I_t * 1{r_t > lambda} / q_t. This is not clipped.
    pub horvitz_thompson_loss_total: f64,
    /// HT total divided by the fixed horizon; this is descriptive, not the risk bound.
    pub empirical_weighted_risk: f64,
    /// Mixture-inverted upper bound for the conditional weighted calibration risk.
    pub upper_risk_bound: f64,
    pub log_mixture_at_zero: f64,
    pub log_mixture_at_one: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SowrcSelectionStatus {
    CertifiedByMixture,
    SafeTerminalFullLabelSet,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SowrcReport {
    pub metric_version: String,
    pub calibration_horizon: usize,
    pub observed_label_count: usize,
    pub q_min: f64,
    pub minimum_observed_inclusion_probability: f64,
    pub target_risk: f64,
    pub total_delta: f64,
    pub per_threshold_delta: f64,
    pub design_manifest_ref: String,
    pub prediction_manifest_ref: String,
    pub threshold_bounds: Vec<SowrcThresholdBound>,
    pub selected_lambda: f64,
    /// The upper risk certificate for the selected rule. A fallback to lambda=1
    /// has exact zero miscoverage because it returns the full label space.
    pub selected_risk_upper_bound: f64,
    pub selection_status: SowrcSelectionStatus,
    pub betting_fractions: Vec<f64>,
    pub mixture_weights: Vec<f64>,
    pub deterministic_c: f64,
    pub deterministic_b: f64,
    pub assumptions: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SowrcError {
    EmptyHorizon,
    EmptyRecordId,
    DuplicateRecordId(String),
    NonSequentialIndex { expected: usize, actual: usize },
    InvalidProbabilityFloor,
    InvalidInclusionProbability(String),
    InclusionProbabilityBelowFloor(String),
    CertainInclusionWithoutLabel(String),
    LabelScoreMismatch(String),
    InvalidNonconformityScore(String),
    InvalidTargetRisk,
    InvalidDelta,
    InvalidThresholdGrid,
    InvalidBettingMixture,
    InvalidDesignManifest,
    InvalidPredictionManifest,
    InvalidMixtureNumerics,
}

impl std::fmt::Display for SowrcError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyHorizon => write!(formatter, "SOWRC requires a nonempty fixed horizon"),
            Self::EmptyRecordId => write!(formatter, "SOWRC record IDs must not be empty"),
            Self::DuplicateRecordId(id) => write!(formatter, "duplicate SOWRC record ID: {id}"),
            Self::NonSequentialIndex { expected, actual } => write!(
                formatter,
                "SOWRC sequence index mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidProbabilityFloor => {
                write!(formatter, "q_min must be finite and in (0, 1]")
            }
            Self::InvalidInclusionProbability(id) => {
                write!(formatter, "invalid inclusion probability for record {id}")
            }
            Self::InclusionProbabilityBelowFloor(id) => {
                write!(formatter, "inclusion probability is below q_min for record {id}")
            }
            Self::CertainInclusionWithoutLabel(id) => {
                write!(formatter, "q=1 but no label was observed for record {id}")
            }
            Self::LabelScoreMismatch(id) => {
                write!(formatter, "label-observed flag and nonconformity score disagree for {id}")
            }
            Self::InvalidNonconformityScore(id) => {
                write!(formatter, "nonconformity score must be finite and in [0, 1] for {id}")
            }
            Self::InvalidTargetRisk => {
                write!(formatter, "target risk must be finite and in [0, 1]")
            }
            Self::InvalidDelta => write!(formatter, "delta must be finite and in (0, 1)"),
            Self::InvalidThresholdGrid => write!(
                formatter,
                "threshold grid must be strictly increasing in [0, 1] and end at 1"
            ),
            Self::InvalidBettingMixture => {
                write!(formatter, "betting fractions and mixture weights are invalid")
            }
            Self::InvalidDesignManifest => {
                write!(formatter, "design manifest reference must not be empty")
            }
            Self::InvalidPredictionManifest => {
                write!(formatter, "prediction manifest reference must not be empty")
            }
            Self::InvalidMixtureNumerics => {
                write!(formatter, "martingale-mixture calculation produced non-finite values")
            }
        }
    }
}

impl std::error::Error for SowrcError {}

/// Validate a fixed-horizon SOWRC cohort and compute simultaneous risk bounds
/// over its predeclared nested nonconformity grid.
pub fn evaluate_sowrc(
    observations: &[SowrcObservation],
    config: &SowrcConfig,
) -> Result<SowrcReport, SowrcError> {
    validate_config(config)?;
    if observations.is_empty() {
        return Err(SowrcError::EmptyHorizon);
    }

    let mut ids = HashSet::with_capacity(observations.len());
    let mut observed_label_count = 0usize;
    let mut minimum_probability = 1.0_f64;
    for (index, observation) in observations.iter().enumerate() {
        if observation.record_id.trim().is_empty() {
            return Err(SowrcError::EmptyRecordId);
        }
        if !ids.insert(observation.record_id.as_str()) {
            return Err(SowrcError::DuplicateRecordId(
                observation.record_id.clone(),
            ));
        }
        if observation.sequence_number != index {
            return Err(SowrcError::NonSequentialIndex {
                expected: index,
                actual: observation.sequence_number,
            });
        }
        let q = observation.inclusion_probability;
        if !q.is_finite() || q <= 0.0 || q > 1.0 {
            return Err(SowrcError::InvalidInclusionProbability(
                observation.record_id.clone(),
            ));
        }
        if q + SUM_TOLERANCE < config.q_min {
            return Err(SowrcError::InclusionProbabilityBelowFloor(
                observation.record_id.clone(),
            ));
        }
        minimum_probability = minimum_probability.min(q);
        match (observation.label_observed, observation.true_label_nonconformity) {
            (true, Some(score)) => {
                if !score.is_finite() || !(0.0..=1.0).contains(&score) {
                    return Err(SowrcError::InvalidNonconformityScore(
                        observation.record_id.clone(),
                    ));
                }
                observed_label_count += 1;
            }
            (false, None) => {
                if q == 1.0 {
                    return Err(SowrcError::CertainInclusionWithoutLabel(
                        observation.record_id.clone(),
                    ));
                }
            }
            _ => {
                return Err(SowrcError::LabelScoreMismatch(
                    observation.record_id.clone(),
                ))
            }
        }
    }

    // v1 uses a_t = 1, so A_n = n, c_n = 1, and b_n = 1/q_min.
    let horizon = observations.len();
    let c_n = 1.0_f64;
    let b_n = 1.0 / config.q_min;
    let per_threshold_delta = config.delta / config.threshold_grid.len() as f64;
    let mut threshold_bounds = Vec::with_capacity(config.threshold_grid.len());

    for &lambda in &config.threshold_grid {
        let ht_loss_total = observations
            .iter()
            .filter_map(|observation| {
                let score = observation.true_label_nonconformity?;
                (observation.label_observed && score > lambda)
                    .then_some(1.0 / observation.inclusion_probability)
            })
            .sum::<f64>();
        let empirical_weighted_risk = ht_loss_total / horizon as f64;
        let (upper_risk_bound, log_mixture_at_zero, log_mixture_at_one) =
            invert_mixture_upper_bound(
                horizon,
                ht_loss_total,
                per_threshold_delta,
                c_n,
                b_n,
                &config.betting_fractions,
                &config.mixture_weights,
            )?;
        threshold_bounds.push(SowrcThresholdBound {
            lambda,
            allocated_delta: per_threshold_delta,
            horvitz_thompson_loss_total: ht_loss_total,
            empirical_weighted_risk,
            upper_risk_bound,
            log_mixture_at_zero,
            log_mixture_at_one,
        });
    }

    // Search only nonterminal thresholds first. Lambda=1 always returns the
    // entire label space, so the exact structural fallback is risk zero even
    // when the statistical mixture is too conservative to certify a smaller set.
    let selected = threshold_bounds
        .iter()
        .take(threshold_bounds.len().saturating_sub(1))
        .find(|report| report.upper_risk_bound <= config.target_risk);
    let (selected_lambda, selected_bound, selection_status) = match selected {
        Some(report) => (
            report.lambda,
            report.upper_risk_bound,
            SowrcSelectionStatus::CertifiedByMixture,
        ),
        None => (
            1.0,
            0.0,
            SowrcSelectionStatus::SafeTerminalFullLabelSet,
        ),
    };

    Ok(SowrcReport {
        metric_version: SOWRC_METRIC_VERSION.to_owned(),
        calibration_horizon: horizon,
        observed_label_count,
        q_min: config.q_min,
        minimum_observed_inclusion_probability: minimum_probability,
        target_risk: config.target_risk,
        total_delta: config.delta,
        per_threshold_delta,
        design_manifest_ref: config.design_manifest_ref.clone(),
        prediction_manifest_ref: config.prediction_manifest_ref.clone(),
        threshold_bounds,
        selected_lambda,
        selected_risk_upper_bound: selected_bound,
        selection_status,
        betting_fractions: config.betting_fractions.clone(),
        mixture_weights: config.mixture_weights.clone(),
        deterministic_c: c_n,
        deterministic_b: b_n,
        assumptions: vec![
            concat!(
                "Every inclusion probability q_t is known before the corresponding label ",
                "and is predictable from pre-label history."
            )
            .to_owned(),
            concat!(
                "Conditional ignorability holds: inclusion is independent of the current ",
                "true label conditional on pre-label history."
            )
            .to_owned(),
            concat!(
                "Every event has q_t >= q_min > 0; the lower support floor is fixed by ",
                "the external audit design."
            )
            .to_owned(),
            "The candidate probabilities and nonconformity function are frozen before labels are observed."
                .to_owned(),
            "The horizon and deterministic envelopes c_n=1 and b_n=1/q_min are fixed before labels."
                .to_owned(),
            concat!(
                "The loss is bounded in [0,1], nonincreasing in lambda, and lambda=1 ",
                "returns the full label space with zero miscoverage."
            )
            .to_owned(),
            concat!(
                "Threshold grid, betting fractions, mixture weights and delta allocation ",
                "are fixed independently of the audit labels."
            )
            .to_owned(),
        ],
        limitations: vec![
            concat!(
                "The evaluator cannot authenticate randomization, the custodian, or ",
                "conditional ignorability; those are external design and governance obligations."
            )
            .to_owned(),
            concat!(
                "The theorem controls the weighted calibration population only; prospective ",
                "deployment risk needs an independent drift certificate and a separate ",
                "failure-probability allocation."
            )
            .to_owned(),
            concat!(
                "This fixed-horizon implementation is not the anytime/time-uniform extension ",
                "and does not support adaptive budget changes."
            )
            .to_owned(),
            "Only constant predictable weights a_t=1 are implemented in v1.".to_owned(),
            concat!(
                "The result is a risk-control bound under declared assumptions, not evidence ",
                "that the predictor is accurate or scientifically valid."
            )
            .to_owned(),
        ],
    })
}

fn validate_config(config: &SowrcConfig) -> Result<(), SowrcError> {
    if !config.q_min.is_finite() || config.q_min <= 0.0 || config.q_min > 1.0 {
        return Err(SowrcError::InvalidProbabilityFloor);
    }
    if !config.target_risk.is_finite() || !(0.0..=1.0).contains(&config.target_risk) {
        return Err(SowrcError::InvalidTargetRisk);
    }
    if !config.delta.is_finite() || config.delta <= 0.0 || config.delta >= 1.0 {
        return Err(SowrcError::InvalidDelta);
    }
    if config.threshold_grid.len() < 2
        || config
            .threshold_grid
            .iter()
            .any(|lambda| !lambda.is_finite() || !(0.0..=1.0).contains(lambda))
        || config
            .threshold_grid
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || config.threshold_grid.last().copied() != Some(1.0)
    {
        return Err(SowrcError::InvalidThresholdGrid);
    }
    if config.betting_fractions.is_empty()
        || config.betting_fractions.len() != config.mixture_weights.len()
    {
        return Err(SowrcError::InvalidBettingMixture);
    }
    let b_n = 1.0 / config.q_min;
    let mut weight_sum = 0.0_f64;
    for (&eta, &weight) in config
        .betting_fractions
        .iter()
        .zip(&config.mixture_weights)
    {
        let kappa = kappa(eta, 1.0, b_n);
        if !eta.is_finite()
            || eta <= 0.0
            || eta >= 3.0
            || !weight.is_finite()
            || weight <= 0.0
            || !kappa.is_finite()
            || kappa <= 0.0
        {
            return Err(SowrcError::InvalidBettingMixture);
        }
        weight_sum += weight;
    }
    if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > SUM_TOLERANCE {
        return Err(SowrcError::InvalidBettingMixture);
    }
    if config.design_manifest_ref.trim().is_empty() {
        return Err(SowrcError::InvalidDesignManifest);
    }
    if config.prediction_manifest_ref.trim().is_empty() {
        return Err(SowrcError::InvalidPredictionManifest);
    }
    Ok(())
}

fn g_c(eta: f64, c_n: f64) -> f64 {
    eta * eta / (2.0 * (1.0 - eta * c_n / 3.0))
}

fn kappa(eta: f64, c_n: f64, b_n: f64) -> f64 {
    eta - b_n * g_c(eta, c_n)
}

fn log_sum_exp(values: &[f64]) -> Result<f64, SowrcError> {
    let maximum = values
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or(SowrcError::InvalidMixtureNumerics)?;
    if !maximum.is_finite() {
        return Err(SowrcError::InvalidMixtureNumerics);
    }
    let sum = values
        .iter()
        .map(|value| (value - maximum).exp())
        .sum::<f64>();
    let result = maximum + sum.ln();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(SowrcError::InvalidMixtureNumerics)
    }
}

fn log_mixture_at_risk(
    risk: f64,
    horizon: usize,
    ht_loss_total: f64,
    c_n: f64,
    b_n: f64,
    betting_fractions: &[f64],
    mixture_weights: &[f64],
) -> Result<f64, SowrcError> {
    let mut terms = Vec::with_capacity(betting_fractions.len());
    for (&eta, &weight) in betting_fractions.iter().zip(mixture_weights) {
        let k = kappa(eta, c_n, b_n);
        let exponent = k * horizon as f64 * risk - eta * ht_loss_total;
        if !exponent.is_finite() {
            return Err(SowrcError::InvalidMixtureNumerics);
        }
        terms.push(weight.ln() + exponent);
    }
    log_sum_exp(&terms)
}

/// Invert E_n(lambda,r) >= 1/delta using log-space bisection.
/// Returns (U_n, log E_n at 0, log E_n at 1).
fn invert_mixture_upper_bound(
    horizon: usize,
    ht_loss_total: f64,
    delta: f64,
    c_n: f64,
    b_n: f64,
    betting_fractions: &[f64],
    mixture_weights: &[f64],
) -> Result<(f64, f64, f64), SowrcError> {
    let log_barrier = (1.0 / delta).ln();
    let log_at_zero = log_mixture_at_risk(
        0.0,
        horizon,
        ht_loss_total,
        c_n,
        b_n,
        betting_fractions,
        mixture_weights,
    )?;
    let log_at_one = log_mixture_at_risk(
        1.0,
        horizon,
        ht_loss_total,
        c_n,
        b_n,
        betting_fractions,
        mixture_weights,
    )?;
    if log_at_zero >= log_barrier {
        return Ok((0.0, log_at_zero, log_at_one));
    }
    if log_at_one < log_barrier {
        return Ok((1.0, log_at_zero, log_at_one));
    }

    let mut lower = 0.0_f64;
    let mut upper = 1.0_f64;
    for _ in 0..BISECTION_STEPS {
        let midpoint = (lower + upper) / 2.0;
        let log_value = log_mixture_at_risk(
            midpoint,
            horizon,
            ht_loss_total,
            c_n,
            b_n,
            betting_fractions,
            mixture_weights,
        )?;
        if log_value >= log_barrier {
            upper = midpoint;
        } else {
            lower = midpoint;
        }
    }
    Ok((upper, log_at_zero, log_at_one))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(q_min: f64) -> SowrcConfig {
        SowrcConfig::for_audit_floor(
            q_min,
            "custody://frozen-randomized-audit/v1",
            "artifact://frozen-predictions/v1",
        )
    }

    fn observation(
        sequence_number: usize,
        id: &str,
        q: f64,
        score: Option<f64>,
    ) -> SowrcObservation {
        SowrcObservation {
            sequence_number,
            record_id: id.to_owned(),
            inclusion_probability: q,
            label_observed: score.is_some(),
            true_label_nonconformity: score,
        }
    }

    #[test]
    fn default_betting_grid_is_admissible_for_multiple_audit_floors() {
        for q_min in [0.05, 0.10, 0.20, 0.50, 0.80, 1.0] {
            let cfg = config(q_min);
            validate_config(&cfg).unwrap();
            assert!(cfg.betting_fractions.iter().all(|eta| {
                kappa(*eta, 1.0, 1.0 / q_min) > 0.0
            }));
        }
    }

    #[test]
    fn horvitz_thompson_loss_uses_only_observed_labels() {
        let cfg = config(0.5);
        let rows = vec![
            observation(0, "a", 0.5, Some(0.8)),
            observation(1, "b", 0.5, None),
        ];
        let report = evaluate_sowrc(&rows, &cfg).unwrap();
        let lambda_half = report
            .threshold_bounds
            .iter()
            .find(|bound| (bound.lambda - 0.5).abs() < 1e-12)
            .unwrap();
        assert_eq!(lambda_half.horvitz_thompson_loss_total, 2.0);
        assert_eq!(lambda_half.empirical_weighted_risk, 1.0);
    }

    #[test]
    fn nonconformity_loss_is_monotone_and_terminal_threshold_is_safe() {
        let cfg = config(1.0);
        let rows = vec![
            observation(0, "a", 1.0, Some(0.9)),
            observation(1, "b", 1.0, Some(0.2)),
            observation(2, "c", 1.0, Some(0.7)),
        ];
        let report = evaluate_sowrc(&rows, &cfg).unwrap();
        assert!(report.threshold_bounds.windows(2).all(|pair| {
            pair[0].horvitz_thompson_loss_total >= pair[1].horvitz_thompson_loss_total
        }));
        let terminal = report.threshold_bounds.last().unwrap();
        assert_eq!(terminal.lambda, 1.0);
        assert_eq!(terminal.horvitz_thompson_loss_total, 0.0);
        assert_eq!(report.selected_lambda, 1.0);
        assert_eq!(report.selected_risk_upper_bound, 0.0);
    }

    #[test]
    fn fixed_horizon_bound_is_nontrivial_with_complete_labels_and_enough_events() {
        let cfg = config(1.0);
        let rows = (0..40)
            .map(|index| {
                observation(
                    index,
                    &format!("event-{index}"),
                    1.0,
                    Some(if index % 5 == 0 { 0.8 } else { 0.1 }),
                )
            })
            .collect::<Vec<_>>();
        let report = evaluate_sowrc(&rows, &cfg).unwrap();
        let target = report
            .threshold_bounds
            .iter()
            .find(|bound| (bound.lambda - 0.5).abs() < 1e-12)
            .unwrap();
        assert!(target.upper_risk_bound < 1.0);
        assert!(target.upper_risk_bound >= target.empirical_weighted_risk);
    }

    #[test]
    fn safe_terminal_fallback_is_used_when_no_nonterminal_grid_point_is_feasible() {
        let mut cfg = config(0.2);
        cfg.target_risk = 0.0;
        let rows = vec![
            observation(0, "a", 0.2, Some(0.95)),
            observation(1, "b", 0.2, None),
            observation(2, "c", 0.2, Some(0.60)),
        ];
        let report = evaluate_sowrc(&rows, &cfg).unwrap();
        assert_eq!(report.selected_lambda, 1.0);
        assert_eq!(report.selected_risk_upper_bound, 0.0);
        assert_eq!(
            report.selection_status,
            SowrcSelectionStatus::SafeTerminalFullLabelSet
        );
    }

    #[test]
    fn threshold_grid_error_allocations_sum_to_total_delta() {
        let cfg = config(0.5);
        let rows = vec![
            observation(0, "a", 0.5, Some(0.2)),
            observation(1, "b", 0.5, None),
        ];
        let report = evaluate_sowrc(&rows, &cfg).unwrap();
        let allocated = report
            .threshold_bounds
            .iter()
            .map(|bound| bound.allocated_delta)
            .sum::<f64>();
        assert!((allocated - report.total_delta).abs() < 1e-12);
    }

    #[test]
    fn zero_or_below_floor_probability_fails_closed() {
        for q in [0.0, 0.01, f64::NAN, 1.01] {
            let rows = vec![observation(0, "a", q, Some(0.5))];
            let error = evaluate_sowrc(&rows, &config(0.05)).unwrap_err();
            assert!(matches!(
                error,
                SowrcError::InvalidInclusionProbability(_)
                    | SowrcError::InclusionProbabilityBelowFloor(_)
            ));
        }
    }

    #[test]
    fn selected_label_status_and_score_must_agree() {
        let rows = vec![observation(0, "a", 0.5, None)];
        let mut inconsistent = rows.clone();
        inconsistent[0].label_observed = true;
        assert!(matches!(
            evaluate_sowrc(&inconsistent, &config(0.5)),
            Err(SowrcError::LabelScoreMismatch(_))
        ));

        let mut inconsistent = rows;
        inconsistent[0].true_label_nonconformity = Some(0.2);
        assert!(matches!(
            evaluate_sowrc(&inconsistent, &config(0.5)),
            Err(SowrcError::LabelScoreMismatch(_))
        ));
    }

    #[test]
    fn certainty_without_a_label_is_impossible_and_rejected() {
        let rows = vec![observation(0, "a", 1.0, None)];
        assert!(matches!(
            evaluate_sowrc(&rows, &config(1.0)),
            Err(SowrcError::CertainInclusionWithoutLabel(_))
        ));
    }

    #[test]
    fn invalid_sequence_and_duplicate_record_ids_fail_closed() {
        let mut rows = vec![
            observation(0, "a", 0.5, Some(0.1)),
            observation(1, "b", 0.5, None),
        ];
        rows[1].sequence_number = 0;
        assert!(matches!(
            evaluate_sowrc(&rows, &config(0.5)),
            Err(SowrcError::NonSequentialIndex { .. })
        ));
        rows[1].sequence_number = 1;
        rows[1].record_id = "a".to_owned();
        assert!(matches!(
            evaluate_sowrc(&rows, &config(0.5)),
            Err(SowrcError::DuplicateRecordId(_))
        ));
    }

    #[test]
    fn invalid_threshold_grid_or_mixture_is_rejected() {
        let rows = vec![observation(0, "a", 0.5, Some(0.1))];
        let mut cfg = config(0.5);
        cfg.threshold_grid = vec![0.0, 0.5];
        assert_eq!(
            evaluate_sowrc(&rows, &cfg).unwrap_err(),
            SowrcError::InvalidThresholdGrid
        );

        let mut cfg = config(0.5);
        cfg.mixture_weights[0] += 0.1;
        assert_eq!(
            evaluate_sowrc(&rows, &cfg).unwrap_err(),
            SowrcError::InvalidBettingMixture
        );
    }

    #[test]
    fn nonconformity_scores_must_be_bounded() {
        for score in [f64::NAN, -0.1, 1.1] {
            let rows = vec![observation(0, "a", 1.0, Some(score))];
            assert!(matches!(
                evaluate_sowrc(&rows, &config(1.0)),
                Err(SowrcError::InvalidNonconformityScore(_))
            ));
        }
    }

    #[test]
    fn fixed_threshold_bound_covers_exact_bernoulli_audit_mask_enumeration() {
        // Exact enumeration over all 2^16 independent audit masks. The
        // probability of each mask is its Bernoulli design probability, not
        // a uniform weight over all masks.
        let n = 16usize;
        let q = 0.8_f64;
        let q_min = q;
        let delta = 0.5_f64;
        let per_threshold_delta = delta / 2.0;
        let eta_max = 2.0 * q_min / (1.0 + 2.0 * q_min / 3.0);
        let etas = ETA_FACTORS
            .iter()
            .map(|factor| eta_max * factor)
            .collect::<Vec<_>>();
        let weights = vec![1.0 / etas.len() as f64; etas.len()];
        let true_losses = (0..n)
            .map(|index| index % 2 == 0)
            .collect::<Vec<_>>();
        let population_risk = 0.5_f64;
        let mut failure_probability = 0.0_f64;
        let mut saw_nontrivial_bound = false;

        for mask in 0..(1usize << n) {
            let mut ht_loss_total = 0.0_f64;
            let mut mask_probability = 1.0_f64;
            for index in 0..n {
                let observed = mask & (1usize << index) != 0;
                mask_probability *= if observed { q } else { 1.0 - q };
                if observed && true_losses[index] {
                    ht_loss_total += 1.0 / q;
                }
            }
            let (upper_bound, _, _) = invert_mixture_upper_bound(
                n,
                ht_loss_total,
                per_threshold_delta,
                1.0,
                1.0 / q_min,
                &etas,
                &weights,
            )
            .unwrap();
            saw_nontrivial_bound |= upper_bound < 1.0;
            if upper_bound + 1e-12 < population_risk {
                failure_probability += mask_probability;
            }
        }

        assert!(saw_nontrivial_bound);
        assert!(
            failure_probability <= per_threshold_delta + 1e-12,
            "exact Bernoulli audit-mask failure probability {failure_probability} exceeded {per_threshold_delta}"
        );
    }
}
