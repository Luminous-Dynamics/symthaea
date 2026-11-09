// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Decision-focused diagnostics for continuous scalar Gaussian observations.
//!
//! This complements the finite-categorical EVSI kernel. Each fault/model
//! hypothesis supplies an action-conditioned Gaussian predictive observation
//! distribution. Expected posterior Bayes risk and model mutual information
//! are integrated over the continuous observation space using bounded
//! adaptive Simpson quadrature.
//!
//! The kernel does not infer predictive distributions, execute probes, or
//! authorize WeightPrograms. Likelihood construction, causal identification,
//! physical safety, and independent verification remain caller responsibilities.

use serde::{Deserialize, Serialize};
use thiserror::Error;

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const RISK_TOLERANCE: f64 = 1e-10;
const TAIL_SIGMAS: f64 = 12.0;
const MAX_QUADRATURE_DEPTH: u32 = 24;

/// One action-conditioned univariate Gaussian predictive observation model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GaussianOutcomePrediction {
    pub mean: f64,
    pub standard_deviation: f64,
}

/// A candidate model/fault hypothesis and its downstream decision-loss table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianDecisionHypothesis {
    pub id: String,
    pub prior_probability: f64,
    pub outcome_predictions_by_action: Vec<GaussianOutcomePrediction>,
    pub loss_by_final_decision: Vec<f64>,
}

/// A continuous-observation diagnostic action with explicit operational cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianDiagnosticAction {
    pub id: String,
    pub observation_cost: f64,
    /// Caller-defined risk in [0, 1], not necessarily a calibrated probability.
    pub safety_risk: f64,
}

/// Numerical and decision-policy controls.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GaussianDecisionDiagnosticConfig {
    pub observation_cost_weight: f64,
    pub safety_risk_weight: f64,
    pub maximum_safety_risk: f64,
    pub risk_absolute_tolerance: f64,
    pub information_absolute_tolerance: f64,
    pub maximum_quadrature_evaluations_per_action: usize,
}

impl Default for GaussianDecisionDiagnosticConfig {
    fn default() -> Self {
        Self {
            observation_cost_weight: 1.0,
            safety_risk_weight: 1.0,
            maximum_safety_risk: 1.0,
            risk_absolute_tolerance: 1e-8,
            information_absolute_tolerance: 1e-8,
            maximum_quadrature_evaluations_per_action: 100_000,
        }
    }
}

/// Evaluation of one continuous diagnostic action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianDiagnosticActionEvaluation {
    pub action_index: usize,
    pub action_id: String,
    pub expected_posterior_bayes_risk: f64,
    pub expected_value_of_sample_information: f64,
    pub model_information_gain_nats: f64,
    pub observation_cost: f64,
    pub safety_risk: f64,
    pub eligible: bool,
    pub total_score: Option<f64>,
    pub risk_quadrature_error_estimate: f64,
    pub information_quadrature_error_estimate: f64,
    pub quadrature_evaluations: usize,
}

/// Comparison of continuous probes against a no-probe Bayes-risk baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianDecisionDiagnosticEvaluation {
    pub prior_bayes_risk: f64,
    pub no_probe_score: f64,
    pub selected_action_index: Option<usize>,
    pub actions: Vec<GaussianDiagnosticActionEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum GaussianDecisionDiagnosticError {
    #[error("at least one model/fault hypothesis is required")]
    EmptyHypotheses,
    #[error("hypothesis {hypothesis_index} has an empty identifier")]
    EmptyHypothesisId { hypothesis_index: usize },
    #[error("duplicate hypothesis identifier: {0}")]
    DuplicateHypothesisId(String),
    #[error("hypothesis {hypothesis_index} has invalid prior probability")]
    InvalidPrior { hypothesis_index: usize },
    #[error("hypothesis prior mass sums to {sum}, expected 1")]
    PriorMassNotNormalized { sum: f64 },
    #[error("at least one diagnostic action is required")]
    EmptyActionSet,
    #[error("hypothesis {hypothesis_index} has {actual} actions; expected {expected}")]
    InconsistentActionCount {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("hypothesis {hypothesis_index} has no terminal decisions")]
    EmptyDecisionSet { hypothesis_index: usize },
    #[error("hypothesis {hypothesis_index} has {actual} decisions; expected {expected}")]
    InconsistentDecisionCount {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("hypothesis {hypothesis_index}, decision {decision_index} has invalid loss")]
    InvalidDecisionLoss {
        hypothesis_index: usize,
        decision_index: usize,
    },
    #[error("hypothesis {hypothesis_index}, action {action_index} has an invalid Gaussian prediction")]
    InvalidPrediction {
        hypothesis_index: usize,
        action_index: usize,
    },
    #[error("action {action_index} has an empty identifier")]
    EmptyActionId { action_index: usize },
    #[error("duplicate action identifier: {0}")]
    DuplicateActionId(String),
    #[error("action {action_index} has invalid observation cost")]
    InvalidObservationCost { action_index: usize },
    #[error("action {action_index} has safety risk outside [0, 1]")]
    InvalidSafetyRisk { action_index: usize },
    #[error("received {actual} action descriptors for {expected} modelled actions")]
    ActionCountMismatch { expected: usize, actual: usize },
    #[error("quadrature/decision configuration is invalid")]
    InvalidConfig,
    #[error("action index {action_index} is outside 0..{action_count}")]
    ActionOutOfRange {
        action_index: usize,
        action_count: usize,
    },
    #[error("observation has zero predictive probability under action {action_index}")]
    ImpossibleObservation { action_index: usize },
    #[error("quadrature evaluation budget {budget} exceeded")]
    QuadratureBudgetExceeded { budget: usize },
    #[error("adaptive quadrature did not converge: estimated error {estimated_error} exceeds tolerance {tolerance}")]
    QuadratureDidNotConverge {
        estimated_error: f64,
        tolerance: f64,
    },
    #[error("quadrature produced a non-finite value")]
    NonFiniteQuadrature,
    #[error("expected posterior Bayes risk {expected_risk} exceeds prior risk {prior_risk}")]
    NonMonotonicBayesRisk {
        prior_risk: f64,
        expected_risk: f64,
    },
    #[error("computed model information gain is materially negative: {0}")]
    NegativeInformationGain(f64),
}

/// Evaluate each probe using expected posterior Bayes risk and model information.
/// A no-probe option is always available.
pub fn evaluate_gaussian_decision_diagnostics(
    hypotheses: &[GaussianDecisionHypothesis],
    actions: &[GaussianDiagnosticAction],
    config: GaussianDecisionDiagnosticConfig,
) -> Result<GaussianDecisionDiagnosticEvaluation, GaussianDecisionDiagnosticError> {
    if actions.is_empty() {
        return Err(GaussianDecisionDiagnosticError::EmptyActionSet);
    }
    validate_config(config)?;
    let (action_count, decision_count) = validate_hypotheses(hypotheses)?;
    if actions.len() != action_count {
        return Err(GaussianDecisionDiagnosticError::ActionCountMismatch {
            expected: action_count,
            actual: actions.len(),
        });
    }
    validate_actions(actions)?;

    let prior: Vec<f64> = hypotheses.iter().map(|h| h.prior_probability).collect();
    let prior_risk = bayes_risk(&prior, hypotheses, decision_count);
    let maximum_loss = hypotheses
        .iter()
        .flat_map(|h| h.loss_by_final_decision.iter().copied())
        .fold(0.0_f64, f64::max);
    let mut best_score = prior_risk;
    let mut selected = None;
    let mut evaluations = Vec::with_capacity(actions.len());

    for (action_index, action) in actions.iter().enumerate() {
        let predictions: Vec<_> = hypotheses
            .iter()
            .map(|h| h.outcome_predictions_by_action[action_index])
            .collect();
        let breakpoints = integration_breakpoints(&predictions)?;
        let mut used = 0;
        let (risk, risk_error) = integrate_piecewise(
            &breakpoints,
            config.risk_absolute_tolerance,
            config.maximum_quadrature_evaluations_per_action,
            &mut used,
            |y| integrated_risk_density(y, &predictions, hypotheses, decision_count),
        )?;
        let (information, information_error) = integrate_piecewise(
            &breakpoints,
            config.information_absolute_tolerance,
            config.maximum_quadrature_evaluations_per_action,
            &mut used,
            |y| integrated_information_density(y, &predictions, hypotheses),
        )?;

        // Chernoff: P(|Z|>=12) <= 2 exp(-72), for each Gaussian component.
        let tail_bound = 2.0 * (-72.0_f64).exp();
        let risk_error_estimate = risk_error + tail_bound * maximum_loss;
        let information_error_estimate = information_error + tail_bound * entropy_upper_bound(&prior);
        if risk > prior_risk + risk_error_estimate + RISK_TOLERANCE {
            return Err(GaussianDecisionDiagnosticError::NonMonotonicBayesRisk {
                prior_risk,
                expected_risk: risk,
            });
        }
        let posterior_risk = risk.min(prior_risk);
        let evsi = (prior_risk - posterior_risk).max(0.0);
        let eig = if information < 0.0 {
            if information.abs() <= information_error_estimate + RISK_TOLERANCE {
                0.0
            } else {
                return Err(GaussianDecisionDiagnosticError::NegativeInformationGain(information));
            }
        } else {
            information
        };

        let eligible = action.safety_risk <= config.maximum_safety_risk;
        let score = if eligible {
            let value = posterior_risk
                + config.observation_cost_weight * action.observation_cost
                + config.safety_risk_weight * action.safety_risk;
            if !value.is_finite() {
                return Err(GaussianDecisionDiagnosticError::NonFiniteQuadrature);
            }
            Some(value)
        } else {
            None
        };
        if let Some(value) = score
            && value + RISK_TOLERANCE < best_score
        {
            best_score = value;
            selected = Some(action_index);
        }

        evaluations.push(GaussianDiagnosticActionEvaluation {
            action_index,
            action_id: action.id.clone(),
            expected_posterior_bayes_risk: posterior_risk,
            expected_value_of_sample_information: evsi,
            model_information_gain_nats: eig,
            observation_cost: action.observation_cost,
            safety_risk: action.safety_risk,
            eligible,
            total_score: score,
            risk_quadrature_error_estimate: risk_error_estimate,
            information_quadrature_error_estimate: information_error_estimate,
            quadrature_evaluations: used,
        });
    }

    Ok(GaussianDecisionDiagnosticEvaluation {
        prior_bayes_risk: prior_risk,
        no_probe_score: prior_risk,
        selected_action_index: selected,
        actions: evaluations,
    })
}

/// Bayesian posterior over hypotheses after one scalar observation.
pub fn gaussian_posterior_model_probabilities(
    hypotheses: &[GaussianDecisionHypothesis],
    action_index: usize,
    observed_value: f64,
) -> Result<Vec<f64>, GaussianDecisionDiagnosticError> {
    let (action_count, _) = validate_hypotheses(hypotheses)?;
    if action_index >= action_count {
        return Err(GaussianDecisionDiagnosticError::ActionOutOfRange {
            action_index,
            action_count,
        });
    }
    if !observed_value.is_finite() {
        return Err(GaussianDecisionDiagnosticError::ImpossibleObservation { action_index });
    }
    let mut masses = Vec::with_capacity(hypotheses.len());
    let mut evidence = 0.0;
    for h in hypotheses {
        let density = normal_density(observed_value, h.outcome_predictions_by_action[action_index]);
        let mass = h.prior_probability * density;
        masses.push(mass);
        evidence += mass;
    }
    if !evidence.is_finite() || evidence <= 0.0 {
        return Err(GaussianDecisionDiagnosticError::ImpossibleObservation { action_index });
    }
    Ok(masses.into_iter().map(|mass| mass / evidence).collect())
}

fn validate_hypotheses(
    hypotheses: &[GaussianDecisionHypothesis],
) -> Result<(usize, usize), GaussianDecisionDiagnosticError> {
    if hypotheses.is_empty() {
        return Err(GaussianDecisionDiagnosticError::EmptyHypotheses);
    }
    let action_count = hypotheses[0].outcome_predictions_by_action.len();
    if action_count == 0 {
        return Err(GaussianDecisionDiagnosticError::EmptyActionSet);
    }
    let decision_count = hypotheses[0].loss_by_final_decision.len();
    if decision_count == 0 {
        return Err(GaussianDecisionDiagnosticError::EmptyDecisionSet { hypothesis_index: 0 });
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut prior_sum = 0.0;
    for (hi, h) in hypotheses.iter().enumerate() {
        if h.id.trim().is_empty() {
            return Err(GaussianDecisionDiagnosticError::EmptyHypothesisId { hypothesis_index: hi });
        }
        if !ids.insert(h.id.clone()) {
            return Err(GaussianDecisionDiagnosticError::DuplicateHypothesisId(h.id.clone()));
        }
        if !h.prior_probability.is_finite() || !(0.0..=1.0).contains(&h.prior_probability) {
            return Err(GaussianDecisionDiagnosticError::InvalidPrior { hypothesis_index: hi });
        }
        prior_sum += h.prior_probability;
        if h.outcome_predictions_by_action.len() != action_count {
            return Err(GaussianDecisionDiagnosticError::InconsistentActionCount {
                hypothesis_index: hi,
                expected: action_count,
                actual: h.outcome_predictions_by_action.len(),
            });
        }
        if h.loss_by_final_decision.len() != decision_count {
            return Err(GaussianDecisionDiagnosticError::InconsistentDecisionCount {
                hypothesis_index: hi,
                expected: decision_count,
                actual: h.loss_by_final_decision.len(),
            });
        }
        for (di, loss) in h.loss_by_final_decision.iter().enumerate() {
            if !loss.is_finite() || *loss < 0.0 {
                return Err(GaussianDecisionDiagnosticError::InvalidDecisionLoss {
                    hypothesis_index: hi,
                    decision_index: di,
                });
            }
        }
        for (ai, prediction) in h.outcome_predictions_by_action.iter().enumerate() {
            if !prediction.mean.is_finite()
                || !prediction.standard_deviation.is_finite()
                || prediction.standard_deviation <= 0.0
                || !(prediction.mean - TAIL_SIGMAS * prediction.standard_deviation).is_finite()
                || !(prediction.mean + TAIL_SIGMAS * prediction.standard_deviation).is_finite()
            {
                return Err(GaussianDecisionDiagnosticError::InvalidPrediction {
                    hypothesis_index: hi,
                    action_index: ai,
                });
            }
        }
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(GaussianDecisionDiagnosticError::PriorMassNotNormalized { sum: prior_sum });
    }
    Ok((action_count, decision_count))
}

fn validate_actions(actions: &[GaussianDiagnosticAction]) -> Result<(), GaussianDecisionDiagnosticError> {
    let mut ids = std::collections::BTreeSet::new();
    for (ai, action) in actions.iter().enumerate() {
        if action.id.trim().is_empty() {
            return Err(GaussianDecisionDiagnosticError::EmptyActionId { action_index: ai });
        }
        if !ids.insert(action.id.clone()) {
            return Err(GaussianDecisionDiagnosticError::DuplicateActionId(action.id.clone()));
        }
        if !action.observation_cost.is_finite() || action.observation_cost < 0.0 {
            return Err(GaussianDecisionDiagnosticError::InvalidObservationCost { action_index: ai });
        }
        if !action.safety_risk.is_finite() || !(0.0..=1.0).contains(&action.safety_risk) {
            return Err(GaussianDecisionDiagnosticError::InvalidSafetyRisk { action_index: ai });
        }
    }
    Ok(())
}

fn validate_config(config: GaussianDecisionDiagnosticConfig) -> Result<(), GaussianDecisionDiagnosticError> {
    if !config.observation_cost_weight.is_finite()
        || config.observation_cost_weight < 0.0
        || !config.safety_risk_weight.is_finite()
        || config.safety_risk_weight < 0.0
        || !config.maximum_safety_risk.is_finite()
        || !(0.0..=1.0).contains(&config.maximum_safety_risk)
        || !config.risk_absolute_tolerance.is_finite()
        || config.risk_absolute_tolerance <= 0.0
        || !config.information_absolute_tolerance.is_finite()
        || config.information_absolute_tolerance <= 0.0
        || config.maximum_quadrature_evaluations_per_action < 64
    {
        return Err(GaussianDecisionDiagnosticError::InvalidConfig);
    }
    Ok(())
}

fn bayes_risk(p: &[f64], hypotheses: &[GaussianDecisionHypothesis], decisions: usize) -> f64 {
    (0..decisions)
        .map(|d| hypotheses.iter().zip(p).map(|(h, probability)| probability * h.loss_by_final_decision[d]).sum::<f64>())
        .fold(f64::INFINITY, f64::min)
}

fn normal_density(y: f64, prediction: GaussianOutcomePrediction) -> f64 {
    let z = (y - prediction.mean) / prediction.standard_deviation;
    if !z.is_finite() || z.abs() > 38.0 {
        return 0.0;
    }
    (-0.5 * z * z).exp() / (prediction.standard_deviation * (2.0 * std::f64::consts::PI).sqrt())
}

fn integrated_risk_density(
    y: f64,
    predictions: &[GaussianOutcomePrediction],
    hypotheses: &[GaussianDecisionHypothesis],
    decisions: usize,
) -> f64 {
    let joint: Vec<f64> = hypotheses.iter().zip(predictions)
        .map(|(h, p)| h.prior_probability * normal_density(y, *p)).collect();
    (0..decisions)
        .map(|d| hypotheses.iter().zip(&joint).map(|(h, mass)| mass * h.loss_by_final_decision[d]).sum::<f64>())
        .fold(f64::INFINITY, f64::min)
}

fn integrated_information_density(
    y: f64,
    predictions: &[GaussianOutcomePrediction],
    hypotheses: &[GaussianDecisionHypothesis],
) -> f64 {
    let mut joint = Vec::with_capacity(hypotheses.len());
    let mut densities = Vec::with_capacity(hypotheses.len());
    let mut evidence = 0.0;
    for (h, p) in hypotheses.iter().zip(predictions) {
        let density = normal_density(y, *p);
        let mass = h.prior_probability * density;
        densities.push(density);
        joint.push(mass);
        evidence += mass;
    }
    if evidence <= 0.0 {
        return 0.0;
    }
    joint.iter().zip(densities)
        .filter_map(|(mass, density)| {
            if *mass > 0.0 && density > 0.0 {
                Some(mass * (density / evidence).ln())
            } else {
                None
            }
        })
        .sum()
}

fn integration_breakpoints(
    predictions: &[GaussianOutcomePrediction],
) -> Result<Vec<f64>, GaussianDecisionDiagnosticError> {
    const OFFSETS: [f64; 9] = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0];
    let mut points = Vec::with_capacity(predictions.len() * 11);
    for p in predictions {
        points.push(p.mean - TAIL_SIGMAS * p.standard_deviation);
        points.push(p.mean + TAIL_SIGMAS * p.standard_deviation);
        for offset in OFFSETS {
            points.push(p.mean + offset * p.standard_deviation);
        }
    }
    if points.iter().any(|point| !point.is_finite()) {
        return Err(GaussianDecisionDiagnosticError::NonFiniteQuadrature);
    }
    points.sort_by(f64::total_cmp);
    points.dedup_by(|left, right| (*left - *right).abs() <= 1e-13 * left.abs().max(right.abs()).max(1.0));
    if points.len() < 2 {
        return Err(GaussianDecisionDiagnosticError::NonFiniteQuadrature);
    }
    Ok(points)
}

fn integrate_piecewise<F>(
    points: &[f64],
    tolerance: f64,
    maximum: usize,
    used: &mut usize,
    f: F,
) -> Result<(f64, f64), GaussianDecisionDiagnosticError>
where
    F: Fn(f64) -> f64,
{
    let local_tolerance = tolerance / (points.len() - 1) as f64;
    let mut total = 0.0;
    let mut error_total = 0.0;
    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if a >= b { continue; }
        let m = a + (b - a) * 0.5;
        let fa = checked_eval(&f, a, used, maximum)?;
        let fm = checked_eval(&f, m, used, maximum)?;
        let fb = checked_eval(&f, b, used, maximum)?;
        let whole = simpson(a, b, fa, fm, fb);
        let (value, error) = adaptive_simpson(
            &f, a, b, fa, fm, fb, whole, local_tolerance,
            MAX_QUADRATURE_DEPTH, used, maximum,
        )?;
        total += value;
        error_total += error;
    }
    if !total.is_finite() || !error_total.is_finite() {
        return Err(GaussianDecisionDiagnosticError::NonFiniteQuadrature);
    }
    Ok((total, error_total))
}

fn checked_eval<F>(
    f: &F,
    x: f64,
    used: &mut usize,
    maximum: usize,
) -> Result<f64, GaussianDecisionDiagnosticError>
where
    F: Fn(f64) -> f64,
{
    if *used >= maximum {
        return Err(GaussianDecisionDiagnosticError::QuadratureBudgetExceeded { budget: maximum });
    }
    *used += 1;
    let y = f(x);
    if y.is_finite() { Ok(y) } else { Err(GaussianDecisionDiagnosticError::NonFiniteQuadrature) }
}

#[allow(clippy::too_many_arguments)]
fn adaptive_simpson<F>(
    f: &F, a: f64, b: f64, fa: f64, fm: f64, fb: f64, whole: f64,
    tolerance: f64, depth: u32, used: &mut usize, maximum: usize,
) -> Result<(f64, f64), GaussianDecisionDiagnosticError>
where
    F: Fn(f64) -> f64,
{
    let m = a + (b - a) * 0.5;
    let lm = a + (m - a) * 0.5;
    let rm = m + (b - m) * 0.5;
    let fl = checked_eval(f, lm, used, maximum)?;
    let fr = checked_eval(f, rm, used, maximum)?;
    let left = simpson(a, m, fa, fl, fm);
    let right = simpson(m, b, fm, fr, fb);
    let delta = left + right - whole;
    let error = delta.abs() / 15.0;
    if error <= tolerance {
        return Ok((left + right + delta / 15.0, error));
    }
    if depth == 0 {
        return Err(GaussianDecisionDiagnosticError::QuadratureDidNotConverge {
            estimated_error: error, tolerance,
        });
    }
    let (lv, le) = adaptive_simpson(
        f, a, m, fa, fl, fm, left, tolerance * 0.5, depth - 1, used, maximum,
    )?;
    let (rv, re) = adaptive_simpson(
        f, m, b, fm, fr, fb, right, tolerance * 0.5, depth - 1, used, maximum,
    )?;
    Ok((lv + rv, le + re))
}

fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

fn entropy_upper_bound(p: &[f64]) -> f64 {
    p.iter().filter(|x| **x > 0.0).map(|x| -*x * x.ln()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary(
        first: GaussianOutcomePrediction,
        second: GaussianOutcomePrediction,
    ) -> Vec<GaussianDecisionHypothesis> {
        vec![
            GaussianDecisionHypothesis {
                id: "fault-a".into(), prior_probability: 0.5,
                outcome_predictions_by_action: vec![first],
                loss_by_final_decision: vec![0.0, 1.0],
            },
            GaussianDecisionHypothesis {
                id: "fault-b".into(), prior_probability: 0.5,
                outcome_predictions_by_action: vec![second],
                loss_by_final_decision: vec![1.0, 0.0],
            },
        ]
    }

    fn prediction(mean: f64, sd: f64) -> GaussianOutcomePrediction {
        GaussianOutcomePrediction { mean, standard_deviation: sd }
    }

    fn action(id: &str) -> GaussianDiagnosticAction {
        GaussianDiagnosticAction { id: id.into(), observation_cost: 0.0, safety_risk: 0.0 }
    }

    #[test]
    fn identical_predictive_distributions_have_zero_eig_and_evsi() {
        let h = binary(prediction(0.0, 1.0), prediction(0.0, 1.0));
        let r = evaluate_gaussian_decision_diagnostics(&h, &[action("inspect")], Default::default()).expect("valid");
        assert!(r.actions[0].model_information_gain_nats.abs() < 1e-8);
        assert!(r.actions[0].expected_value_of_sample_information.abs() < 1e-8);
        assert_eq!(r.selected_action_index, None);
    }

    #[test]
    fn separated_gaussians_resolve_binary_fault_and_reduce_risk() {
        let h = binary(prediction(-5.0, 1.0), prediction(5.0, 1.0));
        let r = evaluate_gaussian_decision_diagnostics(&h, &[action("inspect")], Default::default()).expect("valid");
        assert!((r.prior_bayes_risk - 0.5).abs() < 1e-12);
        assert!(r.actions[0].expected_value_of_sample_information > 0.49);
        assert!(r.actions[0].model_information_gain_nats > 0.68);
        assert_eq!(r.selected_action_index, Some(0));
    }

    #[test]
    fn information_without_decision_value_does_not_trigger_probe() {
        let mut h = binary(prediction(-5.0, 1.0), prediction(5.0, 1.0));
        for model in &mut h { model.loss_by_final_decision = vec![0.0, 1.0]; }
        let r = evaluate_gaussian_decision_diagnostics(&h, &[action("irrelevant")], Default::default()).expect("valid");
        assert!(r.actions[0].model_information_gain_nats > 0.68);
        assert!(r.actions[0].expected_value_of_sample_information.abs() < 1e-8);
        assert_eq!(r.selected_action_index, None);
    }

    #[test]
    fn lower_eig_decision_relevant_probe_beats_higher_eig_irrelevant_probe() {
        let mut h = Vec::new();
        for index in 0..4 {
            let fault_a = index < 2;
            let nuisance_zero = index % 2 == 0;
            h.push(GaussianDecisionHypothesis {
                id: format!("model-{index}"), prior_probability: 0.25,
                outcome_predictions_by_action: vec![
                    prediction(if nuisance_zero { -5.0 } else { 5.0 }, 1.0),
                    prediction(if fault_a { -0.5 } else { 0.5 }, 1.0),
                ],
                loss_by_final_decision: if fault_a { vec![0.0, 1.0] } else { vec![1.0, 0.0] },
            });
        }
        let r = evaluate_gaussian_decision_diagnostics(
            &h, &[action("irrelevant-high-eig"), action("relevant-low-eig")], Default::default()
        ).expect("valid");
        assert!(r.actions[0].model_information_gain_nats > r.actions[1].model_information_gain_nats);
        assert!(r.actions[0].expected_value_of_sample_information.abs() < 1e-8);
        assert!(r.actions[1].expected_value_of_sample_information > 0.05);
        assert_eq!(r.selected_action_index, Some(1));
    }

    #[test]
    fn cost_and_hard_safety_threshold_prevent_selection() {
        let h = binary(prediction(-3.0, 1.0), prediction(3.0, 1.0));
        let mut costly = action("costly");
        costly.observation_cost = 1.0;
        let c = evaluate_gaussian_decision_diagnostics(&h, &[costly], Default::default()).expect("cost");
        assert_eq!(c.selected_action_index, None);

        let mut unsafe_probe = action("unsafe");
        unsafe_probe.safety_risk = 0.9;
        let s = evaluate_gaussian_decision_diagnostics(
            &h, &[unsafe_probe],
            GaussianDecisionDiagnosticConfig { maximum_safety_risk: 0.2, ..Default::default() }
        ).expect("safety");
        assert_eq!(s.selected_action_index, None);
        assert!(!s.actions[0].eligible);
    }

    #[test]
    fn posterior_update_is_normalized_and_tracks_observation() {
        let h = binary(prediction(-2.0, 1.0), prediction(2.0, 1.0));
        let p = gaussian_posterior_model_probabilities(&h, 0, -2.0).expect("posterior");
        assert!(p[0] > p[1]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn quadrature_budget_exhaustion_fails_closed() {
        let h = binary(prediction(-10.0, 0.1), prediction(10.0, 0.1));
        let r = evaluate_gaussian_decision_diagnostics(
            &h, &[action("inspect")],
            GaussianDecisionDiagnosticConfig { maximum_quadrature_evaluations_per_action: 64, ..Default::default() }
        );
        assert!(matches!(r, Err(GaussianDecisionDiagnosticError::QuadratureBudgetExceeded { .. })));
    }

    #[test]
    fn malformed_gaussian_and_loss_inputs_are_rejected() {
        let mut h = binary(prediction(0.0, 1.0), prediction(1.0, 1.0));
        h[1].outcome_predictions_by_action[0].standard_deviation = 0.0;
        assert!(matches!(
            evaluate_gaussian_decision_diagnostics(&h, &[action("inspect")], Default::default()),
            Err(GaussianDecisionDiagnosticError::InvalidPrediction { .. })
        ));
        h[1].outcome_predictions_by_action[0].standard_deviation = 1.0;
        h[0].loss_by_final_decision[0] = f64::NAN;
        assert!(matches!(
            evaluate_gaussian_decision_diagnostics(&h, &[action("inspect")], Default::default()),
            Err(GaussianDecisionDiagnosticError::InvalidDecisionLoss { .. })
        ));
    }
}
