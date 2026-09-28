// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Active-inference decision support for scientific test selection.
//! Predictions remain predictions; external observations remain outside this module.

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::collections::BTreeSet;

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificOutcome {
    pub id: String,
    /// Planner-supplied utility; never a truth value.
    pub utility: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificTrajectoryForecast {
    pub model_id: String,
    pub lineage: String,
    pub prior: f64,
    pub horizon_seconds: f32,
    /// State produced by the temporal world model.
    pub predicted_state: ContinuousHV,
    /// Likelihood of each declared test outcome, in test outcome order.
    pub outcome_probabilities: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificTest {
    pub id: String,
    pub cost: f64,
    pub outcomes: Vec<ScientificOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificActiveInferenceAssessment {
    pub test_id: String,
    pub epistemic_value_bits: f64,
    pub pragmatic_risk: f64,
    /// Lower is preferred by this decision layer.
    pub expected_free_energy: f64,
    pub eig_per_cost: f64,
    pub distinct_lineages: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificTestRanking {
    pub assessments: Vec<ScientificActiveInferenceAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificActiveInferenceError {
    EmptyForecasts, EmptyModelId, EmptyLineage, InvalidPrior, PriorsDoNotNormalize,
    EmptyTestId, NegativeCost, EmptyOutcomes, EmptyOutcomeId, InvalidUtility,
    ProbabilityWidthMismatch, InvalidProbability, ProbabilitiesDoNotNormalize, HorizonMismatch,
}
impl fmt::Display for ScientificActiveInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for ScientificActiveInferenceError {}

pub fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities.iter().filter(|p| **p > 0.0).map(|p| -p * p.log2()).sum()
}

/// FEP-style assessment. The epistemic term is expected model-entropy reduction,
/// matching the scientific EIG convention; the pragmatic term is planner utility
/// plus execution cost. Neither term is evidence of biological truth.
pub fn assess_test(
    forecasts: &[ScientificTrajectoryForecast],
    test: &ScientificTest,
) -> Result<ScientificActiveInferenceAssessment, ScientificActiveInferenceError> {
    validate(forecasts, test)?;
    let priors: Vec<f64> = forecasts.iter().map(|f| f.prior).collect();
    let prior_entropy = entropy_bits(&priors);
    let mut expected_posterior_entropy = 0.0;
    let mut expected_negative_utility = 0.0;

    for (i, outcome) in test.outcomes.iter().enumerate() {
        let p_outcome: f64 = forecasts.iter()
            .map(|f| f.prior * f.outcome_probabilities[i]).sum();
        expected_negative_utility += p_outcome * -outcome.utility;
        if p_outcome <= EPS { continue; }
        let posterior: Vec<f64> = forecasts.iter()
            .map(|f| f.prior * f.outcome_probabilities[i] / p_outcome).collect();
        expected_posterior_entropy += p_outcome * entropy_bits(&posterior);
    }

    let epistemic_value_bits = (prior_entropy - expected_posterior_entropy).max(0.0);
    let pragmatic_risk = test.cost + expected_negative_utility;
    let expected_free_energy = pragmatic_risk - epistemic_value_bits;
    let eig_per_cost = epistemic_value_bits / test.cost.max(EPS);
    let distinct_lineages = forecasts.iter().map(|f| f.lineage.as_str()).collect::<BTreeSet<_>>().len();

    Ok(ScientificActiveInferenceAssessment {
        test_id: test.id.clone(), epistemic_value_bits, pragmatic_risk,
        expected_free_energy, eig_per_cost, distinct_lineages,
    })
}

/// Deterministic decision ordering only; this is not a truth/evidence ranking.
pub fn rank_tests(
    candidates: &[(ScientificTest, Vec<ScientificTrajectoryForecast>)],
) -> Result<ScientificTestRanking, ScientificActiveInferenceError> {
    let mut assessments = candidates.iter()
        .map(|(test, forecasts)| assess_test(forecasts, test))
        .collect::<Result<Vec<_>, _>>()?;
    assessments.sort_by(|a, b| a.expected_free_energy.total_cmp(&b.expected_free_energy)
        .then(b.eig_per_cost.total_cmp(&a.eig_per_cost))
        .then(a.test_id.cmp(&b.test_id)));
    Ok(ScientificTestRanking { assessments })
}

pub fn validate(
    forecasts: &[ScientificTrajectoryForecast],
    test: &ScientificTest,
) -> Result<(), ScientificActiveInferenceError> {
    if forecasts.is_empty() { return Err(ScientificActiveInferenceError::EmptyForecasts); }
    let mut prior_sum = 0.0;
    let mut horizon = None;
    for f in forecasts {
        if f.model_id.trim().is_empty() { return Err(ScientificActiveInferenceError::EmptyModelId); }
        if f.lineage.trim().is_empty() { return Err(ScientificActiveInferenceError::EmptyLineage); }
        if !f.prior.is_finite() || f.prior < 0.0 { return Err(ScientificActiveInferenceError::InvalidPrior); }
        prior_sum += f.prior;
        if let Some(h) = horizon {
            if (h - f.horizon_seconds).abs() > f32::EPSILON { return Err(ScientificActiveInferenceError::HorizonMismatch); }
        } else { horizon = Some(f.horizon_seconds); }
    }
    if (prior_sum - 1.0).abs() > EPS { return Err(ScientificActiveInferenceError::PriorsDoNotNormalize); }
    if test.id.trim().is_empty() { return Err(ScientificActiveInferenceError::EmptyTestId); }
    if !test.cost.is_finite() || test.cost < 0.0 { return Err(ScientificActiveInferenceError::NegativeCost); }
    if test.outcomes.is_empty() { return Err(ScientificActiveInferenceError::EmptyOutcomes); }
    for o in &test.outcomes {
        if o.id.trim().is_empty() { return Err(ScientificActiveInferenceError::EmptyOutcomeId); }
        if !o.utility.is_finite() { return Err(ScientificActiveInferenceError::InvalidUtility); }
    }
    for f in forecasts {
        if f.outcome_probabilities.len() != test.outcomes.len() { return Err(ScientificActiveInferenceError::ProbabilityWidthMismatch); }
        if f.outcome_probabilities.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(ScientificActiveInferenceError::InvalidProbability);
        }
        if (f.outcome_probabilities.iter().sum::<f64>() - 1.0).abs() > EPS {
            return Err(ScientificActiveInferenceError::ProbabilitiesDoNotNormalize);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::unified_hv::HDC_DIMENSION;
    fn state(seed: u64) -> ContinuousHV { ContinuousHV::random(HDC_DIMENSION, seed) }
    fn test() -> ScientificTest {
        ScientificTest { id: "test-a".into(), cost: 1.0,
            outcomes: vec![ScientificOutcome { id: "a".into(), utility: 1.0 },
                            ScientificOutcome { id: "b".into(), utility: 0.0 }] }
    }
    fn forecasts() -> Vec<ScientificTrajectoryForecast> {
        vec![
            ScientificTrajectoryForecast { model_id:"model-a".into(), lineage:"lineage-a".into(), prior:0.5,
                horizon_seconds:10.0, predicted_state:state(1), outcome_probabilities:vec![1.0,0.0] },
            ScientificTrajectoryForecast { model_id:"model-b".into(), lineage:"lineage-b".into(), prior:0.5,
                horizon_seconds:10.0, predicted_state:state(2), outcome_probabilities:vec![0.0,1.0] },
        ]
    }
    #[test] fn perfect_discrimination_is_one_bit() {
        let a = assess_test(&forecasts(), &test()).unwrap();
        assert!((a.epistemic_value_bits - 1.0).abs() < 1e-12);
        assert_eq!(a.distinct_lineages, 2);
    }
    #[test] fn identical_likelihoods_have_zero_epistemic_value() {
        let mut f = forecasts(); for x in &mut f { x.outcome_probabilities = vec![0.5,0.5]; }
        assert!(assess_test(&f, &test()).unwrap().epistemic_value_bits.abs() < 1e-12);
    }
    #[test] fn pragmatic_term_includes_cost_and_utility() {
        let a = assess_test(&forecasts(), &test()).unwrap();
        assert!((a.pragmatic_risk - 0.5).abs() < 1e-12);
    }
    #[test] fn ranking_is_deterministic() {
        let a = test(); let mut b = a.clone(); b.id = "test-b".into();
        let r = rank_tests(&[(b,forecasts()),(a,forecasts())]).unwrap();
        assert_eq!(r.assessments[0].test_id, "test-a");
    }
    #[test] fn width_mismatch_is_rejected() {
        let mut f = forecasts(); f[0].outcome_probabilities.pop();
        assert!(matches!(validate(&f,&test()), Err(ScientificActiveInferenceError::ProbabilityWidthMismatch)));
    }
    #[test] fn horizon_mismatch_is_rejected() {
        let mut f = forecasts(); f[1].horizon_seconds = 11.0;
        assert!(matches!(validate(&f,&test()), Err(ScientificActiveInferenceError::HorizonMismatch)));
    }
}
