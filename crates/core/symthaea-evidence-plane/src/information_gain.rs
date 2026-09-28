// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain-neutral expected-information-gain planning.
//!
//! This module selects tests that discriminate competing models. It does not
//! select what is true and does not convert computational predictions into
//! observations. Real observations must re-enter through the evidence graph.

use serde::{Deserialize, Serialize};
use std::fmt;

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificModel {
    pub id: String,
    pub prior: f64,
    /// Stable ancestry identifier used to prevent shared model ancestry from
    /// being mistaken for independent evidence elsewhere in the graph.
    pub lineage: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeLikelihood {
    pub outcome_id: String,
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscriminatingTest {
    pub id: String,
    pub cost: f64,
    pub outcomes: Vec<OutcomeLikelihood>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationGainAssessment {
    pub test_id: String,
    pub eig_bits: f64,
    pub cost: f64,
    pub eig_per_cost: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InformationGainError {
    EmptyModels,
    EmptyTestId,
    EmptyModelId,
    EmptyLineage,
    InvalidPrior { model: String },
    PriorsDoNotNormalize,
    EmptyTestOutcomes { test: String },
    NegativeCost { test: String },
    EmptyOutcomeId { test: String },
    InvalidLikelihood { test: String, outcome: String, model_index: usize },
    LikelihoodsDoNotNormalize { test: String, model_index: usize },
    OutcomeWidthMismatch { test: String },
}

impl fmt::Display for InformationGainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModels => write!(f, "at least one scientific model is required"),
            Self::EmptyTestId => write!(f, "test id must not be empty"),
            Self::EmptyModelId => write!(f, "model id must not be empty"),
            Self::EmptyLineage => write!(f, "model lineage must not be empty"),
            Self::InvalidPrior { model } => write!(f, "invalid prior for model {model}"),
            Self::PriorsDoNotNormalize => write!(f, "model priors must sum to 1"),
            Self::EmptyTestOutcomes { test } => write!(f, "test {test} has no outcomes"),
            Self::NegativeCost { test } => write!(f, "test {test} has negative cost"),
            Self::EmptyOutcomeId { test } => write!(f, "test {test} contains an empty outcome id"),
            Self::InvalidLikelihood { test, outcome, model_index } =>
                write!(f, "invalid likelihood for test {test}, outcome {outcome}, model {model_index}"),
            Self::LikelihoodsDoNotNormalize { test, model_index } =>
                write!(f, "likelihoods for test {test}, model {model_index} do not sum to 1"),
            Self::OutcomeWidthMismatch { test } =>
                write!(f, "outcome likelihood vectors for test {test} have inconsistent widths"),
        }
    }
}
impl std::error::Error for InformationGainError {}

pub fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities.iter().filter(|p| **p > 0.0).map(|p| -p * p.log2()).sum()
}

pub fn expected_information_gain(
    models: &[ScientificModel],
    test: &DiscriminatingTest,
) -> Result<f64, InformationGainError> {
    validate(models, test)?;
    let prior: Vec<f64> = models.iter().map(|m| m.prior).collect();
    let prior_entropy = entropy_bits(&prior);
    let mut expected_posterior_entropy = 0.0;

    for outcome in &test.outcomes {
        let p_outcome: f64 = models.iter().enumerate()
            .map(|(i, model)| model.prior * outcome.probabilities[i])
            .sum();
        if p_outcome <= EPS {
            continue;
        }
        let posterior: Vec<f64> = models.iter().enumerate()
            .map(|(i, model)| model.prior * outcome.probabilities[i] / p_outcome)
            .collect();
        expected_posterior_entropy += p_outcome * entropy_bits(&posterior);
    }

    Ok((prior_entropy - expected_posterior_entropy).max(0.0))
}

pub fn assess_test(
    models: &[ScientificModel],
    test: &DiscriminatingTest,
) -> Result<InformationGainAssessment, InformationGainError> {
    let eig_bits = expected_information_gain(models, test)?;
    let denominator = test.cost.max(EPS);
    Ok(InformationGainAssessment {
        test_id: test.id.clone(),
        eig_bits,
        cost: test.cost,
        eig_per_cost: eig_bits / denominator,
    })
}

/// Deterministic decision layer: maximize information gain per unit cost,
/// then raw information gain, then test id. This is a planning preference,
/// not a truth score.
pub fn rank_tests(
    models: &[ScientificModel],
    tests: &[DiscriminatingTest],
) -> Result<Vec<InformationGainAssessment>, InformationGainError> {
    let mut assessments: Vec<_> = tests.iter()
        .map(|test| assess_test(models, test))
        .collect::<Result<_, _>>()?;
    assessments.sort_by(|a, b| {
        b.eig_per_cost.total_cmp(&a.eig_per_cost)
            .then(b.eig_bits.total_cmp(&a.eig_bits))
            .then(a.test_id.cmp(&b.test_id))
    });
    Ok(assessments)
}

pub fn validate(
    models: &[ScientificModel],
    test: &DiscriminatingTest,
) -> Result<(), InformationGainError> {
    if models.is_empty() { return Err(InformationGainError::EmptyModels); }

    let mut prior_sum = 0.0;
    for model in models {
        if model.id.trim().is_empty() { return Err(InformationGainError::EmptyModelId); }
        if model.lineage.trim().is_empty() { return Err(InformationGainError::EmptyLineage); }
        if !model.prior.is_finite() || model.prior < 0.0 {
            return Err(InformationGainError::InvalidPrior { model: model.id.clone() });
        }
        prior_sum += model.prior;
    }
    if (prior_sum - 1.0).abs() > EPS {
        return Err(InformationGainError::PriorsDoNotNormalize);
    }

    if test.id.trim().is_empty() { return Err(InformationGainError::EmptyTestId); }
    if !test.cost.is_finite() || test.cost < 0.0 {
        return Err(InformationGainError::NegativeCost { test: test.id.clone() });
    }
    if test.outcomes.is_empty() {
        return Err(InformationGainError::EmptyTestOutcomes { test: test.id.clone() });
    }
    for outcome in &test.outcomes {
        if outcome.outcome_id.trim().is_empty() {
            return Err(InformationGainError::EmptyOutcomeId { test: test.id.clone() });
        }
        if outcome.probabilities.len() != models.len() {
            return Err(InformationGainError::OutcomeWidthMismatch { test: test.id.clone() });
        }
    }
    for model_index in 0..models.len() {
        let mut sum = 0.0;
        for outcome in &test.outcomes {
            let probability = outcome.probabilities[model_index];
            if !probability.is_finite() || probability < 0.0 {
                return Err(InformationGainError::InvalidLikelihood {
                    test: test.id.clone(),
                    outcome: outcome.outcome_id.clone(),
                    model_index,
                });
            }
            sum += probability;
        }
        if (sum - 1.0).abs() > EPS {
            return Err(InformationGainError::LikelihoodsDoNotNormalize {
                test: test.id.clone(),
                model_index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> Vec<ScientificModel> {
        vec![
            ScientificModel { id: "model-a".into(), prior: 0.5, lineage: "root-a".into() },
            ScientificModel { id: "model-b".into(), prior: 0.5, lineage: "root-b".into() },
        ]
    }

    fn discriminating() -> DiscriminatingTest {
        DiscriminatingTest {
            id: "test-a".into(),
            cost: 10.0,
            outcomes: vec![
                OutcomeLikelihood { outcome_id: "a".into(), probabilities: vec![1.0, 0.0] },
                OutcomeLikelihood { outcome_id: "b".into(), probabilities: vec![0.0, 1.0] },
            ],
        }
    }

    #[test]
    fn perfect_discrimination_produces_one_bit() {
        let eig = expected_information_gain(&models(), &discriminating()).unwrap();
        assert!((eig - 1.0).abs() < 1e-12);
    }

    #[test]
    fn indistinguishable_test_produces_zero_information() {
        let test = DiscriminatingTest {
            id: "test-same".into(),
            cost: 10.0,
            outcomes: vec![
                OutcomeLikelihood { outcome_id: "a".into(), probabilities: vec![0.8, 0.8] },
                OutcomeLikelihood { outcome_id: "b".into(), probabilities: vec![0.2, 0.2] },
            ],
        };
        assert!(expected_information_gain(&models(), &test).unwrap().abs() < 1e-12);
    }

    #[test]
    fn ranking_prefers_information_per_cost() {
        let cheap = discriminating();
        let expensive = DiscriminatingTest { id: "test-b".into(), cost: 100.0, outcomes: cheap.outcomes.clone() };
        let ranked = rank_tests(&models(), &[expensive, cheap]).unwrap();
        assert_eq!(ranked[0].test_id, "test-a");
    }

    #[test]
    fn invalid_likelihoods_are_rejected() {
        let mut test = discriminating();
        test.outcomes[0].probabilities[0] = -0.1;
        assert!(matches!(
            validate(&models(), &test),
            Err(InformationGainError::InvalidLikelihood { .. })
        ));
    }

    #[test]
    fn missing_model_probability_is_rejected() {
        let mut test = discriminating();
        test.outcomes[0].probabilities.pop();
        assert!(matches!(
            validate(&models(), &test),
            Err(InformationGainError::OutcomeWidthMismatch { .. })
        ));
    }

    #[test]
    fn zero_cost_does_not_divide_by_zero() {
        let mut test = discriminating();
        test.cost = 0.0;
        let assessment = assess_test(&models(), &test).unwrap();
        assert!(assessment.eig_per_cost.is_finite());
    }

    #[test]
    fn shared_lineage_is_metadata_not_independence() {
        let mut same_lineage = models();
        same_lineage[1].lineage = "root-a".into();
        assert_eq!(same_lineage[0].lineage, same_lineage[1].lineage);
        let eig = expected_information_gain(&same_lineage, &discriminating()).unwrap();
        assert!((eig - 1.0).abs() < 1e-12);
    }
}
