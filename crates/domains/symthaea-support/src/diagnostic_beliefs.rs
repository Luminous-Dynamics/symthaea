// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bound IT diagnostic hypotheses and Bayesian test selection.
//!
//! This module is intentionally not a replacement for `symthaea-causal-reasoning`.
//! It tracks epistemic belief over candidate IT causes and computes diagnostic
//! expected information gain. Structural causal models, interventions, and
//! counterfactual identification remain owned by the causal-reasoning crate.

use crate::change_timeline::{ChangeId, ChangeTimelineV1};
use crate::system_state::{EntityId, ObservationId, SystemStateGraphV1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const PROB_EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HypothesisId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisStatusV1 {
    Proposed,
    Supported,
    Weakened,
    Rejected,
    Indeterminate,
}

/// One explicitly non-factual candidate explanation for an observed IT failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalHypothesisV1 {
    pub id: HypothesisId,
    pub subject: EntityId,
    pub statement: String,
    pub status: HypothesisStatusV1,
    /// Graph revision against which the hypothesis was created.
    pub graph_revision: u64,
    #[serde(default)]
    pub supporting_evidence: BTreeSet<ObservationId>,
    #[serde(default)]
    pub contradicting_evidence: BTreeSet<ObservationId>,
    #[serde(default)]
    pub candidate_changes: BTreeSet<ChangeId>,
}

impl CausalHypothesisV1 {
    pub fn validate(
        &self,
        graph: &SystemStateGraphV1,
        timeline: &ChangeTimelineV1,
    ) -> Result<(), DiagnosticBeliefError> {
        if self.id.0.trim().is_empty() {
            return Err(DiagnosticBeliefError::EmptyIdentifier("hypothesis"));
        }
        if self.subject.0.trim().is_empty() {
            return Err(DiagnosticBeliefError::EmptyIdentifier("subject"));
        }
        if self.statement.trim().is_empty() {
            return Err(DiagnosticBeliefError::EmptyStatement(self.id.clone()));
        }
        if graph.entity(&self.subject).is_none() {
            return Err(DiagnosticBeliefError::UnknownSubject(self.subject.clone()));
        }
        if self.graph_revision > graph.revision {
            return Err(DiagnosticBeliefError::FutureGraphRevision {
                hypothesis: self.id.clone(),
                hypothesis_revision: self.graph_revision,
                graph_revision: graph.revision,
            });
        }
        for evidence in self
            .supporting_evidence
            .iter()
            .chain(self.contradicting_evidence.iter())
        {
            if graph.observation(evidence).is_none() {
                return Err(DiagnosticBeliefError::UnknownObservation(evidence.clone()));
            }
        }
        if let Some(overlap) = self
            .supporting_evidence
            .intersection(&self.contradicting_evidence)
            .next()
        {
            return Err(DiagnosticBeliefError::EvidenceRoleConflict(overlap.clone()));
        }
        for change in &self.candidate_changes {
            if timeline.get(change).is_none() {
                return Err(DiagnosticBeliefError::UnknownChange(change.clone()));
            }
        }
        Ok(())
    }
}

/// Normalized probability distribution over mutually exclusive diagnostic hypotheses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HypothesisDistributionV1 {
    probabilities: BTreeMap<HypothesisId, f64>,
}

impl HypothesisDistributionV1 {
    /// Construct from non-negative weights. Weights are normalized explicitly;
    /// this accepts scores from calibrated upstream models without pretending the
    /// caller already supplied a perfect probability simplex.
    pub fn from_weights<I>(weights: I) -> Result<Self, DiagnosticBeliefError>
    where
        I: IntoIterator<Item = (HypothesisId, f64)>,
    {
        let mut raw = BTreeMap::new();
        let mut total = 0.0;
        for (id, weight) in weights {
            if id.0.trim().is_empty() {
                return Err(DiagnosticBeliefError::EmptyIdentifier("hypothesis"));
            }
            validate_probability_like(weight, "hypothesis weight")?;
            if raw.insert(id.clone(), weight).is_some() {
                return Err(DiagnosticBeliefError::DuplicateHypothesis(id));
            }
            total += weight;
        }
        if raw.is_empty() || !total.is_finite() || total <= 0.0 {
            return Err(DiagnosticBeliefError::EmptyOrZeroDistribution);
        }
        for value in raw.values_mut() {
            *value /= total;
        }
        Ok(Self { probabilities: raw })
    }

    pub fn probability(&self, id: &HypothesisId) -> Option<f64> {
        self.probabilities.get(id).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&HypothesisId, f64)> {
        self.probabilities.iter().map(|(id, probability)| (id, *probability))
    }

    pub fn entropy_bits(&self) -> f64 {
        self.probabilities
            .values()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.log2())
            .sum()
    }

    pub fn most_likely(&self) -> Option<(&HypothesisId, f64)> {
        self.probabilities
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, probability)| (id, *probability))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DiagnosticTestId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DiagnosticOutcomeId(pub String);

/// Likelihood model P(outcome | hypothesis) for one diagnostic test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticTestModelV1 {
    pub id: DiagnosticTestId,
    pub description: String,
    /// outcome -> (hypothesis -> likelihood)
    pub outcomes: BTreeMap<DiagnosticOutcomeId, BTreeMap<HypothesisId, f64>>,
}

impl DiagnosticTestModelV1 {
    pub fn validate(
        &self,
        prior: &HypothesisDistributionV1,
    ) -> Result<(), DiagnosticBeliefError> {
        if self.id.0.trim().is_empty() {
            return Err(DiagnosticBeliefError::EmptyIdentifier("diagnostic test"));
        }
        if self.description.trim().is_empty() {
            return Err(DiagnosticBeliefError::EmptyTestDescription(self.id.clone()));
        }
        if self.outcomes.len() < 2 {
            return Err(DiagnosticBeliefError::InsufficientOutcomes(self.id.clone()));
        }

        for (outcome, likelihoods) in &self.outcomes {
            if outcome.0.trim().is_empty() {
                return Err(DiagnosticBeliefError::EmptyIdentifier("diagnostic outcome"));
            }
            for (hypothesis, likelihood) in likelihoods {
                if prior.probability(hypothesis).is_none() {
                    return Err(DiagnosticBeliefError::UnknownHypothesisInTest {
                        test: self.id.clone(),
                        hypothesis: hypothesis.clone(),
                    });
                }
                validate_unit_interval(*likelihood, "diagnostic likelihood")?;
            }
        }

        // For every hypothesis in the prior, outcomes must form a complete
        // conditional distribution. Missing likelihoods are not treated as zero.
        for (hypothesis, _) in prior.iter() {
            let mut total = 0.0;
            for (outcome, likelihoods) in &self.outcomes {
                let Some(likelihood) = likelihoods.get(hypothesis) else {
                    return Err(DiagnosticBeliefError::MissingLikelihood {
                        test: self.id.clone(),
                        outcome: outcome.clone(),
                        hypothesis: hypothesis.clone(),
                    });
                };
                total += likelihood;
            }
            if (total - 1.0).abs() > PROB_EPSILON {
                return Err(DiagnosticBeliefError::LikelihoodsDoNotSumToOne {
                    test: self.id.clone(),
                    hypothesis: hypothesis.clone(),
                    total,
                });
            }
        }
        Ok(())
    }

    /// P(outcome) under the supplied prior.
    pub fn outcome_probability(
        &self,
        prior: &HypothesisDistributionV1,
        outcome: &DiagnosticOutcomeId,
    ) -> Result<f64, DiagnosticBeliefError> {
        self.validate(prior)?;
        let likelihoods = self
            .outcomes
            .get(outcome)
            .ok_or_else(|| DiagnosticBeliefError::UnknownOutcome {
                test: self.id.clone(),
                outcome: outcome.clone(),
            })?;
        Ok(prior
            .iter()
            .map(|(hypothesis, probability)| {
                probability * likelihoods.get(hypothesis).copied().unwrap_or(0.0)
            })
            .sum())
    }

    /// Bayesian posterior P(hypothesis | outcome).
    pub fn posterior(
        &self,
        prior: &HypothesisDistributionV1,
        outcome: &DiagnosticOutcomeId,
    ) -> Result<HypothesisDistributionV1, DiagnosticBeliefError> {
        self.validate(prior)?;
        let outcome_probability = self.outcome_probability(prior, outcome)?;
        if outcome_probability <= PROB_EPSILON {
            return Err(DiagnosticBeliefError::ZeroProbabilityOutcome {
                test: self.id.clone(),
                outcome: outcome.clone(),
            });
        }
        let likelihoods = self.outcomes.get(outcome).expect("validated outcome");
        HypothesisDistributionV1::from_weights(prior.iter().map(|(hypothesis, probability)| {
            (
                hypothesis.clone(),
                probability * likelihoods.get(hypothesis).copied().unwrap_or(0.0),
            )
        }))
    }

    pub fn expected_information_gain_bits(
        &self,
        prior: &HypothesisDistributionV1,
    ) -> Result<ExpectedInformationGainV1, DiagnosticBeliefError> {
        self.validate(prior)?;
        let prior_entropy = prior.entropy_bits();
        let mut expected_posterior_entropy = 0.0;

        for outcome in self.outcomes.keys() {
            let probability = self.outcome_probability(prior, outcome)?;
            if probability <= PROB_EPSILON {
                continue;
            }
            let posterior = self.posterior(prior, outcome)?;
            expected_posterior_entropy += probability * posterior.entropy_bits();
        }

        let information_gain = (prior_entropy - expected_posterior_entropy).max(0.0);
        Ok(ExpectedInformationGainV1 {
            test_id: self.id.clone(),
            prior_entropy_bits: prior_entropy,
            expected_posterior_entropy_bits: expected_posterior_entropy,
            information_gain_bits: information_gain,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedInformationGainV1 {
    pub test_id: DiagnosticTestId,
    pub prior_entropy_bits: f64,
    pub expected_posterior_entropy_bits: f64,
    pub information_gain_bits: f64,
}

pub fn rank_tests_by_information_gain(
    prior: &HypothesisDistributionV1,
    tests: &[DiagnosticTestModelV1],
) -> Result<Vec<ExpectedInformationGainV1>, DiagnosticBeliefError> {
    let mut ranked = tests
        .iter()
        .map(|test| test.expected_information_gain_bits(prior))
        .collect::<Result<Vec<_>, _>>()?;
    ranked.sort_by(|a, b| {
        b.information_gain_bits
            .partial_cmp(&a.information_gain_bits)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.test_id.0.cmp(&b.test_id.0))
    });
    Ok(ranked)
}

fn validate_probability_like(value: f64, label: &'static str) -> Result<(), DiagnosticBeliefError> {
    if !value.is_finite() || value < 0.0 {
        Err(DiagnosticBeliefError::InvalidProbability { label, value })
    } else {
        Ok(())
    }
}

fn validate_unit_interval(value: f64, label: &'static str) -> Result<(), DiagnosticBeliefError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(DiagnosticBeliefError::InvalidProbability { label, value })
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticBeliefError {
    EmptyIdentifier(&'static str),
    EmptyStatement(HypothesisId),
    UnknownSubject(EntityId),
    FutureGraphRevision {
        hypothesis: HypothesisId,
        hypothesis_revision: u64,
        graph_revision: u64,
    },
    UnknownObservation(ObservationId),
    EvidenceRoleConflict(ObservationId),
    UnknownChange(ChangeId),
    DuplicateHypothesis(HypothesisId),
    EmptyOrZeroDistribution,
    InvalidProbability {
        label: &'static str,
        value: f64,
    },
    EmptyTestDescription(DiagnosticTestId),
    InsufficientOutcomes(DiagnosticTestId),
    UnknownHypothesisInTest {
        test: DiagnosticTestId,
        hypothesis: HypothesisId,
    },
    MissingLikelihood {
        test: DiagnosticTestId,
        outcome: DiagnosticOutcomeId,
        hypothesis: HypothesisId,
    },
    LikelihoodsDoNotSumToOne {
        test: DiagnosticTestId,
        hypothesis: HypothesisId,
        total: f64,
    },
    UnknownOutcome {
        test: DiagnosticTestId,
        outcome: DiagnosticOutcomeId,
    },
    ZeroProbabilityOutcome {
        test: DiagnosticTestId,
        outcome: DiagnosticOutcomeId,
    },
}

impl fmt::Display for DiagnosticBeliefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(kind) => write!(f, "empty {kind} identifier"),
            Self::EmptyStatement(id) => write!(f, "hypothesis {:?} has an empty statement", id.0),
            Self::UnknownSubject(id) => write!(f, "unknown hypothesis subject {:?}", id.0),
            Self::FutureGraphRevision {
                hypothesis,
                hypothesis_revision,
                graph_revision,
            } => write!(
                f,
                "hypothesis {:?} cites future graph revision {} > {}",
                hypothesis.0, hypothesis_revision, graph_revision
            ),
            Self::UnknownObservation(id) => write!(f, "unknown observation {:?}", id.0),
            Self::EvidenceRoleConflict(id) => write!(
                f,
                "observation {:?} cannot simultaneously support and contradict a hypothesis",
                id.0
            ),
            Self::UnknownChange(id) => write!(f, "unknown candidate change {:?}", id.0),
            Self::DuplicateHypothesis(id) => write!(f, "duplicate hypothesis {:?}", id.0),
            Self::EmptyOrZeroDistribution => write!(f, "hypothesis distribution is empty or zero"),
            Self::InvalidProbability { label, value } => {
                write!(f, "invalid {label} value {value}")
            }
            Self::EmptyTestDescription(id) => {
                write!(f, "diagnostic test {:?} has an empty description", id.0)
            }
            Self::InsufficientOutcomes(id) => {
                write!(f, "diagnostic test {:?} needs at least two outcomes", id.0)
            }
            Self::UnknownHypothesisInTest { test, hypothesis } => write!(
                f,
                "test {:?} references unknown hypothesis {:?}",
                test.0, hypothesis.0
            ),
            Self::MissingLikelihood {
                test,
                outcome,
                hypothesis,
            } => write!(
                f,
                "test {:?} outcome {:?} lacks likelihood for hypothesis {:?}",
                test.0, outcome.0, hypothesis.0
            ),
            Self::LikelihoodsDoNotSumToOne {
                test,
                hypothesis,
                total,
            } => write!(
                f,
                "test {:?} likelihoods for hypothesis {:?} sum to {total}, not 1",
                test.0, hypothesis.0
            ),
            Self::UnknownOutcome { test, outcome } => {
                write!(f, "unknown outcome {:?} for test {:?}", outcome.0, test.0)
            }
            Self::ZeroProbabilityOutcome { test, outcome } => write!(
                f,
                "outcome {:?} for test {:?} has zero prior probability",
                outcome.0, test.0
            ),
        }
    }
}

impl Error for DiagnosticBeliefError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(id: &str) -> HypothesisId {
        HypothesisId(id.into())
    }

    fn o(id: &str) -> DiagnosticOutcomeId {
        DiagnosticOutcomeId(id.into())
    }

    fn prior() -> HypothesisDistributionV1 {
        HypothesisDistributionV1::from_weights([
            (h("dns"), 0.4),
            (h("routing"), 0.35),
            (h("server"), 0.25),
        ])
        .unwrap()
    }

    fn dns_test() -> DiagnosticTestModelV1 {
        DiagnosticTestModelV1 {
            id: DiagnosticTestId("resolve-name".into()),
            description: "Resolve the target name using the configured resolver".into(),
            outcomes: BTreeMap::from([
                (
                    o("fails"),
                    BTreeMap::from([
                        (h("dns"), 0.95),
                        (h("routing"), 0.20),
                        (h("server"), 0.05),
                    ]),
                ),
                (
                    o("succeeds"),
                    BTreeMap::from([
                        (h("dns"), 0.05),
                        (h("routing"), 0.80),
                        (h("server"), 0.95),
                    ]),
                ),
            ]),
        }
    }

    fn weak_test() -> DiagnosticTestModelV1 {
        DiagnosticTestModelV1 {
            id: DiagnosticTestId("generic-health".into()),
            description: "Generic health probe with little discriminatory value".into(),
            outcomes: BTreeMap::from([
                (
                    o("bad"),
                    BTreeMap::from([
                        (h("dns"), 0.52),
                        (h("routing"), 0.50),
                        (h("server"), 0.48),
                    ]),
                ),
                (
                    o("good"),
                    BTreeMap::from([
                        (h("dns"), 0.48),
                        (h("routing"), 0.50),
                        (h("server"), 0.52),
                    ]),
                ),
            ]),
        }
    }

    #[test]
    fn raw_weights_are_normalized() {
        let distribution = HypothesisDistributionV1::from_weights([
            (h("a"), 2.0),
            (h("b"), 1.0),
        ])
        .unwrap();
        assert!((distribution.probability(&h("a")).unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_is_one_bit_for_even_binary_distribution() {
        let distribution = HypothesisDistributionV1::from_weights([
            (h("a"), 1.0),
            (h("b"), 1.0),
        ])
        .unwrap();
        assert!((distribution.entropy_bits() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bayesian_posterior_moves_toward_likely_cause() {
        let posterior = dns_test().posterior(&prior(), &o("fails")).unwrap();
        assert!(posterior.probability(&h("dns")).unwrap() > 0.75);
        assert_eq!(posterior.most_likely().unwrap().0, &h("dns"));
    }

    #[test]
    fn true_information_gain_ranks_discriminatory_test_first() {
        let ranked = rank_tests_by_information_gain(&prior(), &[weak_test(), dns_test()]).unwrap();
        assert_eq!(ranked[0].test_id.0, "resolve-name");
        assert!(ranked[0].information_gain_bits > ranked[1].information_gain_bits);
    }

    #[test]
    fn missing_likelihood_is_rejected_not_assumed_zero() {
        let mut test = dns_test();
        test.outcomes
            .get_mut(&o("fails"))
            .unwrap()
            .remove(&h("server"));
        let err = test.validate(&prior()).unwrap_err();
        assert!(matches!(err, DiagnosticBeliefError::MissingLikelihood { .. }));
    }

    #[test]
    fn likelihoods_must_form_distribution_per_hypothesis() {
        let mut test = dns_test();
        *test
            .outcomes
            .get_mut(&o("fails"))
            .unwrap()
            .get_mut(&h("dns"))
            .unwrap() = 0.80;
        let err = test.validate(&prior()).unwrap_err();
        assert!(matches!(
            err,
            DiagnosticBeliefError::LikelihoodsDoNotSumToOne { .. }
        ));
    }
}
