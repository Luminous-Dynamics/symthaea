// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact bounded multi-fault diagnostic belief models for IT incidents.
//!
//! The existing `HypothesisDistributionV1` intentionally models mutually exclusive
//! single-cause candidates. This module preserves that fast/simple path and adds a
//! separate exact joint-state representation for incidents where several candidate
//! faults may coexist.
//!
//! Core non-equivalences:
//!
//! ```text
//! one likely fault != exactly one fault exists
//! marginal probability != fault independence
//! omitted joint state != disproved joint state
//! diagnostic posterior != structural causal proof
//! information gain != execution authority
//! ```
//!
//! V1 does not generate a powerset of fault combinations. Callers provide the
//! bounded joint states they are willing to model explicitly. That avoids silently
//! assuming independence while keeping exact Bayesian updates tractable.

use crate::diagnostic_beliefs::{DiagnosticOutcomeId, DiagnosticTestId, HypothesisId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const PROB_EPSILON: f64 = 1e-9;
pub const MAX_EXACT_MULTI_FAULT_STATES_V1: usize = 4096;
pub const MAX_MULTI_FAULT_HYPOTHESES_V1: usize = 64;

/// One explicitly modeled joint fault state. The empty set is valid and means
/// "none of the modeled faults are active" rather than "the system is healthy".
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct FaultStateV1 {
    #[serde(default)]
    pub active: BTreeSet<HypothesisId>,
}

impl FaultStateV1 {
    pub fn new<I>(active: I) -> Self
    where
        I: IntoIterator<Item = HypothesisId>,
    {
        Self {
            active: active.into_iter().collect(),
        }
    }

    pub fn none() -> Self {
        Self::default()
    }

    pub fn contains(&self, hypothesis: &HypothesisId) -> bool {
        self.active.contains(hypothesis)
    }

    pub fn fault_count(&self) -> usize {
        self.active.len()
    }
}

/// Exact normalized probability distribution over an explicitly supplied set of
/// possible concurrent-fault states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointFaultDistributionV1 {
    known_hypotheses: BTreeSet<HypothesisId>,
    probabilities: BTreeMap<FaultStateV1, f64>,
}

impl JointFaultDistributionV1 {
    pub fn from_weights<I>(
        known_hypotheses: BTreeSet<HypothesisId>,
        states: I,
    ) -> Result<Self, MultiFaultDiagnosticErrorV1>
    where
        I: IntoIterator<Item = (FaultStateV1, f64)>,
    {
        validate_hypothesis_set(&known_hypotheses)?;

        let mut raw = BTreeMap::new();
        let mut total = 0.0;
        for (state, weight) in states {
            validate_state(&known_hypotheses, &state)?;
            validate_nonnegative_finite(weight, "fault-state weight")?;
            if raw.insert(state.clone(), weight).is_some() {
                return Err(MultiFaultDiagnosticErrorV1::DuplicateFaultState(state));
            }
            total += weight;
            if raw.len() > MAX_EXACT_MULTI_FAULT_STATES_V1 {
                return Err(MultiFaultDiagnosticErrorV1::TooManyStates {
                    count: raw.len(),
                    max: MAX_EXACT_MULTI_FAULT_STATES_V1,
                });
            }
        }

        if raw.is_empty() || !total.is_finite() || total <= 0.0 {
            return Err(MultiFaultDiagnosticErrorV1::EmptyOrZeroDistribution);
        }

        for probability in raw.values_mut() {
            *probability /= total;
        }

        Ok(Self {
            known_hypotheses,
            probabilities: raw,
        })
    }

    pub fn known_hypotheses(&self) -> &BTreeSet<HypothesisId> {
        &self.known_hypotheses
    }

    pub fn states(&self) -> impl Iterator<Item = (&FaultStateV1, f64)> {
        self.probabilities
            .iter()
            .map(|(state, probability)| (state, *probability))
    }

    pub fn probability(&self, state: &FaultStateV1) -> Option<f64> {
        self.probabilities.get(state).copied()
    }

    /// Marginal P(hypothesis active) obtained from the exact joint state model.
    /// No independence assumption is introduced.
    pub fn marginal_probability(
        &self,
        hypothesis: &HypothesisId,
    ) -> Result<f64, MultiFaultDiagnosticErrorV1> {
        if !self.known_hypotheses.contains(hypothesis) {
            return Err(MultiFaultDiagnosticErrorV1::UnknownHypothesis(
                hypothesis.clone(),
            ));
        }
        Ok(self
            .probabilities
            .iter()
            .filter(|(state, _)| state.contains(hypothesis))
            .map(|(_, probability)| *probability)
            .sum())
    }

    pub fn expected_fault_count(&self) -> f64 {
        self.probabilities
            .iter()
            .map(|(state, probability)| state.fault_count() as f64 * probability)
            .sum()
    }

    pub fn entropy_bits(&self) -> f64 {
        self.probabilities
            .values()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.log2())
            .sum()
    }

    pub fn most_likely_state(&self) -> Option<(&FaultStateV1, f64)> {
        self.probabilities
            .iter()
            .max_by(|(state_a, probability_a), (state_b, probability_b)| {
                probability_a
                    .partial_cmp(probability_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| state_b.cmp(state_a))
            })
            .map(|(state, probability)| (state, *probability))
    }
}

/// Exact likelihood model P(outcome | joint fault state) for one diagnostic test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFaultDiagnosticTestModelV1 {
    pub id: DiagnosticTestId,
    pub description: String,
    /// outcome -> (joint fault state -> likelihood)
    pub outcomes: BTreeMap<DiagnosticOutcomeId, BTreeMap<FaultStateV1, f64>>,
}

impl MultiFaultDiagnosticTestModelV1 {
    pub fn validate(
        &self,
        prior: &JointFaultDistributionV1,
    ) -> Result<(), MultiFaultDiagnosticErrorV1> {
        if self.id.0.trim().is_empty() {
            return Err(MultiFaultDiagnosticErrorV1::EmptyIdentifier(
                "diagnostic test",
            ));
        }
        if self.description.trim().is_empty() {
            return Err(MultiFaultDiagnosticErrorV1::EmptyTestDescription(
                self.id.clone(),
            ));
        }
        if self.outcomes.len() < 2 {
            return Err(MultiFaultDiagnosticErrorV1::InsufficientOutcomes(
                self.id.clone(),
            ));
        }

        for (outcome, likelihoods) in &self.outcomes {
            if outcome.0.trim().is_empty() {
                return Err(MultiFaultDiagnosticErrorV1::EmptyIdentifier(
                    "diagnostic outcome",
                ));
            }
            for (state, likelihood) in likelihoods {
                if prior.probability(state).is_none() {
                    return Err(MultiFaultDiagnosticErrorV1::UnknownFaultStateInTest {
                        test: self.id.clone(),
                        state: state.clone(),
                    });
                }
                validate_unit_interval(*likelihood, "diagnostic likelihood")?;
            }
        }

        // Missing state likelihoods are unknown/error, never silently zero.
        for (state, _) in prior.states() {
            let mut total = 0.0;
            for (outcome, likelihoods) in &self.outcomes {
                let Some(likelihood) = likelihoods.get(state) else {
                    return Err(MultiFaultDiagnosticErrorV1::MissingLikelihood {
                        test: self.id.clone(),
                        outcome: outcome.clone(),
                        state: state.clone(),
                    });
                };
                total += likelihood;
            }
            if (total - 1.0).abs() > PROB_EPSILON {
                return Err(
                    MultiFaultDiagnosticErrorV1::LikelihoodsDoNotSumToOne {
                        test: self.id.clone(),
                        state: state.clone(),
                        total,
                    },
                );
            }
        }
        Ok(())
    }

    pub fn outcome_probability(
        &self,
        prior: &JointFaultDistributionV1,
        outcome: &DiagnosticOutcomeId,
    ) -> Result<f64, MultiFaultDiagnosticErrorV1> {
        self.validate(prior)?;
        let likelihoods = self.outcomes.get(outcome).ok_or_else(|| {
            MultiFaultDiagnosticErrorV1::UnknownOutcome {
                test: self.id.clone(),
                outcome: outcome.clone(),
            }
        })?;

        Ok(prior
            .states()
            .map(|(state, probability)| {
                probability * likelihoods.get(state).copied().unwrap_or(0.0)
            })
            .sum())
    }

    pub fn posterior(
        &self,
        prior: &JointFaultDistributionV1,
        outcome: &DiagnosticOutcomeId,
    ) -> Result<JointFaultDistributionV1, MultiFaultDiagnosticErrorV1> {
        self.validate(prior)?;
        let outcome_probability = self.outcome_probability(prior, outcome)?;
        if outcome_probability <= PROB_EPSILON {
            return Err(MultiFaultDiagnosticErrorV1::ZeroProbabilityOutcome {
                test: self.id.clone(),
                outcome: outcome.clone(),
            });
        }

        let likelihoods = self.outcomes.get(outcome).expect("validated outcome");
        JointFaultDistributionV1::from_weights(
            prior.known_hypotheses.clone(),
            prior.states().map(|(state, probability)| {
                (
                    state.clone(),
                    probability * likelihoods.get(state).copied().unwrap_or(0.0),
                )
            }),
        )
    }

    pub fn expected_information_gain_bits(
        &self,
        prior: &JointFaultDistributionV1,
    ) -> Result<MultiFaultExpectedInformationGainV1, MultiFaultDiagnosticErrorV1> {
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

        Ok(MultiFaultExpectedInformationGainV1 {
            test_id: self.id.clone(),
            prior_entropy_bits: prior_entropy,
            expected_posterior_entropy_bits: expected_posterior_entropy,
            information_gain_bits: (prior_entropy - expected_posterior_entropy).max(0.0),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiFaultExpectedInformationGainV1 {
    pub test_id: DiagnosticTestId,
    pub prior_entropy_bits: f64,
    pub expected_posterior_entropy_bits: f64,
    pub information_gain_bits: f64,
}

pub fn rank_multifault_tests_by_information_gain_v1(
    prior: &JointFaultDistributionV1,
    tests: &[MultiFaultDiagnosticTestModelV1],
) -> Result<Vec<MultiFaultExpectedInformationGainV1>, MultiFaultDiagnosticErrorV1> {
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

fn validate_hypothesis_set(
    hypotheses: &BTreeSet<HypothesisId>,
) -> Result<(), MultiFaultDiagnosticErrorV1> {
    if hypotheses.is_empty() {
        return Err(MultiFaultDiagnosticErrorV1::EmptyHypothesisSet);
    }
    if hypotheses.len() > MAX_MULTI_FAULT_HYPOTHESES_V1 {
        return Err(MultiFaultDiagnosticErrorV1::TooManyHypotheses {
            count: hypotheses.len(),
            max: MAX_MULTI_FAULT_HYPOTHESES_V1,
        });
    }
    if hypotheses.iter().any(|id| id.0.trim().is_empty()) {
        return Err(MultiFaultDiagnosticErrorV1::EmptyIdentifier("hypothesis"));
    }
    Ok(())
}

fn validate_state(
    known_hypotheses: &BTreeSet<HypothesisId>,
    state: &FaultStateV1,
) -> Result<(), MultiFaultDiagnosticErrorV1> {
    if let Some(unknown) = state
        .active
        .iter()
        .find(|hypothesis| !known_hypotheses.contains(*hypothesis))
    {
        return Err(MultiFaultDiagnosticErrorV1::UnknownHypothesis(
            unknown.clone(),
        ));
    }
    Ok(())
}

fn validate_nonnegative_finite(
    value: f64,
    label: &'static str,
) -> Result<(), MultiFaultDiagnosticErrorV1> {
    if !value.is_finite() || value < 0.0 {
        Err(MultiFaultDiagnosticErrorV1::InvalidProbability { label, value })
    } else {
        Ok(())
    }
}

fn validate_unit_interval(
    value: f64,
    label: &'static str,
) -> Result<(), MultiFaultDiagnosticErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(MultiFaultDiagnosticErrorV1::InvalidProbability { label, value })
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MultiFaultDiagnosticErrorV1 {
    EmptyIdentifier(&'static str),
    EmptyHypothesisSet,
    TooManyHypotheses { count: usize, max: usize },
    TooManyStates { count: usize, max: usize },
    UnknownHypothesis(HypothesisId),
    DuplicateFaultState(FaultStateV1),
    EmptyOrZeroDistribution,
    InvalidProbability { label: &'static str, value: f64 },
    EmptyTestDescription(DiagnosticTestId),
    InsufficientOutcomes(DiagnosticTestId),
    UnknownFaultStateInTest {
        test: DiagnosticTestId,
        state: FaultStateV1,
    },
    MissingLikelihood {
        test: DiagnosticTestId,
        outcome: DiagnosticOutcomeId,
        state: FaultStateV1,
    },
    LikelihoodsDoNotSumToOne {
        test: DiagnosticTestId,
        state: FaultStateV1,
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

impl fmt::Display for MultiFaultDiagnosticErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(kind) => write!(f, "empty {kind} identifier"),
            Self::EmptyHypothesisSet => write!(f, "multi-fault hypothesis set is empty"),
            Self::TooManyHypotheses { count, max } => write!(
                f,
                "multi-fault model has {count} hypotheses; exact V1 maximum is {max}"
            ),
            Self::TooManyStates { count, max } => write!(
                f,
                "multi-fault model has {count} states; exact V1 maximum is {max}"
            ),
            Self::UnknownHypothesis(id) => {
                write!(f, "unknown multi-fault hypothesis {:?}", id.0)
            }
            Self::DuplicateFaultState(state) => {
                write!(f, "duplicate joint fault state {:?}", state.active)
            }
            Self::EmptyOrZeroDistribution => {
                write!(f, "joint fault distribution is empty or zero")
            }
            Self::InvalidProbability { label, value } => {
                write!(f, "invalid {label} value {value}")
            }
            Self::EmptyTestDescription(id) => {
                write!(f, "multi-fault diagnostic test {:?} has an empty description", id.0)
            }
            Self::InsufficientOutcomes(id) => write!(
                f,
                "multi-fault diagnostic test {:?} needs at least two outcomes",
                id.0
            ),
            Self::UnknownFaultStateInTest { test, state } => write!(
                f,
                "test {:?} contains a state not present in the prior: {:?}",
                test.0, state.active
            ),
            Self::MissingLikelihood {
                test,
                outcome,
                state,
            } => write!(
                f,
                "test {:?} outcome {:?} is missing likelihood for state {:?}",
                test.0, outcome.0, state.active
            ),
            Self::LikelihoodsDoNotSumToOne { test, state, total } => write!(
                f,
                "test {:?} likelihoods for state {:?} sum to {total}, not 1",
                test.0, state.active
            ),
            Self::UnknownOutcome { test, outcome } => write!(
                f,
                "test {:?} has no outcome {:?}",
                test.0, outcome.0
            ),
            Self::ZeroProbabilityOutcome { test, outcome } => write!(
                f,
                "test {:?} outcome {:?} has zero probability under the prior",
                test.0, outcome.0
            ),
        }
    }
}

impl Error for MultiFaultDiagnosticErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(name: &str) -> HypothesisId {
        HypothesisId(name.into())
    }

    fn state(names: &[&str]) -> FaultStateV1 {
        FaultStateV1::new(names.iter().map(|name| h(name)))
    }

    fn prior() -> JointFaultDistributionV1 {
        let known = BTreeSet::from([h("dns"), h("mtu")]);
        JointFaultDistributionV1::from_weights(
            known,
            [
                (FaultStateV1::none(), 0.10),
                (state(&["dns"]), 0.25),
                (state(&["mtu"]), 0.25),
                (state(&["dns", "mtu"]), 0.40),
            ],
        )
        .unwrap()
    }

    #[test]
    fn exact_joint_distribution_preserves_concurrent_fault_probability() {
        let prior = prior();
        assert!((prior.marginal_probability(&h("dns")).unwrap() - 0.65).abs() < 1e-9);
        assert!((prior.marginal_probability(&h("mtu")).unwrap() - 0.65).abs() < 1e-9);
        assert!((prior.expected_fault_count() - 1.30).abs() < 1e-9);
        let (most_likely, probability) = prior.most_likely_state().unwrap();
        assert_eq!(most_likely, &state(&["dns", "mtu"]));
        assert!((probability - 0.40).abs() < 1e-9);
    }

    #[test]
    fn no_fault_state_is_explicit_not_implicit_health_claim() {
        let prior = prior();
        assert_eq!(prior.probability(&FaultStateV1::none()), Some(0.10));
    }

    #[test]
    fn posterior_can_increase_dual_fault_state_without_forcing_exclusivity() {
        let prior = prior();
        let positive = DiagnosticOutcomeId("positive".into());
        let negative = DiagnosticOutcomeId("negative".into());
        let test = MultiFaultDiagnosticTestModelV1 {
            id: DiagnosticTestId("combined-evidence".into()),
            description: "Check a signal expected especially when both faults coexist".into(),
            outcomes: BTreeMap::from([
                (
                    positive.clone(),
                    BTreeMap::from([
                        (FaultStateV1::none(), 0.05),
                        (state(&["dns"]), 0.20),
                        (state(&["mtu"]), 0.20),
                        (state(&["dns", "mtu"]), 0.90),
                    ]),
                ),
                (
                    negative,
                    BTreeMap::from([
                        (FaultStateV1::none(), 0.95),
                        (state(&["dns"]), 0.80),
                        (state(&["mtu"]), 0.80),
                        (state(&["dns", "mtu"]), 0.10),
                    ]),
                ),
            ]),
        };
        let posterior = test.posterior(&prior, &positive).unwrap();
        assert!(posterior.probability(&state(&["dns", "mtu"])).unwrap() > 0.70);
        assert!(posterior.marginal_probability(&h("dns")).unwrap() > 0.75);
        assert!(posterior.marginal_probability(&h("mtu")).unwrap() > 0.75);
    }

    #[test]
    fn missing_joint_state_likelihood_is_rejected_not_zero_filled() {
        let prior = prior();
        let test = MultiFaultDiagnosticTestModelV1 {
            id: DiagnosticTestId("incomplete".into()),
            description: "Incomplete likelihood model".into(),
            outcomes: BTreeMap::from([
                (
                    DiagnosticOutcomeId("yes".into()),
                    BTreeMap::from([(FaultStateV1::none(), 0.5)]),
                ),
                (
                    DiagnosticOutcomeId("no".into()),
                    BTreeMap::from([(FaultStateV1::none(), 0.5)]),
                ),
            ]),
        };
        assert!(matches!(
            test.validate(&prior),
            Err(MultiFaultDiagnosticErrorV1::MissingLikelihood { .. })
        ));
    }

    #[test]
    fn unknown_hypothesis_in_joint_state_is_rejected() {
        let known = BTreeSet::from([h("dns")]);
        assert!(matches!(
            JointFaultDistributionV1::from_weights(known, [(state(&["dns", "mtu"]), 1.0)]),
            Err(MultiFaultDiagnosticErrorV1::UnknownHypothesis(_))
        ));
    }

    #[test]
    fn information_gain_is_computed_over_joint_state_entropy() {
        let prior = prior();
        let yes = DiagnosticOutcomeId("yes".into());
        let no = DiagnosticOutcomeId("no".into());
        let sharp = MultiFaultDiagnosticTestModelV1 {
            id: DiagnosticTestId("sharp".into()),
            description: "Separates dual-fault state from the rest".into(),
            outcomes: BTreeMap::from([
                (
                    yes.clone(),
                    BTreeMap::from([
                        (FaultStateV1::none(), 0.05),
                        (state(&["dns"]), 0.05),
                        (state(&["mtu"]), 0.05),
                        (state(&["dns", "mtu"]), 0.95),
                    ]),
                ),
                (
                    no.clone(),
                    BTreeMap::from([
                        (FaultStateV1::none(), 0.95),
                        (state(&["dns"]), 0.95),
                        (state(&["mtu"]), 0.95),
                        (state(&["dns", "mtu"]), 0.05),
                    ]),
                ),
            ]),
        };
        let weak = MultiFaultDiagnosticTestModelV1 {
            id: DiagnosticTestId("weak".into()),
            description: "Nearly uninformative test".into(),
            outcomes: BTreeMap::from([
                (
                    yes,
                    prior
                        .states()
                        .map(|(state, _)| (state.clone(), 0.52))
                        .collect(),
                ),
                (
                    no,
                    prior
                        .states()
                        .map(|(state, _)| (state.clone(), 0.48))
                        .collect(),
                ),
            ]),
        };
        let ranked = rank_multifault_tests_by_information_gain_v1(&prior, &[weak, sharp]).unwrap();
        assert_eq!(ranked[0].test_id.0, "sharp");
        assert!(ranked[0].information_gain_bits > ranked[1].information_gain_bits);
    }
}
