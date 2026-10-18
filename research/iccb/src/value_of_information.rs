//! Decision-theoretic value of information for capital diligence.
//!
//! Estimates the expected value of one noisy measurement before paying for it.
//! Its modeled outcomes are mutually exclusive and exhaustive per state.
//! Inputs are explicit synthetic/elicited priors, likelihoods and payoffs;
//! this module does not infer calibrated probabilities from raw evidence.

const BPS: i128 = 10_000;
const BPS_SQUARED: i128 = 100_000_000;
const MAX_STATES: usize = 16;
const MAX_SIGNAL_OUTCOMES: usize = 16;
const MAX_ACTIONS: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InformationState {
    pub id: String,
    pub prior_probability_bps: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalLikelihood {
    pub state_id: String,
    /// Probability of this outcome conditional on this state, in basis points.
    pub likelihood_bps: u32,
}

/// One possible, mutually exclusive outcome of a single measurement action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalOutcome {
    pub id: String,
    /// P(this outcome | each state); the set of outcome likelihoods sums to 1 per state.
    pub likelihoods: Vec<SignalLikelihood>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionPayoff {
    pub action_id: String,
    pub state_id: String,
    /// Evaluator-defined net value in integer minor units; negative values are allowed.
    pub net_value_minor: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueOfInformationScenario {
    pub id: String,
    /// Direct cost of the survey / measurement in currency minor units.
    pub information_cost_minor: i64,
    /// Economic cost of waiting for the signal (e.g. lost contribution margin).
    pub delay_cost_minor: i64,
    /// Financing / carrying cost attributable to the acquisition delay.
    pub financing_cost_minor: i64,
    pub states: Vec<InformationState>,
    pub signal_outcomes: Vec<SignalOutcome>,
    pub action_ids: Vec<String>,
    pub payoffs: Vec<DecisionPayoff>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueOfInformationError {
    EmptyIdentifier,
    DuplicateIdentifier,
    EmptyStatesOutcomesOrActions,
    ScenarioTooLarge,
    PriorProbabilitiesDoNotSumTo10000,
    OutcomeLikelihoodMassDoesNotSumTo10000,
    IncompleteLikelihoodTable,
    DuplicateLikelihoodEntry,
    IncompletePayoffTable,
    DuplicatePayoffEntry,
    UnknownStateReference,
    UnknownStateOrActionReference,
    NegativeInformationCost,
    InformationCostOverflow,
    ExpectedValueOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeDecision {
    pub outcome_id: String,
    pub chosen_action_id: String,
    /// Exact probability numerator over 100,000,000 (basis-points squared).
    pub outcome_probability_numerator_bp2: u64,
    pub joint_weighted_payoff_numerator: i128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueOfInformationReport {
    pub scenario_id: String,
    pub baseline_action_id: String,
    pub baseline_expected_value_minor: i64,
    pub expected_value_with_signal_minor: i64,
    pub gross_evsi_minor: i64,
    pub information_cost_minor: i64,
    pub delay_cost_minor: i64,
    pub financing_cost_minor: i64,
    pub total_cost_minor: i64,
    pub net_evsi_minor: i64,
    pub perfect_information_expected_value_minor: i64,
    pub evpi_minor: i64,
    pub worthwhile_at_declared_cost: bool,
    pub decisions_by_outcome: Vec<OutcomeDecision>,
}

impl ValueOfInformationScenario {
    pub fn evaluate(&self) -> Result<ValueOfInformationReport, ValueOfInformationError> {
        self.validate()?;

        let baseline_action_id = self.best_action_without_signal();
        let baseline_numerator = self.expected_value_numerator(&baseline_action_id);
        let baseline_expected_value_minor = to_i64(
            baseline_numerator.div_euclid(BPS),
        )?;

        let mut with_signal_numerator = 0_i128;
        let mut decisions_by_outcome = Vec::with_capacity(self.signal_outcomes.len());

        let mut outcomes = self.signal_outcomes.iter().collect::<Vec<_>>();
        outcomes.sort_by(|left, right| left.id.cmp(&right.id));

        for outcome in outcomes {
            let mut marginal_probability_numerator = 0_i128;
            for state in &self.states {
                let likelihood = self.likelihood_bps(outcome, &state.id);
                marginal_probability_numerator +=
                    i128::from(state.prior_probability_bps) * i128::from(likelihood);
            }

            let mut best_action_id: Option<&str> = None;
            let mut best_conditional_numerator = i128::MIN;
            for action_id in &self.action_ids {
                let mut conditional_numerator = 0_i128;
                for state in &self.states {
                    let prior = i128::from(state.prior_probability_bps);
                    let likelihood = i128::from(self.likelihood_bps(outcome, &state.id));
                    let payoff = i128::from(self.payoff(action_id, &state.id));
                    conditional_numerator += prior * likelihood * payoff;
                }

                let replace = conditional_numerator > best_conditional_numerator
                    || (conditional_numerator == best_conditional_numerator
                        && best_action_id.is_none_or(|previous| action_id.as_str() < previous));
                if replace {
                    best_action_id = Some(action_id.as_str());
                    best_conditional_numerator = conditional_numerator;
                }
            }

            // A zero-probability outcome never affects expected value; retain a
            // deterministic action choice for complete, reproducible reports.
            let chosen_action_id = best_action_id
                .expect("validated action set must be non-empty")
                .to_owned();
            with_signal_numerator += best_conditional_numerator;

            decisions_by_outcome.push(OutcomeDecision {
                outcome_id: outcome.id.clone(),
                chosen_action_id,
                outcome_probability_numerator_bp2: u64::try_from(
                    marginal_probability_numerator,
                )
                .map_err(|_| ValueOfInformationError::ExpectedValueOverflow)?,
                joint_weighted_payoff_numerator: best_conditional_numerator,
            });
        }

        let expected_value_with_signal_minor =
            to_i64(with_signal_numerator.div_euclid(BPS_SQUARED))?;
        let gross_evsi_numerator =
            with_signal_numerator - baseline_numerator * BPS;
        let gross_evsi_minor = to_i64(gross_evsi_numerator.div_euclid(BPS_SQUARED))?;
        let total_cost_minor = self
            .information_cost_minor
            .checked_add(self.delay_cost_minor)
            .and_then(|cost| cost.checked_add(self.financing_cost_minor))
            .ok_or(ValueOfInformationError::InformationCostOverflow)?;
        let net_evsi_minor = gross_evsi_minor
            .checked_sub(total_cost_minor)
            .ok_or(ValueOfInformationError::ExpectedValueOverflow)?;

        let mut perfect_information_numerator = 0_i128;
        for state in &self.states {
            let best_payoff = self
                .action_ids
                .iter()
                .map(|action_id| self.payoff(action_id, &state.id))
                .max()
                .expect("validated action set must be non-empty");
            perfect_information_numerator +=
                i128::from(state.prior_probability_bps) * i128::from(best_payoff);
        }
        let perfect_information_expected_value_minor =
            to_i64(perfect_information_numerator.div_euclid(BPS))?;
        let evpi_minor = to_i64(
            (perfect_information_numerator - baseline_numerator).div_euclid(BPS),
        )?;

        Ok(ValueOfInformationReport {
            scenario_id: self.id.clone(),
            baseline_action_id,
            baseline_expected_value_minor,
            expected_value_with_signal_minor,
            gross_evsi_minor,
            information_cost_minor: self.information_cost_minor,
            delay_cost_minor: self.delay_cost_minor,
            financing_cost_minor: self.financing_cost_minor,
            total_cost_minor,
            net_evsi_minor,
            perfect_information_expected_value_minor,
            evpi_minor,
            worthwhile_at_declared_cost: net_evsi_minor > 0,
            decisions_by_outcome,
        })
    }

    fn validate(&self) -> Result<(), ValueOfInformationError> {
        if self.id.trim().is_empty()
            || self.states.iter().any(|state| state.id.trim().is_empty())
            || self.signal_outcomes.iter().any(|outcome| outcome.id.trim().is_empty())
            || self.action_ids.iter().any(|action| action.trim().is_empty())
            || self.payoffs.iter().any(|payoff| {
                payoff.action_id.trim().is_empty() || payoff.state_id.trim().is_empty()
            })
        {
            return Err(ValueOfInformationError::EmptyIdentifier);
        }
        if self.information_cost_minor < 0
            || self.delay_cost_minor < 0
            || self.financing_cost_minor < 0
        {
            return Err(ValueOfInformationError::NegativeInformationCost);
        }
        self.information_cost_minor
            .checked_add(self.delay_cost_minor)
            .and_then(|cost| cost.checked_add(self.financing_cost_minor))
            .ok_or(ValueOfInformationError::InformationCostOverflow)?;
        if self.states.is_empty() || self.signal_outcomes.is_empty() || self.action_ids.is_empty() {
            return Err(ValueOfInformationError::EmptyStatesOutcomesOrActions);
        }
        if self.states.len() > MAX_STATES
            || self.signal_outcomes.len() > MAX_SIGNAL_OUTCOMES
            || self.action_ids.len() > MAX_ACTIONS
            || self.payoffs.len() > MAX_STATES * MAX_ACTIONS
        {
            return Err(ValueOfInformationError::ScenarioTooLarge);
        }
        ensure_unique(self.states.iter().map(|state| state.id.as_str()))?;
        ensure_unique(self.signal_outcomes.iter().map(|outcome| outcome.id.as_str()))?;
        ensure_unique(self.action_ids.iter().map(String::as_str))?;

        if self
            .states
            .iter()
            .map(|state| u64::from(state.prior_probability_bps))
            .sum::<u64>()
            != 10_000
        {
            return Err(ValueOfInformationError::PriorProbabilitiesDoNotSumTo10000);
        }

        let state_ids = self
            .states
            .iter()
            .map(|state| state.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let action_ids = self
            .action_ids
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let mut likelihood_keys = std::collections::BTreeSet::new();
        let mut likelihood_sums = self
            .states
            .iter()
            .map(|state| (state.id.as_str(), 0_u64))
            .collect::<std::collections::BTreeMap<_, _>>();

        for outcome in &self.signal_outcomes {
            for likelihood in &outcome.likelihoods {
                if !state_ids.contains(likelihood.state_id.as_str()) {
                    return Err(ValueOfInformationError::UnknownStateReference);
                }
                if !likelihood_keys.insert((outcome.id.as_str(), likelihood.state_id.as_str())) {
                    return Err(ValueOfInformationError::DuplicateLikelihoodEntry);
                }
                *likelihood_sums
                    .get_mut(likelihood.state_id.as_str())
                    .expect("state identity checked above") += u64::from(likelihood.likelihood_bps);
            }
            if outcome.likelihoods.len() != self.states.len() {
                return Err(ValueOfInformationError::IncompleteLikelihoodTable);
            }
        }

        if likelihood_sums.values().any(|sum| *sum != 10_000) {
            return Err(ValueOfInformationError::OutcomeLikelihoodMassDoesNotSumTo10000);
        }

        let mut payoff_keys = std::collections::BTreeSet::new();
        for payoff in &self.payoffs {
            if !state_ids.contains(payoff.state_id.as_str())
                || !action_ids.contains(payoff.action_id.as_str())
            {
                return Err(ValueOfInformationError::UnknownStateOrActionReference);
            }
            if !payoff_keys.insert((payoff.action_id.as_str(), payoff.state_id.as_str())) {
                return Err(ValueOfInformationError::DuplicatePayoffEntry);
            }
        }
        if payoff_keys.len() != self.states.len() * self.action_ids.len() {
            return Err(ValueOfInformationError::IncompletePayoffTable);
        }

        Ok(())
    }

    fn best_action_without_signal(&self) -> String {
        let mut best_action: Option<&str> = None;
        let mut best_numerator = i128::MIN;

        for action_id in &self.action_ids {
            let numerator = self.expected_value_numerator(action_id);
            let replace = numerator > best_numerator
                || (numerator == best_numerator
                    && best_action.is_none_or(|previous| action_id.as_str() < previous));
            if replace {
                best_action = Some(action_id.as_str());
                best_numerator = numerator;
            }
        }
        best_action
            .expect("validated action set must be non-empty")
            .to_owned()
    }

    fn expected_value_numerator(&self, action_id: &str) -> i128 {
        self.states.iter().fold(0_i128, |total, state| {
            total
                + i128::from(state.prior_probability_bps)
                    * i128::from(self.payoff(action_id, &state.id))
        })
    }

    fn likelihood_bps(&self, outcome: &SignalOutcome, state_id: &str) -> u32 {
        outcome
            .likelihoods
            .iter()
            .find(|likelihood| likelihood.state_id == state_id)
            .map(|likelihood| likelihood.likelihood_bps)
            .expect("likelihood table validated before evaluation")
    }

    fn payoff(&self, action_id: &str, state_id: &str) -> i64 {
        self.payoffs
            .iter()
            .find(|payoff| payoff.action_id == action_id && payoff.state_id == state_id)
            .map(|payoff| payoff.net_value_minor)
            .expect("payoff table validated before evaluation")
    }
}

fn ensure_unique<'a>(
    values: impl Iterator<Item = &'a str>,
) -> Result<(), ValueOfInformationError> {
    let mut seen = std::collections::BTreeSet::new();
    if values.into_iter().all(|value| seen.insert(value)) {
        Ok(())
    } else {
        Err(ValueOfInformationError::DuplicateIdentifier)
    }
}

fn to_i64(value: i128) -> Result<i64, ValueOfInformationError> {
    i64::try_from(value).map_err(|_| ValueOfInformationError::ExpectedValueOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_signal_scenario(information_cost_minor: i64) -> ValueOfInformationScenario {
        ValueOfInformationScenario {
            id: "ICCB-VOI-PERFECT-SIGNAL".into(),
            information_cost_minor,
            delay_cost_minor: 0,
            financing_cost_minor: 0,
            states: vec![
                InformationState {
                    id: "high-demand".into(),
                    prior_probability_bps: 5_000,
                },
                InformationState {
                    id: "low-demand".into(),
                    prior_probability_bps: 5_000,
                },
            ],
            signal_outcomes: vec![
                SignalOutcome {
                    id: "signal-high".into(),
                    likelihoods: vec![
                        SignalLikelihood {
                            state_id: "high-demand".into(),
                            likelihood_bps: 10_000,
                        },
                        SignalLikelihood {
                            state_id: "low-demand".into(),
                            likelihood_bps: 0,
                        },
                    ],
                },
                SignalOutcome {
                    id: "signal-low".into(),
                    likelihoods: vec![
                        SignalLikelihood {
                            state_id: "high-demand".into(),
                            likelihood_bps: 0,
                        },
                        SignalLikelihood {
                            state_id: "low-demand".into(),
                            likelihood_bps: 10_000,
                        },
                    ],
                },
            ],
            action_ids: vec!["build".into(), "avoid".into()],
            payoffs: vec![
                DecisionPayoff {
                    action_id: "build".into(),
                    state_id: "high-demand".into(),
                    net_value_minor: 100,
                },
                DecisionPayoff {
                    action_id: "build".into(),
                    state_id: "low-demand".into(),
                    net_value_minor: -100,
                },
                DecisionPayoff {
                    action_id: "avoid".into(),
                    state_id: "high-demand".into(),
                    net_value_minor: 0,
                },
                DecisionPayoff {
                    action_id: "avoid".into(),
                    state_id: "low-demand".into(),
                    net_value_minor: 0,
                },
            ],
        }
    }

    #[test]
    fn informative_signal_changes_the_choice_and_has_positive_evsi() {
        let report = perfect_signal_scenario(20).evaluate().unwrap();
        assert_eq!(report.baseline_action_id, "avoid");
        assert_eq!(report.baseline_expected_value_minor, 0);
        assert_eq!(report.expected_value_with_signal_minor, 50);
        assert_eq!(report.gross_evsi_minor, 50);
        assert_eq!(report.net_evsi_minor, 30);
        assert_eq!(report.perfect_information_expected_value_minor, 50);
        assert_eq!(report.evpi_minor, 50);
        assert!(report.worthwhile_at_declared_cost);
        assert_eq!(report.decisions_by_outcome.len(), 2);
    }

    #[test]
    fn noisy_signal_has_lower_value_than_perfect_information() {
        let mut input = perfect_signal_scenario(0);
        input.signal_outcomes = vec![
            SignalOutcome {
                id: "survey-high".into(),
                likelihoods: vec![
                    SignalLikelihood {
                        state_id: "high-demand".into(),
                        likelihood_bps: 7_500,
                    },
                    SignalLikelihood {
                        state_id: "low-demand".into(),
                        likelihood_bps: 2_500,
                    },
                ],
            },
            SignalOutcome {
                id: "survey-low".into(),
                likelihoods: vec![
                    SignalLikelihood {
                        state_id: "high-demand".into(),
                        likelihood_bps: 2_500,
                    },
                    SignalLikelihood {
                        state_id: "low-demand".into(),
                        likelihood_bps: 7_500,
                    },
                ],
            },
        ];

        let report = input.evaluate().unwrap();
        assert_eq!(report.gross_evsi_minor, 25);
        assert_eq!(report.evpi_minor, 50);
        assert!(report.gross_evsi_minor < report.evpi_minor);
    }

    #[test]
    fn matches_symtropy_cislunar_survey_reference_fixture() {
        // Shared with Symtropy issue #1738; synthetic contract values only.
        // Rich prior 0.40, build payoff +100 / -150, survey cost 10,
        // sensitivity 0.90 and specificity 0.90 give gross EVSI 27, net 17,
        // and EVPI 40. No real lunar-resource or project estimate is implied.
        let input = ValueOfInformationScenario {
            id: "SYM-EVSI-CISLUNAR-REF-001".into(),
            information_cost_minor: 10,
            delay_cost_minor: 0,
            financing_cost_minor: 0,
            states: vec![
                InformationState {
                    id: "rich-deposit".into(),
                    prior_probability_bps: 4_000,
                },
                InformationState {
                    id: "poor-deposit".into(),
                    prior_probability_bps: 6_000,
                },
            ],
            signal_outcomes: vec![
                SignalOutcome {
                    id: "survey-positive".into(),
                    likelihoods: vec![
                        SignalLikelihood {
                            state_id: "rich-deposit".into(),
                            likelihood_bps: 9_000,
                        },
                        SignalLikelihood {
                            state_id: "poor-deposit".into(),
                            likelihood_bps: 1_000,
                        },
                    ],
                },
                SignalOutcome {
                    id: "survey-negative".into(),
                    likelihoods: vec![
                        SignalLikelihood {
                            state_id: "rich-deposit".into(),
                            likelihood_bps: 1_000,
                        },
                        SignalLikelihood {
                            state_id: "poor-deposit".into(),
                            likelihood_bps: 9_000,
                        },
                    ],
                },
            ],
            action_ids: vec!["build-isru".into(), "import".into()],
            payoffs: vec![
                DecisionPayoff {
                    action_id: "build-isru".into(),
                    state_id: "rich-deposit".into(),
                    net_value_minor: 100,
                },
                DecisionPayoff {
                    action_id: "build-isru".into(),
                    state_id: "poor-deposit".into(),
                    net_value_minor: -150,
                },
                DecisionPayoff {
                    action_id: "import".into(),
                    state_id: "rich-deposit".into(),
                    net_value_minor: 0,
                },
                DecisionPayoff {
                    action_id: "import".into(),
                    state_id: "poor-deposit".into(),
                    net_value_minor: 0,
                },
            ],
        };

        let report = input.evaluate().unwrap();
        assert_eq!(report.baseline_action_id, "import");
        assert_eq!(report.baseline_expected_value_minor, 0);
        assert_eq!(report.expected_value_with_signal_minor, 27);
        assert_eq!(report.gross_evsi_minor, 27);
        assert_eq!(report.information_cost_minor, 10);
        assert_eq!(report.net_evsi_minor, 17);
        assert_eq!(report.perfect_information_expected_value_minor, 40);
        assert_eq!(report.evpi_minor, 40);
        assert!(report.worthwhile_at_declared_cost);
    }

    #[test]
    fn total_diligence_cost_includes_survey_delay_and_financing() {
        let mut input = perfect_signal_scenario(50);
        input.delay_cost_minor = 30;
        input.financing_cost_minor = 25;
        let report = input.evaluate().unwrap();

        assert_eq!(report.gross_evsi_minor, 50);
        assert_eq!(report.information_cost_minor, 50);
        assert_eq!(report.delay_cost_minor, 30);
        assert_eq!(report.financing_cost_minor, 25);
        assert_eq!(report.total_cost_minor, 105);
        assert_eq!(report.net_evsi_minor, -55);
        assert!(!report.worthwhile_at_declared_cost);
    }

    #[test]
    fn negative_delay_or_financing_cost_fails_closed() {
        let mut input = perfect_signal_scenario(0);
        input.delay_cost_minor = -1;
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::NegativeInformationCost)
        );

        let mut input = perfect_signal_scenario(0);
        input.financing_cost_minor = -1;
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::NegativeInformationCost)
        );
    }

    #[test]
    fn cost_component_overflow_fails_closed() {
        let mut input = perfect_signal_scenario(i64::MAX);
        input.delay_cost_minor = 1;
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::InformationCostOverflow)
        );
    }

    #[test]
    fn information_cost_can_exceed_gross_value() {
        let report = perfect_signal_scenario(101).evaluate().unwrap();
        assert_eq!(report.gross_evsi_minor, 50);
        assert_eq!(report.net_evsi_minor, -51);
        assert!(!report.worthwhile_at_declared_cost);
    }

    #[test]
    fn uninformative_signal_has_zero_evsi() {
        let mut input = perfect_signal_scenario(0);
        input.signal_outcomes = vec![SignalOutcome {
            id: "uninformative".into(),
            likelihoods: vec![
                SignalLikelihood {
                    state_id: "high-demand".into(),
                    likelihood_bps: 10_000,
                },
                SignalLikelihood {
                    state_id: "low-demand".into(),
                    likelihood_bps: 10_000,
                },
            ],
        }];
        let report = input.evaluate().unwrap();
        assert_eq!(report.gross_evsi_minor, 0);
        assert_eq!(report.net_evsi_minor, 0);
    }

    #[test]
    fn sample_information_value_does_not_exceed_perfect_information_ceiling() {
        let report = perfect_signal_scenario(0).evaluate().unwrap();
        assert!(report.gross_evsi_minor <= report.evpi_minor);
    }

    #[test]
    fn oversized_state_space_fails_closed() {
        let mut input = perfect_signal_scenario(0);
        input.states = (0..=MAX_STATES)
            .map(|index| InformationState {
                id: format!("state-{index}"),
                prior_probability_bps: 0,
            })
            .collect();
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::ScenarioTooLarge)
        );
    }

    #[test]
    fn invalid_priors_fail_closed() {
        let mut input = perfect_signal_scenario(0);
        input.states[0].prior_probability_bps = 4_999;
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::PriorProbabilitiesDoNotSumTo10000)
        );
    }

    #[test]
    fn likelihood_mass_must_sum_to_one_for_each_state() {
        let mut input = perfect_signal_scenario(0);
        input.signal_outcomes[0].likelihoods[0].likelihood_bps = 9_999;
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::OutcomeLikelihoodMassDoesNotSumTo10000)
        );
    }

    #[test]
    fn payoffs_require_a_complete_action_state_table() {
        let mut input = perfect_signal_scenario(0);
        input.payoffs.pop();
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::IncompletePayoffTable)
        );
    }

    #[test]
    fn informative_signal_has_zero_value_when_it_cannot_change_the_best_action() {
        let mut input = perfect_signal_scenario(0);
        for payoff in &mut input.payoffs {
            if payoff.action_id == "build" {
                payoff.net_value_minor = match payoff.state_id.as_str() {
                    "high-demand" => 100,
                    "low-demand" => 10,
                    _ => unreachable!(),
                };
            }
        }

        let report = input.evaluate().unwrap();
        assert_eq!(report.baseline_action_id, "build");
        assert_eq!(report.gross_evsi_minor, 0);
        assert_eq!(report.evpi_minor, 0);
    }

    #[test]
    fn zero_probability_signal_does_not_change_expected_value() {
        let mut input = perfect_signal_scenario(20);
        input.signal_outcomes.push(SignalOutcome {
            id: "impossible-signal".into(),
            likelihoods: vec![
                SignalLikelihood {
                    state_id: "high-demand".into(),
                    likelihood_bps: 0,
                },
                SignalLikelihood {
                    state_id: "low-demand".into(),
                    likelihood_bps: 0,
                },
            ],
        });

        let report = input.evaluate().unwrap();
        assert_eq!(report.gross_evsi_minor, 50);
        assert_eq!(report.decisions_by_outcome.len(), 3);
        assert_eq!(
            report
                .decisions_by_outcome
                .iter()
                .find(|decision| decision.outcome_id == "impossible-signal")
                .unwrap()
                .outcome_probability_numerator_bp2,
            0
        );
    }

    #[test]
    fn permuting_states_signal_outcomes_actions_and_payoffs_preserves_results() {
        let input = perfect_signal_scenario(20);
        let baseline = input.evaluate().unwrap();

        let mut permuted = input;
        permuted.states.reverse();
        permuted.signal_outcomes.reverse();
        for outcome in &mut permuted.signal_outcomes {
            outcome.likelihoods.reverse();
        }
        permuted.action_ids.reverse();
        permuted.payoffs.reverse();

        assert_eq!(baseline, permuted.evaluate().unwrap());
    }

    #[test]
    fn evaluation_is_deterministic() {
        let input = perfect_signal_scenario(20);
        assert_eq!(input.evaluate(), input.evaluate());
    }
}
