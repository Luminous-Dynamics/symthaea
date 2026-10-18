//! Decision-theoretic value of information for capital diligence.
//!
//! Estimates the expected value of a specified noisy signal before paying for
//! it. Inputs are explicit synthetic/elicited priors, likelihoods and payoffs;
//! this module does not infer calibrated probabilities from raw evidence.

const BPS: i128 = 10_000;
const BPS_SQUARED: i128 = 100_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InformationState {
    pub id: String,
    pub prior_probability_bps: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalLikelihood {
    pub state_id: String,
    /// Probability of this signal conditional on this state, in basis points.
    pub likelihood_bps: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InformationSignal {
    pub id: String,
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
    pub information_cost_minor: i64,
    pub states: Vec<InformationState>,
    pub signals: Vec<InformationSignal>,
    pub action_ids: Vec<String>,
    pub payoffs: Vec<DecisionPayoff>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueOfInformationError {
    EmptyIdentifier,
    DuplicateIdentifier,
    EmptyStatesSignalsOrActions,
    PriorProbabilitiesDoNotSumTo10000,
    SignalLikelihoodsDoNotSumTo10000,
    IncompleteLikelihoodTable,
    DuplicateLikelihoodEntry,
    IncompletePayoffTable,
    DuplicatePayoffEntry,
    UnknownStateReference,
    UnknownSignalOrActionReference,
    NegativeInformationCost,
    ExpectedValueOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignalDecision {
    pub signal_id: String,
    pub chosen_action_id: String,
    /// Exact probability numerator over 100,000,000 (basis-points squared).
    pub signal_probability_numerator_bp2: u64,
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
    pub net_evsi_minor: i64,
    pub perfect_information_expected_value_minor: i64,
    pub evpi_minor: i64,
    pub worthwhile_at_declared_cost: bool,
    pub decisions_by_signal: Vec<SignalDecision>,
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
        let mut decisions_by_signal = Vec::with_capacity(self.signals.len());

        let mut signals = self.signals.iter().collect::<Vec<_>>();
        signals.sort_by(|left, right| left.id.cmp(&right.id));

        for signal in signals {
            let mut marginal_probability_numerator = 0_i128;
            for state in &self.states {
                let likelihood = self.likelihood_bps(signal, &state.id);
                marginal_probability_numerator +=
                    i128::from(state.prior_probability_bps) * i128::from(likelihood);
            }

            let mut best_action_id: Option<&str> = None;
            let mut best_conditional_numerator = i128::MIN;
            for action_id in &self.action_ids {
                let mut conditional_numerator = 0_i128;
                for state in &self.states {
                    let prior = i128::from(state.prior_probability_bps);
                    let likelihood = i128::from(self.likelihood_bps(signal, &state.id));
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

            // A zero-probability signal never affects the expected value; retain a
            // deterministic action choice for complete, reproducible reports.
            let chosen_action_id = best_action_id
                .expect("validated action set must be non-empty")
                .to_owned();
            with_signal_numerator += best_conditional_numerator;

            decisions_by_signal.push(SignalDecision {
                signal_id: signal.id.clone(),
                chosen_action_id,
                signal_probability_numerator_bp2: u64::try_from(
                    marginal_probability_numerator,
                )
                .map_err(|_| ValueOfInformationError::ExpectedValueOverflow)?,
                conditional_expected_value_numerator: best_conditional_numerator,
            });
        }

        let expected_value_with_signal_minor =
            to_i64(with_signal_numerator.div_euclid(BPS_SQUARED))?;
        let gross_evsi_numerator =
            with_signal_numerator - baseline_numerator * BPS;
        let gross_evsi_minor = to_i64(gross_evsi_numerator.div_euclid(BPS_SQUARED))?;
        let net_evsi_minor = gross_evsi_minor
            .checked_sub(self.information_cost_minor)
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
            net_evsi_minor,
            perfect_information_expected_value_minor,
            evpi_minor,
            worthwhile_at_declared_cost: net_evsi_minor > 0,
            decisions_by_signal,
        })
    }

    fn validate(&self) -> Result<(), ValueOfInformationError> {
        if self.id.trim().is_empty()
            || self.states.iter().any(|state| state.id.trim().is_empty())
            || self.signals.iter().any(|signal| signal.id.trim().is_empty())
            || self.action_ids.iter().any(|action| action.trim().is_empty())
            || self.payoffs.iter().any(|payoff| {
                payoff.action_id.trim().is_empty() || payoff.state_id.trim().is_empty()
            })
        {
            return Err(ValueOfInformationError::EmptyIdentifier);
        }
        if self.information_cost_minor < 0 {
            return Err(ValueOfInformationError::NegativeInformationCost);
        }
        if self.states.is_empty() || self.signals.is_empty() || self.action_ids.is_empty() {
            return Err(ValueOfInformationError::EmptyStatesSignalsOrActions);
        }
        ensure_unique(self.states.iter().map(|state| state.id.as_str()))?;
        ensure_unique(self.signals.iter().map(|signal| signal.id.as_str()))?;
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

        for signal in &self.signals {
            for likelihood in &signal.likelihoods {
                if !state_ids.contains(likelihood.state_id.as_str()) {
                    return Err(ValueOfInformationError::UnknownStateReference);
                }
                if !likelihood_keys.insert((signal.id.as_str(), likelihood.state_id.as_str())) {
                    return Err(ValueOfInformationError::DuplicateLikelihoodEntry);
                }
                *likelihood_sums
                    .get_mut(likelihood.state_id.as_str())
                    .expect("state identity checked above") += u64::from(likelihood.likelihood_bps);
            }
            if signal.likelihoods.len() != self.states.len() {
                return Err(ValueOfInformationError::IncompleteLikelihoodTable);
            }
        }

        if likelihood_sums.values().any(|sum| *sum != 10_000) {
            return Err(ValueOfInformationError::SignalLikelihoodsDoNotSumTo10000);
        }

        let mut payoff_keys = std::collections::BTreeSet::new();
        for payoff in &self.payoffs {
            if !state_ids.contains(payoff.state_id.as_str())
                || !action_ids.contains(payoff.action_id.as_str())
            {
                return Err(ValueOfInformationError::UnknownSignalOrActionReference);
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

    fn likelihood_bps(&self, signal: &InformationSignal, state_id: &str) -> u32 {
        signal
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
            signals: vec![
                InformationSignal {
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
                InformationSignal {
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
        assert_eq!(report.decisions_by_signal.len(), 2);
    }

    #[test]
    fn noisy_signal_has_lower_value_than_perfect_information() {
        let mut input = perfect_signal_scenario(0);
        input.signals = vec![
            InformationSignal {
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
            InformationSignal {
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
    fn information_cost_can_exceed_gross_value() {
        let report = perfect_signal_scenario(101).evaluate().unwrap();
        assert_eq!(report.gross_evsi_minor, 50);
        assert_eq!(report.net_evsi_minor, -51);
        assert!(!report.worthwhile_at_declared_cost);
    }

    #[test]
    fn uninformative_signal_has_zero_evsi() {
        let mut input = perfect_signal_scenario(0);
        input.signals = vec![InformationSignal {
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
        input.signals[0].likelihoods[0].likelihood_bps = 9_999;
        assert_eq!(
            input.evaluate(),
            Err(ValueOfInformationError::SignalLikelihoodsDoNotSumTo10000)
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
        input.signals.push(InformationSignal {
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
        assert_eq!(report.decisions_by_signal.len(), 3);
        assert_eq!(
            report
                .decisions_by_signal
                .iter()
                .find(|decision| decision.signal_id == "impossible-signal")
                .unwrap()
                .signal_probability_numerator_bp2,
            0
        );
    }

    #[test]
    fn permuting_states_signals_actions_and_payoffs_preserves_results() {
        let input = perfect_signal_scenario(20);
        let baseline = input.evaluate().unwrap();

        let mut permuted = input;
        permuted.states.reverse();
        permuted.signals.reverse();
        for signal in &mut permuted.signals {
            signal.likelihoods.reverse();
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
