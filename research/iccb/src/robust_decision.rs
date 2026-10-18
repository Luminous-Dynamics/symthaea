//! Robust action ranking when state probabilities are intervals rather than trusted points.
//!
//! This module treats scenario payoffs as fixed and bounds only the probability
//! mass assigned to states. It does not model payoff/model discrepancy,
//! sequential information acquisition, or governance authority.

const BPS: i128 = 10_000;
const TOTAL_BPS: u64 = 10_000;
const MAX_STATES: usize = 16;
const MAX_ACTIONS: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateProbabilityBound {
    pub state_id: String,
    pub lower_bps: u32,
    pub upper_bps: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobustPayoff {
    pub action_id: String,
    pub state_id: String,
    /// Fixed modeled payoff in currency minor units; this module does not bound it.
    pub payoff_minor: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobustDecisionScenario {
    pub id: String,
    pub state_probability_bounds: Vec<StateProbabilityBound>,
    pub action_ids: Vec<String>,
    pub payoffs: Vec<RobustPayoff>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustDecisionError {
    EmptyIdentifier,
    DuplicateIdentifier,
    EmptyStatesOrActions,
    ScenarioTooLarge,
    InvalidProbabilityInterval,
    InfeasibleProbabilityBounds,
    IncompletePayoffTable,
    DuplicatePayoffEntry,
    UnknownStateOrActionReference,
    ProbabilityAllocationInfeasible,
    ExpectedValueOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbabilityAllocation {
    pub state_id: String,
    pub probability_bps: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionPayoffEnvelope {
    pub action_id: String,
    /// Conservative floor of the exact worst-case expectation in minor units.
    pub lower_expected_value_minor: i64,
    /// Conservative ceiling of the exact best-case expectation in minor units.
    pub upper_expected_value_minor: i64,
    /// Exact expectation numerator; divide by 10,000 for currency minor units.
    pub lower_expected_value_numerator: i128,
    /// Exact expectation numerator; divide by 10,000 for currency minor units.
    pub upper_expected_value_numerator: i128,
    /// Conservative ceiling on worst-case expected regret in currency minor units.
    pub worst_case_expected_regret_upper_minor: i64,
    /// Exact worst-case expected-regret numerator; divide by 10,000 for minor units.
    pub worst_case_expected_regret_numerator: i128,
    pub worst_case_probability_allocation: Vec<ProbabilityAllocation>,
    pub best_case_probability_allocation: Vec<ProbabilityAllocation>,
    pub worst_case_regret_probability_allocation: Vec<ProbabilityAllocation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RobustDecisionStatus {
    /// The action's exact worst-case expected value strictly exceeds every
    /// alternative's exact best-case expected value under the declared bounds.
    UniqueRobustWinner { action_id: String },
    /// Probability ambiguity permits overlapping or reversed action rankings.
    NoUniqueRobustWinner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobustDecisionReport {
    pub scenario_id: String,
    pub status: RobustDecisionStatus,
    pub action_envelopes: Vec<ActionPayoffEnvelope>,
    /// All actions tied for minimum worst-case expected regret, in stable ID order.
    pub minimax_regret_action_ids: Vec<String>,
}

impl RobustDecisionScenario {
    pub fn evaluate(&self) -> Result<RobustDecisionReport, RobustDecisionError> {
        self.validate()?;

        let mut envelopes = Vec::with_capacity(self.action_ids.len());
        for action_id in &self.action_ids {
            envelopes.push(self.envelope_for_action(action_id)?);
        }
        envelopes.sort_by(|left, right| left.action_id.cmp(&right.action_id));

        let robust_winners = envelopes
            .iter()
            .filter(|candidate| {
                envelopes.iter().all(|alternative| {
                    candidate.action_id == alternative.action_id
                        || candidate.lower_expected_value_numerator
                            > alternative.upper_expected_value_numerator
                })
            })
            .map(|envelope| envelope.action_id.clone())
            .collect::<Vec<_>>();

        let status = match robust_winners.as_slice() {
            [action_id] => RobustDecisionStatus::UniqueRobustWinner {
                action_id: action_id.clone(),
            },
            _ => RobustDecisionStatus::NoUniqueRobustWinner,
        };

        let lowest_regret = envelopes
            .iter()
            .map(|envelope| envelope.worst_case_expected_regret_numerator)
            .min()
            .expect("validated action set must be non-empty");
        let minimax_regret_action_ids = envelopes
            .iter()
            .filter(|envelope| envelope.worst_case_expected_regret_numerator == lowest_regret)
            .map(|envelope| envelope.action_id.clone())
            .collect();

        Ok(RobustDecisionReport {
            scenario_id: self.id.clone(),
            status,
            action_envelopes: envelopes,
            minimax_regret_action_ids,
        })
    }

    fn validate(&self) -> Result<(), RobustDecisionError> {
        if self.id.trim().is_empty()
            || self
                .state_probability_bounds
                .iter()
                .any(|state| state.state_id.trim().is_empty())
            || self.action_ids.iter().any(|action| action.trim().is_empty())
            || self.payoffs.iter().any(|payoff| {
                payoff.action_id.trim().is_empty() || payoff.state_id.trim().is_empty()
            })
        {
            return Err(RobustDecisionError::EmptyIdentifier);
        }

        if self.state_probability_bounds.is_empty() || self.action_ids.is_empty() {
            return Err(RobustDecisionError::EmptyStatesOrActions);
        }
        if self.state_probability_bounds.len() > MAX_STATES
            || self.action_ids.len() > MAX_ACTIONS
            || self.payoffs.len() > MAX_STATES * MAX_ACTIONS
        {
            return Err(RobustDecisionError::ScenarioTooLarge);
        }

        ensure_unique(
            self.state_probability_bounds
                .iter()
                .map(|state| state.state_id.as_str()),
        )?;
        ensure_unique(self.action_ids.iter().map(String::as_str))?;

        let mut lower_sum = 0_u64;
        let mut upper_sum = 0_u64;
        let state_ids = self
            .state_probability_bounds
            .iter()
            .map(|state| state.state_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let action_ids = self
            .action_ids
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();

        for state in &self.state_probability_bounds {
            if state.lower_bps > state.upper_bps || state.upper_bps > TOTAL_BPS as u32 {
                return Err(RobustDecisionError::InvalidProbabilityInterval);
            }
            lower_sum += u64::from(state.lower_bps);
            upper_sum += u64::from(state.upper_bps);
        }
        if lower_sum > TOTAL_BPS || upper_sum < TOTAL_BPS {
            return Err(RobustDecisionError::InfeasibleProbabilityBounds);
        }

        let mut payoff_keys = std::collections::BTreeSet::new();
        for payoff in &self.payoffs {
            if !state_ids.contains(payoff.state_id.as_str())
                || !action_ids.contains(payoff.action_id.as_str())
            {
                return Err(RobustDecisionError::UnknownStateOrActionReference);
            }
            if !payoff_keys.insert((payoff.action_id.as_str(), payoff.state_id.as_str())) {
                return Err(RobustDecisionError::DuplicatePayoffEntry);
            }
        }
        if payoff_keys.len() != self.state_probability_bounds.len() * self.action_ids.len() {
            return Err(RobustDecisionError::IncompletePayoffTable);
        }

        Ok(())
    }

    fn envelope_for_action(
        &self,
        action_id: &str,
    ) -> Result<ActionPayoffEnvelope, RobustDecisionError> {
        let (lower_expected_value_numerator, worst_case_probability_allocation) =
            self.extreme_allocation(action_id, true)?;
        let (upper_expected_value_numerator, best_case_probability_allocation) =
            self.extreme_allocation(action_id, false)?;

        let lower_expected_value_minor = floor_ratio_to_i64(lower_expected_value_numerator, BPS)?;
        let upper_expected_value_minor = ceil_ratio_to_i64(upper_expected_value_numerator, BPS)?;
        let (worst_case_expected_regret_numerator, worst_case_regret_probability_allocation) =
            self.worst_case_regret_allocation(action_id)?;
        let worst_case_expected_regret_upper_minor =
            ceil_ratio_to_i64(worst_case_expected_regret_numerator, BPS)?;

        Ok(ActionPayoffEnvelope {
            action_id: action_id.to_owned(),
            lower_expected_value_minor,
            upper_expected_value_minor,
            lower_expected_value_numerator,
            upper_expected_value_numerator,
            worst_case_expected_regret_upper_minor,
            worst_case_expected_regret_numerator,
            worst_case_probability_allocation,
            best_case_probability_allocation,
            worst_case_regret_probability_allocation,
        })
    }

    fn extreme_allocation(
        &self,
        action_id: &str,
        minimize: bool,
    ) -> Result<(i128, Vec<ProbabilityAllocation>), RobustDecisionError> {
        let mut allocation = self
            .state_probability_bounds
            .iter()
            .map(|state| state.lower_bps)
            .collect::<Vec<_>>();
        let lower_sum = allocation.iter().map(|value| u64::from(*value)).sum::<u64>();
        let mut remaining = TOTAL_BPS
            .checked_sub(lower_sum)
            .ok_or(RobustDecisionError::InfeasibleProbabilityBounds)?;

        let mut order = (0..self.state_probability_bounds.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            let left_state = &self.state_probability_bounds[*left];
            let right_state = &self.state_probability_bounds[*right];
            let left_payoff = self.payoff(action_id, &left_state.state_id);
            let right_payoff = self.payoff(action_id, &right_state.state_id);
            let payoff_order = if minimize {
                left_payoff.cmp(&right_payoff)
            } else {
                right_payoff.cmp(&left_payoff)
            };
            payoff_order.then_with(|| left_state.state_id.cmp(&right_state.state_id))
        });

        for index in order {
            if remaining == 0 {
                break;
            }
            let state = &self.state_probability_bounds[index];
            let capacity = u64::from(state.upper_bps - allocation[index]);
            let assigned = remaining.min(capacity);
            allocation[index] = allocation[index]
                .checked_add(u32::try_from(assigned).map_err(|_| {
                    RobustDecisionError::ProbabilityAllocationInfeasible
                })?)
                .ok_or(RobustDecisionError::ProbabilityAllocationInfeasible)?;
            remaining -= assigned;
        }

        if remaining != 0
            || allocation.iter().map(|value| u64::from(*value)).sum::<u64>() != TOTAL_BPS
        {
            return Err(RobustDecisionError::ProbabilityAllocationInfeasible);
        }

        let mut numerator = 0_i128;
        let mut recorded = Vec::with_capacity(allocation.len());
        for (state, probability) in self.state_probability_bounds.iter().zip(allocation) {
            numerator += i128::from(probability)
                * i128::from(self.payoff(action_id, &state.state_id));
            recorded.push(ProbabilityAllocation {
                state_id: state.state_id.clone(),
                probability_bps: probability,
            });
        }
        recorded.sort_by(|left, right| left.state_id.cmp(&right.state_id));
        Ok((numerator, recorded))
    }

    fn worst_case_regret_allocation(
        &self,
        action_id: &str,
    ) -> Result<(i128, Vec<ProbabilityAllocation>), RobustDecisionError> {
        let mut allocation = self
            .state_probability_bounds
            .iter()
            .map(|state| state.lower_bps)
            .collect::<Vec<_>>();
        let lower_sum = allocation.iter().map(|value| u64::from(*value)).sum::<u64>();
        let mut remaining = TOTAL_BPS
            .checked_sub(lower_sum)
            .ok_or(RobustDecisionError::InfeasibleProbabilityBounds)?;

        let regrets = self
            .state_probability_bounds
            .iter()
            .map(|state| {
                let best_payoff = self
                    .action_ids
                    .iter()
                    .map(|candidate| i128::from(self.payoff(candidate, &state.state_id)))
                    .max()
                    .expect("validated action set must be non-empty");
                let current_payoff = i128::from(self.payoff(action_id, &state.state_id));
                best_payoff - current_payoff
            })
            .collect::<Vec<_>>();

        let mut order = (0..self.state_probability_bounds.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            regrets[*right]
                .cmp(&regrets[*left])
                .then_with(|| {
                    self.state_probability_bounds[*left]
                        .state_id
                        .cmp(&self.state_probability_bounds[*right].state_id)
                })
        });

        for index in order {
            if remaining == 0 {
                break;
            }
            let state = &self.state_probability_bounds[index];
            let capacity = u64::from(state.upper_bps - allocation[index]);
            let assigned = remaining.min(capacity);
            allocation[index] = allocation[index]
                .checked_add(u32::try_from(assigned).map_err(|_| {
                    RobustDecisionError::ProbabilityAllocationInfeasible
                })?)
                .ok_or(RobustDecisionError::ProbabilityAllocationInfeasible)?;
            remaining -= assigned;
        }

        if remaining != 0
            || allocation.iter().map(|value| u64::from(*value)).sum::<u64>() != TOTAL_BPS
        {
            return Err(RobustDecisionError::ProbabilityAllocationInfeasible);
        }

        let mut numerator = 0_i128;
        let mut recorded = Vec::with_capacity(allocation.len());
        for ((state, probability), regret) in self
            .state_probability_bounds
            .iter()
            .zip(allocation)
            .zip(regrets)
        {
            numerator += i128::from(probability) * regret;
            recorded.push(ProbabilityAllocation {
                state_id: state.state_id.clone(),
                probability_bps: probability,
            });
        }
        recorded.sort_by(|left, right| left.state_id.cmp(&right.state_id));
        Ok((numerator, recorded))
    }

    fn payoff(&self, action_id: &str, state_id: &str) -> i64 {
        self.payoffs
            .iter()
            .find(|payoff| payoff.action_id == action_id && payoff.state_id == state_id)
            .map(|payoff| payoff.payoff_minor)
            .expect("payoff table validated before evaluation")
    }
}

fn ensure_unique<'a>(
    values: impl Iterator<Item = &'a str>,
) -> Result<(), RobustDecisionError> {
    let mut seen = std::collections::BTreeSet::new();
    if values.into_iter().all(|value| seen.insert(value)) {
        Ok(())
    } else {
        Err(RobustDecisionError::DuplicateIdentifier)
    }
}

fn floor_ratio_to_i64(numerator: i128, denominator: i128) -> Result<i64, RobustDecisionError> {
    i64::try_from(numerator.div_euclid(denominator))
        .map_err(|_| RobustDecisionError::ExpectedValueOverflow)
}

fn ceil_ratio_to_i64(numerator: i128, denominator: i128) -> Result<i64, RobustDecisionError> {
    let negated = numerator
        .checked_neg()
        .ok_or(RobustDecisionError::ExpectedValueOverflow)?;
    i64::try_from(
        negated
            .div_euclid(denominator)
            .checked_neg()
            .ok_or(RobustDecisionError::ExpectedValueOverflow)?,
    )
    .map_err(|_| RobustDecisionError::ExpectedValueOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> RobustDecisionScenario {
        RobustDecisionScenario {
            id: "ICCB-ROBUST-PROBABILITY-001".into(),
            state_probability_bounds: vec![
                StateProbabilityBound {
                    state_id: "high-demand".into(),
                    lower_bps: 3_000,
                    upper_bps: 7_000,
                },
                StateProbabilityBound {
                    state_id: "low-demand".into(),
                    lower_bps: 3_000,
                    upper_bps: 7_000,
                },
            ],
            action_ids: vec!["build".into(), "avoid".into()],
            payoffs: vec![
                RobustPayoff {
                    action_id: "build".into(),
                    state_id: "high-demand".into(),
                    payoff_minor: 100,
                },
                RobustPayoff {
                    action_id: "build".into(),
                    state_id: "low-demand".into(),
                    payoff_minor: -100,
                },
                RobustPayoff {
                    action_id: "avoid".into(),
                    state_id: "high-demand".into(),
                    payoff_minor: 0,
                },
                RobustPayoff {
                    action_id: "avoid".into(),
                    state_id: "low-demand".into(),
                    payoff_minor: 0,
                },
            ],
        }
    }

    fn envelope<'a>(report: &'a RobustDecisionReport, action: &str) -> &'a ActionPayoffEnvelope {
        report
            .action_envelopes
            .iter()
            .find(|candidate| candidate.action_id == action)
            .expect("action envelope must exist")
    }

    #[test]
    fn opposite_state_payoffs_make_the_ranking_ambiguous() {
        let report = scenario().evaluate().unwrap();
        assert_eq!(report.status, RobustDecisionStatus::NoUniqueRobustWinner);
        assert_eq!(envelope(&report, "build").lower_expected_value_minor, -40);
        assert_eq!(envelope(&report, "build").upper_expected_value_minor, 40);
    }

    #[test]
    fn action_that_dominates_in_every_state_is_a_robust_winner() {
        let mut input = scenario();
        for payoff in &mut input.payoffs {
            payoff.payoff_minor = match (payoff.action_id.as_str(), payoff.state_id.as_str()) {
                ("build", "high-demand") => 100,
                ("build", "low-demand") => 10,
                ("avoid", "high-demand") => 0,
                ("avoid", "low-demand") => 0,
                _ => unreachable!(),
            };
        }

        let report = input.evaluate().unwrap();
        assert_eq!(
            report.status,
            RobustDecisionStatus::UniqueRobustWinner {
                action_id: "build".into()
            }
        );
    }

    #[test]
    fn point_probability_bounds_match_expected_value_ordering() {
        let mut input = scenario();
        input.state_probability_bounds = vec![
            StateProbabilityBound {
                state_id: "high-demand".into(),
                lower_bps: 4_000,
                upper_bps: 4_000,
            },
            StateProbabilityBound {
                state_id: "low-demand".into(),
                lower_bps: 6_000,
                upper_bps: 6_000,
            },
        ];
        let report = input.evaluate().unwrap();
        assert_eq!(
            report.status,
            RobustDecisionStatus::UniqueRobustWinner {
                action_id: "avoid".into()
            }
        );
        assert_eq!(envelope(&report, "build").lower_expected_value_minor, -20);
        assert_eq!(envelope(&report, "build").upper_expected_value_minor, -20);
    }

    #[test]
    fn wider_probability_bounds_can_turn_a_winner_into_ambiguity() {
        let mut tight = scenario();
        tight.state_probability_bounds = vec![
            StateProbabilityBound {
                state_id: "high-demand".into(),
                lower_bps: 6_000,
                upper_bps: 6_000,
            },
            StateProbabilityBound {
                state_id: "low-demand".into(),
                lower_bps: 4_000,
                upper_bps: 4_000,
            },
        ];
        assert_eq!(
            tight.evaluate().unwrap().status,
            RobustDecisionStatus::UniqueRobustWinner {
                action_id: "build".into()
            }
        );

        let mut wide = tight;
        wide.state_probability_bounds = vec![
            StateProbabilityBound {
                state_id: "high-demand".into(),
                lower_bps: 4_000,
                upper_bps: 7_000,
            },
            StateProbabilityBound {
                state_id: "low-demand".into(),
                lower_bps: 3_000,
                upper_bps: 6_000,
            },
        ];
        assert_eq!(
            wide.evaluate().unwrap().status,
            RobustDecisionStatus::NoUniqueRobustWinner
        );
    }

    #[test]
    fn impossible_probability_mass_fails_closed() {
        let mut input = scenario();
        input.state_probability_bounds[0].lower_bps = 6_000;
        input.state_probability_bounds[1].lower_bps = 6_000;
        assert_eq!(
            input.evaluate(),
            Err(RobustDecisionError::InfeasibleProbabilityBounds)
        );
    }

    #[test]
    fn upper_probability_mass_below_one_fails_closed() {
        let mut input = scenario();
        input.state_probability_bounds[0].upper_bps = 4_000;
        input.state_probability_bounds[1].upper_bps = 4_000;
        assert_eq!(
            input.evaluate(),
            Err(RobustDecisionError::InfeasibleProbabilityBounds)
        );
    }

    #[test]
    fn invalid_probability_interval_fails_closed() {
        let mut input = scenario();
        input.state_probability_bounds[0].lower_bps = 8_000;
        input.state_probability_bounds[0].upper_bps = 7_000;
        assert_eq!(
            input.evaluate(),
            Err(RobustDecisionError::InvalidProbabilityInterval)
        );
    }

    #[test]
    fn duplicate_payoff_entry_fails_closed() {
        let mut input = scenario();
        input.payoffs.push(input.payoffs[0].clone());
        assert_eq!(
            input.evaluate(),
            Err(RobustDecisionError::DuplicatePayoffEntry)
        );
    }

    #[test]
    fn permutation_of_states_actions_and_payoff_rows_preserves_result() {
        let input = scenario();
        let baseline = input.evaluate().unwrap();
        let mut permuted = input;
        permuted.state_probability_bounds.reverse();
        permuted.action_ids.reverse();
        permuted.payoffs.reverse();
        assert_eq!(baseline, permuted.evaluate().unwrap());
    }

    #[test]
    fn greedy_extrema_match_independent_grid_oracle() {
        let mut input = scenario();
        input.state_probability_bounds = vec![
            StateProbabilityBound {
                state_id: "alpha".into(),
                lower_bps: 2_000,
                upper_bps: 6_000,
            },
            StateProbabilityBound {
                state_id: "beta".into(),
                lower_bps: 2_000,
                upper_bps: 6_000,
            },
            StateProbabilityBound {
                state_id: "gamma".into(),
                lower_bps: 2_000,
                upper_bps: 6_000,
            },
        ];
        input.action_ids = vec!["test-action".into()];
        input.payoffs = vec![
            RobustPayoff {
                action_id: "test-action".into(),
                state_id: "alpha".into(),
                payoff_minor: 17,
            },
            RobustPayoff {
                action_id: "test-action".into(),
                state_id: "beta".into(),
                payoff_minor: -9,
            },
            RobustPayoff {
                action_id: "test-action".into(),
                state_id: "gamma".into(),
                payoff_minor: 23,
            },
        ];

        let mut oracle_min = i128::MAX;
        let mut oracle_max = i128::MIN;
        for alpha in (2_000_u32..=6_000).step_by(1_000) {
            for beta in (2_000_u32..=6_000).step_by(1_000) {
                if u64::from(alpha) + u64::from(beta) > TOTAL_BPS {
                    continue;
                }
                let gamma = TOTAL_BPS - u64::from(alpha) - u64::from(beta);
                if !(2_000..=6_000).contains(&gamma) || gamma % 1_000 != 0 {
                    continue;
                }
                let numerator = i128::from(alpha) * 17
                    + i128::from(beta) * -9
                    + i128::from(gamma) * 23;
                oracle_min = oracle_min.min(numerator);
                oracle_max = oracle_max.max(numerator);
            }
        }

        let report = input.evaluate().unwrap();
        let evaluated = envelope(&report, "test-action");
        assert_eq!(evaluated.lower_expected_value_numerator, oracle_min);
        assert_eq!(evaluated.upper_expected_value_numerator, oracle_max);
    }

    #[test]
    fn extreme_allocations_respect_bounds_and_sum_to_one() {
        let input = scenario();
        let report = input.evaluate().unwrap();
        for action in &report.action_envelopes {
            for allocation in [
                &action.worst_case_probability_allocation,
                &action.best_case_probability_allocation,
            ] {
                assert_eq!(
                    allocation
                        .iter()
                        .map(|entry| u64::from(entry.probability_bps))
                        .sum::<u64>(),
                    TOTAL_BPS
                );
                assert_eq!(
                    allocation.len(),
                    input.state_probability_bounds.len()
                );
                for entry in allocation {
                    let bound = input
                        .state_probability_bounds
                        .iter()
                        .find(|bound| bound.state_id == entry.state_id)
                        .expect("allocation must reference a known state");
                    assert!(entry.probability_bps >= bound.lower_bps);
                    assert!(entry.probability_bps <= bound.upper_bps);
                }
            }
        }
    }

    #[test]
    fn negative_nonintegral_expectations_use_conservative_floor_and_ceil() {
        let mut input = scenario();
        input.state_probability_bounds = vec![
            StateProbabilityBound {
                state_id: "one".into(),
                lower_bps: 3_333,
                upper_bps: 3_333,
            },
            StateProbabilityBound {
                state_id: "two".into(),
                lower_bps: 6_667,
                upper_bps: 6_667,
            },
        ];
        input.action_ids = vec!["negative".into()];
        input.payoffs = vec![
            RobustPayoff {
                action_id: "negative".into(),
                state_id: "one".into(),
                payoff_minor: -1,
            },
            RobustPayoff {
                action_id: "negative".into(),
                state_id: "two".into(),
                payoff_minor: 0,
            },
        ];
        let report = input.evaluate().unwrap();
        assert_eq!(envelope(&report, "negative").lower_expected_value_minor, -1);
        assert_eq!(envelope(&report, "negative").upper_expected_value_minor, 0);
    }

    #[test]
    fn out_of_range_probability_fails_closed() {
        let mut input = scenario();
        input.state_probability_bounds[0].upper_bps = 10_001;
        assert_eq!(
            input.evaluate(),
            Err(RobustDecisionError::InvalidProbabilityInterval)
        );
    }
}
