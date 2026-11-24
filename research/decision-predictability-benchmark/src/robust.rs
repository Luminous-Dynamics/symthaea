//! Prior-free robust value of an information-gathering action.
//!
//! This module is deliberately separate from Bayesian EVSI. It accepts sets of
//! hidden states that could produce each outcome, but no prior probabilities or
//! likelihood magnitudes. It computes the worst-case minimax decision loss before
//! and after observing the probe. It must not be reported as expected value of
//! information, mutual information, or EIG.
//!
//! This provides a principled decision-value companion for prior-free
//! discrimination scores (for example, covariance-based maximin separation).
//! A covariance distance alone is not enough: an adapter must still provide
//! outcome-support sets and a frozen decision-loss table.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

const MAX_STATES: usize = 256;
const MAX_DECISIONS: usize = 256;
const MAX_OUTCOMES: usize = 10_000;
const TOLERANCE: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq)]
pub struct RobustDecisionProblem {
    pub state_ids: Vec<String>,
    pub decision_ids: Vec<String>,
    /// decision_loss[decision_id][state_id] is a bounded task loss in [0, 1].
    pub decision_loss: BTreeMap<String, BTreeMap<String, f64>>,
    /// A decision is admissible only if its worst-case loss on the current
    /// possible-state set is no greater than this frozen threshold.
    pub maximum_worst_case_loss: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobustProbeOutcome {
    pub outcome_id: String,
    /// States under which this outcome is possible. No probability is implied.
    pub possible_state_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustProbe {
    pub probe_id: String,
    /// Application-declared cost units; never compared to task loss.
    pub acquisition_cost_units: f64,
    pub result_delay_steps: u64,
    pub outcomes: Vec<RobustProbeOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustProbeContext {
    pub permitted_probe_ids: Vec<String>,
    pub remaining_cost_budget_units: f64,
    pub current_step: u64,
    pub decision_deadline_step: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustProbeAdmission {
    Eligible,
    NotPermitted,
    OverBudget,
    MissesDecisionDeadline,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustProbeValue {
    pub probe_id: String,
    pub admission: RobustProbeAdmission,
    /// Minimax decision loss before acquiring the observation.
    pub current_minimax_loss: f64,
    /// Worst-case, across possible outcomes, of the posterior minimax loss.
    pub worst_case_post_observation_minimax_loss: f64,
    /// Guaranteed reduction in minimax task loss. Not a probabilistic expectation.
    pub guaranteed_worst_case_loss_reduction: f64,
    /// Kept separate because application cost units need not equal task-loss units.
    pub acquisition_cost_units: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RobustEvaluationError {
    EmptyStateSet,
    EmptyDecisionSet,
    TooManyStates(usize),
    TooManyDecisions(usize),
    TooManyOutcomes(usize),
    EmptyIdentifier(&'static str),
    DuplicateIdentifier {
        field: &'static str,
        id: String,
    },
    InvalidThreshold(f64),
    InvalidLoss {
        decision_id: String,
        state_id: String,
        value_bits: u64,
    },
    MissingDecisionLoss {
        decision_id: String,
        state_id: String,
    },
    InvalidCost(f64),
    InvalidBudget(f64),
    InvalidDeadline,
    EmptyOutcomeSupport(String),
    DuplicateStateInSupport {
        outcome_id: String,
        state_id: String,
    },
    UnknownStateInSupport {
        outcome_id: String,
        state_id: String,
    },
    UncoveredState(String),
    NoRiskAdmissibleDecision,
    InformationIncreasedWorstCaseRisk {
        before_bits: u64,
        after_bits: u64,
    },
}

impl Display for RobustEvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStateSet => write!(
                formatter,
                "robust decision problem requires at least one possible state"
            ),
            Self::EmptyDecisionSet => write!(
                formatter,
                "robust decision problem requires at least one decision"
            ),
            Self::TooManyStates(count) => {
                write!(formatter, "state count {count} exceeds {MAX_STATES}")
            }
            Self::TooManyDecisions(count) => {
                write!(formatter, "decision count {count} exceeds {MAX_DECISIONS}")
            }
            Self::TooManyOutcomes(count) => {
                write!(formatter, "outcome count {count} exceeds {MAX_OUTCOMES}")
            }
            Self::EmptyIdentifier(field) => {
                write!(formatter, "{field} identifier must be non-empty")
            }
            Self::DuplicateIdentifier { field, id } => {
                write!(formatter, "duplicate {field} identifier: {id}")
            }
            Self::InvalidThreshold(value) => {
                write!(formatter, "maximum worst-case loss is invalid: {value}")
            }
            Self::InvalidLoss {
                decision_id,
                state_id,
                value_bits,
            } => write!(
                formatter,
                "decision {decision_id} has invalid loss {} for state {state_id}",
                f64::from_bits(*value_bits)
            ),
            Self::MissingDecisionLoss {
                decision_id,
                state_id,
            } => write!(
                formatter,
                "decision {decision_id} has no loss for state {state_id}"
            ),
            Self::InvalidCost(value) => {
                write!(formatter, "probe acquisition cost is invalid: {value}")
            }
            Self::InvalidBudget(value) => {
                write!(formatter, "remaining cost budget is invalid: {value}")
            }
            Self::InvalidDeadline => write!(formatter, "decision deadline precedes current step"),
            Self::EmptyOutcomeSupport(outcome) => {
                write!(formatter, "outcome {outcome} has no possible hidden states")
            }
            Self::DuplicateStateInSupport {
                outcome_id,
                state_id,
            } => write!(formatter, "outcome {outcome_id} repeats state {state_id}"),
            Self::UnknownStateInSupport {
                outcome_id,
                state_id,
            } => write!(
                formatter,
                "outcome {outcome_id} names unknown state {state_id}"
            ),
            Self::UncoveredState(state) => write!(
                formatter,
                "probe outcome supports do not cover possible state {state}"
            ),
            Self::NoRiskAdmissibleDecision => write!(
                formatter,
                "no decision satisfies the frozen worst-case risk constraint"
            ),
            Self::InformationIncreasedWorstCaseRisk {
                before_bits,
                after_bits,
            } => write!(
                formatter,
                "observing a probe increased worst-case minimax risk from {} to {}",
                f64::from_bits(*before_bits),
                f64::from_bits(*after_bits)
            ),
        }
    }
}

impl Error for RobustEvaluationError {}

fn validate_problem(problem: &RobustDecisionProblem) -> Result<(), RobustEvaluationError> {
    if problem.state_ids.is_empty() {
        return Err(RobustEvaluationError::EmptyStateSet);
    }
    if problem.state_ids.len() > MAX_STATES {
        return Err(RobustEvaluationError::TooManyStates(
            problem.state_ids.len(),
        ));
    }
    if problem.decision_ids.is_empty() {
        return Err(RobustEvaluationError::EmptyDecisionSet);
    }
    if problem.decision_ids.len() > MAX_DECISIONS {
        return Err(RobustEvaluationError::TooManyDecisions(
            problem.decision_ids.len(),
        ));
    }
    if !problem.maximum_worst_case_loss.is_finite()
        || !(0.0..=1.0).contains(&problem.maximum_worst_case_loss)
    {
        return Err(RobustEvaluationError::InvalidThreshold(
            problem.maximum_worst_case_loss,
        ));
    }

    let mut states = BTreeSet::new();
    for state_id in &problem.state_ids {
        if state_id.trim().is_empty() {
            return Err(RobustEvaluationError::EmptyIdentifier("state"));
        }
        if !states.insert(state_id.as_str()) {
            return Err(RobustEvaluationError::DuplicateIdentifier {
                field: "state",
                id: state_id.clone(),
            });
        }
    }

    let mut decisions = BTreeSet::new();
    for decision_id in &problem.decision_ids {
        if decision_id.trim().is_empty() {
            return Err(RobustEvaluationError::EmptyIdentifier("decision"));
        }
        if !decisions.insert(decision_id.as_str()) {
            return Err(RobustEvaluationError::DuplicateIdentifier {
                field: "decision",
                id: decision_id.clone(),
            });
        }
        let losses = problem.decision_loss.get(decision_id).ok_or_else(|| {
            RobustEvaluationError::MissingDecisionLoss {
                decision_id: decision_id.clone(),
                state_id: problem.state_ids[0].clone(),
            }
        })?;
        for state_id in &problem.state_ids {
            let loss = losses.get(state_id).copied().ok_or_else(|| {
                RobustEvaluationError::MissingDecisionLoss {
                    decision_id: decision_id.clone(),
                    state_id: state_id.clone(),
                }
            })?;
            if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
                return Err(RobustEvaluationError::InvalidLoss {
                    decision_id: decision_id.clone(),
                    state_id: state_id.clone(),
                    value_bits: loss.to_bits(),
                });
            }
        }
    }
    Ok(())
}

fn minimax_loss(
    problem: &RobustDecisionProblem,
    possible_states: &[String],
) -> Result<f64, RobustEvaluationError> {
    if possible_states.is_empty() {
        return Err(RobustEvaluationError::EmptyStateSet);
    }
    let mut best = f64::INFINITY;
    for decision_id in &problem.decision_ids {
        let losses = &problem.decision_loss[decision_id];
        let mut worst = 0.0_f64;
        for state_id in possible_states {
            let loss = losses.get(state_id).copied().ok_or_else(|| {
                RobustEvaluationError::MissingDecisionLoss {
                    decision_id: decision_id.clone(),
                    state_id: state_id.clone(),
                }
            })?;
            worst = worst.max(loss);
        }
        if worst <= problem.maximum_worst_case_loss + TOLERANCE {
            best = best.min(worst);
        }
    }
    if !best.is_finite() {
        return Err(RobustEvaluationError::NoRiskAdmissibleDecision);
    }
    Ok(best)
}

fn validate_probe(
    problem: &RobustDecisionProblem,
    probe: &RobustProbe,
) -> Result<(), RobustEvaluationError> {
    if probe.probe_id.trim().is_empty() {
        return Err(RobustEvaluationError::EmptyIdentifier("probe"));
    }
    if !probe.acquisition_cost_units.is_finite() || probe.acquisition_cost_units < 0.0 {
        return Err(RobustEvaluationError::InvalidCost(
            probe.acquisition_cost_units,
        ));
    }
    if probe.outcomes.is_empty() || probe.outcomes.len() > MAX_OUTCOMES {
        return Err(RobustEvaluationError::TooManyOutcomes(probe.outcomes.len()));
    }

    let known_states: BTreeSet<&str> = problem.state_ids.iter().map(String::as_str).collect();
    let mut outcomes = BTreeSet::new();
    let mut covered_states = BTreeSet::new();
    for outcome in &probe.outcomes {
        if outcome.outcome_id.trim().is_empty() {
            return Err(RobustEvaluationError::EmptyIdentifier("outcome"));
        }
        if !outcomes.insert(outcome.outcome_id.as_str()) {
            return Err(RobustEvaluationError::DuplicateIdentifier {
                field: "outcome",
                id: outcome.outcome_id.clone(),
            });
        }
        if outcome.possible_state_ids.is_empty() {
            return Err(RobustEvaluationError::EmptyOutcomeSupport(
                outcome.outcome_id.clone(),
            ));
        }
        let mut local_states = BTreeSet::new();
        for state_id in &outcome.possible_state_ids {
            if !known_states.contains(state_id.as_str()) {
                return Err(RobustEvaluationError::UnknownStateInSupport {
                    outcome_id: outcome.outcome_id.clone(),
                    state_id: state_id.clone(),
                });
            }
            if !local_states.insert(state_id.as_str()) {
                return Err(RobustEvaluationError::DuplicateStateInSupport {
                    outcome_id: outcome.outcome_id.clone(),
                    state_id: state_id.clone(),
                });
            }
            covered_states.insert(state_id.as_str());
        }
    }
    for state_id in &problem.state_ids {
        if !covered_states.contains(state_id.as_str()) {
            return Err(RobustEvaluationError::UncoveredState(state_id.clone()));
        }
    }
    Ok(())
}

/// Evaluate a probe using a set-valued observation model, without priors or
/// probability magnitudes.
///
/// For each possible outcome, the evaluator computes the minimax task loss over
/// the states consistent with that outcome. It then takes the worst of those
/// post-observation minimax losses. The difference from the current minimax loss
/// is a guaranteed worst-case reduction. Acquisition cost stays in its own
/// application-defined units and is not subtracted from task loss.
pub fn evaluate_robust_probe(
    problem: &RobustDecisionProblem,
    probe: &RobustProbe,
    context: &RobustProbeContext,
) -> Result<RobustProbeValue, RobustEvaluationError> {
    validate_problem(problem)?;
    validate_probe(problem, probe)?;
    if !context.remaining_cost_budget_units.is_finite() || context.remaining_cost_budget_units < 0.0
    {
        return Err(RobustEvaluationError::InvalidBudget(
            context.remaining_cost_budget_units,
        ));
    }
    if context.decision_deadline_step < context.current_step {
        return Err(RobustEvaluationError::InvalidDeadline);
    }
    let mut permitted = BTreeSet::new();
    for probe_id in &context.permitted_probe_ids {
        if probe_id.trim().is_empty() {
            return Err(RobustEvaluationError::EmptyIdentifier("permitted probe"));
        }
        if !permitted.insert(probe_id.as_str()) {
            return Err(RobustEvaluationError::DuplicateIdentifier {
                field: "permitted probe",
                id: probe_id.clone(),
            });
        }
    }

    let current_minimax_loss = minimax_loss(problem, &problem.state_ids)?;
    let mut worst_case_post_observation_minimax_loss = 0.0_f64;
    for outcome in &probe.outcomes {
        let outcome_loss = minimax_loss(problem, &outcome.possible_state_ids)?;
        worst_case_post_observation_minimax_loss =
            worst_case_post_observation_minimax_loss.max(outcome_loss);
    }
    if worst_case_post_observation_minimax_loss > current_minimax_loss + TOLERANCE {
        return Err(RobustEvaluationError::InformationIncreasedWorstCaseRisk {
            before_bits: current_minimax_loss.to_bits(),
            after_bits: worst_case_post_observation_minimax_loss.to_bits(),
        });
    }

    let admission = if !permitted.contains(probe.probe_id.as_str()) {
        RobustProbeAdmission::NotPermitted
    } else if probe.acquisition_cost_units > context.remaining_cost_budget_units + TOLERANCE {
        RobustProbeAdmission::OverBudget
    } else if context
        .current_step
        .saturating_add(probe.result_delay_steps)
        > context.decision_deadline_step
    {
        RobustProbeAdmission::MissesDecisionDeadline
    } else {
        RobustProbeAdmission::Eligible
    };

    Ok(RobustProbeValue {
        probe_id: probe.probe_id.clone(),
        admission,
        current_minimax_loss,
        worst_case_post_observation_minimax_loss,
        guaranteed_worst_case_loss_reduction: (current_minimax_loss
            - worst_case_post_observation_minimax_loss)
            .max(0.0),
        acquisition_cost_units: probe.acquisition_cost_units,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem(
        state_ids: &[&str],
        decision_ids: &[&str],
        losses: &[(&str, &[(&str, f64)])],
        threshold: f64,
    ) -> RobustDecisionProblem {
        RobustDecisionProblem {
            state_ids: state_ids.iter().map(|id| (*id).to_owned()).collect(),
            decision_ids: decision_ids.iter().map(|id| (*id).to_owned()).collect(),
            decision_loss: losses
                .iter()
                .map(|(decision, entries)| {
                    (
                        (*decision).to_owned(),
                        entries
                            .iter()
                            .map(|(state, loss)| ((*state).to_owned(), *loss))
                            .collect(),
                    )
                })
                .collect(),
            maximum_worst_case_loss: threshold,
        }
    }

    fn context() -> RobustProbeContext {
        RobustProbeContext {
            permitted_probe_ids: vec!["reference".to_owned()],
            remaining_cost_budget_units: 1.0,
            current_step: 0,
            decision_deadline_step: 5,
        }
    }

    fn perfect_probe() -> RobustProbe {
        RobustProbe {
            probe_id: "reference".to_owned(),
            acquisition_cost_units: 0.1,
            result_delay_steps: 1,
            outcomes: vec![
                RobustProbeOutcome {
                    outcome_id: "A".to_owned(),
                    possible_state_ids: vec!["H_A".to_owned()],
                },
                RobustProbeOutcome {
                    outcome_id: "B".to_owned(),
                    possible_state_ids: vec!["H_B".to_owned()],
                },
            ],
        }
    }

    #[test]
    fn prior_free_probe_reduces_worst_case_decision_loss() {
        let problem = problem(
            &["H_A", "H_B"],
            &["choose_A", "choose_B", "abstain"],
            &[
                ("choose_A", &[("H_A", 0.0), ("H_B", 1.0)]),
                ("choose_B", &[("H_A", 1.0), ("H_B", 0.0)]),
                ("abstain", &[("H_A", 0.4), ("H_B", 0.4)]),
            ],
            0.4,
        );
        let value = evaluate_robust_probe(&problem, &perfect_probe(), &context()).unwrap();
        assert_eq!(value.admission, RobustProbeAdmission::Eligible);
        assert!((value.current_minimax_loss - 0.4).abs() < TOLERANCE);
        assert!(value.worst_case_post_observation_minimax_loss.abs() < TOLERANCE);
        assert!((value.guaranteed_worst_case_loss_reduction - 0.4).abs() < TOLERANCE);
    }

    #[test]
    fn perfect_state_discrimination_can_have_zero_decision_value() {
        let problem = problem(
            &["H_A", "H_B"],
            &["continue", "stop"],
            &[
                ("continue", &[("H_A", 0.0), ("H_B", 0.0)]),
                ("stop", &[("H_A", 1.0), ("H_B", 1.0)]),
            ],
            0.0,
        );
        let value = evaluate_robust_probe(&problem, &perfect_probe(), &context()).unwrap();
        assert!((value.guaranteed_worst_case_loss_reduction).abs() < TOLERANCE);
        assert!((value.acquisition_cost_units - 0.1).abs() < TOLERANCE);
    }

    #[test]
    fn nondiscriminating_probe_has_zero_robust_value() {
        let problem = problem(
            &["H_A", "H_B"],
            &["choose_A", "choose_B", "abstain"],
            &[
                ("choose_A", &[("H_A", 0.0), ("H_B", 1.0)]),
                ("choose_B", &[("H_A", 1.0), ("H_B", 0.0)]),
                ("abstain", &[("H_A", 0.4), ("H_B", 0.4)]),
            ],
            0.4,
        );
        let probe = RobustProbe {
            probe_id: "reference".to_owned(),
            acquisition_cost_units: 0.1,
            result_delay_steps: 1,
            outcomes: vec![RobustProbeOutcome {
                outcome_id: "ambiguous".to_owned(),
                possible_state_ids: vec!["H_A".to_owned(), "H_B".to_owned()],
            }],
        };
        let value = evaluate_robust_probe(&problem, &probe, &context()).unwrap();
        assert!(value.guaranteed_worst_case_loss_reduction.abs() < TOLERANCE);
    }

    #[test]
    fn prior_free_probe_respects_permission_budget_and_deadline() {
        let problem = problem(
            &["H_A", "H_B"],
            &["choose_A", "choose_B", "abstain"],
            &[
                ("choose_A", &[("H_A", 0.0), ("H_B", 1.0)]),
                ("choose_B", &[("H_A", 1.0), ("H_B", 0.0)]),
                ("abstain", &[("H_A", 0.4), ("H_B", 0.4)]),
            ],
            0.4,
        );
        let probe = perfect_probe();

        let mut blocked = context();
        blocked.permitted_probe_ids.clear();
        assert_eq!(
            evaluate_robust_probe(&problem, &probe, &blocked)
                .unwrap()
                .admission,
            RobustProbeAdmission::NotPermitted
        );

        let mut over_budget = context();
        over_budget.remaining_cost_budget_units = 0.01;
        assert_eq!(
            evaluate_robust_probe(&problem, &probe, &over_budget)
                .unwrap()
                .admission,
            RobustProbeAdmission::OverBudget
        );

        let mut late = context();
        late.decision_deadline_step = 0;
        assert_eq!(
            evaluate_robust_probe(&problem, &probe, &late)
                .unwrap()
                .admission,
            RobustProbeAdmission::MissesDecisionDeadline
        );
    }

    #[test]
    fn prior_free_outcome_support_must_cover_every_state() {
        let problem = problem(
            &["H_A", "H_B"],
            &["continue"],
            &[("continue", &[("H_A", 0.0), ("H_B", 0.0)])],
            0.0,
        );
        let probe = RobustProbe {
            probe_id: "reference".to_owned(),
            acquisition_cost_units: 0.1,
            result_delay_steps: 1,
            outcomes: vec![RobustProbeOutcome {
                outcome_id: "only_A".to_owned(),
                possible_state_ids: vec!["H_A".to_owned()],
            }],
        };
        assert!(matches!(
            evaluate_robust_probe(&problem, &probe, &context()),
            Err(RobustEvaluationError::UncoveredState(_))
        ));
    }
}
