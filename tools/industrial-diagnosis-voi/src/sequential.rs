// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exhaustive adaptive two-stage measurement policy evaluator.
//!
//! The policy tree chooses a first measurement, observes one of its declared
//! binary outcomes, and then chooses a second measurement or stops on each
//! branch. Distinct evidence origins are assumed conditionally independent
//! given the supplied finite hypothesis model; a reused origin is blocked.

use super::{
    DecisionCase, DecisionEstimate, Measurement, MeasurementDisposition, evaluate_case,
    evaluate_measurement, minimum_expected_loss, posterior,
};

const LOSS_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementOutcome {
    Positive,
    Negative,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SequentialError {
    InvalidCase(Vec<super::CaseInputFailure>),
    InvalidMeasurementBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveBranch {
    pub first_outcome: MeasurementOutcome,
    pub outcome_probability: f64,
    pub second_measurement_id: Option<&'static str>,
    pub stop_decision_id: Option<&'static str>,
    pub second_positive_decision_id: Option<&'static str>,
    pub second_negative_decision_id: Option<&'static str>,
    pub terminal_expected_decision_loss: f64,
    pub second_measurement_cost: f64,
    pub branch_expected_total_loss: f64,
    pub second_positive_probability: Option<f64>,
    pub second_negative_probability: Option<f64>,
    pub second_positive_leaf_total_loss: Option<f64>,
    pub second_negative_leaf_total_loss: Option<f64>,
    pub worst_case_leaf_total_loss: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptivePolicy {
    pub first_measurement_id: &'static str,
    pub first_measurement_cost: f64,
    pub branches: Vec<AdaptiveBranch>,
    pub expected_terminal_decision_loss: f64,
    pub expected_measurement_cost: f64,
    pub expected_total_loss: f64,
    pub worst_case_leaf_total_loss: f64,
    pub worst_case_regret_vs_no_test: f64,
    pub improvement_over_no_test: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SequentialBenchmark {
    pub no_test_decision: DecisionEstimate,
    pub no_test_total_loss: f64,
    pub greedy_one_step_measurement_id: Option<&'static str>,
    pub greedy_one_step_total_loss: f64,
    pub adaptive_depth_two: Option<AdaptivePolicy>,
    pub first_measurements_enumerated: usize,
    pub nonzero_first_outcome_branches: usize,
    pub second_measurements_evaluated: usize,
}

struct BranchChoice {
    branch: AdaptiveBranch,
    terminal_loss: f64,
    expected_second_cost: f64,
}

fn lexically_better_id(candidate: &'static str, incumbent: &'static str) -> bool {
    candidate < incumbent
}

fn best_second_measurement(
    branch_case: &DecisionCase,
    measurements: &[Measurement],
    first_measurement: &Measurement,
    remaining_budget: f64,
    current_decision: &DecisionEstimate,
    evaluated_count: &mut usize,
) -> Option<(Measurement, super::MeasurementEvaluation)> {
    let mut best: Option<(Measurement, super::MeasurementEvaluation, f64)> = None;

    for measurement in measurements {
        // A second test cannot reuse the first test or its evidence origin.
        if measurement.id == first_measurement.id
            || measurement.evidence_origin == first_measurement.evidence_origin
        {
            continue;
        }
        *evaluated_count += 1;
        let evaluation = evaluate_measurement(branch_case, measurement);
        if evaluation.disposition != MeasurementDisposition::Eligible {
            continue;
        }
        let Some(cost) = evaluation.measurement_cost else {
            continue;
        };
        if cost > remaining_budget + LOSS_TOLERANCE {
            continue;
        }
        let Some(posterior_loss) = evaluation.expected_posterior_loss else {
            continue;
        };
        let net_value = current_decision.expected_loss - posterior_loss - cost;
        if net_value <= LOSS_TOLERANCE {
            continue;
        }

        let replace = match &best {
            None => true,
            Some((incumbent_measurement, _, incumbent_net)) => {
                net_value > *incumbent_net + LOSS_TOLERANCE
                    || ((net_value - *incumbent_net).abs() <= LOSS_TOLERANCE
                        && lexically_better_id(measurement.id, incumbent_measurement.id))
            }
        };
        if replace {
            best = Some((measurement.clone(), evaluation, net_value));
        }
    }

    best.map(|(measurement, evaluation, _)| (measurement, evaluation))
}

struct FirstBranchEvaluation<'a> {
    case: &'a DecisionCase,
    measurements: &'a [Measurement],
    first_measurement: &'a Measurement,
    first_outcome: MeasurementOutcome,
    outcome_probability: f64,
    posterior_values: &'a [f64],
    remaining_budget: f64,
    second_evaluated_count: &'a mut usize,
}

fn evaluate_first_branch(context: FirstBranchEvaluation<'_>) -> BranchChoice {
    let FirstBranchEvaluation {
        case,
        measurements,
        first_measurement,
        first_outcome,
        outcome_probability,
        posterior_values,
        remaining_budget,
        second_evaluated_count,
    } = context;
    let mut branch_case = case.clone();
    for (hypothesis, probability) in branch_case.hypotheses.iter_mut().zip(posterior_values) {
        hypothesis.prior = *probability;
    }
    // Preserve the source identity so a second measurement based on the same
    // evidence origin is rejected as a reuse/correlation hazard.
    branch_case
        .existing_evidence_origins
        .push(first_measurement.evidence_origin);

    let current_decision = minimum_expected_loss(&branch_case, posterior_values);
    let second = best_second_measurement(
        &branch_case,
        measurements,
        first_measurement,
        remaining_budget,
        &current_decision,
        second_evaluated_count,
    );

    let branch = match second {
        Some((measurement, evaluation)) => {
            let terminal_loss = evaluation
                .expected_posterior_loss
                .expect("eligible measurement has a posterior-loss estimate");
            let second_cost = evaluation
                .measurement_cost
                .expect("eligible measurement has a cost");
            let second_positive_probability = evaluation.probability_positive;
            let second_negative_probability = evaluation
                .probability_positive
                .map(|probability| (1.0 - probability).max(0.0));
            let second_positive_leaf_total_loss =
                evaluation.positive_decision.as_ref().map(|decision| {
                    first_measurement.cost_units + second_cost + decision.expected_loss
                });
            let second_negative_leaf_total_loss =
                evaluation.negative_decision.as_ref().map(|decision| {
                    first_measurement.cost_units + second_cost + decision.expected_loss
                });
            let worst_case_leaf_total_loss = [
                second_positive_leaf_total_loss,
                second_negative_leaf_total_loss,
            ]
            .into_iter()
            .flatten()
            .max_by(f64::total_cmp)
            .unwrap_or(first_measurement.cost_units + terminal_loss + second_cost);
            AdaptiveBranch {
                first_outcome,
                outcome_probability,
                second_measurement_id: Some(measurement.id),
                stop_decision_id: None,
                second_positive_decision_id: evaluation
                    .positive_decision
                    .as_ref()
                    .map(|decision| decision.decision_id),
                second_negative_decision_id: evaluation
                    .negative_decision
                    .as_ref()
                    .map(|decision| decision.decision_id),
                terminal_expected_decision_loss: terminal_loss,
                second_measurement_cost: second_cost,
                branch_expected_total_loss: first_measurement.cost_units
                    + terminal_loss
                    + second_cost,
                second_positive_probability,
                second_negative_probability,
                second_positive_leaf_total_loss,
                second_negative_leaf_total_loss,
                worst_case_leaf_total_loss,
            }
        }
        None => AdaptiveBranch {
            first_outcome,
            outcome_probability,
            second_measurement_id: None,
            stop_decision_id: Some(current_decision.decision_id),
            second_positive_decision_id: None,
            second_negative_decision_id: None,
            terminal_expected_decision_loss: current_decision.expected_loss,
            second_measurement_cost: 0.0,
            branch_expected_total_loss: first_measurement.cost_units
                + current_decision.expected_loss,
            second_positive_probability: None,
            second_negative_probability: None,
            second_positive_leaf_total_loss: None,
            second_negative_leaf_total_loss: None,
            worst_case_leaf_total_loss: first_measurement.cost_units
                + current_decision.expected_loss,
        },
    };

    BranchChoice {
        terminal_loss: branch.terminal_expected_decision_loss,
        expected_second_cost: branch.second_measurement_cost * outcome_probability,
        branch,
    }
}

/// Exhaustively enumerate all admissible first measurements and the best
/// remaining measurement-or-stop decision on each observed first-outcome
/// branch. Measurement costs are charged once and a branch may stop early.
///
/// "Exact" means exhaustive for this two-stage, binary-outcome, finite
/// hypothesis model. It does not model arbitrary cross-sensor likelihood
/// dependence: distinct evidence origins are assumed conditionally independent.
pub fn evaluate_depth_two(
    case: &DecisionCase,
    measurements: &[Measurement],
    measurement_budget_units: f64,
) -> Result<SequentialBenchmark, SequentialError> {
    if !measurement_budget_units.is_finite() || measurement_budget_units < 0.0 {
        return Err(SequentialError::InvalidMeasurementBudget);
    }
    let baseline = evaluate_case(case, measurements).map_err(SequentialError::InvalidCase)?;
    let no_test_loss = baseline.baseline_decision.expected_loss;

    // Greedy one-step decision-value baseline under the same total budget.
    let mut greedy_choice: Option<(&Measurement, &super::MeasurementEvaluation, f64)> = None;
    for (measurement, evaluation) in measurements.iter().zip(&baseline.measurements) {
        if evaluation.disposition != MeasurementDisposition::Eligible {
            continue;
        }
        let Some(cost) = evaluation.measurement_cost else {
            continue;
        };
        if cost > measurement_budget_units + LOSS_TOLERANCE {
            continue;
        }
        let Some(net) = evaluation.net_value_of_information else {
            continue;
        };
        if net <= LOSS_TOLERANCE {
            continue;
        }
        let replace = match greedy_choice {
            None => true,
            Some((incumbent, _, incumbent_net)) => {
                net > incumbent_net + LOSS_TOLERANCE
                    || ((net - incumbent_net).abs() <= LOSS_TOLERANCE
                        && lexically_better_id(measurement.id, incumbent.id))
            }
        };
        if replace {
            greedy_choice = Some((measurement, evaluation, net));
        }
    }

    let (greedy_id, greedy_total_loss) = match greedy_choice {
        Some((measurement, evaluation, _)) => {
            let posterior_loss = evaluation
                .expected_posterior_loss
                .expect("eligible measurement has a posterior-loss estimate");
            let cost = evaluation
                .measurement_cost
                .expect("eligible measurement has a cost");
            (Some(measurement.id), posterior_loss + cost)
        }
        None => (None, no_test_loss),
    };

    let mut best_policy: Option<AdaptivePolicy> = None;
    let mut second_measurements_evaluated = 0usize;
    let mut nonzero_branches = 0usize;
    let mut first_measurements_enumerated = 0usize;
    let prior: Vec<f64> = case
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior)
        .collect();

    for (first_measurement, first_evaluation) in measurements.iter().zip(&baseline.measurements) {
        if first_evaluation.disposition != MeasurementDisposition::Eligible {
            continue;
        }
        let Some(first_cost) = first_evaluation.measurement_cost else {
            continue;
        };
        if first_cost > measurement_budget_units + LOSS_TOLERANCE {
            continue;
        }
        let Some(likelihoods) = first_measurement.positive_likelihoods.as_ref() else {
            continue;
        };
        first_measurements_enumerated += 1;
        let probability_positive = first_evaluation
            .probability_positive
            .expect("eligible measurement has an outcome probability");
        let probability_negative = (1.0 - probability_positive).max(0.0);
        let positive_posterior = posterior(&prior, likelihoods, true, probability_positive);
        let negative_posterior = posterior(&prior, likelihoods, false, probability_negative);

        let mut branches = Vec::with_capacity(2);
        let mut expected_terminal_loss = 0.0;
        let mut expected_measurement_cost = first_cost;

        if probability_positive > 0.0
            && let Some(posterior_values) = positive_posterior
        {
            let choice = evaluate_first_branch(FirstBranchEvaluation {
                case,
                measurements,
                first_measurement,
                first_outcome: MeasurementOutcome::Positive,
                outcome_probability: probability_positive,
                posterior_values: &posterior_values,
                remaining_budget: measurement_budget_units - first_cost,
                second_evaluated_count: &mut second_measurements_evaluated,
            });
            nonzero_branches += 1;
            expected_terminal_loss += probability_positive * choice.terminal_loss;
            expected_measurement_cost += choice.expected_second_cost;
            branches.push(choice.branch);
        }

        if probability_negative > 0.0
            && let Some(posterior_values) = negative_posterior
        {
            let choice = evaluate_first_branch(FirstBranchEvaluation {
                case,
                measurements,
                first_measurement,
                first_outcome: MeasurementOutcome::Negative,
                outcome_probability: probability_negative,
                posterior_values: &posterior_values,
                remaining_budget: measurement_budget_units - first_cost,
                second_evaluated_count: &mut second_measurements_evaluated,
            });
            nonzero_branches += 1;
            expected_terminal_loss += probability_negative * choice.terminal_loss;
            expected_measurement_cost += choice.expected_second_cost;
            branches.push(choice.branch);
        }

        let expected_total_loss = expected_terminal_loss + expected_measurement_cost;
        let improvement = no_test_loss - expected_total_loss;
        if improvement <= LOSS_TOLERANCE {
            continue;
        }

        let worst_case_leaf_total_loss = branches
            .iter()
            .map(|branch| branch.worst_case_leaf_total_loss)
            .max_by(f64::total_cmp)
            .unwrap_or(expected_total_loss);
        let candidate = AdaptivePolicy {
            first_measurement_id: first_measurement.id,
            first_measurement_cost: first_cost,
            branches,
            expected_terminal_decision_loss: expected_terminal_loss,
            expected_measurement_cost,
            expected_total_loss,
            worst_case_leaf_total_loss,
            worst_case_regret_vs_no_test: (worst_case_leaf_total_loss - no_test_loss).max(0.0),
            improvement_over_no_test: improvement,
        };

        let replace = match &best_policy {
            None => true,
            Some(incumbent) => {
                candidate.expected_total_loss < incumbent.expected_total_loss - LOSS_TOLERANCE
                    || ((candidate.expected_total_loss - incumbent.expected_total_loss).abs()
                        <= LOSS_TOLERANCE
                        && lexically_better_id(
                            candidate.first_measurement_id,
                            incumbent.first_measurement_id,
                        ))
            }
        };
        if replace {
            best_policy = Some(candidate);
        }
    }

    Ok(SequentialBenchmark {
        no_test_decision: baseline.baseline_decision,
        no_test_total_loss: no_test_loss,
        greedy_one_step_measurement_id: greedy_id,
        greedy_one_step_total_loss: greedy_total_loss,
        adaptive_depth_two: best_policy,
        first_measurements_enumerated,
        nonzero_first_outcome_branches: nonzero_branches,
        second_measurements_evaluated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reference_case, reference_measurements};

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn adaptive_depth_two_selects_calibration_then_conditional_process_test() {
        let result = evaluate_depth_two(&reference_case(), &reference_measurements(), 4.0).unwrap();
        assert_eq!(
            result.greedy_one_step_measurement_id,
            Some("INDEPENDENT_PROCESS_SAFETY_TEST")
        );
        let policy = result.adaptive_depth_two.unwrap();
        assert_eq!(
            policy.first_measurement_id,
            "CALIBRATION_REFERENCE_ARTIFACT"
        );
        close(policy.expected_total_loss, 6.598);
        assert_eq!(policy.branches.len(), 2);
        assert_eq!(
            policy.branches[0].first_outcome,
            MeasurementOutcome::Positive
        );
        assert_eq!(policy.branches[0].second_measurement_id, None);
        assert_eq!(
            policy.branches[0].stop_decision_id,
            Some("RECALIBRATE_AND_RETEST")
        );
        assert_eq!(
            policy.branches[1].first_outcome,
            MeasurementOutcome::Negative
        );
        assert_eq!(
            policy.branches[1].second_measurement_id,
            Some("INDEPENDENT_PROCESS_SAFETY_TEST")
        );
    }

    #[test]
    fn adaptive_depth_two_beats_greedy_one_step_on_frozen_case() {
        let result = evaluate_depth_two(&reference_case(), &reference_measurements(), 4.0).unwrap();
        let adaptive = result.adaptive_depth_two.unwrap();
        assert!(adaptive.expected_total_loss < result.greedy_one_step_total_loss);
        close(result.greedy_one_step_total_loss, 9.77);
        close(adaptive.expected_total_loss, 6.598);
        close(adaptive.expected_terminal_decision_loss, 4.248);
        close(adaptive.expected_measurement_cost, 2.35);
        close(adaptive.worst_case_leaf_total_loss, 15.0);
        close(adaptive.worst_case_regret_vs_no_test, 3.6);
    }

    #[test]
    fn adaptive_policy_reports_worst_case_leaf_loss_separately() {
        let result = evaluate_depth_two(&reference_case(), &reference_measurements(), 4.0).unwrap();
        let policy = result.adaptive_depth_two.unwrap();
        let expected_max = policy
            .branches
            .iter()
            .map(|branch| branch.worst_case_leaf_total_loss)
            .max_by(f64::total_cmp)
            .unwrap();
        close(policy.worst_case_leaf_total_loss, expected_max);
        assert!(policy.worst_case_leaf_total_loss >= policy.expected_total_loss);
    }

    #[test]
    fn adaptive_second_test_exposes_both_leaf_losses() {
        let result = evaluate_depth_two(&reference_case(), &reference_measurements(), 4.0).unwrap();
        let policy = result.adaptive_depth_two.unwrap();
        let negative = &policy.branches[1];
        close(negative.second_positive_probability.unwrap(), 0.2077777778);
        close(negative.second_negative_probability.unwrap(), 0.7922222222);
        close(negative.second_positive_leaf_total_loss.unwrap(), 15.0);
        close(
            negative.second_negative_leaf_total_loss.unwrap(),
            5.159887798,
        );
        close(negative.worst_case_leaf_total_loss, 15.0);
    }

    #[test]
    fn measurement_cost_is_charged_once_and_budget_is_respected() {
        let result = evaluate_depth_two(&reference_case(), &reference_measurements(), 4.0).unwrap();
        let policy = result.adaptive_depth_two.unwrap();
        close(policy.expected_measurement_cost, 2.35);
        assert!(policy.expected_measurement_cost <= 4.0);
        assert_eq!(policy.first_measurement_cost, 1.0);
        close(policy.branches[1].second_measurement_cost, 2.0);
    }

    #[test]
    fn second_test_is_not_reused_on_the_same_evidence_origin() {
        let case = reference_case();
        let mut measurements = reference_measurements();
        measurements[1].evidence_origin = measurements[0].evidence_origin;
        measurements.truncate(2);
        let result = evaluate_depth_two(&case, &measurements, 4.0).unwrap();
        let policy = result.adaptive_depth_two.unwrap();
        assert!(
            policy
                .branches
                .iter()
                .all(|branch| branch.second_measurement_id.is_none())
        );
    }

    #[test]
    fn zero_probability_outcome_has_no_posterior_branch() {
        let prior = vec![0.10, 0.30, 0.60];
        let impossible_positive = vec![0.0, 0.0, 0.0];
        assert!(posterior(&prior, &impossible_positive, true, 0.0).is_none());
        let negative = posterior(&prior, &impossible_positive, false, 1.0).unwrap();
        assert_eq!(negative, prior);
    }

    #[test]
    fn budget_too_small_for_a_test_falls_back_to_no_test() {
        let result = evaluate_depth_two(&reference_case(), &reference_measurements(), 0.1).unwrap();
        assert_eq!(result.greedy_one_step_measurement_id, None);
        assert_eq!(result.greedy_one_step_total_loss, result.no_test_total_loss);
        assert_eq!(result.adaptive_depth_two, None);
    }

    #[test]
    fn adaptive_policy_respects_budget_below_two_test_combination_cost() {
        let result =
            evaluate_depth_two(&reference_case(), &reference_measurements(), 1.49).unwrap();
        let policy = result.adaptive_depth_two.unwrap();
        assert!(policy.expected_measurement_cost <= 1.49 + LOSS_TOLERANCE);
        assert!(
            policy
                .branches
                .iter()
                .all(|branch| { branch.second_measurement_id.is_none() })
        );
    }

    #[test]
    fn invalid_budget_fails_closed() {
        assert_eq!(
            evaluate_depth_two(&reference_case(), &reference_measurements(), f64::NAN),
            Err(SequentialError::InvalidMeasurementBudget)
        );
        assert_eq!(
            evaluate_depth_two(&reference_case(), &reference_measurements(), -1.0),
            Err(SequentialError::InvalidMeasurementBudget)
        );
    }

    #[test]
    fn adaptive_depth_two_is_deterministic_under_replay() {
        let first = evaluate_depth_two(&reference_case(), &reference_measurements(), 4.0).unwrap();
        let second = evaluate_depth_two(&reference_case(), &reference_measurements(), 4.0).unwrap();
        assert_eq!(first, second);
    }
}
