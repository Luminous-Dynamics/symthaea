// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emit the exact adaptive depth-two industrial diagnosis policy tree.

use industrial_diagnosis_voi::evaluate_case;
use industrial_diagnosis_voi::reference_case;
use industrial_diagnosis_voi::reference_measurements;
use industrial_diagnosis_voi::sequential::{evaluate_depth_two, MeasurementOutcome};

fn fmt(value: f64) -> String {
    format!("{value:.6}")
}

fn main() {
    let case = reference_case();
    let measurements = reference_measurements();
    let result = evaluate_depth_two(&case, &measurements, 4.0)
        .expect("frozen sequential diagnosis case must be valid");

    println!("record_type\tcase_id\tpolicy_or_branch\tfirst_outcome\tsecond_measurement\tstop_decision\texpected_terminal_loss\texpected_measurement_cost\texpected_total_loss\tworst_case_branch_loss\tprobability");
    println!(
        "BASELINE\t{}\tNO_TEST\t-\t-\t{}\t{}\t0.000000\t{}\t{}\t1.000000",
        case.case_id,
        result.no_test_decision.decision_id,
        fmt(result.no_test_total_loss),
        fmt(result.no_test_total_loss),
        fmt(result.no_test_total_loss)
    );
    let one_step = evaluate_case(&case, &measurements)
        .expect("frozen one-step comparison must be valid");
    if let Some(measurement_id) = result.greedy_one_step_measurement_id {
        let evaluation = one_step
            .measurements
            .iter()
            .find(|item| item.measurement_id == measurement_id)
            .expect("selected greedy measurement must have an evaluation");
        println!(
            "GREEDY_ONE_STEP\t{}\t{}\t-\t-\t-\t{}\t{}\t{}\t-\t1.000000",
            case.case_id,
            measurement_id,
            fmt(evaluation.expected_posterior_loss.unwrap()),
            fmt(evaluation.measurement_cost.unwrap()),
            fmt(result.greedy_one_step_total_loss)
        );
    } else {
        println!(
            "GREEDY_ONE_STEP\t{}\tNO_TEST\t-\t-\t{}\t{}\t0.000000\t{}\t-\t1.000000",
            case.case_id,
            result.no_test_decision.decision_id,
            fmt(result.no_test_total_loss),
            fmt(result.no_test_total_loss)
        );
    }

    match result.adaptive_depth_two {
        Some(policy) => {
            println!(
                "ADAPTIVE_DEPTH_TWO\t{}\t{}\t-\t-\t-\t{}\t{}\t{}\t{}\t1.000000",
                case.case_id,
                policy.first_measurement_id,
                fmt(policy.expected_terminal_decision_loss),
                fmt(policy.expected_measurement_cost),
                fmt(policy.expected_total_loss),
                fmt(policy.worst_case_branch_total_loss)
            );
            for branch in policy.branches {
                let outcome = match branch.first_outcome {
                    MeasurementOutcome::Positive => "POSITIVE",
                    MeasurementOutcome::Negative => "NEGATIVE",
                };
                println!(
                    "BRANCH\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    case.case_id,
                    policy.first_measurement_id,
                    outcome,
                    branch.second_measurement_id.unwrap_or("STOP"),
                    branch.stop_decision_id.unwrap_or("-"),
                    fmt(branch.terminal_expected_decision_loss),
                    fmt(branch.second_measurement_cost),
                    fmt(branch.branch_expected_total_loss),
                    fmt(branch.branch_expected_total_loss),
                    fmt(branch.outcome_probability)
                );
            }
        }
        None => {
            println!(
                "ADAPTIVE_DEPTH_TWO\t{}\tNO_TEST\t-\t-\t{}\t{}\t0.000000\t{}\t{}\t1.000000",
                case.case_id,
                result.no_test_decision.decision_id,
                fmt(result.no_test_total_loss),
                fmt(result.no_test_total_loss),
                fmt(result.no_test_total_loss)
            );
        }
    }
}
