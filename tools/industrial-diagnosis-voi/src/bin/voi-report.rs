// SPDX-License-Identifier: AGPL-3.0-or-later
//! Emit exact one-step diagnosis and decision-value-of-information benchmark.

use industrial_diagnosis_voi::{evaluate_case, reference_case, reference_measurements};

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_else(|| "-".to_owned())
}

fn main() {
    let case = reference_case();
    let measurements = reference_measurements();
    let result = evaluate_case(&case, &measurements).expect("frozen reference case is valid");

    println!("record_type\tcase_id\tpolicy_or_test\tdisposition\tbaseline_decision\tmeasurement_decision_positive\tmeasurement_decision_negative\tinformation_gain_bits\texpected_posterior_loss\tmeasurement_cost\tgross_voi\tnet_voi");
    println!(
        "BASELINE\t{}\tNO_MEASUREMENT\tELIGIBLE\t{}\t-\t-\t-\t{}\t0.000000\t0.000000\t0.000000",
        case.case_id,
        result.baseline_decision.decision_id,
        fmt(Some(result.baseline_decision.expected_loss))
    );

    for evaluation in &result.measurements {
        let disposition = match evaluation.disposition {
            industrial_diagnosis_voi::MeasurementDisposition::Eligible => "ELIGIBLE",
            industrial_diagnosis_voi::MeasurementDisposition::Blocked => "BLOCKED",
        };
        println!(
            "MEASUREMENT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.case_id,
            evaluation.measurement_id,
            disposition,
            result.baseline_decision.decision_id,
            evaluation.positive_decision.as_ref().map(|decision| decision.decision_id).unwrap_or("-"),
            evaluation.negative_decision.as_ref().map(|decision| decision.decision_id).unwrap_or("-"),
            fmt(evaluation.information_gain_bits),
            fmt(evaluation.expected_posterior_loss),
            fmt(evaluation.measurement_cost),
            fmt(evaluation.gross_value_of_information),
            fmt(evaluation.net_value_of_information),
        );
    }

    println!(
        "POLICY\t{}\tMAX_EXPECTED_INFORMATION_GAIN\t{}\t-\t-\t-\t{}\t-\t-\t-\t-",
        case.case_id,
        result.information_gain_choice.measurement_id.unwrap_or("NO_TEST"),
        fmt(result.information_gain_choice.score)
    );
    println!(
        "POLICY\t{}\tMAX_NET_VALUE_OF_INFORMATION\t{}\t-\t-\t-\t-\t-\t-\t-\t{}",
        case.case_id,
        result.decision_value_choice.measurement_id.unwrap_or("NO_TEST"),
        fmt(result.decision_value_choice.score)
    );
}
